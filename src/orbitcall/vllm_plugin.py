# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 Yuhao Xie
"""Explicit suffix scoring for the pinned vLLM 0.29 legacy GPU runner.

The row/chunk handling below adapts vLLM's Apache-2.0
GPUModelRunner._get_prompt_logprobs_dict. Native model.compute_logits,
sampler.compute_logprobs and gather_logprobs are retained; this is not a
second model runtime or a restricted-vocabulary probability approximation.
Original source: https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/worker/gpu_model_runner.py

Enable ORBITCALL_VLLM_SCORING=1 and VLLM_USE_V2_MODEL_RUNNER=0 before creating
an engine with async_scheduling=False. Score requests carry
extra_args={"orbitcall_score_start": S}, where hidden row S predicts input
token S+1. Returned prompt_logprobs has S+1 leading None entries followed by
every requested suffix token's native raw logprob. None means unscored.
Untagged requests delegate to the original methods without changing settings.
The native-shape path supports matching FP16 or BF16, non-sliding Qwen2,
single-cache-group runtimes. Historical development results cover BF16 only.
An optional orbitcall_score_logit_dtype="float32" request marker temporarily
selects vLLM's native FP32-output LM-head path during prompt scoring. Its BF16
operands and backbone are unchanged; generation outside this hook is unchanged.
The separate orbitcall_score_mode="native_shape_suffix" marker preserves the
entire stock FP16/BF16 prompt LM-head projection and crops logits only afterward.
It forbids prefix-cache reads and the FP32-head marker. Prefix-only chunks
still run the full native head; only normalization, gathering and return work
are omitted for their unrequested rows.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import os


MARKER = "orbitcall_score_start"
HEAD_MARKER = "orbitcall_score_logit_dtype"
MODE_MARKER = "orbitcall_score_mode"
ENABLE_ENV = "ORBITCALL_VLLM_SCORING"
_CACHE_BOUND = ContextVar("orbitcall_vllm_cache_bound", default=None)


def _score_mode(params):
    """Validate the independent native-shape path without enabling old modes."""
    extras = getattr(params, "extra_args", None) if params is not None else None
    if not extras or MODE_MARKER not in extras:
        return None
    mode = extras[MODE_MARKER]
    if type(mode) is not str or mode != "native_shape_suffix":
        raise ValueError("orbitcall_score_mode supports only 'native_shape_suffix'")
    if (MARKER not in extras
            or getattr(params, "skip_reading_prefix_cache", None) is not True
            or HEAD_MARKER in extras):
        raise ValueError("native_shape_suffix requires a causal start, explicit "
                         "skip_reading_prefix_cache=True and no score-head marker")
    return mode


def _score_start(params, prompt_len=None):
    """Validate an explicitly marked request; return None for native requests."""
    _score_mode(params)
    extras = getattr(params, "extra_args", None) if params is not None else None
    if not extras or MARKER not in extras:
        return None
    start = extras[MARKER]
    if type(start) is not int or start < 0:
        raise ValueError("orbitcall_score_start must be a non-negative integer")
    if (type(getattr(params, "prompt_logprobs", None)) is not int
            or params.prompt_logprobs != 0
            or getattr(params, "n", None) != 1
            or getattr(params, "max_tokens", None) != 1
            or getattr(params, "detokenize", None) is not False
            or getattr(params, "flat_logprobs", None) is not False
            or type(getattr(params, "skip_reading_prefix_cache", None)) is not bool):
        raise ValueError("Marked scoring requires prompt_logprobs=0, n=1, max_tokens=1, "
                         "detokenize=False, flat_logprobs=False and an explicit cache policy")
    if prompt_len is not None and not 0 <= start < prompt_len - 1:
        raise ValueError("The causal scoring start must leave at least one target token")
    return start


def _head_dtype(params):
    """Read the opt-in score-head policy without changing default requests."""
    _score_mode(params)
    extras = getattr(params, "extra_args", None) if params is not None else None
    if not extras or HEAD_MARKER not in extras:
        return None
    value = extras[HEAD_MARKER]
    if type(value) is not str or value != "float32":
        raise ValueError("orbitcall_score_logit_dtype supports only explicit 'float32'")
    if (type(getattr(params, "prompt_logprobs", None)) is not int
            or params.prompt_logprobs != 0
            or getattr(params, "n", None) != 1
            or getattr(params, "max_tokens", None) != 1
            or getattr(params, "detokenize", None) is not False
            or getattr(params, "flat_logprobs", None) is not False
            or type(getattr(params, "skip_reading_prefix_cache", None)) is not bool):
        raise ValueError("Marked score-head requests require prompt_logprobs=0, n=1, "
                         "max_tokens=1, detokenize=False, flat_logprobs=False "
                         "and an explicit cache policy")
    return value


def _install_cache_hooks(manager_cls, coordinator_cls, full_attention_spec_cls):
    """Bound native APC lookup per request without mutating a request's length."""
    original_lookup = coordinator_cls.find_longest_cache_hit
    if getattr(original_lookup, "_orbitcall_scoring_hook", False):
        return
    original_get = manager_cls.get_computed_blocks

    @wraps(original_lookup)
    def lookup(self, block_hashes, max_cache_hit_length):
        context = _CACHE_BOUND.get()
        if context is not None and context[0] is self:
            max_cache_hit_length = min(max_cache_hit_length, context[1])
        return original_lookup(self, block_hashes, max_cache_hit_length)

    @wraps(original_get)
    def get(self, request):
        start = _score_start(getattr(request, "sampling_params", None),
                             getattr(request, "num_prompt_tokens", None))
        if start is None:
            return original_get(self, request)
        coordinator = self.coordinator
        if (type(coordinator) is not coordinator_cls
                or not isinstance(coordinator.kv_cache_spec, full_attention_spec_cls)):
            raise RuntimeError("OrbitCall APC requires a single full-attention cache group")
        if not request.skip_reading_prefix_cache and not self.enable_caching:
            raise RuntimeError("Marked APC scoring requires enable_prefix_caching=True")
        token = _CACHE_BOUND.set((coordinator, start))
        try:
            result = original_get(self, request)
            if not 0 <= result[1] <= start:
                raise RuntimeError("APC skipped a required causal scoring row")
            return result
        finally:
            _CACHE_BOUND.reset(token)

    lookup._orbitcall_scoring_hook = True
    get._orbitcall_scoring_hook = True
    coordinator_cls.find_longest_cache_hit = lookup
    manager_cls.get_computed_blocks = get

    # Connector/hybrid lookups have a different path and are not this study.
    original_connector = getattr(manager_cls, "get_computed_blocks_for_connector", None)
    if original_connector is not None:
        @wraps(original_connector)
        def connector(self, request):
            if _score_start(getattr(request, "sampling_params", None)) is not None:
                raise RuntimeError("OrbitCall marked scoring does not support KV connectors")
            return original_connector(self, request)
        manager_cls.get_computed_blocks_for_connector = connector


def _install_frontend_hook(processor_cls):
    """Preserve token-index alignment using explicit unscored None positions."""
    original = processor_cls.from_new_request.__func__
    if getattr(original, "_orbitcall_scoring_hook", False):
        return

    @classmethod
    @wraps(original)
    def create(cls, tokenizer, request):
        result = original(cls, tokenizer, request)
        prompt_ids = getattr(request, "prompt_token_ids", None)
        start = _score_start(request.sampling_params,
                             len(prompt_ids) if prompt_ids is not None else None)
        if start is not None:
            if prompt_ids is None:
                raise RuntimeError("OrbitCall suffix scoring requires token-ID prompts")
            if type(result.prompt_logprobs) is not list or result.prompt_logprobs != [None]:
                raise RuntimeError("Unexpected native prompt-logprob container")
            # Native index zero already has its None sentinel. Add only S.
            result.prompt_logprobs.extend([None] * start)
            result.tokenizer = None  # Tagged requests explicitly disable detokenization.
        return result

    create.__func__._orbitcall_scoring_hook = True
    processor_cls.from_new_request = create


def _install_runner_hook(runner_cls, logprobs_tensors_cls, async_h2d):
    """Adapt native chunk handling only for explicitly marked suffix requests."""
    original = runner_cls._get_prompt_logprobs_dict
    if getattr(original, "_orbitcall_scoring_hook", False):
        return

    @wraps(original)
    def compute(self, hidden_states, num_scheduled_tokens):
        tagged = {}
        modes = {}
        for req_id in self.num_prompt_logprobs:
            request = self.requests[req_id]
            prompt = request.prompt_token_ids
            start = _score_start(request.sampling_params,
                                 len(prompt) if prompt is not None else None)
            if start is not None:
                if prompt is None:
                    raise RuntimeError("OrbitCall suffix scoring requires token-ID prompts")
                if getattr(request, "lora_request", None) is not None:
                    raise RuntimeError("OrbitCall marked scoring does not support LoRA")
                tagged[req_id] = start
                modes[req_id] = _score_mode(request.sampling_params)
        if not tagged:
            return original(self, hidden_states, num_scheduled_tokens)

        import torch

        config = self.model_config
        native_shape = any(mode == "native_shape_suffix" for mode in modes.values())
        supported_dtypes = (torch.float16, torch.bfloat16) if native_shape else (torch.bfloat16,)
        if (config.dtype not in supported_dtypes or config.logprobs_mode != "raw_logprobs"
                or getattr(config.hf_config, "model_type", None) != "qwen2"
                or getattr(self.scheduler_config, "async_scheduling", False)
                or getattr(self, "speculative_config", None) is not None
                or getattr(self.cache_config, "kv_sharing_fast_prefill", False)
                or getattr(self.cache_config, "cache_dtype", "auto") != "auto"
                or getattr(config, "quantization", None) is not None):
            raise RuntimeError("OrbitCall supports native FP16/BF16 raw-logprob Qwen2, synchronous V1 "
                               "scheduling, without speculative/KV-sharing fast prefill")
        if config.dtype == torch.float16 and any(mode != "native_shape_suffix" for mode in modes.values()):
            raise RuntimeError("FP16 marked scoring requires native_shape_suffix for every marked request")
        if native_shape:
            processor = getattr(self.model, "logits_processor", None)
            head = getattr(self.model, "lm_head", None)
            if (hidden_states.dtype != config.dtype
                    or processor is None or not hasattr(processor, "head_dtype")
                    or processor.head_dtype not in (None, config.dtype)
                    or getattr(getattr(head, "weight", None), "dtype", None)
                    != config.dtype):
                raise RuntimeError("native_shape_suffix requires matching native FP16/BF16 "
                                   "model, hidden states, LM-head weights and output dtype")

        # Native requests retain the original buffer, cleanup and sync behavior.
        # The legacy runner executes this method serially; no scheduler state
        # or sampling parameters are changed by this temporary dictionary view.
        removed = {key: self.num_prompt_logprobs.pop(key) for key in tagged}
        try:
            output = original(self, hidden_states, num_scheduled_tokens)
        finally:
            self.num_prompt_logprobs.update(removed)

        completed = []
        for req_id, start in tagged.items():
            scheduled = num_scheduled_tokens.get(req_id)
            if scheduled is None:
                continue  # A request preempted in prefill can be unscheduled.
            request = self.requests[req_id]
            prompt = request.prompt_token_ids
            prompt_len, chunk_start = len(prompt), request.num_computed_tokens
            remaining = prompt_len - (chunk_start + 1)
            final_chunk = scheduled > remaining  # Preserve native equality-defer rule.
            buffer = request.in_progress_prompt_logprobs_cpu
            if buffer is None:
                if chunk_start > start:
                    raise RuntimeError("The initial cache hit skipped requested suffix scores")
                buffer = logprobs_tensors_cls.empty_cpu(prompt_len - start - 1, 1)
                request.in_progress_prompt_logprobs_cpu = buffer
                request._orbitcall_written_until = start
            lo, hi = max(chunk_start, start), min(chunk_start + scheduled, prompt_len - 1)
            native_logits = None
            if modes[req_id] == "native_shape_suffix":
                # Preserve the exact upstream projection rows, stride and base
                # offset. Do this even when the chunk contains only prefix rows.
                # No clone, packing, contiguous conversion or head bypass occurs.
                num_logits = scheduled if scheduled <= remaining else remaining
                if num_logits > 0:
                    req_idx = self.input_batch.req_id_to_index[req_id]
                    offset = int(self.query_start_loc.np[req_idx])
                    native_states = hidden_states[offset:offset + num_logits]
                    native_logits = self.model.compute_logits(native_states)
                    if native_logits is None:
                        raise RuntimeError("OrbitCall does not support pipeline-parallel scoring")
            if lo < hi:
                if lo > request._orbitcall_written_until:
                    raise RuntimeError("A prefill gap left suffix scores uncomputed")
                if modes[req_id] == "native_shape_suffix":
                    logits = native_logits[lo - chunk_start:hi - chunk_start]
                else:
                    req_idx = self.input_batch.req_id_to_index[req_id]
                    offset = int(self.query_start_loc.np[req_idx])
                    predictor_states = hidden_states[offset + lo - chunk_start:
                                                    offset + hi - chunk_start]
                    logits = self.model.compute_logits(predictor_states)
                if logits is None:
                    raise RuntimeError("OrbitCall does not support pipeline-parallel scoring")
                targets = async_h2d(prompt[lo + 1:hi + 1], device=self.device)
                full_logprobs = self.sampler.compute_logprobs(logits)
                token_ids, values, ranks, *_ = self.sampler.gather_logprobs(
                    full_logprobs, 0, targets)
                destination = slice(lo - start, hi - start)
                buffer.logprob_token_ids[destination].copy_(token_ids, non_blocking=True)
                buffer.logprobs[destination].copy_(values, non_blocking=True)
                buffer.selected_token_ranks[destination].copy_(ranks, non_blocking=True)
                request._orbitcall_written_until = max(request._orbitcall_written_until, hi)
            if final_chunk:
                if request._orbitcall_written_until != prompt_len - 1:
                    raise RuntimeError("Suffix scoring did not cover every target including EOS")
                output[req_id] = buffer
                completed.append(req_id)

        for req_id in completed:
            del self.num_prompt_logprobs[req_id]
            self.requests[req_id].in_progress_prompt_logprobs_cpu = None
            del self.requests[req_id]._orbitcall_written_until
        if completed:
            self._sync_device()  # Complete native nonblocking D2H before returning.
        return output

    compute._orbitcall_scoring_hook = True
    runner_cls._get_prompt_logprobs_dict = compute


def _install_head_hook(runner_cls):
    """Scope native FP32-output logits to a homogeneous prompt-score batch.

    This wraps the suffix hook, so HEAD-only requests remain the native full
    prompt scorer. vLLM's LogitsProcessor._apply_head uses torch.mm with BF16
    input/weight operands and out_dtype=torch.float32 on CUDA. No weights,
    activations, logits formula, model instance or Torch global flags are
    replaced by this hook. The field is restored on every exit.
    """
    original = runner_cls._get_prompt_logprobs_dict
    if getattr(original, "_orbitcall_score_head_hook", False):
        return

    @wraps(original)
    def compute(self, hidden_states, num_scheduled_tokens):
        requests = [self.requests[req_id] for req_id in self.num_prompt_logprobs]
        policies = [_head_dtype(request.sampling_params) for request in requests]
        if not any(policy is not None for policy in policies):
            return original(self, hidden_states, num_scheduled_tokens)
        if any(policy != "float32" for policy in policies):
            raise ValueError("Mixed stock/FP32 score-head requests are unsupported; "
                             "all tracked prompt-score requests must carry the same marker")

        import torch

        config = self.model_config
        if (config.dtype != torch.bfloat16 or hidden_states.dtype != torch.bfloat16
                or config.logprobs_mode != "raw_logprobs"
                or getattr(config.hf_config, "model_type", None) != "qwen2"
                or getattr(self.scheduler_config, "async_scheduling", False)
                or getattr(self, "speculative_config", None) is not None
                or getattr(self.cache_config, "kv_sharing_fast_prefill", False)
                or getattr(self.cache_config, "cache_dtype", "auto") != "auto"
                or getattr(config, "quantization", None) is not None
                or any(getattr(request, "lora_request", None) is not None
                       for request in requests)):
            raise RuntimeError("FP32 score logits require the supported native BF16 Qwen2 runtime")
        processor = getattr(self.model, "logits_processor", None)
        head = getattr(self.model, "lm_head", None)
        if (processor is None or not hasattr(processor, "head_dtype")
                or head is None or getattr(getattr(head, "weight", None), "dtype", None)
                != torch.bfloat16):
            raise RuntimeError("Missing native LogitsProcessor head_dtype or BF16 LM-head weights")
        saved = processor.head_dtype
        if saved not in (None, torch.bfloat16):
            raise RuntimeError("FP32 score logits expect an otherwise stock BF16 generation head")
        try:
            processor.head_dtype = torch.float32
            return original(self, hidden_states, num_scheduled_tokens)
        finally:
            processor.head_dtype = saved

    compute._orbitcall_score_head_hook = True
    runner_cls._get_prompt_logprobs_dict = compute


def register():
    """vLLM general-plugin entry point; disabled unless explicitly enabled."""
    if os.environ.get(ENABLE_ENV) != "1":
        return
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise RuntimeError("OrbitCall suffix plugin requires VLLM_USE_V2_MODEL_RUNNER=0")
    from importlib.metadata import version
    if version("vllm").split("+")[0] != "0.29.0":
        raise RuntimeError("OrbitCall suffix plugin is validated only against vLLM 0.29.0")
    from vllm.utils.torch_utils import async_tensor_h2d
    from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.engine.logprobs import LogprobsProcessor
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _install_cache_hooks(KVCacheManager, UnitaryKVCacheCoordinator, FullAttentionSpec)
    _install_frontend_hook(LogprobsProcessor)
    _install_runner_hook(GPUModelRunner, LogprobsTensors, async_tensor_h2d)
    _install_head_hook(GPUModelRunner)
