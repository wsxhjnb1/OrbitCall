"""CPU-only hook contracts; run on the authorized Spark, never on Windows.

These fakes exercise vLLM 0.29 chunk and frontend boundaries without importing
vLLM, loading weights, creating CUDA contexts, or claiming runtime performance.
"""

from types import SimpleNamespace

import pytest
import torch

from orbitcall import vllm_plugin as plugin


EOS = 0
PROMPT = [1, 2, 3, 4, 1, EOS]


def params(start=2, **changes):
    values = dict(extra_args={plugin.MARKER: start}, prompt_logprobs=0, n=1,
                  max_tokens=1, detokenize=False, flat_logprobs=False,
                  skip_reading_prefix_cache=False)
    values.update(changes)
    return SimpleNamespace(**values)


def request(start=2, *, prompt=None, computed=0, **changes):
    token_ids = list(PROMPT if prompt is None else prompt)
    sampling = params(start, **changes)
    return SimpleNamespace(sampling_params=sampling, prompt_token_ids=token_ids,
                           num_prompt_tokens=len(token_ids), num_tokens=len(token_ids),
                           num_computed_tokens=computed, block_hashes=object(),
                           skip_reading_prefix_cache=sampling.skip_reading_prefix_cache,
                           in_progress_prompt_logprobs_cpu=None, lora_request=None)


def states():
    # Distinct rows expose both wrong request offsets and wrong causal shifts.
    # Column six never occurs in PROMPT, yet must remain in the denominator.
    return torch.tensor([
        [0., 2., -1., 3., -4., 1., 8.],
        [4., 0., -2., 1., 3., -1., 7.],
        [1., -3., 2., 0., 4., -2., 9.],
        [-2., 3., 1., -1., 0., 2., 6.],
        [2., 1., -4., 0., 3., -1., 10.],
        [3., -2., 0., 2., -1., 1., 5.],
    ], dtype=torch.float32)


class PromptTensors:
    @classmethod
    def empty_cpu(cls, rows, columns):
        return SimpleNamespace(
            logprob_token_ids=torch.empty((rows, columns), dtype=torch.long),
            logprobs=torch.empty((rows, columns), dtype=torch.float32),
            selected_token_ranks=torch.empty(rows, dtype=torch.long),
        )


class Model:
    def __init__(self):
        self.calls = []
        self.input_layouts = []
        self.return_none = False

    def compute_logits(self, hidden):
        self.input_layouts.append(dict(shape=tuple(hidden.shape), stride=tuple(hidden.stride()),
                                       storage_offset=hidden.storage_offset(), data_ptr=hidden.data_ptr()))
        self.calls.append(hidden.clone())
        return None if self.return_none else hidden


class Sampler:
    def __init__(self):
        self.normalized_shapes = []
        self.targets = []

    def compute_logprobs(self, logits):
        self.normalized_shapes.append(tuple(logits.shape))
        return torch.log_softmax(logits.float(), dim=-1)

    def gather_logprobs(self, scores, count, targets):
        assert count == 0
        self.targets.append(targets.clone())
        rows = torch.arange(len(targets))
        selected = scores[rows, targets]
        ranks = (scores > selected[:, None]).sum(dim=-1) + 1
        return targets[:, None], selected[:, None], ranks


def runner_class():
    class Runner:
        def __init__(self, requests):
            self.requests = requests
            self.num_prompt_logprobs = {key: 0 for key in requests}
            self.input_batch = SimpleNamespace(
                req_id_to_index={key: index for index, key in enumerate(requests)})
            self.query_start_loc = SimpleNamespace(np=[0] * len(requests))
            self.model_config = SimpleNamespace(
                dtype=torch.bfloat16, logprobs_mode="raw_logprobs",
                hf_config=SimpleNamespace(model_type="qwen2"), quantization=None)
            self.scheduler_config = SimpleNamespace(async_scheduling=False)
            self.cache_config = SimpleNamespace(kv_sharing_fast_prefill=False,
                                                cache_dtype="auto")
            self.speculative_config = None
            self.model, self.sampler = Model(), Sampler()
            self.device = torch.device("cpu")
            self.original_calls, self.syncs = [], 0
            self.native_output = {}
            self.native_completed = set()
            self.original_exception = None

        def _get_prompt_logprobs_dict(self, hidden, scheduled):
            active = tuple(self.num_prompt_logprobs)
            self.original_calls.append((active, hidden, scheduled))
            if self.original_exception is not None:
                raise self.original_exception
            output = {key: value for key, value in self.native_output.items() if key in active}
            for key in self.native_completed:
                self.num_prompt_logprobs.pop(key, None)
            return output

        def _sync_device(self):
            self.syncs += 1

    plugin._install_runner_hook(
        Runner, PromptTensors,
        lambda values, *, device: torch.tensor(values, dtype=torch.long, device=device))
    return Runner


def processor_class(container=None):
    class Processor:
        last_result = None
        calls = []

        @classmethod
        def from_new_request(cls, tokenizer, new_request):
            cls.calls.append((tokenizer, new_request))
            result = cls()
            result.tokenizer = tokenizer
            result.prompt_logprobs = ([None] if container is None else container())
            cls.last_result = result
            return result

        def _update_prompt_logprobs(self, tensors):
            for token, value, rank in zip(tensors.logprob_token_ids[:, 0].tolist(),
                                          tensors.logprobs[:, 0].tolist(),
                                          tensors.selected_token_ranks.tolist(), strict=True):
                self.prompt_logprobs.append({token: SimpleNamespace(
                    logprob=value, rank=rank, decoded_token=None)})

    plugin._install_frontend_hook(Processor)
    return Processor


def cache_classes(block_size=1):
    class FullSpec:
        pass

    class Coordinator:
        def __init__(self):
            self.kv_cache_spec = FullSpec()
            self.calls = []
            self.hit_override = None
            self.last_result = None

        def find_longest_cache_hit(self, hashes, maximum):
            self.calls.append((hashes, maximum))
            hit = maximum // block_size * block_size
            if self.hit_override is not None:
                hit = self.hit_override
            self.last_result = (([object()],), hit, 0)
            return self.last_result

    class Manager:
        def __init__(self, coordinator):
            self.coordinator = coordinator
            self.enable_caching = True
            self.on_get = None
            self.native_zero = ((), 0, 0)
            self.connector_result = object()

        def get_computed_blocks(self, req):
            if self.on_get is not None:
                return self.on_get(self, req)
            if req.skip_reading_prefix_cache or not self.enable_caching:
                return self.native_zero
            return self.coordinator.find_longest_cache_hit(req.block_hashes, req.num_tokens - 1)

        def get_computed_blocks_for_connector(self, req):
            return self.connector_result

    plugin._install_cache_hooks(Manager, Coordinator, FullSpec)
    return Manager, Coordinator, FullSpec


@pytest.mark.parametrize("start", [True, False, -1, 1.0, "2", None])
def test_marker_requires_nonnegative_builtin_integer(start):
    with pytest.raises(ValueError):
        plugin._score_start(params(start))


@pytest.mark.parametrize("start, length", [(0, 1), (5, 6), (6, 6), (0, 0)])
def test_marker_must_leave_a_target(start, length):
    with pytest.raises(ValueError):
        plugin._score_start(params(start), length)


@pytest.mark.parametrize("changes", [
    {"prompt_logprobs": None}, {"prompt_logprobs": True}, {"prompt_logprobs": 0.0},
    {"prompt_logprobs": 1}, {"n": 2}, {"max_tokens": 2}, {"detokenize": True},
    {"flat_logprobs": True}, {"skip_reading_prefix_cache": None},
    {"skip_reading_prefix_cache": 0},
])
def test_marked_request_contract_is_explicit(changes):
    with pytest.raises(ValueError):
        plugin._score_start(params(**changes), len(PROMPT))


@pytest.mark.parametrize("skip", [True, False])
def test_valid_marker_accepts_explicit_cache_policy(skip):
    assert plugin._score_start(params(0, skip_reading_prefix_cache=skip), 2) == 0
    assert plugin._score_start(params(4, skip_reading_prefix_cache=skip), 6) == 4


def test_untagged_parameters_are_not_validated_as_scoring_requests():
    assert plugin._score_start(None) is None
    assert plugin._score_start(SimpleNamespace(extra_args=None)) is None
    assert plugin._score_start(SimpleNamespace(extra_args={"other": -1})) is None


@pytest.mark.parametrize("enabled", ["0", "true", ""])
def test_registration_disabled_is_a_noop(monkeypatch, enabled):
    monkeypatch.setenv(plugin.ENABLE_ENV, enabled)
    assert plugin.register() is None


def test_registration_rejects_other_runner_before_importing_vllm(monkeypatch):
    monkeypatch.setenv(plugin.ENABLE_ENV, "1")
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    with pytest.raises(RuntimeError, match="VLLM_USE_V2_MODEL_RUNNER"):
        plugin.register()


def test_registration_rejects_other_vllm_version(monkeypatch):
    import importlib.metadata
    monkeypatch.setenv(plugin.ENABLE_ENV, "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "0.30.0")
    with pytest.raises(RuntimeError, match="0.29.0"):
        plugin.register()


@pytest.mark.parametrize("start, block_size, expected", [(4, 4, 4), (5, 4, 4), (0, 4, 0)])
def test_cache_cap_leaves_first_predictor_recomputable(start, block_size, expected):
    Manager, Coordinator, _ = cache_classes(block_size)
    coordinator, req = Coordinator(), request(start, prompt=[1] * 11 + [EOS])
    manager = Manager(coordinator)
    lengths = (req.num_tokens, req.num_prompt_tokens)
    result = manager.get_computed_blocks(req)
    assert coordinator.calls == [(req.block_hashes, start)]
    assert result is coordinator.last_result
    assert result[1] == expected <= start
    assert (req.num_tokens, req.num_prompt_tokens) == lengths
    assert plugin._CACHE_BOUND.get() is None


def test_untagged_cache_result_and_original_bound_are_preserved():
    Manager, Coordinator, _ = cache_classes()
    coordinator = Coordinator()
    manager = Manager(coordinator)
    req = request(extra_args=None)
    result = manager.get_computed_blocks(req)
    assert result is coordinator.last_result
    assert coordinator.calls == [(req.block_hashes, len(PROMPT) - 1)]


def test_skip_cache_policy_does_not_invent_a_hit():
    Manager, Coordinator, _ = cache_classes()
    coordinator = Coordinator()
    manager = Manager(coordinator)
    assert manager.get_computed_blocks(request(skip_reading_prefix_cache=True)) is manager.native_zero
    assert not coordinator.calls


def test_cache_context_is_scoped_to_the_specific_coordinator():
    Manager, Coordinator, _ = cache_classes()
    first, other = Coordinator(), Coordinator()
    manager = Manager(first)
    req = request(2)

    def native_get(instance, current):
        other.find_longest_cache_hit(current.block_hashes, 50)
        return first.find_longest_cache_hit(current.block_hashes, 50)

    manager.on_get = native_get
    manager.get_computed_blocks(req)
    assert first.calls[-1][1] == 2
    assert other.calls[-1][1] == 50
    assert plugin._CACHE_BOUND.get() is None


def test_nested_cache_context_restores_outer_bound():
    Manager, Coordinator, _ = cache_classes()
    coordinator = Coordinator()
    manager = Manager(coordinator)
    outer, inner = request(4), request(1)

    def native_get(instance, current):
        before = coordinator.find_longest_cache_hit(current.block_hashes, 20)
        if current is outer:
            instance.get_computed_blocks(inner)
            return coordinator.find_longest_cache_hit(current.block_hashes, 20)
        return before

    manager.on_get = native_get
    manager.get_computed_blocks(outer)
    assert [maximum for _, maximum in coordinator.calls] == [4, 1, 4]
    assert plugin._CACHE_BOUND.get() is None


def test_cache_exception_cannot_leak_bound_to_native_requests():
    Manager, Coordinator, _ = cache_classes()
    coordinator = Coordinator()
    manager = Manager(coordinator)

    def fail(instance, current):
        coordinator.find_longest_cache_hit(current.block_hashes, 20)
        raise RuntimeError("injected native failure")

    manager.on_get = fail
    with pytest.raises(RuntimeError, match="injected"):
        manager.get_computed_blocks(request(2))
    assert plugin._CACHE_BOUND.get() is None
    manager.on_get = None
    manager.get_computed_blocks(request(extra_args=None))
    assert coordinator.calls[-1][1] == len(PROMPT) - 1


@pytest.mark.parametrize("hit", [-1, 3])
def test_invalid_cache_hit_is_rejected_and_context_restored(hit):
    Manager, Coordinator, _ = cache_classes()
    coordinator = Coordinator()
    coordinator.hit_override = hit
    with pytest.raises(RuntimeError, match="required causal"):
        Manager(coordinator).get_computed_blocks(request(2))
    assert plugin._CACHE_BOUND.get() is None


def test_cache_requires_supported_spec_and_enabled_apc():
    Manager, Coordinator, _ = cache_classes()
    coordinator = Coordinator()
    coordinator.kv_cache_spec = object()
    with pytest.raises(RuntimeError, match="single full-attention"):
        Manager(coordinator).get_computed_blocks(request())
    coordinator = Coordinator()
    manager = Manager(coordinator)
    manager.enable_caching = False
    with pytest.raises(RuntimeError, match="enable_prefix_caching"):
        manager.get_computed_blocks(request())


def test_connector_delegates_native_but_rejects_marked_requests():
    Manager, Coordinator, _ = cache_classes()
    manager = Manager(Coordinator())
    assert manager.get_computed_blocks_for_connector(request(extra_args=None)) is manager.connector_result
    with pytest.raises(RuntimeError, match="KV connectors"):
        manager.get_computed_blocks_for_connector(request())


def test_cache_hook_installation_is_idempotent():
    Manager, Coordinator, FullSpec = cache_classes()
    methods = (Manager.get_computed_blocks, Coordinator.find_longest_cache_hit,
               Manager.get_computed_blocks_for_connector)
    plugin._install_cache_hooks(Manager, Coordinator, FullSpec)
    assert methods == (Manager.get_computed_blocks, Coordinator.find_longest_cache_hit,
                       Manager.get_computed_blocks_for_connector)


def test_frontend_has_total_s_plus_one_unscored_positions():
    Processor = processor_class()
    frontend = Processor.from_new_request(object(), request(2))
    assert frontend.prompt_logprobs == [None, None, None]
    assert frontend.tokenizer is None


def test_frontend_native_result_is_unchanged():
    Processor = processor_class()
    tokenizer = object()
    req = request(extra_args=None, detokenize=True)
    frontend = Processor.from_new_request(tokenizer, req)
    assert frontend is Processor.last_result
    assert frontend.prompt_logprobs == [None]
    assert frontend.tokenizer is tokenizer


@pytest.mark.parametrize("container", [lambda: [], lambda: [None, None], lambda: (None,)])
def test_frontend_rejects_unknown_native_sentinel_container(container):
    Processor = processor_class(container)
    with pytest.raises(RuntimeError, match="container"):
        Processor.from_new_request(None, request())


def test_frontend_rejects_embedding_prompt():
    Processor = processor_class()
    req = request()
    req.prompt_token_ids = None
    with pytest.raises(RuntimeError, match="token-ID"):
        Processor.from_new_request(None, req)


def test_frontend_hook_does_not_pad_twice():
    Processor = processor_class()
    original = Processor.from_new_request.__func__
    plugin._install_frontend_hook(Processor)
    assert Processor.from_new_request.__func__ is original
    assert Processor.from_new_request(None, request(4)).prompt_logprobs == [None] * 5


@pytest.mark.parametrize("start", [0, 2, 4])
def test_suffix_targets_and_full_vocabulary_denominator_including_eos(start):
    Runner = runner_class()
    req, hidden = request(start), states()
    runner = Runner({"score": req})
    output = runner._get_prompt_logprobs_dict(hidden, {"score": len(PROMPT)})
    tensors = output["score"]
    targets = torch.tensor(PROMPT[start + 1:])
    reference = torch.log_softmax(hidden.float(), dim=-1)
    expected = reference[torch.arange(start, len(PROMPT) - 1), targets]
    assert tensors.logprob_token_ids[:, 0].tolist() == PROMPT[start + 1:]
    assert tensors.logprob_token_ids[-1, 0].item() == EOS
    torch.testing.assert_close(tensors.logprobs[:, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(runner.model.calls[0], hidden[start:-1], rtol=0, atol=0)
    assert runner.sampler.normalized_shapes == [(len(targets), hidden.shape[1])]
    assert runner.syncs == 1
    assert not runner.num_prompt_logprobs
    assert req.in_progress_prompt_logprobs_cpu is None
    assert not hasattr(req, "_orbitcall_written_until")


def test_complete_frontend_alignment_contains_no_fabricated_prefix_scores():
    req = request(2)
    Runner, Processor = runner_class(), processor_class()
    frontend = Processor.from_new_request(None, req)
    output = Runner({"score": req})._get_prompt_logprobs_dict(states(), {"score": len(PROMPT)})
    frontend._update_prompt_logprobs(output["score"])
    assert len(frontend.prompt_logprobs) == len(PROMPT)
    assert frontend.prompt_logprobs[:3] == [None] * 3
    assert [next(iter(position)) for position in frontend.prompt_logprobs[3:]] == PROMPT[3:]
    assert next(iter(frontend.prompt_logprobs[-1])) == EOS
    assert all(next(iter(position.values())).decoded_token is None
               for position in frontend.prompt_logprobs[3:])


def test_chunk_before_start_and_chunk_crossing_start_only_project_requested_rows():
    req, hidden = request(2), states()
    runner = runner_class()({"score": req})
    assert runner._get_prompt_logprobs_dict(hidden[:2], {"score": 2}) == {}
    assert not runner.model.calls and runner.syncs == 0
    req.num_computed_tokens = 2
    assert runner._get_prompt_logprobs_dict(hidden[2:4], {"score": 2}) == {}
    req.num_computed_tokens = 4
    output = runner._get_prompt_logprobs_dict(hidden[4:6], {"score": 2})
    torch.testing.assert_close(torch.cat(runner.model.calls), hidden[2:5], rtol=0, atol=0)
    assert output["score"].logprob_token_ids[:, 0].tolist() == PROMPT[3:]
    assert runner.syncs == 1


def test_native_equality_defer_returns_filled_scores_on_final_no_logits_chunk():
    req, hidden = request(2, computed=2), states()
    runner = runner_class()({"score": req})
    assert runner._get_prompt_logprobs_dict(hidden[2:5], {"score": 3}) == {}
    assert req._orbitcall_written_until == len(PROMPT) - 1
    assert "score" in runner.num_prompt_logprobs
    assert runner.syncs == 0
    req.num_computed_tokens = 5
    output = runner._get_prompt_logprobs_dict(hidden[5:], {"score": 1})
    assert output["score"].logprob_token_ids[-1, 0].item() == EOS
    assert len(runner.model.calls) == 1
    assert runner.syncs == 1


def test_initial_cached_boundary_equals_s_but_never_skips_s():
    req = request(2, computed=2)
    runner = runner_class()({"score": req})
    output = runner._get_prompt_logprobs_dict(states()[2:], {"score": 4})
    assert output["score"].logprob_token_ids[:, 0].tolist() == PROMPT[3:]
    runner = runner_class()({"score": request(2, computed=3)})
    with pytest.raises(RuntimeError, match="initial cache hit"):
        runner._get_prompt_logprobs_dict(states()[3:], {"score": 3})


def test_unscheduled_marked_request_keeps_state_without_projecting_rows():
    req = request()
    runner = runner_class()({"score": req})
    assert runner._get_prompt_logprobs_dict(states()[:0], {}) == {}
    assert runner.num_prompt_logprobs == {"score": 0}
    assert req.in_progress_prompt_logprobs_cpu is None
    assert not runner.model.calls and not runner.syncs


def test_native_request_bypasses_all_marked_runner_configuration_checks():
    runner = runner_class()({"native": request(extra_args=None)})
    runner.model_config.dtype = torch.float32
    sentinel = object()
    runner.native_output = {"native": sentinel}
    hidden, scheduled = states(), {"native": len(PROMPT)}
    assert runner._get_prompt_logprobs_dict(hidden, scheduled)["native"] is sentinel
    keys, seen_hidden, seen_scheduled = runner.original_calls[-1]
    assert keys == ("native",)
    assert seen_hidden is hidden and seen_scheduled is scheduled
    assert not runner.model.calls


def test_mixed_native_requests_preserve_original_cleanup_and_batch_offsets():
    native, scored = request(extra_args=None), request(2)
    runner = runner_class()({"native": native, "score": scored})
    runner.query_start_loc.np = [0, 2]
    sentinel = object()
    runner.native_output = {"native": sentinel}
    runner.native_completed = {"native"}
    hidden = torch.cat((states()[:2], states()))
    output = runner._get_prompt_logprobs_dict(hidden, {"native": 2, "score": len(PROMPT)})
    assert runner.original_calls[-1][0] == ("native",)
    assert output["native"] is sentinel
    assert output["score"].logprob_token_ids[:, 0].tolist() == PROMPT[3:]
    torch.testing.assert_close(runner.model.calls[0], states()[2:5], rtol=0, atol=0)
    assert not runner.num_prompt_logprobs


def test_original_runner_exception_restores_marked_dictionary_entries():
    runner = runner_class()({"score": request(), "native": request(extra_args=None)})
    runner.original_exception = RuntimeError("native failure")
    with pytest.raises(RuntimeError, match="native failure"):
        runner._get_prompt_logprobs_dict(states(), {"score": len(PROMPT)})
    assert set(runner.num_prompt_logprobs) == {"score", "native"}


def test_prefill_gap_is_rejected_before_uninitialized_scores_can_escape():
    req = request(2)
    runner = runner_class()({"score": req})
    runner._get_prompt_logprobs_dict(states()[:3], {"score": 3})
    assert req._orbitcall_written_until == 3
    req.num_computed_tokens = 4
    with pytest.raises(RuntimeError, match="prefill gap"):
        runner._get_prompt_logprobs_dict(states()[4:], {"score": 2})
    assert runner.syncs == 0


def test_final_chunk_cannot_return_a_buffer_missing_eos():
    req = request(2)
    runner = runner_class()({"score": req})
    runner._get_prompt_logprobs_dict(states()[:4], {"score": 4})
    req.num_computed_tokens = 5
    with pytest.raises(RuntimeError, match="including EOS"):
        runner._get_prompt_logprobs_dict(states()[5:], {"score": 1})
    assert runner.syncs == 0


def test_preemption_recompute_can_overwrite_already_scored_rows_without_a_gap():
    req, hidden = request(2), states()
    runner = runner_class()({"score": req})
    runner._get_prompt_logprobs_dict(hidden[:4], {"score": 4})
    req.num_computed_tokens = 0
    assert runner._get_prompt_logprobs_dict(hidden[:5], {"score": 5}) == {}
    req.num_computed_tokens = 5
    output = runner._get_prompt_logprobs_dict(hidden[5:], {"score": 1})
    assert output["score"].logprob_token_ids[:, 0].tolist() == PROMPT[3:]
    expected = torch.log_softmax(hidden.float(), dim=-1)[torch.arange(2, 5), torch.tensor(PROMPT[3:])]
    torch.testing.assert_close(output["score"].logprobs[:, 0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("field, value", [
    ("dtype", torch.float32), ("logprobs_mode", "raw_logits"), ("quantization", "awq"),
    ("model_type", "llama"), ("async_scheduling", True),
    ("kv_sharing_fast_prefill", True), ("cache_dtype", "fp8"),
    ("speculative_config", object()), ("lora_request", object()),
])
def test_marked_runner_rejects_unsupported_runtime(field, value):
    req = request()
    runner = runner_class()({"score": req})
    if field in ("dtype", "logprobs_mode", "quantization"):
        setattr(runner.model_config, field, value)
    elif field == "model_type":
        runner.model_config.hf_config.model_type = value
    elif field == "async_scheduling":
        runner.scheduler_config.async_scheduling = value
    elif field in ("kv_sharing_fast_prefill", "cache_dtype"):
        setattr(runner.cache_config, field, value)
    elif field == "lora_request":
        req.lora_request = value
    else:
        setattr(runner, field, value)
    with pytest.raises(RuntimeError):
        runner._get_prompt_logprobs_dict(states(), {"score": len(PROMPT)})
    assert not runner.model.calls


def test_marked_runner_rejects_pipeline_logits_none():
    runner = runner_class()({"score": request()})
    runner.model.return_none = True
    with pytest.raises(RuntimeError, match="pipeline-parallel"):
        runner._get_prompt_logprobs_dict(states(), {"score": len(PROMPT)})


def test_runner_hook_installation_is_idempotent():
    Runner = runner_class()
    original = Runner._get_prompt_logprobs_dict
    plugin._install_runner_hook(Runner, PromptTensors, lambda *args, **kwargs: None)
    assert Runner._get_prompt_logprobs_dict is original


def attach_stock_head(runner, saved=torch.bfloat16):
    runner.model.lm_head = SimpleNamespace(weight=torch.empty((1, 1), dtype=torch.bfloat16))
    runner.model.logits_processor = SimpleNamespace(head_dtype=saved)


def test_unflagged_head_hook_preserves_exact_native_return_and_arguments():
    sentinel = object()

    class Native:
        def __init__(self):
            self.requests = {"native": request(extra_args=None)}
            self.num_prompt_logprobs = {"native": 0}
            self.calls = []

        def _get_prompt_logprobs_dict(self, hidden, scheduled):
            self.calls.append((hidden, scheduled))
            return sentinel

    plugin._install_head_hook(Native)
    installed = Native._get_prompt_logprobs_dict
    plugin._install_head_hook(Native)
    assert Native._get_prompt_logprobs_dict is installed
    runner, hidden, scheduled = Native(), object(), {"native": 6}
    assert runner._get_prompt_logprobs_dict(hidden, scheduled) is sentinel
    assert runner.calls == [(hidden, scheduled)]
    assert plugin._head_dtype(None) is None
    assert plugin._head_dtype(params()) is None


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("saved", [None, torch.bfloat16])
def test_head_only_native_scope_restores_saved_dtype_on_every_exit(raises, saved):
    Runner = runner_class()
    delegated = Runner._get_prompt_logprobs_dict

    def observe(self, hidden, scheduled):
        self.observed_dtypes.append(self.model.logits_processor.head_dtype)
        if raises:
            raise RuntimeError("injected score-head failure")
        return delegated(self, hidden, scheduled)

    Runner._get_prompt_logprobs_dict = observe
    plugin._install_head_hook(Runner)
    req = request(extra_args={plugin.HEAD_MARKER: "float32"})
    assert plugin._score_start(req.sampling_params) is None
    assert plugin._head_dtype(req.sampling_params) == "float32"
    runner = Runner({"native": req})
    attach_stock_head(runner, saved)
    runner.observed_dtypes = []
    sentinel = object()
    runner.native_output = {"native": sentinel}
    hidden = states().to(torch.bfloat16)
    if raises:
        with pytest.raises(RuntimeError, match="injected score-head"):
            runner._get_prompt_logprobs_dict(hidden, {"native": len(PROMPT)})
    else:
        assert runner._get_prompt_logprobs_dict(hidden, {"native": len(PROMPT)})["native"] is sentinel
        assert runner.original_calls[-1][0] == ("native",)
    assert runner.observed_dtypes == [torch.float32]
    assert runner.model.logits_processor.head_dtype is saved
    assert runner.model.lm_head.weight.dtype == torch.bfloat16


def test_fp32_head_and_suffix_hook_keep_full_normalization_and_final_eos():
    Runner = runner_class()
    plugin._install_head_hook(Runner)
    req = request(extra_args={plugin.MARKER: 2, plugin.HEAD_MARKER: "float32"})
    runner = Runner({"score": req})
    attach_stock_head(runner)
    original_head = runner.model.compute_logits
    observed = []

    def project(hidden):
        observed.append(runner.model.logits_processor.head_dtype)
        return original_head(hidden).to(runner.model.logits_processor.head_dtype)

    runner.model.compute_logits = project
    hidden = states().to(torch.bfloat16)
    output = runner._get_prompt_logprobs_dict(hidden, {"score": len(PROMPT)})["score"]
    reference = torch.log_softmax(hidden.float(), dim=-1)
    expected = reference[torch.arange(2, 5), torch.tensor(PROMPT[3:])]
    torch.testing.assert_close(output.logprobs[:, 0], expected, rtol=0, atol=0)
    assert output.logprob_token_ids[:, 0].tolist() == PROMPT[3:]
    assert output.logprob_token_ids[-1, 0].item() == EOS
    assert runner.sampler.normalized_shapes == [(3, hidden.shape[1])]
    assert observed == [torch.float32]
    assert runner.model.logits_processor.head_dtype == torch.bfloat16
    assert runner.syncs == 1 and not runner.num_prompt_logprobs


def test_mixed_head_policies_fail_before_mutating_or_delegating():
    Runner = runner_class()
    plugin._install_head_hook(Runner)
    marked = request(extra_args={plugin.HEAD_MARKER: "float32"})
    runner = Runner({"fp32": marked, "stock": request(extra_args=None)})
    attach_stock_head(runner)
    with pytest.raises(ValueError, match="Mixed stock/FP32"):
        runner._get_prompt_logprobs_dict(states(), {"fp32": 6, "stock": 6})
    assert not runner.original_calls and not runner.model.calls
    assert runner.model.logits_processor.head_dtype == torch.bfloat16
    assert set(runner.num_prompt_logprobs) == {"fp32", "stock"}


@pytest.mark.parametrize("flag", [True, False, None, 0, 1, "bfloat16", "float64"])
def test_head_marker_rejects_invalid_types_and_keeps_score_request_contract(flag):
    with pytest.raises(ValueError, match="float32"):
        plugin._head_dtype(params(extra_args={plugin.HEAD_MARKER: flag}))
    with pytest.raises(ValueError, match="score-head requests"):
        plugin._head_dtype(params(extra_args={plugin.HEAD_MARKER: "float32"},
                                  prompt_logprobs=1))


def native_shape_request(start=2, computed=0, **changes):
    values = dict(extra_args={plugin.MARKER: start,
                              plugin.MODE_MARKER: "native_shape_suffix"},
                  skip_reading_prefix_cache=True)
    values.update(changes)
    return request(start, computed=computed, **values)


def test_native_shape_mode_requires_causal_start_cache_bypass_and_no_head_marker():
    assert plugin._score_mode(params()) is None
    assert plugin._score_mode(native_shape_request().sampling_params) == "native_shape_suffix"
    bad = [
        dict(extra_args={plugin.MODE_MARKER: "native_shape_suffix"}),
        dict(skip_reading_prefix_cache=False),
        dict(extra_args={plugin.MARKER: 2, plugin.MODE_MARKER: "native_shape_suffix",
                         plugin.HEAD_MARKER: "float32"}),
        dict(extra_args={plugin.MARKER: 2, plugin.MODE_MARKER: "native_shape_suffix",
                         plugin.HEAD_MARKER: None}),
    ]
    bad.extend(dict(extra_args={plugin.MARKER: 2, plugin.MODE_MARKER: value})
               for value in (True, None, "suffix_only"))
    for changes in bad:
        sampling = native_shape_request(**changes).sampling_params
        with pytest.raises(ValueError):
            plugin._score_mode(sampling)
        # Invalid mode metadata must not fall through the existing suffix hook.
        with pytest.raises(ValueError):
            plugin._score_start(sampling, len(PROMPT))


def test_native_shape_projects_full_native_view_before_cropping_logits():
    backing = torch.empty((10, 14), dtype=torch.bfloat16)
    hidden = backing[2:8, ::2]
    hidden.copy_(states().to(torch.bfloat16))
    runner = runner_class()({"score": native_shape_request()})
    attach_stock_head(runner)
    output = runner._get_prompt_logprobs_dict(hidden, {"score": len(PROMPT)})["score"]
    native_view = hidden[:len(PROMPT) - 1]
    assert runner.model.input_layouts == [dict(
        shape=tuple(native_view.shape), stride=tuple(native_view.stride()),
        storage_offset=native_view.storage_offset(), data_ptr=native_view.data_ptr())]
    torch.testing.assert_close(runner.model.calls[0], native_view, rtol=0, atol=0)
    expected = torch.log_softmax(hidden.float(), dim=-1)[
        torch.arange(2, 5), torch.tensor(PROMPT[3:])]
    torch.testing.assert_close(output.logprobs[:, 0], expected, rtol=0, atol=0)
    assert output.logprob_token_ids[:, 0].tolist() == PROMPT[3:]
    assert runner.sampler.normalized_shapes == [(3, hidden.shape[1])]
    assert runner.model.logits_processor.head_dtype == torch.bfloat16


def test_native_shape_keeps_prefix_only_head_call_and_native_equality_defer():
    req, hidden = native_shape_request(), states().to(torch.bfloat16)
    runner = runner_class()({"score": req})
    attach_stock_head(runner)
    assert runner._get_prompt_logprobs_dict(hidden[:2], {"score": 2}) == {}
    assert [layout["shape"] for layout in runner.model.input_layouts] == [(2, hidden.shape[1])]
    assert not runner.sampler.normalized_shapes and not runner.syncs
    req.num_computed_tokens = 2
    assert runner._get_prompt_logprobs_dict(hidden[2:5], {"score": 3}) == {}
    assert req._orbitcall_written_until == len(PROMPT) - 1
    assert "score" in runner.num_prompt_logprobs and not runner.syncs
    req.num_computed_tokens = 5
    output = runner._get_prompt_logprobs_dict(hidden[5:], {"score": 1})["score"]
    # Native M=0 on this final chunk: return existing EOS scores, no extra head.
    assert [layout["shape"] for layout in runner.model.input_layouts] == [
        (2, hidden.shape[1]), (3, hidden.shape[1])]
    assert runner.sampler.normalized_shapes == [(3, hidden.shape[1])]
    assert output.logprob_token_ids[:, 0].tolist() == PROMPT[3:]
    assert output.logprob_token_ids[-1, 0].item() == EOS
    assert runner.syncs == 1 and not runner.num_prompt_logprobs


def test_native_shape_mixed_requests_preserve_native_result_and_full_head_offset():
    native = request(prompt=[1, EOS], extra_args=None)
    runner = runner_class()({"native": native, "score": native_shape_request()})
    attach_stock_head(runner)
    runner.query_start_loc.np = [0, 2]
    sentinel = object()
    runner.native_output = {"native": sentinel}
    runner.native_completed = {"native"}
    hidden = torch.cat((states()[:2], states())).to(torch.bfloat16)
    output = runner._get_prompt_logprobs_dict(hidden, {"native": 2, "score": len(PROMPT)})
    native_view = hidden[2:7]
    assert runner.original_calls[-1][0] == ("native",)
    assert output["native"] is sentinel
    assert runner.model.input_layouts == [dict(
        shape=tuple(native_view.shape), stride=tuple(native_view.stride()),
        storage_offset=native_view.storage_offset(), data_ptr=native_view.data_ptr())]
    torch.testing.assert_close(runner.model.calls[0], native_view, rtol=0, atol=0)
    assert output["score"].logprob_token_ids[:, 0].tolist() == PROMPT[3:]
    assert not runner.num_prompt_logprobs


def test_native_shape_rejects_none_logits_even_for_prefix_only_chunk():
    runner = runner_class()({"score": native_shape_request()})
    attach_stock_head(runner)
    runner.model.return_none = True
    with pytest.raises(RuntimeError):
        runner._get_prompt_logprobs_dict(states()[:2].to(torch.bfloat16), {"score": 2})
    assert [layout["shape"] for layout in runner.model.input_layouts] == [(2, states().shape[1])]
    assert not runner.sampler.normalized_shapes and not runner.syncs
