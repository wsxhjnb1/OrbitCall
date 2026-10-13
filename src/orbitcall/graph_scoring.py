"""Opt-in CUDA Graph scheduling for the unchanged IEEE-FP32 FULL backbone.

Only the backbone is captured. Hidden gathering, the original 32-row LM head,
and the explicitly selected full-vocabulary normalizer remain eager. Graphs
are keyed by the *actual* (batch, padded width); no extra padding, tree, cache,
precision conversion, or attention-mask substitution is introduced.

This is an experimental execution path, not a numerical-equivalence claim.
Validate it against SequenceScorer on the authorized Spark before use. Model
weights and modules must remain immutable, and callers must not concurrently
use the shared model or capture other work in the process.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import os
import threading
import time
import warnings

import torch
from transformers.masking_utils import create_causal_mask

from .scoring import Scores, SequenceScorer


class GraphUnavailable(RuntimeError):
    """A known unsupported condition, before an unsafe replay can occur."""


@dataclass
class _PackedGroup:
    inputs: torch.Tensor  # CPU [2, B, W]: token IDs followed by the 2-D mask.
    targets: torch.Tensor
    counts: tuple[int, ...]
    prompt_length: int
    suffix_width: int

    @property
    def key(self):
        return tuple(self.inputs.shape[1:])


def _pack_group(prompt, sequences, pad_token):
    """Bulk-pack exactly the right padding and mask used by SequenceScorer."""
    counts = tuple(map(len, sequences))
    p, suffix_width = len(prompt), max(counts) - 1
    ids = [list(prompt) + list(sequence[:-1])
           + [pad_token] * (suffix_width - count + 1)
           for sequence, count in zip(sequences, counts, strict=True)]
    masks = [[1] * (p + count - 1) + [0] * (suffix_width - count + 1)
             for count in counts]
    return _PackedGroup(
        torch.tensor([ids, masks], dtype=torch.long, device="cpu"),
        torch.tensor([token for sequence in sequences for token in sequence],
                     dtype=torch.long, device="cpu"),
        counts, p, suffix_width,
    )


@dataclass
class _GraphEntry:
    graph: object
    inputs: torch.Tensor
    positions: torch.Tensor
    output: torch.Tensor
    pool_bytes: int
    static_bytes: int
    attention_mask: torch.Tensor | None = None

    @property
    def retained_bytes(self):
        return self.pool_bytes + self.static_bytes

    def close(self):
        # The public score API has already copied results to CPU before an
        # entry can be evicted. Never return this graph-owned output directly.
        if self.graph is not None:
            self.graph.reset()
        self.output = None
        self.inputs = None
        self.positions = None
        self.attention_mask = None


class GraphFullScorer(SequenceScorer):
    """Share a native scorer with ``from_scorer`` or load one explicitly.

    Defaults retain at most two independent graph pools and 2 GiB of graph
    pool reservations plus owned static input storage. ``max_pool_bytes`` is
    an admission/retention bound, NOT a transient capture or process-memory
    limit: the first capture must allocate before its pool can be measured.
    Model weights, eager head workspace, and allocator caches are excluded.

    ``fallback='raise'`` fails closed. Explicit ``fallback='eager'`` permits
    known unsupported allocator/pool-budget cases, emits a warning, and counts
    every fallback batch. CUDA capture/replay errors always propagate; a failed
    CUDA context is not presumed safe to retry. Check ``graph_stats`` before
    attributing any timing to graph replay. Statistics never synchronize for
    replay timing: ``replay_enqueue_seconds`` measures CPU enqueue time only;
    ``score_wall_seconds`` includes packing, cold setup, head work and CPU copy.
    ``graphs_enabled=False`` is the matched bulk-packed eager control, including
    the same shape cache, owned input buffers and native 4-D additive masks.
    """

    def __init__(self, model_path, dtype="float32", attention="eager", head_chunk=32,
                 *, max_graphs=2, max_pool_bytes=2 * 1024**3,
                 warmup_iterations=3, fallback="raise", graphs_enabled=True):
        if (dtype, attention, head_chunk) != ("float32", "eager", 32):
            raise ValueError("Graph FULL requires float32, eager attention, and head_chunk=32")
        self._configure_graphs(max_graphs, max_pool_bytes, warmup_iterations, fallback,
                               graphs_enabled)
        super().__init__(model_path, dtype=dtype, attention=attention, head_chunk=head_chunk)
        self._validate_contract()

    @classmethod
    def from_scorer(cls, scorer, *, max_graphs=2, max_pool_bytes=2 * 1024**3,
                    warmup_iterations=3, fallback="raise", graphs_enabled=True):
        """Reuse native model/tokenizer weights; never allocate a second model.

        The original scorer may be used sequentially for reference checks.
        Neither wrapper owns exclusive access to the shared model; the caller
        must serialize that access and leave its weights/modules unchanged.
        """
        if type(scorer) is not SequenceScorer:
            raise TypeError("from_scorer requires an unmodified native SequenceScorer")
        result = cls.__new__(cls)
        for name in ("model", "tokenizer", "head_chunk", "device", "end_token", "pad_token"):
            setattr(result, name, getattr(scorer, name))
        result._configure_graphs(max_graphs, max_pool_bytes, warmup_iterations, fallback,
                                 graphs_enabled)
        result._validate_contract()
        return result

    def _configure_graphs(self, max_graphs, max_pool_bytes, warmup_iterations, fallback,
                          graphs_enabled=True):
        for name, value in (("max_graphs", max_graphs), ("max_pool_bytes", max_pool_bytes),
                            ("warmup_iterations", warmup_iterations)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if fallback not in {"raise", "eager"}:
            raise ValueError("fallback must be 'raise' or explicitly 'eager'")
        if not isinstance(graphs_enabled, bool):
            raise ValueError("graphs_enabled must be a bool")
        self.max_graphs = max_graphs
        self.max_pool_bytes = max_pool_bytes
        self.warmup_iterations = warmup_iterations
        self.fallback = fallback
        self.graphs_enabled = graphs_enabled
        self._graphs = OrderedDict()
        self._unsupported_keys = {}
        self._signature = None
        self._call_lock = threading.Lock()
        self._stats = {
            "scope": "full_backbone_only; original eager head_chunk=32",
            "captures_attempted": 0, "captures_succeeded": 0, "capture_failures": 0,
            "warmup_seconds": 0.0, "capture_seconds": 0.0, "cold_setup_seconds": 0.0,
            "replay_count": 0, "replay_enqueue_seconds": 0.0,
            "cache_hits": 0, "cache_misses": 0, "evictions": 0,
            "fallback_batches": 0, "score_calls": 0, "score_failures": 0,
            "eager_backbone_batches": 0,
            "score_wall_seconds": 0.0, "peak_retained_bytes": 0,
            "capture_records": [],
        }

    def _validate_contract(self):
        if self.head_chunk != 32:
            raise ValueError("Graph FULL requires head_chunk=32")
        config = self.model.config
        if (config.model_type != "qwen2"
                or getattr(config, "use_sliding_window", False)
                or getattr(config, "_attn_implementation", None) != "eager"):
            raise ValueError("Graph FULL requires non-sliding Qwen2 with eager attention")
        rope = getattr(config, "rope_scaling", None)
        if rope and rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("Dynamic/scaled RoPE is outside the graph FULL contract")
        if self.model.training or any(module.training for module in self.model.modules()):
            raise ValueError("Graph FULL requires eval mode on every module")
        params = list(self.model.parameters())
        if not params or any(parameter.dtype != torch.float32 for parameter in params):
            raise ValueError("Graph FULL requires all model parameters in float32")
        device = params[0].device
        if device.type != "cuda" or any(parameter.device != device for parameter in params):
            raise ValueError("Graph FULL requires one CUDA device")
        # Normalize 'cuda' to the actual index without querying another GPU.
        self.device = device
        if (torch.get_float32_matmul_precision() != "highest"
                or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32
                or torch.is_autocast_enabled("cuda")):
            raise ValueError("Graph FULL requires IEEE FP32: highest, TF32 off, autocast off")
        projection_names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                            "up_proj", "down_proj", "lm_head"}
        if any(type(module) is not torch.nn.Linear
               for name, module in self.model.named_modules()
               if name.rsplit(".", 1)[-1] in projection_names):
            raise ValueError("Graph FULL requires unchanged native torch.nn.Linear projections")

    def _execution_signature(self):
        # Metadata only. Catch module replacement, parameter replacement/in-place
        # edits with version counters, and BLAS policy transitions before replay.
        parameters = []
        for name, parameter in self.model.named_parameters():
            try:
                version = parameter._version
            except RuntimeError:  # Inference tensors have no version counter.
                version = None
            parameters.append((name, parameter.data_ptr(), version, parameter.dtype,
                               tuple(parameter.shape), tuple(parameter.stride())))
        return (
            str(torch.backends.cuda.preferred_blas_library()),
            torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32, torch.is_autocast_enabled("cuda"),
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            self.head_chunk, self.model.config._attn_implementation,
            self.graphs_enabled,
            tuple((name, id(module), type(module)) for name, module in self.model.named_modules()),
            tuple(parameters),
        )

    @property
    def graph_stats(self):
        """JSON-safe snapshot; historical capture/fallback costs survive eviction."""
        result = deepcopy(self._stats)
        result.update({
            "max_graphs": self.max_graphs, "max_pool_bytes": self.max_pool_bytes,
            "warmup_iterations": self.warmup_iterations, "fallback_policy": self.fallback,
            "graphs_enabled": self.graphs_enabled,
            "cached_graphs": sum(entry.graph is not None for entry in self._graphs.values()),
            "cached_shapes": len(self._graphs),
            "cached_keys": [list(key) for key in self._graphs],
            "retained_pool_bytes": sum(entry.pool_bytes for entry in self._graphs.values()),
            "retained_static_bytes": sum(entry.static_bytes for entry in self._graphs.values()),
            "retained_bytes": self._retained_bytes(),
            "unsupported_keys": [{"key": list(key), "reason": reason}
                                 for key, reason in self._unsupported_keys.items()],
        })
        return result

    def _retained_bytes(self):
        return sum(entry.retained_bytes for entry in self._graphs.values())

    def _evict_oldest(self):
        _, entry = self._graphs.popitem(last=False)
        entry.close()
        self._stats["evictions"] += 1

    def clear_graph_cache(self):
        """Release graphs and reset the policy binding; retain all cost statistics."""
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("Cannot clear graph cache during scoring")
        try:
            if self._graphs:
                torch.cuda.synchronize(self.device)
            while self._graphs:
                self._evict_oldest()
            self._unsupported_keys.clear()
            self._signature = None
        finally:
            self._call_lock.release()

    def _native_attention_mask(self, inputs, positions):
        # Transformers 5.15 eager_mask creates CUDA scalar constants from the
        # CPU, which cannot occur during capture. Build its exact native mask
        # outside capture on EVERY batch, in both enabled and disabled paths.
        # The pinned create_causal_mask uses inputs_embeds only for its first
        # two dimensions, dtype and device here; no embedding values are read.
        metadata = torch.empty((*inputs.shape[1:], 1), dtype=torch.float32, device=self.device)
        mask = create_causal_mask(
            config=self.model.config, inputs_embeds=metadata, attention_mask=inputs[1],
            past_key_values=None, position_ids=positions,
        )
        batch, width = inputs.shape[1:]
        if (not isinstance(mask, torch.Tensor) or mask.shape != (batch, 1, width, width)
                or mask.dtype != torch.float32 or mask.device != inputs.device):
            raise GraphUnavailable("Pinned native eager mask did not produce a FP32 [B,1,W,W] tensor")
        return mask

    def _backbone(self, inputs, positions, attention_mask):
        return self.model.model(
            input_ids=inputs[0], attention_mask=attention_mask, position_ids=positions,
            past_key_values=None, use_cache=False,
        ).last_hidden_state

    def _pool_reserved_bytes(self, graph):
        # The pinned Torch 2.13 native allocator exposes private-pool segments.
        # Fail closed rather than silently estimating from process-wide deltas.
        pool = tuple(graph.pool())
        segments = torch.cuda.memory_snapshot(mempool_id=pool, include_traces=False)
        if not segments or any(tuple(segment.get("segment_pool_id", ())) != pool
                               for segment in segments):
            raise GraphUnavailable("Cannot account for this graph's private memory pool")
        reserved = sum(int(segment["total_size"]) for segment in segments)
        if reserved <= 0:
            raise GraphUnavailable("Graph private-pool reservation was not reported")
        return reserved

    def _capture_entry(self, packed):
        if torch.cuda.get_allocator_backend() != "native":
            raise GraphUnavailable("Bounded graph pools require the native CUDA allocator")
        record = {"key": list(packed.key), "status": "started",
                  "warmup_iterations": self.warmup_iterations,
                  "warmup_seconds": 0.0, "capture_seconds": 0.0}
        self._stats["captures_attempted"] += 1
        self._stats["capture_records"].append(record)
        started = time.perf_counter()
        graph, output, entry, inputs, positions, attention_mask = None, None, None, None, None, None
        try:
            inputs = packed.inputs.to(self.device)
            positions = torch.arange(packed.key[1], device=self.device).expand(packed.key[0], -1)
            attention_mask = self._native_attention_mask(inputs, positions)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            warm_started = time.perf_counter()
            record["warmup_iterations_completed"] = 0
            try:
                with torch.cuda.stream(stream):
                    for _ in range(self.warmup_iterations):
                        warm_output = self._backbone(inputs, positions, attention_mask)
                        del warm_output
                        record["warmup_iterations_completed"] += 1
                stream.synchronize()
            finally:
                record["warmup_seconds"] = time.perf_counter() - warm_started
                self._stats["warmup_seconds"] += record["warmup_seconds"]
            graph = torch.cuda.CUDAGraph()
            capture_started = time.perf_counter()
            try:
                with torch.cuda.graph(graph, stream=stream, capture_error_mode="global"):
                    output = self._backbone(inputs, positions, attention_mask)
            finally:
                record["capture_seconds"] = time.perf_counter() - capture_started
                self._stats["capture_seconds"] += record["capture_seconds"]
            pool_bytes = self._pool_reserved_bytes(graph)
            static_bytes = (inputs.untyped_storage().nbytes() + positions.untyped_storage().nbytes()
                            + attention_mask.untyped_storage().nbytes())
            entry = _GraphEntry(graph, inputs, positions, output, pool_bytes, static_bytes, attention_mask)
            record.update(pool_bytes=pool_bytes, static_bytes=static_bytes)
            if entry.retained_bytes > self.max_pool_bytes:
                raise GraphUnavailable(
                    f"Captured shape {packed.key} needs {entry.retained_bytes} retained bytes; "
                    f"limit is {self.max_pool_bytes}"
                )
            self._stats["captures_succeeded"] += 1
            record["status"] = "captured"
            return entry
        except Exception as exc:
            self._stats["capture_failures"] += 1
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if entry is not None:
                entry.close()
            elif graph is not None:
                graph.reset()
            output, entry, inputs, positions, attention_mask = None, None, None, None, None
            raise
        finally:
            record["cold_setup_seconds"] = time.perf_counter() - started
            self._stats["cold_setup_seconds"] += record["cold_setup_seconds"]

    def _eager_entry(self, packed):
        # Matched scheduling control: identical owned buffers, shape cache,
        # padding/mask, positions, hidden gather, and head; no graph or warmup.
        inputs = packed.inputs.to(self.device)
        positions = torch.arange(packed.key[1], device=self.device).expand(packed.key[0], -1)
        attention_mask = self._native_attention_mask(inputs, positions)
        static_bytes = (inputs.untyped_storage().nbytes() + positions.untyped_storage().nbytes()
                        + attention_mask.untyped_storage().nbytes())
        if static_bytes > self.max_pool_bytes:
            raise GraphUnavailable("Owned static inputs exceed the retained-byte limit")
        return _GraphEntry(None, inputs, positions, None, 0, static_bytes, attention_mask)

    def _entry_for(self, packed):
        key = packed.key
        if key in self._unsupported_keys:
            raise GraphUnavailable(self._unsupported_keys[key])
        if key in self._graphs:
            self._stats["cache_hits"] += 1
            self._graphs.move_to_end(key)
            entry = self._graphs[key]
            entry.inputs.copy_(packed.inputs, non_blocking=True)
            entry.attention_mask.copy_(self._native_attention_mask(entry.inputs, entry.positions))
            return entry
        self._stats["cache_misses"] += 1
        while len(self._graphs) >= self.max_graphs:
            self._evict_oldest()
        try:
            entry = self._capture_entry(packed) if self.graphs_enabled else self._eager_entry(packed)
        except GraphUnavailable as exc:
            self._unsupported_keys[key] = str(exc)
            raise
        while self._graphs and self._retained_bytes() + entry.retained_bytes > self.max_pool_bytes:
            self._evict_oldest()
        self._graphs[key] = entry
        self._stats["peak_retained_bytes"] = max(
            self._stats["peak_retained_bytes"], self._retained_bytes()
        )
        return entry

    def _run_backbone(self, packed):
        try:
            entry = self._entry_for(packed)
        except GraphUnavailable as exc:
            if self.fallback != "eager":
                raise
            self._stats["fallback_batches"] += 1
            warnings.warn(f"Graph FULL explicit eager fallback: {exc}", RuntimeWarning, stacklevel=2)
            inputs = packed.inputs.to(self.device)
            positions = torch.arange(packed.key[1], device=self.device).expand(packed.key[0], -1)
            return self._backbone(inputs, positions, self._native_attention_mask(inputs, positions))
        started = time.perf_counter()
        if entry.graph is None:
            self._stats["eager_backbone_batches"] += 1
            return self._backbone(entry.inputs, entry.positions, entry.attention_mask)
        entry.graph.replay()
        self._stats["replay_enqueue_seconds"] += time.perf_counter() - started
        self._stats["replay_count"] += 1
        return entry.output

    @torch.inference_mode()
    def score(self, prompt, sequences, method="full", reducer="torch_ce", batch_size=8):
        if method != "full":
            raise ValueError("Graph FULL implements only independent full-prompt scoring")
        if reducer not in {"torch_ce", "triton"}:
            raise ValueError("Select an explicit full-vocabulary torch_ce or triton normalizer")
        if not prompt or not sequences or any(not sequence for sequence in sequences):
            raise ValueError("Nonempty prompt and suffix paths are required")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("Graph FULL scorer cannot be called concurrently")
        started = time.perf_counter()
        self._stats["score_calls"] += 1
        try:
            self._validate_contract()
            signature = self._execution_signature()
            if self._signature is not None and signature != self._signature:
                raise RuntimeError("Model/BLAS policy changed; clear_graph_cache before recapture")
            self._signature = signature
            all_values, processed_queries, processed_prompts = [], 0, 0
            p = len(prompt)
            with torch.cuda.device(self.device):
                for begin in range(0, len(sequences), batch_size):
                    packed = _pack_group(prompt, sequences[begin:begin + batch_size], self.pad_token)
                    output = self._run_backbone(packed)
                    # Match the original row order, concatenation and each 32-row
                    # (including final partial) head GEMM exactly in shape.
                    hidden = torch.cat([output[row, p - 1:p + count - 1]
                                        for row, count in enumerate(packed.counts)])
                    targets = packed.targets.to(self.device, non_blocking=True)
                    values = self._project(hidden, targets, reducer=reducer).cpu().tolist()
                    offset = 0
                    for count in packed.counts:
                        all_values.append(values[offset:offset + count])
                        offset += count
                    processed_queries += len(packed.counts) * packed.suffix_width
                    processed_prompts += len(packed.counts) * p
                    # Do not keep an evicted graph pool alive through a previous
                    # batch's output view while constructing the next capture.
                    del output, hidden, targets
            self._check(all_values)
            edges = sum(map(len, sequences))
            return Scores([float(sum(values)) for values in all_values], all_values, p,
                          edges, edges, "full", processed_prompts, processed_queries)
        except Exception:
            self._stats["score_failures"] += 1
            raise
        finally:
            self._stats["score_wall_seconds"] += time.perf_counter() - started
            self._call_lock.release()
