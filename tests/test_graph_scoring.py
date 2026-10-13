"""Run all checks, including CPU logic and tiny-model inference, on Spark only.

No checkpoint download is needed. CPU fakes test cache ownership/scheduling;
the CUDA tests construct a small random Qwen2 and exercise actual capture.
"""

from contextlib import nullcontext
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from orbitcall.graph_scoring import (
    GraphFullScorer, GraphUnavailable, _GraphEntry, _pack_group,
)
from orbitcall.scoring import SequenceScorer


def test_bulk_pack_preserves_mask_positions_targets_and_input_ownership():
    prompt, sequences = [11, 12], [[21, 22, 2], [31, 2], [2]]
    packed = _pack_group(prompt, sequences, 0)
    assert packed.key == (3, 4)
    assert packed.inputs.tolist() == [
        [[11, 12, 21, 22], [11, 12, 31, 0], [11, 12, 0, 0]],
        [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]],
    ]
    assert packed.targets.tolist() == [21, 22, 2, 31, 2, 2]
    assert packed.counts == (3, 2, 1)
    prompt[0], sequences[0][0] = 99, 98
    assert packed.inputs[0, 0, 0].item() == 11
    assert packed.targets[0].item() == 21


@pytest.mark.parametrize("kwargs", [
    {"max_graphs": 0}, {"max_graphs": True}, {"max_pool_bytes": 0},
    {"warmup_iterations": 0}, {"fallback": "auto"}, {"graphs_enabled": 1},
])
def test_invalid_cache_configuration_rejected_without_loading_model(kwargs):
    settings = dict(max_graphs=2, max_pool_bytes=4096, warmup_iterations=1, fallback="raise")
    settings.update(kwargs)
    scorer = GraphFullScorer.__new__(GraphFullScorer)
    with pytest.raises(ValueError):
        scorer._configure_graphs(**settings)


def test_from_scorer_shares_model_but_owns_independent_cache(monkeypatch):
    native = SequenceScorer.__new__(SequenceScorer)
    for name in ("model", "tokenizer", "head_chunk", "device", "end_token", "pad_token"):
        setattr(native, name, object())
    monkeypatch.setattr(GraphFullScorer, "_validate_contract", lambda self: None)
    graph = GraphFullScorer.from_scorer(native)
    other = GraphFullScorer.from_scorer(native, graphs_enabled=False)
    assert graph.model is native.model and graph.tokenizer is native.tokenizer
    assert graph._graphs is not other._graphs
    assert graph.graph_stats["graphs_enabled"] is True
    assert other.graph_stats["graphs_enabled"] is False
    with pytest.raises(TypeError, match="native SequenceScorer"):
        GraphFullScorer.from_scorer(graph)


class _FakeGraph:
    def __init__(self, inputs, positions, output, closed, attention_mask):
        self.inputs, self.positions, self.output, self.closed = inputs, positions, output, closed
        self.attention_mask = attention_mask

    def replay(self):
        self.output.copy_(_fake_backbone(self.inputs, self.positions, self.attention_mask))

    def reset(self):
        self.closed.append(id(self))


def _fake_mask(inputs, positions):
    width = inputs.shape[-1]
    allowed = torch.arange(width)[None, :] <= torch.arange(width)[:, None]
    allowed = allowed[None, None] & inputs[1][:, None, None, :].bool()
    return torch.where(allowed, 0.0, torch.finfo(torch.float32).min)


def _fake_backbone(inputs, positions, attention_mask):
    # Sensitive to token IDs, changing masks, and position IDs; deterministic
    # CPU scheduling fixture, not a language-model numerical reference.
    valid_diagonal = attention_mask[:, 0].diagonal(dim1=-2, dim2=-1).eq(0)
    return (inputs[0].float() + 100 * valid_diagonal + positions).unsqueeze(-1)


@pytest.fixture
def logic_scorer(monkeypatch):
    scorer = GraphFullScorer.__new__(GraphFullScorer)
    scorer._configure_graphs(2, 4096, 1, "raise")
    scorer.device, scorer.pad_token, scorer.head_chunk = torch.device("cpu"), 0, 32
    scorer._captured_keys, scorer._closed = [], []
    monkeypatch.setattr(scorer, "_validate_contract", lambda: None)
    monkeypatch.setattr(scorer, "_execution_signature", lambda: ("fixture", scorer.graphs_enabled))
    monkeypatch.setattr(scorer, "_backbone", _fake_backbone)
    monkeypatch.setattr(scorer, "_native_attention_mask", _fake_mask)
    monkeypatch.setattr(scorer, "_project",
                        lambda hidden, targets, reducer: hidden[:, 0] + targets.float() / 1000)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    def capture(packed):
        scorer._captured_keys.append(packed.key)
        inputs = packed.inputs.clone()
        positions = torch.arange(packed.key[1]).expand(packed.key[0], -1)
        output = torch.empty((*packed.key, 1))
        attention_mask = _fake_mask(inputs, positions)
        graph = _FakeGraph(inputs, positions, output, scorer._closed, attention_mask)
        return _GraphEntry(graph, inputs, positions, output, 512, 128, attention_mask)

    monkeypatch.setattr(scorer, "_capture_entry", capture)
    return scorer


def test_replay_replaces_ids_masks_counts_and_preserves_returned_scores(logic_scorer):
    scorer = logic_scorer
    first = scorer.score([11, 12], [[21, 22, 2], [31, 2]])
    saved = deepcopy(first)
    second = scorer.score([41, 42], [[51, 2], [61, 62, 2]])
    assert scorer._captured_keys == [(2, 4)]
    assert scorer.graph_stats["cache_hits"] == 1
    assert scorer.graph_stats["replay_count"] == 2
    assert first == saved
    assert first.token_logprobs != second.token_logprobs
    assert list(map(len, second.token_logprobs)) == [2, 3]
    # The first suffix target is predicted from the final prompt position.
    assert second.token_logprobs[0][0] == pytest.approx(143.051, abs=2e-5)
    # Change prompt length but keep exact backbone shape: gather is not frozen.
    third = scorer.score([7], [[8, 9, 10, 2], [2]])
    assert scorer._captured_keys == [(2, 4)]
    assert third.token_logprobs[0][0] == pytest.approx(107.008, abs=2e-5)
    assert list(map(len, third.token_logprobs)) == [4, 1]


def test_exact_shape_lru_eviction_and_recapture(logic_scorer):
    scorer = logic_scorer
    scorer.score([1, 3], [[4, 2]])  # A: (1, 3)
    scorer.score([1, 3], [[4, 5, 2]])  # B: (1, 4)
    scorer.score([1, 3], [[6, 2]])  # A becomes most recent.
    scorer.score([1, 3], [[4, 5, 6, 2]])  # C: (1, 5); evict B.
    assert scorer.graph_stats["cached_keys"] == [[1, 3], [1, 5]]
    assert len(scorer._closed) == 1
    scorer.score([1, 3], [[4, 5, 2]])
    assert scorer._captured_keys == [(1, 3), (1, 4), (1, 5), (1, 4)]
    assert scorer.graph_stats["cached_graphs"] == 2
    assert scorer.graph_stats["evictions"] == 2
    scorer.clear_graph_cache()
    assert scorer.graph_stats["cached_graphs"] == 0
    assert scorer.graph_stats["replay_count"] == 5  # Clearing retains costs.
    assert len(scorer._closed) == 4


def test_retained_byte_limit_evicts_before_admitting_second_shape(logic_scorer):
    scorer = logic_scorer
    scorer.max_pool_bytes = 1000  # Two 640-byte fake entries would exceed this.
    scorer.score([1], [[2]])
    scorer.score([1], [[3, 2]])
    assert scorer.graph_stats["retained_bytes"] == 640
    assert scorer.graph_stats["cached_graphs"] == 1
    assert scorer.graph_stats["evictions"] == 1
    assert scorer.graph_stats["peak_retained_bytes"] <= scorer.max_pool_bytes


def test_pool_accounting_uses_only_private_pool_segments(logic_scorer, monkeypatch):
    graph = SimpleNamespace(pool=lambda: (3, 9))
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return [{"segment_pool_id": (3, 9), "total_size": 512},
                {"segment_pool_id": (3, 9), "total_size": 1024}]

    monkeypatch.setattr(torch.cuda, "memory_snapshot", snapshot)
    assert logic_scorer._pool_reserved_bytes(graph) == 1536
    assert calls == [{"mempool_id": (3, 9), "include_traces": False}]
    monkeypatch.setattr(torch.cuda, "memory_snapshot",
                        lambda **kwargs: [{"segment_pool_id": (0, 0), "total_size": 999}])
    with pytest.raises(GraphUnavailable, match="Cannot account"):
        logic_scorer._pool_reserved_bytes(graph)


def test_disabled_control_uses_same_buffers_and_eager_backbone(logic_scorer):
    scorer = logic_scorer
    scorer.graphs_enabled = False
    first = scorer.score([11, 12], [[21, 22, 2], [31, 2]])
    buffer = next(iter(scorer._graphs.values())).inputs
    second = scorer.score([41, 42], [[51, 2], [61, 62, 2]])
    assert next(iter(scorer._graphs.values())).inputs is buffer
    assert first.token_logprobs != second.token_logprobs
    assert scorer._captured_keys == []
    stats = scorer.graph_stats
    assert stats["replay_count"] == stats["captures_attempted"] == stats["cached_graphs"] == 0
    assert stats["cached_shapes"] == 1 and stats["eager_backbone_batches"] == 2


def test_explicit_fallback_is_counted_but_cuda_errors_are_not_retried(logic_scorer, monkeypatch):
    scorer = logic_scorer

    def unsupported(packed):
        raise GraphUnavailable("fixture unsupported allocator")

    monkeypatch.setattr(scorer, "_capture_entry", unsupported)
    with pytest.raises(GraphUnavailable):
        scorer.score([1], [[3, 2]])
    scorer.fallback = "eager"
    with pytest.warns(RuntimeWarning, match="explicit eager fallback"):
        scorer.score([1], [[3, 2]])
    assert scorer.graph_stats["fallback_batches"] == 1
    assert scorer.graph_stats["replay_count"] == 0
    scorer.clear_graph_cache()

    def broken_cuda(packed):
        raise RuntimeError("fixture capture runtime failure")

    monkeypatch.setattr(scorer, "_capture_entry", broken_cuda)
    with pytest.raises(RuntimeError, match="capture runtime failure"):
        scorer.score([1], [[3, 2]])
    assert scorer.graph_stats["fallback_batches"] == 1


def test_policy_changes_require_cache_clear_and_stats_are_owned(logic_scorer, monkeypatch):
    scorer = logic_scorer
    scorer.score([1], [[3, 2]])
    monkeypatch.setattr(scorer, "_execution_signature", lambda: ("changed BLAS policy",))
    with pytest.raises(RuntimeError, match="clear_graph_cache"):
        scorer.score([1], [[3, 2]])
    assert scorer.graph_stats["replay_count"] == 1
    scorer.clear_graph_cache()
    scorer.score([1], [[3, 2]])
    stats = scorer.graph_stats
    json.dumps(stats, allow_nan=False)
    stats["cached_keys"].clear()
    assert scorer.graph_stats["cached_keys"] == [[1, 2]]


@pytest.mark.parametrize("kwargs", [
    {"method": "cached"}, {"method": "tree"}, {"reducer": "auto"},
    {"reducer": "torch"}, {"batch_size": 0}, {"batch_size": True},
])
def test_unsupported_call_contract_rejected(logic_scorer, kwargs):
    with pytest.raises(ValueError):
        logic_scorer.score([1], [[2]], **kwargs)
    assert not logic_scorer._captured_keys


def test_head_guard_precedes_model_execution():
    scorer = GraphFullScorer.__new__(GraphFullScorer)
    scorer.head_chunk = 64
    with pytest.raises(ValueError, match="head_chunk=32"):
        scorer._validate_contract()


@pytest.fixture
def tiny_native_scorer():
    if not torch.cuda.is_available():
        pytest.skip("Actual graph checks require the authorized Spark GPU")
    from transformers import AutoModelForCausalLM, Qwen2Config

    device = torch.device("cuda", torch.cuda.current_device())
    old_precision = torch.get_float32_matmul_precision()
    old_tf32, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(417)
            config = Qwen2Config(vocab_size=67, hidden_size=32, intermediate_size=48,
                                 num_hidden_layers=2, num_attention_heads=4,
                                 num_key_value_heads=2, max_position_embeddings=128,
                                 attention_dropout=0.0, use_sliding_window=False,
                                 eos_token_id=2, pad_token_id=0, tie_word_embeddings=False)
            model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
            model = model.to(device=device, dtype=torch.float32).eval()
        native = SequenceScorer.__new__(SequenceScorer)
        native.model, native.tokenizer, native.device = model, None, device
        native.head_chunk, native.end_token, native.pad_token = 32, 2, 0
        yield native
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn


@pytest.mark.parametrize("reducer", ["torch_ce", "triton"])
def test_actual_qwen_graph_replay_matches_native_and_disabled_control(tiny_native_scorer, reducer):
    if reducer == "triton":
        pytest.importorskip("triton")
    native = tiny_native_scorer
    graph = GraphFullScorer.from_scorer(native, warmup_iterations=1)
    control = GraphFullScorer.from_scorer(native, warmup_iterations=1, graphs_enabled=False)
    calls = [
        ([3, 4, 5], [list(range(6, 40)) + [2], [11, 2]]),
        ([6, 7, 8], [[15, 2], list(range(10, 44)) + [2]]),
        ([9, 10], [list(range(11, 46)) + [2], [2]]),
    ]
    head_shapes = []
    hook = native.model.lm_head.register_forward_pre_hook(
        lambda module, inputs: head_shapes.append(tuple(inputs[0].shape))
    )
    previous, previous_copy = None, None
    try:
        for prompt, sequences in calls:
            head_shapes.clear()
            expected = native.score(prompt, sequences, method="full", reducer=reducer, batch_size=2)
            native_shapes = list(head_shapes)
            head_shapes.clear()
            actual = graph.score(prompt, sequences, reducer=reducer, batch_size=2)
            assert head_shapes == native_shapes
            assert any(shape[0] == 32 for shape in head_shapes)
            eager = control.score(prompt, sequences, reducer=reducer, batch_size=2)
            for reference, captured, disabled in zip(expected.token_logprobs, actual.token_logprobs,
                                                     eager.token_logprobs, strict=True):
                torch.testing.assert_close(torch.tensor(captured), torch.tensor(reference),
                                           rtol=0, atol=3e-5)
                torch.testing.assert_close(torch.tensor(disabled), torch.tensor(reference),
                                           rtol=0, atol=3e-5)
            assert (actual.prompt_tokens, actual.independent_suffix_tokens, actual.scored_edges,
                    actual.processed_prompt_positions, actual.processed_query_positions) == (
                        expected.prompt_tokens, expected.independent_suffix_tokens, expected.scored_edges,
                        expected.processed_prompt_positions, expected.processed_query_positions)
            if previous is not None:
                assert previous == previous_copy
            previous, previous_copy = actual, deepcopy(actual)
        stats = graph.graph_stats
        assert stats["captures_succeeded"] == 1 and stats["replay_count"] == 3
        assert stats["fallback_batches"] == stats["capture_failures"] == 0
        assert stats["retained_pool_bytes"] > 0
        assert 0 < stats["retained_bytes"] <= stats["max_pool_bytes"]
        assert stats["cold_setup_seconds"] >= stats["capture_seconds"] > 0
    finally:
        hook.remove()
        graph.clear_graph_cache()
        control.clear_graph_cache()


def test_actual_contract_rejects_tf32_and_changed_head(tiny_native_scorer):
    native = tiny_native_scorer
    native.head_chunk = 64
    with pytest.raises(ValueError, match="head_chunk=32"):
        GraphFullScorer.from_scorer(native)
    native.head_chunk = 32
    torch.backends.cuda.matmul.allow_tf32 = True
    with pytest.raises(ValueError, match="IEEE FP32"):
        GraphFullScorer.from_scorer(native)
