"""Run numerical and CUDA checks on the authorized DGX Spark only."""

import pytest
import torch

from orbitcall.kernels import selected_logprobs, selected_logprobs_cross_entropy


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/Triton checks require the authorized Spark GPU")
    pytest.importorskip("triton")
    return torch.device("cuda")


@pytest.mark.parametrize("vocab_size", [1, 7, 31, 1025, 4095, 4096, 4097, 32003, 151936])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tiled_full_vocabulary_matches_log_softmax(vocab_size, dtype):
    device = _cuda_or_skip()
    generator = torch.Generator(device=device).manual_seed(173 + vocab_size)
    with torch.inference_mode():
        logits = torch.randn(17, vocab_size, device=device, generator=generator).to(dtype)
        targets = torch.randint(vocab_size, (17,), device=device, generator=generator)
        expected = torch.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None])[:, 0]
        actual = selected_logprobs(logits, targets, backend="triton")
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
        assert actual.dtype == torch.float32 and actual.shape == (17,)


@pytest.mark.parametrize("backend", ["torch", "torch_lse", "torch_ce", "auto"])
def test_torch_controls_match_distinct_row_cross_entropy(backend):
    with torch.inference_mode():
        generator = torch.Generator().manual_seed(291)
        logits = torch.randn(13, 521, generator=generator) * 17
        targets = torch.randint(521, (13,), generator=generator)
        expected = selected_logprobs_cross_entropy(logits, targets)
        actual = selected_logprobs(logits, targets, backend=backend)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)


@pytest.mark.parametrize("backend", ["torch", "torch_lse", "triton"])
def test_scrambled_repeated_parent_rows_and_strides(backend):
    device = _cuda_or_skip() if backend == "triton" else torch.device("cpu")
    with torch.inference_mode():
        generator = torch.Generator(device=device).manual_seed(239)
        # Noncontiguous row and vocabulary strides, plus strided index arrays.
        storage = torch.randn(10, 8206, device=device, generator=generator)
        logits = storage[::2, ::2]
        row_storage = torch.tensor([4, 99, 1, 99, 4, 99, 0, 99, 2, 99, 1, 99], device=device)
        target_storage = torch.tensor(
            [4102, 99, 0, 99, 4088, 99, 17, 99, 32, 99, 73, 99], device=device
        )
        rows, targets = row_storage[::2], target_storage[::2]
        expected = torch.log_softmax(logits.float(), dim=-1)[rows, targets]
        actual = selected_logprobs(logits, targets, rows, backend=backend)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
        assert actual.shape == (6,)


@pytest.mark.parametrize("backend", ["torch", "torch_lse", "triton"])
def test_duplicate_targets_do_not_change_full_vocabulary_denominator(backend):
    device = _cuda_or_skip() if backend == "triton" else torch.device("cpu")
    with torch.inference_mode():
        logits = torch.zeros(2, 4097, device=device)
        # High probability belongs to an unrequested token in another tile.
        logits[0, 4096] = 20
        rows = torch.tensor([0, 0, 1, 0], device=device)
        targets = torch.tensor([0, 0, 0, 4096], device=device)
        expected = torch.log_softmax(logits, dim=-1)[rows, targets]
        actual = selected_logprobs(logits, targets, rows, backend=backend)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
        assert actual[0] < -19  # A requested-only softmax would incorrectly give 0.


@pytest.mark.parametrize("backend", ["torch", "torch_lse", "torch_ce", "triton"])
def test_extreme_logits_and_large_common_offset(backend):
    device = _cuda_or_skip() if backend == "triton" else torch.device("cpu")
    with torch.inference_mode():
        logits = torch.full((4, 4097), -10000.0, device=device)
        logits[0, 4096] = 10000
        logits[1] = 1e20  # Correct logprob is -log(V), not zero.
        logits[2, 2] = 10000
        logits[2, 4096] = 10000
        logits[3] = torch.linspace(-100, 100, 4097, device=device)
        targets = torch.tensor([0, 10, 2, 4096], device=device)
        expected = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None])[:, 0]
        actual = selected_logprobs(logits, targets, backend=backend)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=3e-5)


@pytest.mark.parametrize("backend", ["torch", "torch_lse", "torch_ce", "triton"])
def test_infinity_nan_and_invalid_target_propagation(backend):
    device = _cuda_or_skip() if backend == "triton" else torch.device("cpu")
    with torch.inference_mode():
        logits = torch.zeros(8, 4097, device=device)
        logits[0, 4096] = float("nan")
        logits[1, 4096] = float("inf")
        logits[2] = -float("inf")
        logits[3, 1] = -float("inf")
        # A complete -inf tile must contribute zero, not contaminate the row.
        logits[4, :4096] = -float("inf")
        logits[5, 4096] = -float("inf")
        targets = torch.tensor([0, 0, 0, 1, 4096, 0, -1, 4097], device=device)
        expected = torch.log_softmax(logits, dim=-1).gather(
            1, targets.clamp(0, 4096)[:, None]
        )[:, 0]
        expected[6:] = float("nan")
        actual = selected_logprobs(logits, targets, backend=backend)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5, equal_nan=True)
        assert torch.isnan(actual[[0, 1, 2, 6, 7]]).all()
        assert torch.isneginf(actual[3])
        assert actual[4] == 0


@pytest.mark.parametrize("backend", ["torch", "torch_lse", "triton"])
def test_invalid_parent_rows_are_nan_without_out_of_bounds_loads(backend):
    device = _cuda_or_skip() if backend == "triton" else torch.device("cpu")
    with torch.inference_mode():
        logits = torch.zeros(2, 31, device=device)
        targets = torch.tensor([0, 0, 0], device=device)
        rows = torch.tensor([-1, 2, 1], device=device)
        values = selected_logprobs(logits, targets, rows, backend=backend)
        assert torch.isnan(values[:2]).all()
        torch.testing.assert_close(values[2], torch.tensor(31.0, device=device).log().neg())


@pytest.mark.parametrize("backend", ["torch", "torch_lse", "torch_ce", "auto"])
def test_empty_distinct_row_input(backend):
    with torch.inference_mode():
        actual = selected_logprobs(
            torch.empty(0, 7), torch.empty(0, dtype=torch.int64), backend=backend
        )
        assert actual.shape == (0,) and actual.dtype == torch.float32


def test_metadata_validation_and_explicit_backend_contract():
    with torch.inference_mode():
        logits, targets = torch.zeros(2, 5), torch.zeros(2, dtype=torch.int64)
        with pytest.raises(ValueError, match="shape"):
            selected_logprobs(logits[0], targets)
        with pytest.raises(ValueError, match="nonempty"):
            selected_logprobs(logits[:, :0], targets)
        with pytest.raises(TypeError, match="int64"):
            selected_logprobs(logits, targets.to(torch.int32))
        with pytest.raises(ValueError, match="shape"):
            selected_logprobs(logits, targets[:1])
        with pytest.raises(ValueError, match="same shape"):
            selected_logprobs(logits, targets, targets[:1])
        with pytest.raises(ValueError, match="distinct"):
            selected_logprobs(logits, targets, targets, backend="torch_ce")
        with pytest.raises(ValueError, match="unrecognized"):
            selected_logprobs(logits, targets, backend="unknown")
        with pytest.raises(RuntimeError, match="CUDA"):
            selected_logprobs(logits, targets, backend="triton")
