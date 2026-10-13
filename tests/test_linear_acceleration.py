"""Run all numerical tests on Spark; CUDA kernels are explicit opt-in only."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from orbitcall.linear_acceleration import ExperimentalLinear, accelerated_linears, fp32_linear


def cuda():
    if not torch.cuda.is_available():
        pytest.skip("Experimental Tensor-Core tests require the authorized Spark GPU")
    pytest.importorskip("triton")
    return torch.device("cuda")


@pytest.mark.parametrize("precision", ["tf32x3", "tf32x6"])
@pytest.mark.parametrize("shape,n", [((1, 7), 3), ((35, 67), 33), ((2, 3, 95), 65), ((17, 129), 129)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_linear_tails_flatten_bias_against_fp64(shape, n, has_bias, precision):
    device = cuda()
    generator = torch.Generator(device=device).manual_seed(197)
    with torch.inference_mode():
        input = torch.randn(shape, device=device, generator=generator) / (shape[-1] ** 0.5)
        weight = torch.randn(n, shape[-1], device=device, generator=generator)
        bias = torch.randn(n, device=device, generator=generator) if has_bias else None
        expected = torch.nn.functional.linear(input.double(), weight.double(), None if bias is None else bias.double()).float()
        actual = fp32_linear(input, weight, bias, backend="triton", precision=precision)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=3e-6)
        assert actual.dtype == torch.float32 and actual.shape == (*shape[:-1], n)


@pytest.mark.parametrize("precision", ["tf32x3", "tf32x6"])
def test_near_zero_cancellation_has_absolute_error_gate(precision):
    device = cuda()
    with torch.inference_mode():
        # Residual bits below the first TF32 component, cancelling high products.
        input = torch.tensor([[1.0 + 2 ** -14, 1.0 - 2 ** -14, 1.0 + 2 ** -23, 1.0]], device=device)
        weight = torch.tensor([[1.0, -1.0, 1.0, -1.0],
                               [1.0 + 2 ** -14, -1.0 + 2 ** -14, -1.0, 1.0]], device=device)
        expected = torch.nn.functional.linear(input.double(), weight.double()).float()
        actual = fp32_linear(input, weight, backend="triton", precision=precision)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=0)


@pytest.mark.parametrize("precision", ["tf32x3", "tf32x6"])
def test_empty_rows_and_zero_reduction_match_reference(precision):
    device = cuda()
    with torch.inference_mode():
        assert fp32_linear(torch.empty(0, 7, device=device), torch.empty(3, 7, device=device), backend="triton", precision=precision).shape == (0, 3)
        input = torch.empty(2, 3, 0, device=device)
        weight, bias = torch.empty(5, 0, device=device), torch.arange(5, device=device, dtype=torch.float32)
        actual = fp32_linear(input, weight, bias, backend="triton", precision=precision)
        torch.testing.assert_close(actual, bias.expand(2, 3, 5), atol=0, rtol=0)


def test_reference_default_keeps_native_noncontiguous_behavior():
    input = torch.arange(24, dtype=torch.float32).reshape(3, 8)[:, ::2]
    weight = torch.arange(20, dtype=torch.float32).reshape(4, 5).t()
    torch.testing.assert_close(fp32_linear(input, weight), torch.nn.functional.linear(input, weight), atol=0, rtol=0)


def test_explicit_backend_metadata_errors():
    input, weight = torch.ones(2, 3), torch.ones(4, 3)
    with pytest.raises(TypeError):
        fp32_linear(input.double(), weight)
    with pytest.raises(ValueError):
        fp32_linear(input, weight, torch.zeros(5))
    with pytest.raises(ValueError):
        fp32_linear(input, weight, precision="tf32")
    with pytest.raises(RuntimeError):
        fp32_linear(input, weight, backend="triton")
    with pytest.raises(TypeError):
        fp32_linear(input, weight, invented_argument=True)


@pytest.mark.parametrize("which", ["input", "weight", "bias"])
def test_triton_noncontiguous_inputs_fail_without_copies(which):
    device = cuda()
    with torch.inference_mode():
        input, weight, bias = torch.ones(3, 4, device=device), torch.ones(5, 4, device=device), torch.ones(5, device=device)
        if which == "input":
            input = torch.ones(3, 8, device=device)[:, ::2]
        elif which == "weight":
            weight = torch.ones(5, 8, device=device)[:, ::2]
        else:
            bias = torch.ones(10, device=device)[::2]
        with pytest.raises(ValueError, match="contiguous"):
            fp32_linear(input, weight, bias, backend="triton")


def test_device_mismatch_and_grad_mode_fail_closed():
    device = cuda()
    with pytest.raises(ValueError, match="devices"):
        fp32_linear(torch.ones(2, 3, device=device), torch.ones(4, 3), backend="triton")
    with pytest.raises(ValueError, match="inference"):
        fp32_linear(torch.ones(2, 3, device=device, requires_grad=True), torch.ones(4, 3, device=device), backend="triton")


class ToyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen2", use_sliding_window=False)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(5, 4)
        self.model.proj = nn.Linear(4, 4)
        self.model.alias = self.model.proj
        self.lm_head = nn.Linear(4, 5, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.unrelated = nn.Linear(4, 4)
        self.eval()


@pytest.mark.parametrize("scope", ["head", "backbone", "all"])
def test_retrofit_scope_tied_parameters_and_exception_restoration(scope):
    model = ToyQwen()
    original_proj, original_head, original_embedding = model.model.proj, model.lm_head, model.model.embed_tokens
    unrelated, input = model.unrelated, torch.randn(2, 4)
    expected = original_head(input)
    with pytest.raises(RuntimeError, match="deliberate"):
        with accelerated_linears(model, scope=scope, backend="torch") as names:
            assert model.model.embed_tokens is original_embedding and model.unrelated is unrelated
            assert model.lm_head.weight is original_embedding.weight
            if scope in ("backbone", "all"):
                assert isinstance(model.model.proj, ExperimentalLinear)
                assert model.model.proj is model.model.alias and model.model.proj.weight is original_proj.weight
            if scope in ("head", "all"):
                torch.testing.assert_close(model.lm_head(input=input), expected, atol=0, rtol=0)
                assert "lm_head" in names
                with pytest.raises(TypeError):
                    model.lm_head(input, unsupported_keyword=True)
            raise RuntimeError("deliberate restoration test")
    assert model.model.proj is original_proj and model.model.alias is original_proj
    assert model.lm_head is original_head and model.model.embed_tokens is original_embedding


def test_retrofit_rejects_unsupported_model_before_mutation():
    model = ToyQwen()
    original = model.model.proj
    model.config.model_type = "other"
    with pytest.raises(ValueError, match="Qwen2"):
        with accelerated_linears(model, backend="torch"):
            pytest.fail("Unsupported architecture was retrofitted")
    assert model.model.proj is original
