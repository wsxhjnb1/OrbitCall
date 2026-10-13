"""Tests of the prospective error gate, without model inference."""

import copy
import pytest
import torch
from orbitcall.approximation import (TORCH_CONTROL, precision_context, evaluate_accuracy,
                                    summarize_accuracy)


def result(values, winner="a"):
    return {"scores": {"method": "full", "token_logprobs": [values],
                       "sequence_logprobs": [sum(values)]},
            "predictions": {"mass": {"semantic_key": winner, "margin": 0.01}}}


def test_small_error_valid():
    check = evaluate_accuracy(result([-1., -2.]), result([-1.0001, -1.9999]))
    assert summarize_accuracy([check])["valid"]


def test_mean_budget_not_max_only():
    check = evaluate_accuracy(result([-1., -2.]), result([-1.002, -2.002]))
    assert check["numerical_valid"]
    assert not summarize_accuracy([check])["valid"]


def test_token_max_failure():
    check = evaluate_accuracy(result([-1.]), result([-1.02]))
    assert not check["numerical_valid"]


def test_sequence_bound_independent():
    check = evaluate_accuracy(result([-1.] * 100), result([-1.002] * 100))
    assert check["maximum_token_error"] < .01
    assert not check["numerical_valid"]


def test_population_permits_one_changed_winner_per100():
    good = evaluate_accuracy(result([-1.]), result([-1.]))
    changed = evaluate_accuracy(result([-1.]), result([-1.], "b"))
    checks = [good] * 99 + [changed]
    assert summarize_accuracy(checks)["valid"]
    assert not summarize_accuracy(checks, require_all_winners=True)["valid"]
    assert not summarize_accuracy([good] * 98 + [changed])["valid"]


def test_empty_denominator_rejected():
    with pytest.raises(ValueError):
        summarize_accuracy([])


def test_torch_policy_restores_after_exception():
    precision = torch.get_float32_matmul_precision()
    cuda, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    raw = (torch.backends.cuda.matmul.fp32_precision, torch.backends.mkldnn.matmul.fp32_precision)
    with pytest.raises(RuntimeError):
        with precision_context(None, TORCH_CONTROL):
            assert torch.backends.cuda.matmul.allow_tf32
            raise RuntimeError("retained")
    assert torch.get_float32_matmul_precision() == precision
    assert torch.backends.cuda.matmul.allow_tf32 == cuda
    assert torch.backends.cudnn.allow_tf32 == cudnn
    assert (torch.backends.cuda.matmul.fp32_precision, torch.backends.mkldnn.matmul.fp32_precision) == raw
