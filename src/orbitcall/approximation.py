"""Explicit approximate-score contracts; historical strict gates stay unchanged."""

from contextlib import contextmanager
import math

from .ranking import compare_scoring_results

LIMITS = {"max_token_absolute_nats": 0.01, "mean_token_absolute_nats": 0.0005,
          "max_sequence_absolute_nats": 0.1, "minimum_winner_agreement": 0.99}
TORCH_CONTROL = {"kind": "torch_tf32", "precision": "high", "scope": "all"}


@contextmanager
def precision_context(model, config):
    """Explicit FP32-storage opt-in, restoring native precision on every exit."""
    if config["kind"] == "linear":
        from .linear_acceleration import accelerated_linears
        with accelerated_linears(model, scope=config["scope"], precision=config["precision"]):
            yield
    elif config == TORCH_CONTROL:
        import torch
        original = torch.get_float32_matmul_precision()
        cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        cuda_raw = torch.backends.cuda.matmul.fp32_precision
        mkldnn_raw = torch.backends.mkldnn.matmul.fp32_precision
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = False
            yield
        finally:
            torch.set_float32_matmul_precision(original)
            torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
            torch.backends.cuda.matmul.fp32_precision = cuda_raw
            torch.backends.mkldnn.matmul.fp32_precision = mkldnn_raw
    else:
        raise ValueError("Unknown approximate configuration")


def evaluate_accuracy(reference, alternative):
    """Retain every error and all margins; per-item validity is numerical only."""
    comparison = compare_scoring_results(reference, alternative,
                                          tolerance=LIMITS["max_token_absolute_nats"])
    errors = [value for path in comparison["per_sequence_token_errors"] for value in path]
    count = len(errors)
    if not count:
        raise ValueError("Zero-family records must be handled outside the accuracy denominator")
    total = math.fsum(errors)
    numerical = (comparison["maximum_token_error"] <= LIMITS["max_token_absolute_nats"]
                 and comparison["maximum_sequence_error"] <= LIMITS["max_sequence_absolute_nats"])
    return {**comparison, "limits": LIMITS.copy(), "token_count": count,
            "sum_absolute_token_error": total, "mean_absolute_token_error": total / count,
            "numerical_valid": numerical,
            "valid": numerical}  # Population mean and agreement are checked below.


def summarize_accuracy(checks, require_all_winners=False):
    """Population metrics use scored records; never let empty pools inflate agreement."""
    if not checks:
        raise ValueError("No scored observations")
    methods = tuple(checks[0]["winner_agreement"])
    if any(set(x["winner_agreement"]) != set(methods) for x in checks):
        raise ValueError("Changed aggregator population")
    count = sum(x["token_count"] for x in checks)
    mean = math.fsum(x["sum_absolute_token_error"] for x in checks) / count
    agreements = {method: sum(x["winner_agreement"][method] for x in checks) / len(checks)
                  for method in methods}
    agreeing_counts = {method: sum(x["winner_agreement"][method] for x in checks) for method in methods}
    joint_count = sum(all(x["winner_agreement"].values()) for x in checks)
    percentage = 100 if require_all_winners else 99
    return {"valid": (all(x["numerical_valid"] for x in checks)
                      and mean <= LIMITS["mean_token_absolute_nats"]
                      and all(100 * x >= percentage * len(checks) for x in agreeing_counts.values())
                      and 100 * joint_count >= percentage * len(checks)),
            "limits": LIMITS.copy(), "require_all_winners": require_all_winners,
            "scored_records": len(checks), "token_count": count,
            "mean_absolute_token_error": mean,
            "maximum_token_error": max(x["maximum_token_error"] for x in checks),
            "maximum_sequence_error": max(x["maximum_sequence_error"] for x in checks),
            "winner_agreement": agreements,
            "joint_all_controls_agreement": joint_count / len(checks),
            "union_changed_ids": [x.get("id", index) for index, x in enumerate(checks)
                                  if not all(x["winner_agreement"].values())],
            "winner_changed_counts": {method: sum(not x["winner_agreement"][method] for x in checks)
                                      for method in methods}}
