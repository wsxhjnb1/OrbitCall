"""Inspect a finite JSON family and aggregate caller-supplied sequence scores.

This is an API example, not a model experiment or benchmark. No model is loaded,
no network request is made, and no numerical scores are invented. The optional
--logprobs values must correspond to the displayed serialization order after
each sequence has been tokenized under one fixed prompt/template convention and
terminated with exactly one explicit end token.

The project's protocol places numerical use of this example on DGX Spark.
"""

from __future__ import annotations

import argparse
import json

from orbitcall.ranking import logsumexp
from orbitcall.serialization import semantic_key, serialization_orbit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logprobs",
        nargs="+",
        type=float,
        help="Caller-supplied complete sequence logprobs, in displayed family order.",
    )
    args = parser.parse_args()

    call = {"name": "reserve", "arguments": {"city": "Boston", "seats": 2}}
    family = serialization_orbit(call, key_order=("city", "seats"))
    record = {
        "scope": "illustrative_api_example_without_model_measurements",
        "call": call,
        "semantic_key": semantic_key(call),
        "serialized_family_without_end_token": list(family),
        "aggregation": "logsumexp of distinct, explicitly terminated sequence logprobs",
        "complete_semantic_posterior": False,
    }
    if args.logprobs is not None:
        if len(args.logprobs) != len(family):
            parser.error("Supply one logprob for each displayed serialization.")
        if any(value > 0 for value in args.logprobs):
            parser.error("A complete sequence log probability cannot be positive.")
        record["log_mass_from_caller_scores"] = logsumexp(args.logprobs)
        record["score_provenance"] = "caller supplied; not verified by this example"
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

