"""Spark-only mathematical and real-checkpoint scoring checks before timing."""

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from orbitcall.scoring import SequenceScorer
from orbitcall.gpu_guard import require_idle_gpu, require_owned_gpu


def compare(scorer, prompt, sequences, tolerance, groups=None):
    results = {}
    for method in ("full", "cached", "tree"):
        require_owned_gpu()
        results[method] = scorer.score(prompt, sequences, method, reducer="torch")
        require_owned_gpu()
    reference = results["full"]
    comparisons = {}
    for method, row in results.items():
        if len(row.token_logprobs) != len(sequences) or len(row.sequence_logprobs) != len(sequences):
            raise RuntimeError(f"{method} sequence count mismatch")
        if any(len(values) != len(tokens) for values, tokens in zip(row.token_logprobs, sequences)):
            raise RuntimeError(f"{method} token count mismatch")
        if not all(np.isfinite(value) for values in row.token_logprobs for value in values):
            raise RuntimeError(f"{method} non-finite token score")
        if not all(np.isfinite(value) for value in row.sequence_logprobs):
            raise RuntimeError(f"{method} non-finite sequence score")
        error = max(abs(a - b) for x, y in zip(reference.token_logprobs, row.token_logprobs)
                    for a, b in zip(x, y))
        sum_error = float(np.max(np.abs(np.array(reference.sequence_logprobs)
                                       - np.array(row.sequence_logprobs))))
        comparisons[method] = {"max_token_error": error, "max_sequence_error": sum_error,
                               "same_top_sequence": int(np.argmax(reference.sequence_logprobs))
                               == int(np.argmax(row.sequence_logprobs)),
                               "scored_edges": row.scored_edges,
                               "sequence_logprobs": row.sequence_logprobs}
        sorted_scores = sorted(reference.sequence_logprobs, reverse=True)
        comparisons[method]["reference_sequence_margin"] = sorted_scores[0] - sorted_scores[1]
        if error > tolerance or sum_error > max(map(len, sequences)) * tolerance:
            raise RuntimeError(f"{method} token error {error} exceeds {tolerance}")
        if not comparisons[method]["same_top_sequence"]:
            raise RuntimeError(f"{method} selected sequence differs")
        if groups:
            def masses(values):
                return [float(np.logaddexp.reduce([values[i] for i in group])) for group in groups]
            ref_mass, got_mass = masses(reference.sequence_logprobs), masses(row.sequence_logprobs)
            same = int(np.argmax(ref_mass)) == int(np.argmax(got_mass))
            comparisons[method]["same_finite_family_winner"] = same
            comparisons[method]["finite_family_logmasses"] = got_mass
            sorted_masses = sorted(ref_mass, reverse=True)
            comparisons[method]["reference_family_margin"] = sorted_masses[0] - sorted_masses[1]
            if not same:
                raise RuntimeError(f"{method} finite-family winner differs")
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--attention", default="eager")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    require_idle_gpu()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(31337)
    torch.use_deterministic_algorithms(True)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=256,
                         _attn_implementation="sdpa")
    tiny = SequenceScorer.__new__(SequenceScorer)
    tiny.model = Qwen2ForCausalLM(config).float().cuda().eval()
    tiny.head_chunk, tiny.device, tiny.pad_token = 16, torch.device("cuda"), 0
    record = {"started_epoch": time.time(), "gpu": torch.cuda.get_device_name(),
              "tiny_float32": compare(tiny, [9, 8, 7, 6, 5],
                                      [[2, 3, 4, 1], [2, 3, 5, 1],
                                       [2, 8, 1], [7, 3, 4, 5, 1]], 3e-5)}
    del tiny
    torch.cuda.empty_cache()
    if args.model:
        scorer = SequenceScorer(args.model, dtype=args.dtype, attention=args.attention)
        messages = [{"role": "system", "content": "Select a valid JSON function call."},
                    {"role": "user", "content": "Book two seats to Boston tomorrow."}]
        texts = [
            '{"name":"book","arguments":{"city":"Boston","seats":2}}',
            '{"name":"book","arguments":{"seats":2,"city":"Boston"}}',
            '{"name":"book","arguments":{"city":"Chicago","seats":3}}',
            '{"name":"book","arguments":{"seats":3,"city":"Chicago"}}',
        ]
        prompt, sequences = scorer.prompt_ids(messages), scorer.sequence_ids(texts)
        record["real_checkpoint"] = compare(scorer, prompt, sequences, 1e-3, [[0, 1], [2, 3]])
        record["dtype"] = args.dtype
        record["attention"] = args.attention
        record["model"] = Path(args.model).name
        record["finite_family"] = texts
    record["ended_epoch"] = time.time()
    record["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
