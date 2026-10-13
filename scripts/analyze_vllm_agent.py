#!/usr/bin/env python3
"""Independent CPU analysis of the frozen BF16-backbone vLLM Agent study.

No Torch, vLLM engine, model, GPU context, external service or inference is
loaded. Tokenizers and deterministic local task evaluation run on Spark CPU.
All ratios use retained complete online windows, including divergent branches.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4"
ARMS = ("native_full", "suffix_no_cache", "bounded_apc")
SYSTEM_ARMS = ("native_default", *ARMS)
CONTROLS = SYSTEM_ARMS[:3]
AUDIT_ARMS = ("native_default", *ARMS[1:])
POLICIES = (*SYSTEM_ARMS, "vote4", "greedy1")
SEEDS = (17, 29, 43, 71)
LIMITS = {"max_token_absolute_nats": .05, "mean_token_absolute_nats": .005,
          "max_sequence_absolute_nats": .5, "minimum_winner_agreement": .99}
ENGINE = {"dtype": "bfloat16", "kv_cache_dtype": "auto", "tensor_parallel_size": 1,
          "gpu_memory_utilization": .65, "max_model_len": 4096, "max_num_seqs": 64,
          "max_num_batched_tokens": 8192, "enable_prefix_caching": True,
          "enable_chunked_prefill": True, "async_scheduling": False,
          "enforce_eager": False, "trust_remote_code": False, "seed": 20261013,
          "logprobs_mode": "raw_logprobs"}
FIXED_ORDERS = tuple(itertools.permutations(range(3)))
METHODS = ("mass", "max_order", "mean_logprob", "length_normalized", "canonical_alphabetic",
           *("fixed_" + "".join(map(str, order)) for order in FIXED_ORDERS))
RUNTIME_FILES = {"sampling_params.py", "v1/worker/gpu_model_runner.py", "v1/worker/gpu_input_batch.py",
                 "v1/core/kv_cache_manager.py", "v1/core/kv_cache_coordinator.py",
                 "v1/engine/logprobs.py", "v1/sample/sampler.py", "model_executor/layers/logits_processor.py"}
HASHES: dict[str, str] = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def loads(text):
    return json.loads(text, object_pairs_hook=unique_keys,
                      parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))


def read(path):
    path = Path(path)
    value = loads(path.read_text(encoding="utf-8"))
    HASHES[str(path.relative_to(ROOT))] = sha(path)
    return value


def json_lines(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = [loads(row) for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]
    HASHES[str(path.relative_to(ROOT))] = sha(path)
    return rows


def child(root, relative):
    require(type(relative) is str and not Path(relative).is_absolute(), "Expected relative artifact path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Artifact escaped its directory")
    return path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def equal(left, right):
    return canonical(left) == canonical(right)


def near(actual, expected, name, tolerance=1e-10):
    require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) <= tolerance, "Stored/recomputed mismatch: " + name)


def positive(value, name, allow_zero=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value >= 0 if allow_zero else value > 0), "Invalid duration: " + name)


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def distribution(values):
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "p50": quantile(values, .5), "p90": quantile(values, .9), "p99": quantile(values, .99)}


def verify_sources(run, plan):
    mandatory = {"scripts/run_vllm_agent.py", "scripts/vllm_agent_common.py", "docs/VLLM_AGENT_PROTOCOL.md",
                 "pyproject.toml", "src/orbitcall/agent_tasks.py", "src/orbitcall/vllm_scoring.py",
                 "src/orbitcall/vllm_plugin.py", "src/orbitcall/serialization.py", "src/orbitcall/ranking.py"}
    require(mandatory <= plan["source_sha256"].keys(), "Required source dependency omitted")
    for relative, digest in plan["source_sha256"].items():
        archived, current = child(run / "source", relative), child(ROOT, relative)
        require(sha(archived) == sha(current) == digest, "Frozen source changed: " + relative)
        HASHES[str(archived.relative_to(ROOT))] = digest


def verify_model(engine, model, tokenizer_root):
    public_path = ROOT / "metadata/public_models.json"
    public = read(public_path)
    rows = [row for row in public["models"] if row["repo_id"].split("/")[-1] == model]
    require(len(rows) == 1, "Unexpected public checkpoint")
    declared = rows[0]
    receipt_name = {"Qwen2.5-3B-Instruct": "model_3b_verified.json",
                    "Qwen2.5-1.5B-Instruct": "model_1p5b_verified.json"}[model]
    receipt_path = ROOT / "metadata" / receipt_name
    receipt, recorded = read(receipt_path), engine["model"]
    require(recorded["repo_id"] == declared["repo_id"] == receipt["repo_id"]
            and recorded["revision"] == declared["revision"] == receipt["revision"]
            and recorded["all_public_files_match"] is receipt["all_public_files_match"] is True
            and recorded["receipt_file"] == receipt_name and recorded["receipt_sha256"] == sha(receipt_path)
            and recorded["public_manifest_sha256"] == sha(public_path), "Model/public receipt chain changed")
    verified = {row["path"]: row for row in receipt["files"]}
    require(len(verified) == len(receipt["files"]) == len(declared["files"])
            and set(verified) == {row["path"] for row in declared["files"]}, "Checkpoint inventory incomplete")
    files = []
    for row in declared["files"]:
        actual = verified[row["path"]]
        require(actual["matches_public_revision"] is True and actual["bytes"] == row["size"]
                and (row["sha256"] is None or row["sha256"] == actual["sha256"]), "Public file metadata mismatch")
        files.append({name: actual[name] for name in ("path", "bytes", "sha256")})
    require(recorded["files"] == files, "Engine model files differ from original verified checkpoint")
    directory = tokenizer_root / model
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = directory / name
        require(sha(path) == verified[name]["sha256"] and path.stat().st_size == verified[name]["bytes"],
                "Public tokenizer asset changed")
    # Tokenizers is a CPU Rust library. No Transformers/Torch/model is imported.
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    config = loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    eos = config["eos_token"]
    eos = eos["content"] if type(eos) is dict else eos
    eos_id = tokenizer.token_to_id(eos)
    require(eos_id == 151645, "Configured public Qwen EOS changed")
    return tokenizer, config, eos_id


def render_ids(tokenizer, config, messages):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    template = config["chat_template"]
    if type(template) is list:
        template = {row["name"]: row["template"] for row in template}["default"]
    require(type(template) is str, "Unsupported frozen chat template")
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                               extensions=["jinja2.ext.loopcontrols"])
    environment.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
    environment.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    special = {key: value["content"] if type(value) is dict else value
               for key, value in config.items() if key.endswith("_token")}
    rendered = environment.from_string(template).render(messages=messages, add_generation_prompt=True, **special)
    return tokenizer.encode(rendered, add_special_tokens=False).ids


def parse_pool(outputs, schemas):
    from orbitcall.serialization import parse_call, semantic_key
    calls, invalid, votes = {}, [], Counter()
    for output in outputs:
        call = None
        for schema in schemas:
            try:
                call = parse_call(output["text"], schema)
                break
            except ValueError:
                pass
        if call is None:
            invalid.append({"seed": output["seed"], "reason": "invalid strict tool call"})
        else:
            key = semantic_key(call)
            calls[key], votes[key] = call, votes[key] + 1
    ordered = [calls[key] for key in sorted(calls)]
    vote = calls[sorted(votes, key=lambda key: (-votes[key], key))[0]] if votes else None
    return ordered, invalid, vote


def score_vectors(score, calls, prompt, arm, tokenizer, eos):
    from orbitcall.serialization import semantic_key, serialization_orbit
    require(equal(score["calls"], calls), "Score pool differs from parsed candidates")
    expected, groups = [], []
    for call in calls:
        paths = [tuple(tokenizer.encode(text, add_special_tokens=False).ids + [eos])
                 for text in serialization_orbit(call)]
        distinct = list(dict.fromkeys(paths))
        require(len(distinct) == math.factorial(len(call["arguments"])), "Incomplete finite key-order token orbit")
        groups.append((call, len(distinct)))
        expected.extend(map(list, distinct))
    require(score["suffix_ids"] == expected
            and len(score["token_logprobs"]) == len(score["sequence_logprobs"]) == len(score["cached_tokens"]) == len(expected),
            "Missing/misaligned complete suffix/EOS scores")
    positive(score["elapsed_seconds"], "score", allow_zero=True)
    for path, values, total, cached in zip(expected, score["token_logprobs"], score["sequence_logprobs"], score["cached_tokens"], strict=True):
        require(path and path[-1] == eos and path.count(eos) == 1 and len(values) == len(path), "EOS/token vector shape changed")
        require(all(type(value) in (int, float) and math.isfinite(value) for value in [*values, total]), "Nonfinite score")
        near(total, math.fsum(values), "complete sequence sum")
        require(type(cached) is int and 0 <= cached <= len(prompt) - 1, "Cached a required causal row")
        require(arm == "bounded_apc" or cached == 0, "No-cache scoring control read cached tokens")
    diagnostics = {method: [] for method in METHODS}
    masses, offset = [], 0
    for call, count in groups:
        values = score["sequence_logprobs"][offset:offset + count]
        lengths = [len(path) for path in expected[offset:offset + count]]
        maximum = max(values)
        mass = maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))
        key = semantic_key(call)
        basics = (mass, maximum, math.fsum(values) / count,
                  math.fsum(value / length for value, length in zip(values, lengths, strict=True)) / count, values[0])
        for method, value in zip(METHODS[:5], basics, strict=True):
            diagnostics[method].append((value, key))
        orders = list(itertools.permutations(range(len(call["arguments"]))))
        for method, fixed in zip(METHODS[5:], FIXED_ORDERS, strict=True):
            index = orders.index(tuple(index for index in fixed if index < len(call["arguments"])))
            diagnostics[method].append((values[index], key))
        masses.append({"call": call, "semantic_key": key, "log_mass": mass, "family_size": count})
        offset += count
    require(len(score["log_masses"]) == len(masses), "Family mass count changed")
    for saved, actual in zip(score["log_masses"], masses, strict=True):
        require(equal(saved["call"], actual["call"]) and saved["semantic_key"] == actual["semantic_key"]
                and saved["family_size"] == actual["family_size"], "Family mass identity changed")
        near(saved["log_mass"], actual["log_mass"], "stable mass")
    decisions = {method: sorted(values, key=lambda row: (-row[0], row[1]))[0][1] if values else None
                 for method, values in diagnostics.items()}
    winner = next((call for call in calls if semantic_key(call) == decisions["mass"]), None)
    require(equal(score["winner"], winner), "Stored winner differs from independent full-family mass")
    return decisions


def compare_scores(reference, other, reference_decisions, other_decisions):
    require(reference["suffix_ids"] == other["suffix_ids"] and equal(reference["calls"], other["calls"]),
            "Locked audit changed candidate paths")
    errors = [abs(a - b) for x, y in zip(reference["token_logprobs"], other["token_logprobs"], strict=True)
              for a, b in zip(x, y, strict=True)]
    sequence = [abs(a - b) for a, b in zip(reference["sequence_logprobs"], other["sequence_logprobs"], strict=True)]
    agreement = {method: reference_decisions[method] == other_decisions[method] for method in METHODS}
    return {"nonempty": bool(errors), "max_token_error": max(errors, default=0),
            "sum_token_error": math.fsum(errors), "token_count": len(errors),
            "max_sequence_error": max(sequence, default=0), "rule_agreement": agreement,
            "joint_agreement": all(agreement.values()),
            "numerically_valid": not errors or (max(errors) <= LIMITS["max_token_absolute_nats"]
                                                 and max(sequence) <= LIMITS["max_sequence_absolute_nats"]),
            "token_errors": errors, "sequence_errors": sequence,
            "reference_decisions": reference_decisions, "alternative_decisions": other_decisions}


def audit_summary(rows):
    scored = [row for row in rows if row["nonempty"]]
    n, tokens = len(scored), sum(row["token_count"] for row in scored)
    agreements = {method: sum(row["rule_agreement"][method] for row in scored) for method in METHODS} if scored else {}
    joint = sum(row["joint_agreement"] for row in scored)
    mean = math.fsum(row["sum_token_error"] for row in scored) / tokens if tokens else 0
    valid = bool(n) and all(row["numerically_valid"] for row in scored)
    valid = valid and mean <= LIMITS["mean_token_absolute_nats"] and all(100 * count >= 99 * n for count in [joint, *agreements.values()])
    return {"valid": valid, "nonempty_pools": n, "zero_pools": len(rows) - n,
            "joint_agreeing": joint, "agreement_counts": agreements,
            "max_token_error": max((row["max_token_error"] for row in scored), default=0),
            "mean_token_error": mean, "max_sequence_error": max((row["max_sequence_error"] for row in scored), default=0)}


def assert_audit(saved, actual):
    for key in ("valid", "nonempty_pools", "zero_pools", "joint_agreeing", "agreement_counts"):
        require(saved[key] == actual[key], "Stored audit acceptance/count differs: " + key)
    for key in ("max_token_error", "mean_token_error", "max_sequence_error"):
        near(saved[key], actual[key], "pooled audit " + key)


def generation_check(generated, expected_prompt, policy, stage, tokenizer):
    require(generated["prompt_ids"] == expected_prompt, "Generated prompt differs from actual feedback/template")
    seeds = (17,) if policy == "greedy1" or stage == 3 else SEEDS
    require([row["seed"] for row in generated["outputs"]] == list(seeds), "Candidate sample budget/order changed")
    positive(generated["batch_elapsed_seconds"], "native generation batch")
    for sample in generated["outputs"]:
        ids = sample["token_ids"]
        require(type(ids) is list and all(type(token) is int and token >= 0 for token in ids), "Malformed generated token IDs")
        require(len(ids) <= (64 if stage == 3 else 128), "Generation token ceiling changed")
        require(tokenizer.decode(ids, skip_special_tokens=True) == sample["text"], "Generated text differs from retained tokens")
        require(sample["finish_reason"] in {"stop", "length"}, "Uncompleted native generation")
        require(type(sample["cached_tokens"]) is int and 0 <= sample["cached_tokens"] <= len(expected_prompt), "Malformed generation cache count")


def agent_messages(task, history, stage):
    """Independently reconstruct the new generic directive, without answers."""
    from orbitcall.agent_tasks import render_messages, tool_specs
    messages = render_messages(task, history, stage)
    if stage < 3:
        names = ", ".join(schema["name"] for schema in tool_specs(stage))
        messages.append({"role": "user", "content": f"Current step: perform tool round {stage} now. Return only one JSON tool call with exactly name and arguments, choosing from {names}. Do not give the final answer yet. Use the request and actual observations above."})
    return messages


def episode_check(episode, task, policy, tokenizer, config, eos):
    from orbitcall.agent_tasks import execute_tool, tool_specs
    require(all(episode[key] == task[key] for key in ("id", "split", "family"))
            and episode["policy"] == policy and [row["stage"] for row in episode["rounds"]] == [1, 2],
            "Online episode/task/two-round policy changed")
    history, decisions, work = [], {}, Counter()
    for row in episode["rounds"]:
        stage = row["stage"]
        prompt = render_ids(tokenizer, config, agent_messages(task, history, stage))
        generation_check(row["generation"], prompt, policy, stage, tokenizer)
        calls, invalid, vote = parse_pool(row["generation"]["outputs"], tool_specs(stage))
        require(equal(row["candidates"], calls) and row["invalid"] == invalid, "Candidate parsing/deduplication changed")
        if policy in SYSTEM_ARMS:
            require(type(row["scores"]) is dict, "Scored policy omitted scores")
            decisions[stage] = score_vectors(row["scores"], calls, prompt, policy, tokenizer, eos)
            winner = row["scores"]["winner"]
            work["score_requests"] += len(row["scores"]["suffix_ids"])
            work["scored_suffix_tokens"] += sum(map(len, row["scores"]["suffix_ids"]))
            work["independent_score_prompt_tokens"] += len(prompt) * len(row["scores"]["suffix_ids"])
            work["score_cached_tokens"] += sum(row["scores"]["cached_tokens"])
        else:
            require(row["scores"] is None, "Voting/greedy control performed hidden likelihood scoring")
            winner = vote
        require(equal(row["call"], winner), "Executed call differs from stated policy")
        actual = execute_tool(winner, task)
        require(equal(row["observation"], actual), "Tool observation is not the actual deterministic execution")
        history.append({"call": winner, "observation": actual})
        work["generation_requests"] += len(row["generation"]["outputs"])
        work["generated_tool_tokens"] += sum(len(sample["token_ids"]) for sample in row["generation"]["outputs"])
        work["generation_prompt_tokens"] += len(prompt) * len(row["generation"]["outputs"])
        work["generation_cached_tokens"] += sum(sample["cached_tokens"] for sample in row["generation"]["outputs"])
        work["invalid_candidates"] += len(invalid)
        work["zero_candidate_rounds"] += not calls
        work["tool_errors"] += not actual["ok"]
    require(equal(episode["trace"], history), "Retained trace differs from actual round execution")
    prompt = render_ids(tokenizer, config, agent_messages(task, history, 3))
    final = episode["final_generation"]
    generation_check(final, prompt, policy, 3, tokenizer)
    require(episode["final_response"] == final["outputs"][0]["text"], "Final response replaced native model output")
    work["generation_requests"] += 1
    work["final_tokens"] += len(final["outputs"][0]["token_ids"])
    work["generation_prompt_tokens"] += len(prompt)
    work["generation_cached_tokens"] += final["outputs"][0]["cached_tokens"]
    positive(episode["cohort_service_seconds"], "cohort service")
    positive(episode["request_pool_latency_seconds"], "pooled request latency")
    return {"decisions": decisions, "work": dict(work)}


def pool_check(pool, tasks, policy, key, tokenizer, config, eos):
    require(pool["policy"] == policy and pool["key"] == key
            and [row["id"] for row in pool["records"]] == [task["id"] for task in tasks], "Online pool identity/order changed")
    positive(pool["elapsed_seconds"], "complete Agent pool")
    require(set(pool["stage_seconds"]) == {"generation", "parsing", "tools_and_history", "final_generation", *( ["scoring"] if policy in SYSTEM_ARMS else [])},
            "Unexpected/missing online stage timer")
    for key, value in pool["stage_seconds"].items():
        positive(value, key, allow_zero=True)
    near(pool["harness_seconds"], pool["elapsed_seconds"] - math.fsum(pool["stage_seconds"].values()), "harness residual")
    require(pool["harness_seconds"] >= -1e-7, "Overlapping stage clocks")
    checked = [episode_check(row, task, policy, tokenizer, config, eos)
               for row, task in zip(pool["records"], tasks, strict=True)]
    if policy in SYSTEM_ARMS:
        near(pool["stage_seconds"]["scoring"], math.fsum(row["scores"]["elapsed_seconds"] for episode in pool["records"] for row in episode["rounds"]),
             "disjoint complete scoring times", 1e-8)
    for begin in range(0, len(tasks), 4):
        cohort = pool["records"][begin:begin + 4]
        require(len({row["cohort_service_seconds"] for row in cohort}) == 1
                and len({row["request_pool_latency_seconds"] for row in cohort}) == 1,
                "Cohort-shared clocks relabeled as individual service latency")
        for stage in (0, 1):
            require(len({row["rounds"][stage]["generation"]["batch_elapsed_seconds"] for row in cohort}) == 1,
                    "Shared generation batch timer inconsistent")
    budgets = Counter()
    for row in checked:
        budgets.update(row["work"])
    return {"policy": policy, "key": key, "elapsed_seconds": pool["elapsed_seconds"],
            "stage_seconds": pool["stage_seconds"], "harness_seconds": pool["harness_seconds"],
            "work": dict(budgets), "episode_work": {row["id"]: check["work"] for row, check in zip(pool["records"], checked, strict=True)},
            "cohort_service_seconds": distribution([row["cohort_service_seconds"] for row in pool["records"]]),
            "pooled_request_latency_seconds": distribution([row["request_pool_latency_seconds"] for row in pool["records"]]),
            "latency_scope": "four-episode synchronous cohort service and queued pool wait; not independent per-request TTFT"}, checked


def expected_plan(development):
    rng, plan = random.Random(271828), []
    if not development:
        for control in CONTROLS:
            for block in range(6):
                order = [control, "bounded_apc", "bounded_apc", control] if rng.randrange(2) else ["bounded_apc", control, control, "bounded_apc"]
                plan.append({"control": control, "block": block, "order": order})
        rng.shuffle(plan)
    return plan


def freeze_check(path, receipt_path, expected):
    saved, receipt = read(path), read(receipt_path)
    require(receipt["stage"] == "agent-predictions-frozen" and receipt["answers_accessed"] is False
            and receipt["predictions"]["sha256"] == sha(path) and equal(saved["records"], expected),
            "Prediction freeze/hash/complete record chain changed")
    return {"file": path, "receipt": receipt_path, "predictions_sha256": sha(path), "receipt_sha256": sha(receipt_path)}


def inspect_run(run, inputs, input_path, tokenizer_root):
    plan = read(run / "plan.json")
    development = plan["development"]
    require(type(development) is bool and plan["inputs_sha256"] == sha(input_path)
            and plan["answers_accessed"] is False and plan["engine"] == ENGINE and plan["limits"] == LIMITS
            and tuple(plan["policies"]) == POLICIES and plan["cohort_size"] == 4
            and plan["maximum_process_seconds"] == (1800 if development else 4200), "Frozen Agent configuration changed")
    require(tuple(plan["controls"]) == CONTROLS
            and plan["score_logit_dtype_by_arm"] == {arm: "bfloat16" if arm == "native_default" else "float32" for arm in SYSTEM_ARMS}
            and plan["generation_head_dtype"] == plan["backbone_dtype"] == plan["kv_dtype"] == "bfloat16"
            and plan["prompt_policy"] == "explicit_stage_directive_v2"
            and plan["development_selection"] == "first lexicographic development ID per family",
            "New scoring-head/prompt/control prospective contract changed")
    verify_sources(run, plan)
    selected = [row for row in inputs["records"] if row["split"] == ("development" if development else "test")]
    selected = [sorted((row for row in selected if row["family"] == family), key=lambda row: row["id"])[0]
                for family in ("catalog", "calendar")] if development else selected
    require(plan["selected_ids"] == [row["id"] for row in selected]
            and plan["timing_fixture_ids"] == inputs["timing_fixture_ids"]
            and plan["timing_blocks"] == expected_plan(development), "Input-only corpus/timing schedule changed")
    failure = read(run / "failure.json") if (run / "failure.json").exists() else None
    complete = read(run / "complete.json") if (run / "complete.json").exists() else None
    if not (run / "engine.json").exists():
        require(failure is not None and complete is None, "No engine receipt or failure disposition")
        return {"run": run.name, "model": plan["model"], "development": development, "failure": failure,
                "complete": None, "quality_unavailable": "engine failed before complete prediction freeze",
                "process_elapsed_seconds": failure["elapsed_seconds"], "promotion_accepted": False,
                "files_for_quality": [], "windows": [], "policy_reviews": {}}, []
    engine = read(run / "engine.json")
    require(engine["options"] == ENGINE and engine["runtime"]["vllm"] == "0.29.0"
            and GPU_UUID in engine["gpu"] and "GB10" in engine["gpu"]
            and engine["plugin_source_sha256"] == plan["source_sha256"]["src/orbitcall/vllm_plugin.py"], "Engine/actual plugin identity changed")
    require(set(engine["runtime"]) == {"vllm", "torch", "transformers", "triton"}
            and all(importlib.metadata.version(name) == version for name, version in engine["runtime"].items()),
            "Analysis environment differs from the recorded native runtime packages")
    require(set(engine["runtime_source_sha256"]) == RUNTIME_FILES
            and type(engine["actual_compilation_mode"]) is str and type(engine["actual_cudagraph_mode"]) is str,
            "Actual vLLM source/graph settings not retained")
    package = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    for relative, digest in engine["runtime_source_sha256"].items():
        require(sha(child(package, relative)) == digest, "Executed vLLM source differs from analysis environment")
    positive(engine["initialization_seconds"], "engine initialization")
    tokenizer, config, eos = verify_model(engine, plan["model"], tokenizer_root)
    policy_files, policies, policy_reviews, policy_checks = [], {}, {}, {}
    for policy in (SYSTEM_ARMS if development else POLICIES):
        path = run / ("policy_" + policy + ".json")
        if not path.exists():
            require(failure is not None and complete is None, "Missing full policy result")
            break
        pool = read(path)
        review, checks = pool_check(pool, selected, policy, "policy:" + policy, tokenizer, config, eos)
        policy_files.append(path)
        policies[policy], policy_reviews[policy], policy_checks[policy] = pool, review, checks
    expected_policies = SYSTEM_ARMS if development else POLICIES
    require(not complete or set(policies) == set(expected_policies), "Completed run lost a policy")
    policy_freeze = None
    expected_policy_records = [row for path in policy_files for row in policies[path.stem[len("policy_"):]]["records"]]
    if (run / "policy_predictions.json").exists() and (run / "policy_prediction_freeze.json").exists():
        policy_freeze = freeze_check(run / "policy_predictions.json", run / "policy_prediction_freeze.json", expected_policy_records)
    audit, stock_diagnostic = {}, None
    if (run / "audit_checks.json").exists():
        saved = read(run / "audit_checks.json")
        require(saved["limits"] == LIMITS and "native_full" in policies, "Audit reference/limits missing")
        saved_rows = saved["records"]
        expected_identities = [(row["id"], stage, arm) for row in selected for stage in (1, 2) for arm in AUDIT_ARMS]
        require([(row["id"], row["stage"], row["arm"]) for row in saved_rows] == expected_identities,
                "Locked audit pool population/order changed")
        calculated, files = [], set()
        for index, (episode, checks) in enumerate(zip(policies["native_full"]["records"], policy_checks["native_full"], strict=True)):
            for row in episode["rounds"]:
                for arm in AUDIT_ARMS:
                    path = run / "audit" / (episode["id"] + "_r" + str(row["stage"]) + "_" + arm + ".json")
                    files.add(path.name)
                    alternative = read(path)
                    other_decisions = score_vectors(alternative, row["candidates"], row["generation"]["prompt_ids"], arm, tokenizer, eos)
                    result = {"id": episode["id"], "stage": row["stage"], "arm": arm,
                              **compare_scores(row["scores"], alternative, checks["decisions"][row["stage"]], other_decisions)}
                    stored = saved_rows[len(calculated)]
                    for key in ("id", "stage", "arm", "nonempty", "token_count", "rule_agreement", "joint_agreement", "numerically_valid"):
                        require(stored[key] == result[key], "Saved locked audit check differs: " + key)
                    for key in ("max_token_error", "sum_token_error", "max_sequence_error"):
                        near(stored[key], result[key], "locked raw error " + key)
                    calculated.append(result)
        require({path.name for path in (run / "audit").glob("*.json")} == files, "Unplanned/missing audit raw file")
        stored_summary = read(run / "audit_summary.json")
        require(set(stored_summary) == set(ARMS[1:]), "Strong suffix audit control omitted")
        for arm in ARMS[1:]:
            rows = [row for row in calculated if row["arm"] == arm]
            summary = audit_summary(rows)
            assert_audit(stored_summary[arm], summary)
            audit[arm] = {**summary, "pool_count": len(rows),
                          "zero_pool_ids": [{"id": row["id"], "stage": row["stage"]} for row in rows if not row["nonempty"]],
                          "changed_pools": [{key: row[key] for key in ("id", "stage", "rule_agreement", "reference_decisions", "alternative_decisions")}
                                            for row in rows if not row["joint_agreement"]],
                          "token_errors": distribution([value for row in rows for value in row["token_errors"]]),
                          "sequence_errors": distribution([value for row in rows for value in row["sequence_errors"]]),
                          "numerically_failed_pools": [{"id": row["id"], "stage": row["stage"]} for row in rows if not row["numerically_valid"]]}
        stock_rows = [row for row in calculated if row["arm"] == "native_default"]
        stock_diagnostic = {**audit_summary(stock_rows), "pool_count": len(stock_rows),
                            "comparison": "stock BF16-output native_default versus explicitly FP32-output native_full, same locked paths",
                            "acceptance_scope": "diagnostic only; never part of the same-output-precision numerical gate",
                            "token_errors": distribution([value for row in stock_rows for value in row["token_errors"]]),
                            "sequence_errors": distribution([value for row in stock_rows for value in row["sequence_errors"]]),
                            "changed_pools": [{key: row[key] for key in ("id", "stage", "rule_agreement", "reference_decisions", "alternative_decisions")}
                                              for row in stock_rows if not row["joint_agreement"]]}
    elif complete:
        raise ValueError("Completed run omitted locked audit")
    window_records, windows, file_set = json_lines(run / "windows.jsonl"), [], set()
    scheduled = [(block["control"], block["block"], position, policy)
                 for block in plan["timing_blocks"] for position, policy in enumerate(block["order"])]
    require(len(window_records) <= len(scheduled), "Extra timing windows")
    lookup = {row["id"]: row for row in inputs["records"]}
    fixtures = [lookup[key] for key in inputs["timing_fixture_ids"]]
    for saved, expected in zip(window_records, scheduled, strict=False):
        control, block, position, policy = expected
        relative = f"windows/{control}_{block}_{position}.json"
        require(all(saved[key] == value for key, value in zip(("control", "block", "position", "policy"), expected, strict=True))
                and saved["file"] == relative, "Timing prefix differs from prospective randomized schedule")
        path = child(run, relative)
        require(sha(path) == saved["sha256"], "Raw online window hash changed")
        pool = read(path)
        near(saved["elapsed_seconds"], pool["elapsed_seconds"], "indexed complete online wall time")
        review, _ = pool_check(pool, fixtures, policy, f"timing:{control}:{block}:{position}", tokenizer, config, eos)
        windows.append({"control": control, "block": block, "position": position, "policy": policy,
                        "file": str(path.relative_to(ROOT)), "pool": pool, "review": review, "path": path})
        file_set.add(path.name)
    extras = {path.name for path in (run / "windows").glob("*.json")} - file_set
    require(len(extras) <= 1 and (not extras or failure is not None), "Unplanned unindexed raw window")
    raw_only = []
    if extras:
        require(len(windows) < len(scheduled), "Raw-only tail outside prospective schedule")
        control, block, position, policy = scheduled[len(windows)]
        name = f"{control}_{block}_{position}.json"
        require(extras == {name}, "Unindexed raw tail is not the next planned window")
        path = run / "windows" / name
        pool = read(path)
        review, _ = pool_check(pool, fixtures, policy, f"timing:{control}:{block}:{position}", tokenizer, config, eos)
        raw_only.append({"file": str(path.relative_to(ROOT)), "review": review,
                         "disposition": "retained unindexed failed tail, no accepted paired ratio"})
    final_freeze = None
    if (run / "predictions.json").exists() and (run / "prediction_freeze.json").exists():
        expected = expected_policy_records + [row for window in sorted(windows, key=lambda row: row["path"].name) for row in window["pool"]["records"]]
        final_freeze = freeze_check(run / "predictions.json", run / "prediction_freeze.json", expected)
    if complete:
        require(complete["valid"] is True and len(windows) == len(scheduled) == complete["completed_windows"]
                and not raw_only and policy_freeze is not None and final_freeze is not None
                and complete["predictions_sha256"] == final_freeze["predictions_sha256"]
                and len(audit) == 2 and all(row["valid"] for row in audit.values()), "Completed study lacks required accepted evidence")
    else:
        require(failure is not None, "Study is neither complete nor explicitly failed")
    elapsed = complete["elapsed_seconds"] if complete else failure["elapsed_seconds"]
    positive(elapsed, "study process")
    require(not complete or elapsed <= plan["maximum_process_seconds"], "Completed process exceeded frozen ceiling")
    nested = math.fsum([engine["initialization_seconds"], *(row["elapsed_seconds"] for row in policy_reviews.values()),
                       *(window["review"]["elapsed_seconds"] for window in windows), *(row["review"]["elapsed_seconds"] for row in raw_only)])
    require(elapsed + 1e-5 >= nested, "Process interval shorter than retained nested work")
    quality_files = []
    if policy_freeze:
        quality_files.extend({"path": path, "freeze": policy_freeze, "scope": "policy", "name": path.stem[len("policy_"):]} for path in policy_files)
    if final_freeze:
        quality_files.extend({"path": window["path"], "freeze": final_freeze, "scope": "window",
                              "name": window["path"].stem} for window in windows)
    result = {"run": run.name, "model": plan["model"], "development": development,
              "precision_scope": {"generation_weight_backbone_kv": "bfloat16", "generation_logit_output": "bfloat16",
                                  "native_default_score_logit_output": "bfloat16", "same_precision_score_logit_output": "float32",
                                  "same_precision_reference": "native_full", "score_head_mechanism": "request-scoped native BF16 operands with FP32-output logits, restored after scoring"},
              "complete": complete, "failure": failure, "engine": engine, "audit": audit,
              "stock_default_locked_diagnostic": stock_diagnostic,
              "policy_reviews": policy_reviews, "policy_pools": policies, "windows": windows, "raw_only_unaccepted": raw_only,
              "process_elapsed_seconds": elapsed, "measured_nested_seconds": nested,
              "unassigned_process_seconds": elapsed - nested, "files_for_quality": quality_files,
              "quality_unavailable": None if policy_freeze else "no immutable policy prediction receipt",
              "promotion_accepted": False}
    return result, quality_files


def work_identity(episode):
    return {"id": episode["id"], "rounds": [
        {"stage": row["stage"], "prompt_ids": row["generation"]["prompt_ids"],
         "samples": [{key: sample[key] for key in ("seed", "text", "token_ids", "finish_reason")}
                     for sample in row["generation"]["outputs"]], "candidates": row["candidates"],
         "suffix_ids": row["scores"]["suffix_ids"] if row["scores"] else [],
         "call": row["call"], "observation": row["observation"]} for row in episode["rounds"]],
        "trace": episode["trace"], "final_prompt_ids": episode["final_generation"]["prompt_ids"],
        "final_outputs": [{key: sample[key] for key in ("seed", "text", "token_ids", "finish_reason")}
                          for sample in episode["final_generation"]["outputs"]]}


def compare_episodes(left, right):
    details = []
    for a, b in zip(left["records"], right["records"], strict=True):
        require(a["id"] == b["id"], "Paired online episode order differs")
        round_fields = {}
        for stage, (x, y) in enumerate(zip(a["rounds"], b["rounds"], strict=True), 1):
            round_fields[str(stage)] = {
                "same_prompt_tokens": x["generation"]["prompt_ids"] == y["generation"]["prompt_ids"],
                "same_samples": equal([{key: item[key] for key in ("seed", "text", "token_ids", "finish_reason")} for item in x["generation"]["outputs"]],
                                      [{key: item[key] for key in ("seed", "text", "token_ids", "finish_reason")} for item in y["generation"]["outputs"]]),
                "same_candidates": equal(x["candidates"], y["candidates"]),
                "same_scored_tokens": x["scores"]["suffix_ids"] == y["scores"]["suffix_ids"],
                "same_selected_call": equal(x["call"], y["call"]), "same_tool_observation": equal(x["observation"], y["observation"])}
        details.append({"id": a["id"], "matched_complete_workload": equal(work_identity(a), work_identity(b)),
                        "rounds": round_fields, "same_final_text": a["final_response"] == b["final_response"],
                        "same_final_tokens": a["final_generation"]["outputs"][0]["token_ids"] == b["final_generation"]["outputs"][0]["token_ids"]})
    return {"episode_count": len(details), "matched_count": sum(row["matched_complete_workload"] for row in details),
            "all_matched": all(row["matched_complete_workload"] for row in details), "episodes": details}


def quality_summary(rows):
    n = len(rows)
    return {"episodes": n, "task_successes": sum(row["task_success"] for row in rows),
            "task_success_rate": sum(row["task_success"] for row in rows) / n if n else None,
            "final_json_valid": sum(row["final_json_valid"] for row in rows),
            "final_correct": sum(row["final_correct"] for row in rows),
            "round_correct": [sum(row["round_correct"][index] for row in rows) for index in (0, 1)],
            "actual_observation_consistent": sum(all(row["observation_consistent"]) for row in rows),
            "failed_ids": [row["id"] for row in rows if not row["task_success"]],
            "family": {family: {"episodes": sum(row["family"] == family for row in rows),
                                 "task_successes": sum(row["task_success"] and row["family"] == family for row in rows)}
                       for family in ("catalog", "calendar")},
            "scope": "all fixed local task episodes including empty/error pools; no external population or official benchmark claim"}


def exclusive_json(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare_quality_receipts(specifications, directory):
    # Every receipt is prepared before ANY task reference/evaluation is accessed.
    directory.mkdir(parents=True, exist_ok=False)
    for index, spec in enumerate(specifications):
        path = directory / f"{index:04d}.json"
        freeze = spec["freeze"]
        payload = {"stage": "agent-predictions-frozen", "answers_accessed": False,
                   "predictions": {"sha256": sha(spec["path"])}, "derived_receipt": True,
                   "source_prediction_file": str(spec["path"].relative_to(ROOT)),
                   "parent_freeze_file": str(freeze["receipt"].relative_to(ROOT)),
                   "parent_freeze_sha256": freeze["receipt_sha256"],
                   "parent_predictions_sha256": freeze["predictions_sha256"],
                   "derivation": "unchanged original records verified in original aggregate byte-freeze; no new predictions/references"}
        exclusive_json(path, payload)
        spec["derived_receipt"] = path
        HASHES[str(path.relative_to(ROOT))] = sha(path)


def evaluate_study(study, inputs):
    from orbitcall.agent_tasks import evaluate_after_freeze
    lookup = {row["id"]: row for row in inputs["records"]}
    qualities = {}
    for spec in study["files_for_quality"]:
        pool = read(spec["path"])
        evaluated = [evaluate_after_freeze(lookup[row["id"]], row["final_response"], row["trace"],
                                          predictions_path=spec["path"], freeze_receipt_path=spec["derived_receipt"])
                     for row in pool["records"]]
        qualities[(spec["scope"], spec["name"])] = {"summary": quality_summary(evaluated), "records": evaluated,
                                                    "derived_receipt": str(spec["derived_receipt"].relative_to(ROOT))}
    study["policy_quality"] = {name: value for (scope, name), value in qualities.items() if scope == "policy"}
    by_window = {name: value for (scope, name), value in qualities.items() if scope == "window"}
    for window in study["windows"]:
        window["quality"] = by_window.get(window["path"].stem)
    return qualities


def timing_comparisons(study):
    if not study["complete"] or study["development"]:
        return []
    reports = []
    for control in CONTROLS:
        blocks, all_matched, no_quality_drop = [], True, True
        for block in range(6):
            windows = sorted([row for row in study["windows"] if row["control"] == control and row["block"] == block], key=lambda row: row["position"])
            require(len(windows) == 4, "Incomplete accepted four-window block")
            arms = {arm: [row for row in windows if row["policy"] == arm] for arm in (control, "bounded_apc")}
            require(all(len(rows) == 2 for rows in arms.values()), "Block is not two A/two B")
            means = {arm: statistics.mean(row["review"]["elapsed_seconds"] for row in rows) for arm, rows in arms.items()}
            stage_means = {arm: {stage: statistics.mean(row["review"]["stage_seconds"][stage] for row in rows)
                                 for stage in rows[0]["review"]["stage_seconds"]} for arm, rows in arms.items()}
            cross = [compare_episodes(a["pool"], b["pool"]) for a in arms[control] for b in arms["bounded_apc"]]
            repeat = {arm: compare_episodes(rows[0]["pool"], rows[1]["pool"]) for arm, rows in arms.items()}
            matched = all(row["all_matched"] for row in [*cross, *repeat.values()])
            quality = {arm: sum(row["quality"]["summary"]["task_successes"] for row in rows) for arm, rows in arms.items()}
            drop = quality["bounded_apc"] < quality[control]
            all_matched, no_quality_drop = all_matched and matched, no_quality_drop and not drop
            blocks.append({"block": block, "order": [row["policy"] for row in windows],
                           "wall_seconds_mean_of_two": means, "control_over_optimized": means[control] / means["bounded_apc"],
                           "disjoint_stage_seconds_mean_of_two": stage_means,
                           "scoring_control_over_optimized": stage_means[control]["scoring"] / stage_means["bounded_apc"]["scoring"]
                           if stage_means["bounded_apc"]["scoring"] > 0 else None,
                           "cross_arm_workload_comparisons": cross, "AA_BB_repeatability": repeat,
                           "all_workloads_matched": matched, "task_successes_of_16_episode_windows": quality,
                           "optimized_task_success_drop": drop})
        ratios = [row["control_over_optimized"] for row in blocks]
        score_ratios = [row["scoring_control_over_optimized"] for row in blocks]
        corpus = study["policy_quality"]
        corpus_drop = corpus["bounded_apc"]["summary"]["task_successes"] < corpus[control]["summary"]["task_successes"]
        corpus_match = compare_episodes(study["policy_pools"][control], study["policy_pools"]["bounded_apc"])
        reports.append({"control": control, "optimized": "bounded_apc", "block_count": 6, "window_count": 24,
                        "median_control_over_optimized": statistics.median(ratios), "range": [min(ratios), max(ratios)],
                        "median_control_over_optimized_scoring": statistics.median(score_ratios) if all(value is not None for value in score_ratios) else None,
                        "scoring_range": [min(score_ratios), max(score_ratios)] if all(value is not None for value in score_ratios) else None,
                        "score_ratio_scope": "scoring stage inside the online workflow; real prefix population/generation is counted in the whole-Agent timer, not this nested stage",
                        "blocks": blocks, "all_timed_workloads_matched": all_matched, "no_block_task_success_drop": no_quality_drop,
                        "complete_test_corpus_workload_comparison": corpus_match,
                        "complete_test_corpus_task_success_drop": corpus_drop,
                        "same_workload_promotion_accepted": all_matched and corpus_match["all_matched"] and no_quality_drop and not corpus_drop,
                        "ratio_scope": "all retained actual online Agent windows, no divergent episode/block exclusion",
                        "interpretation": "fixed-workload implementation ratio" if all_matched else "adaptive online-policy latency ratio; workload differs"})
    return reports


def retained_failure_history(run):
    """Hash and cost retired/unsupported runs without reinterpreting science."""
    failure = read(run / "failure.json") if (run / "failure.json").exists() else None
    completion = read(run / "complete.json") if (run / "complete.json").exists() else None
    require(failure is not None or completion is not None, "Retired run has no process disposition")
    plan = read(run / "plan.json") if (run / "plan.json").exists() else None
    if plan:
        for relative, digest in plan["source_sha256"].items():
            require(sha(child(run / "source", relative)) == digest, "Retired source archive changed")
    originals = {}
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".py", ".md", ".toml", ".log", ".txt"}:
            digest = sha(path)
            originals[path.relative_to(run).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest}
            HASHES[str(path.relative_to(ROOT))] = digest
    intervals = [row["elapsed_seconds"] for row in (failure, completion) if row is not None and "elapsed_seconds" in row]
    require(intervals, "Retired run lacks a recorded process interval")
    for value in intervals:
        positive(value, "retired process")
    audit = read(run / "audit_summary.json") if (run / "audit_summary.json").exists() else None
    return {"run": run.name, "disposition": "retired preflight/unsupported earlier mode; no accepted current-mode Agent ratio",
            "process_elapsed_seconds": max(intervals), "cost_rule": "latest recorded elapsed interval once, nested stage times not added",
            "has_failure_record": failure is not None, "has_completion_record": completion is not None,
            "stored_errors": completion.get("errors") if completion else None,
            "stored_prior_audit_summary": audit, "stored_summary_independently_recomputed": False,
            "source_verification": "archived plan hashes" if plan else "byte hashes only; smoke producer had no original source manifest",
            "retained_window_count": len(list((run / "windows").glob("*.json"))),
            "raw_artifact_inventory": originals, "accepted_speed_ratios": [],
            "task_quality_recomputed": False, "old_thresholds_or_results_relabelled": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, default=ROOT / "models")
    parser.add_argument("--failed-runs", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "aarch64":
        raise RuntimeError("All numerical/quality analysis runs exclusively on authorized Spark")
    identity = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True, timeout=15).strip().splitlines()
    require(len(identity) == 1 and GPU_UUID in identity[0] and "GB10" in identity[0], "Wrong live CPU analysis host")
    sys.path.insert(0, str(ROOT / "src"))
    from orbitcall.agent_tasks import tasks
    started = time.perf_counter()
    input_path = (ROOT / args.inputs).resolve() if not args.inputs.is_absolute() else args.inputs.resolve()
    output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    require(input_path.is_relative_to(ROOT) and output.is_relative_to(ROOT) and not output.exists(), "Input/output paths outside repo or output exists")
    inputs = read(input_path)
    require(inputs["stage"] == "agent-inputs-frozen" and inputs["answers_accessed"] is False
            and inputs["tool_execution_performed"] is inputs["model_inference_performed"] is False
            and inputs["records"] == tasks() and inputs["record_count"] == 48
            and inputs["split_counts"] == {"development": 12, "test": 36}, "Original input-only fixture freeze changed")
    for relative, digest in inputs["source_sha256"].items():
        require(sha(child(ROOT, relative)) == digest, "Input construction source changed")
    fixtures = [row["id"] for family in ("catalog", "calendar") for row in sorted(
        (row for row in inputs["records"] if row["family"] == family and row["split"] == "test"), key=lambda row: row["id"])[:4]]
    require(inputs["timing_fixture_ids"] == fixtures, "Input-only first-four per-family timing rule changed")
    runs = [(ROOT / path).resolve() if not path.is_absolute() else path.resolve() for path in args.runs]
    require(len(set(runs)) == len(runs) and all(path.is_relative_to(ROOT / "runs") for path in runs), "Duplicate/outside study runs")
    failed_runs = [(ROOT / path).resolve() if not path.is_absolute() else path.resolve() for path in args.failed_runs]
    require(len(set(failed_runs)) == len(failed_runs) and not set(failed_runs) & set(runs)
            and all(path.is_relative_to(ROOT / "runs") for path in failed_runs), "Duplicate/overlapping/outside retired runs")
    tokenizer_root = (ROOT / args.tokenizer_root).resolve() if not args.tokenizer_root.is_absolute() else args.tokenizer_root.resolve()
    studies, specifications = [], []
    for run in runs:
        study, files = inspect_run(run, inputs, input_path, tokenizer_root)
        studies.append(study)
        specifications.extend(files)
    require(len({(row["model"], row["development"]) for row in studies}) == len(studies), "Duplicate model/stage study")
    history = [retained_failure_history(run) for run in failed_runs]
    # Validate ALL provided sources and prediction freezes before any answer use.
    receipt_directory = output.parent / (output.stem + "_quality_receipts")
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_quality_receipts(specifications, receipt_directory)
    for study in studies:
        evaluate_study(study, inputs)
        study["timing_comparisons"] = timing_comparisons(study)
        study["promotion_accepted"] = bool(not study["development"] and study["complete"]
                                            and len(study["timing_comparisons"]) == 3
                                            and all(row["same_workload_promotion_accepted"] for row in study["timing_comparisons"]))
        # Retain original paths/hashes, not huge duplicate raw records in JSON.
        for window in study["windows"]:
            window.pop("pool")
            window.pop("path")
        study.pop("files_for_quality")
        study.pop("policy_pools", None)
    elapsed = time.perf_counter() - started
    models = {row["model"] for row in studies if not row["development"]}
    both = models == {"Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct"}
    result = {"format_version": 1, "analysis_source_sha256": sha(Path(__file__)),
              "execution": "authorized Spark CPU-only; tokenizer assets and deterministic tools, no model/inference/GPU context",
              "live_gpu_identity": identity[0], "inputs_sha256": sha(input_path), "limits": LIMITS,
              "studies": studies, "retired_failed_run_history": history, "both_confirmation_models_present": both,
              "combined_same_workload_promotion_accepted": both and all(row["promotion_accepted"] for row in studies if not row["development"]),
              "provided_run_process_wall_seconds": math.fsum(row["process_elapsed_seconds"] for row in studies),
              "provided_retired_run_process_wall_seconds": math.fsum(row["process_elapsed_seconds"] for row in history),
              "provided_current_plus_retired_wall_lower_bound_seconds": math.fsum(row["process_elapsed_seconds"] for row in [*studies, *history]),
              "independent_analysis_wall_seconds": elapsed, "source_artifact_sha256": HASHES,
              "cost_scope": "distinct provided process intervals; initialization/policy/window costs nested, no double count; not GPU-busy time or all-project budget",
              "claim_limits": ["All online branches retained; different workloads prevent fixed-workload acceleration promotion.",
                               "Local deterministic 36-task suite is not an official Agent benchmark or an external success-probability guarantee.",
                               "Eleven-rule likelihood fidelity uses only nonempty pools; task quality includes empty/error episodes.",
                               "Six blocks and repeated episodes are not hundreds of independent task trials.",
                               "Cohort service and pooled queue latency are not independent per-request TTFT.",
                               "First-round/second-round tool outcomes are actual executions; no invented network delay."]}
    exclusive_json(output, result)
    print(json.dumps({"status": "independent_vllm_agent_analysis_complete", "output_sha256": sha(output),
                      "combined_same_workload_promotion_accepted": result["combined_same_workload_promotion_accepted"]}, indent=2))


if __name__ == "__main__":
    main()
