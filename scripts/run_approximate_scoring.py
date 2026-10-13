"""Spark-only, answer-free confirmation of one frozen approximate scorer.

Audit the selected pipeline and its declared strong controls against all 119
retained IEEE-FP32 records, then time the pre-frozen eight test families.
Numerical failures retain every completed audit observation and timing window.
The selected configuration and bounds are never changed by this driver.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import time
import traceback
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
import triton

from audit_full_triton import (
    GPU_UUID, INPUT_SHA256, validate_item, verify_model,
    write_new,
)
from benchmark_scoring import append_observation, ce_selected_adapter, invoke, measure, sha256
from run_accelerated_scoring import checked_reference, choose_timing_fixtures
from orbitcall.approximation import (
    LIMITS, TORCH_CONTROL, evaluate_accuracy, precision_context, summarize_accuracy,
)
from orbitcall.bfcl import render_messages
from orbitcall.gpu_guard import require_idle_gpu, require_owned_gpu
from orbitcall.ranking import FIXED_METHODS, SCORE_METHODS, aggregate_scores
from orbitcall.scoring import SequenceScorer
from orbitcall.serialization import semantic_key, serialization_orbit, validate_call
from orbitcall.trie import TokenTrie

ROOT = Path(__file__).resolve().parents[1]
MAX_SECONDS = 3600
BLOCKS = 6
TIMING_SEED = 271828
METHODS = (*SCORE_METHODS, *FIXED_METHODS)
CONFIGS = [{"kind": "linear", "precision": precision, "scope": scope}
           for precision in ("tf32x3", "tf32x6")
           for scope in ("head", "backbone", "all")] + [TORCH_CONTROL]
CACHED_CONTROL = {"kind": "pipeline", "method": "cached", "linear": None}
PIPELINE_CONFIGS = [{"kind": "pipeline", "method": method, "linear": linear}
                    for method in ("cached", "tree")
                    for linear in (None, {"kind": "linear", "precision": "tf32x3", "scope": "all"})]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_budget(deadline):
    if time.monotonic() > deadline:
        raise RuntimeError("Predeclared 3600-second confirmation budget exhausted")


def require_ieee():
    if (torch.get_float32_matmul_precision() != "highest"
            or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32
            or torch.is_autocast_enabled("cuda")):
        raise RuntimeError("Native IEEE FP32 settings were not restored")


def effective_precision():
    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "mkldnn_matmul_fp32_precision": torch.backends.mkldnn.matmul.fp32_precision,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_autocast_enabled": torch.is_autocast_enabled("cuda")}


def method_for(config):
    return config["method"] if config["kind"] == "pipeline" else "full"


def context_for(model, config):
    arithmetic = config["linear"] if config["kind"] == "pipeline" else config
    return nullcontext() if arithmetic is None else precision_context(model, arithmetic)


def selected_configuration(path):
    """Reject old strict studies or any changed development implementation."""
    selection = read_json(path)
    plan_path = path.parent / "plan.json"
    plan = read_json(plan_path)
    config = selection.get("configuration")
    is_pipeline = config in PIPELINE_CONFIGS
    if (selection.get("valid") is not True or selection.get("answers_accessed") is not False
            or plan.get("answers_accessed") is not False
            or selection.get("plan_sha256") != sha256(plan_path)
            or config not in CONFIGS + PIPELINE_CONFIGS or config not in plan["configs"]
            or plan.get("limits") != LIMITS
            or plan["model"]["repo_id"].split("/")[-1] != "Qwen2.5-3B-Instruct"
            or plan["fixture_freeze"]["inputs_sha256"] != INPUT_SHA256):
        raise ValueError("Invalid frozen approximate-development selection")
    if is_pipeline and plan.get("comparators") != ["cached"]:
        raise ValueError("Prefix selection must freeze the strong cached comparator")
    declared = plan["source_sha256"]
    required = {path.relative_to(ROOT).as_posix()
                for path in (ROOT / "src/orbitcall").glob("*.py")}
    required.update({"docs/APPROXIMATE_SCORING_PROTOCOL.md",
                     "metadata/acceleration_input_freeze_v1.json"})
    if is_pipeline:
        required.add("docs/PREFIX_APPROXIMATE_PROTOCOL.md")
    if not required <= declared.keys():
        raise ValueError("Development did not freeze all scoring source, bounds and input metadata")
    for relative, digest in declared.items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or sha256(source) != digest:
            raise ValueError(f"Selected development source changed: {relative}")
        archived = (path.parent / "source" / relative).resolve()
        if not archived.is_relative_to((path.parent / "source").resolve()) or sha256(archived) != digest:
            raise ValueError(f"Selected development snapshot changed: {relative}")
    return selection, plan


def inventory_paths(selection_plan):
    paths = [Path(__file__).resolve(), ROOT / "AGENTS.md",
             ROOT / "docs/APPROXIMATE_SCORING_PROTOCOL.md",
             ROOT / "metadata/acceleration_input_freeze_v1.json",
             ROOT / "metadata/public_models.json",
             ROOT / "metadata/model_3b_verified.json",
             ROOT / "metadata/model_1p5b_verified.json"]
    paths += sorted((ROOT / "src/orbitcall").glob("*.py"))
    paths += [ROOT / "scripts" / name for name in
              ("audit_full_triton.py", "benchmark_scoring.py", "run_accelerated_scoring.py")]
    paths += [ROOT / name for name in selection_plan["source_sha256"]]
    if selection_plan.get("comparators") == ["cached"]:
        paths.append(ROOT / "docs/PREFIX_CONFIRMATION_CONTROL_AMENDMENT.md")
    return sorted(set(paths))


def validate_layout(scores, prompt, sequences, batch_size, expected_method="full"):
    """Include every EOS token, full prompt copy and original padded query row."""
    paths, totals = scores["token_logprobs"], scores["sequence_logprobs"]
    if (scores["method"] != expected_method or scores["prompt_tokens"] != len(prompt)
            or len(paths) != len(sequences) or len(totals) != len(sequences)
            or any(len(path) != len(sequence) for path, sequence in zip(paths, sequences, strict=True))):
        raise ValueError("Incomplete, misaligned or relabeled score vectors")
    if any(type(value) not in (float, int) or not math.isfinite(value)
           for value in [*totals, *(value for path in paths for value in path)]):
        raise ValueError("Non-finite score vector")
    if any(abs(math.fsum(path) - total) > 1e-5 for path, total in zip(paths, totals, strict=True)):
        raise ValueError("Sequence score differs from its complete EOS-inclusive sum")
    tokens = sum(map(len, sequences))
    query_rows = sum(len(group) * (max(map(len, group)) - 1)
                     for begin in range(0, len(sequences), batch_size)
                     for group in [sequences[begin:begin + batch_size]])
    edges = tokens
    prompt_rows = len(prompt) * len(sequences) if expected_method == "full" else len(prompt)
    if expected_method == "tree":
        tree = TokenTrie.build(sequences)
        edges = tree.edge_count
        query_rows = len({parent for parent in tree.parents if parent >= 0})
    if (scores["independent_suffix_tokens"] != tokens or scores["scored_edges"] != edges
            or scores["processed_prompt_positions"] != prompt_rows
            or scores["processed_query_positions"] != query_rows):
        raise ValueError("Scoring changed full-prompt/EOS paths or padded batch shapes")


def check_paths(result, expected, expected_method="full"):
    if any(result[key] != expected[key]
           for key in ("id", "prompt_ids", "sequence_ids", "families")):
        raise ValueError("Scoring changed a prompt, token path, order, family or record ID")
    validate_layout(result["scores"], expected["prompt_ids"], expected["sequence_ids"],
                    len(expected["sequence_ids"]), expected_method)
    recomputed = aggregate_scores(expected["families"], result["scores"]["sequence_logprobs"],
                                  list(map(len, expected["sequence_ids"])))
    if result["predictions"] != recomputed or set(recomputed) != set(METHODS):
        raise ValueError("Scorer predictions do not cover the eleven unchanged aggregators")


def accuracy_check(reference, alternative, record_id, split):
    check = evaluate_accuracy(reference, alternative)
    if set(check["winner_agreement"]) != set(METHODS):
        raise ValueError("Approximate check changed the eleven aggregator controls")
    return {"id": record_id, "split": split, **check}


def accuracy_summary(checks, *, require_all_winners=False):
    result = summarize_accuracy(checks, require_all_winners=require_all_winners)
    # The overall scored population is the audit gate. Split summaries expose
    # concentration of errors but do not introduce undeclared subgroup gates.
    result["by_split"] = {}
    for split in ("development", "test"):
        subset = [check for check in checks if check["split"] == split]
        summary = (summarize_accuracy(subset, require_all_winners=require_all_winners)
                   if subset else {"valid": None, "scored_records": 0})
        result["by_split"][split] = {**summary, "descriptive_only": True}
    result["changed_record_ids"] = {
        method: [check["id"] for check in checks if not check["winner_agreement"][method]]
        for method in METHODS
    }
    result["union_changed_ids"] = [check["id"] for check in checks
                                   if not all(check["winner_agreement"].values())]
    return result


def timing_plan(comparators):
    rng, plan = random.Random(TIMING_SEED), []
    for control in comparators:
        for block in range(BLOCKS):
            order = [control, "optimized", "optimized", control]
            if rng.randrange(2):
                order = ["optimized", control, control, "optimized"]
            plan.append({"pair": control + "_vs_optimized", "control": control,
                         "block": block, "order": order})
    rng.shuffle(plan)
    return plan


def paired_summary(observations, control):
    blocks = []
    pair = control + "_vs_optimized"
    for number in range(BLOCKS):
        rows = [row for row in observations if row["pair"] == pair and row["block"] == number]
        if len(rows) != 4 or any(sum(row["arm"] == arm for row in rows) != 2
                                 for arm in (control, "optimized")):
            raise ValueError("Paired timing block is incomplete or duplicated")
        times = {arm: statistics.mean(row["wall_seconds"] for row in rows if row["arm"] == arm)
                 for arm in (control, "optimized")}
        blocks.append({"block": number, "seconds": times,
                       "control_over_optimized": times[control] / times["optimized"]})
    ratios = [block["control_over_optimized"] for block in blocks]
    resampled = sorted(statistics.median(values)
                       for values in itertools.product(ratios, repeat=BLOCKS))
    return {"pair": pair, "blocks": blocks, "median_ratio": statistics.median(ratios),
            "range": [min(ratios), max(ratios)],
            "descriptive_bootstrap_median_95": [resampled[int(.025 * len(resampled))],
                                                 resampled[int(.975 * len(resampled))]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "ranking", "candidates", "selection", "run"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("model", "ranking", "candidates", "selection", "run"):
        setattr(args, name, getattr(args, name).resolve())
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("All approximate-score experiments run on the authorized Spark")
    if args.run.exists():
        raise FileExistsError(args.run)
    require_idle_gpu()
    args.run.mkdir(parents=True, exist_ok=False)
    started, started_epoch = time.monotonic(), time.time()
    deadline, current = started + MAX_SECONDS, None
    inventory, source_hashes, reference_hashes, model_stats = [], {}, {}, {}
    selection_hash, candidate_hash, integrity_check = None, None, None
    try:
        selection, selection_plan = selected_configuration(args.selection)
        config = selection["configuration"]
        selection_hash, candidate_hash = sha256(args.selection), sha256(args.candidates)
        inventory = inventory_paths(selection_plan)
        source_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in inventory}
        for path in inventory:
            destination = args.run / "source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            if sha256(destination) != source_hashes[path.relative_to(ROOT).as_posix()]:
                raise RuntimeError("Source changed while freezing the confirmation snapshot")
        for source, name in ((args.selection, "selection.json"),
                             (args.selection.parent / "plan.json", "selection_plan.json"),
                             (ROOT / "metadata/acceleration_input_freeze_v1.json", "input_freeze.json")):
            shutil.copyfile(source, args.run / name)
        reference_manifest, rows, items, reference_hashes = checked_reference(args.ranking, args.model.name)
        candidates = read_json(args.candidates)
        if candidate_hash != reference_manifest["candidates_sha256"] or candidates.get("answers_accessed") is not False:
            raise ValueError("Candidate hash or answer isolation changed")
        fixtures, dispositions = choose_timing_fixtures(candidates, rows)
        if len(fixtures) != 8:
            raise ValueError("Confirmation requires the predeclared eight test fixtures")
        freeze = read_json(ROOT / "metadata/acceleration_input_freeze_v1.json")
        frozen = [record for record in freeze["records"] if record["model"] == args.model.name]
        if (len(frozen) != 1 or freeze.get("answers_accessed") is not False
                or frozen[0]["candidates_sha256"] != candidate_hash
                or frozen[0]["timing_fixture_ids"] != [fixture["id"] for fixture in fixtures]
                or frozen[0]["test_dispositions"] != dispositions):
            raise ValueError("Frozen answer-free timing population changed")
        checkpoint, model_stats = verify_model(args.model, read_json)
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        ).strip().splitlines()
        if (len(gpu) != 1 or "GB10" not in gpu[0] or GPU_UUID not in gpu[0]
                or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise RuntimeError("Incorrect study GPU identity or visibility")
        # Freeze both possible pair orders before model work. A failed control
        # gate only removes its predeclared pair; it never redraws the schedule.
        is_pipeline = config["kind"] == "pipeline"
        control_configs = ({"cached": CACHED_CONTROL,
                            "structure": {"kind": "pipeline", "method": config["method"], "linear": None}}
                           if is_pipeline else {"torch_tf32": TORCH_CONTROL})
        arm_configs = {"optimized": config}
        control_audit_arms = {}
        for name, control_config in control_configs.items():
            matches = [arm for arm, arm_config in arm_configs.items() if arm_config == control_config]
            if matches:
                control_audit_arms[name] = matches[0]
            else:
                arm_configs[name] = control_config
                control_audit_arms[name] = name
        aliases = {name: arm for name, arm in control_audit_arms.items() if name != arm}
        comparators = ["native", *(name for name, arm in control_audit_arms.items() if name == arm)]
        plan = timing_plan(comparators)
        manifest = {
            "format_version": 1, "stage": "approximate whole119 audit and held-out full-score timing",
            "answers_accessed": False, "started_epoch": started_epoch, "pid": os.getpid(),
            "model": checkpoint, "gpu": gpu[0], "expected_gpu_uuid": GPU_UUID,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "triton": triton.__version__, "python": platform.python_version(),
            "configuration": config, "limits": LIMITS, "audit_batch_size": 8,
            "timing_batch_size": "whole-family", "attention": "eager", "head_chunk": 32,
            "weight_activation_storage": "float32", "normalizer": "torch_ce",
            "shared_parent_ce_adapter": "benchmark_scoring.ce_selected_adapter, applied to all arms",
            "selected_scoring_method": method_for(config),
            "reference_reducer": "torch_lse", "reference_arithmetic": "IEEE FP32, TF32 off",
            "torch_tf32_control": TORCH_CONTROL,
            "torch_tf32_scope": "global matmul policy, including attention matrix products",
            "control_configurations": control_configs, "control_audit_arms": control_audit_arms,
            "control_aliases": aliases,
            "torch_tf32_audited": not is_pipeline,
            "selection_sha256": selection_hash, "selection": selection,
            "selection_plan_sha256": sha256(args.selection.parent / "plan.json"),
            "record_ids": [row["id"] for row in rows], "record_count": len(rows),
            "reference_sha256": reference_hashes, "ranking_run": args.ranking.name,
            "inputs_sha256": INPUT_SHA256, "candidates_sha256": candidate_hash,
            "fixture_ids": [fixture["id"] for fixture in fixtures], "test_dispositions": dispositions,
            "source_sha256": source_hashes, "audit_arms": arm_configs,
            "confirmation_control_amendment": ("docs/PREFIX_CONFIRMATION_CONTROL_AMENDMENT.md"
                                               if is_pipeline else None),
            "timing_plan": plan, "timing_seed": TIMING_SEED, "blocks_per_pair": BLOCKS,
            "max_seconds": MAX_SECONDS, "methods": list(METHODS),
            "zero_family_scope": "retained null predictions; excluded from every accuracy denominator",
            "selection_scope": "one immutable 3B development choice used unchanged on both models",
            "timing_scope": "complete score stage; excludes generation, loading, first-call/JIT setup and validation",
        }
        write_new(args.run / "manifest.json", manifest)
        check_budget(deadline)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        write_new(args.run / "setup/native_precision.json", effective_precision())
        load_started = time.monotonic()
        scorer = SequenceScorer(args.model, dtype="float32", attention="eager", head_chunk=32)
        torch.cuda.synchronize()
        write_new(args.run / "setup/model_load.json", {"elapsed_seconds": time.monotonic() - load_started})

        def recheck_integrity():
            if any(sha256(ROOT / relative) != digest for relative, digest in source_hashes.items()):
                raise RuntimeError("Frozen confirmation source or metadata changed during run")
            if (sha256(args.selection) != selection_hash
                    or sha256(args.selection.parent / "plan.json") != manifest["selection_plan_sha256"]
                    or sha256(args.candidates) != candidate_hash
                    or sha256(ROOT / "data/frozen_inputs.json") != INPUT_SHA256
                    or any(sha256(args.ranking / relative) != digest
                           for relative, digest in reference_hashes.items())):
                raise RuntimeError("Frozen selection, input, candidate or reference changed during run")
            if any((args.model / relative).stat().st_size != size
                   or (args.model / relative).stat().st_mtime_ns != mtime
                   for relative, (size, mtime) in model_stats.items()):
                raise RuntimeError("Pinned model files changed during run")
        integrity_check = recheck_integrity

        all_checks = {arm: [] for arm in arm_configs}
        zero_ids, first_calls, item_paths = [], set(), {}
        audit_started = time.monotonic()
        for index, (source, item) in enumerate(zip(rows, items, strict=True)):
            current = {"stage": "audit", "index": index, "id": item["id"]}
            check_budget(deadline)
            require_owned_gpu()
            require_ieee()
            prompt, sequences, families = validate_item(
                item, source, scorer, aggregate_scores, semantic_key, serialization_orbit,
                validate_call, METHODS,
            )
            if scorer.prompt_ids(render_messages(source)) != prompt:
                raise ValueError("Prompt differs from the frozen tokenizer rendering")
            item_paths[item["id"]] = {"id": item["id"], "prompt_ids": prompt,
                                      "sequence_ids": sequences, "families": families}
            common = {**current, "split": source["split"], "zero_family": not sequences,
                      "source_item_sha256": reference_hashes[f"items/{index:04d}.json"]}
            if not sequences:
                zero_ids.append(item["id"])
                write_new(args.run / "audit_zero" / f"{index:04d}.json", {
                    **common, "disposition": "zero_family_not_scored",
                    "predictions": item["scoring"]["full"]["predictions"],
                    "included_in_accuracy_denominator": False,
                })
                continue
            for arm, arm_config in arm_configs.items():
                current = {**common, "arm": arm}
                check_budget(deadline)
                arm_started = time.monotonic()
                with context_for(scorer.model, arm_config):
                    precision = effective_precision()
                    scores = asdict(scorer.score(prompt, sequences, method_for(arm_config),
                                               reducer="torch_ce", batch_size=8))
                elapsed = time.monotonic() - arm_started
                # Preserve the complete numerical output before evaluating it.
                write_new(args.run / "audit_raw" / arm / f"{index:04d}.json", {
                    **current, "configuration": arm_config, "elapsed_seconds": elapsed,
                    "effective_precision": precision,
                    "prompt_token_ids": prompt, "sequence_token_ids": sequences, "scores": scores,
                })
                if arm not in first_calls:
                    write_new(args.run / "setup" / ("first_call_" + arm + ".json"), {
                        "arm": arm, "id": item["id"], "configuration": arm_config,
                        "elapsed_seconds": elapsed, "includes_first_call_and_possible_jit": True,
                        "included_in_audit_elapsed": True, "accepted_timing_window": False,
                    })
                    first_calls.add(arm)
                require_ieee()
                validate_layout(scores, prompt, sequences, 8, method_for(arm_config))
                alternative = {"scores": scores, "predictions": aggregate_scores(
                    families, scores["sequence_logprobs"], list(map(len, sequences)))}
                check = accuracy_check(item["scoring"]["full"], alternative, item["id"], source["split"])
                write_new(args.run / "audit" / arm / f"{index:04d}.json", {
                    **current, "alternative": alternative, "check": check,
                })
                all_checks[arm].append(check)
                if index % 10 == 0 or not check["valid"] or not all(check["winner_agreement"].values()):
                    print(json.dumps({**current, "numerical_valid": check["valid"],
                                      "maximum_token_error": check["maximum_token_error"]}), flush=True)
        summaries = {arm: accuracy_summary(checks) for arm, checks in all_checks.items()}
        audit_summary = {
            "valid": summaries["optimized"]["valid"], "arms": summaries,
            "population": len(rows), "scored": len(all_checks["optimized"]),
            "zero_family": len(zero_ids), "zero_family_ids": zero_ids,
            "zero_by_split": {split: sum(item["id"] in zero_ids for item in rows if item["split"] == split)
                              for split in ("development", "test")},
            "control_audit_arms": control_audit_arms,
            "elapsed_seconds": time.monotonic() - audit_started,
            "stage_elapsed_seconds": time.monotonic() - started,
        }
        if any(summary["scored_records"] + len(zero_ids) != 119 for summary in summaries.values()):
            raise ValueError("Full-population scored/zero disposition counts changed")
        write_new(args.run / "audit_summary.json", audit_summary)
        if aliases:
            write_new(args.run / "audit_control_alias.json", {
                "audit_summary_sha256": sha256(args.run / "audit_summary.json"),
                "aliases": [{"control": name, "configuration": control_configs[name],
                             "source_arm": arm, "raw_pattern": "audit_raw/" + arm + "/{index:04d}.json",
                             "checked_pattern": "audit/" + arm + "/{index:04d}.json",
                             "extra_inference_calls": 0, "reason": "identical frozen configuration"}
                            for name, arm in aliases.items()],
            })
        recheck_integrity()
        check_budget(deadline)
        if not audit_summary["valid"]:
            raise RuntimeError("Selected configuration failed whole119 approximate gate; no accepted timing")

        active_controls, excluded = ["native"], {}
        for control in comparators[1:]:
            if summaries[control]["valid"]:
                active_controls.append(control)
            else:
                excluded[control] = "failed whole119 approximate gate"
        fresh, gate_checks = {}, {"optimized": []}
        for control in active_controls[1:]:
            gate_checks[control] = []
        setup_outputs = {arm: {} for arm in gate_checks}
        setup_outputs["native"] = fresh
        setup_started = time.monotonic()
        for fixture in fixtures:
            current = {"stage": "whole_family_gate", "id": fixture["id"]}
            check_budget(deadline)
            require_owned_gpu()
            require_ieee()
            native_started = time.monotonic()
            native = invoke(scorer, fixture, "full", "torch_ce")
            write_new(args.run / "timing_gate_raw/native" / (fixture["id"] + ".json"), {
                "output": native, "elapsed_seconds": time.monotonic() - native_started,
            })
            check_paths(native, item_paths[fixture["id"]])
            fresh[fixture["id"]] = native
            for arm in gate_checks:
                current = {"stage": "whole_family_gate", "id": fixture["id"], "arm": arm}
                check_budget(deadline)
                arm_started = time.monotonic()
                with context_for(scorer.model, arm_configs[arm]):
                    precision = effective_precision()
                    result = invoke(scorer, fixture, method_for(arm_configs[arm]), "torch_ce")
                write_new(args.run / "timing_gate_raw" / arm / (fixture["id"] + ".json"), {
                    "output": result, "elapsed_seconds": time.monotonic() - arm_started,
                    "effective_precision": precision,
                })
                require_ieee()
                check_paths(result, native, method_for(arm_configs[arm]))
                check = accuracy_check(native, result, fixture["id"], "test")
                write_new(args.run / "timing_gates" / arm / (fixture["id"] + ".json"), {
                    "native": native, "alternative": result, "check": check,
                })
                gate_checks[arm].append(check)
                setup_outputs[arm][fixture["id"]] = result
        gate_summaries = {arm: accuracy_summary(checks, require_all_winners=True)
                          for arm, checks in gate_checks.items()}
        write_new(args.run / "timing_setup.json", {
            "elapsed_seconds": time.monotonic() - setup_started, "summaries": gate_summaries,
            "includes_first_whole_family_shape_calls": True, "accepted_timing_window": False,
        })
        if not gate_summaries["optimized"]["valid"]:
            raise RuntimeError("Selected configuration failed whole-family approximate gate")
        for control in list(active_controls[1:]):
            if not gate_summaries[control]["valid"]:
                active_controls.remove(control)
                excluded[control] = "failed whole-family approximate gate"
        active_plan = [block for block in plan if block["control"] in active_controls]
        write_new(args.run / "active_timing_plan.json", {
            "controls": active_controls, "excluded_controls": excluded, "plan": active_plan,
            "selection_equals_torch_control": config == TORCH_CONTROL,
            "control_aliases": aliases,
            "plan_rule": "filter the already frozen schedule; no reseeding or redrawing",
        })
        observations = []
        for block in active_plan:
            for position, arm in enumerate(block["order"]):
                current = {"stage": "timing", "pair": block["pair"], "block": block["block"],
                           "position": position, "arm": arm}
                check_budget(deadline)
                require_owned_gpu()
                require_ieee()

                def pool():
                    if arm == "native":
                        return [invoke(scorer, fixture, "full", "torch_ce") for fixture in fixtures]
                    with context_for(scorer.model, arm_configs[arm]):
                        return [invoke(scorer, fixture, method_for(arm_configs[arm]), "torch_ce")
                                for fixture in fixtures]

                timing, outputs = measure(pool, 1)
                append_observation(args.run / "raw_windows.jsonl", {**current, **timing, "outputs": outputs})
                require_ieee()
                if [output["id"] for output in outputs] != manifest["fixture_ids"]:
                    raise ValueError("Measured window changed or reordered the eight-fixture population")
                checks, repeatability = {}, {}
                for output in outputs:
                    reference = fresh[output["id"]]
                    expected_method = "full" if arm == "native" else method_for(arm_configs[arm])
                    check_paths(output, reference, expected_method)
                    checks[output["id"]] = accuracy_check(reference, output, output["id"], "test")
                    repeatability[output["id"]] = accuracy_check(
                        setup_outputs[arm][output["id"]], output, output["id"], "test")
                summary = accuracy_summary(list(checks.values()), require_all_winners=True)
                observation = {**current, **timing, "outputs": outputs, "checks": checks,
                               "repeatability_checks": repeatability,
                               "repeatability_summary": accuracy_summary(list(repeatability.values()),
                                                                         require_all_winners=True),
                               "repeatability_is_diagnostic_only": True,
                               "summary": summary, "valid": summary["valid"]}
                append_observation(args.run / "windows.jsonl", observation)
                observations.append(observation)
                print(json.dumps({**current, "wall_seconds": timing["wall_seconds"],
                                  "valid": summary["valid"]}), flush=True)
                # Continue the immutable plan after a numerical rejection so
                # all raw windows and costs remain visible. Integrity/runtime
                # exceptions and the fixed wall budget still terminate work.
        primary_valid = (len(observations) == len(active_plan) * 4
                         and all(row["valid"] for row in observations
                                 if row["arm"] in {"native", "optimized"}))
        pairs, pair_status = [], []
        for control in active_controls:
            pair = control + "_vs_optimized"
            pair_rows = [row for row in observations if row["pair"] == pair]
            pair_valid = primary_valid and len(pair_rows) == BLOCKS * 4 and all(row["valid"] for row in pair_rows)
            pair_status.append({"pair": pair, "valid": pair_valid, "windows": len(pair_rows),
                                "rejected_windows": [{key: row[key] for key in ("block", "position", "arm")}
                                                     for row in pair_rows if not row["valid"]]})
            if pair_valid:
                pairs.append(paired_summary(observations, control))
            elif control != "native" and any(not row["valid"] for row in pair_rows if row["arm"] == control):
                excluded[control] = "failed measured-window approximate gate"
        recheck_integrity()
        check_budget(deadline)
        completion = {
            "valid": bool(primary_valid), "pairs": pairs, "pair_status": pair_status,
            "excluded_controls": excluded, "configuration": config, "limits": LIMITS,
            "selection_sha256": selection_hash, "fixture_count": len(fixtures),
            "raw_windows": len(observations), "expected_windows": len(active_plan) * 4,
            "elapsed_seconds": time.monotonic() - started, "answers_accessed": False,
            "scope": "approximate complete score stage; no generation, serving-throughput or task-quality claim",
            "native_control_selected": config == TORCH_CONTROL,
            "control_aliases": aliases,
        }
        write_new(args.run / "complete.json", completion)
        if not primary_valid:
            raise RuntimeError("Completed timing retained, but selected/native numerical gates failed")
        return 0
    except Exception as exc:
        final_integrity = {"checked": False}
        if integrity_check is not None:
            try:
                integrity_check()
                final_integrity = {"checked": True, "valid": True}
            except Exception as integrity_exc:
                final_integrity = {"checked": True, "valid": False, "error": str(integrity_exc)}
        write_new(args.run / "failure.json", {
            "exception": str(exc), "traceback": traceback.format_exc(), "current": current,
            "elapsed_seconds": time.monotonic() - started, "answers_accessed": False,
            "selection_sha256": selection_hash, "candidates_sha256": candidate_hash,
            "source_sha256": source_hashes, "reference_sha256": reference_hashes,
            "final_integrity": final_integrity,
        })
        raise


if __name__ == "__main__":
    with patch("orbitcall.scoring.selected_logprobs", ce_selected_adapter):
        raise SystemExit(main())
