#!/usr/bin/env python3
"""Spark CPU-only release provenance check; no inference or gold evaluator.

Reopen the existing archive without extracting it, verify published artifacts,
and audit the retained demo's execution/serialization interface. Demo final
answers are preserved, never corrected or counted as task-quality evidence.
This receipt is additional to the already sealed raw archive.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tarfile
import time
import tomllib
import xml.etree.ElementTree as ET

from export_vllm_agent_evidence import (
    CONFIRMATIONS, GPU_UUID, RUNS, _CONSOLE_PATH, _PRIVATE, child,
    decode_json, digest, file_digest, payload, require, string_values,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = "metadata/vllm_agent_shape_analysis_v1.json"
ANALYSIS_SHA = "40c72261f6f3f64cb6d7dcf1ea80e09abdbe854f889ac5ef98b064f4c460c728"
ANALYZER_SHA = "c6be3282caee6f31e7cb52f0c0a556d1ae6c7cab800e2471460e0ba917c75444"
RENDERER_SHA = "7b5b4a7e5c46dca66200a3bb3646ce5466dcbc4a839507ca61fc4b939d464447"
ARCHIVE_SHA = "5cbc505a541f291ec2505548e1c3b539195beacc5bb724f1e634ea1100e7a4cf"
ARCHIVE_BYTES = 8486482
ARCHIVE_MEMBERS = 551
REPORT = "evidence/vllm_agent/"
REPORT_FILES = {"RESULTS.md", "agent_vs_scoring_ratios.png", "agent_vs_scoring_ratios.svg",
                "stage_wall_times.png", "stage_wall_times.svg", "summary.json", "independent_analysis.json"}
RULES = {"mass", "max_order", "mean_logprob", "length_normalized", "canonical_alphabetic",
         "fixed_012", "fixed_021", "fixed_102", "fixed_120", "fixed_201", "fixed_210"}
LIMITS = {"max_token_absolute_nats": .05, "mean_token_absolute_nats": .005,
          "max_sequence_absolute_nats": .5, "minimum_winner_agreement": .99}


def read(relative):
    return decode_json(child(ROOT, relative).read_bytes())


def file_receipt(relative):
    path = child(ROOT, relative)
    return dict(file=relative, bytes=path.stat().st_size, sha256=file_digest(path))


def safe_member(name):
    path = PurePosixPath(name)
    return (bool(name) and not path.is_absolute() and ".." not in path.parts
            and "\\" not in name and "\x00" not in name
            and not re.match(r"[A-Za-z]:", name))


def check_analysis_sources(analysis):
    require(file_digest(child(ROOT, ANALYSIS)) == ANALYSIS_SHA
            and analysis["analysis_source_sha256"] == ANALYZER_SHA
            and file_digest(child(ROOT, "scripts/analyze_vllm_agent_shape.py")) == ANALYZER_SHA,
            "Pinned analysis/source changed")
    selection = read("metadata/vllm_agent_shape_queue_20261013_v1_selection.json")
    for relative, expected in selection["source_sha256"].items():
        require(file_digest(child(ROOT, relative)) == expected, "Current frozen source changed: " + relative)
    require(selection["mechanism"] == "native_shape_suffix"
            and selection["held_out_outputs_accessed"] is False
            and selection["configuration_selected_before_confirmation"] is True,
            "Prospective selection contract changed")
    project = tomllib.loads(child(ROOT, "pyproject.toml").read_text(encoding="utf-8"))
    require(project["project"]["entry-points"]["vllm.general_plugins"]["orbitcall_scoring"]
            == "orbitcall.vllm_plugin:register", "Installed-plugin declaration changed")
    for relative, expected in analysis["source_artifact_sha256"].items():
        require(file_digest(child(ROOT, relative)) == expected, "Analyzed original changed: " + relative)
    confirmations = {row["model"]: row for row in analysis["studies"] if not row["development"]}
    require(set(confirmations) == {"Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct"}, "Missing confirmation model")
    require(confirmations["Qwen2.5-3B-Instruct"]["promotion_accepted"] is True
            and confirmations["Qwen2.5-1.5B-Instruct"]["promotion_accepted"] is False
            and analysis["combined_same_workload_promotion_accepted"] is False,
            "A protected failed promotion flag changed")
    require(analysis["limits"] == LIMITS, "Numerical gates changed")
    for study in confirmations.values():
        require(study["complete"]["valid"] is True and study["failure"] is None
                and len(study["windows"]) == 24, "Incomplete confirmed raw windows")
        comparison = study["timing_comparisons"]
        require(len(comparison) == 1 and comparison[0]["all_timed_workloads_matched"] is True,
                "Timed workload identity changed")
        engine = study["engine"]
        require(engine["options"]["dtype"] == "bfloat16"
                and engine["options"]["enforce_eager"] is False
                and engine["options"]["async_scheduling"] is False,
                "Native BF16/graph/legacy scheduling scope changed")
        distribution = importlib.metadata.distribution("vllm")
        for relative, expected in engine["runtime_source_sha256"].items():
            require(file_digest(Path(distribution.locate_file("vllm/" + relative))) == expected,
                    "Installed native runtime changed: " + relative)
    return dict(frozen_source_files=len(selection["source_sha256"]),
        analyzed_originals=len(analysis["source_artifact_sha256"]),
        promotion_by_model={model: row["promotion_accepted"] for model, row in confirmations.items()},
        combined_same_workload_promotion_accepted=False,
        baseline="Adapter installed; unmarked native_default delegates original vLLM BF16 full scoring. No adapter-disabled timing arm.")


def check_archive(analysis):
    manifest = read(REPORT + "MANIFEST.json")
    archive_path = child(ROOT, REPORT + manifest["archive"])
    require(archive_path.stat().st_size == manifest["archive_bytes"] == ARCHIVE_BYTES
            and file_digest(archive_path) == manifest["archive_sha256"] == ARCHIVE_SHA,
            "Pinned raw archive changed")
    require(file_digest(child(ROOT, "scripts/export_vllm_agent_evidence.py"))
            == manifest["exporter_source_sha256"], "Exporter provenance changed")
    require(manifest["independent_analysis_sha256"] == ANALYSIS_SHA
            and manifest["scientific_json_byte_preserved"] is True
            and manifest["source_snapshots_byte_preserved"] is True, "Archive science scope changed")
    expected = {row["file"]: row for row in manifest["files"]}
    require(len(expected) == len(manifest["files"]) == ARCHIVE_MEMBERS, "Manifest membership changed")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        require(len(members) == len({row.name for row in members}) == ARCHIVE_MEMBERS
                and set(expected) == {row.name for row in members}
                and all(row.isfile() and safe_member(row.name) for row in members),
                "Unsafe, duplicate or unplanned archive member")
        for member in members:
            row = expected[member.name]
            data = archive.extractfile(member).read()
            require(len(data) == member.size == row["bytes"] and digest(data) == row["sha256"],
                    "Archived member bytes changed: " + member.name)
            console = member.name.endswith(".log") or member.name.endswith("failure.json")
            require(payload(data, member.name, console) == data,
                    "Published member privacy/redaction invalid: " + member.name)
            if row["scientific_json"] or "/source/" in member.name:
                require(row["byte_preserved"] is True and row["sha256"] == row["source_sha256"],
                        "Scientific/source payload was transformed")
    for relative, original_hash in analysis["source_artifact_sha256"].items():
        require(relative in expected and expected[relative]["source_sha256"] == original_hash,
                "Archive lost the analysis-original hash chain: " + relative)
    require(set(RUNS) == {row["run_id"] for row in manifest["run_dispositions"]},
            "Retired/current run omitted")
    backends = {row["attention_backend"] for row in manifest["attention_backend_evidence"]
                if row["attention_backend"] is not None}
    require(backends == {"FLASH_ATTN"}, "Actual attention backend scope changed")
    first_log = "logs/vllm_agent_export_v1.log"
    require(first_log in expected and child(ROOT, first_log).is_file(), "First export infrastructure log lost")
    log_text = child(ROOT, first_log).read_text(encoding="utf-8")
    require("Required evidence file is missing" in log_text and "THIRD_PARTY_NOTICES.md" in log_text,
            "Expected first export infrastructure failure evidence missing")
    current_log_hash = file_digest(child(ROOT, first_log))
    return dict(archive_sha256=ARCHIVE_SHA, archive_bytes=ARCHIVE_BYTES,
        reopened_members=ARCHIVE_MEMBERS, member_privacy_checked=True,
        analyzed_original_hash_links=len(analysis["source_artifact_sha256"]),
        actual_attention_backends=sorted(backends),
        first_export_infrastructure_log=dict(file=first_log,
            archived_original_sha256=expected[first_log]["source_sha256"], current_sha256=current_log_hash,
            unchanged_since_snapshot=current_log_hash == expected[first_log]["source_sha256"],
            scope="Retained non-scientific export failure; any later console append does not rewrite archive bytes."))


def check_report(analysis):
    receipt = read(REPORT + "RENDER_RECEIPT.json")
    require(receipt["status"] == "complete" and receipt["analysis_sha256"] == ANALYSIS_SHA
            and receipt["renderer_source_sha256"] == RENDERER_SHA
            and file_digest(child(ROOT, "scripts/render_vllm_agent_report.py")) == RENDERER_SHA
            and set(receipt["artifacts"]) == REPORT_FILES
            and receipt["model_or_reference_evaluation_performed"] is False,
            "Final render/source scope changed")
    for name, expected in receipt["artifacts"].items():
        actual = file_receipt(REPORT + name)
        require(actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"],
                "Rendered artifact changed: " + name)
    require(child(ROOT, REPORT + "independent_analysis.json").read_bytes()
            == child(ROOT, ANALYSIS).read_bytes(), "Published analysis differs from raw analyzed evidence")
    summary = read(REPORT + "summary.json")
    models = {row["model"]: row for row in summary["models"]}
    require(summary["analysis_sha256"] == ANALYSIS_SHA
            and summary["analysis_source_sha256"] == ANALYZER_SHA
            and summary["renderer_source_sha256"] == RENDERER_SHA
            and models["Qwen2.5-3B-Instruct"]["promotion_accepted"] is True
            and models["Qwen2.5-1.5B-Instruct"]["promotion_accepted"] is False
            and summary["combined_same_workload_promotion_accepted"] is False,
            "Report altered a failed model/combined disposition")
    markdown = child(ROOT, REPORT + "RESULTS.md").read_text(encoding="utf-8")
    require("No separate adapter-disabled engine performance arm was measured." in markdown
            and "35/36" in markdown and "not passed" in markdown,
            "Baseline/corpus distinction missing from report")
    return dict(rendered_artifacts=7, report_files_with_receipt=8,
                artifacts=[file_receipt(REPORT + name) for name in sorted(REPORT_FILES | {"RENDER_RECEIPT.json"})])


def check_locked_arrays():
    count = 0
    for run_id in CONFIRMATIONS:
        base = "runs/" + run_id + "/"
        native = read(base + "policy_native_default.json")
        checks = read(base + "audit_checks.json")["records"]
        require(len(native["records"]) == 36 and len(checks) == 72, "Locked population changed")
        by_identity = {(row["id"], row["stage"]): row for row in checks}
        require(len(by_identity) == 72, "Duplicate locked round")
        for episode in native["records"]:
            require([row["stage"] for row in episode["rounds"]] == [1, 2], "A real tool round was dropped")
            for row in episode["rounds"]:
                fresh = read(base + "audit/" + episode["id"] + "_r" + str(row["stage"]) + "_native_shape_suffix.json")
                original = row["scores"]
                require(bool(original["suffix_ids"]), "Locked empty pool unexpectedly promoted")
                for field in ("calls", "suffix_ids", "token_logprobs", "sequence_logprobs", "log_masses", "winner"):
                    require(original[field] == fresh[field], "Locked raw score/decision differs: " + field)
                require(all(ids and ids[-1] == 151645 and 151645 not in ids[:-1]
                            for ids in fresh["suffix_ids"]), "A complete EOS target is missing")
                require(all(value == 0 for value in fresh["cached_tokens"]), "Released scorer read prefix cache")
                check = by_identity[(episode["id"], row["stage"])]
                require(check["nonempty"] is True and check["numerically_valid"] is True
                        and check["max_token_error"] == check["sum_token_error"] == check["max_sequence_error"] == 0
                        and check["joint_agreement"] is True
                        and set(check["rule_agreement"]) == RULES and all(check["rule_agreement"].values()),
                        "Stored eleven-rule/zero-error locked result changed")
                count += 1
    require(count == 144, "Both locked audit populations were not retained")
    return dict(nonempty_locked_pools=144, exact_raw_score_arrays_match=True,
                all_eleven_stored_decisions_match=True, zero_recorded_errors=True,
                numerical_or_quality_resampling_performed=False)


def check_demo():
    sys.path.insert(0, str(ROOT / "src"))
    from orbitcall.agent_tasks import execute_tool, tool_specs
    from orbitcall.serialization import parse_call, semantic_key, serialization_orbit
    from tokenizers import Tokenizer

    demo_path = REPORT + "demo_agent_v1.json"
    demo = read(demo_path)
    require(demo["scope"] == "development demonstration, not benchmark evidence"
            and demo["answers_accessed"] is False and demo["policy"] == "native_shape_suffix"
            and len(demo["records"]) == 2, "Demo was relabeled as quality/timing evidence")
    inputs = read("data/agent_inputs_v2.json")
    tasks = {row["id"]: row for row in inputs["records"]}
    selected = [sorted((row for row in tasks.values() if row["split"] == "development"
                and row["family"] == family), key=lambda row: row["id"])[0]["id"]
                for family in ("catalog", "calendar")]
    require([row["id"] for row in demo["records"]] == selected, "Demo used test or output-selected tasks")
    model = PurePosixPath(demo["model"]).name
    require(model == "Qwen2.5-1.5B-Instruct", "Unexpected demo model")
    tokenizer_path = child(ROOT, "models/" + model + "/tokenizer.json")
    model_receipt = read("metadata/model_1p5b_verified.json")
    tokenizer_sha = next(row["sha256"] for row in model_receipt["files"] if row["path"] == "tokenizer.json")
    require(file_digest(tokenizer_path) == tokenizer_sha, "Pinned tokenizer changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    observations = []
    for episode in demo["records"]:
        require(episode["split"] == "development" and [row["stage"] for row in episode["rounds"]] == [1, 2]
                and len(episode["trace"]) == 2, "Demo lost a compulsory tool round")
        for index, row in enumerate(episode["rounds"]):
            require(row["call"] == row["scores"]["winner"] and episode["trace"][index]
                    == {"call": row["call"], "observation": row["observation"]}, "Demo trace was repaired")
            actual = execute_tool(row["call"], tasks[episode["id"]])
            require(actual == row["observation"], "Recorded demo observation differs from actual local tool")
            scores = row["scores"]
            require(all(value == 0 for value in scores["cached_tokens"]), "Demo scorer read prefix cache")
            expected_ids = []
            for call, mass in zip(scores["calls"], scores["log_masses"], strict=True):
                require(len(call["arguments"]) in (2, 3), "Demo escaped the finite scalar family")
                family = [tokenizer.encode(text, add_special_tokens=False).ids + [151645]
                          for text in serialization_orbit(call)]
                require(len(family) == math.factorial(len(call["arguments"]))
                        and len({tuple(ids) for ids in family}) == len(family)
                        and mass["call"] == call and mass["family_size"] == len(family),
                        "Demo serialization family was pruned/merged")
                expected_ids.extend(family)
            require(scores["suffix_ids"] == expected_ids
                    and len(scores["token_logprobs"]) == len(expected_ids), "Incomplete demo score family")
            for ids, values in zip(expected_ids, scores["token_logprobs"], strict=True):
                require(ids[-1] == 151645 and 151645 not in ids[:-1] and len(values) == len(ids)
                        and all(type(value) in (int, float) and math.isfinite(value) for value in values),
                        "Demo EOS/finite logprob target is missing")
            valid = {}
            for output in row["generation"]["outputs"]:
                for schema in tool_specs(row["stage"]):
                    try:
                        call = parse_call(output["text"], schema)
                        valid[semantic_key(call)] = call
                        break
                    except ValueError:
                        pass
            require([valid[key] for key in sorted(valid)] == row["candidates"] == scores["calls"],
                    "Demo replaced generated candidates")
            observations.append(dict(id=episode["id"], stage=row["stage"],
                actual_observation_matches_record=True, complete_family_sequences=len(expected_ids)))
        final = episode["final_generation"]["outputs"][0]
        require(episode["final_response"] == final["text"] and final["token_ids"]
                and final["token_ids"][-1] == 151645, "Demo final text/EOS was edited")
    example_path = "examples/vllm_agent.py"
    source = child(ROOT, example_path).read_text(encoding="utf-8")
    parsed = ast.parse(source)
    forbidden = {"_reference", "evaluate_after_freeze"}
    require(not any((isinstance(node, ast.Name) and node.id in forbidden)
                    or (isinstance(node, ast.Attribute) and node.attr in forbidden)
                    for node in ast.walk(parsed)),
            "Portable demo invokes a gold evaluator")
    require(child(ROOT, "logs/portable_agent_demo_v1.log").is_file(), "Actual portable execution log missing")
    return dict(demo=file_receipt(demo_path), development_ids=selected,
        real_tool_observations=observations, source_at_validation=file_receipt(example_path),
        source_chain_limit="Demo had no pre-run script-hash receipt; current example hash is recorded at validation, not retroactively frozen.",
        final_answers_preserved=True, demo_quality_evaluated=False,
        demo_task_success_claim=False, scope="Interface/execution demonstration only; model final-answer errors remain untouched.")


def check_public_files():
    from PIL import Image

    readme = child(ROOT, "README.md").read_text(encoding="utf-8")
    matches = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    require(len(matches) == 4, "README embedded image count changed")
    images = []
    for relative in matches:
        require(not re.match(r"[a-zA-Z]+://", relative), "Expected a local release image")
        path = child(ROOT, relative)
        if path.suffix == ".svg":
            require(ET.parse(path).getroot().tag.rsplit("}", 1)[-1] == "svg", "Invalid SVG image")
        elif path.suffix == ".png":
            with Image.open(path) as picture:
                require(picture.width > 0 and picture.height > 0, "Empty PNG image")
                picture.verify()
        else:
            raise RuntimeError("Unexpected README image format")
        images.append(file_receipt(relative))
    public = ("README.md", "docs/VLLM_REPRODUCE.md", "examples/vllm_agent.py",
              "THIRD_PARTY_NOTICES.md", "LICENSES/Apache-2.0.txt", "LICENSE")
    for relative in public:
        text = child(ROOT, relative).read_text(encoding="utf-8")
        require(not re.search(r"\b(?:TODO|FIXME)\b", text), "Public entry file still contains TODO: " + relative)
    require("adapter installed" in readme and "strict corpus-consistency gate remains failed" in readme,
            "README hides baseline or failed corpus scope")
    log = "logs/vllm_agent_full_unit_v1.log"
    text = child(ROOT, log).read_text(encoding="utf-8")
    require(re.search(r"\b419 passed in [0-9.]+s\b", text) and not re.search(r"\b(?:FAILED|ERROR)\b", text),
            "Full Spark test evidence is not the retained 419-pass suite")
    return dict(readme_images=images, public_entry_files=[file_receipt(path) for path in public],
                full_unit_log=file_receipt(log), tests_passed=419,
                historical_draft_documents_scanned=False, generated_svg_bytes_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(platform.system() == "Linux" and platform.machine() == "aarch64", "Run on Spark CPU only")
    identity = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                                        "--format=csv,noheader"], text=True, timeout=15).strip().splitlines()
    require(len(identity) == 1 and "GB10" in identity[0] and GPU_UUID in identity[0], "Wrong validation host")
    output = (args.output if args.output.is_absolute() else ROOT / args.output).resolve()
    require(output.is_relative_to(ROOT / "evidence/vllm_agent") and not output.exists(),
            "Use a fresh release receipt in the evidence directory")
    begin = time.perf_counter()
    checks, status = {}, "passed"
    try:
        analysis = read(ANALYSIS)
        checks["analysis_and_frozen_sources"] = check_analysis_sources(analysis)
        checks["raw_archive"] = check_archive(analysis)
        checks["published_report"] = check_report(analysis)
        checks["retained_locked_arrays"] = check_locked_arrays()
        checks["portable_demo"] = check_demo()
        checks["public_files_and_tests"] = check_public_files()
    except Exception as error:
        status = "failed"
        checks["blocking_error"] = _CONSOLE_PATH.sub("<host-path>", str(error))
    result = dict(status=status, execution="Original Spark CPU-only artifact/replay validation",
        validator_source_sha256=file_digest(Path(__file__)), live_host=identity[0],
        analysis_sha256=ANALYSIS_SHA, archive_sha256=ARCHIVE_SHA,
        elapsed_seconds=time.perf_counter() - begin, checks=checks,
        model_inference_performed=False, gpu_context_created=False, gold_reference_evaluator_called=False,
        scientific_files_modified=False, thresholds_changed=False,
        additional_receipt_outside_sealed_raw_archive=True,
        limits="Provenance, archive privacy, stored score arrays and deterministic public-tool execution only; no new speed/quality claim, demo-answer correction or failed-promotion rescue.")
    require(not any(_PRIVATE.search(value) for value in string_values(result)), "Private path in release receipt")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(status=status, output=output.relative_to(ROOT).as_posix(), sha256=file_digest(output))))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
