#!/usr/bin/env python3
"""Export the completed native-BF16 Agent study on Spark, without inference.

Run only after the queue and independent analysis finish. Scientific JSON,
JSONL and source snapshots retain their exact bytes. Only console logs and
failure traceback JSON may have private paths redacted; original and exported
hashes are recorded. Owned process/command receipts are excluded with hashes.
No model weights, downloaded answers, virtual environments or other runs are
walked. The archive is exclusive, bounded, and reopened member by member.

Example (on the original Spark):
  python scripts/export_vllm_agent_evidence.py --root . \
    --queue-id vllm_agent_shape_queue_20261013_v1 \
    --analysis metadata/vllm_agent_shape_analysis_v1.json \
    --extra-log logs/vllm_shape_suffix_cpu_v1.log \
    --extra-log logs/vllm_agent_full_unit_v1.log
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import tarfile


RUNS = (
    "vllm_agent_smoke_3b_20261013_v1",
    "vllm_agent_smoke_3b_20261013_v2",
    "vllm_agent_smoke_3b_20261013_v3",
    "vllm_agent_dev_3b_20261013_v1",
    "vllm_agent_dev_3b_20261013_v2",
    "vllm_agent_shape_dev_3b_20261013_v1",
    "vllm_agent_shape_confirmation_3b_20261013_v1",
    "vllm_agent_shape_confirmation_1p5b_20261013_v1",
)
CONFIRMATIONS = RUNS[-2:]
GPU_UUID = "GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4"
MAX_RAW_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
TEXT_SUFFIXES = {".py", ".md", ".toml", ".sty", ".tex", ".sh", ".txt",
                 ".yaml", ".yml", ".ini", ".cfg", ".lock"}
RUNTIME_FILES = (
    "sampling_params.py", "v1/worker/gpu_model_runner.py",
    "v1/worker/gpu_input_batch.py", "v1/core/kv_cache_manager.py",
    "v1/core/kv_cache_coordinator.py", "v1/engine/logprobs.py",
    "v1/sample/sampler.py", "model_executor/layers/logits_processor.py",
)
RUNTIME_PROVENANCE = (*RUNTIME_FILES, "platforms/cuda.py",
    "v1/attention/backends/flash_attn.py", "model_executor/models/qwen2.py",
    "model_executor/layers/linear.py", "plugins/__init__.py", "config/model.py")
SCRIPTS = (
    "export_vllm_agent_evidence.py", "analyze_vllm_agent_shape.py",
    "analyze_vllm_agent.py", "render_vllm_agent_report.py",
    "run_vllm_agent_shape.py", "run_vllm_agent.py", "smoke_vllm_agent.py",
    "vllm_agent_common.py", "audit_full_triton.py", "freeze_agent_tasks.py",
    "queue_vllm_agent_shape.py", "inspect_agent_development.py",
)
DOCS = ("VLLM_AGENT_PROTOCOL.md", "VLLM_AGENT_REFINEMENT.md",
        "VLLM_AGENT_SHAPE_PROTOCOL.md")
_PRIVATE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"
    r"|/(?:home|root|Users|tmp|private|run/user)/"
    r"|file:///", re.IGNORECASE)
_CONSOLE_PATH = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"'\x1b]+"
    r"|/(?:home|root|Users|tmp|private|run/user)/[^\s\"'\x1b]+"
    r"|file:///[^\s\"'\x1b]+", re.IGNORECASE)
_SOURCE_PRIVATE = re.compile(
    r"/(?:home|Users)/[A-Za-z0-9_.-]+/"
    r"|(?<![A-Za-z0-9])[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9_.-]+[\\/]",
    re.IGNORECASE)
_BACKEND = re.compile(r"\bUsing ([A-Z0-9_]+) attention backend\b")
_FA_VERSION = re.compile(r"\bUsing FlashAttention version [0-9]+\b")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def decode_json(data):
    return json.loads(data.decode("utf-8"), object_pairs_hook=unique_keys,
                      parse_constant=reject_constant)


def string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from string_values(key)
            yield from string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from string_values(child)


def scientific_privacy(data, relative, suffix):
    # Inspect actual strings; do not mistake JSON's escaped newlines for paths.
    records = ([decode_json(line) for line in data.splitlines() if line.strip()]
               if suffix == ".jsonl" else [decode_json(data)])
    require(not any(_PRIVATE.search(value) for record in records
                    for value in string_values(record)),
            "Private path in immutable scientific JSON: " + relative)


def redact_value(value):
    if isinstance(value, str):
        return _CONSOLE_PATH.sub("<host-path>", value)
    if isinstance(value, list):
        return [redact_value(child) for child in value]
    if isinstance(value, dict):
        require(not any(_PRIVATE.search(key) for key in value),
                "Private path in console JSON key; preserve privately instead")
        return {key: redact_value(child) for key, child in value.items()}
    return value


def payload(original, relative, console):
    suffix = PurePosixPath(relative).suffix
    if console and suffix == ".json":
        source = decode_json(original)
        redacted = redact_value(source)
        data = (original if redacted == source else
                (json.dumps(redacted, ensure_ascii=False, indent=2,
                            allow_nan=False) + "\n").encode("utf-8"))
        require(not any(_PRIVATE.search(value) for value in string_values(redacted)),
                "Console JSON path redaction incomplete: " + relative)
        return data
    if console:
        text = _CONSOLE_PATH.sub("<host-path>", original.decode("utf-8"))
        require(not _PRIVATE.search(text), "Console path redaction incomplete: " + relative)
        return text.encode("utf-8")
    if suffix in (".json", ".jsonl"):
        scientific_privacy(original, relative, suffix)
    elif suffix in TEXT_SUFFIXES or PurePosixPath(relative).name in ("LICENSE", "NOTICE"):
        require(not _SOURCE_PRIVATE.search(original.decode("utf-8")),
                "Private user path in immutable source/text: " + relative)
    return original


def child(root, relative):
    name = PurePosixPath(relative)
    require(not name.is_absolute() and ".." not in name.parts,
            "Expected an in-repository relative path")
    path = root.joinpath(*name.parts)
    require(path.resolve().is_relative_to(root), "Path escapes the repository: " + relative)
    require(not path.is_symlink() and not any(parent.is_symlink()
                for parent in path.parents if parent != root and parent.is_relative_to(root)),
            "Symlinks are not evidence sources: " + relative)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path,
                        default=Path("evidence/vllm_agent/raw_results_v1.tar.gz"))
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--run", action="append", choices=RUNS)
    parser.add_argument("--extra-log", action="append", default=[])
    args = parser.parse_args()
    require(platform.system() == "Linux" and platform.machine() == "aarch64",
            "Evidence export runs only on the original Spark, without inference")
    identity = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"
    ], text=True, timeout=15).strip().splitlines()
    require(len(identity) == 1 and "GB10" in identity[0] and GPU_UUID in identity[0],
            "Wrong live export host/GPU")
    root = args.root.resolve()
    require(re.fullmatch(r"vllm_agent_shape_queue_[0-9]{8}_v[0-9]+", args.queue_id),
            "Unexpected queue ID")
    run_names = tuple(args.run or RUNS)
    require(len(set(run_names)) == len(run_names) and set(run_names) == set(RUNS),
            "Export all eight predeclared runs; no failed-run omission")
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    require(output.is_relative_to(root / "evidence/vllm_agent")
            and output.name == "raw_results_v1.tar.gz", "Unexpected archive destination")
    manifest_path = output.parent / "MANIFEST.json"
    temporary = output.with_name(output.name + ".partial")
    manifest_partial = manifest_path.with_name(manifest_path.name + ".partial")
    require(not any(path.exists() for path in
                    (output, manifest_path, temporary, manifest_partial)),
            "Retained archive/manifest/partial exists; reconcile without overwrite")

    # Entries carry source paths privately in memory, never in the public receipt.
    entries, cached, excluded, dispositions = {}, {}, [], []

    def add(relative, *, console=False):
        path = child(root, relative)
        require(path.is_file(), "Required evidence file is missing: " + relative)
        previous = entries.setdefault(relative, (path, console))
        require(previous == (path, console), "Conflicting export mode: " + relative)
        return path

    def read(relative):
        path = child(root, relative)
        if relative not in cached:
            cached[relative] = path.read_bytes()
        return decode_json(cached[relative])

    def exclude(relative, reason):
        path = child(root, relative)
        excluded.append(dict(file=relative, bytes=path.stat().st_size,
                             source_sha256=file_digest(path), reason=reason))

    for run_id in RUNS:
        run_prefix = "runs/" + run_id
        directory = child(root, run_prefix)
        require(directory.is_dir(), "Predeclared run is missing: " + run_id)
        complete_name, failure_name = run_prefix + "/complete.json", run_prefix + "/failure.json"
        complete = read(complete_name) if child(root, complete_name).is_file() else None
        failure = read(failure_name) if child(root, failure_name).is_file() else None
        require(complete is not None or failure is not None, "Run is still active/unreconciled: " + run_id)
        terminal_name = complete_name if complete is not None else failure_name
        terminal = complete if complete is not None else failure
        elapsed = terminal.get("elapsed_seconds")
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
                "Missing finite terminal process interval: " + run_id)
        dispositions.append(dict(run_id=run_id, complete=complete is not None,
            failure=failure is not None, terminal_receipt=terminal_name,
            terminal_source_sha256=digest(cached[terminal_name]), process_elapsed_seconds=elapsed,
            terminal_status="complete" if complete is not None else "failure",
            smoke_or_retired_mode=run_id in RUNS[:5],
            completion_valid_field=complete.get("valid") if complete else None,
            interpretation=("Retired API/development evidence; no accepted current-mode speed claim"
                            if run_id in RUNS[:5] else "See independent analysis for numerical/utility acceptance")))
        plan_name = run_prefix + "/plan.json"
        if child(root, plan_name).is_file():
            plan = read(plan_name)
            for relative, expected in plan["source_sha256"].items():
                snapshot = add(run_prefix + "/source/" + relative)
                require(file_digest(snapshot) == expected, "Source snapshot hash differs: " + run_id + "/" + relative)
        for path in sorted(directory.rglob("*")):
            if not path.is_file() and not path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            child(root, relative)
            if path.name.endswith("_owned.json"):
                exclude(relative, "Owned PID/argv receipt; private command paths, not scientific scores")
            elif path.name.endswith(".trace.json") or path.name.endswith(".partial"):
                exclude(relative, "Unrequested trace/partial artifact; original retained privately")
            elif path.suffix in (".json", ".jsonl", ".log") or "/source/" in relative and (
                    path.suffix in TEXT_SUFFIXES or path.name in ("LICENSE", "NOTICE")):
                add(relative, console=path.name == "failure.json" or path.suffix == ".log")
            else:
                exclude(relative, "Outside the declared text/JSON/source evidence allowlist")
        add("logs/" + run_id + ".log", console=True)

    for relative in ("data/agent_inputs_v1.json", "data/agent_inputs_v2.json",
                     "metadata/public_models.json", "metadata/model_3b_verified.json",
                     "metadata/model_1p5b_verified.json", "pyproject.toml", "AGENTS.md",
                     "LICENSE", "THIRD_PARTY_NOTICES.md", "LICENSES/Apache-2.0.txt"):
        add(relative)
    for name in SCRIPTS:
        add("scripts/" + name)
    for name in DOCS:
        add("docs/" + name)
    for path in sorted(child(root, "src/orbitcall").glob("*.py")):
        add(path.relative_to(root).as_posix())
    for name in ("test_vllm_plugin.py", "test_vllm_scoring.py", "test_agent_tasks.py"):
        add("tests/" + name)

    queue_prefix = "metadata/" + args.queue_id
    selection_name = queue_prefix + "_selection.json"
    queue_name = queue_prefix + "_queue.json"
    selection, queue = read(selection_name), read(queue_name)
    require(selection["mechanism"] == "native_shape_suffix"
            and selection["held_out_outputs_accessed"] is False
            and selection["configuration_selected_before_confirmation"] is True,
            "Missing prospective development selection")
    require(queue["source_sha256"] == file_digest(child(root, "scripts/queue_vllm_agent_shape.py")),
            "Queue source differs from its receipt")
    development_prefix = "runs/" + RUNS[5]
    development_plan = read(development_prefix + "/plan.json")
    development_complete = read(development_prefix + "/complete.json")
    development_audit = read(development_prefix + "/audit_summary.json")
    require(selection["development_run"] == development_prefix
            and selection["source_sha256"] == development_plan["source_sha256"]
            and selection["inputs_sha256"] == development_plan["inputs_sha256"]
            and selection["inputs_sha256"] == file_digest(child(root, "data/agent_inputs_v2.json"))
            and selection["complete_sha256"] == digest(cached[development_prefix + "/complete.json"])
            and selection["audit_sha256"] == digest(cached[development_prefix + "/audit_summary.json"])
            and development_complete["valid"] is True
            and all(row["valid"] is True for row in development_audit.values()),
            "Development selection receipt chain changed")
    for run_id in CONFIRMATIONS:
        require(read("runs/" + run_id + "/plan.json")["selection_sha256"]
                == digest(cached[selection_name]), "Confirmation selection receipt differs")
    add(selection_name); add(queue_name)
    queue_terminals = [queue_prefix + suffix for suffix in ("_complete.json", "_failure.json")
                       if child(root, queue_prefix + suffix).is_file()]
    require(bool(queue_terminals), "Queue is still active/unreconciled")
    for relative in queue_terminals:
        add(relative, console=relative.endswith("_failure.json"))
    for path in sorted(child(root, "metadata").glob("vllm_agent*_owned.json")):
        exclude(path.relative_to(root).as_posix(), "Owned PID/argv receipt excluded; hash preserves provenance")

    analysis_path = (args.analysis if args.analysis.is_absolute() else root / args.analysis).resolve()
    require(analysis_path.is_relative_to(root / "metadata")
            and re.fullmatch(r"vllm_agent_shape_analysis_v[0-9]+\.json", analysis_path.name),
            "Unexpected independent analysis artifact")
    analysis_name = analysis_path.relative_to(root).as_posix()
    analysis = read(analysis_name)
    require(analysis["analysis_source_sha256"] == file_digest(child(root, "scripts/analyze_vllm_agent_shape.py"))
            and analysis["both_confirmation_models_present"] is True
            and set(CONFIRMATIONS) <= {row["run"] for row in analysis["studies"]},
            "Final independent analysis does not cover both frozen confirmation runs")
    add(analysis_name)
    receipts = child(root, "metadata/" + analysis_path.stem + "_quality_receipts")
    require(receipts.is_dir(), "Independent quality-freeze receipts missing")
    for path in sorted(receipts.iterdir()):
        require(path.is_file() and re.fullmatch(r"[0-9]{4}\.json", path.name),
                "Unexpected quality receipt member")
        add(path.relative_to(root).as_posix())

    # Narrow console-only log allowlist; no PID files or arbitrary logs directory.
    for path in sorted(child(root, "logs").glob("vllm_*.log")):
        require(re.fullmatch(r"vllm_[A-Za-z0-9_]+\.log", path.name), "Unexpected log name")
        add(path.relative_to(root).as_posix(), console=True)
    for name in ("unit_full_final_v1.log", "unit_release_default_full_v1.log"):
        if child(root, "logs/" + name).is_file():
            add("logs/" + name, console=True)
    for relative in args.extra_log:
        require(re.fullmatch(r"logs/(?:vllm_[A-Za-z0-9_]+|unit_full_final_v[0-9]+)\.log", relative),
                "Extra log is outside the console allowlist")
        add(relative, console=True)

    versions = {name: importlib.metadata.version(name)
                for name in ("vllm", "torch", "transformers", "triton")}
    require(versions["vllm"].split("+")[0] == "0.29.0", "Unexpected vLLM distribution")
    distribution = importlib.metadata.distribution("vllm")
    for relative in RUNTIME_PROVENANCE:
        source = Path(distribution.locate_file("vllm/" + relative)).resolve()
        require(source.is_file(), "Installed public runtime source missing")
        entries["runtime/vllm0.29.0/" + relative] = (source, False)
    for run_id in RUNS[5:]:
        engine = read("runs/" + run_id + "/engine.json")
        require(engine["runtime"] == versions, "Export environment differs from executed study")
        for relative, expected in engine["runtime_source_sha256"].items():
            require(relative in RUNTIME_FILES
                    and file_digest(entries["runtime/vllm0.29.0/" + relative][0]) == expected,
                    "Installed runtime differs from executed source: " + relative)

    analysis_hashes = analysis["source_artifact_sha256"]
    require(type(analysis_hashes) is dict and bool(analysis_hashes),
            "Independent analysis has no original artifact hash map")
    excluded_names = {row["file"] for row in excluded}
    for relative, expected in analysis_hashes.items():
        require(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected),
                "Invalid independent original artifact hash")
        path = child(root, relative)
        original_hash = (digest(cached[relative]) if relative in cached else file_digest(path))
        require(original_hash == expected, "Independent analysis is stale: " + relative)
        require(relative in entries or relative in excluded_names,
                "Analyzed artifact is outside the explicit archive/exclusion receipts: " + relative)

    total = sum(path.stat().st_size for path, _ in entries.values())
    require(total <= MAX_RAW_BYTES, "Archive raw-byte budget exceeded; do not silently omit evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt, backend_lines = [], []
    with temporary.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(mode="w", fileobj=compressed) as archive:
                for relative, (path, console) in sorted(entries.items()):
                    original = cached.get(relative)
                    if original is None:
                        original = path.read_bytes()
                    data = payload(original, relative, console)
                    if console and path.suffix == ".log":
                        for line in data.decode("utf-8").splitlines():
                            backend = _BACKEND.search(line) if "cuda.py" in line else None
                            version = _FA_VERSION.search(line)
                            if backend or version:
                                backend_lines.append(dict(file=relative, line=line,
                                    attention_backend=backend.group(1) if backend else None,
                                    attention_version=version.group(0) if version else None))
                    member = tarfile.TarInfo(relative)
                    member.size, member.mode, member.mtime = len(data), 0o644, 0
                    archive.addfile(member, io.BytesIO(data))
                    receipt.append(dict(file=relative, bytes=len(data), sha256=digest(data),
                        source_sha256=digest(original), console_path_redaction=data != original,
                        byte_preserved=data == original, scientific_json=not console and path.suffix in (".json", ".jsonl")))
        raw.flush(); os.fsync(raw.fileno())
    require(all(any(row["file"] == "logs/" + run_id + ".log"
                        and row["attention_backend"] is not None for row in backend_lines)
                for run_id in RUNS[5:]), "A current study lacks its explicit cuda.py attention backend line")
    require(temporary.stat().st_size <= MAX_ARCHIVE_BYTES, "Compressed archive budget exceeded")
    with tarfile.open(temporary, "r:gz") as archive:
        members = archive.getmembers()
        require(len(members) == len({member.name for member in members}) == len(receipt)
                and {member.name for member in members} == set(entries)
                and all(member.isfile() for member in members), "Archive membership mismatch")
        by_name = {member.name: member for member in members}
        for item in receipt:
            stream = archive.extractfile(by_name[item["file"]])
            require(stream is not None, "Missing archive payload")
            data = stream.read()
            require(len(data) == item["bytes"] and digest(data) == item["sha256"],
                    "Reopened archive payload mismatch: " + item["file"])
            if item["file"] in analysis_hashes:
                require(item["source_sha256"] == analysis_hashes[item["file"]],
                        "Archived original differs from independent analysis: " + item["file"])
            require(file_digest(entries[item["file"]][0]) == item["source_sha256"],
                    "Source changed during export: " + item["file"])
    retired_cost = math.fsum(row["process_elapsed_seconds"] for row in dispositions[:5])
    current_cost = math.fsum(row["process_elapsed_seconds"] for row in dispositions[5:])
    manifest = dict(format_version=1, archive=output.name, archive_sha256=file_digest(temporary),
        archive_bytes=temporary.stat().st_size, raw_input_bytes=total,
        exporter_source_sha256=file_digest(Path(__file__)), live_export_host=identity[0],
        runtime_versions=versions, independent_analysis_file=analysis_name,
        independent_analysis_sha256=digest(cached[analysis_name]),
        independent_analysis_original_hashes_verified=len(analysis_hashes),
        independent_analysis_hash_coverage=dict(
            archived=sum(relative in entries for relative in analysis_hashes),
            excluded_with_receipt=sum(relative in excluded_names for relative in analysis_hashes)),
        scientific_json_byte_preserved=True, source_snapshots_byte_preserved=True,
        console_exception="Only failure JSON decoded string values and console logs redact private paths; failure JSON may be reserialized. Original/exported SHA256 retained.",
        files=receipt, validated_archive_members=len(receipt), run_dispositions=dispositions,
        excluded_file_receipts=excluded,
        attention_backend_evidence=backend_lines,
        attention_backend_rule="Only explicit cuda.py 'Using NAME attention backend' lines identify attention; FlashAttention version lines qualify it. Sampler/autotune FlashInfer lines do not.",
        process_cost_lower_bound=dict(retired_attempt_seconds=retired_cost,
            current_attempt_seconds=current_cost, combined_seconds=math.fsum((retired_cost, current_cost)),
            rule="One complete-or-failure terminal process elapsed interval per distinct run. Completion takes precedence if both exist. Never sum nested initialization, policy, audit, window or queue elapsed times.",
            exclusions="Setup/download/test/export/analysis time and unrecorded shutdown overhead; not GPU-busy time or total project cost."),
        excluded_categories=["Model weights/tokenizer binaries", "downloaded gold answers", "virtual environments",
            "private owned argv/PID receipts (hashes only)", "unrelated runs/projects", "trace/partial/nonallowlisted binaries"],
        claim_scope="Archive integrity and retained evidence only; no new speed, accuracy, numerical acceptance or novelty claim.")
    with manifest_partial.open("xb") as stream:
        stream.write((json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, output)
    os.link(manifest_partial, manifest_path)
    temporary.unlink(); manifest_partial.unlink()
    print(json.dumps({key: manifest[key] for key in
                      ("archive", "archive_bytes", "archive_sha256", "validated_archive_members")}, indent=2))


if __name__ == "__main__":
    main()
