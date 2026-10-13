"""Spark-only follow-up archive; scientific JSON bytes are immutable."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import tarfile

from export_evidence import string_values

RUNS = ["full_scoring_discovery_3b_20261013_v1",
        "tensorlinear_discovery_3b_20261013_v1", "graph_runtime_smoke_v1",
        "graph_runtime_smoke_v2", "graph_scoring_discovery_3b_20261013_v1",
        "approximate_discovery_3b_20261013_v1",
        "prefix_approximate_discovery_3b_20261013_v1",
        "prefix_approximate_confirmation_3b_20261013_v1",
        "prefix_approximate_confirmation_1p5b_20261013_v1"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Archive verification runs on Spark")
    if args.output.exists() or (args.output.parent / "MANIFEST.json").exists():
        raise FileExistsError("Retained archive/manifest already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths, excluded, dispositions = set(), [], []
    for name in RUNS:
        run = args.root / "runs" / name
        if not run.exists():
            raise FileNotFoundError(run)
        complete, failure = (run / "complete.json").exists(), (run / "failure.json").exists()
        if not complete and not failure and not name.startswith("graph_runtime_smoke_"):
            raise RuntimeError("Unreconciled run: " + name)
        dispositions.append({"run_id": name, "complete": complete, "failure": failure})
        for path in run.rglob("*"):
            if not path.is_file():
                continue
            if path.name.endswith(".trace.json"):
                excluded.append({"file": path.relative_to(args.root).as_posix(),
                                 "bytes": path.stat().st_size, "sha256": digest(path.read_bytes()),
                                 "reason": "Chrome traces contain host paths; full originals retained privately"})
            elif path.suffix in (".json", ".jsonl", ".py", ".md"):
                paths.add(path)
        log = args.root / "logs" / (name + ".log")
        if log.exists():
            paths.add(log)
    for name in ("linear_acceleration.py", "graph_scoring.py", "approximation.py", "full_acceleration.py"):
        paths.add(args.root / "src/orbitcall" / name)
    paths.add(args.root / "metadata/acceleration_input_freeze_v1.json")
    paths.add(args.root / "metadata/torch_precision_flagcheck_20261013_v1.json")
    paths.add(args.root / "metadata/torch_precision_flagcheck_source.py")
    paths.add(args.root / "metadata/prefix_driver_validation_v1.json")
    paths.add(args.root / "data/frozen_inputs.json")
    for name in ("analyze_approximate_scoring.py", "export_acceleration_evidence.py",
                 "run_approximate_scoring.py", "screen_approximate_scoring.py", "screen_prefix_approximate.py",
                 "report_acceleration.py", "queue_prefix_confirmation.py", "validate_prefix_driver_receipt.py"):
        paths.add(args.root / "scripts" / name)
    for name in ("unit_acceleration_final_v1.log", "approximation_unit_v1.log", "torch_precision_flagcheck_20261013_v1.log",
                 "prefix_confirmation_queue_v1.log"):
        path = args.root / "logs" / name
        if path.exists():
            paths.add(path)
    receipt = []
    with tarfile.open(args.output, "w:gz") as archive:
        for path in sorted(paths):
            relative = path.relative_to(args.root).as_posix()
            original, data = path.read_bytes(), path.read_bytes()
            console = path.name == "failure.json" or path.suffix == ".log"
            if path.suffix in (".json", ".jsonl") and not console:
                text = data.decode("utf-8")
                records = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else [json.loads(text)]
                if any("/home/" in value or re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", value)
                       for record in records for value in string_values(record)):
                    raise RuntimeError("Private path in scientific data: " + relative)
            if console:
                text = data.decode("utf-8", errors="replace")
                text = re.sub(r"/home/[^\s\"'\\\x1b]+", "<host-path>", text)
                text = re.sub(r"[A-Z]:[\\/][^\s\"'\x1b]+", "<host-path>", text)
                data = text.encode("utf-8")
            member = tarfile.TarInfo(relative)
            member.size, member.mode, member.mtime = len(data), 0o644, 0
            archive.addfile(member, io.BytesIO(data))
            receipt.append({"file": relative, "bytes": len(data), "sha256": digest(data),
                            "source_sha256": digest(original), "console_path_redaction": data != original})
    # Re-open all members and verify hashes after writing the gzip stream.
    with tarfile.open(args.output, "r:gz") as archive:
        members = {x.name: x for x in archive.getmembers()}
        if set(members) != {x["file"] for x in receipt}:
            raise RuntimeError("Archive membership mismatch")
        for item in receipt:
            if digest(archive.extractfile(members[item["file"]]).read()) != item["sha256"]:
                raise RuntimeError("Archive payload mismatch")
    manifest = {"archive": args.output.name, "archive_sha256": digest(args.output.read_bytes()),
                "archive_bytes": args.output.stat().st_size, "scientific_json_byte_preserved": True,
                "files": receipt, "run_dispositions": dispositions, "excluded_traces": excluded,
                "excluded": ["model weights", "gold labels", "virtual environments", "unrelated projects"],
                "validated_archive_members": len(receipt)}
    with (args.output.parent / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: manifest[k] for k in ("archive_bytes", "archive_sha256", "validated_archive_members")}))


if __name__ == "__main__":
    main()
