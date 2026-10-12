"""Spark-only archival of all run JSON, public metadata, and scrubbed logs.

Scientific JSON is byte-preserved; only console paths are redacted. Weights,
downloaded labels, virtual environments and unrelated projects are excluded.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile


def string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from string_values(key)
            yield from string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from string_values(item)


def main():
    if sys.platform != "linux":
        raise RuntimeError("Evidence export runs on the experiment host")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("evidence/raw_results.tar.gz"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for pattern in ("runs/**/*.json", "runs/**/*.jsonl", "runs/**/*.partial",
                    "runs/**/real_reducer_fixture.pt", "runs/**/source/**/*.py",
                    "runs/**/source/**/*.md", "metadata/**/*.json",
                    "metadata/gorilla/LICENSE", "data/frozen_inputs.json", "logs/*.log", "logs/*.pid"):
        files.extend(args.root.glob(pattern))
    receipt = []
    with tarfile.open(args.output, "w:gz") as archive:
        for path in sorted(set(files)):
            relative = path.relative_to(args.root).as_posix()
            original = path.read_bytes()
            data = original
            if relative.endswith((".json", ".jsonl")) and not relative.endswith("failure.json"):
                # Inspect decoded string values. Scanning JSON escapes as path
                # bytes falsely treats prose such as "Innocent III:\\n" as I:\\.
                text = original.decode("utf-8")
                records = [json.loads(line) for line in text.splitlines() if line.strip()] if relative.endswith(".jsonl") else [json.loads(text)]
                if any("/home/" in value or re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", value)
                       for record in records for value in string_values(record)):
                    raise ValueError(f"Private path in frozen scientific JSON: {relative}")
            if relative.endswith("failure.json"):
                # Failure tracebacks are console evidence rather than scores.
                # Preserve original hash and explicitly mark path redaction.
                text = original.decode("utf-8", errors="replace")
                text = re.sub(r"/home/[^\s\"'\\]+", "<host-path>", text)
                data = text.encode("utf-8")
            if relative.startswith("logs/"):
                text = original.decode("utf-8", errors="replace")
                # Absolute private execution paths do not belong to the public
                # release; retain traceback messages and relative filenames.
                text = re.sub(r"/home/[^\s\"'\x1b]+", "<host-path>", text)
                text = re.sub(r"[A-Z]:[\\/][^\s\"'\x1b]+", "<host-path>", text)
                data = text.encode("utf-8")
            member = tarfile.TarInfo(relative)
            member.size, member.mode, member.mtime = len(data), 0o644, 0
            archive.addfile(member, io.BytesIO(data))
            receipt.append({"file": relative, "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "source_sha256": hashlib.sha256(original).hexdigest(),
                            "console_path_redaction": data != original})
    manifest = {"archive": args.output.name,
                "archive_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "scientific_json_byte_preserved": True,
                "exception": "Console traceback fields in failure.json may have host paths redacted; original hashes retained",
                "files": receipt,
                "excluded": ["model weights", "downloaded labels", "virtual environments", "unrelated projects"]}
    manifest_path = args.output.parent / "MANIFEST.json"
    with manifest_path.open("x") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print(json.dumps({"archive_bytes": args.output.stat().st_size, "files": len(receipt),
                      "sha256": manifest["archive_sha256"]}))


if __name__ == "__main__":
    main()
