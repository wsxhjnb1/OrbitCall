"""Spark-only byte validation of the completed public evidence transform."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("evidence/MANIFEST.json"))
    parser.add_argument("--output", type=Path, default=Path("metadata/public_archive_validation.json"))
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "aarch64":
        raise RuntimeError("Archive validation runs on Spark")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    archive_path = args.manifest.parent / manifest["archive"]
    if digest(archive_path.read_bytes()) != manifest["archive_sha256"]:
        raise ValueError("Completed archive hash mismatch")
    expected = {row["file"]: row for row in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("Duplicate archive-manifest names")
    seen, redacted, unchanged_json = set(), [], 0
    with tarfile.open(archive_path) as archive:
        for member in archive:
            name, path = member.name, PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or name in seen or name not in expected):
                raise ValueError("Unlisted or unsafe archive member")
            if name.startswith(("models/", "data/answers/", ".venv/", "private/")):
                raise ValueError("Excluded material in archive")
            if path.suffix in {".bin", ".safetensors", ".pth"} or (path.suffix == ".pt" and path.name != "real_reducer_fixture.pt"):
                raise ValueError("Weights or unlisted binary in archive")
            published = archive.extractfile(member).read()
            row = expected[name]
            original = Path(name).read_bytes()
            if len(published) != row["bytes"] or digest(published) != row["sha256"]:
                raise ValueError(f"Published member byte mismatch: {name}")
            if digest(original) != row["source_sha256"]:
                raise ValueError(f"Retained original changed: {name}")
            transformed = published != original
            if transformed != row["console_path_redaction"]:
                raise ValueError("Undeclared publication transform")
            if transformed:
                if not (name.startswith("logs/") or name.endswith("failure.json")):
                    raise ValueError("Scientific byte transform detected")
                redacted.append(name)
            if path.suffix in {".json", ".jsonl"} and not name.endswith("failure.json"):
                if transformed:
                    raise ValueError("Scientific JSON is not byte-preserved")
                unchanged_json += 1
            if name.startswith("logs/") or name.endswith("failure.json"):
                if b"/home/" in published or re.search(rb"[A-Z]:[\\/]Users[\\/]", published):
                    raise ValueError("Unredacted private console path")
            seen.add(name)
    if seen != set(expected):
        raise ValueError("Incomplete public archive")
    receipt = {"valid": True, "scope": "Publication byte and member validation; no inference or statistical rerun",
               "host": "original DGX Spark workspace", "archive_sha256": manifest["archive_sha256"],
               "manifest_sha256": digest(args.manifest.read_bytes()), "files": len(seen),
               "unchanged_scientific_json_members": unchanged_json, "redacted_console_members": redacted,
               "validator_sha256": digest(Path(__file__).read_bytes()),
               "models_and_gold_excluded": True, "all_original_and_published_hashes_verified": True}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"valid": True, "files": len(seen), "unchanged_scientific_json_members": unchanged_json,
                      "redacted_console_members": len(redacted)}))


if __name__ == "__main__":
    main()
