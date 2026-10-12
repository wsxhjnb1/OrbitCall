"""Verify public checkpoint blobs without loading a model or using a GPU."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    item = next(m for m in json.loads(args.manifest.read_text(encoding="utf-8"))["models"]
                if m["repo_id"] == args.repo_id)
    records = []
    for record in item["files"]:
        path = args.directory / record["path"]
        if not path.is_file():
            raise FileNotFoundError(record["path"])
        blob = hashlib.sha256()
        git_blob = hashlib.sha1()
        git_blob.update(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                blob.update(chunk)
                git_blob.update(chunk)
        valid = (blob.hexdigest() == record["sha256"] if record["sha256"]
                 else git_blob.hexdigest() == record["git_blob_id"])
        records.append({"path": record["path"], "bytes": path.stat().st_size,
                        "sha256": blob.hexdigest(), "matches_public_revision": valid})
        if not valid:
            raise RuntimeError(f"Public revision mismatch: {record['path']}")
    evidence = {"repo_id": item["repo_id"], "revision": item["revision"],
                "all_public_files_match": True, "files": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"repo_id": item["repo_id"], "revision": item["revision"],
                      "verified_files": len(records)}))


if __name__ == "__main__":
    main()
