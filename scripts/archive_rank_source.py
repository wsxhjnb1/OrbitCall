"""Recover byte-exact ranker sources from current files or a retained source tar."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, default=Path("orbitcall_source_v3.tar.gz"))
    args = parser.parse_args()
    sources = json.loads((args.run / "ranking_manifest.json").read_text())["source_sha256"]
    inventory = []
    with tarfile.open(args.fallback) as archive:
        for relative, digest in sources.items():
            current = Path(relative)
            blob = current.read_bytes()
            if sha(blob) != digest:
                member = archive.getmember(relative)
                blob = archive.extractfile(member).read()
            if sha(blob) != digest:
                raise ValueError(f"Cannot reconstruct original source: {relative}")
            output = args.run / "source" / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                if output.read_bytes() != blob:
                    raise ValueError("Refuse to change an existing source snapshot")
            else:
                output.write_bytes(blob)
            inventory.append({"file": relative, "sha256": digest})
    path = args.run / "source_receipt.json"
    data = json.dumps({"byte_exact_manifest_match": True, "files": inventory}, indent=2) + "\n"
    if path.exists() and path.read_text() != data:
        raise FileExistsError(path)
    if not path.exists():
        path.write_text(data)
    print(json.dumps({"run": args.run.name, "exact_sources": len(inventory)}))


if __name__ == "__main__":
    main()
