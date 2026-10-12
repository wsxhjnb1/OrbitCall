#!/usr/bin/env python3
"""Stage only revision-pinned public BFCL inputs; never download answers.

Windows may run `download`. Run `freeze` on DGX Spark before gold is accessed.
The two commands are deliberately separate so downloading cannot select data.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Permit the download-only command before an offline editable installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from orbitcall.bfcl import (  # noqa: E402
    DATA_BASE,
    GORILLA_BASE,
    GORILLA_REVISION,
    LICENSE_SHA256,
    SOURCE_FILES,
    SOURCE_SHA256,
    freeze_inputs,
    sha256_file,
    write_frozen_json,
)


def _download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f"Refusing to replace an unrecognized existing file: {destination.name}")
        return
    partial = destination.with_name(destination.name + ".part")
    # Keep a failed transfer as evidence instead of silently replacing it.
    with partial.open("xb") as output:
        request = urllib.request.Request(url, headers={"User-Agent": "OrbitCall-input-staging/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)
    if sha256_file(partial) != expected_sha256:
        raise ValueError(f"Downloaded hash mismatch; retained {partial.name}")
    if destination.exists():
        raise ValueError("Destination appeared during download; refusing to overwrite")
    partial.rename(destination)


def download_inputs(root: Path) -> dict:
    """Allowlist: exactly two input files and the repository's root LICENSE."""
    receipts = []
    for category, filename in SOURCE_FILES.items():
        relative_path = Path("data") / "raw" / filename
        url = f"{DATA_BASE}/{filename}"
        _download_verified(url, root / relative_path, SOURCE_SHA256[category])
        receipts.append({"category": category, "path": relative_path.as_posix(),
                         "url": url, "sha256": SOURCE_SHA256[category]})
    license_relative = Path("metadata") / "gorilla" / "LICENSE"
    license_url = f"{GORILLA_BASE}/LICENSE"
    _download_verified(license_url, root / license_relative, LICENSE_SHA256)
    receipt = {
        "format_version": 1, "gorilla_revision": GORILLA_REVISION,
        "source_repository": "https://github.com/ShishirPatil/gorilla",
        "data_citation": "Patil et al. (2025), The Berkeley Function Calling Leaderboard, ICML",
        "citation_url": "https://proceedings.mlr.press/v267/patil25a.html",
        "data_license_basis": "Gorilla repository root Apache-2.0 license at the pinned revision",
        "license": {"identifier": "Apache-2.0", "url": license_url,
                    "path": license_relative.as_posix(), "sha256": LICENSE_SHA256},
        "inputs": receipts, "answers_downloaded": False,
        "eligibility_or_split_executed": False,
        "note": "JSONL input byte hashes only; no answer endpoint is used by this script.",
    }
    write_frozen_json(root / "metadata" / "bfcl_inputs_download.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download allowlisted public inputs and license only")
    freeze_parser = commands.add_parser("freeze", help="Spark-only input eligibility and schema split")
    freeze_parser.add_argument("--output", type=Path, default=Path("data/frozen_inputs.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "download":
        receipt = download_inputs(root)
        print(json.dumps(receipt, indent=2))
    else:
        if sys.platform != "linux":
            parser.error("The project protocol permits freeze/analysis only on DGX Spark (Linux).")
        output = args.output if args.output.is_absolute() else root / args.output
        manifest = freeze_inputs(
            {category: root / "data" / "raw" / filename
             for category, filename in SOURCE_FILES.items()}, output,
        )
        print(json.dumps({"eligible": len(manifest["eligible_ids"]),
                          "development": len(manifest["development_ids"]),
                          "test": len(manifest["test_ids"]),
                          "counts_by_category": manifest["counts_by_category"],
                          "freeze_sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
