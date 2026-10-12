"""Download pinned public labels only after every requested prediction freeze.

This asset-transfer stage may run on Windows. It does not parse labels or
evaluate calls. The separate evaluator and every analysis run on Spark.
"""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BASE = f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{REVISION}/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer"
FILES = ("BFCL_v4_simple_python.json", "BFCL_v4_live_simple.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--freeze", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    freezes = []
    models = []
    input_hashes = []
    if len(args.freeze) != 2 or len(set(args.freeze)) != 2:
        raise ValueError("This protocol requires two distinct model freezes")
    for directory in args.freeze:
        receipt_path = directory / "pre_gold_freeze.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        manifest = json.loads((directory / "ranking_manifest.json").read_text(encoding="utf-8"))
        if receipt["answers_accessed"] is not False or receipt["stage"] != "pre-gold":
            raise ValueError("Missing answer-isolated freeze")
        if sha(directory / "ranking_manifest.json") != receipt["ranking_manifest_sha256"]:
            raise ValueError("Ranking manifest freeze mismatch")
        if sha(directory / "predictions.json") != receipt["predictions"]["sha256"]:
            raise ValueError("Prediction freeze mismatch")
        actual_inputs = sha(args.root / "data" / "frozen_inputs.json")
        if receipt["inputs_sha256"] != manifest["inputs_sha256"] or actual_inputs != manifest["inputs_sha256"]:
            raise ValueError("Current input/receipt/manifest freeze mismatch")
        if (manifest.get("answers_accessed") is not False or manifest.get("dtype") != "float32"
                or manifest.get("attention") != "eager" or manifest.get("tf32") is not False
                or manifest.get("selected_method") != "full"):
            raise ValueError("Ranking precision/isolation differs from the protocol")
        predictions = json.loads((directory / "predictions.json").read_text(encoding="utf-8"))
        inputs = json.loads((args.root / "data" / "frozen_inputs.json").read_text(encoding="utf-8"))
        ids = inputs["eligible_ids"]
        if (predictions.get("answers_accessed") is not False
                or predictions.get("selected_scoring_method") != "full"
                or [row["id"] for row in predictions["records"]] != ids
                or receipt["record_count"] != len(ids) or len(receipt["item_sha256"]) != len(ids)):
            raise ValueError("Incomplete frozen prediction population")
        for relative, digest in receipt["item_sha256"].items():
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory.resolve()) or sha(path) != digest:
                raise ValueError("Item checkpoint hash mismatch")
            if json.loads(path.read_text(encoding="utf-8")).get("valid") is not True:
                raise ValueError("Invalid item checkpoint")
        # Locate the corresponding generation by immutable candidate hash,
        # instead of trusting a new caller-computed replacement receipt.
        matches = [p for p in (args.root / "runs").glob("*/candidates.json")
                   if sha(p) == receipt["candidates"]["sha256"]]
        if len(matches) != 1 or manifest["candidates_sha256"] != receipt["candidates"]["sha256"]:
            raise ValueError("Candidate freeze mismatch or ambiguous generation")
        models.append(manifest["model"])
        input_hashes.append(manifest["inputs_sha256"])
        freezes.append({"run": directory.name, "receipt_sha256": sha(receipt_path),
                        "candidates_sha256": receipt["candidates"]["sha256"],
                        "predictions_sha256": receipt["predictions"]["sha256"]})
    if set(models) != {"Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct"} or len(set(input_hashes)) != 1:
        raise ValueError("Both frozen public models must use the same input population")
    receipt_path = args.root / "metadata" / "bfcl_gold_download.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    downloads = []
    for filename in FILES:
        path = args.root / "data" / "answers" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".json.partial")
        if path.exists() or partial.exists():
            raise FileExistsError(path)
        url = f"{BASE}/{filename}"
        request = urllib.request.Request(url, headers={"User-Agent": "OrbitCall-frozen-label-transfer/0.1"})
        with partial.open("xb") as output, urllib.request.urlopen(request, timeout=60) as response:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)
        partial.rename(path)
        downloads.append({"file": filename, "url": url, "sha256": sha(path)})
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as output:
        json.dump({"revision": REVISION, "pre_gold_freezes": freezes,
                   "downloads": downloads, "labels_evaluated": False}, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": "labels_transferred_after_freeze", "files": len(downloads)}))


if __name__ == "__main__":
    main()
