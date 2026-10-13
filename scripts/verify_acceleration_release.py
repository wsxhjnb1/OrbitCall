"""Spark CPU-only archive and follow-up cost receipt, without new inference."""

import hashlib
import json
from pathlib import Path
import platform
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Evidence validation runs on Spark")
    directory = ROOT / "evidence/acceleration"
    output = directory / "RELEASE_VALIDATION.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    archive = directory / manifest["archive"]
    if digest(archive.read_bytes()) != manifest["archive_sha256"]:
        raise RuntimeError("Archive hash mismatch")
    scientific_count = 0
    with tarfile.open(archive, "r:gz") as stream:
        members = {member.name: member for member in stream.getmembers()}
        if set(members) != {row["file"] for row in manifest["files"]}:
            raise RuntimeError("Archive membership mismatch")
        for row in manifest["files"]:
            original = (ROOT / row["file"]).read_bytes()
            published = stream.extractfile(members[row["file"]]).read()
            if digest(original) != row["source_sha256"] or digest(published) != row["sha256"]:
                raise RuntimeError("Original or published hash mismatch: " + row["file"])
            if not row["console_path_redaction"] and published != original:
                raise RuntimeError("Unreported publication transform")
            if row["file"].endswith((".json", ".jsonl")) and not row["file"].endswith("failure.json"):
                if published != original:
                    raise RuntimeError("Scientific data was transformed")
                scientific_count += 1
    analysis = (directory / "independent_analysis.json").read_bytes()
    if analysis != (ROOT / "metadata/approximate_scoring_independent_analysis_v1.json").read_bytes():
        raise RuntimeError("Public analysis differs from independent receipt")
    if json.loads(analysis)["combined_confirmation_accepted"] is not True:
        raise RuntimeError("Incomplete confirmation")
    intervals = []
    missing = []
    for run in manifest["run_dispositions"]:
        root = ROOT / "runs" / run["run_id"]
        completion = root / "complete.json"
        failure = root / "failure.json"
        record = json.loads((completion if completion.exists() else failure).read_text()) if completion.exists() or failure.exists() else {}
        seconds = record.get("elapsed_seconds")
        if isinstance(seconds, (int, float)):
            intervals.append({"run": run["run_id"], "seconds": seconds,
                              "source": (completion if completion.exists() else failure).relative_to(ROOT).as_posix()})
        else:
            missing.append(run["run_id"])
    receipt = {
        "execution": "Spark CPU only; no model or CUDA workload",
        "archive_sha256": manifest["archive_sha256"],
        "archive_members_verified_against_retained_originals": len(members),
        "scientific_json_files_byte_preserved": scientific_count,
        "independent_analysis_sha256": digest(analysis),
        "combined_confirmation_accepted": True,
        "followup_recorded_process_intervals": intervals,
        "followup_recorded_process_interval_subtotal_seconds": sum(row["seconds"] for row in intervals),
        "followup_runs_without_recorded_complete_process_interval": missing,
        "cost_scope": "Lower bound of explicitly recorded, distinct follow-up process intervals, including retired attempts. Excludes original study, tests, model transfer, unrecorded work and overhead. Nested first-call/window durations are not added again; not pure GPU-busy time.",
        "original_study_measured_component_lower_bound_seconds": 3161.201,
        "unit_suite": "299 passed in logs/unit_acceleration_final_v1.log",
        "example_smoke": "logs/example_prefix_score_v1.json; runtime-only demonstration, no new performance or fidelity result",
        "source_sha256": digest(Path(__file__).read_bytes()),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
