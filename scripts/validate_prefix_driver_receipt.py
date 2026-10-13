"""CPU-only validation of the driver layout checks against retained GPU outputs."""

import argparse
import json
from pathlib import Path
import platform

from run_approximate_scoring import check_paths, selected_configuration
from benchmark_scoring import write_new, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Retained numerical checks run on Spark")
    selection, plan = selected_configuration(args.development / "selection.json")
    references = json.loads((args.development / "references.json").read_text())
    records = []
    for path in sorted(args.development.glob("*_outputs.json")):
        for output in json.loads(path.read_text()):
            check_paths(output, references[output["id"]], output["scores"]["method"])
            records.append({"file": path.name, "id": output["id"],
                            "method": output["scores"]["method"], "valid": True})
    for output in references.values():
        check_paths(output, output)
    write_new(args.output, {"valid": True, "driver_sha256": sha256(Path(__file__).with_name("run_approximate_scoring.py")),
                           "development_plan_sha256": sha256(args.development / "plan.json"),
                           "records": records, "native_records": len(references),
                           "configuration": selection["configuration"], "gpu_work": False})
    print(json.dumps({"valid": True, "alternative_records": len(records)}))


if __name__ == "__main__":
    main()
