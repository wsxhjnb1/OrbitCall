"""Read the retained discovery profiler on Spark without new inference."""
import json
from pathlib import Path
import sys

if sys.platform != "linux":
    raise RuntimeError("Profile analysis runs on Spark")
root = Path(sys.argv[1])
for path in sorted(root.glob("*_gate.json")):
    row = json.loads(path.read_text())
    print(path.name)
    print([(x["key"], round(x["device_us"] / 1e6, 3), x["calls"])
           for x in sorted(row["operators"], key=lambda x: x["device_us"], reverse=True)[:12]])
    print({key: value["max_token_error"] for key, value in row["checks"].items()})
