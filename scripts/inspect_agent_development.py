"""Read retained development outputs without new inference."""
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
for filename in ("generation.json","policy_native_full.json"):
    path=root/filename
    if not path.exists(): continue
    record=json.loads(path.read_text())
    if "records" not in record:
        print(filename,[(row["text"],row["finish_reason"]) for row in record["outputs"]])
    else:
        for episode in record["records"][:2]:
            for row in episode["rounds"]:
                print(episode["id"],row["stage"],"candidates",row["candidates"])
                print([(sample["text"],sample["finish_reason"]) for sample in row["generation"]["outputs"]])
