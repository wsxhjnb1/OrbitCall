"""Print retained independent analysis; CPU-only on the experiment host."""
import json
from pathlib import Path
import sys

record=json.loads(Path(sys.argv[1]).read_text())
for study in record["studies"]:
    if study["development"]: continue
    print(study["model"])
    print(json.dumps({"quality":{name:row["summary"] for name,row in study["policy_quality"].items()},
                      "audits":study["audit"],"promotion":study["promotion_accepted"]},indent=2))
    for pair in study["timing_comparisons"]:
        summary={name:value for name,value in pair.items()
                 if name not in ("blocks","complete_test_corpus_workload_comparison")}
        corpus=pair["complete_test_corpus_workload_comparison"]
        summary["corpus_workload"]={name:corpus[name] for name in ("episode_count","matched_count","all_matched")}
        summary["changed_corpus_ids"]=[row["id"] for row in corpus["episodes"] if not row["matched_complete_workload"]]
        print(json.dumps(summary,indent=2))
        for block in pair["blocks"]:
            print(json.dumps({name:block[name] for name in ("block","order","control_over_optimized",
                  "scoring_control_over_optimized","all_workloads_matched","optimized_task_success_drop",
                  "wall_seconds_mean_of_two","disjoint_stage_seconds_mean_of_two")},indent=2))
