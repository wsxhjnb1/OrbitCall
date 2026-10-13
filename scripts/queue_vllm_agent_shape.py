"""Owned sequential Spark confirmations; never rerun retained run IDs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from vllm_agent_common import ROOT,read_json,sha256,write_new


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--development-run",required=True)
    p.add_argument("--queue-id",required=True)
    args=p.parse_args()
    dev=ROOT/"runs"/args.development_run
    complete=read_json(dev/"complete.json")
    audit=read_json(dev/"audit_summary.json")
    plan=read_json(dev/"plan.json")
    if not complete["valid"] or not all(row["valid"] for row in audit.values()):
        raise RuntimeError("Failed development cannot start confirmations")
    selected=ROOT/"metadata"/(args.queue_id+"_selection.json")
    receipt=dict(mechanism="native_shape_suffix",development_run=dev.relative_to(ROOT).as_posix(),
                 source_sha256=plan["source_sha256"],inputs_sha256=plan["inputs_sha256"],
                 complete_sha256=sha256(dev/"complete.json"),audit_sha256=sha256(dev/"audit_summary.json"),
                 held_out_outputs_accessed=False,configuration_selected_before_confirmation=True)
    write_new(selected,receipt)
    queue_path=ROOT/"metadata"/(args.queue_id+"_queue.json")
    write_new(queue_path,dict(start_time_ns=time.time_ns(),source_sha256=sha256(Path(__file__)),runs=[]))
    begin=time.perf_counter()
    for size,model in (("3b","Qwen2.5-3B-Instruct"),("1p5b","Qwen2.5-1.5B-Instruct")):
        run_id="vllm_agent_shape_confirmation_"+size+"_20261013_v1"
        target=ROOT/"runs"/run_id
        if target.exists():
            raise FileExistsError("Retained run ID requires reconciliation: "+run_id)
        command=[sys.executable,str(ROOT/"scripts/run_vllm_agent_shape.py"),"--model",str(ROOT/"models"/model),
                 "--inputs",str(ROOT/"data/agent_inputs_v2.json"),"--run",str(target),"--selection",str(selected)]
        log=ROOT/"logs"/(run_id+".log")
        with log.open("x") as stream:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL,start_new_session=True)
            write_new(ROOT/"metadata"/(run_id+"_owned.json"),dict(pid=proc.pid,run_id=run_id,command=command,
                       selection_sha256=sha256(selected),started_time_ns=time.time_ns()))
            code=proc.wait()
        if code!=0 or not (target/"complete.json").exists() or not read_json(target/"complete.json")["valid"]:
            write_new(ROOT/"metadata"/(args.queue_id+"_failure.json"),dict(run_id=run_id,exit_code=code,
                        elapsed_seconds=time.perf_counter()-begin))
            raise RuntimeError("Owned confirmation failed; later models not started")
    write_new(ROOT/"metadata"/(args.queue_id+"_complete.json"),dict(valid=True,elapsed_seconds=time.perf_counter()-begin))


if __name__=="__main__": main()
