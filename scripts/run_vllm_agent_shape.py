"""Prospective BF16 vLLM closed-loop Agent and locked-score study on Spark."""

import argparse
from collections import Counter
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import signal
import time
import traceback

from vllm_agent_common import ROOT,ENGINE,LIMITS,initialize,owned_gpu_check,shutdown,parse_candidates,read_json,sha256,write_new
from orbitcall.agent_tasks import render_messages,execute_tool,tool_specs
from orbitcall.serialization import semantic_key
from orbitcall.vllm_scoring import SHAPE_ARMS,VLLMScorer,decision_predictions

SYSTEM_ARMS=SHAPE_ARMS
POLICIES=(*SYSTEM_ARMS,"vote4","greedy1")
CONTROLS=("native_default",)
BLOCKS=6


def append(path,value):
    with path.open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")
        stream.flush()


def watchdog(signum,frame):
    raise TimeoutError("Predeclared per-model study budget exhausted")


def agent_messages(task,history,stage):
    messages=render_messages(task,history,stage)
    if stage<3:
        tools=", ".join(schema["name"] for schema in tool_specs(stage))
        messages.append({"role":"user","content":f'Current step: perform tool round {stage} now. Return only one JSON tool call choosing from {tools}. Required structure: {{"name":"<available_tool>","arguments":{{}}}}. Replace <available_tool> with one available tool name and fill arguments with ALL required tool parameters. All parameters MUST be nested inside arguments. Never put parameters at the top level or use a key named parameters. The only top-level keys are name and arguments. Do not give the final answer yet. Use the request and actual observations above.'})
    return messages


def run_pool(scorer,tasks,policy,key):
    # Cache resets are counted. Every cohort's warm P comes from real generation.
    begin=time.perf_counter()
    if not scorer.llm.reset_prefix_cache():
        raise RuntimeError("A fresh window requires a completed prefix-cache reset")
    episodes=[]
    stage_totals=Counter()
    for offset in range(0,len(tasks),4):
        cohort=tasks[offset:offset+4]
        cohort_start=time.perf_counter()
        histories=[[] for _ in cohort]
        records=[dict(id=task["id"],split=task["split"],family=task["family"],policy=policy,rounds=[]) for task in cohort]
        salts=[key+":"+task["id"] for task in cohort]
        for stage in (1,2):
            tick=time.perf_counter()
            messages=[agent_messages(task,history,stage) for task,history in zip(cohort,histories,strict=True)]
            generated=scorer.generate_many(messages,salts,single=policy=="greedy1")
            stage_totals["generation"]+=time.perf_counter()-tick
            for task,history,row,generation,salt in zip(cohort,histories,records,generated,salts,strict=True):
                tick=time.perf_counter()
                calls,errors=parse_candidates(generation["outputs"],tool_specs(stage))
                stage_totals["parsing"]+=time.perf_counter()-tick
                if policy in SYSTEM_ARMS:
                    scores=scorer.score(generation["prompt_ids"],calls,policy,salt)
                    winner=scores.winner
                    score_record=asdict(scores)
                    stage_totals["scoring"]+=scores.elapsed_seconds
                else:
                    score_record=None
                    counts=Counter()
                    callmap={}
                    for sample in generation["outputs"]:
                        valid,_=parse_candidates([sample],tool_specs(stage))
                        if valid:
                            identity=semantic_key(valid[0]); counts[identity]+=1; callmap[identity]=valid[0]
                    winner=callmap[sorted(counts,key=lambda x:(-counts[x],x))[0]] if counts else None
                tick=time.perf_counter()
                observation=execute_tool(winner,task)
                history.append(dict(call=winner,observation=observation))
                stage_totals["tools_and_history"]+=time.perf_counter()-tick
                row["rounds"].append(dict(stage=stage,generation=generation,candidates=calls,invalid=errors,
                                          scores=score_record,call=winner,observation=observation))
        tick=time.perf_counter()
        finals=scorer.generate_many([agent_messages(task,history,3) for task,history in zip(cohort,histories,strict=True)],salts,final=True)
        stage_totals["final_generation"]+=time.perf_counter()-tick
        end=time.perf_counter()
        for row,history,final in zip(records,histories,finals,strict=True):
            row.update(final_response=final["outputs"][0]["text"],trace=history,final_generation=final,
                       cohort_service_seconds=end-cohort_start,request_pool_latency_seconds=end-begin)
            episodes.append(row)
    elapsed=time.perf_counter()-begin
    return dict(policy=policy,key=key,elapsed_seconds=elapsed,stage_seconds=dict(stage_totals),
                harness_seconds=elapsed-math.fsum(stage_totals.values()),records=episodes)


def accuracy(reference,other):
    if reference["suffix_ids"]!=other["suffix_ids"] or reference["calls"]!=other["calls"]:
        raise RuntimeError("Locked candidate family changed")
    differences=[abs(a-b) for x,y in zip(reference["token_logprobs"],other["token_logprobs"],strict=True) for a,b in zip(x,y,strict=True)]
    sequence=[abs(a-b) for a,b in zip(reference["sequence_logprobs"],other["sequence_logprobs"],strict=True)]
    from orbitcall.vllm_scoring import ScoredFamily
    a=decision_predictions(ScoredFamily(**reference)); b=decision_predictions(ScoredFamily(**other))
    valid=(not differences or (max(differences)<=LIMITS["max_token_absolute_nats"]
           and max(sequence)<=LIMITS["max_sequence_absolute_nats"]))
    return dict(nonempty=bool(differences),max_token_error=max(differences,default=0),
                sum_token_error=math.fsum(differences),token_count=len(differences),
                max_sequence_error=max(sequence,default=0),rule_agreement={key:a[key]==b[key] for key in a},
                joint_agreement=a==b,numerically_valid=valid)


def locked_audit(scorer,native,run):
    checks=[]
    for episode in native["records"]:
        for row in episode["rounds"]:
            identity=episode["id"]+"_r"+str(row["stage"])
            # Stock BF16 arrays are already frozen in native_default policy artifact.
            prompt=row["generation"]["prompt_ids"]
            salt="locked:"+identity
            if not scorer.llm.reset_prefix_cache():
                raise RuntimeError("Locked audit requires a completed cache reset")
            from vllm import SamplingParams
            scorer.llm.generate([scorer.token_prompt(prompt,salt)],SamplingParams(max_tokens=1,temperature=0),use_tqdm=False)
            references=row["scores"]
            for arm in SYSTEM_ARMS[1:]:
                fresh=asdict(scorer.score(prompt,row["candidates"],arm,salt))
                write_new(run/"audit"/(identity+"_"+arm+".json"),fresh)
                result=accuracy(references,fresh)
                checks.append(dict(id=episode["id"],stage=row["stage"],arm=arm,**result))
    write_new(run/"audit_checks.json",dict(limits=LIMITS,records=checks))
    gates={}
    for arm in SYSTEM_ARMS[1:]:
        rows=[row for row in checks if row["arm"]==arm and row["nonempty"]]
        rules=rows[0]["rule_agreement"] if rows else {}
        joint=sum(row["joint_agreement"] for row in rows)
        counts={name:sum(row["rule_agreement"][name] for row in rows) for name in rules}
        tokens=sum(row["token_count"] for row in rows)
        mean=math.fsum(row["sum_token_error"] for row in rows)/tokens if tokens else 0
        valid=bool(rows) and all(row["numerically_valid"] for row in rows) and mean<=LIMITS["mean_token_absolute_nats"] and all(count/len(rows)>=.99 for count in [joint,*counts.values()])
        gates[arm]=dict(valid=valid,nonempty_pools=len(rows),zero_pools=sum(row["arm"]==arm and not row["nonempty"] for row in checks),
                       joint_agreeing=joint,agreement_counts=counts,max_token_error=max((row["max_token_error"] for row in rows),default=0),
                       mean_token_error=mean,max_sequence_error=max((row["max_sequence_error"] for row in rows),default=0))
    write_new(run/"audit_summary.json",gates)
    return gates


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--selection",type=Path)
    p.add_argument("--development",action="store_true")
    args=p.parse_args()
    args.run.mkdir(parents=True,exist_ok=False)
    begin=time.perf_counter()
    signal.signal(signal.SIGALRM,watchdog); signal.alarm(1800 if args.development else 4200)
    inputs=read_json(args.inputs)
    for name,digest in inputs["source_sha256"].items():
        if sha256(ROOT/name)!=digest:
            raise RuntimeError("Frozen input source changed: "+name)
    source=[Path(__file__),ROOT/"scripts/vllm_agent_common.py",ROOT/"scripts/audit_full_triton.py",ROOT/"docs/VLLM_AGENT_SHAPE_PROTOCOL.md",ROOT/"docs/VLLM_AGENT_PROTOCOL.md",ROOT/"docs/VLLM_AGENT_REFINEMENT.md",ROOT/"pyproject.toml",*sorted((ROOT/"src/orbitcall").glob("*.py"))]
    hashes={str(path.relative_to(ROOT)):sha256(path) for path in source}
    if not args.development:
        if args.selection is None:
            raise RuntimeError("Confirmation requires a frozen successful development receipt")
        selection=read_json(args.selection)
        if (selection["source_sha256"]!=hashes
                or selection["inputs_sha256"]!=sha256(args.inputs)
                or selection["mechanism"]!="native_shape_suffix"
                or selection["held_out_outputs_accessed"] is not False):
            raise RuntimeError("Confirmation source/configuration differs from development selection")
        dev=ROOT/selection["development_run"]
        if (sha256(dev/"complete.json")!=selection["complete_sha256"]
                or sha256(dev/"audit_summary.json")!=selection["audit_sha256"]
                or not read_json(dev/"complete.json")["valid"]
                or not all(row["valid"] for row in read_json(dev/"audit_summary.json").values())):
            raise RuntimeError("The retained development result is not valid")
    for path in source:
        target=args.run/"source"/path.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
    selected=[row for row in inputs["records"] if row["split"]==("development" if args.development else "test")]
    if args.development: selected=[sorted([row for row in selected if row["family"]==family],key=lambda row:row["id"])[0] for family in ("catalog","calendar")]
    plan=[]; rng=random.Random(271828)
    if not args.development:
        for control in CONTROLS:
            for block in range(BLOCKS):
                order=[control,"native_shape_suffix","native_shape_suffix",control] if rng.randrange(2) else ["native_shape_suffix",control,control,"native_shape_suffix"]
                plan.append(dict(control=control,block=block,order=order))
        rng.shuffle(plan)
    write_new(args.run/"plan.json",dict(model=args.model.name,engine=ENGINE,limits=LIMITS,
               inputs_sha256=sha256(args.inputs),source_sha256=hashes,selected_ids=[row["id"] for row in selected],
               selection_sha256=None if args.development else sha256(args.selection),
               timing_fixture_ids=inputs["timing_fixture_ids"],timing_blocks=plan,development=args.development,
               policies=POLICIES,controls=CONTROLS,cohort_size=4,
               score_logit_dtype_by_arm={arm:"bfloat16" for arm in SYSTEM_ARMS},
               generation_head_dtype="bfloat16",backbone_dtype="bfloat16",kv_dtype="bfloat16",
               prompt_policy="explicit_nested_arguments_v3",development_selection="first lexicographic development ID per family",
               scoring_mode_by_arm={"native_default":"stock_full","native_shape_suffix":"native_shape_suffix"},
               maximum_process_seconds=1800 if args.development else 4200,answers_accessed=False))
    llm=None
    try:
        llm=initialize(args.model,args.run); scorer=VLLMScorer(llm)
        policies=SYSTEM_ARMS if args.development else POLICIES
        native=None
        for policy in policies:
            result=run_pool(scorer,selected,policy,"policy:"+policy)
            write_new(args.run/("policy_"+policy+".json"),result)
            owned_gpu_check()
            if policy=="native_default": native=result
        policy_predictions=[row for path in [args.run/("policy_"+policy+".json") for policy in policies]
                            for row in read_json(path)["records"]]
        write_new(args.run/"policy_predictions.json",dict(records=policy_predictions))
        write_new(args.run/"policy_prediction_freeze.json",dict(stage="agent-predictions-frozen",answers_accessed=False,
                      predictions=dict(sha256=sha256(args.run/"policy_predictions.json"))))
        audit=locked_audit(scorer,native,args.run)
        if not all(row["valid"] for row in audit.values()):
            raise RuntimeError("Frozen BF16 numerical/decision gate failed; timings not accepted")
        if not args.development:
            lookup={row["id"]:row for row in inputs["records"]}
            fixtures=[lookup[id] for id in inputs["timing_fixture_ids"]]
            for scheduled in plan:
                for position,policy in enumerate(scheduled["order"]):
                    result=run_pool(scorer,fixtures,policy,f"timing:{scheduled['control']}:{scheduled['block']}:{position}")
                    relative=f"windows/{scheduled['control']}_{scheduled['block']}_{position}.json"
                    write_new(args.run/relative,result)
                    append(args.run/"windows.jsonl",dict(control=scheduled["control"],block=scheduled["block"],position=position,
                                                      policy=policy,file=relative,sha256=sha256(args.run/relative),elapsed_seconds=result["elapsed_seconds"]))
                    owned_gpu_check()
        predictions=[]
        for path in [args.run/("policy_"+policy+".json") for policy in policies]:
            predictions.extend(read_json(path)["records"])
        for path in sorted((args.run/"windows").glob("*.json")):
            predictions.extend(read_json(path)["records"])
        write_new(args.run/"predictions.json",dict(records=predictions))
        write_new(args.run/"prediction_freeze.json",dict(stage="agent-predictions-frozen",answers_accessed=False,
                                                        predictions=dict(sha256=sha256(args.run/"predictions.json"))))
        if any(sha256(ROOT/name)!=value for name,value in hashes.items()):
            raise RuntimeError("Frozen source changed during study")
        write_new(args.run/"complete.json",dict(valid=True,elapsed_seconds=time.perf_counter()-begin,
                                               completed_windows=sum(len(row["order"]) for row in plan),
                                               predictions_sha256=sha256(args.run/"predictions.json")))
    except Exception:
        write_new(args.run/"failure.json",dict(traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-begin))
        raise
    finally:
        signal.alarm(0)
        if llm is not None: shutdown(llm)


if __name__=="__main__": main()
