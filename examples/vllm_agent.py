"""A small executable two-tool Agent demo, using the released BF16 adapter.

This portable demonstration is separate from the original-host frozen study.
It uses two DEVELOPMENT fixtures, retains every sampled output and selected
tool result, and never calls the reference task evaluator. Provision a
compatible public vLLM0.29/CUDA runtime and download the model first.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_vllm_agent_shape import run_pool
from vllm_agent_common import ENGINE,shutdown
from orbitcall.agent_tasks import tasks
from orbitcall.vllm_scoring import SHAPE_ARMS,VLLMScorer


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True,help="Downloaded local Qwen2.5-Instruct checkpoint")
    p.add_argument("--policy",choices=SHAPE_ARMS,default="native_shape_suffix")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh output path")
    from vllm import LLM
    llm=None
    try:
        llm=LLM(model=args.model,**ENGINE)
        inputs=[sorted([row for row in tasks() if row["split"]=="development" and row["family"]==family],
                       key=lambda row:row["id"])[0] for family in ("catalog","calendar")]
        result=run_pool(VLLMScorer(llm),inputs,args.policy,"portable-development-demo")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as stream:
            json.dump(dict(scope="development demonstration, not benchmark evidence",
                           answers_accessed=False,model=args.model,engine=ENGINE,**result),stream,
                      indent=2,ensure_ascii=False,allow_nan=False)
            stream.write("\n")
        print(json.dumps(dict(episodes=len(result["records"]),policy=args.policy,
                              elapsed_seconds=result["elapsed_seconds"],output=str(args.output))))
    finally:
        if llm is not None: shutdown(llm)


if __name__=="__main__": main()
