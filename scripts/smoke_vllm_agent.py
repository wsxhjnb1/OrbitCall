"""Development-only engine/plugin validation; preserves failures and source."""

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import time
import traceback

from vllm_agent_common import initialize, shutdown, write_new, ROOT, LIMITS, owned_gpu_check
from orbitcall.vllm_scoring import ARMS,VLLMScorer,decision_predictions


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    args=p.parse_args()
    args.run.mkdir(parents=True,exist_ok=False)
    for path in (ROOT/"src/orbitcall/vllm_plugin.py",ROOT/"src/orbitcall/vllm_scoring.py",Path(__file__),ROOT/"scripts/vllm_agent_common.py"):
        target=args.run/"source"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    begin=time.perf_counter()
    llm=None
    try:
        llm=initialize(args.model,args.run)
        scorer=VLLMScorer(llm)
        messages=[{"role":"system","content":"Return a bare JSON tool call with exactly name and arguments. Available: add(a:integer,b:integer)."},
                  {"role":"user","content":"Use add to add 3 and 5."}]
        generation=scorer.generate(messages,"development-smoke")
        write_new(args.run/"generation.json",generation)
        calls=[{"name":"add","arguments":{"a":3,"b":5}},{"name":"add","arguments":{"a":3,"b":4}}]
        results={}
        for arm in ARMS:
            result=scorer.score(generation["prompt_ids"],calls,arm,"development-smoke")
            results[arm]=asdict(result)
            write_new(args.run/(arm+".json"),results[arm])
        errors={}
        for arm in ARMS[1:]:
            differences=[abs(a-b) for x,y in zip(results["native_full"]["token_logprobs"],results[arm]["token_logprobs"],strict=True) for a,b in zip(x,y,strict=True)]
            errors[arm]={"maximum_token_error":max(differences),"mean_token_error":sum(differences)/len(differences),
                         "winner_equal":results[arm]["winner"]==results["native_full"]["winner"]}
        owned_gpu_check()
        write_new(args.run/"complete.json",dict(errors=errors,limits=LIMITS,elapsed_seconds=time.perf_counter()-begin,scope="development only; no held-out speed claim"))
    except Exception:
        write_new(args.run/"failure.json",dict(traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-begin))
        raise
    finally:
        if llm is not None:
            shutdown(llm)


if __name__=="__main__":
    main()
