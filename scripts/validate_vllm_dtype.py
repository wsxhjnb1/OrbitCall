"""Spark-only, development-only compatibility gate for the model-native dtype.

See docs/VLLM_DTYPE_PROTOCOL.md. All raw outputs and failed attempts are retained.
This does not replace the frozen BF16 performance study or read task answers.
"""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import time
import traceback

from vllm_agent_common import ENGINE, ROOT, owned_gpu_check, shutdown
from audit_full_triton import GPU_UUID, sha256, verify_model, write_new
from orbitcall.agent_tasks import tasks
from orbitcall.gpu_guard import require_idle_gpu
from orbitcall.vllm_scoring import ScoredFamily, VLLMScorer, decision_predictions
from run_vllm_agent_shape import run_pool


def read_json(path):
    return json.loads(Path(path).read_text())


def compare_scores(native, optimized, eos_token_id):
    assert native.calls == optimized.calls
    assert native.suffix_ids == optimized.suffix_ids and native.suffix_ids
    for result in (native, optimized):
        assert len(result.suffix_ids) == len(result.token_logprobs) == len(result.sequence_logprobs)
        for ids, values in zip(result.suffix_ids, result.token_logprobs, strict=True):
            assert ids and ids[-1] == eos_token_id and len(values) == len(ids)
            assert all(math.isfinite(value) for value in values)
    differences = [abs(a - b)
                   for left, right in zip(native.token_logprobs, optimized.token_logprobs, strict=True)
                   for a, b in zip(left, right, strict=True)]
    sequence = [abs(a - b) for a, b in zip(native.sequence_logprobs,
                                         optimized.sequence_logprobs, strict=True)]
    assert differences and all(math.isfinite(value) for value in differences + sequence)
    agreement = {key: value == decision_predictions(optimized)[key]
                 for key, value in decision_predictions(native).items()}
    result = dict(token_count=len(differences), max_token_error=max(differences),
                  max_sequence_error=max(sequence), decision_agreement=agreement)
    result['valid'] = (result['max_token_error'] <= 0.05
                       and result['max_sequence_error'] <= 0.5 and all(agreement.values()))
    assert result['valid'], result
    return result


def compare_agents(native, optimized, eos_token_id):
    assert len(native['records']) == len(optimized['records']) == 2
    checks = []
    for left, right in zip(native['records'], optimized['records'], strict=True):
        assert left['id'] == right['id']
        round_checks = []
        for a, b in zip(left['rounds'], right['rounds'], strict=True):
            generated_a = [(row['seed'], row['token_ids']) for row in a['generation']['outputs']]
            generated_b = [(row['seed'], row['token_ids']) for row in b['generation']['outputs']]
            same = dict(generation_tokens=generated_a == generated_b,
                        candidates=a['candidates'] == b['candidates'],
                        selected_call=a['call'] == b['call'], observation=a['observation'] == b['observation'])
            assert all(same.values()), same
            assert a['scores']['calls'] == b['scores']['calls']
            if a['scores']['calls']:
                same['score_checks'] = compare_scores(ScoredFamily(**a['scores']),
                                                      ScoredFamily(**b['scores']), eos_token_id)
            else:
                assert not a['scores']['suffix_ids'] and not b['scores']['suffix_ids']
                same['score_checks'] = {'nonempty': False}
            round_checks.append(same)
        same_final = (left['final_generation']['outputs'][0]['token_ids']
                      == right['final_generation']['outputs'][0]['token_ids'])
        assert same_final
        checks.append(dict(id=left['id'], rounds=round_checks, final_tokens_equal=same_final))
    return checks


def mixed_request_check(scorer, prompt, suffix, run):
    from vllm import SamplingParams
    start = len(prompt) - 1
    options = dict(prompt_logprobs=0, max_tokens=1, temperature=0,
                   detokenize=False, flat_logprobs=False, skip_reading_prefix_cache=True)
    native = SamplingParams(**options)
    marked = SamplingParams(**options, extra_args={'orbitcall_score_start': start,
                            'orbitcall_score_mode': 'native_shape_suffix'})
    outputs = scorer.llm.generate([scorer.token_prompt(prompt + suffix, 'mixed-native'),
                                   scorer.token_prompt(prompt + suffix, 'mixed-marked')],
                                  [native, marked], use_tqdm=False)
    raw_rows = [[None if row is None else {str(token): {'logprob': float(value.logprob),
                                                          'rank': int(value.rank)}
                                               for token, value in row.items()}
                 for row in output.prompt_logprobs] for output in outputs]
    write_new(run / 'mixed_request_raw.json', dict(prompt_ids=prompt, suffix_ids=suffix,
              raw_prompt_logprobs=raw_rows, cached_tokens=[int(out.num_cached_tokens) for out in outputs]))
    arrays = []
    for output in outputs:
        rows = output.prompt_logprobs
        assert len(rows) == len(prompt) + len(suffix)
        arrays.append([float(row[token].logprob)
                       for token, row in zip(suffix, rows[len(prompt):], strict=True)])
    assert all(row is None for row in outputs[1].prompt_logprobs[:len(prompt)])
    assert all(math.isfinite(value) for array in arrays for value in array)
    differences = [abs(a - b) for a, b in zip(*arrays, strict=True)]
    result = dict(token_count=len(suffix), native_token_logprobs=arrays[0],
                  marked_token_logprobs=arrays[1], suffix_ids=suffix,
                  max_token_error=max(differences), marked_cached_tokens=int(outputs[1].num_cached_tokens))
    assert result['max_token_error'] <= 0.05 and result['marked_cached_tokens'] == 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dtype', choices=('auto', 'float16'), required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise RuntimeError('Validation runs only on the authorized Spark')
    require_idle_gpu()
    identity = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid',
                                        '--format=csv,noheader'], text=True, timeout=15).strip()
    if GPU_UUID not in identity or 'GB10' not in identity:
        raise RuntimeError('Unexpected validation GPU')
    args.run.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), ROOT / 'examples/vllm_agent.py',
                    ROOT / 'scripts/vllm_agent_common.py', ROOT / 'scripts/run_vllm_agent_shape.py',
                    ROOT / 'scripts/audit_full_triton.py', ROOT / 'pyproject.toml',
                    ROOT / 'docs/VLLM_DTYPE_PROTOCOL.md', ROOT / 'docs/VLLM_DTYPE_VALIDATION_AMENDMENT.md',
                    *sorted((ROOT / 'src/orbitcall').glob('*.py'))]
    inventory = {path.relative_to(ROOT).as_posix(): sha256(path) for path in source_files}
    for path in source_files:
        target = args.run / 'source' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_new(args.run / 'source_manifest.json', inventory)
    llm = None
    begin = time.perf_counter()
    def watchdog(signum, frame):
        raise TimeoutError('Compatibility validation exceeded 1200-second budget')
    signal.signal(signal.SIGALRM, watchdog)
    signal.alarm(1200)
    try:
        receipt, _ = verify_model(args.model, read_json)
        runtime = {name: importlib.metadata.version(name) for name in ('vllm', 'torch', 'transformers', 'triton')}
        assert runtime['vllm'] == '0.29.0'
        from vllm import LLM
        options = dict(ENGINE, dtype=args.dtype)
        llm = LLM(model=str(args.model), **options)
        resolved = str(llm.llm_engine.vllm_config.model_config.dtype)
        expected = 'torch.bfloat16' if args.dtype == 'auto' else 'torch.float16'
        assert resolved == expected, (resolved, expected)
        write_new(args.run / 'engine.json', dict(model=receipt, runtime=runtime, gpu=identity,
                  requested_dtype=args.dtype, resolved_dtype=resolved, options=options,
                  source_sha256=inventory))
        scorer = VLLMScorer(llm)
        messages = [{'role': 'system', 'content': 'Return a JSON tool call for add(a:integer,b:integer).'},
                    {'role': 'user', 'content': 'Use add to add 3 and 5.'}]
        prompt = scorer.prompt_ids(messages)
        calls = [{'name': 'add', 'arguments': {'a': 3, 'b': 5}},
                 {'name': 'add', 'arguments': {'a': 3, 'b': 4}},
                 {'name': 'add3', 'arguments': {'a': 3, 'b': 4, 'c': 5}}]
        scored = {}
        for policy in ('native_default', 'native_shape_suffix'):
            assert llm.reset_prefix_cache()
            scored[policy] = scorer.score(prompt, calls, policy, 'dtype-locked-development')
            write_new(args.run / f'locked_{policy}.json', asdict(scored[policy]))
        locked = compare_scores(scored['native_default'], scored['native_shape_suffix'], scorer.tokenizer.eos_token_id)
        mixed = mixed_request_check(scorer, prompt, scored['native_default'].suffix_ids[0], args.run)
        write_new(args.run / 'mixed_request.json', mixed)
        selected = [sorted([row for row in tasks() if row['split'] == 'development'
                            and row['family'] == family], key=lambda row: row['id'])[0]
                    for family in ('catalog', 'calendar')]
        agents = {}
        for policy in ('native_default', 'native_shape_suffix'):
            agents[policy] = run_pool(scorer, selected, policy, 'dtype-complete-agent-development')
            write_new(args.run / f'agent_{policy}.json', agents[policy])
        comparisons = compare_agents(agents['native_default'], agents['native_shape_suffix'], scorer.tokenizer.eos_token_id)
        owned_gpu_check()
        write_new(args.run / 'complete.json', dict(valid=True, requested_dtype=args.dtype,
                  resolved_dtype=resolved, locked_score_checks=locked, mixed_request=mixed,
                  complete_agent_checks=comparisons, answers_accessed=False,
                  scope='Development-only compatibility check; no new speed or held-out quality claim',
                  elapsed_seconds=time.perf_counter() - begin))
    except Exception:
        write_new(args.run / 'failure.json', dict(traceback=traceback.format_exc(),
                  elapsed_seconds=time.perf_counter() - begin))
        raise
    finally:
        signal.alarm(0)
        if llm is not None:
            shutdown(llm)


if __name__ == '__main__':
    main()
