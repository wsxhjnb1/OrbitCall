"""Export completed Spark compatibility outputs; no inference or new timing."""
from pathlib import Path
import hashlib
import io
import json
import platform
import subprocess
import tarfile
import xml.etree.ElementTree as ET

from export_vllm_agent_evidence import payload


ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = ('vllm_dtype_compat_20261014_v1', 'vllm_dtype_compat_20261014_v2',
            'vllm_dtype_compat_20261014_v3')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise RuntimeError('Evidence export runs on the authorized Spark')
    identity = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid',
                                        '--format=csv,noheader'], text=True, timeout=15).strip()
    assert 'GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4' in identity and 'GB10' in identity
    final = ROOT / 'runs' / ATTEMPTS[-1]
    assert json.loads((final / 'queue_complete.json').read_text())['valid'] is True
    tree = ET.parse(final / 'tests.xml')
    suites = list(tree.getroot().iter('testsuite'))
    test_counts = {name: sum(int(row.attrib.get(name, 0)) for row in suites)
                   for name in ('tests', 'failures', 'errors', 'skipped')}
    assert test_counts['tests'] > 0 and all(test_counts[key] == 0 for key in ('failures', 'errors', 'skipped'))
    summaries = []
    for mode in ('auto', 'float16'):
        result = json.loads((final / mode / 'complete.json').read_text())
        engine = json.loads((final / mode / 'engine.json').read_text())
        assert result['valid'] is True and result['answers_accessed'] is False
        manifest = json.loads((final / mode / 'source_manifest.json').read_text())
        for name, digest in manifest.items():
            assert sha((ROOT / name).read_bytes()) == digest, name
            assert sha((final / mode / 'source' / name).read_bytes()) == digest, name
        rounds = [row for episode in result['complete_agent_checks'] for row in episode['rounds']]
        scored = [row['score_checks'] for row in rounds if row['score_checks'].get('nonempty', True)]
        assert len(rounds) == 4 and scored
        summaries.append(dict(requested_dtype=mode, resolved_dtype=result['resolved_dtype'],
            model=engine['model']['repo_id'], revision=engine['model']['revision'],
            runtime=engine['runtime'], source_sha256=manifest,
            locked_score_checks=result['locked_score_checks'], mixed_request=result['mixed_request'],
            agent_episode_count=len(result['complete_agent_checks']), agent_round_count=len(rounds),
            agent_scored_round_count=len(scored),
            max_agent_token_error=max(row['max_token_error'] for row in scored),
            max_agent_sequence_error=max(row['max_sequence_error'] for row in scored),
            all_agent_decisions_equal=all(all(row['decision_agreement'].values()) for row in scored),
            all_agent_traces_equal=all(all(row[key] for key in ('generation_tokens','candidates','selected_call','observation')) for row in rounds),
            all_final_tokens_equal=all(row['final_tokens_equal'] for row in result['complete_agent_checks'])))
    target = ROOT / 'evidence/dtype_compatibility'
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('Reconcile an existing evidence export')
    target.mkdir(parents=True, exist_ok=True)
    scientific = ['evidence/vllm_agent/summary.json', 'evidence/vllm_agent/raw_results_v1.tar.gz',
                  'docs/VLLM_AGENT_SHAPE_PROTOCOL.md']
    summary = dict(scope='Development-only model-native dtype compatibility; no FP16 speed or held-out quality claim',
                   host='NVIDIA DGX Spark GB10', final_run=ATTEMPTS[-1], excluded_attempts=list(ATTEMPTS[:-1]),
                   answers_accessed=False, regression_tests=test_counts, modes=summaries,
                   historical_evidence_sha256={name:sha((ROOT/name).read_bytes()) for name in scientific})
    write_new(target / 'summary.json', summary)
    files = []
    for attempt in ATTEMPTS:
        files.extend(sorted((ROOT / 'runs' / attempt).rglob('*')))
    files.extend(ROOT / name for name in ('tests/test_vllm_plugin.py', 'tests/test_vllm_agent_demo.py',
                                          'scripts/export_dtype_compatibility.py'))
    archive = target / 'raw_results_v1.tar.gz'
    records, excluded = [], []
    with tarfile.open(archive, 'x:gz') as out:
        for path in files:
            if not path.is_file():
                continue
            assert not path.is_symlink()
            relative = path.relative_to(ROOT).as_posix()
            original = path.read_bytes()
            if path.name.endswith('_owned.json'):
                excluded.append(dict(path=relative, sha256=sha(original), reason='Owned process command receipt kept privately'))
                continue
            console = path.suffix in ('.log', '.xml') or path.name in ('queue_failure.json', 'failure.json')
            data = payload(original, relative, console)
            header = tarfile.TarInfo(relative)
            header.size = len(data)
            out.addfile(header, io.BytesIO(data))
            records.append(dict(path=relative, original_sha256=sha(original), exported_sha256=sha(data),
                                bytes=len(data), console_paths_redacted=data != original))
    with tarfile.open(archive) as check:
        assert len(check.getmembers()) == len(records)
        for record in records:
            assert sha(check.extractfile(record['path']).read()) == record['exported_sha256']
    validation = dict(valid=True, all_archive_members_reopened=True, members=records, excluded=excluded,
                      raw_archive_sha256=sha(archive.read_bytes()), summary_sha256=sha((target/'summary.json').read_bytes()),
                      scientific_json_and_source_bytes_preserved=True, failed_attempts_retained=True,
                      model_weights_or_answers_exported=False)
    write_new(target / 'RELEASE_VALIDATION.json', validation)
    lines = ['# Model-native dtype compatibility', '',
             'The plugin follows the loaded model dtype. The portable demo defaults to `--dtype auto`;',
             'an explicit `--dtype float16` selects the FP16 path. The same source is used for both.', '',
             'This development-only check used pinned Qwen2.5-1.5B-Instruct on DGX Spark GB10,',
             'vLLM 0.29.0. Comparisons are within each dtype against its own native full scorer.', '',
             '| Requested | Resolved | Locked token error | Agent token error | Agent traces |',
             '| --- | --- | ---: | ---: | --- |']
    for row in summaries:
        lines.append(f"| {row['requested_dtype']} | {row['resolved_dtype']} | {row['locked_score_checks']['max_token_error']:.6g} | {row['max_agent_token_error']:.6g} | equal for both development episodes |")
    lines += ['', f"Regression suite: {test_counts['tests']} passed, zero failures/errors/skips.", '',
              'Locked two- and three-argument families retain full JSON/EOS scoring and all eleven',
              'decision rules. Mixed marked/native requests pass. Complete Agent checks retain',
              'all four rounds per dtype: candidate token IDs, score vectors, selected calls,',
              'actual tool observations and final token IDs agree with their native controls.', '',
              'This is compatibility evidence, not an FP16 speed benchmark or a held-out task score.',
              'Historical BF16 speed ratios and scientific files remain unchanged.', '',
              '[Frozen protocol](../../docs/VLLM_DTYPE_PROTOCOL.md) ·',
              '[Infrastructure amendments](../../docs/VLLM_DTYPE_VALIDATION_AMENDMENT.md) ·',
              '[Summary](summary.json) · [Raw runs](raw_results_v1.tar.gz) ·',
              '[Export validation](RELEASE_VALIDATION.json)', '']
    with (target / 'RESULTS.md').open('x') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps(dict(regression_tests=test_counts, modes=[{key:row[key] for key in ('requested_dtype','resolved_dtype','max_agent_token_error','max_agent_sequence_error')} for row in summaries],
                         archive_bytes=archive.stat().st_size, members=len(records), archive_sha256=validation['raw_archive_sha256']), indent=2))


if __name__ == '__main__':
    main()
