#!/usr/bin/env python3
"""Package a completed long-context cohort, retaining reused and supplemental evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tensorfold/tuning'))
from sweep import headroom
from long_context import select_cases


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(row):
    return row['full_credit'] and not row['truncated'] and not row['reasoning_marker']


def brief(row):
    return {**{k: row[k] for k in ('id', 'task', 'prompt_tokens', 'score', 'truncated', 'reasoning_marker')},
            'clean_full_credit': clean(row),
            'response_sha256': hashlib.sha256(row['response']['choices'][0]['text'].encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'bank', 'receipt', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--controls-only', action='store_true',
                        help='Package all 52 frozen 32K controls without near-1M cases')
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text())
    finished = load(args.run / 'complete.json')
    assert finished['status'] == 'pass'
    results = load(args.run / 'long-context/results.json')
    assert results['manifest'] == load(args.run / 'long-context/manifest.json'), 'Embedded manifest differs from the on-disk long-context manifest'
    bank = load(args.bank / 'manifest.json')
    manifest = results['manifest']
    long_n = manifest['sampling_plan']['long_samples_per_task']
    selected = select_cases(bank, [32768] if args.controls_only else [32768, 1048576],
                            None if args.controls_only else long_n)
    if args.controls_only:
        assert len(selected) == 52 and len({c['id'] for c in selected}) == 52, 'Expected all 52 frozen controls'
        assert len(bank['protocol']['tasks']) == 13 and bank['protocol']['samples_per_task'] == 4
        for task in bank['protocol']['tasks']:
            assert sorted(c['sample'] for c in selected if c['task'] == task) == [0, 1, 2, 3]
    assert results['complete'] and results['cases'] == len(selected)
    assert bank['protocol']['name'] == 'local-ruler-13-v2'
    assert manifest['settings']['TF_CONTEXT'] == '1048576'
    assert manifest['bank_sha256'] == sha(args.bank / 'manifest.json')
    assert manifest['client_sha256'] == sha(ROOT / 'quality/long_context.py')
    requests = args.run / 'long-context/requests.jsonl'
    assert results['requests_sha256'] == sha(requests)
    rows = [json.loads(line) for line in requests.read_text().splitlines()]
    assert [r['id'] for r in rows] == manifest['cases'] == [c['id'] for c in selected]
    if args.controls_only:
        assert all((row['task'], row['length']) == (case['task'], case['length'])
                   for row, case in zip(rows, selected)), 'Control metadata differs from the frozen bank'
    floor = int(manifest['minimum_free_MiB'])
    devices = manifest['gpu_devices']
    memory = headroom(args.run / 'gpu.csv', devices, floor)
    assert memory['passed']
    supplemental = []
    source_gate = None
    files = [args.run / name for name in ('launch-manifest.json', 'complete.json', 'gpu.csv', 'server.log',
             'long-context/manifest.json', 'long-context/results.json')] + [requests, args.bank / 'manifest.json']
    if manifest.get('reuse'):
        source = Path(manifest['reuse']['source'])
        assert sha(source / 'manifest.json') == manifest['reuse']['manifest_sha256']
        assert sha(source / 'requests.jsonl') == manifest['reuse']['requests_sha256']
        old = [json.loads(line) for line in (source / 'requests.jsonl').read_text().splitlines()]
        supplemental = [r for r in old if r['id'] not in manifest['cases']]
        source_gate = headroom(source.parent / 'gpu.csv', devices, floor)
        assert source_gate['passed']
        memory = dict(memory, sampled_min_free_MiB={rank: min(value, source_gate['sampled_min_free_MiB'][rank])
                      for rank, value in memory['sampled_min_free_MiB'].items()},
                      scope='One-second samples across both source and continued runs; intervening transient peaks are not observed')
        files += [source / 'manifest.json', source / 'requests.jsonl', source.parent / 'gpu.csv']
    near = [g for g in results['groups'] if g['configured_context'] == 1048576 and g['near_window_cases'] == g['n']]
    controls = [g for g in results['groups'] if g['configured_context'] == 32768]
    if args.controls_only:
        assert len(results['groups']) == len(controls) == 13
        assert {g['task'] for g in controls} == set(bank['protocol']['tasks'])
        for group in controls:
            task_rows = [r for r in rows if r['task'] == group['task']]
            assert group['n'] == len(task_rows) == 4
            assert math.isclose(group['mean_score'], sum(r['score'] for r in task_rows) / len(task_rows), rel_tol=1e-12, abs_tol=1e-12)
            assert group['full_credit_clean_completion'] == sum(clean(r) for r in task_rows)
            assert group['input_tokens_min'] == min(r['prompt_tokens'] for r in task_rows)
            assert group['input_tokens_max'] == max(r['prompt_tokens'] for r in task_rows)
    else:
        assert len(near) == 11 and sum(g['n'] for g in near) == 11 * long_n
    assert sum(g['n'] for g in controls) == 52
    issues = [brief(r) for r in rows if not clean(r)]
    reused = sum('reused_from' in r for r in rows)
    receipt = {
        'context_label_semantics': 'Group configured_context is the nominal bank label; actual TF_CONTEXT is 1048576 for both groups.',
        'results': results, 'completion': finished, 'headroom': memory, 'source_run_headroom': source_gate,
        'cohort_requests': len(rows), 'reused_requests': reused, 'new_requests': len(rows) - reused,
        'short_controls': {'n': 52, 'mean_task_score': statistics.mean(g['mean_score'] for g in controls),
                           'clean_full_credit': sum(clean(r) for r in rows if r['length'] == 32768)},
        'near_1m': None if args.controls_only else {'tasks': 11, 'n': 11 * long_n, 'mean_task_score': statistics.mean(g['mean_score'] for g in near),
                    'clean_full_credit': sum(clean(r) for r in rows if r['length'] == 1048576 and r['near_window'])},
        'all_nonclean_or_incorrect_cases': issues,
        'supplemental_completed_cases': [brief(r) for r in supplemental],
        'evidence': [{'path': str(path), 'sha256': sha(path)} for path in files],
        'reporter_sha256': sha(Path(__file__)),
        'limits': 'User reduced long examples mid-run by fixed sample index, independent of scores. All shorter controls retained. Small local RULER-derived screen, not a leaderboard or universal qualification. No matched vLLM or older-TensorFold long scores exist for this bank. Substring scores and clean completion are separate.'}
    if args.controls_only:
        receipt['cohort_mode'] = 'controls-only'
        receipt['context_label_semantics'] = 'Group configured_context is the nominal 32768 bank label; actual TF_CONTEXT is 1048576 for all controls.'
        receipt['limits'] = 'All 52 frozen 32K controls under TF_CONTEXT=1048576; near-1M cases were not run. Substring scores and clean completion are separate.'
    if args.receipt.exists():
        raise SystemExit(f'Refusing to overwrite existing receipt: {args.receipt}')
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    if args.controls_only:
        short = receipt['short_controls']
        minima = ', '.join(f"GPU {device}: {memory['sampled_min_free_MiB'][device]:,} MiB" for device in devices)
        lines = ['# Long-context short controls', '',
            f"Completed all 52 frozen controls across 13 tasks: {short['clean_full_credit']}/52 have clean full credit; mean task score {short['mean_task_score']:.2f}. Clean full credit requires full score, no truncation and no generated reasoning marker.", '',
            'Nominal input length is 32,768 tokens; actual server allocation is `TF_CONTEXT=1048576`. Near-1M cases were not run.', '',
            f"Minimum sampled free memory: {minima}. The {floor:,} MiB floor passed; one-second sampling does not observe intervening transient peaks.", '',
            '| Task | Cases | Mean score | Actual input tokens | Clean full credit |',
            '|---|---:|---:|---:|---:|']
        for task in bank['protocol']['tasks']:
            group = next(g for g in controls if g['task'] == task)
            lines.append(f"| {task} | {group['n']} | {group['mean_score']:.2f} | {group['input_tokens_min']:,}–{group['input_tokens_max']:,} | {group['full_credit_clean_completion']}/{group['n']} |")
        lines += ['', '## Incorrect or nonclean responses', '',
            '| Case | Score | Truncated | Reasoning marker |', '|---|---:|---|---|']
        lines += [f"| {r['id']} | {r['score']} | {r['truncated']} | {r['reasoning_marker']} |" for r in issues] if issues else ['| None | — | — | — |']
        lines += ['', f"Requests: {len(rows) - reused} new, {reused} reused. Raw responses: `{requests.as_posix()}`.", '',
            f'[Portable receipt](../{args.receipt.as_posix()}) · [Protocol](../quality/LONG-CONTEXT.md).', '']
        args.report.write_text('\n'.join(lines))
        return
    short, long = receipt['short_controls'], receipt['near_1m']
    lines = ['# Long-context accuracy screen', '',
        f"Completed the selected {len(rows)}-request cohort on `host-1m`: {short['clean_full_credit']}/52 shorter controls and {long['clean_full_credit']}/{long['n']} near-1M cases have clean full credit. This requires full official score, no truncation and no generated reasoning marker.", '',
        f"The user reduced the plan to {long_n} fixed examples per long task after the run began. Selection uses sample indices 0 through {long_n - 1}, independent of scores. All 52 shorter controls are retained. {reused} valid requests were reused without new inference; {len(rows) - reused} were newly executed. {len(supplemental)} additional completed cases are preserved separately below, rather than discarded or mixed into the balanced headline.", '',
        f"Mean task score: {short['mean_task_score']:.2f} for 13 shorter-control tasks; {long['mean_task_score']:.2f} for 11 near-1M tasks. Only {long_n} long examples per task makes this an initial regression baseline, not a precise general accuracy estimate. These are different examples, not repeated trials of an identical prompt.", '',
        '| Task | Shorter mean score (n=4) | Longer mean score | Actual longer input tokens | Clean full credit, longer |',
        '|---|---:|---:|---:|---:|']
    for task in bank['protocol']['tasks']:
        small = next(g for g in controls if g['task'] == task)
        large = next(g for g in results['groups'] if g['task'] == task and g['configured_context'] == 1048576)
        suffix = '' if large['near_window_cases'] == large['n'] else ' (below 1M)'
        lines.append(f"| {task} | {small['mean_score']:.2f} | {large['mean_score']:.2f} | {large['input_tokens_min']:,}–{large['input_tokens_max']:,}{suffix} | {large['full_credit_clean_completion']}/{large['n']} |")
    minima = '/'.join(f"{memory['sampled_min_free_MiB'][device]:,}" for device in devices)
    lines += ['', 'CWE and SQuAD are corpus-limited and excluded from the near-1M aggregate. Both groups use the same 1M server allocation; shorter controls contain roughly 28K tokens. Each request allows 256 output tokens. QA question IDs are shared across lengths; synthetic golds need not be identical.', '',
        f"Minimum sampled free memory across source and continued runs: {minima} MiB. The {floor:,} MiB floor passed; one-second sampling does not observe intervening transient peaks. Capacity and task accuracy remain separate results.", '',
        '## Incorrect or nonclean core responses', '', '| Case | Official score | Truncated | Reasoning marker |', '|---|---:|---|---|']
    lines += [f"| {r['id']} | {r['score']} | {r['truncated']} | {r['reasoning_marker']} |" for r in issues] if issues else ['| None | — | — | — |']
    lines += ['', '## Supplemental completed cases', '', '| Case | Official score | Clean full credit |', '|---|---:|---|']
    lines += [f"| {r['id']} | {r['score']} | {clean(r)} |" for r in supplemental] if supplemental else ['| None | — | — |']
    lines += ['', 'All responses remain in the raw runs. No requests were retried to select a better answer. The official substring metric can accept an answer mentioned in extra text; clean completion is an additional diagnostic, not a semantic judge.', '',
        'The original open-reasoning attempt is invalid and excluded. The corrected bank preserves its task bodies/golds; only the wrapper changed. The later four-example run was stopped at the user’s scope reduction, and its completed valid results were reused after checking protocol, image, settings, prompt hashes and raw responses. [Protocol](../quality/LONG-CONTEXT.md).', '',
        'These results cover the current TensorFold configuration only and do not establish superiority over vLLM or older TensorFold at long context. The [three-way short-quality comparison](QUALITY-BAKEOFF-V2.md) remains the matched engine comparison.', '',
        f'[Portable receipt](../{args.receipt.as_posix()}) · raw continuation: `{args.run.as_posix()}`.', '']
    args.report.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
