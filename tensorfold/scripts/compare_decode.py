#!/usr/bin/env python3
"""Compare bracketed 4-warp / 1-warp / 4-warp experiments without hiding drift."""
import argparse
import json
from pathlib import Path
import statistics


def load(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    rows = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
    rows = [x for x in rows if not x['warmup']]
    calls = {(r['case'], r['body']['draft'], r['round']): r for r in rows}
    if len(calls) != len(rows) or not rows:
        raise ValueError('Empty or duplicate measurement rows')
    expected = {(name, draft, i) for name in ('hash-table', 'database', 'archive')
                for draft in (True, False) for i in range(manifest['rounds'])}
    if set(calls) != expected:
        raise ValueError('Incomplete measurement set')
    return manifest, calls


def compare(folders):
    runs = [load(p) for p in folders]
    first = runs[0][0]
    for (manifest, calls), warps in zip(runs, ('4', '1', '4')):
        if manifest['settings']['TF_NVFP4_DECODE_WARPS'] != warps:
            raise ValueError('Expected 4, 1, 4 warp order')
        for key in ('image_id', 'workload_sha256', 'client_sha256', 'rounds'):
            if manifest[key] != first[key]:
                raise ValueError(f'Incomparable {key}')
        settings = {k: v for k, v in manifest['settings'].items() if k != 'TF_NVFP4_DECODE_WARPS'}
        baseline = {k: v for k, v in first['settings'].items() if k != 'TF_NVFP4_DECODE_WARPS'}
        if settings != baseline or calls.keys() != runs[0][1].keys():
            raise ValueError('Settings or request sets differ')
    disagreements = []
    for key, baseline in runs[0][1].items():
        for i, (_, calls) in enumerate(runs[1:], 1):
            row = calls[key]
            if row['request_sha256'] != baseline['request_sha256']:
                raise ValueError(f'Request mismatch: {key}')
            if row['response']['usage']['prompt_tokens'] != baseline['response']['usage']['prompt_tokens']:
                raise ValueError(f'Token count mismatch: {key}')
            if row['response']['tensorfold']['token_sha256'] != baseline['response']['tensorfold']['token_sha256']:
                disagreements.append({'case': key, 'run': i})
    summaries = []
    for name, draft in sorted({(k[0], k[1]) for k in runs[0][1]}):
        modes = []
        for manifest, calls in runs:
            rows = [r for k, r in calls.items() if k[:2] == (name, draft)]
            rates = [r['response']['tensorfold']['tokens_per_second'] for r in rows]
            modes.append({'warps': manifest['settings']['TF_NVFP4_DECODE_WARPS'],
                          'median_decode_tps': statistics.median(rates), 'min_decode_tps': min(rates),
                          'max_decode_tps': max(rates),
                          'median_client_seconds': statistics.median(r['client_seconds'] for r in rows),
                          'median_prefill_seconds': statistics.median(r['response']['tensorfold']['prefill_seconds'] for r in rows)})
        a, b, c = [r['median_decode_tps'] for r in modes]
        summaries.append({'case': name, 'draft': draft, 'runs': modes,
                          'one_warp_vs_before_percent': 100 * (b / a - 1),
                          'one_warp_vs_after_percent': 100 * (b / c - 1),
                          'baseline_after_vs_before_percent': 100 * (c / a - 1)})
    plain_disagreements = []
    for i, (_, calls) in enumerate(runs):
        for key, row in calls.items():
            name, draft, round_id = key
            if draft and row['response']['tensorfold']['token_sha256'] != calls[(name, False, round_id)]['response']['tensorfold']['token_sha256']:
                plain_disagreements.append({'run': i, 'case': name, 'round': round_id})
    return {'runs': [str(x) for x in folders], 'manifests': [x[0] for x in runs],
            'output_disagreements': disagreements, 'draft_plain_disagreements': plain_disagreements,
            'measurements_per_run': len(runs[0][1]), 'summary': summaries,
            'scope': 'Three synthetic prompts, 256 generated tokens, one slot, 16K context. Four repeats per case. Bracketed runs expose drift; they do not establish a universal speedup.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs=3, type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = compare(args.runs)
    with args.output.open('x') as out:
        json.dump(report, out, indent=2); out.write('\n')
    print(json.dumps(report['summary'], indent=2))
    print('Output disagreements:', len(report['output_disagreements']),
          'draft/plain:', len(report['draft_plain_disagreements']))


if __name__ == '__main__':
    main()
