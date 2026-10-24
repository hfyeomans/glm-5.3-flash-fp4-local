#!/usr/bin/env python3
"""Summarize a served sweep without treating screening samples as qualification."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    runs = []
    for file in sorted(args.input.glob('*/complete.json')):
        run = json.loads(file.read_text())
        run['folder'] = file.parent.name
        summary = file.parent / 'bench/summary.json'
        run['measurements'] = json.loads(summary.read_text()) if summary.exists() else []
        runs.append(run)
    controls = [r for r in runs if r['profile'] == 'control' and r['status'] == 'pass']
    if not controls:
        raise ValueError('No passing control')
    reference = controls[0]['token_hashes']
    baseline = {}
    for r in controls[-1:]:
        for x in r['measurements']:
            baseline.setdefault((x['case'], x['mtp']), []).append(x)
    lines = ['# Served TensorFold tuning screen', '',
             'Screening medians versus the latest completed control (the closing control once finished). The opening control ran cooler; retain both controls to assess drift. No accuracy or release qualification is implied.', '',
             '| Profile | Prompt | MTP | Decode tok/s | Decode change | Prefill tok/s | Prefill change | TTFT s | Output matches control |',
             '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in runs:
        r['matches_control'] = r.get('token_hashes') == reference
        for x in r['measurements']:
            b = baseline.get((x['case'], x['mtp']))
            if not b:
                continue
            x['decode_change_percent'] = 100 * (x['decode_tokens_per_second'] / statistics.mean(y['decode_tokens_per_second'] for y in b) - 1)
            x['prefill_change_percent'] = 100 * (x['prefill_tokens_per_second'] / statistics.mean(y['prefill_tokens_per_second'] for y in b) - 1)
            same = r.get('token_hashes', {}).get(x['case']) == reference.get(x['case'])
            lines.append(f'| {r["profile"]} | {x["case"]} | {"on" if x["mtp"] else "off"} | {x["decode_tokens_per_second"]:.1f} | {x["decode_change_percent"]:+.2f}% | {x["prefill_tokens_per_second"]:.0f} | {x["prefill_change_percent"]:+.2f}% | {x["client_ttft_seconds"]:.3f} | {same} |')
        if not r['measurements']:
            lines.append(f'| {r["profile"]} | {r["status"]} | | | | | | | |')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'screen.json').write_text(json.dumps(runs, indent=2) + '\n')
    (args.output / 'screen.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
