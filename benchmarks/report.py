#!/usr/bin/env python3
"""Summarize shared engine measurements without conflating prefill, decode or TTFT."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifests, cells, consistency = [], [], []
    for folder in args.runs:
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifests:
            for key in ('protocol', 'workloads_sha256', 'model_revision', 'client_sha256'):
                if manifest[key] != manifests[0][key]:
                    raise ValueError(f'Incomparable {key}')
        manifests.append(manifest)
        summary = json.loads((folder / 'summary.json').read_text())
        rows = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
        for row in summary:
            selected = [r for r in rows if not r['warmup'] and r['case'] == row['case'] and r['draft'] == row['mtp']]
            if len(selected) != manifest['repeats'] or len(selected) != row['n']:
                raise ValueError('Incomplete measured sample set')
            if any(r['response']['usage']['prompt_tokens'] != row['prompt_tokens'] or
                   r['response']['usage']['completion_tokens'] != 256 or
                   r['response']['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) != 0 for r in selected):
                raise ValueError('Counts or cold-cache contract differs')
            cells.append(dict(backend=manifest['backend'], run=str(folder), **row))
        if manifest['backend'] == 'tensorfold':
            for case in manifest['cases']:
                selected = [r for r in rows if r['case'] == case]
                hashes = {r['response']['runtime']['token_sha256'] for r in selected}
                consistency.append({'run': str(folder), 'case': case, 'mtp_modes': sorted({r['draft'] for r in selected}),
                                    'distinct_native_token_hashes_including_warmups': len(hashes)})
    keys = [(x['backend'], x['case'], x['mtp']) for x in cells]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate backend/case/mode; create a separate candidate comparison')
    cells.sort(key=lambda x: (x['prompt_tokens'], x['case'], x['backend'], not x['mtp']))
    lines = ['# Shared cold-prompt benchmark', '',
             'Same pinned NVIDIA checkpoint, tokenizer IDs and raw prompts; one request at a time, 256 output tokens, '
             'greedy generation, seed 1729, three measured repeats per cell after warmup.', '',
             '| Prompt | Engine | MTP | Decode tok/s | Prefill tok/s | Prefill seconds | Client TTFT seconds |',
             '|---|---|---|---:|---:|---:|---:|']
    for x in cells:
        lines.append(f"| {x['case']} ({x['prompt_tokens']:,}) | {x['backend']} | {'on' if x['mtp'] else 'off'} | "
                     f"{x['decode_tokens_per_second']:.1f} | {x['prefill_tokens_per_second']:,.0f} | "
                     f"{x['prefill_seconds']:.3f} | {x['client_ttft_seconds']:.3f} |")
    lines += ['', 'These are per-cell medians, not a universal engine ranking. Short-prompt prefill rates include fixed '
              'overhead and should not be extrapolated to long prompts.', '',
              'Decode uses each server\'s post-first-token interval and (output tokens minus one). vLLM timings are '
              'isolated deltas of exactly one request in its Prometheus histograms; TensorFold returns runtime timings '
              'with the completion. Client TTFT is first nonempty raw text, including local HTTP/queue overhead. '
              'Server prefill boundaries are implementation-specific; client TTFT supplies a common external measurement.', '',
              'Every measured request had zero reused prompt tokens. vLLM retains its qualified MTP3/B12X budget, '
              '1M configured window and eight slots, with one client request active. Its plain run removes only the '
              'speculative configuration. Unique cache salts prevent prefix reuse. TensorFold uses one decode warp, '
              '128K configured context, one slot, FP8 KV, default reserve and zero retained prompt-cache budget.', '',
              'Some repeated vLLM greedy outputs differ despite the fixed seed; distinct text counts are retained '
              'in the JSON. The timing samples are valid fixed-input/fixed-output-length measurements, not a claim of '
              'bit-deterministic generation or an accuracy score. TensorFold token-hash consistency is reported separately. '
              'The 256-token cap is intentional for throughput. The earlier quality suite remains the accuracy reference.', '',
              'Raw manifests, requests, SSE frames, counter snapshots and responses are retained under `logs/tuning/`. '
              'No clocks, power limits, checkpoint values or shared caches were changed.', '']
    report = {'protocol': manifests[0]['protocol'], 'manifests': manifests, 'cells': cells,
              'native_consistency': consistency,
              'limits': 'Bounded synthetic performance, not accuracy or a universal speed ranking; vLLM outputs varied in some repeated cells.'}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'comparison.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
