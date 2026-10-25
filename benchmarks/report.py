#!/usr/bin/env python3
"""Summarize shared engine measurements without conflating prefill, decode or TTFT."""
import argparse
import json
from pathlib import Path


def comparable_settings(manifest):
    """Permit only the MTP option/value to differ between vLLM on/off runs."""
    settings = dict(manifest['settings'])
    if manifest['backend'] != 'vllm':
        return settings
    command = list(settings['command'])
    count = command.count('--speculative-config')
    if manifest['mtp'] not in ('on', 'off') or count != (manifest['mtp'] == 'on'):
        raise ValueError('vLLM launch and recorded MTP mode differ')
    if count:
        index = command.index('--speculative-config')
        if index + 1 == len(command):
            raise ValueError('Missing vLLM speculative configuration')
        spec = json.loads(command[index + 1])
        if spec.get('method') != 'mtp':
            raise ValueError('MTP comparison refuses other drafters')
        del command[index:index + 2]
    settings['command'] = command
    return settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifests, cells, consistency = [], [], []
    for folder in args.runs:
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['backend'] not in ('vllm', 'tensorfold') or manifest['mtp'] not in ('on', 'off', 'both'):
            raise ValueError('Unknown engine or MTP mode')
        comparable_settings(manifest)
        if manifests:
            for key in ('protocol', 'workloads_sha256', 'model_revision', 'client_sha256'):
                if manifest[key] != manifests[0][key]:
                    raise ValueError(f'Incomparable {key}')
        manifests.append(manifest)
        summary = json.loads((folder / 'summary.json').read_text())
        rows = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
        modes = (True, False) if manifest['mtp'] == 'both' else (manifest['mtp'] == 'on',)
        expected = {(case, mode) for case in manifest['cases'] for mode in modes}
        if not expected or {(r['case'], r['mtp']) for r in summary} != expected:
            raise ValueError('Summary coverage differs from manifest cases/modes')
        if {(r['case'], r['draft']) for r in rows if not r['warmup']} != expected:
            raise ValueError('Measured coverage differs from manifest cases/modes')
        for row in summary:
            selected = [r for r in rows if not r['warmup'] and r['case'] == row['case'] and r['draft'] == row['mtp']]
            if (len(selected) != manifest['repeats'] or len(selected) != row['n']
                    or {r['repeat'] for r in selected} != set(range(manifest['repeats']))):
                raise ValueError('Incomplete measured sample set')
            if any(r['response']['usage']['prompt_tokens'] != row['prompt_tokens'] or
                   r['response']['usage']['completion_tokens'] != 256 or
                   (r['response']['usage'].get('prompt_tokens_details') or {}).get('cached_tokens') != 0 or
                   r['response'].get('finish_reason') != 'length' for r in selected):
                raise ValueError('Counts or cold-cache contract differs')
            cells.append(dict(backend=manifest['backend'], run=str(folder), **row))
        if manifest['backend'] == 'tensorfold':
            for case in manifest['cases']:
                selected = [r for r in rows if r['case'] == case]
                hashes = {r['response']['runtime']['token_sha256'] for r in selected}
                consistency.append({'run': str(folder), 'case': case, 'mtp_modes': sorted({r['draft'] for r in selected}),
                                    'distinct_native_token_hashes_including_warmups': len(hashes)})
    for backend in ('vllm', 'tensorfold'):
        backend_runs = [m for m in manifests if m['backend'] == backend]
        if len({m['image_id'] for m in backend_runs}) > 1:
            raise ValueError(f'Incomparable {backend} image_id')
        seen = {json.dumps(comparable_settings(m), sort_keys=True) for m in backend_runs}
        specs = {json.dumps(json.loads(m['settings']['command'][m['settings']['command'].index('--speculative-config') + 1]), sort_keys=True)
                 for m in backend_runs if backend == 'vllm' and m['mtp'] == 'on'}
        if len(specs) > 1:
            raise ValueError('Incomparable vllm MTP configurations across runs')
        if len(seen) > 1:
            raise ValueError(f'Incomparable {backend} launch settings across runs')
    keys = [(x['backend'], x['case'], x['mtp']) for x in cells]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate backend/case/mode; create a separate candidate comparison')
    if {m['backend'] for m in manifests} != {'vllm', 'tensorfold'} or any(m['repeats'] != 3 for m in manifests):
        raise ValueError('The published comparison requires both engines and three measured repeats')
    coverage = {backend: {(x['case'], x['mtp']) for x in cells if x['backend'] == backend}
                for backend in ('vllm', 'tensorfold')}
    if coverage['vllm'] != coverage['tensorfold'] or any(
            {mode for name, mode in coverage['vllm'] if name == case} != {True, False}
            for case, _ in coverage['vllm']):
        raise ValueError('Engines need matching cases with both MTP modes')
    native = next(m for m in manifests if m['backend'] == 'tensorfold')
    settings = native['settings']
    overlap = settings.get('TF_GLM_OVERLAP_PIECES', 'upstream default')
    native_context = int(settings['TF_CONTEXT'])
    native_slots = int(settings['TF_PARALLEL'])
    native_warps = settings.get('TF_NVFP4_DECODE_WARPS', 'upstream default')
    native_reserve = settings.get('TF_MEMORY_RESERVE_GIB') or 'native default'
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
              'speculative configuration. Unique cache salts prevent prefix reuse. '
              f'TensorFold settings: decode warps={native_warps}, configured context={native_context:,} tokens, '
              f'slots={native_slots}, FP8 KV, reserve={native_reserve}, zero retained prompt-cache budget and overlap pieces={overlap}.', '',
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
