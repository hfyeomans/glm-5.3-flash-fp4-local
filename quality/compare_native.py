#!/usr/bin/env python3
"""Validate paired TensorFold tuning runs without relabeling either engine as vLLM."""
import argparse
import json
import math
from pathlib import Path
import re

from report import METRICS, load_run


def compare(reference, candidate):
    a, ra, ca, sa = load_run(reference)
    b, rb, cb, sb = load_run(candidate)
    if a['backend'] != 'tensorfold' or b['backend'] != 'tensorfold':
        raise ValueError('Both runs must be TensorFold')
    keys = ('profile_sha256', 'template_sha256', 'selected_ids', 'lock_sha256', 'client_sha256', 'pilot')
    for key in keys:
        if a[key] != b[key]:
            raise ValueError(f'Incomparable {key}')
    if set(ca) != set(cb):
        raise ValueError('Different request sets')
    changed = []
    for key in sorted(ca):
        x, y = ca[key], cb[key]
        if any(x[k] != y[k] for k in ('request_sha256', 'prompt_token_sha256', 'prompt_tokens')):
            raise ValueError(f'Different request/token identity for {key}')
        hx, hy = [r['response'].get('tensorfold', {}).get('token_sha256', '') for r in (x, y)]
        if not all(re.fullmatch(r'[0-9a-f]{64}', h) for h in (hx, hy)):
            raise ValueError('Missing native output token hash')
        if hx != hy:
            changed.append({'task': key[0], 'doc_id': key[1], 'reference_hash': hx, 'candidate_hash': hy})
    scores = []
    for task, filt, metric, title in METRICS:
        ids = a['selected_ids'][task]
        pairs = [[int(s[(task, i, filt)][metric]) for i in ids] for s in (sa, sb)]
        for result, values in zip((ra, rb), pairs):
            if not math.isclose(sum(values) / len(values), result['results'][task][metric + ',' + filt], abs_tol=1e-12):
                raise ValueError('Official score differs from per-item grades')
        x, y = pairs
        scores.append({'title': title, 'n': len(ids), 'reference_correct': sum(x), 'candidate_correct': sum(y),
                       'changed_grades': sum(u != v for u, v in zip(x, y)),
                       'reference_only_correct': sum(u == 1 and v == 0 for u, v in zip(x, y)),
                       'candidate_only_correct': sum(u == 0 and v == 1 for u, v in zip(x, y))})
    return {'reference': str(reference), 'candidate': str(candidate), 'pilot': a['pilot'],
            'reference_image_id': a['image_id'], 'candidate_image_id': b['image_id'],
            'reference_settings': a['engine_settings'], 'candidate_settings': b['engine_settings'],
            'profile_sha256': a['profile_sha256'], 'client_sha256': a['client_sha256'],
            'requests': len(ca), 'identical_output_token_hashes': len(ca) - len(changed),
            'changed_outputs': changed, 'scores': scores,
            'diagnostics': {name: {'truncated': sum(r['truncated'] for r in calls.values()),
                                  'unclosed_reasoning': sum(not r['reasoning_closed'] for r in calls.values())}
                            for name, calls in [('reference', ca), ('candidate', cb)]},
            'limits': 'Exact agreement applies only to the recorded sample and settings. It is not a proof of universal equivalence, a long-context quality score or a comparison to unquantized weights.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2); out.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
