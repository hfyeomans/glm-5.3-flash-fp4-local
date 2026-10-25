#!/usr/bin/env python3
"""Compare paired official task scores; refuse changed prompts or incomplete runs."""
import argparse
import json
import math
from pathlib import Path
import random

from common import wilson

METRICS = [
    ('gsm8k_cot', 'flexible-extract', 'exact_match', 'GSM8K flexible exact match'),
    ('gsm8k_cot', 'strict-match', 'exact_match', 'GSM8K strict exact match'),
    ('ifeval', 'none', 'prompt_level_strict_acc', 'IFEval prompt strict'),
    ('ifeval', 'none', 'prompt_level_loose_acc', 'IFEval prompt loose'),
]


def paired_stats(a, b, iterations=10000):
    if len(a) != len(b) or not a:
        raise ValueError('Need equally sized nonempty paired results')
    if any(x not in (0, 1) for x in a + b):
        raise ValueError('Binary grades required')
    differences = [y - x for x, y in zip(a, b)]
    rng = random.Random(1729)
    estimates = sorted(sum(rng.choices(differences, k=len(a))) / len(a) for _ in range(iterations))
    a_only = sum(x == 1 and y == 0 for x, y in zip(a, b))
    b_only = sum(x == 0 and y == 1 for x, y in zip(a, b))
    discordant = a_only + b_only
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / 2 ** discordant)
    return {'n': len(a), 'vllm_correct': sum(a), 'tensorfold_correct': sum(b),
            'vllm_accuracy': sum(a) / len(a), 'tensorfold_accuracy': sum(b) / len(b),
            'vllm_wilson_95': wilson(sum(a), len(a)), 'tensorfold_wilson_95': wilson(sum(b), len(b)),
            'difference_tensorfold_minus_vllm': sum(differences) / len(a),
            'paired_bootstrap_95': [estimates[int(iterations * .025)], estimates[int(iterations * .975) - 1]],
            'vllm_only_correct': a_only, 'tensorfold_only_correct': b_only,
            'both_correct': sum(x == y == 1 for x, y in zip(a, b)),
            'both_incorrect': sum(x == y == 0 for x, y in zip(a, b)),
            'mcnemar_exact_two_sided_p': p}


def load_run(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    results = json.loads((folder / 'results.json').read_text())
    rows = [json.loads(x) for x in (folder / 'requests.jsonl').read_text().splitlines()]
    calls = {(x['task'], x['doc_id']): x for x in rows}
    expected = {(task, i) for task, ids in manifest['selected_ids'].items() for i in ids}
    if len(calls) != len(rows) or set(calls) != expected:
        raise ValueError('Missing or duplicate request evidence')
    samples = {}
    for task, items in results['samples'].items():
        for item in items:
            key = (task, item['doc_id'], item['filter'])
            if key in samples:
                raise ValueError('Duplicate score evidence')
            call = calls[(task, item['doc_id'])]
            if item['arguments'][0][0] != call['body']['prompt'] or item['resps'][0][0] != call['scored_answer']:
                raise ValueError('Scored sample does not match recorded prompt/answer')
            samples[key] = item
    return manifest, results, calls, samples


def validate_pair(a, b, calls_a, calls_b):
    if a['backend'] != 'vllm' or b['backend'] != 'tensorfold':
        raise ValueError('Expected vLLM first and TensorFold second')
    for key in ('profile_sha256', 'template_sha256', 'selected_ids', 'lock_sha256', 'client_sha256', 'pilot'):
        if a[key] != b[key]:
            raise ValueError(f'Runs are not comparable: {key} differs')
    if set(calls_a) != set(calls_b):
        raise ValueError('Paired sample sets differ')
    for key, row in calls_a.items():
        if any(row[field] != calls_b[key][field] for field in ('request_sha256', 'prompt_token_sha256', 'prompt_tokens')):
            raise ValueError(f'Paired requests/tokens differ for {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vllm', type=Path)
    parser.add_argument('tensorfold', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    a, ra, ca, sa = load_run(args.vllm)
    b, rb, cb, sb = load_run(args.tensorfold)
    validate_pair(a, b, ca, cb)
    report = {'vllm_run': str(args.vllm), 'tensorfold_run': str(args.tensorfold),
              'profile': a['profile'], 'pilot': a['pilot'], 'metrics': [],
              'official_scores': {'vllm': ra['results'], 'tensorfold': rb['results']},
              'diagnostics': {}, 'disagreements': []}
    for backend, calls in [('vllm', ca), ('tensorfold', cb)]:
        report['diagnostics'][backend] = {
            'requests': len(calls), 'truncated': sum(x['truncated'] for x in calls.values()),
            'unclosed_reasoning': sum(not x['reasoning_closed'] for x in calls.values()),
            'total_completion_tokens': sum(x['response']['usage']['completion_tokens'] for x in calls.values()),
            'request_seconds': sum(x['seconds'] for x in calls.values()),
            'cached_prompt_tokens': sum(x['response']['usage']['prompt_tokens_details']['cached_tokens'] for x in calls.values())}
    for task, filt, metric, title in METRICS:
        ids = a['selected_ids'][task]
        scores_a = [int(sa[(task, i, filt)][metric]) for i in ids]
        scores_b = [int(sb[(task, i, filt)][metric]) for i in ids]
        stats = paired_stats(scores_a, scores_b)
        for name, score in [('vllm', sum(scores_a) / len(ids)), ('tensorfold', sum(scores_b) / len(ids))]:
            if not math.isclose(score, report['official_scores'][name][task][metric + ',' + filt], abs_tol=1e-12):
                raise ValueError('Per-item grades disagree with official aggregate')
        report['metrics'].append(dict(task=task, filter=filt, metric=metric, title=title, **stats))
        for i, x, y in zip(ids, scores_a, scores_b):
            if x != y:
                report['disagreements'].append({'task': task, 'doc_id': i, 'filter': filt,
                                               'metric': metric, 'vllm_correct': x, 'tensorfold_correct': y,
                                               'vllm_answer': ca[(task, i)]['scored_answer'],
                                               'tensorfold_answer': cb[(task, i)]['scored_answer']})
    lines = ['# Initial quality comparison', '',
             'PILOT ONLY.' if a['pilot'] else '128 fixed examples per task; same checkpoint, prompts and tokenizer IDs.', '',
             '| Metric | vLLM (95% Wilson interval) | TensorFold (95% Wilson interval) | TF − vLLM (paired 95% bootstrap interval) |',
             '|---|---|---|---|']
    def score(m, prefix):
        low, high = m[prefix + '_wilson_95']
        return f"{m[prefix + '_correct']}/{m['n']} = {100 * m[prefix + '_accuracy']:.2f}% ({100 * low:.1f}–{100 * high:.1f}%)"
    for m in report['metrics']:
        low, high = m['paired_bootstrap_95']
        lines.append(f"| {m['title']} | {score(m, 'vllm')} | {score(m, 'tensorfold')} | {100 * m['difference_tensorfold_minus_vllm']:+.2f} pp ({100 * low:+.2f} to {100 * high:+.2f}) |")
    lines += ['', 'Diagnostics:', '']
    for name, d in report['diagnostics'].items():
        lines.append(f"- {name}: {d['requests']} responses; {d['truncated']} output-budget truncations; {d['unclosed_reasoning']} missing reasoning closures.")
    lines += ['', 'Intervals describe uncertainty in this selected benchmark sample, not all model capabilities. '
              'The paired bootstrap uses 10,000 resamples with seed 1729. An interval spanning zero is inconclusive, '
              'not proof of equivalence. With no observed paired differences the bootstrap interval is degenerate; '
              'it cannot rule out unseen disagreements. These short-context scores do not qualify 1M context or '
              'measure quantization loss against an unquantized model.', '']
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    (args.output / 'comparison.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
