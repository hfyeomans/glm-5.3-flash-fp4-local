#!/usr/bin/env python3
"""Compare next-token distributions at fixed histories; a diagnostic, not an accuracy score."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REVISION = '76348ddb2be2e35e39ecfc00e75dc07703f4f3d5'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def call(path, body):
    req = urllib.request.Request('http://127.0.0.1:8001' + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def prepare(source, output):
    from transformers import AutoTokenizer
    model = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'scripts/model.py')], text=True))
    tok = AutoTokenizer.from_pretrained(model['path'], local_files_only=True, trust_remote_code=False)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    cases = []
    for task in sorted({r['task'] for r in rows}):
        for r in [r for r in rows if r['task'] == task][:8]:
            prompt = r['body']['prompt']
            answer_ids = tok.encode(r['response']['choices'][0]['text'], add_special_tokens=False)
            text = prompt + tok.decode(answer_ids[:128], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            ids = tok.encode(text, add_special_tokens=False)
            original = tok.encode(prompt, add_special_tokens=False)
            if ids[:len(original)] != original:
                raise ValueError('Prompt/answer boundary retokenizes; fixed history not preserved')
            cases.append({'task': task, 'doc_id': r['doc_id'], 'prompt': text, 'ids': ids, 'score_from': len(original)})
    value = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'cases': cases}
    value['sha256'] = digest(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(value, f)
    print(f'Prepared {len(cases)} fixed histories')


def run(source, output):
    data = json.loads(source.read_text())
    if data['sha256'] != digest({k: v for k, v in data.items() if k != 'sha256'}):
        raise ValueError('History digest mismatch')
    state = json.loads(subprocess.check_output(['docker', 'inspect', 'glm53-fp4-tensorfold'], text=True))[0]
    env = dict(x.split('=', 1) for x in state['Config']['Env'])
    if not state['State']['Running'] or env.get('TF_PROMPT_CACHE_GIB') != '0' or REVISION not in env.get('MODEL_PATH', ''):
        raise ValueError('Require running pinned TensorFold checkpoint with retained cache disabled')
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'history_sha256': data['sha256'], 'image_id': state['Image'], 'model_revision': REVISION,
                'client_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'cases': [[c['task'], c['doc_id']] for c in data['cases']],
                'settings': {k: v for k, v in env.items() if k.startswith(('TF_', 'NCCL_')) or k == 'MODEL_PATH'}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (output / 'rows.jsonl').open('w') as f:
        for c in data['cases']:
            ids = call('/tokenize', {'model': 'glm-5.3-flash-fp4-local', 'prompt': c['prompt'], 'add_special_tokens': False})['tokens']
            if ids != c['ids']:
                raise ValueError('Server tokenizer differs')
            response = call('/v1/completions', {'model': 'glm-5.3-flash-fp4-local', 'prompt': c['prompt'],
                            'add_special_tokens': False, 'max_tokens': 1, 'temperature': 0, 'draft': False, 'prompt_logprobs': 2})
            lp = response['choices'][0].get('prompt_logprobs')
            if lp is None or len(lp) != len(ids) or lp[0] is not None or response['usage']['prompt_tokens'] != len(ids):
                raise ValueError('Missing/alignment-invalid prompt logprobs')
            if response['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) != 0:
                raise ValueError('Expected cold fixed-history request')
            for index in range(c['score_from'], len(ids)):
                if str(ids[index]) not in lp[index] or len(lp[index]) < 2:
                    raise ValueError('Missing actual next-token probability or alternatives')
                if any(not math.isfinite(v['logprob']) for v in lp[index].values()):
                    raise ValueError('Nonfinite next-token probability')
            f.write(json.dumps({'task': c['task'], 'doc_id': c['doc_id'], 'score_from': c['score_from'], 'ids': ids, 'response': response}) + '\n'); f.flush()
            print(c['task'], c['doc_id'], len(ids) - c['score_from'], 'scored positions', flush=True)


def compare(left, right, output):
    lm, rm = [json.loads((p / 'manifest.json').read_text()) for p in (left, right)]
    if any(lm[k] != rm[k] for k in ('history_sha256', 'model_revision', 'client_sha256', 'cases')):
        raise ValueError('Different fixed histories, model or client')
    a, b = [[json.loads(line) for line in (p / 'rows.jsonl').read_text().splitlines()] for p in (left, right)]
    if [[r['task'], r['doc_id']] for r in a] != lm['cases'] or [[r['task'], r['doc_id']] for r in b] != rm['cases']:
        raise ValueError('Incomplete or reordered runs')
    deltas, flips = [], []
    for x, y in zip(a, b):
        if any(x[k] != y[k] for k in ('task', 'doc_id', 'score_from', 'ids')):
            raise ValueError('Unmatched histories')
        for index in range(x['score_from'], len(x['ids'])):
            lx, ly = [z['response']['choices'][0]['prompt_logprobs'][index] for z in (x, y)]
            token = str(x['ids'][index])
            deltas.append(abs(lx[token]['logprob'] - ly[token]['logprob']))
            tops = [sorted(lp, key=lambda t: (-lp[t]['logprob'], int(t))) for lp in (lx, ly)]
            if tops[0][0] != tops[1][0]:
                flips.append({'task': x['task'], 'doc_id': x['doc_id'], 'position': index,
                              'left_top': tops[0][0], 'right_top': tops[1][0],
                              'left_margin': lx[tops[0][0]]['logprob'] - lx[tops[0][1]]['logprob']})
    result = {'scope': 'Fixed-history up to the first 128 completion tokens of 8 GSM8K and 8 IFEval baseline outputs; sampled diagnostic, not full model equivalence',
              'history_sha256': lm['history_sha256'], 'positions': len(deltas), 'top1_flips': flips,
              'max_abs_target_logprob_change': max(deltas), 'mean_abs_target_logprob_change': sum(deltas) / len(deltas)}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'compare'])
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--right', type=Path)
    args = p.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if args.command == 'compare':
        if not args.right:
            p.error('compare requires --right')
        compare(args.source, args.right, args.output)
    else:
        globals()[args.command](args.source, args.output)


if __name__ == '__main__':
    main()
