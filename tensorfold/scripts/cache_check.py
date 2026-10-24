#!/usr/bin/env python3
"""Bounded cold/warm, drafted/plain retrieval at the recipe's normal 8K settings."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from context_check import call, fit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8001')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = json.loads(subprocess.check_output(['docker', 'inspect', 'glm53-fp4-tensorfold'], text=True))[0]
    env = dict(x.split('=', 1) for x in state['Config']['Env'])
    expected = {'TF_CONTEXT': '8192', 'TF_PARALLEL': '1', 'TF_PROMPT_CACHE_GIB': '1',
                'TF_MEMORY_RESERVE_GIB': '', 'TF_PROFILE': '0', 'TF_NVFP4_DECODE_WARPS': '1'}
    if not state['State']['Running'] or any(env.get(k) != v for k, v in expected.items()):
        raise ValueError('Expected the one-warp 8K, one-slot, 1 GiB prompt-cache profile')
    report = {'image_id': state['Image'], 'settings': expected, 'model_path': env['MODEL_PATH'],
              'client_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'checks': [], 'passed': False,
              'scope': 'Synthetic retrieval and cached/drafted consistency, not broad task accuracy'}
    for target in (3648, 8000):
        body, secrets, positions, count, records = fit(args.base_url, target, 'glm-5.3-flash-fp4-local', 60)
        check = {'target': target, 'count': count, 'expected': secrets, 'needle_rows': positions,
                 'records': records, 'body': body, 'runs': []}
        report['checks'].append(check)
        for i, draft in enumerate((True, True, False, False)):
            start = time.monotonic()
            response = call(args.base_url, '/v1/chat/completions', dict(body, draft=draft), 300)
            cached = response['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
            content = response['choices'][0]['message'].get('content', '')
            passed = (all(s in content for s in secrets)
                      and response['usage']['prompt_tokens'] == count
                      and response['choices'][0]['finish_reason'] == 'stop'
                      and response['tensorfold']['drafts'] == draft
                      and ((cached == 0) if i == 0 else (cached > 0)))
            check['runs'].append({'draft': draft, 'seconds': time.monotonic() - start,
                                  'passed': passed, 'cached_tokens': cached, 'response': response})
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f'{count} prompt tokens, draft={draft}, cached={cached}, passed={passed}', flush=True)
            if not passed:
                raise ValueError('Retrieval, usage, mode, finish or required cache state differs')
        hashes = {x['response']['tensorfold'].get('token_sha256') for x in check['runs']}
        check['all_output_hashes_match'] = len(hashes) == 1 and None not in hashes and '' not in hashes
        if not check['all_output_hashes_match']:
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            raise ValueError('Cold/warm or drafted/plain output hashes differ')
    report['passed'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS both lengths: cold/warm and drafted/plain retrieval and token hashes', flush=True)


if __name__ == '__main__':
    main()
