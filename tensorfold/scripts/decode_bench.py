#!/usr/bin/env python3
"""Fixed cold requests for the bounded NVFP4 decode-warp experiment."""
import argparse
import hashlib
import json
from urllib.parse import urlparse
from pathlib import Path
import random
import statistics
import subprocess
import time

from context_check import call


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def workloads(model):
    topics = [
        ('hash-table', 'Explain hash table collisions, chaining, open addressing and resizing. Include Python examples.'),
        ('database', 'Explain database transactions, isolation levels, write-ahead logging and recovery. Include SQL examples.'),
        ('archive', '\n'.join(f'Record {i}: the archive entry is blue.' for i in range(650))
         + '\nExplain a robust design for indexing this archive, validating records and recovering damaged indexes.'),
    ]
    return [(name, {'model': model, 'messages': [{'role': 'user', 'content': prompt +
             '\nWrite a detailed tutorial of at least 1000 words.'}],
             'temperature': 0, 'top_p': 1, 'max_tokens': 256,
             'chat_template_kwargs': {'enable_thinking': False}}) for name, prompt in topics]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8001')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=4)
    args = p.parse_args()
    if args.rounds < 2:
        p.error('At least two rounds required')
    args.base_url = args.base_url.rstrip('/')
    endpoint = urlparse(args.base_url)
    if endpoint.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Local benchmark endpoint required')
    cases = workloads('glm-5.3-flash-fp4-local')
    state = json.loads(subprocess.check_output(['docker', 'inspect', 'glm53-fp4-tensorfold'], text=True))[0]
    env = dict(x.split('=', 1) for x in state['Config']['Env'])
    expected = {'TF_CONTEXT': '16384', 'TF_PARALLEL': '1', 'TF_PROMPT_CACHE_GIB': '0',
                'TF_MEMORY_RESERVE_GIB': '', 'TF_PROFILE': '0'}
    if not state['State']['Running'] or any(env.get(k) != v for k, v in expected.items()):
        raise ValueError('Expected fixed 16K, one-slot, cold-cache, default-reserve unprofiled server')
    if env.get('TF_NVFP4_DECODE_WARPS') not in ('1', '4'):
        raise ValueError('Missing explicit decode warp setting')
    if str(endpoint.port if endpoint.port is not None else (443 if endpoint.scheme == 'https' else 80)) != env.get('TF_PORT'):
        raise ValueError('Benchmark endpoint port does not match the recorded container')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'image_id': state['Image'], 'settings': {k: v for k, v in env.items() if k.startswith('TF_') or k == 'MODEL_PATH'},
                'rounds': args.rounds, 'workload_sha256': digest(cases),
                'client_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'note': 'Bounded 256-token synthetic workload, not an accuracy score. Server decode timing and client end-to-end time are distinct.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    specs = [(name, dict(body, draft=draft)) for name, body in cases for draft in (True, False)]
    counts = {}
    for name, body in specs:
        counts[name] = call(args.base_url, '/tokenize', body, 60)['count']
        if counts[name] + body['max_tokens'] > 16384:
            raise ValueError('Workload exceeds the common context')
    rows = []
    with (args.output / 'requests.jsonl').open('w') as evidence:
        for round_id in range(-1, args.rounds):
            order = list(specs)
            random.Random(1729 + round_id).shuffle(order)
            for name, body in order:
                started_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                start = time.monotonic()
                response = call(args.base_url, '/v1/chat/completions', body, 300)
                seconds = time.monotonic() - start
                usage, tf = response['usage'], response['tensorfold']
                if (usage['prompt_tokens'] != counts[name] or usage['completion_tokens'] != 256
                    or (usage.get('prompt_tokens_details') or {}).get('cached_tokens') is None
                    or usage['prompt_tokens_details']['cached_tokens'] != 0
                    or response['choices'][0]['finish_reason'] != 'length'
                    or not tf.get('token_sha256') or tf['drafts'] != body['draft']):
                    raise ValueError('Incomplete, cached, mismatched or unidentifiable benchmark response')
                row = {'case': name, 'round': round_id, 'warmup': round_id == -1,
                       'request_sha256': digest(body), 'body': body, 'response': response,
                       'client_seconds': seconds, 'started_utc': started_utc}
                evidence.write(json.dumps(row) + '\n'); evidence.flush()
                if round_id >= 0:
                    rows.append(row)
                print(f'{name} round={round_id} draft={body["draft"]} '
                      f'decode={tf["tokens_per_second"]:.2f} tok/s client={seconds:.3f}s', flush=True)
    summary = []
    for name, body in specs:
        selected = [x for x in rows if x['case'] == name and x['body']['draft'] == body['draft']]
        hashes = {x['response']['tensorfold']['token_sha256'] for x in selected}
        if len(hashes) != 1:
            raise ValueError('Repeated greedy outputs changed within the run')
        summary.append({'case': name, 'draft': body['draft'], 'n': len(selected),
                        'prompt_tokens': counts[name], 'token_sha256': next(iter(hashes)),
                        'median_decode_tokens_per_second': statistics.median(x['response']['tensorfold']['tokens_per_second'] for x in selected),
                        'median_prefill_seconds': statistics.median(x['response']['tensorfold']['prefill_seconds'] for x in selected),
                        'median_client_seconds': statistics.median(x['client_seconds'] for x in selected)})
    manifest['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
