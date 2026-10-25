#!/usr/bin/env python3
"""Cold shared-prompt engine benchmark with per-request server metrics and client TTFT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from urllib.parse import urlparse

from prepare import digest

ROOT = Path(__file__).resolve().parents[1]

METRICS = ('vllm:request_prefill_time_seconds', 'vllm:request_decode_time_seconds',
           'vllm:time_to_first_token_seconds', 'vllm:request_prompt_tokens', 'vllm:request_generation_tokens')


def request(base, path, body=None):
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('BENCH_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['BENCH_API_KEY']
    return urllib.request.Request(base + path, headers=headers,
                                  data=None if body is None else json.dumps(body).encode())


def call(base, path, body=None):
    with urllib.request.urlopen(request(base, path, body), timeout=600) as response:
        return json.load(response)


def metrics(base):
    with urllib.request.urlopen(request(base, '/metrics'), timeout=30) as response:
        raw = response.read().decode()
    totals = {}
    for line in raw.splitlines():
        match = re.match(r'([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)', line)
        if match:
            key = match[1]
            if (key in ('vllm:num_requests_running', 'vllm:num_requests_waiting') or
                    any(key == name + suffix for name in METRICS for suffix in ('_sum', '_count'))):
                totals[key] = totals.get(key, 0.0) + float(match[2])
    if len(totals) != 2 * len(METRICS) + 2:
        raise ValueError('Missing required vLLM metrics or scheduler gauges')
    return totals


def metric_delta(before, after, prompt_tokens, completion_tokens):
    for name in METRICS:
        if after[name + '_count'] - before[name + '_count'] != 1:
            raise ValueError('Metrics do not isolate exactly one request; concurrent traffic or delayed metrics')
    values = {name: after[name + '_sum'] - before[name + '_sum'] for name in METRICS}
    if values['vllm:request_prompt_tokens'] != prompt_tokens or values['vllm:request_generation_tokens'] != completion_tokens:
        raise ValueError('Server metric tokens disagree with completion usage')
    return {'prefill_seconds': values[METRICS[0]], 'decode_seconds': values[METRICS[1]],
            'server_ttft_seconds': values[METRICS[2]],
            'decode_tokens_per_second': (completion_tokens - 1) / values[METRICS[1]]}


def stream(base, body):
    frames, texts = [], []
    usage = runtime = None
    finish = None
    first = last = None
    start = time.monotonic()
    with urllib.request.urlopen(request(base, '/v1/completions', body), timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            value = line[5:].strip()
            if value == '[DONE]':
                break
            frame = json.loads(value)
            if frame.get('error'):
                raise ValueError(str(frame['error']))
            now = time.monotonic() - start
            frames.append({'seconds': now, 'data': frame})
            if frame.get('usage'):
                usage = frame['usage']
            if frame.get('tensorfold'):
                runtime = frame['tensorfold']
            for choice in frame.get('choices', []):
                text = choice.get('text', '')
                if text:
                    first = now if first is None else first
                    last = now; texts.append(text)
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
    if first is None or usage is None:
        raise ValueError('Missing streamed text or usage')
    return {'frames': frames, 'text': ''.join(texts), 'usage': usage, 'runtime': runtime,
            'finish_reason': finish, 'client_ttft_seconds': first,
            'client_seconds': time.monotonic() - start, 'last_text_seconds': last}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=('vllm', 'tensorfold'), required=True)
    p.add_argument('--mtp', choices=('on', 'off', 'both'), required=True)
    p.add_argument('--workloads', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--cases', nargs='+')
    args = p.parse_args()
    if args.repeats < 1 or (args.backend == 'vllm' and args.mtp == 'both'):
        p.error('vLLM requires a separate on/off launch; repeats must be positive')
    base = (args.base_url or ('http://127.0.0.1:8000' if args.backend == 'vllm' else 'http://127.0.0.1:8001')).rstrip('/')
    if urlparse(base).hostname not in ('127.0.0.1', 'localhost', '::1'):
        p.error('Benchmark endpoints must be local')
    workload = json.loads(args.workloads.read_text())
    unsigned = {k: v for k, v in workload.items() if k != 'workloads_sha256'}
    if digest(unsigned) != workload['workloads_sha256']:
        raise ValueError('Workload hash mismatch')
    cases = [x for x in workload['cases'] if not args.cases or x['name'] in args.cases]
    if not cases or (args.cases and set(args.cases) != {x['name'] for x in cases}):
        raise ValueError('Unknown or empty case selection')
    container = 'glm53-fp4-tp2' if args.backend == 'vllm' else 'glm53-fp4-tensorfold'
    state = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True))[0]
    env = dict(x.split('=', 1) for x in state['Config']['Env'])
    command = state['Config']['Cmd'] or []
    if not state['State']['Running']:
        raise ValueError('Expected running recipe container')
    if args.backend == 'vllm' and state['Config']['Labels'].get('local.glm53.recipe') != 'glm53-fp4-local-v1':
        raise ValueError('vLLM container is not owned by this recipe')
    model = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'scripts/model.py')], text=True))
    if args.backend == 'vllm':
        enabled = '--speculative-config' in command
        if enabled and json.loads(command[command.index('--speculative-config') + 1]).get('method') != 'mtp':
            raise ValueError('This MTP comparison refuses other drafters; DFlash2 needs a separately labelled benchmark mode')
        if enabled != (args.mtp == 'on'):
            raise ValueError('Requested MTP mode does not match vLLM launch')
        settings = {'command': command}
        if command[2] != model['container_path']:
            raise ValueError('vLLM container does not serve the locked checkpoint path')
    else:
        settings = {k: v for k, v in env.items() if k.startswith(('TF_', 'NCCL_')) or k == 'MODEL_PATH'}
        if env.get('TF_PROMPT_CACHE_GIB') != '0' or env.get('TF_PROFILE') != '0':
            raise ValueError('TensorFold needs retained cache off and profiling off')
        if env.get('MODEL_PATH') != model['container_path']:
            raise ValueError('TensorFold container does not serve the locked checkpoint path')
    endpoint = urlparse(base)
    if str(endpoint.port or (443 if endpoint.scheme == 'https' else 80)) != env.get('PORT' if args.backend == 'vllm' else 'TF_PORT'):
        raise ValueError('Benchmark endpoint port does not match the recorded container')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'protocol': workload['protocol'], 'backend': args.backend, 'mtp': args.mtp,
                'workloads_sha256': workload['workloads_sha256'], 'model_revision': workload['model_revision'],
                'image_id': state['Image'], 'settings': settings, 'repeats': args.repeats,
                'cases': [x['name'] for x in cases], 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'client_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for case in cases:
        remote = call(base, '/tokenize', {'model': 'glm-5.3-flash-fp4-local', 'prompt': case['prompt'], 'add_special_tokens': False})
        if remote['tokens'] != case['tokens']:
            raise ValueError('Server tokenizer differs from frozen local token IDs')
    modes = [True, False] if args.mtp == 'both' else [args.mtp == 'on']
    rows = []
    with (args.output / 'requests.jsonl').open('w') as evidence:
        for repeat in range(-1, args.repeats):
            cells = [(case, draft) for case in cases for draft in modes]
            random.Random(1729 + repeat).shuffle(cells)
            for case, draft in cells:
                body = {'model': 'glm-5.3-flash-fp4-local', 'prompt': case['prompt'],
                        'temperature': 0, 'top_p': 1, 'seed': workload['seed'], 'max_tokens': workload['max_tokens'],
                        'add_special_tokens': False, 'stream': True, 'stream_options': {'include_usage': True}}
                if args.backend == 'tensorfold':
                    body['draft'] = draft
                    before = None
                else:
                    body['cache_salt'] = str(uuid.uuid4())
                    before = metrics(base)
                    if before['vllm:num_requests_running'] or before['vllm:num_requests_waiting']:
                        raise ValueError('Other traffic active at timing start')
                response = stream(base, body)
                usage = response['usage']
                row = {'case': case['name'], 'draft': draft, 'repeat': repeat, 'warmup': repeat < 0,
                       'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                       'prompt_sha256': case['prompt_sha256'], 'token_sha256': case['token_sha256'],
                       'request_body': body, 'response': response, 'text_sha256': digest(response['text'])}
                if args.backend == 'vllm':
                    deadline = time.monotonic() + 30
                    while True:
                        after = metrics(base)
                        if after[METRICS[0] + '_count'] > before[METRICS[0] + '_count']:
                            break
                        if time.monotonic() > deadline:
                            raise ValueError('vLLM metrics did not report the completed request')
                        time.sleep(.2)
                    if after['vllm:num_requests_running'] or after['vllm:num_requests_waiting']:
                        raise ValueError('Other traffic active at timing end')
                    timing = metric_delta(before, after, case['prompt_tokens'], workload['max_tokens'])
                    row['metrics_before'], row['metrics_after'] = before, after
                else:
                    tf = response['runtime']
                    if tf is None or tf['drafts'] != draft or not tf.get('token_sha256'):
                        raise ValueError('Missing native runtime/mode/token identity')
                    timing = {'prefill_seconds': tf['prefill_seconds'], 'decode_seconds': tf['seconds'],
                              'server_ttft_seconds': tf['time_to_first_token'], 'decode_tokens_per_second': tf['tokens_per_second']}
                row['timing'] = timing
                evidence.write(json.dumps(row) + '\n'); evidence.flush()
                details = usage.get('prompt_tokens_details') or {}
                if 'cached_tokens' not in details:
                    raise ValueError('Response is missing prompt_tokens_details.cached_tokens')
                if (usage['prompt_tokens'] != case['prompt_tokens'] or usage['completion_tokens'] != workload['max_tokens']
                    or details['cached_tokens'] != 0 or response['finish_reason'] != 'length'):
                    raise ValueError('Token counts, cold cache or output length contract failed')
                if repeat >= 0:
                    rows.append(row)
                print(f'{case["name"]} mtp={draft} repeat={repeat} decode={timing["decode_tokens_per_second"]:.2f} '
                      f'prefill={timing["prefill_seconds"]:.3f}s TTFT={response["client_ttft_seconds"]:.3f}s', flush=True)
    summary = []
    for case in cases:
        for draft in modes:
            selected = [x for x in rows if x['case'] == case['name'] and x['draft'] == draft]
            summary.append({'case': case['name'], 'prompt_tokens': case['prompt_tokens'], 'mtp': draft,
                            'n': len(selected), 'distinct_output_texts': len({x['text_sha256'] for x in selected}),
                            **{k: statistics.median(x['timing'][k] for x in selected) for k in selected[0]['timing']},
                            'prefill_tokens_per_second': statistics.median(case['prompt_tokens'] / x['timing']['prefill_seconds'] for x in selected),
                            'client_ttft_seconds': statistics.median(x['response']['client_ttft_seconds'] for x in selected),
                            'end_to_end_tokens_per_second': statistics.median(workload['max_tokens'] / x['response']['client_seconds'] for x in selected)})
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
