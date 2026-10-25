#!/usr/bin/env python3
"""Independent long-context vLLM clients with bounded runtime and scheduler evidence."""
import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'glm-5.3-flash-fp4-local'
REVISION = '76348ddb2be2e35e39ecfc00e75dc07703f4f3d5'


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def call(base, path, body=None, timeout=15):
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get('BENCH_API_KEY') or os.environ.get('VLLM_API_KEY')
    if key:
        headers['Authorization'] = 'Bearer ' + key
    req = urllib.request.Request(base + path, headers=headers,
                                 data=None if body is None else json.dumps(body).encode())
    return urllib.request.urlopen(req, timeout=timeout)


def parse_metrics(raw):
    values = {}
    for line in raw.splitlines():
        m = re.match(r'([^\s{]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)(?:\s|$)', line)
        if m and m[1].startswith('vllm:') and not m[1].endswith(('_bucket', '_created')):
            values[m[1]] = values.get(m[1], 0.) + float(m[2])
    for key in ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:num_preemptions_total'):
        if key not in values:
            raise ValueError('Missing scheduler metric: ' + key)
    return values


def metrics(base):
    with call(base, '/metrics') as response:
        return parse_metrics(response.read().decode())


def gpu_sample(devices):
    raw = subprocess.check_output(['nvidia-smi', '--id=' + ','.join(map(str, devices)),
        '--query-gpu=index,memory.free,memory.used,utilization.gpu,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = []
    for line in raw.strip().splitlines():
        vals = [int(x.strip()) for x in line.split(',')]
        rows.append(dict(zip(('index', 'free_mib', 'used_mib', 'util_percent', 'temperature_c'), vals)))
    if {r['index'] for r in rows} != set(devices):
        raise ValueError('Expected both GPU samples')
    return rows


def prompt_case(tok, window, output_tokens, identity, sustained):
    placeholder = 'CAPACITY_ARCHIVE_CONTENT_PLACEHOLDER'
    task = (f'Independent client {identity}. Read this archive and find the three ARCHIVE SECRET values.\n'
            + placeholder + '\nReturn the three ARCHIVE SECRET values in numbered order, one per line.')
    if sustained:
        task += '\nThen write an extremely long tutorial about archive indexing. Continue with detailed examples.'
    else:
        task += ' No other text.'
    rendered = tok.apply_chat_template([{'role': 'user', 'content': task}], tokenize=False,
        add_generation_prompt=True, reasoning_effort='low', clear_thinking=False)
    if not rendered.endswith('<|assistant|><think>') or rendered.count(placeholder) != 1:
        raise ValueError('Unexpected template; refuse ambiguous reasoning prefix')
    prefix, suffix = (rendered + '</think>\n\n').split(placeholder)
    encode = lambda text: tok.encode(text, add_special_tokens=False)
    ids, end = encode(prefix), encode(suffix)
    target = window - output_tokens
    filler = encode(f'Archive {identity}: an ordinary blue record with no secret value.\n')
    expected = [f'{name}-{identity}-{i}7' for i, name in enumerate(('cedar', 'orchid', 'willow'), 1)]
    positions = []

    def pad_to(size):
        count = size - len(ids)
        if count < 0:
            raise ValueError('Insufficient window for markers')
        whole, partial = divmod(count, len(filler))
        ids.extend(filler * whole + filler[:partial])

    for i, value in enumerate(expected):
        pad_to(int(target * (.1, .5, .9)[i]))
        positions.append(len(ids))
        ids.extend(encode(f'\nARCHIVE SECRET {i + 1} is {value}.\n'))
    pad_to(target - len(end))
    ids.extend(end)
    assert len(ids) == target
    return {'prompt_tokens': target, 'token_sha256': sha(ids), 'expected': expected,
            'needle_token_positions': positions, 'request': {
                'model': MODEL, 'prompt': ids, 'add_special_tokens': False,
                'temperature': 0, 'top_p': 1, 'seed': 1729, 'max_tokens': output_tokens,
                'ignore_eos': sustained, 'cache_salt': identity,
                'return_token_ids': sustained,
                'stream': True, 'stream_options': {'include_usage': True}}}


def worker(base, case_path, output, barrier, timeout):
    output = Path(output)
    with gzip.open(case_path, 'rt') as f:
        case = json.load(f)
    result = {'client': output.name, 'prompt_tokens': case['prompt_tokens'],
              'token_sha256': case['token_sha256'], 'expected': case['expected'],
              'token_ids_required': case['request'].get('return_token_ids', False),
              'emitted_token_count': 0}
    texts = []
    try:
        barrier.wait(timeout=60)
        result['start_monotonic'] = time.monotonic()
        with call(base, '/v1/completions', case['request'], timeout=timeout) as response, \
                (output / 'frames.jsonl').open('w') as frames:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                now = time.monotonic()
                if data == '[DONE]':
                    result['done'] = True
                    break
                frame = json.loads(data)
                frames.write(json.dumps({'monotonic': now, 'frame': frame}) + '\n')
                frames.flush()
                if frame.get('error'):
                    raise ValueError(str(frame['error']))
                if frame.get('usage'):
                    result['usage'] = frame['usage']
                for choice in frame.get('choices', []):
                    if choice.get('prompt_token_ids') is not None:
                        result['server_prompt_ids_match'] = sha(choice['prompt_token_ids']) == case['token_sha256']
                    if choice.get('token_ids'):
                        result.setdefault('first_token_monotonic', now)
                        result['last_token_monotonic'] = now
                        result['emitted_token_count'] += len(choice['token_ids'])
                    text = choice.get('text', '')
                    if text:
                        result.setdefault('first_text_monotonic', now)
                        result['last_text_monotonic'] = now
                        texts.append(text)
                    if choice.get('finish_reason'):
                        result['finish_reason'] = choice['finish_reason']
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        result['end_monotonic'] = time.monotonic()
        result['text'] = ''.join(texts)
        result['retrieved'] = [v in result['text'][:8192] for v in case['expected']]
        (output / 'response.json').write_text(json.dumps(result, indent=2) + '\n')


def summarize(rows, samples, before, after, sustained, output_tokens):
    complete = all(r.get('done') and not r.get('error') and r.get('usage') and
        r['usage']['prompt_tokens'] == r['prompt_tokens'] and
        r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', -1) == 0 and
        0 < r['usage']['completion_tokens'] <= output_tokens and
        (not r.get('token_ids_required') or (r.get('server_prompt_ids_match') and
         r.get('emitted_token_count') == r['usage']['completion_tokens'])) and
        (r.get('finish_reason') == 'length' and r['usage']['completion_tokens'] == output_tokens
         if sustained else r.get('finish_reason') in ('stop', 'length')) for r in rows)
    intervals = [(r.get('first_token_monotonic', r.get('first_text_monotonic')),
                  r.get('last_token_monotonic', r.get('last_text_monotonic'))) for r in rows]
    overlap = (max(0., min(b for _, b in intervals) - max(a for a, _ in intervals))
               if all(a is not None and b is not None for a, b in intervals) else 0.)
    preemptions = after['vllm:num_preemptions_total'] - before['vllm:num_preemptions_total']
    successes = after.get('vllm:request_success_total', 0) - before.get('vllm:request_success_total', 0)
    return {'requests_completed_cleanly': complete, 'retrieval_all_passed': all(all(r['retrieved']) for r in rows),
            'retrieval_format_clean': all(r.get('finish_reason') == 'stop' and not any(m in r['text'] for m in ('<think>', '</think>')) for r in rows),
            'request_isolation_passed': not any(marker in row['text'] for i, row in enumerate(rows)
                for j, other in enumerate(rows) if i != j for marker in other.get('expected', [])),
            'server_idle_after': not after.get('vllm:num_requests_running', 0) and not after.get('vllm:num_requests_waiting', 0),
            'preemptions_delta': preemptions, 'successes_delta': successes,
            'isolated_success_count': successes == len(rows),
            'all_client_decode_interval_overlap_seconds': overlap,
            'decode_interval_basis': 'token_ids' if all(r.get('token_ids_required') for r in rows) else 'visible_text',
            'overlapping_decode_without_preemption': len(rows) > 1 and complete and overlap > .5 and preemptions == 0,
            'peak_running': max(s['metrics']['vllm:num_requests_running'] for s in samples),
            'peak_waiting': max(s['metrics']['vllm:num_requests_waiting'] for s in samples),
            'peak_kv_cache_usage_fraction': max(s['metrics'].get('vllm:kv_cache_usage_perc', 0) for s in samples),
            'prompt_tokens_processed_delta': after.get('vllm:prompt_tokens_total', 0) - before.get('vllm:prompt_tokens_total', 0),
            'min_free_mib': {str(g): min(r['free_mib'] for s in samples for r in s['gpus'] if r['index'] == g) for g in (0, 1)},
            'clients': [{'client': r['client'], 'prompt_tokens': r['prompt_tokens'], 'usage': r.get('usage'),
                'ttft_seconds': r.get('first_text_monotonic', r['end_monotonic']) - r.get('start_monotonic', r['end_monotonic']),
                'server_prompt_ids_match': r.get('server_prompt_ids_match'),
                'emitted_token_count': r.get('emitted_token_count'),
                'elapsed_seconds': r['end_monotonic'] - r.get('start_monotonic', r['end_monotonic']),
                'finish_reason': r.get('finish_reason'), 'retrieved': r['retrieved'],
                'text_sha256': sha(r['text']), 'error': r.get('error')} for r in rows],
            'scope': 'Synthetic capacity/retrieval screen, not broad accuracy. Decode intervals may include stalls. Running count is not fully resident context count; cache occupancy may include inactive prefixes.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--clients', type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument('--output-tokens', type=int, default=256)
    parser.add_argument('--sustained', action='store_true')
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--min-free-mib', type=int, default=1536)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    if urlparse(base).hostname not in ('localhost', '127.0.0.1', '::1'):
        parser.error('Local endpoints only')
    if not 8192 <= args.window <= 1048576 or not 1 <= args.output_tokens < args.window // 2:
        parser.error('Invalid token budget')
    state = json.loads(subprocess.check_output(['docker', 'inspect', 'glm53-fp4-tp2'], text=True))[0]
    command = state['Config']['Cmd']
    if not state['State']['Running'] or state['Config']['Labels'].get('local.glm53.recipe') != 'glm53-fp4-local-v1':
        raise ValueError('Expected owned running recipe')
    devices = sorted({int(i) for req in (state['HostConfig'].get('DeviceRequests') or []) for i in (req.get('DeviceIDs') or [])})
    if len(devices) != 2:
        raise ValueError('Expected exactly two GPU device ids on the serving container')
    if REVISION not in ' '.join(command) or '--speculative-config' not in command:
        raise ValueError('Expected pinned checkpoint and MTP-on baseline')
    for key, expected in {'--max-model-len': '1048576', '--max-num-seqs': '8',
                          '--gpu-memory-utilization': '0.985', '--kv-cache-memory-bytes': '7650410496'}.items():
        if command[command.index(key) + 1] != expected:
            raise ValueError('Unexpected baseline setting: ' + key)
    before = metrics(base)
    if before['vllm:num_requests_running'] or before['vllm:num_requests_waiting']:
        raise ValueError('Other traffic active; refuse mixed measurement')
    args.output.mkdir(parents=True, exist_ok=False)
    model = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'scripts/model.py')], text=True))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model['path'], local_files_only=True, trust_remote_code=False)
    cases = []
    for i in range(args.clients):
        identity = uuid.uuid4().hex[:12]
        case = prompt_case(tok, args.window, args.output_tokens, identity, args.sustained)
        path = args.output / f'client-{i}'
        path.mkdir()
        with gzip.open(path / 'prompt.json.gz', 'wt') as f:
            json.dump(case, f)
        cases.append((path, {k: v for k, v in case.items() if k != 'request'}))
        print(f'Prepared client {i}: {case["prompt_tokens"]} exact input tokens, {args.output_tokens} output budget', flush=True)
    manifest = {'protocol': 'independent-context-v1', 'image_id': state['Image'], 'command': command,
        'model_revision': REVISION, 'client_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'tokenizer_sha256': hashlib.sha256((Path(model['path']) / 'tokenizer.json').read_bytes()).hexdigest(),
        'template_sha256': hashlib.sha256((Path(model['path']) / 'chat_template.jinja').read_bytes()).hexdigest(),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'window': args.window, 'clients': args.clients, 'sustained': args.sustained,
        'max_output_tokens': args.output_tokens, 'timeout_seconds': args.timeout,
        'min_free_mib': args.min_free_mib, 'cases': [case for _, case in cases],
        'prompt_protocol': 'Exact token-ID input, independent salted filler/markers, checkpoint low reasoning with explicitly closed think prefix. Sustained mode ignores EOS solely to hold decode active; its tail is not quality-scored.'}
    (args.output / 'client-source.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    before = metrics(base)
    if before['vllm:num_requests_running'] or before['vllm:num_requests_waiting']:
        raise ValueError('Other traffic arrived during prompt preparation')
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(args.clients + 1)
    processes = [ctx.Process(target=worker, args=(base, str(path / 'prompt.json.gz'), str(path), barrier, args.timeout)) for path, _ in cases]
    samples, failure = [], None
    start = time.monotonic()
    heartbeat = start
    try:
        for p in processes:
            p.start()
        barrier.wait(timeout=60)
        with (args.output / 'telemetry.jsonl').open('w') as trace:
            while True:
                sample = {'monotonic': time.monotonic(), 'metrics': metrics(base), 'gpus': gpu_sample(devices)}
                samples.append(sample)
                trace.write(json.dumps(sample) + '\n'); trace.flush()
                if min(g['free_mib'] for g in sample['gpus']) < args.min_free_mib:
                    raise ValueError('Sampled physical GPU headroom below floor')
                if time.monotonic() - heartbeat > 25:
                    m = sample['metrics']
                    print(f'{time.monotonic()-start:.0f}s running={m["vllm:num_requests_running"]:.0f} waiting={m["vllm:num_requests_waiting"]:.0f} preemptions={m["vllm:num_preemptions_total"]-before["vllm:num_preemptions_total"]:.0f} free={[g["free_mib"] for g in sample["gpus"]]}', flush=True)
                    heartbeat = time.monotonic()
                if not any(p.is_alive() for p in processes):
                    break
                if time.monotonic() - start > args.timeout:
                    raise TimeoutError('Cell wall-time limit exceeded; close only these test clients')
                time.sleep(1)
    except Exception as exc:
        failure = type(exc).__name__ + ': ' + str(exc)
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            if p.pid:
                p.join(timeout=10)
    # Flush metrics and allow disconnected test requests to be retired.
    after = samples[-1]['metrics'] if samples else before
    try:
        after = metrics(base)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (after.get('vllm:request_success_total', 0) - before.get('vllm:request_success_total', 0) == args.clients
                    and not after['vllm:num_requests_running'] and not after['vllm:num_requests_waiting']):
                break
            time.sleep(1)
            after = metrics(base)
    except Exception as exc:
        failure = (failure or '') + '; final metrics unavailable: ' + type(exc).__name__
    rows = []
    for path, case in cases:
        if (path / 'response.json').exists():
            rows.append(json.loads((path / 'response.json').read_text()))
        else:
            rows.append(dict(case, client=path.name, error='Client aborted without final response',
                             text='', retrieved=[False]*3, end_monotonic=time.monotonic()))
    result = summarize(rows, samples, before, after, args.sustained, args.output_tokens) if samples else {}
    result.update(failure=failure, metrics_before=before, metrics_after=after)
    result['capacity_passed'] = bool(result.get('requests_completed_cleanly') and result.get('isolated_success_count') and result.get('server_idle_after') and not failure)
    result['passed'] = bool(result['capacity_passed'] and result.get('retrieval_all_passed') and result.get('request_isolation_passed')
                            and (args.sustained or result.get('retrieval_format_clean')))
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('metrics_')}, indent=2), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
