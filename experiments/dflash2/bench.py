#!/usr/bin/env python3
"""Matched single-client vLLM drafting comparison with explicit mode attribution."""
import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import run as shared

REVISION = '76348ddb2be2e35e39ecfc00e75dc07703f4f3d5'
LOCK = json.loads((ROOT / 'config/dflash2-lock.json').read_text())
IMAGE = 'sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29'


def validate_server(state, mode, base):
    config = state['Config']
    command = config['Cmd']
    if not state['State']['Running'] or config['Labels'].get('local.glm53.recipe') != 'glm53-fp4-local-v1':
        raise ValueError('Expected owned running recipe')
    endpoint = urlparse(base)
    if endpoint.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Local endpoint required')
    if str(endpoint.port if endpoint.port is not None else (443 if endpoint.scheme == 'https' else 80)) != command[command.index('--port') + 1]:
        raise ValueError('Endpoint differs from inspected container port')
    if state['Image'] != IMAGE or command[:2] != ['vllm', 'serve'] or REVISION not in command[2]:
        raise ValueError('Expected pinned image and NVIDIA checkpoint')
    spec = json.loads(command[command.index('--speculative-config') + 1]) if '--speculative-config' in command else None
    actual = 'none' if spec is None else {'mtp': 'mtp', 'dflash': 'dflash2'}.get(spec.get('method'))
    if actual != mode:
        raise ValueError('Requested drafter differs from launch')
    if mode == 'dflash2' and ('models--incoai--GLM-5.3-Flash-DFlash2/snapshots/' + LOCK['revision']) not in spec.get('model', ''):
        raise ValueError('Expected locked DFlash2 snapshot')
    devices = sorted({int(x) for r in state['HostConfig']['DeviceRequests'] for x in r.get('DeviceIDs', [])})
    if len(devices) != 2:
        raise ValueError('Expected two GPU IDs')
    return spec, devices


def read_metrics(base):
    with shared.urllib.request.urlopen(shared.request(base, '/metrics'), timeout=30) as response:
        raw = response.read().decode()
    totals = {}
    for line in raw.splitlines():
        match = re.match(r'([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)', line)
        if match and match[1].startswith('vllm:'):
            totals[match[1]] = totals.get(match[1], 0) + float(match[2])
    for key in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
        if key not in totals:
            raise ValueError('Missing scheduler gauge: ' + key)
    return totals



def require_idle(totals):
    if totals['vllm:num_requests_running'] or totals['vllm:num_requests_waiting']:
        raise ValueError('Other traffic active at timing start')


def completion_metrics(base, before, timeout=30):
    deadline = time.monotonic() + timeout
    count_key = shared.METRICS[0] + '_count'
    while True:
        after = read_metrics(base)
        completed = after[count_key] - before[count_key]
        if completed > 1:
            raise ValueError('Metrics do not isolate exactly one request')
        if (completed == 1 and not after['vllm:num_requests_running']
                and not after['vllm:num_requests_waiting']):
            return after
        if time.monotonic() >= deadline:
            raise ValueError('Request metrics or idle scheduler boundary delayed')
        time.sleep(.2)


def gpu_sample(devices):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = {int(a): int(b) for a, b in (line.split(',') for line in raw.strip().splitlines())}
    return {str(i): rows[i] for i in devices}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('mtp', 'dflash2', 'none'), required=True)
    parser.add_argument('--container', default='glm53-fp4-dflash2-probe')
    parser.add_argument('--base-url', default='http://127.0.0.1:8002')
    parser.add_argument('--workloads', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-tokens', type=int, default=256)
    parser.add_argument('--cases', nargs='+', default=['prefill-8192', 'prefill-32768'])
    args = parser.parse_args()
    if args.repeats < 1 or args.output_tokens < 2:
        parser.error('Positive repeats and at least two output tokens required')
    state = json.loads(subprocess.check_output(['docker', 'inspect', args.container], text=True))[0]
    spec, devices = validate_server(state, args.mode, args.base_url)
    workload = json.loads(args.workloads.read_text())
    if workload['model_revision'] != REVISION or shared.digest({k:v for k,v in workload.items() if k != 'workloads_sha256'}) != workload['workloads_sha256']:
        raise ValueError('Workload identity mismatch')
    cases = [x for x in workload['cases'] if x['name'] in args.cases]
    if {x['name'] for x in cases} != set(args.cases):
        raise ValueError('Unknown workload case')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'mode': args.mode, 'image': state['Image'], 'container_id': state['Id'],
                'command': state['Config']['Cmd'], 'speculative_config': spec,
                'workloads_sha256': workload['workloads_sha256'], 'output_tokens': args.output_tokens,
                'client_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'repeats': args.repeats,
                'target_revision': REVISION, 'draft_lock': LOCK if args.mode == 'dflash2' else None}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for case in cases:
        got = shared.call(args.base_url, '/tokenize', {'model': 'glm-5.3-flash-fp4-local', 'prompt': case['prompt'], 'add_special_tokens': False})
        if got['tokens'] != case['tokens']:
            raise ValueError('Tokenizer ID mismatch')
    rows, samples, errors = [], [], []
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                sample = gpu_sample(devices)
                samples.append({'time': time.time(), 'free_mib': sample})
                if min(sample.values()) < 1536:
                    errors.append('Free memory below 1536 MiB')
            except Exception as exc:
                errors.append(str(exc))
            stop.wait(.5)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with (args.output/'requests.jsonl').open('x') as out:
            for repeat in range(-1, args.repeats):
                for case in cases:
                    if errors:
                        raise ValueError(errors[0])
                    before = read_metrics(args.base_url)
                    require_idle(before)
                    body = {'model':'glm-5.3-flash-fp4-local','prompt':case['prompt'],'temperature':0,'top_p':1,
                            'seed':workload['seed'],'max_tokens':args.output_tokens,'add_special_tokens':False,
                            'stream':True,'stream_options':{'include_usage':True},'cache_salt':str(uuid.uuid4())}
                    response = shared.stream(args.base_url, body)
                    after = completion_metrics(args.base_url, before)
                    timing = shared.metric_delta(before,after,case['prompt_tokens'],args.output_tokens)
                    usage=response['usage']
                    if usage['prompt_tokens']!=case['prompt_tokens'] or usage['completion_tokens']!=args.output_tokens or usage.get('prompt_tokens_details',{}).get('cached_tokens')!=0 or response['finish_reason']!='length':
                        raise ValueError('Token length, cold cache or finish contract failed')
                    row={'mode':args.mode,'case':case['name'],'repeat':repeat,'warmup':repeat<0,'timing':timing,
                         'response':response,'metrics_before':before,'metrics_after':after,'body':body,
                         'prompt_tokens':case['prompt_tokens'],'text_sha256':shared.digest(response['text'])}
                    out.write(json.dumps(row)+'\n'); out.flush()
                    if repeat>=0: rows.append(row)
                    print(f'{args.mode} {case["name"]} repeat={repeat} decode={timing["decode_tokens_per_second"]:.2f} TTFT={response["client_ttft_seconds"]:.3f}s total={response["client_seconds"]:.3f}s',flush=True)
        if errors:
            raise ValueError(errors[0])
    finally:
        stop.set(); thread.join(timeout=15)
        (args.output/'gpu-samples.json').write_text(json.dumps({'samples':samples,'errors':errors},indent=2)+'\n')
    summaries=[]
    for case in cases:
        selected=[r for r in rows if r['case']==case['name']]
        summaries.append({'mode':args.mode,'case':case['name'],'n':len(selected),'output_tokens':args.output_tokens,
            'decode_tokens_per_second':statistics.median(r['timing']['decode_tokens_per_second'] for r in selected),
            'prefill_tokens_per_second':statistics.median(r['prompt_tokens']/r['timing']['prefill_seconds'] for r in selected),
            'client_ttft_seconds':statistics.median(r['response']['client_ttft_seconds'] for r in selected),
            'end_to_end_seconds':statistics.median(r['response']['client_seconds'] for r in selected),
            'distinct_outputs':len({r['text_sha256'] for r in selected})})
    (args.output/'summary.json').write_text(json.dumps({'results':summaries,'minimum_free_mib':{str(i):min(s['free_mib'][str(i)] for s in samples) for i in devices}},indent=2)+'\n')


if __name__=='__main__':
    main()
