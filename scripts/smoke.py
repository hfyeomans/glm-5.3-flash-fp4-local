#!/usr/bin/env python3
"""Small OpenAI-compatible API checks; Python standard library only."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get('BASE_URL', 'http://127.0.0.1:8000').rstrip('/')
MODEL = os.environ.get('SERVED_MODEL_NAME', 'glm-5.3-flash-fp4-local')
TIMEOUT = float(os.environ.get('SMOKE_TIMEOUT', '180'))


def headers():
    result = {'Content-Type': 'application/json'}
    if os.environ.get('VLLM_API_KEY'):
        result['Authorization'] = 'Bearer ' + os.environ['VLLM_API_KEY']
    return result


def request(path, body=None, timeout=TIMEOUT):
    req = urllib.request.Request(BASE + path, headers=headers(),
                                 data=None if body is None else json.dumps(body).encode())
    return urllib.request.urlopen(req, timeout=timeout)


def call(path, body=None, timeout=TIMEOUT):
    with request(path, body, timeout) as response:
        return json.load(response)


def payload(prompt, max_tokens=512):
    return {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'top_p': 1, 'max_tokens': max_tokens,
            'chat_template_kwargs': {'reasoning_effort': 'low', 'clear_thinking': False}}


def ready(wait=0):
    deadline = time.monotonic() + wait
    heartbeat = time.monotonic()
    while True:
        try:
            with request('/health', timeout=5) as response:
                if response.status != 200: raise ValueError('Health endpoint is not ready')
            models = call('/v1/models', timeout=5)
            if MODEL not in [m['id'] for m in models.get('data', [])]:
                raise ValueError(f'API does not list expected model {MODEL}')
            return models
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f'API readiness failed: {exc}') from exc
            name = os.environ.get('CONTAINER_NAME')
            if name:
                state = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', name],
                                       text=True, capture_output=True, timeout=10)
                if state.returncode or state.stdout.strip() != 'true':
                    raise RuntimeError('Serving container exited before readiness; inspect ./logs.sh') from exc
            if time.monotonic() - heartbeat >= 30:
                print('Waiting for weights, kernel compilation and graph capture...', flush=True)
                heartbeat = time.monotonic()
            time.sleep(min(5, max(0, deadline-time.monotonic())))


def stream(body):
    body = dict(body, stream=True, stream_options={'include_usage': True})
    start = time.monotonic()
    first = None
    content = []
    usage = None
    done = False
    with request('/v1/chat/completions', body) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'): continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            event = json.loads(data)
            if 'error' in event: raise ValueError(str(event['error']))
            if event.get('usage'): usage = event['usage']
            for choice in event.get('choices', []):
                text = choice.get('delta', {}).get('content')
                if text:
                    if first is None: first = time.monotonic()-start
                    content.append(text)
    if not done or not content:
        raise ValueError('Stream must return nonempty content and finish with [DONE]')
    return {'content': ''.join(content), 'usage': usage, 'ttft_seconds': first,
            'elapsed_seconds': time.monotonic()-start}


def run_checks(suite=False, output=None):
    results = []
    def save(name, result):
        if output:
            output.mkdir(parents=True, exist_ok=True)
            path = output/(name+'.json')
            if path.exists():
                raise ValueError(f'Refusing to overwrite existing smoke receipt: {path}')
            path.write_text(json.dumps(result, indent=2)+'\n')
    def check(name, fn):
        start = time.monotonic()
        try:
            value = fn()
            save(name, value)
            result = {'check': name, 'passed': True, 'seconds': time.monotonic()-start}
            print('PASS ' + name)
        except (KeyError, IndexError, TypeError, ValueError, OSError, RuntimeError) as exc:
            result = {'check': name, 'passed': False, 'error': str(exc), 'seconds': time.monotonic()-start}
            print('FAIL ' + name + ': ' + str(exc), file=sys.stderr)
        results.append(result)
    def completion():
        r = call('/v1/chat/completions', payload('Reply with exactly the single word: ready'))
        save('completion-response', r)
        text = r['choices'][0]['message']['content']
        if not isinstance(text, str) or text.strip().lower() != 'ready':
            raise ValueError('Completion did not obey the ready instruction; inspect completion-response.json')
        return r
    check('completion', completion)
    if suite:
        def json_check():
            b = payload('Return a JSON object with exactly one key, answer, whose integer value is 17 + 25.', 512)
            b['response_format'] = {'type': 'json_object'}
            r = call('/v1/chat/completions', b)
            save('json-response', r)
            if json.loads(r['choices'][0]['message']['content']) != {'answer': 42}:
                raise ValueError('Expected exactly {"answer": 42}')
            return r
        def tool_check():
            b = payload('Call get_temperature for Paris with unit celsius.', 512)
            b['tools'] = [{'type': 'function', 'function': {'name': 'get_temperature',
                'description': 'Read the temperature in a city.', 'parameters': {'type': 'object',
                'properties': {'city': {'type': 'string'}, 'unit': {'type': 'string', 'enum': ['celsius']}},
                'required': ['city', 'unit'], 'additionalProperties': False}}}]
            b['tool_choice'] = 'required'
            r = call('/v1/chat/completions', b)
            save('tools-response', r)
            tools = r['choices'][0]['message']['tool_calls']
            if len(tools) != 1 or tools[0]['function']['name'] != 'get_temperature':
                raise ValueError('Expected one get_temperature tool call')
            args = json.loads(tools[0]['function']['arguments'])
            if args.get('city', '').lower() != 'paris' or args.get('unit') != 'celsius':
                raise ValueError('Tool arguments failed validation')
            return r
        check('json', json_check)
        check('tools', tool_check)
        check('streaming', lambda: stream(payload('Write one sentence about the Moon.')))
    save('checks', results)
    return all(r['passed'] for r in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--suite', action='store_true')
    mode.add_argument('--ready', action='store_true')
    parser.add_argument('--wait', type=float, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.wait < 0: parser.error('--wait must be nonnegative')
    try:
        ready(args.wait)
        print(f'PASS readiness and model identity: {MODEL}')
        return 0 if args.ready or run_checks(args.suite, args.output) else 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print('FAIL ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
