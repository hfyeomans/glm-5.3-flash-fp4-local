#!/usr/bin/env python3
"""Cold multi-position retrieval with server-counted tokens; no engine lifecycle changes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def call(base, path, body, timeout):
    request = urllib.request.Request(base.rstrip('/') + path, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def payload(rows, model, salt):
    secrets = [f'cedar-{salt}-41827', f'orchid-{salt}-63951', f'willow-{salt}-82746']
    positions = [rows // 10, rows // 2, rows * 9 // 10]
    markers = dict(zip(positions, secrets))
    lines = [f'Record {i}: the archive entry is blue.' if i not in markers else
             f'Record {i}: ARCHIVE SECRET {secrets.index(markers[i]) + 1} is {markers[i]}.'
             for i in range(rows)]
    text = ('Read the archive and find its three ARCHIVE SECRET values.\n'
            + '\n'.join(lines)
            + '\nReturn the three ARCHIVE SECRET values in numbered order, one per line. No other text.')
    return {'model': model, 'messages': [{'role': 'user', 'content': text}],
            'temperature': 0, 'top_p': 1, 'max_tokens': 128,
            'chat_template_kwargs': {'enable_thinking': False}}, secrets, positions


def fit(base, target, model, timeout):
    # Fit the actual rendered chat, with enough room for the requested output.
    lo, hi = 10, target // 8 + 128
    best = None
    while lo <= hi:
        rows = (lo + hi) // 2
        body, secrets, positions = payload(rows, model, str(target))
        count = call(base, '/tokenize', body, timeout)['count']
        if count <= target:
            best = (body, secrets, positions, count, rows)
            lo = rows + 1
        else:
            hi = rows - 1
    if best is None or best[3] < target - 32:
        raise ValueError('Could not construct a prompt sufficiently close to target')
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8001')
    parser.add_argument('--model', default='glm-5.3-flash-fp4-local')
    parser.add_argument('--targets', nargs='+', type=int, required=True,
                        help='Prompt token targets; leave at least 128 tokens below served context')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--compare-plain', action='store_true')
    args = parser.parse_args()
    if any(t < 1024 or t > 1048448 for t in args.targets):
        parser.error('Targets must be between 1024 and 1048448')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'base_url': args.base_url, 'model': args.model, 'checks': [],
              'scope': 'Synthetic retrieval, not a general long-context accuracy qualification'}
    ok = True
    for target in args.targets:
        body, expected, positions, count, rows = fit(args.base_url, target, args.model, args.timeout)
        raw = json.dumps(body, sort_keys=True).encode()
        with gzip.open(args.output / f'prompt-{target}.json.gz', 'wb') as stream:
            stream.write(raw)
        result = {'target': target, 'tokenized_count': count, 'rows': rows,
                  'needle_rows': positions, 'expected': expected,
                  'request_sha256': hashlib.sha256(raw).hexdigest(), 'runs': []}
        print(f'Testing {count} prompt tokens; markers near 10%, 50%, 90%', flush=True)
        for draft in ([True, False] if args.compare_plain else [True]):
            start = time.monotonic()
            response = call(args.base_url, '/v1/chat/completions', dict(body, draft=draft), args.timeout)
            seconds = time.monotonic() - start
            content = response['choices'][0]['message'].get('content', '')
            usage = response['usage']
            retrieved = all(secret in content for secret in expected)
            same_count = usage['prompt_tokens'] == count
            cold = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0
            passed = retrieved and same_count and cold and response['choices'][0]['finish_reason'] == 'stop'
            result['runs'].append({'draft': draft, 'seconds': seconds, 'retrieval_passed': retrieved,
                                   'count_matches': same_count, 'cold': cold, 'passed': passed,
                                   'response': response})
            ok = ok and passed
            print(f'  draft={draft}: pass={passed}, {seconds:.2f}s, {content!r}', flush=True)
        if args.compare_plain:
            hashes = [r['response'].get('tensorfold', {}).get('token_sha256') for r in result['runs']]
            result['draft_equals_plain'] = bool(hashes[0]) and hashes[0] == hashes[1]
            ok = ok and result['draft_equals_plain']
        report['checks'].append(result)
        report['passed'] = ok
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if not ok:
            break
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
