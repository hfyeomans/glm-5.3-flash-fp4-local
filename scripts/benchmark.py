#!/usr/bin/env python3
"""Bounded synthetic streaming benchmark; reports client-observed rates."""
import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path
from smoke import MODEL, payload, ready, stream


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--concurrency', type=int, choices=[1, 2, 4, 8], default=1)
    p.add_argument('--requests', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.requests < args.concurrency or args.max_tokens < 1:
        p.error('requests must cover concurrency; max-tokens must be positive')
    ready()
    prompt = 'Write a detailed explanation of how a hash table resolves collisions, with a Python example.'
    stream(payload('Reply with exactly the single word: ready', 128))  # excluded warmup
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(stream, payload(prompt+f' Use variable suffix {i}.', args.max_tokens))
                   for i in range(args.requests)]
        rows = [f.result() for f in futures]  # Any failed request fails the run.
    wall = time.monotonic()-start
    if any(not r['usage'] or not r['usage'].get('completion_tokens') for r in rows):
        raise ValueError('Missing server usage; refusing to estimate token counts')
    tokens = sum(r['usage']['completion_tokens'] for r in rows)
    report = {'model': MODEL, 'concurrency': args.concurrency, 'requests': args.requests,
              'max_tokens': args.max_tokens, 'wall_seconds': wall, 'completion_tokens': tokens,
              'aggregate_end_to_end_tokens_per_second': tokens/wall,
              'median_ttft_seconds': statistics.median(r['ttft_seconds'] for r in rows),
              'median_request_seconds': statistics.median(r['elapsed_seconds'] for r in rows),
              'note': 'Includes reasoning tokens, prompt processing and network time. TTFT is first visible content; this is not isolated decode throughput.',
              'samples': rows}
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite existing receipt: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}, indent=2))


if __name__ == '__main__':
    main()
