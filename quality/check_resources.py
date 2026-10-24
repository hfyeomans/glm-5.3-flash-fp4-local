#!/usr/bin/env python3
"""Verify the sentence-tokenization data used by the official IFEval grader."""
import hashlib
import json
import os
from pathlib import Path


def resource_digest(root):
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if not files:
        raise ValueError('NLTK punkt_tab data missing; run quality/setup.sh')
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(root)).encode() + b'\0')
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def main():
    cache = Path(os.environ.get('HF_CACHE_ROOT', '/models_engrams/huggingface'))
    root = cache / 'benchmarks/nltk/tokenizers/punkt_tab'
    lock = json.loads(Path(__file__).with_name('resources.json').read_text())
    if resource_digest(root) != lock['punkt_tab_tree_sha256']:
        raise ValueError('NLTK grader data differs from the benchmark resource lock')


if __name__ == '__main__':
    main()
