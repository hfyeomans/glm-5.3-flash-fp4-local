#!/usr/bin/env python3
"""Use the baseline's acceptance workloads with native reasoning parameter spelling."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
smoke = importlib.import_module('smoke')
base_payload = smoke.payload


def payload(*args, **kwargs):
    body = base_payload(*args, **kwargs)
    body['reasoning_effort'] = 'low'
    return body


smoke.payload = payload
command = sys.argv.pop(1)
if command == 'smoke':
    sys.exit(smoke.main())
benchmark = importlib.import_module('benchmark')
sys.exit(benchmark.main())
