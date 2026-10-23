#!/usr/bin/env bash
# CPU-only checks; no real Docker calls or model inference.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1
for script in ./*.sh scripts/*.sh; do bash -n "$script"; done
shellcheck -S warning ./*.sh scripts/*.sh
python3 - <<'PYSYNTAX'
import ast
from pathlib import Path
for root in ['scripts', 'tests']:
    for path in Path(root).glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
print('PASS shell lint and Python syntax')
PYSYNTAX
python3 -m unittest discover -s tests -v
