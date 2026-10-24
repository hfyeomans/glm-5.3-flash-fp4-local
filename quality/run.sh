#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONHASHSEED=0
.state/quality-venv/bin/python quality/check_resources.py
exec .state/quality-venv/bin/python quality/run.py "$@"
