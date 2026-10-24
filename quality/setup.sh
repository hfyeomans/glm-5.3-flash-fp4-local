#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export UV_PROJECT_ENVIRONMENT="$PWD/.state/quality-venv"
uv sync --project quality --python 3.12 --locked
export HF_HOME="${HF_CACHE_ROOT:-/models_engrams/huggingface}"
export NLTK_DATA="$HF_HOME/benchmarks/nltk"
.state/quality-venv/bin/python - <<'PY'
import os
import nltk
nltk.download('punkt_tab', download_dir=os.environ['NLTK_DATA'], raise_on_error=True)
PY
.state/quality-venv/bin/python quality/check_resources.py
