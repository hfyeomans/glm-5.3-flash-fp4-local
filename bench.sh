#!/usr/bin/env bash
# Usage: ./bench.sh --concurrency 1 --requests 8 --output logs/bench-c1.json
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
validate_config
exec python3 "$REPO_ROOT/scripts/benchmark.py" "$@"
