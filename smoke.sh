#!/usr/bin/env bash
# Usage: ./smoke.sh [--suite|--ready] [--output PATH]
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
validate_config
exec python3 "$REPO_ROOT/scripts/smoke.py" "$@"
