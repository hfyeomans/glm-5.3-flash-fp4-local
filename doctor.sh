#!/usr/bin/env bash
# Read-only checks: Docker, image, GPUs, offline model and API port.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
validate_config
python3 "$REPO_ROOT/scripts/doctor.py"
