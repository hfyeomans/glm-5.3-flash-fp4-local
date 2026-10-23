#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
validate_config
assert_owned
docker inspect --format 'Container: {{.Name}} | state: {{.State.Status}} | restarts: {{.RestartCount}} | image: {{.Image}}' "$CONTAINER_NAME"
python3 "$REPO_ROOT/scripts/smoke.py" --ready
