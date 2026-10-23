#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
validate_config
assert_owned
exec docker logs --timestamps --tail 100 --follow "$CONTAINER_NAME"
