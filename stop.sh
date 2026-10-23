#!/usr/bin/env bash
# Stop only this recipe's container. Preserve container, weights and caches.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
validate_config
if [[ "${1:-}" == --dry-run || "${DRY_RUN:-0}" == 1 ]]; then plan docker stop --time 30 "$CONTAINER_NAME"; exit 0; fi
[[ $# == 0 ]] || fail 'Usage: ./stop.sh [--dry-run]'
lock_recipe
docker info >/dev/null 2>&1 || fail 'Docker daemon is unavailable.'
if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then log "$CONTAINER_NAME is absent."; exit 0; fi
assert_owned
docker stop --time 30 "$CONTAINER_NAME" >/dev/null
mkdir -p "$LOG_DIR"
docker logs --timestamps "$CONTAINER_NAME" > "$LOG_DIR/stopped-$(date -u +%Y%m%dT%H%M%SZ)-$$.log" 2>&1
log "Stopped $CONTAINER_NAME. Container, logs, weights and runtime volume are preserved."
