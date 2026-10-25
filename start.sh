#!/usr/bin/env bash
# Usage: ./start.sh [restart|--dry-run]. Starts only this recipe's container.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
MODE=start
case "${1:-}" in
  '') ;;
  restart) MODE=restart ;;
  --dry-run) DRY_RUN=1 ;;
  -h|--help) log 'Usage: ./start.sh [restart|--dry-run]'; exit 0 ;;
  *) fail 'Usage: ./start.sh [restart|--dry-run]' ;;
esac
(( $# <= 1 )) || fail 'Too many arguments.'
validate_config
# Lifecycle acceptance must check the endpoint this launch is configured to bind.
export BASE_URL=$RECIPE_BASE_URL
log "GLM-5.3-Flash FP4 | 2 GPUs ($GPUS) | drafter=$DRAFTER | port $PORT"
RUN_IMAGE=$IMAGE
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  resolve_model --plan
  resolve_drafter --plan
  build_serve_args
  build_docker_args
  log 'Plan only; no Docker calls, file changes, downloads or GPU work.'
  log "Runtime ENV: $REPO_ROOT/config/b12x.env"
  plan "${DOCKER_ARGS[@]}"
  log "Would wait up to $WAIT_TIMEOUT seconds for /health, check /v1/models, then send a smoke completion."
  exit 0
fi
# Refuse a missing/wrong external draft before stopping an existing service.
resolve_drafter
lock_recipe
docker info >/dev/null 2>&1 || fail 'Docker daemon is unavailable.'
if container_exists; then
  assert_owned
  if [[ "$MODE" == start && "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == true ]]; then
    RUNNING_COMMAND=$(docker inspect -f '{{json .Config.Cmd}}' "$CONTAINER_NAME")
    python3 - "$RUNNING_COMMAND" "$PORT" <<'PYPORT'
import json, sys
try:
    command = json.loads(sys.argv[1])
    if not isinstance(command, list) or command[:2] != ['vllm', 'serve']:
        raise ValueError('Unexpected serving command')
    port = int(command[command.index('--port') + 1])
except (ValueError, IndexError, TypeError):
    sys.exit('Cannot determine the running container serving port. Inspect its launch before accepting readiness.')
if port != int(sys.argv[2]):
    sys.exit(f'Running container serves port {port}, but configured PORT={sys.argv[2]}. Run ./start.sh restart to apply settings.')
PYPORT
    log "$CONTAINER_NAME already exists and is running. Use ./start.sh restart to apply settings."
    python3 "$REPO_ROOT/scripts/smoke.py" --ready
    exit 0
  fi
  if [[ "$MODE" == restart ]]; then
    log "Stopping recipe container $CONTAINER_NAME before launch checks."
    docker stop --time 30 "$CONTAINER_NAME" >/dev/null
  fi
  mkdir -p "$LOG_DIR"
  docker logs --timestamps "$CONTAINER_NAME" > "$LOG_DIR/previous-$(date -u +%Y%m%dT%H%M%SZ)-$$.log" 2>&1
  docker rm "$CONTAINER_NAME" >/dev/null
fi
python3 "$REPO_ROOT/scripts/doctor.py" || fail 'Preflight failed. Correct the listed blockers, then rerun.'
resolve_model
# Resolve the prepared reference to its exact local image ID; refuse silent tag drift.
RUN_IMAGE=$(docker image inspect "$IMAGE" --format '{{.Id}}')
if [[ -f "$STATE_DIR/image.json" ]]; then
  python3 - "$STATE_DIR/image.json" "$IMAGE" "$RUN_IMAGE" <<'PYPIN'
import json,sys
p=json.load(open(sys.argv[1]))
if p['requested_image']!=sys.argv[2] or p['image_id']!=sys.argv[3]:
    sys.exit('Image reference changed since preparation. Review it, then run ./prepare.sh to refresh the lock.')
PYPIN
fi
build_serve_args
build_docker_args
RUN_DIR=$LOG_DIR/run-$(date -u +%Y%m%dT%H%M%SZ)-$$
mkdir -p "$RUN_DIR"
ln -sfn "${RUN_DIR##*/}" "$LOG_DIR/latest"
printf '%s\n' "$MODEL_METADATA" > "$RUN_DIR/model.json"
if [[ "$DRAFTER" == dflash2 ]]; then
  printf '%s\n' "$DRAFT_METADATA" > "$RUN_DIR/drafter.json"
fi
printf '{"drafter":"%s"}\n' "$DRAFTER" > "$RUN_DIR/speculation.json"
plan "${DOCKER_ARGS[@]}" > "$RUN_DIR/launch-command.txt"
cp "$REPO_ROOT/config/b12x.env" "$RUN_DIR/runtime.env"
docker image inspect "$RUN_IMAGE" --format '{"id":{{json .Id}},"repo_digests":{{json .RepoDigests}}}' > "$RUN_DIR/image.json"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,driver_version --format=csv > "$RUN_DIR/gpus.csv"
log "Weights: $MODEL_HOST_PATH"
log "Evidence: $RUN_DIR"
"${DOCKER_ARGS[@]}" > "$RUN_DIR/container-id.txt"
if ! python3 "$REPO_ROOT/scripts/smoke.py" --wait "$WAIT_TIMEOUT" --output "$RUN_DIR"; then
  docker logs --timestamps "$CONTAINER_NAME" > "$RUN_DIR/server.log" 2>&1 || true
  fail "Startup or smoke failed; inspect $RUN_DIR/server.log. Container retained for diagnosis."
fi
log "Ready: $BASE_URL/v1 | model: $SERVED_MODEL_NAME"
log 'Use ./logs.sh to follow server logs; ./smoke.sh --suite for API checks.'
