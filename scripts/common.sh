#!/usr/bin/env bash
# shellcheck disable=SC2034
# Shared configuration. .env is literal data, never sourced or evaluated.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
MODEL_ID=nvidia/GLM-5.3-Flash-FP4-Local
RECIPE_LABEL=glm53-fp4-local-v1
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '%s\n' "$*"; }
plan() { printf '  '; printf '%q ' "$@"; printf '\n'; }
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line=${line%$'\r'}
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == *=* ]] || fail '.env requires literal KEY=value lines.'
    key=${line%%=*}; value=${line#*=}
    case "$key" in
      HF_CACHE_ROOT|MODEL_DIR|MODEL_REVISION|LOAD_FORMAT|MTP_ENABLED|DRAFTER|DFLASH2_MODEL_DIR|DFLASH2_NUM_SPECULATIVE_TOKENS|DFLASH2_KV_CACHE_DTYPE|IMAGE|CONTAINER_NAME|GPUS|HOST|PORT|SERVED_MODEL_NAME|RUNTIME_VOLUME|MAX_MODEL_LEN|MAX_NUM_SEQS|MAX_NUM_BATCHED_TOKENS|GPU_MEMORY_UTILIZATION|KV_CACHE_MEMORY_BYTES|MIN_GPU_FREE_MIB|WAIT_TIMEOUT|SMOKE_TIMEOUT|RESTART_POLICY) ;;
      *) fail "Unsupported .env key: $key (credentials belong in exported environment variables)." ;;
    esac
    if [[ ! -v "$key" ]]; then export "$key=$value"; fi
  done < "$REPO_ROOT/.env"
fi
export HF_CACHE_ROOT=${HF_CACHE_ROOT:-/models_engrams/huggingface}
# Keep this recipe on its configured drive even when the login shell uses another cache.
export HF_HOME=$HF_CACHE_ROOT
export HF_HUB_CACHE=$HF_CACHE_ROOT/hub
export MODEL_DIR=${MODEL_DIR:-}
export LOAD_FORMAT=${LOAD_FORMAT:-safetensors}
export MTP_ENABLED=${MTP_ENABLED:-1}
# Explicit DRAFTER wins; old MTP_ENABLED=0/1 remains supported when it is unset.
if [[ -z "${DRAFTER:-}" ]]; then
  DRAFTER=mtp
  [[ "$MTP_ENABLED" != 0 ]] || DRAFTER=none
fi
export DRAFTER
export DFLASH2_MODEL_DIR=${DFLASH2_MODEL_DIR:-}
export DFLASH2_NUM_SPECULATIVE_TOKENS=${DFLASH2_NUM_SPECULATIVE_TOKENS:-7}
export DFLASH2_KV_CACHE_DTYPE=${DFLASH2_KV_CACHE_DTYPE:-auto}
export MODEL_REVISION=${MODEL_REVISION:-76348ddb2be2e35e39ecfc00e75dc07703f4f3d5}
export IMAGE=${IMAGE:-ghcr.io/local-inference-lab/vllm:release-10152026@sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29}
export CONTAINER_NAME=${CONTAINER_NAME:-glm53-fp4-tp2}
export GPUS=${GPUS:-0,1}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm-5.3-flash-fp4-local}
export RUNTIME_VOLUME=${RUNTIME_VOLUME:-glm53-fp4-runtime-10152026}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.985}
export KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-7650410496}
export MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-96000}
export WAIT_TIMEOUT=${WAIT_TIMEOUT:-3600}
export SMOKE_TIMEOUT=${SMOKE_TIMEOUT:-180}
export RESTART_POLICY=${RESTART_POLICY:-no}
export MODEL_ID RECIPE_LABEL
export PYTHONDONTWRITEBYTECODE=1
STATE_DIR=$REPO_ROOT/.state
LOG_DIR=$REPO_ROOT/logs
API_HOST=$HOST
[[ "$API_HOST" != 0.0.0.0 && "$API_HOST" != :: ]] || API_HOST=127.0.0.1
[[ "$API_HOST" != *:* ]] || API_HOST=[$API_HOST]
RECIPE_BASE_URL=http://$API_HOST:$PORT
export BASE_URL=${BASE_URL:-$RECIPE_BASE_URL}
validate_config() {
  python3 - <<'PYCFG'
import os, re
from pathlib import Path

def require(ok, message):
    if not ok: raise SystemExit('ERROR: '+message)
e=os.environ
for key in ['PORT','MAX_MODEL_LEN','MAX_NUM_SEQS','MAX_NUM_BATCHED_TOKENS','KV_CACHE_MEMORY_BYTES','MIN_GPU_FREE_MIB','WAIT_TIMEOUT','SMOKE_TIMEOUT']:
    require(e[key].isdigit() and int(e[key])>0, key+' must be a positive integer')
require(int(e['PORT'])<65536, 'PORT must be below 65536')
require(0<float(e['GPU_MEMORY_UTILIZATION'])<=1, 'GPU_MEMORY_UTILIZATION must be in (0,1]')
require(re.fullmatch(r'\d+,\d+',e['GPUS']) and len(set(e['GPUS'].split(',')))==2, 'GPUS must contain exactly two distinct numeric IDs, e.g. 0,1')
for key in ['CONTAINER_NAME','RUNTIME_VOLUME']:
    require(re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*',e[key]), key+' must be a Docker name')
require(re.fullmatch(r'[0-9a-f]{40}',e['MODEL_REVISION']), 'MODEL_REVISION must be a full commit SHA')
require(e['MTP_ENABLED'] in ['0','1'], 'MTP_ENABLED must be 0 or 1')
require(e['DRAFTER'] in ['mtp','dflash2','none'], 'DRAFTER must be mtp, dflash2 or none')
require(e['DFLASH2_KV_CACHE_DTYPE'] in ['auto','fp8'], 'DFLASH2_KV_CACHE_DTYPE must be auto or fp8')
require(e['DFLASH2_NUM_SPECULATIVE_TOKENS'].isdigit() and 1 <= int(e['DFLASH2_NUM_SPECULATIVE_TOKENS']) <= 7, 'DFLASH2_NUM_SPECULATIVE_TOKENS must be 1..7')
require(e['LOAD_FORMAT'] in ['instanttensor','safetensors'], 'LOAD_FORMAT must be instanttensor or safetensors')
require(e['RESTART_POLICY'] in ['no','unless-stopped','on-failure','on-failure:3'], 'Unsupported RESTART_POLICY')
for key in ['HF_HOME','HF_HUB_CACHE','MODEL_DIR','DFLASH2_MODEL_DIR']:
    require(not e[key] or (Path(e[key]).is_absolute() and not any(x in e[key] for x in [',','\n','\r'])),key+' must be an absolute path without commas or newlines')
require(bool(e['IMAGE']) and bool(e['HOST']) and bool(e['SERVED_MODEL_NAME']), 'IMAGE, HOST and SERVED_MODEL_NAME cannot be empty')
PYCFG
}
lock_recipe() {
  mkdir -p "$STATE_DIR"
  exec 9>"$STATE_DIR/lifecycle.lock"
  flock -n 9 || fail 'Another recipe lifecycle operation is running.'
}
container_exists() {
  local error
  if error=$(docker container inspect "$CONTAINER_NAME" 2>&1 >/dev/null); then
    return 0
  fi
  case "$error" in
    "Error response from daemon: No such container: $CONTAINER_NAME"|"Error: No such container: $CONTAINER_NAME"|\
    "Error response from daemon: No such object: $CONTAINER_NAME"|"Error: No such object: $CONTAINER_NAME") return 1 ;;
    *) fail "Cannot inspect container $CONTAINER_NAME: $error" ;;
  esac
}

assert_owned() {
  local owner
  owner=$(docker inspect -f '{{index .Config.Labels "local.glm53.recipe"}}' "$CONTAINER_NAME")
  [[ "$owner" == "$RECIPE_LABEL" ]] || fail "Container $CONTAINER_NAME is not owned by this recipe. Choose another CONTAINER_NAME."
}
resolve_model() {
  MODEL_METADATA=$(python3 "$REPO_ROOT/scripts/model.py" "$@") || return
  MODEL_HOST_PATH=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])' <<< "$MODEL_METADATA")
  MODEL_CONTAINER_PATH=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["container_path"])' <<< "$MODEL_METADATA")
  MODEL_MOUNT_SOURCE=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["mount_source"])' <<< "$MODEL_METADATA")
  MODEL_MOUNT_TARGET=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["mount_target"])' <<< "$MODEL_METADATA")
}
resolve_drafter() {
  [[ "$DRAFTER" == dflash2 ]] || return 0
  DRAFT_METADATA=$(python3 "$REPO_ROOT/scripts/draft_model.py" "$@") || return
  DRAFT_CONTAINER_PATH=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["container_path"])' <<< "$DRAFT_METADATA")
  DRAFT_MOUNT_SOURCE=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["mount_source"])' <<< "$DRAFT_METADATA")
  DRAFT_MOUNT_TARGET=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["mount_target"])' <<< "$DRAFT_METADATA")
}
build_serve_args() {
  SERVE_ARGS=(vllm serve "$MODEL_CONTAINER_PATH"
    --host "$HOST" --port "$PORT" --served-model-name "$SERVED_MODEL_NAME"
    --additional-config '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"b12x"}'
    --attention-backend B12X --block-size 256
    --chat-template /opt/lil/runtime/templates/glm53-flash.jinja
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
    --cp-kv-cache-interleave-size 4
    --cudagraph-capture-sizes 1 2 4 8 12 16 20 24 28 32
    --dcp-kv-cache-interleave-size 4 --decode-context-parallel-size 2
    --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":false}'
    --dtype bfloat16 --enable-auto-tool-choice --enable-chunked-prefill
    --no-enable-flashinfer-autotune --enable-prefix-caching --enable-prompt-tokens-details
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --kv-cache-dtype fp8 --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"
    --linear-backend b12x --load-format "$LOAD_FORMAT" --mamba-cache-mode align
    --max-cudagraph-capture-size 32 --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --max-num-seqs "$MAX_NUM_SEQS"
    --max-parallel-prefills 1 --moe-backend b12x
    --override-generation-config '{"temperature":1.0,"top_p":0.95}'
    --prefill-compute-share 0.4 --prefill-schedule-interval 1
    --quantization modelopt_mixed --reasoning-parser glm45
    --recurrent-checkpoint-policy request_boundaries)
  if [[ "$DRAFTER" == mtp ]]; then
    SERVE_ARGS+=(--speculative-config '{"method":"mtp","draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"b12x","attention_backend":"B12X","num_speculative_tokens":3}')
  elif [[ "$DRAFTER" == dflash2 ]]; then
    local spec
    spec=$(python3 - "$DRAFT_CONTAINER_PATH" "$DFLASH2_NUM_SPECULATIVE_TOKENS" "$DFLASH2_KV_CACHE_DTYPE" <<'PYSPEC'
import json, sys
print(json.dumps({'method':'dflash', 'model':sys.argv[1], 'num_speculative_tokens':int(sys.argv[2]),
                  'draft_tensor_parallel_size':2, 'quantization':None, 'kv_cache_dtype':sys.argv[3],
                  'attention_backend':'TRITON_ATTN', 'draft_sample_method':'probabilistic',
                  'rejection_sample_method':'standard'}, separators=(',',':')))
PYSPEC
)
    SERVE_ARGS+=(--speculative-config "$spec")
  fi
  SERVE_ARGS+=(--tensor-parallel-size 2 --tool-call-parser glm47)
}
build_docker_args() {
  DOCKER_ARGS=(docker run -d --name "$CONTAINER_NAME" --init --gpus "\"device=$GPUS\""
    --restart "$RESTART_POLICY" --network host --ipc host --shm-size 32g --stop-timeout 30
    --log-driver json-file --log-opt max-size=100m --log-opt max-file=3
    --ulimit memlock=-1 --ulimit stack=67108864:67108864
    --label "local.glm53.recipe=$RECIPE_LABEL"
    --mount "type=bind,src=$MODEL_MOUNT_SOURCE,dst=$MODEL_MOUNT_TARGET,readonly"
    --mount "type=volume,src=$RUNTIME_VOLUME,dst=/cache"
    --env-file "$REPO_ROOT/config/b12x.env")
  # Hub paths may have different host aliases but share one container mount.
  if [[ "$DRAFTER" == dflash2 && "$DRAFT_MOUNT_TARGET" != "$MODEL_MOUNT_TARGET" ]]; then
    DOCKER_ARGS+=(--mount "type=bind,src=$DRAFT_MOUNT_SOURCE,dst=$DRAFT_MOUNT_TARGET,readonly")
  fi
  [[ -z "${VLLM_API_KEY:-}" ]] || DOCKER_ARGS+=(-e VLLM_API_KEY)
  DOCKER_ARGS+=("${RUN_IMAGE:-$IMAGE}" "${SERVE_ARGS[@]}")
}
