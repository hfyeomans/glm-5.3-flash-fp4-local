#!/usr/bin/env bash
# One native process per device; rank zero owns HTTP.
set -euo pipefail
: "${MODEL_PATH:?}" "${TF_PORT:?}" "${TF_COMM_PORT:?}" "${TF_CONTEXT:?}" "${TF_PARALLEL:?}"
# Check mixed recipes and a real compressed scale before creating CUDA contexts.
/opt/tensorfold/bin/glm-loader-probe "$MODEL_PATH"
export TF_GLM_WORLD=2 NCCL_COMM_ID="127.0.0.1:${TF_COMM_PORT}"
export TF_ROCE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo CUDA_DEVICE_ORDER=PCI_BUS_ID
export TF_GLM_PROF="${TF_PROFILE:-0}" NCCL_DEBUG="${TF_NCCL_DEBUG:-WARN}"
if [[ -n "${TF_MEMORY_RESERVE_GIB:-}" ]]; then
    export TENSORFOLD_MEMORY_RESERVE_GIB="$TF_MEMORY_RESERVE_GIB"
fi
args=(serve "$MODEL_PATH" --backend cuda --host "${TF_HOST:-0.0.0.0}" --port "$TF_PORT"
      --name "${TF_MODEL_NAME:-glm-5.3-flash-fp4-local}" --context "$TF_CONTEXT"
      --parallel "$TF_PARALLEL" --kv-cache fp8 --prompt-cache-gib "${TF_PROMPT_CACHE_GIB:-1}"
      --thinking --reasoning-effort xhigh --temperature 1 --top-p 0.95 --no-update-check)
pids=()
cleanup() {
    trap - TERM INT
    if ((${#pids[@]})); then
        kill -TERM "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
for rank in 0 1; do
    TF_GLM_RANK="$rank" TF_CUDA_DEVICE="$rank" /opt/tensorfold/bin/tensorfold-native "${args[@]}" &
    pids+=("$!")
done
set +e
wait -n "${pids[@]}"
rc=$?
set -e
# An unsolicited rank exit, even code zero, terminates the complete service.
if ((rc == 0)); then rc=1; fi
exit "$rc"
