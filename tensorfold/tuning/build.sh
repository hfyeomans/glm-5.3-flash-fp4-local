#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
expected=sha256:c829536f5a214755ff50f4e72e622814a9226b8aec43ed646cdc6cd20c418e11
actual=$(docker image inspect --format '{{.Id}}' glm53-fp4-tensorfold:1.1.3-pr1050-sm120-context)
[[ "$actual" == "$expected" ]] || { printf 'Refusing an unpinned parent image.\n' >&2; exit 1; }
docker build --progress=plain -t glm53-fp4-tensorfold:sm120-tuning-v1 .
