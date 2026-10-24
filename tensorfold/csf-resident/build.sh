#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
expected=sha256:2d9b848d07d66c6072ab7757017dd5dffbfb30453305022cbe2befc8904899ee
actual=$(docker image inspect --format '{{.Id}}' glm53-fp4-tensorfold:sm120-tuning-v1)
[[ "$actual" == "$expected" ]] || { printf 'Refusing an unpinned parent image.\n' >&2; exit 1; }
docker build --progress=plain -t glm53-fp4-tensorfold:csf-resident-v1 .
