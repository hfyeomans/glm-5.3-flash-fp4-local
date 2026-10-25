# Detailed vLLM operations and provenance

Detailed reference for the pinned NVIDIA checkpoint, vLLM launch and operational
checks on two RTX PRO 6000 Blackwell GPUs. Use the root [quick start](../README.md)
for normal startup. [Validation](VALIDATION.md) records loading and acceptance
results; [performance](PERFORMANCE-CURRENT.md) and [quality](QUALITY-BASELINE.md)
use separate protocols. Native experiments are in [TensorFold](../tensorfold/README.md)
and its [tuning recipe](../tensorfold/tuning/README.md).

## Fixed configuration

| Ingredient | Fixed starting point |
| --- | --- |
| Checkpoint | [`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) |
| Model revision | `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` |
| Weight files | 42 safetensors shards, 177,463,841,024 bytes including headers (165.28 GiB) |
| Container | `ghcr.io/local-inference-lab/vllm:release-10152026` |
| Parallelism | TP2 / DCP2, selected host GPUs `0,1` |
| Weights / attention | ModelOpt mixed precision with compressed FP4 scales; B12X attention, MoE and linear kernels |
| Loader | CPU-staged `safetensors`; the original and bounded InstantTensor runs failed during MTP loading on this host |
| Speculation | Checkpoint's MTP head, three draft tokens; no separate drafter download |
| Context ceiling | 1,048,576 tokens per request, subject to shared cache capacity |
| Scheduler | 8 active sequences; 4,096 batched tokens; one parallel prefill |
| KV budget | FP8, **7,650,410,496 bytes per GPU** (7.125 GiB) |
| GPU memory fraction | `0.985` |
| Request defaults | Temperature `1.0`, top-p `0.95`, reasoning effort `max`, `clear_thinking=false` |
| Endpoint | `http://127.0.0.1:8000/v1` |
| API model name | `glm-5.3-flash-fp4-local` |

Eight slots do not mean eight simultaneous million-token prompts fit. Read the capacity reported by vLLM at startup. The context ceiling and total KV capacity are different limits.

NVIDIA's card lists `release-10142026` in its heading, but its actual launch commands use **`release-10152026`**. This recipe follows the commands and pins the verified Linux/amd64 registry digest:

```text
sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
```

The default image reference includes both the release tag and this digest. The compressed scale format needs the specialized image; replacing it with an arbitrary upstream vLLM image is not a supported substitution. [Sources and deliberate choices](../docs/PROVENANCE.md).

## Requirements and model cache

Use Linux x86_64 with working NVIDIA drivers, Docker, NVIDIA Container Toolkit, Bash, Python 3.10+, `flock` and ordinary GNU utilities. The service smoke/performance clients use only the standard library; the optional quality benchmark installs its own isolated client environment. ShellCheck is needed for the repository checks.

Keep both GPUs almost empty. The preflight requires the greater of `MIN_GPU_FREE_MIB=96000` and the configured vLLM memory fraction on each GPU. On this host's 97,887 MiB cards, the default fraction requires **96,419 MiB free per card**. It reports other containers and never stops them.

The checkpoint already lives here:

```text
/models_engrams/huggingface/
└── hub/
    └── models--nvidia--GLM-5.3-Flash-FP4-Local/
        ├── blobs/
        ├── refs/main
        └── snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5/
            ├── config.json
            ├── model.safetensors.index.json
            ├── tokenizer.json
            └── model-*.safetensors -> ../../blobs/...
```

This is the standard Hub layout for the namespace `nvidia/`; a second literal `nvidia/` directory is unnecessary. The complete cache entry was moved from `/data/huggingface/hub/` into this shared cache. All **42 weight-shard SHA-256 checksums passed after the move**.

The recipe's **`HF_CACHE_ROOT`** defaults to `/models_engrams/huggingface`. It derives host `HF_HOME` and `HF_HUB_CACHE` from that setting so an older login-shell cache setting cannot send it back to `/data`. The container sees the Hub cache at `/root/.cache/huggingface/hub`, mounted read-only. Relative blob links stay intact. Kernels and compilation caches go into the separate Docker volume `glm53-fp4-runtime-10152026`.

Offline serving needs no `HF_TOKEN`. The launcher runs with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; it does not forward a token or download replacement weights. Reserve disk space for the image and JIT caches on Docker's filesystem; use `docker system df` and `df -h` before pulling. Allow roughly 60 GiB initially; measure actual image/cache use.

## Start the service

From this repository:

```bash
# Optional: customize the literal KEY=value settings.
test -f .env || cp .env.example .env

# Inspect the full launch without Docker calls, writes or GPU work.
./start.sh --dry-run

# Pull the supported image; record its exact image ID and registry digest.
# This uses no GPUs and does not download model weights.
./prepare.sh

# Read-only checks. Resolve every FAIL before starting.
./doctor.sh

# Load weights, compile kernels, capture graphs, wait, then send a completion.
./start.sh

# The small acceptance suite: identity, text, JSON, tools and streaming.
./smoke.sh --suite --output logs/acceptance
```

If another model service occupies the selected GPUs, arrange its shutdown before the GLM launch. `./doctor.sh` reports the memory blocker. The pinned image is prepared locally; repeat the preflight before startup because GPU and port availability can change.

First startup can spend minutes loading and compiling; the default readiness deadline is one hour. The script prints a heartbeat while waiting. Follow progress from a second terminal with `./logs.sh`. A readiness failure retains the container and saves its logs for inspection. The first-run restart policy is `no`, which keeps a failed run inspectable. After acceptance, set `RESTART_POLICY=on-failure:3` in `.env` and run `./start.sh restart` if bounded automatic retries are desired.

The explicit `vllm serve` command is passed through the image's normal entrypoint, retaining its CUDA/NCCL bootstrap. **`PRESET=glm53-tp2` is deliberately omitted:** that preset selects a different Local Inference Lab checkpoint and may start an additional LMCache RAM tier. This recipe selects the NVIDIA snapshot directly and uses GPU prefix caching.

## API acceptance

`./start.sh` checks `/health`, model identity and a short greedy `ready` completion.

`./smoke.sh --suite` additionally requires:

- A JSON response with `{"answer":42}` for a small arithmetic prompt.
- One parsed `get_temperature` tool call with the requested city and unit. It never executes a tool.
- A nonempty streaming response terminating with `[DONE]`.

Every failed check counts; the suite does not retry model-output failures. Read [the test service plan](../docs/TESTING.md) before scaling the workload.

Try the API:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"glm-5.3-flash-fp4-local",
    "messages":[{"role":"user","content":"Write a short Python fibonacci function."}],
    "temperature":0,
    "max_tokens":512,
    "chat_template_kwargs":{"reasoning_effort":"low","clear_thinking":false}
  }'
```

Short checks use `reasoning_effort=low` and 512 output tokens; normal requests default to `max`. Avoid `enable_thinking=false`: the template still opens reasoning, causing parser mismatch. The default bind is `0.0.0.0`, matching the model card, and the API has no key by default. Set `HOST=127.0.0.1` for local-only clients. Export `VLLM_API_KEY` before startup to enable API authentication; the supplied Python clients read it without printing it.

## Lifecycle

```bash
./status.sh                 # container state plus API/model readiness
./logs.sh                   # timestamped server logs, follow mode
./smoke.sh                  # one short completion
./start.sh restart          # recreate only this recipe's container
./stop.sh                   # stop; preserve container, logs, model and runtime cache
```

Lifecycle operations use a lock and verify an ownership label before stopping or removing a container. `restart` stops this recipe's existing server before checking free GPU memory and the port; if the checks fail, the service stays stopped. Plain `start` checks an already-running recipe's readiness and leaves its configuration in place.

Each new launch writes `logs/run-<UTC>-<pid>/` and updates `logs/latest`:

```text
model.json             selected snapshot, shard sizes and metadata hashes
image.json             exact Docker image ID and registry digest
launch-command.txt     shell-escaped command; credentials are passed by name
runtime.env            frozen B12X/runtime environment, no credentials
gpus.csv               GPU identity, driver and memory before startup
container-id.txt       container identity
completion-response.json / checks.json
server.log             captured on startup failure; stop/restart also save logs
```

`prepare.sh` records `.state/image.json`. If the tag later points at a different local image ID, `start.sh` refuses silent drift; review the image and rerun `prepare.sh` to update the lock. Normal launch resolves the image to its immutable local ID and does not pull it automatically.

## Configuration changes

`.env` is literal `KEY=value` data. No shell expansion, quote removal or shell execution occurs. Use `.env.example` as the template; exported recipe settings take precedence. Keep credentials in the exported environment.

For another shared cache, set `HF_CACHE_ROOT=/absolute/path`. For a complete standalone checkpoint directory, set `MODEL_DIR=/absolute/path/to/checkpoint`. Do not detach just a snapshot directory from its Hub blobs; the validator rejects links that would escape the container mount. Automatic local-folder discovery also recognizes `nvidia/glm-5.3-flash-fp4-local` and `glm-5.3-flash-fp4-local` under the cache root, but the standard Hub layout is preferred.

Model validation checks each shard's header, indexed tensor names and exact data
extent. Use relative links within the mounted tree; absolute internal links are
rejected because the container uses a different path. Receipts retain the requested
revision; directories without matching snapshot-path evidence are marked
`local-directory-unverified-revision`.

`config/model-lock.json` pins the NVIDIA config and index hashes. Startup checks metadata identity, required tokenizer files, shard presence, sizes and container-visible links; it does not rehash 165 GiB on every launch. An intentional checkpoint upgrade needs reviewed metadata and its own full revision. `config/b12x.env` holds the supplied kernel/NCCL settings. The default `LOAD_FORMAT=safetensors` passed on this host. Both the original InstantTensor settings and a 256 MiB/eight-slot staging experiment failed while loading MTP; `LOAD_FORMAT=instanttensor` remains available for further investigation. Inference uses the same B12X kernels with either loader. Keep those fixed for the initial baseline.

If graph capture runs out of memory, preserve the failure logs first. Check other GPU users before experimenting with fewer sequences or a smaller graph configuration. Reducing `GPU_MEMORY_UTILIZATION` alone does **not** reduce the explicitly allocated `KV_CACHE_MEMORY_BYTES`.

## Repository checks

```bash
./scripts/check.sh         # shell syntax, ShellCheck, Python syntax and CPU-only tests
```

The repository checks use fixtures and a local mock HTTP server. They perform no model inference or Docker operations on the host. See [validation status](../docs/VALIDATION.md), [troubleshooting](../docs/TROUBLESHOOTING.md), and the [TensorFold build/qualification plan](../docs/TENSORFOLD.md).

The operational structure is inspired by the neighboring [Aevonix/Mia's AI Lab two-GPU TensorFold recipe](https://github.com/Aevonix/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold). Its EXL3 patches and performance results belong to that checkpoint/runtime. The NVIDIA weights, container and TensorFold retain their respective licenses; see [provenance](../docs/PROVENANCE.md).
