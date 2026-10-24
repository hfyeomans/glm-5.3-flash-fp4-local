# GLM-5.3-Flash FP4 Local · the two-GPU recipe

A repeatable service recipe for **`nvidia/GLM-5.3-Flash-FP4-Local`** on **two 96 GB NVIDIA RTX PRO 6000 Blackwell GPUs**. The first course uses the model card's supported Local Inference Lab vLLM image, B12X kernels and MTP3. [A separate native TensorFold recipe](tensorfold/README.md) integrates pinned PR #1050 for this checkpoint; its local qualification is recorded independently from the measured vLLM baseline.

The kitchen rule: weigh the ingredients, check the station, then fire the service. Every launch records its checkpoint identity, image ID, settings and results. The measured acceptance checks, loading results and bounded concurrency sample are recorded in [the validation record](docs/VALIDATION.md).

The matched [MTP on/off performance report](docs/PERFORMANCE-CURRENT.md) separates decode speed, prefill and client TTFT. The [quality benchmark](docs/QUALITY-BASELINE.md) measures answer accuracy separately. Local SM120 experiments have their own [pinned tuning recipe](tensorfold/tuning/README.md).

## The ingredients

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

The default image reference includes both the release tag and this digest. The compressed scale format needs the specialized image; replacing it with an arbitrary upstream vLLM image is not a supported substitution. [Sources and deliberate choices](docs/PROVENANCE.md).

## Mise en place

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

**`HF_TOKEN` is a credential, not a directory.** Serving these local files does not need it. The launcher runs with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; it does not forward a token or download replacement weights. Reserve disk space for the image and JIT caches on Docker's filesystem; use `docker system df` and `df -h` before pulling. A 60 GiB planning allowance is a starting budget, not a measured image size.

## The first service

From this repository:

```bash
# Optional: customize the literal KEY=value settings.
cp .env.example .env

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

## Taste before serving

A successful `./start.sh` proves `/health` responds, `/v1/models` lists the expected model, and a short greedy completion returns `ready`. It does not establish long-context capacity or model quality.

`./smoke.sh --suite` additionally requires:

- A JSON response with `{"answer":42}` for a small arithmetic prompt.
- One parsed `get_temperature` tool call with the requested city and unit. It never executes a tool.
- A nonempty streaming response terminating with `[DONE]`.

Every failed check counts; the suite does not retry model-output failures. Read [the test service plan](docs/TESTING.md) before scaling the workload.

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

The short checks select `reasoning_effort=low` with a 512-token budget. This image's template always opens a reasoning block; do not send `enable_thinking=false`, which disables parser separation without removing the template's reasoning prefix. Normal requests retain the server's `max` reasoning default. The default bind is `0.0.0.0`, matching the model card, and the API has no key by default. Set `HOST=127.0.0.1` for local-only clients. Export `VLLM_API_KEY` before startup to enable API authentication; the supplied Python clients read it without printing it.

## During service

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

## Change one ingredient at a time

`.env` is literal `KEY=value` data. No shell expansion, quote removal or shell execution occurs. Use `.env.example` as the template; exported recipe settings take precedence. Keep credentials in the exported environment.

For another shared cache, set `HF_CACHE_ROOT=/absolute/path`. For a complete standalone checkpoint directory, set `MODEL_DIR=/absolute/path/to/checkpoint`. Do not detach just a snapshot directory from its Hub blobs; the validator rejects links that would escape the container mount. Automatic local-folder discovery also recognizes `nvidia/glm-5.3-flash-fp4-local` and `glm-5.3-flash-fp4-local` under the cache root, but the standard Hub layout is preferred.

`config/model-lock.json` pins the NVIDIA config and index hashes. Startup checks metadata identity, required tokenizer files, shard presence, sizes and container-visible links; it does not rehash 165 GiB on every launch. An intentional checkpoint upgrade needs reviewed metadata and its own full revision. `config/b12x.env` holds the supplied kernel/NCCL settings. The default `LOAD_FORMAT=safetensors` passed on this host. Both the original InstantTensor settings and a 256 MiB/eight-slot staging experiment failed while loading MTP; `LOAD_FORMAT=instanttensor` remains available for further investigation. Inference uses the same B12X kernels with either loader. Keep those fixed for the initial baseline.

If graph capture runs out of memory, preserve the failure logs first. Check other GPU users before experimenting with fewer sequences or a smaller graph configuration. Reducing `GPU_MEMORY_UTILIZATION` alone does **not** reduce the explicitly allocated `KV_CACHE_MEMORY_BYTES`.

## Kitchen checks and next course

```bash
./scripts/check.sh         # shell syntax, ShellCheck, Python syntax and CPU-only tests
```

The repository checks use fixtures and a local mock HTTP server. They perform no model inference or Docker operations on the host. See [validation status](docs/VALIDATION.md), [troubleshooting](docs/TROUBLESHOOTING.md), and the [TensorFold build/qualification plan](docs/TENSORFOLD.md).

The operational structure is inspired by the neighboring [Aevonix/Mia's AI Lab two-GPU TensorFold recipe](https://github.com/Aevonix/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold). Its EXL3 patches and performance results belong to that checkpoint/runtime. The NVIDIA weights, container and TensorFold retain their respective licenses; see [provenance](docs/PROVENANCE.md).

## Quality and accuracy

The [initial quality benchmark](quality/README.md) compares 128 fixed GSM8K questions and 128 fixed IFEval prompts between the two engines using official pinned task graders and identical rendered prompts. It saves per-item evidence and reports paired differences and uncertainty. The [measured initial baseline](docs/QUALITY-BASELINE.md) records the completed two-engine comparison. Use it separately from the throughput benchmark and long-context retrieval tests.

[Precision and quality notes](docs/PRECISION-AND-QUALITY.md) explain W4A16, BF16, NVFP4, lossless scale compression, KV-cache precision, speculative decoding, memory headroom and what each test does—and does not—measure.


The separate [TensorFold resident-CSF32 experiment](tensorfold/csf-resident/RESULTS.md)
now passes a 1,048,311-token prompt in a 1M window on these two cards, with unchanged
short-quality hashes and 1,693/1,846 MiB sampled free. It trades about 7% decode speed
and 3–4% longer-prompt prefill throughput for memory, using one slot, mapped embedding
and no retained prompt cache. It is an explicit capacity profile; the vLLM recipe
above remains the supported starting point.


The newer [TensorFold serving profiles](tensorfold/serving/README.md) qualify 512K
with embeddings in VRAM; only the explicit 1M profile maps them to host memory.
The [three-way short-quality comparison](docs/QUALITY-BAKEOFF-V2.md) shows no
convincing accuracy advantage over the archived vLLM baseline. The
[prefill trace](docs/PREFILL-INVESTIGATION-V2.md) records the remaining native
performance costs. Broader 1M accuracy is measured separately from capacity.


The [reduced long-context accuracy screen](docs/LONG-CONTEXT-ACCURACY-V2.md)
finished with 21/22 near-1M core cases fully correct and one partial aggregation
answer. This is a small native regression baseline, separate from capacity and
from the archived matched vLLM comparison.
