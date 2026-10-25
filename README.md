# GLM-5.3-Flash FP4 Local on two RTX PRO 6000 Blackwell GPUs

Run **`nvidia/GLM-5.3-Flash-FP4-Local`** on two NVIDIA RTX PRO 6000 Blackwell
Max-Q cards. Start with the vLLM recipe. TensorFold is experimental; its current
profiles and limitations are listed below.

This repository contains launch scripts, source patches and compact result
receipts. Model weights, credentials, local builds and raw run logs stay out of Git.

## vLLM quick start

Requirements: Linux x86_64, working NVIDIA drivers and Container Toolkit, Docker,
Bash, Python 3.10+, `flock`, and the downloaded checkpoint. The preflight checks
available GPU memory and reports other GPU users.

Run from the repository root:

```bash
# Create local settings only if they do not already exist.
test -f .env || cp .env.example .env
# Edit .env for your model cache, GPU IDs, bind address and context limits.

./start.sh --dry-run       # preview the launch
./prepare.sh               # pull and pin the container image
./doctor.sh                # check model, image, GPUs and port
./start.sh                 # start, wait for readiness, send a short completion
./smoke.sh --suite --output logs/acceptance
```

The checkpoint is read from `HF_CACHE_ROOT=/models_engrams/huggingface`, using its
standard Hugging Face Hub layout. For a standalone complete snapshot, set
`MODEL_DIR=/absolute/path/to/checkpoint` in `.env`. Keep Hub snapshot symlinks and
their blobs together. Serving local weights requires no `HF_TOKEN`.

| Fixed ingredient | Value |
|---|---|
| Checkpoint revision | `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` |
| Container | `ghcr.io/local-inference-lab/vllm:release-10152026` |
| Image pin | Tag plus verified registry digest in [.env.example](.env.example) |
| Weight format | Original NVIDIA mixed ModelOpt / CSF32 compressed scales |
| GPU execution | TP2 / DCP2; supplied B12X kernels and bootstrap |
| Loader | `safetensors`; InstantTensor failed at MTP loading on this host |
| Speculation | Built-in MTP, three draft tokens; no separate draft model |
| Client endpoint | `http://127.0.0.1:8000/v1` |
| API model name | `glm-5.3-flash-fp4-local` |

See [provenance](docs/PROVENANCE.md) for image selection and checkpoint pins.

## vLLM settings and lifecycle

Put literal, unquoted `KEY=value` lines in root **`.env`**. Exported values take
precedence. Restart to apply changes; plain `start` leaves an already-running
container's configuration unchanged.

| Setting | Default | Purpose |
|---|---|---|
| `HF_CACHE_ROOT` | `/models_engrams/huggingface` | Existing host model cache |
| `GPUS` | `0,1` | Two host GPU IDs |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Listen address; use `127.0.0.1` for local-only access |
| `MAX_MODEL_LEN` | `1048576` | Per-request prompt **plus output** token ceiling |
| `MAX_NUM_SEQS` | `8` | Maximum active sequences |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | Scheduler token budget per iteration |
| `GPU_MEMORY_UTILIZATION` | `0.985` | Preserve the measured baseline fraction |
| `KV_CACHE_MEMORY_BYTES` | `7650410496` | Fixed FP8 KV allocation **per GPU** |
| `MTP_ENABLED` | `1` | Use `0` for a controlled plain-decoding comparison |
| `LOAD_FORMAT` | `safetensors` | Qualified loader; InstantTensor remains experimental here |
| `RESTART_POLICY` | `no` | Inspect failures; optional `on-failure:3` after acceptance |

For **262,144 tokens per request and four active sequences**, change only these
lines in `.env`, retaining the baseline memory and kernel settings:

```dotenv
MAX_MODEL_LEN=262144
MAX_NUM_SEQS=4
```

Then run `./start.sh restart`. Leaving the defaults also permits four 262K
requests, but permits larger requests and up to eight active sequences too.
The recipe allows one parallel prefill (`--max-parallel-prefills 1`).
Four-full-context capacity testing is pending.

The MTP-on baseline reported **1,844,162 shared KV tokens**, or 1.76 full 1M
windows. Two independent full 1M requests exceed that pool. Monitor queueing,
cache usage and preemption when increasing concurrency. The fixed KV budget is
controlled by `KV_CACHE_MEMORY_BYTES`, independently of the memory fraction.

```bash
./status.sh                # container and API readiness
./logs.sh                  # follow server logs
./smoke.sh --suite         # identity, text, JSON, tool parsing and streaming
./start.sh restart         # recreate only this recipe's container
./stop.sh                  # stop; preserve container, model and runtime cache
```

Startup can spend minutes loading/compiling. Logs and launch receipts go to
`logs/run-<UTC>-<pid>/`, linked by `logs/latest`. Failed startup retains evidence.
Lifecycle commands control only this recipe's container.
[Detailed operations](docs/VLLM-OPERATIONS.md) · [Troubleshooting](docs/TROUBLESHOOTING.md).

## API example

The default bind accepts network clients and has no API key. Set `HOST=127.0.0.1`
for local use, or export `VLLM_API_KEY` before startup to require authentication.
Keep credentials out of `.env` and Git; the supplied Python clients read the
exported key. If authentication is enabled, add its bearer header to curl requests.

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"glm-5.3-flash-fp4-local",
    "messages":[{"role":"user","content":"Explain tensor parallelism briefly."}],
    "temperature":0,
    "max_tokens":512,
    "chat_template_kwargs":{"reasoning_effort":"low","clear_thinking":false}
  }'
```

Normal requests default to maximum reasoning effort. Use `reasoning_effort=low`
for short responses. Avoid `enable_thinking=false`: the pinned template still
opens a reasoning block, causing incorrect parser behavior.

## TensorFold: current status and limitations

The native recipe builds pinned TensorFold **v1.1.3 + PR #1050** for SM120. The PR
supplies NVIDIA checkpoint integration; our local work adds measured RTX settings,
optional mapped embeddings and a separate resident compressed-scale implementation.

| Current resident-CSF32 profile | Placement / concurrency | Measured status |
|---|---|---|
| `gpu-512k` | GPU embeddings; one slot; 524,288-token window | Capacity/retrieval passed with MTP on/off; minimum free VRAM 4,177 / 4,330 MiB including probability diagnostic |
| `host-1m` | Host-mapped embeddings; one slot; 1,048,576-token window | Near-window retrieval passed; initial long-task screen 21/22 fully correct, one partial; minimum free 1,693 / 1,846 MiB |
| Proposed `host-262k-x4` | Four 262,144-token windows; host embeddings candidate | **Blocked / not runnable**: current resident-scale recipe and engine permit one slot only |

Build the pinned experimental image locally using the
[build guide](tensorfold/README.md) and
[resident-scale recipe](tensorfold/csf-resident/README.md), then switch from vLLM:

```bash
./stop.sh
python3 tensorfold/serving/run.py gpu-512k start
python3 tensorfold/serving/run.py gpu-512k smoke --wait 1200 --suite
python3 tensorfold/serving/run.py gpu-512k stop
# Explicit larger-context alternative: substitute host-1m in those commands.
```

Current limits:

- Resident compressed scales are qualified only for TP2, one slot, no external
  draft model or vision. Native MTP uses this checkpoint's draft component.
- The 1M profile has only 157 MiB above our 1,536 MiB sampled headroom floor on
  the limiting card. The reserve setting alone does not enforce that floor.
- Compressed scales save about 3,896 MiB/card but cost 6.6–7.9% decode and 3.2–4.1%
  longer-prompt prefill throughput in controlled same-image comparisons.
- vLLM remains substantially faster on measured long-prefill workloads. The
  short matched quality comparison found **no convincing TensorFold advantage**;
  new native results reproduced all 256 older native answers and 1,439 scored
  probabilities. No matched vLLM near-1M quality comparison has been completed.
- Native CUDA numerical checks passed; the full suite still reports inherited
  resource-record failures.

[Serving profiles](tensorfold/serving/README.md) ·
[Four-stream proposal and qualification gates](tensorfold/serving/FOUR-STREAMS.md) ·
[PR #1050 integration, RTX changes and iteration history](tensorfold/PR1050-SM120-BRIEF.md).
The PR brief is maintained here for a possible upstream contribution.

## Evidence and development

```bash
./scripts/check.sh         # shell lint, Python syntax and CPU-only recipe tests
```

GPU validation results and compact receipts are linked below.

- [Validation record](docs/VALIDATION.md): actual checks, failures and scope.
- [Performance](docs/PERFORMANCE-CURRENT.md) and [prefill investigation](docs/PREFILL-INVESTIGATION-V2.md).
- [Three-way short-quality comparison](docs/QUALITY-BAKEOFF-V2.md) and [initial long-context screen](docs/LONG-CONTEXT-ACCURACY-V2.md).
- [Precision guide](docs/PRECISION-AND-QUALITY.md): W4A16, BF16, FP4, compressed scales and MTP.
- [Credits and model terms](docs/PROVENANCE.md). PR #1050's checkpoint integration
  is credited to Bizuayeu; the operational guide was inspired by
  [Aevonix's two-GPU EXL3 recipe](https://github.com/Aevonix/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold).
