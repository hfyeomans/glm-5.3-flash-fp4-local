# TensorFold · PR #1050 on two RTX PRO 6000 Blackwell cards

Build **TensorFold 1.1.3 plus [PR #1050](https://github.com/ashhart/TensorFold/pull/1050)**
for SM120 with the original NVIDIA `GLM-5.3-Flash-FP4-Local` checkpoint.

| Image/profile stage | Tested prompt | Minimum free VRAM, GPU 0 / 1 | Status |
|---|---:|---:|---|
| Original PR image | 524,027 | 339 / 492 MiB | Prefer its 256K working profile; 1M startup failed |
| Mapped embedding only | 524,027 | 1,547 / 1,700 MiB | Separate capacity experiment |
| Current `gpu-512k` resident scales | 524,027 | 4,177 / 4,330 MiB | GPU embeddings; includes probability diagnostic |
| Current `host-1m` resident scales | 1,048,311 | 1,693 / 1,846 MiB | Host embeddings; one slot, retained cache off |

Current [serving profiles](serving/README.md) passed MTP on/off retrieval and
preserved 256 short-quality output hashes and 1,439 fixed-history probabilities.
Use `python3 tensorfold/serving/run.py gpu-512k start` after building its image.
[Quality comparison](../docs/QUALITY-BAKEOFF-V2.md) ·
[Performance](../docs/PERFORMANCE-CURRENT.md) ·
[Prefill investigation](../docs/PREFILL-INVESTIGATION-V2.md).

Plain 1.1.3's GLM loader still calls the older NVFP4 reader. PR #1050 connects
mixed formats and CSF32 scales, preserves FP32 norms and handles MTP layout.
The [SM120 brief](PR1050-SM120-BRIEF.md) records our extensions and retained
upstream choices. [Tuning](tuning/README.md),
[tuning results](TUNING-RESULTS.md) and the
[resident-scale recipe](csf-resident/README.md)/[results](csf-resident/RESULTS.md)
retain separate image pins and historical measurements. The original
[loader/kernel plan](COMPRESSED-SCALES-PLAN.md) and
[validation record](../docs/VALIDATION.md) preserve failures and subsequent fixes.

External DFlash2 integration is deferred; the [implementation and qualification plan](DFLASH2-PLAN.md)
records the resident-scale blocker, head dependencies and memory requirements.

## Fixed ingredients

| Ingredient | Pin |
| --- | --- |
| Base engine | v1.1.3, `36415df4f76e4458e82d0fa729ecb28e287ebf38` |
| PR #1050 | `Bizuayeu/TensorFold`, branch `pr/glm-fp4local`, commit `9b23b6a2e7ff226ca61934189c4cd57bf2933980` |
| Clean merged source tree | `1d52effd182d9319d775d5787904c36783e532f1` |
| Compiler | Zig 0.17.0, official archive SHA256 `1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026` |
| CUDA base | `nvidia/cuda:13.3.0-devel-ubuntu24.04@sha256:69e9e39eb8fe2cda271654a0f5eac2f1bb946b2fb9c460eb19c7c3c155f4e64e` |
| CUDA target / NCCL | SM120 / `2.31.2-1+cuda13.3` |
| Grammar library | xgrammar v0.2.8, `97787376faee5ed8466cfad57c99855e4ce2f6aa`, recursive submodules |
| Model | `nvidia/GLM-5.3-Flash-FP4-Local@76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` |
| Checkpoint format | ModelOpt mixed precision: NVFP4 with `csf_32x2_v1` scales, W4A16_NVFP4, MXFP8, FP8_PB_WO and FP32 norms |
| Cache root | `/models_engrams/huggingface`, read-only model mount |
| Image | `glm53-fp4-tensorfold:1.1.3-pr1050-sm120-context` (launcher v2; native binary unchanged) |
| Container / cache volume | `glm53-fp4-tensorfold` / `glm53-fp4-tensorfold-pr1050` |
| HTTP / NCCL / watchdog | `8001` / `29600` / `29601` |

The Docker build fetches both exact commits, checks the clean merge's tree hash and builds that tree. The binary reports numeric version `1.1.3`, as required by upstream's build system; the image tag and labels identify the PR and merged tree. Ubuntu support package versions are recorded in `/opt/tensorfold/build-packages.txt`. The development image retains the compiler and sources.

## Build and inspect

From the parent repository:

```bash
cp tensorfold/.env.example tensorfold/.env  # optional, do not overwrite existing settings blindly
./scripts/check.sh
./tensorfold/build.sh
./tensorfold/inspect.sh
./tensorfold/compatibility.sh
```

`build.sh` prints a log path under `logs/tensorfold/`. It compiles the native server, grammar library and upstream CUDA test runner for SM120, plus the small CPU format probe. `inspect.sh` runs without a GPU or network; capability output should include `glm5_next: modelopt-mixed`. `compatibility.sh` verifies the locked snapshot, expected tensor headers and the PR image labels, then admits the real mixed recipes and losslessly decodes/re-encodes a real expert scale matrix. GPU loader and arithmetic tests remain separate.

The historical `scripts/legacy_loader_probe.zig` reproduces the old missing-scale error; startup uses the PR loader.

The parent `.env` is not loaded. `tensorfold/.env` accepts only its example keys, as literal unquoted values; shell environment values for those keys win. `HF_CACHE_ROOT` controls the model location even if an old ambient `HF_HOME` points elsewhere. `HF_TOKEN` is a credential, not a path; offline model access needs no token and passes none to the container.

## First GPU service

The initial configuration is **8,192 tokens, one active request, FP8 KV, built-in MTP and a 1 GiB prompt-state cache**, with the native default device-memory reserve. Native defaults use temperature 1, top-p 0.95, thinking and `xhigh` reasoning effort. Vision is not enabled. The recipe selects one expert decode warp and four communication-overlap pieces, both locally measured and checked against the fixed quality suite. Set `TF_GLM_OVERLAP_PIECES=3` to restore the original overlap policy.

`doctor.sh` checks the model, image, three ports, card identity, free memory, compute processes and CPU format probe. It never stops another service. It requires 90,000 MiB free before startup; final fit is checked during loading.

To deliberately switch this repository's service from vLLM to TensorFold:

```bash
./stop.sh
./tensorfold/doctor.sh
./tensorfold/gpu-check.sh             # arithmetic and representative real-loader checks
./tensorfold/start.sh
./tensorfold/logs.sh --follow
```

Then, in another terminal:

```bash
./tensorfold/smoke.sh --wait 1200 --suite --output logs/tensorfold/acceptance-pr1050
./tensorfold/bench.sh --concurrency 1 --requests 8 --output logs/tensorfold/bench-pr1050-c1.json
```

API: `http://127.0.0.1:8001/v1`, model `glm-5.3-flash-fp4-local`. Default binding is `0.0.0.0`; use `TF_HOST=127.0.0.1` for local access only.

Startup records the checkpoint, image ID, configuration and exact command in `logs/tensorfold/run-*/launch.json`. The container has no automatic restart during qualification. Rank zero serves HTTP; each process opens its own CUDA device, and either rank's exit terminates its peer. `TF_ROCE=0`, `NCCL_IB_DISABLE=1` and loopback bootstrap select local NCCL communication. No B12X settings or historical Python/EXL3 patches are passed.

Save complete logs before stopping; `tensorfold/stop.sh` removes only its owned container, preserving weights and the cache volume:

```bash
docker logs glm53-fp4-tensorfold > logs/tensorfold/server-pr1050.log 2>&1
./tensorfold/stop.sh
./start.sh
./smoke.sh --wait 1200 --suite
```

## Qualification and comparison

The image includes `/opt/tensorfold/bin/tf-cuda-test`. The upstream `glm-loader-gate` compares actual GPU uploads against independently cut/packed checkpoint tensors, with `TF_GATE_WORLDS`, `TF_GATE_RANKS` and `TF_GATE_LAYERS` selecting TP geometry and layers. `glm-ref`, `quant-ref`, `mxfp8` and `projection-f32` check relevant arithmetic. `quant-ref` includes NVFP4 expert decode and prefill against float64 references, using the selected decode warp setting. Run them with the intended GPUs free, followed by served checks.

API checks reuse the baseline's text, JSON, required-tool and streaming requests, adding native top-level `reasoning_effort=low`. Further tests should compare drafted/plain output, concurrent/solo requests, repeated-prefix behavior and progressively longer contexts. Native `xhigh`/vLLM `max` reasoning and MTP settings require explicit matching for comparisons.

The PR reports TP2 loading and serving on **GB10/SM121**, not these discrete SM120 cards. Its historical memory receipt gives 90.27 GiB at load end and 92.96 GiB peak per rank. Our bounded concurrency-one test measured 152.11 end-to-end completion tokens/s here versus 184.05 for vLLM. Native loading took 175.9 s; sampled free memory stayed above 3,043 / 3,196 MiB.

## Research findings

Upstream [tracker #1015](https://github.com/ashhart/TensorFold/issues/1015) maps the work. [PR #1021](https://github.com/ashhart/TensorFold/pull/1021) added the shared lossless CSF32 codec, explicitly without a serving caller; [PR #1023](https://github.com/ashhart/TensorFold/pull/1023) added mixed-format config recipes. These foundations are present in 1.1.3. [PR #1050](https://github.com/ashhart/TensorFold/pull/1050) supplies the missing GLM integration.

On 2026-10-24 UTC the upstream repository listed `main`, `ds41-four-ranks` and `codex/beta9-google-recovery`; the relevant work is the contributor fork's `pr/glm-fp4local` branch. Related fork branches address KV paging and TP3 transport; they are not included in this first TP2 test. The PR's older base also needed nested-shard support (#1179), which our 1.1.3 base already includes.

Header accounting on the actual snapshot found 36,288 compressed expert scale matrices: 10,321,055,136 stored bytes expand losslessly to 19,025,362,944 scale bytes, an extra 8.11 GiB across the model (about 4.05 GiB per rank if split equally). Decompressing scales alone cannot make the old loader work: FP8 projections, MTP formats and precision-preserving norms also need the PR's implementation. The recipe preserves the downloaded files.

## SM120 tuning and the context ladder

[SM120-NOTES.md](SM120-NOTES.md) records every local adaptation, upstream behavior already selected for our cards, measured evidence, and untested optimization candidates. It is intended to support a future upstream report; nothing has been posted.

The default remains the reproducible 8K envelope. For context experiments, the launcher now accepts `TF_PROMPT_CACHE_GIB` (default 1), `TF_MEMORY_RESERVE_GIB` (empty keeps native default), `TF_PROFILE` (0/1, default 0) and `TF_NCCL_DEBUG` (WARN/INFO, default WARN). The new image uses the same native binary and kernels; it adds explicit launcher controls and a version label to reject older launchers that would ignore them.

Use `python3 tensorfold/scripts/context_check.py --help` for the cold multi-position retrieval test. It obtains rendered token counts from `/tokenize`, saves compressed request bodies and response evidence, and can compare MTP and plain token hashes. Leave at least 128 tokens for output below the configured window.

The tested 256K working profile (with the intended GPUs free) is:

```bash
TF_NVFP4_DECODE_WARPS=4 TF_CONTEXT=262144 TF_PROMPT_CACHE_GIB=0 TF_MEMORY_RESERVE_GIB=1.5 ./tensorfold/start.sh
./tensorfold/smoke.sh --wait 1200 --suite
python3 tensorfold/scripts/context_check.py --targets 261888 --compare-plain --output logs/tensorfold/recheck-256k
```

GLM allocates the first sequence before reserve admission; use measured free-memory margins from the tuning notes.

The [initial quality baseline](../docs/QUALITY-BASELINE.md) adds official GSM8K/IFEval scoring against vLLM with identical prompts.

## Decode warp experiment

`TF_NVFP4_DECODE_WARPS` accepts only `1` (measured SM120 recipe default) or `4`
(upstream default and the setting used for the historical context ladder).
The existing image contains both variants, and its launcher inherits the explicit
container environment variable; no rebuild or checkpoint conversion is needed.
The recipe passes the setting to serving and GPU checks and records it in launch
evidence. Changing it requires restarting this owned TensorFold service.

With the intended GPUs free, a fixed experiment run is:

```bash
TF_NVFP4_DECODE_WARPS=1 ./tensorfold/gpu-check.sh
TF_CONTEXT=16384 TF_PROMPT_CACHE_GIB=0 TF_MEMORY_RESERVE_GIB= TF_PROFILE=0 TF_NVFP4_DECODE_WARPS=1 ./tensorfold/start.sh
./tensorfold/smoke.sh --wait 1200 --suite
python3 tensorfold/scripts/decode_bench.py --output logs/tensorfold/my-w1
```

Use fresh output directories. Repeat the same fixed configuration with `4`
before and after the candidate, preserving full server logs before each stop.
Each run excludes six warmup calls and measures four repeats of three prompts,
each with drafts on and off. The client requires exactly 256 output tokens,
zero cached tokens, matching prompt counts and repeatable output hashes. It
reports server decode timing separately from client end-to-end time.

```bash
python3 tensorfold/scripts/compare_decode.py logs/tensorfold/my-w4-before logs/tensorfold/my-w1 logs/tensorfold/my-w4-after --output logs/tensorfold/my-warp-comparison.json
```

The comparison rejects changed requests, images, settings and incomplete runs;
it reports output differences and baseline drift. Run the fixed
[quality suite](../quality/README.md) for the candidate and preserve its actual
warp setting alongside the quality manifest (the original quality client does
not include that newer setting). Read measured conclusions in
[SM120-NOTES.md](SM120-NOTES.md).

The bracketed experiment measured a 0.65–1.27% decode-rate gain for one warp,
with all 256 fixed quality responses matching the original TensorFold token
hashes. The normal 8K/1 GiB cache profile also passed cold/warm and drafted/plain
retrieval checks. The original
long-context reproduction command above explicitly retains four warps.
An existing `tensorfold/.env` or shell override takes precedence over the new
default; use `TF_NVFP4_DECODE_WARPS=4` to roll back without rebuilding.

To repeat the normal-profile cache qualification with that one-warp server:

```bash
python3 tensorfold/scripts/cache_check.py --output logs/tensorfold/my-cached-profile
```

The test requires 8K context, one slot, 1 GiB prompt cache, default reserve,
profiling off and one warp. Run it on a fresh service to establish a cold first
request. It checks actual cache reuse and output hashes.

The [initial 1M accuracy screen](../docs/LONG-CONTEXT-ACCURACY-V2.md) is complete:
21/22 core cases fully correct, with one partial frequency-aggregation answer.
