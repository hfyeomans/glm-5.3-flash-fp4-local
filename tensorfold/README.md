# TensorFold · PR #1050 on two RTX PRO 6000 Blackwell cards

This recipe builds **TensorFold 1.1.3 plus [PR #1050](https://github.com/ashhart/TensorFold/pull/1050)** for SM120 and reuses the exact NVIDIA `GLM-5.3-Flash-FP4-Local` checkpoint. It passed the initial 8K two-card test and a later context ladder through **524,027 prompt tokens** with one active slot and identical drafted/plain retrieval outputs. **256K is the preferable tested working profile**: the 512K capacity experiment left only 339 / 492 MiB of VRAM free. **The 1M configuration was attempted and failed during CUDA allocation before readiness.** It remains an experimental integration of an open PR. Local GPU results are recorded separately in [VALIDATION.md](../docs/VALIDATION.md).

The plain 1.1.3 image built previously cannot load this checkpoint. It contains a CSF32 decoder and a mixed-format config reader, but its GLM CUDA path still calls the older NVFP4 reader. PR #1050 connects the formats to the actual model loader, preserves FP32 norms and handles MTP's mixed layout. No weights are converted or replaced.

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

The Docker build fetches both exact commits, checks the clean merge's tree hash and builds that tree. It does not track a moving PR branch or manually change upstream kernels. The binary reports numeric version `1.1.3`, as required by upstream's build system; the image tag and labels identify the PR and merged tree. Ubuntu support package versions are recorded in `/opt/tensorfold/build-packages.txt`. The development image retains the compiler and sources.

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

The historical `scripts/legacy_loader_probe.zig` reproduces the old reader's missing plain-scale error. It is not the PR loader and is not used for the new image's startup checks.

The parent `.env` is not loaded. `tensorfold/.env` accepts only its example keys, as literal unquoted values; shell environment values for those keys win. `HF_CACHE_ROOT` controls the model location even if an old ambient `HF_HOME` points elsewhere. `HF_TOKEN` is a credential, not a path; offline model access needs no token and passes none to the container.

## First GPU service

The initial configuration is **8,192 tokens, one active request, FP8 KV, built-in MTP and a 1 GiB prompt-state cache**, with the native default device-memory reserve. This does not copy vLLM's fixed KV allocation or 98.5% fraction: the two allocators and weight layouts differ. Native defaults use temperature 1, top-p 0.95, thinking and `xhigh` reasoning effort. Vision is not enabled.

`doctor.sh` checks the model, image, three ports, card identity, free memory, compute processes and CPU format probe. It never stops another service. The 90,000 MiB minimum-free check is a coarse guard; it does not guarantee fit.

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

The image includes `/opt/tensorfold/bin/tf-cuda-test`. The upstream `glm-loader-gate` compares actual GPU uploads against independently cut/packed checkpoint tensors, with `TF_GATE_WORLDS`, `TF_GATE_RANKS` and `TF_GATE_LAYERS` selecting TP geometry and layers. `glm-ref`, `mxfp8` and `projection-f32` check relevant arithmetic. Run them only with the intended GPUs free; a loader test does not establish served correctness.

API checks reuse the baseline's text, JSON, required-tool and streaming requests, adding native top-level `reasoning_effort=low`. Further tests should compare drafted/plain output, concurrent/solo requests, repeated-prefix behavior and progressively longer contexts. A successful short request does not qualify the model's 1M architectural context ceiling. Native `xhigh` and vLLM `max` reasoning settings are not guaranteed identical budgets; built-in MTP behavior also differs from vLLM's explicit MTP3 control.

The PR reports TP2 loading and serving on **GB10/SM121**, not these discrete SM120 cards. Its historical memory receipt gives 90.27 GiB at load end and 92.96 GiB peak per rank. Those numbers are useful planning evidence, not a fit guarantee here. Our bounded concurrency-one test measured 152.11 end-to-end completion tokens/s here versus 184.05 for vLLM. Native loading took 175.9 s; sampled free memory stayed above 3,043 / 3,196 MiB. These results cover the documented initial envelope only.

## Research findings

Upstream [tracker #1015](https://github.com/ashhart/TensorFold/issues/1015) maps the work. [PR #1021](https://github.com/ashhart/TensorFold/pull/1021) added the shared lossless CSF32 codec, explicitly without a serving caller; [PR #1023](https://github.com/ashhart/TensorFold/pull/1023) added mixed-format config recipes. These foundations are present in 1.1.3. [PR #1050](https://github.com/ashhart/TensorFold/pull/1050) supplies the missing GLM integration.

On 2026-10-24 UTC the upstream repository listed `main`, `ds41-four-ranks` and `codex/beta9-google-recovery`; the relevant work is the contributor fork's `pr/glm-fp4local` branch. Related fork branches address KV paging and TP3 transport; they are not included in this first TP2 test. The PR's older base also needed nested-shard support (#1179), which our 1.1.3 base already includes.

Header accounting on the actual snapshot found 36,288 compressed expert scale matrices: 10,321,055,136 stored bytes expand losslessly to 19,025,362,944 scale bytes, an extra 8.11 GiB across the model (about 4.05 GiB per rank if split equally). That is representation accounting, not measured runtime VRAM. Decompressing scales alone cannot make the old loader work: FP8 projections, MTP formats and precision-preserving norms also need the PR's implementation. The recipe preserves the downloaded files.

## SM120 tuning and the context ladder

[SM120-NOTES.md](SM120-NOTES.md) records every local adaptation, upstream behavior already selected for our cards, measured evidence, and untested optimization candidates. It is intended to support a future upstream report; nothing has been posted.

The default remains the reproducible 8K envelope. For context experiments, the launcher now accepts `TF_PROMPT_CACHE_GIB` (default 1), `TF_MEMORY_RESERVE_GIB` (empty keeps native default), `TF_PROFILE` (0/1, default 0) and `TF_NCCL_DEBUG` (WARN/INFO, default WARN). The new image uses the same native binary and kernels; it adds explicit launcher controls and a version label to reject older launchers that would ignore them.

Use `python3 tensorfold/scripts/context_check.py --help` for the cold multi-position retrieval test. It obtains rendered token counts from `/tokenize`, saves compressed request bodies and response evidence, and can compare MTP and plain token hashes. Leave at least 128 tokens for output below the configured window. A passing synthetic retrieval test does not establish broad accuracy at that length.

The tested 256K working profile (with the intended GPUs free) is:

```bash
TF_CONTEXT=262144 TF_PROMPT_CACHE_GIB=0 TF_MEMORY_RESERVE_GIB=1.5 ./tensorfold/start.sh
./tensorfold/smoke.sh --wait 1200 --suite
python3 tensorfold/scripts/context_check.py --targets 261888 --compare-plain --output logs/tensorfold/recheck-256k
```

The reserve is an admission setting, not a guaranteed physical free-memory floor for GLM's already-allocated first sequence. See the measured margins in the tuning notes.

The [initial quality baseline](../docs/QUALITY-BASELINE.md) adds official GSM8K/IFEval scoring against vLLM with identical prompts. It complements the loader, arithmetic and retrieval checks; it does not qualify long-context accuracy or quantify loss against an unquantized reference.
