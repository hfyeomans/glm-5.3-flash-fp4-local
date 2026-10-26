# Validation record

Chronological results from 2026-10-23 EDT / 2026-10-24 UTC onward. Service states are recorded at each phase’s completion.

Summaries: [MTP performance](PERFORMANCE-CURRENT.md), [SM120 tuning](../tensorfold/TUNING-RESULTS.md), [short quality](QUALITY-BAKEOFF-V2.md), [long-task screen](LONG-CONTEXT-ACCURACY-V2.md).

## Initial vLLM baseline

| Check | Result |
| --- | --- |
| Model revision | `nvidia/GLM-5.3-Flash-FP4-Local@76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` |
| Weight integrity after cache move | **42/42 SHA-256 checksums passed**; 177,463,841,024 bytes; metadata hashes match `config/model-lock.json` |
| Model cache | Complete Hub entry under `/models_engrams/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local` |
| Image | `release-10152026`, immutable digest `sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29` |
| Host | Two RTX PRO 6000 Blackwell Max-Q Workstation Edition cards; 97,887 MiB each; driver `615.71.09` |
| Loader | **safetensors**, CPU-staged; same NVIDIA weights and B12X inference kernels |
| Serving settings | TP2, DCP2, MTP3, 8 slots, 4,096 batched tokens, max context 1,048,576 |
| Memory settings | GPU utilization **0.985**; fixed FP8 KV budget **7,650,410,496 bytes per GPU** |
| Bash / ShellCheck / Python syntax | Passed |
| CPU-only operational tests | **15 passed**, including model validation, container ownership, live-port rejection and immediate restart after socket closure |
| Readiness and model identity | Passed |
| Text / JSON / tool call / streaming | **4/4 passed** on both successful safetensors runs |
| Default max-effort reasoning | Passed: reasoning separated from final answer `42`, normal stop |
| Eight concurrent requests | **8/8 completed**, 256 completion tokens each, no request failures or OOM |
| KV capacity reported by engine | **1,844,162 tokens shared**; reported concurrency 1.76x at the maximum per-request context |
| Model memory | **82.73 GiB per GPU** reported during load |
| CUDA graphs | Captured successfully; **0.36 GiB per GPU** |
| Observed free VRAM during concurrency test | Minimum sampled **1,635 MiB / 1,788 MiB** on GPU 0 / 1 |
| Restart and compiled-cache reuse | Passed; final run used cached kernels and completed engine initialization in 34.48 s after model loading |
| Long-context retrieval and vision | Not tested |
| TensorFold | Plain 1.1.3 failed format checks; **1.1.3 + pinned PR #1050 passed bounded SM120 loading and serving checks**. See the separate records below. |

The eight-slot short-request test retained the original KV budget and GPU fraction. Free memory was sampled periodically.

## Loading experiments and fixes

All runs used the same image, checkpoint, TP2/DCP2/MTP3, 98.5% utilization and fixed KV budget.

| Loader experiment | Target weight load | MTP load / outcome |
| --- | ---: | --- |
| Original InstantTensor defaults | 27.50 s | Failed: `RuntimeError: Device memory is not enough` in InstantTensor I/O setup |
| CPU-staged safetensors reference | 24.71 s | 0.33 s; server started and corrected API suite passed |
| InstantTensor, 256 MiB buffer / eight I/O slots | 27.30 s | Same I/O-setup memory failure during MTP load |
| Final safetensors restart | 24.75 s | 0.31 s; startup smoke, API suite, default reasoning and concurrency test passed |

The file cache was warm. Only safetensors completed loading and remains the default; `LOAD_FORMAT=instanttensor` remains available for investigation.

The first safetensors server started correctly, but the original smoke client sent `enable_thinking=false`. The image's template always appends `<think>`; disabling parser separation put reasoning and the final answer together in `content`. The client now uses supported `reasoning_effort=low`, keeps parser separation enabled and allows 512 output tokens. The server's default remains `max`.

An immediate restart also exposed a preflight socket check that treated a recently closed connection as an occupied port. The check now uses `SO_REUSEADDR`, matching server behavior. Regression tests verify that recently closed sockets permit restart while an active listener still fails.

## Bounded concurrency sample

Eight parallel requests, eight total requests, 256 output tokens each; one excluded warmup. Related synthetic coding prompts, low reasoning effort, greedy sampling. Prefix reuse is possible.

- Completion tokens: **2,048**, including reasoning tokens.
- Total measured wall time: **4.51 s**.
- Aggregate end-to-end completion rate: **453.97 tokens/s**.
- Median first visible content: **1.285 s**.
- Median request duration: **4.307 s**.

Rates include prompt processing and network time.

## Local evidence

Generated artifacts are ignored by Git and retained in `logs/`:

- `run-20261023T234553Z-135306/`: original InstantTensor launch and failure.
- `run-20261023T234946Z-141180/`, `safetensors-reference-server.log`, `acceptance-safetensors/`: reference run, initial client failure and corrected acceptance.
- `run-20261023T235816Z-166869/`, `instanttensor-bounded-server.log`: bounded InstantTensor failure.
- `run-20261024T000109Z-170881/` (`logs/latest`): final successful launch, exact command, metadata, GPU inventory and server log.
- `acceptance-final/`: all four successful API checks and raw responses.
- `default-reasoning-response.json`: successful server-default reasoning check.
- `bench-c8.json`, `bench-c8-memory.json`: every concurrency sample and GPU memory observation.

At this phase’s completion, `glm53-fp4-tp2` was running with zero automatic restarts. Use `./status.sh`, `./logs.sh` and `./stop.sh` to manage it.

## TensorFold build and compatibility investigation

Recorded 2026-10-24 UTC (2026-10-23 EDT). The separate `tensorfold/` recipe was built without allocating GPUs or stopping the vLLM baseline. **The pinned native GLM CUDA loader is incompatible with this exact NVIDIA checkpoint.** The failure occurred in format/loader checks before GPU allocation.

| Check | Actual result |
| --- | --- |
| Engine | TensorFold 1.1.3, unmodified commit `36415df4f76e4458e82d0fa729ecb28e287ebf38` |
| Final local image | `glm53-fp4-tensorfold:1.1.3-sm120`, Docker-reported ID `sha256:4f4ad0784c0b44e6e63e542045e065025d65c4e41e5021152a982d4717dc71cd` |
| Image size | 12,852,963,747 bytes, including source and build tools |
| Build inputs | Zig 0.17.0; digest-pinned CUDA 13.3.0 devel base; installed NVCC package `13.3.33-1`; NCCL `2.31.2-1+cuda13.3`; SM120; xgrammar v0.2.8 commit `97787376faee5ed8466cfad57c99855e4ce2f6aa` |
| Native server / grammar build | Passed; upstream source tree remained clean |
| Shared compressed-scale decoder tests | **9/9 passed** |
| Shared ModelOpt config/recipe tests | **6/6 passed**, including the exact NVIDIA FP4 Local fixture |
| Native grammar install host check | Passed |
| Repository checks after adding TensorFold | Shell lint/Python syntax passed; **25/25 operational tests passed**, including compatibility refusal, busy-GPU refusal, container ownership and failed-rank peer cleanup |
| Native version / capabilities | Passed with no GPU and no network; version 1.1.3, CUDA backend, GLM family listed; NCCL, cuBLASLt and grammar library found |
| Offline model validation | Same pinned NVIDIA revision, 42 shards and metadata hashes as the baseline; read-only Hub mount |
| Actual native NVFP4 reader probe | **Failed with `MissingTensor`**, reproducibly on the downloaded checkpoint, without GPU allocation |
| Header contracts | Missing plain expert `weight_scale`; KDA q projection and LM head stored as `F8_E4M3`, while the GLM CUDA loader expects BF16/F32 at those sites |
| Startup guard | `tensorfold/start.sh` exits 1 with a specific compatibility error before GPU/container lifecycle work |
| Existing vLLM service | Still running; `/health` passed after the final checks |
| TensorFold GPU loading, API tests and performance | **Not run**; known loader mismatch blocks this stage |

The shared ModelOpt fixture and CSF32 decoder pass, but `families/glm/cuda_weights.zig` calls `core.nvfp4.Linear.read`, which directly requests a plain scale tensor. `core.Checkpoint.get` performs an exact lookup and does not synthesize that tensor from CSF32 components.

`./tensorfold/probe-loader.sh` executes that exact native reader and produces:

```text
error: checkpoint has no tensor model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale
FAIL native GLM NVFP4 reader: model.language_model.layers.3.mlp.experts.0.gate_proj: MissingTensor
```

The actual index instead contains `.weight_scale.csf32_bases`, `.csf32_codes`, `.csf32_offsets`, `.csf32_page_bases` and `.csf32_exceptions`. Separately, safetensors headers show the first KDA q projection and the LM head use `F8_E4M3`. Source inspection shows the loader's conversion branch only handles BF16/F32 at those sites. The missing-scale failure was reproduced by the probe; the projection mismatch was found by source/header inspection.

Upstream `info` identifies the GLM family but reports `format: full precision`, 165.33 GiB of directory files and a generic 167.33 GiB memory floor. Its quantization summary reads MLX-style metadata, so those lines do not describe the NVIDIA mixed quantization or a per-rank GPU requirement. The recipe continues to identify the model using its pinned ModelOpt/CSF32 metadata and the 165.28 GiB of actual weight shards.

Native loader integration for compressed scales, mixed FP8 projections and MTP layouts is required before inference qualification. No checkpoint bytes were converted, no replacement model was downloaded and no upstream arithmetic was patched. The future starting envelope is 8K context / one slot / native FP8 KV / built-in MTP, with the native default device reserve; none of those settings has been GPU-qualified here. The vLLM memory budget remains unchanged.

Final artifact SHA256s:

- `tensorfold-native`: `7cba7a917f7407007c57c1f9eb0a2ef1f3288111138be148c1c7b7521fab02fe`
- `glm-loader-probe`: `1524dc28df66698450df6869489af0205cdcec008838d3ac80fed5170441da47`
- `libtfgrammar.so`: `9e6d7b7a649e79ad8150704c2f143e49f0c75a4c87d794bf1968b28780974419`

Evidence retained under `logs/tensorfold/` (ignored by Git): `build.log` for the first complete native compilation, `build-20261024T002559Z.log` for all decoder/ModelOpt host tests, `build-20261024T002913Z.log` for the final image and diagnostic binary, `artifacts.log`, `inspect.log`, `compatibility.log`, `loader-probe.log`, `start-refusal.log` and `recipe-check.log`. `doctor-with-vllm.log` records the earlier successful busy-GPU refusal (1,635 MiB free on GPU 0), before adding the stricter format gate.

## PR #1050: successful RTX PRO 6000 Blackwell Max-Q experiment

Validated 2026-10-24 UTC. The user authorized trying the existing open PR on the two local cards. The vLLM container was stopped only for GPU testing, with its configuration, weights and cache preserved, then restarted afterward. No power/clock settings were changed. The exact NVIDIA revision and all checkpoint bytes remained unchanged.

| Check | Actual result |
| --- | --- |
| Source | v1.1.3 base `36415df4f76e4458e82d0fa729ecb28e287ebf38` + PR #1050 head `9b23b6a2e7ff226ca61934189c4cd57bf2933980` |
| Source integration | Clean `git merge-tree`; exact merged tree `1d52effd182d9319d775d5787904c36783e532f1`, verified in the final container; no manual kernel/source edits |
| Image | `glm53-fp4-tensorfold:1.1.3-pr1050-sm120`, Docker ID `sha256:6f5b4b2d833edd0a984bd745184ead48b7a93c89a0b9e50fab8af534b79d0f80`; 13,435,592,523 bytes |
| Toolchain | Zig 0.17.0, CUDA devel 13.3.0 / NVCC package 13.3.33-1, SM120 kernels, NCCL 2.31.2, xgrammar v0.2.8 |
| Native version string | `1.1.3`; upstream only accepts numeric MAJOR.MINOR.PATCH, so PR identity is in the image tag/labels and tree hash |
| Local recipe checks | **26/26 passed**, plus shell lint and Python syntax |
| Upstream host validation | `zig build test test-golden test-kernel-contract fuzz-cortheon -j4 --summary all`: **322/322 steps succeeded; 3,652/3,668 tests passed, 16 skipped**, no GPU or network |
| Actual checkpoint CPU probe | Admitted **494 mixed-format modules**; decoded/re-encoded **524,288 scales** from a real expert matrix, all five components byte-identical |
| Advertised GLM formats | `modelopt-nvfp4` and `modelopt-mixed` |
| Actual GPU identity | Both RTX PRO 6000 Blackwell Max-Q cards identified as **SM120**, with embedded kernels |
| GPU arithmetic | `glm-ref`, `mxfp8`, `projection-f32` passed on **both GPUs**; includes GLM FP32 norm weights and FP64-reference checks |
| GPU real-loader gate | TP2 rank 0 on GPU 0, rank 1 on GPU 1; decoder layers 0, 3, 4, 43, 44 plus MTP. Each: **115 checkpoint-cut + 72 independently packed uploads, 11,922 MiB compared, 0 differences; 3 uploads outside this checker** |
| Native configuration | TP2, **8,192 context**, **one active slot**, FP8 KV, built-in MTP, 1 GiB prompt-state cache, native default CUDA reserve |
| Full-model startup | **175.9 s** reported by server; initial checkpoint digest read 165.3 GiB in **22.6 s**, reused by the other rank |
| Memory admission | Rank-zero startup reports **89.80 GiB model**, **0.23 GiB per stream**, **3.0 GiB reserve**, 1.0 GiB available past reserve; prompt passes up to 2,048 rows |
| Sampled GPU memory | Peak used **94,296 / 94,146 MiB**; minimum free **3,043 / 3,196 MiB**, GPUs 0 / 1; 850 samples per card at approximately 500 ms intervals |
| API acceptance | **4/4 passed:** text, structured JSON, required tool call, SSE streaming |
| Draft/queue consistency | Upstream client, code and chat, 128 tokens greedy, one/two submitted requests with **one active slot**: all recorded drafted/plain and queued/alone comparisons equal, no failed or unmeasured requests |
| Repeated-prefix retrieval | 3,640-token prompt: correct password on both runs, identical token hash; second run reused **2,048 tokens**; 0.889 s cold, 0.460 s repeated |
| Near-window retrieval | **7,481 prompt + 8 completion tokens**, correct password, no cached tokens, **1.757 s** |
| Not qualified | More than one simultaneously active stream, context above 8K, a full-model GPU loader comparison on every layer, full GPU kernel-contract suite, vision/video, broad accuracy or sustained performance |

Launch: `logs/tensorfold/run-20261024T010539Z/launch.json`. Full serving loaded all layers; the selected equality gate excluded three uploads per rank.

The native default reserve and original quantization were retained. FP32 norms and mixed projection kernels came from the pinned PR.

### Matched short client workload

Both runs used this repository's benchmark prompt, concurrency one, eight requests and 256 completion tokens per request (2,048 total), with an excluded warm-up. vLLM was measured before stopping it, after compilation/host tests finished. TensorFold used the same benchmark with its native spelling of low reasoning effort. Each runtime retained its own speculation and context/cache settings.

| Metric | vLLM baseline | TensorFold PR #1050 |
| --- | ---: | ---: |
| End-to-end completion tokens/s | **184.05** | **152.11** |
| Total wall time | 11.127 s | 13.464 s |
| Median first visible content | 0.188 s | 0.238 s |
| Median request duration | 1.375 s | 1.680 s |

Rates include reasoning, prompt processing and network time. Native load-time scale expansion also means its resident weight representation differs from vLLM's B12X path; the 1M vLLM KV reservation and 8K native window are not comparable memory budgets.

### Why this build works

[Tracker #1015](https://github.com/ashhart/TensorFold/issues/1015) distinguishes shared codecs from family integration. [PR #1050](https://github.com/ashhart/TensorFold/pull/1050), still open when checked, connects mixed recipe admission, CSF32 expert scales, MXFP8/FP8 block projections, FP32 norms and MTP's W4A16 layout to the GLM CUDA engine. Its pinned contributor branch is `Bizuayeu/TensorFold:pr/glm-fp4local`. The upstream branch list also contained `main`, `ds41-four-ranks` and `codex/beta9-google-recovery`; no alternate upstream GLM FP4-Local branch was listed. Related KV-paging and TP3 transport branches were not included.

The old reader requests plain scale tensors; the new loader reconstructs their exact bytes from CSF32 components and retains the other stored precisions. Header accounting found 36,288 compressed matrices, 10,321,055,136 compressed bytes and 19,025,362,944 decoded scale bytes: an 8.11 GiB representation difference across the model.

### Reproduction and artifacts

`./tensorfold/gpu-check.sh` reproduces the bounded device checks after the intended GPUs are free. `./tensorfold/start.sh`, `smoke.sh` and `bench.sh` reproduce service validation; the recipe documents explicit switching and rollback.

All detailed evidence is retained under `logs/tensorfold/`: `build-20261024T004923Z.log`, `pr1050-host-tests.log`, `pr1050-recipe-check.log`, `pr1050-compatibility.log`, `pr1050-inspect.log`, `gpu-check-20261024T010231Z/`, `pr1050-gpu-check.log`, `acceptance-pr1050/`, `pr1050-server.log`, `pr1050-container-state.json`, `pr1050-gpu-memory.csv`, `pr1050-memory-summary.json`, `vllm-reference-c1.json`, `bench-pr1050-c1.json`, `pr1050-draft-queue-consistency.json`, `pr1050-repeated-prefix.json`, `pr1050-near-window.json` and `pr1050-artifacts.log`.

Final artifact SHA256s:

- Native server: `535e8a051e157b27fcd43fa47cda695dbaf7b1bb7ea0af8979260cc8d5c5fe75`
- CUDA test runner: `12ab35fd9f727339e0d8309df68ee7d6e76ed641cca2efe391bad8446e452aea`
- Grammar library: `9e6d7b7a649e79ad8150704c2f143e49f0c75a4c87d794bf1968b28780974419`

Rollback verification: vLLM was restored in `logs/run-20261024T011259Z-266204/`. Its readiness and all four API checks passed again (`logs/tensorfold/vllm-restored-acceptance/`). Final running model service: `glm53-fp4-tp2` on port 8000. The TensorFold service was stopped and removed after its logs were saved; its PR image and runtime volume remain available.

## SM120 context ladder and profiling (2026-10-24 UTC)

This continuation preserves the same NVIDIA revision, PR head, merged engine tree, native binary and CUDA kernels. [SM120-NOTES.md](../tensorfold/SM120-NOTES.md) is the detailed local change/tuning journal for a possible future upstream report. No report has been posted.

The separate `glm53-fp4-tensorfold:1.1.3-pr1050-sm120-context` image (`sha256:c829536f5a214755ff50f4e72e622814a9226b8aec43ed646cdc6cd20c418e11`) adds launcher controls for prompt-cache budget, native reserve and profiling. Its native server and CUDA test-runner hashes match the previous image (`logs/tensorfold/context-image-artifacts.log`). The initial 8K defaults remain available. vLLM's configuration is unchanged.

| Configured context | Actual prompt tokens exercised | Result | Minimum sampled free VRAM, GPU 0 / 1 |
|---|---|---|---|
| 131,072 | 8,182; 32,764; 130,812 | Three-position retrieval and drafted/plain token equality passed at every length; all four API checks passed | 3,153 / 3,306 MiB |
| 262,144 | 261,888 | Three-position retrieval, drafted/plain token equality, and all four API checks passed | 2,215 / 2,368 MiB |
| 524,288 | 524,027 | Same checks passed; capacity experiment with minimal remaining VRAM | 339 / 492 MiB |
| 1,048,576 | None | CUDA out of memory during startup; no readiness or inference | Not a successful resident footprint |

Each synthetic archive contains three distinct secrets near 10%, 50% and 90% of its records. The client counts the rendered chat with `/tokenize`, verifies the completion usage matches that count, and requires zero cached tokens. Requests are greedy, thinking disabled, with 128 output tokens allowed. Each cold prompt is sent with MTP enabled and then `draft=false`; the complete generated token SHA256s match.

128K uses no retained prompt cache, the unchanged native reserve (~3 GiB), `TF_PROFILE=1` and `TF_NCCL_DEBUG=INFO`. Startup took 164.3 s, model allocations were reported as 90.66 GiB and sequence state as 1.03 GiB. Near-window cold wall times: 31.38 / 32.13 s (drafted/plain). NCCL reports P2P/IPC in both directions. Representative 3–4-row verification summaries put routed experts at 6.28–7.07 ms out of 15.18–16.21 ms GPU time. The diagnostic 8-request decode workload overlapped submission with the last long request and had profiling enabled: do not use its client throughput as a clean engine comparison.

256K uses no retained prompt cache, an explicit 1.5 GiB native reserve and profiling disabled. Startup took 176.8 s, model allocations were 91.58 GiB and sequence state 1.88 GiB. Near-window cold wall times: 64.09 / 66.22 s. The reserve change is specific to this native experiment; vLLM remains at 0.985 and its original fixed KV budget.

512K retains no prompt cache and requests the same 1.5 GiB reserve. Startup took 174.3 s, reported model allocations were 93.41 GiB and sequence state 3.57 GiB. Both cold retrievals passed in 138.32 / 140.79 s, with identical generated token hashes. The 1.5 GiB reserve was **not** maintained as physical free VRAM: GLM allocates its first sequence before admission, then admits it as an existing stream even when remaining budget is zero. No upstream code was changed to obtain this result. The 339 / 492 MiB margin is too small for the working profile.

Evidence: `logs/tensorfold/context-{128k,256k,512k}-{start.log,server.log,acceptance.log,memory.csv,memory-summary.json}`, the corresponding `acceptance/` and `retrieval/` directories, and `run-*/launch.json`. Retrieval folders contain gzipped exact request bodies, request SHA256s and full response/usage/token-hash reports. Memory was sampled every 500 ms. Recipe checks: 27/27 passed after adding the controls and retrieval client, including rejection of incorrect retrieval, prompt-count mismatch and warm-cache evidence.

The full 1M attempt used the same native binary, exact NVIDIA revision, FP8 KV, one slot, no retained cache and a 1.5 GiB requested reserve. Both ranks reported `cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY` at 01:50:40 UTC; the container exited code 1 at 01:50:41, with `OOMKilled=false`. Readiness failed, so no near-1M prompt was sent. The shutdown path additionally reported a 136-byte SafeAllocator leak with an empty stack trace; its origin is not established. GPU memory was released when the processes exited, and the following vLLM preflight passed. See `logs/tensorfold/context-1m-{attempt.json,start.log,acceptance.log,server.log,stop.log,memory.csv,memory-summary.json}`.

The pinned source increases identified allocations by 7,500 bytes per additional capacity token per rank once fixed ring thresholds are exceeded. From the measured 512K run, 1M needs approximately 3.66 GiB more per GPU against only 339 MiB free on GPU 0. Roughly 3.33 GiB/rank must be saved just to fit, or 4.83 GiB to retain a 1.5 GiB margin, excluding new implementation overhead. These are allocation estimates; actual startup OOM independently confirms this configuration does not fit. Device-side lossless compressed scales, embedding placement and live-KV sharding/offload remain unimplemented capacity candidates. The KV-paging PRs discussed in the notes do not currently page GLM's FP8 cache.

The accuracy-harness research is recorded in the notes: lm-evaluation-harness for generated-answer tasks such as GSM8K/IFEval and NVIDIA RULER for broader long-context tasks. At the end of that capacity ladder, those suites had not been run; the subsequent GSM8K/IFEval results are recorded below. MMLU and RULER remain unrun.

Restoration after the ladder: vLLM started successfully in `logs/run-20261024T015042Z-306221/` and all four API checks passed again (`logs/tensorfold/context-vllm-restored-acceptance/`, matching `.log`). `context-1m-attempt.json` records both restore exit codes as zero. The final running service is `glm53-fp4-tp2` on port 8000. TensorFold and the owned memory samplers were stopped; all model files, images, runtime volumes and evidence were retained.

## Initial paired quality benchmark (2026-10-24 UTC)

The [quality baseline report](QUALITY-BASELINE.md) records 128 fixed GSM8K CoT test items and 128 fixed IFEval prompts on each engine, using the pinned official lm-evaluation-harness task definitions and graders. The [precision guide](PRECISION-AND-QUALITY.md) explains W4A16 and the other numerical/runtime concepts behind the experiment. Reproduction is in [quality/README.md](../quality/README.md).

| Primary metric | vLLM | TensorFold | Paired interpretation |
|---|---|---|---|
| GSM8K flexible exact match | 119/128 (92.97%) | 120/128 (93.75%) | One additional correct item for TensorFold; exact McNemar p=1.0 |
| IFEval strict prompt accuracy | 112/128 (87.50%) | 113/128 (88.28%) | TensorFold-only passes on two items, vLLM-only on one; exact McNemar p=1.0 |

The paired bootstrap intervals include zero. Both engines completed all 256 formal requests with no HTTP failures, output-budget truncations or missing reasoning closures. Every paired request body and tokenizer-ID hash matched, and completion input counts matched local tokenization. Actual prompts were at most 784 tokens.

Protocol: checkpoint revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`; harness commit `d6de81643928d653435c431bae19945d41d32520`; GSM8K revision `740312add88f781978c0658806c59bc2815b9866`; IFEval revision `966cd89545d6b6acfd7638bc708b98261ca58e84`; seed 1729, 128 unique rows per task, standard eight-shot GSM8K CoT and zero-shot IFEval. The runner renders the locked checkpoint template locally at low reasoning and uses raw completions on both servers, avoiding server chat-template differences. Generation is greedy with a 4,096-token total output budget. The final answer after `</think>` is scored; raw reasoning/output is retained in the local receipts.

The native run used unchanged SM120 kernels with 16K context, one slot, FP8 KV, existing MTP, no retained prompt cache and default reserve. vLLM retained its measured baseline configuration.  Two questionable GSM8K labels were annotated after inspecting misses, without dropping items or changing official grades.

Evidence: `logs/quality/initial-v1-{vllm,tensorfold}/{manifest.json,requests.jsonl,results.json,server.log}`, corresponding run `.log` files, and `logs/quality/initial-v1-comparison/{comparison.json,comparison.md}`. The committed `quality/baselines/initial-v1.json` preserves scores, uncertainty, source hashes, selected IDs, image IDs and settings without the full benchmark text. Initial adapter pilots caught task-loading API issues before the formal scored runs; valid four-item pilots were excluded from the final statistics. Dependency lock and grader-resource checksum are committed.

Verification: `./scripts/check.sh` passed **32/32** CPU tests, including final-answer extraction, uncertainty calculation, changed-token rejection and paired-score direction. TensorFold passed all four API checks before benchmarking. After preserving its logs it was stopped; vLLM was restored and passed readiness and all four API checks again (`logs/quality/vllm-restored-start.log`, `logs/quality/vllm-restored-acceptance/` and `.log`). Final running service: `glm53-fp4-tp2` on port 8000.

Optimization status remains unchanged: the recipe compiles for SM120 and configures local P2P/NCCL, but no GB10-derived kernel tiles, split-K constants or arithmetic have been manually tuned/replaced. This supplies the accuracy regression reference for subsequent tuning.

## 2026-10-24 — first SM120 decode-grouping optimization

The recipe now selects `TF_NVFP4_DECODE_WARPS=1`, with `4` available for rollback.
The NVIDIA checkpoint/revision/CSF32 format, native source tree, image
`sha256:c829536f5a214755ff50f4e72e622814a9226b8aec43ed646cdc6cd20c418e11`,
precision, memory budgets, GPU clocks/power settings and vLLM settings are unchanged.
This selects an existing compiled kernel variant; no upstream kernel was edited
or rebuilt. Details and portable receipts are in
[SM120-NOTES.md](../tensorfold/SM120-NOTES.md) and
[decode-warps-v1.json](../tensorfold/results/decode-warps-v1.json).

- **GPU gates:** the one-warp variant passed `info`, `glm-ref`, `quant-ref`,
  `mxfp8`, `projection-f32` and selected real-checkpoint `glm-loader-gate` on both
  cards. Each rank matched 115 checkpoint-cut and 72 host-packed uploads, with
  three exclusions and zero unknown/mismatched uploads. Four-warps `quant-ref`
  also passed on both cards. Evidence: `logs/tensorfold/gpu-check-20261024T023808Z/`
  and `logs/tensorfold/decode-warps-v1/gpu{0,1}-quant-ref-w4.log`.
- **Performance:** 4 → 1 → 4, identical image and settings except warp count:
  16K context, one slot, FP8 KV, zero retained cache, default reserve, profiling
  off. Three prompts (42, 43, 6,931 tokens), drafts on/off, four measured repeats
  each, exactly 256 output tokens: 72 measured requests plus 18 excluded warmups.
  One-warp median server decode rates were **0.65–1.27% higher** than the two
  references; absolute baseline median drift was at most **0.22%**. All measured
  output hashes and speculation counts matched across settings; drafted/plain
  outputs matched.
- **Quality:** all **256 output token hashes**, extracted answers and official
  scores matched the original four-warp TensorFold quality run. GSM8K 120/128;
  strict IFEval 113/128; loose IFEval 114/128. Zero truncations or missing reasoning
  closures. Prompts, tokenizer IDs, dataset/harness/client pins and image matched.
- **Cold context checks:** 8,182 and 16,125 actual prompt tokens recovered all
  three hidden values with matching drafted/plain hashes.
- **Normal-profile cache gate before adoption:** 8K context, one slot, 1 GiB
  prompt cache, default reserve and one warp. All four API checks passed. At
  3,648 and 7,993 prompt tokens, cold/warm and drafted/plain retrieval recovered
  all values and produced identical hashes. Warm calls reused **2,048 / 6,144
  tokens**, respectively. Evidence: `cached-profile/report.json` and logs.
- **Memory and scope:** measured speed-run free-memory minima were unchanged at
  3,975 / 4,128 MiB. Automatic clock
  and temperature variation is recorded; clocks/power were never adjusted.
  The old long-context reproduction commands explicitly retain four warps.
- **Code checks:** `./scripts/check.sh` passed shell lint, Python syntax and all
  **32 CPU tests** after selecting the new default. `git diff --check` passed.

Raw requests, responses, launch records, server logs, comparison reports and GPU
telemetry remain under `logs/tensorfold/decode-warps-v1/`. The native benchmark
reports server decode timing separately from prefill and client latency. Hitting
the output cap is intentional for that speed test.
The precision guide now explains warps, occupancy, tiles, split-K and the
checkpoint's bundled MTP versus an external DFlash drafter. No draft model was
downloaded and no upstream comment or PR was posted.

Final service state: vLLM restored on port 8000; readiness, completion, JSON,
required-tool and streaming checks all passed. TensorFold is stopped, with its
logs preserved. Final restoration evidence: `final-vllm-restore.log`,
`final-vllm-api.log` and `final-vllm-api/` under the experiment directory; vLLM
run folder `logs/run-20261024T031041Z-391354/`. No unrelated service, model file,
cache volume, GPU power setting or clock setting was changed.

## 2026-10-24 — shared cold performance, MTP on/off

Completed the matched raw-completion protocol in
[PERFORMANCE-BASELINE.md](PERFORMANCE-BASELINE.md), with portable results in
`benchmarks/results/cold-shared-v1.json`. The checkpoint revision, image IDs,
launch settings and client hash are retained in that receipt. Six prompts
(31/33 tokens and exact 2K/8K/32K/64K), 256 output tokens, one active request,
greedy sampling and seed 1729; three measured repeats after each cell's warmup.
All server tokenizer arrays match the frozen local tokenizer. Every timed
request reports zero reused input tokens and exactly 256 output tokens.

At 8K input, median decode is 174.1/117.7 tok/s for vLLM MTP on/off and
120.6/68.6 tok/s for TensorFold on/off. Median client TTFT is 0.809/0.791 s and
1.837/1.819 s, respectively. Full lengths and prefill rates are in the
linked report.

vLLM keeps the qualified B12X settings, 0.985 fraction, fixed 7,650,410,496-byte
KV budget, configured 1M window and eight slots. The new `MTP_ENABLED=0` switch
removes only speculative configuration; default remains 1 (MTP3). TensorFold
uses the qualified image, one decode warp, configured 128K window, one slot,
zero retained prompt cache and default reserve. Shared input/output lengths
and one active client are matched; the configured capacity profiles differ.

The native output token hash is identical across repetitions and MTP modes for
all six cases. Five of six vLLM cases have text variation across some repeated
greedy calls despite a fixed seed. This is preserved in raw evidence and distinct
text counts; the cause remains unresolved.

Raw evidence: `logs/tuning/vllm-mtp-on`, `vllm-mtp-off`,
`tensorfold-baseline`, and `shared-baseline`. The two-case vLLM pilot is excluded.
The workload semantic hash is
`5c9cea16d0486aee749dd9543db76705f35e74f2e044233e70ee5e35b5151545`.
CPU shell/syntax and lifecycle checks pass (34 tests, including the experimental
image guard under development). GPU API checks passed before measured runs.

## 2026-10-24 — native tuning gate evidence

The local experimental image `glm53-fp4-tensorfold:sm120-tuning-v1` has image ID
`sha256:2d9b848d07d66c6072ab7757017dd5dffbfb30453305022cbe2befc8904899ee`;
patch, source-tree and native/test binary hashes are recorded in
`tensorfold/results/sm120-native-gates-v1.json`. The qualified image remains
separate and is still the recipe default. Details and commands are in
[tensorfold/tuning](../tensorfold/tuning/README.md).

Completed pre-serve checks:

- All screened expert prefill widths passed byte equality against the original
  template instance on both GPUs, including repeated runs, valid partial tiles,
  non-finite fixtures, output guards and unchanged inputs.
- The combined gate/up 8, down 16 tile image passed `info`, `glm-ref`, `quant-ref`,
  `mxfp8`, `projection-f32` and selected real-checkpoint `glm-loader-gate` checks
  on both GPUs. Each rank's loader gate reported 115 direct and 72 packed uploads,
  three excluded entries and zero mismatches.
- `zig build test-cuda-host` completed with exit code 0. Its deliberate negative
  controls print failure/refusal messages; they are not failures of the command.
- Split targets 192, 384 and 768 passed the quantized reference fixtures on GPU 0.
  A separate tiled-BF16 probe checked eight matrix shapes and seven row counts
  against independent FP64 dots for every column of the first three rows (or all
  rows when fewer than three). All passed its bound; larger batches have sampled
  FP64 coverage. Each larger policy had increased maximum absolute error in six
  of 56 shape/row cells. Per-cell evidence is in
  `tensorfold/results/split-k-fp64-v1.json`.
- Mapped-host BF16 embedding passed the selected real-checkpoint loader gate on
  both GPUs; full-model checks followed separately.

These are local SM120 checks. Full upstream kernel-contract registration, all
supported architectures and release qualification have not been completed.
The read-only ATC review and existing PR #1050 scope are recorded in the tuning
contract; no upstream claim, issue, comment or PR was published.

## 2026-10-24 — completed 18-configuration served screen

All 18 fresh launches completed API, draft/plain retrieval and repeated cold
throughput checks. The screen includes two controls and 16 single-setting or
paired-tile candidates; 144 measured requests plus 72 warmups, each with 256
output tokens. [TUNING-RESULTS.md](../tensorfold/TUNING-RESULTS.md) contains the
full interpretation and portable receipt. Main comparisons use the closing
control because the opening run averaged about 4°C cooler and had 0.90–1.42%
faster long prefill.

Wider expert prefill tiles were essentially tied with the closing control.
Split-K 384/768 and shape-dependent prompt reduction changed all four screening
outputs and reduced long-prefill speed; these were not adopted. Gather was
6.5–6.8% slower on long prefill, and narrow KDA lost decode speed. Other launch,
collective and NCCL allocation changes had no clear adoption-worthy benefit.
Four-piece overlap proceeds to a separate five-repeat bracketed confirmation
because its initial 0.58–0.80% prefill gain is small.

Mapped-host embedding saved **1,208 MiB per GPU**, increasing minimum free VRAM
at the same 128K configuration from 3,153 / 3,306 to 4,361 / 4,514 MiB. Every
screening output hash matched control. Decode was approximately unchanged;
prefill was 1.11–1.29% slower. Accuracy and near-512K qualification followed.

## 2026-10-24 — four-piece overlap confirmation

Completed fresh control / four pieces / control launches at the same 128K
configuration, using five measured repeats per 8K/32K prompt after warmup:
30 measured requests plus six warmups. All output token hashes match. Prefill
improved **1.05% at 8K and 0.87% at 32K** versus the mean control medians;
control-to-control drift was **−0.01% and +0.14%**. Decode differences were
−0.12% and −0.09%.
Read-only clock telemetry was added to this confirmation. Receipt:
`tensorfold/results/overlap4-confirmation-v1.json`; raw evidence:
`logs/tuning/overlap4-confirmation-v1`. Normal-image quality/cache qualification
is required before changing the default.

## 2026-10-24 — normal-image four-piece quality qualification

The unchanged qualified image (`sha256:c829536f5a214755ff50f4e72e622814a9226b8aec43ed646cdc6cd20c418e11`)
passed API and paired retrieval checks with `TF_GLM_OVERLAP_PIECES=4`, one decode
warp, 16K context, one slot, zero retained cache and default reserve. Its shared
short/2K/8K timing run completed three measured repeats with MTP on/off after
warmup; every prompt had one identical output hash across modes and repeats.
The 16-history prompt-log-probability diagnostic completed for later placement
comparison.

The unchanged 256-item quality client produced **256/256 full output token hashes
identical to the original TensorFold baseline**, and every per-item grade matched.
Scores remained 120/128 GSM8K, 113/128 strict IFEval and 114/128 loose IFEval,
with zero truncations and zero missing reasoning closures.
Receipt: `tensorfold/results/overlap4-quality-v1.json`; raw run:
`logs/tuning/overlap4-quality-v1/00-overlap4-quality`. The receipt includes the
actual tuning environment because the original quality manifest predates the
new knobs. Normal 8K cache qualification still precedes a default change.

## 2026-10-24 — fixed-history host-embedding diagnostic

Completed the paired native diagnostic on 16 fixed histories (eight GSM8K and
eight IFEval examples, up to the first 128 saved completion tokens). Both runs
used four-piece overlap and matched 16K settings; the reference used the normal
image and device embedding, while the candidate used the pinned experimental
image and mapped-host embedding. The client verified checkpoint identity,
complete case sets, tokenizer arrays, prompt counts and cold-cache state.

Across **1,439 scored positions**, there were **zero top-one changes** and
**zero change in the actual target token's log probability** (maximum and mean
absolute change both 0). Full manifests and raw
row/histories hashes are in
`tensorfold/results/host-embedding-probabilities-v1.json`. The independent
256-item candidate quality run follows separately.

## 2026-10-24 — combined host-embedding quality qualification

The pinned experimental image with mapped BF16 embedding and four-piece overlap
passed the same 16K API, paired retrieval, MTP-on/off throughput and fixed-history
checks. The unchanged full quality suite produced **256/256 output token hashes
identical to the original TensorFold baseline**, with every per-item grade
unchanged: 120/128 GSM8K, 113/128 strict IFEval and 114/128 loose IFEval. There
were no truncations or missing reasoning closures.

The [native quality receipt](../tensorfold/results/host-embedding-quality-v1.json)
contains actual launch settings, image identity, native timing cells and raw
artifact hashes. Raw run:
`logs/tuning/host-embed-quality-v1/00-host-embed-overlap4-quality`.

## 2026-10-24 — host-embedding 512K capacity qualification

The combined mapped-embedding/four-piece profile passed both **261,888** and
**524,027** actual prompt tokens with MTP on and off. All three markers near
10%, 50% and 90% were recovered; both lengths had identical drafted/plain output
hashes, matching prompt counts, zero reused prompt tokens and normal stop finishes.
Wall times were 62.19 / 63.83 seconds at 256K and 140.58 / 140.52 seconds near
512K (MTP on/off), dominated by prefill.

Settings: context 524,288, one slot, FP8 KV, zero retained prompt cache, reserve
1.5 GiB, one decode warp, four overlap pieces and mapped BF16 embedding. Minimum
sampled free VRAM was **1,547 / 1,700 MiB**, versus **339 / 492 MiB** in the earlier
512K experiment: **1,208 MiB more per GPU**.

Receipt: `tensorfold/results/host-embedding-capacity-v1.json`; raw evidence:
`logs/tuning/host-embed-capacity-v1/00-host-embed-overlap4-512k`. The 256-item
short-context quality suite and fixed-history diagnostic passed separately.
RULER and broad long-context reasoning accuracy remain unmeasured.

**1M still needs additional allocation savings.** The prior 7,500-byte-per-token,
per-rank allocation analysis predicts another 3.662 GiB/rank from 512K to 1M.
With the newly measured headroom, the limiting GPU remains about **2.15 GiB
short just to fit**, or **3.65 GiB short while retaining 1.5 GiB free**, before
new representation/workspace overhead.
Resident compressed scales or live latent-cache sharding require further engine
work.

## 2026-10-24 — four-piece normal-cache acceptance

The normal qualified image passed the 8K / one-slot / 1 GiB prompt-cache profile
with one decode warp and four overlap pieces. At 3,648 and 7,993 input tokens,
first requests were cold, repeated requests reused 2,048 and 6,144 tokens,
respectively, and all cold/warm and drafted/plain output hashes matched. All
markers, prompt counts, mode flags and stop finishes passed. Receipt:
`tensorfold/results/overlap4-cache-v1.json`; raw evidence:
`logs/tuning/overlap4-cache-v1/00-overlap4-default`.

## 2026-10-24 — adopted overlap default and refreshed performance

After quality, fixed-history and normal-cache acceptance, the normal image
completed the full shared 2K/8K/32K/64K plus short-code/prose benchmark with MTP
on/off, three measured repeats and excluded warmups. **All 48 responses matched
the original native output token hashes**, including 64K and both MTP modes.
The report is [PERFORMANCE-CURRENT.md](PERFORMANCE-CURRENT.md), with portable data
in `benchmarks/results/cold-shared-current-v1.json`. The unchanged vLLM cohorts
are reused transparently; native measurements were refreshed after tuning.

At 8K input / 256 output, native median decode is **122.27 / 69.70 tok/s** and
client TTFT is **1.822 / 1.806 s** with MTP on/off. vLLM remains **174.12 / 117.72
tok/s**, TTFT **0.809 / 0.791 s**, on its unchanged qualified settings. The
controlled causal tuning result remains the 0.87–1.05% prefill improvement;
other differences from the earlier full run are not attributed to overlap.

The normal recipe now defaults to **`TF_GLM_OVERLAP_PIECES=4`**. Rollback is
`TF_GLM_OVERLAP_PIECES=3`. One decode warp remains selected. The normal image,
checkpoint revision, target precision and all vLLM budgets are unchanged.
The experimental mapped-embedding profile is separate and explicitly pinned.
Historical screening controls explicitly pin three pieces so this new default
cannot silently alter their reproduction. Final shell/syntax and CPU checks:
**36 tests passed**.

## Final service restoration after tuning

Restored `glm53-fp4-tp2` on port 8000 with the original image
`sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29`.
Readiness/model identity and completion, JSON, tool-call and streaming checks
all passed. Inspection confirmed MTP method `mtp` with three speculative tokens,
`safetensors`, memory fraction 0.985, fixed KV budget 7,650,410,496 bytes and
configured context 1,048,576. The owned TensorFold container is absent after its
logs were saved. No unrelated service, GPU power/clocks, checkpoint files or
shared cache was changed.

Evidence: `logs/tuning/final-vllm-restore.log`, `final-vllm-api/`,
`final-service-state.json` and `final-check.log`. The normal native default is
now four overlap pieces; mapped embedding remains an explicit experimental
capacity profile. All changes and portable receipts are local to this repository;
no upstream issue, comment or PR was published.

## 2026-10-24 — compressed-scale research and contribution record

Added the [SM120 PR discussion brief](../tensorfold/PR1050-SM120-BRIEF.md) and
[resident-CSF32 implementation/validation plan](../tensorfold/COMPRESSED-SCALES-PLAN.md).
These distinguish adopted settings, isolated source patches, rejected candidates
and future work. Live upstream inspection found PR #1050 open at our pinned
head, with GitHub reporting merge conflicts against current main. Nothing was
posted or claimed upstream.

The CPU-only [inventory utility](../tensorfold/tuning/csf_inventory.py) read
safetensors headers and compressed-scale page indexes from the exact pinned
NVIDIA checkpoint. All 36,288 routed matrices across 42 layers were accounted
for, including actual TP2 exception splits and duplicated down-projection row
bases. [The result](../tensorfold/results/csf-residency-plan-audit-v1.json)
records source/metadata hashes, raw byte totals and projection assumptions.
It checks relevant page boundaries/extents, not every codec record or weight.

Projected net savings with 216 MiB shared scale scratch are 3.819 GiB/rank;
combined with the previously qualified mapped-embedding profile, the projected
1M free memory is 1.668 / 1.817 GiB before new metadata/padding/graph overhead.
Compressed residency and its kernels were still unimplemented at this planning stage.

`./scripts/check.sh` passed shell lint, Python syntax and **41 CPU tests**,
including five new accounting tests for TP cuts, byte/record units, empty
exceptions and invalid extents/origins/boundaries. Local document links,
inventory provenance and accounting totals also passed checks.

A read-only `./status.sh` confirmed the existing vLLM container running with
zero restarts, image `sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29`,
and passing readiness/model identity. No service restart, new GPU inference,
checkpoint write, power/clock change or engine configuration change occurred
in this research step.

## 2026-10-24 — resident CSF32 implementation, loader/operator gates

The user authorized local implementation and GPU testing and stopping this recipe's
vLLM container. `glm53-fp4-tp2` is stopped and preserved; no unrelated container,
clock/power setting, checkpoint, vLLM parameter or normal TensorFold image changed.

Experimental source tree `b02cfca0f02235fa8f4b87c0ae921664a73d1807`, patch
`567b5953034033ed5432fc31255149c9315d5e7fc93e1aaaffbc2f05c5d46905`, image
`sha256:e80420cb0f54909e06f5c5aeafa8f0fc5b27c031c5f7e8199d227b6175d9652f`.
It builds on tuned tree `0c2bc6ddf016c0d96d63fbd7d1238b6355603578`; exact
NVIDIA revision and compressed-scale format are unchanged.

Completed evidence (raw logs under `logs/csf-resident-v1/`):

- Separate-layout contract: 14/14 cases on each SM120 GPU, including five identical
  repeats, guards, unchanged inputs and FP64/FP32 accumulation stress gates.
- CSF32 integer expansion: nine shapes per GPU, exact bytes, all 256 byte encodings,
  active/duplicate/empty grouped plans and scratch reuse.
- Real checkpoint loader: each rank's 36,288 matrices / 9,512,681,472 expanded scale
  bytes matched the CPU legacy oracle. All 42 routed layers audited on both ranks.
  Per rank, 1,380 non-CSF raw uploads (80,455 MiB) also matched. The three existing
  draft-head GPU quantizer upload exclusions remain explicit, not counted as passes.
- Mode-1 separated expanded storage: all 1,464 raw uploads / 89,527 MiB per rank
  matched. This mode isolates layout and saves no scale payload memory.
- Host slicing/ABI tests; upstream `test`, `test-kernel-contract`, `test-golden`,
  `fuzz-cortheon`; separate plain build: all exit zero. Negative-fixture output is
  intentional. Lean: 438 inherited findings in both trees, zero new findings.
- Final experimental Docker build succeeded; source and patch identities checked.
  Recipe checks passed 46 tests at this stage.

The [portable loader receipt](../tensorfold/results/csf-resident-loader-v1.json)
records source/binary/log hashes and the per-layer allocation ledger. Net requested
saving is 3.817650/3.817674 GiB per rank, after descriptors, alignment and 216 MiB
scratch. Full GPU contract, memory and served tests follow below.

Preserved failed attempts: a test callback lifetime bug (fixed by static callback
storage), a weak down-projection accumulation stress fixture (strengthened inputs,
unchanged gate), and a full-suite prebuilt-assets invocation missing an unrelated H3
image (rebuilt with complete assets).

### Resident scales: full contract and served quality

The full CUDA suite on GPU0 passed 146/146 numerical kernels, all 33 planted
controls, and its additional gradient/operator checks. It **exited 1**, reporting
885 static-table problems and 64 stale live resource records. All live launches
still fit the device's observed thread/shared-memory limits. An isolated parent-tree
diagnostic reproduced all 949 problems on each GPU, with identical live failure
lines; the 14 added launch records passed. The full numerical suite on GPU1 was not
reached because the original command used `&&`. New operator contracts and real
loader audits independently passed on both GPUs. See [contract receipt](../tensorfold/results/csf-resident-contract-v1.json).

The experimental image then passed completion, JSON, tool and streaming smoke tests;
cold three-marker retrieval at 2,046/8,182 actual prompt tokens, MTP/plain identical;
and all 32 benchmark output hashes (including warmups) matched the mapped-embedding
baseline. The initial 16K-context timing screen overlapped CPU-only diagnostic
compilation and is not used to claim a causal performance improvement.

Full quality: all **256/256 output hashes and grades identical** to the original
TensorFold baseline, GSM8K 120/128; IFEval strict 113/128, loose 114/128. Fixed-history
comparison: 1,439 positions, zero top-one flips, zero target-token log-probability
difference.
Receipts: [quality](../tensorfold/results/csf-resident-quality-v1.json),
[probabilities](../tensorfold/results/csf-resident-probabilities-v1.json),
[initial screen](../tensorfold/results/csf-resident-screen-v1.json).

### Resident scales: cache, physical memory and completed 1M ladder

Cold/warm cache checks at 3,648 and 7,993 tokens passed, with identical MTP/plain
output hashes. The 512K allocation passed cold retrieval at 261,888 and 524,027
actual input tokens in both modes. Minimum sampled free memory: **5,443/5,596 MiB**,
a gain of **3,896 MiB per GPU** over the earlier mapped-embedding 512K run.

The final **1,048,576-token pool** then started with the same one-slot/cache-off
settings and 1.5 GiB reserve target. Serial prompt rungs at **655,104; 786,168;
917,247; and 1,048,311 actual input tokens** all passed three-marker retrieval,
usage/cold-cache/finish checks and exact MTP/plain token-hash comparison. The final
pair took 342.88/342.92 seconds end-to-end. Intermediate rungs used the same 1M pool.

Minimum sampled free memory stayed **1,693/1,846 MiB**, leaving **157/310 MiB** above
the 1.5 GiB target. Telemetry sampled once per second and the harness enforced the
floor after startup smoke and after every rung. Rank-0 native counters reported
98,861,412,570 peak device bytes and 1,269,439,488 pinned-host bytes during late-ladder
sampling; these are allocator counters, not total physical device/host memory.
Mapped BF16 embedding, FP8 KV and the native MTP component remained loaded throughout.
No checkpoint/activation precision, GPU clocks/power, or unrelated processes changed.

[512K receipt](../tensorfold/results/csf-resident-capacity-512k-v1.json) ·
[1M receipt](../tensorfold/results/csf-resident-capacity-1m-v1.json) ·
[cache receipt](../tensorfold/results/csf-resident-cache-v1.json) ·
[experiment results](../tensorfold/csf-resident/RESULTS.md).
This phase used one slot and synthetic retrieval; task accuracy and timing followed separately.

### Resident scales: matched performance and corrected cancellation checks

Same experimental image, 128K context, mapped embedding, one slot, cache off,
overlap four: legacy→resident→legacy, six fixed workloads, MTP on/off, three measured
repeats per cell. **108 measured requests + 36 excluded warmups; all 144 output hashes
match by workload across layouts, modes and repeats.** Returning legacy decode
medians drifted at most 0.25%; 8K–64K prefill controls differed by less than 0.7%.
Automatic clock/temperature telemetry is retained.

Resident scales measured **6.6–7.9% lower decode throughput** and **3.2–4.1% lower
prefill throughput at 2K–64K**. At 8K/MTP-on: returning legacy reference 121.12 decode
tok/s, 4,437 prefill tok/s, 1.847 s client TTFT; resident 112.72, 4,278 and 1.915 s.
At 128K both
legacy controls left 4,361/4,514 MiB sampled free; resident left 8,257/8,410 MiB,
confirming the same 3,896 MiB/card gain seen at 512K.

The first post-timing cancellation fixture was invalid for this server: upstream's
raw `Host: x` requests received HTTP 403 before inference. Each of the three original
sweep status files remains **failed**. Their timing stages had completed successfully
before the fixture error; the report independently validates all rows and preserves
both failures and hashes. The original probes never reached inference.

A fixture-only adapter now uses the URL authority in that raw Host header, preserving
the original file and both hashes. It changes neither server host policy nor test
thresholds (drop after 0.4 s, release and follow-up slack 3 s). Unit checks exercise the
actual wire bytes, preserved body and unknown-fixture refusal. Corrected tests ran
separately on resident and legacy profiles: three observed-running release trials
plus three disconnect/follow-up trials per layout; six `client-left` server events
each. All follow-up text and native hashes matched. Resident release: 0.027 s;
legacy: 0.017–0.356 s; maximum follow-up latency 0.165/0.626 s.

Portable [timing receipt](../tensorfold/results/csf-resident-performance-v1.json) and
[corrected cancellation receipt](../tensorfold/results/csf-resident-cancellation-v2.json).
Final recipe code checks passed **48 tests** (`logs/csf-resident-v1/recipe-check-cancel-host.log`).
The native patch/image remained unchanged through these runs. Full-suite inherited
resource-record failures and unrun architectures remain as documented above.

Final state: vLLM stopped and preserved; TensorFold test container removed; temporary
CPU builders stopped without pruning their caches. No GPU compute processes remain;
free memory returned to 97,187/97,340 MiB. The unrelated graphics allocation remains.
[Evidence index](../tensorfold/results/csf-resident-qualification-v1.json) links the
portable results and final service-state log hash. To start vLLM again from this repo,
run `./start.sh`; the normal image and B12X/memory settings are unchanged.

## GPU-embedding quality and 512K qualification (2026-10-24)

The separate `tensorfold/serving/profiles.json` makes GPU embeddings explicit for
512K and smaller profiles; only `host-1m` uses mapped host embeddings. The immutable
resident-scale image, checkpoint revision and W4A16 arithmetic are unchanged.

The fresh 256-case short-quality run preserved every earlier native output hash
and grade: GSM8K 120/128, IFEval strict 113/128 and loose 114/128. Archived vLLM was
119/128, 112/128 and 114/128 respectively. All 1,439 fixed-history positions matched exactly.
[Three-way comparison](QUALITY-BAKEOFF-V2.md).

At configured 524,288 tokens with GPU embeddings, cold 261,888- and 524,027-token
retrieval passed with MTP both on and off, with identical output hashes. Free
memory stayed at least 4,235 / 4,388 MiB through retrieval. The subsequent
log-probability diagnostic temporarily used 58 MiB more per rank: whole-run minima
were **4,177 / 4,330 MiB**, above the 1,536 MiB floor. Those 1,439 probabilities and
top-1 decisions were again identical to the earlier reference. Sampling is 1 Hz;
transient peaks between samples are not observed.
[Receipt](../tensorfold/results/gpu-512k-qualification-v2.json).

The separate [1M task screen](../quality/LONG-CONTEXT.md) followed with actual
input lengths and per-task results.

## Native prefill diagnosis (2026-10-24)

Two normal 8K/32K controls passed, with <0.05% prefill drift. The first profiling
benchmark was correctly rejected by the normal client's profiling-off guard;
its failure and interleaved combined logs are retained. A separate diagnostic
keeps that guard intact, redirects each rank's stderr to its own file, and validates
complete chunk/stage records. All eight diagnostic requests passed and all 24
control/diagnostic output hashes matched within their workload. Instrumented TTFT
overhead was 2.35%/1.59%; no instrumented timing is presented as production speed.
[Findings and next experiments](PREFILL-INVESTIGATION-V2.md).

## Long-context prompt-protocol correction (2026-10-24)

The initial RULER-derived attempt was cancelled after discovering that this
checkpoint's Jinja ignores `enable_thinking=False`: the raw prompt retained an
open `<think>` under its default Max header. Runtime metadata alone had not caught
this. All 52 shorter substring scores were 100, but five responses truncated;
these are **not valid results for the intended thinking-off protocol**. The first
near-1M request was cancelled before a scored completion. Its failed sweep,
responses and logs remain preserved.

The corrected bank retains every task body/gold/seed/question selection, selects
the Low header and explicitly closes the think prefix, adding two tokens per
case. The client now validates actual prompt content and reports truncated or
reasoning-marker responses separately from official substring scores. Corrected scoring began separately. Recipe checks pass 54 tests. The earlier valid
256-case bakeoff, 512K qualification and prefill measurements are unaffected.
[Failure record](../tensorfold/results/long-context-protocol-failure-v1.json) ·
[Corrected bank](../tensorfold/results/long-context-bank-v2.json).

## Reduced long-context cohort (user request, 2026-10-24)

The user approved two fixed examples per long task instead of four after the
corrected run had completed 71 valid responses. The full-plan client was stopped;
its status records SIGTERM and is not presented as a completed full benchmark.
All 52 shorter controls, including the one incorrect SQuAD answer, remain in the
selected cohort. The first two long sample indices per task define 22 near-1M
cases plus four undersized cases, independent of observed scores. Nine completed
extra long cases are retained separately.

Reuse preflight passed for 62 valid responses, leaving only 16 new scored requests.
It checks the pinned corrected client, bank, image, settings, raw responses,
prompt hashes, golds and memory floor. Invalid-prefix or different-engine results
cannot be reused. Recipe checks pass 56 tests. The continued run uses the same
immutable resident-scale image and `host-1m` profile; generation and grading are
unchanged. [Reduction receipt](../tensorfold/results/long-context-reduction-v1.json).

## Completed reduced long-context screen (2026-10-24)

The agreed balanced core completed: **21/22 near-1M cases had clean full credit**.
All retrieval, variable-tracking and HotpotQA cases passed; one frequency case
returned two of three gold words (66.67%), choosing the fourth-ranked word instead
of third. An independent word-count audit confirms the gold. The second frequency
case passed. No corrected-cohort response truncated or emitted reasoning markers.

All 52 shorter controls remain in the report (51 fully correct, one SQuAD miss).
Both CWE and both SQuAD corpus-limited checks passed at their actual ~143K/~205K
lengths and are excluded from the near-1M headline. Nine already completed extra
long cases passed and remain supplemental. The selected cohort contains 78
results: 62 reused after strict protocol validation and 16 newly executed. No
completed valid case was rerun to select a better answer. The native image,
checkpoint, generation settings and grading are unchanged.

Both source and continued runs passed the sampled 1,536 MiB floor; combined minima
were **1,693/1,846 MiB**. The continuation ended successfully and stopped/removed
only its owned TensorFold container. The original vLLM container remains stopped
and preserved. No GPU compute jobs remain; free memory returned to
97,187/97,340 MiB, preserving unrelated graphics allocations. Recipe checks pass
56 tests; no new native kernel code was changed in this phase.

The core used two examples per task. The matched short bakeoff remained unchanged. [Results](LONG-CONTEXT-ACCURACY-V2.md) ·
[Portable receipt](../tensorfold/results/long-context-accuracy-v2.json) ·
[Word-count audit](../tensorfold/results/long-context-fwe-audit-v1.json).

## Public recipe documentation preparation (2026-10-24)

Refreshed the root vLLM quick start/settings guide, retained detailed operations
in `docs/VLLM-OPERATIONS.md`, and updated the local PR #1050 companion brief with
integration history, RTX-specific decisions, rejected experiments and current
qualification limits. The four-stream TensorFold proposal is documentation only;
its existing single-slot refusal and runnable profiles are unchanged. Read-only
upstream metadata still identifies the pinned PR head; no upstream mutation or
comment was made.

`./scripts/check.sh` passed shell lint, Python syntax and all 56 CPU-only tests.
The sweep fixture emitted existing unclosed-lock-file ResourceWarnings, without
test failures. Documentation link checks and ignore-rule checks passed. A pattern
scan of all 288 reachable historical Git blobs (3,348,883 bytes before this docs
commit) found no matches for the tested credential/private-key patterns, no
sensitive checkpoint/credential filenames and no blobs over 5 MiB. `.env` variants,
editor swap/backup files, model payloads, local environments, builds, caches and
raw logs are ignored; example settings, B12X configuration and compact receipts
remain tracked. Pre-existing unrelated untracked notes are excluded from this
publication change.

No GPU tests, service restarts, new performance/accuracy measurements, engine
changes or model revisions were made in this documentation phase.


## vLLM independent-client context screen (2026-10-25)

The original MTP-on baseline was started from its stopped state, retaining 1M
per-request context, eight slots, 98.5% utilization and the fixed FP8 KV budget.
The single near-1M control and all four two-client retrieval cells (262K, 512K,
768K, near-1M) passed: nine requests, all markers retrieved, zero cached prompt
tokens, normal stops and no preemption. Ordinary pairs had no decode overlap.

The sustained small control showed 7.20 s of token-stream overlap. For the
sustained full-size pair, each client processed 1,032,192 input + 16,384 output
tokens with exact input-ID/output-count verification and isolated markers. Both
completed, at 219.16/438.59 s, but ran sequentially: one running, one waiting,
zero decode overlap and zero preemptions. Pinned source inspection confirmed
full-input cache admission enabled by default; live waiting reason was capacity.
Startup reported 1,844,162 shared KV tokens, below the pair's combined input.

The initial 1,536 MiB headroom probe stopped before completion. After workspace
growth, free memory settled at 903/1,056 MiB. Further capacity cells used a
recorded 512 MiB stop floor without changing server allocation. One ordinary 8K
probe retrieved its markers but repeated reasoning and truncated; it remains a
formatting failure. Exact earlier client sources were restored and verified
against their recorded hashes; raw reports were preserved unchanged.

Post-test readiness and all four API checks passed. The owned vLLM container was
stopped cleanly, exit 0, preserving its model/runtime caches. No compute processes
remained and free memory returned to 97,187/97,340 MiB. CPU checks passed 62 tests.

[Results](VLLM-CONCURRENCY.md) · [Protocol](../benchmarks/CONCURRENCY.md) ·
[Portable receipt](../benchmarks/results/vllm-concurrent-context-v1.json).
The TensorFold engine and serving profiles were unchanged.

## Code-hardening review pass (2026-10-25)

A four-stream adversarial code review (serving/benchmark path, measurement/quality path,
shell/security) produced 16 jev-supported findings; all are fixed on the
`fix/code-review-hardening` branch. CPU-side verification: `./scripts/check.sh` green
with 65 tests, including three new offline guard tests; receipt writers refuse existing
outputs and `context_check` refuses reused directories, exercised by direct invocation.
The measurement-facing guards — endpoint port binding to the recorded container,
zero running/waiting timing boundaries, telemetry-failure propagation, GPU device-id
sampling from the container, `mtp=off` prefill controls, strict
`prompt_tokens_details.cached_tokens` and the pinned `squad.json` hash — are enforced
by code but not yet exercised against live servers; they take effect on the next
qualification run and its receipts are the confirmation.

## DFlash2 feasibility and selector (2026-10-25)

Isolated branch `research/dflash2-feasibility`, based on `4153ef5`: added
`DRAFTER=mtp|dflash2|none`, retaining the MTP default and legacy switch. The pinned
Inco AI GLM-5.3-Flash-DFlash2 weights were already cached; the full 2,342,169,800-byte
file passed SHA256 verification. No weights were downloaded.

`./scripts/check.sh` passed **78 CPU tests**, including offline draft identity,
symlink containment, selector precedence, unchanged target budgets and refusal
before container stop when the draft is missing. Both MTP timing/capacity
harnesses reject external drafts. Initial checks exposed a duplicate dry-run Hub
mount through a host path alias and missing `sys`/`ROOT` definitions in the
existing benchmark startup; both were fixed. All three launch previews passed.

Read-only inspection of the pinned image found dedicated DFlash2 execution,
GLM mHC auxiliary capture and DCP-replicated draft-cache grouping. TensorFold's
resident-CSF path explicitly rejects an external drafter. No GPU inference,
service restart or TensorFold modification was performed. The 128K/one-sequence,
3 GiB-KV probe profile remains untested on GPUs.

[Review and qualification gates](DFLASH2-FEASIBILITY.md) ·
[Source hashes, pins and check receipt](../experiments/dflash2/results/feasibility-v1.json).

## vLLM DFlash2 GPU qualification (2026-10-25)

Research branch only. The original idle MTP container was stopped temporarily,
then restored with its exact ID, image and 262,144-context/four-sequence command.
Readiness and text/JSON/tools/streaming checks passed after restoration. The
experimental container is stopped. TensorFold changes are documentation only.

The first DFlash2 launch failed because regular B12X attention rejects 256-token
pages and replicated DCP KV. Draft-only `TRITON_ATTN` fixed startup; target B12X,
image bootstrap, NVIDIA checkpoint/revision and compressed scales were retained.
The cached draft remained pinned, with BF16 weights. No engine image was patched.

A matched 128K/one-slot/3 GiB-KV comparison ran 32 requests: eight excluded warmups
and 24 measured requests across MTP/DFlash2, 8K/32K inputs and 256/1,024 outputs.
MTP was faster: DFlash2 end-to-end medians were about 7–10% longer. Prefill and TTFT
were similar. Both modes produced varying greedy output hashes across repeats;
broader task accuracy and probability comparisons remain pending.

DFlash2 passed API checks and retrieval at 128K, 512K and near-1M. The 1M profile
processed 1,048,320 input tokens, retrieved all three markers and stopped normally,
with zero cached input/preemption and 1,591/1,744 MiB minimum free memory. It uses
BF16 draft KV, 7.125 GiB explicit KV/card and 98.5%. A 99% attempt failed the CUDA
startup free-memory gate before loading weights; retrying at 98.5% retained the
same explicit KV allocation.

At 262K/four slots, BF16 draft KV planned only 2.70 full windows. Changing only
draft KV to FP8 increased that to 4.35. Four independent clients then each
completed 245,760 input + 16,384 output tokens. Input IDs and output counts matched,
all twelve markers were retrieved, isolation passed, cached input/preemptions were
zero, and all four token-stream intervals overlapped for 40.122 seconds. Minimum
free memory was 1,387/1,540 MiB, passing the recorded 1,024 MiB probe floor. GPU 0
was below the earlier 1,536 MiB working headroom target.

`./scripts/check.sh` passed **84 CPU tests**. Added mode/endpoint/window attribution
checks, an explicit draft-KV precision switch and a labelled DFlash2 capacity mode.
The legacy timing reader now retains the scheduler gauges required by its existing
guard; this fix was CPU-tested. The GPU timing client used its own metric reader.

[Measurements and profiles](../experiments/dflash2/RESULTS.md) ·
[Immutable GPU receipt](../experiments/dflash2/results/gpu-v1.json) ·
[TensorFold implementation plan](../tensorfold/DFLASH2-PLAN.md).

## DFlash2 PR review fixes (2026-10-25)

PR #3 review found a timing-client race: scheduler gauges can remain nonzero
briefly after the response stream ends. The client now waits for one completed
request and idle gauges, rejects extra completions and times out after 30 seconds.
Busy start boundaries still fail. Three regression tests cover delayed gauges,
missing gauges, overlapping traffic and timeout behavior.

`./scripts/check.sh` passed **87 CPU tests**. README profile instructions now copy
only the three profile-specific settings, preserving the existing container, port
and bind address. The 128K example identifies its measured timing profile.
Historical GPU receipts are unchanged; this follow-up made no serving changes.

## Swarm review and regression fixes (2026-10-25)

Reviewed `17d312e` against the 15 findings originally reported at `4153ef5`.
#1 was already fixed; #7 was partly fixed. Confirmed remaining cases were
reproduced before correction across separate lifecycle, measurement and native
worktrees, followed by independent Sol review and a Luna mechanical pass.

The combined branch passed **130 CPU tests**, shell lint and Python syntax checks.
This included the exact Zig agreement helper compiled with Zig 0.17.0: the old
nonfinite comparison failed its reproduction, and three corrected numerical tests
passed. A read-only inspection validated the actual 42-shard NVIDIA cache and
DFlash2 metadata/header/size. This initial review phase used no GPU inference or
container lifecycle actions; subsequent GPU qualification follows below.

Fixes cover endpoint attribution, relocatable model mounts, per-shard truncation,
revision accounting, Docker inspect failures, malformed API tool results, report
comparability/coverage, late telemetry errors and resumed-run protocol/image drift.
Independent follow-ups caught automatic local-folder dry-run discovery and an
already-running container's changed-port acceptance. Cleanup failures retain
telemetry shutdown and no longer print a successful stop message.

Historical receipts are unchanged. The full CUDA probe is pending.
[Finding ledger and tests](SWARM-REVIEW-20261025.md) ·
[Zig maintainer note](../tensorfold/FP64-AGREEMENT-FIX.md).

## Swarm vLLM GPU qualification (2026-10-25)

Both drafters passed from the review worktree at code commit `fb1823a`, using
the pinned NVIDIA CSF32 checkpoint and release-10152026 image. GPU fraction
remained **0.985**, with **7,650,410,496 bytes of KV cache per GPU** and a
1,024 MiB physical-memory stop floor. Hardware: two RTX PRO 6000 Blackwell Max-Q
cards, driver 615.71.09.

| Drafter / profile | Input + output per request | Four-way decode overlap | Minimum free MiB, GPU 0 / 1 | Result |
|---|---|---:|---:|---|
| MTP, 262K × 4 | 229,376 + 32,768 | 126.70 s | 1,187 / 1,270 | Pass |
| MTP, 1M × 1 | 1,048,320 + 44 (256 budget) | — | 1,219 / 1,302 | Pass |
| DFlash2, 262K × 4, FP8 draft KV | 245,760 + 16,384 | 47.31 s | 1,461 / 1,540 | Pass |
| DFlash2, 1M × 1, BF16 draft KV | 1,048,320 + 44 (256 budget) | — | 1,649 / 1,730 | Pass |

All runs passed marker retrieval, request isolation, exact input counts, zero
cache reuse and zero preemptions. The four-client runs also matched streamed
input/output token IDs and counts. Each profile passed completion, JSON, tools
and streaming API checks. Single-request 1M TTFT was 145.58 s for MTP and
139.50 s for DFlash2; the four-client tests used different output lengths.

The initial MTP four-client run passed capacity/retrieval with 16K outputs, but
its first request finished before the fourth began decoding. Its receipt is
retained alongside the successful 32K-output repeat. Both keep each request's
total ceiling at 262,144 tokens.

The explicit `--mtp-profile capacity` option validates matching context/slot
limits, pinned image/checkpoint, loopback endpoint and unchanged MTP3/B12X
configuration. The historical baseline guard remains the default. **136 CPU
tests passed**, including the compiled Zig helper tests; independent Sol review
found no issues in this harness addition. The full TensorFold CUDA probe remains
pending.

[Portable receipt](../benchmarks/results/swarm-vllm-validation-v1.json) ·
[Reproduction commands](../benchmarks/CONCURRENCY.md#matched-mtp-capacity-profiles).
Raw logs, prompts, frames and telemetry remain under
`logs/swarm-validation-20261025/` in the review worktree. Historical receipts are
unchanged.

The original MTP container was restored with identical container ID, image and
launch command, and passed the full API suite. MTP remains the recipe default;
the restored service keeps its previous 262,144-token/four-sequence settings.

### PR #4 follow-up checks

Devin review found two additional cases, reproduced before correction: a failed
TensorFold stop could strand the next profile's folder, and the MTP capacity
guard accepted a changed target attention backend. Cleanup now halts before the
next profile after a stop failure and checks container absence before creating
an attempt folder. Capacity validation now checks target B12X kernels, TP2/DCP2,
FP8 KV, BF16 compute, ModelOpt mixed quantization and 256-token blocks.

**140 CPU tests passed**, including compiled Zig checks. Replaying all three
recorded MTP launch manifests passed the stricter guard; substituting target
TRITON_ATTN was rejected. These follow-ups changed validation/cleanup only;
the GPU receipts and serving configuration are unchanged.

CodeRabbit's two follow-ups also reproduced: alternate exact Docker absence
messages were rejected, and explicit endpoint port `0` became the scheme default.
The lifecycle helper now accepts both known error prefixes while rejecting mixed
errors or a different container name. All five measurement clients preserve
explicit ports and reject mismatches before HTTP/output creation. **144 CPU
tests passed**, including compiled Zig checks; all four PR findings are fixed.

## Review-hardening validation (2026-10-26 UTC)

**V1–V5 passed** on two RTX PRO 6000 Blackwell Max-Q cards. The 52 short
TensorFold controls produced **51/52 clean full-credit responses**, mean task
score **98.08**. The sole miss remains `qa_squad-32768-2`; no responses were
truncated. MTP vLLM was restored and passed completion, JSON, tools and streaming.

| Check | Evidence |
|---|---|
| V1 — launch | Same pinned image, command, mounts and B12X settings; `json-file`, `max-size=100m`, `max-file=3` appear in both launch record and Docker inspect. Live unauthenticated/non-loopback launch warned; key-set and loopback dry runs were silent. Full API suite passed. |
| V2 — fixture | Copied without execution from the stopped `glm53-csf-limits-parent`, whose image matches the serving profile `sha256:e80420cb…`. SHA-256 matches `545c9a0aaa7df67261f80bb489e976403e4c6c1e662dddbbb327e8d9943182a4`. The handoff's vLLM source path lacked this TensorFold fixture. |
| V3 — selected devices | Actual sweep child used `--devices 0,1`; manifest and Docker GPU IDs agree. Smoke, 2K/8K sanity rungs and all 52 controls passed their gates. Minimum free memory **1,765 / 1,846 MiB**, above **1,536 MiB**. |
| V4 — receipt | Floor and GPU IDs match the manifest. A changed on-disk manifest was rejected. Adding an unrelated zero-free GPU to copied telemetry left the selected-device gate passing; selecting that device failed both sampled and physical checks. |
| V5 — suite | **159 CPU tests passed**, including compiled Zig 0.17.0 checks using the existing compiler via `ZIG`. Baseline before the reporter fix: 151 tests, also including Zig. |

The vLLM container was already stopped on arrival. V1 ran `./stop.sh`,
`./start.sh` and `./smoke.sh --suite`; the running service was then stopped for V3
and restored afterward. The NVIDIA revision remains
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, with original compressed scales,
release-10152026 image `sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29`,
GPU fraction 0.985 and 7,650,410,496-byte per-GPU KV budget.

The reporter originally rejected a complete 52-control run because it required
the full short-plus-long cohort. The new `--controls-only` mode was reproduced
failing-first, preserves the default full report, checks the entire frozen
short cohort and records `near_1m: null`. PR review also tightened group-counter
checks and corrected inconsistent test-fixture flags. Inference code and client
pins were unchanged; the first report draft is retained in the raw run folder.

Commands run from the validation worktree (existing dependency overlay and bank):

```bash
./stop.sh
./start.sh
./smoke.sh --suite --output logs/hardening-validation-20261026/v1-api

docker cp glm53-csf-limits-parent:/src/tensorfold/zig/tests/qwen35/serve_cancel_release.py \
  logs/hardening-validation-20261026/serve_cancel_release.py
sha256sum logs/hardening-validation-20261026/serve_cancel_release.py

PYTHONPATH="$PWD/.state/long-quality-deps" .state/quality-venv/bin/python \
  tensorfold/tuning/sweep.py --profile-file tensorfold/serving/profiles.json \
  --profiles host-1m --skip-benchmark --minimum-free-mib 1536 \
  --long-context-bank /home/hank/dev/src/glm-5.3-flash-fp4-local/logs/qualification-v2/ruler-bank-v2 \
  --long-context-lengths 32768 \
  --output logs/hardening-validation-20261026/v3-short-controls

.state/quality-venv/bin/python quality/report_long.py --controls-only \
  --run logs/hardening-validation-20261026/v3-short-controls/00-host-1m \
  --bank /home/hank/dev/src/glm-5.3-flash-fp4-local/logs/qualification-v2/ruler-bank-v2 \
  --receipt tensorfold/results/review-hardening-controls-v1.json \
  --report docs/REVIEW-HARDENING-CONTROLS-20261026.md

ZIG=/home/hank/dev/src/glm-5.3-flash-fp4-local/.state/worktrees/swarm-native/.state/cpu-zig/zig \
  ./scripts/check.sh
```

The frozen bank hash is `90ae1b9aed4a6b5649a4a2e8abcd258d4aaa7982ad08be23fc00253be74d0961`.
All 52 requests were new; actual inputs were 28,471–28,674 tokens. The existing
`host-1m` allocation stayed in place, with no near-1M prompts. This two-GPU host
used a synthetic third-device replay; no physical third GPU was tested.
TensorFold `TF_HOST` parity was outside this pass.

[Validation receipt](../tensorfold/results/review-hardening-validation-v1.json) ·
[Control scores and misses](REVIEW-HARDENING-CONTROLS-20261026.md) ·
[Control receipt](../tensorfold/results/review-hardening-controls-v1.json).
Raw evidence is under `logs/hardening-validation-20261026/` in
`.state/worktrees/hardening-validation`, including `v1-result.json`,
`v3-device-forwarding.json`, `v4-negative-gates.json`, `restored.json`, and the
sweep's `*-headroom.json` / `long-context/headroom.json`. Historical receipts
and qualification sections are unchanged.
