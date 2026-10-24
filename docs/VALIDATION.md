# Validation record

Validated on 2026-10-23 EDT, finishing 2026-10-24 UTC. The server is running at `http://127.0.0.1:8000/v1` with API model `glm-5.3-flash-fp4-local`.

## Final configuration and results

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

The tested workload did not require reducing the KV cache or changing GPU utilization. The memory observations cover short requests at the configured eight-slot limit; they do not qualify every long-context or multimodal workload. GPU memory was sampled periodically, not continuously.

## Loading experiments and fixes

All runs used the same image, checkpoint, TP2/DCP2/MTP3, 98.5% utilization and fixed KV budget.

| Loader experiment | Target weight load | MTP load / outcome |
| --- | ---: | --- |
| Original InstantTensor defaults | 27.50 s | Failed: `RuntimeError: Device memory is not enough` in InstantTensor I/O setup |
| CPU-staged safetensors reference | 24.71 s | 0.33 s; server started and corrected API suite passed |
| InstantTensor, 256 MiB buffer / eight I/O slots | 27.30 s | Same I/O-setup memory failure during MTP load |
| Final safetensors restart | 24.75 s | 0.31 s; startup smoke, API suite, default reasoning and concurrency test passed |

These are bounded observations, not repeated statistical loader benchmarks. The file cache was warm. They show no loading-speed advantage for InstantTensor in these runs. Loader selection does not itself accelerate steady-state inference. `LOAD_FORMAT=instanttensor` remains available for investigation; `safetensors` is the tested default.

The first safetensors server started correctly, but the original smoke client sent `enable_thinking=false`. The image's template always appends `<think>`; disabling parser separation put reasoning and the final answer together in `content`. The client now uses supported `reasoning_effort=low`, keeps parser separation enabled and allows 512 output tokens. The server's default remains `max`.

An immediate restart also exposed a preflight socket check that treated a recently closed connection as an occupied port. The check now uses `SO_REUSEADDR`, matching server behavior. Regression tests verify that recently closed sockets permit restart while an active listener still fails.

## Bounded concurrency sample

Eight parallel requests, eight total requests, 256 output tokens each; one excluded warmup. Related synthetic coding prompts, low reasoning effort, greedy sampling. Prefix reuse is possible.

- Completion tokens: **2,048**, including reasoning tokens.
- Total measured wall time: **4.51 s**.
- Aggregate end-to-end completion rate: **453.97 tokens/s**.
- Median first visible content: **1.285 s**.
- Median request duration: **4.307 s**.

This includes prompt processing and network time; it is not isolated decode throughput or a broad performance qualification.

## Local evidence

Generated artifacts are ignored by Git and retained in `logs/`:

- `run-20261023T234553Z-135306/`: original InstantTensor launch and failure.
- `run-20261023T234946Z-141180/`, `safetensors-reference-server.log`, `acceptance-safetensors/`: reference run, initial client failure and corrected acceptance.
- `run-20261023T235816Z-166869/`, `instanttensor-bounded-server.log`: bounded InstantTensor failure.
- `run-20261024T000109Z-170881/` (`logs/latest`): final successful launch, exact command, metadata, GPU inventory and server log.
- `acceptance-final/`: all four successful API checks and raw responses.
- `default-reasoning-response.json`: successful server-default reasoning check.
- `bench-c8.json`, `bench-c8-memory.json`: every concurrency sample and GPU memory observation.

The final container is `glm53-fp4-tp2`, running with zero automatic restarts. Use `./status.sh`, `./logs.sh` and `./stop.sh` to manage it.


## TensorFold build and compatibility investigation

Recorded 2026-10-24 UTC (2026-10-23 EDT). The separate `tensorfold/` recipe was built without allocating GPUs or stopping the vLLM baseline. **The pinned native GLM CUDA loader is incompatible with this exact NVIDIA checkpoint.** This is a format/loader failure, not a measured GPU-memory failure.

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

The key distinction is between shared format utilities and their integration into a model's actual CUDA path. The shared ModelOpt fixture and CSF32 decoder pass, but `families/glm/cuda_weights.zig` calls `core.nvfp4.Linear.read`, which directly requests a plain scale tensor. `core.Checkpoint.get` performs an exact lookup and does not synthesize that tensor from CSF32 components.

`./tensorfold/probe-loader.sh` executes that exact native reader and produces:

```text
error: checkpoint has no tensor model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale
FAIL native GLM NVFP4 reader: model.language_model.layers.3.mlp.experts.0.gate_proj: MissingTensor
```

The actual index instead contains `.weight_scale.csf32_bases`, `.csf32_codes`, `.csf32_offsets`, `.csf32_page_bases` and `.csf32_exceptions`. Separately, safetensors headers show the first KDA q projection and the LM head use `F8_E4M3`. Source inspection shows the loader's conversion branch only handles BF16/F32 at those sites. The probe reproduces the missing-scale failure; the projection finding is a source/header contract check, not an observed GPU execution error.

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

The model was served from `logs/tensorfold/run-20261024T010539Z/launch.json`. The three omitted uploads in each representative loader check are reported by the upstream checker; the gate is not a claim of complete tensor coverage. Full-model serving subsequently loaded all layers successfully, but that does not extend the upload-equality gate's coverage.

The memory values are sampled observations, not an allocation trace or a promise for larger contexts. The native default reserve remained in force; no smaller reserve or re-quantization was needed. Runtime FP32 norm handling and mixed projection kernels came from the pinned PR.

### Matched short client workload

Both runs used this repository's benchmark prompt, concurrency one, eight requests and 256 completion tokens per request (2,048 total), with an excluded warm-up. vLLM was measured before stopping it, after compilation/host tests finished. TensorFold used the same benchmark with its native spelling of low reasoning effort. The runtimes retain their own speculative behavior and context/cache allocation; this is a bounded comparison of the tested configurations, not a general engine ranking.

| Metric | vLLM baseline | TensorFold PR #1050 |
| --- | ---: | ---: |
| End-to-end completion tokens/s | **184.05** | **152.11** |
| Total wall time | 11.127 s | 13.464 s |
| Median first visible content | 0.188 s | 0.238 s |
| Median request duration | 1.375 s | 1.680 s |

Rates include reasoning, prompt processing and network time. Native load-time scale expansion also means its resident weight representation differs from vLLM's B12X path; the 1M vLLM KV reservation and 8K native window are not comparable memory budgets.

### Why this build works

[Tracker #1015](https://github.com/ashhart/TensorFold/issues/1015) distinguishes shared codecs from family integration. [PR #1050](https://github.com/ashhart/TensorFold/pull/1050), still open when checked, connects mixed recipe admission, CSF32 expert scales, MXFP8/FP8 block projections, FP32 norms and MTP's W4A16 layout to the GLM CUDA engine. Its pinned contributor branch is `Bizuayeu/TensorFold:pr/glm-fp4local`. The upstream branch list also contained `main`, `ds41-four-ranks` and `codex/beta9-google-recovery`; no alternate upstream GLM FP4-Local branch was listed. Related KV-paging and TP3 transport branches were not included.

The release build's failure is therefore an integration gap, not a need for different GPUs or different weights. The old reader requests plain scale tensors; the new loader reconstructs their exact bytes from CSF32 components and retains the other stored precisions. Header accounting found 36,288 compressed matrices, 10,321,055,136 compressed bytes and 19,025,362,944 decoded scale bytes: an 8.11 GiB representation difference across the model. This accounting alone is not a runtime allocation estimate.

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

Each synthetic archive contains three distinct secrets near 10%, 50% and 90% of its records. The client counts the rendered chat with `/tokenize`, verifies the completion usage matches that count, and requires zero cached tokens. Requests are greedy, thinking disabled, with 128 output tokens allowed. Each cold prompt is sent with MTP enabled and then `draft=false`; the complete generated token SHA256s match. This proves these exercised requests, not broad model accuracy at every position or task.

128K uses no retained prompt cache, the unchanged native reserve (~3 GiB), `TF_PROFILE=1` and `TF_NCCL_DEBUG=INFO`. Startup took 164.3 s, model allocations were reported as 90.66 GiB and sequence state as 1.03 GiB. Near-window cold wall times: 31.38 / 32.13 s (drafted/plain). NCCL reports P2P/IPC in both directions. Representative 3–4-row verification summaries put routed experts at 6.28–7.07 ms out of 15.18–16.21 ms GPU time. The diagnostic 8-request decode workload overlapped submission with the last long request and had profiling enabled: do not use its client throughput as a clean engine comparison.

256K uses no retained prompt cache, an explicit 1.5 GiB native reserve and profiling disabled. Startup took 176.8 s, model allocations were 91.58 GiB and sequence state 1.88 GiB. Near-window cold wall times: 64.09 / 66.22 s. The reserve change is specific to this native experiment; vLLM remains at 0.985 and its original fixed KV budget.

512K retains no prompt cache and requests the same 1.5 GiB reserve. Startup took 174.3 s, reported model allocations were 93.41 GiB and sequence state 3.57 GiB. Both cold retrievals passed in 138.32 / 140.79 s, with identical generated token hashes. The 1.5 GiB reserve was **not** maintained as physical free VRAM: GLM allocates its first sequence before admission, then admits it as an existing stream even when remaining budget is zero. No upstream code was changed to obtain this result. The 339 / 492 MiB sampled margin makes this a capacity test, not a comfortable default service configuration.

Evidence: `logs/tensorfold/context-{128k,256k,512k}-{start.log,server.log,acceptance.log,memory.csv,memory-summary.json}`, the corresponding `acceptance/` and `retrieval/` directories, and `run-*/launch.json`. Retrieval folders contain gzipped exact request bodies, request SHA256s and full response/usage/token-hash reports. Memory was sampled every 500 ms; this is not a continuous allocation trace. Recipe checks: 27/27 passed after adding the controls and retrieval client, including rejection of incorrect retrieval, prompt-count mismatch and warm-cache evidence.

The full 1M attempt used the same native binary, exact NVIDIA revision, FP8 KV, one slot, no retained cache and a 1.5 GiB requested reserve. Both ranks reported `cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY` at 01:50:40 UTC; the container exited code 1 at 01:50:41, with `OOMKilled=false`. Readiness failed, so no near-1M prompt was sent. The shutdown path additionally reported a 136-byte SafeAllocator leak with an empty stack trace; its origin is not established. GPU memory was released when the processes exited, and the following vLLM preflight passed. See `logs/tensorfold/context-1m-{attempt.json,start.log,acceptance.log,server.log,stop.log,memory.csv,memory-summary.json}`.

The pinned source increases identified allocations by 7,500 bytes per additional capacity token per rank once fixed ring thresholds are exceeded. From the measured 512K run, 1M needs approximately 3.66 GiB more per GPU against only 339 MiB free on GPU 0. Roughly 3.33 GiB/rank must be saved just to fit, or 4.83 GiB to retain a 1.5 GiB margin, excluding new implementation overhead. These are allocation estimates; actual startup OOM independently confirms this configuration does not fit. Device-side lossless compressed scales, embedding placement and live-KV sharding/offload remain unimplemented capacity candidates. The KV-paging PRs discussed in the notes do not currently page GLM's FP8 cache.

The accuracy-harness research is recorded in the notes: lm-evaluation-harness for generated-answer tasks such as GSM8K/IFEval and NVIDIA RULER for broader long-context tasks. At the end of that capacity ladder, those suites had not been run; the subsequent GSM8K/IFEval results are recorded below. MMLU and RULER remain unrun. The passed retrieval ladder must not be presented as a general accuracy benchmark.

Restoration after the ladder: vLLM started successfully in `logs/run-20261024T015042Z-306221/` and all four API checks passed again (`logs/tensorfold/context-vllm-restored-acceptance/`, matching `.log`). `context-1m-attempt.json` records both restore exit codes as zero. The final running service is `glm53-fp4-tp2` on port 8000. TensorFold and the owned memory samplers were stopped; all model files, images, runtime volumes and evidence were retained.

## Initial paired quality benchmark (2026-10-24 UTC)

The [quality baseline report](QUALITY-BASELINE.md) records 128 fixed GSM8K CoT test items and 128 fixed IFEval prompts on each engine, using the pinned official lm-evaluation-harness task definitions and graders. The [precision guide](PRECISION-AND-QUALITY.md) explains W4A16 and the other numerical/runtime concepts behind the experiment. Reproduction is in [quality/README.md](../quality/README.md).

| Primary metric | vLLM | TensorFold | Paired interpretation |
|---|---|---|---|
| GSM8K flexible exact match | 119/128 (92.97%) | 120/128 (93.75%) | One additional correct item for TensorFold; exact McNemar p=1.0 |
| IFEval strict prompt accuracy | 112/128 (87.50%) | 113/128 (88.28%) | TensorFold-only passes on two items, vLLM-only on one; exact McNemar p=1.0 |

The paired bootstrap intervals include zero; these small differences do not establish an engine accuracy advantage. Both engines completed all 256 formal requests with no HTTP failures, output-budget truncations or missing reasoning closures. Every paired request body and tokenizer-ID hash matched, and completion input counts matched local tokenization. The actual prompts were at most 784 tokens, so this is short-context quality evidence, not long-context accuracy qualification.

Protocol: checkpoint revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`; harness commit `d6de81643928d653435c431bae19945d41d32520`; GSM8K revision `740312add88f781978c0658806c59bc2815b9866`; IFEval revision `966cd89545d6b6acfd7638bc708b98261ca58e84`; seed 1729, 128 unique rows per task, standard eight-shot GSM8K CoT and zero-shot IFEval. The runner renders the locked checkpoint template locally at low reasoning and uses raw completions on both servers, avoiding server chat-template differences. Generation is greedy with a 4,096-token total output budget. The final answer after `</think>` is scored; raw reasoning/output is retained in the local receipts.

The native run used unchanged SM120 kernels with 16K context, one slot, FP8 KV, existing MTP, no retained prompt cache and default reserve. vLLM retained its measured baseline configuration. This was not an unprofiled throughput A/B: configurations and cache behavior differ. It also does not compare the quantized checkpoint against a higher-precision reference. Two questionable GSM8K labels were annotated after inspecting misses, without dropping items or changing official grades.

Evidence: `logs/quality/initial-v1-{vllm,tensorfold}/{manifest.json,requests.jsonl,results.json,server.log}`, corresponding run `.log` files, and `logs/quality/initial-v1-comparison/{comparison.json,comparison.md}`. The committed `quality/baselines/initial-v1.json` preserves scores, uncertainty, source hashes, selected IDs, image IDs and settings without the full benchmark text. Initial adapter pilots caught task-loading API issues before the formal scored runs; valid four-item pilots were excluded from the final statistics. Dependency lock and grader-resource checksum are committed.

Verification: `./scripts/check.sh` passed **32/32** CPU tests, including final-answer extraction, uncertainty calculation, changed-token rejection and paired-score direction. TensorFold passed all four API checks before benchmarking. After preserving its logs it was stopped; vLLM was restored and passed readiness and all four API checks again (`logs/quality/vllm-restored-start.log`, `logs/quality/vllm-restored-acceptance/` and `.log`). Final running service: `glm53-fp4-tp2` on port 8000.

Optimization status remains unchanged: the recipe compiles for SM120 and configures local P2P/NCCL, but no GB10-derived kernel tiles, split-K constants or arithmetic have been manually tuned/replaced. This accuracy baseline qualifies the existing integration and supplies a regression reference for future performance work; it is not evidence of a custom kernel speedup.
