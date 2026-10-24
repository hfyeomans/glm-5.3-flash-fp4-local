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
| TensorFold | Native 1.1.3 image built; exact CUDA loader reader fails on NVIDIA compressed scales. No GPU inference attempted; see the separate record below. |

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
