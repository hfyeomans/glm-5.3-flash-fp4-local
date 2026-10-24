# TensorFold · the native-engine tasting menu

This separate recipe successfully builds **TensorFold 1.1.3** for two RTX PRO 6000 Blackwell cards. **This release's native GLM CUDA loader cannot load the exact NVIDIA GLM-5.3-Flash-FP4-Local checkpoint.** The reproducible CPU probe fails with `MissingTensor`; the startup commands refuse this known incompatibility before allocating GPUs. The working vLLM baseline remains the serving option.

The shared ModelOpt parser recognizes the NVIDIA format and its CSF decoder passes tests, but GLM's CUDA loader does not use those newer paths. This recipe keeps the original NVIDIA files unchanged and does not substitute older NVFP4, ZAI or EXL3 weights.

## Ingredients and provenance

| Ingredient | Pin |
| --- | --- |
| Engine | [TensorFold v1.1.3](https://github.com/ashhart/TensorFold/releases/tag/v1.1.3), commit `36415df4f76e4458e82d0fa729ecb28e287ebf38` |
| Native compiler | Zig `0.17.0`; official archive SHA256 `1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026` |
| CUDA base | `nvidia/cuda:13.3.0-devel-ubuntu24.04`, amd64 digest `sha256:69e9e39eb8fe2cda271654a0f5eac2f1bb946b2fb9c460eb19c7c3c155f4e64e` |
| NCCL | `2.31.2-1+cuda13.3`; native TensorFold requires at least 2.30 |
| Structured output | xgrammar `v0.2.8`, commit `97787376faee5ed8466cfad57c99855e4ce2f6aa`, pinned recursive submodules, built into `libtfgrammar.so` |
| Build | `zig build native grammar -Dversion=1.1.3 -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=120 -Dxgrammar=/src/xgrammar -j4` |
| Checkpoint | `nvidia/GLM-5.3-Flash-FP4-Local`, revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` |
| Quantization | ModelOpt mixed precision, including NVIDIA `csf_32x2_v1` compressed NVFP4 scales |
| Source model location | `HF_CACHE_ROOT=/models_engrams/huggingface`, mounted read-only, including Hub blob links |
| Image / container | `glm53-fp4-tensorfold:1.1.3-sm120` / `glm53-fp4-tensorfold` |
| Runtime cache | Docker volume `glm53-fp4-tensorfold-1.1.3` |
| HTTP / rank coordination | Port `8001` / loopback ports `29600` and `29601` |

Engine, compiler, CUDA base and NCCL are pinned. Ubuntu support packages come from the build-time apt repository; `/opt/tensorfold/build-packages.txt` records their actual versions. The image keeps its compiler and unmodified upstream sources for diagnostics. This is a large development image, not a minimal deployment artifact.

The existing local image named `tensorfold-glm53:1.1.0-2x` contains the older Python 0.6.6 engine with EXL3 patches. Its tag does not identify a native 1.1.0 build. This recipe uses its own image.

## Mise en place: build without taking the GPUs

From the parent repository:

```bash
cp tensorfold/.env.example tensorfold/.env  # optional configuration
./scripts/check.sh
./tensorfold/build.sh
./tensorfold/inspect.sh
./tensorfold/compatibility.sh          # expected failure for the pinned checkpoint
./tensorfold/probe-loader.sh           # reproduces native MissingTensor, no GPU
```

`build.sh` prints its log path under `logs/tensorfold/`. Follow that file with `tail -f` while CUDA kernels compile. It builds the native server, grammar library and a small diagnostic executable that calls the exact NVFP4 reader used by GLM CUDA. It runs host tests for grammar installation, compressed-scale decoding and ModelOpt admission. `inspect.sh` checks the exact local snapshot and image provenance, then runs native version, capabilities and model-info commands with **no GPU and no network**.

The parent `.env` is not loaded. `tensorfold/.env` accepts only the keys in its example, as literal, unquoted values; host environment values for those keys take precedence. `HF_CACHE_ROOT` controls model lookup, even when an old ambient `HF_HOME` points elsewhere. `HF_TOKEN` is an authentication credential, not a directory; offline model access requires no token and does not pass one into this container.

## The compatibility gate

Two independent loader assumptions conflict with the actual snapshot:

- `core.nvfp4.Linear.read` requests `model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale`. That tensor is absent; the checkpoint stores `.weight_scale.csf32_bases`, `codes`, `offsets`, `page_bases` and `exceptions`. The native probe calls this exact unmodified reader and fails with `MissingTensor`.
- GLM CUDA's `Loader.add` handles BF16/F32 at its KDA projection and LM-head loading sites. The actual `layers.0.self_attn.q_proj.weight` and `lm_head.weight` tensors are `F8_E4M3` with quantization metadata. Supporting them requires the appropriate scale-aware decoding or kernels; casting their bytes is not equivalent.

`compatibility.sh` checks these contracts using actual safetensors headers. `doctor.sh` and `start.sh` refuse the known mismatch before GPU/container lifecycle work. The container entrypoint also runs the native probe before launching ranks. A memory-fraction change cannot resolve this problem.

`inspect.sh` remains informational: upstream `info` labels this mixed-format model `full precision` because its generic summary does not recognize this config layout. Its 165.33 GiB directory size includes ancillary files, and its 167.33 GiB memory-floor line is not a TP2 allocation estimate. The recipe's locked identity remains **ModelOpt mixed precision with CSF32 scales, 42 weight shards totaling 165.28 GiB**. Source fixtures, capability listings and `info` output do not establish end-to-end loader support.

The next implementation step is a separately reviewed GLM CUDA loader integration for this checkpoint's mixed projection recipes, compressed expert scales and MTP layouts, followed by memory and numerical qualification. There are no upstream source or arithmetic patches in this build.

## Future service: after the loader is implemented and qualified

The initial envelope is **8,192 tokens, one active request, FP8 KV and the built-in MTP drafter**. TensorFold keeps its own default CUDA memory reserve. The prompt-state cache is limited to 1 GiB. These are starting test settings, not a measured context limit.

The measured vLLM recipe remains at 98.5% utilization, a fixed 7.125 GiB KV allocation per GPU and a 1M context ceiling. TensorFold does not expose equivalent vLLM allocation or B12X controls. Its native allocator, dense/scale layouts, embedding placement and drafting implementation differ, so copying that budget would not establish fit. Grow context and parallelism only after observing actual allocations on both cards.

Build and inspection can run alongside vLLM. After the loader compatibility gate passes, GPU inference needs the selected cards free. `doctor.sh` also checks the model, image, all three ports, GPU model, free memory and existing compute processes. It never stops another service. Its 90,000 MiB free-memory precheck is a coarse admission guard, not a prediction that the model fits.

**Do not stop vLLM to try this pinned release: it is already known to fail.** The following lifecycle is retained for a future compatible, explicitly pinned build after its format gate is updated with evidence:

```bash
./stop.sh                            # explicitly stop this recipe's vLLM service
./tensorfold/doctor.sh
./tensorfold/start.sh
./tensorfold/logs.sh --follow
```

In another terminal:

```bash
./tensorfold/smoke.sh --wait 1200 --suite --output logs/tensorfold/acceptance
./tensorfold/bench.sh --concurrency 1 --requests 8 --output logs/tensorfold/bench-c1.json
```

The endpoint is `http://127.0.0.1:8001/v1`, with model ID `glm-5.3-flash-fp4-local`. Default binding is `0.0.0.0`, matching the baseline. Set `TF_HOST=127.0.0.1` for a local-only service.

`start.sh` records the model metadata, resolved image ID and exact command in `logs/tensorfold/run-*/launch.json`. Native logs use `docker logs`; save them before stopping, because `stop.sh` removes only this recipe's named container after stopping it. It preserves model files and the runtime volume. Automatic restart is disabled during qualification so a loading error does not become a retry loop.

```bash
./tensorfold/status.sh
./tensorfold/logs.sh > logs/tensorfold/server-tail.log
# Save the complete log before removing the container (default name shown):
docker logs glm53-fp4-tensorfold > logs/tensorfold/server.log 2>&1
./tensorfold/stop.sh
./start.sh                           # restore the measured vLLM service
./smoke.sh --wait 1200 --suite
```

## Native controls and comparison boundaries

Two native processes run inside one container. Each sees both selected GPUs and opens its own ordinal using `TF_CUDA_DEVICE=0` or `1`; `TF_GLM_RANK` identifies the rank and `TF_GLM_WORLD=2` the group. Rank zero serves HTTP. If either rank exits, the supervisor terminates its peer and the container fails. A failed-rank host test verifies this cleanup.

Local traffic uses NCCL over the PCIe topology. `TF_ROCE=0` disables the optional multi-host RoCE path, `NCCL_IB_DISABLE=1` disables IB, and bootstrap uses loopback. TensorFold's own communicator settings remain in force. No B12X environment variables, Python-era patches, external drafter or checkpoint substitution are used.

Request defaults use temperature 1, top-p 0.95, thinking enabled and native `xhigh` reasoning effort. The native CLI offers `low/medium/high/xhigh`, whereas this vLLM template uses `max`; these are not evidence of identical reasoning budgets. The native engine uses its own supported template handling for the checkpoint. Built-in GLM MTP is enabled, but this release does not accept the vLLM MTP3 configuration as an equivalent tuning knob. Vision is not enabled in this initial experiment.

The API wrappers reuse the baseline text, JSON, required-tool and streaming checks. They add native `reasoning_effort=low` at the request's top level, retaining the baseline's template kwargs. Client-observed benchmarks include reasoning, prompt processing and network time; they are not isolated decode speed. Begin at concurrency one and qualify higher concurrency before comparing against the existing eight-request vLLM measurement.

Record GPU evidence in [docs/VALIDATION.md](../docs/VALIDATION.md): first-load result, per-rank memory, actual context/slots, structured-output checks, full logs and identical-workload measurements. Test progressively larger prompts and repeated prefixes before raising `TF_CONTEXT`; the model's 1,048,576-token architectural window does not prove that a given native allocation supports it.

## Source evidence

The [release-specific shared ModelOpt reader and NVIDIA fixture](https://github.com/ashhart/TensorFold/blob/36415df4f76e4458e82d0fa729ecb28e287ebf38/zig/src/core/modelopt.zig) explicitly admit `GLM-5.3-Flash-FP4-Local`; [the scale decoder](https://github.com/ashhart/TensorFold/blob/36415df4f76e4458e82d0fa729ecb28e287ebf38/zig/src/core/nvfp4_csf32.zig) implements `csf_32x2_v1`. [Native GLM rank loading](https://github.com/ashhart/TensorFold/blob/36415df4f76e4458e82d0fa729ecb28e287ebf38/zig/src/families/glm/cuda_native.zig) defines the process/rank setup, and [native memory admission](https://github.com/ashhart/TensorFold/blob/36415df4f76e4458e82d0fa729ecb28e287ebf38/zig/src/native/cuda_memory.zig) defines its reserve. The [GLM CUDA weight loader](https://github.com/ashhart/TensorFold/blob/36415df4f76e4458e82d0fa729ecb28e287ebf38/zig/src/families/glm/cuda_weights.zig) calls the [older NVFP4 reader](https://github.com/ashhart/TensorFold/blob/36415df4f76e4458e82d0fa729ecb28e287ebf38/zig/src/core/nvfp4.zig), which requests the missing plain scale. Connecting the shared format support to this loader is required before GPU qualification.
