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
| TensorFold | Not built or run; see `TENSORFOLD.md` |

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
