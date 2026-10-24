# Diagnose from the station outward

| Symptom | First check | Action |
| --- | --- | --- |
| Doctor cannot access Docker | `docker info` in the same login | Fix daemon/socket access or the session's Docker permissions |
| Driver inaccessible | Host `nvidia-smi` | Restore host driver access before testing the container |
| Almost no free VRAM | Doctor's GPU rows and running containers | Arrange shutdown of the owning GPU workload; the recipe does not do this automatically |
| Model not found | `HF_CACHE_ROOT`, `MODEL_DIR`, snapshot revision | Keep the whole Hub entry including `blobs`; point to the existing files rather than downloading a substitute |
| Wrong ModelOpt format | Selected `config.json` | Use `nvidia/GLM-5.3-Flash-FP4-Local`; the original ZAI FP8 and EXL3 checkpoints need other serving paths |
| Metadata lock mismatch | `config/model-lock.json` and selected revision | Investigate the files; update pins only for an intentional, reviewed checkpoint change |
| Broken symlinks in container | Model validator's mount error | Mount the Hub root; detached snapshots can point outside their mount |
| Unsupported compressed scales or B12X | Selected image and exact image ID | Use the model card's specialized release; do not silently switch engines |
| CUDA/NCCL startup failure | `logs/latest/server.log`, driver, GPU selection/topology | Confirm GPU access and PCIe peer support; retain the supplied NCCL settings for the baseline |
| InstantTensor fails during MTP load | Loader GPU staging limits | Use the tested default `LOAD_FORMAT=safetensors`; both original and bounded InstantTensor staging failed on this host |
| OOM during capture | Free VRAM, graph and KV settings | Clear other GPU workloads first; then make one documented memory-budget change |
| Port in use | Doctor's bind check | Set another `PORT` or stop the known owner |
| Long compilation | `./logs.sh`, heartbeat | Wait while compilation progresses; increase `WAIT_TIMEOUT` only with evidence of ongoing work |
| Empty/truncated smoke reply | `completion-response.json`, reasoning mode and token budget | Use low reasoning effort and retain reasoning-parser separation; this template always opens a thinking block |
| Tool/JSON failure | Saved raw response and arguments | Treat as an acceptance failure; do not hide misses with selective retries |
| Image drift | `.state/image.json` vs local image ID | Review the changed image, then explicitly refresh with `./prepare.sh` |

`./stop.sh` preserves the stopped container and caches. `./start.sh restart` saves old container logs before removing that recipe-owned container. Nothing prunes Docker volumes or model caches. If a new image needs a clean compilation cache, choose a new `RUNTIME_VOLUME` and retain the old one until the experiment is settled.
