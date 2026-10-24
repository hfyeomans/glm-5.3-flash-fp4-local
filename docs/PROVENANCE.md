# Sources and deliberate choices

Verified 2026-10-23. The reference recipe was inspected for operational structure; this repository's launcher targets the NVIDIA checkpoint and its supplied vLLM stack.

| Source | Evidence used |
| --- | --- |
| [NVIDIA model card, pinned revision](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5/README.md) | Supported image, exact two-GPU environment and vLLM settings, compressed scale requirement and model terms |
| [NVIDIA config](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5/config.json) and [index](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5/model.safetensors.index.json) | `glm5_next`, ModelOpt mixed precision, `csf_32x2_v1`, 42 shards and metadata hashes |
| [Local Inference Lab plain-command guide](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm-5.3-flash/plain-vllm-command.md) | Preset checkpoint differs; preserve the image bootstrap; plain command uses GPU prefix cache |
| [Release-specific shell command](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm-5.3-flash/glm53-tp2-plain-vllm.sh) | Confirmation of the B12X settings and launch shape for release-10152026 |
| [Aevonix two-GPU TensorFold recipe](https://github.com/Aevonix/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold) | Preflight, persistent caches, readiness and smoke-test workflow; local reference commit `87c287855a484381d72b444f48131dd8cd52ed90` |

The model metadata was fetched with `hf` and hashed into `config/model-lock.json`. Temporary research files live in ignored `.firecrawl/`. The existing model cache was relocated as a complete Hub repository, then every weight blob was SHA-256 verified against its content-addressed filename.

Differences from the supplied commands are intentional:

- Use the NVIDIA snapshot directory explicitly. The upstream preset's checkpoint ID and revision are different.
- Mount the existing host Hub cache read-only and serve offline. Store JIT caches separately.
- Retain `release-10152026`, matching the actual commands despite the model card's older heading; pin its verified Linux/amd64 registry digest to prevent tag drift.
- Retain the normal image entrypoint. Supplying `vllm serve` after the image preserves its runtime bootstrap.
- Use CPU-staged `safetensors` loading after both original and bounded InstantTensor loading failed at the MTP head on this host. The model weights, quantization and inference kernels are unchanged.
- Add a stable API alias, metadata/image recording, launch checks and bounded readiness.
- Use low reasoning effort for short client checks. The pinned template always opens a thinking block; setting `enable_thinking=false` caused reasoning to leak into the answer field.
- Default to no automatic restart during testing. The user can choose bounded retries after acceptance.
- Pin per-request context to the supplied 1,048,576-token limit. A newer upstream plain script uses automatic sizing; that is a separate configuration.

The model card describes the checkpoint as governed by the NVIDIA Open Model Agreement, with the ZAI base model under MIT. Follow the linked model terms. This recipe does not transfer the reference recipe's EXL3/DFlash2 licenses or measurements to the NVIDIA model.
