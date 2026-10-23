# Working on this recipe

The initial target is nvidia/GLM-5.3-Flash-FP4-Local with the model card's vLLM release-10152026. TensorFold is a later, separately qualified experiment.

- Keep the NVIDIA checkpoint, revision and compressed-scale format explicit. Do not silently substitute the preset's checkpoint or the original ZAI/EXL3 weights.
- Reuse /models_engrams/huggingface via HF_CACHE_ROOT. Keep model files out of Git and credential values out of logs.
- Never stop unrelated containers, change GPU power/clocks or prune caches as part of preparation or diagnosis.
- Preserve image bootstrap, B12X settings and the supplied memory budget until the baseline is measured. Explain every change to them.
- Run ./scripts/check.sh for code changes. GPU inference is a separate validation step; record actual evidence in docs/VALIDATION.md.
