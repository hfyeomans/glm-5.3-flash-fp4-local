# Working on this recipe

The initial target is nvidia/GLM-5.3-Flash-FP4-Local with the model card's vLLM release-10152026. TensorFold is a later, separately qualified experiment.

- Keep the NVIDIA checkpoint, revision and compressed-scale format explicit. Do not silently substitute the preset's checkpoint or the original ZAI/EXL3 weights.
- Reuse /models_engrams/huggingface via HF_CACHE_ROOT. Keep model files out of Git and credential values out of logs.
- Never stop unrelated containers, change GPU power/clocks or prune caches as part of preparation or diagnosis.
- Preserve image bootstrap, B12X settings and the supplied memory budget until the baseline is measured. Explain every change to them.
- Run ./scripts/check.sh for code changes. GPU inference is a separate validation step; record actual evidence in docs/VALIDATION.md.

## README and documentation style

- Lead with the result, command or decision the reader needs. Prefer short paragraphs,
  compact tables and direct instructions; remove repeated background and obvious explanations.
- Remove generic caveats such as "small tests do not establish universal accuracy"
  and "X does not prove/establish/guarantee Y." State the measured result and its
  concrete scope instead: hardware, configuration, sample count and actual failures.
- Keep actionable limitations: unsupported modes, untested configurations, memory
  constraints, failed checks and requirements that change how someone should run
  or interpret the recipe. Never turn an untested setting into a supported claim.
- Keep commands, settings, source/checkpoint pins, attribution and evidence links
  accurate. Put detailed history and methodology in their dedicated documents and
  link to them instead of repeating them throughout the guides.
- Prefer substantial cuts over cosmetic rewrites. Explain a concept once, where
  readers need it, and avoid defensive qualifications about things they can readily infer.
