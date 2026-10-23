# Next course: TensorFold

The linked project is [ashhart/TensorFold](https://github.com/ashhart/TensorFold). This phase starts after the NVIDIA/vLLM baseline passes. No TensorFold code is installed, built or patched by this recipe yet.

The neighboring recipe uses the **Python 0.6.6 engine plus EXL3 patches**. Current upstream TensorFold uses a **native Zig engine**; its installation, rank control and checkpoint readers have changed. Copying the old Python recipe's flags or patch stack into a current native build is not a reproducible upgrade.

Evidence checked 2026-10-23:

- The [current README](https://github.com/ashhart/TensorFold#qualified-models) includes a GLM NVFP4 two-GB10 contributed result, with limited qualification.
- [Issue #1040](https://github.com/ashhart/TensorFold/issues/1040) records a complete NVFP4 checkpoint loading and serving at TP2 on two GB10s using native 1.1.0. It does not establish fit, accuracy or throughput on two 96 GB RTX PRO 6000 cards.
- [Release 1.1.3](https://github.com/ashhart/TensorFold/releases/tag/v1.1.3) is the latest release observed. That is a research observation, not a selected build pin or a qualification of this checkpoint on this machine.
- The [historical GLM recipe](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/glm-5.3-flash.md) mixes historical Python guidance with native references. Use release-specific source and capabilities to resolve any conflict.

Build and qualification plan:

1. Freeze the passing vLLM image ID, NVIDIA revision, test prompts and measurements in a baseline record.
2. Inspect the existing local `tensorfold-glm53:1.1.0-2x` image read-only for provenance. Choose a native release/commit deliberately; record the Zig compiler, CUDA toolkit, driver, dependencies and build flags. Do not infer compatibility from the tag alone.
3. Before modifying TensorFold, read its contributor instructions and recipe skill at the chosen revision. Build in a separate checkout/image so the vLLM service remains reproducible.
4. Verify the checkpoint reader accepts **this NVIDIA compressed-scale `csf_32x2_v1` format**, ModelOpt mixed precision, vision layout and MTP weights. Ordinary NVFP4 support alone is insufficient evidence. If the reader lacks support, document the gap before planning a conversion or patch.
5. Check the two-rank loader's memory footprint on these 96 GB cards. The GB10 report has a different memory budget. Derive context and concurrency from actual allocation; start with one stream and short context.
6. Determine rank launch and PCIe collectives from that release's native CLI/source. Verify peer access, preserve the same checkpoint bytes, and use a distinct container name, port and cache volume. Do not pass the old `TF_GLM_*` settings wholesale.
7. Run the same text, JSON, tools, streaming, reasoning, progressive context and repeated-prefix checks. Compare solo/concurrent and drafted/serial behavior where the engine supports those controls. Include vision only after explicit qualification.
8. Measure identical workloads and report errors, memory, TTFT and throughput alongside any numerical, feature or context differences. Keep the existing vLLM recipe available for rollback.

If the native reader needs another checkpoint or a conversion, treat that as a different model experiment and record the fidelity implications. The first comparison should aim to reuse the NVIDIA files already on disk.
