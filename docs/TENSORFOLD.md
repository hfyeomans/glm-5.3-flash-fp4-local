# TensorFold experiment

The separate [native TensorFold recipe](../tensorfold/README.md) successfully builds release **1.1.3**, commit `36415df4f76e4458e82d0fa729ecb28e287ebf38`, for SM120. **Its GLM CUDA loader cannot directly load the pinned NVIDIA FP4 Local checkpoint.** A CPU probe of the exact reader fails with `MissingTensor` for a plain `weight_scale`; the checkpoint stores CSF32 components. Its FP8 projections also conflict with the loader's BF16/F32 assumptions.

Use `./tensorfold/build.sh` and `./tensorfold/inspect.sh` without GPU allocation. Reproduce the incompatibility with `./tensorfold/compatibility.sh` and `./tensorfold/probe-loader.sh`. Startup refuses this mismatch before allocating GPUs. Keep the measured vLLM service running; changing memory settings will not fix a missing loader path.

The native image, container, port, runtime volume and configuration file are separate. No upstream source patches, checkpoint conversion or replacement weights were introduced. See [VALIDATION.md](VALIDATION.md) for actual results and the [recipe](../tensorfold/README.md) for pins, source evidence and the future qualification sequence after native loader implementation.

The older neighboring recipe and local `tensorfold-glm53:1.1.0-2x` image use Python 0.6.6 plus EXL3 patches. They are not used by this native build.
