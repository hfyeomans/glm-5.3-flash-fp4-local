# TensorFold experiment

The separate [TensorFold recipe](../tensorfold/README.md) now targets **v1.1.3 plus open PR #1050** at `9b23b6a2e7ff226ca61934189c4cd57bf2933980`, compiled for RTX PRO 6000 Blackwell SM120. The clean merged tree is pinned as `1d52effd182d9319d775d5787904c36783e532f1`.

Plain v1.1.3's GLM CUDA loader cannot load the exact NVIDIA FP4 Local checkpoint; its old-reader failure remains recorded in [VALIDATION.md](VALIDATION.md). The upstream PR connects CSF32 scale decoding, mixed FP8 projections, FP32 norms and MTP layout support to the GLM engine. The user authorized testing that existing work on the local cards.

The PR image, port, cache and configuration are isolated from vLLM. It reuses the exact NVIDIA checkpoint unchanged. The [recipe](../tensorfold/README.md) documents build pins, branch research, launch/rollback commands and qualification limits. The local SM120 test passed full loading, all four API checks, draft/queue consistency and retrieval through 7,481 prompt tokens at an 8K/one-slot configuration. The same short client workload measured 152.11 tokens/s versus vLLM’s 184.05. Full evidence and untested limits are recorded in VALIDATION.md; the experiment does not establish performance parity or long-context qualification.


Later isolated SM120 tuning and the [resident-CSF32 experiment](../tensorfold/csf-resident/RESULTS.md)
now pass a 1,048,311-token prompt in a 1M window with MTP on/off and identical
three-marker retrieval outputs. Minimum sampled free memory is 1,693/1,846 MiB;
all 256 short-quality hashes and 1,439 sampled target log probabilities remain
unchanged. This uses a separate image, mapped embedding, one slot and no retained
prompt cache. The original normal image and its limits above are unchanged.
