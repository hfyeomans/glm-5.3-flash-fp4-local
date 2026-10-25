# Compression and CSF scales: context capacity, not more experts

Research checked 2026-10-24. Compression can let a larger trained model, including more experts, fit into the same VRAM; for this GLM experiment the intended use of saved VRAM is **more context capacity with the same checkpoint**, not adding experts. The practical challenge is making the compressed representation inexpensive enough to use during inference. This guide records the compression discussion and adds primary-source research; it is explanatory documentation, not a new engine benchmark or a claim that the ongoing TensorFold experiment has passed its gates. For implementation status, consult [the validation record](VALIDATION.md) and [the compressed-scale plan](../tensorfold/COMPRESSED-SCALES-PLAN.md).

## Three different meanings of compression

| Technique | What it does | Quality implications |
|---|---|---|
| **Quantization**, such as FP4 | Represents weights with fewer bits | Can change accuracy; needs evaluation |
| **Lossless compression**, including CSF | Encodes existing values more compactly and reconstructs them exactly | Adds no numerical loss when implemented correctly |
| **MoE routing** | Executes only selected experts for each token | Saves computation relative to executing every expert; inactive experts still need storage somewhere |

MoE routing is conditional computation rather than a file codec. Quantization and lossless encoding can be combined: first obtain a quantized checkpoint, then encode those quantized values without further loss. Lossless recovery of an FP4 checkpoint does not recover the original higher-precision weights or undo quantization error.

Pruning, distillation and merging experts are other approaches, but they change the model or its parameters. They are not equivalent to storing the same checkpoint more efficiently.

## What CSF means for this checkpoint

The model is **`nvidia/GLM-5.3-Flash-FP4-Local`**, revision **`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`**. It is a mixed-precision checkpoint, not an entirely four-bit model. The routed experts include FP4 weight codes and compressed block scales. The [precision guide](PRECISION-AND-QUALITY.md) explains W4A16, BF16 activations and FP32 accumulation.

A quantized weight is interpreted using its code and scale factors. For example, one eight-bit scale per sixteen four-bit weights adds 0.5 bits per weight: 4.5 bits/weight before global scales and other metadata. This simple calculation illustrates why scales matter even when the weights are already compact; it is not the memory footprint of the whole model.

The checkpoint's `csf_32x2_v1` format compresses **scale bytes**. For NVFP4 it uses a row base, packed four-bit offsets and exact replacement bytes for exceptions. Its five components hold bases, codes, page origins, exception prefixes and exceptions. It leaves the FP4 weight nibbles and global calibration tensors unchanged. It does not reduce the number of experts or combine their learned functions. See the [pinned CSF format specification](https://github.com/local-inference-lab/flashinfer/blob/e36134561b80174f29abec41f193f8dc61ce861e/docs/csf-scale-blocks.md).

In a correctly implemented lossless path, the reconstructed scale bytes must equal the originals.

## Is this like gzip?

The resemblance is exact reconstruction from a smaller representation. The execution requirements differ. Conventional DEFLATE streams have decoding dependencies that are difficult to parallelize; NVIDIA's [GDeflate](https://docs.nvidia.com/cuda/nvcomp/gdeflate.html) rearranges the stream to expose 32-way parallelism. GPU decompression is therefore a real technique, not just a metaphor. It is not a drop-in CSF replacement.

| Where compression is used | What can improve | What does not follow automatically |
|---|---|---|
| Download or archive, expanded before loading | Disk space and transfer volume | Lower steady-state VRAM |
| CPU storage, selected data transferred to GPU | CPU capacity and transfer volume, depending on execution | Lower latency; transfer and decoding can dominate |
| GPU-resident compressed data, bounded expansion scratch | VRAM capacity and potentially memory traffic | Faster inference or unchanged end-to-end behavior |
| Decode inside the consuming GPU kernel | Potentially avoids a separate expanded buffer/write | Better performance; extra instructions and registers can reduce throughput |

This is why the TensorFold plan distinguishes compressed residency from loading a compressed checkpoint. Expanding everything once and retaining the expanded model removes the steady-state memory benefit. The format owner's implementation reference describes compressed resident scales, reusable expansion buffers and route-dependent expansion. Overlapping consumers need separate storage or explicit ordering.

## Would this let us fit more experts?

Yes, **when comparing trained models and compatible engines**. If expert storage falls, a fixed memory budget can accommodate more expert parameters, provided everything else also fits.

Scale encoding alone is bounded by the scales' share of expert storage: 0.5 of 4.5 bits per weight, about 11%. The audited CSF32 residency saving for this checkpoint is about 4.030 GiB/rank of roughly 80 GiB/rank in per-rank expert weight codes and scales — about 5% — which would hold roughly 1.05 times as many experts in the same expert-only budget, if added experts compress at the same ratio. That arithmetic ignores routing tables, non-expert layers, cache, scratch, alignment and communication. The [checkpoint-index audit](../tensorfold/results/csf-residency-plan-audit-v1.json) and the compressed-scale plan hold the detailed accounting.

There are two distinct outcomes:

- **Run a larger existing MoE checkpoint:** compression may make it fit, subject to its architecture and runtime requirements.
- **Add experts to this existing checkpoint:** available memory is only a prerequisite. Useful new experts and compatible routing require training or adaptation and evaluation. Changing an expert-count setting does not create learned capability.

Keeping top-k fixed while increasing the total expert pool need not increase expert matrix work proportionally, if expert dimensions remain fixed. Routing, communication, storage and load balance can still become more expensive.

The resident-scale implementation measured a **3,896 MiB (3.80 GiB) per-card free-memory gain** at 512K and 128K context, against a projected 3.819 GiB/rank net scale saving after 216 MiB of reusable scratch. Metadata, allocation padding and live-buffer peaks still matter; see [resident-scale results](../tensorfold/csf-resident/RESULTS.md).

## What the public research establishes

This is a bounded reading list, not an exhaustive survey. Paper-specific compression ratios should not be transferred to our model without measurement.

| Primary source | Relevant finding | How to interpret it here |
|---|---|---|
| [Deep Compression — Han, Mao and Dally, 2015/2016](https://arxiv.org/abs/1510.00149v5) | Combines pruning, trained quantization and Huffman coding | Neural-network entropy coding is not new. Its combined pipeline changes parameters; retained benchmark accuracy is not bitwise preservation of the original model. |
| [ZipNN — Hershcovitch et al., 2024; revised 2025](https://arxiv.org/abs/2411.05239v2) | Studies lossless model compression for storage and network transfer; reports substantial savings on evaluated models | Supports the “zip for model files” idea. It does not establish identical savings for already-packed FP4 weights or lower live VRAM in our engine. |
| [Lossless Compression of Neural Network Components — Heilper and Singer, 2025](https://arxiv.org/abs/2508.19263v1) | Investigates entropy coding of lower-precision formats, including FP8/FP4, and compressibility of KV caches | Low precision does not automatically eliminate all compressibility. This is a research lead, not a qualified GLM GPU implementation. |
| NVIDIA nvCOMP / GDeflate | Makes a DEFLATE-related stream more parallel for GPU decompression | General-purpose GPU compression exists. We would still need measured ratios, chunk access, scratch budgets and decoder costs for our tensors. |
| [DeepSeek-V3 technical report, 2024; revised 2025](https://arxiv.org/html/2412.19437v2) | Describes MoE, latent attention and FP8 mixed-precision training | Public large-model systems combine several memory/compute techniques. These are different mechanisms. |
| CSF format/execution specification, pinned revision above | Specifies exact scale encoding and GPU execution integration | Directly relevant to the checkpoint format; TensorFold still needs its own validated implementation and arithmetic contract. |

The historical and recent work supports a distinction: the broad compression idea is established, while obtaining useful **resident-memory savings with low decode overhead** remains a substantive systems problem. These sources do not substantiate a blanket claim that proprietary frontier models generally keep their experts gzip-compressed during inference.

## What deserves measurement before another compression experiment

The following are research questions, not changes enabled by this document:

1. **Which bytes remain redundant?** Measure weight codes, block scales, metadata and caches separately. A nearly uniform four-bit code stream has little room below four bits per symbol without exploiting additional structure; scale distributions can differ. Do not assume FP4 codes are either universally compressible or universally incompressible.
2. **What is the dependent cost?** Time decode plus its consumer and full requests, including cold prefill and MTP on/off. A fast standalone decompressor does not establish faster inference.
3. **Would a general codec actually help?** Compare CSF with any proposed alternative on the actual tensors and required access granularity before adding a new dependency or changing the checkpoint format.

The compressed-scale plan owns the remaining stages: net-resident accounting beyond the audit's projections, route-dependent expansion safety and end-to-end reconstruction validation.

A rough bandwidth-bound timing model is `time saved reading bytes − decoding and extra-memory-traffic cost`. Compute overlap and scheduling can change that balance. Compression may improve speed, trade speed for capacity, or be a net loss; the outcome must be measured on the target cards.

For the RTX qualification path and measured decode costs, see [the RTX optimization record](../tensorfold/PR1050-SM120-BRIEF.md).
