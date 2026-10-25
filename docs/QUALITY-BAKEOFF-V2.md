# Quality comparison after resident-scale and embedding changes

The new TensorFold run reproduced all 256 earlier TensorFold output-token hashes
and all grades. At 1,439 fixed-history positions, target log probabilities were
identical, with no top-1 flips.

| Metric | Samples | Archived vLLM | Earlier TensorFold | New TensorFold, GPU embeddings |
|---|---:|---:|---:|---:|
| GSM8K exact match | 128 | 119 (92.97%) | 120 (93.75%) | 120 (93.75%) |
| IFEval prompt strict | 128 | 112 (87.50%) | 113 (88.28%) | 113 (88.28%) |
| IFEval prompt loose | 128 | 114 (89.06%) | 114 (89.06%) | 114 (89.06%) |

Paired McNemar p=1.0; complete paired results and intervals are in the
[receipt](../tensorfold/results/quality-bakeoff-v2.json).
There were no output-cap truncations or unclosed reasoning blocks in either
TensorFold run. The archived vLLM and earlier native runs were reused.

The protocol remains `initial-v1-low-reasoning`: pinned checkpoint, frozen task
samples, exact rendered prompts, greedy generation and a 4,096-token output cap.
The native quality profile uses 16K configured context, one slot, GPU embeddings,
resident compressed expert scales and the immutable `csf-resident-v1` image.
Actual inputs were 30–784 tokens (median 418). See the separate
[long-context protocol](../quality/LONG-CONTEXT.md).

Evidence:

- [Output hashes and task grades](../tensorfold/results/gpu-embedding-quality-v2.json)
- [Fixed-history probability comparison](../tensorfold/results/gpu-embedding-probabilities-v2.json)
- Raw run: `logs/qualification-v2/gpu-quality/00-gpu-quality/quality`
- Frozen baseline: `quality/baselines/initial-v1.json`

Both engines use the same quantized checkpoint. TensorFold's routed expert path
is W4A16: 4-bit stored weights with BF16 activations. The supplied vLLM configuration
also enables a hybrid A4 prefill path when its calibration and scheduler conditions
are met. Compressed-scale storage preserves the reconstructed scale bytes.

The separate [long-context screen](LONG-CONTEXT-ACCURACY-V2.md) completed its
user-approved 22-case near-1M core: 21 fully correct and one frequency response
matching two of three words. A matched long-context engine comparison remains pending.
