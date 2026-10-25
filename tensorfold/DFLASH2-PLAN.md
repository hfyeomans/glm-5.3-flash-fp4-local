# DFlash2 integration work

**Status:** blocked by the resident-CSF engine's external-drafter guard. The work
below is required before adding a DFlash2 profile; existing MTP profiles stay unchanged.

Use `incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a`
with the existing NVIDIA target revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` and original CSF32 scales.
[Draft identity, license and cross-engine review](../docs/DFLASH2-FEASIBILITY.md).

## Required implementation

1. **Resident-scale compatibility.** `families/glm/cuda_engine.zig` currently
   rejects `scaleLayout()==2` with `o.drafter`. Audit resident-scale scratch
   ownership and lifetimes across target prefill, eight-row verification and
   draft execution before lifting `UnsupportedCsfResidentMode`.
2. **Target feature/head integration.** Check `cuda_taps.zig`, `dflash_load.zig`
   and `dflash_block.zig` against PR #1050's projection layout and FP32 norms.
   Verify the five mHC taps, embedding sharing, target vocabulary and draft-head
   math against the pinned vLLM path. The native loader requires
   `e.w.mtp.?.draft`; an external draft still needs that component allocated.
3. **Memory accounting.** Measure weights, replicated attention, sharded MLP,
   draft ring, selector scratch, graph/workspace and loading peaks per GPU.
   Start with original BF16 draft weights; evaluate runtime draft quantization
   separately. Preserve the NVIDIA target's precision throughout.
4. **Serving selector.** Add an explicit `mtp|dflash2|none` mode with pinned
   read-only draft mounting, metadata validation and launch receipts. Keep MTP
   the default. Build and pin a separate SM120 image for the implementation.
5. **Four streams.** After the single-slot path passes, address
   `CsfResidentSingleSlot` with per-slot draft/recurrent scratch, cache ownership
   and cancellation cleanup. Test isolation before enabling a 262K/four-slot profile.

## Qualification sequence

- Run native numerical checks for the loader, two-tap convolution, candidate
  selector and multi-row target verifier. Retain individual failure evidence.
- Test target-only/MTP/DFlash2 greedy outputs and fixed-history probabilities;
  cover accepted and rejected drafts, mixed acceptance lengths, cache resume,
  cancellation, JSON and tool calls. Evaluate sampling separately.
- Measure matched 8K/32K prefill, TTFT, decode and end-to-end time against both
  TensorFold MTP and qualified vLLM results, including draft acceptance and
  per-card memory telemetry.
- Start at short context with GPU embeddings. Then qualify 512K with GPU
  embeddings before attempting the optional 1M mapped-embedding profile.
  Require at least 1,536 MiB sampled free memory per GPU.

Current measured free memory is 4,177/4,330 MiB at 512K and 1,693/1,846 MiB at 1M.
The 1M profile has only 157 MiB above that floor on its limiting card; an external
draft requires additional savings or a smaller context. Multi-slot work remains
separate from this single-slot integration.

[Resident-scale results](csf-resident/RESULTS.md) ·
[PR #1050 / RTX adaptation history](PR1050-SM120-BRIEF.md).

Capacity targets for later DFlash2 work are four concurrent 262,144-token windows
and one 1,048,576-token window. Track TensorFold's native reserve and KV/draft
allocations in bytes; vLLM's memory-fraction setting is separate.

The [vLLM GPU reference](../experiments/dflash2/RESULTS.md) now includes near-1M
retrieval and four concurrent 262K windows. Its FP8 draft-KV change addresses
vLLM's shared block pool; reassess TensorFold's separate draft-ring allocation
rather than copying that setting directly.
