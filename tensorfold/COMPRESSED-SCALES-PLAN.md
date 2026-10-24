# Resident CSF32 scales on SM120: scope and validation plan

Status, 2026-10-24: **research and CPU storage accounting only**. No resident-CSF32 GPU loader/kernel has been implemented or qualified. This work targets TensorFold; the running vLLM service, image, loader, B12X settings and memory budget remain unchanged. The engine is TensorFold, not TensorFlow.

## Decision and motivation

Proceed first with compressed GPU scale storage plus bounded, reusable GPU expansion scratch, preserving TensorFold's existing numerical operations. Separate packed weight codes from scale storage so removing expanded scales actually releases VRAM. Consider decoding scales inside the expert kernel only after the simpler implementation has correctness and timing evidence.

The measured RTX tuning pass is complete for its tested candidates, not for every kernel or workload. The [PR discussion brief](PR1050-SM120-BRIEF.md) records adopted settings, retained settings, experimental source changes and their evidence. At 8K input, the [current shared benchmark](../docs/PERFORMANCE-CURRENT.md) shows vLLM/TensorFold MTP-on prefill at about 10,195/4,499 tok/s. We have no matched GPU trace establishing how much of that gap belongs to routed experts, dense projections, attention, communication or scheduling. Wider expert tiles improved isolated kernels without improving served prefill; this is evidence against selecting kernels by microbenchmarks alone.

PR #1050 supplies checkpoint compatibility, mixed-format routing and precision handling. Its host scale expansion is deliberate. It does not promise compressed GPU residency or the fastest SM120 prefill path. A storage optimization must preserve quality; it does not inherently improve accuracy or close the prefill gap.

## Source pins and live upstream findings

- Model: `nvidia/GLM-5.3-Flash-FP4-Local@76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, unchanged ModelOpt mixed checkpoint and `csf_32x2_v1` NVFP4 scales.
- TensorFold base `36415df4f76e4458e82d0fa729ecb28e287ebf38` plus [PR #1050](https://github.com/ashhart/TensorFold/pull/1050) head `9b23b6a2e7ff226ca61934189c4cd57bf2933980`; local merged tree `1d52effd182d9319d775d5787904c36783e532f1`. The optional mapped-embedding baseline is separately pinned in [tuning/README.md](tuning/README.md).
- Live GitHub inspection on 2026-10-24 found PR #1050 open, unmerged, at the same head; GitHub reported `mergeable:false` against main `bd264368ae7382672faef6cc99c2add8b24c5ec0`. Our qualified merge with v1.1.3 does not establish clean application to current main.
- Bounded searches of CSF/compressed-scale/SM120/GLM-prefill issues and PRs found [#1021](https://github.com/ashhart/TensorFold/pull/1021), [#1023](https://github.com/ashhart/TensorFold/pull/1023) and #1050 as the relevant host-codec/config/loader chain. [#904](https://github.com/ashhart/TensorFold/pull/904) concerns a different slab codec; [#906](https://github.com/ashhart/TensorFold/pull/906) is rank-local host loading, not a resident compressed-scale consumer.
- Upstream branch-name search for `csf` returned none. The contributor fork has `pr/fp4local-csf`; its [codec source](https://github.com/Bizuayeu/TensorFold/blob/pr/fp4local-csf/zig/src/core/nvfp4_csf32.zig) explicitly implements host decoding/encoding. Fork GLM branches also include prefill-overlap, ring-split, KDA and prompt-program work. Branch names alone do not establish qualification or applicability. No ready TensorFold GPU-resident CSF32 implementation was located in this bounded search; this is not an exhaustive audit of every branch.
- The format owner's [CSF scale-block specification](https://github.com/local-inference-lab/flashinfer/blob/e36134561b80174f29abec41f193f8dc61ce861e/docs/csf-scale-blocks.md), pinned to `e36134561b80174f29abec41f193f8dc61ce861e`, describes compressed resident payloads, caller-owned scale scratch, active-expert expansion and native packed output. This is a design reference, not proof of our deployed October 15 vLLM image's internals: the fetched documentation was updated October 22.

Use TensorFold's architecture and implement its contracts. Do not copy another engine's normalization, arithmetic or source into this patch by assumption. Reusing external code would require license/attribution review and upstream maintainer agreement. Preserve the existing PR author's credit. Before any eventual upstream submission, refresh coordination and the issue/ATC status; no comment, issue, claim or PR was published during this investigation.

## Why the current loader consumes extra memory

Paths below are relative to the pinned TensorFold source tree:

| Owner | Current behavior | Proposed responsibility |
|---|---|---|
| `zig/src/core/nvfp4_csf32.zig` | CPU codec and structural validation | Independent scale-byte oracle; reusable geometry/TP-slice tests |
| `zig/src/families/glm/cuda_formats.zig`, `csfLinear` | Decodes five CSF components into row-major E4M3 bytes | Admit a typed compressed view while retaining the legacy decoder |
| `zig/src/families/glm/cuda_weights.zig`, `nvfp4Experts` | Host expansion, TP slicing, packing, persistent upload | Stream rank-local compressed components and codes; retain plain MTP fallback |
| `zig/src/cuda/experts.zig`, `Nvfp4Experts` | Each 32×32 weight tile holds 128 code words plus 16 scale words | Separate immutable code storage and block-scale descriptor/scratch arguments |
| `zig/kernels/cuda/nvfp4_experts.cu` | Decode and prompt paths consume interleaved codes/scales | Fetch codes and expanded scratch separately; preserve compute/reduction order |
| `zig/src/families/glm/cuda_experts.zig`, `cuda_moe.zig`, `cuda_forward.zig` | Routing, sequential gate/up and down launches, forward/overlap streams | Expansion ordering, scratch ownership, fixed-storage replay and lifetime proof |
| `zig/tests/cuda/contract_tests.zig`, `zig/src/core/contract/registry.zig`, `zig/build/kernel_contract.zig` | Operator gates and compiled launch limits | Register new/replaced operators, controls and CUDA 13.3 resource records |

The current GPU tile uses 576 bytes: 512 bytes of FP4 weight codes and 64 bytes of expanded E4M3 scales. Leaving empty scale slots in this layout saves nothing. Codes-only global storage would use 128 u32 words rather than 144; the shared-memory compute layout can initially remain unchanged. Both decode loads and prompt asynchronous copies must honor the new global layout.

`Layer.up_scale` and `down_scale` currently address global FP32 calibration values, not the compressed block scales. Add explicit typed fields/ownership; do not overload those pointers. Keep weight codes, global scales, BF16 activations, FP32 accumulation and existing rounding/reduction order unchanged. The MTP experts use plain W4A16_NVFP4 and remain on the legacy path. MXFP8, FP8 and FP32 tensors are outside this storage change.

## Audited memory budget

The [read-only inventory](tuning/csf_inventory.py) examined safetensors headers and CSF page indexes for **36,288 matrices across 42 routed layers**. It does not load weight tensors, use GPUs or fully validate every compressed record. The [portable result](results/csf-residency-plan-audit-v1.json) pins model metadata and audit-client hashes.

Decoded scales total 19,025,362,944 bytes. Checkpoint compressed components total 10,321,055,136 bytes. Simply dividing their difference by two overstates savings: TP down-projections need row bases on both ranks. The audit counts actual per-rank exception records and duplicates these bases, costing an additional 23.625 MiB per rank relative to naive halving. Gross savings are approximately **4.030 GiB/rank**, before scratch and implementation overhead.

For one routed layer, expanded gate/up scales require 144 MiB/rank and down scales 72 MiB/rank. The baseline is the qualified mapped-embedding 512K profile, one slot, no retained prompt cache, sampled free memory 1,547/1,700 MiB. The measured allocation slope projects another 3.662 GiB/rank for 512K→1M.

| Proposed scratch strategy | Scratch per rank | Net scale saving, rank 0 | Projected free at 1M, rank 0 / rank 1 |
|---|---:|---:|---:|
| Separate reusable gate/up and down outputs | 216 MiB | 3.819 GiB | 1.668 / 1.817 GiB |
| Alias serial gate/up and down lifetimes | 144 MiB | 3.890 GiB | 1.738 / 1.888 GiB |
| Two complete layer scratch sets | 432 MiB | 3.608 GiB | 1.457 / 1.606 GiB |
| Copy entire interleaved code+scale layer into scratch | 1,944 MiB | 2.132 GiB | −0.020 / 0.130 GiB |

These are **projections, not measured candidate allocations**. They exclude descriptor tables, allocator padding, graph state, retained staging and future workload peaks. The first strategy leaves only about **172 MiB above a 1.5 GiB free-memory target** on rank 0. Replacing compact u16 prefixes with u32 entries for every block would alone add about 283.5 MiB/rank. Preserve compact indexing or account for the cost before implementation. A negative projected free value means even bare fit fails under these assumptions.

The audit assumes aligned TP cuts can rebuild compact rank-local pages without padding growth. Gate/up split rows; down splits scale columns. Preserve the codec's block/tile order while regrouping down-projection pages. Reindex origins/prefixes and maintain correct byte-versus-record units. GPU decode tests must prove this proposed representation, not just the size calculation.

Reproduce to a **new** output path (existing outputs are deliberately not overwritten):

```bash
python3 tensorfold/tuning/csf_inventory.py --output logs/csf-inventory-new.json
```

## Implementation stages and exit gates

1. **Freeze the baseline and allocation ledger.** Pin model, source, binaries and tuning settings. Measure physical allocations by owner and load/steady/request peaks; verify the slope at more than one context. Include mapped host RAM. Keep the normal image and vLLM service available for rollback. Use a separate experimental image and patch; no checkpoint conversion or new weights.
2. **Separate storage without compression first.** Add codes-only packing and separate expanded-scale pointers. Compare every packed code and scale byte against the legacy packer, then both prompt/decode outputs on identical inputs. Retain the exact MMA/fmaf/reduction sequence. This isolates layout mistakes before adding a codec.
3. **Add compressed residency and bounded GPU expansion.** CPU-validate and TP-slice/reindex one expert or bounded chunk at a time; CPU decode may remain a validation oracle during initial loading. Upload compressed parts without retaining a full expanded GPU model. Expand directly into TensorFold's native scale swizzle in the 144/72 MiB outputs. Release upload staging before KV allocation. Reject unsupported layouts explicitly.
4. **Prove ownership before reuse.** `cuda_moe.body` currently routes then launches gate/up and down on its supplied stream, which is promising for reuse. Nevertheless, trace all forward, overlap, MTP, cancellation and cache paths before aliasing buffers. One API slot does not prove one GPU stream. Start with separate up/down outputs; only use 144 MiB aliasing after an event/lifetime proof. Require fixed addresses and no replay-time allocations/readbacks; refresh scales before every dependent replay. Unsupported concurrent consumers must be rejected or receive budgeted independent scratch.
5. **Optimize expansion only with dependent timing.** Decode should expand the GPU route's active expert set; long prefill may justify all experts. Test empty, repeated and changing routes so stale scales cannot be consumed. Benchmark expansion plus expert consumers and whole requests. A fused in-kernel decoder is a later option if scratch/launch costs dominate, with explicit register/shared-memory and indexing costs; it is not the first implementation.
6. **Qualify capacity and performance.** Start at the known 512K profile, then try configured 640K, 768K, 896K and 1,048,576, subject to actual memory. Use near-window prompts with output allowance, sampled free memory and allocation peaks. Stop on insufficient headroom or incorrect output. Neither startup nor a configured 1M window proves near-1M inference or long-context accuracy.

## Validation contract

| Gate | Required evidence before adoption |
|---|---|
| Format and TP geometry | Independent goldens, malformed extents/dtypes/overflow, base bounds, prefix/page units, exception ordering/duplicates/bounds, odd-column codec fixtures, row/column cuts, projection fusion; supported runtime geometry explicit |
| Real checkpoint byte preservation | Compare all 36,288 matrices through both TP slices against CPU decoding and the old packed-scale layout; unchanged weight codes/global calibration; no requantization |
| GPU operator contract | Both cards; exact legacy output bytes, five-run repeatability, unchanged inputs, guards/aliasing, tile edges and served shapes, non-finite handling, FP32 arithmetic gates with planted failing controls, registered compiled limits |
| Stateful behavior | Prefill/chunk seams, decode, MTP verify/accept/reject, cache cold/warm/resume, cancellation/reset, route changes and scratch reuse; target logits and KV/recurrent/convolution state checks where affected |
| Quality | Repeat the frozen 256-item GSM8K/IFEval suite and fixed-history diagnostics (1,439 positions), compare hashes, per-item grades, target log probabilities and top-one margins; report every changed case |
| Timing | Shared cold short/2K/8K/32K/64K workloads, 256 output tokens, one request, MTP on/off, excluded warmup and ≥3 measured repeats; bracket small gains with returning baselines; record prefill, decode, client TTFT, expansion time and telemetry |
| Capacity | Allocation ledger including load peak, scratch, metadata, padding and pinned RAM; near-window retrieval at every admitted rung; ≥1.5 GiB measured free target on both cards, not just projected fit |
| Broader accuracy | Longer and more varied retrieval plus RULER or an equivalent long-context suite before broad accuracy claims; three synthetic markers and a short-context suite are insufficient |

Run the applicable upstream host, golden, fuzz/security and GPU contract suites, retain failures and skips, and list unrun architectures. New/replaced kernels require the upstream contract runner and launch registry, not only standalone tests. For storage-only changes, byte preservation is the intended contract; a changed reduction requires a separately reviewed numerical contract. Do not relax correctness/performance gates to make a capacity experiment appear ready. An observed speed cost can be recorded as an experimental tradeoff, but does not qualify a faster default.

## Separate SM120 prefill investigation

Use TensorFold's existing profiling spans (`TF_GLM_PROF`) and GPU tracing to separate routed expert compute, MXFP8/FP8 projections, KDA/MLA work, collective overlap and launch gaps at fixed 8K/32K/64K inputs. Confirm profiler overhead against an unprofiled control. Inspect achieved bandwidth, occupancy, register/shared-memory use and route distributions before selecting new tiles, split policies or fusion.

Compare like-for-like MTP settings and cache state; a matched vLLM trace is observational evidence, not an isolated kernel comparison. Cross-engine implementation differences must be isolated with same-input operator tests or controlled ablations before attributing causality. Do not import another engine's FP4 activation quantization or nondeterministic reduction into a storage-preserving patch. Attention/cache sharding, external drafting, precision changes and clock/power tuning are separate workstreams.

The next implementation deliverable is a small, reviewable storage-layout patch with byte-exact tests and an allocation ledger. Resident CSF32 expansion follows only after that gate. This document authorizes no public posting or service switch by itself.
