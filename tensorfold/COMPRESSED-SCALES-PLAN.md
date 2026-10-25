# Resident CSF32 scales on SM120: scope and validation plan

Status, 2026-10-24: the isolated implementation passed byte, short-quality,
cache and near-1M checks; matched timing and corrected cancellation tests are
complete. [Results](csf-resident/RESULTS.md). The original design/projections below
explain the implementation sequence.

## Design decision

Keep compressed GPU scales with bounded reusable expansion scratch, preserving
TensorFold arithmetic. Separate weight codes from scales to release the expanded
storage. Direct decoding inside expert kernels is a later optimization candidate.

PR #1050 supplies checkpoint loading and host scale expansion. Our extension
adds compressed GPU residency. The [RTX brief](PR1050-SM120-BRIEF.md) records
adopted/rejected tuning; the archived [shared benchmark](../docs/PERFORMANCE-CURRENT.md)
measured 8K MTP-on prefill at 10,195/4,499 tok/s for vLLM/TensorFold. Wider expert
tiles improved isolated kernels but not served prefill. Matched cross-engine
traces are still needed to apportion the gap.

## Source pins and live upstream findings

- Model: `nvidia/GLM-5.3-Flash-FP4-Local@76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, unchanged ModelOpt mixed checkpoint and `csf_32x2_v1` NVFP4 scales.
- TensorFold base `36415df4f76e4458e82d0fa729ecb28e287ebf38` plus [PR #1050](https://github.com/ashhart/TensorFold/pull/1050) head `9b23b6a2e7ff226ca61934189c4cd57bf2933980`; local merged tree `1d52effd182d9319d775d5787904c36783e532f1`. The optional mapped-embedding baseline is separately pinned in [tuning/README.md](tuning/README.md).
- Live GitHub inspection on 2026-10-24 found PR #1050 open, unmerged, at the same head; GitHub reported `mergeable:false` against main `bd264368ae7382672faef6cc99c2add8b24c5ec0`. The qualified local merge uses v1.1.3.
- Bounded searches of CSF/compressed-scale/SM120/GLM-prefill issues and PRs found [#1021](https://github.com/ashhart/TensorFold/pull/1021), [#1023](https://github.com/ashhart/TensorFold/pull/1023) and #1050 as the relevant host-codec/config/loader chain. [#904](https://github.com/ashhart/TensorFold/pull/904) concerns a different slab codec; [#906](https://github.com/ashhart/TensorFold/pull/906) is rank-local host loading, not a resident compressed-scale consumer.
- Upstream branch-name search for `csf` returned none. The contributor fork has `pr/fp4local-csf`; its [codec source](https://github.com/Bizuayeu/TensorFold/blob/pr/fp4local-csf/zig/src/core/nvfp4_csf32.zig) explicitly implements host decoding/encoding. Fork GLM branches also include prefill-overlap, ring-split, KDA and prompt-program work. No ready TensorFold GPU-resident CSF32 implementation was found in those searches.
- The format owner's [CSF scale-block specification](https://github.com/local-inference-lab/flashinfer/blob/e36134561b80174f29abec41f193f8dc61ce861e/docs/csf-scale-blocks.md), pinned to `e36134561b80174f29abec41f193f8dc61ce861e`, describes compressed resident payloads, caller-owned scale scratch, active-expert expansion and native packed output. The design reference was updated October 22, after our October 15 vLLM image.

Preserve TensorFold's numerical contracts and the PR author's credit. Reusing
external code needs license/attribution review; a future upstream submission needs
coordination and refreshed issue/ATC status. This investigation remained local.

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

The [read-only inventory](tuning/csf_inventory.py) examined safetensors headers and CSF page indexes for **36,288 matrices across 42 routed layers**. It reads metadata on the CPU; full byte validation follows during implementation. The [portable result](results/csf-residency-plan-audit-v1.json) pins model metadata and audit-client hashes.

Decoded scales total 19,025,362,944 bytes. Checkpoint compressed components total 10,321,055,136 bytes. Simply dividing their difference by two overstates savings: TP down-projections need row bases on both ranks. The audit counts actual per-rank exception records and duplicates these bases, costing an additional 23.625 MiB per rank relative to naive halving. Gross savings are approximately **4.030 GiB/rank**, before scratch and implementation overhead.

For one routed layer, expanded gate/up scales require 144 MiB/rank and down scales 72 MiB/rank. The baseline is the qualified mapped-embedding 512K profile, one slot, no retained prompt cache, sampled free memory 1,547/1,700 MiB. The measured allocation slope projects another 3.662 GiB/rank for 512K→1M.

| Proposed scratch strategy | Scratch per rank | Net scale saving, rank 0 | Projected free at 1M, rank 0 / rank 1 |
|---|---:|---:|---:|
| Separate reusable gate/up and down outputs | 216 MiB | 3.819 GiB | 1.668 / 1.817 GiB |
| Alias serial gate/up and down lifetimes | 144 MiB | 3.890 GiB | 1.738 / 1.888 GiB |
| Two complete layer scratch sets | 432 MiB | 3.608 GiB | 1.457 / 1.606 GiB |
| Copy entire interleaved code+scale layer into scratch | 1,944 MiB | 2.132 GiB | −0.020 / 0.130 GiB |

These **projections** exclude descriptors, allocator padding, graph state, retained staging and workload peaks. The first strategy leaves only about **172 MiB above a 1.5 GiB free-memory target** on rank 0. Replacing compact u16 prefixes with u32 entries for every block would alone add about 283.5 MiB/rank. Preserve compact indexing or account for the cost before implementation.

The audit assumes aligned TP cuts can rebuild compact rank-local pages without padding growth. Gate/up split rows; down splits scale columns. Preserve the codec's block/tile order while regrouping down-projection pages. Reindex origins/prefixes and maintain correct byte-versus-record units. Validate the representation with GPU decode tests.

Reproduce to a **new** output path (existing outputs are deliberately not overwritten):

```bash
python3 tensorfold/tuning/csf_inventory.py --output logs/csf-inventory-new.json
```

## Implementation stages and exit gates

1. **Freeze the baseline and allocation ledger.** Pin model, source, binaries and tuning settings. Measure physical allocations by owner and load/steady/request peaks; verify the slope at more than one context. Include mapped host RAM. Keep the normal image and vLLM service available for rollback. Use a separate experimental image and patch; no checkpoint conversion or new weights.
2. **Separate storage without compression first.** Add codes-only packing and separate expanded-scale pointers. Compare every packed code and scale byte against the legacy packer, then both prompt/decode outputs on identical inputs. Retain the exact MMA/fmaf/reduction sequence. This isolates layout mistakes before adding a codec.
3. **Add compressed residency and bounded GPU expansion.** CPU-validate and TP-slice/reindex one expert or bounded chunk at a time; CPU decode may remain a validation oracle during initial loading. Upload compressed parts without retaining a full expanded GPU model. Expand directly into TensorFold's native scale swizzle in the 144/72 MiB outputs. Release upload staging before KV allocation. Reject unsupported layouts explicitly.
4. **Validate ownership before reuse.** Trace forward, overlap, MTP, cancellation and cache consumers. Start with separate up/down outputs; 144 MiB aliasing requires verified event/lifetime ordering. Require fixed addresses, no replay-time allocations/readbacks and fresh scales before replay. Reject unsupported concurrent consumers or budget independent scratch.
5. **Optimize expansion only with dependent timing.** Decode should expand the GPU route's active expert set; long prefill may justify all experts. Test empty, repeated and changing routes so stale scales cannot be consumed. Benchmark expansion plus expert consumers and whole requests. A fused in-kernel decoder is a later option if scratch/launch costs dominate, with explicit register/shared-memory and indexing costs; it is not the first implementation.
6. **Qualify capacity and performance.** Start at the known 512K profile, then try configured 640K, 768K, 896K and 1,048,576, subject to actual memory. Use near-window prompts with output allowance, sampled free memory and allocation peaks. Stop on insufficient headroom or incorrect output.

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
| Longer-task accuracy | Varied retrieval plus RULER or an equivalent suite, scored by actual prompt length |

Run the applicable upstream host, golden, fuzz/security and GPU contract suites, retain failures and skips, and list unrun architectures. New/replaced kernels require the upstream contract runner and launch registry, not only standalone tests. For storage-only changes, byte preservation is the intended contract; a changed reduction requires a separately reviewed numerical contract. Retain failed gates and measured speed costs alongside capacity gains.

## Separate SM120 prefill investigation

Use TensorFold's existing profiling spans (`TF_GLM_PROF`) and GPU tracing to separate routed expert compute, MXFP8/FP8 projections, KDA/MLA work, collective overlap and launch gaps at fixed 8K/32K/64K inputs. Confirm profiler overhead against an unprofiled control. Inspect achieved bandwidth, occupancy, register/shared-memory use and route distributions before selecting new tiles, split policies or fusion.

Match MTP and cache settings; use same-input operator tests or controlled ablations to isolate cross-engine differences. Keep activation quantization, changed reductions, attention/cache sharding, external drafting and clock/power tuning separate from storage work.

The planned sequence is a storage-layout patch with byte-exact tests and an allocation ledger, followed by resident CSF32 expansion.

## Execution adjustment: context ladder allocation

After the measured 512K rung, the implementation harness can allocate the final 1M
pool once, then test 640K, 768K, 896K and near-1M prompts serially. This avoids three
extra checkpoint reloads and tests each request against the tighter final memory
budget. Intermediate rungs share that final allocation. The
harness checks the 1.5 GiB sampled-free floor after smoke and each request pair and
stops on failure. 512K and 1M provide separate allocation measurements.
