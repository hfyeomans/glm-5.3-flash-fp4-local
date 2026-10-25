# DFlash2 feasibility and remaining work

**Keep MTP as the default.** DFlash2 runs in the pinned vLLM image and passed
near-1M retrieval and four concurrent 262K windows. MTP was 7–10% faster
end-to-end in matched 8K/32K timing cells. TensorFold still needs engine work.
[Measured results](../experiments/dflash2/RESULTS.md) ·
[Settings and switching modes](../README.md#dflash2-settings-and-profiles).

## Draft identity and terms

The draft is **Inco AI's [GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/bf582e4eacc1810f76656d1811693ff6c6737d2a/README.md)**,
published for `zai-org/GLM-5.3-Flash`. We tested it with the original NVIDIA FP4
checkpoint and its CSF32 compressed scales.

| Ingredient | Pin |
|---|---|
| NVIDIA target revision | `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` |
| DFlash2 draft revision | `bf582e4eacc1810f76656d1811693ff6c6737d2a` |
| vLLM image | `ghcr.io/local-inference-lab/vllm:release-10152026` |
| Image digest | `sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29` |

The [draft lock](../config/dflash2-lock.json) records 1,171,080,448 BF16 parameters
and 2.18 GiB of tensor payload. The cached weights passed full SHA256 verification;
no weights were downloaded. The [configuration](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/bf582e4eacc1810f76656d1811693ff6c6737d2a/config.json)
has five layers, taps `[5,14,24,33,42]`, block size eight, a top-16/rank-256 selector
and a 2,048-token draft sliding window. Target context is configured separately.

The draft weights use [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/):
non-commercial use, attribution and restrictions on distributing adapted material.
Our use is research/evaluation; the recipe references the official weights without
bundling them. The [DFlash reference code is MIT](https://github.com/z-lab/dflash/blob/main/LICENSE).
MTP uses the target's built-in head, and the NVIDIA [target terms](PROVENANCE.md)
apply in every mode.

[DFlash2](https://inco.ai/blog/dflash2/) adds two-tap dynamic convolution and a
candidate-path selector to parallel block drafting. vLLM still accepts
`method: "dflash"`; the `DFlash2DraftModel` architecture selects
`DFlash2Speculator`.

## Implementation findings

The pinned image provides GLM auxiliary-state capture that completes deferred mHC
post-processing, dedicated DFlash2 model/selector code and DCP-replicated draft
cache grouping. Graph capture, API requests and retrieval ran on our SM120 cards.

The first launch rejected regular B12X draft attention because it lacks support
for 256-token pages and replicated DCP KV. Selecting `TRITON_ATTN` for the draft
fixed startup. Target attention, image bootstrap and B12X environment stayed intact.
No engine image was patched.

Target and draft groups still share a pool sized for the widest group. BF16 draft
KV limited the four-slot profile to 2.70 full 262K windows. FP8 draft KV raised
planned capacity to 4.35; four sustained requests then passed. Draft weights and
target precision were unchanged. The one-request 1M profile passed with BF16
draft KV. Both use 98.5% and 7.125 GiB explicit KV per GPU; 99% failed CUDA's
startup free-memory gate.

The upstream review recorded [vLLM #56983](https://github.com/vllm-project/vllm/pull/56983)
as merged with auxiliary capture scope, [#55682](https://github.com/vllm-project/vllm/pull/55682)
as open, and [#55423](https://github.com/vllm-project/vllm/pull/55423) as closed
unmerged. The custom image includes draft-cache handling beyond those older
reports. Exact revisions and source hashes are in the
[review receipt](../experiments/dflash2/results/feasibility-v1.json).

## Next qualification and tuning

| Work | Evidence needed |
|---|---|
| Task accuracy | Matched target-only, MTP and DFlash2 scores; fixed-history probabilities, including FP8 draft KV |
| State handling | Rejected-draft, cache-resume and cancellation tests; JSON/tools passed the initial API screen |
| Decode tuning | Compare shorter drafts against seven tokens using identical inputs, sampling and memory settings |
| Concurrent latency | Measure prefill scheduling changes against the recorded four-client TTFTs |
| Memory margin | Improve the four-stream profile's 1,387 MiB GPU-0 minimum toward the 1,536 MiB working target |

The [timing client](../experiments/dflash2/bench.py) records drafter identity and
container/endpoint binding. The capacity harness requires `--drafter dflash2`
for experimental runs; its default remains the MTP baseline. Preserve raw results
and record acceptance, prefill, TTFT, decode, end-to-end time and memory together.

## TensorFold

The resident-CSF path rejects an external drafter with `UnsupportedCsfResidentMode`
and additional slots with `CsfResidentSingleSlot`. Native DFlash2 also depends on
`e.w.mtp.?.draft`; account for that head when budgeting memory. Implementation,
SM120 validation and four-stream requirements are in the
[TensorFold work plan](../tensorfold/DFLASH2-PLAN.md).
