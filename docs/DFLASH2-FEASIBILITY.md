# DFlash2: vLLM first, TensorFold deferred

**Decision:** qualify DFlash2 in the pinned vLLM image first. Keep MTP as the default.
The selector is implemented on `research/dflash2-feasibility`. GPU tests passed
API checks, near-1M retrieval and four concurrent 262K windows; MTP remains
faster in matched short timing cells.
See [measured results](../experiments/dflash2/RESULTS.md).

## Which draft and which license?

The draft is **Inco AI's [GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/bf582e4eacc1810f76656d1811693ff6c6737d2a/README.md)**,
published for `zai-org/GLM-5.3-Flash`. We will pair it with the existing NVIDIA
FP4 target, revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, preserving CSF32 scales.
Its model card currently says checkpoint benchmarks are being remeasured.

| Mode | Draft component | License distinction |
|---|---|---|
| `mtp` | Target checkpoint's built-in head; three draft tokens | Target model terms continue to apply |
| `dflash2` | Separate Inco AI checkpoint; seven draft tokens by default | CC BY-NC-ND 4.0; our stated use is non-commercial research/evaluation |
| `none` | Target-only decoding | Target model terms continue to apply |

MTP is a drafting method, not a license. The [DFlash reference code is MIT](https://github.com/z-lab/dflash/blob/main/LICENSE),
while this draft's weights carry separate terms. [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
requires attribution and non-commercial use and restricts sharing adapted material.
Private experimentation and redistribution are distinct; ND is not a blanket ban
on private runtime changes. This recipe references the official checkpoint without
bundling weights. Switching to MTP removes the external draft from the launch;
it does not change the NVIDIA target's [terms](PROVENANCE.md).

[DFlash2](https://inco.ai/blog/dflash2/) adds two-tap dynamic convolution and a
candidate-path selector to parallel block drafting. The pinned vLLM API still uses
`method: "dflash"`; the `DFlash2DraftModel` architecture selects its dedicated
`DFlash2Speculator` implementation.

## Pinned ingredients

[Draft lock](../config/dflash2-lock.json): revision
`bf582e4eacc1810f76656d1811693ff6c6737d2a`, 1,171,080,448 BF16 parameters,
**2.18 GiB tensor payload**. The existing local weights passed the full SHA256
check; no weights were downloaded. The configuration has five layers, taps
`[5,14,24,33,42]`, block size eight, a top-16/rank-256 selector and a 2,048-token
sliding window. That sliding window belongs to the drafter; target context remains
independently configured. See the [pinned config](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/bf582e4eacc1810f76656d1811693ff6c6737d2a/config.json).

The vLLM image remains `release-10152026`, digest
`sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29`.
Its copied source has dedicated DFlash2 model/selector code, GLM auxiliary-state
capture that completes deferred mHC post-processing and separate DCP-replicated
draft KV grouping. The first GPU launch rejected regular B12X draft attention
for its 256-token pages and replicated DCP KV. The draft now uses `TRITON_ATTN`;
target attention stays B12X. These are encouraging integration
points; the FP4 target's shared head, host embeddings, graph capture and rejection
rollback still need execution tests on our SM120 cards.

Upstream [vLLM #56983](https://github.com/vllm-project/vllm/pull/56983) merged the
auxiliary-capture scope. Its earlier end-to-end description used different hardware
and weights. [#55682](https://github.com/vllm-project/vllm/pull/55682) remains open;
[#55423](https://github.com/vllm-project/vllm/pull/55423) closed unmerged. The custom
image already contains draft-cache handling beyond those older reports. Source
hashes and inspected revisions are in the [review receipt](../experiments/dflash2/results/feasibility-v1.json).

## Switch and first probe

Set **`DRAFTER=mtp|dflash2|none`** in `.env` or the exported environment. Explicit
`DRAFTER` wins over legacy `MTP_ENABLED`; when unset, `MTP_ENABLED=0` still selects
plain decoding. Restart applies a selection. Safe previews:

```bash
DRAFTER=mtp ./start.sh --dry-run
DRAFTER=dflash2 ./start.sh --dry-run
DRAFTER=none ./start.sh --dry-run
python3 scripts/draft_model.py --verify
```

The draft defaults to its pinned snapshot under `HF_CACHE_ROOT/hub` and mounts
read-only. `DFLASH2_MODEL_DIR` can point to a complete copy of that same checkpoint.
Startup validates config, tensor header and file size before any container stop;
`--verify` additionally hashes the full payload. `DFLASH2_NUM_SPECULATIVE_TOKENS`
accepts 1–7. `DFLASH2_KV_CACHE_DTYPE=auto|fp8` independently selects BF16 or FP8
draft-cache storage; the default is `auto`. Target memory, kernels and bootstrap settings stay unchanged by the
selector. New launch receipts identify the selected drafter explicitly.

Use [the candidate probe settings](../experiments/dflash2/vllm-probe.env.example)
for the first GPU run: **128K, one sequence, 3 GiB KV per GPU**, with a separate
container name, port and runtime volume. In the research worktree:

```bash
# Create probe settings only if this worktree has no existing .env.
test -f .env || cp experiments/dflash2/vllm-probe.env.example .env
./start.sh --dry-run
# After the serving GPUs are available:
./doctor.sh
./start.sh
```

This explicitly reduces the 7.125 GiB baseline KV budget by **4.125 GiB/card** to
leave room for draft weights and scratch. The 98.5% fraction, B12X environment,
TP2/DCP2 and target FP4 format remain intact. Draft parameters are partly shared or
replicated; 2.18 GiB divided by two is not a reliable peak allocation estimate.
Use the same probe budget for MTP, plain decoding and DFlash2 comparisons.

## Adversarial review and qualification gates

| Risk | Required evidence |
|---|---|
| More drafts may lose to MTP3 after selector/verification/PCIe work | Compare accepted tokens, decode throughput and complete request latency against MTP3, not only plain decoding |
| Auxiliary capture/draft prefill can raise TTFT | Record cold/warm TTFT and prefill separately; DFlash2 primarily targets decode |
| Draft KV could be expanded or grouped incorrectly | Record actual cache geometry, DCP replication, peak memory and sampled free memory; require at least 1.5 GiB free/card for a working profile |
| Wrong target taps, head precision or rollback can change outputs | Check greedy divergence, fixed-history probabilities, rejected drafts, cache resume, cancellation, JSON and tools before speed claims |
| Near-1M capacity already has little slack | Start at 128K; ladder context only after fit/correctness pass, including a separate long-context quality screen |

First confirm logs select **DFlash2Speculator**, its selector executes, and the
server uses the pinned target/draft. Then run matched 8K/32K prompts with 256/1,024
output tokens, one warmup and three measured repeats per mode. Freeze input token
IDs, sampling settings and prefix-cache state; record TTFT, prefill, decode,
end-to-end latency, acceptance and memory. For sampling, compare scored outcomes
and probabilities rather than requiring identical random text across algorithms.
A useful promotion target is at least 10% lower end-to-end latency on decode-heavy
workloads, with any TTFT regression above 5% called out as a tradeoff.

The existing MTP comparison and concurrency harnesses now reject external drafters.
The [experimental comparison client](../experiments/dflash2/bench.py) records
DFlash2 explicitly, including container/endpoint binding and draft identity.

## Why TensorFold comes later

The current resident-CSF engine explicitly rejects an external drafter with
`UnsupportedCsfResidentMode`. Existing native DFlash2 support also loads its draft
head through `e.w.mtp`; switching algorithms cannot be assumed to free that whole
allocation. Its TP2 implementation partly replicates weights and scratch.

A separate integration needs scratch-lifetime analysis, eight-row target
verification, mHC tap/head checks, rejection rollback, cancellation and numerical
validation before lifting that guard. The adjacent EXL3 recipe uses another target
format and engine path; its patches and speed results need independent evaluation.

Measured resident-CSF free memory is 4,177/4,330 MiB at 512K with GPU embeddings,
and 1,693/1,846 MiB at 1M with mapped embeddings. The latter leaves only 157 MiB
above the headroom floor. Start any native exploration at short context, retaining
GPU embeddings below the 1M profile. No TensorFold source or profile was changed
in this review. [Capacity evidence](../tensorfold/csf-resident/RESULTS.md).

Both capacity targets passed: **262,144 × four concurrent requests** with FP8
draft KV and **1,048,576 × one request** with BF16 draft KV. The 99% attempt
failed CUDA startup's free-memory gate; both successful profiles use 98.5% with
7.125 GiB explicit KV allocation per GPU. See the measured results above.
