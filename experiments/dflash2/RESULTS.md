# First vLLM DFlash2 GPU screen

**MTP remains the faster default in the matched screen.** DFlash2 runs with the
pinned NVIDIA FP4 checkpoint after selecting Triton for draft attention. Target
attention remains B12X. TensorFold integration work is listed in
[its own folder](../../tensorfold/DFLASH2-PLAN.md).

## Matched performance

Two RTX PRO 6000 Blackwell Max-Q cards; pinned release-10152026 image; TP2/DCP2;
128K context, one sequence, 3 GiB KV/card, 98.5%; safetensors; identical frozen
8K/32K input IDs. Each cell has one excluded warmup and three measured repeats,
greedy requests and zero cached input tokens. MTP uses three draft tokens;
DFlash2 uses seven, BF16 draft weights and BF16 draft KV.

| Input / output tokens | Mode | Decode tok/s | Prefill tok/s | Client TTFT | End-to-end |
|---|---|---:|---:|---:|---:|
| 8,192 / 256 | MTP | 173.3 | 10,216 | 0.807 s | 2.280 s |
| 8,192 / 256 | DFlash2 | 152.8 | 10,353 | 0.797 s | 2.470 s |
| 32,768 / 256 | MTP | 171.5 | 10,228 | 3.215 s | 4.687 s |
| 32,768 / 256 | DFlash2 | 137.3 | 10,429 | 3.154 s | 5.004 s |
| 8,192 / 1,024 | MTP | 172.8 | 9,982 | 0.826 s | 6.745 s |
| 8,192 / 1,024 | DFlash2 | 154.5 | 10,129 | 0.813 s | 7.432 s |
| 32,768 / 1,024 | MTP | 173.6 | 10,051 | 3.272 s | 9.173 s |
| 32,768 / 1,024 | DFlash2 | 152.9 | 10,216 | 3.221 s | 9.861 s |

Values are medians. DFlash2 end-to-end time was about 7–10% longer. Both modes
produced three distinct greedy output hashes per cell.

Across the measured requests, DFlash2 accepted 27–29% of its seven-token proposals,
versus MTP's 52–53% of three-token proposals. Including the target's bonus token,
DFlash2 advanced about 2.89–3.03 tokens per draft round versus 2.56–2.58 for MTP.
That modest gain did not cover the larger draft/verification work in this setup.
Testing a shorter DFlash2 block is a next tuning candidate.

## Function and 128K capacity

DFlash2 passed readiness, text completion, JSON, tool-call and streaming checks.
A near-128K retrieval request processed **130,816 input tokens**, retrieved all
three markers and stopped normally after 47 output tokens, with zero cached input
tokens. TTFT was 14.512 s; complete request time was 14.642 s.

Minimum sampled free memory for DFlash2's timing/retrieval runs was
**5,757 / 5,912 MiB**. At the same 3 GiB KV budget, startup reported **194,699**
effective KV tokens for DFlash2 versus **351,085** for MTP. Source inspection
shows separate target/draft groups still share a block pool whose stride is sized
for the widest group; the draft's five BF16 layers can increase that stride.

Broader task accuracy, fixed-history probability comparison and cancellation/
rejection-edge tests remain pending.

## First failure and fix

The initial all-B12X launch loaded the target but rejected draft attention with
`block_size not supported` and `replicated DCP KV not supported`. This image's
regular B12X attention supports 64/128-token pages; our target recipe uses 256.
`TRITON_ATTN` explicitly supports both 256-token pages and replicated DCP KV.
Changing **only the draft backend** allowed startup, DFlash2 graph capture and
successful requests. No image or checkpoint patch was required.

## Larger-context targets

Both targets passed: 1,048,576 tokens × one request and 262,144 × four concurrent requests.
The initial 99% attempt failed before loading weights: CUDA reported
93.78/93.93 GiB free versus 94.11 GiB required. Both successful profiles use
98.5% and 7.125 GiB explicit KV/card. The fixed byte budget controls KV allocation.

At 98.5%, the 1M profile started and reported **1,167,222 effective KV tokens**.
The larger-window ladder passed:

| Requested window | Actual input tokens | Result | Minimum free MiB, GPU 0 / 1 |
|---|---:|---|---:|
| 524,288 | 524,032 | All three markers; normal stop; no cached input or preemption | 1,591 / 1,744 |
| 1,048,576 | 1,048,320 | All three markers; 41 output tokens; normal stop; no cached input or preemption | 1,591 / 1,744 |

Near-1M TTFT was **140.494 s**, with **140.499 s** end-to-end time. These capacity
cells used a 1,024 MiB stop floor; their observed minima also exceeded 1,536 MiB.
[The 1M profile](vllm-1m.env.example) is a working capacity candidate; broad
near-1M task accuracy is still pending.

The 262K/four-slot profile with BF16 draft KV started, but its planner reported
**706,817 effective tokens / 2.70 full windows**. Four full requests were not
submitted under that insufficient budget. Switching only the draft KV to FP8
raised planned capacity to **1,139,840 tokens / 4.35 full windows**. Draft weights
remain BF16 and target precision stays unchanged.

The FP8 draft-KV profile passed all four API checks and the sustained four-client
screen. Each client processed **245,760 input + 16,384 output = 262,144 tokens**.
All input IDs and output counts were verified; all twelve markers were retrieved;
there was no cross-client marker leakage, cached input or preemption. All four
token-stream intervals overlapped for **40.122 seconds**.

| Client | TTFT | End-to-end | Output tokens |
|---|---:|---:|---:|
| 0 | 170.735 s | 379.131 s | 16,384 |
| 1 | 28.520 s | 287.017 s | 16,384 |
| 2 | 242.669 s | 392.680 s | 16,384 |
| 3 | 98.909 s | 282.791 s | 16,384 |

The unchanged single-prefill policy staggered admission: up to three waited
initially, then all four ran together. Minimum free memory was **1,387 / 1,540 MiB**,
above this probe's 1,024 MiB floor; GPU 0 was below the earlier 1,536 MiB working
headroom target. This profile retains 98.5% and 7.125 GiB KV/card.

## Profiles and next work

- [1M × one](vllm-1m.env.example): BF16 draft KV; near-window retrieval passed.
- [262K × four](vllm-262k-x4.env.example): FP8 draft KV; sustained concurrency passed.
- [128K comparison](vllm-probe.env.example): BF16 draft KV, 3 GiB/card; matched timing settings.

Keep MTP as the normal service. Next tuning candidates are shorter DFlash2 drafts,
concurrent prefill scheduling, and cache-group sizing to improve memory headroom.
Run matched task accuracy and fixed-history probability checks for the FP8 draft-KV
profile before broader use. The existing NVIDIA checkpoint and container image
were retained throughout; TensorFold changes are documentation only.

[Portable GPU receipt](results/gpu-v1.json) ·
[Source review and draft checksum](results/feasibility-v1.json).

The original MTP container was restored with its exact prior ID, image and
262K/four-sequence command; readiness and all four API checks passed. The
experimental container is stopped. Final CPU checks passed **84 tests**.
