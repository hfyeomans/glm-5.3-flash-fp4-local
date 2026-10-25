# vLLM concurrent-context results

Measured 2026-10-25 on two RTX PRO 6000 Blackwell Max-Q cards.
**Two full 1M requests completed through queueing, one at a time.** The smaller
sustained control generated concurrently.

The unchanged baseline used the pinned NVIDIA checkpoint, release-10152026,
MTP3, eight slots, a 1,048,576-token per-request limit, GPU fraction 0.985 and
7,650,410,496 bytes of FP8 KV per GPU. Startup reported **1,844,162 shared KV
tokens** (1.76 full 1M windows).

## Measured cells

All inputs were cold and independent, with three retrieval markers per client.
Each cell ran once. TTFT includes queueing and prompt processing; values below
are ordered by first response, not client ID.

| Workload | Clients | Input tokens/client | Output budget/client | TTFT, seconds | Decode overlap |
|---|---:|---:|---:|---|---:|
| Near-1M control | 1 | 1,048,320 | 256 | 145.74 | — |
| 262K pair | 2 | 261,888 | 256 | 31.77 / 64.02 | 0 s |
| 512K pair | 2 | 524,032 | 256 | 68.39 / 136.65 | 0 s |
| 768K pair | 2 | 786,176 | 256 | 106.62 / 212.66 | 0 s |
| Near-1M pair | 2 | 1,048,320 | 256 | 147.29 / 293.95 | 0 s |
| Sustained small control | 2 | 7,168 | 1,024 forced | 0.75 / 2.18 | **7.20 s** |
| Sustained full-1M pair | 2 | 1,032,192 | 16,384 forced | 144.02 / 363.70 | **0 s** |

All nine ordinary long requests retrieved their markers and stopped normally.
Both sustained full-size clients also retrieved all markers and emitted exactly
16,384 tokens, reaching **1,048,576 total tokens each**. Their returned prompt IDs
matched the submitted IDs, and streamed output-ID counts matched usage. No
cross-client markers appeared. The forced continuation was used to hold context
active; only its initial retrieval was scored.

The sustained full-size pair completed at **219.16 and 438.59 seconds** from
submission. Telemetry showed at most **one running and one waiting** request;
preemptions remained zero. The small sustained pair reached two running requests
and overlapping token streams under the same server settings.

## Why the large requests queued

The pinned scheduler defaults `scheduler_reserve_full_isl=True`. Waiting admission
checks that the full input fits in the available KV pool; otherwise allocation
returns no slot and the request waits. The live waiting reason was `capacity`.
The two sustained inputs total 2,064,384 tokens, exceeding the reported shared
pool before output growth. No admission setting was changed.

Keep the distinction between **eight active-sequence slots**, the **1M limit per
request**, and the **1.84M shared cache**. Increasing sequence slots alone adds no
cache capacity. An agent with a 1M limit can run alongside others while their
actual contexts fit; two independently filled 1M contexts queued in this test.

## Capacity options

Scaling the reported pool gives these **planning estimates** for the current TP2
layout. Candidate budgets still need startup and concurrent-load measurement.

| Full windows | Estimated KV budget per GPU | Increase over current 7.125 GiB |
|---|---:|---:|
| Two × 1M | 8.102 GiB | 0.977 GiB |
| Four × 1M | 16.205 GiB | 9.080 GiB |

The observed 903/1,056 MiB free leaves almost no margin for even the two-window
increase. Options are freeing other engine allocations, distributing the model
across more GPU memory, or running separate instances on separate GPU pairs.
A larger tensor-parallel group needs qualification with this custom B12X image;
the current recipe targets TP2. Two separately qualified TP2 instances would
provide independent cache pools and endpoints. GPU/host resources and concurrent
operation would still need checking on that expanded machine.

## Headroom, failures and final state

- The first 8K-pair probe stopped at the initial **1,536 MiB** headroom threshold,
  before completion. Sampled free memory was 1,191/1,344 MiB then; subsequent
  workspace growth settled at **903/1,056 MiB**.
- The remaining capacity cells used an explicit **512 MiB stop floor**, retaining
  the original vLLM allocation. All completed above that floor, with no OOM or
  preemption. Memory was sampled about once per second.
- A second ordinary 8K-pair probe retrieved all markers, but one response repeated
  reasoning text and hit its 256-token cap. That formatting failure is retained.
- Post-test readiness, text, JSON, tool parsing and streaming passed. The owned
  vLLM container then stopped cleanly (exit 0), restoring the initial stopped
  state. Model/runtime caches remain; no GPU compute processes remain.

[Reproduction commands and protocol](../benchmarks/CONCURRENCY.md) ·
[Portable receipt](../benchmarks/results/vllm-concurrent-context-v1.json).
Raw prompts, streamed frames, telemetry, exact client sources and scheduler-source
snapshots are retained under `logs/concurrent-context-v1/`.
