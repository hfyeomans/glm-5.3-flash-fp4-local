# Resident CSF32: measured local results

NVIDIA checkpoint revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, two RTX PRO
6000 Blackwell Max-Q cards, SM120 TP2. Source tree and image pins are in
[profiles.json](profiles.json); implementation checkpoint: `32fc48a`.
Normal recipes remain available. This is an opt-in capacity experiment.

## Numerical and quality evidence

- All 36,288 routed expert matrices per rank reconstructed their original scales
  exactly on both GPUs: 19,025,362,944 scale bytes across the pair. Weight codes and
  global calibration upload checks also passed. Three existing draft-head quantizer
  upload exclusions are explicit in the loader receipt.
- Separate consumers passed 14 native contract cases per card; integer expansion
  passed nine shapes per card, including scratch reuse and empty/duplicate routes.
- All 256 frozen quality output hashes and grades match the original native baseline:
  GSM8K 120/128; IFEval strict 113/128, loose 114/128. No truncated/unclosed outputs.
- Fixed-history probabilities: 1,439 positions, zero top-one flips and zero target
  log-probability changes. This is sampled evidence, not universal equivalence.
- Cold/warm cache checks at 3,648/7,993 tokens passed with identical MTP/plain hashes.

See portable [loader](../results/csf-resident-loader-v1.json),
[quality](../results/csf-resident-quality-v1.json),
[probability](../results/csf-resident-probabilities-v1.json) and
[cache](../results/csf-resident-cache-v1.json) receipts.

## Capacity and physical headroom

| Configuration | GPU 0 minimum free, MiB | GPU 1 minimum free, MiB |
|---|---:|---:|
| Previous mapped embedding, 512K | 1,547 | 1,700 |
| Resident CSF32 + mapped embedding, 512K | 5,443 | 5,596 |
| Resident CSF32 + mapped embedding, 1M | 1,693 | 1,846 |

The observed gain at 512K is **3,896 MiB (3.8047 GiB) per card**. The loader requests
3.81765 GiB/rank less storage after scratch and metadata; runtime/allocator overhead
accounts for the distinction, without assigning it to a particular component.
Doubling the configured window consumed another 3,750 MiB/rank, matching the earlier
allocation-slope estimate. The limiting card has only **157 MiB above the 1.5 GiB
free-memory target**. These are one-second samples, not proof against shorter peaks.

One slot, no retained prompt cache, immutable BF16 mapped embedding, FP8 KV and
native MTP loaded throughout. Rank-0 native counters reported a device allocation
peak of 98,861,412,570 bytes and 1,269,439,488 pinned-host bytes held during the late
ladder. Those counters exclude some driver allocations and are not total host RSS
or a host-load peak. No GPU clocks, power settings or unrelated processes changed.

The 512K profile passed at 261,888 and 524,027 actual prompt tokens. Then one final
1,048,576-token pool was allocated and the following cold prompt ladder completed:

| Actual input tokens | MTP on wall time, s | MTP off wall time, s | Result |
|---:|---:|---:|---|
| 655,104 | 188.14 | 189.45 | Identical; all three markers |
| 786,168 | 236.99 | 237.27 | Identical; all three markers |
| 917,247 | 288.06 | 288.68 | Identical; all three markers |
| 1,048,311 | 342.88 | 342.92 | Identical; all three markers |

Each request reserved room for up to 128 output tokens, retrieved three markers near
10/50/90% of the input, reported the expected token count and zero cached tokens,
finished normally, and matched its paired output hash. The final prompt had
**1,048,311 input tokens**; this demonstrates near-ceiling inference in the 1M window.
The intermediate prompt lengths ran in that same final pool, not separate pool sizes.
This does not qualify broad long-context reasoning or multiple active slots.

Receipts: [512K](../results/csf-resident-capacity-512k-v1.json),
[1M ladder](../results/csf-resident-capacity-1m-v1.json).

## Performance qualification

The same-image legacy→resident→legacy comparison completed all **144 requests**,
including 36 excluded warmups and 108 measured requests. All output token hashes
match by workload across modes, repeats and layouts. Both layouts used mapped
embedding, 128K context, one slot, zero retained prompt cache and four-piece overlap.
The table uses three-repeat resident medians and the mean of the returning legacy
medians. It measures the combined layout/expansion tradeoff, not expansion alone.

| Prompt | MTP | Legacy decode tok/s | Resident decode tok/s | Legacy prefill tok/s | Resident prefill tok/s | Legacy TTFT s | Resident TTFT s |
|---|---|---:|---:|---:|---:|---:|---:|
| short-code | on | 128.48 | 119.16 | 734 | 605 | 0.046 | 0.055 |
| short-code | off | 73.96 | 68.16 | 773 | 644 | 0.043 | 0.052 |
| short-prose | on | 145.09 | 134.63 | 753 | 605 | 0.042 | 0.052 |
| short-prose | off | 73.98 | 68.14 | 795 | 652 | 0.040 | 0.048 |
| prefill-2048 | on | 123.54 | 115.25 | 4929 | 4726 | 0.416 | 0.434 |
| prefill-2048 | off | 69.48 | 64.28 | 5141 | 4956 | 0.399 | 0.414 |
| prefill-8192 | on | 121.12 | 112.72 | 4437 | 4278 | 1.847 | 1.915 |
| prefill-8192 | off | 69.09 | 63.90 | 4481 | 4327 | 1.829 | 1.894 |
| prefill-32768 | on | 119.79 | 111.55 | 4506 | 4352 | 7.273 | 7.531 |
| prefill-32768 | off | 67.97 | 63.03 | 4519 | 4361 | 7.252 | 7.515 |
| prefill-65536 | on | 133.31 | 124.56 | 4466 | 4316 | 14.677 | 15.185 |
| prefill-65536 | off | 66.36 | 61.51 | 4462 | 4318 | 14.690 | 15.180 |

Observed decode cost: **6.6–7.9%** across cells. At 2K–64K, prefill throughput cost
was **3.2–4.1%**; tiny-prompt TTFT increased roughly 8–10 ms. Returning legacy decode
medians differed by at most 0.25%; 8K–64K prefill controls differed by less than 0.7%.
Short/2K timing is noisier. Automatic clocks, temperatures and power were sampled;
no settings were changed. This is a capacity tradeoff, **not a faster default**.

At 128K, sampled free memory was 4,361/4,514 MiB in each legacy control and
8,257/8,410 MiB with resident scales: the same 3,896 MiB/card gain measured at 512K.
[Portable paired timing receipt](../results/csf-resident-performance-v1.json).
The earlier 16K screen overlapped CPU compilation and is not the causal comparison.

The original upstream cancellation fixture sent `Host: x`; the server rejected its
long requests with HTTP 403 before inference. All three original sweep status files
therefore remain **failed**, even though their timing stages completed successfully
first. The report verifies all timing rows independently and preserves the failures.
Only the fixture's Host header has been adapted to the URL authority, with original
and adapted hashes retained. Corrected cancellation checks passed separately on both layouts: six client-left
events per layout, observed-running disconnects, and identical follow-up text/token
hashes. Resident release was 0.027 s in all three observed trials; legacy ranged
0.017–0.356 s. The longest follow-up request was 0.165/0.626 s respectively. These
small samples establish recovery within the declared bounds, not a recovery-speed
ranking. No timing data was overwritten or rerun, and no server policy changed.
[Corrected cancellation receipt](../results/csf-resident-cancellation-v2.json).

## Qualification limits and preserved failures

Recipe checks passed 48 tests. Upstream host tests, golden/fuzz checks and plain build
passed; the lean check found zero new findings (438 inherited in both trees).
The full CUDA suite on GPU0 passed 146 numerical kernels and caught all 33 planted
controls, but **exited 1** on 949 incomplete/stale resource records. The parent build
reproduced the same 949 findings on both cards, with identical live failures. All
14 newly registered launches passed; none of the 64 live mismatches exceeded the
observed device launch limits. The GPU1 full numerical suite was not reached after
GPU0's nonzero exit; new operator and loader gates separately passed on both cards.
This is not a clean full upstream release qualification.

[The contract receipt](../results/csf-resident-contract-v1.json) and
[chronological validation](../../docs/VALIDATION.md) preserve the failed test-harness,
stress-fixture and missing-asset attempts and their corrections. Other architectures,
broad long-context accuracy, exhaustive concurrency and direct KV/state-array audits
remain outside the demonstrated scope. The storage patch does not change KV/state
arithmetic. No upstream publication has been made.


[Final evidence index](../results/csf-resident-qualification-v1.json) pins all portable
receipts and the code-check log. At handoff both model services are stopped, the
vLLM container is preserved, and no GPU compute jobs remain. GPU free memory returned
to 97,187/97,340 MiB; the unrelated existing graphics allocation was preserved.


## Later serving and accuracy qualification

The [serving profiles](../serving/README.md) now measure 512K with GPU embeddings
(4,177/4,330 MiB whole-run minimum free, including the probability diagnostic)
and reserve host embedding for the explicit 1M profile. The
[initial longer-task screen](../../docs/LONG-CONTEXT-ACCURACY-V2.md) completed
21/22 core near-1M cases with full credit; one aggregation answer matched two of
three words. This extends the earlier capacity-only evidence without claiming
exhaustive long-context quality. Native image/arithmetic and historical receipts
above are unchanged.
