# SM120 served tuning results

The complete initial screen ran 18 fresh process configurations (two controls plus 16 candidates) on the two RTX PRO 6000 Blackwell Max-Q cards. The exact NVIDIA FP4-Local checkpoint, CSF32 representation, BF16 activations and FP8 KV remained unchanged.

Each profile passed readiness, four API checks and paired draft/plain three-needle retrieval at roughly 2K and 8K. Timed requests used the frozen shared raw prompts, one active request, 256 output tokens, greedy sampling and MTP on. There were two measured repeats per cell after one excluded warmup. Full settings, medians, token hashes, GPU telemetry summaries and raw-evidence hashes are in [the portable receipt](results/served-screen-v1.json).

## Initial screen

The opening control was cooler during timing than the later profiles (about 4°C in the averages). Its prefill was 1.42% faster at 8K and 0.90% faster at 32K than the closing control. The table below uses the **closing control**, measured at 4,536 / 4,617 prefill tok/s for 8K / 32K, and about 122.4 tok/s decode after the 8K prompt. Small differences remain screening observations, not established gains.

| Profile | 8K prefill change | 32K prefill change | 8K decode change | All four output hashes match control |
|---|---:|---:|---:|---|
| tiles-up | +0.14% | -0.05% | -0.05% | yes |
| tiles | -0.19% | -0.30% | +0.07% | yes |
| split384 | -1.46% | -1.92% | -1.81% | no |
| split768 | -1.58% | -1.96% | -11.09% | no |
| overlap1 | -0.37% | -0.66% | -0.07% | yes |
| overlap2 | +0.39% | -0.00% | +0.00% | yes |
| overlap4 | +0.80% | +0.58% | -0.16% | yes |
| overlap6 | -0.67% | -0.73% | -0.49% | yes |
| pdl-off | +0.36% | +0.08% | -0.38% | yes |
| kda-narrow | +0.02% | -0.05% | -1.58% | yes |
| prompt-shape | -2.03% | -2.46% | +5.62% | no |
| reduce-gather | -6.51% | -6.76% | -0.14% | yes |
| reduce-ring | +0.49% | +0.03% | -0.15% | yes |
| cumem-device | +0.45% | +0.05% | -0.01% | yes |
| cumem-host | +0.18% | -0.00% | -0.09% | yes |
| host-embed | -1.11% | -1.29% | -0.09% | yes |

Every profile was internally repeatable on these timed requests. A passing profile status means its requested API/retrieval checks and repeatability checks completed; it does **not** imply equivalence to control or adoption. Split-K 384/768 and shape-dependent prompt reduction changed all four output hashes. A changed greedy output is not by itself an accuracy loss, but those policies also lack a useful overall speed result here.

Wider prefill tiles did not translate the isolated kernel improvements into a meaningful served-model gain. Four-piece overlap was the only scheduling candidate retained for a bracketed confirmation (0.58–0.80% prefill improvement). Narrow KDA lost decode speed; gather lost about 6.5–6.8% prefill throughput. PDL, ring and cuMem changes showed no clear benefit sufficient to adopt.

## Four-piece overlap confirmation

A separate control–candidate–control sequence used five measured repeats per
8K/32K prompt after warmup, with read-only clock telemetry added. All output
hashes matched across the three launches. The [confirmation receipt](results/overlap4-confirmation-v1.json)
retains complete settings, measurements, telemetry summaries and evidence hashes.

| Input | Four-piece prefill gain versus mean control medians | Control-to-control prefill drift | Decode change |
|---|---:|---:|---:|
| 8K | +1.05% | −0.01% | −0.12% |
| 32K | +0.87% | +0.14% | −0.09% |

This supports a small local prefill improvement. It does not establish the same
gain for every workload or a decode improvement. The normal image then passed the fixed 256-item accuracy suite with every
output token hash and per-item grade identical to the original baseline:
120/128 GSM8K, 113/128 strict IFEval and 114/128 loose IFEval. Its paired MTP
on/off timing outputs also matched. The [quality receipt](results/overlap4-quality-v1.json)
records actual settings and evidence hashes. Normal 8K prompt-cache qualification also passed: cold/warm and MTP on/off
outputs matched at 3,648 and 7,993 input tokens. The final 2K–64K shared run
matched all 48 original native output hashes. **Four overlap pieces are now
the normal recipe default**; set `TF_GLM_OVERLAP_PIECES=3` to roll back. The
normal image and target precision are unchanged.

## Embedding placement

Mapped-host embedding saved **1,208 MiB per GPU** at the same 128K configuration. Minimum measured free memory increased from 3,153 / 3,306 MiB to **4,361 / 4,514 MiB**. It preserved every screening output hash and had approximately unchanged decode, with **1.11–1.29% lower prefill throughput** than the closing control. This is a capacity/performance tradeoff, not a precision change.

The embedding values are still BF16. Each rank owns an immutable mapped pinned-host copy (about 1.18 GiB of host RAM per rank), and the existing GPU gather kernel reads that pointer. The source patch owns and releases the allocation, and rank agreement includes the placement/tile/split settings. Selected real-loader gates passed on both GPUs before full-model testing.

The combined four-piece/host-embedding configuration passed the complete
256-item quality suite with all output hashes and grades identical to the
original TensorFold baseline; see [the quality receipt](results/host-embedding-quality-v1.json).
At 1,439 fixed-history positions it had zero top-one changes and zero target-token
log-probability difference versus the normal four-piece image; see
[the probability diagnostic](results/host-embedding-probabilities-v1.json).
The combined profile also passed 261,888 and 524,027 prompt tokens with MTP on/off,
recovering all three markers with identical output hashes. At configured 512K,
one slot, zero retained cache and a 1.5 GiB reserve, minimum sampled free VRAM was
**1,547 / 1,700 MiB**, up from **339 / 492 MiB**. The [capacity receipt](results/host-embedding-capacity-v1.json)
records all responses, actual settings and raw evidence hashes. This is bounded
retrieval/capacity evidence, not broad long-context accuracy.

The 1M target remains about **2.15 GiB/rank short just to fit** on the limiting
GPU, or **3.65 GiB/rank short with 1.5 GiB headroom**, using the prior allocation
slope and assuming no additional workspace. Resident compressed-scale storage
or latent-cache sharding remains a separate engine task. The normal image retains device embedding; mapped embedding remains an explicit experimental profile. This embedding saving alone does not close the previously measured 1M memory gap.

## Reproduction and scope

See [the experimental recipe](tuning/README.md), [numerical contract](tuning/CONTRACT.md), [current vLLM/TensorFold benchmark](../docs/PERFORMANCE-CURRENT.md) and [precision guide](../docs/PRECISION-AND-QUALITY.md). Raw screen evidence is under `logs/tuning/served-screen-v1`; its original profile file is archived there. The screen used runner code from local commit `ba1a549`; later qualification options do not change those completed requests.

No GPU power/clocks, unrelated containers, checkpoint values or shared caches were changed. No upstream submission was made. The experiments qualify only the available SM120 setup, not other architectures or a full upstream release.
