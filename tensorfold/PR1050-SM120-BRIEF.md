# PR #1050 discussion brief: RTX PRO 6000 Blackwell SM120

Prepared locally on 2026-10-24; **not posted upstream**. This is a record of measured local changes and proposed follow-up work, not a claim of exhaustive optimization or qualification on GB10/other GPUs. TensorFold is the engine name; TensorFlow is not involved.

## Reproduction baseline and credit

The checkpoint integration is the work of PR #1050's author, Bizuayeu: [PR #1050](https://github.com/ashhart/TensorFold/pull/1050), head `9b23b6a2e7ff226ca61934189c4cd57bf2933980`. Our local recipe combines it with TensorFold v1.1.3 `36415df4f76e4458e82d0fa729ecb28e287ebf38`, yielding source tree `1d52effd182d9319d775d5787904c36783e532f1`. Do not attribute the upstream loader/codec implementation to these local tuning experiments.

Hardware: two RTX PRO 6000 Blackwell Max-Q cards, 188 SMs/card, SM120, PCIe NODE topology; observed native transport NCCL P2P/IPC. CUDA 13.3, Zig 0.17.0, NCCL 2.31.2. No GPU power/clock changes, copied GB10 clock caps/CPU-affinity policies, NVLink or RoCE assumptions.

Model: `nvidia/GLM-5.3-Flash-FP4-Local@76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, ModelOpt mixed precision with CSF32 NVFP4 routed experts, BF16 activations, FP32 accumulation and FP8 KV. No checkpoint conversion, precision substitution or external draft model. MTP uses the checkpoint's own draft component.

## What changed and what we retained

| Area | RTX decision | Evidence and scope |
|---|---|---|
| GPU target and launch sizing | Build SM120; retain architecture dispatch and actual-device SM-count sizing | This is target selection, not deletion of every GB10/SM121 path. The SM121-specific draft-head PDL path stays off on SM120. |
| Expert decode grouping | Select existing one-warp variant instead of upstream four | About +0.65–1.27% decode on measured workloads; full quality hashes unchanged. Normal recipe default `TF_NVFP4_DECODE_WARPS=1`; rollback `4`. |
| Prefill communication overlap | Select four pieces instead of three | Bracketed five-repeat confirmation: +1.05% at 8K, +0.87% at 32K; decode approximately unchanged. Normal default `TF_GLM_OVERLAP_PIECES=4`; rollback `3`. |
| Expert prefill tiles | Retain gate/up 4, down 8 | Wider 8/16 variants improved isolated kernels but did not materially improve served prefill. A GB10-associated setting can still be the best measured setting here. |
| Split-K / prompt policy | Retain target 192 and fixed prompt-K `one` | 384/768 and shape-dependent policies changed output hashes and hurt prefill; no adoption or accuracy-loss claim from hashes alone. |
| PDL and KDA | Retain supported BF16 PDL and wide KDA decode | PDL-off showed no clear gain; narrow KDA lost roughly 1.3–2% decode. |
| Collectives / allocation | Retain split prefill reduction, NCCL cuMem flags off | Gather lost 6.5–6.8% prefill; ring and cuMem variants showed no compelling benefit. cuMem flags select allocation mechanisms, not extra VRAM. |
| Embedding placement | Optional immutable BF16 mapped-host embedding in a separate image | Saves 1,208 MiB/GPU; roughly 1.11–1.29% prefill cost in screening, decode approximately unchanged. Capacity tradeoff, not default placement or quantization. |
| Resident CSF32 | Implemented and locally tested in a separate image | All routed scale bytes exact; all 256 quality hashes unchanged. Near-1M retrieval passed in both MTP modes, with 1,693/1,846 MiB sampled free. Matched timing shows a 6.6–7.9% decode and 3.2–4.1% 2K–64K prefill cost; retain it as an explicit capacity tradeoff. |

Screening covered 18 fresh configurations: two controls and 16 candidates, 144 measured timing requests plus 72 excluded warmups. Small percentages in the initial screen are not causal gains; retained overlap received a separate control–candidate–control confirmation. Automatic clocks/temperatures varied and were recorded. The full [tuning results](TUNING-RESULTS.md) preserve rejected candidates and limitations.

## Source patches versus recipe settings

The adopted one-warp and four-piece choices select existing upstream controls. The normal image remains `sha256:c829536f5a214755ff50f4e72e622814a9226b8aec43ed646cdc6cd20c418e11`.

The isolated source experiment is preserved in [sm120-experiments.patch](tuning/sm120-experiments.patch), SHA-256 `4e1cf2a5666d6ddb1527bbf7192f785caa928614feb68ec2912608a95d8cfdef`. It adds matched host/device prefill-tile choices, a split target, mapped embedding ownership/cleanup and rank-setting agreement (fingerprint layout 12). It does not implement compressed-resident scales. Experimental tree: `0c2bc6ddf016c0d96d63fbd7d1238b6355603578`; image: `sha256:2d9b848d07d66c6072ab7757017dd5dffbfb30453305022cbe2befc8904899ee`.

Local history separates `7e06307` (one-warp adoption), `ba1a549` (isolated source controls), `f19564d` (overlap/embedding qualification and results), and `bcda842` (performance/memory explanations). [The experiment recipe](tuning/README.md) pins build/profile reproduction. A future upstream patch should extract the justified change and relevant tests rather than submit all screening knobs as universal defaults.

## Validation and measured limits

- Normal four-piece and experimental mapped-embedding profiles each reproduced all 256 original quality output hashes and grades: GSM8K 120/128; IFEval strict 113/128, loose 114/128. These are bounded task scores, not universal accuracy proof.
- The combined profile's fixed-history comparison covered 1,439 positions: zero top-one flips and zero target-token log-probability difference. See [probability receipt](results/host-embedding-probabilities-v1.json).
- Normal-cache tests at 3,648/7,993 input tokens matched cold/warm and MTP/plain output hashes. The final shared 2K–64K refresh matched all 48 original native responses, including warmups and both MTP modes.
- Mapped embedding plus four-piece overlap passed retrieval at 261,888 and 524,027 actual prompt tokens, MTP on/off, with all three markers and identical outputs. The 512K configuration left 1,547/1,700 MiB sampled free. This does not qualify broad long-context reasoning or multiple active slots.
- The old device-embedding 1M configuration failed allocation. The new compressed-resident image passed near-1M retrieval at 1,048,311 actual input tokens in both MTP modes. This is a one-slot capacity/retrieval result, not broad long-context accuracy.

Portable evidence: [decode](results/decode-warps-v1.json), [screen](results/served-screen-v1.json), [overlap confirmation](results/overlap4-confirmation-v1.json), [normal quality](results/overlap4-quality-v1.json), [normal cache](results/overlap4-cache-v1.json), [mapped quality](results/host-embedding-quality-v1.json), [capacity](results/host-embedding-capacity-v1.json). The chronological [validation record](../docs/VALIDATION.md) includes actual GPU checks and failures; raw local logs remain outside Git.

At 8K input/256 output, the current matched benchmark reports:

| Engine | MTP | Decode tok/s | Prefill tok/s | Client TTFT, seconds |
|---|---|---:|---:|---:|
| vLLM | on | 174.12 | 10,195 | 0.809 |
| vLLM | off | 117.72 | 10,422 | 0.791 |
| TensorFold | on | 122.27 | 4,499 | 1.822 |
| TensorFold | off | 69.70 | 4,538 | 1.806 |

The [full report](../docs/PERFORMANCE-CURRENT.md) describes frozen prompts, three measured repeats, cold cache, tokenizer checks and reused unchanged vLLM cohorts. vLLM is faster on this cell; TensorFold has lower TTFT on some tiny prompts. We have not profiled enough to assign the cross-engine gap to a particular kernel or PR omission. Changed vLLM greedy outputs occurred with and without MTP; their cause remains unestablished.

## Implemented storage extension and next work

The [compressed-scale plan](COMPRESSED-SCALES-PLAN.md) led to the separate
[resident-CSF32 patch and recipe](csf-resident/README.md). It uses codes-only packing,
compact rank-local descriptors and 216 MiB shared expansion scratch per rank while
preserving expert arithmetic. Actual requested saving is 3.81765 GiB/rank; measured
physical headroom improves by 3.8047 GiB/rank. Combined with mapped embedding, it
passed 1,048,311 input tokens with 1,693/1,846 MiB minimum sampled free memory.

All 256 quality hashes and 1,439 sampled target log probabilities are unchanged.
Same-image legacy→resident→legacy timing matched all 144 response hashes but measured
a 6.6–7.9% decode cost and 3.2–4.1% 2K–64K prefill cost. It is a capacity option, not
a faster default. The retained prompt gate/up kernel already uses 254 registers per
thread on SM120; direct compressed consumption needs resource/profiling evidence
before fusion can be called an optimization. The full upstream GPU suite's inherited
resource-record failures and original HTTP-fixture failures remain in the receipts.

Questions for eventual maintainer coordination: preferred descriptor/owner boundary; scratch lifetime across native stream modes; supported TP slicing envelope; whether storage separation should land independently; required platforms and performance gates. These are topics for a later authorized upstream discussion, not requests already sent.

As checked on 2026-10-24, PR #1050 is still open and GitHub reports merge conflicts against current main. Refresh that state and coordinate with its author before rebasing or proposing an upstream change. No public posting, issue claim, merge or release was performed.

## Resident-scale implementation record

The [separate recipe](csf-resident/README.md) adds compact TP slices, typed GPU descriptors, shared expansion scratch, codes-only consumers and native contract tests. It extends the credited PR integration and the earlier mapped-embedding/tuning patch. The [loader receipt](results/csf-resident-loader-v1.json) records actual payload allocations, exact scale reconstruction and explicit draft-head exclusions. The original planning estimates above are preserved for comparison. Normal defaults remain legacy storage; this is an opt-in one-slot TP2 experiment.

A pinned vLLM source audit also found hybrid A4 prefill enabled by the supplied flags. That is a separate arithmetic choice from TensorFold's W4A16 expert path; see [precision notes](../docs/PRECISION-AND-QUALITY.md). No timing ablation attributes the full cross-engine gap to this policy.

The [resident-scale results](csf-resident/RESULTS.md) record the implemented patch, measured 3.80 GiB/rank headroom gain at 512K, exact quality comparisons and completed 640K→768K→896K→near-1M prompt ladder. The 1M pool has only 157 MiB above the 1.5 GiB target on the limiting GPU; scope remains explicit.


## Follow-up qualification and profiling (2026-10-24)

The separate serving profiles now keep embeddings in VRAM through 512K and map
them only for 1M. Actual 512K cold retrieval passed in both MTP modes, with
4,235/4,388 MiB free through retrieval and 4,177/4,330 MiB whole-run minima after
the extra probability diagnostic. All 256 short-quality outputs and 1,439
probabilities matched the earlier native reference. This is not broad 512K accuracy.

[Complete native prefill traces](../docs/PREFILL-INVESTIGATION-V2.md) show routed
experts at about 27% of chunk wall time, alongside substantial projection and
attention costs. These results support targeted kernel/attention investigation,
not a claim that removing a GB10-associated setting or adding A4 alone solves the
whole gap. The diagnostic launcher changes stderr routing only; no native kernel
or precision change was made in this phase. The short vLLM comparison shows no
convincing quality advantage for the slower native configuration.


The subsequent [initial long-context accuracy screen](../docs/LONG-CONTEXT-ACCURACY-V2.md)
completed 22 core near-1M cases (two fixed samples per task): 21 fully correct,
one frequency-aggregation miss (two of three words). All core retrieval, tracking
and QA cases passed. Four corpus-limited checks and nine previously completed
supplemental long cases also passed. All 52 shorter controls were retained,
including one incorrect SQuAD answer. The mean/clean-completion metrics and actual
lengths are explicit; no truncations or reasoning-marker outputs occurred in the
corrected cohort. A prompt-wrapper bug in the initial attempt was corrected and
its invalid scores excluded. The user reduced the larger plan by fixed sample
index; completed valid evidence was reused without selecting by score.

This adds bounded accuracy evidence for the RTX 1M host-embedding/resident-scale
profile, with the same 1,693/1,846 MiB sampled free floor measurements. It is not
full release qualification, a statistical quality advantage or a matched
vLLM long-context result. The inherited CUDA resource-record findings remain as
previously documented; this phase changed recipe/evaluation tooling, not native
kernel arithmetic. No upstream posting has been made.
