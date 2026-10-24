# TensorFold prefill investigation on the two SM120 cards

The new trace shows a distributed bottleneck: routed experts, projections and
attention all take substantial time. The evidence does not support treating the
entire vLLM gap as an embedding-placement or compressed-scale problem, or assuming
that a faster expert kernel alone will close it.

## Measured timing

Same frozen 8K/32K raw prompts and token IDs, 256 output tokens, greedy generation,
cold cache, one request, resident scales and GPU embeddings, configured 128K.
Two uninstrumented controls each used three measured repetitions after warmup.
Their prefill rates differed by less than 0.05%. vLLM numbers below are archived;
its baseline was not restarted or changed.

| Input / decode drafting | New TensorFold prefill tok/s | New TensorFold TTFT | Archived vLLM prefill tok/s | Archived vLLM TTFT |
|---|---:|---:|---:|---:|
| 8,192 / off | 4,442 | 1.845 s | 10,422 | 0.791 s |
| 32,768 / off | 4,481 | 7.313 s | 10,387 | 3.166 s |

Values summarize the two control medians. TensorFold's request-level `draft:false`
disables draft decoding, while the loaded engine still absorbs prompt rows into
its MTP state. vLLM's archived off run removes its speculative configuration at
launch. These are served-mode comparisons, not identical internal work schedules.

The successful diagnostic follows those controls and uses the same immutable image,
settings and stream measurement helper, with profiling enabled and separate stderr
files per rank. Measured TTFT was 1.889/7.430 s: about **2.35%/1.59% higher** than the
mean control medians. All 24 output hashes across controls and diagnostics matched
within their workload, including warmups. Every captured chunk and stage record
was checked for completeness on both ranks.

## Where native prefill spends time

Representative rank-0 medians below are average milliseconds per 2,048-row chunk
within each measured request. Rank 1 is similar. Traced chunk wall time was
463.1 ms at 8K and 455.3 ms at 32K. CUDA events include the interval between their
markers; they are not hardware-counter measurements of arithmetic alone.

| Stage | 8K ms/chunk | 32K ms/chunk | Interpretation |
|---|---:|---:|---|
| Routed experts | 126.8 | 122.0 | Routing-plan execution, CSF expansion and gate/up/down consumers; about 27% of chunk wall time |
| Mixer projections | 87.1 | 87.4 | Attention input projections |
| Mixer output projections | 33.9 | 34.0 | Separate from input projections |
| MLA | 64.9 | 57.6 | Attention stage; not the separate sparse-index score/select spans |
| KDA | 42.7 | 42.7 | Recurrent attention stage |
| Shared expert | 26.6 | 26.6 | Nested within FFN output projection; do not count both |
| MTP prompt absorption | 17.5 | 16.9 | Present even with request-level draft decoding off |
| Main-stream wait_shared + wait_gathered | 32.5 | 32.3 | Exposed waits across mixer and FFN sites |
| GPU embedding gather | 0.034 | 0.030 | Very small in this GPU-placement profile |

These are selected spans, not a partition that should sum to 100%. In particular,
FFN parent spans include router/experts, output_projection includes shared_expert,
and side-stream spans overlap main-stream work. The large side `wait_filled`
interval mostly waits for main-stream work to become ready; calling that interval
PCIe transfer time would be incorrect. Side partials are not added to main wall time.

## Next optimization work, in evidence-based order

1. **Measure real-shape expert and projection kernels on SM120.** Together these
   occupy much of the trace. Capture achieved bandwidth, occupancy, registers,
   spills and route distributions before choosing another tile. Previous wider
   tiles and split-K sweeps are retained; they did not establish a served speed gain.
2. **Reduce CSF expansion/consumer overhead while preserving W4A16 arithmetic.**
   Matched earlier tests put the residency prefill cost at 3.2–4.1%, so this is a
   bounded recovery opportunity, not an explanation for a greater-than-twofold gap.
   Reuse the existing integer byte-oracle, scratch-lifetime and full-model gates.
3. **Investigate MLA/KDA and projection scheduling at real prompt shapes.** Compare
   launch counts and main-stream stalls with a matched vLLM trace. The current
   native 2,048-row chunk and vLLM 4,096 batched-token budget are different scheduler
   concepts; simply doubling a constant is not a qualified optimization.
4. **Consider a separately qualified FP4-activation expert prefill path.** The
   pinned vLLM source enables hybrid A4 prefill when calibration/metadata conditions
   are met. Native experts use BF16 MMA after weight reconstruction. That is a
   plausible contributor, but experts are only about 27% of the traced wall time;
   this change alone cannot be assumed to close the whole gap. It changes activation
   arithmetic and needs operator tolerances, fixed-history diagnostics, paired
   short/long accuracy, and end-to-end timing before adoption.
5. **Treat communication and MTP-off specialization as secondary investigations.**
   Four-piece overlap is already retained. Main exposed waits are much smaller
   than the misleading side wait total. Skipping MTP prompt absorption for plain
   requests could only help that mode and needs cache/state transition validation;
   it cannot be removed from the normal MTP-on profile by assumption.

For workloads with repeated prefixes, separately qualify retained prompt-cache
reuse. Earlier resident-scale cold/warm tests passed at 3,648/7,993 input tokens,
but that does not qualify a 512K or 1M cache budget. Reuse can avoid repeated
prefill; it does not accelerate the first cold prompt or demonstrate a kernel
speedup. The current serving qualification keeps retained cache off, and the
1M profile has little spare memory above its floor.

Do not infer a quality benefit from slower execution or from BF16 activations.
The [short-quality comparison](QUALITY-BAKEOFF-V2.md) found no convincing advantage;
the long-context task screen is separate. No native arithmetic/kernel change was
made during this profiling and evaluation phase.

## Reproduction and preserved failures

The first profiling attempt passed smoke/retrieval but the production benchmark
correctly rejected `TF_PROFILE=1`. Its status remains failed. Combined rank stderr
also interleaved some trace records; those records are not used for this breakdown.
The normal benchmark guard was not relaxed.

```bash
.state/quality-venv/bin/python tensorfold/tuning/prefill_profile.py \
  --output logs/qualification-v2/prefill-diagnostic
python3 tensorfold/tuning/prefill_report.py \
  --diagnostic logs/qualification-v2/prefill-diagnostic \
  --controls logs/qualification-v2/prefill-profile/00-gpu-perf \
             logs/qualification-v2/prefill-profile/02-gpu-perf \
  --output tensorfold/results/prefill-profile-v2.json
```

Use fresh output paths; do not overwrite evidence. The diagnostic mounts a copy of
the pinned launcher with only per-rank stderr redirection, records both hashes,
uses the normal guarded lifecycle, and stops its owned container afterward. The
offline case generator was paused during timing; no clocks/power limits changed.
[Trace/control receipt](../tensorfold/results/prefill-profile-v2.json) ·
[Pinned vLLM source audit](../tensorfold/results/vllm-prefill-policy-audit-v1.json).
