# TensorFold: four streams at 262,144 tokens

Status: **proposed, blocked by the current resident-CSF32 single-slot restriction,
not a runnable or qualified profile**. This document changes no engine code,
profile, active configuration or service. The existing one-slot profiles remain
as qualified. Recorded 2026-10-24.

## Intended settings and location

262K means **262,144 tokens per request**, including the rendered prompt and
output. Four full windows total 1,048,576 tokens. Context is per stream; setting
`TF_CONTEXT=1048576` with four slots would request four 1M windows, not divide one
1M window among them.

The proposed profile name is `host-262k-x4`. After multi-slot support and
qualification, it belongs in [profiles.json](profiles.json), alongside the
existing profiles, with these overrides:

```json
{
  "TF_CONTEXT": "262144",
  "TF_PARALLEL": "4",
  "TF_MEMORY_RESERVE_GIB": "1.5",
  "TF_TUNE_HOST_EMBED": "1"
}
```

Retain resident compressed scales (`TF_TUNE_SCALE_LAYOUT=2`), zero retained prompt
cache (`TF_PROMPT_CACHE_GIB=0`), FP8 KV, TP2 and the existing SM120 tuning. Host
embeddings are a **candidate for this aggregate-1M workload**, not a new default
for ordinary one-slot 262K/512K serving. GPU embeddings can be reconsidered only
with measured concurrent headroom. MTP remains a per-request setting and needs
both on/off qualification.

The serving wrapper installs pinned defaults and profile values before invoking
the recipe. Editing `tensorfold/.env` or exporting `TF_PARALLEL=4` does not override
those profile-controlled settings through `serving/run.py`. Do not add the above
entry yet or relabel `host-1m`: the current image cannot run this configuration.
A future implementation must pin its new image, source tree and patch without
silently changing the images used to reproduce existing results.

Keep `nvidia/GLM-5.3-Flash-FP4-Local` at revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, its original NVIDIA compressed-scale
format, and `HF_CACHE_ROOT=/models_engrams/huggingface`.

## Current blocker and memory limits

[Recipe validation](../scripts/recipe.py) rejects resident scale mode 2 whenever
`TF_PARALLEL` is not 1. The [native patch](../csf-resident/resident-scales.patch)
also rejects binding any slot other than slot 0 (`CsfResidentSingleSlot`). This
is an explicit scope restriction in our local experiment, not a claim that all
TensorFold engines lack concurrent serving.

The implementation shares 216 MiB of scale-expansion scratch per rank across
routed layers. Supporting more slots requires reviewing scratch producer/consumer
ordering, concurrent prefill/decode, graph replay, and per-slot recurrent/MTP
state. Shared scratch might be safe under proven serialization, or require
separate storage/events; removing the guards alone does not establish either.
Disabling compressed scales loses the measured roughly 3,896 MiB per card saving
and is not a qualified capacity workaround.

Four streams have additional per-slot state and scheduling/workspace costs.
Equal aggregate token counts do **not** prove equal memory consumption. The
one-slot `host-1m` run measured only 1,693/1,846 MiB free, versus the 1,536 MiB
floor: GPU 0 had just 157 MiB margin. That evidence does not establish a four-slot
fit. `TF_MEMORY_RESERVE_GIB=1.5` alone does not enforce the physical free-memory
floor in the current allocation path.

## Qualification before promotion

1. **Implementation and operator checks.** Audit multi-slot state ownership and
   scale scratch ordering on both SM120 ranks. Test byte-exact scale expansion,
   mixed prefill/decode, duplicate/empty routes and reused scratch with multiple
   slots. Exercise graph replay, MTP on/off and slot teardown/reuse. Preserve the
   current refusal until a separately pinned candidate supports the new scope.
2. **Capacity ladder.** Start with four short requests, then four requests at
   approximately 64K, 128K and 262K total window use. Reserve output tokens within
   each 262,144-token limit. Use distinct cold prompts and a synchronized client
   barrier; verify four live slots actually overlap instead of silently queueing
   serially. Record actual token counts, per-card memory throughout load/prefill/
   decode/reuse, and allocation failures. Require at least 1,536 MiB sampled free
   on each card, no OOMs and no unexpected eviction/recompute. Sampling does not
   prove the absence of shorter allocation peaks; retain allocator/server logs.
3. **Isolation and recovery.** Give each stream distinct retrieval values. Check
   that answers do not leak across requests, including staggered arrivals,
   cancellation, disconnects and reuse of a freed slot while others decode.
   Check responses and server health after each transition.
4. **Quality separately from capacity.** Reuse frozen tasks and the corrected
   prompt protocol from the existing long-context harness. Compare identical
   262K-or-shorter prompts served serially and concurrently on the candidate;
   include retrieval at varied depths, tracking and aggregation, with MTP on/off.
   Record output hashes, original task scores, truncation and reasoning markers;
   investigate differences rather than assuming batching preserves exact tokens.
   Use existing older-engine results only for identical prompts/settings; there
   is no existing four-stream quality result to reuse as proof. Expand the suite
   when failures or variation warrant it, rather than automatically rerunning
   the completed 22-case one-slot 1M screen.
5. **Performance and release evidence.** Report per-stream TTFT, prefill rate,
   decode rate and completion latency, plus aggregate throughput and fairness.
   Separate cold simultaneous arrivals from staggered arrivals and cache reuse.
   Compare against matched one-slot measurements, not archived timings at other
   lengths. Save image/source/checkpoint pins, configuration, raw outputs and GPU
   telemetry. Run `./scripts/check.sh` for implementation changes and append actual
   GPU outcomes to [VALIDATION.md](../../docs/VALIDATION.md). Promote a separately
   named profile only after capacity, isolation and quality checks pass.

These are future qualification requirements. No four-stream GPU inference,
capacity result, quality result or speed claim is established by this document.
