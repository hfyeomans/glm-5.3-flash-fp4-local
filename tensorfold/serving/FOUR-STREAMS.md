# TensorFold: four streams at 262,144 tokens

**Proposed; blocked by the resident-CSF32 single-slot restriction.** Recorded
2026-10-24. The engine and existing profiles are unchanged.

## Proposed profile

`host-262k-x4` would use these overrides in [profiles.json](profiles.json), after
multi-slot support and qualification:

```json
{
  "TF_CONTEXT": "262144",
  "TF_PARALLEL": "4",
  "TF_MEMORY_RESERVE_GIB": "1.5",
  "TF_TUNE_HOST_EMBED": "1"
}
```

Each window includes prompt and output. Four × 262,144 totals 1,048,576 tokens;
`TF_CONTEXT=1048576` with four slots instead requests four 1M windows.
Retain `TF_TUNE_SCALE_LAYOUT=2`, `TF_PROMPT_CACHE_GIB=0`, FP8 KV, TP2 and the
existing SM120 tuning. Host embeddings are a candidate for this aggregate-1M
workload; one-slot 262K/512K profiles keep GPU embeddings. Test MTP on and off.

`serving/run.py` installs pinned profile values, overriding corresponding
`tensorfold/.env` and exported settings. Keep this proposal out of the runnable
manifest until supported; pin the candidate image, tree and patch separately.

Checkpoint: `nvidia/GLM-5.3-Flash-FP4-Local` at
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, original NVIDIA compressed-scale format,
with `HF_CACHE_ROOT=/models_engrams/huggingface`.

## Blocker and memory budget

[Recipe validation](../scripts/recipe.py) refuses mode 2 with `TF_PARALLEL!=1`;
the [native patch](../csf-resident/resident-scales.patch) refuses slots beyond 0
with `CsfResidentSingleSlot`. These guards belong to our resident-scale extension.

The 216 MiB/rank scale scratch is shared across routed layers. Multi-slot work
needs validated producer/consumer ordering, mixed prefill/decode, graph replay
and per-slot recurrent/MTP state. Use proven serialization or independent
storage/events. Disabling compressed scales gives back about 3,896 MiB/card.

Four slots add state and workspace costs. The one-slot 1M run left
1,693/1,846 MiB free—only 157 MiB above the 1,536 MiB floor on GPU 0.
`TF_MEMORY_RESERVE_GIB=1.5` does not enforce that physical floor in the current
allocation path; four-slot fit remains untested.

## Qualification gates

1. **Operators and state:** test exact scale expansion, duplicate/empty routes,
   reused scratch, mixed prefill/decode, graph replay, MTP on/off and slot teardown
   on both SM120 ranks. Keep the guard until a pinned candidate passes.
2. **Capacity:** ladder four simultaneous cold requests through short, 64K, 128K
   and 262K windows, reserving output space. Verify four live slots overlap.
   Record token counts and load/prefill/decode/reuse memory; require at least
   1,536 MiB sampled free/card, no OOMs or unexpected eviction/recompute. Retain
   allocator/server logs alongside sampled telemetry.
3. **Isolation/recovery:** use distinct retrieval values and check staggered
   arrivals, cancellation, disconnects and slot reuse while peers decode.
4. **Quality:** compare frozen identical prompts serially and concurrently, using
   the corrected reasoning wrapper, varied retrieval depths, tracking and
   aggregation with MTP on/off. Record hashes, task scores, truncation and reasoning
   markers. Reuse older results only for identical prompts/settings; expand cases
   when failures or variation warrant it.
5. **Performance/release:** report per-stream TTFT, prefill, decode and latency,
   aggregate throughput and fairness. Separate cold, staggered and reused-prefix
   workloads; compare matching one-slot cases. Save pins, configuration, responses
   and telemetry. Run `./scripts/check.sh` for implementation changes and record
   GPU outcomes in [VALIDATION.md](../../docs/VALIDATION.md) before promoting a
   separately named profile.
