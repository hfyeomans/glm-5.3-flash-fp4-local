# Serving and qualification profiles

These profiles pin the resident-CSF32 experimental image and retain the NVIDIA
checkpoint revision checked by the normal recipe. Historical measurements under
`../csf-resident/profiles.json` are unchanged.

| Profile | Configured context | Embeddings | Purpose |
|---|---:|---|---|
| gpu-quality | 16,384 | GPU | Frozen short-quality comparison |
| gpu-perf | 131,072 | GPU | Performance measurement |
| gpu-profile | 131,072 | GPU | Instrumented diagnostic; not a production timing |
| gpu-512k | 524,288 | GPU | Passed capacity/retrieval and short-probability checks |
| host-1m | 1,048,576 | Host mapped | Separate capacity/accuracy qualification |
| fast-gpu | 131,072 | GPU | Legacy expanded scales; diagnostic alternative |

Only `host-1m` maps embeddings into host memory. All profiles use one slot, FP8 KV,
zero retained prompt-cache budget and four overlap pieces. All except `fast-gpu`
retain compressed expert scales. MTP is controlled per request and remains enabled
by default. The memory reserve setting alone does not enforce physical headroom;
qualification samples both cards and gates on at least 1,536 MiB free.

Run from the repository root, with the owned vLLM service stopped:

```bash
python3 tensorfold/serving/run.py gpu-512k start
python3 tensorfold/serving/run.py gpu-512k smoke --wait 1200 --suite
python3 tensorfold/serving/run.py gpu-512k stop
```

Substitute `host-1m` explicitly for the larger profile. Do not add host embeddings
to the smaller profiles to inherit the 1M capacity workaround.

See [validation](../../docs/VALIDATION.md) for measured qualification and limitations.
A configured context is not an accuracy guarantee. Near-window allocation and
retrieval tests are separate from the broader task evaluation.

Measured `gpu-512k` free memory was 4,235 / 4,388 MiB through retrieval.
The additional fixed-history log-probability diagnostic briefly used 58 MiB more
per card, giving whole-run minima of **4,177 / 4,330 MiB**. All 1,439 scored
positions matched the earlier native reference exactly.
[Qualification receipt](../results/gpu-512k-qualification-v2.json).


The `host-1m` initial accuracy screen is complete: **21/22 core near-1M cases fully
correct**, with one frequency-aggregation response matching two of three words.
No corrected-cohort truncations or reasoning-marker outputs occurred. Sampled
free memory remained at least **1,693/1,846 MiB**. These are two examples per task,
not an exhaustive 1M quality qualification or a comparison to vLLM at 1M.
[Accuracy results](../../docs/LONG-CONTEXT-ACCURACY-V2.md).


The proposed **four streams × 262,144 tokens** configuration is documented in
[FOUR-STREAMS.md](FOUR-STREAMS.md), including settings, scratch/state requirements
and capacity/quality qualification gates. It is blocked by the current resident-
CSF32 single-slot restriction and is not an executable profile. Existing serving
profiles and embedding placement are unchanged.
