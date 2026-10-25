# Local SM120 tuning experiments

SM120 experiments for two RTX PRO 6000 Blackwell Max-Q cards, using a separate
pinned image.
See [CONTRACT.md](CONTRACT.md) for scope and acceptance criteria.

The checkpoint remains NVIDIA `GLM-5.3-Flash-FP4-Local`, revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, with CSF32 compressed scales.

## Build and select

The base image must already exist from the qualified TensorFold recipe. Build
from this directory; the Dockerfile verifies both the base tree and patch hash.

```bash
docker build --progress=plain -t glm53-fp4-tensorfold:sm120-tuning-v1 tensorfold/tuning
python3 tensorfold/tuning/run.py tiles gpu-check
python3 tensorfold/tuning/run.py tiles start
python3 tensorfold/tuning/run.py tiles smoke --wait 1200 --suite
# Preserve full logs before stop removes this recipe's container:
docker logs glm53-fp4-tensorfold > logs/my-tiles-server.log 2>&1
python3 tensorfold/tuning/run.py tiles stop
```

Stop the owned vLLM recipe before GPU experiments. Do not stop unrelated
services. At the end restore it with `MTP_ENABLED=1 ./start.sh` and
`./smoke.sh --suite`. No clock, power or cache-pruning changes are required.

`profiles.json` pins the image, patch and source tree. `run.py` explicitly resets
all recipe settings before applying each profile, preventing ambient knobs from
leaking between runs. The recipe rejects non-default source controls without
both experimental identity pins and matching image labels.

## What changes

| Control | Starting policy | Experimental choices | Implication |
|---|---|---|---|
| Expert prefill width | gate/up 4, down 8 warps | gate/up 8, down 16 | Matching CUDA instances and host launch geometry; isolated byte-equivalence screen required |
| Projection split-K target | 192 column tiles × slices | 384, 768 | Changes FP32 addition order; numerical and model checks required |
| Communication overlap | 3 pieces | 1, 2, 4, 6 | Existing native control; no precision change intended |
| PDL | enabled | disabled | Existing launch control |
| KDA decode | wide | narrow | Existing attention launch control |
| Prompt reduction | fixed K | shape-dependent K | Can change arithmetic; retain separate evidence |
| Prefill collective | split | gather, ring | Existing communication control |
| NCCL allocations | cuMem device/host disabled | enable each separately | Allocation and stability experiment |
| Embedding location | device BF16 | mapped pinned host BF16 | Same values and GPU gather; potential 1.18 GiB/rank VRAM saving, PCIe/lifetime checks required |

The earlier measured one-warp expert decode setting remains enabled throughout.
Native screen context is 128K, one slot, zero retained prompt cache, default
reserve. MTP uses the checkpoint's bundled prediction head; no external draft
model is added.

## Reproduce a served screen

Prepare the shared workloads once using `benchmarks/prepare.py`; preserve their
hash. Use the existing quality virtual environment for tokenizer dependencies.
Every profile gets API and draft/plain retrieval checks, then warmup and fixed
length timed requests. All attempts, server logs and one-second GPU telemetry
are retained. A final control measures drift.

```bash
.state/quality-venv/bin/python tensorfold/tuning/sweep.py \
  --profiles control tiles-up tiles split384 split768 overlap1 overlap2 \
    overlap4 overlap6 pdl-off kda-narrow prompt-shape reduce-gather reduce-ring \
    cumem-device cumem-host host-embed control \
  --output logs/tuning/my-screen
python3 tensorfold/tuning/report.py logs/tuning/my-screen \
  --output logs/tuning/my-screen-report
```

The sweep controls only the recipe-owned TensorFold container. It refuses an
existing container, serializes sweeps, skips completed matching attempts and
preserves interrupted attempts for inspection. Successful sweeps finish with TensorFold stopped; failed cleanup is recorded
in the receipt and `stop.log`. Restore vLLM after checking the final status.

Screen with two measured repeats per cell. Confirm candidates with bracketed
controls, the shared MTP on/off benchmark, fixed histories, the 256-item quality
suite and relevant cache/capacity checks.

## Numerical probes

`prefill_tiles.cu` instantiates the pinned upstream NVFP4 expert template at
multiple widths. It checks output identity, repeatability, guards and unchanged
inputs on both GPUs, including valid partial tiles and non-finite fixtures.
Time the full model separately.

`glm_mm_bench_fp64.zig` is a separate probe, adapted from the pinned upstream
matrix benchmark. It computes independent FP64 references for the first three
rows of each matrix shape, checking every output column of those sampled rows.
Its whole-output cross-product check now rejects nonfinite values in both
arrays; timing and the FP64 bound are unchanged. See the
[maintainer note and CPU regression](../FP64-AGREEMENT-FIX.md).

The 192/384/768 policies all passed the sampled bound, but each larger target
increased maximum absolute error in six of 56 shape/row cells.

`teacher_force.py` freezes up to the first 128 output tokens from eight GSM8K and eight
IFEval baseline examples, then requests top-two prompt log probabilities at
those identical histories. It reports target-log-probability differences and
top-one changes with the baseline margin.

## Candidate qualification stages

The `overlap4-quality` and `host-embed-overlap4-quality` profiles use the same
16K context, zero retained cache and default reserve as the original native
accuracy run, with four-piece overlap. `--baseline-image` selects the normal
qualified image; source-dependent knobs are refused on that image. The second
profile selects the experimental image with mapped embedding.
Use fresh output folders; `--teacher-histories` requires a previously frozen
history file from `teacher_force.py prepare`.

```bash
.state/quality-venv/bin/python tensorfold/tuning/sweep.py \
  --baseline-image --profiles overlap4-quality --mtp both --repeats 3 \
  --cases short-code short-prose prefill-2048 prefill-8192 \
  --teacher-histories logs/tuning/teacher-histories-v1.json --quality \
  --output logs/tuning/my-control-qualification
.state/quality-venv/bin/python tensorfold/tuning/sweep.py \
  --profiles host-embed-overlap4-quality --mtp both --repeats 3 \
  --cases short-code short-prose prefill-2048 prefill-8192 \
  --teacher-histories logs/tuning/teacher-histories-v1.json --quality \
  --output logs/tuning/my-host-qualification
```

The optional quality stage runs the unchanged 256-item client and saves a
sidecar benchmark manifest with the full tuning settings. The optional teacher
stage compares fixed histories separately from throughput; its extra head passes
are outside the timed benchmark. Outcomes are in the validation record.

`host-embed-overlap4-512k` is the capacity experiment: 524,288 configured context, one slot,
zero retained cache, 1.5 GiB reserve and mapped embedding. To exercise long prompts
with draft/plain retrieval without rerunning the short throughput screen:

```bash
.state/quality-venv/bin/python tensorfold/tuning/sweep.py \
  --profiles host-embed-overlap4-512k --skip-benchmark \
  --context-targets 261888 524032 --output logs/tuning/my-host-capacity
```

Keep full logs and GPU telemetry. This embedding-only profile still falls short of 1M.

For the normal 8K profile, retained-cache acceptance is a separate stage:

```bash
.state/quality-venv/bin/python tensorfold/tuning/sweep.py \
  --baseline-image --profiles overlap4-default --skip-benchmark --cache-check \
  --output logs/tuning/my-normal-cache-check
```

The recorded outcomes are in [TUNING-RESULTS.md](../TUNING-RESULTS.md). The
common screening settings explicitly pin three-piece overlap, so a later normal
recipe default does not silently change the historical control.
