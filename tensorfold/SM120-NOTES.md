# PR #1050 on two RTX PRO 6000 Blackwell Max-Q cards

Local evidence for a future upstream comment or PR. Nothing has been posted upstream.
The comparison here is with the PR author's **DGX Spark / GB10 (SM121)** machines, not a DGX B200 system.

## Reproducible identity

- Checkpoint: `nvidia/GLM-5.3-Flash-FP4-Local@76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, ModelOpt mixed recipes, `csf_32x2_v1` scales, all 42 shards. No substituted ZAI, EXL3 or older NVFP4 checkpoint.
- TensorFold base: v1.1.3, `36415df4f76e4458e82d0fa729ecb28e287ebf38`.
- PR [#1050](https://github.com/ashhart/TensorFold/pull/1050): `9b23b6a2e7ff226ca61934189c4cd57bf2933980`.
- Clean merged source tree: `1d52effd182d9319d775d5787904c36783e532f1`.
- Hardware: two RTX PRO 6000 Blackwell Max-Q, SM120, 188 SMs each; driver 615.71.09. CUDA-visible totals are about 97,337 MiB/card, distinct from the nominal 96 GB product specification.
- Build: x86-64, Zig 0.17.0, pinned CUDA 13.3 container, NCCL 2.31.2-1+cuda13.3, `-Dsm=120`.
- Native binary SHA256: `535e8a051e157b27fcd43fa47cda695dbaf7b1bb7ea0af8979260cc8d5c5fe75`.
- Initial image: `glm53-fp4-tensorfold:1.1.3-pr1050-sm120`, ID `sha256:6f5b4b2d833edd0a984bd745184ead48b7a93c89a0b9e50fab8af534b79d0f80`.
- Context-test image: `glm53-fp4-tensorfold:1.1.3-pr1050-sm120-context`, ID `sha256:c829536f5a214755ff50f4e72e622814a9226b8aec43ed646cdc6cd20c418e11`. Only the launcher/label differs; native binary and CUDA test-runner hashes match the initial image.

## Changes made locally

| Change | Reason | Evidence / qualification |
|---|---|---|
| Build for SM120 rather than SM121 | Target the installed discrete Blackwell GPUs | Native info reports SM120; both GPUs pass arithmetic and selected real-checkpoint loader gates |
| Two processes in one container, rank/device 0 and 1 | Both cards are in one host | TP2 startup and served acceptance pass |
| `TF_ROCE=0`, `NCCL_IB_DISABLE=1`, loopback NCCL bootstrap | The PR author's inter-machine RoCE setup is inapplicable here | 128K NCCL log explicitly reports `P2P/IPC` in both directions; loopback is bootstrap, not the GPU data path |
| FP8 KV, one active stream | Establish fit and accuracy before increasing concurrency | 8K and 128K retrieval and drafted/plain equality pass; this does not qualify multiple active streams |
| Explicit device prompt-cache budget | Avoid spending scarce VRAM on retained prefixes during context qualification | Original 8K run used 1 GiB; ladder runs use 0 and verify reported cached tokens are zero |
| Explicit native reserve override | Make any headroom reduction reviewable | 128K retains native default, about 3 GiB; 256K and 512K trials select 1.5 GiB; actual memory sampling is recorded separately |
| Opt-in `TF_GLM_PROF` and NCCL INFO logging | Identify local bottlenecks and transport | Enabled for 128K only; instrumented wall times are not an unprofiled performance comparison |
| Recipe image checks include launcher version | Older launchers would otherwise silently ignore new memory controls | Launcher label `2` required |
| `TF_NVFP4_DECODE_WARPS=1` recipe default | Select the faster existing expert-decode grouping measured on our SM120 cards | 0.65–1.27% observed decode gain; all 256 quality token hashes unchanged; 8K cached-profile checks pass. Four warps remains the historical ladder setting |

**No manual changes to upstream kernels, precision, split-K targets, expert tiles, clocks or power limits have been made.** The compressed-scale reader is the PR's implementation, not a local SM120 fix. The full target model retains its stored numerical values; the PR's existing speculative draft-head transformation is unchanged. The initial qualification established hardware compatibility. The subsequent decode-grouping experiment below measures a small runtime-setting gain without editing upstream kernels.

## Upstream behavior already adapting to these cards

- Both GB10 and these cards are Blackwell. The current weight-only paths reconstruct weights and use BF16 activation arithmetic; `qmm_frag.cuh` emits BF16 MMA. Switching to quantized FP4 activations to chase an advertised peak would be a numerical change, not a free SM120 optimization.
- `cuda/experts.zig` computes NVFP4 decode residency from CUDA-reported occupancy multiplied by the detected SM count. It already uses our 188 SMs; it is not universally limited to a GB10-sized grid.
- `cuda/qmmf.zig` enables thread-block clusters on compute capability major >=9, including SM120. A comment mentioning GB10 does not mean this path is disabled on our GPUs.
- `native/cuda_memory.zig` distinguishes discrete from unified memory. The discrete default is `max(1 GiB, total/32)`; the unified-memory default is `max(4 GiB, total/10)`.
- `families/glm/cuda_q4.zig` enables one draft-head PDL path specifically on SM121; that path is off on SM120. In contrast, the BF16 `cuda_mm.zig` PDL default is already on. These are separate dispatch decisions, not a single global PDL switch.

## Tuning candidates and qualification status

| Candidate | Source / observation | Required experiment |
|---|---|---|
| NVFP4 decode warp grouping | `cuda/experts.zig` supports `TF_NVFP4_DECODE_WARPS=1` or `4`; upstream default 4, now recipe default 1. Each warp owns independent tiles. Routed experts dominate our sampled verify passes. | Completed: small measured decode gain with arithmetic, quality and cache checks; see the experiment below |
| Expert prefill tile geometry | `prompt_wn={8,4,8}` is explicitly described as GB10's fastest tiles in `cuda/experts.zig` | Screened matching compiled variants on both GPUs and in the full model: wider tiles showed no meaningful served gain; retain original sizes |
| Projection split-K | `families/glm/cuda_mm.zig` and `cuda/qmmf.zig` use a fixed target of 192 column tiles times K slices | 192/384/768 numerical and served screen completed: larger targets changed outputs and lost long-prefill speed; retain 192 |
| Local communication / overlap | Actual path is PCIe P2P/IPC, topology `NODE` (same NUMA node, across host bridges). Upstream overlap pieces=3; the qualified recipe now selects 4. | Screen completed for overlap 1/2/4/6 and split/gather/ring; gather was slower, four pieces confirmed a 0.87–1.05% prefill gain and passed quality/cache qualification |
| NCCL cuMem defaults | `cuda_native.zig` defaults cuMem host/device allocation off, with a GB10 driver-pool explanation | Device/host flags screened separately: no clear gain; default flags remain off |
| Lossless resident scale compression | PR decodes CSF32 on the host and packs expanded scales for current GPU kernels | New loader/kernel support would be needed to retain compressed scales on device. Whole-checkpoint expansion is 8.1065 GiB, about 4.0533 GiB/rank under equal TP2; this is representation accounting, not measured recoverable VRAM |
| Host embedding lookup or sharded latent cache | Full BF16 embedding is resident on each rank; latent KV is replicated | Mapped BF16 embedding implemented in the isolated image: 1,208 MiB/rank saved with matching screen outputs, further qualification below. Latent-cache sharding remains unimplemented |

Our cards have published memory bandwidth of 1,792 GB/s versus DGX Spark's 273 GB/s, but only 96 GB of dedicated VRAM per card versus the referenced Spark's 128 GB unified memory. Bandwidth advantage does not establish an inference-speed multiplier or solve the capacity constraint. Sources: [RTX PRO Max-Q](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/), [DGX Spark](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).

## RTX PRO tuning scope and order

Here, adapting GB10 tuning means reviewing inherited scheduling, tile geometry,
communication and memory-layout decisions against our discrete SM120 cards. It
includes choices without a GB10 label. The goal is measured improvement on our
hardware with preserved correctness and task quality.

1. **Expert decode grouping:** compare the existing 4-warp default with
   `TF_NVFP4_DECODE_WARPS=1`. This is the first bounded experiment because local
   profiling identifies routed experts as a substantial decode cost, and both
   variants already exist upstream. Completed below: one warp gave a small
   consistent decode gain and passed the quality/cache checks.
2. **Expert prefill tiles:** sweep suitable compiled tile shapes around
   `prompt_wn={8,4,8}`, whose source explicitly identifies GB10 tuning. Change
   host launch geometry and matching device kernel instantiations together.
   Measure short and long prefills separately from decode.
3. **Projection split-K:** review the fixed target of 192 in GLM BF16,
   quantized projection and draft-head paths. Derive candidates from actual
   matrix shapes, residency and our 188 SMs; multiplying the target by an SM
   ratio is not sufficient. Additional splits add reduction work and can
   alter floating-point rounding.
4. **PCIe communication overlap:** benchmark the current overlap setting and
   NCCL choices using the verified local P2P/IPC path. Check end-to-end time,
   memory and stability rather than isolated transfer bandwidth alone.
5. **1M capacity:** separately investigate resident compressed scales,
   embedding placement and latent-cache sharding. The measured capacity gap
   requires actual allocation savings; kernel scheduling changes do not
   inherently provide those savings.

Keep the detected-SM occupancy logic and valid architecture dispatch unless an
experiment provides evidence to change them. Evaluate NCCL allocation defaults
and SM121-only launch paths individually; a GB10 origin alone does not establish
that a setting is unsuitable for SM120.

For each candidate, preserve a baseline image and change one variable. Record
source/binary identity, settings, exact workload, cold versus warm cache state,
GPU memory, prompt and decode timing, and repeated measurements under comparable
thermal conditions. Leave clocks and power limits unchanged. Run relevant
arithmetic/loader gates and draft/plain consistency checks before the fixed
[quality benchmark](../quality/README.md); compare against the unchanged
TensorFold baseline as well as the existing vLLM reference. Repeat long-context
qualification for changes affecting allocation or attention. Keep only gains
that exceed run-to-run variation without an unexplained correctness regression.

**Status:** one decode warp and four overlap pieces are the qualified recipe defaults. The subsequent
18-run prefill, split-K, communication and placement screen is recorded in
[TUNING-RESULTS.md](TUNING-RESULTS.md). Its source changes live in a separately
pinned experimental image; the qualified image remains unchanged. No broad
scheduling gain was established by the initial screen. Mapped embedding saves
1,208 MiB per rank; qualification and any confirmed overlap result are recorded
in that results document.

## Measured profiling evidence

At configured 128K, one slot, FP8 KV, no retained cache, profiling enabled:

- Representative 64-call summaries for 3–4-row verify passes: GPU 15.18–16.21 ms/pass; routed experts 6.28–7.07 ms, KDA 3.00–3.12 ms, MLA 1.85–1.94 ms, gather A+B approximately 1.08–1.47 ms.
- This suggests prioritizing expert kernels for this short decode workload. It does not establish the bottleneck of all long-prefill or concurrent workloads.
- Do not interpret the coarse prompt summary's `router` field as isolated router cost: overlapped prompt work has additional stage traces, and their intervals can overlap. Some two-rank log lines interleave.
- The bounded profile workload submitted 8 x 256-token requests. It was started while the last long-context request was finishing; its client wall time is unsuitable as a clean throughput benchmark. The per-kernel event evidence remains diagnostic.
- Raw evidence: `logs/tensorfold/context-128k-server.log`, `context-128k-profile-bench.json`, `context-128k-memory.csv`.

## Context ladder

Configured context alone is not an exercised window. Each retrieval prompt hides independent values near 10%, 50%, and 90% of its records, checks the server's exact rendered token count against usage, requires a cold cache, and compares generated token hashes with drafts on/off. This synthetic workload is not a general long-context accuracy benchmark.

| Configured window | Prompt cache / reserve | Actual prompt tokens | Result |
|---|---|---|---|
| 8,192 | 1 GiB / native default | 7,481 | Initial single-value retrieval passed; 4 API checks, draft/queue consistency and repeated-prefix test also passed |
| 131,072 | 0 / native default (~3 GiB) | 8,182; 32,764; 130,812 | All three values recovered at every length; draft/plain token hashes identical; 4 API checks passed |
| 262,144 | 0 / 1.5 GiB | 261,888 | All three values recovered; draft/plain token hashes identical; 4 API checks passed |
| 524,288 | 0 / 1.5 GiB requested | 524,027 | All three values recovered; draft/plain token hashes identical; 4 API checks passed; very little physical headroom |
| 1,048,576 | 0 / 1.5 GiB | None | Startup failed with `cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY`; no service readiness or 1M inference |

128K startup: 164.3 s, reported model 90.66 GiB, sequence state 1.03 GiB. Minimum sampled free memory: 3,153 / 3,306 MiB. The near-window cold requests took 31.38 / 32.13 s with drafts on/off under profiling. A memory sample is not a continuous allocator trace.

256K startup: 176.8 s, reported model 91.58 GiB, sequence state 1.88 GiB. Minimum sampled free memory: 2,215 / 2,368 MiB. The near-window cold requests took 64.09 / 66.22 s with drafts on/off, with profiling disabled.

512K startup: 174.3 s, reported model 93.41 GiB, sequence state 3.57 GiB. Minimum sampled free memory: **339 / 492 MiB**, despite the 1.5 GiB reserve setting. Near-window cold requests: 138.32 / 140.79 s with drafts on/off, profiling disabled. This passed bounded capacity testing; 256K has substantially more measured headroom and is the preferable tested working profile.

The reserve observation has a source explanation: GLM allocates its own sequence inside `Engine.init`, then declares `free_streams=1`; native admission accepts that already-allocated stream even when `Pool.room` has saturated to zero. The startup line explicitly reports 0.0 GiB left after the reserve. The reserve therefore is not a hard free-VRAM guarantee for that initial stream. This is an upstream-report candidate, not a claim that we fixed admission.

The 1M attempt ran from 01:47:40 to 01:50:41 UTC on 2026-10-24. Both ranks emitted `cuMemAlloc: CUDA_ERROR_OUT_OF_MEMORY`; the peer watchdog terminated the other rank and the container exited with code 1 (`OOMKilled=false`: this was a CUDA allocation failure, not the host OOM killer). The failure path also emitted a **136-byte SafeAllocator leak diagnostic**, with no usable stack trace. Its origin has not been diagnosed. The processes exited and the subsequent baseline preflight found the GPUs free; no persistent GPU allocation leak was observed.

The sampled peak during the failed 1M load is not a successful 1M footprint: a large allocation can fail while some VRAM remains free. Evidence: `logs/tensorfold/context-1m-{attempt.json,start.log,acceptance.log,server.log,memory.csv,memory-summary.json}`. No 1M request was submitted because readiness failed.

Full receipts belong in [VALIDATION.md](../docs/VALIDATION.md).

## Why 1M needs explicit capacity work

At this pinned source and model shape, context capacity is aligned to 64 after adding 8 verify rows. There are 11 backbone MLA caches plus MTP. With FP8 KV, each layer uses 512 bytes plus a 4-byte scale per latent row; its pooled index keys add 64 bytes/token. That is 6,960 bytes/token/rank, or about 6.80 GiB/rank at 1M, before fixed state.

The four forward scratch arenas add 536 bytes/token/rank in sparse score storage and the rank link adds another 4. For contexts above the fixed ring/dense thresholds, the identified allocation increase is **7,500 bytes per extra capacity token per rank**. This excludes allocator rounding, profiler/driver allocations, and any new feature. The observed 128K-to-256K increase is 938 MiB/rank, consistent with the predicted 937.5 MiB plus rounding. Moving from 128K to 1M therefore adds about 6.41 GiB/rank, exceeding the observed ~3.1 GiB free at 128K. Even a zero reserve would not close that gap by itself.

[PR #1163](https://github.com/ashhart/TensorFold/pull/1163) and [#1211](https://github.com/ashhart/TensorFold/pull/1211) improve allocation sharing across streams, but their current GLM implementation keeps FP8 caches on windows. Their reported 1M configuration is TP3 on another checkpoint, not a near-1M prompt qualification of our TP2 FP4-Local setup. They are not included in this image.

## Reproduction

With the intended GPUs free:

```bash
./tensorfold/build.sh
TF_NVFP4_DECODE_WARPS=4 TF_CONTEXT=131072 TF_PROMPT_CACHE_GIB=0 TF_PROFILE=1 TF_NCCL_DEBUG=INFO ./tensorfold/start.sh
./tensorfold/smoke.sh --wait 1200 --suite --output logs/tensorfold/recheck-128k-api
python3 tensorfold/scripts/context_check.py --targets 8192 32768 130816 --compare-plain --output logs/tensorfold/recheck-128k
# Save complete Docker logs before stopping; stop removes only this owned TensorFold container.
./tensorfold/stop.sh
```

Set `TF_PROFILE=0` for throughput measurements. Set `TF_MEMORY_RESERVE_GIB` explicitly when reproducing a reduced-reserve profile. Never infer it from vLLM's GPU-memory fraction. Preserve the checkpoint, engine tree, binary hash, launch JSON, request hashes, responses and memory sampling together when preparing an upstream report.

## Accuracy evaluation beyond these checks

TensorFold has NVFP4 CUDA paths. The original failure was integration of FP4-Local's mixed recipes and CSF32 scales, not lack of NVFP4 support in principle. PR #1050 connects those readers to this GLM engine. Numerical kernel tests, selected real-tensor upload checks, speculative/plain equality and synthetic retrieval are necessary evidence; none is a broad model-accuracy score.

Two external harnesses are appropriate next qualification tools:

- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md): its local OpenAI-compatible adapters can exercise both services. `gsm8k_cot` measures generated-answer math accuracy, and `ifeval` measures verifiable instruction following. Likelihood-based MMLU requires a completion endpoint with compatible prompt log-probabilities; the chat adapter alone is insufficient. That API compatibility must be checked before claiming support for those tasks.
- [NVIDIA RULER](https://github.com/NVIDIA/RULER): configurable long-context lengths and 13 tasks across four categories, extending beyond our three-value recall check. Use fixed generated examples and seeds at each successful ladder rung, and record task scores by length rather than treating a configured limit as effective context. The stock published table through 128K is not evidence of our model's accuracy at 1M.

A defensible engine comparison pins the same NVIDIA revision, evaluation harness revision, dataset revision/split, sampled item IDs, tokenizer, rendered prompts, thinking setting, greedy/sampling parameters, output allowance and scoring/extraction rules. Start at a comfortable common context and one active request, then qualify longer contexts separately. Record per-item answers, errors, truncation and confidence intervals; do not convert HTTP success into an accuracy pass. Different valid engine arithmetic can change generated text, so compare task scores as well as targeted numerical evidence.

These harnesses were researched but not run during the capacity ladder. Subsequently, the [initial quality baseline](../docs/QUALITY-BASELINE.md) ran 128 GSM8K and 128 IFEval items on both engines with matching prompts and tokenizer IDs. MMLU and RULER remain unrun. A comparison to an unquantized ZAI checkpoint would assess a different question (quantization quality) and must not replace the same-checkpoint TensorFold/vLLM comparison.

## Capacity engineering estimate for the 1M target

From the successful 512K run, another 524,288 capacity tokens require about 3.6621 GiB/rank in the identified allocations. GPU 0 had only 339 MiB free. The projected deficit is therefore about **3.33 GiB/rank just to fit**, or **4.83 GiB/rank to retain 1.5 GiB free**, before new representation/transfer overhead. Lowering the reserve alone cannot supply physical memory. These are source-derived estimates, not a measured 1M resident footprint.

The representation accounting suggests lossless device-side CSF32 storage could be a substantial part of the solution (~4.05 GiB/rank under equal TP2), but it does not by itself promise a comfortable margin. The replicated BF16 embedding has 154,880 x 4,096 x 2 bytes, about 1.18 GiB per rank; an exact host-backed lookup is another possible saving, with PCIe cost to measure. A sharded live latent cache is an alternative structural route. Host-backed **live** KV would be a separate engine feature; changing the retained prompt-cache budget does not offload live KV. At the time of this ladder none of these changes was implemented. A later isolated mapped-embedding experiment is recorded in [TUNING-RESULTS.md](TUNING-RESULTS.md); resident scale compression and latent-cache sharding remain unimplemented.

Before a performance claim: measure the new resident layout, run the real-loader and arithmetic gates, repeat draft/plain and cache/cold checks, exercise a near-1M prompt, then score a fixed long-context suite such as RULER. Keep any reserve-admission fix separate from a speed or capacity optimization so its behavior is reviewable.

Final local state after this ladder: vLLM restored on port 8000, readiness and all four API checks passed. TensorFold stopped after preserving logs. No unrelated services, GPU power/clocks, model files or caches were changed. The recipe's 27 CPU checks passed; no upstream kernel edits or accuracy-harness scores are part of this change.

Subsequent quality qualification: the initial paired sample measured TensorFold at 120/128 GSM8K and 113/128 strict IFEval, versus vLLM at 119/128 and 112/128. These one-item differences do not establish an accuracy advantage. Exact protocol and uncertainty are in the linked quality baseline; no target precision or kernel changes were made for the run.

## Measured decode warp comparison — 2026-10-24 UTC

We compared the existing upstream 4-warp and 1-warp NVFP4 expert decode kernels
on this pinned SM120 image without rebuilding or changing precision. The order
was 4 → 1 → 4, at 16K context, one slot, FP8 KV, no retained prompt cache, native
default reserve and profiling disabled. Each run excluded six warmups and then
measured four repeats of three identical prompts with MTP on and off: 24 measured
requests per run, 72 overall. Every measured request generated exactly 256 tokens.
Prompt lengths were 42, 43 and 6,931 tokens. The cap is deliberate for this speed
workload; it is not an accuracy pass.

| Prompt | MTP | 4 warps before (tok/s) | 1 warp (tok/s) | 4 warps after (tok/s) | 1-warp gain versus both baselines |
|---|---|---:|---:|---:|---:|
| archive | off | 69.35 | 70.04 | 69.36 | +0.98–0.99% |
| archive | on | 126.70 | 127.83 | 126.48 | +0.89–1.07% |
| database | off | 73.83 | 74.34 | 73.86 | +0.65–0.68% |
| database | on | 144.11 | 145.17 | 144.13 | +0.72–0.73% |
| hash-table | off | 73.80 | 74.46 | 73.69 | +0.90–1.05% |
| hash-table | on | 145.83 | 147.68 | 146.15 | +1.05–1.27% |

These are medians of **server-reported decode rates**, which count output tokens
after the first token and exclude prefill. Full client latency and prefill times
are retained separately. This is a small observed gain (0.65–1.27%), not a broad
engine-speed ranking. The absolute change between the two 4-warp baseline medians
was at most 0.22%. All measured output hashes match across settings, repetitions
and drafted/plain modes; speculation counts also match between settings.

The 1-warp candidate passed `glm-ref`, `quant-ref`, `mxfp8`, `projection-f32` and
the selected real-checkpoint loader gate on both cards. Each rank had 115 direct
checkpoint-cut uploads and 72 host-packed uploads matching, with three excluded
uploads and zero unknown/mismatched uploads. The 4-warp reference additionally
passed `quant-ref` on both cards. All three served runs passed the four API checks.

The candidate also recovered all three hidden values at 8,182 and 16,125 actual
prompt tokens with matching drafted/plain hashes. Its complete fixed quality run
matched **all 256 original TensorFold output token hashes and every official
score**: GSM8K 120/128, strict IFEval 113/128, loose IFEval 114/128. No truncation
or missing reasoning closure occurred. The same original quality client, task
and dataset pins, prompts, tokenizer IDs and image were verified. The newly
exposed warp setting is recorded alongside the quality manifest in
`w1-quality/experiment-settings.json`.

One-second telemetry during measured requests found the same minimum free memory
in all three runs: 3,975 / 4,128 MiB. Thus this experiment demonstrates no capacity
saving toward 1M. GPU 0 median SM clock was 2,317 MHz in every run; GPU 1 medians
were 2,283.5 / 2,287 / 2,280 MHz. Automatic clocks and temperatures varied; no
power or clock controls were changed. The full quality run occurred between the
candidate timing run and the returning baseline. The repeated baseline helps
expose drift but cannot eliminate every thermal or temporal confounder.

MTP was already active, using the checkpoint's own draft component. On these
prompts it provided roughly 1.8–2.0× the plain decode rate; the warp change adds
only the much smaller percentage above. Its draft acceptance rates were about
64.9–73.1%, with 2.10–2.31 output tokens per verification round. No external
DFlash model was loaded.

Raw evidence: `logs/tensorfold/decode-warps-v1/`, including per-request responses,
configuration manifests, `performance-comparison.json`, `quality-regression.json`,
GPU telemetry and full server logs. The arithmetic/loader suite is under
`logs/tensorfold/gpu-check-20261024T023808Z/`. These bounded checks do not qualify
long-context task accuracy, multi-slot performance or all prompt/cache patterns.

### Adoption and rollback

The recipe now defaults to `TF_NVFP4_DECODE_WARPS=1`. After the speed/quality
comparison, a separate normal-profile run (8K context, one slot, 1 GiB prompt
cache, native reserve) passed all four API checks and cold/warm, drafted/plain
retrieval at **3,648 and 7,993 prompt tokens**. The repeated requests reused
**2,048 and 6,144 tokens**, respectively. All four output hashes at each length
were identical and all three hidden values were recovered. Evidence:
`cached-profile/report.json` and `cached-profile-{api,check,server}.log` under the
experiment directory.

This is selection of an existing upstream kernel variant for our SM120 cards,
not a kernel rewrite or a precision change. The retained cache and memory
budgets are unchanged. Set `TF_NVFP4_DECODE_WARPS=4` to restore the upstream
setting without rebuilding. Historical 128K/256K/512K ladder results used four
warps, and their reproduction commands keep that setting explicit. The one-warp
choice has not been requalified at those lengths or at multiple active slots.
The 1M allocation failure remains unresolved.

Portable settings, scores and measurement summaries are in
[results/decode-warps-v1.json](results/decode-warps-v1.json). Full responses remain
in the ignored local evidence directory.

## Qualified four-piece overlap and optional mapped embedding

The [complete tuning record](TUNING-RESULTS.md) covers 18 initial profiles, a
five-repeat control/candidate/control confirmation, full 256-item quality
reruns, fixed-history probabilities, normal-cache checks and a 512K retrieval
ladder. The normal-image overlap default changes from 3 to 4; rollback is
`TF_GLM_OVERLAP_PIECES=3`. Wider expert prefill tiles, larger split-K targets,
PDL/KDA changes, alternate collectives and cuMem flags were not adopted.

The separate mapped-embedding profile saves 1,208 MiB per GPU and passed near
512K with 1,547 / 1,700 MiB measured free, at a small prefill cost. It remains an
explicit experimental image/profile, not the default weight placement. The
qualified image, checkpoint, precision, GPU clocks/power and vLLM memory budget
remain unchanged. The 1M target still needs additional engine allocation savings.

## Consolidated contribution record and next scope

See [PR1050-SM120-BRIEF.md](PR1050-SM120-BRIEF.md) for the dated, evidence-linked record of adopted RTX settings, experimental source patches, retained GB10-associated choices and unrun platforms. The [compressed-scale plan](COMPRESSED-SCALES-PLAN.md) scopes the next TensorFold-only loader/kernel work. Nothing has been posted upstream. The later [resident-CSF32 implementation](csf-resident/RESULTS.md) now passes the near-1M prompt ladder, with 1,693/1,846 MiB sampled free and unchanged 256-item quality hashes. The earlier allocation failures and projections above remain historical evidence for the original images.
