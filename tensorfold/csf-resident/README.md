# Resident CSF32 experiment for RTX PRO 6000 Blackwell

This isolated TensorFold patch retains the NVIDIA checkpoint's compressed block
scales on the GPUs. It is a memory optimization with an expansion cost, not a new
quantization method. The normal TensorFold recipe and vLLM image remain unchanged.
The engine is TensorFold, not TensorFlow.

## Pins and scope

`profiles.json` pins the parent image, source tree, patch and built image ID.
The patch applies after `../tuning/sm120-experiments.patch`, on v1.1.3 plus PR #1050.
Checkpoint: `nvidia/GLM-5.3-Flash-FP4-Local`, revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, original mixed ModelOpt/CSF32 format.
Reuse `HF_CACHE_ROOT=/models_engrams/huggingface`; no checkpoint conversion or token
is needed for the existing local snapshot. Model mounts are read-only.

Qualification targets two SM120 RTX PRO 6000 Blackwell Max-Q cards, TP2, one serving
slot, no external draft model or vision, native MTP unchanged. CUDA resource records
were compiled for SM80/86/89/120/121, but only SM120 hardware is being tested.
This does not establish a universal replacement for GB10/DGX tuning.

## Storage and lifetime

`TF_TUNE_SCALE_LAYOUT=0` preserves legacy interleaved code/scale packing.
Mode 1 separates unchanged FP4 codes and expanded scales, isolating layout effects.
Mode 2 stores rank-local compressed components plus 40-byte descriptors, with
144 MiB gate/up and 72 MiB down scratch shared across all 42 routed layers.
Prompt launches expand all experts; decode launches expand active expert groups.
Expansion runs on the consumer stream without CPU routing readback or hot allocation.
The separate scratch buffers avoid aliasing across up/down consumers. MTP experts
remain on their original plain NVFP4 path. Mode 2 refuses unqualified multi-slot use.

The CPU codec remains a structural validator and independent byte oracle. TP slicing
retains original bases, nibbles and exception records, rebuilding compact u16/u32
indexes. GPU expansion uses integer operations and writes the existing scale swizzle.
BF16 activations, FP32 accumulation, calibration values and compute/reduction order
are preserved. This is W4A16, with exact storage reconstruction rather than A4.

The actual loader ledger requests about **3.81765 GiB less per rank**, including
216 MiB scratch, alignment and descriptors. That figure is not an allocator-peak
measurement or proof that a 1M request succeeds.

## Reproduce

Build from the already pinned tuned parent image:

```bash
./tensorfold/csf-resident/build.sh
python3 tensorfold/csf-resident/run.py resident-quality gpu-check
python3 tensorfold/csf-resident/run.py resident-quality start
python3 tensorfold/csf-resident/run.py resident-quality smoke --wait 1200 --suite
python3 tensorfold/csf-resident/run.py resident-quality stop
```

Run serialized served qualification (requires the owned TensorFold container to be
absent and the two GPUs available):

```bash
.state/quality-venv/bin/python tensorfold/tuning/sweep.py \
  --profile-file tensorfold/csf-resident/profiles.json \
  --profiles resident-quality --mtp both --repeats 3 \
  --cases short-code short-prose prefill-2048 prefill-8192 \
  --teacher-histories logs/tuning/teacher-histories-v1.json --quality \
  --output logs/csf-resident-v1/quality-v1
```

Profiles include matched legacy/separate/resident performance controls, cache tests,
and 512K/640K/768K/896K/1M capacity stages. Configuration is not validation: advance
only after output checks and measured free memory meet the 1.5 GiB target.

## Evidence and failure record

The [loader receipt](../results/csf-resident-loader-v1.json) records exact GPU/CPU
agreement for all 36,288 expert matrices per rank (19,025,362,944 scale bytes across
two ranks), raw code/calibration upload checks and allocation requests. Three
pre-existing draft-head quantizer upload exclusions remain explicit.

The separate-layout native contract passed all 14 cases on each GPU. The CSF32
contract passed nine shapes per GPU, including all byte values, TP cuts, partial
activity, duplicate/empty groups and reused scratch. Host tests, upstream host
contract/golden/fuzz checks, plain build, and the lean check passed (zero new lean
findings; 438 inherited). Full GPU and served-model qualification are recorded in
[VALIDATION.md](../../docs/VALIDATION.md) as they complete.

Preserved local failures include a test callback lifetime bug, an insufficiently
sensitive stress fixture, and a prebuilt-image full-suite attempt missing an
unrelated H3 asset. The callback/fixture were corrected without relaxing numerical
gates; the full suite was rebuilt with complete CUDA assets. Raw logs remain under
`logs/csf-resident-v1/`, outside Git. No upstream publication has been made.


## Patch ownership and further optimization

| Files in the pinned native tree | Change |
|---|---|
| `core/nvfp4_csf32_slice.zig` | Validated aligned TP slices with compact index rebuilding |
| `families/glm/cuda_expert_csf.zig`, `cuda_expert_separate.zig`, `cuda_weights.zig` | Persistent storage ownership, upload ledger and shared scratch |
| `cuda/nvfp4_csf32.zig`, `nvfp4_separate.zig`, `experts.zig` | Typed descriptors, ordered expansion and separate consumers |
| `kernels/cuda/nvfp4_csf32.cuh`, `nvfp4_separate.cuh`, `nvfp4_experts.cu` | Exact integer expansion and unchanged expert math with separate scale reads |
| `cuda_engine.zig`, `cuda_ranks.zig`, `sm120_tuning.zig` | Scope refusal and rank-setting agreement, fingerprint layout 13 |
| Native CUDA contracts, loader oracle and resource registry | Exactness, bounds, accumulation and launch-resource evidence |

A future direct compressed-scale consumer could avoid expansion writes/reads and
possibly scratch memory. It would need a separate profile, operator contracts and
end-to-end qualification. The current implementation deliberately measures the
simpler reusable-scratch design first. No A4 arithmetic, new split-K reduction order,
GB10 clock policy or DGX transport assumption is introduced by this storage patch.


Once the 512K rung passes with measured headroom, the one-slot 1M profile can allocate
its final pool once and then test progressively longer requests:

```bash
.state/quality-venv/bin/python tensorfold/tuning/sweep.py \
  --profile-file tensorfold/csf-resident/profiles.json \
  --profiles resident-1m --skip-benchmark --minimum-free-mib 1536 \
  --context-targets 655104 786176 917248 1048320 \
  --context-stage-timeout 7200 --output logs/csf-resident-v1/capacity-ladder-v1
```

This tests four prompt lengths in an already allocated 1M pool; it does not claim
four different allocation configurations. The harness checks all sampled free-memory
values after smoke and each rung, including startup peaks, and stops before the next
rung if either card misses the floor. Sampling is once per second, not a guarantee
against shorter transient peaks. Each rung checks three markers, server token counts,
cold cache, finish reason and MTP/plain output identity.
