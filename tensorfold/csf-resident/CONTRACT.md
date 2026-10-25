# Local resident-scale implementation contract

User authorization on 2026-10-24 covers local implementation and GPU experiments,
including stopping this recipe's vLLM service. No upstream publication is requested.
This extends the pinned PR #1050 integration in an isolated checkout, with credit
to its author.
ATC's broad snapshot has incomplete file coverage; the scoped #1015 snapshot is
complete and `takenOutside` through #1050. No claim or public session was created.

Baseline: experimental tree `0c2bc6ddf016c0d96d63fbd7d1238b6355603578`, native
v1.1.3 plus PR #1050 and the already qualified local tuning patch. Exact NVIDIA
checkpoint revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, CSF32 mixed format,
SM120 TP2, CUDA 13.3, BF16 activations and FP32 accumulation. Other GPU classes
are unavailable. No checkpoint writes, precision changes or clock/power changes.

Stage one separates persistent code and expanded-scale storage, retaining the
same tile swizzle and arithmetic and leaving plain MTP experts on the legacy path.
It is expected to save no scale payload memory by itself. Stage two replaces
persistent expanded scales with compact CSF32 plus bounded expansion scratch.
See ../COMPRESSED-SCALES-PLAN.md for owners, memory estimates and acceptance gates.

Correctness must be byte-exact to the legacy operators and model baseline. Register
new operators in the native GPU contract and audit actual compiled launch limits.
Run host, real-loader, GPU and model gates; measure end-to-end speed and cold
prefill with fixed prompts. Preserve failed and unrun checks.
