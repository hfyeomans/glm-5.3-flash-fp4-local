# Maintainer note: matrix agreement can accept infinite disagreement

**The matrix-comparison gate can pass when one output is infinite and the other
is finite.** This review fixes the gate in our
[FP64 probe](tuning/glm_mm_bench_fp64.zig). TensorFold inference kernels and tuning
constants were unchanged.

## Upstream location and scope

Our probe is adapted from TensorFold's Apache-2.0 `zig/tests/cuda/glm_mm_bench.zig`.
The same reduction was verified in:

- [Pinned v1.1.3 source](https://github.com/ashhart/TensorFold/blob/36415df4f76e4458e82d0fa729ecb28e287ebf38/zig/tests/cuda/glm_mm_bench.zig#L74).
- [Upstream main](https://github.com/ashhart/TensorFold/blob/main/zig/tests/cuda/glm_mm_bench.zig#L74), read on 2026-10-25; file blob `3f65eaa4d4ba5e79c51965133b1a410a8811fec4`.

The comparison checks the `b16.matmul` and tiled `glm_mm.matmul` outputs.
Our added FP64 reference covers `gb`'s first three rows. The whole-output gate
must check both arrays for nonfinite values.

## Failing case

For `ga = [+Inf]` and `gb = [1]`, the old reduction computes:

```text
most = max(abs(ga - gb)) = Inf
big  = max(abs(ga))      = Inf
most <= big * 1e-3      = true
```

Thus a divergent result satisfies the agreement condition. NaNs and overflow in
`u - v` also need explicit rejection.

## Local fix

`productsAgree` now rejects empty/mismatched arrays, nonfinite values in either
array, and nonfinite differences before applying the existing relative tolerance:

```zig
if (!std.math.isFinite(u) or !std.math.isFinite(v)) return false;
const difference = @abs(u - v);
if (!std.math.isFinite(difference)) return false;
```

The `1e-3` agreement threshold, independent FP64 bound, matrix shapes, timing loops
and GPU kernels remain unchanged. The fix is in integration commit `1dcd1a0`.

## Reproduction and validation

The old reduction was extracted and compiled with **Zig 0.17.0**; the nonfinite
regression failed. The exact replacement helper passed three compiled CPU tests:

1. `NaN`, `+Inf` and `-Inf` in every position of either/both arrays.
2. Finite subtraction overflow, ordinary disagreement, empty and unequal arrays.
3. Equal finite values, signed zero and values inside the retained tolerance.

Run the host regression without CUDA:

```bash
ZIG=/path/to/zig-0.17.0/zig python3 -m unittest discover \
  -s tests -p test_fp64_agreement.py -v
```

The [test driver](../tests/test_fp64_agreement.py) compiles the production helper
and its Zig tests directly from the probe source. The full CUDA benchmark has not
been rerun with this fix; existing GPU receipts retain their original source hashes.

For upstream discussion, the proposed change is a finite-value guard and regression
cases for the benchmark's agreement oracle. The sampled FP64 reference added by
this recipe is a separate extension. No upstream issue or PR was opened.
