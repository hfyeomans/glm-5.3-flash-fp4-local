# Swarm code review — 2026-10-25

Reviewed current main **`17d312e`** against the supplied findings from `4153ef5`.
Finding #1 was already fixed; #7 was partly fixed. The remaining defects were
reproduced and corrected on **`fix/swarm-review-20261025`**.

## Workflow

Three isolated validation/fix lanes covered lifecycle/model checks, measurement/
reporting, and TensorFold telemetry/numerics. They inherited the session model
with high reasoning effort. An independent **GPT-6.1 Sol / high** pass reviewed
the commits; **GPT-6 Luna / low** handled the mechanical audit and a cleanup-message
fix. Integration used separate commits and a final combined check.

Each fix started with a failing fixture or mocked invocation. The independent
pass caught a remaining automatic model-discovery mismatch. A targeted follow-up
also caught readiness accepting a changed port on an already-running container.

## Supplied findings

| ID | Status at review start | Resolution and evidence |
|---|---|---|
| #1 | Already fixed | `benchmarks/run.py` already imported `sys` and defined `ROOT`; valid invocation regression now reaches measurement. |
| #2 | Confirmed | vLLM clients read the effective port from `Cmd --port`; valid launches without `PORT` in Env pass. Quality checks it before HTTP. |
| #3 | Confirmed | Reports enforce image identity within each backend; synthetic vLLM/TensorFold comparisons with distinct images pass. |
| #4 | Confirmed | Reports permit only the MTP option/value difference between on/off commands; other settings and differing MTP configurations still fail. |
| #5 | Confirmed | Delayed telemetry failure and unfinished threads fail the receipt; cleanup waits beyond the query timeout and halts further profiles if telemetry remains alive. |
| #6 | Confirmed | Startup derives its endpoint from HOST/PORT, checks container state around readiness, and verifies an already-running container's launched port before HTTP. |
| #7 | Partly fixed | Concurrent-context port binding already existed. Decode now rejects remote hosts; concurrent/DFlash2 guards also use HTTPS's effective default port. |
| #8 | Confirmed | Resumes and reports validate all completed receipts, including disjoint profiles, against one protocol/image; image drift fails before launch. |
| #9 | Confirmed | The Zig comparison rejects nonfinite operands and subtraction overflow across both output arrays. Original reduction fails the CPU regression; replacement passes. |
| #10 | Confirmed | NVIDIA and DFlash2 validation reject absolute internal symlinks, including intermediate directories; relative Hub links and host cache-root aliases pass. |
| #11 | Confirmed | Per-shard header, tensor names, offsets and exact physical extent catch one-byte truncation despite aggregate header slack. |
| #12 | Confirmed | Stop and startup distinguish explicit absence from inspect failures; permission/daemon errors return failure. |
| #13 | Confirmed | Malformed tool arguments become failed checks; streaming continues and `checks.json` is written. |
| #14 | Confirmed | Planning and inspection share mount resolution and candidate discovery, including automatic local folders and ambiguity; absent weights still permit planning. |
| #15 | Confirmed | Requested snapshot revisions retain their identity. Manual directories without revision evidence report an unverified revision and preserve `requested_revision`. |

Additional reproduced cases fixed: shard/index tensor mismatches, forbidden tool
argument properties, missing cache telemetry, early benchmark finishes, duplicate
repeat IDs, omitted report cells, and cleanup failures bypassing telemetry shutdown.
Failed cleanup also no longer prints a successful stop message.

## Validation

**130 CPU tests passed**, plus shell lint and Python syntax checks. The suite
included three compiled Zig 0.17.0 tests of the exact production agreement helper.
The compiler was copied read-only from an existing stopped build container.

Read-only inspection of the real NVIDIA cache passed all 42 shards and
177,463,841,024 physical bytes. DFlash2's pinned metadata/header and 2,342,169,800-byte
size passed. These initial checks used no GPU inference or container lifecycle
actions. Subsequent vLLM GPU qualification is recorded in
[VALIDATION.md](VALIDATION.md#swarm-vllm-gpu-qualification-2026-10-25).
Historical receipts are unchanged. The follow-up passed 136 CPU tests and all
four MTP/DFlash2 capacity profiles (262K × four and 1M × one). See the linked
validation record for overlap, memory minima and the retained shorter MTP attempt.

Regression suites:

- [Lifecycle/model/API](../tests/test_recipe.py) and [draft mounts](../tests/test_draft_model.py).
- [Endpoint and report compatibility](../tests/test_measurement_regressions.py).
- [Telemetry/resume/report integrity](../tests/test_tuning_sweep.py).
- [Compiled numerical gate](../tests/test_fp64_agreement.py).

```bash
# Run in the review worktree. ZIG enables the compiled CPU numerical checks.
ZIG=/path/to/zig-0.17.0/zig ./scripts/check.sh
```

Without `ZIG` or a compiler on PATH, the numerical test is explicitly skipped.
The full CUDA numerical probe remains pending.
[TensorFold maintainer note](../tensorfold/FP64-AGREEMENT-FIX.md).
