# Code and security review — 2026-10-25

Reviewed current main `022f7eb` with four independent read-only lanes: lifecycle/ops
shell, Python measurement tooling, native probes and build inputs, and a security
lane. Nine findings came back; seven were fixed on `fix/review-hardening-20261025`,
one was dismissed with recorded evidence, one is partially addressed with the
remaining default left as a documented operator choice. The baseline suite was
144 CPU tests; the branch passes 149.

## Dispositions

| ID | Lane | Severity | Location | Disposition |
|---|---|---|---|---|
| S1 | security | medium | `tensorfold/tuning/sweep.py:43` | Fixed |
| P1 | python | medium | `quality/report_long.py:52` | Fixed |
| P2 | python | low | `quality/long_context.py:60,166` | Fixed |
| A | lifecycle | low | `stop.sh:15`, `start.sh:60` | Fixed |
| B | lifecycle | low | `start.sh:55` | Fixed |
| S2 | security | low | `scripts/common.sh:36` | Partial (log rotation plus warning) |
| S3 | security | low | `.gitignore` | Fixed |
| C1 | native | low | `tensorfold/tuning/prefill_tiles.cu:40` | Dismissed with evidence |
| C2 | native | low | `tensorfold/tuning/Dockerfile` | Fixed |

## Fixes

**S1 — host code execution from an unpinned container fixture.** `--cancel-check`
copied `serve_cancel_release.py` out of the running TensorFold container, rewrote one
byte string, and executed it on the host with no identity check; a compromised
container could serve arbitrary code that the host then ran. `prepare_cancel_fixture`
now pins `EXPECTED_CANCEL_FIXTURE_SHA256` (the hash recorded by the
`csf-resident-cancellation-v2` receipts) and refuses any other source before
compiling or executing. New tests cover both the pin and the preserved adaptation
behavior.

**P1 — reporter hardcoded the memory floor.** `report_long.py` re-checked headroom
against a literal `1536` MiB instead of the run's recorded `minimum_free_MiB`, so a
run with a lower declared floor could pass its own gate and fail the report, and a
higher floor was misreported in the receipt. The reporter now reads the floor and
devices from the run manifest and formats the floor value into the report sentence.

**P2 — headroom devices hardcoded to `0,1`.** The telemetry gate in
`long_context.py` and both reporter gates sampled fixed devices while the sweep
gate correctly used `TF_GPUS`. `long_context.py` now takes `--devices` (the sweep
passes `TF_GPUS`; standalone runs default to every local GPU), records
`gpu_devices` in the manifest, and reuse refuses runs sampled on different devices.

**A — failed log capture aborted a successful stop.** `docker logs` capture in
`stop.sh` and in `start.sh`'s pre-removal path ran unguarded under `set -e`, so a
logging-driver or disk failure produced a raw docker error after the stop had
already succeeded, and skipped the following `docker rm`. Both captures are now
best effort (`|| true`), matching the existing post-failure capture in `start.sh`.

**B — crash-looping container blocked a plain `./start.sh`.** A container in
docker's `restarting` state reports `Running=false`, skipped the stop step, and
then failed `docker rm` without `-f`. The pre-removal stop now runs whenever the
container is not running, not only for `MODE=restart`; `docker stop` is idempotent
on already-stopped containers.

**S2 — unauthenticated API on a non-loopback bind.** `HOST` defaults to `0.0.0.0`
under `--network host`, documented in the README and `docs/VLLM-OPERATIONS.md` as
an operator choice, so the default itself is unchanged. Two concrete gaps are
closed: container logs now rotate (`--log-opt max-size=100m --log-opt max-file=3`,
matching the TensorFold launcher), and `start.sh` prints a warning when `HOST` is
non-loopback and `VLLM_API_KEY` is unset. TensorFold's `TF_HOST` default parity is
left open.

**S3 — clone hygiene.** `.gitignore` was ignored by itself, so fresh clones did
not ignore `.env`, `logs/`, `.state/`, editor backups, or `.netrc`. The file is now
tracked with the same rules minus the self-ignore.

**C2 — unpinned tuning parent image.** `tensorfold/tuning/Dockerfile` built on the
`1.1.3-pr1050-sm120-context` tag with no parent-ID check, unlike
`csf-resident/build.sh`. New `tensorfold/tuning/build.sh` pins the parent to the
image ID recorded in `docs/VALIDATION.md` (`sha256:c829536f…`) before building, and
`scripts/check.sh` now lints `tensorfold/tuning/*.sh`.

## Dismissed

**C1 — `shared()` exceeds its own 49152-byte threshold.** The stage-selection
fallback returns 53248 bytes for the `wn=8`/`wn=16` variants. The recorded probe
runs show those exact sizes launched and passing byte-equal on both GPUs
(`logs/tuning/tiles-gpu0.log`, `tiles-gpu1.log`, `ATTR down wn16 … shared=53248`,
all `PASS … byte_equal guards inputs`), so the constant is a selection heuristic,
not a live constraint, and the launch opts in through `cudaFuncSetAttribute`.
Changing the threshold would change which variants the probe measures and break
comparability with recorded receipts for no runtime gain.

## Bot review follow-up

CodeRabbit and Devin reviewed the PR before merge. Three findings accepted, one dismissed.

- Devin: the post-request physical headroom check took the minimum across every
  local GPU, so a valid run on selected GPUs could abort because an unrelated GPU
  was low. The check now covers only the selected devices and fails loudly when a
  selected index is missing.
- CodeRabbit: the reporter certified headroom from the manifest embedded in
  `results.json` without comparing the on-disk `long-context/manifest.json`. It
  now refuses a mismatch before certifying anything.
- CodeRabbit: `max-size` and `max-file` log options are only valid on the
  `json-file` driver. The launch now pins `--log-driver json-file` instead of
  relying on the daemon default.
- Dismissed: the docstring coverage check. The repo deliberately has no
  docstrings; that check predates this diff.

## Validation

`./scripts/check.sh` passes with 149 tests (baseline 144, five new), shell lint and
Python syntax pass, and the ZIG numerical gate is skipped as before (no compiler on
PATH). Each fix landed with a failing-first fixture: the cancel-fixture hash pin,
the reporter floor/devices gate, stop with failing log capture, start over a
non-running container, and the open-bind warning. No GPU inference ran in this
round; GPU qualification remains a separate step recorded in
[VALIDATION.md](VALIDATION.md).

## Compatibility notes

- Editing `quality/long_context.py` invalidates `REUSE_CLIENT_SHA256` and the
  recorded `long_context_client_sha256` protocol hashes. Completed historical runs
  stay valid; reusing their results under the new client requires re-qualification,
  and in-flight sweep resumes across this change fail closed by design.
- Long-context manifests now record `gpu_devices`. Re-reporting a pre-change run
  with `report_long.py` fails on the missing key instead of silently assuming
  devices `0,1`.
- The cancellation fixture pin is deliberately fail-closed: a deliberate image
  update that changes the upstream fixture requires consciously refreshing
  `EXPECTED_CANCEL_FIXTURE_SHA256`, the same discipline as the image pins.
- `--log-opt` rotation flags appear in new runs' `launch-command.txt`; they do not
  affect measurement, and mixed old/new comparisons were already refused by the
  protocol checks.
