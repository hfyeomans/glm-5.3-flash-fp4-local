# DFlash2 GPU qualification

[Measured results](RESULTS.md).

Use the pinned NVIDIA target and Inco AI draft from
[the feasibility review](../../docs/DFLASH2-FEASIBILITY.md).
The initial comparison uses 128K / one sequence / 3 GiB KV per GPU.
Capacity profiles passed near-1M retrieval and four concurrent 262K windows. Target attention stays
B12X; the draft uses Triton because this image's regular B12X backend rejects
256-token pages and replicated DCP KV.

## Capacity targets

Measured capacity profiles:

- **262,144 tokens × four concurrent requests**, with measured token-stream overlap.
- **1,048,576 tokens × one request**, including output tokens in the window.

Start at 98.5%; 99% is authorized if needed. Tune the explicit per-GPU KV budget
alongside draft/state/workspace allocation. Record memory minima, preemption,
queueing and failures at each step. Qualify concurrent full windows with measured token-stream overlap.

## Matched performance screen

The client uses the existing frozen workload format and explicitly records
`mtp`, `dflash2` or `none`. It checks container ownership, image/checkpoint/draft
pins, endpoint port, input IDs, cold-cache accounting and isolated server timing.
It samples GPU memory and stops issuing requests below 1,536 MiB free/card.

```bash
python3 experiments/dflash2/bench.py --mode dflash2 \
  --workloads /path/to/frozen/workloads.json \
  --output logs/dflash2/my-dflash2-256
# Use a fresh output directory for --output-tokens 1024.
# Relaunch the same probe profile with DRAFTER=mtp or DRAFTER=none,
# then repeat with the matching --mode and fresh output directories.
```

Default cases: 8K and 32K input, 256 output, one warmup and three measured repeats.
The baseline MTP harness remains restricted to MTP/plain launches. Reuse the
original frozen workloads for comparable input IDs; preserve every raw result.

[TensorFold work plan](../../tensorfold/DFLASH2-PLAN.md).

## Capacity client

Select the [1M/one-sequence](vllm-1m.env.example) or
[262K/four-sequence](vllm-262k-x4.env.example) settings in this worktree's
`.env`, then restart. Read [the results](RESULTS.md) before choosing a profile.
The probe's name/port differ from the normal MTP service; release its GPUs first.

```bash
# Single near-1M retrieval screen, with a recorded 1 GiB stop floor.
python3 benchmarks/concurrent_context.py --drafter dflash2 \
  --container glm53-fp4-dflash2-probe --base-url http://127.0.0.1:8002 \
  --window 1048576 --clients 1 --min-free-mib 1024 \
  --output logs/dflash2/my-1m-retrieval
# For the four-stream profile: --window 262144 --clients 4.
# Sustained decode: add --sustained --output-tokens 16384 and use a fresh output.
```

This client needs the existing tokenizer environment (transformers), such as the
main worktree's `.state/quality-venv/bin/python`. All windows include the output
budget. Concurrent capacity requires `overlapping_decode_without_preemption=true`
plus successful completion/retrieval and isolation checks in the report.
