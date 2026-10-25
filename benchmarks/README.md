# Shared vLLM / TensorFold performance benchmark

See [the current measured comparison](../docs/PERFORMANCE-CURRENT.md) and the [original baseline](../docs/PERFORMANCE-BASELINE.md). This is a cold
single-request performance protocol, separate from the [accuracy suite](../quality/README.md).

```bash
# Uses the pinned tokenizer already in HF_CACHE_ROOT; no new model download.
.state/quality-venv/bin/python benchmarks/prepare.py --output logs/tuning/workloads.json

# Start the vLLM recipe with MTP_ENABLED=1, then benchmark the ready server:
.state/quality-venv/bin/python benchmarks/run.py --backend vllm --mtp on \
  --workloads logs/tuning/workloads.json --output logs/tuning/my-vllm-on
# Stop only this recipe, restart with MTP_ENABLED=0, wait for smoke acceptance,
# then repeat with --mtp off and a fresh output folder.

# For TensorFold: context 131072, one slot, retained prompt cache 0, profiling 0.
.state/quality-venv/bin/python benchmarks/run.py --backend tensorfold --mtp both \
  --workloads logs/tuning/workloads.json --output logs/tuning/my-tensorfold

python3 benchmarks/report.py logs/tuning/my-vllm-on logs/tuning/my-vllm-off \
  logs/tuning/my-tensorfold --output logs/tuning/my-comparison
```

Use three measured repeats for the published baseline protocol. Each case/mode
also has an excluded warmup. All requests have 256 output tokens, greedy
sampling, seed 1729, verified identical input token IDs and zero reused prompt
tokens. The raw prompt explicitly closes the reasoning prefix.

The client saves raw SSE frames, text, server timings, token counts, per-request
vLLM metrics and native output token hashes. It refuses concurrent/delayed vLLM
metrics that cannot isolate exactly one request. vLLM requires separate process
launches for MTP on/off; TensorFold supports a per-request draft switch.

Client TTFT means the first nonempty raw text frame. Decode excludes the first
token. Prefill timing boundaries are server-specific, so compare client TTFT as
well as each server's reported prefill throughput. Short-prompt prefill rates
include fixed overhead. Reports retain thermal drift, output variability and
context/slot settings.

After testing, restore the normal vLLM service with `MTP_ENABLED=1 ./start.sh`
and `./smoke.sh --suite`.

The original TensorFold baseline used three overlap pieces. Set
`TF_GLM_OVERLAP_PIECES=3` explicitly to reproduce it; the current qualified recipe
default is four. Exact settings and image IDs accompany every result.


For multiple active clients, use the [concurrent-context harness](CONCURRENCY.md).
[Measured results](../docs/VLLM-CONCURRENCY.md) cover independent requests through
two full 1M windows, including sustained generation and scheduler telemetry.
