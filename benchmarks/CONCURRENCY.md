# Concurrent long-context vLLM screen

`concurrent_context.py` tests independent active requests against the local vLLM
service. Its default MTP profile requires the historical 1M context/eight-slot
launch, GPU fraction 0.985 and 7,650,410,496-byte per-GPU KV budget.
Keep other client traffic off the endpoint during measurement.

## Run

Use the existing quality environment and downloaded tokenizer. Each run needs a
new output directory. The harness leaves server configuration and lifecycle alone.

```bash
MAX_MODEL_LEN=1048576 MAX_NUM_SEQS=8 MTP_ENABLED=1 ./start.sh

# Single near-1M control: 1,048,320 input tokens plus 256 output tokens.
.state/quality-venv/bin/python benchmarks/concurrent_context.py \
  --window 1048576 --clients 1 --min-free-mib 512 \
  --output logs/concurrent-context-v1/my-1m-control

# Pair ladder: 262144, 524288, 786432, then 1048576. Review each report.
.state/quality-venv/bin/python benchmarks/concurrent_context.py \
  --window 262144 --clients 2 --min-free-mib 512 \
  --output logs/concurrent-context-v1/my-262k-pair

# Hold both requests active: reserve 16K output tokens and ignore EOS.
.state/quality-venv/bin/python benchmarks/concurrent_context.py \
  --window 1048576 --clients 2 --output-tokens 16384 --sustained \
  --min-free-mib 512 --timeout 1200 \
  --output logs/concurrent-context-v1/my-sustained-1m-pair
```

`BENCH_API_KEY` or `VLLM_API_KEY` supplies optional authentication from the
environment. The 512 MiB setting is a **capacity-screen stop floor**. The initial
probe stopped at the default 1,536 MiB floor; vLLM workspace growth left
903/1,056 MiB free on this unchanged server budget.

## Matched MTP capacity profiles

For a separately launched MTP service with limits matching the test, opt in with
`--mtp-profile capacity`. This requires the pinned image and NVIDIA Hub snapshot,
loopback serving host/endpoint, MTP3 and target B12X kernels, TP2/DCP2, FP8 KV,
BF16 compute, ModelOpt mixed quantization, 256-token blocks, the same
0.985/7,650,410,496-byte memory budget, and launch limits exactly equal to
`--window` and `--clients`. The selected profile is recorded in `manifest.json`.

```bash
DRAFTER=mtp MAX_MODEL_LEN=262144 MAX_NUM_SEQS=4 \
  CONTAINER_NAME=glm53-fp4-swarm-probe HOST=127.0.0.1 PORT=8002 ./start.sh

.state/quality-venv/bin/python benchmarks/concurrent_context.py \
  --mtp-profile capacity --container glm53-fp4-swarm-probe \
  --base-url http://127.0.0.1:8002 --window 262144 --clients 4 \
  --sustained --output-tokens 32768 --min-free-mib 1024 \
  --output logs/concurrent-context-v1/my-mtp-262k-x4
# For the 1M/one-client profile, relaunch with MAX_MODEL_LEN=1048576 MAX_NUM_SEQS=1
# and use --window 1048576 --clients 1 with a fresh output directory.
```

The 32K output allowance kept all four MTP clients decoding together in the
[qualification run](../docs/VALIDATION.md#swarm-vllm-gpu-qualification-2026-10-25);
with 16K outputs, the first client finished before the fourth began decoding.

DFlash2 capacity runs select `--drafter dflash2`; see its
[capacity client commands](../experiments/dflash2/README.md#capacity-client).

## Protocol and results

Prompts use exact token IDs, distinct archive identities/filler, unique cache
salts, and three retrieval markers near 10%, 50% and 90%. The checkpoint's
low-reasoning template has its thinking prefix explicitly closed. Each input
contains `window - output_tokens` tokens. A process barrier synchronizes clients.

The harness saves image/checkpoint/tokenizer/client pins, compressed prompts,
raw streaming frames, timing and about one GPU/scheduler sample per second.
Sustained mode also checks streamed token IDs, including tokens with no visible
text, against final usage and the submitted prompt IDs.
Memory or timeout failures close only the test clients.

| Report field | Meaning |
|---|---|
| `capacity_passed` | Completed responses, exact input counts, zero cached tokens, isolated success counts, an idle server afterward and no runtime/headroom failure |
| `request_isolation_passed` | No other client's retrieval markers appeared in a response |
| `retrieval_all_passed` | All three markers returned by each client |
| `retrieval_format_clean` | Normal stop, with no thinking markers in the response |
| `passed` | Capacity and retrieval passed; ordinary mode also requires clean formatting |
| `overlapping_decode_without_preemption` | All client decode intervals overlap by over 0.5 seconds, with zero preemptions |

Sustained mode forces its output length; score its marker retrieval separately
from the generated continuation. Decode intervals can contain stalls. Inspect
running/waiting counts, cache usage and preemptions together: inactive prefixes
may occupy cache, and submitted clients may wait before executing.

The MTP-on launch reports **1,844,162 shared KV tokens**, below two independent
full 1M windows (2,097,152 tokens). The experiment measures how the scheduler
handles that workload. Raw records stay under ignored `logs/`; completed results
are recorded in [VALIDATION.md](../docs/VALIDATION.md).
