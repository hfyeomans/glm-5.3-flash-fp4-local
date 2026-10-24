# Shared cold-prompt benchmark

Same pinned NVIDIA checkpoint, tokenizer IDs and raw prompts; one request at a time, 256 output tokens, greedy generation, seed 1729, three measured repeats per cell after warmup.

| Prompt | Engine | MTP | Decode tok/s | Prefill tok/s | Prefill seconds | Client TTFT seconds |
|---|---|---|---:|---:|---:|---:|
| short-prose (31) | tensorfold | on | 144.0 | 747 | 0.041 | 0.042 |
| short-prose (31) | tensorfold | off | 73.4 | 778 | 0.040 | 0.040 |
| short-prose (31) | vllm | on | 181.7 | 394 | 0.079 | 0.081 |
| short-prose (31) | vllm | off | 117.6 | 437 | 0.071 | 0.073 |
| short-code (33) | tensorfold | on | 127.8 | 734 | 0.045 | 0.046 |
| short-code (33) | tensorfold | off | 73.5 | 778 | 0.042 | 0.043 |
| short-code (33) | vllm | on | 172.3 | 179 | 0.184 | 0.186 |
| short-code (33) | vllm | off | 117.2 | 195 | 0.170 | 0.172 |
| prefill-2048 (2,048) | tensorfold | on | 122.9 | 5,029 | 0.407 | 0.408 |
| prefill-2048 (2,048) | tensorfold | off | 69.0 | 5,154 | 0.397 | 0.398 |
| prefill-2048 (2,048) | vllm | on | 168.1 | 8,311 | 0.246 | 0.249 |
| prefill-2048 (2,048) | vllm | off | 117.7 | 8,761 | 0.234 | 0.237 |
| prefill-8192 (8,192) | tensorfold | on | 120.6 | 4,460 | 1.837 | 1.837 |
| prefill-8192 (8,192) | tensorfold | off | 68.6 | 4,504 | 1.819 | 1.819 |
| prefill-8192 (8,192) | vllm | on | 174.1 | 10,195 | 0.804 | 0.809 |
| prefill-8192 (8,192) | vllm | off | 117.7 | 10,422 | 0.786 | 0.791 |
| prefill-32768 (32,768) | tensorfold | on | 119.1 | 4,522 | 7.247 | 7.248 |
| prefill-32768 (32,768) | tensorfold | off | 67.5 | 4,534 | 7.228 | 7.229 |
| prefill-32768 (32,768) | vllm | on | 170.7 | 10,257 | 3.195 | 3.207 |
| prefill-32768 (32,768) | vllm | off | 117.4 | 10,387 | 3.155 | 3.166 |
| prefill-65536 (65,536) | tensorfold | on | 132.5 | 4,491 | 14.593 | 14.595 |
| prefill-65536 (65,536) | tensorfold | off | 65.9 | 4,489 | 14.600 | 14.602 |
| prefill-65536 (65,536) | vllm | on | 167.9 | 10,161 | 6.450 | 6.476 |
| prefill-65536 (65,536) | vllm | off | 117.7 | 10,341 | 6.338 | 6.361 |

These are per-cell medians, not a universal engine ranking. Short-prompt prefill rates include fixed overhead and should not be extrapolated to long prompts.

Decode uses each server's post-first-token interval and (output tokens minus one). vLLM timings are isolated deltas of exactly one request in its Prometheus histograms; TensorFold returns runtime timings with the completion. Client TTFT is first nonempty raw text, including local HTTP/queue overhead. Server prefill boundaries are implementation-specific; client TTFT supplies a common external measurement.

Every measured request had zero reused prompt tokens. vLLM retains its qualified MTP3/B12X budget, 1M configured window and eight slots, with one client request active. Its plain run removes only the speculative configuration. Unique cache salts prevent prefix reuse. TensorFold uses one decode warp, 128K configured context, one slot, FP8 KV, default reserve and zero retained prompt-cache budget.

Some repeated vLLM greedy outputs differ despite the fixed seed; distinct text counts are retained in the JSON. The timing samples are valid fixed-input/fixed-output-length measurements, not a claim of bit-deterministic generation or an accuracy score. TensorFold token-hash consistency is reported separately. The 256-token cap is intentional for throughput. The earlier quality suite remains the accuracy reference.

Raw manifests, requests, SSE frames, counter snapshots and responses are retained under `logs/tuning/`. No clocks, power limits, checkpoint values or shared caches were changed.
