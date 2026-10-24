# Current shared cold-prompt benchmark

Updated after four-piece overlap qualification on 2026-10-24 UTC. vLLM uses the unchanged MTP-on/off cohorts from the original comparison; TensorFold was rerun on the normal image at the original 128K profile with four overlap pieces. The controlled A/B/A test established a 0.87–1.05% prefill gain; other differences from the earlier full run are not attributed to that setting. [Original baseline](PERFORMANCE-BASELINE.md) · [Tuning evidence](../tensorfold/TUNING-RESULTS.md).

Same pinned NVIDIA checkpoint, tokenizer IDs and raw prompts; one request at a time, 256 output tokens, greedy generation, seed 1729, three measured repeats per cell after warmup.

| Prompt | Engine | MTP | Decode tok/s | Prefill tok/s | Prefill seconds | Client TTFT seconds |
|---|---|---|---:|---:|---:|---:|
| short-prose (31) | tensorfold | on | 146.6 | 744 | 0.042 | 0.042 |
| short-prose (31) | tensorfold | off | 74.6 | 803 | 0.039 | 0.039 |
| short-prose (31) | vllm | on | 181.7 | 394 | 0.079 | 0.081 |
| short-prose (31) | vllm | off | 117.6 | 437 | 0.071 | 0.073 |
| short-code (33) | tensorfold | on | 129.7 | 716 | 0.046 | 0.047 |
| short-code (33) | tensorfold | off | 74.5 | 756 | 0.044 | 0.044 |
| short-code (33) | vllm | on | 172.3 | 179 | 0.184 | 0.186 |
| short-code (33) | vllm | off | 117.2 | 195 | 0.170 | 0.172 |
| prefill-2048 (2,048) | tensorfold | on | 125.1 | 5,041 | 0.406 | 0.407 |
| prefill-2048 (2,048) | tensorfold | off | 70.1 | 5,206 | 0.393 | 0.394 |
| prefill-2048 (2,048) | vllm | on | 168.1 | 8,311 | 0.246 | 0.249 |
| prefill-2048 (2,048) | vllm | off | 117.7 | 8,761 | 0.234 | 0.237 |
| prefill-8192 (8,192) | tensorfold | on | 122.3 | 4,499 | 1.821 | 1.822 |
| prefill-8192 (8,192) | tensorfold | off | 69.7 | 4,538 | 1.805 | 1.806 |
| prefill-8192 (8,192) | vllm | on | 174.1 | 10,195 | 0.804 | 0.809 |
| prefill-8192 (8,192) | vllm | off | 117.7 | 10,422 | 0.786 | 0.791 |
| prefill-32768 (32,768) | tensorfold | on | 120.7 | 4,565 | 7.178 | 7.179 |
| prefill-32768 (32,768) | tensorfold | off | 68.6 | 4,577 | 7.160 | 7.161 |
| prefill-32768 (32,768) | vllm | on | 170.7 | 10,257 | 3.195 | 3.207 |
| prefill-32768 (32,768) | vllm | off | 117.4 | 10,387 | 3.155 | 3.166 |
| prefill-65536 (65,536) | tensorfold | on | 134.3 | 4,533 | 14.456 | 14.458 |
| prefill-65536 (65,536) | tensorfold | off | 66.9 | 4,528 | 14.474 | 14.476 |
| prefill-65536 (65,536) | vllm | on | 167.9 | 10,161 | 6.450 | 6.476 |
| prefill-65536 (65,536) | vllm | off | 117.7 | 10,341 | 6.338 | 6.361 |

These are per-cell medians, not a universal engine ranking. Short-prompt prefill rates include fixed overhead and should not be extrapolated to long prompts.

Decode uses each server's post-first-token interval and (output tokens minus one). vLLM timings are isolated deltas of exactly one request in its Prometheus histograms; TensorFold returns runtime timings with the completion. Client TTFT is first nonempty raw text, including local HTTP/queue overhead. Server prefill boundaries are implementation-specific; client TTFT supplies a common external measurement.

Every measured request had zero reused prompt tokens. vLLM retains its qualified MTP3/B12X budget, 1M configured window and eight slots, with one client request active. Its plain run removes only the speculative configuration. Unique cache salts prevent prefix reuse. TensorFold settings: decode warps=1, configured context=131,072 tokens, slots=1, FP8 KV, reserve=native default, zero retained prompt-cache budget and overlap pieces=4.

Some repeated vLLM greedy outputs differ despite the fixed seed; distinct text counts are retained in the JSON. The timing samples are valid fixed-input/fixed-output-length measurements, not a claim of bit-deterministic generation or an accuracy score. TensorFold token-hash consistency is reported separately. The 256-token cap is intentional for throughput. The earlier quality suite remains the accuracy reference.

Raw manifests, requests, SSE frames, counter snapshots and responses are retained under `logs/tuning/`. No clocks, power limits, checkpoint values or shared caches were changed.


## Separate compressed-residency capacity experiment

The [resident-CSF32 result](../tensorfold/csf-resident/RESULTS.md) now demonstrates
near-1M TensorFold inference. It has a measured speed cost: in the same experimental
image with mapped embedding, returning legacy controls averaged 121.12 decode tok/s,
4,437 prefill tok/s and 1.847 s client TTFT at 8K/MTP-on; resident scales measured
112.72, 4,278 and 1.915 s. Across the six workloads, decode was 6.6–7.9% slower;
2K–64K prefill was 3.2–4.1% slower. All 144 output hashes matched. The original
normal-image and vLLM table above remains its own unchanged cohort comparison.

The [pinned vLLM source audit](../tensorfold/results/vllm-prefill-policy-audit-v1.json)
found hybrid A4 prefill enabled by the supplied flags. TensorFold's expert path here
remains W4A16. This is a plausible contributor to the prefill gap, not a measured
causal attribution of the whole cross-engine difference.
