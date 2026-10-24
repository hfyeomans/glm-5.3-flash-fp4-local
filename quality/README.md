# Initial GLM-5.3-Flash quality benchmark

This measures the same locked NVIDIA FP4-Local checkpoint through vLLM and our experimental TensorFold PR #1050 build. It is an initial short-context engine comparison, separate from the long-context capacity ladder. It does not compare against the original higher-precision model.

## Protocol

- Official [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) tasks and graders, pinned to `d6de81643928d653435c431bae19945d41d32520`; dependency versions/hashes in `uv.lock`.
- 128 unique examples per task, selected without replacement with seed 1729 and then sorted by dataset index. The runner verifies selected dataset rows against the harness requests.
- `gsm8k_cot`: official eight few-shot examples, pinned `openai/gsm8k` test split. Primary metric: flexible-extraction exact match; strict extraction is also reported.
- `ifeval`: official zero-shot task, pinned `google/IFEval` train split (the published evaluation set). Primary metric: strict prompt-level accuracy; loose and instruction-level scores remain in the raw results.
- Same checkpoint tokenizer and chat template, rendered locally with `reasoning_effort=low`, then sent to **raw completion endpoints**. This bypasses different server chat-template defaults. Every server tokenization and completion input count must match the locally tokenized prompt.
- Greedy generation (`temperature=0`, `top_p=1`), seed 1729, one request at a time, 4,096 output tokens including reasoning; no silent input truncation. Input plus allowance must fit 16,384 tokens.
- Grade only the final answer after `</think>`, with surrounding whitespace stripped. Preserve raw output, flag missing markers and output truncation. No retries that select a better answer.
- Existing engine kernels, target weight precisions, FP8 KV and speculative decoding remain in use. The manifest records image IDs and relevant settings. This compares the serving implementations, not perfectly identical internal execution strategies or performance.
- The native benchmark profile uses 16K context, one slot, no retained prompt cache and the default reserve. vLLM retains its measured baseline settings. Cached-token counts are saved; the quality benchmark is not a cold-cache speed test.

The subset and reasoning/output settings differ from a full published leaderboard run. Do not compare these percentages directly to model-card numbers. No overall “quality score” is formed by averaging unrelated tasks.

## Run

The setup requires `uv` and selects Python 3.12. It installs only client/tokenizer/scorer dependencies into `.state/quality-venv`. It does not install a new inference engine or require PyTorch in the client environment; inference remains inside the existing GPU containers. HF data uses `HF_CACHE_ROOT` (default `/models_engrams/huggingface`); model weights remain in the existing cache. No HF token is required for these public datasets. NLTK sentence-tokenization data is stored under that cache's `benchmarks/nltk` directory. The extracted grader data is checked against `resources.json`; a changed resource is refused rather than silently changing scoring.

```bash
./quality/setup.sh
# Existing vLLM must be ready:
./quality/run.sh --backend vllm --output logs/quality/my-vllm

# Switch only the recipe-owned model services:
./stop.sh
TF_CONTEXT=16384 TF_PARALLEL=1 TF_PROMPT_CACHE_GIB=0 ./tensorfold/start.sh
./tensorfold/smoke.sh --wait 1200 --suite
./quality/run.sh --backend tensorfold --output logs/quality/my-tensorfold

# Preserve native logs before its stop removes the container, then restore vLLM:
./tensorfold/stop.sh
./start.sh
./smoke.sh --suite
```

Use a new output directory for every run. `--limit 2` selects two examples per task from the fixed draw for a pipeline pilot; those are explicitly marked as pilot results. A pilot is not a useful accuracy estimate. `QUALITY_API_KEY`, if needed, is sent only to the selected local server and is never written to the manifest. Ambient OpenAI credentials are not used.

Outputs include `manifest.json` (pins, IDs, settings, client hash), `requests.jsonl` (exact prompts, raw responses, token hashes, usage, finish reasons and extracted answers), and `results.json` (official task scores and per-item grades). These artifacts are ignored by Git. The committed profile, client, dependency lock and documentation reproduce the protocol.

See [PRECISION-AND-QUALITY.md](../docs/PRECISION-AND-QUALITY.md) for W4A16, BF16, NVFP4, scale compression, KV cache, speculative decoding and how to read the evidence.

## Compare the completed runs

```bash
.state/quality-venv/bin/python quality/report.py logs/quality/my-vllm logs/quality/my-tensorfold --output logs/quality/my-comparison
```

The reporter refuses changed profiles, dependency locks, runner hashes, selected rows, prompts or tokenizer IDs. It checks per-item grades against the official aggregate, reports Wilson intervals and a paired bootstrap difference, and retains every scoring disagreement in `comparison.json`. Truncation and missing reasoning closures remain visible as diagnostics. Do not interpret a statistically inconclusive difference as proof of engine equivalence.

## Review benchmark misses as well as percentages

Official labels remain unchanged. Manual inspection of the initial vLLM GSM8K misses flagged two reference-answer problems worth considering when interpreting scores:

- Test row 952 says 8 AM to 11 PM, but the reference works from 8 AM to 11 AM. The model's 1,800 matches the literal 15-hour interval; the official label is 360.
- Test row 119 describes one salary as 30% higher than the other, but the reference reverses that relationship as 30% lower, which is not algebraically equivalent.

These annotations do not remove items or revise scores after seeing the answers. They illustrate why benchmark-label agreement is not identical to correctness on every question. The raw responses and dataset rows remain available for review.
