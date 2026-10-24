# Initial quality baseline — 2026-10-24 UTC

128 fixed examples per task; same checkpoint, prompts and tokenizer IDs.

| Metric | vLLM (95% Wilson interval) | TensorFold (95% Wilson interval) | TF − vLLM (paired 95% bootstrap interval) |
|---|---|---|---|
| GSM8K flexible exact match | 119/128 = 92.97% (87.2–96.3%) | 120/128 = 93.75% (88.2–96.8%) | +0.78 pp (+0.00 to +2.34) |
| GSM8K strict exact match | 119/128 = 92.97% (87.2–96.3%) | 120/128 = 93.75% (88.2–96.8%) | +0.78 pp (+0.00 to +2.34) |
| IFEval prompt strict | 112/128 = 87.50% (80.7–92.2%) | 113/128 = 88.28% (81.6–92.8%) | +0.78 pp (-1.56 to +3.12) |
| IFEval prompt loose | 114/128 = 89.06% (82.5–93.4%) | 114/128 = 89.06% (82.5–93.4%) | +0.00 pp (-2.34 to +2.34) |

Diagnostics:

- vllm: 256 responses; 0 output-budget truncations; 0 missing reasoning closures.
- tensorfold: 256 responses; 0 output-budget truncations; 0 missing reasoning closures.

Intervals describe uncertainty in this selected benchmark sample, not all model capabilities. The paired bootstrap uses 10,000 resamples with seed 1729. An interval spanning zero is inconclusive, not proof of equivalence. With no observed paired differences the bootstrap interval is degenerate; it cannot rule out unseen disagreements. These short-context scores do not qualify 1M context or measure quantization loss against an unquantized model.

## What this reading establishes

The two engines performed similarly on this fixed initial sample. TensorFold had one additional correct GSM8K answer and one additional strict IFEval pass. The exact paired McNemar test is p=1.0 on each primary metric: these few disagreements do not establish an engine accuracy advantage. This is also not a proof of equivalence or a test of all model capabilities.

All 512 formal responses came from the same pinned NVIDIA FP4-Local checkpoint. The clients verified identical request bodies and tokenizer IDs for every paired item and checked input usage against local token counts. No HTTP failures, input truncations, output-budget truncations or missing reasoning closures occurred in the completed formal runs. Four-item pilot runs are excluded from the scores.

The profile is low reasoning, greedy generation, up to 4,096 output tokens including reasoning, one request at a time, official eight-shot GSM8K CoT and zero-shot IFEval. Actual input prompts were at most 784 tokens. These are short-context quality results; they do not establish 256K, 512K or 1M task accuracy. They also do not measure quantization loss against the original higher-precision model.

## Reviewable disagreements

| Item | Observed scoring difference |
|---|---|
| GSM8K test 425 | TensorFold matched the feeder-count reference; vLLM interpreted the baseline bird count differently |
| IFEval 261 | vLLM included the required keyword; TensorFold omitted it |
| IFEval 305 | TensorFold passed strict JSON formatting; vLLM passed only the loose check |
| IFEval 496 | TensorFold met the minimum word count; vLLM did not |

The [benchmark instructions](../quality/README.md) also record two questionable GSM8K reference answers (rows 119 and 952). Official scores were left unchanged; no items were removed after inspecting responses. All scores should be read as agreement with the benchmark's rules and labels, not independent proof that every reference is correct.

## Engine and artifact identity

- NVIDIA revision: `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`.
- vLLM: release-10152026, TP2/DCP2, FP8 KV, MTP3; measured production configuration retained, including 1M configured limit and 8 slots, with one client request at a time.
- TensorFold: v1.1.3 plus PR #1050 head `9b23b6a2e7ff226ca61934189c4cd57bf2933980`; SM120 image with unchanged native kernels, 16K configured context, one slot, FP8 KV, existing MTP, no retained prompt cache and default native reserve.
- Harness: `d6de81643928d653435c431bae19945d41d32520`. Dataset pins and selection seed 1729 are in [profile.json](../quality/profile.json); dependencies are in `quality/uv.lock`, grader resource checksum in `quality/resources.json`.
- Shared prompt template comes from the locked NVIDIA snapshot. Raw completion requests bypass the two servers' different chat-template defaults. Final-answer extraction is the same on both engines.

The portable [baseline JSON](../quality/baselines/initial-v1.json) contains scores, confidence intervals, selected IDs, image IDs, settings and source hashes. Full local evidence lives in `logs/quality/initial-v1-{vllm,tensorfold}/`: manifests, exact requests, raw responses, extracted answers, official per-item grades and server logs. `logs/quality/initial-v1-comparison/` contains the detailed paired report and response disagreements. Raw benchmark material stays out of Git.

Request times and token counts are retained for diagnosis, but this was not a controlled throughput comparison: retained-cache behavior and internal engine configurations differ. vLLM reported 719 cached input tokens across the run; TensorFold reported zero.

## Reproduce or extend

Follow [quality/README.md](../quality/README.md) for setup, both engine runs and the comparison command. Use fresh output directories. Changes to the dataset selection, prompt, reasoning effort, output allowance or grader require a new protocol version; do not compare its percentages as though nothing changed.

The [precision and quality guide](PRECISION-AND-QUALITY.md) explains W4A16, BF16, NVFP4, scale compression, KV precision and the difference between correctness checks and accuracy benchmarks. Broader next qualifications include held-out workload-specific prompts, more benchmark items and a separately controlled long-context suite such as RULER. No RULER or MMLU score is claimed here.

Final service state: vLLM was restored and all four API acceptance checks passed after the benchmark. The repository's 32 CPU checks passed. No target-kernel changes or GB10-derived kernel retuning were made for this experiment; this is the accuracy reference for future optimization work.
