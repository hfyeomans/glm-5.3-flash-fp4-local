# The test service

Use this order for GPU validation. Preserve every failure and change one setting at a time.

1. Free the two selected GPUs, then run `./prepare.sh` and `./doctor.sh`. Confirm the exact image, driver, checkpoint and available GPU memory. Preparation pulls only the container.
2. Run `./start.sh`. Capture loading, B12X compilation, graph capture and the reported logical KV capacity with `./logs.sh`. Confirm TP2, DCP2, MTP3, FP8 KV and the chosen model directory in the server log.
3. Run `./smoke.sh --suite --output logs/acceptance`. Require all four generation checks to pass, including parsed tool arguments and streaming termination.
4. Send a normal thinking-enabled request with a generous output budget. Check that reasoning and final content are separated and an actual final answer arrives. This checks the server default separately from the low-effort short suite.
5. Measure warm synthetic requests at concurrency 1, 2, 4 and 8. Use the same request count, output cap and prompt family for every comparison:

```bash
./bench.sh --concurrency 1 --requests 8 --max-tokens 256 --output logs/bench-c1.json
./bench.sh --concurrency 2 --requests 8 --max-tokens 256 --output logs/bench-c2.json
./bench.sh --concurrency 4 --requests 8 --max-tokens 256 --output logs/bench-c4.json
./bench.sh --concurrency 8 --requests 8 --max-tokens 256 --output logs/bench-c8.json
```

The client reports aggregate end-to-end completion tokens/second, median TTFT and request latency. These rates include prompt processing and network time; completion-token usage includes reasoning tokens, and TTFT measures the first visible content. It requires server token usage, retains every sample and fails the run on a request failure. One warmup is excluded. Related prompts may reuse prefixes.

6. Test long prompts progressively: approximately 8K, 32K, 128K, then larger only after successful admission and complete responses. Count prompt tokens with the running server's tokenizer or response usage. Insert fresh retrieval facts and verify the actual answer. Save prompts, usage, response, TTFT and admission failures.
7. Send the same long conversation prefix twice, then append a turn. Compare fresh and resumed answers, prompt accounting and latency. Repeat after other requests displace the cache. Repeat after restart to check cache lifetime.
8. If vision is required, add local test images and verify answers and token use independently.
9. Stop and restart the recipe; rerun acceptance. Check JIT cache reuse and unchanged model/image identities. Establish an acceptable repeated-run range before trying TensorFold.

For an engine comparison, preserve the same checkpoint files, prompts, sampler settings, thinking mode, token budgets and tool schemas. Record differences in precision, context, cache budgets and speculation. Investigate changed greedy outputs on the tested hardware.

`./scripts/check.sh` is the independent CPU-only repository check. It checks local cache selection, missing shards, escaped symlinks, literal configuration handling, safe container ownership, refusal to launch on busy GPUs and the HTTP client against a local server. It never loads the model.

## Initial quality comparison

See [quality/README.md](../quality/README.md) for the pinned official GSM8K/IFEval benchmark and paired-report commands. This is separate from throughput, API acceptance and long-context retrieval tests. The 128-item subsets, reasoning effort, output limit, dataset revisions and tokenization checks are fixed and recorded. [Precision and quality notes](PRECISION-AND-QUALITY.md) explain the terms and interpretation.
