# Local SM120 tuning contract

This work measures the user's two RTX PRO 6000 Blackwell Max-Q cards against the
qualified recipe at commit `7e06307`. It keeps NVIDIA FP4-Local revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`, CSF32, BF16 activations and FP32
accumulation explicit. Native baseline: v1.1.3 plus PR #1050, merged tree
`1d52effd182d9319d775d5787904c36783e532f1`, CUDA 13.3, Zig 0.17.0, SM120, TP2.

The requested work is a local experiment on the existing PR, not an independent
implementation of its checkpoint-loading issue or a publication. The ATC board
was read with ATC 0.7.0, revision `e4f5a4a114eb2bcfef7a8ebcc2dbdfb718df28c4`.
The broad board reported incomplete file coverage (`pullFilesLimit`); scoped
issue #1015 was complete and `takenOutside` because PR #1050 already covers it.
No claim, public session, issue, comment or competing PR was created. Local
experiments do not constitute upstream clearance or maintainer approval. Any
future upstream submission must coordinate with that PR and satisfy its full
contract/launch-record and supported-platform requirements.

## Measurements and acceptance

- Establish shared raw prompts at exact 2K/8K/32K/64K lengths and short code/prose
  cases. Freeze token arrays, template and generation settings. Verify each
  server's tokenizer IDs. Measure cold prefill, server decode and client TTFT
  separately with MTP on/off. Preserve all attempted runs and errors.
- Screen prefill tile widths with the same upstream kernel template, real
  rank-local GLM matrix dimensions, varying routed rows and valid tile edges.
  Require byte identity to the original instance, repeatability, guards and
  unchanged input checks. Microbenchmarks are screening evidence only.
- For a served tile candidate, keep matching device instantiations and host
  launch geometry together. Run arithmetic/real-loader gates, API checks,
  draft/plain and cache/cold comparisons, and the fixed quality suite. Adopt
  only repeatable end-to-end improvements with no unexplained regression.
- Split-K changes addition order. Screen its speed separately and retain
  independent FP64 numerical-reference results by projection class. A changed
  reduction is not a bit-preserving replacement and is not ready to land merely
  because it passes a same-engine draft/plain check or a small accuracy sample.
- Screen existing local transport/overlap and launch controls independently;
  preserve the baseline and use fixed workloads. Do not change clocks, power,
  drivers, precision or unrelated services.
- Capacity work (mapped immutable embedding or resident compressed scales) needs
  separate allocation, lifetime, loader and long-context evidence. A scheduling
  gain does not prove progress toward 1M; a configured window is not a measured
  near-window request or a broad long-context quality score.

Only these SM120 cards are available. Other architectures, full upstream release
qualification and maintainer agreement remain explicitly outside local evidence.
