# Long-context short controls

Completed all 52 frozen controls across 13 tasks: 51/52 have clean full credit; mean task score 98.08. Clean full credit requires full score, no truncation and no generated reasoning marker.

Nominal input length is 32,768 tokens; actual server allocation is `TF_CONTEXT=1048576`. Near-1M cases were not run.

Minimum sampled free memory: GPU 0: 1,765 MiB, GPU 1: 1,846 MiB. The 1,536 MiB floor passed; one-second sampling does not observe intervening transient peaks.

| Task | Cases | Mean score | Actual input tokens | Clean full credit |
|---|---:|---:|---:|---:|
| niah_single_1 | 4 | 100.00 | 28,651–28,653 | 4/4 |
| niah_single_2 | 4 | 100.00 | 28,674–28,674 | 4/4 |
| niah_single_3 | 4 | 100.00 | 28,673–28,674 | 4/4 |
| niah_multikey_1 | 4 | 100.00 | 28,673–28,674 | 4/4 |
| niah_multikey_2 | 4 | 100.00 | 28,660–28,671 | 4/4 |
| niah_multikey_3 | 4 | 100.00 | 28,610–28,674 | 4/4 |
| niah_multivalue | 4 | 100.00 | 28,674–28,674 | 4/4 |
| niah_multiquery | 4 | 100.00 | 28,674–28,674 | 4/4 |
| vt | 4 | 100.00 | 28,652–28,674 | 4/4 |
| cwe | 4 | 100.00 | 28,657–28,669 | 4/4 |
| fwe | 4 | 100.00 | 28,660–28,674 | 4/4 |
| qa_squad | 4 | 75.00 | 28,505–28,674 | 3/4 |
| qa_hotpot | 4 | 100.00 | 28,471–28,666 | 4/4 |

## Incorrect or nonclean responses

| Case | Score | Truncated | Reasoning marker |
|---|---:|---|---|
| qa_squad-32768-2 | 0 | False | False |

Requests: 52 new, 0 reused. Raw responses: `logs/hardening-validation-20261026/v3-short-controls/00-host-1m/long-context/requests.jsonl`.

[Portable receipt](../tensorfold/results/review-hardening-controls-v1.json) · [Protocol](../quality/LONG-CONTEXT.md).
