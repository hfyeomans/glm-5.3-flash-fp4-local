# Long-context accuracy screen

**The agreed 22-case near-1M core is complete: 21 fully correct, with one frequency case returning two of three expected words.** All 22 responses finished without truncation or reasoning markers.

The full selected cohort contains 78 results: those 22 core cases, 52 shorter controls (51 fully correct), and four corpus-limited checks (all fully correct). Nine extra long cases completed before the scope reduction are retained separately. Clean full credit requires full official score, no truncation and no generated reasoning marker.

The user reduced the plan to 2 fixed examples per long task after the run began. Selection uses sample indices 0 through 1, independent of scores. All 52 shorter controls are retained. 62 valid requests were reused without new inference; 16 were newly executed. 9 additional completed cases are preserved separately below, rather than discarded or mixed into the balanced headline.

Mean task score: 98.08 for 13 shorter-control tasks; 98.48 for 11 near-1M tasks. These aggregates use different task mixes and distinct examples: four per shorter task, two per long task.

| Task | Shorter mean score (n=4) | Longer mean score | Actual longer input tokens | Clean full credit, longer |
|---|---:|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | 1,044,476–1,044,477 | 2/2 |
| niah_single_2 | 100.00 | 100.00 | 1,044,482–1,044,482 | 2/2 |
| niah_single_3 | 100.00 | 100.00 | 1,044,481–1,044,482 | 2/2 |
| niah_multikey_1 | 100.00 | 100.00 | 1,044,482–1,044,482 | 2/2 |
| niah_multikey_2 | 100.00 | 100.00 | 1,044,474–1,044,480 | 2/2 |
| niah_multikey_3 | 100.00 | 100.00 | 1,044,443–1,044,448 | 2/2 |
| niah_multivalue | 100.00 | 100.00 | 1,044,482–1,044,482 | 2/2 |
| niah_multiquery | 100.00 | 100.00 | 1,044,482–1,044,482 | 2/2 |
| vt | 100.00 | 100.00 | 1,044,473–1,044,474 | 2/2 |
| cwe | 100.00 | 100.00 | 143,602–143,683 (below 1M) | 2/2 |
| fwe | 100.00 | 83.34 | 1,044,481–1,044,482 | 1/2 |
| qa_squad | 75.00 | 100.00 | 204,875–204,877 (below 1M) | 2/2 |
| qa_hotpot | 100.00 | 100.00 | 1,044,433–1,044,456 | 2/2 |

CWE and SQuAD are corpus-limited and excluded from the near-1M aggregate. Both groups use the same 1M server allocation; shorter controls contain roughly 28K tokens. Each request allows 256 output tokens. QA question IDs are shared across lengths; synthetic golds need not be identical.

Minimum free memory at one-second sampling: 1,693/1,846 MiB, above the 1,536 MiB floor.

## Incorrect or nonclean core responses

| Case | Official score | Truncated | Reasoning marker |
|---|---:|---|---|
| qa_squad-32768-2 | 0 | False | False |
| fwe-1048576-0 | 66.67 | False | False |

The frequency miss is independently verified against the frozen prompt: the
model selected the fourth-ranked word, `ucudvr` (13,570 occurrences), instead of
third-ranked `lqwtql` (21,204). Gold top-three counts were checked for both long
frequency cases and all four shorter controls. This is a real aggregation miss,
with verified gold counts. A matched reference run is needed to locate its cause.
[Count audit](../tensorfold/results/long-context-fwe-audit-v1.json).

## Supplemental completed cases

| Case | Official score | Clean full credit |
|---|---:|---|
| niah_single_1-1048576-2 | 100.0 | True |
| niah_single_1-1048576-3 | 100.0 | True |
| niah_single_2-1048576-2 | 100.0 | True |
| niah_single_2-1048576-3 | 100.0 | True |
| niah_single_3-1048576-2 | 100.0 | True |
| niah_single_3-1048576-3 | 100.0 | True |
| niah_multikey_1-1048576-2 | 100.0 | True |
| niah_multikey_1-1048576-3 | 100.0 | True |
| niah_multikey_2-1048576-2 | 100.0 | True |

All responses remain in the raw runs. No requests were retried to select a better answer. The official metric accepts answer substrings; clean completion additionally checks truncation and reasoning markers.

The original open-reasoning attempt is invalid and excluded. The corrected bank preserves its task bodies/golds; only the wrapper changed. The later four-example run was stopped at the user’s scope reduction, and its completed valid results were reused after checking protocol, image, settings, prompt hashes and raw responses. [Protocol](../quality/LONG-CONTEXT.md).

These results cover current TensorFold. For the matched engine comparison, see
[three-way short quality](QUALITY-BAKEOFF-V2.md).

[Portable receipt](../tensorfold/results/long-context-accuracy-v2.json) · raw continuation: `logs/qualification-v2/long-accuracy-22/00-host-1m`.
