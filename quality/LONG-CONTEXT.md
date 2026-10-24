# Local long-context accuracy screen

The corrected v2 screen uses all 13 RULER task families, four frozen examples per task,
at configured 32K and 1M lengths: 104 requests. This is a local RULER-derived
screen, not the full published benchmark or a leaderboard score. Four examples
per task can expose failures but cannot establish a reliable task-level accuracy
estimate. Capacity tests and simple three-marker retrieval are separate evidence.

The source is [NVIDIA RULER](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a),
using task constructors from [the pinned lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/ruler).
The original NVIDIA substring metrics are used: fraction of required answers for
retrieval/tracking/aggregation, and **any accepted alias** for QA. The pinned
harness's QA helper averages aliases; that helper is deliberately not used.
Every response score is checked against the original NVIDIA scorer.

## Frozen protocol

- Exact local NVIDIA checkpoint and tokenizer; both engines' earlier short-quality
  baselines are reused without changing the environment or re-running them.
- Deterministic per-case seeds, shared NVIDIA low-effort chat template with explicitly closed thinking prefix, greedy
  sampling, MTP on, 256 output-token cap, cold cache, one request at a time.
- Complete generated examples are sized by binary search; no truncation of evidence
  or answers. The 4,096-token preparation margin makes the smaller controls about
  28.7K input tokens. Report measured lengths, not the nominal 32K label.
- The essay distractor corpus is repeated to extend the task to 1M. CWE retains
  its upstream vocabulary cap, and QA retains its source-document corpus.
  Undersized tasks are reported and excluded from
  any near-1M aggregate. A case is marked near-window only at >=90% of its label.
- QA question IDs are shared between lengths. Synthetic golds can differ between
  lengths because their construction changes; these are task controls, not paired
  identical-answer experiments across lengths.
- Prompt text, token IDs, source files, datasets, image, configuration and client
  are hashed. Native tokenization must match the frozen token IDs before inference.
- Raw answers, wrong answers and output-limit truncations are retained. There is
  no retry-based selection. An uncommitted raw response requires inspection before
  any restart; protocol failures are not silently converted into accuracy failures.
- Both cards' free memory is sampled once per second, including startup; a sample
  below 1,536 MiB stops the run. Peaks between samples are not observed.

The substring scorer is intentionally the upstream metric and can accept an
answer embedded in extra text. Inspect failed and suspicious successful answers;
do not describe this screen as an unrestricted reasoning or factuality evaluation.
No vLLM near-1M accuracy conclusion follows from TensorFold-only results.

## Reproduction

Use the existing short-quality environment without modifying its lockfile. Install
the additional word generator into a separate overlay:

```bash
uv pip install --target .state/long-quality-deps -r quality/long-context-requirements.txt --require-hashes
.state/quality-venv/bin/python quality/rewrap_ruler_bank.py --source logs/qualification-v2/ruler-bank --output logs/qualification-v2/ruler-bank-v2
.state/quality-venv/bin/python tensorfold/tuning/sweep.py \
  --profile-file tensorfold/serving/profiles.json --profiles host-1m \
  --skip-benchmark --minimum-free-mib 1536 \
  --long-context-bank logs/qualification-v2/ruler-bank-v2 \
  --long-context-lengths 32768 1048576 \
  --output logs/qualification-v2/long-accuracy-v2
```

The generator requires the pinned RULER checkout at `.state/ruler-upstream`, the
existing pinned lm-eval installation, NLTK assets, and the datasets recorded in
its manifest. Dataset files remain under `/models_engrams/huggingface`, outside
Git. The bank is complete only when `complete.json` matches its manifest hash.
The runner owns only `glm53-fp4-tensorfold` and stops it after the sweep.


The frozen bank has 44 near-1M cases across 11 task families. CWE reaches about
143K input tokens and SQuAD about 205K; their eight configured-1M cases are reported
at those actual lengths and excluded from the near-1M aggregate. The 52 shorter
controls use roughly 28K inputs because of the explicit preparation margin.
[Corrected bank and preservation receipt](../tensorfold/results/long-context-bank-v2.json).
The parent v1 receipt retains dependency versions and input provenance; its original
reasoning wrapper is invalid for the intended scoring protocol.

For a fresh checkout, prepare the pinned scorer and HotpotQA source before the
commands above (skip cloning if the already-pinned checkout exists):

```bash
export HF_CACHE_ROOT=/models_engrams/huggingface
export HF_HOME="$HF_CACHE_ROOT"
git clone https://github.com/NVIDIA/RULER.git .state/ruler-upstream
git -C .state/ruler-upstream checkout --detach c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
hf download namlh2004/hotpotqa hotpot_dev_distractor_v1.json   --repo-type dataset --revision 7e54db4656209750ff487f6fdf8e39a66dba136b   --cache-dir "$HF_CACHE_ROOT/hub"
```

The generator fetches the pinned essay dataset and SQuAD source if absent. Their
content hashes are retained in the bank receipt. Existing short-quality setup
provides the frozen lm-eval environment and NLTK assets. No existing shared-cache
files are deleted or replaced during qualification.


## Preserved protocol failure and correction

The first attempt used `enable_thinking=False` with the checkpoint Jinja template.
That template ignores the keyword and always emits an opening `<think>`; it also
selected its default Max header. The runtime's `enable_thinking:false` field on a
raw completion did not validate the supplied prompt. The 52 shorter responses
contained the gold substrings, but five hit the output cap and some contained
unfinished reasoning. Those scores are **invalid for the intended thinking-off
protocol**. The first near-1M request was cancelled and the owned service stopped.
[Failure receipt](../tensorfold/results/long-context-protocol-failure-v1.json).

The corrected renderer explicitly selects the Low header and closes `<think>`
before the unchanged answer prefix, exactly as the established performance
benchmark does. The rewrapper preserves all 104 task bodies, gold answers, seeds
and question selections, adding two tokens per case. It does not read inference
outputs. The client validates the actual prefix before every request; regression
tests reject an open prefix or unexpected header. All 104 cases are rerun in a
separate directory. Existing valid short-quality and performance baselines are
not rerun.

The reproduction command above uses the preserved v1 input bank. To generate a
**new cohort** from scratch, use the corrected `quality/ruler_bank.py` with a fresh
output directory; fresh binary sizing can choose slightly different task bodies,
so do not expect its prompt hashes to equal this rewrapped cohort. The original
v1 generator is preserved in commit `f7aab64` if its inputs need reconstruction.

The official substring metric remains unchanged. The corrected report also lists
truncations, generated reasoning markers and full-credit responses that have
neither. A full substring score with an unfinished answer is not called clean
completion or broad accuracy qualification.
