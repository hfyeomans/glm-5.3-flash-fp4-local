# Validation record

Recorded 2026-10-23. Model inference is the upcoming test pass.

| Check | Result |
| --- | --- |
| Cache relocation | Complete Hub entry moved from `/data/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local` to the matching entry under `/models_engrams/huggingface/hub` |
| Model revision | `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` |
| Weight integrity after move | **42/42 SHA-256 checksums passed**, checked against the content-addressed Hub blob filenames; 177,463,841,024 bytes |
| Metadata identity | Config and index SHA-256 match `config/model-lock.json` |
| Offline model validation | Passed; all required tokenizer files, shards and blob links accessible through the planned mount |
| Bash syntax / ShellCheck | Passed |
| Python syntax | Passed |
| CPU-only operational tests | **13 passed**; local model fixtures, fake Docker commands and local mock API only |
| Dry-run | Passed; correct model cache, explicit NVIDIA snapshot, TP2/DCP2/MTP3, no Docker calls or file changes, credential values absent |
| Host | Linux x86_64; NVIDIA driver `615.71.09`; Docker and NVIDIA runtime accessible |
| GPUs at latest preflight | Two RTX PRO 6000 Blackwell Max-Q Workstation Edition cards, 97,887 MiB each; 97,187 / 97,340 MiB free; both pass the 96,419 MiB threshold |
| Port 8000 | Available at latest preflight |
| Registry manifest | Linux/amd64 release-10152026 verified as `sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29`; default image reference pinned to that digest |
| Serving image | **Not present locally**. Pull with `./prepare.sh` before the GPU pass |
| Actual container startup / inference | **Not run** |
| Long context, vision, reasoning and performance | **Not measured** |
| TensorFold build / inference | **Not run**; see `TENSORFOLD.md` |

The final host preflight returned one expected blocker: the missing serving image. No model workload was started or stopped by the repository checks. GPU availability can change; repeat `./doctor.sh` immediately before startup.
