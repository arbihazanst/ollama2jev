<p align="center"><img src="assets/Ollama2Jev.png" alt="Ollama2Jev" width="100%"></p>

<div align="center">

# Ollama2Jev: Bridging Generative Language Models and System-One Decision Inference

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square)](LICENSE)
[![Backend](https://img.shields.io/badge/backend-Ollama-green?style=flat-square)](src/llm2jev/backend/ollama)
[![Experiments](https://img.shields.io/badge/experiments-E1--E5%20complete-success?style=flat-square)](experiments/README.md)

**A training-free backend for structured Choice, Noul, and Score inference using an already installed Ollama-served generative checkpoint.**

</div>

> **Project relationship:** Ollama2Jev extends the independently released [Yinsongxu/LLM2Jev](https://github.com/Yinsongxu/LLM2Jev) implementation. Existing binary-question compilation, prompt rendering, normalization, and typed assembly are retained; the new Ollama binary backend extracts exact lowercase `yes`/`no` scores from first-position top-20 log-probabilities. This is an independent research extension, not an official upstream project.

## What the bridge does

An Ollama model already installed for text generation can be invoked for a deterministic one-token scoring request (`logprobs=true`, `top_logprobs=20`). Ollama2Jev derives the conditional score `P(yes | {yes, no})` when both exact verbalizers are present and passes the ordered scores to the existing LLM2Jev typed assembler. Missing verbalizers produce explicit coverage failures. No second decision-specialized checkpoint or training step is required for this pathway.

## Completed experiments (E1–E5)

| Phase | Scope | Main evidence | Status |
|---|---|---|---|
| [E1](experiments/01_general_llm_benchmark/README.md) | 11 general-purpose models × 231 public JevBench tasks | 2,538/2,541 typed attempts completed; best Qwen3.8 27B: 197/231 (85.28%) | Complete |
| [E2](experiments/02_zero_shot_ablation/README.md) | Direct-label Z0 versus conditional Z1 and archived scoring ablations | Z0 1,722/2,541 (67.77%) versus Z1 1,648/2,541 (64.86%); model-dependent trade-offs | Complete |
| [E3](experiments/03_finetuned_decision_models/README.md) | Six native System-One deployed pipelines × 231 tasks | 1,386 valid outcomes; PPLX Decider: 205/231 (88.74%); selected native advantages significant | Complete |
| [E4](experiments/04_backend_fidelity/README.md) | E4A numerical audit; E4B 11 Ollama–Transformers nominal pairs × 231 tasks × 3 repetitions | 7,617/7,623 valid pairs; 27,987 paired binary candidates; 87.71% valid-pair-weighted typed agreement | Complete |
| [E5](experiments/05_deployment_audit/README.md) | 11 installed tags, five Qwen generation/scoring verifications across archived runs, six pairwise storage audits | Five model configurations verified; additional pairwise specialized blob content 0.78–18.78 GiB | Complete |

### E5: same-checkpoint verification

Successful archived dual-use records cover **all five** general-purpose Qwen configurations:

| Model tag | Ordinary generation | Exact `yes`/`no` binary scoring | Stable tag, digest and inventory |
|---|---|---|---|
| `qwen3.5:0.8b` | Pass | Pass | Yes |
| `qwen3.5:2b` | Pass | Pass | Yes |
| `qwen3.5:4b` | Pass | Pass | Yes |
| `qwen3.5:9b` | Pass | Pass | Yes |
| `qwen3.8:latest` (27B) | Pass | Pass | Yes |

The successful observations were recorded **across 20 and 22 October 2026 (UTC)**. The [20 October run](experiments/05_deployment_audit/outputs/dual_use_corrected/probe_events.json) passed 9B and 27B. The [22 October consolidated run](experiments/05_deployment_audit/outputs/dual_use_all_qwen_20261023/probe_events.json) passed 0.8B, 2B, 4B and 9B; its 27B attempt timed out despite the earlier successful verification. **Do not describe this as a single uninterrupted five-for-five successful batch.** These probes verify generation and first-position binary probability extraction, **not** five additional 231-task JevBench evaluations or complete typed assembler runs.

The six manifest-based comparisons show **0.78–18.78 GiB** of content-addressed specialized checkpoint blobs not referenced by the respective paired general-purpose checkpoint. This is **not** a controlled measurement of global disk reclamation, accelerator memory saved, training-time savings, or deployment latency. E5 used Ollama 0.40.5; E1–E4 used Ollama 0.35.0.

## Inference architecture

```text
Existing LLM2Jev: JevRequest → binary compiler → prompt renderer
                                                |
Ollama2Jev:                         OllamaBackend → Ollama /api/chat
                                                → 1-token top-20 logprobs
                                                → exact yes/no extraction
                                                → conditional P(yes)
                                                |
Existing LLM2Jev:                    typed assembly → Choice / Noul / Score
```

## Reproducibility

All committed evidence, scripts, denominators and caveats are indexed under [experiments/README.md](experiments/README.md). E1/E2 accuracy comparisons are *not* a controlled prompt-only effect, E3 compares deployed pipelines rather than isolating fine-tuning, and E4 does not establish identical-weight logit equivalence. Latency comparisons are observational. E5 supports checkpoint reuse feasibility, not measured resource savings.

<!--
## Citation and DOI

Software citation metadata is provided in [`CITATION.cff`](CITATION.cff). **This repository has no registered software DOI yet.** To obtain one, connect the repository to [Zenodo](https://zenodo.org/), enable GitHub archiving and publish a release from the `ollama2jev` branch. Zenodo assigns a DOI after archiving. Add the actual DOI badge and version-specific citation once registered; do not cite a guessed DOI.
-->
