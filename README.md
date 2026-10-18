<p align="center"><img src="assets/Ollama2Jev.png" alt="Ollama2Jev" width="100%"></p>

<div align="center">

# Ollama2Jev: Extending LLM2Jev with an Ollama Backend

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square)](LICENSE)
[![Backend](https://img.shields.io/badge/backend-Ollama-green?style=flat-square)](src/llm2jev/backend/ollama)
[![E1--E4](https://img.shields.io/badge/experiments-E1--E4%20complete-success?style=flat-square)](experiments/README.md)

**Ollama2Jev extends the open-source LLM2Jev stack with an Ollama-compatible binary scoring backend for probabilistic typed decisions.**

</div>

> **Project relationship:** This project builds upon [Yinsongxu/LLM2Jev](https://github.com/Yinsongxu/LLM2Jev). The existing typed `Choice`, `Score` and `Noul` compilation, normalization and response-assembly pipeline is retained; the additional `OllamaBackend` accesses one-token next-token log-probabilities via Ollama. It is not an official component of upstream LLM2Jev, Jev, TypeSafe or Ollama.

## Features

- Ollama-compatible binary scoring using a deterministic one-token request.
- Top-20 first-token log-probability diagnostics with exact lowercase `yes` and `no` extraction.
- Numerically stable binary conditional probabilities and explicit missing-verbalizer coverage errors.
- Original upstream typed response assembly for Choice, Noul and Score.
- Reproducible zero-shot benchmarks, native decision-model comparisons, scoring ablations and cross-backend numerical checks.

## Experimental evaluation — all complete

| Experiment | Design | Status |
|---|---|---|
| [E1](experiments/01_general_llm_benchmark/README.md) | 11 general-purpose LLMs × 231 public JevBench tasks | **COMPLETE** |
| [E2](experiments/02_zero_shot_ablation/README.md) | Z0 direct generation plus Z1–Z4 scoring variants (11 models) | **COMPLETE** |
| [E3](experiments/03_finetuned_decision_models/README.md) | Six native fine-tuned System-One decision models × 231 tasks | **COMPLETE** |
| [E4](experiments/04_backend_fidelity/README.md) | Offline numerical audit plus four Qwen-family cross-backend pairs × 231 tasks × three repetitions | **COMPLETE** |

Highlights from committed measurements:

- **E1:** 2,541 task attempts, 2,538 valid decisions. Highest general-purpose E1 score: `qwen3.8:latest` **197/231 (85.28%)**.
- **E2:** Z0 direct generation **1,722/2,541 (67.77%)**, versus original Z1 **1,648/2,541 (64.86%)**; improvement depends on model, not universal.
- **E3:** Six native models, all 1,386 final decisions valid; the strongest was `pplx-decider:latest` **205/231 (88.74%)**.
- **E4:** 2,772/2,772 paired typed evaluations across four model pairs, 10,188 binary candidate comparisons, no recorded inference errors. It measures **cross-stack agreement**, not exact identical-weight logit fidelity.

All primary evidence, reproduction scripts, dataset assumptions and caveats are documented in the [experimental evaluation directory](experiments/README.md).

## Relationship to LLM2Jev

```text
Original LLM2Jev
  JevRequest → binary question compiler → prompt renderer
                     |
                     v
Ollama2Jev           OllamaBackend → Ollama chat API
                                     → one-token next-token logprobs
                                     → lowercase yes/no extraction
                                     → conditional P(yes)
                     |
                     v
Original LLM2Jev  typed normalization and response assembly
                     |
                     v
                Choice / Score / Noul
```

## Interpretation boundaries

The Ollama backend makes typed scoring practical for models available through Ollama, but **does not guarantee superior accuracy over direct zero-shot generation for every model**. E3 compares differing deployment pipelines, not fine-tuning in isolation. E4 uses separately prepared model checkpoints whose quantization, kernels and tokenization may differ; it is not exact same-checkpoint fidelity evidence. See the experiment reports for complete statistical and reproducibility details.
