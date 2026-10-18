#!/usr/bin/env python3
"""Update Ollama2Jev experiment documentation to verified 2026-10-18 status.

Only writes 6 README.md files, never modifies result files or code.
Run from repo root: python /path/to/apply_readmes.py
"""
from pathlib import Path
import sys
from datetime import datetime

root=Path.cwd()
if not (root/'experiments/04_backend_fidelity/results/paired/summary/e4_cross_stack_summary.json').is_file():
    sys.exit('Run from the Ollama2Jev repo root; required E4 results not found')

DOCS={
'experiments/01_general_llm_benchmark/README.md': '''# E1 — General-purpose Ollama2Jev benchmark

**Status: COMPLETE (verified 2026-10-18).** Eleven general-purpose Ollama model configurations were evaluated zero-shot on the **231 public JevBench tasks** (Choice 139, Noul 74, Score 18) using the Ollama2Jev one-token binary backend. Each binary candidate requires both lowercase `yes` and `no` log-probabilities among the top 20.

## Verified results

- **2,541 model–task attempts** (11 × 231), **2,538 successful typed decisions**, and **3 failed decisions** due to missing verbalizer coverage in `granite4.2:8b`.
- **1,648/2,541 correct** using the complete-attempt denominator (64.86% pooled accuracy).
- The strongest E1 general-purpose configuration was **`qwen3.8:latest`: 197/231 (85.28%)**. The installed tag is `qwen3.8:latest`, not `qwen3.8:27b`.
- The first-token probability and top-k diagnostics are audited by E4A. E2 Z1–Z4 reuse the original E1 records.

## Authoritative artifacts

- [11-model summary](results/summary/ollama2jev_jevbench_11models_summary.json)
- [Per-model JSONL outputs](results/raw/)
- [JevBench evaluation runner](../../examples/run_ollama2jev_jevbench.py)
- [Additional model runner](../../examples/run_ollama2jev_jevbench_additional.py)

E1 results are stored **inside this experiment directory**, not at the repository root. Keep the archived E1 outputs unchanged to preserve experimental provenance.

## Interpretation

The backend requests one generated token per binary candidate, reads the first-token top-20 log-probabilities, computes the two-verbalizer conditional probability, and applies upstream LLM2Jev's typed assembler. Full-denominator accuracy treats uncovered tasks as incorrect; probability metrics exclude invalid distributions.

**E4 is a separate run.** In particular, `qwen3.8:latest` scored 197/231 in E1 but 200/231 in E4; never silently substitute E4 accuracy into E1. Preserve exact model digests, dataset hashes, quantization, and software versions when comparing runs.
''',
'experiments/02_zero_shot_ablation/README.md': '''# E2 — Zero-shot baselines and scoring ablations

**Status: COMPLETE (verified 2026-10-18).** All Z0–Z4 variants have committed outputs. There are **11 model configurations × 231 public JevBench tasks = 2,541 model–task attempts** per variant. Z1 is the archived E1 method; Z2–Z4 use its stored diagnostics.

## Variants and verified results

| Variant | Method | Correct / 2,541 | Accuracy | Execution |
|---|---|---:|---:|---|
| Z0 | Direct zero-shot final-label generation | **1,722** | **67.77%** | New Ollama inference |
| Z1 | Conditional `yes`/`no` probability, original typed assembler | 1,648 | 64.86% | Original E1 run |
| Z2 | Yes-only likelihood, same typed assembler | 1,643 | 64.66% | Offline replay |
| Z3 | Hard score `1[p_yes >= 0.5]`, same typed assembler | 1,388 | 54.62% | Offline replay |

Z0 generated **2,513 valid responses**, **28 invalid responses**, and **zero transport errors**. Unlike Z1–Z3, Z0 does not yield class probability distributions without a separate probabilistic extraction procedure; do not report Brier/NLL/ECE for it as if measured.

### Z4 — Effective top-k rank-truncation replay

| Effective k | Binary candidates covered / 9,327 | Typed decisions complete / 2,541 | Strict correct / 2,541 |
|---|---:|---:|---:|
| 2 | 7,989 | 1,985 | 1,208 (47.54%) |
| 5 | 9,106 | 2,421 | 1,552 (61.08%) |
| 10 | 9,292 | 2,518 | 1,632 (64.23%) |
| 20 | 9,324 | 2,538 | 1,648 (64.86%) |

**Z4 is post-hoc replay of stored top-20 ranks**, not new inference with a smaller API top-k value; it cannot establish runtime improvements.

## Interpretation

The original Z1 scoring method is **not uniformly better than Z0**. The paired exact McNemar tests with Holm correction across 11 models found significantly higher Z0 accuracy for Qwen3.5 4B, Qwen3.5 9B, and Gemma 4 E4B (adjusted p < 0.05). Z1 scored numerically higher than Z0 for Qwen3.8 27B, but the Holm-adjusted p-value did not reach 0.05. Z2 changes relatively few correct predictions while Z3 substantially reduces accuracy.

Z2 substitutes `exp(logp_yes)` for conditional `p_yes` while retaining the original typed assembler's L1 normalization. Z3 hard-thresholds binary candidates before the same assembly; zero-sum and tie handling still follow the assembler.

## Authoritative results

- [Z0 raw records and model summary](results/z0_raw/)
- [Z0 vs Z1 paired comparison](results/comparisons/z0_z1_paired_comparison.csv)
- [Z1–Z3 offline summary](results/ablations/z1_z3_precomputed_summary.csv)
- [Z4 pooled coverage and accuracy](results/ablations/z4_topk_pooled_summary.csv)
- [Archived E1 results](../01_general_llm_benchmark/results/raw/)

## Reproduce offline analysis

From repository root (public dataset in `../jevbench/datasets/public/`):

```bash
python experiments/02_zero_shot_ablation/scripts/reproduce_z2_z4_ablation.py \\
  --results-dir experiments/01_general_llm_benchmark/results/raw \\
  --dataset-dir ../jevbench/datasets/public \\
  --out-dir experiments/02_zero_shot_ablation/results/recomputed
```

The prior [ablation manuscript notes](manuscript/ablation_atomic_revision.md) predate completion of Z0 and should not supersede the committed final results.
''',
'experiments/03_finetuned_decision_models/README.md': '''# E3 — Fine-tuned System-One decision models

**Status: COMPLETE (verified 2026-10-18).** Six native decision models were evaluated through Ollama's `/v1/systemone` interface, each on **231 public JevBench tasks**, with predeclared paired comparisons against general-purpose E1 models.

## Final benchmark

| Native model | E1 comparator | Correct / 231 | Accuracy | Delta vs E1 (pp) | Holm-adjusted p |
|---|---|---:|---:|---:|---:|
| `tev1:0.8b` | `qwen3.5:0.8b` | 140 | 60.61% | +8.66 | 0.0821 |
| `tev1:4b` | `qwen3.5:4b` | 175 | 75.76% | +4.76 | 0.2346 |
| `nimble-v3:9b` | `qwen3.5:9b` | 202 | 87.45% | +15.15 | **0.00000217** |
| `clef-flash:9b` | `qwen3.5:9b` | 190 | 82.25% | +9.96 | **0.00544** |
| `clef:27b` | `qwen3.8:latest` | 203 | 87.88% | +2.60 | 0.2379 |
| `pplx-decider:latest` | `qwen3.8:latest` | 205 | 88.74% | +3.46 | 0.2304 |

All **1,386 unique model–task evaluations** have valid successful final outcomes, with **zero final invalid responses or errors**. The `tev1:4b` raw JSONL includes 12 earlier failed attempts later retried successfully: the **latest row per task** is authoritative, rather than counting retries as new independent observations.

After Holm correction across six planned pairs, Nimble v3 9B and Clef Flash 9B have statistically significant positive differences versus their E1 comparators. The other numerical improvements should not be presented as Holm-significant.

## Protocol

The native endpoint receives one named `decision` question per request, preserving state, instructions, criteria, type, and candidate order. Choice uses the native chosen option checked against probabilities; Noul thresholds at 0.5; Score uses **argmax rubric level** rather than rounded expectation. Accuracy includes any invalid/errors in the full denominator, while calibration statistics use valid probability distributions.

## Artifacts

- [Six-model summary](results/summary/e3_model_summary.json)
- [Raw records](results/raw/)
- [E3 vs E1 paired results (CSV)](results/comparisons/e3_vs_e1_paired_comparison.csv)
- [E3 vs E1 paired results (JSON)](results/comparisons/e3_vs_e1_paired_comparison.json)
- [Native evaluation runner](scripts/run_e3_native.py)
- [Paired comparison script](scripts/compare_e3_to_e1.py)

## Verification / reproduction

Run from the repository root, with JevBench in `../jevbench/datasets/public/`:

```bash
# Recompute summaries from existing E3 results (no Ollama calls):
python experiments/03_finetuned_decision_models/scripts/run_e3_native.py --summary-only

# Recompute six paired comparisons:
python experiments/03_finetuned_decision_models/scripts/compare_e3_to_e1.py

# Optional: reproduce inference with required model tags installed:
python -u experiments/03_finetuned_decision_models/scripts/run_e3_native.py --timeout 240 --retries 1
```

**Limitation:** This is a comparison of deployed inference pipelines, **not a causal estimate of fine-tuning alone**. Model training, checkpoint precision, prompt formatting, runtime, and probability generation differ. The exact inference tag is `pplx-decider:latest` (not `pplx-decider:27b`). Report benchmark exposure risk and model licensing conditions. [Native API documentation](https://docs.ollama.com/api/systemone).
''',
'experiments/04_backend_fidelity/README.md': '''# E4 — Numerical integrity, cross-backend agreement and efficiency

**Status: COMPLETE (verified 2026-10-18).** E4A numerical replay and E4B paired cross-stack evaluation have committed results. E4B covers **four Qwen-family configurations × 231 tasks × 3 repetitions = 2,772 paired typed evaluations** and **10,188 binary candidate comparisons**. Both Ollama and Transformers succeeded on all 2,772 paired evaluations with **zero missing-verbalizer coverage or errors**.

## E4A — First-token probability integrity

The independent audit reconstructs the conditional two-token score from recorded first-token log-probabilities:

`q = exp(l_yes-m)/(exp(l_yes-m)+exp(l_no-m))`, where `m = max(l_yes, l_no)`.

For the 11 archived E1 models, **9,324 / 9,327** attempted binary candidates have both target verbalizers. The three uncovered evaluations belong to Granite 4.2 8B. The committed audit reports zero reconstruction discrepancy at recorded precision. This checks the numerical conversion, **not independently measured model logits**.

[Audit JSON](results/offline/e4a_e1_logprob_audit.json) · [Audit CSV](results/offline/e4a_e1_logprob_audit.csv).

```bash
python experiments/04_backend_fidelity/scripts/audit_e1_logits.py
```

## E4B — Ollama versus Transformers full-vocabulary logits

Both backends use the same LLM2Jev message-level binary prompts, candidate order, and typed response assembler:

- **Ollama:** one generated token, temperature 0, thinking disabled, top-20 first-token log-probabilities, lowercase `yes` and `no` both required.
- **Transformers:** prefill-only full-vocabulary next-token logits at `yes` and `no` single-token IDs, binary conditional softmax, batch size 1.
- **Protocol:** 231 distinct public tasks (139 Choice, 74 Noul, 18 Score), three paired repetitions per model. Timing excludes model initialization/warmup. Inference order alternates to reduce ordering effects.
- **Compute:** server has four NVIDIA RTX A6000 48 GB GPUs. The 27B HF checkpoint uses `CUDA_VISIBLE_DEVICES=1,2,3` with `--device-map balanced`; the recorded module device map confirms use of all three visible CUDA devices.

### Final full-run results

Accuracy is measured on the **first repetition of 231 independent tasks**. Probability MAE, agreement and per-decision latency aggregate all three repetitions.

| Model | Ollama correct / 231 | HF correct / 231 | Binary MAE | Pearson | Typed agreement | Ollama median (s) | HF median (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen3.5:0.8b` | 120 | 119 | 0.0220 | 0.9930 | 97.69% | 0.217 | 0.308 |
| `qwen3.5:4b` | 165 | 173 | 0.0519 | 0.9773 | 94.37% | 0.430 | 0.459 |
| `qwen3.5:9b` | 169 | 175 | 0.0674 | 0.9557 | 90.91% | 0.504 | 0.494 |
| `qwen3.8:latest` (27B) | 200 | 197 | 0.0329 | 0.9745 | 96.25% | 1.813 | 1.181 |

Each model has **693/693 successful paired task/repetition evaluations**, **2,547 candidate comparisons**, and **100% prompt-token count agreement**. Matching *counts* does not establish matching token ID sequences. The 27B score of 200/231 in E4 is a different run from its 197/231 E1 result.

### Authoritative artifacts

- [Combined full and pilot summaries (JSON)](results/paired/summary/e4_cross_stack_summary.json) / [CSV](results/paired/summary/e4_cross_stack_summary.csv)
- [Raw paired records](results/paired/raw/)
- [Model digests, checkpoint metadata and device maps](results/paired/metadata/)
- [Paired E4 runner](scripts/run_e4_paired.py) / [summary script](scripts/summarize_e4.py)

The one-task smoke tests and small pilot are **not independent additional full-run observations**.

## Reproduction

Run from repo root, with the public JevBench set in `../jevbench/datasets/public/` and the separately downloaded HF post-trained checkpoints:

```bash
# Summarize committed runs without new inference:
python experiments/04_backend_fidelity/scripts/summarize_e4.py

# Optional 27B full rerun; adjust resources/model path for your server:
CUDA_VISIBLE_DEVICES=1,2,3 python -u experiments/04_backend_fidelity/scripts/run_e4_paired.py \\
  --ollama-model qwen3.8:latest \\
  --hf-model "$HOME/hf_models/Qwen3.8-27B" \\
  --device cuda --device-map balanced --hf-dtype bfloat16 \\
  --full --repetitions 3 --timeout 180
```

## Important limitations

E4 measures **cross-stack agreement**, not exact same-weight logit fidelity. Ollama GGUF quantization differs from HF native checkpoints (0.8B Q8_0; 4B, 9B and 27B Q4_K_M), and the runtimes may differ in weight precision, chat serialization, token IDs, and kernels. The full-run prompt-token *count* agreement is not sufficient to prove identical tokenization. HF acceleration kernels may differ; its 27B model is sharded across three GPUs while Ollama's GPU placement is controlled by its daemon. Report latency as an observational property of these configurations, not an intrinsic speedup. Three repetitions improve stability estimates but are not 693 independent task samples per model.
''',
'experiments/README.md': '''# Ollama2Jev — Experimental evaluation

**Status: E1–E4 COMPLETE (verified 2026-10-18).** All four experimental phases have committed raw results, summaries and analysis scripts on branch `ollama2jev`.

| Phase | Scope | Final status | Primary evidence |
|---|---|---|---|
| **E1** | 11 general-purpose Ollama models × 231 public JevBench tasks | **COMPLETE** | [11-model summary](01_general_llm_benchmark/results/summary/ollama2jev_jevbench_11models_summary.json) |
| **E2** | 11-model direct zero-shot Z0 plus Z1–Z4 scoring ablations | **COMPLETE** | [Z0 summary](02_zero_shot_ablation/results/z0_raw/z0_model_summary.json), [offline ablations](02_zero_shot_ablation/results/ablations/), [paired tests](02_zero_shot_ablation/results/comparisons/) |
| **E3** | 6 native System-One models × 231 tasks and six paired comparisons | **COMPLETE** | [Six-model summary](03_finetuned_decision_models/results/summary/e3_model_summary.json), [paired results](03_finetuned_decision_models/results/comparisons/) |
| **E4A** | Archived E1 conditional-probability reconstruction for 9,327 binary attempts | **COMPLETE** | [E4A log-probability audit](04_backend_fidelity/results/offline/e4a_e1_logprob_audit.json) |
| **E4B** | 4 Qwen-family models × 231 tasks × 3 paired repetitions | **COMPLETE** | [Cross-backend summary](04_backend_fidelity/results/paired/summary/e4_cross_stack_summary.json) |

## Verified totals

- **E1:** 2,541 attempted model–task decisions; 2,538 successful typed results and 3 Granite 4.2 8B coverage failures; 1,648 correct.
- **E2 Z0:** 2,541 completed direct-generation attempts; 2,513 valid and 28 invalid labels; 1,722 correct. **Z1:** 1,648 correct; **Z2:** 1,643; **Z3:** 1,388. Z4 top-20 coverage is 9,324/9,327 candidates.
- **E3:** 1,386/1,386 unique model–task pairs have valid final results after recorded retries. Nimble v3 9B and Clef Flash 9B are Holm-significantly better than their paired E1 baselines.
- **E4B:** 2,772/2,772 paired typed task/repetition measurements, **10,188 paired binary candidates**, zero backend failures; all four 231-task model sets have three repetitions. The Transformers 27B checkpoint was sharded across three visible GPUs.

## How to reproduce analyses

Run from the repository root with the public JevBench dataset in the sibling directory `../jevbench/datasets/public/`:

```bash
# E2: recompute Z1–Z4 offline from archived E1 raw files.
python experiments/02_zero_shot_ablation/scripts/reproduce_z2_z4_ablation.py \\
  --results-dir experiments/01_general_llm_benchmark/results/raw \\
  --dataset-dir ../jevbench/datasets/public \\
  --out-dir experiments/02_zero_shot_ablation/results/recomputed

# E3: verify final native-model summaries and paired contrasts.
python experiments/03_finetuned_decision_models/scripts/run_e3_native.py --summary-only
python experiments/03_finetuned_decision_models/scripts/compare_e3_to_e1.py

# E4: verify numerical reconstruction and regenerate paired comparison summaries.
python experiments/04_backend_fidelity/scripts/audit_e1_logits.py
python experiments/04_backend_fidelity/scripts/summarize_e4.py
```

## Research-integrity and provenance notes

- **E1 and E4 are different inference runs**; do not replace E1 model scores with E4 first-repetition results. `qwen3.8:latest` was 197/231 correct in E1 and 200/231 in E4.
- **Z0 does not produce probability distributions**, while Z1–Z3 use probabilistic typed assembly. Comparisons should distinguish accuracy from calibration metrics.
- **Z4 is offline top-k rank truncation**, not evidence of runtime speedup at altered Ollama top-k settings.
- **E3 compares deployed systems**, not the isolated effect of fine-tuning. Different model formats and inference paths prevent causal attribution.
- **E4 is cross-stack agreement, not exact same-weight logit equivalence**, because HF model checkpoints and Ollama GGUF representations may differ. Prompt-token *count* parity does not prove token ID parity. Accuracy uses one observation per task (first repetition) and repeats are retained for stability and latency, not inflated sample size.
- Preserve model digests, checkpoint metadata, seeds, versions, GPU allocations, dataset fingerprints and underlying raw outputs.

Read the individual [E1](01_general_llm_benchmark/README.md), [E2](02_zero_shot_ablation/README.md), [E3](03_finetuned_decision_models/README.md) and [E4](04_backend_fidelity/README.md) reports for methods, exact artifacts and limitations.
''',
'README.md': '''<p align="center"><img src="assets/Ollama2Jev.png" alt="Ollama2Jev" width="100%"></p>

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
'''
}

for rel, content in DOCS.items():
    path=root/rel
    if not path.is_file():
        sys.exit(f'Missing README: {path}')
    backup=path.with_suffix(path.suffix+'.pre_e1_e4_docs.bak')
    if not backup.exists():
        backup.write_text(path.read_text(encoding='utf-8'),encoding='utf-8')
    path.write_text(content,encoding='utf-8')
    print(f'UPDATED {rel}: {len(content)} chars (backup {backup.name})')
print('Done. Review git diff and commit six README files only; backups are untracked.')
