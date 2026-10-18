# E2 — Zero-shot baselines and scoring ablations

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
python experiments/02_zero_shot_ablation/scripts/reproduce_z2_z4_ablation.py \
  --results-dir experiments/01_general_llm_benchmark/results/raw \
  --dataset-dir ../jevbench/datasets/public \
  --out-dir experiments/02_zero_shot_ablation/results/recomputed
```

The prior [ablation manuscript notes](manuscript/ablation_atomic_revision.md) predate completion of Z0 and should not supersede the committed final results.
