# E2 — Zero-shot baseline and scoring ablations

## Status

- **Z0 (direct zero-shot output generation): PLANNED; no results yet.**
- **Z1 (full Ollama2Jev): COMPLETE.** Reconstructed from the 11 model-specific JevBench JSONL results.
- **Z2 (yes-only unconditional likelihood): COMPLETE, offline replay.**
- **Z3 (hard-thresholded binary scores): COMPLETE, offline replay.**
- **Z4 (effective top-k sensitivity): COMPLETE, offline rank-truncation replay.**

The saved first-token yes/no log-probabilities and ranks permit Z2–Z4 offline analysis; no additional model calls are required. See `scripts/reproduce_z2_z4_ablation.py` and `results/`.

## Variant definitions

**Z0 — not yet executed:** Ask each general-purpose model for the final allowed label or integer directly with the same JevBench state, instructions, and options. Use no demonstrations, temperature=0, no thinking if supported, and sufficient `num_predict` for a complete answer. Record the exact prompt renderer, extraction policy, invalid-output rate, time, and model digests. Do not pretend that generated labels provide probability distributions: Brier/ECE/NLL are not available without an additional defensible probabilistic extraction procedure.

**Z1 — full:** For candidate *i*, `q_i = exp(logp_yes_i)/(exp(logp_yes_i)+exp(logp_no_i))`; use the original typed assembler.

**Z2 — yes-only:** Replace `q_i` with `s_i = exp(logp_yes_i)`; keep the same L1 normalization for multi-candidate questions and Noul rules. This removes *binary yes/no renormalization* but **not** the typed assembler's L1 normalization.

**Z3 — hard:** Replace `q_i` with `1[q_i >= 0.5]`; keep the assembler, including uniform zero-sum fallback and deterministic tie rules. Interpret probability-quality metrics cautiously: this variant intentionally destroys continuous information.

**Z4 — top-k replay:** At effective k in `{2,5,10,20}`, a binary candidate counts as covered iff **both** saved verbalizer ranks are at most k. A typed decision counts as complete iff all its candidates are covered. For covered tasks, use the original Z1 decision. Incomplete decisions count as incorrect for strict accuracy. **This is not a new Ollama inference timing experiment.**

## Primary result (pooled 11 models)

| Variant | Correct / 2,541 attempts | Accuracy | Mean Brier among 2,538 completed decisions |
|---|---:|---:|---:|
| Z1 | 1,648 | 0.6486 | 0.4794 |
| Z2 | 1,643 | 0.6466 | 0.4817 |
| Z3 | 1,388 | 0.5462 | 0.5342 |

| Effective k | Typed decisions completed | Candidate checks covered | Strict accuracy |
|---|---:|---:|---:|
| 2 | 1,985/2,541 | 7,989/9,327 | 0.4754 |
| 5 | 2,421/2,541 | 9,106/9,327 | 0.6108 |
| 10 | 2,518/2,541 | 9,292/9,327 | 0.6423 |
| 20 | 2,538/2,541 | 9,324/9,327 | 0.6486 |

### Interpretation boundaries

Z2 causes a small net decrease (5 correct decisions) and does not establish that conditional yes/no renormalization is consistently superior. Z3 reduces correct decisions by 260, indicating that candidate probability magnitudes matter for the typed assembly. Z4 demonstrates a coverage/accuracy trade-off under an **offline** truncation assumption; it does not measure latency differences from changing the actual top-k API parameter.

## Reproduction

From repository root:

```bash
python experiments/02_zero_shot_ablation/scripts/reproduce_z2_z4_ablation.py \
  --results-dir . --dataset-dir ../jevbench/datasets/public \
  --out-dir experiments/02_zero_shot_ablation/results/recomputed
```

The committed `results/z1_z3_precomputed_summary.csv` and `results/z4_topk_pooled_summary.csv` are compact checked summaries from the initial offline analysis. The script writes a more detailed set of results under `results/recomputed/` for verification. See `manuscript/ablation_atomic_revision.md` for LaTeX insertion and wording.
