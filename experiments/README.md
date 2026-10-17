# Ollama2Jev evaluation experiments

This directory separates **completed measurements**, **offline re-analysis**, and **proposed (not executed)** evaluations. The original 11-model raw JevBench JSONL files stay at the repository root to preserve existing links and provenance; do not duplicate or overwrite them here.

| ID | Experiment | Status | Evidence |
|---|---|---|---|
| E1 | General-purpose LLM benchmark (11 Ollama configurations × 231 decisions) | **COMPLETED** | Root `ollama2jev_jevbench_*_full.jsonl` and `ollama2jev_jevbench_11models_summary.json` |
| E2 Z0 | Direct zero-shot label-generation baseline | **PLANNED — NOT RUN** | Protocol in `02_zero_shot_ablation/README.md` |
| E2 Z1 | Full Ollama2Jev scoring (original method) | **COMPLETED** | Reconstructed from E1 raw outputs |
| E2 Z2–Z3 | Yes-only and hard-score offline ablations | **COMPLETED (OFFLINE)** | `02_zero_shot_ablation/results/` and replay script |
| E2 Z4 | Effective top-k = 2, 5, 10, 20 coverage replay | **COMPLETED (OFFLINE)** | `02_zero_shot_ablation/results/` and replay script |
| E3 | Native fine-tuned decision-model comparison | **PLANNED — NOT RUN** | `03_finetuned_decision_models/README.md` |
| E4 | Controlled exact-logit fidelity / scalability evaluation | **PLANNED — NOT RUN** | `04_backend_fidelity/README.md` |

## Reproduce completed offline ablations

From the **repository root**, with the original JevBench public data available in the sibling directory `../jevbench/datasets/public/`:

```bash
python experiments/02_zero_shot_ablation/scripts/reproduce_z2_z4_ablation.py \
  --results-dir . \
  --dataset-dir ../jevbench/datasets/public \
  --out-dir experiments/02_zero_shot_ablation/results/recomputed
```

The command reads the original 11 JSONL artifacts, requires 231 matching JevBench item IDs, and checks that replayed Z1 correctness matches recorded outputs. It does not contact Ollama or fine-tune models. Output includes model metrics, paired McNemar counts, top-k coverage, and a JSON summary.

## Research-integrity rules

- Never describe Z0, E3, or E4 as completed until their outputs are committed and checked.
- Z4 is a **post-hoc rank-truncation replay** using saved top-20 diagnostics, not a new `top_logprobs=k` inference run and not evidence of speedups.
- Z3 hard-thresholding creates ties and point masses; NLL results depend on clipping and should not be used as primary calibration claims.
- A **full 231-task denominator per model** counts coverage failures as incorrect. Probability metrics only use completed distributions (2,538 across all models).
- Record exact model digests, quantization, software versions, source commits, hardware, and random seeds (when applicable) for future experiments.

See each experiment README for task protocol, metrics, and comparability caveats.
