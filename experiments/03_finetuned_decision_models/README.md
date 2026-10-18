# E3 — Fine-tuned System-One decision models

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
