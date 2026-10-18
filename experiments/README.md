# Ollama2Jev — Experimental evaluation

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
python experiments/02_zero_shot_ablation/scripts/reproduce_z2_z4_ablation.py \
  --results-dir experiments/01_general_llm_benchmark/results/raw \
  --dataset-dir ../jevbench/datasets/public \
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
