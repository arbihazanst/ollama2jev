# Ollama2Jev — Experimental evaluation

**Status: E1–E4 COMPLETE (updated 2026-10-18).** All four phases have committed model outputs and analysis artifacts on the `ollama2jev` branch. E4B now includes the full **11-model E1-aligned comparison**, not just the initial four-Qwen pilot scope.

## Experiment matrix

| Phase | Research purpose | Experimental scope | Status and evidence |
|---|---|---|---|
| **E1** | General-purpose LLM feasibility through one-token binary probability scoring | 11 Ollama models × 231 public JevBench tasks | **Complete** — [11-model results](01_general_llm_benchmark/results/summary/ollama2jev_jevbench_11models_summary.json) |
| **E2** | Direct-generation baseline and probability-scoring ablations | 11 models; direct Z0 versus archived Z1, offline Z2/Z3 and top-k rank replay Z4 | **Complete** — [Z0 results](02_zero_shot_ablation/results/z0_raw/z0_model_summary.json), [paired comparisons](02_zero_shot_ablation/results/comparisons/), [ablations](02_zero_shot_ablation/results/ablations/) |
| **E3** | Specialized native System-One decision-model comparison | 6 native decision models × 231 tasks; 6 predefined paired E1 comparator tests | **Complete** — [summary](03_finetuned_decision_models/results/summary/e3_model_summary.json), [paired tests](03_finetuned_decision_models/results/comparisons/) |
| **E4A** | Numerical integrity of E1 conditional two-verbalizer probabilities | 9,327 archived E1 binary candidate attempts | **Complete** — [logprob audit](04_backend_fidelity/results/offline/e4a_e1_logprob_audit.json) |
| **E4B** | Cross-stack agreement and probability consistency (Ollama versus Transformers) | **11 model pairs × 231 tasks × 3 repetitions = 7,623 paired attempts** | **Complete, with six coverage failures** — [11-model summary](04_backend_fidelity/results/paired/summary/e4_cross_stack_summary.json) |

## Final verified results

### E1 — General-purpose Ollama2Jev

- **2,541 attempted typed decisions** (11 models × 231 JevBench tasks), of which **2,538 completed successfully**; three missing-verbalizer failures occur in `granite4.2:8b`.
- **1,648/2,541 = 64.86%** pooled strict accuracy, counting unsuccessful tasks as incorrect.
- Highest E1 accuracy: **`qwen3.8:latest` with 197/231 = 85.28%**.
- E1 uses one-token, first-token top-20 `yes`/`no` logprob scoring and typed decision assembly. [Detailed E1 README](01_general_llm_benchmark/README.md).

### E2 — Direct generation and score ablations

| Variant | Method | Correct / 2,541 | Strict accuracy |
|---|---|---:|---:|
| Z0 | Direct zero-shot final-label generation | 1,722 | **67.77%** |
| Z1 | Archived E1 conditional `yes`/`no` scoring and typed assembly | 1,648 | 64.86% |
| Z2 | Yes-only likelihood with the same typed assembler (offline replay) | 1,643 | 64.66% |
| Z3 | Hard-thresholded binary scores with the same typed assembler (offline replay) | 1,388 | 54.62% |

- Z0 includes **2,513 valid labels and 28 invalid responses** out of 2,541 attempts. Z0 does **not** provide directly measured probability distributions comparable to Z1–Z3.
- After paired exact McNemar tests and Holm correction over 11 models, direct Z0 is significantly more accurate than Z1 for **Qwen3.5 4B**, **Qwen3.5 9B**, and **Gemma 4 E4B**. Z1 is not uniformly superior to direct generation.
- Z4 is an **offline rank-truncation replay** of existing top-20 scores; strict accuracies at effective k = 2/5/10/20 are **47.54% / 61.08% / 64.23% / 64.86%**. Its results do not establish latency changes from a live smaller top-k API setting.
- [Detailed E2 README](02_zero_shot_ablation/README.md).

### E3 — Native System-One decision models

| Native decision model | E1 comparator | E3 correct / 231 | E3 accuracy | Gain over E1 (pp) | Holm-adjusted p |
|---|---|---:|---:|---:|---:|
| `tev1:0.8b` | `qwen3.5:0.8b` | 140 | 60.61% | +8.66 | 0.0821 |
| `tev1:4b` | `qwen3.5:4b` | 175 | 75.76% | +4.76 | 0.2346 |
| `nimble-v3:9b` | `qwen3.5:9b` | 202 | 87.45% | **+15.15** | **0.00000217** |
| `clef-flash:9b` | `qwen3.5:9b` | 190 | 82.25% | **+9.96** | **0.00544** |
| `clef:27b` | `qwen3.8:latest` | 203 | 87.88% | +2.60 | 0.2379 |
| `pplx-decider:latest` | `qwen3.8:latest` | 205 | **88.74%** | +3.46 | 0.2304 |

- All **1,386 unique model–task outcomes** have valid final results (after documented retries where required).
- **Nimble v3 9B** and **Clef Flash 9B** have the two Holm-significant positive paired contrasts. PPLX Decider has the highest absolute native-model accuracy, but its paired advantage is not Holm-significant.
- The comparisons evaluate **deployed inference systems**, not the isolated causal contribution of fine-tuning. [Detailed E3 README](03_finetuned_decision_models/README.md).

### E4 — Numerical replay and cross-stack consistency

- **E4A:** **9,324/9,327** archived E1 binary candidates expose both `yes` and `no`; three uncovered candidates occur in Granite 4.2 8B. Conditional-softmax reconstruction discrepancy is zero at recorded precision; it is **not an independent exact-logit comparison**.
- **E4B:** **7,623 attempted paired task/repetition evaluations**, **7,617 valid pairs**, and **27,987 valid paired binary candidate comparisons** across **11 models**. Six additional Ollama top-20 verbalizer-coverage failures occur in Granite 4.2 8B; all HF evaluations succeeded and zero non-coverage backend errors were recorded.
- Successful-pair-weighted cross-stack typed agreement: approximately **87.71%**. Per-model agreement ranges from **75.47% (Llama 3.2 1B)** to **97.69% (Qwen3.5 0.8B)**. The smallest candidate-level probability MAE is **0.0140 (Qwen3.5 2B)**.
- The two Llama 3.2 model pairs report **0% prompt-token-count parity**; the other nine report **100%** for their valid pairs. Count parity is *not* identical tokenization; the precise cause of Llama's mismatch is not established.
- [Detailed 11-model E4 report and full table](04_backend_fidelity/README.md).

## Reproduce offline analyses

From the repository root, with the public JevBench set at `../jevbench/datasets/public/`:

```bash
# E2: reconstruct archived score ablations; no new inference.
python experiments/02_zero_shot_ablation/scripts/reproduce_z2_z4_ablation.py \
  --results-dir experiments/01_general_llm_benchmark/results/raw \
  --dataset-dir ../jevbench/datasets/public \
  --out-dir experiments/02_zero_shot_ablation/results/recomputed

# E3: use final latest-per-task native results and paired exact comparisons.
python experiments/03_finetuned_decision_models/scripts/run_e3_native.py --summary-only
python experiments/03_finetuned_decision_models/scripts/compare_e3_to_e1.py

# E4: audit archived E1 logprobs and recompute the 11-model cross-stack summary.
python experiments/04_backend_fidelity/scripts/audit_e1_logits.py
python experiments/04_backend_fidelity/scripts/summarize_e4.py
```

The E4 summarizer also processes small smoke-test and pilot files in `results/paired/raw/`. For scientific reporting, use only summary records whose `source_jsonl` ends in **`_full_r3.jsonl`** (exactly 11 model pairs).

## Research-integrity and provenance notes

1. **Separate E1 and E4 inference runs.** Do not replace original E1 accuracy with E4 first-repetition accuracy. For instance, `qwen3.8:latest` scored **197/231** in E1 versus **200/231** on E4's Ollama first repetition.
2. **Separate methodological questions.** E1 establishes typed local scoring feasibility; E2 compares decision-scoring methods; E3 compares native specialized deployed pipelines; E4 checks numerical reconstruction and cross-stack consistency.
3. **Use full-attempt denominators.** Count invalid or uncovered decisions as incorrect for strict accuracy; restrict probability calibration, MAE, and agreement to cases with valid outputs. Report the denominator for each quantity.
4. **Preserve paired statistical inference.** The E2 and E3 comparisons use paired exact McNemar tests with Holm correction. Descriptive pooled accuracy and unadjusted p-values are not substitutes for the planned paired significance tests.
5. **No causal fine-tuning claim from E3.** Training/checkpoint format, prompts, quantization, and endpoints may all differ among the compared deployed systems.
6. **No exact same-weight equivalence claim from E4.** The two backends can differ in quantization, token IDs, chat serialization, and kernels. Identical *prompt-token counts* do not prove identical token sequences; Llama token-count mismatches remain a documented diagnostic.
7. **Account for correlated repetitions.** E4 has 231 independent tasks per model; three repetitions are for stability/latency characterization and should not be treated as 693 statistically independent tasks.
8. **Observational latency only.** GPU placement, sharding and daemon scheduling can differ among runs. Keep model digests, dataset fingerprints, configuration hashes, precision, software versions, and raw outputs intact.

For details consult the [E1](01_general_llm_benchmark/README.md), [E2](02_zero_shot_ablation/README.md), [E3](03_finetuned_decision_models/README.md), and [E4](04_backend_fidelity/README.md) reports.
