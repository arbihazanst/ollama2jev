# Ollama2Jev — Experimental evaluation

**Status: E1–E5 COMPLETE (26 October 2026).** Committed results are on the [`ollama2jev` branch](https://github.com/arbihazanst/ollama2jev/tree/ollama2jev). Different experiments answer different systems and evaluation questions; their denominators must not be combined.

## Experiment matrix

| Phase | Scope | Result | Detailed report |
|---|---|---|---|
| E1 | 11 general-purpose Ollama configurations × 231 public JevBench tasks (2,541 attempts) | 2,538 valid decisions; best Qwen3.8 27B 197/231 (85.28%) | [E1](01_general_llm_benchmark/README.md) |
| E2 | Same 11 models: Z0 direct generation; Z1 conditional scoring; offline Z2/Z3/Z4 replays | Z0 67.77% pooled accuracy vs Z1 64.86%; effects depend on model | [E2](02_zero_shot_ablation/README.md) |
| E3 | 6 native specialized System-One configurations × 231 tasks | 1,386 valid outcomes; PPLX Decider 205/231 (88.74%) | [E3](03_finetuned_decision_models/README.md) |
| E4A | Algebraic reconstruction from 9,327 archived E1 binary candidate attempts | 9,324 covered, 3 uncovered, exact arithmetic at saved precision | [E4](04_backend_fidelity/README.md) |
| E4B | 11 nominal Ollama–Transformers pairs × 231 tasks × 3 repetitions | 7,617 valid / 7,623 attempted paired decisions; 27,987 paired binary candidates; 87.71% typed agreement | [E4](04_backend_fidelity/README.md) |
| E5 | 11 installed tags; 5 Qwen model dual-use verifications across runs; 6 manifest storage pairs | All five general-purpose model tags have successful archived same-checkpoint generation/scoring records; 0.78–18.78 GiB extra pairwise specialized blob content | [E5](05_deployment_audit/README.md) |

## E5 provenance and verification

The following five Qwen configurations each passed archived generation and first-position binary-scoring checks using unchanged served tags, SHA-256 model digests and installed-tag sets: `qwen3.5:0.8b`, `qwen3.5:2b`, `qwen3.5:4b`, `qwen3.5:9b`, and `qwen3.8:latest` (27B).

- [20 October 2026 corrected-run probe events](05_deployment_audit/outputs/dual_use_corrected/probe_events.json): 9B and 27B pass.
- [22 October 2026 consolidated-run probe events](05_deployment_audit/outputs/dual_use_all_qwen_20261023/probe_events.json): 0.8B, 2B, 4B and 9B pass; the 27B request times out in this batch, but its earlier verification passes.
- [Separate small-model verification](05_deployment_audit/outputs/dual_use_small_models/probe_events.json): successful 0.8B and 4B records.

**Interpretation:** all five configurations have successful archived verification, **not** five successes in one uninterrupted batch. E5 probes the required serving interfaces but is **not** another full typed JevBench benchmark. E1–E4 used Ollama 0.35.0, E5 used Ollama 0.40.5. Six pairwise specialized-model blob differences range from 0.78 to 18.78 GiB and cannot be read as physical disk or GPU-memory savings.

## Important boundaries

- **E1:** Strict model-level accuracy counts all 231 attempts as the denominator; Granite 4.2 8B has three missing-verbalizer cases.
- **E2:** Z0 and Z1 differ in prompt and elicitation, not solely the probability readout. Qwen3.8 27B's +6.93 pp Z1 advantage is not Holm-significant; Qwen3.5 4B and 9B have statistically supported Z0 advantages. Z2–Z4 are offline replays.
- **E3:** Comparisons use different model artifacts and native endpoints; they are not a controlled causal evaluation of fine-tuning.
- **E4:** E4A checks saved arithmetic; E4B measures cross-stack *deployed-system* agreement, not identical weights/token IDs. Three repetitions per task do not yield three statistically independent items.
- **E5:** Checkpoint model-tag/digest stability is verified; no full JevBench rerun, training cost, physical disk savings, GPU-memory savings, or power savings are measured.

## Reproduce archived analyses

From the repository root, with public JevBench data located at `../jevbench/datasets/public/`:

```bash
python experiments/02_zero_shot_ablation/scripts/reproduce_z2_z4_ablation.py \
  --results-dir experiments/01_general_llm_benchmark/results/raw \
  --dataset-dir ../jevbench/datasets/public \
  --out-dir experiments/02_zero_shot_ablation/results/recomputed

python experiments/03_finetuned_decision_models/scripts/run_e3_native.py --summary-only
python experiments/03_finetuned_decision_models/scripts/compare_e3_to_e1.py
python experiments/04_backend_fidelity/scripts/audit_e1_logits.py
python experiments/04_backend_fidelity/scripts/summarize_e4.py
```

E4 summary consumers should select only full model-pair raw files ending `_full_r3.jsonl` (11 full runs), excluding smaller pilots. See the detailed [E4 report](04_backend_fidelity/README.md) for exact totals and scripts. The E5 audit script and interpretation guidance are in [E5 README](05_deployment_audit/README.md).
