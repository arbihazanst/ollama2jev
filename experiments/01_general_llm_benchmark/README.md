# E1 — General-purpose Ollama2Jev benchmark

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
