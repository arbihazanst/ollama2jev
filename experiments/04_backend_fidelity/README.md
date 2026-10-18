# E4 — Numerical integrity, cross-backend agreement, and efficiency

**Status: COMPLETE (11-model full-run results verified 2026-10-18).** E4 has two complementary components. **E4A** checks the numerical reconstruction of conditional `yes`/`no` probabilities from archived E1 top-logprob data. **E4B** compares Ollama and Hugging Face Transformers implementations for all **11 general-purpose E1 model configurations** over **231 JevBench tasks × 3 repetitions**. The raw records, model-level metadata, and aggregate summaries are committed to the repository.

> **Full-run E4B totals:** 7,623 attempted paired task/repetition evaluations; **7,617 successful pairs**; **27,987 paired binary candidate comparisons**; six Ollama missing-verbalizer coverage failures (all in `granite4.2:8b`); no recorded non-coverage backend errors. A run can be *complete* (all attempts recorded) without all attempts being *valid pairs*.

## E4A — First-token probability integrity

The E4A independent arithmetic replay reconstructs the conditional binary score from recorded first-token log-probabilities:

\[
P(\mathrm{yes}\mid\{\mathrm{yes},\mathrm{no}\})
= \frac{\exp(\ell_{\mathrm{yes}}-m)}{\exp(\ell_{\mathrm{yes}}-m)+\exp(\ell_{\mathrm{no}}-m)},
\quad m=\max(\ell_{\mathrm{yes}},\ell_{\mathrm{no}}).
\]

Across the 11 archived E1 configurations, **9,324 of 9,327** attempted binary candidates exposed both required lowercase verbalizers in the returned top-20 token list. The three uncovered candidates belong to `granite4.2:8b`. The reconstructed scores match the recorded probabilities at the precision available in the logs (zero reported reconstruction discrepancy).

**Interpretation:** this verifies the arithmetic and archived coverage, **not** equivalence with independently measured full-vocabulary logits.

- [E4A audit JSON](results/offline/e4a_e1_logprob_audit.json) · [CSV](results/offline/e4a_e1_logprob_audit.csv)
- [E4A audit script](scripts/audit_e1_logits.py)

## E4B — Paired Ollama versus Transformers evaluation

### Protocol

- **Tasks:** 231 distinct public JevBench items: 139 Choice, 74 Noul, and 18 Score.
- **Models:** the same 11 nominal configurations evaluated in E1, paired with their locally downloaded Hugging Face checkpoints.
- **Repetitions:** three per model/task, for 693 attempts per model. The paired execution order alternates between backends.
- **Ollama backend:** one generated token, temperature 0, thinking disabled, top-20 first-token log-probabilities; both lowercase `yes` and `no` must be returned for each binary candidate.
- **Transformers backend:** prefill-only full-vocabulary next-token logits for the respective `yes` and `no` token IDs, with conditional binary normalization and batch size 1.
- **Typed decision assembly:** both backends reuse the LLM2Jev binary-question compiler, prompt messages, candidate ordering, and assembler.
- **Hardware:** four NVIDIA RTX A6000 48 GB GPUs. The 27B Hugging Face run was sharded across three visible GPUs; different model runs may use different placement and precision.
- **Latency:** timed requests exclude initialization/warmup; latency comparisons remain observational under these configurations.

Accuracy is calculated from **the first repetition only**, counting an unsuccessful attempted task as incorrect (denominator 231). Agreement, candidate probability MAE, Pearson correlation, and latency use successful backend outputs across the three repetitions. Repetitions do **not** increase the independent task sample size from 231 to 693.

### E4B final results — all 11 models

| Ollama model | Ollama correct / 231 | HF correct / 231 | Valid pairs / 693 | Typed agreement | Binary MAE | Pearson | Prompt-token *count* match | Ollama p50 (s) | HF p50 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen3.5:0.8b` | 120 | 119 | 693 | 97.69% | 0.0220 | 0.9930 | 100% | 0.217 | 0.308 |
| `qwen3.5:2b` | 144 | 151 | 693 | 94.08% | **0.0140** | 0.9970 | 100% | 0.227 | 0.373 |
| `qwen3.5:4b` | 165 | 173 | 693 | 94.37% | 0.0519 | 0.9773 | 100% | 0.430 | 0.459 |
| `qwen3.5:9b` | 169 | 175 | 693 | 90.91% | 0.0674 | 0.9557 | 100% | 0.504 | 0.494 |
| `qwen3.8:latest` (27B) | 200 | 197 | 693 | 96.25% | 0.0329 | 0.9745 | 100% | 1.813 | 1.181 |
| `llama3.2:1b` | 92 | 86 | 693 | 75.47% | 0.1011 | 0.9001 | **0%** | 0.099 | 0.142 |
| `llama3.2:3b` | 154 | 146 | 693 | 76.77% | 0.1031 | 0.8995 | **0%** | 0.118 | 0.219 |
| `granite4.2:3b` | 150 | 150 | 693 | 76.91% | 0.1834 | 0.7397 | 100% | 0.154 | 0.294 |
| `granite4.2:8b` | 147 | 154 | **687** | 90.54% | 0.0419 | 0.9199 | 100%* | 0.182 | 0.366 |
| `gemma4:e2b` | 146 | 149 | 693 | 84.85% | 0.0991 | 0.7378 | 100% | 0.227 | 0.475 |
| `gemma4:e4b` | 165 | 176 | 693 | 87.01% | 0.0753 | 0.8808 | 100% | 0.284 | 0.539 |

**Notes:** "Ollama correct" and "HF correct" are first-repetition counts and belong to **E4, not E1**. Typed agreement is conditional on valid paired outcomes (7,617 pairs overall). Binary probability statistics are computed over **27,987** aligned binary candidates. The prompt-token count match is computed for available valid pairs: *Granite 8B's 100% applies to the **687** valid pairs*, not its six coverage failures.

### E4B coverage and probability diagnostics

- **Model coverage:** ten models contribute 693 successful pairs each; `granite4.2:8b` contributes 687 and has six Ollama top-20 coverage failures. All HF evaluations succeeded; no non-coverage backend errors were recorded.
- **Failure mechanism:** all six Granite failures have `yes_found=False` and `no_found=True`; lower-case `yes` does not occur in the returned top-20 first-token candidates. They affect **four distinct tasks**: `easy-intent-11`, `easy-tool_selection-00`, `easy-tool_selection-02`, and `easy-tool_selection-05`.
- **Overall agreement:** the successful-pair-weighted typed-decision agreement is approximately **87.71%**. Agreement can be high even when both outputs are incorrect.
- **Smallest probability error:** `qwen3.5:2b` has the lowest candidate-level MAE, **0.0140**.
- **Largest probability error:** `granite4.2:3b` has the highest MAE, **0.1834**. The Qwen 0.8B model has the highest typed agreement (**97.69%**).
- **Prompt-token count parity:** Llama 3.2 1B and 3B have **0%** reported count matches; the other nine nominal pairs have **100%** over their valid paired evaluations. This diagnostic is an equality of **counts, not token ID sequences or serialized chat templates**; the underlying reason for the Llama mismatch is not yet established.

These differences characterize **cross-stack consistency under the measured configurations**, not exact-logit equivalence. Model quantization, chat rendering, token accounting, and numerical kernels may differ.

### Authoritative artifacts

- [Full and pilot summaries (JSON)](results/paired/summary/e4_cross_stack_summary.json) · [CSV](results/paired/summary/e4_cross_stack_summary.csv) — **filter for filenames ending in `_full_r3.jsonl` to obtain exactly 11 full runs**; smaller pilots are not additional model results.
- [Canonical consolidated E4B raw JSONL records](results/paired/raw/) — includes all 11 full runs and additional small pilots.
- [Consolidated metadata (partial collection)](results/paired/metadata/) — contains metadata for several original and additional runs.
- [Original per-model metadata](results/full/) — metadata missing from the consolidated metadata directory for some additional models remain preserved under `results/full/<model>/metadata/`; copy them only after verifying matching configuration fingerprints.
- [E4 runner](scripts/run_e4_paired.py) · [E4 summarizer](scripts/summarize_e4.py)

## Reproduce the offline E4 analyses

Run from repository root with the public JevBench data at `../jevbench/datasets/public/`:

```bash
# E4A: algebraic audit of E1 archived first-token probabilities.
python experiments/04_backend_fidelity/scripts/audit_e1_logits.py

# E4B: regenerate JSON/CSV from already committed paired/raw JSONL files.
python experiments/04_backend_fidelity/scripts/summarize_e4.py

# Count only full (231-task, three-repetition) runs, not pilots.
python - <<'PY'
import json
from pathlib import Path
p = Path('experiments/04_backend_fidelity/results/paired/summary/e4_cross_stack_summary.json')
a = json.loads(p.read_text())
full = [r for r in a if r['source_jsonl'].endswith('_full_r3.jsonl')]
assert len(full) == 11
print('Models:', len(full))
print('Attempts:', sum(r['recorded_model_task_repetitions'] for r in full))
print('Valid pairs:', sum(r['paired_type_decisions'] for r in full))
print('Binary comparisons:', sum(r['paired_binary_candidates'] for r in full))
print('Ollama coverage failures:', sum(r['ollama_status']['coverage_failure'] for r in full))
PY
```

Expected: `11` models; `7623` attempts; `7617` valid pairs; `27987` binary comparisons; `6` Ollama coverage failures.

For model-level reproductions, see [the paired runner](scripts/run_e4_paired.py) and use the correct local Hugging Face checkpoint path, Ollama tag, visible GPUs, and dtype. Rerunning models may change outputs because hardware allocation, quantization, software, or prompt handling can differ.

## Important limitations

1. **Not an exact same-weight experiment:** Ollama GGUF and HF checkpoint representations can differ in quantization, formats, runtimes, token IDs, or prompts. E4 assesses cross-stack consistency rather than claiming mathematically identical logit distributions.
2. **Prompt-token count is only a diagnostic:** 100% matching *counts* for nine models does not certify identical input token sequences. Both Llama model pairs show systematic count mismatches; explaining their cause requires a separate tokenized-input inspection.
3. **Valid-pair conditioning:** agreement and probability errors exclude six Granite coverage-failed pairs, while accuracy counts all 231 first-repetition attempts. E4A's three missing E1 candidates and E4B's six missing evaluations have different denominators and should not be combined.
4. **Nonindependent repetitions:** three repeats assess variability and runtime rather than independent task accuracy; task-level statistical analyses should use one task observation per model or cluster by task.
5. **Observational latency:** device sharding and Ollama daemon GPU placement may differ, so request latencies should not be interpreted as isolated backend speed advantages.
6. **Separate experimental phases:** E1 first-run accuracy and E4 first-repetition accuracy are not interchangeable; e.g., `qwen3.8:latest` is 197/231 in E1 versus 200/231 in E4.
