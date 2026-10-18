# E4 — Numerical integrity, cross-backend agreement and efficiency

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
CUDA_VISIBLE_DEVICES=1,2,3 python -u experiments/04_backend_fidelity/scripts/run_e4_paired.py \
  --ollama-model qwen3.8:latest \
  --hf-model "$HOME/hf_models/Qwen3.8-27B" \
  --device cuda --device-map balanced --hf-dtype bfloat16 \
  --full --repetitions 3 --timeout 180
```

## Important limitations

E4 measures **cross-stack agreement**, not exact same-weight logit fidelity. Ollama GGUF quantization differs from HF native checkpoints (0.8B Q8_0; 4B, 9B and 27B Q4_K_M), and the runtimes may differ in weight precision, chat serialization, token IDs, and kernels. The full-run prompt-token *count* agreement is not sufficient to prove identical tokenization. HF acceleration kernels may differ; its 27B model is sharded across three GPUs while Ollama's GPU placement is controlled by its daemon. Report latency as an observational property of these configurations, not an intrinsic speedup. Three repetitions improve stability estimates but are not 693 independent task samples per model.
