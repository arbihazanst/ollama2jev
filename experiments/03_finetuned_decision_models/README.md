# E3 — Fine-tuned System-One decision models

**Status: protocol and scripts ready; no E3 benchmark outcomes reported until the server run completes.**

## Goal

Compare native decision-model inference using Ollama's `/v1/systemone` endpoint with the original general-purpose Qwen checkpoints scored using Ollama2Jev in E1.

| Native decision model | E1 comparison model |
| --- | --- |
| `tev1:0.8b` | `qwen3.5:0.8b` |
| `tev1:4b` | `qwen3.5:4b` |
| `nimble-v3:9b` | `qwen3.5:9b` |
| `clef-flash:9b` | `qwen3.5:9b` |
| `clef:27b` | `qwen3.8:latest` |
| `pplx-decider:27b` | `qwen3.8:latest` |

## Reproducibility details

- Public JevBench `original.jsonl` → `easy.jsonl` → `hard.jsonl` (exactly 231 tasks: 139 Choice, 74 Noul, 18 Score).
- One named `decision` question per HTTP request, preserving original `state`, `type`, `instructions`, `criteria`, and candidate order.
- Use native model probabilities without forcing temperature, thinking mode, one-token decoding, or logprob arguments: these are not supported for `/v1/systemone`.
- Choice: native chosen option (validated against probability maximum). Noul: probability of true >= 0.5. Score: **argmax rubric level**, not rounded weighted-mean score, matching E1.
- Every output includes per-class probabilities; Brier (multiclass squared sum), NLL (log probability of gold), and ECE-10 computed on valid outputs only. Full-denominator accuracy includes invalid responses and HTTP errors as incorrect.
- Original results belong to E1 and are not duplicated here. E3 includes six per-model JSONL files, a native scoring summary, and six predeclared paired E1 comparisons (Holm correction).
- Appends one JSONL row after each task; on restart, the latest saved row per task ID is authoritative. No requests are repeated unless `--retry-errors` or `--retry-invalid` is specified.
- Environment metadata is captured automatically, including Ollama version and installed model digest.

## Server prerequisites

`nimble-v3:9b` requires **Ollama >= 0.40.4**. Local `/v1/systemone` is supported in Ollama 0.35+, but individual model version requirements vary. Check `ollama --version` and `ollama list` before running.

Pull tags *exactly* (the PPLX tag must be verified locally; if unavailable use `pplx-decider:latest` deliberately and update `MODELS` and the comparison mapping to reflect the actual tag):

```bash
ollama pull tev1:0.8b
ollama pull tev1:4b
ollama pull nimble-v3:9b
ollama pull clef-flash:9b
ollama pull clef:27b
ollama pull pplx-decider:27b
ollama list
```

This project uses only the native HTTP API through Python's standard library; no SDK changes to the existing Ollama2Jev backend are necessary.

## Smoke test

From the repository root:

```bash
python -u experiments/03_finetuned_decision_models/scripts/run_e3_native.py \
  --models tev1:0.8b --smoke --stop-on-error --timeout 120
```

The 15-task smoke run is stored under `results/smoke/`, separate from final results.

## Full run in tmux

```bash
# Run in a new tmux session to protect against SSH disconnects.
tmux new -s e3
cd ~/LLM2Jev
python -u experiments/03_finetuned_decision_models/scripts/run_e3_native.py \
  --timeout 240 --retries 1 2>&1 | tee experiments/03_finetuned_decision_models/e3_run.log
```

Detach: `Ctrl+B`, then `D`. Reattach: `tmux attach -t e3`.

If it stops, rerun the same command; completed rows will be skipped. Use `--retry-errors` only after addressing HTTP failures.

Run just one model: `--models clef:27b`.

## Paired comparison, only after ALL six models finish

```bash
python experiments/03_finetuned_decision_models/scripts/compare_e3_to_e1.py
```

This checks that all six paired contrasts each contain the exact same 231 task IDs. It outputs `results/comparisons/e3_vs_e1_paired_comparison.csv` and `.json`.

## Limitations

These are *deployed-system* comparisons, **not controlled estimates of the effect of fine-tuning alone**. Parent and native decision models differ in training, format/precision, possibly runtime, and scoring mechanism. The `latest` alias used in the E1 Qwen3.8 baseline should be pinned with its digest.

Prompt formatting is delegated to Ollama's native decision runtime, whereas E1 uses an LLM2Jev binary prompt compiler. This is an intentional systems comparison; it does not isolate prompt effects. JevBench is publicly used by some model publishers, so benchmark exposure must be disclosed. Always cite the publishers and respect their model licenses (Nimble v3 is CC BY-NC 4.0).

Official interface reference: https://docs.ollama.com/api/systemone
