# E1 — General-purpose Ollama2Jev benchmark

**Status: COMPLETE.** This is the original 11-model experiment: 231 public JevBench items (Choice=139, Noul=74, Score=18) per model, zero-shot inference, one generated token per binary candidate, exact `yes` and `no` verbalizers, top-20 first-token log-probabilities. The model used for the large Qwen configuration was `qwen3.8:latest`, not `qwen3.8:27b`.

## Results (do not relocate)

The repository root contains:

- `ollama2jev_jevbench_11models_summary.json`
- Eleven `ollama2jev_jevbench_<model>_full.jsonl` files
- `examples/run_ollama2jev_jevbench.py`
- `examples/run_ollama2jev_jevbench_additional.py`

The original E1 experiment contains 2,541 model–task attempts and 2,538 successful typed outputs; Granite 4.2 8B accounts for three top-20 coverage failures. These remain the source-of-truth for E2 offline re-analysis.

## Reproducibility

Source snapshot when this experiment folder was prepared: `b6f3070dbc4ceaba85d9c2c8003547300a300d52` on branch `ollama2jev`. Save installed model digests (especially aliases ending in `:latest`) and the JevBench dataset revision in subsequent release metadata.
