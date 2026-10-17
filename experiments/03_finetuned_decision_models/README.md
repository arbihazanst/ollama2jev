# E3 — Specialized decision-model comparison

**Status: PLANNED / NOT EXECUTED. No accuracy, latency, or calibration results are available for these models in this experiment.**

## Candidate pairs (verify local Ollama tags before running)

| Fine-tuned decision model | Proposed native Ollama tag | Related general-purpose comparator | Pairing caveat |
|---|---|---|---|
| Tev1 0.8B | `tev1:0.8b` | `qwen3.5:0.8b` | Different training/output mechanism |
| Strands Decider | `strands-decider:2b` | `qwen3.5:2b` | Derived from **Qwen3.5-2B-Base**, not necessarily identical comparator checkpoint; Apple Silicon may be required |
| Tev1 4B | `tev1:4b` | `qwen3.5:4b` | Different training/output mechanism |
| Nimble v3 | `nimble-v3:9b` | `qwen3.5:9b` | Different training/output mechanism |
| Clef Flash | `clef-flash:9b` | `qwen3.5:9b` | Different training/output mechanism |
| Clef | `clef:27b` | `qwen3.8:latest` | Pin exact model digest; verify parent |

## Primary design

Use each task's original JevBench state, instructions, and criteria. Query a **native structured-decision endpoint** (check current Ollama documentation and locally installed version before implementation; the proposed endpoint is `/v1/systemone`) rather than the Ollama2Jev one-token chat adapter. Log complete raw requests/responses, native structured label and probability distributions, timings, failures, exact model digest, runtime version, and hardware. Do not force `yes`/`no` token extraction onto a native decision model if it uses a different API.

Use the same 231-item denominator; unsuccessful or unsupported requests count as incorrect, with an explicit failure taxonomy. For successfully completed distributions compute Brier, 10-bin ECE, and NLL using the same metric implementation as E1; report completed counts and missing-data policy. Compare paired correctness by task ID; preregister six primary model pairs and apply exact two-sided McNemar with Holm correction across those six contrasts. Consider scenario-group dependence in interpretations.

## Hardware restrictions

For a model that requires a different host or framework (especially Strands Decider on Apple Silicon), do not directly rank cross-hardware raw latencies. Report host-specific latency separately or benchmark both comparable systems on the same compatible machine when possible.

## Before running

1. Verify availability of each tag and minimum Ollama version; capture `ollama show <tag>` plus digest.
2. Freeze JevBench revision and original model digest; record system/hardware environment.
3. Smoke-test one Choice, one Noul, and one Score item and validate the returned schema.
4. Execute all 231 items per model and save raw per-item JSONL, status/errors, and metrics with the model ID.
5. Compare against existing E1 parent models, noting that this is a **deployed-system comparison**, not an isolated causal estimate of fine-tuning.
