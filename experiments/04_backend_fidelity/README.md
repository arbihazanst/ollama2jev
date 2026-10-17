# E4 — Exact-logit fidelity and robustness

**Status: PLANNED / NOT EXECUTED.** Recommended in response to the editor's concerns about limited technical novelty and insufficient comparisons.

## Primary controlled comparison

Compare the original open-source LLM2Jev backend capable of exact/token-logit scoring (e.g. Transformers or SGLang) against the Ollama2Jev first-token top-k backend. Match **the same model checkpoint/digest or equivalent exported weights and quantization**, tokenizer, special tokens, rendered prompts, scoring candidates, and decoding settings. Use the same 231 JevBench decisions and independent scenario groups where possible.

Measure (i) candidate-level binary probability difference (absolute and distributional), (ii) typed-decision agreement/accuracy, (iii) failure and token coverage, (iv) end-to-end latency, (v) total binary calls, and (vi) peak RAM/VRAM. On identical logit distributions, conditional yes/no scoring is exact when both labels occur in top-k; differences may arise from serving implementation, tokenization, prompt rendering, logits processors, or missing verbalizers rather than top-k normalization itself.

## Secondary robustness analyses

- Context length: fixed controlled lengths, report truncation and prompt-equivalence handling.
- Candidate count: 2/4/8 available choices, matching the same task semantics where possible.
- Verbalizer/prompt sensitivity: document exact forms and tokenization; do not conflate prompt perturbation with the offline Z4 replay.
- Repeated warm latency: multiple trials, model warm-state policy, GPU utilization and percentile definition.

Do not populate result tables before executing these controlled experiments.
