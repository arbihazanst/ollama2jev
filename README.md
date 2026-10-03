  <p align="center">
    <img src="assets/Ollama2Jev.png" alt="Ollama2Jev" width="100%">
  </p>

<div align="center">

# Ollama2Jev: Extending LLM2Jev with an Ollama Backend

<br/>

[![Python](https://img.shields.io/badge/python-3.12%2B-blue?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square)](LICENSE)
[![Backend](https://img.shields.io/badge/Backend-Ollama-green?style=flat-square)](src/llm2jev/backend/ollama)
[![Benchmark](https://img.shields.io/badge/JevBench-231%20public%20tasks-orange?style=flat-square)](ollama2jev_jevbench_model_summary.json)

**Ollama2Jev extends the open-source LLM2Jev software stack with an Ollama-compatible binary-scoring backend for Jev-style structured decisions.**

</div>

> **Project relationship.**
> Ollama2Jev is an extension of the open-source
> [Yinsongxu/LLM2Jev](https://github.com/Yinsongxu/LLM2Jev) repository.
> We retain LLM2Jev's typed `Choice`, `Score`, and `Noul` pipeline, prompt
> rendering, normalization, and response assembly, while adding an
> `OllamaBackend` that works with models already served by Ollama.
>
> Ollama2Jev is not an official component of LLM2Jev, Jev, TypeSafe, or Ollama.

## What Ollama2Jev adds

The upstream LLM2Jev repository provides Transformers, SGLang, and MLX
backends that can access model logits directly or through backend-specific
scoring interfaces.

Ollama2Jev adds a practical alternative for researchers who already use
Ollama:

- **Ollama backend:** implements the LLM2Jev binary backend interface using
  the Ollama chat API.
- **One-token inference:** requests exactly one output token with
  deterministic decoding.
- **First-token log-probabilities:** reads Ollama's returned top-$k$
  distribution.
- **Exact `yes`/`no` extraction:** retrieves the lowercase `yes` and `no`
  log-probabilities used by LLM2Jev's binary candidate formulation.
- **Stable binary scoring:** converts the pair of log-probabilities into
  \(P(\mathrm{yes})\).
- **Coverage diagnostics:** records whether both verbalizers are present in
  the returned top-$k$ set.
- **JevBench evaluation:** includes full results on all 231 public JevBench
  tasks across multiple Ollama-served models.

## Relationship to LLM2Jev

Ollama2Jev does **not** replace the LLM2Jev pipeline. It extends it at the
backend layer.

```text
Existing LLM2Jev
JevRequest
   |
   v
binary question compiler
   |
   v
prompt renderer
   |
   v
------------------------------
Ollama2Jev extension
OllamaBackend
   |
   +--> Ollama server
   |       |
   |       v
   |   one-token decode
   |   top-k logprobs
   |       |
   +<------+
   |
   v
yes/no extraction
   |
   v
P(yes) binary score
------------------------------
   |
   v
Existing LLM2Jev
normalization + typed response assembly
   |
   v
Choice / Score / Noul
