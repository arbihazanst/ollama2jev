# Ollama2Jev — installed-checkpoint and loaded-memory audit

This is an **inventory of models already installed on the existing server**. It does not download, duplicate, fine-tune, register, or remove any model. The default run is read-only. A separate **opt-in** dual-use probe verifies that **the same installed model tag** can respond to normal generation and restricted-top-logprob binary scoring calls, without installing a second checkpoint. (That probe is a serving-interface smoke test, not a full JevBench evaluation.)

## Requirements

- Python 3.10+ standard library; access to an Ollama service (`/api/tags`, `/api/ps`, `/api/show`).
- Optional: `nvidia-smi` in PATH to record device-level memory/usage and compute processes (e.g., RTX A6000 GPUs).
- Optional: read access to the Ollama server's **actual** models directory for on-disk blob accounting. This is usually on the **server machine or inside its container**, and may differ from the calling user's home directory.

## 1. Default: audit installed models without inference

```bash
bash run_deployment_audit.sh
```

Or specify a server and destination:

```bash
python3 audit_ollama_deployment.py \
  --host http://127.0.0.1:11434 \
  --output-dir outputs/deployment_audit
```

The defaults include the **five general-purpose Qwen variants** and **six Ollama-served specialized System-One variants** used as E3 comparators or references:

`qwen3.5:0.8b`, `qwen3.5:2b`, `qwen3.5:4b`, `qwen3.5:9b`, `qwen3.8:latest`, `tev1:0.8b`, `tev1:4b`, `nimble-v3:9b`, `clef-flash:9b`, `clef:27b`, `pplx-decider:latest`.

`--all-models` audits every installed tag; `--models <tag ...>` restricts the inventory. Missing tags are explicitly flagged. `--host` uses port 11434 by default; use `--host http://127.0.0.1:11435` if that is the active research Ollama instance.

## 2. Optional: record physically shared model blobs

If the Ollama *server* stores its models in `/home/arbi/.ollama/models`, run locally on that machine:

```bash
python3 audit_ollama_deployment.py \
  --models-dir /home/arbi/.ollama/models \
  --output-dir outputs/storage_audit
```

If `OLLAMA_MODELS` is set in the running shell, the script also uses it automatically. Supply `--models-dir` explicitly when the server service uses another account, path, Docker volume, or mounted drive. Requires existing readable `manifests/` and `blobs/`. No files in the model store are modified. If manifests are unavailable, the report still includes API-reported installed sizes but **does not claim actual physical disk savings**.

The paired comparison accounts for shared content-addressed blobs *between each general-purpose model and E3 comparator*. It **does not** account for blobs shared with other installed tags or filesystem-level compression/deduplication. Do not report the simple sum of model sizes as exact reclaimable disk capacity.

## 3. Optional: demonstrate same-checkpoint dual use (may load models)

Only run this **when the server is not executing other experiments**:

```bash
python3 audit_ollama_deployment.py \
  --output-dir outputs/dual_use_qwen9b \
  --probe-dual-use qwen3.5:9b
```

The script makes two requests to the **same tag**, without pulling or creating models: a normal `/api/chat` generation and a `/api/chat` binary first-token scoring request with `logprobs=true`, `top_logprobs=20`, and `num_predict=1`. It records whether the first-position top-20 candidates cover **both exact lowercase** `yes` and `no` and whether both requests report the same model tag. The audit additionally checks that the Ollama model digest **and installed tag set remain unchanged** before and after the two requests. This is an interface-level reuse observation, not a full LLM2Jev decision-quality evaluation. It does **not** run the full LLM2Jev assembler. You may repeat for another supported model, such as `qwen3.8:latest`.

You can also make an individual opt-in probe:

```bash
python3 audit_ollama_deployment.py --output-dir outputs/probe_one \
  --probe-chat qwen3.5:9b

python3 audit_ollama_deployment.py --output-dir outputs/probe_native \
  --probe-native nimble-v3:9b
```

Any probe may load or evict models and alter GPU residency; the script does not automatically unload them. It does **not** fine-tune or pull models. To compare loaded-memory footprints fairly, run at low server utilization and record separate baseline/after snapshots; avoid attributing whole-device `nvidia-smi` usage exclusively to a particular model.

## 4. Outputs and correct interpretation

| File | What it records |
|---|---|
| `baseline_models.csv` | Exact installed model tag, digest, model parameters, quantization, API-reported checkpoint size, loaded state, Ollama-reported runtime size and VRAM allocation, context length, optional verified manifest/blob size |
| `baseline_details.json` | Machine-readable snapshot including `/api/ps` raw entries and GPU telemetry |
| `baseline_gpus.csv` | Device-wide NVIDIA memory and utilization, if available |
| `baseline_gpu_processes.csv` | CUDA compute process usage, if available; may contain process IDs and names |
| `pairwise_checkpoint_comparison.csv` | General-purpose Qwen versus E3 comparator; hypothetical additional specialized checkpoint size, plus optional unique-blob accounting |
| `deployment_audit.json` | Full archive of inventory, data provenance, limitations, and opt-in probe events |
| `probe_events.json` | Opt-in probe results (empty for a read-only inventory) |
| `after_*.csv/json` | Snapshot of loaded models and GPUs after each explicit probe |

**Read these cautions before writing the paper:**

1. `/api/tags.size` is **the size reported by Ollama for an installed model**. It does not prove the model occupies that much *additional physical disk space* relative to another model, especially when blobs are shared.
2. `/api/ps.size` and `/api/ps.size_vram` report Ollama's runtime memory and VRAM estimate **for models currently loaded**. An installed-but-unloaded model correctly has no runtime VRAM figure in the CSV, not 0 bytes measured while loaded.
3. `size - size_vram` is only a derived indicator, **not measured host RAM RSS** or system-memory consumption attributable solely to that model.
4. `nvidia-smi memory.used` is **whole-GPU allocation**. Its changes can reflect context windows, KV caches, other models, processes and unrelated workloads.
5. The E3 native specialized checkpoint is **not** a fine-tuned copy of exactly the same installed general-purpose model merely because both have Qwen-related architectures or similar parameter counts. Comparators are matched by experimental design, not necessarily checkpoint lineage or exact model-weight provenance.
6. *No new model required* is an architectural deployment advantage. Measured savings in *training time, total disk space, and GPU residency* require a carefully specified baseline. Avoid presenting the hypothetical avoided checkpoint size as an observed hardware saving.

For IEEE Access, a strong claim supported by the audit is: **"The same installed Ollama model tag was used to perform ordinary generation and to expose first-position token scores for the Ollama2Jev pipeline, without creating or loading a second decision-specific checkpoint."** (Only make this claim for models whose optional dual-use probe succeeded.)

## Research reproducibility

Record: experiment date/time in UTC, Ollama version (`curl -s http://127.0.0.1:11434/api/version`), model tag and digest, installed reported size, quantization, loaded runtime allocation, hardware memory utilization, and a clear note about active concurrent processes. Keep raw audit JSON locally; redact machine paths or process information if publishing raw results.

API reference: [models](https://docs.ollama.com/api/tags), [loaded models](https://docs.ollama.com/api/ps), [chat](https://docs.ollama.com/api/chat), [native decisions](https://docs.ollama.com/capabilities/decision).
