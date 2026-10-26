# E5 — Deployment feasibility and same-checkpoint reuse

**Status: COMPLETE (20 and 22 October 2026 UTC).** E5 evaluates whether an *already installed* general-purpose Ollama checkpoint can serve ordinary text generation and first-position `yes`/`no` binary probability extraction without training or installing an additional decision-specialized checkpoint. The archived audit also inventories 11 model tags and compares six pairs of content-addressed checkpoint manifests.

## Verified same-checkpoint evidence

| Ollama model tag | Ordinary generation | Exact lowercase `yes` / `no` top-20 coverage | Stable tag, SHA-256 digest and installed-tag set | E5 outcome |
|---|---|---|---|---|
| `qwen3.5:0.8b` | Pass | Pass | Yes | **PASS** (22 Oct) |
| `qwen3.5:2b` | Pass | Pass | Yes | **PASS** (22 Oct) |
| `qwen3.5:4b` | Pass | Pass | Yes | **PASS** (22 Oct) |
| `qwen3.5:9b` | Pass | Pass | Yes | **PASS** (20/22 Oct) |
| `qwen3.8:latest` (27B) | Pass | Pass | Yes | **PASS** (20 Oct) |

**Evidence:** [20 October corrected probe events](outputs/dual_use_corrected/probe_events.json) and [22 October consolidated probe events](outputs/dual_use_all_qwen_20261023/probe_events.json). The 22 October run contains four successful Qwen3.5 probes and a 27B timeout, whereas the 27B model had previously passed on 20 October. All five have successful evidence **across runs**, not one successful uninterrupted five-model batch. The archived [small-model events](outputs/dual_use_small_models/probe_events.json) independently confirm 0.8B and 4B.

Success requires nonempty ordinary generation, one-token top-20 probability readout containing both exact lowercase `yes` and `no`, the same served tag, unchanged installed-model digest, and unchanged tag inventory. **These checks exercise the two serving interfaces only**; the probe does not run the full LLM2Jev Choice/Noul/Score assembler or the full 231-task JevBench.

## Manifest-based storage comparisons

The installed-model inventory includes five general-purpose Qwen configurations and six specialized native System-One comparators. Pairwise comparison of manifest-referenced blobs found additional specialized content absent from each matched general-purpose manifest:

| Specialized checkpoint | Additional pairwise blob content (GiB) |
|---|---:|
| `tev1:0.8b` | 0.776 |
| `tev1:4b` | 4.294 |
| `nimble-v3:9b` | 8.873 |
| `clef-flash:9b` | 9.852 |
| `clef:27b` | 18.779 |
| `pplx-decider:latest` | 16.275 |

These measurements do **not** establish globally reclaimable filesystem bytes, disk savings, RAM/VRAM savings, energy savings, training-time savings, or causal inference-speed advantages. The server may have shared blobs or concurrent GPU workloads. E5 used Ollama 0.40.5; E1–E4 used Ollama 0.35.0.

## Run the audit

The standard-library script reads current Ollama tags, metadata and process residency and optionally examines local manifest/blob files. It does not download, delete, register, duplicate or train models. The inventory run is read-only; explicit dual-use probes issue inference requests and may alter which checkpoints are resident in memory.

```bash
# Inventory only, from the repository root:
python3 experiments/05_deployment_audit/audit_ollama_deployment.py \
  --host http://127.0.0.1:11434 \
  --models-dir /usr/share/ollama/.ollama/models \
  --output-dir experiments/05_deployment_audit/outputs/local_inventory

# Optional verification of an already installed model:
python3 experiments/05_deployment_audit/audit_ollama_deployment.py \
  --host http://127.0.0.1:11434 \
  --models-dir /usr/share/ollama/.ollama/models \
  --probe-dual-use qwen3.5:2b \
  --output-dir experiments/05_deployment_audit/outputs/local_probe
```

The models directory must point to the server's actual Ollama model store and be readable. Probes may load or evict checkpoints, so do not run them while other researchers depend on stable GPU allocations. No E5 rerun is required merely to document the archived successful outcomes.

## Archived outputs

- `probe_events.json`: one event per explicit model verification, including pass/fail and coverage detail.
- `deployment_audit.json`: complete model inventory, tag/digest provenance, comparisons and probe records.
- `baseline_models.csv`, `baseline_details.json`: installed-model and initial residency snapshots.
- `pairwise_checkpoint_comparison.csv`: specialized/general-purpose manifest blob comparison; **pairwise only**.
- `after_*.csv` and `after_*.json`: post-probe model and GPU snapshots, where collected.

For the full study and inference comparison boundaries, see [the experiment index](../README.md).
