#!/usr/bin/env bash
# Run exactly one identical JevBench task for each of four Qwen models.
# Run from ~/LLM2Jev after applying enable_e4_one_task_multigpu.py.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

RUNNER=experiments/04_backend_fidelity/scripts/run_e4_paired.py
LOG_DIR=experiments/04_backend_fidelity/smoke_logs
mkdir -p "$LOG_DIR"

# GPU 0 is preferred for the separate Ollama daemon.
# CUDA_VISIBLE_DEVICES only restricts this Transformers Python process, not Ollama's existing service.
# The 27B HF model is sharded across GPUs 1,2,3 with Accelerate.
run_one() {
  local ollama_tag="$1" hf_path="$2" gpu_ids="$3" map="$4" dtype="$5" label="$6"
  test -f "$hf_path/config.json" || { echo "Missing HF checkpoint: $hf_path (download it first)" >&2; return 2; }
  echo "==================== $ollama_tag / HF $hf_path / CUDA_VISIBLE_DEVICES=$gpu_ids ===================="
  CUDA_VISIBLE_DEVICES="$gpu_ids" python -u "$RUNNER" \
    --ollama-model "$ollama_tag" \
    --hf-model "$hf_path" \
    --device cuda \
    --device-map "$map" \
    --hf-dtype "$dtype" \
    --one-task \
    --repetitions 1 \
    --timeout 180 \
    2>&1 | tee "$LOG_DIR/$label.log"
}

run_one 'qwen3.5:0.8b' "$HOME/hf_models/Qwen3.5-0.8B" '1' 'auto' 'auto' 'qwen35_08b'
run_one 'qwen3.5:4b'   "$HOME/hf_models/Qwen3.5-4B"   '1' 'auto' 'auto' 'qwen35_4b'
run_one 'qwen3.5:9b'   "$HOME/hf_models/Qwen3.5-9B"   '1' 'auto' 'auto' 'qwen35_9b'
run_one 'qwen3.8:27b'  "$HOME/hf_models/Qwen3.8-27B"  '1,2,3' 'balanced' 'bfloat16' 'qwen38_27b'

python experiments/04_backend_fidelity/scripts/summarize_e4.py
printf '\nAll four one-task E4 smoke tests finished. Review logs in %s\n' "$LOG_DIR"
