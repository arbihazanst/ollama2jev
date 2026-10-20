#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
OUT="${1:-outputs/deployment_audit_$(date -u +%Y%m%dT%H%M%SZ)}"

# READ-ONLY inventory. Does not pull, serve, reload or unload models.
python3 audit_ollama_deployment.py --host "$HOST" --output-dir "$OUT"

echo "Saved to: $OUT"
echo "To verify dual-use of the same installed model (OPT-IN; can load/evict models):"
echo "python3 audit_ollama_deployment.py --host $HOST --output-dir outputs/dual_use --probe-dual-use qwen3.5:9b"
