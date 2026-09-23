#!/usr/bin/env bash
# Native Linux (host arch) EMBER same-sample run.
# Re-scores the PE corpus on Linux, then attacks only Linux-detected malware.
#
#   bash scripts/audit_ember_linux_native.sh /path/to/pe_dir
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PE="${1:?usage: bash scripts/audit_ember_linux_native.sh /path/to/pe_or_dir}"
shift || true
IMAGE="${EMBER_AUDIT_IMAGE:-neurinspectre:ember-linux}"
OUT="${EMBER_AUDIT_OUT:-$ROOT/results/audit/ember_gbdt_linux_same_sample}"
N_QUERIES="${EMBER_N_QUERIES:-500}"
BUDGETS="${EMBER_QUERY_BUDGETS:-100,500}"
EXTRA=()
if [[ "${EMBER_PROBLEM_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--problem-only)
fi
if [[ -n "${EMBER_REUSE_INVENTORY:-}" ]]; then
  EXTRA+=(--reuse-inventory "$EMBER_REUSE_INVENTORY")
fi
if [[ -n "${EMBER_PRIOR_CHECKPOINT:-}" ]]; then
  EXTRA+=(--prior-checkpoint "$EMBER_PRIOR_CHECKPOINT")
fi
if [[ -n "${EMBER_CHECKPOINT_NAME:-}" ]]; then
  EXTRA+=(--checkpoint-name "$EMBER_CHECKPOINT_NAME")
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[audit] docker not found" >&2
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[audit] building $IMAGE (native $(uname -m))" >&2
  docker build -f "$ROOT/scripts/Dockerfile.ember-linux" -t "$IMAGE" "$ROOT/scripts"
fi

mkdir -p "$OUT"
echo "[audit] image=$IMAGE pe=$PE out=$OUT n_queries=$N_QUERIES budgets=$BUDGETS" >&2
exec docker run --rm \
  -v "$ROOT:/app" \
  -v "$PE:$PE:ro" \
  -w /app \
  "$IMAGE" \
  /app/scripts/run_ember_linux_same_sample.py \
    --pe-dir "$PE" \
    --model-path /app/data/ember/ember2018/ember_model_2018.txt \
    --output-dir "$OUT" \
    --n-queries "$N_QUERIES" \
    --query-budgets "$BUDGETS" \
    "${EXTRA[@]}" \
    "$@"
