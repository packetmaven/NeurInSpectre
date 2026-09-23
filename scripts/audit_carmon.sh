#!/usr/bin/env bash
# Audit Carmon2019 and JPEG-Carmon as security pipelines.
#
# Smoke (whitebox + short scores/labels Square):
#   bash scripts/audit_carmon.sh
#
# Practical scores-only smoke:
#   bash scripts/audit_carmon.sh --mode scores --n-examples 4
#
# Camera-style n=1000 whitebox (slow; GPU recommended):
#   bash scripts/audit_carmon.sh --full
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CKPT="${CARMON_CKPT:-models/cifar10/Linf/Carmon2019Unlabeled.pt}"
if [[ ! -f "$CKPT" ]]; then
  echo "[audit] missing $CKPT — run: python scripts/download_carmon2019.py" >&2
  exit 1
fi

FULL=0
PASSTHRU=()
for arg in "$@"; do
  if [[ "$arg" == "--full" || "$arg" == "full" ]]; then
    FULL=1
  else
    PASSTHRU+=("$arg")
  fi
done

if [[ "$FULL" -eq 1 ]]; then
  EXTRA=(--n-examples 1000 --mode whitebox)
  OUT_SUFFIX="full"
else
  EXTRA=(--smoke --n-examples 8)
  OUT_SUFFIX="smoke"
fi
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then
  EXTRA+=("${PASSTHRU[@]}")
fi

PYTHON="${PYTHON:-python}"
if [[ -x "$ROOT/venv/bin/python" ]]; then
  PYTHON="$ROOT/venv/bin/python"
fi

"$PYTHON" -m neurinspectre.cli audit --target carmon \
  --model-path "$CKPT" \
  --output-dir "results/audit/carmon_${OUT_SUFFIX}" \
  "${EXTRA[@]}"

"$PYTHON" -m neurinspectre.cli audit --target jpeg-carmon \
  --model-path "$CKPT" \
  --output-dir "results/audit/jpeg_carmon_${OUT_SUFFIX}" \
  "${EXTRA[@]}"

echo "[audit] reports:"
echo "  results/audit/carmon_${OUT_SUFFIX}/audit_report.json"
echo "  results/audit/jpeg_carmon_${OUT_SUFFIX}/audit_report.json"
