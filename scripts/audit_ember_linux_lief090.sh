#!/usr/bin/env bash
# EMBER 2018 same-sample on Linux amd64 + Python 3.6 + LIEF 0.9.0.
# official_reproduction is true only inside this image.
#
#   bash scripts/audit_ember_linux_lief090.sh /path/to/pe_dir
#
# Smoke:
#   EMBER_MAX_DETECTED=1 EMBER_N_QUERIES=10 EMBER_AUDIT_OUT=.../smoke \
#     bash scripts/audit_ember_linux_lief090.sh /path/to/pe_dir
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PE="${1:?usage: bash scripts/audit_ember_linux_lief090.sh /path/to/pe_dir}"
shift || true
IMAGE="${EMBER_AUDIT_IMAGE:-neurinspectre:ember-lief090}"
OUT="${EMBER_AUDIT_OUT:-$ROOT/results/audit/ember_gbdt_linux_lief090}"
N_QUERIES="${EMBER_N_QUERIES:-500}"
MAX_DETECTED="${EMBER_MAX_DETECTED:-0}"
MAX_FILES="${EMBER_MAX_FILES:-0}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[lief090] docker not found" >&2
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[lief090] building $IMAGE (linux/amd64, python 3.6, lief 0.9.0)" >&2
  docker build --platform linux/amd64 \
    -f "$ROOT/scripts/Dockerfile.ember-lief090" \
    -t "$IMAGE" \
    "$ROOT/scripts"
fi

mkdir -p "$OUT"
case "$OUT" in
  "$ROOT"/*) CONTAINER_OUT="/work/${OUT#"$ROOT"/}" ;;
  *)
    echo "[lief090] output dir must live under $ROOT so the /work bind mount keeps the report" >&2
    exit 1
    ;;
esac
echo "[lief090] image=$IMAGE pe=$PE out=$OUT n_queries=$N_QUERIES max_detected=$MAX_DETECTED" >&2
exec docker run --rm --platform linux/amd64 \
  -v "$ROOT:/work" \
  -v "$PE:$PE:ro" \
  -w /work \
  "$IMAGE" \
  /work/scripts/run_ember2018_lief090.py \
    --pe-dir "$PE" \
    --model-path /work/data/ember/ember2018/ember_model_2018.txt \
    --output-dir "$CONTAINER_OUT" \
    --n-queries "$N_QUERIES" \
    --max-detected "$MAX_DETECTED" \
    --max-files "$MAX_FILES" \
    "$@"
