#!/bin/bash
set -euo pipefail

# =============================================================================
# Detection experiments (synthetic sequences + real-defense characterization).
# The 12-defense evasion harness is scripts/reproduce_all.sh
# (or scripts/reproduce_table8.sh for Table 8 only).
# Output directory: results/detection/ (default; set RESULTS_DIR to override)
# =============================================================================

echo "=== NeurInSpectre Reproduction Script ==="
echo "=== Detection experiment harness ==="
echo "=== Table 8 harness: scripts/reproduce_all.sh ==="
echo ""

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="${RESULTS_DIR:-results/detection}"
mkdir -p "$RESULTS_DIR"

# Prefer a repo-local venv if present; otherwise rely on PATH.
PY_BIN="${PY_BIN:-}"
if [[ -z "${PY_BIN}" ]]; then
  if [[ -x ".venv-neurinspectre/bin/python" ]]; then
    PY_BIN=".venv-neurinspectre/bin/python"
  else
    PY_BIN="python"
  fi
fi

# Harness controls (override via env)
SKIP_SYNTHETIC="${SKIP_SYNTHETIC:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_REAL="${SKIP_REAL:-0}"
SKIP_FIGURE4="${SKIP_FIGURE4:-0}"

is_true() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

FAILS=0

# -----------------------------------------------------------------------------
# Step 1: Tables 1-4 (synthetic calibration) — ~5 seconds
# -----------------------------------------------------------------------------
echo "--- Step 1: Tables 1-4 (synthetic signatures, ~5 seconds) ---"
if is_true "$SKIP_SYNTHETIC"; then
  echo "  Skipped (SKIP_SYNTHETIC=$SKIP_SYNTHETIC)"
else
  if ! "$PY_BIN" scripts/run_synthetic_experiments.py \
      --output-dir "$RESULTS_DIR" \
      > "$RESULTS_DIR/synthetic.log" 2>&1; then
    echo "  FAILED: see $RESULTS_DIR/synthetic.log"
    FAILS=$((FAILS + 1))
  else
    echo "  OK: $RESULTS_DIR/synthetic_experiments.json"
  fi
fi

# -----------------------------------------------------------------------------
# Step 2a: Carmon2019 checkpoint download (~146 MB) — ~1 minute
# -----------------------------------------------------------------------------
echo ""
echo "--- Step 2a: Carmon2019Unlabeled.pt download (~1 min) ---"
if is_true "$SKIP_DOWNLOAD"; then
  echo "  Skipped (SKIP_DOWNLOAD=$SKIP_DOWNLOAD)"
elif [[ -f "models/cifar10/Linf/Carmon2019Unlabeled.pt" ]]; then
  echo "  Already present at models/cifar10/Linf/Carmon2019Unlabeled.pt"
  # Verify SHA256 unconditionally.
  if ! "$PY_BIN" scripts/download_carmon2019.py \
      > "$RESULTS_DIR/download.log" 2>&1; then
    echo "  FAILED: SHA256 verification failed (see $RESULTS_DIR/download.log)"
    FAILS=$((FAILS + 1))
  fi
else
  if ! "$PY_BIN" scripts/download_carmon2019.py \
      > "$RESULTS_DIR/download.log" 2>&1; then
    echo "  FAILED: see $RESULTS_DIR/download.log"
    FAILS=$((FAILS + 1))
  else
    echo "  OK: models/cifar10/Linf/Carmon2019Unlabeled.pt (SHA256 verified)"
  fi
fi

# -----------------------------------------------------------------------------
# Step 2b: Table 5 (real CIFAR-10 defenses) — ~3 hours on MPS / ~30 min on A100
# -----------------------------------------------------------------------------
echo ""
echo "--- Step 2b: Table 5 (real CIFAR-10 defenses; ~3 h MPS / ~30 min A100) ---"
if is_true "$SKIP_REAL"; then
  echo "  Skipped (SKIP_REAL=$SKIP_REAL)"
else
  if ! "$PY_BIN" -u scripts/run_real_defense_experiments.py \
      --output-dir "$RESULTS_DIR" \
      > "$RESULTS_DIR/real_defense.log" 2>&1; then
    echo "  FAILED: see $RESULTS_DIR/real_defense.log"
    FAILS=$((FAILS + 1))
  else
    echo "  OK: $RESULTS_DIR/real_defense_characterization.json"
  fi
fi

# -----------------------------------------------------------------------------
# Step 3: Figure 4 (dissipative Laplacian 3-panel) — <1 minute
# -----------------------------------------------------------------------------
echo ""
echo "--- Step 3: Figure 4 (Laplacian 3-panel; <1 min) ---"
if is_true "$SKIP_FIGURE4"; then
  echo "  Skipped (SKIP_FIGURE4=$SKIP_FIGURE4)"
else
  if ! "$PY_BIN" scripts/LaPlacian.py \
      > "$RESULTS_DIR/figure4.log" 2>&1; then
    echo "  FAILED: see $RESULTS_DIR/figure4.log"
    FAILS=$((FAILS + 1))
  else
    echo "  OK: figures/laplacian_dissipative_3panel.png"
  fi
fi

# -----------------------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------------------
echo ""
echo "=== All results saved to: $RESULTS_DIR ==="
echo "=== SHA256 manifest ==="
"$PY_BIN" - "$RESULTS_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = root / "sha256_manifest.txt"
paths = []
for p in sorted(root.rglob("*")):
    if not p.is_file() or p.name == manifest.name:
        continue
    if p.suffix.lower() not in {".json", ".log", ".md", ".txt", ".png", ".yaml"}:
        continue
    paths.append(p)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

lines = [f"{sha256_file(p)}  {p.relative_to(root)}" for p in paths]
manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
print(manifest)
for line in lines:
    print(line)
PY

if [[ "$FAILS" -ne 0 ]]; then
  echo ""
  echo "ERROR: $FAILS step(s) failed."
  exit 1
fi

echo ""
echo "OK"
