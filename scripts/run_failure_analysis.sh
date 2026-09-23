#!/bin/bash
set -euo pipefail

# Failure-analysis harness.
#
# Goal: generate concrete "this can fail / become meaningless" artifacts:
# - short gradient histories (N < 64) downweight confidence
# - Volterra fit quality can be poor under heavy stochasticity
# - Square (black-box) requires a minimum query budget
# - certified radius computation has numerical/argument edge cases

RESULTS_DIR="${RESULTS_DIR:-results/failure_analysis_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"

DEVICE="${DEVICE:-auto}"

# Prefer a repo-local venv if present; otherwise rely on PATH.
NS_BIN="${NS_BIN:-}"
if [[ -z "${NS_BIN}" ]]; then
  if [[ -x ".venv-neurinspectre/bin/neurinspectre" ]]; then
    NS_BIN=".venv-neurinspectre/bin/neurinspectre"
  elif [[ -x ".venv/bin/neurinspectre" ]]; then
    NS_BIN=".venv/bin/neurinspectre"
  else
    NS_BIN="neurinspectre"
  fi
fi

PY_BIN="${PY_BIN:-}"
if [[ -z "${PY_BIN}" ]]; then
  # If NS_BIN is a venv console script, prefer the sibling python.
  if [[ "${NS_BIN}" == */neurinspectre ]]; then
    CAND_PY="$(dirname "${NS_BIN}")/python"
    if [[ -x "${CAND_PY}" ]]; then
      PY_BIN="${CAND_PY}"
    else
      PY_BIN="python3"
    fi
  else
    PY_BIN="python3"
  fi
fi

run_capture() {
  # Usage:
  #   run_capture <prefix> <cmd...>
  #
  # Writes:
  #   <prefix>.log  (stdout+stderr)
  #   <prefix>.rc   (numeric exit code)
  local prefix="$1"
  shift
  local log="${prefix}.log"
  local rc_file="${prefix}.rc"

  set +e
  "$@" >"$log" 2>&1
  local rc=$?
  set -e

  echo "$rc" >"$rc_file"
  return 0
}

echo "--- Environment ---"
run_capture "$RESULTS_DIR/doctor_json" "$NS_BIN" doctor --as-json
run_capture "$RESULTS_DIR/doctor_txt" "$NS_BIN" doctor

MODEL_CIFAR10="${MODEL_CIFAR10:-models/cifar10_resnet20_norm_ts.pt}"
DATA_CIFAR10="${DATA_CIFAR10:-./data/cifar10}"

if [[ ! -f "${MODEL_CIFAR10}" ]]; then
  echo "WARN: missing CIFAR-10 model at ${MODEL_CIFAR10}; skipping CIFAR-10 based failure cases."
else
  echo ""
  echo "--- Failure Mode 1: short gradient history (N < 64) ---"
  run_capture "$RESULTS_DIR/char_short_grad" "$NS_BIN" characterize \
    --model "${MODEL_CIFAR10}" \
    --dataset cifar10 \
    --data-path "${DATA_CIFAR10}" \
    --defense jpeg \
    --num-samples 32 \
    --device "${DEVICE}" \
    --report-format text \
    --no-progress \
    --output "${RESULTS_DIR}/char_short_grad.json"

  echo ""
  echo "--- Failure Mode 2: Volterra high-RMSE under stochastic defense ---"
  # Baseline: clean model (helps interpret rmse_scaled values).
  run_capture "$RESULTS_DIR/char_clean_volterra" "$NS_BIN" characterize \
    --model "${MODEL_CIFAR10}" \
    --dataset cifar10 \
    --data-path "${DATA_CIFAR10}" \
    --defense none \
    --num-samples 100 \
    --device "${DEVICE}" \
    --report-format text \
    --no-progress \
    --output "${RESULTS_DIR}/char_clean_volterra.json"

  RS_CFG="${RESULTS_DIR}/defense_randsmooth_sigma5.yaml"
  cat >"${RS_CFG}" <<'YAML'
defense:
  # Intentionally large noise to stress Volterra fit quality.
  sigma: 5.0
  # Single-sample smoothing keeps the forward stochastic per call.
  n_samples: 1
YAML
  run_capture "$RESULTS_DIR/char_randsmooth_sigma5" "$NS_BIN" characterize \
    --model "${MODEL_CIFAR10}" \
    --dataset cifar10 \
    --data-path "${DATA_CIFAR10}" \
    --defense randsmooth \
    --defense-config "${RS_CFG}" \
    --num-samples 100 \
    --device "${DEVICE}" \
    --report-format text \
    --no-progress \
    --output "${RESULTS_DIR}/char_randsmooth_sigma5.json"

  echo ""
  echo "--- Failure Mode 3: black-box query budget too low (Square) ---"
  SQUARE_CFG="${RESULTS_DIR}/eval_square_too_few_queries.yaml"
  cat >"${SQUARE_CFG}" <<'YAML'
seed: 42

datasets:
  cifar10:
    path: ./data/cifar10
    split: test
    num_samples: 128
    batch_size: 128
    num_workers: 0

models:
  cifar10: models/cifar10_resnet20_norm_ts.pt

defenses:
  - name: none
    type: none
    dataset: cifar10

attacks:
  - name: square
    # Intentionally below the minimum enforced by SquareAttack.
    n_queries: 250

perturbation:
  epsilon: 0.03137254901960784  # 8/255
  norm: Linf

iterations: 10

validity_gates:
  enabled: false

baseline_validation:
  enabled: false
YAML
  run_capture "$RESULTS_DIR/eval_square_low_queries" "$NS_BIN" evaluate \
    --config "${SQUARE_CFG}" \
    --output-dir "${RESULTS_DIR}/eval_square_low_queries" \
    --device "${DEVICE}" \
    --report-format text \
    --no-report --no-progress
fi

echo ""
echo "--- Failure Mode 4: certified radius edge cases (toy, deterministic) ---"
set +e
"${PY_BIN}" - <<'PY' >"${RESULTS_DIR}/certified_radius_edge_cases.json" 2>"${RESULTS_DIR}/certified_radius_edge_cases.log"
import json
import math

import torch
import torch.nn as nn

from neurinspectre.defenses.wrappers import RandomizedSmoothingDefense


class ConstantLogits(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = int(x.shape[0])
        logits = torch.zeros((b, 2), device=x.device, dtype=torch.float32)
        logits[:, 0] = 10.0
        return logits


out = {
    "notes": [
        "This is a deterministic toy model used only to hit p_a==1.0 boundary behavior.",
        "Real evaluation runs should use real models + real datasets; this output is numerical hygiene evidence.",
    ]
}

defense = RandomizedSmoothingDefense(ConstantLogits(), sigma=0.25, n_samples=1, device="cpu")
x = torch.zeros((1, 3, 8, 8), dtype=torch.float32)

try:
    defense.certified_radius(x, n_samples=0)
    out["n_samples_zero"] = {"raised": False}
except Exception as e:
    out["n_samples_zero"] = {"raised": True, "error": str(e)}

try:
    defense.certified_radius(torch.zeros((2, 3, 8, 8), dtype=torch.float32), n_samples=10)
    out["batch_size_two"] = {"raised": False}
except Exception as e:
    out["batch_size_two"] = {"raised": True, "error": str(e)}

r = float(defense.certified_radius(x, n_samples=25))
out["p_a_one_clamp"] = {"radius": r, "finite": bool(math.isfinite(r))}

print(json.dumps(out, indent=2, sort_keys=True))
PY
rc=$?
set -e
echo "$rc" >"${RESULTS_DIR}/certified_radius_edge_cases.rc"

echo ""
echo "Failure analysis artifacts saved to: ${RESULTS_DIR}"
