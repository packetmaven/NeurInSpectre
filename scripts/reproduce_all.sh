#!/bin/bash
set -euo pipefail

# =============================================================================
# Table 8 and toolkit harness (7-module toolkit + 12-defense evasion).
# Detection experiments live in scripts/reproduce_detection.sh.
# Default output directory: results/offensive_<timestamp>/.
# =============================================================================

echo "=== NeurInSpectre Reproduction Script ==="
echo "=== Table 8 and toolkit harness ==="
echo "=== Detection experiments: scripts/reproduce_detection.sh ==="
echo ""

RESULTS_DIR="${RESULTS_DIR:-results/offensive_$(date +%Y%m%d_%H%M%S)}"
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

# Harness controls (override via env)
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_CORE_EVASION="${SKIP_CORE_EVASION:-0}"
SKIP_TABLE2="${SKIP_TABLE2:-0}"
TABLE2_REUSE_DIR="${TABLE2_REUSE_DIR:-}"
EXPECTED_TABLE2_DEFENSES="${EXPECTED_TABLE2_DEFENSES:-12}"
TABLE2_NUM_SEEDS="${TABLE2_NUM_SEEDS:-1}"

is_true() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

validate_table2_triplet() {
  local out_dir="$1"
  local expected="$2"
  python3 - "$out_dir" "$expected" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
required = {"pgd", "autoattack", "neurinspectre"}

def _rows_from_summary():
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    results = obj.get("results")
    if not isinstance(results, list):
        return None
    rows = []
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            continue
        attacks = r.get("attacks")
        if not isinstance(attacks, dict):
            continue
        name = (
            r.get("defense")
            or r.get("name")
            or r.get("defense_name")
            or r.get("defense_id")
            or f"row_{i}"
        )
        rows.append((str(name), sorted(attacks.keys())))
    return rows


rows = _rows_from_summary()
source = "summary.json"
if rows is None:
    source = "per-defense JSON"
    rows = []
    for p in sorted(out_dir.glob("*.json")):
        if p.name in {"summary.json", "run_metadata.json"}:
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        attacks = obj.get("attacks")
        if not isinstance(attacks, dict):
            continue
        rows.append((p.name, sorted(attacks.keys())))

missing_triplet = []
for name, keys in rows:
    missing = sorted(required - set(keys))
    if missing:
        missing_triplet.append({"file": name, "missing_attacks": missing})

status = {
    "source": source,
    "expected_defenses": expected,
    "defense_count": len(rows),
    "triplet_complete_count": len(rows) - len(missing_triplet),
    "triplet_missing_count": len(missing_triplet),
    "triplet_missing": missing_triplet,
    "required_attacks": sorted(required),
    "complete": len(rows) == expected and not missing_triplet,
}

out_path = out_dir / "table8_validation.json"
out_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
print(f"[table8-validation] wrote {out_path}")
print(f"[table8-validation] defenses={len(rows)} expected={expected} complete={status['complete']}")
if missing_triplet:
    print("[table8-validation] missing triplets:")
    for entry in missing_triplet:
        print(f"  - {entry['file']}: missing={entry['missing_attacks']}")

if not status["complete"]:
    sys.exit(2)
PY
}

# CIFAR-10 core evasion defaults (override via env if desired)
MODEL_CIFAR10="${MODEL_CIFAR10:-}" # optional override; analyze has defaults for CIFAR-10
DATA_CIFAR10="${DATA_CIFAR10:-}"   # optional override; omit to allow torchvision download
EPS_CIFAR10="${EPS_CIFAR10:-0.03137254901960784}" # 8/255
ITERATIONS="${ITERATIONS:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"

echo "--- Environment ---"
"$NS_BIN" doctor --as-json > "$RESULTS_DIR/doctor.json"
"$NS_BIN" doctor > "$RESULTS_DIR/doctor.txt" 2>&1 || true

FAILS=0

echo ""
echo "--- Smoke Test (Click evaluate --smoke-test) ---"
if is_true "$SKIP_SMOKE"; then
  echo "  Skipped (SKIP_SMOKE=$SKIP_SMOKE)"
elif [[ -f "configs/evaluation_smoke.yaml" ]]; then
  "$NS_BIN" evaluate \
    --config configs/evaluation_smoke.yaml \
    --smoke-test \
    --output-dir "$RESULTS_DIR/evaluate_smoke" \
    --no-report --no-progress --report-format text \
    --device "$DEVICE" \
    > "$RESULTS_DIR/evaluate_smoke.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (missing configs/evaluation_smoke.yaml)"
fi

echo ""
echo "--- Core Evasion (CIFAR-10, 12 defenses) ---"
if is_true "$SKIP_CORE_EVASION"; then
  echo "  Skipped (SKIP_CORE_EVASION=$SKIP_CORE_EVASION)"
elif [[ -n "${DEFENSES:-}" ]]; then
  # Space-separated list: DEFENSES="jpeg bitdepth ..."
  read -r -a DEFENSE_LIST <<< "${DEFENSES}"
else
  DEFENSE_LIST=(
    "jpeg"
    "bitdepth"
    "randsmooth"
    "ensemble"
    "feature_squeezing"
    "gradient_regularization"
    "distillation"
    "at_transform"
    "spatial_smoothing"
    "random_pad_crop"
    "thermometer"
    "certified_defense"
  )
fi

if ! is_true "$SKIP_CORE_EVASION"; then
  for defense in "${DEFENSE_LIST[@]}"; do
    echo "  Running: $defense"
    out_json="$RESULTS_DIR/core_evasion_${defense}.json"
    out_log="$RESULTS_DIR/core_evasion_${defense}.log"
    cmd=(neurinspectre analyze
      --dataset cifar10
      --defense "$defense"
      --epsilon "$EPS_CIFAR10" --norm Linf
      --iterations "$ITERATIONS"
      --num-samples "$NUM_SAMPLES"
      --batch-size "$BATCH_SIZE"
      --device "$DEVICE"
      --no-report --no-progress
      --output "$out_json"
    )
    cmd[0]="$NS_BIN"
    if [[ -n "${MODEL_CIFAR10}" ]]; then
      cmd+=(--model "$MODEL_CIFAR10")
    fi
    if [[ -n "${DATA_CIFAR10}" ]]; then
      cmd+=(--data-path "$DATA_CIFAR10")
    fi

    if ! "${cmd[@]}" > "$out_log" 2>&1; then
        echo "    FAILED: $defense (see $out_log)"
        FAILS=$((FAILS + 1))
    fi
  done
fi

echo ""
echo "--- Table 8 (table2 orchestrator; 12 defenses x 3 attacks) ---"
if is_true "$SKIP_TABLE2"; then
  echo "  Skipped (SKIP_TABLE2=$SKIP_TABLE2)"
elif [[ -f "table2_config.yaml" ]]; then
  TABLE2_OUT="$RESULTS_DIR/table8_table2"
  mkdir -p "$TABLE2_OUT"

  if [[ -n "$TABLE2_REUSE_DIR" ]]; then
    echo "  Reusing existing Table 8 artifacts from: $TABLE2_REUSE_DIR"
    if [[ ! -d "$TABLE2_REUSE_DIR" ]]; then
      echo "  FAILED: TABLE2_REUSE_DIR does not exist: $TABLE2_REUSE_DIR"
      FAILS=$((FAILS + 1))
    else
      shopt -s nullglob
      for f in "$TABLE2_REUSE_DIR"/*.json "$TABLE2_REUSE_DIR"/*.log "$TABLE2_REUSE_DIR"/*.yaml "$TABLE2_REUSE_DIR"/*.tex; do
        [[ -f "$f" ]] && cp -f "$f" "$TABLE2_OUT/"
      done
      for d in "$TABLE2_REUSE_DIR"/seed_*; do
        [[ -d "$d" ]] && cp -R "$d" "$TABLE2_OUT/"
      done
      shopt -u nullglob
    fi
  else
    # Run incrementally so an interruption doesn't waste hours:
    # - stage 1: PGD
    # - stage 2: AutoAttack
    # - stage 3: NeurInSpectre
    #
    # `--resume` merges per-attack results into existing `<defense>.json` files.
    if ! "$NS_BIN" table2 \
        --config table2_config.yaml \
        --output-dir "$TABLE2_OUT" \
        --strict-real-data --strict-dataset-budgets \
        --no-report --no-progress --report-format text \
        --device "$DEVICE" \
        --num-seeds "$TABLE2_NUM_SEEDS" \
        --attacks pgd \
        > "$TABLE2_OUT/table2_pgd.log" 2>&1; then
      echo "  FAILED: table2 (pgd stage) (see $TABLE2_OUT/table2_pgd.log)"
      FAILS=$((FAILS + 1))
    fi

    if ! "$NS_BIN" table2 \
        --config table2_config.yaml \
        --output-dir "$TABLE2_OUT" \
        --strict-real-data --strict-dataset-budgets \
        --no-report --no-progress --report-format text \
        --device "$DEVICE" \
        --num-seeds "$TABLE2_NUM_SEEDS" \
        --resume \
        --attacks autoattack \
        > "$TABLE2_OUT/table2_autoattack.log" 2>&1; then
      echo "  FAILED: table2 (autoattack stage) (see $TABLE2_OUT/table2_autoattack.log)"
      FAILS=$((FAILS + 1))
    fi

    if ! "$NS_BIN" table2 \
        --config table2_config.yaml \
        --output-dir "$TABLE2_OUT" \
        --strict-real-data --strict-dataset-budgets \
        --no-report --no-progress --report-format text \
        --device "$DEVICE" \
        --num-seeds "$TABLE2_NUM_SEEDS" \
        --resume \
        --attacks neurinspectre \
        > "$TABLE2_OUT/table2_neurinspectre.log" 2>&1; then
      echo "  FAILED: table2 (neurinspectre stage) (see $TABLE2_OUT/table2_neurinspectre.log)"
      FAILS=$((FAILS + 1))
    fi
  fi

  if ! validate_table2_triplet "$TABLE2_OUT" "$EXPECTED_TABLE2_DEFENSES"; then
    echo "  FAILED: table2 artifact validation (see $TABLE2_OUT/table8_validation.json)"
    FAILS=$((FAILS + 1))
  fi

  # Export a Table 10-style "attack-strength" aggregate from the Table 8 runner summary.
  # This is intended to replace any draft table that implies runnable component-ablation toggles.
  python3 - "$TABLE2_OUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
summary_path = out_dir / "summary.json"
if not summary_path.exists():
    print(f"[table10-export] Skipped (missing {summary_path})")
    raise SystemExit(0)

obj = json.loads(summary_path.read_text(encoding="utf-8"))
results = obj.get("results", [])

def _valid(row: dict) -> bool:
    try:
        return bool(row["attacks"]["pgd"]["validity"]["passed"])
    except Exception:
        return False

def _asr(row: dict, attack: str) -> float:
    # Stored as fraction in [0,1] (conditional ASR).
    return float(row["attacks"][attack]["attack_success_rate"])

def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0

def _slice(name: str, rows: list[dict]) -> dict:
    aa = _mean([_asr(r, "autoattack") for r in rows])
    ni = _mean([_asr(r, "neurinspectre") for r in rows])
    pgd = _mean([_asr(r, "pgd") for r in rows])
    return {
        "name": str(name),
        "n": int(len(rows)),
        "pgd_asr_mean": pgd,
        "autoattack_asr_mean": aa,
        "neurinspectre_asr_mean": ni,
        "delta_neurinspectre_minus_autoattack": float(ni - aa),
        "pgd_asr_mean_pct": float(100.0 * pgd),
        "autoattack_asr_mean_pct": float(100.0 * aa),
        "neurinspectre_asr_mean_pct": float(100.0 * ni),
        "delta_neurinspectre_minus_autoattack_pct": float(100.0 * (ni - aa)),
    }

valid_rows = [r for r in results if _valid(r)]
cifar_valid = [r for r in valid_rows if r.get("dataset") == "cifar10"]
nuscenes_valid = [r for r in valid_rows if r.get("dataset") == "nuscenes"]
ember_valid = [r for r in valid_rows if r.get("dataset") == "ember"]
vision_valid = [r for r in valid_rows if r.get("dataset") in {"cifar10", "nuscenes"}]

payload = {
    "source": "table2_summary",
    "metric": "attack_success_rate (conditional; fraction in [0,1])",
    "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    "validity_filter": "attacks.pgd.validity.passed == true",
    "slices": [
        _slice("cifar10_valid", cifar_valid),
        _slice("nuscenes_valid", nuscenes_valid),
        _slice("vision_valid", vision_valid),
        _slice("ember_valid", ember_valid),
        _slice("all_valid", valid_rows),
    ],
}

out_json = out_dir / "table10_attack_strength.json"
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[table10-export] wrote {out_json}")

def _fmt(x: float) -> str:
    # One decimal place, matching the paper-facing tables.
    return f"{x:.1f}"

def _row(pretty: str, sl: dict) -> str:
    aa = _fmt(sl["autoattack_asr_mean_pct"])
    ni = _fmt(sl["neurinspectre_asr_mean_pct"])
    d = sl["delta_neurinspectre_minus_autoattack_pct"]
    d_str = ("+" if d >= 0 else "") + _fmt(d)
    return f"{pretty} & {aa} & {ni} & {d_str} \\\\"

sl_map = {s["name"]: s for s in payload["slices"]}
lines = []
lines.append("\\begin{table}[t]")
lines.append("\\centering")
lines.append("\\caption{Attack-strength comparison using Table 8 runner artifacts. Values are mean conditional ASR (\\%) over defenses that pass validity gates. $\\Delta$ = NeurInSpectre $-$ AutoAttack. \\newline\\textit{EMBER note: Under the current EMBER threat model ($\\ell_2$, $\\epsilon=0.5$), validity-passing EMBER defenses have ASR=0.0 for PGD/AutoAttack/NeurInSpectre. AutoAttack perturbation is 0 when ASR=0 because failures return $x_{\\mathrm{adv}}=x$.}}")
lines.append("\\label{tab:ablation}")
lines.append("\\small")
lines.append("\\begin{tabular}{@{}lrrr@{}}")
lines.append("\\toprule")
lines.append("\\textbf{Slice} & \\textbf{AutoAttack} & \\textbf{\\NI{}} & \\textbf{$\\Delta$} \\\\")
lines.append("\\midrule")
lines.append(_row(f"CIFAR-10 (valid, n={sl_map['cifar10_valid']['n']})", sl_map["cifar10_valid"]))
lines.append(_row(f"nuScenes (valid, n={sl_map['nuscenes_valid']['n']})", sl_map["nuscenes_valid"]))
lines.append(_row(f"Vision pooled (valid, n={sl_map['vision_valid']['n']})", sl_map["vision_valid"]))
lines.append(_row(f"EMBER (valid, n={sl_map['ember_valid']['n']})", sl_map["ember_valid"]))
lines.append(_row(f"All domains (valid, n={sl_map['all_valid']['n']})", sl_map["all_valid"]))
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
lines.append("")

out_tex = out_dir / "table10_attack_strength.tex"
out_tex.write_text("\n".join(lines), encoding="utf-8")
print(f"[table10-export] wrote {out_tex}")
PY

  python3 - "$TABLE2_OUT" "$RESULTS_DIR/table8_table2_artifacts.tgz" <<'PY'
import tarfile
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
include = []
for p in sorted(src.rglob("*")):
    if p.is_file() and p.suffix in {".json", ".log", ".yaml", ".tex"}:
        include.append(p)
with tarfile.open(dst, "w:gz") as tar:
    for p in include:
        tar.add(p, arcname=str(p.relative_to(src)))
print(f"[table8-archive] wrote {dst} with {len(include)} files")
PY
else
  echo "  Skipped (missing table2_config.yaml)"
fi

echo ""
echo "--- Optional: Gradient inversion (requires GRADIENTS_FILE) ---"
if [[ -n "${GRADIENTS_FILE:-}" && -f "${GRADIENTS_FILE:-}" ]]; then
  "$NS_BIN" gradient-inversion recover \
    --gradients "$GRADIENTS_FILE" \
    --out-prefix "$RESULTS_DIR/gradinv_" \
    > "$RESULTS_DIR/gradient_inversion.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set GRADIENTS_FILE=/path/to/gradients.npy)"
fi

echo ""
echo "--- Optional: RL obfuscation detection (requires RL_GRADIENT_FILE) ---"
if [[ -n "${RL_GRADIENT_FILE:-}" && -f "${RL_GRADIENT_FILE:-}" ]]; then
  "$NS_BIN" rl-obfuscation analyze \
    --input-file "$RL_GRADIENT_FILE" \
    --sensitivity high \
    --output-report "$RESULTS_DIR/rl_obfuscation_report.json" \
    --output-plot "$RESULTS_DIR/rl_obfuscation_plot.png" \
    > "$RESULTS_DIR/rl_obfuscation.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set RL_GRADIENT_FILE=/path/to/gradient.npy)"
fi

echo ""
echo "--- Optional: Attention security (requires ATTENTION_PROMPT + HF model access) ---"
if [[ -n "${ATTENTION_PROMPT:-}" ]]; then
  ATTENTION_MODEL="${ATTENTION_MODEL:-gpt2}"
  "$NS_BIN" attention-security \
    --model "$ATTENTION_MODEL" \
    --prompt "$ATTENTION_PROMPT" \
    --device "$DEVICE" \
    --output-png "$RESULTS_DIR/attention_security.png" \
    --out-json "$RESULTS_DIR/attention_security.json" \
    --out-html "$RESULTS_DIR/attention_security.html" \
    > "$RESULTS_DIR/attention_security.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set ATTENTION_PROMPT='...'; optional ATTENTION_MODEL=gpt2)"
fi

echo ""
echo "--- Optional: Subnetwork hijack (requires ACTIVATIONS_FILE) ---"
if [[ -n "${ACTIVATIONS_FILE:-}" && -f "${ACTIVATIONS_FILE:-}" ]]; then
  "$NS_BIN" subnetwork_hijack identify \
    --activations "$ACTIVATIONS_FILE" \
    --n_clusters "${SNH_CLUSTERS:-8}" \
    --out-prefix "$RESULTS_DIR/snh_" \
    --interactive \
    > "$RESULTS_DIR/subnetwork_hijack.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set ACTIVATIONS_FILE=/path/to/activations.npy)"
fi

echo ""
echo "--- Optional: EDNN (requires EDNN_EMBEDDINGS_FILE) ---"
if [[ -n "${EDNN_EMBEDDINGS_FILE:-}" && -f "${EDNN_EMBEDDINGS_FILE:-}" ]]; then
  EDNN_MODEL="${EDNN_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
  "$NS_BIN" adversarial-ednn \
    --attack-type inversion \
    --data "$EDNN_EMBEDDINGS_FILE" \
    --model "$EDNN_MODEL" \
    --device "$DEVICE" \
    --output-dir "$RESULTS_DIR/ednn" \
    --verbose \
    > "$RESULTS_DIR/ednn.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set EDNN_EMBEDDINGS_FILE=/path/to/embeddings.npy; optional EDNN_MODEL=...)"
fi

echo ""
echo "--- Optional: Statistical evasion scoring (requires STAT_ATTACK + STAT_REFERENCE) ---"
if [[ -n "${STAT_ATTACK:-}" && -f "${STAT_ATTACK:-}" && -n "${STAT_REFERENCE:-}" && -f "${STAT_REFERENCE:-}" ]]; then
  "$NS_BIN" statistical_evasion score \
    --data "$STAT_ATTACK" \
    --reference "$STAT_REFERENCE" \
    --method ks \
    --alpha 0.05 \
    --out-prefix "$RESULTS_DIR/stat_" \
    > "$RESULTS_DIR/statistical_evasion.log" 2>&1 || FAILS=$((FAILS + 1))
else
  echo "  Skipped (set STAT_ATTACK=/path/to/attack.npy and STAT_REFERENCE=/path/to/reference.npy)"
fi

echo ""
echo "--- Single-command-per-table mapping ---"
cat > "$RESULTS_DIR/table_command_map.md" <<EOF
# Single Command Per Table (Artifact Harness)

This file records the canonical single command (or harness target) used to
reproduce each table from this repository.

## Table 8 (Core defenses x attacks)

- Command: \`$NS_BIN table2 --config table2_config.yaml --output-dir $RESULTS_DIR/table8_table2 --strict-real-data --strict-dataset-budgets\`
- Artifact bundle: \`$RESULTS_DIR/table8_table2_artifacts.tgz\`
- Validation: \`$RESULTS_DIR/table8_table2/table8_validation.json\`

## Table 3 (Toolkit modules)

- Command: \`bash scripts/reproduce_all.sh\` (this harness executes the runnable CLI matrix and optional module checks).
- Artifacts: \`$RESULTS_DIR/*.json\`, \`$RESULTS_DIR/*.log\`, and module-specific outputs in subdirectories.

## Notes

- Set \`TABLE2_REUSE_DIR=<existing_table8_dir>\` to package an already-complete Table 8 run without rerunning compute.
- Set \`SKIP_CORE_EVASION=1\` and/or \`SKIP_SMOKE=1\` to scope execution.
- Set \`TABLE2_NUM_SEEDS=5\` (or pass \`--seeds\`) for multi-seed replication; outputs \`seed_<seed>/\` subdirs and adds \`*_std\` and \`*_ci95\` fields to \`summary.json\`.
EOF

echo ""
echo "=== All results saved to: $RESULTS_DIR ==="
echo "=== SHA256 manifest ==="
python3 - "$RESULTS_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = root / "sha256_manifest.txt"
paths = []
for p in sorted(root.rglob("*")):
    if not p.is_file():
        continue
    if p.name == manifest.name:
        continue
    if p.suffix.lower() not in {".json", ".log", ".md", ".txt", ".tgz", ".yaml", ".tex"}:
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
  echo "ERROR: $FAILS run(s) failed."
  exit 1
fi

echo ""
echo "OK"

