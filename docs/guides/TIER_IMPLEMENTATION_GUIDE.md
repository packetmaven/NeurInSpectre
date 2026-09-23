# Tier 1/2/3 Implementation Guide

This guide turns the audit checklist into concrete, reproducible execution tiers.
All commands are copy/paste-ready from repo root.

## Tier 1: Artifact-Grade Reproduction (execution first)

Goal: produce a reviewer-checkable bundle for the completed Table 8 pipeline.

### 1) Environment sanity

```bash
source .venv-neurinspectre/bin/activate
neurinspectre doctor --as-json > results/tier1_doctor.json
```

### 2) Produce the Table 8 harness bundle

If `results/table8_run_v2` is already complete, package it directly:

```bash
RESULTS_DIR=results/repro_harness_table8_v2 \
SKIP_SMOKE=1 \
SKIP_CORE_EVASION=1 \
TABLE2_REUSE_DIR=results/table8_run_v2 \
bash scripts/reproduce_all.sh
```

If you need to rerun full compute from scratch, remove `TABLE2_REUSE_DIR` and the
`SKIP_*` flags.

### 3) Acceptance checks

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("results/repro_harness_table8_v2/table8_table2/table8_validation.json")
o = json.loads(p.read_text())
print(o["complete"], o["triplet_complete_count"], o["defense_json_count"])
assert o["complete"] is True
assert o["triplet_complete_count"] == 12
assert o["defense_json_count"] == 12
PY
```

Expected key artifacts:

- `results/repro_harness_table8_v2/table8_table2/table8_validation.json`
- `results/repro_harness_table8_v2/table8_table2_artifacts.tgz`
- `results/repro_harness_table8_v2/table_command_map.md`
- `results/repro_harness_table8_v2/sha256_manifest.txt`

## Tier 2: Claim/Number Reconciliation (audit rigor)

Goal: verify claim→code→command→artifact traceability and quantify mismatches.

### 1) Pull finalized Table 8 metrics

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path("results/table8_run_v2")
for p in sorted(root.glob("*.json")):
    if p.name in {"summary.json", "run_metadata.json"}:
        continue
    o = json.loads(p.read_text())
    a = o.get("attacks", {})
    if not isinstance(a, dict):
        continue
    req = {"pgd", "autoattack", "neurinspectre"}
    if not req.issubset(a.keys()):
        continue
    print(p.name, a["pgd"]["attack_success_rate"], a["autoattack"]["attack_success_rate"], a["neurinspectre"]["attack_success_rate"])
PY
```

### 2) Explicit mismatch report for JPEG row

```bash
python3 - <<'PY'
import json
o = json.loads(open("results/table8_run_v2/cm_jpeg_compression.json").read())
a = o["attacks"]
print("conditional", {k: a[k]["attack_success_rate"] for k in ("pgd","autoattack","neurinspectre")})
print("overall", {k: a[k]["attack_success_rate_overall"] for k in ("pgd","autoattack","neurinspectre")})
print("robust_acc", {k: a[k]["robust_accuracy"] for k in ("pgd","autoattack","neurinspectre")})
PY
```

### 3) Confirm runnable model wiring used for claims

```bash
rg "^(models:|  cifar10:|  ember:|  nuscenes:)" results/table8_run_v2/resolved_table2_config.yaml
```

## Tier 3: Parity-Upgrade Execution Plan (implementation work)

Goal: convert remaining paper/code mismatches into tracked engineering tasks.

### A) Defense realism upgrades

- Per-defense checkpoint loading in runnable `table2` is implemented (spec `checkpoint_tag` -> runnable `model_path`),
  including an audit-friendly fallback to the per-dataset model when an artifact is missing.
  - Tests: `tests/test_table2_model_provenance_wiring.py`, `tests/test_checkpoint_tag_resolution.py`
- Real EMBER defense-training routines are implemented for:
  - `gradient_regularization` (Jacobian penalty)
  - `defensive_distillation` (teacher->student)
  - `at_transform` (noise transform + L2 PGD-k adversarial training)
  - Script: `scripts/train_ember_defense_models.py` writes TorchScript artifacts for the `md_*_ember` checkpoint tags.

Validation gate:

```bash
./.venv-neurinspectre/bin/python -m pytest tests/test_attacks.py tests/test_characterization.py -q
```

### B) Statistical evasion parity

- Implemented per-dimension KS/AD/CvM aggregation + Fisher + BH/FDR + iterative evasion loop.
  - Code: `neurinspectre/statistical/drift_detection_enhanced.py`, `neurinspectre/statistical/evasion.py`
  - Tests: `tests/test_statistical_evasion_parity.py`

### C) Tooling parity

- ECC-backed activation steganography implemented (Hamming(7,4)) and wired into the CLI.
  - Code: `neurinspectre/ecc_activation_steganography.py`
  - Tests: `tests/test_activation_steganography_ecc.py`
- Per-head attention-security feature extraction mode implemented alongside token-level mode.
  - Code: `neurinspectre/cli/attention_security_analysis.py` (`anomaly_level=head`)
  - Tests: `tests/test_attention_security_head_mode.py`

### D) Reporting policy

- Until A/B/C are complete, keep claims constrained to runnable behavior.

