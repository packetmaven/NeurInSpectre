# Authorized red team — EMBER GBDT same-sample audit

Operator guide for **written ROE** engagements against static PE scorers (EMBER 2018 / 2024 LightGBM). Use `venv/bin/neurinspectre` (Python 3.10).

Full multi-engagement playbook (vision, Table 5, Capa lane, design-target modules):
[REDTEAM_PLAYBOOK.md](REDTEAM_PLAYBOOK.md). A frozen working copy may also exist under
`results/offensive_overnight_20260920/` (gitignored).

## Two frames (SOW closeout)

| Frame | Client question | NeurInSpectre |
|---|---|---|
| Engagement | Does the file still run, and did **deployed** AV/EDR miss it? | **Out of scope** for execution and live AV (read-only VT metadata sidecar is optional context only). |
| NeurInSpectre | On this **named checkpoint**, **byte regions**, **parse gate**, **query budget**, did malware score cross 0.5? | **Yes** — see `measurement_scope` in `audit_report.json`. |

GBDT audits set `pipeline.gradient_available: false` and `recommended_recipe: problem_space` (no JPEG BPDA routing).

`measurement_scope` lists engagement boundaries in `not_measured` (sandbox, commercial AV/EDR, graph/byte models, and transforms you did not enable). With `--enable-gamma-sections` or `--enable-iat-edits`, the corresponding bounded transform is recorded under `measured`; that still does not prove sandbox execution or AV miss.

## Operator loop

```bash
neurinspectre doctor --as-json -o doctor.json

neurinspectre scope-pe-corpus /client/malware_pe \
  --challenge-dir data/ember/ember2024/dataset/challenge \
  --supplement-index data/ember/ember2024/capa_supplement_index.json \
  -o results/pe_scope.json

neurinspectre run-capa /client/malware_pe --sidecar pe_tags.json \
  --supplement-index data/ember/ember2024/capa_supplement_index.json

neurinspectre audit --target ember2024-gbdt \
  --pe-sample /client/malware_pe \
  --require-detected \
  --query-budgets 10,50,100,500,5000 \
  --save-best-bytes --crossing-matrix --write-diagnosis \
  -o results/audit/ember2024_client

python scripts/diagnose_ember_audit.py results/audit/ember2024_client

neurinspectre transferability results/audit/ember2024_client/audit_report.json \
  --default-crossing -o results/audit/ember2024_client/crossing_matrix_cli.json
```

One-shot client bundle (same steps + `vt_sidecar.json` + zip):

```bash
neurinspectre redteam-bundle /client/malware_pe \
  -o results/audit/ember2024_client_bundle \
  --target ember2024-gbdt --require-detected \
  --query-budgets 10,50,100,500,5000
```

Optional extensions (see playbook for full flags):

```bash
neurinspectre vt-sidecar /client/malware_pe -o results/vt_sidecar.json
neurinspectre iat-probe /client/malware_pe/sample.exe
neurinspectre audit --target ember2024-gbdt --pe-sample /client/malware_pe \
  --vt-sidecar results/vt_sidecar.json --enable-iat-edits --require-detected \
  --save-best-bytes --sow-adapter sandbox_handoff -o results/audit/ember2024_iat
```

GAMMA section injection requires `pip install -e '.[gamma]'` and a real benign donor corpus (`--enable-gamma-sections`, `gamma-inject`).

Optional Capa diff on original vs `best_bytes/` (not a sandbox gate):

```bash
neurinspectre audit --target ember2024-gbdt --pe-sample /client/malware_pe \
  --n-examples 4 --require-detected --save-best-bytes \
  --capa-diff-best --capa-diff-backend file_level \
  -o results/audit/ember2024_capa_diff
```

`--crossing-matrix` and `--capa-diff-best` require `--save-best-bytes`.

## EMBER 2018 official reproduction

```bash
neurinspectre audit --target ember-gbdt --pe-sample /pe --require-official-reproduction
bash scripts/audit_ember_linux_lief090.sh /pe   # quote_as_ember2018 cell (LIEF 0.9.0 Linux)
```

Do not quote Mac or lief 0.13.x scores as official EMBER2018. Do not quote FeatureSquare ASR as PE-valid.

## Crossing rule (transferability / crossing_matrix)

A finding requires: clean `p ≥ 0.5`, bytes changed, mutated `p < 0.5` on the named model. `score_below_0.5` alone is not a finding if the clean file was already missed.

## Do not sell

- Sandbox execution or commercial AV success inferred from GBDT JSON alone (handoff export and `commercial_av_edr` placeholder are provenance only).
- Graph-model / MalConv / byte-model evasion from EMBER GBDT audit JSON alone.
- PE-valid evasion from FeatureSquare or from VT sidecar metadata without a parse-valid flip on the named model.
- GAMMA or IAT results unless those flags were enabled and cited from `measurement_scope` on that run.

Gap ids are explicit on each report (`measurement_scope.not_measured_ids`). Preflight:

```bash
neurinspectre engagement-gaps
```

- Crossing matrix **0/N** as multi-detector evasion without per-model clean scores.
