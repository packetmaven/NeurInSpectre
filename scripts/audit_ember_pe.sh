#!/usr/bin/env bash
# Same-sample EMBER GBDT audit. Public EMBER2018 has no PE binaries.
# Put PE files in a directory and pass --pe-sample.
#
#   bash scripts/audit_ember_pe.sh /path/to/pe_dir
#
# Official 2381-d features need lief + elastic/ember. Without them the report
# records ember_extractor_unavailable and both ASRs stay null. That is not a
# finding. Synthetic fixture PEs are not a paper corpus.
#
# Optional GAMMA-padding payloads (not section injection):
#   bash scripts/audit_ember_pe.sh /path/to/malware_dir --benign-corpus /path/to/benign_dir
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PE="${1:?usage: bash scripts/audit_ember_pe.sh /path/to/pe_or_dir}"
shift || true
PYTHON="${PYTHON:-python}"
if [[ -x "$ROOT/venv/bin/python" ]]; then
  PYTHON="$ROOT/venv/bin/python"
fi
"$PYTHON" -m neurinspectre.cli audit --target ember-gbdt --smoke \
  --pe-sample "$PE" \
  --output-dir results/audit/ember_gbdt_same_sample \
  --device cpu \
  "$@"
