#!/usr/bin/env bash
# Linux same-sample EMBER audit. Elastic documented that EMBER v2 features
# are not consistent on Mac. This wrapper is the Linux path.
#
#   docker build --build-arg INSTALL_MALWARE=1 -t neurinspectre:ember .
#   bash scripts/audit_ember_linux.sh /path/to/pe_dir
#
# Official reproduction still requires lief 0.9.0 or 0.10.1. lief 0.13.x
# (what ember's current API needs on modern Python) is recorded, not claimed
# as Elastic-verified. Add --require-official-reproduction only on a host
# that actually meets that bar.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PE="${1:?usage: bash scripts/audit_ember_linux.sh /path/to/pe_or_dir}"
shift || true
IMAGE="${EMBER_AUDIT_IMAGE:-neurinspectre:ember}"
if command -v docker >/dev/null 2>&1; then
  exec docker run --rm \
    -v "$ROOT:/app" \
    -v "$PE:$PE:ro" \
    -w /app \
    "$IMAGE" \
    audit --target ember-gbdt --pe-sample "$PE" --device cpu "$@"
fi
echo "[audit] docker not found; running local CLI (may be Mac / not official)" >&2
exec neurinspectre audit --target ember-gbdt --pe-sample "$PE" --device cpu "$@"
