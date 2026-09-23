"""Operator bundle: zip audit artifacts after scope + audit + diagnosis + crossing."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def collect_bundle_files(audit_dir: Path) -> List[Path]:
    audit_dir = Path(audit_dir)
    names = [
        "audit_report.json",
        "audit_config.yaml",
        "ember_audit_diagnosis.json",
        "crossing_matrix.json",
        "capa_diff_audit.json",
        "pe_scope.json",
        "vt_sidecar.json",
        "sow_adapter_results.json",
        "bundle_manifest.json",
    ]
    out: List[Path] = []
    for name in names:
        p = audit_dir / name
        if p.is_file():
            out.append(p)
    best = audit_dir / "best_bytes"
    if best.is_dir():
        for f in sorted(best.glob("*.bin")):
            out.append(f)
    handoff = audit_dir / "sandbox_handoff"
    if handoff.is_dir():
        for f in sorted(handoff.rglob("*")):
            if f.is_file():
                out.append(f)
    return out


def write_bundle_manifest(audit_dir: Path, extra: Optional[Dict[str, Any]] = None) -> Path:
    audit_dir = Path(audit_dir)
    files = collect_bundle_files(audit_dir)
    manifest = {
        "kind": "redteam_bundle_manifest",
        "audit_dir": str(audit_dir.resolve()),
        "files": [
            str(p.relative_to(audit_dir))
            for p in files
            if str(p).startswith(str(audit_dir.resolve()))
        ],
        "extra": extra or {},
    }
    path = audit_dir / "bundle_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def zip_audit_bundle(audit_dir: Path, zip_path: Optional[Path] = None) -> Path:
    audit_dir = Path(audit_dir)
    write_bundle_manifest(audit_dir)
    files = collect_bundle_files(audit_dir)
    out_zip = Path(zip_path) if zip_path else audit_dir / "redteam_bundle.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arc = str(f.relative_to(audit_dir))
            zf.write(f, arcname=arc)
    return out_zip
