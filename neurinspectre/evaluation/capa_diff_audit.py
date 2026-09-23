"""One-shot Capa diff: original PE vs audit best_bytes (vivisect/full backend)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def capa_diff_best_bytes_report(
    report_path: Path,
    *,
    rules_dir: Optional[Path] = None,
    backend: str = "full",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    detail = report.get("same_sample_detail") or {}
    manifest = list(detail.get("best_bytes_manifest") or [])
    if max_samples is not None:
        manifest = manifest[: int(max_samples)]

    try:
        from neurinspectre.malware.capa_scan import CapaUnavailable, capa_diff
    except ImportError as exc:
        return {
            "kind": "capa_diff_audit",
            "source_report": str(report_path),
            "error": f"capa_scan import failed: {exc}",
            "per_sample": [],
        }

    capa_backend = "file_level" if backend == "file_level" else "full"
    rules_p = Path(rules_dir) if rules_dir else None
    per_sample: List[Dict[str, Any]] = []
    errors = 0

    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        orig = Path(str(entry.get("sample_path") or ""))
        mut = Path(str(entry.get("path") or ""))
        sha = entry.get("sha256_original")
        row: Dict[str, Any] = {
            "sha256_original": sha,
            "sample_path": str(orig) if orig else None,
            "mutated_path": str(mut) if mut else None,
            "chosen_attack": entry.get("chosen_attack"),
        }
        if not orig.is_file() or not mut.is_file():
            row["error"] = "missing_path"
            errors += 1
            per_sample.append(row)
            continue
        try:
            ob = orig.read_bytes()
            mb = mut.read_bytes()
            diff = capa_diff(
                ob, mb, backend=capa_backend, rules_dir=rules_p,
            )
            row["backend"] = capa_backend
            row["capa_diff"] = diff
        except CapaUnavailable as exc:
            row["error"] = str(exc)
            errors += 1
        per_sample.append(row)

    return {
        "kind": "capa_diff_audit",
        "source_report": str(report_path),
        "backend": backend,
        "n_manifest": len(manifest),
        "n_scanned": len(per_sample),
        "n_errors": errors,
        "note": (
            "Post-hoc diff only; not a sandbox or AV gate. "
            "full backend matches function-level rules (slow)."
        ),
        "per_sample": per_sample,
    }
