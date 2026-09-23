"""`neurinspectre redteam-bundle` — scope → audit → diagnosis → crossing → zip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import click

from ..evaluation.redteam_bundle import zip_audit_bundle
from ..malware.pe_scope import scope_pe_corpus
from ..malware.vt_sidecar import build_vt_sidecar_from_pe_dir


def run_redteam_bundle(
    *,
    pe_sample: str,
    output_dir: str,
    target: str = "ember2024-gbdt",
    smoke: bool = False,
    n_examples: int = 8,
    query_budgets: Optional[str] = None,
    enable_gamma_sections: bool = False,
    gamma_donor_dir: Optional[str] = None,
    enable_iat_edits: bool = False,
    save_best_bytes: bool = True,
    crossing_matrix: bool = True,
    write_diagnosis: bool = True,
    supplement_index: Optional[str] = None,
    challenge_dir: Optional[str] = None,
    vt_dataset_dir: Optional[str] = None,
    build_vt_sidecar: bool = True,
    sow_adapter: Optional[List[str]] = None,
    sow_adapter_ack: bool = False,
    av_system_name: Optional[str] = None,
    require_detected: bool = False,
) -> Path:
    from .audit_cmd import run_audit

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scope_path = out / "pe_scope.json"
    scope = scope_pe_corpus(
        Path(pe_sample),
        challenge_dir=Path(challenge_dir) if challenge_dir else None,
        supplement_index=Path(supplement_index) if supplement_index else None,
    )
    scope_path.write_text(json.dumps(scope, indent=2), encoding="utf-8")

    vt_path = out / "vt_sidecar.json"
    if build_vt_sidecar:
        vt_payload = build_vt_sidecar_from_pe_dir(
            Path(pe_sample),
            dataset_dir=Path(vt_dataset_dir) if vt_dataset_dir else None,
        )
        vt_path.write_text(json.dumps(vt_payload, indent=2), encoding="utf-8")
        vt_records = vt_payload.get("records") or {}

    ctx = click.Context(click.Command("audit"))
    audit_kwargs: dict[str, Any] = {
        "target": target,
        "output_dir": str(out),
        "pe_sample": pe_sample,
        "smoke": smoke,
        "n_examples": n_examples,
        "query_budgets": query_budgets,
        "enable_gamma_sections": enable_gamma_sections,
        "gamma_donor_dir": gamma_donor_dir,
        "enable_iat_edits": enable_iat_edits,
        "save_best_bytes": save_best_bytes,
        "crossing_matrix": crossing_matrix,
        "write_diagnosis": write_diagnosis,
        "require_detected": require_detected,
        "vt_sidecar": str(vt_path) if vt_path.is_file() else None,
        "sow_adapter": list(sow_adapter or []),
        "sow_adapter_ack": sow_adapter_ack,
        "av_system_name": av_system_name,
    }
    run_audit(ctx, **audit_kwargs)

    zip_path = zip_audit_bundle(out)
    click.echo(f"[redteam-bundle] wrote {zip_path}")
    return zip_path
