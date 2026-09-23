"""Compact EMBER same-sample audit diagnosis (shared by CLI and script)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, Union


def load_ember_audit_report(report_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(report_path)
    if path.is_dir():
        path = path / "audit_report.json"
    if not path.is_file():
        raise FileNotFoundError(f"audit_report.json not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_ember_audit_report(report: Dict[str, Any]) -> Dict[str, Any]:
    same = report.get("same_sample_detail") or {}
    samples = same.get("samples") or []
    kept = [s for s in samples if s.get("kept")]
    if not kept:
        return {
            "kind": "ember_audit_diagnosis",
            "n": 0,
            "note": "no GBDT-detected malware in same_sample_detail.samples",
            "measurement_scope": report.get("measurement_scope"),
        }

    feature = (report.get("attacks") or {}).get("feature_square") or {}
    problem = report.get("problem_space") or {}
    fv = report.get("feature_vs_problem_space") or {}

    drops = []
    top = []
    transforms: Dict[str, int] = {}
    closest = None
    for s in kept:
        pr = s.get("problem") or {}
        clean = s.get("clean_p_malware")
        best = pr.get("best_p_malware")
        if clean is None or best is None:
            continue
        drop = float(clean) - float(best)
        drops.append(drop)
        chosen = pr.get("chosen_attack")
        transforms[chosen] = transforms.get(chosen, 0) + 1
        name = Path(str(s.get("path") or s.get("name") or "?")).name
        row = {
            "name": name,
            "clean": float(clean),
            "best": float(best),
            "drop": drop,
            "chosen": chosen,
            "queries": pr.get("queries_used"),
            "success": pr.get("success"),
            "realizable": pr.get("realizable"),
        }
        top.append(row)
        if closest is None or best < closest["best"]:
            closest = row

    top.sort(key=lambda r: -r["drop"])

    return {
        "kind": "ember_audit_diagnosis",
        "target": report.get("target"),
        "n": len(kept),
        "query_budgets": report.get("query_budgets"),
        "official_reproduction": report.get("official_reproduction"),
        "quote_as_ember2018": report.get("quote_as_ember2018"),
        "feature_asr": feature.get("attack_success_rate")
        if feature
        else fv.get("feature_space_asr"),
        "problem_asr": problem.get("attack_success_rate"),
        "problem_valid_asr": problem.get("valid_success_rate"),
        "drop_mean": statistics.mean(drops) if drops else None,
        "drop_median": statistics.median(drops) if drops else None,
        "drop_max": max(drops) if drops else None,
        "n_drop_gt_0.01": sum(1 for d in drops if d > 0.01),
        "n_drop_gt_0.1": sum(1 for d in drops if d > 0.1),
        "n_best_below_0.5": sum(1 for r in top if r["best"] < 0.5),
        "closest": closest,
        "transforms": transforms,
        "top_drops": top[:10],
        "extractor_status": (same.get("extractor") or {}).get("shims"),
        "query_curve": (problem.get("query_curve") if isinstance(problem, dict) else None),
        "measurement_scope": report.get("measurement_scope"),
        "closest_still_malicious": (
            closest
            if closest and float(closest.get("best", 1.0)) >= 0.5
            else None
        ),
        "notes": [
            "Feature-space ASR is unrealizable L-inf on mixed-scale features; not PE-valid.",
            "Problem-space is Full DOS + padding/overlay. Validity is parse-only.",
            "measurement_scope in the report lists what this CLI does not claim (sandbox, AV, section injection).",
        ],
    }
