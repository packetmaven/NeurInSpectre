#!/usr/bin/env python3
"""Summarize an EMBER same-sample audit report into a compact diagnosis JSON.

Works for both ``--target ember-gbdt`` (v2, dim 2381) and
``--target ember2024-*-gbdt`` (v3, dim 2568). Reads ``audit_report.json``,
walks ``same_sample_detail.samples`` for the GBDT-detected subset, and
records feature/problem ASR, the score-drop distribution, and the closest
approach with its transform.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load_report(report_path: Path) -> dict:
    if report_path.is_dir():
        report_path = report_path / "audit_report.json"
    if not report_path.is_file():
        raise SystemExit(f"audit_report.json not found at {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _summarize(report: dict) -> dict:
    same = report.get("same_sample_detail") or {}
    samples = same.get("samples") or []
    kept = [s for s in samples if s.get("kept")]
    if not kept:
        return {
            "kind": "ember_audit_diagnosis",
            "n": 0,
            "note": "no GBDT-detected malware in same_sample_detail.samples",
        }

    feature = (report.get("attacks") or {}).get("feature_square") or {}
    problem = report.get("problem_space") or {}
    fv = report.get("feature_vs_problem_space") or {}

    drops = []
    top = []
    transforms: dict[str, int] = {}
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="audit_report.json or its parent directory")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write diagnosis JSON here (default: sibling ember_audit_diagnosis.json)",
    )
    args = parser.parse_args()

    report = _load_report(args.report)
    diag = _summarize(report)

    out = args.output
    if out is None:
        base = args.report if args.report.is_dir() else args.report.parent
        out = base / "ember_audit_diagnosis.json"
    out.write_text(json.dumps(diag, indent=2, default=str))
    print(f"wrote {out}")
    if diag.get("n") == 0:
        return 1

    for key in (
        "target",
        "n",
        "feature_asr",
        "problem_valid_asr",
        "drop_median",
        "drop_max",
        "n_best_below_0.5",
        "closest",
        "transforms",
    ):
        print(f"  {key}: {diag.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
