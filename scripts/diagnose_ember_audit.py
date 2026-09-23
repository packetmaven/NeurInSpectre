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
from pathlib import Path

from neurinspectre.evaluation.ember_audit_diagnosis import (
    load_ember_audit_report,
    summarize_ember_audit_report,
)


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

    report = load_ember_audit_report(args.report)
    diag = summarize_ember_audit_report(report)

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
