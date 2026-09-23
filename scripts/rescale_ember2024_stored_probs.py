"""Check stored 2024 malware scores against live predict_proba, then invert.

Historical audits stored softmax([-logit(p), +logit(p)]), i.e.
sigmoid(2·logit(p)). This script re-extracts every kept PE, requires
inverse(stored clean_p) to match live predict_proba, and only then writes
ember_audit_diagnosis_rescaled.json with inverted clean and best scores.

ASR and success flags are copied from the stored report. A file that fails
the clean check is not inverted.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from neurinspectre.malware.ember2024_extract import extract_ember2024_features
from neurinspectre.models.ember_gbdt import (
    EmberGBDT2024,
    lightgbm_probability_from_stored_softmax,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "ember2024-gbdt": ROOT / "data/ember/ember2024/EMBER2024_PE.model",
    "ember2024-win32-gbdt": ROOT / "data/ember/ember2024/EMBER2024_Win32.model",
    "ember2024-win64-gbdt": ROOT / "data/ember/ember2024/EMBER2024_Win64.model",
}
DEFAULT_REPORTS = (
    ROOT / "results/ember2024/audit_100q/audit_report.json",
    ROOT / "results/ember2024/audit_500q/audit_report.json",
    ROOT / "results/ember2024/audit_win32_100q/audit_report.json",
    ROOT / "results/ember2024/audit_win64_100q/audit_report.json",
    ROOT / "results/ember2024/C7/audit_slack/audit_report.json",
    ROOT / "results/ember2024/D10/audit_default/audit_report.json",
    ROOT / "results/ember2024/D10/audit_combined/audit_report.json",
)


def _live_p(model, path: Path, cache: dict) -> float:
    key = (str(path), id(model))
    if key not in cache:
        extracted = extract_ember2024_features(path.read_bytes())
        feats = extracted.get("features")
        if feats is None:
            raise RuntimeError(f"extract failed for {path}: {extracted.get('reasons')}")
        cache[key] = float(model.predict_proba(np.asarray(feats, dtype=np.float32).reshape(1, -1))[0, 1])
    return cache[key]


def _old_stored_softmax(p: float) -> float:
    """Softmax malware score of the old [-logit(p), +logit(p)] forward."""
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    s = float(np.log(p / (1.0 - p)))
    logits = np.array([-s, s], dtype=np.float64)
    exps = np.exp(logits - logits.max())
    return float(exps[1] / exps.sum())


def rescale_report(report_path: Path, *, atol: float, cache: dict, models: dict) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    target = str(report.get("target") or "")
    model_path = MODELS.get(target)
    if model_path is None or not model_path.is_file():
        raise SystemExit(f"{report_path}: no model for target {target!r}")
    if target not in models:
        models[target] = EmberGBDT2024.from_file(model_path)
    model = models[target]
    kept = [s for s in (report.get("same_sample_detail") or {}).get("samples") or [] if s.get("kept")]
    rows = []
    failures = []
    for sample in kept:
        problem = sample.get("problem") or {}
        stored_clean = sample.get("clean_p_malware")
        stored_best = problem.get("best_p_malware")
        path = Path(str(sample.get("path") or ""))
        if stored_clean is None or stored_best is None or not path.is_file():
            failures.append({"path": str(path), "reason": "missing_score_or_file"})
            continue
        live = _live_p(model, path, cache)
        inverted_clean = lightgbm_probability_from_stored_softmax(float(stored_clean))
        err = abs(inverted_clean - live)
        saturated = float(stored_clean) >= 1.0 - 1e-6
        forward_err = abs(_old_stored_softmax(live) - float(stored_clean)) if saturated else None
        ok = err <= atol or (saturated and forward_err is not None and forward_err <= 1e-6)
        row = {
            "name": path.name,
            "path": str(path),
            "stored_clean": float(stored_clean),
            "stored_best": float(stored_best),
            "live_clean": live,
            "clean_abs_error": err,
            "forward_abs_error": forward_err,
            "clean_check": "pass" if ok and not saturated else ("pass_saturated" if ok else "fail"),
            "chosen": problem.get("chosen_attack"),
            "queries": problem.get("queries_used"),
            "success": problem.get("success"),
            "realizable": problem.get("realizable"),
        }
        if ok:
            # A stored 1.0 has no inverse. The live probability is the score
            # of those bytes. best_p is inverted only when it did not saturate.
            if saturated:
                row["clean"] = live
            else:
                row["clean"] = inverted_clean
            if float(stored_best) >= 1.0 - 1e-6 and problem.get("chosen_attack") == "clean":
                row["best"] = live
            else:
                row["best"] = lightgbm_probability_from_stored_softmax(float(stored_best))
            row["drop"] = row["clean"] - row["best"]
        else:
            failures.append({"path": str(path), "reason": "clean_inverse_mismatch", "abs_error": err})
        rows.append(row)
    checked = [r for r in rows if r["clean_check"] != "fail"]
    drops = [r["drop"] for r in checked]
    closest = min(checked, key=lambda r: r["best"]) if checked else None
    transforms: dict[str, int] = {}
    for r in checked:
        transforms[r["chosen"]] = transforms.get(r["chosen"], 0) + 1
    ranked = sorted(checked, key=lambda r: -r["drop"])
    return {
        "kind": "ember_audit_diagnosis_rescaled",
        "scale": "lightgbm_probability_from_stored_softmax",
        "source_report": str(report_path),
        "target": target,
        "model": str(model_path),
        "n": len(kept),
        "n_checked": len(checked),
        "n_failed_clean_check": len(failures),
        "clean_check_atol": atol,
        "max_clean_abs_error": max((r["clean_abs_error"] for r in rows), default=None),
        "failures": failures,
        "query_budgets": report.get("query_budgets"),
        "official_reproduction": report.get("official_reproduction"),
        "quote_as_ember2018": report.get("quote_as_ember2018"),
        "feature_asr": ((report.get("attacks") or {}).get("feature_square") or {}).get("attack_success_rate"),
        "problem_asr": (report.get("problem_space") or {}).get("attack_success_rate"),
        "problem_valid_asr": (report.get("problem_space") or {}).get("valid_success_rate"),
        "stored_success_flags_unchanged": True,
        "drop_mean": statistics.mean(drops) if drops else None,
        "drop_median": statistics.median(drops) if drops else None,
        "drop_max": max(drops) if drops else None,
        "n_best_below_0.5": sum(1 for r in checked if r["best"] < 0.5),
        "closest": closest,
        "transforms": transforms,
        "top_drops": ranked[:10],
        "files": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    cache: dict = {}
    models: dict = {}
    any_fail = False
    for report_path in DEFAULT_REPORTS:
        diag = rescale_report(report_path, atol=args.atol, cache=cache, models=models)
        out = report_path.parent / "ember_audit_diagnosis_rescaled.json"
        if diag["n_failed_clean_check"]:
            any_fail = True
            # Still write the check, but best scores are absent on failed files.
        out.write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")
        print(
            f"{report_path.parent.name} n={diag['n']} "
            f"fail={diag['n_failed_clean_check']} "
            f"max_err={diag['max_clean_abs_error']} "
            f"closest_best={None if not diag['closest'] else diag['closest']['best']}",
            flush=True,
        )
    if any_fail:
        raise SystemExit("clean-score check failed for at least one file; those best_p values were not inverted")


if __name__ == "__main__":
    main()
