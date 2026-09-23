#!/usr/bin/env python3
"""Score the EMBER 2024 challenge set (6,315 evasive-in-the-wild files) with
the shipped 2024 LightGBM sub-models.

The challenge JSONLs already include v3 raw features per record; we feed them
through ``thrember.PEFeatureExtractor.process_raw_features`` — no PE binaries
required. Runs offline against a pre-downloaded copy at
``data/ember/ember2024/dataset/challenge/`` (see
``scripts/download_ember2024_challenge.py``).

Reports per-model detection rate at threshold 0.5, per-file-type breakdown,
per-filter matched-cohort detection rate, and the fraction of challenge
files that continue to evade the 2024 model.

Capa-informed filters
---------------------

Every EMBER 2024 record carries per-file tags (Capa capabilities,
ATT&CK TTPs, MBC objectives/behaviors, packer/property/exploit/group
labels, family, file_type). A red-team engagement usually scopes to a
subset. Filters compose as OR within a field, AND across fields; matching
is case-insensitive substring and also honors bracketed ATT&CK/MBC IDs
(``T1055`` matches ``"Process Injection [T1055]"``).

Examples::

    # Ransomware-adjacent cohort
    python scripts/score_ember2024_challenge.py \
        --filter-mbc "ransom,C0055" \
        -o results/ember2024/challenge_scoring_ransom.json

    # ATT&CK T1055 Process Injection, Win32 only
    python scripts/score_ember2024_challenge.py \
        --filter-ttp T1055 --filter-file-type Win32

    # Nation-state grouped samples
    python scripts/score_ember2024_challenge.py \
        --filter-tag lazarusgroup,knotweed
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping

import lightgbm as lgb
import numpy as np

from neurinspectre.malware.capa_filters import TagFilter, apply_tag_filter
from neurinspectre.malware.miss_cohorts import score_all_namespaces

DEFAULT_MODELS = {
    "PE": "data/ember/ember2024/EMBER2024_PE.model",
    "Win32": "data/ember/ember2024/EMBER2024_Win32.model",
    "Win64": "data/ember/ember2024/EMBER2024_Win64.model",
}


def _iter_records(challenge_dir: Path):
    for jsonl in sorted(challenge_dir.glob("*.jsonl")):
        with jsonl.open() as fh:
            for line in fh:
                yield json.loads(line)


def _score(
    challenge_dir: Path,
    models: Mapping[str, str],
    flt: TagFilter,
    *,
    miss_min_support: int = 20,
    miss_limit_per_namespace: int = 25,
) -> dict:
    from neurinspectre.malware.ember2024_extract import _apply_authenticode_shim

    _apply_authenticode_shim()
    from thrember.features import PEFeatureExtractor

    extr = PEFeatureExtractor()

    # Two-stage read: filter first (cheap), then extract only what survived.
    n_seen = 0
    kept_records: List[Mapping] = []
    for row in _iter_records(challenge_dir):
        n_seen += 1
        if flt.match(row):
            kept_records.append(row)
    n_after_filter = len(kept_records)

    X_rows: List[np.ndarray] = []
    meta: List[Dict] = []
    errors: Counter = Counter()
    for row in kept_records:
        try:
            vec = extr.process_raw_features(row).astype(np.float32)
        except Exception as exc:  # pragma: no cover — corpus noise
            errors[type(exc).__name__] += 1
            continue
        if vec.size != 2568 or not np.isfinite(vec).all():
            errors["bad_shape_or_nonfinite"] += 1
            continue
        X_rows.append(vec)
        meta.append(
            {
                "sha256": row.get("sha256"),
                "file_type": row.get("file_type"),
                "family": row.get("family"),
                "week_id": row.get("week_id"),
                "detection_ratio": row.get("detection_ratio"),
            }
        )
    if not X_rows:
        return {
            "kind": "ember2024_challenge_scoring",
            "filter": flt.as_dict(),
            "filter_active": flt.is_active(),
            "n_records_total": n_seen,
            "n_after_filter": n_after_filter,
            "n_features_built": 0,
            "errors": dict(errors),
            "per_model": {},
            "note": "no records survived filter+extract; nothing to score",
        }
    X = np.stack(X_rows, axis=0)

    per_model = {}
    scores: Dict[str, np.ndarray] = {}
    for name, path in models.items():
        booster = lgb.Booster(model_file=str(path))
        preds = booster.predict(X)
        scores[name] = preds
        per_model[name] = {
            "detected_ge_0.5": int((preds >= 0.5).sum()),
            "detected_ge_0.9": int((preds >= 0.9).sum()),
            "median_p": float(np.median(preds)),
            "mean_p": float(np.mean(preds)),
            "p_lt_0.1": int((preds < 0.1).sum()),
        }

    by_type: Dict[str, Dict[str, List[float]]] = {name: {} for name in models}
    for idx, entry in enumerate(meta):
        ft = entry["file_type"]
        for name in models:
            by_type[name].setdefault(ft, []).append(float(scores[name][idx]))
    per_model_by_type = {
        name: {
            ft: {
                "n": len(vals),
                "detected_ge_0.5": int((np.array(vals) >= 0.5).sum()),
                "median_p": float(np.median(vals)),
            }
            for ft, vals in cells.items()
        }
        for name, cells in by_type.items()
    }

    miss_idx = [i for i, p in enumerate(scores["PE"]) if p < 0.5]
    miss_families = Counter(
        (meta[i].get("family") or "?") for i in miss_idx
    ).most_common(15)

    # A2 — per-label all-model-miss scoreboard. Uses the ORIGINAL record dicts
    # (not the trimmed meta) so all tag namespaces are available.
    all_missed_mask = [
        all(scores[name][i] < 0.5 for name in models) for i in range(len(kept_records))
    ]
    # kept_records is aligned with the filtered iteration; some rows may have
    # been dropped by the extractor before scoring, so we need the mask to
    # match ``meta`` (the successfully-extracted subset). Rebuild against
    # ``meta``'s sha256 index into kept_records.
    kept_by_sha = {r.get("sha256"): r for r in kept_records}
    scored_records = [kept_by_sha.get(m["sha256"], {}) for m in meta]
    all_missed_mask = [
        all(scores[name][i] < 0.5 for name in models) for i in range(len(scored_records))
    ]
    miss_cohorts = score_all_namespaces(
        scored_records,
        all_missed_mask,
        min_support=int(miss_min_support),
        limit_per_namespace=int(miss_limit_per_namespace),
    )

    return {
        "kind": "ember2024_challenge_scoring",
        "filter": flt.as_dict(),
        "filter_active": flt.is_active(),
        "n_records_total": n_seen,
        "n_after_filter": n_after_filter,
        "n_features_built": len(X_rows),
        "errors": dict(errors),
        "file_type_dist": dict(Counter(m["file_type"] for m in meta)),
        "per_model": per_model,
        "per_model_by_file_type": per_model_by_type,
        "n_miss_all_models": int(sum(all_missed_mask)),
        "pe_miss_top_families": miss_families,
        "miss_cohorts": miss_cohorts,
        "miss_cohorts_config": {
            "min_support": int(miss_min_support),
            "limit_per_namespace": int(miss_limit_per_namespace),
            "note": (
                "Per-namespace label miss rate: fraction of files carrying "
                "that label where ALL three sub-models score < 0.5. Sorted "
                "by miss_rate desc, min_support filters low-n labels."
            ),
        },
        "notes": [
            "Challenge set = 6315 files initially undetected by ALL VirusTotal AVs, "
            "later re-scanned and labeled malicious by enough AVs to qualify (30+ days).",
            "Scoring uses thrember v3 process_raw_features on pre-computed raw features "
            "(no PE binaries required).",
            "Non-PE file types (APK/ELF/PDF) are still passed to the PE detector for "
            "reporting; degraded detection there is expected, not a bug.",
            "Filters are OR within a field, AND across fields; matching is "
            "case-insensitive substring and honors bracketed ATT&CK/MBC IDs.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--challenge-dir",
        type=Path,
        default=Path("data/ember/ember2024/dataset/challenge"),
    )
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--pe-model", default=DEFAULT_MODELS["PE"])
    parser.add_argument("--win32-model", default=DEFAULT_MODELS["Win32"])
    parser.add_argument("--win64-model", default=DEFAULT_MODELS["Win64"])

    # Capa-informed filters. Each flag is repeatable AND accepts a comma list;
    # ``--filter-tag a --filter-tag b,c`` collects to ``[a, b, c]``.
    parser.add_argument("--filter-file-type", action="append", default=[],
                        help="Repeatable / comma list: Win32,Win64,Dot_Net,PDF,ELF,APK")
    parser.add_argument("--filter-family", action="append", default=[],
                        help="Repeatable / comma list of family substrings")
    parser.add_argument("--filter-tag", action="append", default=[],
                        help="Repeatable / comma list matching behavior/property/packer/exploit/group")
    parser.add_argument("--filter-ttp", action="append", default=[],
                        help="Repeatable / comma list matching ATT&CK tactic/technique/ID")
    parser.add_argument("--filter-mbc", action="append", default=[],
                        help="Repeatable / comma list matching MBC objective/behavior/ID")
    parser.add_argument("--filter-capability", action="append", default=[],
                        help="Repeatable / comma list matching Capa capability or namespace")
    parser.add_argument("--min-vt-detected", type=int, default=None,
                        help="Minimum VirusTotal numerator (from detection_ratio 'X/Y')")
    parser.add_argument("--miss-min-support", type=int, default=20,
                        help="Minimum #files carrying a label before it enters the miss scoreboard")
    parser.add_argument("--miss-limit-per-namespace", type=int, default=25,
                        help="How many top-miss-rate labels to keep per namespace in the JSON")

    args = parser.parse_args()

    challenge_dir = args.challenge_dir
    if not challenge_dir.is_dir():
        raise SystemExit(
            f"Challenge dir not found: {challenge_dir}. "
            "Run: python scripts/download_ember2024_challenge.py"
        )
    models = {"PE": args.pe_model, "Win32": args.win32_model, "Win64": args.win64_model}

    from neurinspectre.malware.capa_filters import _parse_csv

    def _flat(values):
        out = []
        for v in values or []:
            out.extend(_parse_csv(v))
        return out

    flt = TagFilter(
        file_type=_flat(args.filter_file_type),
        family=_flat(args.filter_family),
        tag=_flat(args.filter_tag),
        ttp=_flat(args.filter_ttp),
        mbc=_flat(args.filter_mbc),
        capability=_flat(args.filter_capability),
        min_vt_detected=args.min_vt_detected,
    )
    summary = _score(
        challenge_dir, models, flt,
        miss_min_support=args.miss_min_support,
        miss_limit_per_namespace=args.miss_limit_per_namespace,
    )
    out = args.output or (Path("results/ember2024/challenge_scoring.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    if flt.is_active():
        print(
            f"  filter: {flt.as_dict()}"
        )
        print(
            f"  seen={summary['n_records_total']}  after_filter={summary['n_after_filter']}  "
            f"features_built={summary['n_features_built']}"
        )
    if summary["n_features_built"] == 0:
        print("  no records survived filter+extract — nothing scored")
        return 0
    for name, cell in summary["per_model"].items():
        n = summary["n_features_built"]
        print(
            f"  {name:5s}  det>=0.5={cell['detected_ge_0.5']}/{n} "
            f"({cell['detected_ge_0.5']/n:.3f})  median={cell['median_p']:.4f}"
        )
    print(f"  n_miss_all_models: {summary['n_miss_all_models']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
