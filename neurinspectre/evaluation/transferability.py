"""D9 — Transferability: re-score audit best-of-search mutated PEs
against additional detectors.

An audit run with ``best_bytes_dir=<dir>`` writes each sample's best-of-
search mutated bytes to ``<dir>/<sha256_original>.mutated.bin`` and records
a manifest inside the report. This module loads those mutated bytes and
scores them against a fresh set of EMBER 2024 sub-model checkpoints,
producing a per-file × per-model matrix of malware probabilities. This
tells us whether an evasion tuned against, e.g., ``EMBER2024_PE.model``
transfers to ``EMBER2024_Win64.model`` and ``EMBER2024_Win32.model``.

The re-scorer does not run any search. It extracts features from the
clean file and the saved bytes and reads ``predict_proba``. A finding is
a crossing: clean p >= 0.5, the bytes changed, and mutated p < 0.5.
Manifest rows with a null ``chosen_attack`` or ``sample_path`` are
excluded. ``p < 0.5`` on the mutated bytes alone is not a finding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..malware.ember2024_extract import extract_ember2024_features


def _load_report(report_path: Path) -> Dict[str, Any]:
    return json.loads(Path(report_path).read_text(encoding="utf-8"))


_REQUIRED_MANIFEST_FIELDS = (
    "sample_path",
    "chosen_attack",
    "best_p_malware",
    "sha256_original",
    "path",
)


def manifest_entry_problems(entry: Dict[str, Any]) -> List[str]:
    """Reasons a manifest row cannot support a transfer finding.

    A row with a null ``chosen_attack`` or ``sample_path`` is not an attack
    record. The current writer sets both; older partial manifests did not.
    Those rows stay out of the denominator.
    """
    if not isinstance(entry, dict):
        return ["not_an_object"]
    problems: List[str] = []
    for field in _REQUIRED_MANIFEST_FIELDS:
        value = entry.get(field)
        if field == "best_p_malware":
            if value is None:
                problems.append("best_p_malware_missing")
            continue
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{field}_missing")
    return problems


def _extract_manifest(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    ss = report.get("same_sample_detail") or {}
    manifest = ss.get("best_bytes_manifest") or []
    return list(manifest)


def _load_model(model_path: Path, name: str):
    """Load an EMBER GBDT by path.

    Deferred import so unit tests can stub. Returns a model exposing
    ``predict_proba(features_2d_np)``.
    """
    from ..models.ember_gbdt import EmberGBDT2024
    m = EmberGBDT2024.from_file(Path(model_path))
    m.classifier_name = name
    return m


def score_transferability(
    report_path: Path,
    models: Iterable[Tuple[str, Path]],
    *,
    dim_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the transferability matrix for an audit report.

    ``models`` is an iterable of ``(model_name, model_path)`` pairs.

    A transfer finding is a threshold crossing on one model: the clean file
    scored at least 0.5, the saved bytes differ from the clean file, and the
    mutated file scores below 0.5. ``p < 0.5`` on the mutated bytes alone is
    recorded as ``score_below_0.5_by_model`` and is not a finding: a model
    that already missed the clean file, or an unmodified file, stays a
    baseline miss.

    Rows missing ``chosen_attack``, ``sample_path``, or the byte paths are
    excluded from the denominator.

    Output structure::

      {
        "source_report": "<path>",
        "n_manifest": int,
        "n_samples": int,          # scored rows only
        "n_excluded": int,
        "excluded": [{"sha256_original": ..., "problems": [...]}, ...],
        "models": [{"name": ..., "path": ...}, ...],
        "per_sample": [
            {
                "sample_path": ...,
                "sha256_original": ...,
                "chosen_attack": ...,
                "identical_to_original": bool,
                "clean_p_by_model": {"PE": 0.99, ...},
                "best_p_by_model": {"PE": 0.87, ...},
                "transferred_by_model": {"PE": False, ...},
                "baseline_miss_by_model": {"Win32": True, ...},
                "score_below_0.5_by_model": {"Win32": True, ...},
            }, ...
        ],
        "summary": {
            "flip_rate_by_model": {"PE": 0.0, ...},   # transfer rate
            "transfer_rate_by_model": {"PE": 0.0, ...},
            "baseline_miss_rate_by_model": {...},
            "score_below_0.5_rate_by_model": {...},
            "any_model_flip_rate": 0.0,
            "all_model_flip_rate": 0.0,
        }
      }
    """
    report = _load_report(report_path)
    manifest = _extract_manifest(report)
    loaded = [(name, _load_model(Path(p), name)) for name, p in models]
    per_sample: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    transfer_by_model: Dict[str, List[bool]] = {name: [] for name, _ in loaded}
    baseline_miss_by_model: Dict[str, List[bool]] = {name: [] for name, _ in loaded}
    below_by_model: Dict[str, List[bool]] = {name: [] for name, _ in loaded}
    any_transfers: List[bool] = []
    all_transfers: List[bool] = []

    def _score_vec(model, features) -> float:
        model_dim = getattr(model, "feature_dim", None) or int(np.asarray(features).reshape(-1).size)
        vec = np.asarray(features, dtype=np.float32).reshape(-1)
        if model_dim < len(vec):
            vec = vec[:model_dim]
        elif model_dim > len(vec):
            vec = np.pad(vec, (0, model_dim - len(vec)))
        probs = model.predict_proba(vec[None, :])
        return float(probs[0, 1])

    for entry in manifest:
        problems = manifest_entry_problems(entry)
        sha = entry.get("sha256_original") if isinstance(entry, dict) else None
        if problems:
            excluded.append({"sha256_original": sha, "problems": problems})
            continue
        mutated_path = Path(entry["path"])
        original_path = Path(entry["sample_path"])
        if not mutated_path.is_file():
            excluded.append({"sha256_original": sha, "problems": ["mutated_bytes_missing"]})
            continue
        if not original_path.is_file():
            excluded.append({"sha256_original": sha, "problems": ["original_bytes_missing"]})
            continue
        mutated_bytes = mutated_path.read_bytes()
        original_bytes = original_path.read_bytes()
        extracted_m = extract_ember2024_features(mutated_bytes)
        extracted_c = extract_ember2024_features(original_bytes)
        if extracted_m.get("features") is None or extracted_c.get("features") is None:
            excluded.append({
                "sha256_original": sha,
                "problems": ["extract_failed"],
                "mutated_reasons": extracted_m.get("reasons") or [],
                "clean_reasons": extracted_c.get("reasons") or [],
            })
            continue
        identical = bool(entry.get("identical_to_original")) or mutated_bytes == original_bytes
        row_out: Dict[str, Any] = {
            **entry,
            "extract_ok": True,
            "identical_to_original": identical,
            "clean_p_by_model": {},
            "best_p_by_model": {},
            "transferred_by_model": {},
            "baseline_miss_by_model": {},
            "score_below_0.5_by_model": {},
            # Kept so older readers see the transfer bit, not "mutated p < 0.5".
            "flipped_by_model": {},
        }
        this_all = True
        this_any = False
        for name, model in loaded:
            p_clean = _score_vec(model, extracted_c["features"])
            p_mut = _score_vec(model, extracted_m["features"])
            below = bool(p_mut < 0.5)
            baseline_miss = bool(p_clean < 0.5)
            transferred = bool((not identical) and (p_clean >= 0.5) and below)
            row_out["clean_p_by_model"][name] = p_clean
            row_out["best_p_by_model"][name] = p_mut
            row_out["score_below_0.5_by_model"][name] = below
            row_out["baseline_miss_by_model"][name] = baseline_miss
            row_out["transferred_by_model"][name] = transferred
            row_out["flipped_by_model"][name] = transferred
            transfer_by_model[name].append(transferred)
            baseline_miss_by_model[name].append(baseline_miss)
            below_by_model[name].append(below)
            this_all = this_all and transferred
            this_any = this_any or transferred
        per_sample.append(row_out)
        any_transfers.append(this_any)
        all_transfers.append(this_all)

    def _rate(groups: Dict[str, List[bool]]) -> Dict[str, Optional[float]]:
        return {
            name: (float(sum(v) / len(v)) if v else None)
            for name, v in groups.items()
        }

    n = len(per_sample)
    summary = {
        "definition": (
            "flip_rate_by_model is the transfer rate: clean p >= 0.5, "
            "bytes changed, mutated p < 0.5. Rows with a null chosen_attack "
            "or sample_path are excluded. score_below_0.5_rate_by_model is "
            "not a finding."
        ),
        "n_manifest": len(manifest),
        "n_excluded": len(excluded),
        "flip_rate_by_model": _rate(transfer_by_model),
        "transfer_rate_by_model": _rate(transfer_by_model),
        "baseline_miss_rate_by_model": _rate(baseline_miss_by_model),
        "score_below_0.5_rate_by_model": _rate(below_by_model),
        "any_model_flip_rate": float(sum(any_transfers) / n) if n else None,
        "all_model_flip_rate": float(sum(all_transfers) / n) if n else None,
    }
    return {
        "source_report": str(report_path),
        "n_manifest": len(manifest),
        "n_samples": n,
        "n_excluded": len(excluded),
        "excluded": excluded,
        "models": [{"name": n_, "path": str(p_)} for n_, p_ in
                   ((name, mp) for name, mp in [(n_, Path(p_)) for n_, p_ in models])],
        "per_sample": per_sample,
        "summary": summary,
    }
