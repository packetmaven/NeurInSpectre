"""Same-sample evaluation for non-PE EMBER 2024 detectors (APK / ELF / PDF).

thrember's ``PEFeatureExtractor`` produces a 2568-dim vector for any bytes;
on non-PE files the PE-specific submodules degrade to zero-records. The
shipped non-PE LightGBM detectors accept exactly that vector shape.

Problem-space transforms (Full DOS, overlay/padding) are **PE-specific and
not applied here** — a Full DOS mutation on an APK/PDF/ELF is nonsense.
Only the feature-space column runs, which is unrealizable by design
(mixed-scale L∞ over the 2568-d hash-bucket space); the intent of this
evaluation is to characterize the model's response, not to claim a real
bypass.

CLI:  neurinspectre audit --target ember2024-apk-gbdt --pe-sample /apk
      neurinspectre audit --target ember2024-elf-gbdt --pe-sample /elf
      neurinspectre audit --target ember2024-pdf-gbdt --pe-sample /pdf
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..attacks.feature_square import FeatureSquareAttack
from ..attacks.problem_space_pe import predict_malware
from ..evaluation.problem_space import compute_asr_query_curve
from ..malware.capa_filters import TagFilter
from ..malware.ember2024_extract import extract_ember2024_features, extractor_status
from ..malware.file_families import FileFamily, load_samples_by_family


def evaluate_ember2024_nonpe_same_sample(
    source,
    model,
    family: FileFamily,
    *,
    n_queries: int = 100,
    feature_eps: float = 1.0,
    query_budgets: Optional[Sequence[int]] = None,
    seed: int = 42,
    max_samples: Optional[int] = None,
    extractor: Optional[Callable[[bytes], Dict[str, Any]]] = None,
    tag_filter: Optional[TagFilter] = None,
    tags_by_sha256: Optional[Dict[str, Dict[str, Any]]] = None,
    filter_include_untagged: bool = False,
) -> Dict[str, Any]:
    budgets = list(query_budgets or [10, 25, 50])
    samples = load_samples_by_family(source, family)

    detect_cap = None if max_samples is None else int(max_samples)
    scan_cap = None if detect_cap is None else max(detect_cap * 25, detect_cap)
    ext_status = extractor_status() if extractor is None else {"available": True, "reasons": []}

    filter_active = bool(tag_filter and tag_filter.is_active())
    tag_lookup: Dict[str, Dict[str, Any]] = {
        str(k).lower(): dict(v) for k, v in (tags_by_sha256 or {}).items()
    }
    tags_active = bool(tag_lookup)
    n_filtered_out = 0
    n_untagged_seen = 0

    indexed: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    stopped_early = False
    for i, sample in enumerate(samples):
        if scan_cap is not None and i >= scan_cap:
            stopped_early = True
            break
        data = sample.get("bytes")
        row: Dict[str, Any] = {"index": i, "path": sample.get("path"), "family": family.name}
        if data is None:
            row.update({"kept": False, "reason": sample.get("error") or "file_missing"})
            indexed.append(row)
            continue

        if tags_active or filter_active:
            import hashlib as _hashlib

            sha = _hashlib.sha256(data).hexdigest()
            row["sha256"] = sha
            tags = tag_lookup.get(sha)
            if tags is None:
                if filter_active:
                    n_untagged_seen += 1
                    if not filter_include_untagged:
                        n_filtered_out += 1
                        row.update({"kept": False, "reason": "filter_untagged"})
                        indexed.append(row)
                        continue
            else:
                from ..malware.capa_filters import compact_operator_tags
                row["tags"] = compact_operator_tags(tags)
                row["tags_full"] = dict(tags)
                if filter_active and not tag_filter.match(tags):
                    n_filtered_out += 1
                    row.update({"kept": False, "reason": "filter_tag_mismatch"})
                    indexed.append(row)
                    continue

        extracted = (extractor or extract_ember2024_features)(data)
        row["extraction"] = {k: v for k, v in extracted.items() if k != "features"}
        feats = extracted.get("features")
        if feats is None:
            row.update({"kept": False, "reason": (extracted.get("reasons") or ["extract_failed"])[0]})
            indexed.append(row)
            continue
        feats = np.asarray(feats, dtype=np.float32).reshape(-1)
        if feats.size == 0 or not np.isfinite(feats).all():
            row.update({"kept": False, "reason": "features_nonfinite"})
            indexed.append(row)
            continue

        pred, p_mal = predict_malware(model, feats)
        kept_ok = bool(pred == 1)
        row.update({
            "clean_pred": pred, "clean_p_malware": p_mal,
            "features": feats, "kept": kept_ok,
            "reason": None if kept_ok else "not_detected_as_malware",
        })
        indexed.append(row)
        if row["kept"]:
            kept.append(row)
            if detect_cap is not None and len(kept) >= detect_cap:
                stopped_early = i + 1 < len(samples)
                break

    feature_block: Dict[str, Any] = {
        "space": "feature",
        "realizable": False,
        "n_detected": len(kept),
        "attack_success_rate": None,
        "query_curve": [],
    }
    if kept:
        x = torch.from_numpy(np.stack([r["features"] for r in kept], axis=0))
        y = torch.ones(len(kept), dtype=torch.long)
        fs = FeatureSquareAttack(
            model, eps=float(feature_eps), n_queries=int(n_queries),
            device="cpu", allow_short_budget=int(n_queries) < 1000, seed=int(seed),
        )
        _x_adv, fs_stats = fs(x, y, verbose=False)
        fs_success = [bool(v) for v in np.asarray(fs_stats["success"]).reshape(-1).tolist()]
        fs_queries = [int(v) for v in np.asarray(fs_stats["queries_used"]).reshape(-1).tolist()]
        feature_block.update({
            "attack_success_rate": float(sum(fs_success) / len(fs_success)),
            "query_curve": compute_asr_query_curve(fs_queries, fs_success, budgets),
            "queries_used": fs_queries,
            "success": fs_success,
            "epsilon": float(feature_eps),
        })

    unpaired = [r for r in indexed if not r.get("kept")]
    return {
        "kind": "ember2024_nonpe_same_sample",
        "file_family": family.name,
        "n_listed": len(samples),
        "n_scanned": len(indexed),
        "n_extracted": sum(1 for r in indexed if r.get("features") is not None),
        "n_detected_malware": len(kept),
        "n_skipped": len(unpaired),
        "stopped_early": bool(stopped_early),
        "skip_reasons": _count_reasons(unpaired),
        "extractor": ext_status,
        "tag_filter": (tag_filter.as_dict() if filter_active else None),
        "tag_filter_active": filter_active,
        "n_filtered_out_by_tag": int(n_filtered_out) if filter_active else 0,
        "n_untagged_seen": int(n_untagged_seen) if filter_active else 0,
        "filter_include_untagged": bool(filter_include_untagged) if filter_active else False,
        "same_sample": True,
        "feature_space": feature_block,
        "problem_space": {
            "space": "problem",
            "n_detected": len(kept),
            "attack_success_rate": None,
            "valid_success_rate": None,
            "note": (
                f"Problem-space transforms are PE-specific; not applied to "
                f"{family.name} files. This lane reports feature-space only."
            ),
        },
        "feature_vs_problem_space": {
            "n": len(kept),
            "same_sample": True,
            "feature_space_asr": feature_block.get("attack_success_rate"),
            "feature_space_realizable": False,
            "problem_space_asr": None,
            "problem_space_valid_asr": None,
            "note": (
                f"{family.name} same-sample: feature-space only. FeatureSquare "
                f"L-inf is unrealizable; no PE-specific Full DOS/overlay path."
            ),
        },
        "samples": [
            {k: v for k, v in r.items() if k not in {"features"}}
            for r in indexed
        ],
    }


def _count_reasons(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("reason") or "unknown")
        out[key] = out.get(key, 0) + 1
    return out
