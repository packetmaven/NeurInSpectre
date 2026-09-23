"""Same-sample feature-space vs problem-space evaluation on PE files.

EMBER2018 vectors have no binaries. This module is the Month 3 finding path:
extract EMBER features from PE bytes, attack those vectors (unrealizable),
attack the same bytes with Full DOS / padding, and count only PE-valid flips.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..attacks.feature_square import FeatureSquareAttack
from ..attacks.problem_space_pe import ProblemSpacePESearch, load_pe_samples, predict_malware
from ..evaluation.problem_space import compute_asr_query_curve, evaluate_pe_parse
from ..malware.capa_filters import TagFilter
from ..malware.ember_extract import extract_ember_features, extractor_status
from ..malware.pe_transforms import read_e_lfanew


def evaluate_ember_same_sample(
    pe_source,
    model,
    *,
    n_queries: int = 50,
    feature_eps: float = 1.0,
    query_budgets: Optional[Sequence[int]] = None,
    seed: int = 42,
    payload_size: int = 256,
    benign_payloads: Optional[Sequence[bytes]] = None,
    extractor: Optional[Callable[[bytes], Dict[str, Any]]] = None,
    max_samples: Optional[int] = None,
    tag_filter: Optional[TagFilter] = None,
    tags_by_sha256: Optional[Dict[str, Dict[str, Any]]] = None,
    filter_include_untagged: bool = False,
    capa_preserve: bool = False,
    capa_rules_dir=None,
    capa_preserve_mode: str = "all",
    enable_section_slack: bool = False,
    transform_set: str = "default",
    fulldos_quiet_only: bool = False,
    capa_supplement_index: Optional[Dict[str, list]] = None,
    supplement_root: Optional[Any] = None,
    best_bytes_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    budgets = list(query_budgets or [10, 25, 50])
    samples = load_pe_samples(pe_source)
    # n_examples is detected-malware PEs, not the first N files in the directory.
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
        row: Dict[str, Any] = {"index": i, "path": sample.get("path")}
        if data is None:
            row.update({"kept": False, "reason": sample.get("error") or "pe_sample_missing"})
            indexed.append(row)
            continue
        try:
            read_e_lfanew(data)
        except ValueError:
            row.update({"kept": False, "reason": "not_a_valid_pe"})
            indexed.append(row)
            continue

        # A1 + A3 — Capa tag attachment.
        # A1 (filter): applied before the expensive extraction so filtered-out
        # files consume neither thrember nor GBDT budget.
        # A3 (claim ledger): tags attach to the row whenever a sidecar is
        # loaded, whether or not a filter is active, so successful bypasses
        # carry the original file's ATT&CK/MBC/capa provenance into the
        # ledger. Full tag record is preserved; the truncated `tags` field
        # keeps back-compat with A1 report consumers.
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
        parse = evaluate_pe_parse(data)
        if parse.get("available") and parse.get("passed") is False:
            row.update(
                {
                    "kept": False,
                    "reason": "not_a_valid_pe",
                    "pe_parse": {k: v for k, v in parse.items() if k != "path"},
                }
            )
            indexed.append(row)
            continue
        extracted = extract_ember_features(data, extractor=extractor)
        row["extraction"] = {k: v for k, v in extracted.items() if k != "features"}
        feats = extracted.get("features")
        if feats is None:
            row.update({"kept": False, "reason": (extracted.get("reasons") or ["extract_failed"])[0]})
            indexed.append(row)
            continue
        feats = np.asarray(feats, dtype=np.float32).reshape(-1)
        if feats.size == 0 or not bool(np.isfinite(feats).all()):
            row.update({"kept": False, "reason": "ember_features_nonfinite"})
            indexed.append(row)
            continue
        pred, p_mal = predict_malware(model, feats)
        kept_ok = bool(pred == 1)
        row.update(
            {
                "clean_pred": pred,
                "clean_p_malware": p_mal,
                "features": feats,
                "bytes": data,
                "kept": kept_ok,
                "reason": None if kept_ok else "not_detected_as_malware",
                "pe_parse": {k: v for k, v in parse.items() if k != "path"},
            }
        )
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
    problem_block: Dict[str, Any] = {
        "space": "problem",
        "n_detected": len(kept),
        "attack_success_rate": None,
        "valid_success_rate": None,
        "query_curve": [],
        "secml_gamma_available": False,
    }
    # D9 — always define so the return can reference it whether or not the
    # problem-space branch runs.
    best_bytes_manifest: List[Dict[str, Any]] = []
    best_bytes_rejected: List[Dict[str, Any]] = []

    if kept:
        x = torch.from_numpy(np.stack([r["features"] for r in kept], axis=0))
        y = torch.ones(len(kept), dtype=torch.long)
        fs = FeatureSquareAttack(
            model,
            eps=float(feature_eps),
            n_queries=int(n_queries),
            device="cpu",
            allow_short_budget=int(n_queries) < 1000,
            seed=int(seed),
        )
        _x_adv, fs_stats = fs(x, y, verbose=False)
        fs_success = [bool(v) for v in np.asarray(fs_stats["success"]).reshape(-1).tolist()]
        fs_queries = [int(v) for v in np.asarray(fs_stats["queries_used"]).reshape(-1).tolist()]
        feature_block.update(
            {
                "attack_success_rate": float(sum(fs_success) / len(fs_success)),
                "query_curve": compute_asr_query_curve(fs_queries, fs_success, budgets),
                "queries_used": fs_queries,
                "success": fs_success,
                "epsilon": float(feature_eps),
            }
        )
        for row, ok, q in zip(kept, fs_success, fs_queries):
            row["feature_success"] = ok
            row["feature_queries"] = q

        # C7 — if a supplement index is loaded, precompute per-file byte
        # payloads by carving from the supplement's raw disassembly bytes.
        # For now we do NOT read the raw bytes from the 23.8 GB shards at
        # attack time (that would defeat the point of the index); instead
        # supplement_payloads is passed in whole from the caller if desired.
        supplement_payloads_by_sha = capa_supplement_index or {}
        search = ProblemSpacePESearch(
            model,
            n_queries=int(n_queries),
            payload_size=int(payload_size),
            seed=int(seed),
            benign_payloads=benign_payloads,
            extractor=extractor,
            capa_preserve=bool(capa_preserve),
            capa_rules_dir=capa_rules_dir,
            capa_preserve_mode=str(capa_preserve_mode or "all"),
            enable_section_slack=bool(enable_section_slack),
            transform_set=str(transform_set or "default"),
            fulldos_quiet_only=bool(fulldos_quiet_only),
            supplement_payloads=None,
        )
        ps_success = []
        ps_valid_success = []
        ps_queries = []
        ps_rows = []
        # D9 — persist best-of-search mutated bytes so downstream tooling
        # (transferability, ledger, deep inspection) can re-score them.
        best_bytes_root = None
        if best_bytes_dir is not None:
            from pathlib import Path as _P
            best_bytes_root = _P(best_bytes_dir)
            best_bytes_root.mkdir(parents=True, exist_ok=True)
        for row in kept:
            result = search.run_bytes(row["bytes"], y=1, rng=rng)
            best_ref = None
            if best_bytes_root is not None:
                bb = result.get("best_bytes")
                if isinstance(bb, (bytes, bytearray)) and len(bb) > 0:
                    import hashlib as _h
                    from .transferability import manifest_entry_problems
                    sha_orig = _h.sha256(row["bytes"]).hexdigest()
                    sha_mut = _h.sha256(bytes(bb)).hexdigest()
                    out_path = best_bytes_root / f"{sha_orig}.mutated.bin"
                    entry = {
                        "sample_path": row.get("path"),
                        "chosen_attack": result.get("chosen_attack"),
                        "best_p_malware": result.get("best_p_malware"),
                        "clean_p_malware": result.get("clean_p_malware"),
                        "sha256_original": sha_orig,
                        "sha256_mutated": sha_mut,
                        "path": str(out_path),
                        "size": len(bb),
                        "identical_to_original": sha_orig == sha_mut,
                    }
                    problems = manifest_entry_problems(entry)
                    if problems:
                        best_bytes_rejected.append({
                            "sha256_original": sha_orig,
                            "problems": problems,
                        })
                    else:
                        out_path.write_bytes(bytes(bb))
                        best_ref = {
                            "sha256_original": sha_orig,
                            "sha256_mutated": sha_mut,
                            "path": str(out_path),
                            "size": len(bb),
                            "identical_to_original": sha_orig == sha_mut,
                        }
                        best_bytes_manifest.append(entry)
            row["problem"] = {k: v for k, v in result.items() if k != "best_bytes"}
            if best_ref is not None:
                row["problem"]["best_bytes_ref"] = best_ref
            ok = bool(result.get("success"))
            valid = bool(result.get("realizable") and ok)
            ps_success.append(ok)
            ps_valid_success.append(valid)
            ps_queries.append(int(result.get("queries_used") or n_queries))
            ps_rows.append(result)
        n = len(kept)
        problem_block.update(
            {
                "attack_success_rate": float(sum(ps_success) / n),
                "valid_success_rate": float(sum(ps_valid_success) / n),
                "query_curve": compute_asr_query_curve(ps_queries, ps_valid_success, budgets),
                "queries_used": ps_queries,
                "success": ps_success,
                "valid_success": ps_valid_success,
                "transforms": [r.get("chosen_attack") for r in ps_rows],
                "gamma_padding": bool(benign_payloads),
            }
        )

    unpaired = [r for r in indexed if not r.get("kept")]
    return {
        "kind": "ember_same_sample",
        "n_listed": len(samples),
        "n_scanned": len(indexed),
        "n_pe_files": len(indexed),
        "n_extracted": sum(1 for r in indexed if r.get("features") is not None),
        "n_detected_malware": len(kept),
        "n_skipped": len(unpaired),
        "tag_filter": (tag_filter.as_dict() if filter_active else None),
        "tag_filter_active": filter_active,
        "n_filtered_out_by_tag": int(n_filtered_out) if filter_active else 0,
        "n_untagged_seen": int(n_untagged_seen) if filter_active else 0,
        "filter_include_untagged": bool(filter_include_untagged) if filter_active else False,
        "stopped_early": bool(stopped_early),
        "skip_reasons": _count_reasons(unpaired),
        "extractor": ext_status,
        "threat_model": "malware_evasion",
        "same_sample": True,
        "feature_space": feature_block,
        "problem_space": problem_block,
        "best_bytes_manifest": best_bytes_manifest,
        "best_bytes_rejected": best_bytes_rejected,
        "best_bytes_dir": str(best_bytes_dir) if best_bytes_dir is not None else None,
        "feature_vs_problem_space": {
            "n": len(kept),
            "same_sample": True,
            "feature_space_asr": feature_block.get("attack_success_rate"),
            "feature_space_realizable": False,
            "problem_space_asr": problem_block.get("attack_success_rate"),
            "problem_space_valid_asr": problem_block.get("valid_success_rate"),
            "note": (
                "Both columns use the same PE files that the GBDT detected as malware. "
                "Feature-space L-inf is not a valid PE. Problem-space counts only "
                "Full DOS / padding mutations that still parse with EP and e_lfanew intact."
            ),
        },
        "samples": [
            {k: v for k, v in r.items() if k not in {"features", "bytes"}}
            for r in indexed
        ],
    }


def _count_reasons(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("reason") or "unknown")
        out[key] = out.get(key, 0) + 1
    return out
