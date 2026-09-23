#!/usr/bin/env python3
"""Linux same-sample EMBER run with inventory + checkpointed audit.

Do not execute PE files. Read bytes, extract EMBER v2 features, score the
official LightGBM, then attack feature vectors and PE bytes separately.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _load_repo():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _ensure_pkg(name: str, path: Path) -> None:
    """Register a package without executing a heavy ``__init__.py``."""
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    pkg.__package__ = name
    pkg.__file__ = str(path / "__init__.py")
    sys.modules[name] = pkg


def _load_module(fullname: str, path: Path):
    if fullname in sys.modules:
        return sys.modules[fullname]
    spec = importlib.util.spec_from_file_location(fullname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {fullname} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_ember_stack(root: Path):
    """Import GBDT + attacks without torchvision / AutoAttack package inits."""
    _ensure_pkg("neurinspectre.models", root / "neurinspectre" / "models")
    _ensure_pkg("neurinspectre.attacks", root / "neurinspectre" / "attacks")
    ember_gbdt = _load_module(
        "neurinspectre.models.ember_gbdt",
        root / "neurinspectre" / "models" / "ember_gbdt.py",
    )
    _load_module("neurinspectre.attacks.base", root / "neurinspectre" / "attacks" / "base.py")
    return ember_gbdt


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def inventory_corpus(pe_dir: Path, model, extractor_status) -> Dict[str, Any]:
    from neurinspectre.evaluation.problem_space import evaluate_pe_parse
    from neurinspectre.malware.ember_extract import extract_ember_features
    from neurinspectre.malware.pe_transforms import read_e_lfanew

    files = sorted(p for p in pe_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    rows: List[Dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    t0 = time.time()
    print(f"[linux-inventory] scoring {len(files)} files on {extractor_status.get('platform')} "
          f"lief={extractor_status.get('lief_version')}", flush=True)
    for i, path in enumerate(files, 1):
        t1 = time.time()
        row: Dict[str, Any] = {"path": str(path), "name": path.name, "size": path.stat().st_size}
        try:
            data = path.read_bytes()
            try:
                read_e_lfanew(data)
            except ValueError:
                row.update({"kept": False, "reason": "not_a_valid_pe"})
            else:
                parse = evaluate_pe_parse(data)
                row["pe_parse_passed"] = bool(parse.get("passed"))
                if parse.get("available") and parse.get("passed") is False:
                    row.update({"kept": False, "reason": "not_a_valid_pe"})
                else:
                    extracted = extract_ember_features(data)
                    feats = extracted.get("features")
                    row["extract_available"] = bool(extracted.get("available"))
                    row["extract_reasons"] = list(extracted.get("reasons") or [])
                    if feats is None:
                        row.update({"kept": False, "reason": (extracted.get("reasons") or ["extract_failed"])[0]})
                    else:
                        feats = np.asarray(feats, dtype=np.float32).reshape(-1)
                        if feats.size == 0 or not bool(np.isfinite(feats).all()):
                            row.update({"kept": False, "reason": "ember_features_nonfinite"})
                        else:
                            raw = float(model.booster.predict(feats.reshape(1, -1))[0])
                            pred = int(raw > 0.5)
                            row.update(
                                {
                                    "kept": pred == 1,
                                    "reason": None if pred == 1 else "not_detected_as_malware",
                                    "p_malware": raw,
                                    "pred": pred,
                                    "feat_dim": int(feats.size),
                                }
                            )
        except Exception as exc:  # noqa: BLE001 — inventory must not die on one file
            row.update({"kept": False, "reason": f"exception:{type(exc).__name__}", "error": str(exc)})
        row["extract_s"] = round(time.time() - t1, 3)
        reasons[row.get("reason") or "detected_malware"] += 1
        rows.append(row)
        if i % 10 == 0 or i == len(files):
            n_det = sum(1 for r in rows if r.get("kept"))
            print(
                f"[linux-inventory] {i}/{len(files)} detected={n_det} "
                f"last={path.name} {row.get('reason') or 'detected'} "
                f"p={row.get('p_malware')} {row['extract_s']}s",
                flush=True,
            )
    p_mals = [r["p_malware"] for r in rows if "p_malware" in r]
    summary = {
        "n_listed": len(files),
        "n_scored": len(p_mals),
        "n_detected_malware": sum(1 for r in rows if r.get("kept")),
        "reasons": dict(reasons),
        "p_malware_min": min(p_mals) if p_mals else None,
        "p_malware_max": max(p_mals) if p_mals else None,
        "p_malware_median": float(np.median(p_mals)) if p_mals else None,
        "p_ge_0_9": sum(1 for p in p_mals if p >= 0.9),
        "p_ge_0_5": sum(1 for p in p_mals if p >= 0.5),
        "elapsed_s": round(time.time() - t0, 2),
        "extractor": extractor_status,
        "official_reproduction": bool(extractor_status.get("official_reproduction")),
        "quote_as_ember2018": bool(extractor_status.get("official_reproduction")),
    }
    return {"summary": summary, "samples": rows}


def _feature_attack(model, feats: np.ndarray, n_queries: int, seed: int) -> Dict[str, Any]:
    from neurinspectre.attacks.feature_square import FeatureSquareAttack

    x = torch.from_numpy(np.asarray(feats, dtype=np.float32).reshape(1, -1))
    y = torch.ones(1, dtype=torch.long)
    fs = FeatureSquareAttack(
        model,
        eps=1.0,
        n_queries=int(n_queries),
        device="cpu",
        allow_short_budget=int(n_queries) < 1000,
        seed=int(seed),
    )
    _x_adv, stats = fs(x, y, verbose=False)
    return {
        "success": bool(np.asarray(stats["success"]).reshape(-1)[0]),
        "queries_used": int(np.asarray(stats["queries_used"]).reshape(-1)[0]),
        "realizable": False,
        "space": "feature",
        "epsilon": 1.0,
    }


def _problem_attack(model, pe_bytes: bytes, n_queries: int, payload_size: int, seed: int) -> Dict[str, Any]:
    from neurinspectre.attacks.problem_space_pe import ProblemSpacePESearch

    search = ProblemSpacePESearch(
        model,
        n_queries=int(n_queries),
        payload_size=int(payload_size),
        seed=int(seed),
    )
    result = search.run_bytes(pe_bytes, y=1)
    return {k: v for k, v in result.items() if k != "best_bytes"}


def _curve(queries: List[int], success: List[bool], budgets: List[int]) -> List[Dict[str, Any]]:
    from neurinspectre.evaluation.problem_space import compute_asr_query_curve

    return compute_asr_query_curve(queries, success, budgets)


def _load_prior_features(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    if isinstance(payload.get("done"), dict):
        rows = list(payload["done"].values())
    elif isinstance(payload.get("same_sample_detail"), dict):
        rows = list(payload["same_sample_detail"].get("samples") or [])
    out: Dict[str, Any] = {}
    for row in rows:
        name = row.get("name")
        feat = row.get("feature")
        if name and isinstance(feat, dict):
            out[str(name)] = feat
    return out


def audit_detected(
    detected_rows: List[Dict[str, Any]],
    model,
    *,
    n_queries: int,
    budgets: List[int],
    payload_size: int,
    seed: int,
    checkpoint: Path,
    extractor_status: Dict[str, Any],
    n_listed: int,
    problem_only: bool = False,
    prior_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from neurinspectre.malware.ember_extract import extract_ember_features

    prior_features = prior_features or {}
    if checkpoint.is_file():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
    else:
        state = {"done": {}, "started_s": time.time(), "n_queries": int(n_queries)}
    done: Dict[str, Any] = state.setdefault("done", {})
    state["n_queries"] = int(n_queries)

    for i, meta in enumerate(detected_rows):
        name = meta["name"]
        if name in done:
            print(f"[linux-audit] resume skip {i+1}/{len(detected_rows)} {name}", flush=True)
            continue
        t1 = time.time()
        path = Path(meta["path"])
        data = path.read_bytes()
        if problem_only:
            feature = dict(prior_features.get(name) or {})
            if not feature:
                feature = {"success": True, "queries_used": None, "realizable": False, "space": "feature", "reused": False, "missing_prior": True}
            else:
                feature = dict(feature)
                feature["reused"] = True
            print(
                f"[linux-audit] {i+1}/{len(detected_rows)} problem-only {name} "
                f"p={meta.get('p_malware')} reused_feature={feature.get('success')}",
                flush=True,
            )
        else:
            extracted = extract_ember_features(data)
            feats = extracted.get("features")
            if feats is None:
                done[name] = {"name": name, "path": str(path), "kept": False, "reason": "extract_failed_on_audit"}
                _json_dump(checkpoint, state)
                continue
            feats = np.asarray(feats, dtype=np.float32).reshape(-1)
            print(f"[linux-audit] {i+1}/{len(detected_rows)} feature {name} p={meta.get('p_malware')}", flush=True)
            feature = _feature_attack(model, feats, n_queries, seed + i)
            print(
                f"[linux-audit] {i+1}/{len(detected_rows)} problem {name} "
                f"feature_ok={feature['success']} q={feature['queries_used']}",
                flush=True,
            )
        problem = _problem_attack(model, data, n_queries, payload_size, seed + i)
        done[name] = {
            "name": name,
            "path": str(path),
            "size": meta.get("size"),
            "clean_p_malware": meta.get("p_malware"),
            "clean_pred": 1,
            "kept": True,
            "feature": feature,
            "problem": problem,
            "elapsed_s": round(time.time() - t1, 3),
        }
        _json_dump(checkpoint, state)
        print(
            f"[linux-audit] {i+1}/{len(detected_rows)} done {name} "
            f"feat={feature.get('success')} prob={problem.get('success')} "
            f"valid={bool(problem.get('realizable') and problem.get('success'))} "
            f"{done[name]['elapsed_s']}s",
            flush=True,
        )

    kept = [done[m["name"]] for m in detected_rows if m["name"] in done and done[m["name"]].get("kept")]
    fs_ok = [bool((r.get("feature") or {}).get("success")) for r in kept]
    fs_q = [int((r.get("feature") or {}).get("queries_used") or n_queries) for r in kept]
    ps_ok = [bool(r["problem"].get("success")) for r in kept]
    ps_valid = [bool(r["problem"].get("success") and r["problem"].get("realizable")) for r in kept]
    ps_q = [int(r["problem"].get("queries_used") or n_queries) for r in kept]
    n = len(kept)
    feature_block = {
        "space": "feature",
        "realizable": False,
        "n_detected": n,
        "attack_success_rate": (sum(fs_ok) / n) if n else None,
        "query_curve": _curve(fs_q, fs_ok, budgets) if n else [],
        "queries_used": fs_q,
        "success": fs_ok,
        "epsilon": 1.0,
    }
    problem_block = {
        "space": "problem",
        "n_detected": n,
        "attack_success_rate": (sum(ps_ok) / n) if n else None,
        "valid_success_rate": (sum(ps_valid) / n) if n else None,
        "query_curve": _curve(ps_q, ps_valid, budgets) if n else [],
        "queries_used": ps_q,
        "success": ps_ok,
        "valid_success": ps_valid,
        "transforms": [r["problem"].get("chosen_attack") for r in kept],
        "gamma_padding": False,
        "secml_gamma_available": False,
    }
    return {
        "kind": "neurinspectre_audit",
        "target": "ember-gbdt",
        "dataset": "ember",
        "defense": "md_ember2018_gbdt",
        "same_sample": True,
        "smoke": False,
        "n_examples": n,
        "query_budgets": list(budgets),
        "official_reproduction": bool(extractor_status.get("official_reproduction")),
        "quote_as_ember2018": bool(extractor_status.get("official_reproduction")),
        "chosen_attack": "problem_space",
        "attacks": {"feature_square": feature_block},
        "problem_space": problem_block,
        "feature_vs_problem_space": {
            "n": n,
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
        "same_sample_detail": {
            "extractor": extractor_status,
            "n_listed": n_listed,
            "n_scanned": n_listed,
            "n_extracted": n_listed,
            "n_detected_malware": n,
            "n_skipped": 0,
            "stopped_early": False,
            "skip_reasons": {},
            "samples": kept,
        },
        "notes": [
            "Official Elastic EMBER 2018 LightGBM (ember_model_2018.txt), not the Table 2 MLP.",
            "Linux extraction. quote_as_ember2018 is true only if official_reproduction is true.",
            "Feature-space FeatureSquare is unrealizable. Problem-space is Full DOS + padding.",
            "Problem-only long-budget runs reuse FeatureSquare results from the 500-query table.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pe-dir", required=True)
    parser.add_argument("--model-path", default="data/ember/ember2018/ember_model_2018.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--query-budgets", default="100,500")
    parser.add_argument("--payload-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-detected", type=int, default=0, help="0 = all Linux-detected PEs")
    parser.add_argument("--reuse-inventory", type=str, default=None, help="Existing linux_pe_inventory.json")
    parser.add_argument("--prior-checkpoint", type=str, default=None, help="Reuse FeatureSquare rows from a prior run")
    parser.add_argument("--checkpoint-name", type=str, default="linux_audit_checkpoint.json")
    parser.add_argument("--problem-only", action="store_true", help="Do not rerun FeatureSquare")
    args = parser.parse_args()

    root = _load_repo()
    from neurinspectre.malware.ember_extract import extractor_status

    ember_gbdt = _load_ember_stack(root)
    EmberGBDT = ember_gbdt.EmberGBDT

    pe_dir = Path(args.pe_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise SystemExit(f"EMBER model not found: {model_path}")
    if not pe_dir.is_dir():
        raise SystemExit(f"PE directory not found: {pe_dir}")

    status = extractor_status()
    print(json.dumps({"extractor": status, "cwd": str(Path.cwd()), "repo": str(root)}, indent=2), flush=True)
    if not status.get("available"):
        raise SystemExit(f"EMBER extractor unavailable: {status}")

    model = EmberGBDT.from_file(model_path)
    inv_path = Path(args.reuse_inventory) if args.reuse_inventory else (out_dir / "linux_pe_inventory.json")
    if args.reuse_inventory:
        inventory = json.loads(inv_path.read_text(encoding="utf-8"))
        print(f"[linux-audit] reused inventory {inv_path}", flush=True)
    else:
        inventory = inventory_corpus(pe_dir, model, status)
        _json_dump(out_dir / "linux_pe_inventory.json", inventory)
    print(json.dumps(inventory["summary"], indent=2), flush=True)

    detected = [r for r in inventory["samples"] if r.get("kept")]
    if args.max_detected:
        detected = detected[: int(args.max_detected)]
    if not detected:
        raise SystemExit("Linux GBDT detected no malware PEs; refusing to invent a table.")

    budgets = [int(x) for x in str(args.query_budgets).split(",") if x.strip()]
    prior = _load_prior_features(Path(args.prior_checkpoint) if args.prior_checkpoint else None)
    report = audit_detected(
        detected,
        model,
        n_queries=int(args.n_queries),
        budgets=budgets,
        payload_size=int(args.payload_size),
        seed=int(args.seed),
        checkpoint=out_dir / str(args.checkpoint_name),
        extractor_status=status,
        n_listed=int(inventory["summary"]["n_listed"]),
        problem_only=bool(args.problem_only),
        prior_features=prior,
    )
    _json_dump(out_dir / "audit_report.json", report)
    print(
        f"[linux-audit] wrote {out_dir / 'audit_report.json'} "
        f"n={report['n_examples']} "
        f"feature_asr={report['feature_vs_problem_space']['feature_space_asr']} "
        f"problem_valid_asr={report['feature_vs_problem_space']['problem_space_valid_asr']} "
        f"official_reproduction={report['official_reproduction']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
