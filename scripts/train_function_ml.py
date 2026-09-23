#!/usr/bin/env python3
"""E11 — train per-capability LightGBM classifiers on Capa supplement functions.

Streams the supplement, samples a stratified subset keyed on
``--target-labels`` (defaults to the 20 most common capabilities in
the index), extracts opcode n-gram features, splits by SHA-256, trains
one-vs-rest LightGBM per label, and writes a JSON of per-label
accuracy/precision/recall/F1/AUC to ``--output``.

Runtime is dominated by the shard scan; on 500 K sampled functions with
feature_dim=4096 and 20 labels, expect ~5-10 min.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurinspectre.malware.capa_supplement_index import load_index, summarize
from neurinspectre.malware.function_ml import (
    stratified_sample,
    train_multilabel_functionml,
)


def _default_target_labels(index_path: Path, top_k: int) -> list[str]:
    idx = load_index(index_path)
    summ = summarize(idx)
    return [name for name, _ in summ["top_capabilities"][:top_k]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement", type=Path,
                        default=Path("data/ember/ember2024/capa"))
    parser.add_argument("--index", type=Path,
                        default=Path("data/ember/ember2024/capa_supplement_index.json"))
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("results/ember2024/E11/function_ml_report.json"))
    parser.add_argument("--top-k-labels", type=int, default=20)
    parser.add_argument("--target-per-capability", type=int, default=5000)
    parser.add_argument("--negative-pool-size", type=int, default=20000)
    parser.add_argument("--feature-dim", type=int, default=4096)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-records-scan", type=int, default=None,
                        help="Cap total records read from shards (for smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.supplement.is_dir():
        raise SystemExit(f"supplement not found at {args.supplement}")
    if not args.index.is_file():
        raise SystemExit(f"index not found at {args.index}. Run: neurinspectre index-capa-supplement")

    target_labels = _default_target_labels(args.index, args.top_k_labels)
    print(f"target labels ({len(target_labels)}): {target_labels[:5]}... +{max(0, len(target_labels)-5)} more")

    print(f"sampling: quota={args.target_per_capability}/label, negatives={args.negative_pool_size} ...")
    samples, counts = stratified_sample(
        args.supplement,
        target_labels,
        target_per_capability=args.target_per_capability,
        negative_pool_size=args.negative_pool_size,
        max_records=args.max_records_scan,
    )
    print(f"kept {len(samples)} functions; per-label: {counts['per_label']}  negatives: {counts['negatives']}")

    print(f"training {len(target_labels)} binary LightGBMs (feature_dim={args.feature_dim})...")
    report = train_multilabel_functionml(
        samples, target_labels,
        feature_dim=args.feature_dim,
        n_estimators=args.n_estimators,
        seed=args.seed,
    )
    report["sampling_counts"] = counts
    report["args"] = {
        "top_k_labels": args.top_k_labels,
        "target_per_capability": args.target_per_capability,
        "negative_pool_size": args.negative_pool_size,
        "feature_dim": args.feature_dim,
        "n_estimators": args.n_estimators,
        "max_records_scan": args.max_records_scan,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output}")
    print(f"n_train={report['n_train']} n_val={report['n_val']} n_test={report['n_test']}")
    print("\nper-label (F1, AUC, support_pos):")
    for label in target_labels:
        m = report["per_label_metrics"].get(label) or {}
        if "skipped_reason" in m:
            print(f"  [skip] {label}: {m['skipped_reason']}")
            continue
        print(f"  F1={m['f1']:.3f}  AUC={m['auc']:.3f}  n+={m['support_pos']:5d}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
