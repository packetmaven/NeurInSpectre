#!/usr/bin/env python3
"""
Generate a deterministic ConcreTizer detector exerciser dataset.

Why this exists
---------------
`neurinspectre adversarial-detect --detector-type concretizer` expects a numeric array
interpretable as a (T, D) sequence of model outputs (and optionally query patterns via
--reference-path). Users often want an end-to-end artifact that reliably exercises the
ConcreTizer scoring path at high confidence.

This generator produces a **synthetic, deterministic** `.npy` file designed to trigger the
current ConcreTizer heuristic model-inversion signal detector:
  - High |corr| structure across output dimensions
  - Strong periodicity / “grid-like” autocorrelation structure
  - Highly systematic (mostly self-similar) query vectors
  - Low discrete entropy (few distinct values) → higher reconstruction-confidence proxy

It is NOT a real-world dataset and it does NOT claim to reproduce a specific paper.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ConcretizerDataInfo:
    out_path: str
    timesteps: int
    features: int
    dip_every: int
    baseline_value: float
    dip_value: float
    expected: Dict[str, Any]
    notes: list[str]


def _clean_array_mean_std(arr: np.ndarray) -> np.ndarray:
    """Match the default `adversarial-detect` preprocessing (mean/std scaling per feature)."""
    x = np.asarray(arr, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        x = x.reshape(int(x.shape[0]), -1)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x - mu) / sd


def generate_concretizer_attack_data(
    out: str = "attack_data/concretizer_attack_data.npy",
    *,
    timesteps: int = 300,
    features: int = 64,
    dip_every: int = 100,
    baseline_value: float = 1.0,
    dip_value: float = 0.0,
    threshold: float = 0.9,
) -> Dict[str, Any]:
    """Write a ConcreTizer detector exerciser `.npy` file and return metadata."""
    T = int(timesteps)
    D = int(features)
    if T < 4:
        raise ValueError("timesteps must be >= 4")
    if D < 4:
        raise ValueError("features must be >= 4")
    if dip_every <= 0:
        raise ValueError("dip_every must be > 0")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Feature pattern: alternating signs ensures each row has non-zero variance across features,
    # avoiding degenerate row-correlation in query systematicity.
    s = np.ones(D, dtype=np.float32)
    s[1::2] = -1.0

    # Row amplitude: mostly constant with rare, periodic dips.
    # This keeps most rows strongly self-similar, but ensures non-zero per-feature variance across time
    # so mean/std scaling does not collapse the data to zeros.
    a = np.full((T,), float(baseline_value), dtype=np.float32)
    for t in range(0, T, int(dip_every)):
        a[t] = float(dip_value)

    X = a[:, None] * s[None, :]  # (T, D)
    np.save(out_path, X.astype(np.float32, copy=False))

    # Verify expected detector metrics under the same preprocessing used by adversarial-detect.
    # We intentionally run ConcreTizer on (model_outputs, query_patterns) = (X_clean, X_clean),
    # matching the default CLI behavior when --reference-path is not provided.
    X_clean = _clean_array_mean_std(X)
    from ..security.adversarial_detection import ConcreTizerDetector

    det = ConcreTizerDetector(voxel_resolution=32, inversion_threshold=float(threshold))
    res = det.detect_model_inversion(X_clean, X_clean)

    expected = {
        "threshold": float(threshold),
        "is_inversion_attack": bool(res.get("is_inversion_attack", False)),
        "inversion_score": float(res.get("inversion_score", 0.0)),
        "voxel_occupancy": float(res.get("voxel_occupancy", 0.0)),
        "query_systematicity": float(res.get("query_systematicity", 0.0)),
        "reconstruction_confidence": float(res.get("reconstruction_confidence", 0.0)),
        "information_leakage": float(res.get("information_leakage", 0.0)),
        "attack_complexity": res.get("attack_complexity"),
    }

    info = ConcretizerDataInfo(
        out_path=str(out_path),
        timesteps=T,
        features=D,
        dip_every=int(dip_every),
        baseline_value=float(baseline_value),
        dip_value=float(dip_value),
        expected=expected,
        notes=[
            "Synthetic/deterministic detector exerciser. Not a real-world dataset.",
            "Designed to trigger ConcreTizer via high cross-dim correlation, strong periodicity, and systematic row similarity.",
        ],
    )
    return asdict(info)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Generate ConcreTizer detector exerciser .npy data")
    p.add_argument("--out", default="attack_data/concretizer_attack_data.npy", help="Output .npy path")
    p.add_argument("--timesteps", type=int, default=300, help="Number of timesteps/queries (T)")
    p.add_argument("--features", type=int, default=64, help="Number of features (D)")
    p.add_argument("--dip-every", type=int, default=100, help="Set every Nth row amplitude to dip_value (default: 100)")
    p.add_argument("--baseline", type=float, default=1.0, help="Baseline amplitude value (default: 1.0)")
    p.add_argument("--dip-value", type=float, default=0.0, help="Dip amplitude value (default: 0.0)")
    p.add_argument("--threshold", type=float, default=0.9, help="Detector threshold you plan to use (default: 0.9)")
    p.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        help="Optional JSON metadata output. Use `--json` to print, or `--json <path>` to write.",
    )
    args = p.parse_args(argv)

    info = generate_concretizer_attack_data(
        out=str(args.out),
        timesteps=int(args.timesteps),
        features=int(args.features),
        dip_every=int(args.dip_every),
        baseline_value=float(args.baseline),
        dip_value=float(args.dip_value),
        threshold=float(args.threshold),
    )

    j = getattr(args, "json", None)
    if j is None:
        print(str(info.get("out_path", args.out)))
    elif str(j).strip() == "-":
        print(json.dumps(info, indent=2))
        print(str(info.get("out_path", args.out)))
    else:
        jp = Path(str(j))
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(info, indent=2) + "\n")
        print(str(jp))
        print(str(info.get("out_path", args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


