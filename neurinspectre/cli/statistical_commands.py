#!/usr/bin/env python3
"""
NeurInSpectre Statistical Analysis CLI Commands

Wires the `neurinspectre.statistical` subpackage into the main `neurinspectre` CLI:
- Drift detection (Hotelling T², energy-distance two-sample test, Bayesian CP)
- Enhanced Z-score analysis (univariate/multivariate/robust/temporal)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _safe_makedirs(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _save_drift_plot(
    ref: np.ndarray,
    cur: np.ndarray,
    *,
    consensus: Any,
    per_method: dict[str, Any],
    out_path: str,
    feature_index: int = 0,
) -> None:
    """Create a concise drift plot (distribution + per-method scores) and save as PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    try:
        from scipy.stats import gaussian_kde
        _HAS_KDE = True
    except Exception:
        _HAS_KDE = False

    ref = np.asarray(ref, dtype=np.float64)
    cur = np.asarray(cur, dtype=np.float64)
    p = int(min(ref.shape[1], cur.shape[1])) if ref.ndim == 2 and cur.ndim == 2 else 1
    fi = int(np.clip(int(feature_index), 0, max(0, p - 1)))

    x_ref = ref[:, fi] if ref.ndim == 2 else ref.reshape(-1)
    x_cur = cur[:, fi] if cur.ndim == 2 else cur.reshape(-1)
    x_ref = x_ref[np.isfinite(x_ref)]
    x_cur = x_cur[np.isfinite(x_cur)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel 1: distribution drift for one feature
    ax = axes[0]
    if x_ref.size and x_cur.size:
        lo = float(min(np.min(x_ref), np.min(x_cur)))
        hi = float(max(np.max(x_ref), np.max(x_cur)))
        pad = 0.05 * (hi - lo + 1e-12)
        xs = np.linspace(lo - pad, hi + pad, 250)

        if _HAS_KDE and x_ref.size > 10 and x_cur.size > 10:
            try:
                kde_ref = gaussian_kde(x_ref)
                kde_cur = gaussian_kde(x_cur)
                ax.plot(xs, kde_ref(xs), label="Reference", lw=2)
                ax.plot(xs, kde_cur(xs), label="Current", lw=2, ls="--")
            except Exception:
                ax.hist(x_ref, bins=40, alpha=0.55, density=True, label="Reference")
                ax.hist(x_cur, bins=40, alpha=0.55, density=True, label="Current")
        else:
            ax.hist(x_ref, bins=40, alpha=0.55, density=True, label="Reference")
            ax.hist(x_cur, bins=40, alpha=0.55, density=True, label="Current")

    ax.set_title(f"Feature {fi} Distribution Drift")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(loc="best", frameon=True)

    # Panel 2: per-method drift scores
    ax2 = axes[1]
    method_names = list(per_method.keys())
    scores = []
    pvals = []
    for m in method_names:
        r = per_method[m]
        scores.append(float(getattr(r, "drift_score", 0.0)))
        pvals.append(float(getattr(r, "p_value", 1.0)))
    ax2.bar(method_names, scores, alpha=0.85)
    ax2.set_title("Per-Method Drift Scores")
    ax2.set_ylabel("Drift score")
    ax2.tick_params(axis="x", rotation=25)

    # Annotate p-values
    for i, (s, pv) in enumerate(zip(scores, pvals)):
        ax2.text(i, s, f"p={pv:.3g}", ha="center", va="bottom", fontsize=9)

    # Suptitle
    detected = bool(getattr(consensus, "drift_detected", False))
    score = float(getattr(consensus, "drift_score", 0.0))
    pval = float(getattr(consensus, "p_value", 1.0))
    fig.suptitle(
        f"NeurInSpectre Drift Detect — {'DRIFT DETECTED' if detected else 'No drift'} "
        f"(score={score:.3f}, p={pval:.3g})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    outp = Path(out_path)
    _safe_makedirs(outp)
    fig.savefig(outp, dpi=200)
    plt.close(fig)


def _save_zscore_plot(
    X: np.ndarray,
    res: Any,
    *,
    out_path: str,
    method: str,
    robust_threshold: float,
    confidence_level: float,
) -> None:
    """Create a z-score plot (score timeline + feature contributions) and save as PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=np.float64)
    n = int(X.shape[0]) if X.ndim >= 1 else 0

    method = str(method or "multivariate").lower()
    series = None
    label = method
    if hasattr(res, "multivariate_zscores") and method in ("multivariate", "hybrid"):
        series = np.asarray(res.multivariate_zscores, dtype=np.float64).reshape(-1)
        label = "multivariate_z"
    elif hasattr(res, "robust_zscores") and method == "robust":
        rz = np.asarray(res.robust_zscores, dtype=np.float64)
        series = np.mean(rz, axis=1) if rz.ndim == 2 else rz.reshape(-1)
        label = "robust_z (mean abs)"
    elif hasattr(res, "temporal_zscores") and method == "temporal":
        tz = np.asarray(res.temporal_zscores, dtype=np.float64)
        series = np.mean(tz, axis=1) if tz.ndim == 2 else tz.reshape(-1)
        label = "temporal_z (mean abs)"
    elif hasattr(res, "univariate_zscores") and method == "univariate":
        uz = np.asarray(res.univariate_zscores, dtype=np.float64)
        series = np.mean(np.abs(uz), axis=1) if uz.ndim == 2 else np.abs(uz.reshape(-1))
        label = "univariate_z (mean abs)"

    if series is None:
        series = np.zeros((n,), dtype=np.float64)

    flags = np.asarray(getattr(res, "anomaly_flags", np.zeros((series.size,), dtype=bool))).astype(bool).reshape(-1)
    if flags.size != series.size:
        flags = np.resize(flags, series.shape).astype(bool)

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.8), sharex=True)
    ax = axes[0]
    ax.plot(series, lw=1.6, color="#1f77b4")
    ax.set_ylabel("Score")
    ax.set_title(f"{label} over samples")

    # Thresholds: the CLI uses robust_threshold for the hybrid detector.
    ax.axhline(float(robust_threshold), color="#d62728", ls="--", lw=1.8, label=f"threshold={robust_threshold:g}")
    try:
        from scipy.stats import norm

        z_crit = float(norm.ppf(1.0 - (1.0 - float(confidence_level)) / 2.0))
        ax.axhline(z_crit, color="#ff7f0e", ls=":", lw=1.6, label=f"z_crit={z_crit:.2f}")
    except Exception:
        pass

    # Mark anomalies
    idx = np.where(flags)[0]
    if idx.size:
        ax.scatter(idx, series[idx], color="#d62728", s=18, zorder=3, label="flagged")
    ax.legend(loc="upper right", frameon=True)

    ax2 = axes[1]
    contrib = np.asarray(getattr(res, "feature_contributions", np.array([], dtype=np.float64)), dtype=np.float64).reshape(-1)
    if contrib.size:
        ax2.bar(np.arange(contrib.size), contrib, color="#2ca02c", alpha=0.85)
        ax2.set_ylabel("Contribution")
        ax2.set_title("Feature contributions (mean abs univariate Z)")
    else:
        ax2.text(0.5, 0.5, "No feature contributions available", ha="center", va="center")
        ax2.set_axis_off()

    ax2.set_xlabel("Sample index" if contrib.size == 0 else "Feature index")

    n_anom = int(np.sum(flags)) if flags.size else 0
    fig.suptitle(
        f"NeurInSpectre Z-Score Analysis — anomalies={n_anom}/{int(series.size)}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    outp = Path(out_path)
    _safe_makedirs(outp)
    fig.savefig(outp, dpi=200)
    plt.close(fig)


def _load_array(path: str) -> np.ndarray:
    """Load a .npy/.npz array-like object and sanitize non-finite numeric values."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() == ".npz":
        npz = np.load(str(p), allow_pickle=True)
        for k in (
            "data",
            "X",
            "x",
            "A",
            "arr",
            "activations",
            "series",
            "reference",
            "current",
            "benign",
            "attack",
        ):
            if k in npz:
                arr = np.asarray(npz[k])
                break
        else:
            if len(npz.files) == 0:
                raise ValueError(f"Empty .npz: {p}")
            arr = np.asarray(npz[npz.files[0]])
    else:
        obj = np.load(str(p), allow_pickle=True)
        if getattr(obj, "dtype", None) is object and getattr(obj, "shape", ()) == ():
            obj = obj.item()
        if isinstance(obj, dict):
            for k in (
                "data",
                "X",
                "x",
                "A",
                "arr",
                "activations",
                "series",
                "reference",
                "current",
                "benign",
                "attack",
            ):
                if k in obj:
                    obj = obj[k]
                    break
            else:
                obj = next(iter(obj.values()))
        arr = np.asarray(obj)

    if np.issubdtype(arr.dtype, np.number):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _to_2d_samples(X: np.ndarray) -> np.ndarray:
    """Coerce input into (n_samples, n_features)."""
    X = np.asarray(X)
    if X.ndim == 0:
        return X.reshape(1, 1).astype(np.float64, copy=False)
    if X.ndim == 1:
        return X.reshape(-1, 1).astype(np.float64, copy=False)
    if X.ndim == 2:
        return X.astype(np.float64, copy=False)
    return X.reshape(X.shape[0], -1).astype(np.float64, copy=False)


def _jsonify(obj: Any) -> Any:
    """Best-effort JSON conversion for numpy containers."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def register_statistical_commands(subparsers) -> None:
    """Register statistical analysis commands with the main CLI."""
    drift = subparsers.add_parser(
        "drift-detect",
        aliases=["drift_detect", "drift-detection", "drift_detection"],
        help="📈 Drift detection (Hotelling T² / energy distance / Bayesian CP)",
    )
    drift.add_argument("--reference", "-r", required=True, help="Reference data (.npy/.npz)")
    drift.add_argument("--current", "-c", required=True, help="Current data (.npy/.npz)")
    drift.add_argument(
        "--methods",
        default="hotelling,ks,bayesian",
        help="Comma-separated methods: hotelling, ks, bayesian (default: all)",
    )
    drift.add_argument("--confidence-level", type=float, default=0.95, help="Confidence level (default: 0.95)")
    drift.add_argument("--output", "-o", help="Write JSON results to this path (otherwise prints JSON)")
    drift.add_argument("--plot", help="Optional PNG plot path for a drift summary visualization")
    drift.add_argument(
        "--plot-feature-index",
        type=int,
        default=0,
        help="Feature index to plot for distribution drift (default: 0)",
    )
    drift.set_defaults(func=_handle_drift_detect)

    z = subparsers.add_parser(
        "zscore",
        aliases=["z-score", "z_score", "zscore-analysis", "zscore_analysis"],
        help="📊 Enhanced Z-score analysis (uni/multi/robust/temporal)",
    )
    z.add_argument("--input", "-i", required=True, help="Input data (.npy/.npz)")
    z.add_argument("--reference", "-r", help="Optional reference/training data for fitting (.npy/.npz)")
    z.add_argument(
        "--method",
        choices=["multivariate", "robust", "temporal", "univariate", "hybrid"],
        default="multivariate",
        help="Which score series to emphasize in plots/summary (default: multivariate)",
    )
    z.add_argument("--confidence-level", type=float, default=0.95, help="Confidence level (default: 0.95)")
    z.add_argument("--window-size", type=int, default=100, help="Temporal window size (default: 100)")
    z.add_argument("--overlap", type=float, default=0.5, help="Temporal window overlap ratio (default: 0.5)")
    z.add_argument(
        "--robust-threshold",
        "--threshold",
        dest="robust_threshold",
        type=float,
        default=3.0,
        help="Anomaly threshold used by the hybrid detector (default: 3.0). `--threshold` is accepted as an alias.",
    )
    z.add_argument("--output", "-o", help="Write JSON results to this path (otherwise prints JSON)")
    z.add_argument("--plot", help="Optional PNG plot path for a z-score/anomaly visualization")
    z.set_defaults(func=_handle_zscore)


def _handle_drift_detect(args) -> int:
    try:
        from ..statistical.drift_detection_enhanced import create_enhanced_drift_detector

        ref = _to_2d_samples(_load_array(args.reference))
        cur = _to_2d_samples(_load_array(args.current))

        # Align feature dimensions (best-effort).
        p = int(min(ref.shape[1], cur.shape[1]))
        ref = ref[:, :p]
        cur = cur[:, :p]

        methods = [m.strip().lower() for m in str(getattr(args, "methods", "") or "").split(",") if m.strip()]
        if not methods:
            methods = ["hotelling", "ks", "bayesian"]

        detector = create_enhanced_drift_detector(confidence_level=float(args.confidence_level), methods=methods)
        per_method = detector.detect_drift_ensemble(ref, cur)
        consensus = detector.get_consensus_result(per_method)

        payload = {
            "reference_shape": list(ref.shape),
            "current_shape": list(cur.shape),
            "methods": methods,
            "consensus": _jsonify(asdict(consensus)),
            "per_method": {k: _jsonify(asdict(v)) for k, v in per_method.items()},
        }

        out = getattr(args, "output", None)
        plot = getattr(args, "plot", None)
        if out:
            outp = Path(str(out))
            _safe_makedirs(outp)
            outp.write_text(json.dumps(payload, indent=2))
            print(str(outp))
        else:
            print(json.dumps(payload, indent=2))

        if plot:
            try:
                _save_drift_plot(
                    ref,
                    cur,
                    consensus=consensus,
                    per_method=per_method,
                    out_path=str(plot),
                    feature_index=int(getattr(args, "plot_feature_index", 0)),
                )
            except Exception as e:
                logger.error(f"Drift plot failed: {e}")
        return 0
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        return 1


def _handle_zscore(args) -> int:
    try:
        from ..statistical.enhanced_zscore_analysis import EnhancedZScoreAnalyzer, ZScoreResults

        X = _to_2d_samples(_load_array(args.input))
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        analyzer = EnhancedZScoreAnalyzer(
            confidence_level=float(args.confidence_level),
            window_size=int(args.window_size),
            overlap_ratio=float(args.overlap),
            robust_threshold=float(args.robust_threshold),
        )

        ref_path = getattr(args, "reference", None)
        if ref_path:
            ref = _to_2d_samples(_load_array(str(ref_path)))
            ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
            analyzer.fit(ref)

        res = analyzer.analyze(X, return_detailed=True)
        if not isinstance(res, ZScoreResults):
            payload: dict[str, Any] = {"anomaly_flags": _jsonify(res)}
        else:
            mv = np.asarray(res.multivariate_zscores, dtype=np.float64)
            payload = {
                "input_shape": list(X.shape),
                "summary": {
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "anomaly_count": int(np.sum(res.anomaly_flags)),
                    "anomaly_rate": float(np.mean(res.anomaly_flags)) if X.shape[0] else 0.0,
                    "multivariate_z_mean": float(np.mean(mv)) if mv.size else 0.0,
                    "multivariate_z_p99": float(np.percentile(mv, 99)) if mv.size else 0.0,
                    "multivariate_z_max": float(np.max(mv)) if mv.size else 0.0,
                },
                "confidence_intervals": _jsonify(res.confidence_intervals),
                "statistical_significance": _jsonify(res.statistical_significance),
                "feature_contributions": _jsonify(res.feature_contributions),
                "anomaly_flags": _jsonify(res.anomaly_flags.astype(bool)),
                "multivariate_zscores": _jsonify(mv),
            }

        out = getattr(args, "output", None)
        plot = getattr(args, "plot", None)
        if out:
            outp = Path(str(out))
            _safe_makedirs(outp)
            outp.write_text(json.dumps(payload, indent=2))
            print(str(outp))
        else:
            print(json.dumps(payload, indent=2))

        if plot and isinstance(res, ZScoreResults):
            try:
                _save_zscore_plot(
                    X,
                    res,
                    out_path=str(plot),
                    method=str(getattr(args, "method", "multivariate")),
                    robust_threshold=float(getattr(args, "robust_threshold", 3.0)),
                    confidence_level=float(getattr(args, "confidence_level", 0.95)),
                )
            except Exception as e:
                logger.error(f"Z-score plot failed: {e}")
        return 0
    except Exception as e:
        logger.error(f"Z-score analysis failed: {e}")
        return 1


