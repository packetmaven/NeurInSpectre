#!/usr/bin/env python3
"""
NeurInSpectre Cross-Module Correlation Analysis
Part of the Time Travel Debugger suite
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def _load_array(path: Optional[str]) -> Optional["Any"]:
    import numpy as np
    if path is None:
        return None

    def _sanitize(arr):
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.number):
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    try:
        if str(path).lower().endswith('.npz'):
            data = np.load(path, allow_pickle=True)
            for key in ['data','X','x','A','arr','activations','series','primary','secondary']:
                if key in data:
                    return _sanitize(np.array(data[key]))
            for k in data.files:
                try:
                    return _sanitize(np.array(data[k]))
                except Exception:
                    continue
            return None
        else:
            obj = np.load(path, allow_pickle=True)
            # Allow dict-like objects saved via np.save(..., allow_pickle=True)
            if isinstance(obj, np.ndarray) and obj.dtype == np.dtype("O") and obj.shape == ():
                obj = obj.item()
            if isinstance(obj, dict):
                for key in ['data','X','x','A','arr','activations','series']:
                    if key in obj:
                        return _sanitize(np.array(obj[key]))
                for v in obj.values():
                    try:
                        return _sanitize(np.array(v))
                    except Exception:
                        continue
                return None
            return _sanitize(np.array(obj))
    except Exception:
        return None


def run_correlation(args):
    """Run cross-module correlation analysis"""
    import json
    import numpy as np

    logger.info("🔍 Running cross-module correlation analysis...")
    logger.info("📊 Primary analysis: %s", getattr(args, "primary", None))
    logger.info("📈 Secondary analysis: %s", getattr(args, "secondary", None))
    logger.info("⏱️ Temporal window: %s", getattr(args, "temporal_window", None))
    logger.info("📍 Spatial threshold: %s", getattr(args, "spatial_threshold", None))
    logger.info("🖥️ Device: %s", getattr(args, "device", None))

    if getattr(args, "correlate_action", None) != "run":
        logger.error("❌ Unknown correlation action: %s", getattr(args, "correlate_action", None))
        return 1

    primary_path = getattr(args, "primary_file", None)
    secondary_path = getattr(args, "secondary_file", None)
    if not primary_path or not secondary_path:
        logger.error("No input files provided. Provide --primary-file and --secondary-file (no synthetic/demo fallback).")
        return 1

    prim = _load_array(primary_path)
    sec = _load_array(secondary_path)
    if prim is None or sec is None:
        logger.error("Failed to load one or both inputs. primary=%s secondary=%s", primary_path, secondary_path)
        return 1

    def _to_2d(a: Any) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 0:
            a = a.reshape(1)
        if a.ndim == 1:
            return a.reshape(-1, 1)
        if a.ndim > 2:
            return a.reshape(a.shape[0], -1)
        return a

    prim2 = _to_2d(prim).astype(np.float64, copy=False)
    sec2 = _to_2d(sec).astype(np.float64, copy=False)

    # Align time axis (T) by truncation.
    T = int(min(prim2.shape[0], sec2.shape[0]))
    prim2 = prim2[:T]
    sec2 = sec2[:T]

    if T < 2:
        logger.error("Inputs are too short after alignment (T=%s). Need at least 2 timesteps.", T)
        return 1

    # Output prefix: anchor outputs to either --out-prefix directory (if provided) or the --plot directory.
    out_prefix_raw = str(getattr(args, "out_prefix", "_cli_runs/corr_") or "_cli_runs/corr_")
    out_prefix_path = Path(out_prefix_raw)
    if out_prefix_path.parent == Path("."):
        base_dir = Path(getattr(args, "plot", None)).parent if getattr(args, "plot", None) else Path(".")
        out_prefix = base_dir / out_prefix_path.name
    else:
        out_prefix = out_prefix_path
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_path = Path(str(out_prefix) + "summary.json")
    default_plot_path = Path(str(out_prefix) + "overlay.png")
    html_path = Path(str(out_prefix) + "interactive.html")
    plot_path = Path(getattr(args, "plot", None)) if getattr(args, "plot", None) else default_plot_path

    # Convert float temporal_window into a lag search bound.
    # Interpretation:
    # - temporal_window >= 1.0 → treated as "max lag in timesteps" (rounded)
    # - temporal_window < 1.0  → treated as a fraction of T (e.g., 0.1 => 10% of T)
    tw = float(getattr(args, "temporal_window", 0.0) or 0.0)
    if tw < 0:
        tw = 0.0
    lag_max = int(round(tw)) if tw >= 1.0 else int(round(tw * T))
    lag_max = int(max(0, min(lag_max, T - 2)))

    def _zscore_2d(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64, copy=False)
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return (x - mu) / sd

    def _corr_1d(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        if denom <= 0:
            return 0.0
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))

    def _best_lag_corr(a: np.ndarray, b: np.ndarray, lag: int) -> Tuple[float, int]:
        best = -1.0
        best_lag = 0
        for shift in range(-lag, lag + 1):
            if shift >= 0:
                aa = a[: (T - shift)]
                bb = b[shift:T]
            else:
                aa = a[(-shift):T]
                bb = b[: (T + shift)]
            if aa.size < 2 or bb.size < 2:
                continue
            score = _corr_1d(aa, bb)
            if score > best:
                best = score
                best_lag = shift
        if not np.isfinite(best):
            best = 0.0
            best_lag = 0
        return float(best), int(best_lag)

    def _best_lag_cos(A: np.ndarray, B: np.ndarray, lag: int) -> Tuple[float, int]:
        # Average per-timestep cosine similarity, choosing the best lag within ±lag.
        best = -1.0
        best_lag = 0
        for shift in range(-lag, lag + 1):
            if shift >= 0:
                AA = A[: (T - shift)]
                BB = B[shift:T]
            else:
                AA = A[(-shift):T]
                BB = B[: (T + shift)]
            if AA.shape[0] < 2:
                continue
            num = np.sum(AA * BB, axis=1)
            den = (np.linalg.norm(AA, axis=1) * np.linalg.norm(BB, axis=1)) + 1e-12
            cos = num / den
            score = float(np.mean(np.clip(cos, -1.0, 1.0)))
            if score > best:
                best = score
                best_lag = shift
        if not np.isfinite(best):
            best = 0.0
            best_lag = 0
        return float(best), int(best_lag)

    # Normalize per-feature for pattern/temporal scoring.
    Pn = _zscore_2d(prim2)
    Sn = _zscore_2d(sec2)

    dims_equal = (Pn.shape[1] == Sn.shape[1])

    corr_score: float
    spatial: float
    temporal: float
    best_lag: int

    # Always compute a 1D proxy series for plotting and for the mismatched-dim case.
    prim_series = np.linalg.norm(Pn, axis=1)
    sec_series = np.linalg.norm(Sn, axis=1)

    if not dims_equal:
        # Pattern correlation: correlate scalar proxy series at lag 0
        corr_score = _corr_1d(prim_series, sec_series)

        # Spatial coherence (mismatched-dim fallback): **dimension-invariant** scale similarity.
        #
        # IMPORTANT: using global L2 norms is NOT dimension-invariant; it scales with sqrt(N_elements).
        # Here we use RMS magnitudes (sqrt(mean(x^2))) so the ratio reflects per-element scale,
        # not array size.
        e1_rms = float(np.sqrt(np.mean(prim2 * prim2)))
        e2_rms = float(np.sqrt(np.mean(sec2 * sec2)))
        spatial = float(min(e1_rms, e2_rms) / (max(e1_rms, e2_rms) + 1e-12))

        # Temporal alignment: best-lag correlation between proxy series
        temporal, best_lag = _best_lag_corr(prim_series, sec_series, lag_max)
    else:
        # Pattern correlation: mean feature-wise correlation of normalized series
        # For standardized features, E[Pn*Sn] = correlation (per feature).
        per_feature_corr = np.mean(Pn * Sn, axis=0)
        corr_score = float(np.clip(float(np.mean(per_feature_corr)), -1.0, 1.0))

        # Spatial coherence: cosine similarity of per-feature energy profiles (0..1)
        e1 = np.sqrt(np.sum(prim2 ** 2, axis=0))
        e2 = np.sqrt(np.sum(sec2 ** 2, axis=0))
        spatial = float(np.dot(e1, e2) / (float(np.linalg.norm(e1) * np.linalg.norm(e2)) + 1e-12))
        spatial = float(np.clip(spatial, 0.0, 1.0))

        # Temporal alignment: best-lag average cosine similarity between vectors (on standardized series)
        temporal, best_lag = _best_lag_cos(Pn, Sn, lag_max)

    logger.info("🔍 Pattern correlation: %.3f", corr_score)
    logger.info("📊 Spatial coherence: %.3f", spatial)
    logger.info("⏱️ Temporal alignment: %.3f (best_lag=%s within ±%s)", temporal, best_lag, lag_max)

    files_identical = (str(primary_path) == str(secondary_path))
    if files_identical and corr_score > 0.99:
        logger.warning("⚠️ Primary and Secondary files are identical! Perfect correlation is expected.")
        logger.info("💡 For meaningful correlation, use different artifacts (e.g., gradients vs activations).")

    # Save a machine-readable summary (always).
    summary = {
        "primary": getattr(args, "primary", None),
        "secondary": getattr(args, "secondary", None),
        "primary_path": str(primary_path),
        "secondary_path": str(secondary_path),
        "primary_shape": [int(x) for x in prim2.shape],
        "secondary_shape": [int(x) for x in sec2.shape],
        "dims_equal": bool(dims_equal),
        "pattern_correlation": float(corr_score),
        "spatial_coherence": float(spatial),
        "spatial_metric": ("energy_profile_cosine" if dims_equal else "rms_ratio"),
        "temporal_alignment": float(temporal),
        "temporal_best_lag": int(best_lag),
        "temporal_lag_max": int(lag_max),
        "spatial_threshold": float(getattr(args, "spatial_threshold", 0.75) or 0.75),
    }
    try:
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("🧾 Summary JSON: %s", summary_path)
    except Exception as e:
        logger.warning("Failed to write summary JSON (%s): %s", summary_path, e)

    # Classification: "high correlation" requires both pattern correlation and spatial coherence thresholds.
    corr_thr = 0.75
    spatial_thr = float(getattr(args, "spatial_threshold", 0.75) or 0.75)
    high = (corr_score >= corr_thr) and (spatial >= spatial_thr)

    # Optional PNG overlay plot (only when --plot is provided).
    if getattr(args, "plot", None) is not None:
        try:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            from contextlib import nullcontext

            def _robust_z(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=np.float64).reshape(-1)
                if x.size < 2:
                    return np.zeros_like(x)
                med = float(np.median(x))
                mad = float(np.median(np.abs(x - med)) + 1e-12)
                # Normal-consistent scaling: MAD ≈ 0.67449σ
                _MAD_TO_SIGMA = 0.6744897501960817
                return (_MAD_TO_SIGMA * (x - med) / mad).astype(np.float64)

            # Proxy-series metrics (always defined; aligns with what this plot shows).
            proxy_corr0 = _corr_1d(prim_series, sec_series)
            proxy_best, proxy_best_lag = _best_lag_corr(prim_series, sec_series, lag_max)

            # For plotting, use robust Z-scores so the two proxies are comparable even if scales differ.
            pz = _robust_z(prim_series)
            sz = _robust_z(sec_series)

            # Optional: show a lag-aligned version of secondary for interpretability.
            sz_shifted = None
            if int(proxy_best_lag) != 0:
                sz_shifted = np.full_like(sz, np.nan, dtype=np.float64)
                sh = int(proxy_best_lag)
                if sh > 0:
                    # Secondary lags primary by sh => shift secondary left by sh to align.
                    sz_shifted[: T - sh] = sz[sh:T]
                else:
                    k = -sh
                    # Secondary leads primary by k => shift secondary right by k.
                    sz_shifted[k:T] = sz[: T - k]

            # Spike detection (salient events): robust Z threshold.
            spike_z = 3.0
            p_spikes = np.where(pz >= spike_z)[0]
            s_spikes = np.where(sz >= spike_z)[0]

            # Rolling correlation at best lag (proxy series), to surface non-stationarity.
            # Window size: ~1/6 of available length, bounded for stability.
            # Use lag-aligned series for rolling computation.
            if proxy_best_lag >= 0:
                pa = pz[: T - int(proxy_best_lag)]
                sa = sz[int(proxy_best_lag) : T]
                t0 = 0
            else:
                k = int(-proxy_best_lag)
                pa = pz[k:T]
                sa = sz[: T - k]
                t0 = k
            L = int(min(pa.size, sa.size))
            W = int(max(24, min(96, round(L / 6)))) if L >= 24 else max(4, L)
            roll_t = []
            roll_c = []
            if L >= max(8, W):
                for i in range(0, L - W + 1):
                    roll_t.append(int(t0 + i + W // 2))
                    roll_c.append(_corr_1d(pa[i : i + W], sa[i : i + W]))

            # Cross-correlation vs lag curve (proxy series).
            lags = list(range(-lag_max, lag_max + 1))
            xcorr = []
            for sh in lags:
                if sh >= 0:
                    aa = pz[: T - sh]
                    bb = sz[sh:T]
                else:
                    kk = -sh
                    aa = pz[kk:T]
                    bb = sz[: T - kk]
                xcorr.append(_corr_1d(aa, bb))

            # Interpretation hint (kept conservative + test-focused).
            interp = "Trends > snapshots; validate vs baseline runs."
            if spatial >= 0.85 and abs(corr_score) < 0.2 and abs(proxy_corr0) < 0.2:
                interp = "High spatial similarity but low coupling: investigate modality-specific anomalies + lagged effects."
            elif temporal >= 0.7 or proxy_best >= 0.7:
                interp = "Strong lagged coupling: check persistence across windows/runs and pipeline delays."

            style_ctx = nullcontext()
            try:
                style_ctx = plt.style.context("seaborn-v0_8-whitegrid")
            except Exception:
                pass

            with style_ctx, mpl.rc_context(
                {
                    "axes.titlesize": 13,
                    "axes.labelsize": 11,
                    "xtick.labelsize": 9,
                    "ytick.labelsize": 9,
                    "legend.fontsize": 9,
                    "figure.titlesize": 16,
                    "font.family": "DejaVu Sans",
                }
            ):
                fig = plt.figure(figsize=(12.8, 7.6), dpi=180)
                gs = fig.add_gridspec(
                    2,
                    2,
                    height_ratios=[3.1, 2.1],
                    width_ratios=[3.35, 1.65],
                    hspace=0.18,
                    wspace=0.18,
                )

                ax = fig.add_subplot(gs[0, :])
                sub = gs[1, 0].subgridspec(2, 1, hspace=0.30)
                ax_roll = fig.add_subplot(sub[0, 0])
                ax_lag = fig.add_subplot(sub[1, 0])
                ax_steps = fig.add_subplot(gs[1, 1])
                ax_steps.axis("off")

                title = (
                    "NeurInSpectre — Correlation Triage (Proxy + Lag + Windows)\n"
                    f"pattern={corr_score:.3f} | spatial={spatial:.3f} | temporal={temporal:.3f} (lag={best_lag})  •  "
                    f"proxy corr0={proxy_corr0:.3f} | proxy best={proxy_best:.3f} (lag={proxy_best_lag})"
                )
                ax.set_title(title)

                t = np.arange(T, dtype=int)
                ax.plot(t, pz, color="#D62728", lw=2.2, alpha=0.90, label="Primary proxy (robust Z)")
                ax.plot(t, sz, color="#F59E0B", lw=2.0, alpha=0.85, label="Secondary proxy (robust Z)")
                if sz_shifted is not None:
                    ax.plot(t, sz_shifted, color="#F59E0B", lw=1.4, alpha=0.35, ls="--", label=f"Secondary shifted (lag={proxy_best_lag})")

                # Mark salient spikes
                if p_spikes.size:
                    ax.scatter(p_spikes, pz[p_spikes], s=26, color="#D62728", edgecolor="white", linewidth=0.6, zorder=5)
                if s_spikes.size:
                    ax.scatter(s_spikes, sz[s_spikes], s=26, color="#F59E0B", edgecolor="white", linewidth=0.6, zorder=5)
                ax.axhline(spike_z, color="#6B7280", lw=1.2, ls=(0, (6, 3)), alpha=0.7, label=f"spike threshold = {spike_z:g}σ (robust)")

                ax.set_xlabel("Time index")
                ax.set_ylabel("Proxy magnitude (robust Z)")
                ax.grid(alpha=0.25)

                # Summary box (kept concise)
                sum_lines = [
                    "Summary",
                    f"T={T}  shapes: P={tuple(prim2.shape)}  S={tuple(sec2.shape)}",
                    f"dims_equal={dims_equal}  spatial_metric={summary['spatial_metric']}",
                    f"proxy spikes: P={int(p_spikes.size)}  S={int(s_spikes.size)}",
                    f"rolling window={W}  lag_max=±{lag_max}",
                    f"interpretation: {interp}",
                ]
                ax.text(
                    0.995,
                    0.98,
                    "\n".join(sum_lines),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D0D0D0", alpha=0.96),
                )

                leg = ax.legend(loc="upper right", frameon=True, framealpha=0.92)
                leg.get_frame().set_edgecolor("#D0D0D0")

                # Rolling correlation panel
                if roll_t:
                    ax_roll.plot(roll_t, roll_c, color="#1F77B4", lw=2.0, label=f"rolling corr (W={W}, lag={proxy_best_lag})")
                    ax_roll.axhline(0.7, color="#6B7280", lw=1.0, ls=(0, (6, 3)), alpha=0.6, label="high-coupling guide = 0.7")
                else:
                    ax_roll.text(0.5, 0.5, "rolling corr: insufficient length", ha="center", va="center", transform=ax_roll.transAxes)
                ax_roll.set_title("Windowed coupling (proxy)")
                ax_roll.set_ylabel("corr")
                ax_roll.set_xlim(0, max(1, T - 1))
                ax_roll.set_ylim(-1.05, 1.05)
                ax_roll.grid(alpha=0.25)
                ax_roll.legend(frameon=False, loc="upper right")

                # Lag curve panel
                ax_lag.plot(lags, xcorr, color="#6C5CE7", lw=2.0, marker="o", markersize=3.5, alpha=0.9)
                ax_lag.axvline(int(proxy_best_lag), color="#B91C1C", lw=1.2, ls=(0, (6, 3)), alpha=0.8)
                ax_lag.axhline(float(proxy_best), color="#B91C1C", lw=1.0, ls=":", alpha=0.6)
                ax_lag.set_title("Cross-correlation vs lag (proxy)")
                ax_lag.set_xlabel("lag (timesteps)")
                ax_lag.set_ylabel("corr")
                ax_lag.grid(alpha=0.25)
                ax_lag.set_ylim(-1.05, 1.05)

                # Next steps (kept safe; no evasion instructions)
                blue_text = (
                    "Blue team — practical next steps\n"
                    "• Establish baselines for corr/spatial/temporal metrics per model/run\n"
                    "• Alert on sustained rolling corr increases (not single spikes)\n"
                    "• Investigate spike windows (≥3σ) + whether lagged coupling appears\n"
                    "• Localize: re-run on narrower slices (layer/block/feature groups)\n"
                    "• Correlate with drift/anomaly + spectral triage before actioning"
                )
                red_text = (
                    "Red team — evaluation checklist (safe)\n"
                    "• Run controlled scenarios; record metrics + rolling windows\n"
                    "• Report stability across seeds/runs (avoid one-off snapshots)\n"
                    "• Measure FP/FN vs benign baselines; track best lag consistency\n"
                    "• Share artifacts + summary.json (inputs, window, lag_max, plots)"
                )
                ax_steps.text(
                    0.0,
                    1.0,
                    blue_text,
                    va="top",
                    ha="left",
                    fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.55", facecolor="#ECF3FF", edgecolor="#1F5FBF", alpha=0.95),
                )
                ax_steps.text(
                    0.0,
                    0.48,
                    red_text,
                    va="top",
                    ha="left",
                    fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.55", facecolor="#FFF1F2", edgecolor="#B91C1C", alpha=0.95),
                )

                fig.text(
                    0.01,
                    0.01,
                    "Note: Metrics are heuristic and data-dependent. Prefer baseline comparisons and persistence over single-run snapshots.",
                    fontsize=8.5,
                    color="#555",
                )

                # Avoid tight_layout warnings for mixed text/axes layouts.
                fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.08)
                fig.savefig(str(plot_path), dpi=220, bbox_inches="tight")
                plt.close(fig)
            logger.info("🖼️ Saved plot: %s", plot_path)
        except Exception as e:
            logger.warning("Plot generation failed: %s", e)

    # Optional interactive HTML (independent of --plot).
    if bool(getattr(args, "interactive", False)):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[[{"type": "xy", "colspan": 2}, None], [{"type": "table"}, {"type": "table"}]],
                subplot_titles=(
                    "Cross-Modal Temporal Correlation Analysis",
                    "🔴 Red Team Actionable Intelligence",
                    "🔵 Blue Team Defense Recommendations",
                ),
                row_heights=[0.55, 0.45],
                vertical_spacing=0.12,
            )

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(prim_series))),
                    y=list(prim_series),
                    mode="lines+markers",
                    name="Primary (proxy)",
                    line=dict(color="#E74C3C", width=3),
                    marker=dict(size=6, opacity=0.6),
                    hovertemplate=(
                        "<b>Primary (proxy)</b><br>"
                        "Time Index: %{x}<br>"
                        "Value: %{y:.6f}<br>"
                        f"<b>Pattern Correlation: {corr_score:.4f}</b><br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(sec_series))),
                    y=list(sec_series),
                    mode="lines+markers",
                    name="Secondary (proxy)",
                    line=dict(color="#F39C12", width=3),
                    marker=dict(size=6, opacity=0.6),
                    hovertemplate=(
                        "<b>Secondary (proxy)</b><br>"
                        "Time Index: %{x}<br>"
                        "Value: %{y:.6f}<br>"
                        f"<b>Pattern Correlation: {corr_score:.4f}</b><br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

            # Reuse the existing narrative tables (already built below in the original code).
            # Build tables based on corr_score (same thresholds as above).
            if corr_score >= 0.7:
                threat = "HIGH"
                red_actions = [
                    ["Status", "🔴 HIGH COUPLING (corr ≥ 0.7)"],
                    ["Interpretation", "Gradients and activations move together; a single perturbation may spill across modalities"],
                    ["Risk proxy", "Higher coupling can increase cross-modal leakage/transfer (validate on your environment)"],
                    ["Validation", "Measure whether gradients predict activation drift (and vice versa) against a clean baseline"],
                    ["Tooling", "neurinspectre correlate run --primary activations --secondary gradients --primary-file acts.npy --secondary-file grads.npy"],
                    ["Optional (authorized)", "neurinspectre gradient_inversion recover --gradients grads.npy --out-prefix _cli_runs/ginv_"],
                    ["Optional (authorized)", "neurinspectre latent-jailbreak --model <hf-id> --prompt <text> --magnitude 1.0"],
                    ["Note", "Correlation is not exploitability; treat this as a prioritization signal, not a proof"],
                ]
            elif corr_score >= 0.4:
                threat = "MEDIUM"
                red_actions = [
                    ["Status", "🟡 MODERATE COUPLING (0.4–0.7)"],
                    ["Interpretation", "Partial coupling; transfer may be inconsistent across time/layers/features"],
                    ["Risk proxy", "Multi-signal evaluation is usually required (avoid overfitting to one correlation snapshot)"],
                    ["Validation", "Track correlation drift across runs and compare to benign baselines"],
                    ["Tooling", "neurinspectre correlate run --primary activations --secondary gradients --primary-file acts.npy --secondary-file grads.npy"],
                    ["Tooling", "neurinspectre fusion_attack --primary grads.npy --secondary acts.npy --alpha 0.5 --sweep --out-prefix _cli_runs/fusion_"],
                    ["Optional", "Investigate time windows with lowest coupling for modality-specific anomalies"],
                    ["Note", "Treat outputs as diagnostics; confirm with domain-appropriate ground truth"],
                ]
            else:
                threat = "LOW"
                red_actions = [
                    ["Status", "🟢 LOW COUPLING (corr < 0.4)"],
                    ["Interpretation", "Gradients and activations appear decoupled on this sample/window"],
                    ["Risk proxy", "Single-modality monitoring can miss modality-specific anomalies; validate each modality independently"],
                    ["Validation", "Check whether coupling changes under suspect prompts/inputs vs benign baselines"],
                    ["Tooling", "neurinspectre activation_anomaly_detection --model <hf-id> --baseline-prompt <b> --test-prompt <t> --out _cli_runs/anom.html"],
                    ["Tooling", "neurinspectre anomaly --input acts_or_grads.npy --method auto --topk 20 --out-prefix _cli_runs/anom_"],
                    ["Optional (authorized)", "neurinspectre evasion-detect activations.npy --detector-type all --threshold 0.75 --output-dir _cli_runs/evasion"],
                    ["Note", "Low coupling can be benign; prefer trends over time and baseline comparisons"],
                ]

            if corr_score >= 0.7:
                blue_actions = [
                    ["Status", "🔵 HIGH COUPLING - Single-Modal Defense Often Sufficient"],
                    ["Meaning", "Monitoring one modality can capture the other (verify with baselines)"],
                    ["Defense Priority", "MEDIUM"],
                    ["Primary Action", "Deploy gradient monitoring with clipping/DP (if applicable)"],
                    ["Secondary Action", "Keep activation monitoring enabled to detect decoupling"],
                    ["Alert", "Investigate sharp coupling changes (Δcorr) across runs/windows"],
                    ["Note", "Treat correlation as a signal; confirm with trend + baseline comparisons"],
                ]
            elif corr_score >= 0.4:
                blue_actions = [
                    ["Status", "🟡 MODERATE COUPLING - Multi-Signal Defense Recommended"],
                    ["Meaning", "Partial independence requires both defenses"],
                    ["Defense Priority", "HIGH"],
                    ["Primary Action", "Deploy gradient AND activation monitoring"],
                    ["Gradient Defense", "Clipping + privacy accounting if applicable (calibrate ε/σ)"],
                    ["Activation Defense", "Baseline layers and alert on drift (|Z|>3 is a common starting point)"],
                    ["Alert", "Investigate correlation shifts >0.2 over short windows"],
                    ["Note", "Stability over time matters more than a single corr snapshot"],
                ]
            else:
                blue_actions = [
                    ["Status", "⚠️ LOW COUPLING - Multi-Signal Monitoring Recommended"],
                    ["Meaning", "Modalities may carry independent anomaly signals"],
                    ["Defense Priority", "HIGH"],
                    ["Primary Action", "Multi-layered monitoring: gradients + activations + temporal drift"],
                    ["Gradient Defense", "Clipping + privacy accounting (if applicable)"],
                    ["Activation Defense", "Layer-wise anomaly monitoring and time-travel debugging"],
                    ["Monitoring", "neurinspectre evasion-detect activations.npy --detector-type all --threshold 0.75 --output-dir _cli_runs/evasion"],
                    ["Note", "Low coupling is not inherently malicious; confirm with trend + baseline comparisons"],
                ]

            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["<b>Metric</b>", "<b>Value / Action</b>"],
                        fill_color="#dc3545",
                        align="left",
                        font=dict(color="white", size=13, family="monospace"),
                    ),
                    cells=dict(
                        values=[[r[0] for r in red_actions], [r[1] for r in red_actions]],
                        fill_color=[["#2c2c2c" if i % 2 == 0 else "#1a1a1a" for i in range(len(red_actions))]],
                        align="left",
                        font=dict(size=11, family="monospace", color="white"),
                        height=28,
                    ),
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["<b>Metric</b>", "<b>Value / Action</b>"],
                        fill_color="#007bff",
                        align="left",
                        font=dict(color="white", size=13, family="monospace"),
                    ),
                    cells=dict(
                        values=[[b[0] for b in blue_actions], [b[1] for b in blue_actions]],
                        fill_color=[["#2c2c2c" if i % 2 == 0 else "#1a1a1a" for i in range(len(blue_actions))]],
                        align="left",
                        font=dict(size=11, family="monospace", color="white"),
                        height=28,
                    ),
                ),
                row=2,
                col=2,
            )

            fig.update_layout(
                title=(
                    "<b>NeurInSpectre — Cross-Modal Correlation Analysis</b><br>"
                    f"<sub>corr={corr_score:.3f} | spatial={spatial:.3f} | temporal={temporal:.3f} (lag={best_lag}) | level={threat}</sub>"
                ),
                xaxis_title="Time Index",
                yaxis_title="Proxy magnitude (z-scored space)",
                template="plotly_dark",
                hovermode="x unified",
                height=900,
                showlegend=True,
                font=dict(family="monospace"),
            )

            fig.update_xaxes(gridcolor="rgba(128,128,128,0.3)")
            fig.update_yaxes(gridcolor="rgba(128,128,128,0.3)")

            fig.write_html(str(html_path))
            print(f"🌐 Interactive HTML: {html_path}")
            logger.info("🌐 Interactive HTML: %s", html_path)
        except Exception as e:
            logger.warning("⚠️ Interactive HTML failed: %s", e)

    logger.info("✅ Correlation analysis complete")
    if high:
        logger.info("🎯 High-confidence correlations detected")
    else:
        logger.info("ℹ️ Moderate/low correlation; review parameters and inputs")

    return 0

    # NOTE: legacy flow below kept for reference; the active implementation returns above.
