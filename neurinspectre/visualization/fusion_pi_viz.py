"""Fusion attack π-viz visualization.

"π-viz" is a lightweight time-series view comparing per-step magnitudes of two
modalities (A/B). In practice, modalities can be:
- Two embedding/activation streams (e.g., text vs vision encoder outputs)
- Two prompts' hidden-state trajectories at a chosen layer
- Two saved numpy arrays representing time×feature sequences

Why this helps (security + interpretability):
- The L2 norm per step is a *coarse but fast* internal signal. Divergence between
  modalities can indicate cross-modal mismatch, multi-modal prompt injection,
  or fusion-based evasion attempts.
- This plot is best used for *triage*: it tells you **when** (which step/token)
  and **where** (which layer) modalities diverge, so you can drill down.

Design constraints:
- **No simulation**: callers must provide real arrays or real model hidden states.

Outputs:
- Plotly figure for interactive HTML.
- Matplotlib helper for static PNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

try:
    import plotly.graph_objects as go  # type: ignore
except Exception as _e:  # pragma: no cover
    go = None  # type: ignore
    _PLOTLY_IMPORT_ERROR = _e
else:  # pragma: no cover
    _PLOTLY_IMPORT_ERROR = None


def _require_plotly():
    if go is None:  # pragma: no cover
        raise ImportError(
            "plotly is required for interactive visualization. Install with: pip install plotly"
        ) from _PLOTLY_IMPORT_ERROR
    return go


def _to_numpy(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 0:
        raise ValueError("Expected array-like, got scalar")
    if not np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float32)
    if np.any(~np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32, copy=False)


def _select_layer(arr: np.ndarray, *, layer: Optional[int], layer_axis: int) -> np.ndarray:
    if arr.ndim < 3:
        return arr
    if layer is None:
        raise ValueError(
            f"Input array has {arr.ndim} dims; provide --layer to select a layer (axis {layer_axis})."
        )
    if layer_axis < 0 or layer_axis >= arr.ndim:
        raise ValueError(f"layer_axis={layer_axis} out of range for ndim={arr.ndim}")
    return np.take(arr, int(layer), axis=int(layer_axis))


def _to_time_feature(arr: np.ndarray) -> np.ndarray:
    """Normalize to shape [T, D]."""
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Common case: [batch, time, dim]
        return arr.mean(axis=0)
    raise ValueError(f"Unsupported array rank for π-viz: ndim={arr.ndim}, shape={arr.shape}")


def l2_timeseries(
    arr: Any,
    *,
    layer: Optional[int] = None,
    layer_axis: int = 0,
    max_steps: Optional[int] = None,
) -> np.ndarray:
    """Compute per-step L2 norm time-series from an array."""
    a = _to_numpy(arr)
    a = _select_layer(a, layer=layer, layer_axis=layer_axis)
    a = _to_time_feature(a)

    if max_steps is not None:
        a = a[: int(max_steps)]

    return np.linalg.norm(a, axis=1).astype(np.float32)


def robust_z(x: np.ndarray, *, sigma_floor: float = 1e-6) -> np.ndarray:
    """Robust z-score using median/MAD.

    This produces an interpretable "how unusually large" score without requiring
    a separate baseline run (useful when you only have one trace).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = max(float(1.4826 * mad), float(sigma_floor))
    return (x - med) / scale


@dataclass
class PiVizMetrics:
    steps: int
    corr: float
    mean_gap: float
    max_gap: float
    max_gap_step: int
    spike_z_threshold: float
    spike_steps: list[int]


def compute_metrics(
    a: np.ndarray,
    b: np.ndarray,
    *,
    z_threshold: float = 3.0,
    sigma_floor: float = 1e-6,
) -> PiVizMetrics:
    """Compute interpretable divergence metrics for two time-series."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1D time-series")
    n = int(min(a.size, b.size))
    if n == 0:
        raise ValueError("Empty time-series")

    a = np.asarray(a, dtype=np.float32)[:n]
    b = np.asarray(b, dtype=np.float32)[:n]

    gap = np.abs(a - b)
    mean_gap = float(np.mean(gap))
    max_gap_step = int(np.argmax(gap))
    max_gap = float(gap[max_gap_step])

    # Correlation (guard near-constant series)
    if n < 2 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        corr = float('nan')
    else:
        with np.errstate(all="ignore"):
            corr = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(corr):
            corr = float('nan')

    # Spike steps: robust z on |Δ| time-series
    z = robust_z(gap, sigma_floor=sigma_floor)
    spike_steps = np.where(z > float(z_threshold))[0].astype(int).tolist()

    return PiVizMetrics(
        steps=n,
        corr=corr,
        mean_gap=mean_gap,
        max_gap=max_gap,
        max_gap_step=max_gap_step,
        spike_z_threshold=float(z_threshold),
        spike_steps=spike_steps,
    )


def plot_pi_viz(
    a: np.ndarray,
    b: np.ndarray,
    *,
    title: str = "Fusion Attack Analysis: π-viz",
    label_a: str = "||Modality A||",
    label_b: str = "||Modality B||",
    x_label: str = "Timestep",
    y_label: str = "L2 Norm",
    tokens_a: Optional[Sequence[str]] = None,
    tokens_b: Optional[Sequence[str]] = None,
    z_threshold: float = 3.0,
    max_spikes: int = 6,
) -> "go.Figure":
    """Interactive π-viz Plotly figure.

    Args:
        tokens_a/tokens_b: optional token strings (prompt mode) for hover.
        z_threshold: robust spike threshold on |Δ|.
        max_spikes: max number of spike windows to highlight.
    """
    go_ = _require_plotly()

    n = int(min(a.size, b.size))
    a = np.asarray(a, dtype=np.float32)[:n]
    b = np.asarray(b, dtype=np.float32)[:n]

    metrics = compute_metrics(a, b, z_threshold=z_threshold)
    gap = np.abs(a - b)
    z = robust_z(gap)

    subtitle = (
        f"<span style='font-size:12px;color:#555'>steps={metrics.steps} | "
        f"mean|Δ|={metrics.mean_gap:.3f} | max|Δ|={metrics.max_gap:.3f} @t={metrics.max_gap_step} | "
        f"spikes(z&gt;{metrics.spike_z_threshold:g})={len(metrics.spike_steps)} | "
        + (f"corr={metrics.corr:.3f}</span>" if np.isfinite(metrics.corr) else "corr=n/a</span>")
    )
    subtitle += "<br><span style='font-size:11px;color:#777'>dashed=max|Δ| step; shaded=spike step(s)</span>"

    xs = list(range(n))

    # Build hover templates
    if tokens_a is not None:
        toks_a = list(tokens_a)[:n]
        a_custom = toks_a
        a_hover = (
            "<b>Modality A</b><br>t=%{x}<br>L2=%{y:.4f}<br>token=%{customdata}<extra></extra>"
        )
    else:
        a_custom = None
        a_hover = "<b>Modality A</b><br>t=%{x}<br>L2=%{y:.4f}<extra></extra>"

    if tokens_b is not None:
        toks_b = list(tokens_b)[:n]
        b_custom = toks_b
        b_hover = (
            "<b>Modality B</b><br>t=%{x}<br>L2=%{y:.4f}<br>token=%{customdata}<extra></extra>"
        )
    else:
        b_custom = None
        b_hover = "<b>Modality B</b><br>t=%{x}<br>L2=%{y:.4f}<extra></extra>"

    fig = go_.Figure()
    fig.add_trace(
        go_.Scatter(
            x=xs,
            y=a,
            mode='lines',
            name=label_a,
            line=dict(width=3, color='#4C6EF5'),
            customdata=a_custom,
            hovertemplate=a_hover,
        )
    )
    fig.add_trace(
        go_.Scatter(
            x=xs,
            y=b,
            mode='lines',
            name=label_b,
            line=dict(width=3, color='#F76707'),
            customdata=b_custom,
            hovertemplate=b_hover,
        )
    )

    # Highlight max-gap step for fast triage
    fig.add_vline(
        x=metrics.max_gap_step,
        line=dict(color='rgba(0,0,0,0.35)', width=2, dash='dot'),
    )

    # Highlight top spike steps (by robust z) with light shading
    if metrics.spike_steps:
        spike_steps = [s for s in metrics.spike_steps if 0 <= s < n]
        # rank by z value
        spike_steps = sorted(spike_steps, key=lambda s: float(z[s]), reverse=True)
        spike_steps = spike_steps[: int(max(0, max_spikes))]
        for s in spike_steps:
            fig.add_vrect(
                x0=s - 0.5,
                x1=s + 0.5,
                fillcolor='rgba(255,165,0,0.12)',
                line_width=0,
                layer='below',
            )

    fig.update_layout(
        template='plotly',
        title=dict(text=f"{title}<br>{subtitle}", x=0.5, xanchor='center'),
        xaxis=dict(title=dict(text=x_label)),
        yaxis=dict(title=dict(text=y_label)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0),
        margin=dict(l=70, r=40, t=110, b=460),
        height=760,
        width=1100,
        plot_bgcolor='white',
    )

    # Guidance boxes below axes using yshift (robust)
    blue_text = (
        "<b>🔵 BLUE TEAM — HOW/WHY</b><br>"
        "<b>WHY</b>: |Δ| spikes + low corr can indicate cross-modal prompt injection or fused evasion.<br>"
        "<b>HOW</b>: Start at max|Δ| @t; in prompt mode, inspect the token at that step and re-run adjacent layers.<br>"
        "<b>NEXT</b>: Treat LLMs as confusable deputies (residual prompt-injection risk); constrain actions and isolate modalities."
    )
    red_text = (
        "<b>🔴 RED TEAM — HOW/WHY</b><br>"
        "<b>WHY</b>: Persistent gaps reveal controllable cross-modal channels; large single-step spikes are high-signal to defenders.<br>"
        "<b>HOW</b>: Keep spike z below threshold by smoothing influence across steps and neurons; test transfer across paraphrases.<br>"
        "<b>OPSEC</b>: Avoid repeatable spike steps (same t across runs) and avoid a single dominant trigger token."
    )

    fig.add_annotation(
        text=blue_text,
        xref='paper', yref='paper',
        x=0.5, y=0.0,
        xanchor='center', yanchor='top',
        yshift=-120,
        showarrow=False,
        align='left',
        font=dict(size=11, color='white', family='Arial'),
        bgcolor='rgba(20,60,140,1.0)',
        bordercolor='#3399ff',
        borderwidth=2,
        borderpad=12,
        width=1020,
    )
    fig.add_annotation(
        text=red_text,
        xref='paper', yref='paper',
        x=0.5, y=0.0,
        xanchor='center', yanchor='top',
        yshift=-255,
        showarrow=False,
        align='left',
        font=dict(size=11, color='white', family='Arial'),
        bgcolor='rgba(140,20,20,1.0)',
        bordercolor='#ff3333',
        borderwidth=2,
        borderpad=12,
        width=1020,
    )

    return fig


def save_pi_viz_png(
    a: np.ndarray,
    b: np.ndarray,
    out_path: str,
    *,
    title: str = "Fusion Attack Analysis: π-viz",
    label_a: str = "||Modality A||",
    label_b: str = "||Modality B||",
    z_threshold: float = 3.0,
) -> str:
    """Save a static PNG (screenshot-style) with a max-gap marker."""
    import matplotlib.pyplot as plt  # type: ignore

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    metrics = compute_metrics(a, b, z_threshold=z_threshold)

    n = int(metrics.steps)
    a = a[:n]
    b = b[:n]

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.plot(range(n), a, linewidth=2.5, label=label_a)
    ax.plot(range(n), b, linewidth=2.5, label=label_b)

    # Mark max-gap step
    ax.axvline(metrics.max_gap_step, color='black', alpha=0.25, linestyle='--', linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('L2 Norm')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path
