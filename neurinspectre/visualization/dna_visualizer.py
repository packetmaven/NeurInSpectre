"""DNA / activation visualization helpers.

This module restores the layer-wise *Neuron Activation Anomaly Detection* plot
(bar: anomaly count, line: max Z-score) seen in historical NeurInSpectre outputs
(e.g. ``output/dna_analysis/anomaly_detection.html``).

Design goals:
- **No simulation**: functions operate only on provided activations.
- **Realistic anomaly accounting**: per-layer anomaly count is computed as the
  number of hidden dimensions whose *max* |Z| over test samples exceeds a
  threshold, where baseline (normal) statistics are estimated from the baseline
  samples.
- **Modern hardening** (research-informed): optional robust MAD-based Z-scores
  and a configurable variance floor to reduce brittleness to near-zero baseline
  variance (a common failure mode and an evasion vector).

Inputs:
- ``normal_patterns`` / ``test_patterns`` are mappings from layer identifiers
  (e.g., ``"layer_0"``) to activation tensors/arrays of shape like:
  ``[batch, seq_len, hidden]`` or ``[seq_len, hidden]``.

The visualization is intentionally compact: it answers *where* anomalies
concentrate across layers.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple
import math
import re

import numpy as np

import torch  # type: ignore

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
            "plotly is required for visualization. Install with: pip install plotly"
        ) from _PLOTLY_IMPORT_ERROR
    return go


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch/array-like to a float numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim == 0:
        raise ValueError("Activation value is scalar; expected at least 1D")
    if not np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float32)
    # sanitize inf/nan
    if np.any(~np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32, copy=False)


def _to_2d_samples(x: np.ndarray) -> np.ndarray:
    """Normalize activations to shape [N_samples, D_hidden]."""
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim == 2:
        return x.reshape(-1, x.shape[-1])
    # Treat last dim as hidden; flatten all preceding dims.
    return x.reshape(-1, x.shape[-1])


def _layer_sort_key(layer_id: Any) -> Tuple[int, str]:
    s = str(layer_id)
    # Prefer trailing integer (layer_12 -> 12)
    m = re.search(r"(\d+)(?!.*\d)", s)
    if m:
        return (int(m.group(1)), s)
    return (10**9, s)


def _compute_layer_anomaly_metrics(
    baseline: np.ndarray,
    test: np.ndarray,
    *,
    threshold: float,
    robust: bool,
    sigma_floor: Optional[float],
) -> Tuple[int, float, float, float, int, int, int]:
    """Compute per-layer anomaly count and max Z.

    Returns:
      (anomaly_count, max_abs_z, mean_shift_l2, cosine_mean, n_base, n_test, d)
    """
    base = _to_2d_samples(baseline)
    tst = _to_2d_samples(test)

    if base.size == 0 or tst.size == 0:
        raise ValueError("Empty activation array after preprocessing")

    d = int(min(base.shape[1], tst.shape[1]))
    base = base[:, :d]
    tst = tst[:, :d]

    if robust:
        center = np.median(base, axis=0)
        mad = np.median(np.abs(base - center), axis=0)
        # MAD -> sigma approximation for normal distributions
        scale = 1.4826 * mad
    else:
        center = np.mean(base, axis=0)
        scale = np.std(base, axis=0)

    # Variance floor: avoids division by ~0 which produces meaningless huge Z.
    if sigma_floor is None:
        global_scale = float(np.std(base))
        floor = max(1e-6, global_scale * 1e-2)
    else:
        floor = float(sigma_floor)
    scale = np.maximum(scale, floor)

    z = (tst - center) / scale
    max_abs_z_dim = np.max(np.abs(z), axis=0)

    anomaly_count = int(np.sum(max_abs_z_dim > float(threshold)))
    max_abs_z = float(np.max(max_abs_z_dim)) if max_abs_z_dim.size else float('nan')

    base_mean = np.mean(base, axis=0)
    test_mean = np.mean(tst, axis=0)
    mean_shift_l2 = float(np.linalg.norm(test_mean - base_mean))
    denom = float(np.linalg.norm(base_mean) * np.linalg.norm(test_mean) + 1e-12)
    cosine_mean = float(np.dot(base_mean, test_mean) / denom) if denom > 0 else float('nan')

    return anomaly_count, max_abs_z, mean_shift_l2, cosine_mean, int(base.shape[0]), int(tst.shape[0]), d


def plot_anomaly_detection(
    normal_patterns: Mapping[Any, Any],
    test_patterns: Mapping[Any, Any],
    *,
    threshold: float = 2.5,
    robust: bool = False,
    sigma_floor: Optional[float] = None,
    title: Optional[str] = None,
    numeric_layer_sort: bool = True,
) -> "go.Figure":
    """Create the dual-axis anomaly chart across layers.

    The bar chart shows the number of hidden dimensions whose max |Z| across the
    *test* samples exceeds ``threshold``. The line shows the maximum |Z| in that
    layer.

    Notes (security / evasion-aware):
    - Simple per-neuron Z thresholds are vulnerable to **distributed, low-and-slow
      activation drift** (changes spread across many dimensions). Use this panel
      as a triage tool and pair it with multivariate drift metrics when possible.
    - Setting ``robust=True`` improves resilience to heavy tails/outliers.

    Args:
        normal_patterns: dict-like mapping layer id -> activations (baseline).
        test_patterns: dict-like mapping layer id -> activations (test).
        threshold: Z threshold (labelled σ for continuity with historical plots).
        robust: If True, use median/MAD (robust Z).
        sigma_floor: Optional minimum scale for z-score denominator.
        title: Override plot title.
        numeric_layer_sort: Sort layers by numeric suffix if possible.

    Returns:
        Plotly Figure.
    """
    go_ = _require_plotly()

    if normal_patterns is None or test_patterns is None:
        raise ValueError("normal_patterns and test_patterns are required")

    common = set(normal_patterns.keys()) & set(test_patterns.keys())
    if not common:
        raise ValueError("No overlapping layer keys between normal_patterns and test_patterns")

    layers = sorted(common, key=_layer_sort_key if numeric_layer_sort else lambda k: str(k))

    x_labels = [str(k) for k in layers]

    anomaly_counts = []
    max_z = []
    hover_rows = []

    for k in layers:
        base = _to_numpy(normal_patterns[k])
        tst = _to_numpy(test_patterns[k])
        (
            count,
            max_abs_z,
            mean_shift_l2,
            cosine_mean,
            n_base,
            n_test,
            d,
        ) = _compute_layer_anomaly_metrics(
            base,
            tst,
            threshold=float(threshold),
            robust=bool(robust),
            sigma_floor=sigma_floor,
        )
        anomaly_counts.append(count)
        max_z.append(max_abs_z)
        hover_rows.append((count, max_abs_z, mean_shift_l2, cosine_mean, n_base, n_test, d))

    finite_max = [v for v in max_z if isinstance(v, (int, float)) and math.isfinite(v)]
    y2_max = max(finite_max) if finite_max else 1.0
    y2_range = [0.0, float(y2_max * 1.1 if y2_max > 0 else 1.0)]

    suffix = " (robust MAD)" if robust else ""
    if title is None:
        title = f"NeurInSpectre — Neuron Activation Anomaly Detection (Threshold: {float(threshold):g}σ){suffix}"

    fig = go_.Figure()

    fig.add_trace(
        go_.Bar(
            x=x_labels,
            y=anomaly_counts,
            name="Anomaly Count",
            opacity=0.7,
            marker=dict(color="indianred"),
            customdata=hover_rows,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Anomaly Count: %{y}<br>"
                "Max |Z|: %{customdata[1]:.2f}<br>"
                "Δmean L2: %{customdata[2]:.3f}<br>"
                "Cosine(mean): %{customdata[3]:.3f}<br>"
                "Baseline samples: %{customdata[4]}<br>"
                "Test samples: %{customdata[5]}<br>"
                "Hidden dim: %{customdata[6]}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go_.Scatter(
            x=x_labels,
            y=max_z,
            name="Max Z-Score",
            mode="lines+markers",
            line=dict(color="royalblue", width=2),
            yaxis="y2",
            customdata=hover_rows,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Max |Z|: %{y:.2f}<br>"
                "Anomaly Count: %{customdata[0]}<br>"
                "Δmean L2: %{customdata[2]:.3f}<br>"
                "Cosine(mean): %{customdata[3]:.3f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly",
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text="Layer Index")),
        yaxis=dict(title=dict(text="Anomaly Count")),
        yaxis2=dict(
            title=dict(text="Max Z-Score"),
            overlaying="y",
            side="right",
            range=y2_range,
        ),
        legend=dict(
            orientation="h",
            # Keep legend safely above the plotting area so it never collides with callouts
            # on resize or with long titles.
            yanchor="top",
            y=1.16,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=80, r=80, t=120, b=340),
        height=700,
        width=1000,
    )

    # Visual threshold marker on y2 (Max |Z|) for quick interpretation
    try:
        thr = float(threshold)
        fig.add_shape(
            type="line",
            xref="paper", yref="y2",
            x0=0.0, x1=1.0,
            y0=thr, y1=thr,
            line=dict(color="rgba(255,165,0,0.85)", dash="dash", width=2),
        )
        fig.add_annotation(
            xref="paper", yref="y2",
            x=1.0, y=thr,
            text=f"Z-threshold = {thr:g}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=10, color="rgba(255,165,0,0.95)"),
        )
    except Exception:
        pass

    # Red/Blue team guidance boxes (kept below x-axis).
    # NOTE: We intentionally reserve a large bottom margin to avoid cropping.
    _mono = "font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;"
    blue_text = (
        "<b>🔵 BLUE TEAM — WHY/HOW</b><br>"
        "• <b>WHY</b>: Steering can shift safety; peaks flag leverage layers<br>"
        "• <b>HOW</b>: Use <span style='" + _mono + "'>--baseline-file</span> for stable baselines;<br>"
        "&nbsp;&nbsp;alert on repeatable Count↑ + Max|Z|↑ across prompts<br>"
        "• <b>NEXT</b>: Localize with <span style='" + _mono + "'>activation_attack_patterns</span>;<br>"
        "&nbsp;&nbsp;harden via salting / conditional steering"
    )
    red_text = (
        "<b>🔴 RED TEAM — WHY/HOW</b><br>"
        "• <b>WHY</b>: Repeatable layer peaks suggest controllable internal features<br>"
        "• <b>HOW</b>: Test prompt variants; look for peaks that transfer across prompts<br>"
        "• <b>NOTE</b>: Distributed drift can reduce |Z|;<br>"
        "&nbsp;&nbsp;defenders should also use multivariate drift checks"
    )

    fig.add_annotation(
        text=blue_text,
        xref="paper", yref="paper",
        x=0.02, y=-0.30,
        showarrow=False,
        font=dict(size=11, color="white", family="Arial"),
        align="left",
        bgcolor='rgba(20,60,140,0.92)',
        bordercolor='#3399ff',
        borderwidth=2,
        borderpad=10,
        xanchor='left',
        yanchor='top',
    )

    fig.add_annotation(
        text=red_text,
        xref="paper", yref="paper",
        x=0.98, y=-0.30,
        showarrow=False,
        font=dict(size=11, color="white", family="Arial"),
        align="left",
        bgcolor='rgba(140,20,20,0.92)',
        bordercolor='#ff3333',
        borderwidth=2,
        borderpad=10,
        xanchor='right',
        yanchor='top',
    )

    return fig

# ------------------------------
# Additional restored visualizations
# ------------------------------

def _to_seq_hidden(x: Any) -> np.ndarray:
    """Convert an activation tensor/array to shape [seq, hidden].

    Accepts common shapes:
    - [batch, seq, hidden]
    - [seq, hidden]
    - [hidden] (treated as seq=1)
    """
    arr = _to_numpy(x)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    # Fallback: flatten everything except last dim.
    return arr.reshape(-1, arr.shape[-1])


def _reduce_seq(mat: np.ndarray, reduce: str) -> np.ndarray:
    """Reduce [seq, hidden] -> [hidden] using a named reduction."""
    r = str(reduce).lower()
    if r in ('last', 'last_token'):
        return mat[-1]
    if r in ('mean', 'avg', 'average'):
        return np.mean(mat, axis=0)
    if r in ('maxabs', 'max_abs', 'max-abs'):
        return mat[np.argmax(np.abs(mat), axis=0), np.arange(mat.shape[1])]
    if r in ('max',):
        return np.max(mat, axis=0)
    raise ValueError(f"Unknown reduce='{reduce}'. Use one of: mean, last, maxabs, max")


def plot_neuron_heatmap(
    activations: Mapping[Any, Any],
    *,
    top_k: int = 50,
    layer_range: Optional[Tuple[int, int]] = None,
    reduce: str = 'mean',
    title: Optional[str] = None,
    colorbar_title: str = 'Activation',
    numeric_layer_sort: bool = True,
) -> "go.Figure":
    """Neural Persistence heatmap (neurons × layers).

    This restores the historical `neuron_heatmap.html` concept: pick a stable set
    of top-K neuron indices across layers (by max |activation| across layers)
    and render a heatmap of their per-layer activations.

    Args:
        activations: mapping layer id -> activations (hidden states).
        top_k: number of neurons to display.
        layer_range: optional (start,end) filter on numeric layer suffix.
        reduce: how to reduce [seq, hidden] into per-neuron values.
        title: optional title override.
        colorbar_title: label for heatmap colorbar.
        numeric_layer_sort: sort layers by numeric suffix when possible.

    Returns:
        Plotly Figure.
    """
    go_ = _require_plotly()
    if not activations:
        raise ValueError('activations is empty')

    # Filter + sort layers
    items = list(activations.items())
    if layer_range is not None:
        lo, hi = int(layer_range[0]), int(layer_range[1])
        kept = []
        for k, v in items:
            idx, _ = _layer_sort_key(k)
            if idx != 10**9 and lo <= idx <= hi:
                kept.append((k, v))
        items = kept
        if not items:
            raise ValueError('No layers left after applying layer_range')

    items = sorted(items, key=_layer_sort_key if numeric_layer_sort else lambda kv: str(kv[0]))
    layer_keys = [k for k, _ in items]

    # Build [L, H] matrix
    vecs = []
    for _, v in items:
        mat = _to_seq_hidden(v)
        vecs.append(_reduce_seq(mat, reduce=reduce))

    # Align hidden sizes
    h = min(int(v.shape[-1]) for v in vecs)
    vecs = [v[:h] for v in vecs]
    A = np.vstack(vecs)  # [layers, hidden]

    # Select top-K neurons globally (max |activation| across layers)
    scores = np.max(np.abs(A), axis=0)
    k = max(1, min(int(top_k), scores.shape[0]))
    idxs = np.argsort(scores)[-k:][::-1]

    Z = A[:, idxs].T  # [k, layers]
    x = [str(k) for k in layer_keys]
    y = [f"Neuron {int(i)}" for i in idxs.tolist()]

    if title is None:
        title = "NeurInSpectre — Neural Persistence Heatmap"

    fig = go_.Figure(
        data=go_.Heatmap(
            z=Z,
            x=x,
            y=y,
            colorscale='Viridis',
            hoverongaps=False,
            colorbar=dict(title=dict(text=colorbar_title)),
            hovertemplate=(
                'Layer: %{x}<br>'
                'Neuron: %{y}<br>'
                'Activation: %{z:.4f}<extra></extra>'
            ),
        )
    )

    fig.update_layout(
        template='plotly',
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis=dict(title=dict(text='Layer'), tickangle=45, showgrid=False),
        yaxis=dict(title=dict(text='Neuron Index'), autorange='reversed', showgrid=False, tickfont=dict(size=9)),
        margin=dict(l=100, r=60, t=90, b=470),
        height=950,
        width=1200,
        plot_bgcolor='white',
    )

    # Guidance boxes placed BELOW x-axis using pixel shifts (robust under responsive layout)
    blue_text = (
        "<b>🔵 BLUE TEAM — HOW/WHY</b><br>"
        "<b>WHY</b>: Hot-neuron bands recurring across layers can indicate stable control features/backdoors.<br>"
        "<b>HOW</b>: Run benign suites (<span style='font-family:monospace'>--prompts-file</span>); alert on neurons recurring across prompts + layers.<br>"
        "<b>NEXT</b>: Intersect with suspect prompts; confirm causality; patch/salt or monitor at runtime."
    )
    red_text = (
        "<b>🔴 RED TEAM — HOW/WHY</b><br>"
        "<b>WHY</b>: Persistent hot neurons are candidate internal control points.<br>"
        "<b>HOW</b>: Test prompt variants; seek signatures that persist across prompts/layers.<br>"
        "<b>OPSEC</b>: Avoid one repeatable signature; distribute drift across many neurons."
    )

    # Anchor at bottom of plot area (y=0) then shift down into bottom margin.
    fig.add_annotation(
        text=blue_text,
        xref='paper', yref='paper',
        x=0.5, y=0.0,
        xanchor='center', yanchor='top',
        xshift=0, yshift=-130,
        showarrow=False,
        align='left',
        font=dict(size=12, color='white', family='Arial'),
        bgcolor='rgba(20,60,140,1.0)',
        bordercolor='#3399ff',
        borderwidth=2,
        borderpad=14,
        width=1040,
    )

    fig.add_annotation(
        text=red_text,
        xref='paper', yref='paper',
        x=0.5, y=0.0,
        xanchor='center', yanchor='top',
        xshift=0, yshift=-290,
        showarrow=False,
        align='left',
        font=dict(size=12, color='white', family='Arial'),
        bgcolor='rgba(140,20,20,1.0)',
        bordercolor='#ff3333',
        borderwidth=2,
        borderpad=14,
        width=1040,
    )

    return fig


def plot_attack_patterns(
    original_activations: Mapping[Any, Any],
    modified_activations: Mapping[Any, Any],
    *,
    layer_idx: int,
    top_k: int = 10,
    compare: str = 'prefix',
    title: Optional[str] = None,
    baseline_tokens: Optional[list[str]] = None,
    test_tokens: Optional[list[str]] = None,
) -> "go.Figure":
    """Neuron Activation Changes plot (baseline vs test) for a single layer.

    This restores the historical `attack_patterns.html` concept: compare the
    per-token hidden activations of a chosen layer, flatten token×hidden into a
    single vector, then display the top-K absolute changes.

    Args:
        original_activations: mapping layer id -> hidden states.
        modified_activations: mapping layer id -> hidden states.
        layer_idx: layer index to compare.
        top_k: number of largest changes to display.
        compare: 'prefix' (compare overlapping prefix tokens) or 'last' (compare last-token only).
        title: optional title override.
        baseline_tokens: optional token strings (baseline) for hover.
        test_tokens: optional token strings (test) for hover.

    Returns:
        Plotly Figure.
    """
    go_ = _require_plotly()

    # Resolve layer key
    cand_keys = [f'layer_{int(layer_idx)}', str(int(layer_idx)), int(layer_idx)]
    base_key = next((k for k in cand_keys if k in original_activations), None)
    test_key = next((k for k in cand_keys if k in modified_activations), None)
    if base_key is None or test_key is None:
        raise ValueError(f'Layer {layer_idx} not found in both activation dicts')

    base_mat = _to_seq_hidden(original_activations[base_key])  # [seq, hidden]
    test_mat = _to_seq_hidden(modified_activations[test_key])

    # Align hidden sizes
    hidden = int(min(base_mat.shape[1], test_mat.shape[1]))
    base_mat = base_mat[:, :hidden]
    test_mat = test_mat[:, :hidden]

    mode = str(compare).lower()
    if mode in ('last', 'last_token'):
        base_cmp = base_mat[-1].reshape(1, -1)
        test_cmp = test_mat[-1].reshape(1, -1)
        seq_used = 1
    elif mode in ('prefix', 'overlap'):
        seq_used = int(min(base_mat.shape[0], test_mat.shape[0]))
        base_cmp = base_mat[:seq_used]
        test_cmp = test_mat[:seq_used]
    else:
        raise ValueError("compare must be 'prefix' or 'last'")

    base_flat = base_cmp.reshape(-1)
    test_flat = test_cmp.reshape(-1)

    delta = test_flat - base_flat
    abs_delta = np.abs(delta)

    eps = 1e-8
    changed_total = int(np.sum(abs_delta > eps))
    k = max(1, min(int(top_k), abs_delta.shape[0]))
    top_idx = np.argsort(abs_delta)[-k:][::-1]

    orig_vals = base_flat[top_idx]
    mod_vals = test_flat[top_idx]

    max_d = float(np.max(abs_delta)) if abs_delta.size else 0.0
    mean_d = float(np.mean(abs_delta)) if abs_delta.size else 0.0

    # Build informative labels
    tick_text = []
    hover_info = []
    for flat_i in top_idx.tolist():
        t = int(flat_i // hidden)
        n = int(flat_i % hidden)
        if mode in ('last', 'last_token'):
            label = f"n{n}"
            b_tok = ''
            t_tok = ''
        else:
            label = f"t{t}:n{n}"
            b_tok = baseline_tokens[t] if baseline_tokens and t < len(baseline_tokens) else ''
            t_tok = test_tokens[t] if test_tokens and t < len(test_tokens) else ''
        tick_text.append(label)
        hover_info.append((flat_i, t, n, b_tok, t_tok))

    x = list(range(k))

    if title is None:
        title = (
            f"Neuron Activation Changes (Top {k} of {changed_total} Changed Positions)"
            f"<br><span style='font-size:14px;color:#666'>Layer {int(layer_idx)} | Max Δ: {max_d:.4f} | Mean |Δ|: {mean_d:.4f} | Compared tokens: {seq_used}</span>"
        )

    fig = go_.Figure()

    fig.add_trace(
        go_.Bar(
            x=x,
            y=orig_vals,
            name='Baseline',
            opacity=0.75,
            marker=dict(color='#1f77b4'),
            customdata=hover_info,
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'pos t=%{customdata[1]} n=%{customdata[2]}<br>'
                'baseline token: %{customdata[3]}<br>'
                'test token: %{customdata[4]}<br>'
                'baseline: %{y:.4f}<extra></extra>'
            ),
        )
    )

    fig.add_trace(
        go_.Bar(
            x=x,
            y=mod_vals,
            name='Test',
            opacity=0.75,
            marker=dict(color='#ff7f0e'),
            customdata=hover_info,
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'pos t=%{customdata[1]} n=%{customdata[2]}<br>'
                'baseline token: %{customdata[3]}<br>'
                'test token: %{customdata[4]}<br>'
                'test: %{y:.4f}<extra></extra>'
            ),
        )
    )

    # Delta connector lines + delta annotations
    shapes = []
    ann = []
    for i, (y0, y1, info) in enumerate(zip(orig_vals, mod_vals, hover_info)):
        shapes.append(
            dict(
                type='line', xref='x', yref='y',
                x0=float(i)-0.4, x1=float(i)+0.4,
                y0=float(y0), y1=float(y1),
                line=dict(color='red', width=2, dash='dash'),
            )
        )
        d = float(y1 - y0)
        denom = abs(float(y0)) + 1e-9
        pct = d / denom * 100.0
        color = 'red' if d > 0 else 'green'
        ann.append(
            dict(
                x=float(i),
                y=float(max(y0, y1)) + (0.05 * max(1e-6, abs(float(max(y0, y1))))),
                xref='x', yref='y',
                text=f"Δ{d:+.2f} ({pct:+.1f}%)",
                showarrow=False,
                font=dict(size=10, color=color),
                bgcolor='rgba(255,255,255,0.75)',
                bordercolor='#ccc',
                borderwidth=1,
                borderpad=2,
            )
        )

    fig.update_layout(
        template='plotly',
        title=dict(text=title, x=0.5, xanchor='center'),
        barmode='group',
        xaxis=dict(
            title=dict(text='Position (token/neuron)'),
            tickmode='array',
            tickvals=x,
            ticktext=tick_text,
            tickangle=45,
        ),
        yaxis=dict(title=dict(text='Activation Value')),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=80, r=80, t=110, b=320),
        height=650,
        width=1000,
        shapes=shapes,
        annotations=ann,
    )

    # Red/Blue team guidance boxes (kept below x-axis).
    blue_text = (
        "<b>🔵 BLUE TEAM — HOW/WHY</b><br>"
        "• <b>WHY</b>: Concentrated Δ on few positions can indicate targeted steering/backdoor features<br>"
        "• <b>HOW</b>: Re-run across a prompt suite; if the <i>same</i> positions recur, treat as a control point<br>"
        "• <b>NEXT</b>: Validate with <code>activation_anomaly_detection</code>; mitigate via conditional steering/salting/patching"
    )
    red_text = (
        "<b>🔴 RED TEAM — HOW/WHY</b><br>"
        "• <b>WHY</b>: Large Δ concentrated in few positions suggests strong leverage<br>"
        "• <b>HOW</b>: Test transfer across prompts; avoid leaving a single repeatable signature"
    )

    fig.add_annotation(
        text=blue_text,
        xref='paper', yref='paper',
        x=0.01, y=-0.28,
        showarrow=False,
        font=dict(size=10, color='white', family='Arial'),
        align='left',
        bgcolor='rgba(20,60,140,0.92)',
        bordercolor='#3399ff',
        borderwidth=2,
        borderpad=10,
        xanchor='left',
        yanchor='top',
        width=420,
    )

    fig.add_annotation(
        text=red_text,
        xref='paper', yref='paper',
        x=0.99, y=-0.28,
        showarrow=False,
        font=dict(size=10, color='white', family='Arial'),
        align='left',
        bgcolor='rgba(140,20,20,0.92)',
        bordercolor='#ff3333',
        borderwidth=2,
        borderpad=10,
        xanchor='right',
        yanchor='top',
        width=420,
    )

    return fig
