"""Co-attention fusion trace visualization.

This visualization mirrors the historical "Original Traces" vs "Fused Traces"
side-by-side plot.

Given two sequences of feature vectors A[t] and B[t] (e.g., token-level hidden
states from a chosen layer), we compute a co-attention alignment between time
steps and fuse each trace with a context vector derived from the other trace.

This is useful for security analysis of multimodal / multi-source pipelines
(RAG, tool transcripts, document + user instruction mixtures) because it exposes
*where* a representation becomes sensitive to cross-source alignment.

Important:
- This is not a standalone "maliciousness detector".
- It is best used comparatively (benign vs injection-style prompts; different
  wrappers; different layer selections).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    denom = np.sum(e, axis=axis, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return e / denom


def co_attention_fuse(
    trace_a: np.ndarray,
    trace_b: np.ndarray,
    *,
    alpha: float = 0.5,
    temperature: float = 0.25,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Fuse two traces with a simple cosine-sim co-attention.

    Args:
        trace_a: [T_a, D]
        trace_b: [T_b, D]
        alpha: fusion strength in [0, 1]
        temperature: softmax temperature (>0). Smaller -> sharper alignments.

    Returns:
        fused_a: [T_a, D]
        fused_b: [T_b, D]
        info: dict with similarity and attention weights
    """

    A = np.asarray(trace_a, dtype=np.float64)
    B = np.asarray(trace_b, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError('trace_a and trace_b must be 2D arrays [T, D]')

    if A.shape[0] < 2 or B.shape[0] < 2:
        raise ValueError('Need at least 2 time steps in each trace')

    D = int(min(A.shape[1], B.shape[1]))
    A = A[:, :D]
    B = B[:, :D]

    # Cosine similarity matrix
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + float(eps))
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + float(eps))

    sim = An @ Bn.T  # [T_a, T_b]

    temp = float(max(float(temperature), 1e-6))
    sim_scaled = sim / temp

    w_a2b = _softmax(sim_scaled, axis=1)  # each row sums to 1
    w_b2a = _softmax(sim_scaled.T, axis=1)  # [T_b, T_a]

    ctx_a = w_a2b @ B
    ctx_b = w_b2a @ A

    a = float(np.clip(alpha, 0.0, 1.0))
    fused_a = (1.0 - a) * A + a * ctx_a
    fused_b = (1.0 - a) * B + a * ctx_b

    info = {
        'similarity': sim,
        'a2b': w_a2b,
        'b2a': w_b2a,
    }
    return fused_a.astype(np.float64), fused_b.astype(np.float64), info


def _scale_group(values: List[np.ndarray], *, mode: str = 'tanh_z', eps: float = 1e-12) -> List[np.ndarray]:
    """Scale a group of 1D series with shared statistics.

    Modes:
    - none: no scaling
    - zscore: standard z-score
    - tanh_z: z-score then tanh (keeps [-1, 1] range-ish)
    - minmax: scale to [-1, 1]
    """

    m = str(mode or 'tanh_z').lower()
    xs = [np.asarray(v, dtype=np.float64) for v in values]
    if m == 'none':
        return xs

    cat = np.concatenate(xs, axis=0)
    if cat.size == 0:
        return xs

    if m == 'minmax':
        lo = float(np.min(cat))
        hi = float(np.max(cat))
        den = max(hi - lo, float(eps))
        return [((v - lo) / den) * 2.0 - 1.0 for v in xs]

    mu = float(np.mean(cat))
    sd = float(np.std(cat))
    sd = max(sd, float(eps))
    out = [(v - mu) / sd for v in xs]
    if m == 'tanh_z':
        out = [np.tanh(v) for v in out]
    return out


@dataclass
class CoAttentionTracesMetrics:
    title: str
    strategy: str

    model: Optional[str]
    tokenizer: Optional[str]
    layer: Optional[int]

    prompt_a: Optional[str]
    prompt_b: Optional[str]
    prompt_a_sha16: Optional[str]
    prompt_b_sha16: Optional[str]

    seq_len: int
    feature: int
    feature2: Optional[int]
    scale: str

    alpha: float
    temperature: float

    x: List[int]

    # original
    a_f1: List[float]
    b_f1: List[float]
    a_f2: Optional[List[float]] = None
    b_f2: Optional[List[float]] = None

    # fused
    fa_f1: List[float] = None  # type: ignore
    fb_f1: List[float] = None  # type: ignore
    fa_f2: Optional[List[float]] = None
    fb_f2: Optional[List[float]] = None

    subtitle: Optional[str] = None


def plot_co_attention_traces(
    metrics: CoAttentionTracesMetrics,
    *,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Render the co-attention traces plot (original vs fused)."""

    import matplotlib.pyplot as plt

    x = np.asarray(metrics.x, dtype=np.int64)

    a1 = np.asarray(metrics.a_f1, dtype=np.float64)
    b1 = np.asarray(metrics.b_f1, dtype=np.float64)
    fa1 = np.asarray(metrics.fa_f1, dtype=np.float64)
    fb1 = np.asarray(metrics.fb_f1, dtype=np.float64)

    has_f2 = metrics.feature2 is not None and metrics.a_f2 is not None and metrics.b_f2 is not None and metrics.fa_f2 is not None and metrics.fb_f2 is not None

    if has_f2:
        a2 = np.asarray(metrics.a_f2, dtype=np.float64)
        b2 = np.asarray(metrics.b_f2, dtype=np.float64)
        fa2 = np.asarray(metrics.fa_f2, dtype=np.float64)
        fb2 = np.asarray(metrics.fb_f2, dtype=np.float64)

    # Layout
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), dpi=160)
    ax0, ax1p = axes

    feat_label = f"feat {int(metrics.feature)}"
    if metrics.feature2 is not None:
        feat2_label = f"feat {int(metrics.feature2)}"
    else:
        feat2_label = ""

    def _panel(ax, y_a1, y_b1, y_a2, y_b2, *, panel_title: str, a_name: str, b_name: str):
        ax.grid(True, alpha=0.35)
        ax.set_xlim(int(x[0]), int(x[-1]))
        ax.plot(x, y_a1, color='#1f77b4', linewidth=2.0, label=f"{a_name} ({feat_label})")
        ax.plot(x, y_b1, color='#ff7f0e', linewidth=2.0, label=f"{b_name} ({feat_label})")
        ax.set_xlabel('Time / token index', labelpad=-2)
        ax.set_ylabel(feat_label)

        # Optional secondary feature on right axis
        ax2p = None
        if y_a2 is not None and y_b2 is not None:
            ax2p = ax.twinx()
            ax2p.plot(x, y_a2, color='green', linestyle=':', linewidth=1.8, label=f"{a_name} ({feat2_label})")
            ax2p.plot(x, y_b2, color='purple', linestyle=':', linewidth=1.8, label=f"{b_name} ({feat2_label})")
            ax2p.set_ylabel(feat2_label)

        # Combined legend placed below to avoid covering traces
        handles, labels = ax.get_legend_handles_labels()
        if ax2p is not None:
            h2, l2 = ax2p.get_legend_handles_labels()
            handles = list(handles) + list(h2)
            labels = list(labels) + list(l2)

        ax.legend(
            handles,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            framealpha=0.9,
            fontsize=9,
        )

        ax.set_title(panel_title + "\n" + str(metrics.strategy), fontsize=13, fontweight='bold')

    _panel(
        ax0,
        a1,
        b1,
        a2 if has_f2 else None,
        b2 if has_f2 else None,
        panel_title=f"Original Traces ({feat_label})",
        a_name='Trace A',
        b_name='Trace B',
    )

    _panel(
        ax1p,
        fa1,
        fb1,
        fa2 if has_f2 else None,
        fb2 if has_f2 else None,
        panel_title=f"Fused Traces ({feat_label})",
        a_name='Fused A',
        b_name='Fused B',
    )

    # Peak divergence marker (scaled traces): where fused deviates most from original
    peak_step = None
    try:
        d = np.abs(fa1 - a1) + np.abs(fb1 - b1)
        peak_step = int(x[int(np.argmax(d))])
        for _ax in (ax0, ax1p):
            _ax.axvline(peak_step, color='black', linestyle='--', linewidth=1.0, alpha=0.28)
    except Exception:
        peak_step = None

    fig_title = title or metrics.title
    if 'neurinspectre' not in str(fig_title).lower():
        fig_title = f"NeurInSpectre — {fig_title}"
    if metrics.subtitle:
        fig.suptitle(fig_title + "\n" + metrics.subtitle, fontsize=15, y=0.98)
    else:
        fig.suptitle(fig_title, fontsize=15, y=0.98)

    blue = (
        'Blue team next steps:\n'
        '• Start at dashed peak Δ (vertical line).\n'
        '• Map span via: peak_divergence_a2b_pair\n'
        '  (fallback: alignment_pairs).\n'
        '• Sweep layers; pivot: AGA + FFT + TTD.\n'
        '• Validate: peak Δ shrinks/shifts after mitigations.'
    )
    red = (
        'Red team (authorized) next steps:\n'
        '• Run sanitized IPI suites across wrappers/languages.\n'
        '• Record stable layer band + peak Δ window.\n'
        '• Stress-test multi-step tool workflows safely.\n'
        '• Report prompt SHA16s + mitigation deltas.'
    )

    fig.text(
        0.01,
        0.01,
        blue,
        ha='left',
        va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#1f77b4', alpha=0.95),
    )
    fig.text(
        0.63,
        0.01,
        red,
        ha='left',
        va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#d62728', alpha=0.95),
    )

    fig.tight_layout(rect=[0, 0.32, 1, 0.94])

    if out_path:
        Path2 = __import__('pathlib').Path
        out = Path2(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        return str(out)

    return ''



def plot_co_attention_traces_interactive(
    metrics: CoAttentionTracesMetrics,
    *,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    '''Write an interactive Plotly HTML version of the traces plot.'''

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x = list(metrics.x)

    feat_label = f"feat {int(metrics.feature)}"
    has_f2 = metrics.feature2 is not None and metrics.a_f2 is not None and metrics.b_f2 is not None and metrics.fa_f2 is not None and metrics.fb_f2 is not None
    feat2_label = f"feat {int(metrics.feature2)}" if metrics.feature2 is not None else ""

    st = str(metrics.strategy)
    subplot_titles = (
        f"Original Traces ({feat_label})<br><span style='font-size:12px'>{st}</span>",
        f"Fused Traces ({feat_label})<br><span style='font-size:12px'>{st}</span>",
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.10,
    )

    # Left: original
    fig.add_trace(
        go.Scatter(x=x, y=metrics.a_f1, mode='lines', name=f"Trace A ({feat_label})", line=dict(color='#1f77b4', width=2.2)),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=metrics.b_f1, mode='lines', name=f"Trace B ({feat_label})", line=dict(color='#ff7f0e', width=2.2)),
        row=1,
        col=1,
        secondary_y=False,
    )

    if has_f2:
        fig.add_trace(
            go.Scatter(x=x, y=metrics.a_f2, mode='lines', name=f"Trace A ({feat2_label})", line=dict(color='green', width=2.0, dash='dot')),
            row=1,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=x, y=metrics.b_f2, mode='lines', name=f"Trace B ({feat2_label})", line=dict(color='purple', width=2.0, dash='dot')),
            row=1,
            col=1,
            secondary_y=True,
        )

    # Right: fused
    fig.add_trace(
        go.Scatter(x=x, y=metrics.fa_f1, mode='lines', name=f"Fused A ({feat_label})", line=dict(color='#1f77b4', width=2.2)),
        row=1,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=metrics.fb_f1, mode='lines', name=f"Fused B ({feat_label})", line=dict(color='#ff7f0e', width=2.2)),
        row=1,
        col=2,
        secondary_y=False,
    )

    if has_f2:
        fig.add_trace(
            go.Scatter(x=x, y=metrics.fa_f2, mode='lines', name=f"Fused A ({feat2_label})", line=dict(color='green', width=2.0, dash='dot')),
            row=1,
            col=2,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=x, y=metrics.fb_f2, mode='lines', name=f"Fused B ({feat2_label})", line=dict(color='purple', width=2.0, dash='dot')),
            row=1,
            col=2,
            secondary_y=True,
        )

    # Axes titles
    fig.update_xaxes(title_text='Time / token index', row=1, col=1)
    fig.update_xaxes(title_text='Time / token index', row=1, col=2)

    fig.update_yaxes(title_text=feat_label, row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text=feat_label, row=1, col=2, secondary_y=False)

    if has_f2:
        fig.update_yaxes(title_text=feat2_label, row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text=feat2_label, row=1, col=2, secondary_y=True)

    fig_title = title or metrics.title
    if 'neurinspectre' not in str(fig_title).lower():
        fig_title = f"NeurInSpectre — {fig_title}"
    if metrics.subtitle:
        title_text = f"{fig_title}<br><span style='font-size:12px'>{metrics.subtitle}</span>"
    else:
        title_text = fig_title

    # Keep legends out of the plot area (no overlap) and use unified hover.
    fig.update_layout(
        template='plotly_white',
        title=dict(text=title_text, x=0.5, xanchor='center'),
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center', yanchor='top'),
        margin=dict(l=60, r=50, t=115, b=260),
        height=700,
    )

    # Peak divergence marker (scaled traces): where fused deviates most from original
    peak_x = None
    try:
        import numpy as _np

        d = _np.abs(_np.asarray(metrics.fa_f1, dtype=_np.float64) - _np.asarray(metrics.a_f1, dtype=_np.float64))
        d = d + _np.abs(_np.asarray(metrics.fb_f1, dtype=_np.float64) - _np.asarray(metrics.b_f1, dtype=_np.float64))
        peak_idx = int(_np.argmax(d))
        if 0 <= peak_idx < len(x):
            peak_x = x[peak_idx]
    except Exception:
        peak_x = None

    if peak_x is not None:
        for c in (1, 2):
            fig.add_vline(
                x=peak_x,
                row=1,
                col=c,
                line_width=1,
                line_dash='dash',
                line_color='rgba(0,0,0,0.35)',
            )

    blue_html = (
        "<b>Blue team next steps</b><br>"
        "• Start at the <i>vertical dashed line</i> (peak Δ).<br>"
        "• Map span with: peak_divergence_a2b_pair<br>"
        "&nbsp;&nbsp;(fallback: alignment_pairs).<br>"
        "• Sweep layers; pivot: AGA + FFT + TTD.<br>"
        "• Validate: peak Δ shrinks/shifts after mitigations."
    )
    red_html = (
        "<b>Red team (authorized) next steps</b><br>"
        "• Run sanitized IPI suites across wrappers/languages.<br>"
        "• Record stable layer band + peak Δ window.<br>"
        "• Stress-test multi-step tool workflows safely.<br>"
        "• Report prompt SHA16s + mitigation deltas."
    )

    fig.add_annotation(
        x=0.0,
        y=-0.30,
        xref='paper',
        yref='paper',
        xanchor='left',
        yanchor='top',
        text=blue_html,
        showarrow=False,
        align='left',
        bordercolor='rgba(31,119,180,0.85)',
        borderwidth=1,
        bgcolor='rgba(31,119,180,0.06)',
        font=dict(size=11),
    )
    fig.add_annotation(
        x=1.0,
        y=-0.30,
        xref='paper',
        yref='paper',
        xanchor='right',
        yanchor='top',
        text=red_html,
        showarrow=False,
        align='left',
        bordercolor='rgba(214,39,40,0.85)',
        borderwidth=1,
        bgcolor='rgba(214,39,40,0.06)',
        font=dict(size=11),
    )

    if out_path:
        out = __import__('pathlib').Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out))
        return str(out)

    return ''
