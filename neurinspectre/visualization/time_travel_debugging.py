"""Time‑Travel Debugging — layer-wise activation Δ and attention variance.

This restores a common internal-debugging view used in LLM/VLM security work:
identify **which layers** show the largest representation shift (activation Δ)
and **where attention becomes unstable / unusually concentrated**.

Design goals:
- **No simulation**: callers must provide real hidden states/attentions.
- **Operational**: produces a compact layer-wise triage chart and a JSON summary
  that can be diffed across runs.

Definitions (technical precision):
- Activation Δ (L1) for layer ℓ:

    Δ_ℓ = mean_t || h_test[ℓ,t] − h_base[ℓ,t] ||_1

  where token positions are aligned by index over the shared prefix (truncate to
  min sequence length).

- Attention variance (default): for layer ℓ attention probabilities A[head,q,k]

    V_ℓ = mean_{head,q} Var_k(A[head,q,k]) * (seq_len^2)

  The seq_len^2 factor puts values into a readable range while preserving the
  ordering; it also makes the scale less sensitive to seq_len.

Notes:
- Attention outputs are post-softmax probabilities from HuggingFace models.
- This chart is **triage**: it tells you where to drill down (layer/window).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np


AttnVarMode = Literal['per_query', 'global']
AttnVarScale = Literal['seq2', 'seq', 'none']
DeltaMode = Literal['token_l1_mean', 'token_l1_mean_x100', 'mean_vec_l1', 'mean_vec_l1_x100']


def _to_numpy(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 0:
        raise ValueError('Expected array-like, got scalar')
    if not np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float32)
    if np.any(~np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32, copy=False)


def activation_delta_l1(
    h_base: Any,
    h_test: Any,
    *,
    mode: DeltaMode = 'token_l1_mean_x100',
) -> float:
    """Compute activation Δ between baseline and test hidden states.

    Inputs are expected as [seq, hidden] arrays (or compatible array-like).

    Modes:
    - token_l1_mean: mean_t sum_d |Δ|
    - token_l1_mean_x100: 100 * mean_{t,d} |Δ|   (more interpretable across hidden sizes)
    - mean_vec_l1: sum_d | mean_t(h_test) - mean_t(h_base) |
    - mean_vec_l1_x100: 100 * mean_d | mean_t(h_test) - mean_t(h_base) |
    """
    hb = _to_numpy(h_base)
    ht = _to_numpy(h_test)
    if hb.ndim != 2 or ht.ndim != 2:
        raise ValueError(f'Expected [seq, hidden] arrays, got {hb.shape} and {ht.shape}')

    if mode.startswith('mean_vec'):
        vb = hb.mean(axis=0)
        vt = ht.mean(axis=0)
        diff = np.abs(vt - vb)
        if mode.endswith('_x100'):
            return float(np.mean(diff) * 100.0)
        return float(np.sum(diff))

    # token-aligned modes
    n = int(min(hb.shape[0], ht.shape[0]))
    if n <= 0:
        return 0.0
    diff = np.abs(ht[:n] - hb[:n])

    if mode.endswith('_x100'):
        return float(np.mean(diff) * 100.0)

    token_l1 = np.sum(diff, axis=1)  # [n]
    return float(np.mean(token_l1))
def attention_variance(
    attn: Any,
    *,
    mode: AttnVarMode = 'per_query',
    scale: AttnVarScale = 'seq2',
) -> float:
    """Compute an attention concentration/variance proxy from attention probabilities.

    Expects attention probabilities for a single layer:
    - [heads, seq, seq] or [batch, heads, seq, seq]

    Returns a scalar (possibly scaled for readability).
    """
    a = _to_numpy(attn)
    if a.ndim == 4:
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f'Expected [heads, seq, seq] attention, got shape={a.shape}')

    seq_len = int(a.shape[-1])
    if seq_len <= 0:
        return 0.0

    if mode == 'global':
        v = float(np.var(a))
    else:
        # per_query: Var over keys for each (head, query)
        v = float(np.mean(np.var(a, axis=-1)))

    if scale == 'seq2':
        v *= float(seq_len * seq_len)
    elif scale == 'seq':
        v *= float(seq_len)
    return float(v)


@dataclass
class TimeTravelMetrics:
    layers: List[int]
    activation_delta_l1: List[float]
    attention_variance: List[float]
    attention_variance_baseline: Optional[List[float]] = None
    delta_mode: DeltaMode = 'token_l1_mean'
    attn_var_mode: AttnVarMode = 'per_query'
    attn_var_scale: AttnVarScale = 'seq2'


def plot_time_travel_debugging(
    metrics: TimeTravelMetrics,
    *,
    title: str = 'Time-Travel Debugging – Layer-wise Activation Δ & Attention Variance',
    out_path: Optional[str] = None,
    blue_text: Optional[str] = None,
    red_text: Optional[str] = None,
) -> str:
    """Render the Matplotlib chart (screenshot-style) and optionally save it."""
    import matplotlib.pyplot as plt  # type: ignore

    layers = list(metrics.layers)
    if not layers:
        raise ValueError('No layers to plot')

    deltas = np.asarray(metrics.activation_delta_l1, dtype=np.float32)
    av = np.asarray(metrics.attention_variance, dtype=np.float32)

    x = np.arange(len(layers))
    labels = [f'layer_{i}' for i in layers]

    fig, (ax1, ax_footer) = plt.subplots(
        nrows=2,
        figsize=(14, 8),
        gridspec_kw={'height_ratios': [3.3, 1.3], 'hspace': 0.25},
    )
    ax_footer.axis('off')
    ax2 = ax1.twinx()

    bars = ax1.bar(x, deltas, width=0.55, color='#1f77b4', label='Activation Δ (L1)')
    ax2.plot(x, av, color='#d62728', marker='o', linewidth=2.0, label='Attention variance')

    # Mark maxima for fast triage (smart offsets avoid title/edge overlap)
    try:
        def _smart_annotate(ax, *, xi: float, yi: float, text: str, color: str, idx: int, n: int, force_down: bool = False):
            # Horizontal: flip alignment near the right edge
            if idx >= int(0.7 * max(1, n - 1)):
                xoff, ha = -14, 'right'
            else:
                xoff, ha = 14, 'left'

            # Vertical: default auto; optionally force label downward (avoids title overlap)
            if force_down:
                yoff, va = -32, 'top'
            else:
                ymin, ymax = ax.get_ylim()
                span = float(ymax - ymin) if float(ymax) > float(ymin) else 1.0
                if yi >= float(ymin) + 0.85 * span:
                    yoff, va = -28, 'top'
                else:
                    yoff, va = 18, 'bottom'

            ax.annotate(
                text,
                xy=(float(xi), float(yi)),
                xycoords='data',
                textcoords='offset points',
                xytext=(xoff, yoff),
                ha=ha,
                va=va,
                fontsize=9,
                color=color,
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='none', alpha=0.78),
                arrowprops=dict(arrowstyle='->', color=color, alpha=0.65, linewidth=1.2),
            )

        npts = int(len(layers))

        i_d = int(np.argmax(deltas))
        ax1.scatter([x[i_d]], [float(deltas[i_d])], color='#0b4f8a', zorder=5)
        _smart_annotate(
            ax1,
            xi=float(x[i_d]),
            yi=float(deltas[i_d]),
            text=f"max Δ\nlayer_{layers[i_d]}",
            color='#0b4f8a',
            idx=i_d,
            n=npts,
        )

        i_a = int(np.argmax(av))
        ax2.scatter([x[i_a]], [float(av[i_a])], color='#a11b1b', zorder=5)
        _smart_annotate(
            ax2,
            xi=float(x[i_a]),
            yi=float(av[i_a]),
            text=f"max var\nlayer_{layers[i_a]}",
            color='#a11b1b',
            idx=i_a,
            n=npts,
            force_down=True,
        )
    except Exception:
        pass

    ax1.set_title(title, fontsize=16)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Activation Δ (L1)', color='#1f77b4', fontsize=12)
    ax2.set_ylabel('Attention variance', color='#d62728', fontsize=12)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)

    # Combined legend (upper right)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', framealpha=0.9)

    # Footnote on spike interpretation + scaling
    fig.text(
        0.5,
        0.93,
        f"Δ mode={metrics.delta_mode} | attn_var={metrics.attn_var_mode} × {metrics.attn_var_scale}",
        ha='center',
        fontsize=10,
        color='#444',
    )

    # Guidance boxes (optional)
    if blue_text is None:
        blue_text = """🔵 BLUE TEAM — HOW/WHY
WHY: Layer spikes (Δ↑) + attention instability can indicate prompt-injection control surfaces.
HOW: Start with top-Δ layers; rerun adjacent layers and compare to benign baselines.
NEXT: Constrain actions/tools; add per-layer monitors and drill down with activation_attack_patterns."""

    if red_text is None:
        red_text = """🔴 RED TEAM — HOW/WHY
WHY: High-Δ layers often reflect where an injection steers internal representations.
HOW: Target layers with consistent Δ peaks; avoid single-layer/step spikes that are easy to fingerprint.
OPSEC: Test transfer across paraphrases and context shifts; watch for attention-concentration artifacts."""

    # Guidance boxes rendered in a dedicated footer axis (avoids cropping/overlap)
    ax_footer.text(
        0.01,
        0.98,
        blue_text,
        ha='left',
        va='top',
        fontsize=10,
        color='white',
        transform=ax_footer.transAxes,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#143c8c', edgecolor='#3399ff', linewidth=2),
    )
    ax_footer.text(
        0.01,
        0.02,
        red_text,
        ha='left',
        va='bottom',
        fontsize=10,
        color='white',
        transform=ax_footer.transAxes,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#8c1414', edgecolor='#ff3333', linewidth=2),
    )

    if out_path:
        Path = __import__('pathlib').Path
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        return str(out)

    return ''
