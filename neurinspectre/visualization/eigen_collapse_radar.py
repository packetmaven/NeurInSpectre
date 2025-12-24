"""Eigen‑Collapse Rank Shrinkage Radar.

This visualization summarizes the *spectral geometry* of per-layer hidden states.
It is useful for spotting **representation collapse** (low effective rank / strong
anisotropy), which can correlate with abnormal generation regimes and certain
classes of adversarial prompt/interaction patterns.

No simulation:
- This module never generates synthetic data. Callers must provide real hidden
  states or precomputed eigenvalue metrics.

Technical definition (what we compute):
- For each layer ℓ, take hidden states H_ℓ ∈ R^{T×D} (T tokens, D hidden dims).
- Center across tokens: X = H_ℓ − mean_t(H_ℓ)
- Compute top-k eigenvalues of the token covariance C = XᵀX/(T−1).
  Using SVD of X (economy SVD), eigenvalues(C) = s²/(T−1).
- Normalize (default): divide by eig1 so eig1=1 and the remaining axes show
  relative shrinkage ("petal size").

Interpretation:
- Small petals (eig2..eigk ≪ eig1) indicate a more rank‑collapsed / anisotropic
  representation at that layer.

This is a **triage view**: it tells you *which layers* to drill into next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np


NormalizeMode = Literal['eig1', 'sum', 'none']


def _to_numpy(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 0:
        raise ValueError('Expected array-like, got scalar')
    if not np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float32)
    if np.any(~np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32, copy=False)


def topk_cov_eigvals(hidden: Any, *, k: int = 5) -> np.ndarray:
    """Top-k eigenvalues of token covariance for a single layer.

    hidden: array-like shaped [seq, hidden] (or [batch, seq, hidden]).
    Returns shape [k].
    """
    h = _to_numpy(hidden)
    if h.ndim == 3:
        h = h[0]
    if h.ndim != 2:
        raise ValueError(f'Expected [seq, hidden] array, got shape={h.shape}')

    t, d = int(h.shape[0]), int(h.shape[1])
    k = int(max(1, k))

    if t < 2 or d < 1:
        return np.zeros((k,), dtype=np.float32)

    x = h - h.mean(axis=0, keepdims=True)

    # Economy SVD: X is [T, D] with typically T << D (fast)
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    eig = (s ** 2) / float(max(t - 1, 1))
    eig = eig[:k]
    if eig.size < k:
        eig = np.pad(eig, (0, k - eig.size), constant_values=0.0)
    return eig.astype(np.float32, copy=False)


def normalize_eigvals(eig: np.ndarray, *, mode: NormalizeMode = 'eig1') -> np.ndarray:
    e = np.asarray(eig, dtype=np.float32)
    if e.size == 0:
        return e

    if mode == 'none':
        return e

    denom = float(e[0]) if mode == 'eig1' else float(np.sum(e))
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(e)
    return (e / denom).astype(np.float32, copy=False)


@dataclass
class EigenCollapseRadarMetrics:
    model: str
    layers: List[int]
    k: int
    normalize: NormalizeMode
    eigvals: List[List[float]]  # shape [n_layers, k]
    subtitle: Optional[str] = None


def plot_eigen_collapse_radar(
    metrics: EigenCollapseRadarMetrics,
    *,
    title: str = 'Eigen-Collapse Rank Shrinkage Radar',
    out_path: Optional[str] = None,
    guidance: bool = True,
) -> str:
    """Render radar chart (matplotlib) and optionally save PNG."""
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.gridspec import GridSpec  # type: ignore

    layers = list(metrics.layers)
    if not layers:
        raise ValueError('No layers provided')

    k = int(metrics.k)
    if k <= 0:
        raise ValueError('k must be >= 1')

    vals = np.asarray(metrics.eigvals, dtype=np.float32)
    if vals.ndim != 2 or vals.shape[1] != k:
        raise ValueError(f'Expected eigvals shape [n_layers, k], got {vals.shape} for k={k}')

    # Angles for radar
    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])
    labels = [f'eig{i+1}' for i in range(k)]

    # Layout: polar plot + optional footer guidance
    fig = plt.figure(figsize=(14, 9 if guidance else 7.5))
    gs = GridSpec(2, 1, height_ratios=[3.3, 1.2] if guidance else [1, 0.0001], hspace=0.10)

    ax = fig.add_subplot(gs[0], projection='polar')
    ax.set_facecolor('#e9e9f2')

    footer_ax = None
    if guidance:
        footer_ax = fig.add_subplot(gs[1])
        footer_ax.axis('off')

    # Style
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(['0.00', '0.25', '0.50', '0.75', '1.00'])
    ax.set_rlabel_position(18)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=14)
    ax.grid(True, alpha=0.55)

    # Colors: stable cycle for up to many layers
    cmap = plt.get_cmap('tab20')

    for i, layer in enumerate(layers):
        v = vals[i]
        v_closed = np.concatenate([v, v[:1]])
        color = cmap(i % 20)
        ax.plot(angles_closed, v_closed, color=color, linewidth=2.5, alpha=0.85, label=f'Layer {layer}')

    # Title/subtitle
    model_short = str(metrics.model).split('/')[-1]
    subtitle = metrics.subtitle or f"{model_short} | k={k} eigenvalues"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=22, y=0.97)

    # Legend on the right
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.10), framealpha=0.95)

    # Small footnote (always)
    fig.text(
        0.5,
        0.035 if guidance else 0.02,
        "Red: small petals -> low-rank/anisotropy; Blue: investigate if petals shrink vs baseline.",
        ha='center',
        fontsize=11,
        color='#666',
    )

    if guidance and footer_ax is not None:
        blue = """BLUE TEAM - HOW/WHY
WHY: Petal shrinkage (eig2..k << eig1) is a fast proxy for representation collapse / OOD regimes.
HOW: Baseline per model+prompt suite; alert on layers whose petals shrink over time or vs benign baseline.
NEXT: Drill into flagged layers (attack_patterns / attention heads); consider regularization/orthogonalization + runtime monitors."""

        red = """RED TEAM (authorized) - HOW/WHY
WHY: Collapsed spectra can indicate narrow internal channels; they may correlate with brittle control surfaces.
HOW: Use this as a measurement view across layers/prompts; test whether techniques produce concentrated shrinkage.
OPSEC: Avoid a single dominant layer signature; test transfer across paraphrases and contexts."""

        footer_ax.text(
            0.01,
            0.96,
            blue,
            ha='left',
            va='top',
            fontsize=10,
            color='white',
            transform=footer_ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#143c8c', edgecolor='#3399ff', linewidth=2),
        )
        footer_ax.text(
            0.01,
            0.05,
            red,
            ha='left',
            va='bottom',
            fontsize=10,
            color='white',
            transform=footer_ax.transAxes,
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
