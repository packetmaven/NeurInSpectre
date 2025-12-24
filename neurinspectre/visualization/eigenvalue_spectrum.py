"""Eigenvalue Spectrum (Histogram) — security-relevant anomaly triage.

This visualization plots the distribution of covariance eigenvalues for per-layer hidden
states, which is a compact view of *spectral geometry*.

Real-data only:
- Callers must provide real hidden states or compute them from a real model.

Definition:
- For a given layer ℓ, take hidden states H_ℓ ∈ R^{T×D} (T tokens, D hidden dims)
- Center across tokens: X = H_ℓ − mean_t(H_ℓ)
- Token covariance: C = XᵀX/(T−1)
- Eigenvalues(C) = s²/(T−1) where s are the singular values of X (economy SVD)

Interpretation (triage):
- Mean/variance drift or multi-modal / spiky spectra can indicate regime changes,
  distribution shift, or unusual control surfaces.
- This is not a standalone detector. It is a *measurement view* that pairs well with:
  attention heatmaps, AGA, FFT security spectrum, and eigen-collapse radar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import html
import textwrap

import numpy as np

_EPS = 1e-12


def _to_numpy(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 0:
        raise ValueError('Expected array-like, got scalar')
    if not np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float32)
    if np.any(~np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32, copy=False)


def cov_eigenvalues(hidden: Any) -> np.ndarray:
    """Return covariance eigenvalues for a single layer.

    hidden: array-like shaped [seq, hidden] or [batch, seq, hidden]
    Returns shape [min(seq, hidden)].
    """
    h = _to_numpy(hidden)
    if h.ndim == 3:
        h = h[0]
    if h.ndim != 2:
        raise ValueError(f'Expected [seq, hidden] array, got shape={h.shape}')

    t, d = int(h.shape[0]), int(h.shape[1])
    if t < 2 or d < 1:
        return np.zeros((0,), dtype=np.float32)

    x = h - h.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    eig = (s ** 2) / float(max(t - 1, 1))
    return eig.astype(np.float32, copy=False)


def transform_eigs(values: Sequence[float], x_scale: str) -> np.ndarray:
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return v
    if x_scale == 'log10':
        v = np.maximum(v, 0.0)
        return np.log10(v + _EPS)
    return v


def _gini_positive(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    v = v[v >= 0]
    if v.size == 0:
        return 0.0
    s = float(v.sum())
    if s <= 0:
        return 0.0
    v = np.sort(v)
    n = int(v.size)
    cum = np.cumsum(v)
    g = (n + 1 - 2.0 * float(np.sum(cum)) / s) / float(n)
    return float(np.clip(g, 0.0, 1.0))


def summarize_eigenvalues(eig: Sequence[float]) -> Dict[str, float]:
    """Summary stats + spectral geometry features."""
    v = np.asarray(list(eig), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            'count': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'q05': 0.0,
            'q50': 0.0,
            'q95': 0.0,
            'trace': 0.0,
            'entropy': 0.0,
            'effective_rank': 0.0,
            'participation_ratio': 0.0,
            'top1_frac': 0.0,
            'eig1': 0.0,
            'eig2': 0.0,
            'eig1_eig2_ratio': 0.0,
            'log10_mean': 0.0,
            'log10_std': 0.0,
            'gini': 0.0,
        }

    mu = float(np.mean(v))
    sd = float(np.std(v))
    q05 = float(np.quantile(v, 0.05))
    q50 = float(np.quantile(v, 0.50))
    q95 = float(np.quantile(v, 0.95))

    vp = np.maximum(v, 0.0)
    trace = float(vp.sum())

    if trace > 0:
        p = vp / float(trace)
        entropy = float(-np.sum(p * np.log(p + _EPS)))
        effective_rank = float(np.exp(entropy))
        participation_ratio = float((trace**2) / float(np.sum(vp**2) + _EPS))
    else:
        entropy = 0.0
        effective_rank = 0.0
        participation_ratio = 0.0

    vs = np.sort(vp)[::-1]
    eig1 = float(vs[0]) if vs.size > 0 else 0.0
    eig2 = float(vs[1]) if vs.size > 1 else 0.0
    top1_frac = float(eig1 / float(trace + _EPS))
    eig1_eig2_ratio = float(eig1 / float(eig2 + _EPS))

    logv = np.log10(vp + _EPS)
    log10_mean = float(np.mean(logv))
    log10_std = float(np.std(logv))

    return {
        'count': float(v.size),
        'mean': mu,
        'std': sd,
        'min': float(np.min(v)),
        'max': float(np.max(v)),
        'q05': q05,
        'q50': q50,
        'q95': q95,
        'trace': trace,
        'entropy': entropy,
        'effective_rank': effective_rank,
        'participation_ratio': participation_ratio,
        'top1_frac': top1_frac,
        'eig1': eig1,
        'eig2': eig2,
        'eig1_eig2_ratio': eig1_eig2_ratio,
        'log10_mean': log10_mean,
        'log10_std': log10_std,
        'gini': _gini_positive(vp),
    }


def ks_statistic(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    b = np.asarray(list(y), dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    a = np.sort(a)
    b = np.sort(b)
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side='right') / float(a.size)
    cdf_b = np.searchsorted(b, grid, side='right') / float(b.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein_q(x: Sequence[float], y: Sequence[float], *, n_quantiles: int = 512) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    b = np.asarray(list(y), dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    nq = int(max(64, min(int(n_quantiles), 4096)))
    qs = np.linspace(0.0, 1.0, nq)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.mean(np.abs(qa - qb)))


def js_divergence_hist(x: Sequence[float], y: Sequence[float], *, bins: int = 40) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    b = np.asarray(list(y), dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    mn = float(min(a.min(), b.min()))
    mx = float(max(a.max(), b.max()))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return 0.0
    nb = int(max(10, bins))
    edges = np.linspace(mn, mx, nb + 1)
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    pa = ha.astype(np.float64); pb = hb.astype(np.float64)
    pa /= float(pa.sum() + _EPS)
    pb /= float(pb.sum() + _EPS)
    m = 0.5 * (pa + pb)

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log((p[mask] + _EPS) / (q[mask] + _EPS))))

    return float(max(0.0, 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)))


def robust_zscore(values: Sequence[float]) -> Tuple[float, float, np.ndarray]:
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.0, np.zeros((0,), dtype=np.float64)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    mad_sigma = float(1.4826 * mad)
    z = (v - med) / float(mad_sigma + _EPS)
    return med, mad_sigma, z


@dataclass(frozen=True)
class EigenvalueSpectrumMetrics:
    model: str
    label: str
    layer_mode: str  # 'single' or 'all'
    layer_indices: List[int]
    eigenvalues: List[float]
    bins: int
    stats: Dict[str, float]
    subtitle: Optional[str] = None


def _wrap(lines: List[str], width: int = 118) -> str:
    return '\n'.join(textwrap.fill(str(ln), width=width) for ln in lines if str(ln).strip())


def _guidance_lines(
    *,
    drift: Optional[Dict[str, Any]],
    top_prompt_anomalies: Optional[List[Dict[str, Any]]],
) -> Tuple[List[str], List[str], List[str]]:
    findings: List[str] = []
    red: List[str] = []
    blue: List[str] = []

    if drift and drift.get('summary'):
        s = drift.get('summary') or {}
        parts = []
        if s.get('ks') is not None:
            parts.append(f"KS={float(s['ks']):.2f}")
        if s.get('wasserstein_q') is not None:
            parts.append(f"W1~{float(s['wasserstein_q']):.3g}")
        if s.get('js') is not None:
            parts.append(f"JS={float(s['js']):.3g}")
        if parts:
            findings.append('Baseline drift: ' + ' | '.join(parts))

        tops = drift.get('top_layers') or []
        if tops:
            layers = [str(int(r.get('layer'))) for r in tops[:6] if r.get('layer') is not None]
            if layers:
                findings.append('Most shifted layers: ' + ','.join(layers))

    if top_prompt_anomalies:
        idxs = [str(int(r.get('index'))) for r in top_prompt_anomalies[:5] if r.get('index') is not None]
        if idxs:
            findings.append('Top anomalous prompts (idx): ' + ','.join(idxs))

    blue.extend([
        'Baseline per app/model; alert on sustained drift (KS/JS) and anisotropy changes.',
        'Pivot from top shifted layers into FFT / attention-security / TTD for root cause.',
    ])

    red.extend([
        'Use as an internal-signature meter for injection scanners (Promptmap2/MPIT).',
        'Stealth loop: minimize drift across layers and across paraphrases; avoid single-layer spikes.',
    ])

    return findings, red, blue


def plot_eigenvalue_spectrum(
    metrics: EigenvalueSpectrumMetrics,
    *,
    title: str = 'NeurInSpectre Eigenvalue Spectrum',
    out_path: Optional[str] = None,
    guidance: bool = True,
    x_scale: str = 'linear',
    baseline: Optional[EigenvalueSpectrumMetrics] = None,
    layer_summary: Optional[Dict[int, Dict[str, float]]] = None,
    baseline_layer_summary: Optional[Dict[int, Dict[str, float]]] = None,
    drift: Optional[Dict[str, Any]] = None,
    top_prompt_anomalies: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render PNG (matplotlib) with baseline overlay + per-layer panel."""
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.gridspec import GridSpec  # type: ignore

    test_x = transform_eigs(metrics.eigenvalues, x_scale=x_scale)
    base_x = transform_eigs(baseline.eigenvalues, x_scale=x_scale) if baseline else None

    fig = plt.figure(figsize=(14, 10.2 if guidance else 8.0))
    gs = GridSpec(3, 1, height_ratios=[3.0, 1.3, 1.65] if guidance else [3.0, 1.3, 0.001], hspace=0.24)

    ax = fig.add_subplot(gs[0])
    ax_layer = fig.add_subplot(gs[1])
    footer = fig.add_subplot(gs[2]) if guidance else None
    if footer is not None:
        footer.axis('off')

    bins = int(max(10, metrics.bins))
    if base_x is not None and base_x.size > 0:
        ax.hist(base_x, bins=bins, color='#9e9e9e', alpha=0.35, edgecolor='white', label='baseline')
    ax.hist(test_x, bins=bins, color='#4C78A8', alpha=0.85, edgecolor='white', label='test')

    def _mu_sd(v: np.ndarray) -> Tuple[float, float]:
        if v.size == 0:
            return 0.0, 0.0
        return float(np.mean(v)), float(np.std(v))

    mu_t, sd_t = _mu_sd(test_x)
    mu_b, sd_b = _mu_sd(base_x) if base_x is not None else (0.0, 0.0)
    band_mu = mu_b if base_x is not None else mu_t
    band_sd = sd_b if base_x is not None else sd_t

    ax.axvspan(band_mu - band_sd, band_mu + band_sd, color='#E45756', alpha=0.12, label='±1σ (baseline)' if base_x is not None else '±1σ')
    if base_x is not None:
        ax.axvline(mu_b, color='#424242', linestyle=':', linewidth=2.0, label='baseline μ')
    ax.axvline(mu_t, color='#E45756', linestyle='--', linewidth=2.0, label='test μ')

    xlab = 'eigenvalue' if x_scale == 'linear' else 'log10(eigenvalue + eps)'
    ax.set_xlabel(xlab)
    ax.set_ylabel('count')

    fig.suptitle(
        f"{title}: {metrics.label}\nSecurity-Relevant Spectral Forensics (baseline-vs-test recommended)",
        fontsize=18,
        y=0.97,
    )
    if metrics.subtitle:
        ax.set_title(str(metrics.subtitle), fontsize=11, pad=6)

    ax.legend(loc='upper right', fontsize=10, frameon=True)

    # Per-layer panel: top1_frac baseline vs test
    ax_layer.set_title('Per-layer anisotropy proxy (top1_frac; higher = more collapse)', fontsize=11, pad=6)
    ax_layer.set_xlabel('layer')
    ax_layer.set_ylabel('top1_frac')

    if layer_summary:
        layers = sorted(int(k) for k in layer_summary.keys())
        test_vals = [float(layer_summary[int(k)].get('top1_frac', 0.0)) for k in layers]

        if baseline_layer_summary:
            base_vals = [float(baseline_layer_summary.get(int(k), {}).get('top1_frac', 0.0)) for k in layers]
            w = 0.38
            x = np.arange(len(layers))
            ax_layer.bar(x - w / 2, base_vals, width=w, color='#9e9e9e', alpha=0.55, label='baseline')
            ax_layer.bar(x + w / 2, test_vals, width=w, color='#4C78A8', alpha=0.85, label='test')
            ax_layer.set_xticks(x)
            ax_layer.set_xticklabels([str(l) for l in layers])
            ax_layer.legend(loc='upper right', fontsize=9)

            if drift and drift.get('top_layers'):
                tops = {int(r.get('layer')) for r in (drift.get('top_layers') or [])[:3] if r.get('layer') is not None}
                for i, l in enumerate(layers):
                    if int(l) in tops:
                        ax_layer.axvline(i, color='#E45756', linestyle='--', alpha=0.35)
        else:
            ax_layer.bar([str(l) for l in layers], test_vals, color='#4C78A8', alpha=0.85)

        ax_layer.set_ylim(0.0, 1.0)
    else:
        ax_layer.axis('off')

    if guidance and footer is not None:
        findings, red, blue = _guidance_lines(drift=drift, top_prompt_anomalies=top_prompt_anomalies)
        hdr = [f"Axis: {xlab}.", f"Test μ={mu_t:.3g}; σ={sd_t:.3g}."]
        if base_x is not None and base_x.size > 0:
            hdr.append("Legend: dotted line = baseline mean (μ_baseline); red dashed line = test mean (μ_test); shaded band = μ_baseline ± 1σ.")
        else:
            hdr.append("Legend: red dashed line = mean (μ); shaded band = μ ± 1σ.")
        box = _wrap(hdr + findings + [''] + ['RED: ' + r for r in red] + ['BLUE: ' + b for b in blue])
        footer.text(
            0.5,
            0.5,
            box,
            ha='center',
            va='center',
            fontsize=10.5,
            transform=footer.transAxes,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#bdbdbd', linewidth=1.5),
        )

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        return str(out)

    return ''


def plot_eigenvalue_spectrum_interactive(
    metrics: EigenvalueSpectrumMetrics,
    *,
    title: str = 'NeurInSpectre Eigenvalue Spectrum',
    out_html: Optional[str] = None,
    x_scale: str = 'linear',
    baseline: Optional[EigenvalueSpectrumMetrics] = None,
    layer_summary: Optional[Dict[int, Dict[str, float]]] = None,
    baseline_layer_summary: Optional[Dict[int, Dict[str, float]]] = None,
    drift: Optional[Dict[str, Any]] = None,
    top_prompt_anomalies: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render an interactive HTML report (Plotly + playbook sections)."""
    if not out_html:
        return ''

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio

        test_x = transform_eigs(metrics.eigenvalues, x_scale=x_scale)
        base_x = transform_eigs(baseline.eigenvalues, x_scale=x_scale) if baseline else None

        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.62, 0.38],
            subplot_titles=(
                'Eigenvalue distribution' + (' (baseline vs test)' if base_x is not None else ''),
                'Per-layer top1_frac (anisotropy proxy)' if layer_summary else 'Per-layer summary',
            ),
        )

        if base_x is not None and base_x.size > 0:
            fig.add_trace(
                go.Histogram(
                    x=base_x,
                    nbinsx=int(max(10, metrics.bins)),
                    marker_color='#9e9e9e',
                    opacity=0.35,
                    name='baseline',
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Histogram(
                x=test_x,
                nbinsx=int(max(10, metrics.bins)),
                marker_color='#4C78A8',
                opacity=0.85,
                name='test',
            ),
            row=1,
            col=1,
        )

        def _mu_sd(v: np.ndarray) -> Tuple[float, float]:
            if v.size == 0:
                return 0.0, 0.0
            return float(np.mean(v)), float(np.std(v))

        mu_t, sd_t = _mu_sd(test_x)
        mu_b, sd_b = _mu_sd(base_x) if base_x is not None else (0.0, 0.0)
        band_mu = mu_b if base_x is not None else mu_t
        band_sd = sd_b if base_x is not None else sd_t

        fig.add_vrect(x0=band_mu - band_sd, x1=band_mu + band_sd, fillcolor='#E45756', opacity=0.12, line_width=0, row=1, col=1)
        if base_x is not None:
            fig.add_vline(x=mu_b, line_dash='dot', line_color='#E45756', line_width=2, row=1, col=1)
        fig.add_vline(x=mu_t, line_dash='dash', line_color='#E45756', line_width=2, row=1, col=1)

        # Legend-only traces for mean lines / σ-band (Plotly shapes don't show in legend).
        if base_x is not None and base_x.size > 0:
            fig.add_trace(
                go.Bar(
                    x=[0],
                    y=[0],
                    marker_color='rgba(228,87,86,0.18)',
                    name='μ_baseline ± 1σ band (pink)',
                    visible='legendonly',
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[mu_b, mu_b],
                    y=[0, 1],
                    mode='lines',
                    line=dict(color='#E45756', dash='dot', width=2),
                    name='μ_baseline (red dotted)',
                    visible='legendonly',
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=[mu_t, mu_t],
                y=[0, 1],
                mode='lines',
                line=dict(color='#E45756', dash='dash', width=2),
                name='μ_test (red dashed)',
                visible='legendonly',
            ),
            row=1,
            col=1,
        )

        if layer_summary:
            layers = sorted(int(k) for k in layer_summary.keys())
            test_vals = [float(layer_summary[int(k)].get('top1_frac', 0.0)) for k in layers]
            if baseline_layer_summary:
                base_vals = [float(baseline_layer_summary.get(int(k), {}).get('top1_frac', 0.0)) for k in layers]
                fig.add_trace(go.Bar(x=layers, y=base_vals, name='baseline top1_frac', marker_color='#BDBDBD', opacity=0.55), row=1, col=2)
            fig.add_trace(go.Bar(x=layers, y=test_vals, name='test top1_frac', marker_color='#F58518', opacity=0.85), row=1, col=2)
            fig.update_yaxes(range=[0, 1], row=1, col=2)

        xlab = 'eigenvalue' if x_scale == 'linear' else 'log10(eigenvalue + eps)'
        fig.update_xaxes(title_text=xlab, row=1, col=1)
        fig.update_yaxes(title_text='count', row=1, col=1)
        fig.update_xaxes(title_text='layer', row=1, col=2)
        fig.update_yaxes(title_text='top1_frac', row=1, col=2)

        fig.update_layout(
            barmode='overlay',
            template='plotly_white',
            height=680,
            margin=dict(l=55, r=35, t=70, b=180),
            title_text='',
            legend=dict(orientation='h', yanchor='top', y=-0.22, xanchor='left', x=0.0),
        )

        findings, red, blue = _guidance_lines(drift=drift, top_prompt_anomalies=top_prompt_anomalies)

        def _ul(lines: List[str]) -> str:
            return '<ul>' + ''.join(f"<li>{html.escape(str(x))}</li>" for x in lines if str(x).strip()) + '</ul>'

        layer_tbl = ''
        if drift and drift.get('top_layers'):
            rows = ''
            for r in (drift.get('top_layers') or [])[:10]:
                rows += (
                    '<tr>'
                    f"<td>{int(r.get('layer'))}</td>"
                    f"<td>{float(r.get('ks', 0.0)):.2f}</td>"
                    f"<td>{float(r.get('wasserstein_q', 0.0)):.3g}</td>"
                    f"<td>{float(r.get('js', 0.0)):.3g}</td>"
                    f"<td>{float(r.get('delta_top1_frac', 0.0)):+.3f}</td>"
                    '</tr>'
                )
            layer_tbl = (
                "<table class='tbl'><thead><tr><th>Layer</th><th>KS</th><th>W1~</th><th>JS</th><th>Δtop1_frac</th></tr></thead><tbody>"
                + rows
                + "</tbody></table>"
            )

        prompt_tbl = ''
        if top_prompt_anomalies:
            rows = ''
            for r in top_prompt_anomalies[:10]:
                rows += (
                    '<tr>'
                    f"<td>{int(r.get('index'))}</td>"
                    f"<td>{float(r.get('anomaly_score', 0.0)):.2f}</td>"
                    f"<td><code>{html.escape(str(r.get('prompt_sha16', '')))}</code></td>"
                    f"<td>{html.escape(str(r.get('snippet', '')))}</td>"
                    '</tr>'
                )
            prompt_tbl = (
                "<table class='tbl'><thead><tr><th>Prompt idx</th><th>Anomaly</th><th>sha16</th><th>snippet</th></tr></thead><tbody>"
                + rows
                + "</tbody></table>"
            )

        fig_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)

        doc = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>{html.escape(title)}: {html.escape(str(metrics.label))}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; color: #111; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 18px; }}
    .card {{ border: 1px solid #e3e3e3; border-radius: 10px; padding: 14px; margin: 12px 0; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    @media (max-width: 920px) {{ .two {{ grid-template-columns: 1fr; }} }}
    .tbl {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
    .tbl th, .tbl td {{ border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; vertical-align: top; }}
    code {{ background: #f7f7f7; padding: 2px 5px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class='wrap'>
    <div class='card'>
      <h2 style='margin:0 0 6px 0; font-size: 18px;'>NeurInSpectre Eigenvalue Spectrum: <code>{html.escape(str(metrics.label))}</code></h2>
      <div style='color:#555; font-size: 13px;'>Spectral Forensics Report (baseline-vs-test recommended). Axis: <code>{html.escape(xlab)}</code>. Lines: red dotted = μ_baseline (when provided); red dashed = μ_test; pink band = μ_baseline ± 1σ.</div>
    </div>
    <div class='card'>{fig_html}</div>

    <div class='two'>
      <div class='card'>
        <h3>Findings</h3>
        {_ul(findings)}
        {layer_tbl}
      </div>
      <div class='card'>
        <h3>Top anomalous prompts (suite triage)</h3>
        <div style='color:#555; font-size: 12.5px; margin-bottom: 8px;'>Shown when you provide a baseline suite and run with <code>--prompts-file</code>.</div>
        {prompt_tbl}
      </div>
    </div>

    <div class='two'>
      <div class='card'>
        <h3>Blue team: practical next steps</h3>
        {_ul(blue)}
      </div>
      <div class='card'>
        <h3>Red team: practical next steps</h3>
        {_ul(red)}
      </div>
    </div>

    <div class='card'>
      <h3>Recent research + conference context (verified links)</h3>
      <ul>
        <li><a href='https://arxiv.org/abs/2509.15735' target='_blank' rel='noopener noreferrer'>EigenTrack (spectral activation tracking)</a></li>
        <li><a href='https://arxiv.org/abs/2509.13154' target='_blank' rel='noopener noreferrer'>HSAD (FFT hidden-layer temporal signals)</a></li>
        <li><a href='https://arxiv.org/abs/2505.06311' target='_blank' rel='noopener noreferrer'>Instruction detection for indirect prompt injection</a></li>
        <li><a href='https://infocondb.org/con/def-con/def-con-33/promptmap2' target='_blank' rel='noopener noreferrer'>DEF CON 33 Promptmap2</a></li>
        <li><a href='https://blackhat.com/html/webcast/06102025.html' target='_blank' rel='noopener noreferrer'>Black Hat webcast (Jun 2025): Advanced prompt injection exploits</a></li>
      </ul>
    </div>
  </div>
</body>
</html>
"""

        outp = Path(out_html)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(doc, encoding='utf-8')
        return str(outp)

    except Exception:
        return ''
