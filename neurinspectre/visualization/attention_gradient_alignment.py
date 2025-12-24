"""Attention-Gradient Alignment (AGA) visualization.

AGA is a **heuristic diagnostic**: for each attention head, compute a cosine similarity between
the attention weights and the gradient of a scalar objective w.r.t. those weights.

High \(|alignment|\) suggests the objective is *sensitive* to the head's attention pattern.
This does **not** prove an attack by itself — treat it as a triage signal and corroborate with
other detectors (prompt-injection analysis, attention-token anomaly checks, drift/anomaly modules).

Outputs are designed to be readable in incident-response contexts:
- heatmap with flagged heads
- distribution + per-layer summary
- top-K heads table
- clear, safe next steps for blue-team triage and red-team evaluation
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


def aga_cosine_alignment(attn: np.ndarray, grad: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Cosine similarity per head between attention and gradient."""
    a = np.asarray(attn, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)
    if a.shape != g.shape:
        raise ValueError(f"attn and grad must have same shape, got {a.shape} vs {g.shape}")
    if a.ndim != 3:
        raise ValueError(f"expected [H,S,S], got {a.shape}")

    h = int(a.shape[0])
    a2 = a.reshape(h, -1)
    g2 = g.reshape(h, -1)

    num = np.sum(a2 * g2, axis=1)
    den = (np.linalg.norm(a2, axis=1) * np.linalg.norm(g2, axis=1))
    den = np.maximum(den, float(eps))
    return (num / den).astype(np.float64)


def compute_trigger_attention_ratio(attn: np.ndarray, trigger_indices: Optional[List[int]] = None, eps: float = 1e-12) -> float:
    """Heuristic: fraction of total attention mass landing on a small set of key positions.

    If `trigger_indices` is not provided, we pick the top-k key positions by aggregate attention
    mass (summed across heads and query positions).
    """
    if trigger_indices is None or len(trigger_indices) == 0:
        a_sum = np.sum(attn, axis=(0, 1))
        k = max(1, min(3, len(a_sum) // 4))
        trigger_indices = np.argsort(a_sum)[-k:].tolist()
    
    total_attention = np.sum(attn)
    trigger_attention = np.sum(attn[:, :, trigger_indices])
    tar = float(trigger_attention) / (float(total_attention) + eps)
    return np.clip(tar, 0.0, 1.0)


def compute_head_similarity_matrix(attn: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute pairwise cosine similarity between heads (flattened attention maps)."""
    h = attn.shape[0]
    a_flat = attn.reshape(h, -1)
    norms = np.linalg.norm(a_flat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    a_norm = a_flat / norms
    similarity = a_norm @ a_norm.T
    return similarity.astype(np.float64)


def compute_gradient_attention_anomaly_score(attn: np.ndarray, grad: np.ndarray, baseline_alignment: Optional[np.ndarray] = None, eps: float = 1e-12) -> Tuple[float, np.ndarray]:
    """Heuristic anomaly score: attention-magnitude-weighted deviation in alignment.

    When `baseline_alignment` is provided, we score deviation from baseline; otherwise we score
    absolute alignment.
    """
    alignment = aga_cosine_alignment(attn, grad, eps=eps)
    
    if baseline_alignment is not None:
        deviation = np.abs(alignment - baseline_alignment)
    else:
        deviation = np.abs(alignment)
    
    attn_magnitude = np.sum(np.abs(attn), axis=(1, 2))
    weights = attn_magnitude / (np.sum(attn_magnitude) + eps)
    gaas_score = float(np.sum(weights * deviation))
    per_head_anomaly = (weights * deviation).astype(np.float64)
    
    return gaas_score, per_head_anomaly


def aga_matrix(attentions: List[np.ndarray], attention_grads: List[np.ndarray], *, eps: float = 1e-12) -> np.ndarray:
    """Compute AGA heatmap matrix across layers."""
    if len(attentions) != len(attention_grads):
        raise ValueError("attentions and attention_grads must have same length")
    if not attentions:
        raise ValueError("no attention tensors provided")

    rows: List[np.ndarray] = []
    h_ref: Optional[int] = None
    for a, g in zip(attentions, attention_grads):
        al = aga_cosine_alignment(a, g, eps=eps)
        if h_ref is None:
            h_ref = int(al.shape[0])
        if int(al.shape[0]) < int(h_ref):
            pad = np.full((int(h_ref) - int(al.shape[0]),), np.nan, dtype=np.float64)
            al = np.concatenate([al, pad], axis=0)
        elif int(al.shape[0]) > int(h_ref):
            al = al[: int(h_ref)]
        rows.append(al)

    return np.stack(rows, axis=0)


def identify_high_risk_heads(alignment_matrix: np.ndarray, risk_threshold: float = 0.25, percentile: float = 90.0) -> Dict[str, Any]:
    """Identify high-risk attention heads."""
    flat = alignment_matrix.flatten()
    flat = flat[np.isfinite(flat)]
    
    if len(flat) == 0:
        return {'high_risk_count': 0, 'high_risk_heads': [], 'max_alignment': 0.0, 'mean_alignment': 0.0}
    
    max_alignment = float(np.max(np.abs(flat)))
    mean_alignment = float(np.mean(np.abs(flat)))
    pct_value = float(np.percentile(np.abs(flat), percentile))
    threshold_used = max(float(risk_threshold), float(pct_value))
    
    high_risk_heads = []
    for layer_idx in range(alignment_matrix.shape[0]):
        for head_idx in range(alignment_matrix.shape[1]):
            val = alignment_matrix[layer_idx, head_idx]
            if np.isfinite(val) and abs(val) >= threshold_used:
                high_risk_heads.append({'layer': int(layer_idx), 'head': int(head_idx), 'alignment': float(val), 'abs_alignment': float(abs(val))})
    
    high_risk_heads.sort(key=lambda x: x['abs_alignment'], reverse=True)
    
    # Backward compatibility: keep `percentile_threshold` as an alias for the used threshold.
    return {
        'high_risk_count': len(high_risk_heads),
        'high_risk_heads': high_risk_heads,
        'max_alignment': max_alignment,
        'mean_alignment': mean_alignment,
        'risk_threshold': float(risk_threshold),
        'percentile': float(percentile),
        'percentile_value': float(pct_value),
        'threshold_used': float(threshold_used),
        'percentile_threshold': float(threshold_used),
    }


@dataclass
class AGAMetrics:
    title: str
    model: str
    tokenizer: str
    prompt: str
    layer_start: int
    layer_end: int
    seq_len: int
    num_layers: int
    num_heads: int
    objective: str
    attn_source: str
    risk_threshold: float
    clip_percentile: float
    alignment: List[List[float]]
    trigger_attention_ratio: Optional[float] = None
    gaas_score: Optional[float] = None
    head_similarity_mean: Optional[float] = None
    high_risk_analysis: Optional[Dict] = None
    subtitle: Optional[str] = None
    prompt_sha16: Optional[str] = None


def plot_attention_gradient_alignment(metrics: AGAMetrics, *, title: Optional[str] = None, out_path: Optional[str] = None, guidance: bool = True) -> str:
    """Render a triage dashboard for AGA (static PNG)."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    from matplotlib.patches import Rectangle

    mat = np.asarray(metrics.alignment, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("metrics.alignment must be 2D")

    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        finite = np.asarray([0.0], dtype=np.float64)

    q = float(np.clip(metrics.clip_percentile, 0.5, 1.0))
    vmax = float(np.quantile(np.abs(finite), q))
    vmax = max(vmax, 1e-6)
    vmin = -vmax

    # --- derived thresholds + ranking (for consistent annotations)
    abs_mat = np.abs(mat[np.isfinite(mat)])
    pct90 = float(np.percentile(abs_mat, 90.0)) if abs_mat.size else 0.0
    thr_floor = float(metrics.risk_threshold)
    thr_used = float(max(thr_floor, pct90))
    if metrics.high_risk_analysis:
        try:
            thr_used = float(metrics.high_risk_analysis.get("threshold_used", metrics.high_risk_analysis.get("percentile_threshold", thr_used)))
        except Exception:
            pass

    # Build Top-K list from high_risk_heads if available; else fall back to global ranking.
    topk = 12
    top_rows: List[Tuple[int, int, float]] = []
    if metrics.high_risk_analysis and isinstance(metrics.high_risk_analysis.get("high_risk_heads", None), list):
        for r in metrics.high_risk_analysis["high_risk_heads"][:topk]:
            try:
                top_rows.append((int(r["layer"]), int(r["head"]), float(r["alignment"])))
            except Exception:
                continue
    if not top_rows:
        # Global ranking by |alignment|
        flat_idx = np.argsort(np.abs(np.nan_to_num(mat, nan=0.0)).ravel())[::-1]
        for idx in flat_idx[:topk]:
            li, hi = np.unravel_index(int(idx), mat.shape)
            top_rows.append((int(li), int(hi), float(mat[li, hi])))

    # Normalize prompt snippet for subtitle (collapse whitespace)
    prompt_one_line = " ".join(str(metrics.prompt).split())
    prompt_short = (prompt_one_line[:61] + "...") if len(prompt_one_line) > 64 else prompt_one_line
    fig_title = title or metrics.title
    if not str(fig_title).startswith("NeurInSpectre"):
        fig_title = f"NeurInSpectre — {fig_title}"
    subtitle = metrics.subtitle or f"{metrics.model.split('/')[-1]} | '{prompt_short}'"

    # --- layout (dashboard)
    fig_height = 10.6 if guidance else 9.2
    fig = plt.figure(figsize=(13.6, fig_height), dpi=160)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[3.4, 1.25, 1.30 if guidance else 0.25],
        width_ratios=[3.25, 1.75],
        hspace=0.22,
        wspace=0.18,
    )

    ax_hm = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_layer = fig.add_subplot(gs[1, 1])

    ax_side.axis("off")

    im = ax_hm.imshow(mat, cmap="coolwarm", aspect="auto", vmin=vmin, vmax=vmax)
    ax_hm.set_xlabel("Head index", fontsize=11)
    ax_hm.set_ylabel("Layer index", fontsize=11)

    h, n_layers = int(mat.shape[1]), int(mat.shape[0])
    max_ticks = 32
    x_step = max(1, int(np.ceil(h / max_ticks)))
    y_step = max(1, int(np.ceil(n_layers / max_ticks)))
    x_ticks = list(range(0, h, x_step))
    y_ticks = list(range(0, n_layers, y_step))
    ax_hm.set_xticks(x_ticks)
    ax_hm.set_xticklabels([str(i) for i in x_ticks], fontsize=8)
    ax_hm.set_yticks(y_ticks)
    ax_hm.set_yticklabels([str(int(metrics.layer_start) + i) for i in y_ticks], fontsize=8)

    ax_hm.set_title(f"{fig_title}\n{subtitle}", fontsize=14, fontweight="bold", pad=14)

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    # Avoid unicode arrows that can render as tofu boxes on some macOS Matplotlib fonts.
    cbar.set_label("Alignment\nattn <-> grad", fontsize=10)

    # Highlight flagged heads (outline)
    # Red outline: strong positive alignment; Blue outline: strong negative alignment.
    if np.isfinite(thr_used) and thr_used > 0.0:
        for li in range(n_layers):
            for hi in range(h):
                v = float(mat[li, hi])
                if not np.isfinite(v):
                    continue
                if v >= thr_used:
                    ax_hm.add_patch(Rectangle((hi - 0.5, li - 0.5), 1, 1, fill=False, edgecolor="#B91C1C", linewidth=1.4))
                elif v <= -thr_used:
                    ax_hm.add_patch(Rectangle((hi - 0.5, li - 0.5), 1, 1, fill=False, edgecolor="#1F5FBF", linewidth=1.4))

    # Side panel (summary + Top-K heads)
    hi_cnt = int(metrics.high_risk_analysis.get("high_risk_count", 0)) if isinstance(metrics.high_risk_analysis, dict) else 0
    tar = metrics.trigger_attention_ratio
    gaas = metrics.gaas_score
    sim = metrics.head_similarity_mean
    frac = float(np.mean(np.abs(mat[np.isfinite(mat)]) >= thr_used)) if np.isfinite(thr_used) and thr_used > 0 else 0.0

    side_lines: List[str] = []
    side_lines.append("Triage summary")
    side_lines.append(f"layers={int(metrics.num_layers)} heads={int(metrics.num_heads)} seq={int(metrics.seq_len)}")
    side_lines.append(f"threshold_used={thr_used:.3f} (floor={thr_floor:.3f}, p90={pct90:.3f})")
    side_lines.append(f"flagged_heads={hi_cnt} ({frac*100:.1f}% of cells)")
    if tar is not None:
        side_lines.append(f"TAR={float(tar):.3f}")
    if gaas is not None:
        side_lines.append(f"GAAS={float(gaas):.3f}")
    if sim is not None:
        side_lines.append(f"mean_head_similarity={float(sim):.3f}")
    side_lines.append("")
    side_lines.append("Top heads by |alignment|")
    for rank, (li, hi, v) in enumerate(top_rows, 1):
        layer_abs = int(metrics.layer_start) + int(li)
        side_lines.append(f"{rank:>2d}. L{layer_abs:>2d} H{hi:>2d}  align={v:+.3f}")
    side_lines.append("")
    side_lines.append("Legend")
    side_lines.append("  red outline: align >= threshold_used")
    side_lines.append("  blue outline: align <= -threshold_used")

    ax_side.text(
        0.0,
        1.0,
        "\n".join(side_lines),
        va="top",
        ha="left",
        fontsize=9.2,
        family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFFF", edgecolor="#D0D0D0", alpha=0.98),
    )

    # Histogram (|alignment|)
    abs_all = np.abs(mat[np.isfinite(mat)])
    if abs_all.size:
        ax_hist.hist(abs_all, bins=min(40, max(12, int(np.sqrt(abs_all.size)))), color="#6C5CE7", alpha=0.85)
        ax_hist.axvline(thr_floor, color="#666666", lw=1.2, ls="--", label=f"floor={thr_floor:.2f}")
        ax_hist.axvline(thr_used, color="#B91C1C", lw=1.5, ls="-", label=f"used={thr_used:.2f}")
        ax_hist.set_title("Distribution of |alignment|", fontsize=11)
        ax_hist.set_xlabel("|alignment|")
        ax_hist.set_ylabel("count")
        ax_hist.grid(alpha=0.2)
        ax_hist.legend(frameon=False, fontsize=9, loc="upper right")
    else:
        ax_hist.axis("off")

    # Per-layer counts above threshold
    if np.isfinite(thr_used) and thr_used > 0 and mat.size:
        per_layer = []
        for li in range(n_layers):
            row = mat[li, :]
            row = row[np.isfinite(row)]
            per_layer.append(int(np.sum(np.abs(row) >= thr_used)))
        ax_layer.bar(list(range(n_layers)), per_layer, color="#1F77B4", alpha=0.85)
        ax_layer.set_title("Flagged heads per layer", fontsize=11)
        ax_layer.set_xlabel("layer (relative)")
        ax_layer.set_ylabel("count")
        ax_layer.grid(axis="y", alpha=0.2)
    else:
        ax_layer.axis("off")

    if guidance:
        # Bottom row: practical next steps (safe; avoid procedural attack steps)
        gs_b = gs[2, :].subgridspec(1, 2, wspace=0.14)
        ax_blue = fig.add_subplot(gs_b[0, 0])
        ax_red = fig.add_subplot(gs_b[0, 1])
        ax_blue.axis("off")
        ax_red.axis("off")

        blue_lines = [
            "Blue team: practical next steps",
            "1) Establish a baseline distribution of |alignment| for benign prompts (same model/version).",
            "2) Treat spikes as triage signals: check whether flagged heads cluster in specific layers/heads.",
            "3) Corroborate with prompt-injection / tool-use checks (untrusted context, retrieval boundaries).",
            "4) If repeatable: consider mitigations (input sanitization, retrieval filtering, tool-call gating).",
            "5) Validate fixes: re-run AGA + anomaly/drift modules and confirm the spike resolves.",
        ]
        red_lines = [
            "Red team: evaluation checklist (safe)",
            "1) Run controlled tests (benign vs suspect prompts) and record stability across seeds/runs.",
            "2) Report top heads (layer/head) and thresholds used; include prompt hash for traceability.",
            "3) Stress test with variations (paraphrases, delimiter changes) and measure false positives.",
            "4) If used for model-hardening exercises: document how mitigations changed the head rankings.",
        ]

        ax_blue.text(
            0.0,
            1.0,
            "\n".join(blue_lines),
            va="top",
            ha="left",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#ECF3FF", edgecolor="#1F5FBF", alpha=0.97),
        )
        ax_red.text(
            0.0,
            1.0,
            "\n".join(red_lines),
            va="top",
            ha="left",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#FFF1F2", edgecolor="#B91C1C", alpha=0.97),
        )

        fig.text(
            0.01,
            0.01,
            "Note: AGA is heuristic. Use as a triage signal; corroborate with other modules and re-test after mitigations.",
            fontsize=8.5,
            color="#555",
        )

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(out)
    return ""


def write_attention_gradient_alignment_html(metrics: AGAMetrics, *, out_path: str, title: Optional[str] = None) -> str:
    """Write an interactive AGA triage dashboard as a self-contained HTML file.

    Requires Plotly. This is optional and should not be required for the core CLI to function.
    """
    try:
        import html as _html
        from pathlib import Path as _Path

        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
    except Exception as e:
        raise ImportError("AGA interactive HTML requires Plotly. Install with: pip install plotly") from e

    mat = np.asarray(metrics.alignment, dtype=np.float64)
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        finite = np.asarray([0.0], dtype=np.float64)

    q = float(np.clip(metrics.clip_percentile, 0.5, 1.0))
    vmax = float(np.quantile(np.abs(finite), q))
    vmax = max(vmax, 1e-6)
    vmin = -vmax

    abs_all = np.abs(finite)
    pct90 = float(np.percentile(abs_all, 90.0)) if abs_all.size else 0.0
    thr_floor = float(metrics.risk_threshold)
    thr_used = float(max(thr_floor, pct90))
    if metrics.high_risk_analysis:
        try:
            thr_used = float(metrics.high_risk_analysis.get("threshold_used", metrics.high_risk_analysis.get("percentile_threshold", thr_used)))
        except Exception:
            pass

    # Axis labels (actual layer numbers)
    layer_labels = [int(metrics.layer_start) + i for i in range(int(mat.shape[0]))]
    head_labels = list(range(int(mat.shape[1])))

    # Top-K heads (always rank by |alignment|, independent of whether anything crosses the threshold).
    topk = 12
    rows_ranked: List[Dict[str, Any]] = []
    flat_idx = np.argsort(np.abs(np.nan_to_num(mat, nan=0.0)).ravel())[::-1]
    for idx in flat_idx[:topk]:
        li, hi = np.unravel_index(int(idx), mat.shape)
        rows_ranked.append(
            {
                "layer": int(li),
                "head": int(hi),
                "alignment": float(mat[li, hi]),
                "abs_alignment": float(abs(mat[li, hi])),
            }
        )

    fig = make_subplots(
        rows=2,
        cols=2,
        # NOTE: avoid Plotly `Table` subplots here for compatibility across Plotly versions
        # (some versions attempt to bind xaxis/yaxis to tables and error).
        specs=[[{"type": "heatmap"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.62, 0.38],
        row_heights=[0.62, 0.38],
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
        subplot_titles=("Alignment heatmap", "Top heads (|alignment|)", "|alignment| distribution", "Flagged heads per layer"),
    )

    fig.add_trace(
        go.Heatmap(
            z=mat,
            x=head_labels,
            y=layer_labels,
            zmin=vmin,
            zmax=vmax,
            zmid=0.0,
            colorscale="RdBu",
            colorbar=dict(title="align", len=0.62, y=0.80),
            hovertemplate="layer=%{y}<br>head=%{x}<br>align=%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Flagged markers
    flagged_x = []
    flagged_y = []
    flagged_c = []
    for li in range(mat.shape[0]):
        for hi in range(mat.shape[1]):
            v = float(mat[li, hi])
            if not np.isfinite(v):
                continue
            if abs(v) >= thr_used:
                flagged_x.append(int(hi))
                flagged_y.append(int(metrics.layer_start) + int(li))
                flagged_c.append("#B91C1C" if v >= 0 else "#1F5FBF")
    if flagged_x:
        fig.add_trace(
            go.Scatter(
                x=flagged_x,
                y=flagged_y,
                mode="markers",
                marker=dict(size=9, color=flagged_c, line=dict(color="white", width=0.8)),
                name="flagged",
                hovertemplate="layer=%{y}<br>head=%{x}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Top-heads bar (ranked by |alignment|) — interactive hover gives exact layer/head.
    top_labels = []
    top_abs = []
    top_signed = []
    for r in rows_ranked:
        li = int(r.get("layer", 0))
        hi = int(r.get("head", 0))
        v = float(r.get("alignment", 0.0))
        top_labels.append(f"L{int(metrics.layer_start) + li}:H{hi}")
        top_abs.append(abs(v))
        top_signed.append(v)
    if top_labels:
        fig.add_trace(
            go.Bar(
                x=top_labels,
                y=top_abs,
                marker_color=["#B91C1C" if v >= 0 else "#1F5FBF" for v in top_signed],
                name="top |alignment|",
                hovertemplate="head=%{x}<br>|align|=%{y:.3f}<br>align=%{customdata:+.3f}<extra></extra>",
                customdata=top_signed,
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=thr_used, line_dash="dash", line_color="#666666", row=1, col=2)

    fig.add_trace(
        go.Histogram(x=abs_all, nbinsx=40, marker_color="#6C5CE7", name="|alignment|"),
        row=2,
        col=1,
    )
    fig.add_vline(x=thr_floor, line_dash="dash", line_color="#666666", row=2, col=1)
    fig.add_vline(x=thr_used, line_dash="solid", line_color="#B91C1C", row=2, col=1)

    # Per-layer flagged counts
    per_layer_counts = []
    for li in range(mat.shape[0]):
        row = mat[li, :]
        row = row[np.isfinite(row)]
        per_layer_counts.append(int(np.sum(np.abs(row) >= thr_used)))
    if int(np.sum(per_layer_counts)) > 0:
        fig.add_trace(
            go.Bar(x=layer_labels, y=per_layer_counts, marker_color="#1F77B4", name="flagged count"),
            row=2,
            col=2,
        )
    else:
        # Accurate: no bars when there are no flagged heads; show a clear note instead.
        fig.update_xaxes(visible=False, row=2, col=2)
        fig.update_yaxes(visible=False, row=2, col=2)
        fig.add_annotation(
            text="No heads exceeded threshold_used",
            x=0.5,
            y=0.5,
            xref="x4 domain",
            yref="y4 domain",
            showarrow=False,
            font=dict(size=12, color="#666"),
        )

    # Title/subtitle
    prompt_one_line = " ".join(str(metrics.prompt).split())
    prompt_short = (prompt_one_line[:61] + "...") if len(prompt_one_line) > 64 else prompt_one_line
    fig_title = title or metrics.title
    if not str(fig_title).startswith("NeurInSpectre"):
        fig_title = f"NeurInSpectre — {fig_title}"
    subtitle = metrics.subtitle or f"{metrics.model.split('/')[-1]} | '{prompt_short}'"

    fig.update_layout(
        title=dict(text=f"{_html.escape(str(fig_title))}<br><span style='font-size:12px'>{_html.escape(str(subtitle))}</span>", x=0.5),
        template="plotly_white",
        height=900,
        margin=dict(l=60, r=30, t=90, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0.0),
    )

    fig.update_xaxes(tickangle=35, row=1, col=2)
    # Only label the per-layer subplot axes if we rendered bars.
    if int(np.sum(per_layer_counts)) > 0:
        fig.update_xaxes(title_text="layer", row=2, col=2)
        fig.update_yaxes(title_text="count", row=2, col=2)

    plot_div = pio.to_html(fig, full_html=False, include_plotlyjs=True, config={"displaylogo": False, "responsive": True})

    blue_lines = [
        "Establish a baseline distribution for benign prompts (same model/version).",
        "Treat spikes as triage signals; check clustering by layer/head.",
        "Corroborate with prompt-injection and tool-use boundary checks.",
        "If repeatable: apply mitigations (retrieval filtering, tool-call gating) and re-test.",
    ]
    red_lines = [
        "Run controlled tests and check stability across seeds/runs.",
        "Report top heads + thresholds used + prompt hash for traceability.",
        "Stress test with paraphrases/delimiter variants; measure false positives.",
        "Document how mitigations changed head rankings (model-hardening evaluation).",
    ]

    summary_lines = [
        f"layers={int(metrics.num_layers)} heads={int(metrics.num_heads)} seq={int(metrics.seq_len)}",
        f"threshold_used={thr_used:.3f} (floor={thr_floor:.3f}, p90={pct90:.3f})",
        f"flagged_heads={int(metrics.high_risk_analysis.get('high_risk_count', 0)) if isinstance(metrics.high_risk_analysis, dict) else 0}",
        f"TAR={float(metrics.trigger_attention_ratio):.3f}" if metrics.trigger_attention_ratio is not None else "TAR=n/a",
        f"GAAS={float(metrics.gaas_score):.3f}" if metrics.gaas_score is not None else "GAAS=n/a",
    ]

    # HTML table for Top-K heads (kept outside Plotly for compatibility)
    def _top_table_html() -> str:
        if not rows_ranked:
            return "<div style='color:#666; font-size:12px'>(no heads ranked)</div>"
        t_rows = []
        t_rows.append("<tr><th style='text-align:left'>Layer</th><th style='text-align:left'>Head</th><th style='text-align:left'>Align</th><th style='text-align:left'>|Align|</th></tr>")
        for r in rows_ranked:
            li = int(metrics.layer_start) + int(r.get("layer", 0))
            hi = int(r.get("head", 0))
            v = float(r.get("alignment", 0.0))
            t_rows.append(
                "<tr>"
                f"<td>{li}</td>"
                f"<td>{hi}</td>"
                f"<td>{_html.escape(f'{v:+.3f}')}</td>"
                f"<td>{_html.escape(f'{abs(v):.3f}')}</td>"
                "</tr>"
            )
        return (
            "<table style='border-collapse:collapse; width:100%; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size:12px'>"
            + "".join(
                f"<tr style='border-bottom:1px solid #e5e5e5'>{row}</tr>" if row.startswith("<tr>") else row
                for row in t_rows
            )
            + "</table>"
        )

    def _ul(items: List[str]) -> str:
        return "<ul>" + "".join(f"<li>{_html.escape(x)}</li>" for x in items) + "</ul>"

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>NeurInSpectre — AGA Triage</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 18px; color: #111; }}
    .grid {{ display: grid; grid-template-columns: 1.15fr 1.45fr 1.45fr; gap: 14px; margin-top: 14px; }}
    .box {{ border-radius: 10px; padding: 14px 14px; border: 1px solid #d0d0d0; background: #fff; }}
    .blue {{ background: #ECF3FF; border-color: #1F5FBF; }}
    .red {{ background: #FFF1F2; border-color: #B91C1C; }}
    .note {{ margin-top: 10px; font-size: 12.5px; color: #555; }}
    ul {{ margin: 0; padding-left: 18px; }}
    li {{ margin: 4px 0; }}
    code {{ background: #f5f5f5; padding: 0 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  {plot_div}
  <div class="grid">
    <div class="box">
      <div style="font-weight:600; margin-bottom: 6px;">Summary</div>
      {_ul(summary_lines)}
      <div style="font-weight:600; margin-top: 10px; margin-bottom: 6px;">Top heads (table)</div>
      {_top_table_html()}
    </div>
    <div class="box blue">
      <div style="font-weight:600; margin-bottom: 6px;">Blue team: practical next steps</div>
      {_ul(blue_lines)}
    </div>
    <div class="box red">
      <div style="font-weight:600; margin-bottom: 6px;">Red team: evaluation checklist (safe)</div>
      {_ul(red_lines)}
    </div>
  </div>
  <div class="note">Note: AGA is heuristic. Use as a triage signal; corroborate with other modules and re-test after mitigations.</div>
</body>
</html>
"""

    outp = _Path(str(out_path))
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(page, encoding="utf-8")
    return str(outp)
