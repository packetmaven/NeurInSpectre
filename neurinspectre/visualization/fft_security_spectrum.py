"""FFT-based activation security spectrum visualization.

This module implements a lightweight, real-data visualization that treats a
layer's token-wise hidden-state dynamics as a 1D signal and analyzes its
frequency-domain power spectrum via an rFFT.

Primary intent: provide a coarse, layer-adjustable *shape* diagnostic.

- Dominant low-frequency / DC mass: persistent, slowly-varying activation regimes.
- High-frequency tail energy: rapid token-to-token regime switching (often where
  prompt boundaries, retrieval inserts, or injected instruction blocks show up).

This is **not** a standalone "malicious prompt detector". It becomes more
useful as a security signal when you:
- compare a *test* suite against a benign *baseline* suite, and
- use a signal mode that emphasizes regime changes (e.g. cosine_delta), and
- avoid truncation artifacts (use longer prompts and/or fixed FFT size).

No simulation: spectra are computed from real model hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def detrend_signal(x: np.ndarray, *, mode: str = "none") -> np.ndarray:
    """Detrend a 1D signal.

    - none: return as-is
    - mean: subtract mean (removes DC)

    For security use-cases, `mean` often makes injection-boundary switching
    clearer by de-emphasizing absolute activation scale.
    """

    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("x must be a 1D array")
    m = str(mode or "none").lower()
    if m == "none":
        return arr
    if m == "mean":
        return arr - float(np.mean(arr))
    raise ValueError("detrend mode must be one of: none, mean")


def token_signal(hidden_states: np.ndarray, *, mode: str = "token_norm", eps: float = 1e-12) -> np.ndarray:
    """Convert hidden states [seq, hidden] into a 1D per-token signal.

    Modes:
    - token_norm:           s_t = ||h_t||_2
    - delta_token_norm:     s_t = | ||h_t||_2 - ||h_{t-1}||_2 |
    - cosine_delta:         s_t = 1 - cos(h_t, h_{t-1})
    - mean_abs_delta:       s_t = mean_d |h_t[d] - h_{t-1}[d]|

    These signals are intentionally simple, fast, and model-agnostic. In practice,
    `cosine_delta` is often a better “regime shift” proxy than raw norms.
    """

    hs = np.asarray(hidden_states, dtype=np.float64)
    if hs.ndim != 2:
        raise ValueError("hidden_states must have shape [seq, hidden]")
    if hs.shape[0] < 2:
        return np.zeros((hs.shape[0],), dtype=np.float64)

    m = str(mode or "token_norm").lower()

    if m == "token_norm":
        return np.linalg.norm(hs, axis=-1)

    if m == "delta_token_norm":
        s = np.linalg.norm(hs, axis=-1)
        out = np.empty_like(s)
        out[0] = 0.0
        out[1:] = np.abs(s[1:] - s[:-1])
        return out

    if m == "cosine_delta":
        n = np.linalg.norm(hs, axis=-1)
        n = np.maximum(n, float(eps))
        v = hs / n[:, None]
        cos = np.sum(v[1:] * v[:-1], axis=-1)
        cos = np.clip(cos, -1.0, 1.0)
        d = 1.0 - cos
        out = np.empty((hs.shape[0],), dtype=np.float64)
        out[0] = 0.0
        out[1:] = d
        return out

    if m == "mean_abs_delta":
        d = np.mean(np.abs(hs[1:] - hs[:-1]), axis=-1)
        out = np.empty((hs.shape[0],), dtype=np.float64)
        out[0] = 0.0
        out[1:] = d
        return out

    raise ValueError("signal mode must be one of: token_norm, delta_token_norm, cosine_delta, mean_abs_delta")


def rfft_power(signal_1d: np.ndarray, *, n_fft: Optional[int] = None, window: str = "none") -> Tuple[np.ndarray, np.ndarray]:
    """Return (freqs, power) for a 1D signal using rFFT.

    - Frequencies are normalized for d=1.0, so they fall in [0, 0.5].
    - If `n_fft` is set, the FFT is computed with a fixed length (pads/truncates).
    - Optional windowing (e.g., Hann) reduces leakage and often stabilizes tail metrics.
    """

    x = np.asarray(signal_1d, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal_1d must be a 1D array")

    n = int(n_fft) if n_fft is not None and int(n_fft) > 0 else int(x.shape[0])
    if n < 4:
        raise ValueError("FFT length must be >= 4")

    buf = np.zeros((n,), dtype=np.float64)
    k = min(int(x.shape[0]), n)
    buf[:k] = x[:k]

    w = str(window or "none").lower()
    if w == "none":
        pass
    elif w in ("hann", "hanning"):
        buf = buf * np.hanning(n)
    else:
        raise ValueError("window must be one of: none, hann")

    fft_vals = np.fft.rfft(buf)
    freqs = np.fft.rfftfreq(n, d=1.0)
    power = (np.abs(fft_vals) ** 2).astype(np.float64)
    return freqs, power


def high_freq_tail_ratio(freqs: np.ndarray, power: np.ndarray, *, tail_start: float = 0.25, exclude_dc: bool = True, eps: float = 1e-12) -> float:
    """Compute tail-energy ratio: sum_{f>=tail_start} P(f) / sum P(f).

    By default, excludes DC from the denominator because DC is often dominant
    for non-negative signals and can mask tail changes.
    """

    f = np.asarray(freqs, dtype=np.float64)
    p = np.asarray(power, dtype=np.float64)
    if f.shape != p.shape:
        raise ValueError("freqs and power must have the same shape")
    if f.ndim != 1:
        raise ValueError("freqs and power must be 1D arrays")

    base = p[1:] if exclude_dc and p.shape[0] > 1 else p
    denom = float(np.sum(base)) + float(eps)

    tail_mask = f >= float(tail_start)
    if exclude_dc:
        tail_mask = tail_mask & (f > 0.0)
    tail = float(np.sum(p[tail_mask]))
    return float(tail / denom)


@dataclass
class FFTSecuritySpectrumMetrics:
    """Serializable metrics for plotting and auditability."""

    title: str
    model: str
    tokenizer: str
    layer: int

    # FFT/sample semantics
    seq_len: int
    prompt_count: int
    seq_mode: str

    # Signal processing controls (audit)
    signal_mode: str = "token_norm"
    detrend: str = "none"
    window: str = "none"
    fft_size: int = 0
    segment: str = "prefix"

    # Scoring
    tail_start: float = 0.25
    z_threshold: float = 2.0
    z_mode: str = "standard"

    # Spectra (test suite)
    freqs: List[float] = None  # type: ignore
    spectra: List[List[float]] = None  # type: ignore
    mean_spectrum: List[float] = None  # type: ignore

    dominant_freqs: List[float] = None  # type: ignore
    dominant_powers: List[float] = None  # type: ignore
    tail_ratios: List[float] = None  # type: ignore
    dominant_z: List[float] = None  # type: ignore
    tail_z: List[float] = None  # type: ignore

    subtitle: Optional[str] = None
    prompt_sha16: Optional[List[str]] = None

    # Optional baseline comparison (benign suite)
    baseline_prompt_count: Optional[int] = None
    baseline_mean_spectrum: Optional[List[float]] = None
    baseline_dominant_powers: Optional[List[float]] = None
    baseline_tail_ratios: Optional[List[float]] = None
    baseline_prompt_sha16: Optional[List[str]] = None
    baseline_label: Optional[str] = None

    def clamp_prompt_index(self, idx: int) -> int:
        if int(self.prompt_count) <= 0:
            return 0
        return int(max(0, min(int(idx), int(self.prompt_count) - 1)))


def plot_fft_security_spectrum(metrics: FFTSecuritySpectrumMetrics, *, prompt_index: int = 0, title: Optional[str] = None, out_path: Optional[str] = None, guidance: bool = True) -> str:
    """Render a 2-panel FFT spectrum plot.

    Left: selected prompt spectrum.
    Right: mean spectrum (optionally baseline-vs-test if baseline is present).
    """

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    freqs = np.asarray(metrics.freqs, dtype=np.float64)
    if freqs.ndim != 1:
        raise ValueError("metrics.freqs must be 1D")

    prompt_index = metrics.clamp_prompt_index(prompt_index)
    sel = np.asarray(metrics.spectra[prompt_index], dtype=np.float64)
    mean_test = np.asarray(metrics.mean_spectrum, dtype=np.float64)

    if sel.shape != freqs.shape or mean_test.shape != freqs.shape:
        raise ValueError("Spectrum shapes must match freqs shape")

    baseline_mean = None
    if metrics.baseline_mean_spectrum is not None:
        bm = np.asarray(metrics.baseline_mean_spectrum, dtype=np.float64)
        if bm.shape == freqs.shape:
            baseline_mean = bm

    dom_idx = int(np.argmax(sel))
    dom_f = float(freqs[dom_idx])
    dom_p = float(sel[dom_idx])
    dom_z = float(metrics.dominant_z[prompt_index])
    tail_z = float(metrics.tail_z[prompt_index])

    dom_label = "suspicious" if abs(dom_z) >= float(metrics.z_threshold) else "benign"
    tail_label = "suspicious" if abs(tail_z) >= float(metrics.z_threshold) else "benign"

    tail_mask = freqs >= float(metrics.tail_start)
    tail_peak_f = None
    tail_peak_p = None
    if bool(np.any(tail_mask)):
        tail_sel = sel[tail_mask]
        if tail_sel.size > 0:
            j = int(np.argmax(tail_sel))
            tail_peak_f = float(freqs[tail_mask][j])
            tail_peak_p = float(tail_sel[j])

    # Threshold line: use baseline distribution if present
    if metrics.baseline_dominant_powers is not None and len(metrics.baseline_dominant_powers) > 0:
        dom_vals = np.asarray(metrics.baseline_dominant_powers, dtype=np.float64)
        thr_scope = "baseline"
    else:
        dom_vals = np.asarray(metrics.dominant_powers, dtype=np.float64) if metrics.dominant_powers else np.asarray([0.0], dtype=np.float64)
        thr_scope = "suite"

    if str(metrics.z_mode).lower() == "robust":
        dom_med = float(np.median(dom_vals))
        dom_mad = float(np.median(np.abs(dom_vals - dom_med)))
        dom_scale = max(1.4826 * dom_mad, 1e-12)
        dom_thr = dom_med + float(metrics.z_threshold) * dom_scale
        dom_thr_label = f"median + {metrics.z_threshold:g}*(1.4826*MAD)"
    else:
        dom_mu = float(np.mean(dom_vals))
        dom_sd = float(np.std(dom_vals))
        dom_thr = dom_mu + float(metrics.z_threshold) * max(dom_sd, 1e-12)
        dom_thr_label = f"mean + {metrics.z_threshold:g}*std"

    fig = plt.figure(figsize=(15.5, 7.8), dpi=160)
    gs = GridSpec(3, 2, height_ratios=[12, 1.7, 2.7], hspace=0.25, wspace=0.18)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    leg_ax = fig.add_subplot(gs[1, :])
    guid_ax = fig.add_subplot(gs[2, :])
    leg_ax.axis("off")
    guid_ax.axis("off")

    ax_l.axvspan(0.0, float(metrics.tail_start), color="#ffcccc", alpha=0.35, linewidth=0)
    ax_l.axvspan(float(metrics.tail_start), float(freqs[-1]), color="#ffe7cc", alpha=0.35, linewidth=0)
    ax_r.axvspan(0.0, float(metrics.tail_start), color="#ffcccc", alpha=0.35, linewidth=0)
    ax_r.axvspan(float(metrics.tail_start), float(freqs[-1]), color="#ffe7cc", alpha=0.35, linewidth=0)

    ax_l.plot(freqs, sel, color="#1f3a93", linewidth=2.0)
    if baseline_mean is not None:
        ax_r.plot(freqs, baseline_mean, color="#666666", linewidth=1.8, linestyle="--", alpha=0.9, label="Baseline mean")
        ax_r.plot(freqs, mean_test, color="#1f3a93", linewidth=2.0, label="Test mean")
        ax_r.legend(loc="upper right", framealpha=0.9, fontsize=9)
    else:
        ax_r.plot(freqs, mean_test, color="#1f3a93", linewidth=2.0)

    ax_l.scatter([dom_f], [dom_p], s=55, color="red", zorder=5)
    if tail_peak_f is not None and tail_peak_p is not None:
        ax_l.scatter([tail_peak_f], [tail_peak_p], s=55, color="orange", zorder=5)

    dom_idx_m = int(np.argmax(mean_test))
    ax_r.scatter([float(freqs[dom_idx_m])], [float(mean_test[dom_idx_m])], s=55, color="red", zorder=5)
    if bool(np.any(tail_mask)):
        tail_mean = mean_test[tail_mask]
        if tail_mean.size > 0:
            j2 = int(np.argmax(tail_mean))
            ax_r.scatter([float(freqs[tail_mask][j2])], [float(tail_mean[j2])], s=55, color="orange", zorder=5)

    ax_l.axhline(dom_thr, color="#ff9933", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_r.axhline(dom_thr, color="#ff9933", linestyle="--", linewidth=1.2, alpha=0.8)

    ax_l.set_title(f"NeurInSpectre FFT Spectrum: Prompt {prompt_index + 1}", fontsize=13, fontweight="bold")
    ax_r.set_title("NeurInSpectre: Mean FFT Security Spectrum" if baseline_mean is None else "NeurInSpectre: Mean FFT Security Spectrum (baseline vs test)", fontsize=13, fontweight="bold")
    ax_l.set_xlabel("Frequency")
    ax_r.set_xlabel("Frequency")
    ax_l.set_ylabel("Power")
    ax_r.set_ylabel("Power")
    ax_l.grid(True, alpha=0.25)
    ax_r.grid(True, alpha=0.25)
    ax_l.set_xlim(float(freqs[0]), float(freqs[-1]))
    ax_r.set_xlim(float(freqs[0]), float(freqs[-1]))

    fig_title = title or metrics.title
    subtitle = metrics.subtitle or f"{metrics.model.split('/')[-1]} | layer={metrics.layer} | seq={metrics.seq_len} ({metrics.seq_mode}) | z={metrics.z_mode}"
    fig.suptitle(f"{fig_title}\n{subtitle}", fontsize=16, y=0.98)

    if guidance:
        legend_text = (
            "Red marker   = Dominant frequency (possible persistent trigger / repeated regime)\n"
            "Orange marker= High-frequency tail peak (possible rapid switching / covert signaling)\n"
            f"Dashed line  = Dominant power threshold ({dom_thr_label}) over {thr_scope}"
        )
        leg_ax.text(0.5, 0.5, legend_text, ha="center", va="center", fontsize=9.5, color="#222", bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#bbbbbb", linewidth=1.5), transform=leg_ax.transAxes)

        z_scope = "vs baseline" if metrics.baseline_prompt_count is not None else "within suite"
        z_text = (
            f"Dominant Spike Z-score: {dom_z:+.2f} ({dom_label}) [{z_scope}]\n"
            f"Tail Z-score:           {tail_z:+.2f} ({tail_label}) [{z_scope}]\n\n"
            "Red Team: Measure which prompts/layers create repeatable tail changes; avoid single, obvious spectral fingerprints.\n"
            "Blue Team: Baseline on benign prompts; alert on large |Z| for dominant/tail and correlate with injection boundaries."
        )
        guid_ax.text(0.5, 0.5, z_text, ha="center", va="center", fontsize=9.5, color="#222", bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#999999", linewidth=1.5), transform=guid_ax.transAxes)

    if out_path:
        Path = __import__("pathlib").Path
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(outp), dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(outp)

    return ""
