#!/usr/bin/env python3
"""
NeurInSpectre: Attack Vector Analysis (MITRE ATLAS / OWASP mapping)

What this module does
- Computes robust statistical + spectral indicators over numeric arrays
  (activations, gradients, embeddings).
- Maps **strong signals** to MITRE ATLAS technique IDs using the vendored STIX bundle
  in `neurinspectre/mitre_atlas/`.
- Optionally maps technique IDs to OWASP Top 10 for LLM Applications categories.

What this module does NOT do
- It does **not** emit real CVE identifiers. CVEs are software vulnerabilities assigned to
  specific products/versions. For dependency CVEs, use dedicated scanners instead of this
  ML attack-vector heuristic (e.g., `pip-audit`, `npm audit`).
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import re

# MITRE ATLAS (official) registry helpers (vendored STIX bundle)
#
# We intentionally do NOT hardcode technique names here; we normalize IDs->names/tactics
# against the official MITRE ATLAS STIX bundle shipped in `neurinspectre/mitre_atlas/`.
_ATLAS_TECH_IDX = None
_ATLAS_PHASE_TO_TACTIC = None


def _get_atlas_indexes():
    """Lazy-load ATLAS technique index + phase->tactic mapping."""
    global _ATLAS_TECH_IDX, _ATLAS_PHASE_TO_TACTIC
    if _ATLAS_TECH_IDX is not None and _ATLAS_PHASE_TO_TACTIC is not None:
        return _ATLAS_TECH_IDX, _ATLAS_PHASE_TO_TACTIC

    try:
        from ..mitre_atlas.registry import load_stix_atlas_bundle, technique_index, tactic_by_phase_name

        bundle = load_stix_atlas_bundle()
        _ATLAS_TECH_IDX = technique_index(bundle)
        _ATLAS_PHASE_TO_TACTIC = tactic_by_phase_name(bundle)
    except Exception:
        _ATLAS_TECH_IDX = {}
        _ATLAS_PHASE_TO_TACTIC = {}

    return _ATLAS_TECH_IDX, _ATLAS_PHASE_TO_TACTIC


def _atlas_meta(tech_id: str) -> Dict[str, Any]:
    idx, phase_to_tactic = _get_atlas_indexes()
    tech = idx.get(tech_id)
    if tech is None:
        return {'id': tech_id, 'name': 'Unknown', 'tactics': [], 'url': None}

    tactics = []
    for ph in tech.tactic_phase_names:
        t = phase_to_tactic.get(ph)
        if t and t.name not in tactics:
            tactics.append(t.name)

    return {'id': tech_id, 'name': tech.name, 'tactics': tactics, 'url': tech.url}


# Note: we intentionally do NOT maintain a "CVE database" here.
# CVEs are software vulnerabilities assigned to specific products/versions.
# For CVEs, use dependency scanners (pip-audit / npm audit).

# OWASP Top 10 for LLM Applications
OWASP_LLM_TOP10 = {
    "LLM01": {
        "name": "Prompt Injection",
        "description": "Manipulating LLM through crafted inputs",
        "mitre_atlas": ["AML.T0051", "AML.T0051.001"],
        "severity": "CRITICAL"
    },
    "LLM02": {
        "name": "Insecure Output Handling",
        "description": "Insufficient validation of LLM outputs",
        "mitre_atlas": [],
        "severity": "HIGH"
    },
    "LLM03": {
        "name": "Training Data Poisoning",
        "description": "Corrupting training data to affect model behavior",
        "mitre_atlas": ["AML.T0010", "AML.T0020", "AML.T0019"],
        "severity": "HIGH"
    },
    "LLM04": {
        "name": "Model Denial of Service",
        "description": "Exhausting model resources",
        "mitre_atlas": [],
        "severity": "MEDIUM"
    },
    "LLM05": {
        "name": "Supply Chain Vulnerabilities",
        "description": "Risks from third-party components",
        "mitre_atlas": ["AML.T0010"],
        "severity": "HIGH"
    },
    "LLM06": {
        "name": "Sensitive Information Disclosure",
        "description": "LLM revealing sensitive data",
        "mitre_atlas": ["AML.T0057", "AML.T0024.000", "AML.T0024.001"],
        "severity": "HIGH"
    },
    "LLM07": {
        "name": "Insecure Plugin Design",
        "description": "Vulnerabilities in LLM plugins/tools",
        "mitre_atlas": ["AML.T0053", "AML.T0085.001"],
        "severity": "HIGH"
    },
    "LLM08": {
        "name": "Excessive Agency",
        "description": "LLM with too much autonomous capability",
        "mitre_atlas": [],
        "severity": "HIGH"
    },
    "LLM09": {
        "name": "Overreliance",
        "description": "Excessive trust in LLM outputs",
        "mitre_atlas": [],
        "severity": "MEDIUM"
    },
    "LLM10": {
        "name": "Model Theft",
        "description": "Unauthorized extraction of model",
        "mitre_atlas": ["AML.T0024.002"],
        "severity": "HIGH"
    }
}


class AttackVectorAnalyzer:
    """
    Attack vector analyzer with MITRE ATLAS / OWASP mapping (heuristic, signal-based).
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.detected_vectors: List[Dict[str, Any]] = []
        # Stores the most recent `analyze_data()` output so `generate_report()` can include
        # input-profile + threshold-check explanations even when there are 0 detections.
        self._last_results: Dict[str, Any] = {}
        
    def analyze_data(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze activation/gradient data for vulnerabilities.
        
        Args:
            data: Input data (activations, gradients, embeddings)
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "data_shape": data.shape,
            "analysis": {}
        }

        # Robustness: sanitize non-finite values early so histogram/FFT/stats never crash.
        data = np.asarray(data)
        if data.size:
            nonfinite_mask = ~np.isfinite(data)
            nonfinite_frac = float(np.mean(nonfinite_mask))
        else:
            nonfinite_frac = 0.0
        if nonfinite_frac > 0.0:
            results["analysis"]["nonfinite_fraction"] = nonfinite_frac
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize shape for analysis: treat axis-0 as samples/time, remaining dims as features.
        # This avoids dimension-dependent thresholds like ||x|| exploding with array size.
        if data.ndim == 1:
            X = data.reshape(1, -1)
        else:
            X = data.reshape(data.shape[0], -1)

        n_samples = int(X.shape[0])
        n_features = int(X.shape[1]) if X.ndim == 2 else 0

        # Input profiling: help users understand what kind of artifact this looks like
        # (raw gradients/activations vs low-dimensional summary features).
        profile: Dict[str, Any] = {
            "ndim": int(data.ndim),
            "data_shape": list(data.shape),
            "samples_axis": 0,
            "n_samples": n_samples,
            "n_features": n_features,
            "flattening": "axis-0 is treated as samples/time; remaining dims are flattened into features",
            "low_dimensional": bool(n_features <= 8),
        }
        if n_samples >= 2 and n_features >= 1:
            dX = np.diff(X.astype(np.float64, copy=False), axis=0)
            dx_rms = float(np.sqrt(np.mean(dX * dX))) if dX.size else 0.0
            x_rms = float(np.sqrt(np.mean(X.astype(np.float64, copy=False) ** 2))) if X.size else 0.0
            profile["diff_rms_over_signal_rms"] = float(dx_rms / (x_rms + 1e-12))

        # Heuristic: common 3-column capture format in this repo is [mean, std, max] per step.
        # We don't assume this is always true, but we provide a hint if it strongly matches.
        if data.ndim == 2 and data.shape[1] == 3 and data.shape[0] >= 20:
            cols = np.asarray(data, dtype=np.float64)
            frac_neg = np.mean(cols < 0, axis=0)
            max_ge_mean = float(np.mean(cols[:, 2] >= cols[:, 0]))
            profile["likely_summary_stats_3col"] = bool(frac_neg[1] < 0.01 and frac_neg[2] < 0.01 and max_ge_mean > 0.90)
            profile["summary_stats_3col_hint"] = "Often produced as per-step [mean, std, max] summaries (e.g., capture-adversarial)."
            profile["summary_stats_3col_diagnostics"] = {
                "frac_negative_by_col": [float(x) for x in frac_neg.tolist()],
                "frac_col3_ge_col1": max_ge_mean,
            }

        if profile.get("low_dimensional", False):
            profile["interpretation_hint"] = (
                "Input is low-dimensional. Treat it as a multivariate time series or precomputed features "
                "(not raw per-parameter gradients/activations). Some heuristics may not trigger because "
                "high-dimensional structure is not present."
            )

        results["analysis"]["input_profile"] = profile

        flat = X.reshape(-1)

        # Basic magnitude stats (dimension-invariant via RMS)
        if flat.size:
            mean_val = float(np.mean(flat))
            std_val = float(np.std(flat))
            rms_val = float(np.sqrt(np.mean(flat * flat)))
        else:
            mean_val = 0.0
            std_val = 0.0
            rms_val = 0.0

        if std_val > 0.0:
            z = (flat - mean_val) / std_val
            skewness = float(np.mean(z ** 3))
            kurtosis_excess = float(np.mean(z ** 4) - 3.0)
        else:
            skewness = 0.0
            kurtosis_excess = 0.0

        results["analysis"]["statistics"] = {
            "mean": mean_val,
            "std": std_val,
            "rms": rms_val,
            "skewness": skewness,
            "kurtosis_excess": kurtosis_excess,
        }

        # Value-distribution entropy (normalized to [0,1] via log2(n_bins))
        n_bins = 50
        if flat.size:
            counts, _ = np.histogram(flat, bins=n_bins)
            pmf = counts.astype(np.float64)
            pmf = pmf / (pmf.sum() + 1e-12)
            pmf = pmf[pmf > 0]
            entropy_bits = float(-np.sum(pmf * np.log2(pmf)))
            entropy_norm = float(entropy_bits / np.log2(n_bins)) if n_bins > 1 else 0.0
        else:
            entropy_bits = 0.0
            entropy_norm = 0.0
        results["analysis"]["entropy_bits"] = entropy_bits
        results["analysis"]["entropy_norm"] = entropy_norm

        # Outlier ratio (IQR rule)
        if flat.size:
            q1, q3 = np.percentile(flat, [25, 75])
            iqr = float(q3 - q1)
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            outlier_ratio = float(np.mean((flat < lo) | (flat > hi)))
        else:
            outlier_ratio = 0.0
        results["analysis"]["outlier_ratio"] = outlier_ratio

        # Per-sample norms (robust outliering) → poisoning/contamination signals
        mean_abs_feature_corr = 0.0
        hf_ratio_mean = None
        spectral_entropy_norm = None
        spectral_axis = None
        row_outlier_frac = None
        row_outlier_max_rz = None

        if X.shape[0] >= 8:
            row_norms = np.linalg.norm(X, axis=1)
            med = float(np.median(row_norms))
            mad = float(np.median(np.abs(row_norms - med))) + 1e-12
            rz = (row_norms - med) / (1.4826 * mad)
            row_outlier_frac = float(np.mean(rz > 6.0))
            row_outlier_max_rz = float(np.max(rz))
            results["analysis"]["row_norm_median"] = med
            results["analysis"]["row_norm_mad"] = mad
            results["analysis"]["row_norm_outlier_frac"] = row_outlier_frac
            results["analysis"]["row_norm_outlier_max_rz"] = row_outlier_max_rz

            if row_outlier_frac >= 0.02 and row_outlier_max_rz >= 8.0:
                frac_score = min(1.0, (row_outlier_frac - 0.02) / 0.08)  # 2%→10%
                rz_score = min(1.0, (row_outlier_max_rz - 8.0) / 8.0)     # 8→16
                conf = float(0.5 * frac_score + 0.5 * rz_score)
                self._add_detection(
                    "AML.T0020",
                    f"Extreme row-norm outliers ({row_outlier_frac:.1%}, max RZ={row_outlier_max_rz:.1f}) — possible poisoned/contaminated batches",
                    confidence=conf,
                )

        # Spectral indicators.
        #
        # Primary path (preferred for time-series artifacts):
        #   - Treat axis-0 as samples/time and compute FFT along that axis.
        #
        # Fallback path (for high-dimensional tensors with short sample axis, e.g., attention logits):
        #   - Compute FFT along the *feature axis* per sample and aggregate.
        #
        # Both paths are heuristic and are reported with an explicit `spectral_axis` field for auditability.

        # --- Temporal spectrum over sample axis (requires enough samples for meaningful frequency bins).
        # We aggregate across a small feature subset to keep runtime bounded.
        if X.shape[0] >= 64:
            n = int(X.shape[0])
            k = int(min(X.shape[1], 64))
            Xs = X[:, :k]
            F = np.fft.rfft(Xs, axis=0)
            P = (np.abs(F) ** 2).astype(np.float64)
            freqs = np.fft.rfftfreq(n, d=1.0)
            if freqs.size >= 2:
                f_nyq = float(freqs.max())
                f_theta = 0.25 * f_nyq  # Nyquist/4
                mask = freqs >= f_theta
                denom = np.sum(P, axis=0) + 1e-12
                hf_ratio_feat = np.sum(P[mask, :], axis=0) / denom
                hf_ratio_mean = float(np.mean(hf_ratio_feat))
                results["analysis"]["hf_ratio_mean"] = hf_ratio_mean
                results["analysis"]["hf_ratio_threshold_frac_nyquist"] = 0.25

                P_sum = np.sum(P, axis=1)
                P_pmf = P_sum / (np.sum(P_sum) + 1e-12)
                P_pmf = P_pmf[P_pmf > 0]
                spec_ent_bits = float(-np.sum(P_pmf * np.log2(P_pmf)))
                spectral_entropy_norm = float(spec_ent_bits / np.log2(len(freqs))) if len(freqs) > 1 else 0.0
                results["analysis"]["spectral_entropy_bits"] = spec_ent_bits
                results["analysis"]["spectral_entropy_norm"] = spectral_entropy_norm
                spectral_axis = "samples"
                results["analysis"]["spectral_axis"] = spectral_axis
                results["analysis"]["spectral_features_used"] = int(k)
                results["analysis"]["spectral_samples_used"] = int(n)

        # --- Feature-axis spectrum fallback (works even when sample axis is short).
        if (hf_ratio_mean is None or spectral_entropy_norm is None) and X.shape[1] >= 256 and X.shape[0] >= 1:
            try:
                # Bound runtime: cap both samples and feature length while keeping determinism.
                max_samples = 64
                max_feat_len = 16384

                n0 = int(X.shape[0])
                d0 = int(X.shape[1])

                # Select up to max_samples samples evenly spaced across axis-0.
                if n0 > max_samples:
                    step_s = int(np.ceil(n0 / max_samples))
                    Xs = X[::step_s, :]
                else:
                    step_s = 1
                    Xs = X

                # Downsample features by stride if too long.
                if d0 > max_feat_len:
                    step_f = int(np.ceil(d0 / max_feat_len))
                    Xf = Xs[:, ::step_f]
                else:
                    step_f = 1
                    Xf = Xs

                # Demean per sample for stability; apply Hann window to reduce edge artifacts.
                Xf = Xf.astype(np.float64, copy=False)
                Xf = Xf - np.mean(Xf, axis=1, keepdims=True)
                n_feat = int(Xf.shape[1])
                if n_feat < 2:
                    raise ValueError("Too few features for feature-axis spectrum")
                win = np.hanning(n_feat).astype(np.float64)
                Xw = Xf * win[None, :]

                F = np.fft.rfft(Xw, axis=1)
                P = (np.abs(F) ** 2).astype(np.float64)
                freqs = np.fft.rfftfreq(n_feat, d=1.0)
                if freqs.size >= 2:
                    f_nyq = float(freqs.max())
                    f_theta = 0.25 * f_nyq  # Nyquist/4
                    mask = freqs >= f_theta
                    denom = np.sum(P, axis=1) + 1e-12
                    hf_ratio_s = np.sum(P[:, mask], axis=1) / denom
                    hf_ratio_mean = float(np.mean(hf_ratio_s))
                    results["analysis"]["hf_ratio_mean"] = hf_ratio_mean
                    results["analysis"]["hf_ratio_threshold_frac_nyquist"] = 0.25

                    P_sum = np.sum(P, axis=0)
                    P_pmf = P_sum / (np.sum(P_sum) + 1e-12)
                    P_pmf = P_pmf[P_pmf > 0]
                    spec_ent_bits = float(-np.sum(P_pmf * np.log2(P_pmf)))
                    spectral_entropy_norm = float(spec_ent_bits / np.log2(len(freqs))) if len(freqs) > 1 else 0.0
                    results["analysis"]["spectral_entropy_bits"] = spec_ent_bits
                    results["analysis"]["spectral_entropy_norm"] = spectral_entropy_norm
                    spectral_axis = "features"
                    results["analysis"]["spectral_axis"] = spectral_axis
                    results["analysis"]["spectral_samples_used"] = int(Xs.shape[0])
                    results["analysis"]["spectral_feature_length_used"] = int(n_feat)
                    results["analysis"]["spectral_downsample_stride_samples"] = int(step_s)
                    results["analysis"]["spectral_downsample_stride_features"] = int(step_f)
            except Exception:
                # If fallback spectrum fails, keep metrics as None.
                pass

        # Feature correlation (mean |corr| off-diagonal) — high redundancy can amplify leakage risk.
        if X.shape[0] >= 8 and X.shape[1] >= 2:
            try:
                k = int(min(X.shape[1], 64))
                with np.errstate(all="ignore"):
                    corr = np.corrcoef(X[:, :k], rowvar=False)
                corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                off = corr[~np.eye(corr.shape[0], dtype=bool)]
                mean_abs_feature_corr = float(np.mean(np.abs(off))) if off.size else 0.0
                results["analysis"]["mean_abs_feature_corr"] = mean_abs_feature_corr
            except Exception:
                pass

        # Conservative ATLAS mappings (avoid overclaiming; only fire on strong signals)
        # AML.T0024.000 (membership inference) — very low value-entropy can correlate with privacy risk in some pipelines.
        if flat.size and entropy_norm <= 0.35:
            conf = float(min(1.0, (0.35 - entropy_norm) / 0.35))
            self._add_detection(
                "AML.T0024.000",
                f"Low value-distribution entropy (H_norm={entropy_norm:.2f}) — privacy risk indicator",
                confidence=conf,
            )

        # AML.T0043 (craft adversarial data) — broadband/high-frequency energy + heavy tails/outliers
        if hf_ratio_mean is not None and spectral_entropy_norm is not None:
            if hf_ratio_mean >= 0.35 and spectral_entropy_norm >= 0.75:
                hf_score = min(1.0, (hf_ratio_mean - 0.35) / 0.25)
                ent_score = min(1.0, (spectral_entropy_norm - 0.75) / 0.25)
                out_score = min(1.0, max(0.0, (outlier_ratio - 0.05) / 0.15))
                conf = float(max(0.5 * hf_score + 0.5 * ent_score, out_score))
                self._add_detection(
                    "AML.T0043",
                    f"Broadband/high-frequency anomaly (R_HF={hf_ratio_mean:.2f}, H_spec_norm={spectral_entropy_norm:.2f}, outliers={outlier_ratio:.1%})",
                    confidence=conf,
                )

        # AML.T0024.001 (invert AI model) — only if redundancy is extreme and indicators suggest low-noise updates.
        if mean_abs_feature_corr >= 0.95 and entropy_norm <= 0.60:
            if hf_ratio_mean is None or hf_ratio_mean <= 0.25:
                corr_score = min(1.0, (mean_abs_feature_corr - 0.95) / 0.05)
                ent_score = min(1.0, (0.60 - entropy_norm) / 0.60)
                hf_score = 1.0 if hf_ratio_mean is None else min(1.0, (0.25 - hf_ratio_mean) / 0.25)
                conf = float(max(0.0, min(1.0, 0.5 * corr_score + 0.3 * ent_score + 0.2 * hf_score)))
                self._add_detection(
                    "AML.T0024.001",
                    f"Extremely high feature redundancy (mean|corr|={mean_abs_feature_corr:.2f}) with low-noise indicators — potential inversion surface",
                    confidence=conf,
                )

        # Threshold-check explanations (always available, even if nothing triggered).
        checks: List[Dict[str, Any]] = []

        def _rule(metric: str, op: str, threshold: float, value: Any) -> Dict[str, Any]:
            try:
                v = float(value) if value is not None else None
            except Exception:
                v = None
            passed = None
            if v is not None:
                if op == "<=":
                    passed = bool(v <= threshold)
                elif op == ">=":
                    passed = bool(v >= threshold)
            return {"metric": metric, "op": op, "threshold": float(threshold), "value": v, "passed": passed}

        # AML.T0020 – Poison Training Data
        poison_trigger = bool(
            row_outlier_frac is not None
            and row_outlier_max_rz is not None
            and float(row_outlier_frac) >= 0.02
            and float(row_outlier_max_rz) >= 8.0
        )
        poison_conf = 0.0
        if poison_trigger:
            frac_score = min(1.0, (float(row_outlier_frac) - 0.02) / 0.08)  # 2%→10%
            rz_score = min(1.0, (float(row_outlier_max_rz) - 8.0) / 8.0)     # 8→16
            poison_conf = float(0.5 * frac_score + 0.5 * rz_score)
        checks.append(
            {
                "mitre_id": "AML.T0020",
                "name": _atlas_meta("AML.T0020").get("name", "Unknown"),
                "triggered": poison_trigger,
                "confidence_if_triggered": poison_conf,
                "prereqs": {"n_samples>=8": bool(n_samples >= 8)},
                "rules": [
                    _rule("row_norm_outlier_frac", ">=", 0.02, row_outlier_frac),
                    _rule("row_norm_outlier_max_rz", ">=", 8.0, row_outlier_max_rz),
                ],
                "notes": "Uses robust row-norm outliering as a poisoning/contamination proxy.",
            }
        )

        # AML.T0024.000 – Infer Training Data Membership
        mem_trigger = bool(flat.size and float(entropy_norm) <= 0.35)
        mem_conf = float(min(1.0, (0.35 - float(entropy_norm)) / 0.35)) if mem_trigger else 0.0
        checks.append(
            {
                "mitre_id": "AML.T0024.000",
                "name": _atlas_meta("AML.T0024.000").get("name", "Unknown"),
                "triggered": mem_trigger,
                "confidence_if_triggered": mem_conf,
                "prereqs": {"nonempty": bool(flat.size > 0)},
                "rules": [_rule("entropy_norm", "<=", 0.35, entropy_norm)],
                "notes": "Low value-distribution entropy is treated as a privacy-risk indicator (heuristic).",
            }
        )

        # AML.T0043 – Craft Adversarial Data
        has_spec = bool(hf_ratio_mean is not None and spectral_entropy_norm is not None)
        craft_trigger = bool(has_spec and float(hf_ratio_mean) >= 0.35 and float(spectral_entropy_norm) >= 0.75)
        craft_conf = 0.0
        if craft_trigger:
            hf_score = min(1.0, (float(hf_ratio_mean) - 0.35) / 0.25)
            ent_score = min(1.0, (float(spectral_entropy_norm) - 0.75) / 0.25)
            out_score = min(1.0, max(0.0, (float(outlier_ratio) - 0.05) / 0.15))
            craft_conf = float(max(0.5 * hf_score + 0.5 * ent_score, out_score))
        checks.append(
            {
                "mitre_id": "AML.T0043",
                "name": _atlas_meta("AML.T0043").get("name", "Unknown"),
                "triggered": craft_trigger,
                "confidence_if_triggered": craft_conf,
                "prereqs": {
                    "spectral_available": bool(has_spec),
                    "spectral_axis": (spectral_axis if spectral_axis is not None else None),
                },
                "rules": [
                    _rule("hf_ratio_mean", ">=", 0.35, hf_ratio_mean),
                    _rule("spectral_entropy_norm", ">=", 0.75, spectral_entropy_norm),
                    _rule("outlier_ratio", ">=", 0.05, outlier_ratio),
                ],
                "notes": "Requires strong broadband/high-frequency + high spectral entropy signals.",
            }
        )

        # AML.T0024.001 – Invert AI Model
        has_corr = bool(n_samples >= 8 and n_features >= 2)
        inv_prereq = bool(has_corr)
        inv_trigger = bool(
            inv_prereq
            and float(mean_abs_feature_corr) >= 0.95
            and float(entropy_norm) <= 0.60
            and (hf_ratio_mean is None or float(hf_ratio_mean) <= 0.25)
        )
        inv_conf = 0.0
        if inv_trigger:
            corr_score = min(1.0, (float(mean_abs_feature_corr) - 0.95) / 0.05)
            ent_score = min(1.0, (0.60 - float(entropy_norm)) / 0.60)
            hf_score = 1.0 if hf_ratio_mean is None else min(1.0, (0.25 - float(hf_ratio_mean)) / 0.25)
            inv_conf = float(max(0.0, min(1.0, 0.5 * corr_score + 0.3 * ent_score + 0.2 * hf_score)))
        checks.append(
            {
                "mitre_id": "AML.T0024.001",
                "name": _atlas_meta("AML.T0024.001").get("name", "Unknown"),
                "triggered": inv_trigger,
                "confidence_if_triggered": inv_conf,
                "prereqs": {"n_samples>=8": bool(n_samples >= 8), "n_features>=2": bool(n_features >= 2)},
                "rules": [
                    _rule("mean_abs_feature_corr", ">=", 0.95, mean_abs_feature_corr if has_corr else None),
                    _rule("entropy_norm", "<=", 0.60, entropy_norm),
                    _rule("hf_ratio_mean", "<=", 0.25, hf_ratio_mean),
                ],
                "notes": "Only fires on extreme redundancy + low-noise indicators (conservative).",
            }
        )

        results["analysis"]["threshold_checks"] = checks

        # Persist for `generate_report()`
        self._last_results = results

        return results
    
    def _add_detection(self, mitre_id: str, description: str, confidence: float):
        """Add a vulnerability detection with mappings."""
        technique = _atlas_meta(mitre_id)
        
        detection = {
            "mitre_id": mitre_id,
            "mitre_name": technique.get("name", "Unknown"),
            "mitre_tactics": technique.get("tactics", []),
            "mitre_url": technique.get("url"),
            "description": description,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        # Map to OWASP LLM Top 10
        for owasp_id, owasp_info in OWASP_LLM_TOP10.items():
            if mitre_id in owasp_info["mitre_atlas"]:
                detection["owasp_llm"] = detection.get("owasp_llm", [])
                detection["owasp_llm"].append({
                    "id": owasp_id,
                    "name": owasp_info["name"],
                    "severity": owasp_info["severity"]
                })
        
        self.detected_vectors.append(detection)
        
        if self.verbose:
            print(f"   🔴 {mitre_id}: {detection['mitre_name']}")
            print(f"      {description}")
            print(f"      Confidence: {confidence:.1%}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive vulnerability report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_detections": len(self.detected_vectors),
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "detections": self.detected_vectors,
            "analysis": (self._last_results.get("analysis", {}) if isinstance(self._last_results, dict) else {}),
            "mitre_atlas_coverage": {},
            "owasp_coverage": [],
            "red_team_guidance": [],
            "blue_team_guidance": []
        }
        
        # Calculate severity distribution
        for detection in self.detected_vectors:
            conf = detection["confidence"]
            if conf >= 0.9:
                report["summary"]["critical"] += 1
            elif conf >= 0.7:
                report["summary"]["high"] += 1
            elif conf >= 0.5:
                report["summary"]["medium"] += 1
            else:
                report["summary"]["low"] += 1
            
            # MITRE ATLAS coverage
            mitre_id = detection["mitre_id"]
            if mitre_id not in report["mitre_atlas_coverage"]:
                report["mitre_atlas_coverage"][mitre_id] = {
                    "name": detection["mitre_name"],
                    "tactics": detection["mitre_tactics"],
                    "count": 0,
                    "max_confidence": 0
                }
            report["mitre_atlas_coverage"][mitre_id]["count"] += 1
            report["mitre_atlas_coverage"][mitre_id]["max_confidence"] = max(
                report["mitre_atlas_coverage"][mitre_id]["max_confidence"],
                detection["confidence"]
            )
            
            # OWASP coverage
            for owasp in detection.get("owasp_llm", []):
                if owasp["id"] not in [o["id"] for o in report["owasp_coverage"]]:
                    report["owasp_coverage"].append(owasp)
        
        # Generate guidance
        report["red_team_guidance"] = self._generate_red_team_guidance()
        report["blue_team_guidance"] = self._generate_blue_team_guidance()
        
        return report
    
    def _generate_red_team_guidance(self) -> List[Dict[str, str]]:
        """Generate red team actionable guidance (test-focused; no evasion/exfil instructions)."""
        guidance = []
        
        mitre_to_guidance = {
            "AML.T0020": {
                "attack": "Training Data Poisoning",
                "technique": "Evaluate poisoning resilience using controlled, authorized test sets (do not deploy unreviewed data).",
                "tools": "neurinspectre analyze-attack-vectors --target-data <train_artifact.npy> --verbose; neurinspectre drift-detect --data <x.npy> --reference <clean.npy>",
                "next_step": "Report detection performance + mitigations; validate that alerts fire on known-bad cases with low false positives."
            },
            "AML.T0024.001": {
                "attack": "Model Inversion Attack",
                "technique": "Run an approved privacy evaluation to measure reconstruction risk and validate mitigations.",
                "tools": "neurinspectre gradient_inversion recover --gradients <grads.npy> --out-prefix _cli_runs/ginv_",
                "next_step": "Document conditions that increase risk (batching, clipping/noise, aggregation granularity) and verify improvements after mitigations."
            },
            "AML.T0043": {
                "attack": "Adversarial Example Crafting",
                "technique": "Perform controlled robustness evaluation and measure detection sensitivity under your authorized threat model.",
                "tools": "neurinspectre adversarial-detect <data.npy> --detector-type all --threshold <t> --output-dir _cli_runs/adv; neurinspectre spectral --input <x.npy> --baseline <clean.npy> --plot _cli_runs/spectral.png",
                "next_step": "Report FP/FN + stability across seeds/runs; share flagged frequencies/metrics and recommended defensive controls."
            },
            "AML.T0024.000": {
                "attack": "Training Data Membership Inference",
                "technique": "Evaluate membership-inference risk on approved datasets and quantify confidence leakage.",
                "tools": "neurinspectre analyze-attack-vectors --target-data <x.npy> --verbose; neurinspectre anomaly --input <x.npy> --method robust_z --z 3.0",
                "next_step": "Validate mitigations (regularization, output perturbation, privacy accounting) by re-running the same evaluation suite."
            },
        }
        
        for detection in self.detected_vectors:
            mitre_id = detection["mitre_id"]
            if mitre_id in mitre_to_guidance:
                g = mitre_to_guidance[mitre_id].copy()
                g["mitre_id"] = mitre_id
                g["confidence"] = detection["confidence"]
                if g not in guidance:
                    guidance.append(g)
        
        return guidance
    
    def _generate_blue_team_guidance(self) -> List[Dict[str, str]]:
        """Generate blue team defensive guidance."""
        guidance = []
        
        mitre_to_defense = {
            "AML.T0020": {
                "defense": "Gradient Privacy Protection",
                "technique": "Apply differential privacy (ε=1.0), gradient clipping (max_norm=1.0)",
                "monitoring": "Alert on gradient norm > 10, spectral anomalies",
                "tools": "neurinspectre comprehensive-scan <data.npy> -o <dir>; neurinspectre realtime-monitor <dir> --threshold 0.9 -o <out>"
            },
            "AML.T0024.001": {
                "defense": "Model Inversion Prevention",
                "technique": "Add output noise, limit query rates, restrict API access",
                "monitoring": "Monitor for reconstruction patterns in queries",
                "tools": "neurinspectre realtime-monitor <dir> --threshold 0.9 -o <out>"
            },
            "AML.T0043": {
                "defense": "Adversarial Robustness",
                "technique": "Adversarial training, input sanitization, ensemble models",
                "monitoring": "Detect unusual input distributions",
                "tools": "neurinspectre adversarial-detect <data.npy> -o <dir>; neurinspectre evasion-detect <data.npy> -o <dir>"
            },
            "AML.T0024.000": {
                "defense": "Membership Inference Mitigation",
                "technique": "Regularization, early stopping, output perturbation",
                "monitoring": "Track confidence distributions",
                "tools": "NeurInSpectre anomaly (privacy) + evaluation harness"
            },
        }
        
        for detection in self.detected_vectors:
            mitre_id = detection["mitre_id"]
            if mitre_id in mitre_to_defense:
                g = mitre_to_defense[mitre_id].copy()
                g["mitre_id"] = mitre_id
                g["priority"] = "HIGH" if detection["confidence"] >= 0.7 else "MEDIUM"
                if g not in guidance:
                    guidance.append(g)
        
        return guidance


def run_attack_vector_analysis(args) -> int:
    """
    Run attack vector analysis CLI command.
    """
    print("\n🔴 NeurInSpectre: Attack Vector Analysis (MITRE ATLAS / OWASP mapping)")
    print("=" * 70)
    
    # Load target data
    target_data = getattr(args, "target_data", None)
    if target_data:
        try:
            data = np.load(target_data)
            print(f"✅ Loaded: {target_data} ({data.shape})")
        except Exception as e:
            print(f"⚠️  Could not load {target_data}: {e}")
            print("   No synthetic/demo fallback is generated.")
            return 1
    else:
        print("⚠️  No --target-data provided. Nothing to analyze.")
        print("   No synthetic/demo fallback is generated.")
        return 1
    
    # Initialize analyzer
    verbose = getattr(args, 'verbose', False)
    analyzer = AttackVectorAnalyzer(verbose=verbose)
    
    # Run analysis
    print(f"\n📊 Analyzing data ({data.shape})...")
    _ = analyzer.analyze_data(data)
    
    # Generate report
    report = analyzer.generate_report()
    
    # Display results
    print(f"\n{'='*70}")
    print("📋 ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"   Total Detections: {report['summary']['total_detections']}")
    print(f"   🔴 Critical: {report['summary']['critical']}")
    print(f"   🟠 High: {report['summary']['high']}")
    print(f"   🟡 Medium: {report['summary']['medium']}")
    print(f"   🟢 Low: {report['summary']['low']}")
    
    # CVE Mapping (intentionally unsupported for correctness)
    if getattr(args, 'cve_mapping', False):
        print(f"\n{'='*70}")
        print("🛡️  CVE MAPPING (NOT SUPPORTED)")
        print(f"{'='*70}")
        print("   This module does not emit real CVEs (CVEs are product/version specific).")
        print("   Use dependency scanners instead, e.g. `pip-audit` / `npm audit`.")

    # Verbose: explain what the input looks like and why thresholds did/didn't trigger.
    if verbose:
        analysis = report.get("analysis", {}) or {}
        prof = analysis.get("input_profile", {}) or {}
        checks = list(analysis.get("threshold_checks") or [])

        print(f"\n{'='*70}")
        print("🧪 VERBOSE: INPUT PROFILE + THRESHOLD CHECKS")
        print(f"{'='*70}")

        if prof:
            print("Input profile:")
            print(f"   • data_shape: {prof.get('data_shape')}")
            print(f"   • n_samples: {prof.get('n_samples')}  n_features: {prof.get('n_features')}")
            print(f"   • low_dimensional: {prof.get('low_dimensional')}")
            if prof.get("diff_rms_over_signal_rms") is not None:
                print(f"   • diff_rms_over_signal_rms: {float(prof.get('diff_rms_over_signal_rms')):.4f}")
            if prof.get("likely_summary_stats_3col"):
                print("   • format_hint: likely 3-column summary-stat time series")
                print(f"     - hint: {prof.get('summary_stats_3col_hint')}")
                diag = prof.get("summary_stats_3col_diagnostics") or {}
                if diag:
                    print(f"     - frac_negative_by_col: {diag.get('frac_negative_by_col')}")
                    print(f"     - frac_col3_ge_col1: {diag.get('frac_col3_ge_col1')}")
            if prof.get("interpretation_hint"):
                print(f"   • note: {prof.get('interpretation_hint')}")
            print()

        print("Computed metrics (selected):")
        stats = analysis.get("statistics") or {}
        if stats:
            print(f"   • mean={float(stats.get('mean', 0.0)):.6g} std={float(stats.get('std', 0.0)):.6g} rms={float(stats.get('rms', 0.0)):.6g}")
            print(f"   • skewness={float(stats.get('skewness', 0.0)):.3f} kurtosis_excess={float(stats.get('kurtosis_excess', 0.0)):.3f}")
        if "entropy_norm" in analysis:
            print(f"   • entropy_norm={float(analysis.get('entropy_norm', 0.0)):.3f} (entropy_bits={float(analysis.get('entropy_bits', 0.0)):.3f})")
        if "outlier_ratio" in analysis:
            print(f"   • outlier_ratio={float(analysis.get('outlier_ratio', 0.0)):.3f}")
        if "row_norm_outlier_frac" in analysis:
            print(f"   • row_norm_outlier_frac={float(analysis.get('row_norm_outlier_frac', 0.0)):.3f}  row_norm_outlier_max_rz={float(analysis.get('row_norm_outlier_max_rz', 0.0)):.2f}")
        if "mean_abs_feature_corr" in analysis:
            print(f"   • mean_abs_feature_corr={float(analysis.get('mean_abs_feature_corr', 0.0)):.3f}")
        if "hf_ratio_mean" in analysis:
            print(f"   • hf_ratio_mean={float(analysis.get('hf_ratio_mean', 0.0)):.3f}")
        if "spectral_entropy_norm" in analysis:
            print(f"   • spectral_entropy_norm={float(analysis.get('spectral_entropy_norm', 0.0)):.3f} (spectral_entropy_bits={float(analysis.get('spectral_entropy_bits', 0.0)):.3f})")
        if "spectral_axis" in analysis:
            print(f"   • spectral_axis={analysis.get('spectral_axis')}")
        print()

        if checks:
            print("Threshold checks (why techniques did/didn't trigger):")
            for c in checks:
                mid = c.get("mitre_id", "Unknown")
                name = c.get("name", "Unknown")
                trig = bool(c.get("triggered", False))
                conf_if = float(c.get("confidence_if_triggered", 0.0))
                print(f"   • {mid} — {name}: {'TRIGGERED' if trig else 'not triggered'} (confidence_if_triggered={conf_if:.3f})")
                prereqs = c.get("prereqs") or {}
                if prereqs:
                    print(f"     - prereqs: {prereqs}")
                for r in (c.get("rules") or []):
                    metric = r.get("metric")
                    op = r.get("op")
                    thr = r.get("threshold")
                    val = r.get("value")
                    passed = r.get("passed")
                    status = "PASS" if passed is True else "FAIL" if passed is False else "N/A"
                    if val is None:
                        print(f"     - {metric} {op} {thr}: value=N/A ({status})")
                    else:
                        print(f"     - {metric} {op} {thr}: value={float(val):.6g} ({status})")
                if c.get("notes"):
                    print(f"     - notes: {c.get('notes')}")
            print()
    
    # MITRE ATLAS Coverage
    print(f"\n{'='*70}")
    print("🎯 MITRE ATLAS COVERAGE")
    print(f"{'='*70}")
    if not report["mitre_atlas_coverage"]:
        print("   (no MITRE ATLAS techniques flagged for this dataset)")
    else:
        for mitre_id, info in report["mitre_atlas_coverage"].items():
            conf = info["max_confidence"]
            severity = "CRITICAL" if conf >= 0.9 else "HIGH" if conf >= 0.7 else "MEDIUM" if conf >= 0.5 else "LOW"
            print(f"   {mitre_id}: {info['name']}")
            print(f"      Tactics: {', '.join(info['tactics'])}")
            print(f"      Confidence: {conf:.1%} ({severity})")
            print(f"      Occurrences: {info['count']}")
    
    # OWASP LLM Top 10
    if report["owasp_coverage"]:
        print(f"\n{'='*70}")
        print("📜 OWASP LLM TOP 10 MAPPING")
        print(f"{'='*70}")
        for owasp in report["owasp_coverage"]:
            print(f"   {owasp['id']}: {owasp['name']} ({owasp['severity']})")
    
    # Red Team Guidance
    print(f"\n{'='*70}")
    print("🔴 RED TEAM GUIDANCE")
    print(f"{'='*70}")
    if not report["red_team_guidance"]:
        print("   (no red-team guidance generated — no techniques flagged)")
    else:
        for guidance in report["red_team_guidance"][:5]:
            print(f"   [{guidance['mitre_id']}] {guidance['attack']}")
            print(f"      Technique: {guidance['technique']}")
            print(f"      Tools: {guidance['tools']}")
            print(f"      Next: {guidance['next_step']}")
            print()
    
    # Blue Team Guidance
    print(f"{'='*70}")
    print("🔵 BLUE TEAM GUIDANCE")
    print(f"{'='*70}")
    if not report["blue_team_guidance"]:
        print("   (no blue-team guidance generated — no techniques flagged)")
    else:
        for guidance in report["blue_team_guidance"][:5]:
            print(f"   [{guidance['mitre_id']}] {guidance['defense']} ({guidance['priority']})")
            print(f"      Technique: {guidance['technique']}")
            print(f"      Monitoring: {guidance['monitoring']}")
            print()
    
    # Save report
    output_dir = Path(getattr(args, 'output_dir', '_cli_runs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "attack_vector_analysis.json"
    with open(output_file, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            return obj
        
        json.dump(report, f, indent=2, default=convert)
    
    print(f"\n💾 Report saved: {output_file}")
    print(f"{'='*70}\n")
    
    return 0


def _normalize_attack_vector(v: str) -> str:
    v = (v or "").strip().lower()
    v = re.sub(r"[^a-z0-9]+", "_", v).strip("_")
    return v


def _split_attack_vectors(raw: str) -> List[str]:
    if raw is None:
        return []
    parts = [p.strip() for p in str(raw).split(",")]
    out: List[str] = []
    for p in parts:
        key = _normalize_attack_vector(p)
        if key:
            out.append(key)
    # stable unique
    uniq: List[str] = []
    seen = set()
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def _attack_vector_catalog() -> Dict[str, Dict[str, Any]]:
    """Small, CLI-facing catalog for mapping common vectors -> ATLAS IDs + controls.

    Keep this precise and only reference commands that exist in the unified CLI.
    """
    return {
        # Data leakage / reconstruction via gradients
        "gradient_inversion": {
            "summary": "Reconstruct private training data from leaked gradients/updates.",
            "atlas_ids": ["AML.T0024.001"],
            "controls": [
                ("P0", "Stop per-sample gradient exposure", "Avoid exposing raw gradients; aggregate and sanitize updates."),
                ("P0", "Clip + noise", "Apply gradient clipping and calibrated noise (DP) to reduce leakage."),
                ("P1", "Rate-limit and authenticate", "Reduce attacker iteration speed and restrict access to sensitive endpoints."),
                ("P1", "Monitor leakage signals", "Track unusually high norms / repeated queries / reconstruction-like patterns."),
            ],
            "neurinspectre_commands": [
                "neurinspectre analyze-attack-vectors --target-data <gradients.npy> -o <dir>",
                "neurinspectre comprehensive-scan <activations.npy> -o <dir> --threshold 0.9",
                "neurinspectre realtime-monitor <data_dir> --threshold 0.9 -o <dir>",
            ],
        },
        # Black-box extraction / model theft
        "model_extraction": {
            "summary": "Steal model behavior/weights via repeated queries and distillation.",
            "atlas_ids": ["AML.T0024.002"],
            "controls": [
                ("P0", "Strict auth + quota", "Enforce API keys, per-tenant quotas, and burst rate limits."),
                ("P0", "Abuse monitoring", "Detect scraping/query automation (high volume, low diversity, systematic probing)."),
                ("P1", "Output hardening", "Reduce over-precise outputs where possible (rounding, top-k, calibrated confidence)."),
                ("P1", "Watermarking / canaries", "Embed detectable behavior to flag unauthorized replicas (defender validation)."),
            ],
            "neurinspectre_commands": [
                "neurinspectre analyze-attack-vectors --target-data <observations.npy> -o <dir>",
                "neurinspectre mitre-atlas list techniques",
            ],
        },
        # Common synonym
        "model_theft": {"alias": "model_extraction"},
        "model_stealing": {"alias": "model_extraction"},
        "gradient_leakage": {"alias": "gradient_inversion"},
    }


def _resolve_catalog_entry(key: str, catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    entry = catalog.get(key)
    if entry is None:
        return {
            "attack_vector": key,
            "summary": "Unknown/unsupported vector label (no catalog entry).",
            "atlas": [],
            "countermeasures": [],
            "neurinspectre_commands": ["neurinspectre analyze-attack-vectors --target-data <file.npy> -o <dir>"],
        }
    if "alias" in entry:
        return _resolve_catalog_entry(str(entry["alias"]), catalog)
    return entry


def _collect_owasp_for_atlas_ids(atlas_ids: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for owasp_id, info in OWASP_LLM_TOP10.items():
        if any(tid in (info.get("mitre_atlas") or []) for tid in atlas_ids):
            out.append({"id": owasp_id, "name": info.get("name"), "severity": info.get("severity")})
    return out


def run_recommend_countermeasures(args) -> int:
    """Recommend countermeasures for an explicit set of attack vectors (or auto-detected via --target-data)."""
    print("\n🔵 NeurInSpectre: Countermeasure Recommendations Playbook")
    print("=" * 70)

    threat_level = str(getattr(args, "threat_level", "high") or "high").lower()
    output_dir = Path(getattr(args, "output_dir", "_cli_runs") or "_cli_runs")
    verbose = bool(getattr(args, "verbose", False))

    catalog = _attack_vector_catalog()

    # 1) Determine which vectors to build a playbook for.
    vectors = _split_attack_vectors(getattr(args, "attack_vectors", None))
    target_data = getattr(args, "target_data", None)

    if not vectors and target_data:
        # Auto-detect via analysis
        try:
            data = np.load(target_data, allow_pickle=True)
        except Exception as e:
            print(f"❌ Failed to load --target-data {target_data!r}: {e}")
            return 2

        analyzer = AttackVectorAnalyzer(verbose=verbose)
        _ = analyzer.analyze_data(np.asarray(data))
        report = analyzer.generate_report()
        atlas_ids = sorted(list(report.get("mitre_atlas_coverage", {}).keys()))

        # Map detected ATLAS IDs back to catalog keys (best-effort, many-to-one).
        id_to_key = {
            "AML.T0024.001": "gradient_inversion",
            "AML.T0024.002": "model_extraction",
        }
        vectors = []
        for tid in atlas_ids:
            k = id_to_key.get(tid)
            if k:
                vectors.append(k)
        # de-dup
        vectors = _split_attack_vectors(",".join(vectors))

    if not vectors:
        print("❌ Missing inputs.")
        print("   Provide either:")
        print("   • --attack-vectors \"gradient_inversion,model_extraction\"")
        print("   • --target-data <file.npy>   (to auto-detect vectors)")
        return 2

    # 2) Build playbook entries.
    entries: List[Dict[str, Any]] = []
    all_atlas_ids: List[str] = []
    for vec in vectors:
        entry = _resolve_catalog_entry(vec, catalog)
        atlas_ids = entry.get("atlas_ids") or []
        atlas_meta = [_atlas_meta(tid) for tid in atlas_ids]
        all_atlas_ids.extend(atlas_ids)

        countermeasures = []
        for prio, control, how in entry.get("controls", []):
            countermeasures.append(
                {
                    "priority": prio,
                    "control": control,
                    "how": how,
                    "threat_level": threat_level,
                }
            )

        entries.append(
            {
                "attack_vector": vec,
                "summary": entry.get("summary"),
                "mitre_atlas": atlas_meta,
                "owasp_llm_top10": _collect_owasp_for_atlas_ids(atlas_ids),
                "countermeasures": countermeasures,
                "neurinspectre_commands": entry.get("neurinspectre_commands", []),
            }
        )

    all_atlas_ids = sorted(list({tid for tid in all_atlas_ids if tid}))

    # 3) Save outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "countermeasures_playbook.json"
    out_md = output_dir / "countermeasures_playbook.md"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "threat_level": threat_level,
        "attack_vectors": vectors,
        "mitre_atlas_ids": all_atlas_ids,
        "entries": entries,
    }

    out_json.write_text(json.dumps(payload, indent=2))

    # Human-readable markdown playbook
    md_lines: List[str] = []
    md_lines.append("# NeurInSpectre — Countermeasure Recommendations Playbook")
    md_lines.append("")
    md_lines.append(f"- Threat level: **{threat_level.upper()}**")
    md_lines.append(f"- Attack vectors: **{', '.join(vectors)}**")
    md_lines.append("")
    for ent in entries:
        md_lines.append(f"## {ent['attack_vector']}")
        md_lines.append("")
        md_lines.append(f"**Summary**: {ent.get('summary')}")
        md_lines.append("")
        if ent.get("mitre_atlas"):
            md_lines.append("**MITRE ATLAS mapping**:")
            for t in ent["mitre_atlas"]:
                tactics = ", ".join(t.get("tactics") or []) if isinstance(t, dict) else ""
                name = t.get("name") if isinstance(t, dict) else "Unknown"
                tid = t.get("id") if isinstance(t, dict) else ""
                md_lines.append(f"- `{tid}` — {name}" + (f" (tactics: {tactics})" if tactics else ""))
            md_lines.append("")
        if ent.get("owasp_llm_top10"):
            md_lines.append("**OWASP LLM Top 10**:")
            for o in ent["owasp_llm_top10"]:
                md_lines.append(f"- `{o['id']}` — {o.get('name')} ({o.get('severity')})")
            md_lines.append("")
        md_lines.append("**Countermeasures**:")
        for c in ent.get("countermeasures", []):
            md_lines.append(f"- **{c['priority']}**: {c['control']} — {c['how']}")
        md_lines.append("")
        if ent.get("neurinspectre_commands"):
            md_lines.append("**NeurInSpectre commands**:")
            for cmd in ent["neurinspectre_commands"]:
                md_lines.append(f"- `{cmd}`")
            md_lines.append("")

    out_md.write_text("\n".join(md_lines).rstrip() + "\n")

    print(f"💾 Playbook JSON: {out_json}")
    print(f"📄 Playbook MD:   {out_md}")
    return 0

