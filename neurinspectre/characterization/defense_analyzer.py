"""
Automated defense characterization and obfuscation detection.

This is Layer 1 of NeurInSpectre's pipeline. It collects gradient samples,
computes ETD scores, fits Volterra kernels, and recommends attack settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mathematical.krylov import analyze_krylov_projection
from ..mathematical.volterra import compute_volterra_correlation, fit_volterra_kernel
from .layer1_spectral import compute_spectral_features


class ObfuscationType(Enum):
    """Defense obfuscation taxonomy."""

    SHATTERED = "shattered"
    STOCHASTIC = "stochastic"
    VANISHING = "vanishing"
    RL_TRAINED = "rl_trained"
    EXPLODING = "exploding"
    HYBRID = "hybrid"
    NONE = "none"


@dataclass
class DefenseCharacterization:
    """Complete defense characterization profile."""

    obfuscation_types: List[ObfuscationType]
    etd_score: float
    alpha_volterra: float
    gradient_variance: float
    jacobian_rank: float
    autocorr_timescale: float

    requires_bpda: bool
    requires_eot: bool
    requires_mapgd: bool

    recommended_eot_samples: int
    recommended_memory_length: int

    confidence: float
    metadata: Dict

    def __repr__(self) -> str:
        obf_str = ", ".join([o.value for o in self.obfuscation_types])
        return (
            "DefenseCharacterization(\n"
            f"  obfuscation=[{obf_str}],\n"
            f"  ETD={self.etd_score:.3f}, alpha={self.alpha_volterra:.3f},\n"
            f"  BPDA={self.requires_bpda}, EOT={self.requires_eot}, MA-PGD={self.requires_mapgd},\n"
            f"  confidence={self.confidence:.2f}\n"
            ")"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "obfuscation_types": [o.value for o in self.obfuscation_types],
            "etd_score": float(self.etd_score),
            "alpha_volterra": float(self.alpha_volterra),
            "gradient_variance": float(self.gradient_variance),
            "jacobian_rank": float(self.jacobian_rank),
            "autocorr_timescale": float(self.autocorr_timescale),
            "requires_bpda": bool(self.requires_bpda),
            "requires_eot": bool(self.requires_eot),
            "requires_mapgd": bool(self.requires_mapgd),
            "recommended_eot_samples": int(self.recommended_eot_samples),
            "recommended_memory_length": int(self.recommended_memory_length),
            "confidence": float(self.confidence),
            "chosen_attack": getattr(self, "chosen_attack", None) or (self.metadata or {}).get("chosen_attack"),
            "selected_attack_impl": getattr(self, "selected_attack_impl", None)
            or (self.metadata or {}).get("selected_attack_impl"),
            "metadata": self.metadata,
        }


class DefenseAnalyzer:
    """
    Automated defense characterization system.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 50,
        n_probe_images: int = 100,
        device: str = "cuda",
        verbose: bool = True,
        krylov_dim: int = 20,
        thresholds: Dict[str, Any] | None = None,
        volterra_gradient_source: str = "pre_optimizer",
        volterra_optimizer: str = "sgd",
        volterra_optimizer_beta1: float = 0.9,
        volterra_optimizer_beta2: float = 0.999,
        volterra_optimizer_eps: float = 1e-8,
    ):
        self.model = model.to(device)
        self.n_samples = int(n_samples)
        self.n_probe_images = int(n_probe_images)
        self.device = device
        self.verbose = verbose
        self.krylov_dim = int(krylov_dim)

        # Tier 2: optimizer-confound control for Volterra alpha.
        #
        # Default behavior (pre_optimizer) matches the paper intent: fit alpha on the
        # raw ∇x L gradients, not on any optimizer-smoothed update direction.
        self.volterra_gradient_source = str(volterra_gradient_source).lower().strip()
        if self.volterra_gradient_source not in {"pre_optimizer", "post_optimizer"}:
            self.volterra_gradient_source = "pre_optimizer"
        self.volterra_optimizer = str(volterra_optimizer).lower().strip().replace("-", "_")
        self.volterra_optimizer_beta1 = float(volterra_optimizer_beta1)
        self.volterra_optimizer_beta2 = float(volterra_optimizer_beta2)
        self.volterra_optimizer_eps = float(volterra_optimizer_eps)

        # Thresholds
        self.ETD_THRESHOLD_MODERATE = 0.3
        self.ETD_THRESHOLD_SEVERE = 0.6
        # Layer 1 spectral features (Paper Section 3.1)
        self.SPECTRAL_ENTROPY_OBFUSCATED_THRESHOLD = 0.50
        self.HIGH_FREQ_RATIO_SHATTERED_THRESHOLD = 0.30
        self.ALPHA_RL_THRESHOLD = 0.7
        self.ALPHA_SHATTERED_THRESHOLD = 0.3
        self.GRAD_NORM_SHATTERED_THRESHOLD = 1e-5
        self.VARIANCE_SHATTERED_THRESHOLD = 1e-6
        self.VARIANCE_STOCHASTIC_THRESHOLD = 1e-5
        self.RANK_VANISHING_THRESHOLD = 0.5
        self.AUTOCORR_RL_THRESHOLD = 0.5
        # Spectral/Krylov heuristics (Paper Section 3.1.2)
        self.KRYLOV_REL_ERROR_SHATTERED = 0.5
        self.KRYLOV_NORM_RATIO_VANISHING = 0.3
        self.KRYLOV_NORM_GROWTH_SHATTERED = 0.3

        # Practical stability heuristics:
        # Below this, Volterra/autocorr estimates are often too noisy to support
        # strong claims (used for failure-mode reporting and conservative gating).
        self.MIN_GRADIENT_SEQUENCE_LEN = 64
        # Scale-free RMSE (RMSE / std(grad_history)) threshold for flagging a
        # poor Volterra fit (error comparable to the signal scale).
        #
        # This is used for failure-analysis reporting and confidence downweighting,
        # not as a hard gate for the full pipeline.
        # Empirically, clean/stable runs tend to produce rmse_scaled closer to ~1,
        # while heavily stochastic defenses push it higher. We flag >1.2 as "high"
        # to support failure-mode reporting without affecting core attack logic.
        self.VOLTERRA_RMSE_SCALED_MAX = 1.2

        self._threshold_overrides: Dict[str, Any] = {}
        if thresholds:
            self._apply_threshold_overrides(dict(thresholds))
        # Captures diagnostics from the most recent gradient sampling pass so
        # characterization artifacts can explain "degenerate" fields.
        self._last_grad_sample_info: Dict[str, Any] = {}

    def _forward_logits(self, x: torch.Tensor, *, use_bpda: bool) -> torch.Tensor:
        """
        Best-effort forward pass that optionally enables BPDA-style approximations.

        Many defenses are wrapped in `DefenseWrapper.forward(x, use_approximation=...)`.
        For plain `nn.Module` models (no such kwarg), we fall back to `model(x)`.
        """
        if use_bpda:
            try:
                return self.model(x, use_approximation=True)  # type: ignore[misc]
            except (TypeError, RuntimeError):
                return self.model(x)
        return self.model(x)

    @staticmethod
    def _is_no_grad_runtime_error(exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        return (
            "does not require grad" in msg
            or "does not have a grad_fn" in msg
            or "element 0 of tensors" in msg
        )

    def _safe_backward(self, loss: torch.Tensor) -> bool:
        """
        Backprop helper for characterization.

        Some defenses intentionally break autograd (e.g., hard/argmax-style transforms).
        For those, we return False so callers can try BPDA/fallback logic.
        """
        try:
            loss.backward()
            return True
        except RuntimeError as exc:
            if self._is_no_grad_runtime_error(exc):
                return False
            raise

    def _volterra_preprocess_gradients(self, grad_array: np.ndarray) -> np.ndarray:
        """
        Optionally transform gradients before Volterra fitting.

        Motivation (Tier 2 confound control):
        - Fitting alpha on optimizer-smoothed update directions can "manufacture"
          temporal memory effects from the optimizer itself (momentum/Adam/RMSProp),
          which is not defense-induced.
        - The default ("pre_optimizer") uses raw gradients unchanged.
        """
        src = str(getattr(self, "volterra_gradient_source", "pre_optimizer")).lower().strip()
        if src != "post_optimizer":
            return grad_array

        opt = str(getattr(self, "volterra_optimizer", "sgd")).lower().strip().replace("-", "_")
        beta1 = float(getattr(self, "volterra_optimizer_beta1", 0.9))
        beta2 = float(getattr(self, "volterra_optimizer_beta2", 0.999))
        eps = float(getattr(self, "volterra_optimizer_eps", 1e-8))
        eps = max(eps, 1e-12)

        T, D = int(grad_array.shape[0]), int(grad_array.shape[1])
        if T <= 0 or D <= 0:
            return grad_array

        out = np.zeros_like(grad_array, dtype=np.float64)

        if opt in {"sgd", "vanilla_sgd"}:
            out[:] = grad_array
            return out

        if opt in {"sgd_momentum", "momentum", "sgdm"}:
            v = np.zeros((D,), dtype=np.float64)
            b = float(np.clip(beta1, 0.0, 0.9999))
            for t in range(T):
                v = b * v + grad_array[t]
                out[t] = v
            return out

        if opt in {"rmsprop", "rms_prop"}:
            s = np.zeros((D,), dtype=np.float64)
            b = float(np.clip(beta2, 0.0, 0.9999))
            for t in range(T):
                g = grad_array[t]
                s = b * s + (1.0 - b) * (g * g)
                out[t] = g / (np.sqrt(s) + eps)
            return out

        if opt in {"adam"}:
            m = np.zeros((D,), dtype=np.float64)
            v = np.zeros((D,), dtype=np.float64)
            b1 = float(np.clip(beta1, 0.0, 0.9999))
            b2 = float(np.clip(beta2, 0.0, 0.9999))
            for t in range(T):
                g = grad_array[t]
                m = b1 * m + (1.0 - b1) * g
                v = b2 * v + (1.0 - b2) * (g * g)
                # Bias correction (standard Adam)
                t1 = float(t + 1)
                m_hat = m / (1.0 - (b1**t1)) if b1 < 1.0 else m
                v_hat = v / (1.0 - (b2**t1)) if b2 < 1.0 else v
                out[t] = m_hat / (np.sqrt(v_hat) + eps)
            return out

        # Unknown optimizer key: fall back to raw gradients (do not crash characterization).
        return grad_array

    def _apply_threshold_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Apply user-provided threshold overrides.

        This exists to support Tier 2 calibration: ROC/AUC-derived operating points
        can be loaded from JSON and injected here instead of hard-coding values.
        """

        def _canon(key: str) -> str:
            return str(key).strip().upper().replace("-", "_")

        allowed = {
            "ETD_THRESHOLD_MODERATE": float,
            "ETD_THRESHOLD_SEVERE": float,
            "SPECTRAL_ENTROPY_OBFUSCATED_THRESHOLD": float,
            "HIGH_FREQ_RATIO_SHATTERED_THRESHOLD": float,
            "ALPHA_RL_THRESHOLD": float,
            "ALPHA_SHATTERED_THRESHOLD": float,
            "GRAD_NORM_SHATTERED_THRESHOLD": float,
            "VARIANCE_SHATTERED_THRESHOLD": float,
            "VARIANCE_STOCHASTIC_THRESHOLD": float,
            "RANK_VANISHING_THRESHOLD": float,
            "AUTOCORR_RL_THRESHOLD": float,
            "KRYLOV_REL_ERROR_SHATTERED": float,
            "KRYLOV_NORM_RATIO_VANISHING": float,
            "KRYLOV_NORM_GROWTH_SHATTERED": float,
            "MIN_GRADIENT_SEQUENCE_LEN": int,
            "VOLTERRA_RMSE_SCALED_MAX": float,
        }

        for k, v in overrides.items():
            ck = _canon(str(k))
            caster = allowed.get(ck)
            if caster is None:
                continue
            try:
                casted = caster(v)
            except Exception:
                continue
            setattr(self, ck, casted)
            self._threshold_overrides[ck] = casted

    def _thresholds_dict(self) -> Dict[str, Any]:
        return {
            "ETD_THRESHOLD_MODERATE": float(self.ETD_THRESHOLD_MODERATE),
            "ETD_THRESHOLD_SEVERE": float(self.ETD_THRESHOLD_SEVERE),
            "SPECTRAL_ENTROPY_OBFUSCATED_THRESHOLD": float(self.SPECTRAL_ENTROPY_OBFUSCATED_THRESHOLD),
            "HIGH_FREQ_RATIO_SHATTERED_THRESHOLD": float(self.HIGH_FREQ_RATIO_SHATTERED_THRESHOLD),
            "ALPHA_RL_THRESHOLD": float(self.ALPHA_RL_THRESHOLD),
            "ALPHA_SHATTERED_THRESHOLD": float(self.ALPHA_SHATTERED_THRESHOLD),
            "GRAD_NORM_SHATTERED_THRESHOLD": float(self.GRAD_NORM_SHATTERED_THRESHOLD),
            "VARIANCE_SHATTERED_THRESHOLD": float(self.VARIANCE_SHATTERED_THRESHOLD),
            "VARIANCE_STOCHASTIC_THRESHOLD": float(self.VARIANCE_STOCHASTIC_THRESHOLD),
            "RANK_VANISHING_THRESHOLD": float(self.RANK_VANISHING_THRESHOLD),
            "AUTOCORR_RL_THRESHOLD": float(self.AUTOCORR_RL_THRESHOLD),
            "KRYLOV_REL_ERROR_SHATTERED": float(self.KRYLOV_REL_ERROR_SHATTERED),
            "KRYLOV_NORM_RATIO_VANISHING": float(self.KRYLOV_NORM_RATIO_VANISHING),
            "KRYLOV_NORM_GROWTH_SHATTERED": float(self.KRYLOV_NORM_GROWTH_SHATTERED),
            "MIN_GRADIENT_SEQUENCE_LEN": int(self.MIN_GRADIENT_SEQUENCE_LEN),
            "VOLTERRA_RMSE_SCALED_MAX": float(self.VOLTERRA_RMSE_SCALED_MAX),
            "overrides_applied": dict(self._threshold_overrides),
        }

    def characterize(
        self,
        data_loader: torch.utils.data.DataLoader,
        eps: float = 8 / 255,
    ) -> DefenseCharacterization:
        if self.verbose:
            print("[DefenseAnalyzer] Starting characterization...")
            print(
                f"  Collecting {self.n_samples} gradient samples from {self.n_probe_images} images"
            )

        gradients, images, labels = self._collect_gradient_samples(data_loader, eps)

        # Layer 1 (Paper Section 3.1): spectral entropy + high-frequency ratio.
        # We compute these from the gradient history (T, D) by averaging across D.
        spectral_layer1: Dict[str, Any] = {}
        try:
            if gradients and len(gradients) >= 2:
                grad_seq = np.asarray(gradients, dtype=np.float64)
                spectral_layer1 = compute_spectral_features(grad_seq, fs=1.0, hf_ratio=0.25)
        except Exception:
            spectral_layer1 = {}

        if self.verbose:
            print("[DefenseAnalyzer] Computing ETD score...")
        etd_score = self._compute_etd_score(images, labels, eps)

        if self.verbose:
            print("[DefenseAnalyzer] Fitting Volterra kernel...")
        alpha_volterra, volterra_rmse, volterra_rmse_scaled, volterra_fit_info = (
            self._fit_volterra_kernel(gradients)
        )
        n_grad = int(len(gradients))
        short_grad_history = n_grad < int(self.MIN_GRADIENT_SEQUENCE_LEN)
        volterra_opt_success = bool(volterra_fit_info.get("success", False))
        # "OK for use" means: scale isn't degenerate (so rmse_scaled is finite).
        # The optimizer may still report success=False while returning a usable estimate.
        volterra_fit_ok = bool(np.isfinite(volterra_rmse_scaled))
        volterra_high_rmse = (
            bool(volterra_fit_ok)
            and np.isfinite(volterra_rmse_scaled)
            and float(volterra_rmse_scaled) > float(self.VOLTERRA_RMSE_SCALED_MAX)
        )

        if self.verbose:
            print("[DefenseAnalyzer] Analyzing gradient statistics...")
        grad_variance = self._compute_gradient_variance(images, labels)
        stochastic_score = self._compute_stochastic_score(images, labels)
        grad_norm_mean = self._compute_gradient_norm_mean(gradients)
        grad_sample_info = dict(getattr(self, "_last_grad_sample_info", {}) or {})
        true_grad_none_fraction = float(grad_sample_info.get("true_grad_none_fraction", 0.0))
        true_grad_zero_fraction = float(grad_sample_info.get("true_grad_zero_fraction", 0.0))
        jacobian_rank = self._estimate_jacobian_rank(images, labels)
        autocorr, timescale = self._compute_autocorrelation(gradients)
        spectral_signals = self._compute_spectral_signals(gradients)

        if grad_norm_mean < self.GRAD_NORM_SHATTERED_THRESHOLD:
            alpha_volterra = min(alpha_volterra, 0.2)

        if self.verbose:
            print("[DefenseAnalyzer] Classifying obfuscation mechanisms...")
        obfuscation_types = self._classify_obfuscation(
            etd_score,
            alpha_volterra,
            grad_variance,
            grad_norm_mean,
            stochastic_score,
            jacobian_rank,
            timescale,
            spectral_signals,
            volterra_fit_ok,
            true_grad_none_fraction=true_grad_none_fraction,
            true_grad_zero_fraction=true_grad_zero_fraction,
        )

        requires_bpda, requires_eot, requires_mapgd = self._recommend_attacks(
            obfuscation_types, etd_score, alpha_volterra, volterra_fit_ok
        )

        eot_samples = self._recommend_eot_samples(grad_variance, requires_eot)
        memory_length = self._recommend_memory_length(alpha_volterra, requires_mapgd)

        confidence = self._compute_confidence(
            etd_score,
            alpha_volterra,
            volterra_rmse,
            volterra_rmse_scaled,
            grad_variance,
            grad_norm_mean,
            jacobian_rank,
            n_grad,
        )

        # Paper-aligned (Layer 1) scalars for reporting + calibration.
        spectral_entropy = float(spectral_layer1.get("spectral_entropy", 0.0)) if spectral_layer1 else 0.0
        spectral_entropy_norm = (
            float(spectral_layer1.get("spectral_entropy_norm", 0.0)) if spectral_layer1 else 0.0
        )
        high_freq_ratio = float(spectral_layer1.get("high_freq_ratio", 0.0)) if spectral_layer1 else 0.0
        wavelet_energy = spectral_layer1.get("wavelet_energy", {}) if spectral_layer1 else {}
        if not isinstance(wavelet_energy, dict):
            wavelet_energy = {}

        # Paper-style composite verdict (Section 3.5): record as metadata for auditability.
        # Routing/attack recommendations still use `obfuscation_types` (existing logic).
        paper_entropy_flag = bool(spectral_entropy_norm > float(self.SPECTRAL_ENTROPY_OBFUSCATED_THRESHOLD))
        paper_hf_flag = bool(high_freq_ratio > float(self.HIGH_FREQ_RATIO_SHATTERED_THRESHOLD))
        paper_alpha_flag = bool(volterra_fit_ok and float(alpha_volterra) < float(self.ALPHA_RL_THRESHOLD))
        paper_triggers: List[str] = []
        if paper_entropy_flag:
            paper_triggers.append("spectral_entropy_norm")
        if paper_hf_flag:
            paper_triggers.append("high_freq_ratio")
        if paper_alpha_flag:
            paper_triggers.append("alpha_volterra")

        characterization = DefenseCharacterization(
            obfuscation_types=obfuscation_types,
            etd_score=etd_score,
            alpha_volterra=alpha_volterra,
            gradient_variance=grad_variance,
            jacobian_rank=jacobian_rank,
            autocorr_timescale=timescale,
            requires_bpda=requires_bpda,
            requires_eot=requires_eot,
            requires_mapgd=requires_mapgd,
            recommended_eot_samples=eot_samples,
            recommended_memory_length=memory_length,
            confidence=confidence,
            metadata={
                "spectral_entropy": float(spectral_entropy),
                "spectral_entropy_norm": float(spectral_entropy_norm),
                "high_freq_ratio": float(high_freq_ratio),
                # Draft Section 3.1 (Morlet): include the per-scale CWT energy so
                # paper↔artifact comparisons are mechanical.
                "wavelet_energy": {str(k): float(v) for k, v in wavelet_energy.items()},
                "paper_style": {
                    "composite_obfuscated": bool(paper_entropy_flag or paper_hf_flag or paper_alpha_flag),
                    "triggers": list(paper_triggers),
                    "entropy_obfuscated": bool(paper_entropy_flag),
                    "high_freq_obfuscated": bool(paper_hf_flag),
                    "alpha_memory_obfuscated": bool(paper_alpha_flag),
                },
                "volterra_rmse": volterra_rmse,
                "volterra_rmse_scaled": volterra_rmse_scaled,
                "volterra_fit": volterra_fit_info,
                "volterra_optimizer_success": bool(volterra_opt_success),
                "volterra_fit_ok": bool(volterra_fit_ok),
                "volterra_high_rmse": bool(volterra_high_rmse),
                "volterra_rmse_scaled_threshold": float(self.VOLTERRA_RMSE_SCALED_MAX),
                "short_gradient_history": bool(short_grad_history),
                "min_recommended_gradient_samples": int(self.MIN_GRADIENT_SEQUENCE_LEN),
                "thresholds": self._thresholds_dict(),
                "gradient_sampling": grad_sample_info,
                "autocorrelation": autocorr.tolist()
                if isinstance(autocorr, np.ndarray)
                else autocorr,
                "n_samples_analyzed": n_grad,
                "grad_norm_mean": grad_norm_mean,
                "stochastic_score": stochastic_score,
                "krylov_rel_error_mean": spectral_signals.get("rel_error_mean"),
                "krylov_norm_ratio_mean": spectral_signals.get("norm_ratio_mean"),
                "krylov_norm_growth_fraction": spectral_signals.get("norm_growth_fraction"),
                "krylov_dissipation_anomaly_score": spectral_signals.get("dissipation_anomaly_score"),
                "krylov_steps": spectral_signals.get("steps"),
            },
        )

        if self.verbose:
            print("\n[DefenseAnalyzer] Characterization complete:")
            print(characterization)

        return characterization

    def _collect_gradient_samples(
        self,
        data_loader: torch.utils.data.DataLoader,
        eps: float,
    ) -> Tuple[List[np.ndarray], torch.Tensor, torch.Tensor]:
        all_images = []
        all_labels = []

        for x_batch, y_batch in data_loader:
            all_images.append(x_batch)
            all_labels.append(y_batch)
            if sum(img.size(0) for img in all_images) >= self.n_probe_images:
                break

        images = torch.cat(all_images)[: self.n_probe_images].to(self.device)
        labels = torch.cat(all_labels)[: self.n_probe_images].to(self.device)

        x_min = float(images.min().item())
        x_max = float(images.max().item())
        delta = torch.zeros_like(images).uniform_(-eps, eps)
        delta = torch.clamp(images + delta, x_min, x_max) - images
        delta.requires_grad = True

        gradients: List[np.ndarray] = []
        none_steps = 0
        zero_steps = 0
        bpda_fallback_steps = 0
        invalid_steps = 0
        backward_error_steps = 0

        # Cache whether the defended model *appears* to support the BPDA kwarg.
        # (We still guard with try/except in `_forward_logits`.)
        bpda_kwarg_supported = True
        try:
            _ = self.model(images[:1], use_approximation=False)  # type: ignore[misc]
        except (TypeError, RuntimeError):
            bpda_kwarg_supported = False

        for _ in range(self.n_samples):
            # First try the "true" backward pass (no approximation). If the defense
            # breaks autograd (e.g., PIL JPEG), `delta.grad` can become None.
            logits = self._forward_logits(images + delta, use_bpda=False)
            loss = F.cross_entropy(logits, labels)
            self.model.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            true_backward_ok = self._safe_backward(loss)
            if not true_backward_ok:
                backward_error_steps += 1

            grad = delta.grad if true_backward_ok else None
            grad_ok = False
            if grad is None:
                none_steps += 1
            else:
                if not torch.isfinite(grad).all():
                    invalid_steps += 1
                else:
                    # If the graph does not depend on `delta` but `delta.grad` was
                    # previously allocated, it can remain as an all-zero tensor.
                    # Treat exact-all-zero as a broken/shattered gradient signal.
                    if float(grad.detach().abs().sum().item()) == 0.0:
                        zero_steps += 1
                    else:
                        grad_ok = True
            if not grad_ok:
                # Fall back to BPDA-style approximation if the model supports it.
                if bpda_kwarg_supported:
                    self.model.zero_grad()
                    if delta.grad is not None:
                        delta.grad.zero_()
                    logits = self._forward_logits(images + delta, use_bpda=True)
                    loss = F.cross_entropy(logits, labels)
                    bpda_backward_ok = self._safe_backward(loss)
                    if not bpda_backward_ok:
                        backward_error_steps += 1
                    grad = delta.grad if bpda_backward_ok else None
                    if (
                        grad is not None
                        and torch.isfinite(grad).all()
                        and float(grad.detach().abs().sum().item()) > 0.0
                    ):
                        bpda_fallback_steps += 1

            if grad is None or not torch.isfinite(grad).all() or float(grad.detach().abs().sum().item()) == 0.0:
                grad = torch.zeros_like(delta)
            else:
                grad = grad.detach().clone()
            grad_flat = grad.view(grad.size(0), -1).cpu().numpy()
            gradients.append(grad_flat.mean(axis=0))

            with torch.no_grad():
                # Use the best-effort gradient to "walk" delta; otherwise a single
                # None/zero gradient would freeze the entire history at zeros.
                delta.data = delta + (2 / 255) * grad.sign()
                delta.data = torch.clamp(delta, -eps, eps)
                delta.data = torch.clamp(images + delta, x_min, x_max) - images

            delta.requires_grad = True

        # Publish sampling diagnostics for auditability.
        total_steps = int(max(1, self.n_samples))
        self._last_grad_sample_info = {
            "bpda_kwarg_supported": bool(bpda_kwarg_supported),
            "true_grad_none_steps": int(none_steps),
            "true_grad_zero_steps": int(zero_steps),
            "true_grad_invalid_steps": int(invalid_steps),
            "backward_error_steps": int(backward_error_steps),
            "bpda_fallback_steps": int(bpda_fallback_steps),
            "total_steps": int(total_steps),
            "true_grad_none_fraction": float(none_steps) / float(total_steps),
            "true_grad_zero_fraction": float(zero_steps) / float(total_steps),
            "bpda_fallback_fraction": float(bpda_fallback_steps) / float(total_steps),
        }

        return gradients, images, labels

    def collect_gradient_samples(
        self,
        data_loader: torch.utils.data.DataLoader,
        eps: float,
    ) -> Tuple[List[np.ndarray], torch.Tensor, torch.Tensor]:
        return self._collect_gradient_samples(data_loader, eps)

    def _compute_etd_score(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        eps: float,
    ) -> float:
        from ..attacks.pgd import PGD

        was_training = self.model.training
        pgd = PGD(self.model, eps=eps, steps=40, device=self.device)
        try:
            x_adv = pgd(images, labels)
        except RuntimeError as exc:
            if self._is_no_grad_runtime_error(exc):
                warnings.warn(
                    "ETD score fallback: PGD backward unavailable for this defense; "
                    "using etd_score=0.0"
                )
                return 0.0
            raise

        dist_adv = (images - x_adv).pow(2).view(images.size(0), -1).sum(dim=1)
        view_shape = (-1,) + (1,) * (images.ndim - 1)
        directions = (x_adv - images) / (dist_adv.sqrt().view(view_shape) + 1e-8)
        x_min = float(images.min().item())
        x_max = float(images.max().item())

        cf_distances = []
        for i in range(images.size(0)):
            x_i = images[i : i + 1]
            y_i = labels[i : i + 1]
            dir_i = directions[i : i + 1]

            low, high = 0.0, eps
            for _ in range(20):
                mid = (low + high) / 2
                x_test = x_i + mid * dir_i
                x_test = torch.clamp(x_test, x_min, x_max)
                with torch.no_grad():
                    pred = self.model(x_test).argmax(1)
                if pred != y_i:
                    high = mid
                else:
                    low = mid

            cf_distances.append(((high + low) / 2) ** 2)

        cf_distances = torch.tensor(cf_distances, device=self.device)
        etd_ratios = cf_distances / (dist_adv + 1e-8)
        etd_score = etd_ratios.mean().item()
        if was_training:
            self.model.train()
        else:
            self.model.eval()
        return float(np.clip(etd_score, 0.0, 2.0))

    def _fit_volterra_kernel(self, gradients: List[np.ndarray]) -> Tuple[float, float, float, Dict[str, Any]]:
        if not gradients:
            return 0.5, np.nan, np.nan, {"success": False, "message": "empty gradient sequence"}

        grad_array_raw = np.array(gradients, dtype=np.float64)
        grad_array = self._volterra_preprocess_gradients(grad_array_raw)
        info_out: Dict[str, Any] = {
            "gradient_source": str(getattr(self, "volterra_gradient_source", "pre_optimizer")),
            "optimizer": str(getattr(self, "volterra_optimizer", "sgd")),
            "optimizer_beta1": float(getattr(self, "volterra_optimizer_beta1", 0.9)),
            "optimizer_beta2": float(getattr(self, "volterra_optimizer_beta2", 0.999)),
            "optimizer_eps": float(getattr(self, "volterra_optimizer_eps", 1e-8)),
        }
        try:
            kernel, rmse, info = fit_volterra_kernel(
                grad_array,
                kernel_type="power_law",
                method="L-BFGS-B",
                verbose=False,
            )
            info_out.update(dict(info or {}))
            if not info_out.get("success", False):
                warnings.warn("Volterra fitting did not converge; fit may be unreliable.")
            alpha = float(getattr(kernel, "alpha", 0.5))
        except Exception as exc:
            info_out.setdefault("success", False)
            info_out["message"] = str(exc)
            warnings.warn(f"Volterra fitting failed: {exc}, using alpha=0.5")
            return 0.5, np.nan, np.nan, info_out

        rmse = float(rmse)
        # Scale-free fit error: avoids thresholds tied to raw gradient scale.
        scale = float(np.std(grad_array))
        if np.isfinite(scale) and scale > 1e-12 and np.isfinite(rmse):
            rmse_scaled = float(rmse / scale)
        else:
            rmse_scaled = np.nan
        info_out["rmse_scaled"] = float(rmse_scaled) if np.isfinite(rmse_scaled) else np.nan
        return alpha, float(rmse), float(rmse_scaled), info_out

    def _compute_gradient_variance(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        if images.size(0) == 0:
            return 0.0
        delta = torch.zeros_like(images, requires_grad=True)
        logits = self._forward_logits(images + delta, use_bpda=False)
        loss = F.cross_entropy(logits, labels)
        self.model.zero_grad()
        if delta.grad is not None:
            delta.grad.zero_()
        backward_ok = self._safe_backward(loss)

        grad = delta.grad if backward_ok else None
        if grad is None or not torch.isfinite(grad).all() or float(grad.detach().abs().sum().item()) == 0.0:
            # Try BPDA approximation so variance isn't silently zeroed.
            self.model.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            logits = self._forward_logits(images + delta, use_bpda=True)
            loss = F.cross_entropy(logits, labels)
            bpda_ok = self._safe_backward(loss)
            grad = delta.grad if bpda_ok else None

        if grad is None or not torch.isfinite(grad).all() or float(grad.detach().abs().sum().item()) == 0.0:
            grad = torch.zeros_like(delta)
        else:
            grad = grad.detach()
        grad_flat = grad.view(grad.size(0), -1)
        var = grad_flat.var(unbiased=False).item()
        scale = float(grad_flat.shape[1]) * 1_000_000.0
        return float(var * scale)

    def _compute_stochastic_score(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        n_repeats: int = 5,
    ) -> float:
        if images.size(0) == 0:
            return 0.0
        logits_list = []
        for _ in range(int(n_repeats)):
            logits = self.model(images)
            logits_list.append(logits.detach())
        logit_stack = torch.stack(logits_list, dim=0)
        logit_variance = logit_stack.var(dim=0).mean().item()
        return float(logit_variance)

    def _compute_gradient_norm_mean(self, gradients: List[np.ndarray]) -> float:
        if len(gradients) == 0:
            return 0.0
        grad_array = np.array(gradients)
        norms = np.linalg.norm(grad_array, axis=1)
        return float(np.mean(norms))

    def _estimate_jacobian_rank(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        n_samples: int = 10,
    ) -> float:
        n = int(min(n_samples, images.size(0)))
        if n <= 1:
            return 0.0
        indices = torch.randperm(images.size(0))[:n]
        x_sample = images[indices].requires_grad_(True)
        y_sample = labels[indices]

        logits = self._forward_logits(x_sample, use_bpda=False)
        loss = F.cross_entropy(logits, y_sample)
        self.model.zero_grad()
        backward_ok = self._safe_backward(loss)

        grad = x_sample.grad if backward_ok else None
        if grad is None or not torch.isfinite(grad).all() or float(grad.detach().abs().sum().item()) == 0.0:
            # Best-effort BPDA fallback for non-differentiable transforms.
            self.model.zero_grad()
            x_sample.grad = None
            logits = self._forward_logits(x_sample, use_bpda=True)
            loss = F.cross_entropy(logits, y_sample)
            bpda_ok = self._safe_backward(loss)
            grad = x_sample.grad if bpda_ok else None

        if grad is None or not torch.isfinite(grad).all() or float(grad.detach().abs().sum().item()) == 0.0:
            grad = torch.zeros_like(x_sample)
        else:
            grad = grad.detach()
        grad_flat = grad.view(n, -1).cpu().numpy()
        if not np.any(grad_flat):
            return 0.0
        try:
            _, s, _ = np.linalg.svd(grad_flat, full_matrices=False)
            s_norm = s / (s.sum() + 1e-8)
            effective_rank = np.exp(-np.sum(s_norm * np.log(s_norm + 1e-8)))
            effective_rank_normalized = effective_rank / n
        except np.linalg.LinAlgError:
            effective_rank_normalized = 0.1

        return float(effective_rank_normalized)

    def _compute_autocorrelation(
        self,
        gradients: List[np.ndarray],
        max_lag: int = 20,
    ) -> Tuple[np.ndarray, float]:
        grad_array = np.array(gradients)
        autocorr, timescale = compute_volterra_correlation(grad_array, max_lag=max_lag)
        return autocorr, timescale

    def _compute_spectral_signals(self, gradients: List[np.ndarray]) -> Dict[str, float]:
        if len(gradients) < 2:
            return {
                "rel_error_mean": 0.0,
                "norm_ratio_mean": 1.0,
                "norm_growth_fraction": 0.0,
                "dissipation_anomaly_score": 0.0,
                "steps": 0,
            }
        grad_array = np.array(gradients, dtype=np.float64)
        try:
            summary, _steps, _eigvals = analyze_krylov_projection(
                grad_array,
                krylov_dim=self.krylov_dim,
                dt=1.0,
                damping=0.1,
                steps=min(25, max(2, grad_array.shape[0] - 1)),
                stride=1,
            )
        except Exception as exc:
            warnings.warn(f"Krylov analysis failed: {exc}")
            return {
                "rel_error_mean": 0.0,
                "norm_ratio_mean": 1.0,
                "norm_growth_fraction": 0.0,
                "dissipation_anomaly_score": 0.0,
                "steps": 0,
            }
        reconstruction = summary.get("reconstruction_error", {}) or {}
        dissipation = summary.get("dissipation", {}) or {}
        return {
            "rel_error_mean": float(reconstruction.get("mean", 0.0)),
            "norm_ratio_mean": float(dissipation.get("norm_ratio_mean", 1.0)),
            "norm_growth_fraction": float(dissipation.get("norm_growth_fraction", 0.0)),
            "dissipation_anomaly_score": float(dissipation.get("dissipation_anomaly_score", 0.0)),
            "steps": int(summary.get("steps_analyzed", 0)),
        }

    def _classify_obfuscation(
        self,
        etd_score: float,
        alpha_volterra: float,
        grad_variance: float,
        grad_norm_mean: float,
        stochastic_score: float,
        jacobian_rank: float,
        timescale: float,
        spectral_signals: Dict[str, float],
        volterra_fit_ok: bool,
        *,
        true_grad_none_fraction: float = 0.0,
        true_grad_zero_fraction: float = 0.0,
    ) -> List[ObfuscationType]:
        obfuscation_types: List[ObfuscationType] = []

        # If autograd breaks (grad is None) for a non-trivial fraction of probes,
        # treat this as strong evidence of shattered / non-differentiable defenses.
        if float(true_grad_none_fraction) > 0.10 or float(true_grad_zero_fraction) > 0.50:
            obfuscation_types.append(ObfuscationType.SHATTERED)
            if self.verbose:
                print(
                    "  [DETECTED] Shattered gradients "
                    f"(true_none={float(true_grad_none_fraction):.2f}, true_zero={float(true_grad_zero_fraction):.2f})"
                )

        if grad_norm_mean < self.GRAD_NORM_SHATTERED_THRESHOLD:
            obfuscation_types.append(ObfuscationType.SHATTERED)
            if self.verbose:
                print(
                    "  [DETECTED] Shattered gradients "
                    f"(alpha={alpha_volterra:.3f}, grad_norm={grad_norm_mean:.6f})"
                )

        if stochastic_score > self.VARIANCE_STOCHASTIC_THRESHOLD:
            obfuscation_types.append(ObfuscationType.STOCHASTIC)
            if self.verbose:
                print(
                    "  [DETECTED] Stochastic obfuscation "
                    f"(score={stochastic_score:.6f} > {self.VARIANCE_STOCHASTIC_THRESHOLD:.6f})"
                )

        if jacobian_rank < self.RANK_VANISHING_THRESHOLD:
            obfuscation_types.append(ObfuscationType.VANISHING)
            if self.verbose:
                print(
                    f"  [DETECTED] Vanishing gradients (rank={jacobian_rank:.3f} < 0.5)"
                )

        _rel_error_mean = float(spectral_signals.get("rel_error_mean", 0.0))
        norm_ratio_mean = float(spectral_signals.get("norm_ratio_mean", 1.0))
        _norm_growth_fraction = float(spectral_signals.get("norm_growth_fraction", 0.0))

        # Note: we compute Krylov diagnostics for reporting and future analysis,
        # but we do not classify "SHATTERED" purely from reconstruction error /
        # norm growth. In practice those signals are highly sensitive to generic
        # non-stationarity (even for clean models) and need careful calibration
        # on real, task-specific gradients.

        if norm_ratio_mean <= self.KRYLOV_NORM_RATIO_VANISHING:
            if ObfuscationType.VANISHING not in obfuscation_types:
                obfuscation_types.append(ObfuscationType.VANISHING)
                if self.verbose:
                    print(
                        "  [DETECTED] Spectral vanishing (Krylov norm_ratio="
                        f"{norm_ratio_mean:.3f})"
                    )

        if (
            self.ALPHA_SHATTERED_THRESHOLD <= alpha_volterra < self.ALPHA_RL_THRESHOLD
            and timescale > self.AUTOCORR_RL_THRESHOLD
            and bool(volterra_fit_ok)
        ):
            obfuscation_types.append(ObfuscationType.RL_TRAINED)
            if self.verbose:
                print(
                    f"  [DETECTED] RL-trained obfuscation (alpha={alpha_volterra:.3f}, "
                    f"timescale={timescale:.2f})"
                )

        if len(obfuscation_types) == 0:
            obfuscation_types.append(ObfuscationType.NONE)
            if self.verbose:
                print("  [NO OBFUSCATION DETECTED]")

        return obfuscation_types

    def _recommend_attacks(
        self,
        obfuscation_types: List[ObfuscationType],
        etd_score: float,
        alpha_volterra: float,
        volterra_fit_ok: bool,
    ) -> Tuple[bool, bool, bool]:
        if len(obfuscation_types) == 1 and obfuscation_types[0] == ObfuscationType.NONE:
            return False, False, False
        requires_bpda = (
            ObfuscationType.SHATTERED in obfuscation_types
            or etd_score >= self.ETD_THRESHOLD_SEVERE
        )
        requires_eot = ObfuscationType.STOCHASTIC in obfuscation_types
        requires_mapgd = (
            ObfuscationType.RL_TRAINED in obfuscation_types
            or (bool(volterra_fit_ok) and alpha_volterra < self.ALPHA_RL_THRESHOLD)
        )
        return requires_bpda, requires_eot, requires_mapgd

    def _recommend_eot_samples(self, grad_variance: float, requires_eot: bool) -> int:
        if not requires_eot:
            return 1
        n_samples = int(20 * grad_variance / self.VARIANCE_STOCHASTIC_THRESHOLD)
        return int(np.clip(n_samples, 10, 50))

    def _recommend_memory_length(self, alpha_volterra: float, requires_mapgd: bool) -> int:
        if not requires_mapgd:
            return 1
        from ..attacks.memory_gradient import memory_length_schedule

        return memory_length_schedule(alpha_volterra, max_length=50)

    def _compute_confidence(
        self,
        etd_score: float,
        alpha_volterra: float,
        volterra_rmse: float,
        volterra_rmse_scaled: float,
        grad_variance: float,
        grad_norm_mean: float,
        jacobian_rank: float,
        n_grad: int,
    ) -> float:
        # Prefer the scale-free RMSE when available.
        if np.isfinite(volterra_rmse_scaled):
            volterra_conf = float(np.exp(-float(volterra_rmse_scaled)))
        elif np.isnan(volterra_rmse):
            volterra_conf = 0.5  # unknown fit quality
        else:
            volterra_conf = np.exp(-volterra_rmse * 10)
        etd_conf = 1.0 - np.exp(-abs(etd_score - 0.5) * 3)
        grad_scale = max(self.GRAD_NORM_SHATTERED_THRESHOLD * 5.0, 1e-12)
        grad_conf = 1.0 - np.exp(-float(grad_norm_mean) / grad_scale)
        rank_scale = max(self.RANK_VANISHING_THRESHOLD, 1e-6)
        rank_conf = float(np.clip(jacobian_rank / rank_scale, 0.0, 1.0))
        n_factor = float(np.clip(float(n_grad) / float(self.MIN_GRADIENT_SEQUENCE_LEN), 0.0, 1.0))
        confidence = (
            np.sqrt(volterra_conf * etd_conf)
            * max(grad_conf, 1e-3)
            * max(rank_conf, 1e-3)
            * max(n_factor, 1e-3)
        )
        return float(np.clip(confidence, 0.0, 1.0))


def quick_characterize(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> DefenseCharacterization:
    """Quick defense characterization with default settings."""
    analyzer = DefenseAnalyzer(model, device=device, verbose=True)
    return analyzer.characterize(data_loader)
