"""
Unified base attack interface for NeurInSpectre.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AttackCapability(Enum):
    """Enumeration of attack capabilities."""

    UNTARGETED = auto()
    TARGETED = auto()
    BPDA = auto()
    EOT = auto()
    ADAPTIVE = auto()
    QUERY_EFFICIENT = auto()
    CERTIFIED = auto()


@dataclass
class AttackConfig:
    """
    Universal attack configuration (superset of attack settings).
    """

    norm: str = "linf"
    epsilon: float = 8 / 255
    n_iterations: int = 100
    n_restarts: int = 1
    step_size: Optional[float] = None
    random_init: bool = True
    loss: Any = "dlr"
    loss_temperature: float = 1.0
    loss_softmax_weighting: bool = False
    kappa: float = 0.0
    use_tg: bool = False
    rho: float = 0.75
    targeted: bool = False
    target_class: Optional[int] = None
    seed: int = 42
    batch_size: int = 128
    early_stop: bool = True
    use_bpda: bool = False
    bpda_approximation: str = "defense"
    use_eot: bool = False
    eot_samples: int = 20
    eot_importance_weighted: bool = True
    auto_step_size: bool = False
    input_range: Tuple[float, float] = (0.0, 1.0)
    auto_detect_range: bool = True

    def __post_init__(self) -> None:
        if self.use_tg:
            logger.warning(
                "Transformed Gradients (TG) enabled. "
                "This deviates from standard AutoAttack and may affect comparisons. "
                "Set use_tg=False for strict AutoAttack reproduction."
            )

    def ensure_step_size(self) -> float:
        """Compute step size if auto_step_size is enabled."""
        if self.step_size is not None:
            return float(self.step_size)
        if not self.auto_step_size:
            return 0.0
        if str(self.norm).lower() in {"linf", "l_inf", "inf"}:
            self.step_size = 2.5 * self.epsilon / max(self.n_iterations, 1)
        elif str(self.norm).lower() in {"l2", "l_2"}:
            self.step_size = 2.0 * self.epsilon / max(self.n_iterations, 1) ** 0.5
        else:
            self.step_size = self.epsilon / 10.0
        return float(self.step_size)

    def detect_input_range(self, x: torch.Tensor) -> Tuple[float, float]:
        """
        Auto-detect input range from sample batch.

        Heuristics:
        - If max ≤ 1.1 and min ≥ -0.1 → [0, 1] (standard images)
        - If max ≤ 3.0 and min ≥ -3.0 → Normalized (CIFAR/ImageNet style)
        - Otherwise → [min, max] with 10% margin
        """
        x_min = float(x.min().item())
        x_max = float(x.max().item())

        if x_max <= 1.1 and x_min >= -0.1:
            return (0.0, 1.0)

        span = max(x_max - x_min, 1e-12)
        margin = 0.1 * span
        if x_max <= 3.0 and x_min >= -3.0:
            return (x_min - margin, x_max + margin)

        return (x_min - margin, x_max + margin)


@dataclass
class AttackResult:
    """
    Standardized attack result container.
    """

    x_adv: torch.Tensor
    predictions: Optional[torch.Tensor] = None
    success_mask: Optional[torch.Tensor] = None
    perturbation: Optional[torch.Tensor] = None
    queries: Optional[int] = None
    iterations: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    num_successful: int = 0

    def __post_init__(self) -> None:
        if self.success_mask is not None:
            self.success_rate = float(self.success_mask.float().mean().item())
            self.num_successful = int(self.success_mask.sum().item())

    @property
    def success(self) -> Optional[torch.Tensor]:
        return self.success_mask


class BaseAdversarialAttack(ABC):
    """
    Abstract base class for all adversarial attacks.
    """

    def __init__(self, config: AttackConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self._name = self.__class__.__name__
        self._capabilities = self._get_capabilities()
        self._total_queries = 0
        self._total_iterations = 0
        logger.debug("Initialized %s on %s", self._name, device)

    @abstractmethod
    def run(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> AttackResult:
        raise NotImplementedError

    def perturb(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        result = self.run(model, x, y, targeted, target_labels, **kwargs)
        info = {
            "success": result.success_mask,
            "success_rate": result.success_rate,
            "queries": result.queries,
            "iterations": result.iterations,
            **result.metadata,
        }
        return result.x_adv, info

    @abstractmethod
    def _get_capabilities(self) -> List[AttackCapability]:
        raise NotImplementedError

    def supports_targeted(self) -> bool:
        return AttackCapability.TARGETED in self._capabilities

    def supports_bpda(self) -> bool:
        return AttackCapability.BPDA in self._capabilities

    def supports_eot(self) -> bool:
        return AttackCapability.EOT in self._capabilities

    def is_adaptive(self) -> bool:
        return AttackCapability.ADAPTIVE in self._capabilities

    def has_capability(self, capability: AttackCapability) -> bool:
        return capability in self._capabilities

    def _check_success(
        self,
        model: nn.Module,
        x_adv: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            logits = model(x_adv)
            preds = logits.argmax(dim=1)
            if targeted:
                return preds == target_labels
            return preds != y

    def _validate_inputs(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool,
        target_labels: Optional[torch.Tensor],
    ) -> None:
        if targeted and not self.supports_targeted():
            raise NotImplementedError(
                f"{self._name} does not support targeted attacks."
            )
        if targeted and target_labels is None:
            raise ValueError("target_labels must be provided when targeted=True")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x has {x.shape[0]} samples, y has {y.shape[0]} labels"
            )
        if targeted and target_labels.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x has {x.shape[0]} samples, target_labels has {target_labels.shape[0]} labels"
            )
        expected = torch.device(self.device)
        # On some backends (notably MPS) tensors may report as "mps:0" while
        # `torch.device("mps")` stringifies as "mps". Compare by device type first.
        mismatched = x.device.type != expected.type
        # Preserve index mismatch warnings for CUDA where it matters.
        if not mismatched and expected.type == "cuda":
            if expected.index is not None and x.device.index is not None and expected.index != x.device.index:
                mismatched = True
        if mismatched:
            logger.warning(
                "Input device %s doesn't match attack device %s. Moving to %s.",
                x.device,
                self.device,
                self.device,
            )

    def __repr__(self) -> str:
        caps = ", ".join(c.name for c in self._capabilities)
        return (
            f"{self._name}(norm={self.config.norm}, "
            f"epsilon={self.config.epsilon}, iterations={self.config.n_iterations}, "
            f"capabilities=[{caps}])"
        )


class GradientBasedAttack(BaseAdversarialAttack):
    """
    Base class for gradient-based attacks (PGD/APGD).
    """

    def __init__(self, config: AttackConfig, device: str = "cuda"):
        super().__init__(config, device)
        self._input_min: Optional[float] = None
        self._input_max: Optional[float] = None
        self._range_detected = False

    def _detect_and_set_range(self, x: torch.Tensor) -> None:
        """
        Detect (or update) the clamp range used during projection.

        Important: this method must be safe across *multiple batches*.
        Some datasets (notably EMBER) contain rare but very large feature values.
        If we only infer the range once (from the first batch), later batches can
        contain values outside that range and the projection step will silently
        clamp `x_adv`, producing enormous (and budget-violating) perturbations.
        """
        if not bool(self.config.auto_detect_range):
            if not self._range_detected:
                self._input_min, self._input_max = self.config.input_range
                logger.info(
                    "Using configured input range: [%.3f, %.3f]", self._input_min, self._input_max
                )
                self._range_detected = True
            return

        detected_min, detected_max = self.config.detect_input_range(x)
        if (not self._range_detected) or (self._input_min is None) or (self._input_max is None):
            self._input_min, self._input_max = float(detected_min), float(detected_max)
            logger.info(
                "Auto-detected input range: [%.3f, %.3f]", self._input_min, self._input_max
            )
            self._range_detected = True
            return

        # Never shrink the range; only expand to include new extrema.
        new_min = min(float(self._input_min), float(detected_min))
        new_max = max(float(self._input_max), float(detected_max))
        if new_min != float(self._input_min) or new_max != float(self._input_max):
            self._input_min, self._input_max = new_min, new_max
            logger.debug(
                "Expanded input range to: [%.3f, %.3f]", self._input_min, self._input_max
            )

    def _get_loss_type(self) -> str:
        loss = getattr(self.config, "loss", "dlr")
        if isinstance(loss, Enum):
            return loss.value
        return str(loss)

    def _compute_gradient(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x.requires_grad_(True)
        logits = model(x)
        from ..losses.cw_loss import cw_loss
        from ..losses.dlr_loss import dlr_loss, ce_loss
        from ..losses.logit_margin_loss import (
            enhanced_margin_loss,
            enhanced_margin_targeted_loss,
            logit_margin_loss,
        )
        from ..losses.md_loss import md_loss
        from ..losses.mm_loss import minimum_margin_loss, minimum_margin_targeted_loss

        loss_type = self._get_loss_type().lower()
        if loss_type in {"dlr"}:
            loss = dlr_loss(logits, y, targeted=targeted, target=target_labels, reduction="sum")
        elif loss_type in {"ce"}:
            labels = target_labels if targeted else y
            loss = ce_loss(logits, labels, targeted=targeted, reduction="sum")
        elif loss_type in {"cw"}:
            loss = cw_loss(
                logits,
                y,
                targeted=targeted,
                target=target_labels,
                kappa=float(getattr(self.config, "kappa", 0.0)),
                reduction="sum",
            )
        elif loss_type in {"logit", "logit_margin"}:
            loss = logit_margin_loss(
                logits,
                y,
                targeted=targeted,
                target=target_labels,
                reduction="sum",
            )
        elif loss_type in {"logit_enhanced", "enhanced_margin"}:
            temperature = float(getattr(self.config, "loss_temperature", 1.0))
            use_softmax_weighting = bool(getattr(self.config, "loss_softmax_weighting", False))
            if targeted:
                loss = enhanced_margin_targeted_loss(
                    logits,
                    y,
                    target_labels,
                    temperature=temperature,
                    use_softmax_weighting=use_softmax_weighting,
                    reduction="sum",
                )
            else:
                loss = enhanced_margin_loss(
                    logits,
                    y,
                    temperature=temperature,
                    use_softmax_weighting=use_softmax_weighting,
                    reduction="sum",
                )
        elif loss_type in {"mm", "minimum_margin", "minimum-margin"}:
            if targeted:
                loss = minimum_margin_targeted_loss(
                    logits,
                    y,
                    target_labels,
                    use_rescaling=True,
                    reduction="sum",
                )
            else:
                loss = minimum_margin_loss(
                    logits,
                    y,
                    use_rescaling=True,
                    reduction="sum",
                )
        elif loss_type in {"md", "minimal_difference", "minimal-difference"}:
            labels = target_labels if targeted else y
            loss = md_loss(logits, labels, reduction="sum")
        else:
            loss = dlr_loss(logits, y, targeted=targeted, target=target_labels, reduction="sum")
        if not loss.requires_grad:
            x.requires_grad_(False)
            return torch.zeros_like(x)
        grad = torch.autograd.grad(loss, x, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(x)
        x.requires_grad_(False)
        return grad

    def _project(self, x: torch.Tensor, x_adv: torch.Tensor, epsilon: Optional[float] = None) -> torch.Tensor:
        # Ensure the clamp range includes this batch's extrema (see docstring).
        self._detect_and_set_range(x)
        eps = float(epsilon if epsilon is not None else self.config.epsilon)
        norm = str(self.config.norm).lower()
        if norm in {"linf", "l_inf", "inf"}:
            delta = torch.clamp(x_adv - x, -eps, eps)
            x_projected = x + delta
            return torch.clamp(x_projected, min=self._input_min, max=self._input_max)
        if norm in {"l2", "l_2"}:
            delta = x_adv - x
            delta_flat = delta.view(x.size(0), -1)
            norms = delta_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            factors = torch.min(eps / norms, torch.ones_like(norms))
            delta = (delta_flat * factors).view_as(delta)
            x_projected = x + delta
            return torch.clamp(x_projected, min=self._input_min, max=self._input_max)
        if norm in {"l1", "l_1"}:
            delta = x_adv - x
            delta_flat = delta.view(x.size(0), -1)
            norms = delta_flat.norm(p=1, dim=1, keepdim=True).clamp_min(1e-12)
            factors = torch.min(eps / norms, torch.ones_like(norms))
            delta = (delta_flat * factors).view_as(delta)
            x_projected = x + delta
            return torch.clamp(x_projected, min=self._input_min, max=self._input_max)
        raise ValueError(f"Unsupported norm: {self.config.norm}")

    def _normalize_gradient(self, grad: torch.Tensor, norm_type: Optional[str] = None) -> torch.Tensor:
        norm = str(norm_type or self.config.norm).lower()
        if norm in {"linf", "l_inf", "inf", "sign"}:
            return grad.sign()
        if norm in {"l2", "l_2"}:
            grad_flat = grad.view(grad.size(0), -1)
            grad_norm = grad_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            return (grad_flat / grad_norm).view_as(grad)
        if norm in {"l1", "l_1"}:
            grad_flat = grad.view(grad.size(0), -1)
            grad_norm = grad_flat.norm(p=1, dim=1, keepdim=True).clamp_min(1e-12)
            return (grad_flat / grad_norm).view_as(grad)
        return grad

    def _l2_ball_random_init(self, x: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Uniform random initialization in L2 ball.
        """
        self._detect_and_set_range(x)
        eps = float(epsilon)
        delta = torch.randn_like(x)
        delta_flat = delta.view(x.size(0), -1)
        norms = delta_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        delta_flat = delta_flat / norms
        delta_dim = delta_flat.shape[1]
        r = torch.rand(x.size(0), 1, device=x.device) ** (1.0 / float(delta_dim))
        delta_flat = delta_flat * r * eps
        delta = delta_flat.view_as(x)
        x_projected = x + delta
        return torch.clamp(x_projected, min=self._input_min, max=self._input_max)


class PGDAttack(GradientBasedAttack):
    """
    Projected Gradient Descent (PGD) attack.
    """

    def _get_capabilities(self) -> List[AttackCapability]:
        return [
            AttackCapability.UNTARGETED,
            AttackCapability.TARGETED,
        ]

    def run(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> AttackResult:
        self._validate_inputs(model, x, y, targeted, target_labels)
        x = x.to(self.device)
        y = y.to(self.device)
        if targeted:
            target_labels = target_labels.to(self.device)

        step_size = self.config.step_size
        if step_size is None:
            step_size = self.config.ensure_step_size()
            if step_size == 0.0:
                norm = str(self.config.norm).lower()
                if norm in {"linf", "l_inf", "inf"}:
                    step_size = 2.5 * self.config.epsilon / max(self.config.n_iterations, 1)
                elif norm in {"l2", "l_2"}:
                    step_size = 2.0 * self.config.epsilon / max(self.config.n_iterations, 1) ** 0.5
                else:
                    step_size = self.config.epsilon / 10.0
        self.config.step_size = float(step_size)

        if self.config.random_init:
            norm = str(self.config.norm).lower()
            if norm in {"l2", "l_2"}:
                x_adv = self._l2_ball_random_init(x, self.config.epsilon)
            else:
                delta = torch.empty_like(x).uniform_(
                    -self.config.epsilon, self.config.epsilon
                )
                x_adv = self._project(x, x + delta)
        else:
            x_adv = x.clone()

        iterations_run = 0
        for iteration in range(self.config.n_iterations):
            grad = self._compute_gradient(
                model, x_adv, y, targeted, target_labels
            )
            grad_normalized = self._normalize_gradient(grad)
            x_adv = x_adv + self.config.step_size * grad_normalized
            x_adv = self._project(x, x_adv)
            iterations_run = iteration + 1
            if self.config.early_stop and iteration % 10 == 0:
                success = self._check_success(
                    model, x_adv, y, targeted, target_labels
                )
                if success.all():
                    logger.debug("PGD converged at iteration %s", iteration)
                    break

        success = self._check_success(model, x_adv, y, targeted, target_labels)
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)

        return AttackResult(
            x_adv=x_adv,
            predictions=preds,
            success_mask=success,
            perturbation=x_adv - x,
            queries=iterations_run,
            iterations=iterations_run,
            metadata={
                "attack_type": "PGD",
                "targeted": targeted,
                "step_size": self.config.step_size,
            },
        )


class PGDWithRestarts(BaseAdversarialAttack):
    """
    PGD attack with random restarts for improved success rate.
    """

    def __init__(
        self,
        config: AttackConfig,
        n_restarts: Optional[int] = None,
        device: str = "cuda",
    ):
        super().__init__(config, device)
        self.n_restarts = int(n_restarts if n_restarts is not None else config.n_restarts)
        self.pgd = PGDAttack(config, device)

    def _get_capabilities(self) -> List[AttackCapability]:
        return [
            AttackCapability.UNTARGETED,
            AttackCapability.TARGETED,
        ]

    def run(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> AttackResult:
        self._validate_inputs(model, x, y, targeted, target_labels)
        x = x.to(self.device)
        y = y.to(self.device)
        if targeted:
            target_labels = target_labels.to(self.device)

        best_x_adv = x.clone()
        best_loss = torch.full((len(x),), -float("inf"), device=self.device)
        best_success = torch.zeros(len(x), dtype=torch.bool, device=self.device)
        total_iterations = 0
        base_random_init = bool(self.config.random_init)
        loss_type = self.pgd._get_loss_type()
        restarts_run = 0

        for restart in range(self.n_restarts):
            logger.debug("PGD restart %s/%s", restart + 1, self.n_restarts)
            restarts_run = restart + 1
            if restart > 0:
                self.pgd.config.random_init = True
            else:
                self.pgd.config.random_init = base_random_init

            result = self.pgd.run(model, x, y, targeted, target_labels)

            total_iterations += int(result.iterations or 0)
            with torch.no_grad():
                logits = model(result.x_adv)
                from ..losses.cw_loss import cw_loss
                from ..losses.dlr_loss import dlr_loss, ce_loss
                from ..losses.logit_margin_loss import (
                    enhanced_margin_loss,
                    enhanced_margin_targeted_loss,
                    logit_margin_loss,
                )
                from ..losses.md_loss import md_loss
                from ..losses.mm_loss import minimum_margin_loss, minimum_margin_targeted_loss

                loss_type = str(loss_type).lower()
                if loss_type in {"ce"}:
                    labels = target_labels if targeted else y
                    loss = ce_loss(logits, labels, targeted=targeted, reduction="none")
                elif loss_type in {"cw"}:
                    loss = cw_loss(
                        logits,
                        y,
                        targeted=targeted,
                        target=target_labels,
                        kappa=float(getattr(self.config, "kappa", 0.0)),
                        reduction="none",
                    )
                elif loss_type in {"logit", "logit_margin"}:
                    loss = logit_margin_loss(
                        logits,
                        y,
                        targeted=targeted,
                        target=target_labels,
                        reduction="none",
                    )
                elif loss_type in {"logit_enhanced", "enhanced_margin"}:
                    temperature = float(getattr(self.config, "loss_temperature", 1.0))
                    use_softmax_weighting = bool(getattr(self.config, "loss_softmax_weighting", False))
                    if targeted:
                        loss = enhanced_margin_targeted_loss(
                            logits,
                            y,
                            target_labels,
                            temperature=temperature,
                            use_softmax_weighting=use_softmax_weighting,
                            reduction="none",
                        )
                    else:
                        loss = enhanced_margin_loss(
                            logits,
                            y,
                            temperature=temperature,
                            use_softmax_weighting=use_softmax_weighting,
                            reduction="none",
                        )
                elif loss_type in {"mm", "minimum_margin", "minimum-margin"}:
                    if targeted:
                        loss = minimum_margin_targeted_loss(
                            logits,
                            y,
                            target_labels,
                            use_rescaling=True,
                            reduction="none",
                        )
                    else:
                        loss = minimum_margin_loss(
                            logits,
                            y,
                            use_rescaling=True,
                            reduction="none",
                        )
                elif loss_type in {"md", "minimal_difference", "minimal-difference"}:
                    labels = target_labels if targeted else y
                    loss = md_loss(logits, labels, reduction="none")
                else:
                    loss = dlr_loss(
                        logits,
                        y,
                        targeted=targeted,
                        target=target_labels,
                        reduction="none",
                    )

            improved = loss > best_loss
            best_x_adv[improved] = result.x_adv[improved]
            best_loss[improved] = loss[improved]
            best_success[improved] = result.success_mask[improved]

            if best_success.all():
                logger.debug("All samples successful at restart %s", restart + 1)
                break

        with torch.no_grad():
            preds = model(best_x_adv).argmax(dim=1)

        return AttackResult(
            x_adv=best_x_adv,
            predictions=preds,
            success_mask=best_success,
            perturbation=best_x_adv - x,
            queries=total_iterations,
            iterations=total_iterations,
            metadata={
                "attack_type": "PGD_with_restarts",
                "n_restarts": restarts_run,
                "targeted": targeted,
            },
        )


class APGDAttack(GradientBasedAttack):
    """
    Auto-PGD attack with adaptive step size and momentum.
    """

    def __init__(
        self,
        config: AttackConfig,
        device: str = "cuda",
        n_restarts: Optional[int] = None,
        loss_type: Any = None,
        eot_iter: Optional[int] = None,
        rho: Optional[float] = None,
        verbose: bool = False,
        use_tg: Optional[bool] = None,
        momentum_decay: float = 0.75,
    ):
        resolved_eot = int(
            eot_iter if eot_iter is not None else (config.eot_samples if config.use_eot else 1)
        )
        self.eot_iter = max(1, resolved_eot)
        super().__init__(config, device)
        self.n_restarts = int(n_restarts if n_restarts is not None else config.n_restarts)
        resolved_loss = loss_type
        if resolved_loss is None:
            resolved_loss = super()._get_loss_type()
        if isinstance(resolved_loss, Enum):
            resolved_loss = resolved_loss.value
        self.loss_type = str(resolved_loss).lower()
        self.rho = float(rho if rho is not None else config.rho)
        self.verbose = bool(verbose)
        self.use_tg = bool(use_tg if use_tg is not None else getattr(config, "use_tg", False))
        self.momentum_decay = float(momentum_decay)
        self.n_iter = int(config.n_iterations)
        self._rollback_count = 0
        self._oscillation_count = 0
        checkpoints = [
            int(0.22 * self.n_iter),
            int(0.5 * self.n_iter),
            int(0.75 * self.n_iter),
        ]
        self.checkpoints = sorted({c for c in checkpoints if 0 < c < self.n_iter})

    def _get_loss_type(self) -> str:
        return self.loss_type

    def _get_capabilities(self) -> List[AttackCapability]:
        caps = [
            AttackCapability.UNTARGETED,
            AttackCapability.TARGETED,
        ]
        if int(getattr(self, "eot_iter", 1)) > 1:
            caps.append(AttackCapability.EOT)
        return caps

    def _resolve_step_size(self) -> float:
        step_size = self.config.step_size
        if step_size is None or float(step_size) == 0.0:
            step_size = self.config.ensure_step_size()
            if step_size == 0.0:
                step_size = 2.0 * float(self.config.epsilon)
            self.config.step_size = float(step_size)
        return float(step_size)

    def _loss_from_logits(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        targeted: bool,
        target_labels: Optional[torch.Tensor],
        reduction: str,
    ) -> torch.Tensor:
        from ..losses.cw_loss import cw_loss
        from ..losses.dlr_loss import dlr_loss, ce_loss
        from ..losses.logit_margin_loss import (
            enhanced_margin_loss,
            enhanced_margin_targeted_loss,
            logit_margin_loss,
        )
        from ..losses.md_loss import md_loss
        from ..losses.mm_loss import minimum_margin_loss, minimum_margin_targeted_loss

        loss_type = self.loss_type
        if loss_type in {"dlr"}:
            return dlr_loss(logits, y, targeted=targeted, target=target_labels, reduction=reduction)
        if loss_type in {"ce"}:
            labels = target_labels if targeted else y
            return ce_loss(logits, labels, targeted=targeted, reduction=reduction)
        if loss_type in {"md", "minimal_difference", "minimal-difference"}:
            labels = target_labels if targeted else y
            return md_loss(logits, labels, reduction=reduction)
        if loss_type in {"cw"}:
            return cw_loss(
                logits,
                y,
                targeted=targeted,
                target=target_labels,
                kappa=float(getattr(self.config, "kappa", 0.0)),
                reduction=reduction,
            )
        if loss_type in {"logit", "logit_margin"}:
            return logit_margin_loss(
                logits,
                y,
                targeted=targeted,
                target=target_labels,
                reduction=reduction,
            )
        if loss_type in {"logit_enhanced", "enhanced_margin"}:
            temperature = float(getattr(self.config, "loss_temperature", 1.0))
            use_softmax_weighting = bool(getattr(self.config, "loss_softmax_weighting", False))
            if targeted:
                if target_labels is None:
                    raise ValueError("Targeted enhanced margin loss requires target labels")
                return enhanced_margin_targeted_loss(
                    logits,
                    y,
                    target_labels,
                    temperature=temperature,
                    use_softmax_weighting=use_softmax_weighting,
                    reduction=reduction,
                )
            return enhanced_margin_loss(
                logits,
                y,
                temperature=temperature,
                use_softmax_weighting=use_softmax_weighting,
                reduction=reduction,
            )
        if loss_type in {"mm", "minimum_margin", "minimum-margin"}:
            if targeted:
                if target_labels is None:
                    raise ValueError("Targeted minimum-margin loss requires target labels")
                return minimum_margin_targeted_loss(
                    logits,
                    y,
                    target_labels,
                    use_rescaling=True,
                    reduction=reduction,
                )
            return minimum_margin_loss(logits, y, use_rescaling=True, reduction=reduction)
        return dlr_loss(logits, y, targeted=targeted, target=target_labels, reduction=reduction)

    def _compute_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool,
        target_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.eot_iter > 1:
            return self._compute_loss_eot(model, x, y, targeted, target_labels, n_samples=self.eot_iter)
        with torch.no_grad():
            logits = model(x)
        return self._loss_from_logits(logits, y, targeted, target_labels, reduction="none")

    def _compute_loss_eot(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool,
        target_labels: Optional[torch.Tensor],
        n_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute loss with EOT averaging (for stochastic defenses).

        Loss tracking uses the same EOT sampling as gradient computation.
        """
        samples = int(n_samples if n_samples is not None else self.eot_iter)
        samples = max(1, samples)
        loss_sum = torch.zeros(x.shape[0], device=self.device)
        with torch.no_grad():
            for _ in range(samples):
                logits = model(x)
                loss = self._loss_from_logits(
                    logits, y, targeted, target_labels, reduction="none"
                )
                loss_sum += loss
        return loss_sum / float(samples)

    def run(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> AttackResult:
        self._validate_inputs(model, x, y, targeted, target_labels)

        x = x.to(self.device)
        y = y.to(self.device)
        if targeted and target_labels is not None:
            target_labels = target_labels.to(self.device)

        self._resolve_step_size()
        batch_size = x.size(0)

        best_loss = torch.full((batch_size,), -float("inf"), device=self.device)
        best_x_adv = x.clone()
        total_iterations = 0

        for restart_idx in range(self.n_restarts):
            if self.verbose:
                logger.info("APGD restart %s/%s", restart_idx + 1, self.n_restarts)

            x_adv, loss, iterations_run = self._apgd_iteration(
                model, x, y, targeted, target_labels
            )
            total_iterations += iterations_run

            improved = loss > best_loss
            best_x_adv[improved] = x_adv[improved]
            best_loss[improved] = loss[improved]

        success = self._check_success(model, best_x_adv, y, targeted, target_labels)
        with torch.no_grad():
            preds = model(best_x_adv).argmax(dim=1)

        return AttackResult(
            x_adv=best_x_adv,
            predictions=preds,
            success_mask=success,
            perturbation=best_x_adv - x,
            queries=total_iterations * self.eot_iter,
            iterations=total_iterations,
            metadata={
                "attack_type": f"APGD-{self.loss_type.upper()}",
                "n_restarts": self.n_restarts,
                "targeted": targeted,
                "adaptive_step_size": True,
                "eot_iter": self.eot_iter,
            },
        )

    def _apgd_iteration(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool,
        target_labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        from .numerics import check_grad_sanity, transformed_gradient

        batch_size = x.size(0)
        eps = float(self.config.epsilon)

        if self.config.random_init:
            norm = str(self.config.norm).lower()
            if norm in {"l2", "l_2"}:
                x_adv = self._l2_ball_random_init(x, eps)
            else:
                delta = torch.zeros_like(x).uniform_(-eps, eps)
                x_adv = self._project(x, x + delta)
        else:
            x_adv = x.clone()

        x_best_adv = x_adv.clone()
        step_size = torch.full((batch_size,), float(self.config.step_size), device=self.device)
        momentum = torch.zeros_like(x_adv)
        loss_best = self._compute_loss(model, x_best_adv, y, targeted, target_labels)
        x_checkpoint = x_best_adv.clone()
        loss_checkpoint = loss_best.clone()
        loss_history: list[torch.Tensor] = []
        n_reduced = torch.zeros(batch_size, device=self.device)
        iterations_run = 0
        self._rollback_count = 0
        self._oscillation_count = 0

        for iteration in range(self.n_iter):
            grad = torch.zeros_like(x_adv)
            for _ in range(self.eot_iter):
                x_adv = x_adv.detach()
                x_adv.requires_grad = True
                with torch.enable_grad():
                    logits = model(x_adv)
                    loss = self._loss_from_logits(
                        logits, y, targeted, target_labels, reduction="sum"
                    )
                    if not loss.requires_grad:
                        grad_iter = torch.zeros_like(x_adv)
                    else:
                        try:
                            grad_iter = torch.autograd.grad(
                                loss, x_adv, allow_unused=True, retain_graph=False
                            )[0]
                        except RuntimeError:
                            grad_iter = torch.zeros_like(x_adv)
                x_adv.requires_grad = False
                if grad_iter is None:
                    grad_iter = torch.zeros_like(x_adv)
                grad += grad_iter

            grad = grad / float(self.eot_iter)
            if not check_grad_sanity(grad, f"APGD-{self.loss_type}-step{iteration}"):
                grad = torch.sign(torch.randn_like(grad)) * 1e-3

            grad_norm = grad.abs().sum(dim=list(range(1, grad.ndim)), keepdim=True).clamp_min(1e-12)
            grad_normalized = grad / grad_norm
            if self.use_tg:
                grad_normalized = transformed_gradient(grad_normalized)

            momentum = self.momentum_decay * momentum + grad_normalized

            step_size_expanded = step_size.view(-1, *([1] * (x_adv.ndim - 1)))
            x_adv = x_adv + step_size_expanded * momentum
            x_adv = self._project(x, x_adv)
            iterations_run = iteration + 1

            loss_curr = self._compute_loss(model, x_adv, y, targeted, target_labels)
            loss_history.append(loss_curr.detach())
            improved = loss_curr > loss_best
            x_best_adv[improved] = x_adv[improved]
            loss_best[improved] = loss_curr[improved]

            if (iteration + 1) in self.checkpoints:
                improved_since_check = loss_best > loss_checkpoint + 1e-4

                oscillating = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                if len(loss_history) >= 10:
                    recent = torch.stack(loss_history[-10:], dim=0)
                    max_recent = recent.max(dim=0)[0]
                    min_recent = recent.min(dim=0)[0]
                    final_recent = recent[-1]
                    oscillating = (max_recent > final_recent) & (final_recent > min_recent)

                reduce_step = (~improved_since_check) | oscillating
                if reduce_step.any():
                    step_size[reduce_step] *= self.rho
                    n_reduced[reduce_step] += 1
                    x_adv[reduce_step] = x_checkpoint[reduce_step]
                    momentum[reduce_step] = 0.0
                    self._rollback_count += int(reduce_step.sum().item())
                    if self.verbose:
                        logger.debug(
                            "  Iter %s: Reduced step size for %s/%s samples (%s oscillating)",
                            iteration + 1,
                            int(reduce_step.sum().item()),
                            batch_size,
                            int(oscillating.sum().item()),
                        )
                if oscillating.any():
                    self._oscillation_count += int(oscillating.sum().item())
                x_checkpoint = x_best_adv.clone()
                loss_checkpoint = loss_best.clone()

            if (step_size < 1e-6).all():
                if self.verbose:
                    logger.debug("  Early stop: step size < 1e-6 at iter %s", iteration + 1)
                break

            if self.config.early_stop and (iteration + 1) % 10 == 0:
                success = self._check_success(
                    model, x_best_adv, y, targeted, target_labels
                )
                if success.all():
                    if self.verbose:
                        logger.debug("  Early stop at iteration %s", iteration + 1)
                    break

        return x_best_adv, loss_best, iterations_run


class APGDTargeted(APGDAttack):
    """
    Targeted APGD variant with multi-target strategy.
    """

    def __init__(
        self,
        config: AttackConfig,
        device: str = "cuda",
        n_target_classes: int = 9,
        **kwargs: Any,
    ):
        super().__init__(config, device, **kwargs)
        self.n_target_classes = int(n_target_classes)

    def run(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = True,
        target_labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> AttackResult:
        if not self.supports_targeted():
            raise NotImplementedError(f"{self._name} does not support targeted attacks.")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x has {x.shape[0]} samples, y has {y.shape[0]} labels"
            )

        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            logits = model(x)
            num_classes = logits.shape[1]

        batch_size = x.size(0)
        best_x_adv = x.clone()
        best_success = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        best_perturbation_norm = torch.full(
            (batch_size,), float("inf"), device=self.device
        )
        total_iterations = 0

        target_list = self._select_target_labels_topk(
            logits, y, k=min(self.n_target_classes, max(1, num_classes - 1))
        )

        for target_idx, target_labels in enumerate(target_list):

            result = super().run(model, x, y, targeted=True, target_labels=target_labels)
            total_iterations += int(result.iterations or 0)

            delta = (result.x_adv - x).view(batch_size, -1)
            norm_type = str(self.config.norm).lower()
            if norm_type in {"linf", "l_inf", "inf"}:
                pert_norm = delta.abs().max(dim=1)[0]
            elif norm_type in {"l2", "l_2"}:
                pert_norm = delta.norm(p=2, dim=1)
            else:
                pert_norm = delta.norm(p=1, dim=1)

            improved = (result.success_mask) & (pert_norm < best_perturbation_norm)
            best_x_adv[improved] = result.x_adv[improved]
            best_success[improved] = result.success_mask[improved]
            best_perturbation_norm[improved] = pert_norm[improved]

            if self.verbose:
                logger.info(
                    "  Target %s/%s: ASR = %.2f%%",
                    target_idx + 1,
                    self.n_target_classes,
                    best_success.float().mean().item() * 100.0,
                )

        with torch.no_grad():
            preds = model(best_x_adv).argmax(dim=1)

        return AttackResult(
            x_adv=best_x_adv,
            predictions=preds,
            success_mask=best_success,
            perturbation=best_x_adv - x,
            queries=total_iterations * self.eot_iter,
            iterations=total_iterations,
            metadata={
                "attack_type": "APGD-T",
                "n_target_classes": self.n_target_classes,
                "target_strategy": "topk",
                "targeted": True,
            },
        )

    def _select_target_labels_topk(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        k: int = 9,
    ) -> List[torch.Tensor]:
        """
        Select top-k target classes (AutoAttack APGD-T strategy).
        """
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        logits_masked = logits.clone()
        logits_masked[torch.arange(batch_size, device=logits.device), y] = -float("inf")

        _, top_indices = logits_masked.topk(k=min(k, num_classes - 1), dim=1)

        return [top_indices[:, i] for i in range(top_indices.shape[1])]

    def _select_target_labels(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        strategy: str = "random",
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        if strategy == "random":
            target_labels = torch.randint(0, num_classes, (batch_size,), device=logits.device)
            same_as_true = target_labels == y
            while same_as_true.any():
                target_labels[same_as_true] = torch.randint(
                    0,
                    num_classes,
                    (int(same_as_true.sum().item()),),
                    device=logits.device,
                )
                same_as_true = target_labels == y
        elif strategy == "least_likely":
            logits_except_y = logits.clone()
            logits_except_y[torch.arange(batch_size, device=logits.device), y] = float("inf")
            target_labels = logits_except_y.argmin(dim=1)
        else:
            raise ValueError(f"Unknown target selection strategy: {strategy}")

        return target_labels


__all__ = [
    "AttackCapability",
    "AttackConfig",
    "AttackResult",
    "BaseAdversarialAttack",
    "GradientBasedAttack",
    "PGDAttack",
    "PGDWithRestarts",
    "APGDAttack",
    "APGDTargeted",
]
