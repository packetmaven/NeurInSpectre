"""Official fra31 AutoAttack wrapper and BPDA-wrapped model for AA+BPDA."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn


def official_aa_norm(norm: str) -> str:
    key = str(norm).lower().replace("_", "")
    if key in {"linf", "inf", "l∞"}:
        return "Linf"
    if key in {"l2", "2"}:
        return "L2"
    if key in {"l1", "1"}:
        return "L1"
    raise ValueError(f"Unsupported AutoAttack norm: {norm!r}")


def import_official_autoattack():
    try:
        from autoattack import AutoAttack as OfficialAutoAttack
    except ImportError as exc:
        raise ImportError(
            "Official AutoAttack (fra31/auto-attack) is required for aa_official/aa_bpda. "
            "Install with: pip install 'autoattack @ git+https://github.com/fra31/auto-attack'"
        ) from exc
    return OfficialAutoAttack


class IdentityDefenseAdapter(nn.Module):
    """Minimal defense surface so AA+BPDA can run on an undefended model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.base_model = model
        self.model = model

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda x: x

    def forward(self, x: torch.Tensor, use_approximation: bool = False) -> torch.Tensor:
        return self.base_model(x)


class BPDAWrappedModel(nn.Module):
    """True defense forward, BPDA approximation on the backward pass."""

    def __init__(self, defense, approx_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        if defense is None:
            raise ValueError("BPDAWrappedModel requires a defense with transform()")
        self.defense = defense
        self.approx_fn = approx_fn or defense.get_bpda_approximation()
        base = getattr(defense, "base_model", None) or getattr(defense, "model", None)
        if base is None:
            raise ValueError("Defense must expose base_model or model")
        self.base_model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_actual = self.defense.transform(x)
        x_approx = self.approx_fn(x)
        if not torch.is_tensor(x_approx):
            x_approx = torch.as_tensor(x_approx, device=x.device, dtype=x.dtype)
        x_defended = x_actual + (x_approx - x_approx.detach())
        return self.base_model(x_defended)


def build_official_autoattack(
    model: nn.Module,
    *,
    norm: str,
    eps: float,
    version: str = "standard",
    device: str = "cpu",
    raw_config: Optional[Dict[str, Any]] = None,
):
    raw_config = raw_config or {}
    OfficialAutoAttack = import_official_autoattack()
    aa_norm = official_aa_norm(norm)
    version = str(raw_config.get("version", version) or "standard")
    kwargs: Dict[str, Any] = {
        "norm": aa_norm,
        "eps": float(eps),
        "version": version,
        "device": device,
        "verbose": bool(raw_config.get("verbose", False)),
    }
    attacks_to_run = raw_config.get("attacks_to_run")
    if version == "custom" or attacks_to_run:
        kwargs["version"] = "custom"
        kwargs["attacks_to_run"] = list(attacks_to_run or ["apgd-ce"])
    return OfficialAutoAttack(model, **kwargs)


def is_gradient_unavailable_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "does not require grad",
            "does not have a grad_fn",
            "element 0 of tensors",
            "not have been used in the graph",
            "allow_unused",
            "grad_fn",
        )
    )


def run_official_autoattack(adversary, x: torch.Tensor, y: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    bs = max(1, min(int(batch_size), int(x.size(0))))
    x_adv = adversary.run_standard_evaluation(x, y, bs=bs)
    if not torch.is_tensor(x_adv):
        x_adv = torch.as_tensor(x_adv, device=x.device, dtype=x.dtype)
    return x_adv
