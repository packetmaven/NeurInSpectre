"""
AutoAttack ensemble implementation.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .apgd import APGD
from .base_interface import APGDTargeted, AttackConfig
from .fab import FABEnsemble
from .square import SquareAttack, SquareAttackL2


_GRADIENT_SUBATTACKS = frozenset({"apgd-ce", "apgd-dlr", "apgd-t", "apgd-md", "fab"})


class _APGDTargetedAdapter:
    """Make ``APGDTargeted.run`` look like the callable APGD/FAB interface."""

    def __init__(
        self,
        model: nn.Module,
        *,
        eps: float,
        norm: str,
        device: str,
        n_target_classes: int = 9,
        steps: int = 100,
        n_restarts: int = 1,
    ):
        cfg = AttackConfig(
            norm=norm,
            epsilon=float(eps),
            n_iterations=int(steps),
            n_restarts=int(n_restarts),
            loss="ce",
        )
        self.model = model
        self._inner = APGDTargeted(cfg, device=device, n_target_classes=int(n_target_classes))

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        result = self._inner.run(self.model, x, y, targeted=True)
        return result.x_adv


def _square_for_norm(model: nn.Module, *, eps: float, norm: str, n_queries: int, device: str):
    norm_key = str(norm).lower().replace("_", "")
    if norm_key in {"linf", "inf", "l∞"}:
        return SquareAttack(model, eps=eps, n_queries=n_queries, device=device)
    if norm_key in {"l2", "2"}:
        return SquareAttackL2(model, eps=eps, n_queries=n_queries, device=device)
    return None


def _project_lp(
    x0: torch.Tensor,
    x: torch.Tensor,
    *,
    eps: float,
    norm: str,
    clip_min: float,
    clip_max: float,
) -> torch.Tensor:
    """
    Project `x` into an Lp ball of radius `eps` around `x0`, then clamp to
    `[clip_min, clip_max]`.

    This is intentionally implemented here (rather than relying on individual
    sub-attacks) so the "AutoAttack" wrapper never reports unconstrained ASR.
    """
    norm_key = str(norm).lower().replace("_", "")
    eps_f = float(eps)
    if norm_key in {"linf", "inf", "l∞"}:
        delta = torch.clamp(x - x0, -eps_f, eps_f)
        out = x0 + delta
    elif norm_key in {"l2", "2"}:
        delta = x - x0
        flat = delta.view(delta.size(0), -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        factors = torch.minimum(torch.ones_like(norms), torch.full_like(norms, eps_f) / norms)
        out = x0 + (flat * factors).view_as(delta)
    else:
        raise ValueError(f"Unsupported norm for AutoAttack projection: {norm!r}")
    return torch.clamp(out, min=float(clip_min), max=float(clip_max))


class AutoAttackEnsemble:
    """
    Simple AutoAttack-style ensemble that applies a list of attacks sequentially.
    """

    def __init__(self, model, attacks, device: str = "cuda"):
        self.model = model.to(device)
        self.attacks = attacks
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)

        adv = x.clone()
        still_robust = torch.ones(x.size(0), dtype=torch.bool, device=self.device)

        for attack in self.attacks:
            if not still_robust.any():
                break

            adv_subset = adv[still_robust]
            y_subset = y[still_robust]

            adv_out = attack(adv_subset, y_subset)
            adv_attacked = adv_out[0] if isinstance(adv_out, (tuple, list)) else adv_out

            with torch.no_grad():
                logits = self.model(adv_attacked)
                preds = logits.argmax(1)
                newly_adv = preds != y_subset

            adv[still_robust] = adv_attacked
            still_robust_indices = still_robust.nonzero(as_tuple=False).squeeze(1)
            still_robust[still_robust_indices[newly_adv]] = False

        return adv


class AutoAttack:
    """
    AutoAttack ensemble for robust evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        norm: str = "linf",
        eps: float = 8 / 255,
        version: str = "standard",
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.norm = norm
        self.eps = float(eps)
        self.version = version
        self.device = device

        self._init_attacks()

    def _init_attacks(self):
        if self.version == "standard":
            self.attacks = {
                "apgd-ce": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="ce",
                    n_restarts=1,
                    device=self.device,
                ),
                "apgd-dlr": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="dlr",
                    n_restarts=1,
                    device=self.device,
                ),
                "apgd-t": _APGDTargetedAdapter(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    device=self.device,
                    n_target_classes=9,
                    steps=100,
                    n_restarts=1,
                ),
                "fab": FABEnsemble(self.model, norm=self.norm, device=self.device),
                "square": _square_for_norm(
                    self.model, eps=self.eps, norm=self.norm, n_queries=5000, device=self.device
                ),
            }
        elif self.version == "plus":
            self.attacks = {
                "apgd-ce": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="ce",
                    n_restarts=5,
                    device=self.device,
                ),
                "apgd-dlr": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="dlr",
                    n_restarts=5,
                    device=self.device,
                ),
                "apgd-md": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="md",
                    n_restarts=1,
                    device=self.device,
                ),
                "apgd-t": _APGDTargetedAdapter(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    device=self.device,
                    n_target_classes=9,
                    steps=100,
                    n_restarts=1,
                ),
                "fab": FABEnsemble(self.model, norm=self.norm, device=self.device),
                "square": _square_for_norm(
                    self.model, eps=self.eps, norm=self.norm, n_queries=10000, device=self.device
                ),
            }
        elif self.version == "rand":
            self.attacks = {
                "apgd-ce": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="ce",
                    n_restarts=10,
                    device=self.device,
                ),
                "apgd-dlr": APGD(
                    self.model,
                    eps=self.eps,
                    norm=self.norm,
                    steps=100,
                    loss="dlr",
                    n_restarts=10,
                    device=self.device,
                ),
                "fab": FABEnsemble(self.model, norm=self.norm, device=self.device),
            }
        else:
            raise ValueError("version must be one of: standard, plus, rand")

        self.attacks = {k: v for k, v in self.attacks.items() if v is not None}

    def run(self, x: torch.Tensor, y: torch.Tensor, verbose: bool = True) -> tuple:
        x = x.to(self.device)
        y = y.to(self.device)
        batch_size = x.size(0)
        x0 = x.detach().clone()
        clip_min = float(x0.min().item())
        clip_max = float(x0.max().item())

        attacks = self.attacks
        if x.ndim != 4:
            attacks = {k: v for k, v in self.attacks.items() if k.startswith("apgd")}
            if verbose:
                print(
                    "[AutoAttack] Non-image input detected; "
                    "running APGD-only subset for compatibility."
                )

        is_adversarial = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        x_adv = x.clone()

        asr_per_attack = {}
        samples_per_attack = {}
        skipped: List[str] = []
        skip_reasons: Dict[str, str] = {}

        for attack_name, attack in attacks.items():
            if verbose:
                print(f"\n[AutoAttack] Running {attack_name}...")
                print(f"  Currently adversarial: {is_adversarial.sum()}/{batch_size}")

            remaining_mask = ~is_adversarial
            if not remaining_mask.any():
                if verbose:
                    print(f"  All samples already adversarial, skipping {attack_name}")
                asr_per_attack[attack_name] = 0.0
                samples_per_attack[attack_name] = 0
                continue

            # Always constrain around the *original* clean inputs.
            x0_remaining = x0[remaining_mask]
            x_remaining = x0_remaining
            y_remaining = y[remaining_mask]

            try:
                if attack_name == "square":
                    x_attacked, _stats = attack(x_remaining, y_remaining, verbose=False)
                else:
                    adv_out = attack(x_remaining, y_remaining)
                    x_attacked = adv_out[0] if isinstance(adv_out, (tuple, list)) else adv_out
            except RuntimeError as exc:
                # Some defenses (e.g., hard-vote ensembles) are non-differentiable and can
                # break gradient-based sub-attacks like FAB/APGD. AutoAttack should keep
                # running remaining sub-attacks instead of aborting the full evaluation.
                msg = str(exc).lower()
                grad_missing = (
                    "does not require grad" in msg
                    or "does not have a grad_fn" in msg
                    or "element 0 of tensors" in msg
                )
                if grad_missing:
                    if verbose:
                        print(
                            f"  [AutoAttack] Skipping {attack_name}: gradient unavailable "
                            f"for this defense ({type(exc).__name__}: {exc})"
                        )
                    asr_per_attack[attack_name] = 0.0
                    samples_per_attack[attack_name] = 0
                    skipped.append(attack_name)
                    skip_reasons[attack_name] = "gradient_unavailable"
                    continue
                raise

            # Enforce the threat-model constraint in the wrapper (even if a sub-attack
            # drifts outside the allowed budget).
            x_attacked = _project_lp(
                x0_remaining,
                x_attacked,
                eps=self.eps,
                norm=self.norm,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            with torch.no_grad():
                preds = self.model(x_attacked).argmax(1)
                newly_adversarial = preds != y_remaining

                remaining_indices = remaining_mask.nonzero(as_tuple=False).squeeze(1)
                is_adversarial[remaining_indices[newly_adversarial]] = True
                x_adv[remaining_indices[newly_adversarial]] = x_attacked[newly_adversarial]

                asr_per_attack[attack_name] = newly_adversarial.float().mean().item()
                samples_per_attack[attack_name] = newly_adversarial.sum().item()

            if verbose:
                print(
                    f"  {attack_name} ASR: {asr_per_attack[attack_name]*100:.1f}% "
                    f"({samples_per_attack[attack_name]} new adversarial)"
                )

        total_asr = is_adversarial.float().mean().item()
        robust_accuracy = 1.0 - total_asr
        gradient_names = [name for name in attacks if name in _GRADIENT_SUBATTACKS]
        gradient_skipped = [name for name in skipped if name in _GRADIENT_SUBATTACKS]

        metrics = {
            "robust_accuracy": robust_accuracy,
            "asr": total_asr,
            "asr_per_attack": asr_per_attack,
            "samples_adversarial_per_attack": samples_per_attack,
            "total_adversarial": is_adversarial.sum().item(),
            "aa_subattacks_skipped": list(skipped),
            "aa_skip_reasons": dict(skip_reasons),
            "gradient_unavailable": bool(gradient_skipped),
            "aa_all_gradient_skipped": bool(gradient_names) and len(gradient_skipped) == len(gradient_names),
        }

        if verbose:
            print("\n[AutoAttack] Final Results:")
            print(f"  Total ASR: {total_asr*100:.1f}%")
            print(f"  Robust Accuracy: {robust_accuracy*100:.1f}%")
            print(f"  Adversarial samples: {is_adversarial.sum()}/{batch_size}")

        return x_adv, metrics

    def run_standard_eval(self, test_loader: torch.utils.data.DataLoader, n_examples: int = 10000) -> Dict:
        self.model.eval()

        total_samples = 0
        total_correct_clean = 0
        total_correct_robust = 0

        all_metrics = {"asr_per_attack": {name: [] for name in self.attacks.keys()}}

        for x_batch, y_batch in test_loader:
            if total_samples >= n_examples:
                break

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                preds_clean = self.model(x_batch).argmax(1)
                correct_clean = preds_clean == y_batch
                x_correct = x_batch[correct_clean]
                y_correct = y_batch[correct_clean]

            if x_correct.size(0) == 0:
                continue

            _, metrics = self.run(x_correct, y_correct, verbose=False)

            total_samples += x_batch.size(0)
            total_correct_clean += correct_clean.sum().item()
            total_correct_robust += correct_clean.sum().item() * metrics["robust_accuracy"]

            for attack_name, asr in metrics["asr_per_attack"].items():
                all_metrics["asr_per_attack"][attack_name].append(asr)

            print(
                f"Processed {total_samples}/{n_examples} samples | "
                f"Clean: {total_correct_clean/total_samples*100:.1f}% | "
                f"Robust: {total_correct_robust/total_correct_clean*100:.1f}%"
            )

        clean_accuracy = total_correct_clean / total_samples if total_samples > 0 else 0.0
        robust_accuracy = (
            total_correct_robust / total_correct_clean if total_correct_clean > 0 else 0.0
        )

        final_metrics = {
            "clean_accuracy": clean_accuracy,
            "robust_accuracy": robust_accuracy,
            "total_samples": total_samples,
            "avg_asr_per_attack": {
                name: float(np.mean(scores)) if scores else 0.0
                for name, scores in all_metrics["asr_per_attack"].items()
            },
        }

        return final_metrics
