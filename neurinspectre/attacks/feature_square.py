"""Query-based random search on feature vectors (not images).

Image Square assumes NCHW and [0, 1] pixels. EMBER features are 2D and
mixed-scale, so that clamp would be a silent correctness bug. This search
stays inside an L-inf ball around the original vector and does not claim
the perturbation is a valid PE.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Attack


class FeatureSquareAttack(Attack):
    """Score- or label-based random block search on ``[N, F]`` features."""

    def __init__(
        self,
        model: nn.Module,
        eps: float = 1.0,
        n_queries: int = 5000,
        p_init: float = 0.1,
        loss_type: str = "margin",
        device: str = "cpu",
        allow_short_budget: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(model, device)
        self.eps = float(eps)
        self.n_queries = int(n_queries)
        self.p_init = float(p_init)
        self.loss_type = str(loss_type)
        self.allow_short_budget = bool(allow_short_budget)
        self.seed = None if seed is None else int(seed)
        self._np_rng = np.random.RandomState(self.seed)
        self._torch_gen = torch.Generator(device="cpu")
        if self.seed is not None:
            self._torch_gen.manual_seed(self.seed + 1)
        if self.loss_type in {"labels", "hard"}:
            self.loss_type = "label"
        if self.loss_type not in {"margin", "ce", "label"}:
            raise ValueError("loss_type must be 'margin', 'ce', or 'label'.")
        if self.n_queries < 1000 and not self.allow_short_budget:
            raise ValueError(
                "FeatureSquare requires >=1000 queries for reliable results "
                f"(got n_queries={self.n_queries}). "
                "Pass allow_short_budget=True for budget curves."
            )

    def _margin_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b = logits.size(0)
        z_y = logits[torch.arange(b, device=logits.device), y]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(b, device=logits.device), y] = False
        z_max_other = logits[mask].view(b, -1).max(dim=1)[0]
        return -(z_y - z_max_other)

    def _attack_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 1 or (logits.ndim == 2 and logits.size(1) == 1):
            scores = logits.view(-1)
            # Binary: treat score>0 as class 1.
            if self.loss_type == "label":
                preds = (scores > 0).long()
                return (preds != y).float()
            signed = torch.where(y > 0, -scores, scores)
            return signed
        if self.loss_type == "margin":
            return self._margin_loss(logits, y)
        if self.loss_type == "label":
            return (logits.argmax(1) != y).float()
        return F.cross_entropy(logits, y, reduction="none")

    def _preds(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 1 or (logits.ndim == 2 and logits.size(1) == 1):
            return (logits.view(-1) > 0).long()
        return logits.argmax(1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        if targeted:
            raise ValueError("FeatureSquareAttack is untargeted (malware evasion).")
        x = x.to(self.device)
        y = y.to(self.device)
        if x.ndim != 2:
            raise ValueError("FeatureSquareAttack expects 2D inputs (N, F).")
        batch_size, n_features = x.shape
        delta = torch.zeros_like(x)

        with torch.no_grad():
            logits_init = self.model(x + delta)
            loss_best = self._attack_loss(logits_init, y)
            success = self._preds(logits_init) != y

        queries_used = torch.ones(batch_size, device=self.device)

        for query in range(self.n_queries - 1):
            active_idx = (~success).nonzero(as_tuple=False).squeeze(1)
            if active_idx.numel() == 0:
                break
            p = self.p_init * (1.0 - query / max(self.n_queries, 1))
            width = max(1, int(max(1, n_features) * max(p, 1.0 / n_features)))
            starts = self._np_rng.randint(0, max(1, n_features - width + 1), size=(int(active_idx.numel()),))
            delta_new = delta[active_idx].clone()
            for i, start in enumerate(starts):
                noise = torch.empty(width, dtype=delta.dtype)
                noise.uniform_(-self.eps, self.eps, generator=self._torch_gen)
                delta_new[i, int(start) : int(start) + width] = noise.to(self.device)
            delta_new.clamp_(-self.eps, self.eps)
            with torch.no_grad():
                logits_new = self.model(x[active_idx] + delta_new)
                loss_new = self._attack_loss(logits_new, y[active_idx])
                cand = self._preds(logits_new) != y[active_idx]
                # Keep a label flip even if the surrogate loss did not increase.
                keep = (loss_new > loss_best[active_idx]) | cand
                if keep.any():
                    delta[active_idx[keep]] = delta_new[keep]
                    loss_best[active_idx[keep]] = loss_new[keep]
                queries_used[active_idx] += 1
                if cand.any():
                    success[active_idx[cand]] = True

        stats = {
            "queries_used": queries_used.cpu().numpy(),
            "success": success.cpu().numpy(),
            "asr": success.float().mean().item(),
            "loss_type": self.loss_type,
            "realizable": False,
            "space": "feature",
            "query_floor_relaxed": bool(self.allow_short_budget and self.n_queries < 1000),
        }
        if verbose:
            print(f"[FeatureSquare] ASR={stats['asr']*100:.1f}%")
        return (x + delta).detach(), stats
