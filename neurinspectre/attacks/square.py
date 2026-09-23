"""Square Attack (query-efficient, black-box)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Attack


class SquareAttack(Attack):
    """
    Query-efficient black-box adversarial attack via random search.

    Square Attack perturbs random square regions of the input and accepts
    improvements in a query-efficient manner. This is critical for validating
    that improvements aren't due to gradient masking.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        n_queries: int = 5000,
        p_init: float = 0.8,
        loss_type: str = "margin",
        device: str = "cuda",
        allow_short_budget: bool = False,
    ):
        super().__init__(model, device)
        self.eps = float(eps)
        self.n_queries = int(n_queries)
        self.p_init = float(p_init)
        self.loss_type = str(loss_type)
        self.allow_short_budget = bool(allow_short_budget)

        if self.loss_type not in {"margin", "ce", "label", "labels", "hard"}:
            raise ValueError("loss_type must be 'margin', 'ce', or 'label'.")
        if self.loss_type in {"labels", "hard"}:
            self.loss_type = "label"

        # Ensure sufficient queries for convergence.
        #
        # NOTE: Don't use `assert` here: python -O disables asserts, which would
        # silently permit meaningless "ran but didn't search" outcomes.
        # Practical Grosse-style modes may relax this via allow_short_budget.
        if self.n_queries < 1000 and not self.allow_short_budget:
            raise ValueError(
                "Square Attack requires >=1000 queries for reliable results "
                f"(got n_queries={self.n_queries}). "
                "Pass allow_short_budget=True for Grosse-style budget curves."
            )

    def _margin_loss(self, logits: torch.Tensor, y: torch.Tensor, targeted: bool = False) -> torch.Tensor:
        """
        Compute margin loss for Square Attack.

        Margin loss = z_y - max_{i≠y} z_i
        """
        b = logits.size(0)
        z_y = logits[torch.arange(b), y]

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(b), y] = False
        z_max_other = logits[mask].view(b, -1).max(dim=1)[0]

        margin = z_y - z_max_other
        return -margin if not targeted else margin

    def _label_loss(self, logits: torch.Tensor, y: torch.Tensor, targeted: bool = False) -> torch.Tensor:
        preds = logits.argmax(dim=1)
        flipped = preds != y
        return flipped.float() if not targeted else (~flipped).float()

    def _attack_loss(self, logits: torch.Tensor, y: torch.Tensor, targeted: bool = False) -> torch.Tensor:
        if self.loss_type == "margin":
            return self._margin_loss(logits, y, targeted)
        if self.loss_type == "label":
            return self._label_loss(logits, y, targeted)
        loss = F.cross_entropy(logits, y, reduction="none")
        return -loss if targeted else loss

    def _square_size_schedule(self, query: int) -> float:
        """Compute square size ratio p(q) = p_0 * (1 - q/Q)."""
        return self.p_init * (1.0 - query / self.n_queries)

    def _get_square_coordinates(self, img_size: Tuple[int, int], p: float) -> Tuple[int, int, int, int]:
        """Sample random square coordinates."""
        h, w = img_size
        s_h = max(1, int(np.sqrt(p) * h))
        s_w = max(1, int(np.sqrt(p) * w))
        h_start = np.random.randint(0, h - s_h + 1)
        w_start = np.random.randint(0, w - s_w + 1)
        return h_start, h_start + s_h, w_start, w_start + s_w

    def _initialize_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize perturbation with random uniform noise in [-eps, eps]."""
        delta = torch.zeros_like(x).uniform_(-self.eps, self.eps)
        delta = torch.clamp(x + delta, 0, 1) - x
        return delta

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial examples via Square Attack.

        Returns:
            (adversarial examples, stats dict)
        """
        x = x.to(self.device)
        y = y.to(self.device)
        batch_size = x.size(0)
        if x.ndim != 4:
            raise ValueError("SquareAttack expects 4D inputs (N,C,H,W).")
        _, c, h, w = x.shape

        delta = self._initialize_delta(x)

        with torch.no_grad():
            logits_init = self.model(x + delta)
            loss_best = self._attack_loss(logits_init, y, targeted)
            preds_init = logits_init.argmax(1)
            success = preds_init == y if targeted else preds_init != y

        # Official Square counts the initial evaluation as query 1.
        queries_used = torch.ones(batch_size, device=self.device)

        for query in range(max(self.n_queries - 1, 0)):
            p = self._square_size_schedule(query)
            # Optimization: only evaluate the remaining not-yet-successful subset.
            # This preserves the algorithm's semantics while dramatically reducing
            # compute for attacks where many samples become adversarial early.
            active_idx = (~success).nonzero(as_tuple=False).squeeze(1)
            if active_idx.numel() == 0:
                if verbose:
                    print(f"[Square Attack] All samples adversarial at query {query}/{self.n_queries}")
                break

            x_active = x[active_idx]
            y_active = y[active_idx]
            delta_active = delta[active_idx]
            loss_best_active = loss_best[active_idx]

            delta_new_active = delta_active.clone()
            active_bs = int(delta_new_active.size(0))

            # Vectorized patch update (no Python loop over samples).
            s_h = max(1, int(np.sqrt(float(p)) * h))
            s_w = max(1, int(np.sqrt(float(p)) * w))
            h_start = torch.from_numpy(
                np.random.randint(0, h - s_h + 1, size=(active_bs,), dtype=np.int64)
            ).to(device=delta_new_active.device)
            w_start = torch.from_numpy(
                np.random.randint(0, w - s_w + 1, size=(active_bs,), dtype=np.int64)
            ).to(device=delta_new_active.device)

            rows = h_start[:, None] + torch.arange(s_h, device=delta_new_active.device)[None, :]
            cols = w_start[:, None] + torch.arange(s_w, device=delta_new_active.device)[None, :]

            patch = torch.empty(
                (active_bs, c, s_h, s_w),
                device=delta_new_active.device,
                dtype=delta_new_active.dtype,
            ).uniform_(-self.eps, self.eps)

            b_idx = torch.arange(active_bs, device=delta_new_active.device)[:, None, None, None]
            ch_idx = torch.arange(c, device=delta_new_active.device)[None, :, None, None]
            r_idx = rows[:, None, :, None]
            c_idx = cols[:, None, None, :]
            delta_new_active[b_idx, ch_idx, r_idx, c_idx] = patch

            delta_new_active.clamp_(-self.eps, self.eps)
            delta_new_active = torch.clamp(x_active + delta_new_active, 0, 1) - x_active

            with torch.no_grad():
                logits_new_active = self.model(x_active + delta_new_active)
                loss_new_active = self._attack_loss(logits_new_active, y_active, targeted)

                improved_active = loss_new_active > loss_best_active
                if improved_active.any():
                    improved_idx = active_idx[improved_active]
                    delta[improved_idx] = delta_new_active[improved_active]
                    loss_best[improved_idx] = loss_new_active[improved_active]

                # Each active sample consumes one query this iteration.
                queries_used[active_idx] += 1.0

                preds = logits_new_active.argmax(1)
                if targeted:
                    candidate_success = preds == y_active
                else:
                    candidate_success = preds != y_active
                newly_success = candidate_success & improved_active
                if newly_success.any():
                    success[active_idx[newly_success]] = True

            if success.all():
                if verbose:
                    print(f"[Square Attack] All samples adversarial at query {query+1}/{self.n_queries}")
                break

        with torch.no_grad():
            logits_final = self.model(x + delta)
            final_margin = self._margin_loss(logits_final, y, targeted)

        stats = {
            "queries_used": queries_used.cpu().numpy(),
            "success": success.cpu().numpy(),
            "final_margin": final_margin.cpu().numpy(),
            "asr": success.float().mean().item(),
            "loss_type": self.loss_type,
            "query_floor_relaxed": bool(self.allow_short_budget and self.n_queries < 1000),
        }

        if verbose:
            print(f"[Square Attack] ASR: {stats['asr']*100:.1f}% | Avg queries: {queries_used.mean().item():.0f}")

        return (x + delta).detach(), stats


class SquareAttackL2(Attack):
    """Square Attack variant for L2 perturbations."""

    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.5,
        n_queries: int = 5000,
        p_init: float = 0.8,
        device: str = "cuda",
    ):
        super().__init__(model, device)
        self.eps = float(eps)
        self.n_queries = int(n_queries)
        self.p_init = float(p_init)

    def _project_l2(self, delta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        norms = delta.view(delta.size(0), -1).norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        factors = torch.min(self.eps / norms, torch.ones_like(norms))
        delta_proj = delta * factors.view(-1, 1, 1, 1)
        delta_proj = torch.clamp(x + delta_proj, 0, 1) - x
        return delta_proj

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x = x.to(self.device)
        y = y.to(self.device)
        batch_size = x.size(0)
        if x.ndim != 4:
            raise ValueError("SquareAttackL2 expects 4D inputs (N,C,H,W).")
        _, _, h, w = x.shape

        delta = torch.randn_like(x)
        delta = self._project_l2(delta, x)

        with torch.no_grad():
            logits_init = self.model(x + delta)
            loss_best = -(logits_init[torch.arange(batch_size), y] - logits_init.scatter(1, y.unsqueeze(1), -1e10).max(1)[0])
            success = logits_init.argmax(1) != y

        queries_used = torch.ones(batch_size, device=self.device)

        for query in range(max(self.n_queries - 1, 0)):
            p = self.p_init * (1.0 - query / self.n_queries)
            delta_new = delta.clone()

            for b in range(batch_size):
                if success[b]:
                    continue

                s = max(1, int(np.sqrt(p) * h))
                h_start = np.random.randint(0, h - s + 1)
                w_start = np.random.randint(0, w - s + 1)

                square_delta = torch.randn_like(delta[b, :, h_start : h_start + s, w_start : w_start + s])
                delta_new[b, :, h_start : h_start + s, w_start : w_start + s] = square_delta

                delta_new[b] = self._project_l2(delta_new[b].unsqueeze(0), x[b].unsqueeze(0)).squeeze(0)

            with torch.no_grad():
                logits_new = self.model(x + delta_new)
                loss_new = -(logits_new[torch.arange(batch_size), y] - logits_new.scatter(1, y.unsqueeze(1), -1e10).max(1)[0])

                improved = loss_new > loss_best
                if improved.any():
                    delta[improved] = delta_new[improved]
                    loss_best[improved] = loss_new[improved]

                queries_used += (~success).float()
                preds = logits_new.argmax(1)
                candidate_success = preds != y
                success = success | (candidate_success & improved)

            if success.all():
                break

        stats = {
            "queries_used": queries_used.cpu().numpy(),
            "success": success.cpu().numpy(),
            "asr": success.float().mean().item(),
        }

        return (x + delta).detach(), stats
