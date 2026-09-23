"""Official EMBER LightGBM GBDTs (2018 v2 and 2024 v3), wrapped as torch modules.

Both are binary sigmoid boosters (num_class=1). They have no input gradients.
PGD/APGD/official AA must record that, not invent a differentiable surrogate.

EMBER 2018: `PEFeatureExtractor(2)` (LIEF 0.9.0), dim 2381,
`ember_model_2018.txt`, Elastic-verified on non-Darwin + LIEF 0.9.0 / 0.10.1.

EMBER 2024 ("thrember", v3): `PEFeatureExtractor()`, dim discovered at load
time (2568 for the full PE extractor), `EMBER2024_PE.model` /
`EMBER2024_Win32.model` / `EMBER2024_Win64.model`. pefile-based, OS-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn


DEFAULT_EMBER_GBDT_PATH = "data/ember/ember2018/ember_model_2018.txt"
DEFAULT_EMBER2024_GBDT_PATH = "data/ember/ember2024/EMBER2024_PE.model"
EMBER_FEATURE_DIM = 2381
EMBER2024_PE_FEATURE_DIM = 2568


class EmberGBDT(nn.Module):
    """LightGBM booster → two-class logits. Backward is intentionally absent."""

    def __init__(self, booster: Any, *, feature_dim: int = EMBER_FEATURE_DIM):
        super().__init__()
        self.booster = booster
        self.feature_dim = int(feature_dim)

    @classmethod
    def from_file(cls, path: Union[str, Path], *, feature_dim: int = EMBER_FEATURE_DIM) -> "EmberGBDT":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "EMBER GBDT requires lightgbm. Install with: pip install lightgbm"
            ) from exc
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"EMBER LightGBM model not found: {path}")
        booster = lgb.Booster(model_file=str(path))
        return cls(booster, feature_dim=feature_dim)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.booster.predict(x), dtype=np.float64).reshape(-1)
        # Official EMBER model is binary sigmoid: P(malware).
        p1 = np.clip(raw, 1e-6, 1.0 - 1e-6)
        return np.stack([1.0 - p1, p1], axis=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"EmberGBDT expects (N, F) features, got {tuple(x.shape)}")
        if int(x.size(1)) != self.feature_dim:
            raise ValueError(
                f"EmberGBDT expected {self.feature_dim} features, got {int(x.size(1))}"
            )
        probs = self.predict_proba(x.detach().cpu().numpy())
        # Log-probabilities. softmax recovers P(benign), P(malware).
        # [-logit(p), +logit(p)] does not: softmax of that pair is sigmoid(2·logit(p)).
        out = np.log(probs)
        return torch.as_tensor(out, device=x.device, dtype=x.dtype)


def lightgbm_probability_from_stored_softmax(stored_p: float) -> float:
    """Recover the LightGBM probability from an old ``predict_malware`` score.

    ``EmberGBDT.forward`` used to emit ``[-logit(p), +logit(p)]``. Softmax of
    that pair is ``q = sigmoid(2·logit(p))``, which is what historical audit
    JSON stored as ``clean_p_malware`` / ``best_p_malware``. The inverse is
    ``p = sigmoid(0.5·logit(q))``. The 0.5 decision boundary is unchanged.
    """
    q = float(np.clip(stored_p, 1e-12, 1.0 - 1e-12))
    logit_q = float(np.log(q / (1.0 - q)))
    return float(1.0 / (1.0 + np.exp(-0.5 * logit_q)))


def load_ember_gbdt(path: Optional[Union[str, Path]] = None) -> EmberGBDT:
    return EmberGBDT.from_file(path or DEFAULT_EMBER_GBDT_PATH)


class EmberGBDT2024(EmberGBDT):
    """EMBER 2024 (v3, ``thrember``) LightGBM detector.

    Feature dim is discovered from the loaded booster because thrember publishes
    Win32/Win64/PE/all variants with slightly different sub-feature choices.
    We do not hardcode 2568; we read ``booster.num_feature()``.
    """

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        feature_dim: Optional[int] = None,
    ) -> "EmberGBDT2024":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "EMBER 2024 GBDT requires lightgbm. Install with: pip install lightgbm"
            ) from exc
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"EMBER 2024 LightGBM model not found: {path}. "
                "Fetch with: python scripts/download_ember2024.py"
            )
        booster = lgb.Booster(model_file=str(path))
        n = int(booster.num_feature())
        if feature_dim is not None and int(feature_dim) != n:
            raise ValueError(
                f"EMBER 2024 model {path.name} has {n} features; caller asked for {feature_dim}"
            )
        return cls(booster, feature_dim=n)


def load_ember2024_gbdt(path: Optional[Union[str, Path]] = None) -> EmberGBDT2024:
    return EmberGBDT2024.from_file(path or DEFAULT_EMBER2024_GBDT_PATH)
