"""
Unified AttackFactory for standardized attack creation and execution.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional

import torch

from .base_interface import AttackConfig, AttackResult, PGDAttack, PGDWithRestarts, APGDAttack
from .pgd import PGD
from .apgd import APGD
from .fab import FAB, FABEnsemble
from .square import SquareAttack, SquareAttackL2
from .feature_square import FeatureSquareAttack
from .bpda import BPDA
from .eot import EOT, AdaptiveEOT
from .ma_pgd import MAPGD
from .autoattack import AutoAttack
from .hybrid import HybridBPDAEOT, HybridBPDAEOTVolterra
from .official_aa import (
    BPDAWrappedModel,
    IdentityDefenseAdapter,
    build_official_autoattack,
    is_gradient_unavailable_error,
    run_official_autoattack,
)
from ..characterization.defense_analyzer import DefenseAnalyzer, ObfuscationType

logger = logging.getLogger(__name__)


class LossFunction(Enum):
    CROSS_ENTROPY = "ce"
    DLR = "dlr"
    CW = "cw"
    LOGIT_MARGIN = "logit"
    MINIMUM_MARGIN = "mm"
    ENHANCED_MARGIN = "logit_enhanced"




def _normalize_norm(norm: str) -> str:
    norm = str(norm).lower().replace("_", "")
    if norm in {"linf", "inf", "l∞"}:
        return "linf"
    if norm in {"l2", "2"}:
        return "l2"
    if norm in {"l1", "1"}:
        return "l1"
    return norm


def _parse_loss(loss: Any) -> LossFunction:
    if isinstance(loss, LossFunction):
        return loss
    loss_str = str(loss).lower()
    if loss_str in {"ce", "cross_entropy"}:
        return LossFunction.CROSS_ENTROPY
    if loss_str in {"dlr"}:
        return LossFunction.DLR
    if loss_str in {"cw", "carlini_wagner"}:
        return LossFunction.CW
    if loss_str in {"logit", "logit_margin"}:
        return LossFunction.LOGIT_MARGIN
    if loss_str in {"mm", "minimum_margin", "minimum-margin"}:
        return LossFunction.MINIMUM_MARGIN
    if loss_str in {"logit_enhanced", "enhanced_margin", "enhanced-margin"}:
        return LossFunction.ENHANCED_MARGIN
    raise ValueError(f"Unknown loss type: {loss}")


def _to_attack_config(config: Optional[Dict[str, Any]] | AttackConfig) -> AttackConfig:
    if config is None:
        return AttackConfig()
    if isinstance(config, AttackConfig):
        return config
    if not isinstance(config, dict):
        raise TypeError("config must be a dict or AttackConfig")

    use_tg_value = bool(config.get("use_tg", False))
    if bool(config.get("no_tg", False)):
        use_tg_value = False

    eot_samples = config.get("eot_samples", config.get("n_eot_samples", config.get("num_samples", 20)))
    eot_weighted = config.get("eot_importance_weighted", config.get("eot_importance_weighting", True))

    return AttackConfig(
        norm=_normalize_norm(config.get("norm", "linf")),
        epsilon=float(config.get("epsilon", config.get("eps", 8 / 255))),
        n_iterations=int(config.get("n_iterations", config.get("steps", config.get("attack_steps", 100)))),
        n_restarts=int(config.get("n_restarts", config.get("restarts", 1))),
        step_size=config.get("step_size", config.get("alpha")),
        random_init=bool(config.get("random_init", config.get("random_start", True))),
        loss=_parse_loss(config.get("loss", "ce")),
        loss_temperature=float(config.get("loss_temperature", 1.0)),
        loss_softmax_weighting=bool(config.get("loss_softmax_weighting", False)),
        kappa=float(config.get("kappa", config.get("cw_kappa", 0.0))),
        use_tg=use_tg_value,
        use_bpda=bool(config.get("use_bpda", False)),
        bpda_approximation=str(config.get("bpda_approximation", "defense")),
        use_eot=bool(config.get("use_eot", False)),
        eot_samples=int(eot_samples),
        eot_importance_weighted=bool(eot_weighted),
        rho=float(config.get("rho", 0.75)),
        targeted=bool(config.get("targeted", False)),
        target_class=config.get("target_class"),
        early_stop=bool(config.get("early_stop", True)),
        seed=int(config.get("seed", 42)),
        batch_size=int(config.get("batch_size", 128)),
        auto_step_size=bool(config.get("auto_step_size", False)),
        input_range=tuple(map(float, config.get("input_range", (0.0, 1.0)))),
        auto_detect_range=bool(config.get("auto_detect_range", True)),
    )


def _resolve_defense(model, defense):
    if defense is not None:
        return defense
    if hasattr(model, "transform") and hasattr(model, "get_bpda_approximation"):
        return model
    return None


def _resolve_base_model(model, defense):
    if defense is None:
        return model
    if hasattr(defense, "base_model"):
        return defense.base_model
    if hasattr(defense, "model"):
        return defense.model
    return model


def _resolve_bpda_approximation(defense, cfg: AttackConfig):
    approx_fn = defense.get_bpda_approximation()
    mode = str(getattr(cfg, "bpda_approximation", "defense")).lower()
    if mode in {"identity", "id"}:
        return lambda x: x
    if mode in {"defense", "auto", "default"}:
        return approx_fn
    return approx_fn


class AttackFactory:
    @staticmethod
    def create_attack(
        attack_type: str,
        model,
        *,
        config: Dict[str, Any],
        characterization=None,
        characterization_loader=None,
        defense=None,
        device: str = "cpu",
    ):
        attack_type = str(attack_type).lower()
        cfg = _to_attack_config(config)
        if attack_type == "pgd":
            return _PGDAttackRunner(model, cfg, device=device)
        if attack_type == "apgd":
            return _APGDAttackRunner(model, cfg, device=device, raw_config=config)
        if attack_type == "cw":
            cfg.loss = LossFunction.CW
            return _APGDAttackRunner(model, cfg, device=device, raw_config=config)
        if attack_type in {"mm", "minimum_margin", "minimum-margin"}:
            cfg.loss = LossFunction.MINIMUM_MARGIN
            return _APGDAttackRunner(model, cfg, device=device, raw_config=config)
        if attack_type in {"logit_enhanced", "enhanced_margin", "enhanced-margin"}:
            cfg.loss = LossFunction.ENHANCED_MARGIN
            return _APGDAttackRunner(model, cfg, device=device, raw_config=config)
        if attack_type == "autoattack":
            return _AutoAttackRunner(model, cfg, device=device, raw_config=config)
        if attack_type in {"aa_official", "official_autoattack", "autoattack_official"}:
            return _OfficialAutoAttackRunner(model, cfg, device=device, raw_config=config)
        if attack_type in {"aa_bpda", "autoattack_bpda", "aa_official_bpda"}:
            return _AABPDARunner(model, cfg, defense=defense, device=device, raw_config=config)
        if attack_type == "square":
            return _SquareAttackRunner(model, cfg, device=device, raw_config=config)
        raw_cfg = dict(config) if isinstance(config, dict) else {}
        if attack_type in {"scores", "square_scores"}:
            raw = dict(raw_cfg)
            raw.setdefault("loss_type", "margin")
            raw.setdefault("access", "scores")
            n_q = int(raw.get("n_queries", 5000))
            if n_q < 1000:
                raw.setdefault("allow_short_budget", True)
            return _SquareAttackRunner(model, cfg, device=device, raw_config=raw)
        if attack_type in {"labels", "square_labels"}:
            raw = dict(raw_cfg)
            raw["loss_type"] = "label"
            raw.setdefault("access", "labels")
            n_q = int(raw.get("n_queries", 5000))
            if n_q < 1000:
                raw.setdefault("allow_short_budget", True)
            return _SquareAttackRunner(model, cfg, device=device, raw_config=raw)
        if attack_type in {"feature_square", "square_features"}:
            raw = dict(raw_cfg)
            raw.setdefault("access", "scores")
            n_q = int(raw.get("n_queries", 5000))
            if n_q < 1000:
                raw.setdefault("allow_short_budget", True)
            return _FeatureSquareRunner(model, cfg, device=device, raw_config=raw)
        if attack_type == "fab":
            return _FABAttackRunner(model, cfg, device=device, raw_config=config)
        if attack_type == "mapgd":
            return _MAPGDAttackRunner(model, cfg, device=device, raw_config=config)
        if attack_type == "bpda":
            return _BPDAAttackRunner(model, cfg, defense=defense, device=device)
        if attack_type == "eot":
            return _EOTAttackRunner(model, cfg, defense=defense, device=device, raw_config=config)
        if attack_type == "hybrid":
            return _HybridAttackRunner(model, cfg, defense=defense, device=device, raw_config=config)
        if attack_type in {"hybrid_volterra", "hybrid-volterra"}:
            return _HybridVolterraAttackRunner(model, cfg, defense=defense, device=device, raw_config=config)
        if attack_type == "neurinspectre":
            return _NeurInSpectreRunner(
                model,
                cfg,
                characterization=characterization,
                characterization_loader=characterization_loader,
                defense=defense,
                device=device,
                raw_config=config,
            )
        raise ValueError(f"Unknown attack type: {attack_type}")

    @classmethod
    def create(
        cls,
        attack_name: str,
        config: Optional[AttackConfig] = None,
        device: str = "cpu",
        **kwargs,
    ):
        cfg = config or AttackConfig()
        cfg.loss = _parse_loss(cfg.loss)
        attack_name = str(attack_name).lower()
        if attack_name == "pgd":
            return PGD(kwargs["model"], eps=cfg.epsilon, alpha=cfg.step_size or 2 / 255, steps=cfg.n_iterations)
        if attack_name == "apgd":
            return APGD(kwargs["model"], eps=cfg.epsilon, steps=cfg.n_iterations, loss=cfg.loss.value, device=device)
        if attack_name == "bpda":
            return BPDA(kwargs["model"], kwargs["defense"], eps=cfg.epsilon, alpha=cfg.step_size or 2 / 255)
        if attack_name == "eot":
            return EOT(kwargs["model"], kwargs["transform_fn"], num_samples=cfg.eot_samples, eps=cfg.epsilon)
        raise ValueError(f"Unknown attack: {attack_name}")

    @classmethod
    def create_for_defense(
        cls,
        defense,
        config: Optional[AttackConfig] = None,
        device: str = "cpu",
    ):
        cfg = config or AttackConfig()
        cfg.loss = _parse_loss(cfg.loss)
        obf_types = getattr(defense, "obfuscation_types", [])
        if ObfuscationType.SHATTERED in obf_types and ObfuscationType.STOCHASTIC in obf_types:
            return HybridBPDAEOT(
                _resolve_base_model(defense, defense),
                defense,
                approx_fn=_resolve_bpda_approximation(defense, cfg),
                num_samples=cfg.eot_samples,
                eps=cfg.epsilon,
                alpha=cfg.step_size or 2 / 255,
                steps=cfg.n_iterations,
                norm=cfg.norm,
                device=device,
            )
        if ObfuscationType.SHATTERED in obf_types:
            return BPDA(
                _resolve_base_model(defense, defense),
                defense.transform,
                approx_fn=_resolve_bpda_approximation(defense, cfg),
                eps=cfg.epsilon,
                alpha=cfg.step_size or 2 / 255,
                steps=cfg.n_iterations,
                norm=cfg.norm,
                device=device,
            )
        if ObfuscationType.STOCHASTIC in obf_types:
            return EOT(
                _resolve_base_model(defense, defense),
                defense.transform,
                num_samples=cfg.eot_samples,
                importance_sampling=cfg.eot_importance_weighted,
                eps=cfg.epsilon,
                alpha=cfg.step_size or 2 / 255,
                steps=cfg.n_iterations,
                norm=cfg.norm,
                device=device,
            )
        if ObfuscationType.VANISHING in obf_types:
            cfg.loss = LossFunction.LOGIT_MARGIN
        return APGD(
            _resolve_base_model(defense, defense),
            eps=cfg.epsilon,
            steps=cfg.n_iterations,
            norm=cfg.norm,
            loss=cfg.loss.value,
            device=device,
        )

    @classmethod
    def create_autoattack(cls, model, config: Optional[AttackConfig] = None, device: str = "cpu"):
        cfg = config or AttackConfig()
        return AutoAttack(model, norm=cfg.norm, eps=cfg.epsilon, version="standard", device=device)


class _BaseRunner:
    def __init__(self, model, attack, eval_model=None):
        self.model = model
        self.attack = attack
        self.eval_model = eval_model or model

    def _predict(self, x_adv: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.eval_model(x_adv).argmax(1)


class _PGDAttackRunner(_BaseRunner):
    def __init__(self, model, cfg: AttackConfig, device: str = "cpu"):
        if cfg.n_restarts > 1:
            attack = PGDWithRestarts(cfg, n_restarts=cfg.n_restarts, device=device)
        else:
            attack = PGDAttack(cfg, device=device)
        super().__init__(model, attack)
        self.cfg = cfg

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        target_labels = None
        if self.cfg.targeted and self.cfg.target_class is not None:
            target_labels = torch.full_like(y, int(self.cfg.target_class))
        result = self.attack.run(
            self.model,
            x,
            y,
            targeted=self.cfg.targeted,
            target_labels=target_labels,
        )
        preds = self._predict(result.x_adv)
        result.predictions = preds
        probe = x[:1].detach().requires_grad_(True)
        try:
            logits = self.model(probe)
            no_grad = not torch.is_tensor(logits) or not bool(getattr(logits, "requires_grad", False))
        except Exception:
            no_grad = True
        if no_grad:
            result.metadata = dict(result.metadata or {})
            result.metadata["gradient_unavailable"] = True
            result.metadata["chosen_attack"] = "pgd"
        return result


class _APGDAttackRunner(_BaseRunner):
    def __init__(
        self,
        model,
        cfg: AttackConfig,
        device: str = "cpu",
        raw_config: Optional[Dict[str, Any]] = None,
    ):
        raw_config = raw_config or {}
        loss_params: Dict[str, Any] = {}
        if isinstance(raw_config.get("loss_params"), dict):
            loss_params.update(raw_config["loss_params"])
        if "loss_temperature" in raw_config:
            loss_params["temperature"] = raw_config.get("loss_temperature")
        if "loss_softmax_weighting" in raw_config:
            loss_params["use_softmax_weighting"] = raw_config.get("loss_softmax_weighting")
        if loss_params:
            cfg.loss_temperature = float(loss_params.get("temperature", cfg.loss_temperature))
            cfg.loss_softmax_weighting = bool(
                loss_params.get("use_softmax_weighting", cfg.loss_softmax_weighting)
            )
        if "use_tg" in raw_config or "no_tg" in raw_config:
            cfg.use_tg = bool(raw_config.get("use_tg", not raw_config.get("no_tg", False)))

        eot_iter = int(raw_config.get("eot_iter", cfg.eot_samples if cfg.use_eot else 1))
        attack = APGDAttack(
            cfg,
            device=device,
            n_restarts=cfg.n_restarts,
            loss_type=cfg.loss,
            eot_iter=eot_iter,
            rho=cfg.rho,
            verbose=bool(raw_config.get("verbose", False)),
            use_tg=cfg.use_tg,
        )
        super().__init__(model, attack)
        self.cfg = cfg

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        target_labels = None
        if self.cfg.targeted and self.cfg.target_class is not None:
            target_labels = torch.full_like(y, int(self.cfg.target_class))
        result = self.attack.run(
            self.model,
            x,
            y,
            targeted=self.cfg.targeted,
            target_labels=target_labels,
        )
        preds = self._predict(result.x_adv)
        result.predictions = preds
        return result


class _AutoAttackRunner(_BaseRunner):
    def __init__(self, model, cfg: AttackConfig, device: str = "cpu", raw_config: Optional[Dict[str, Any]] = None):
        raw_config = raw_config or {}
        attack = AutoAttack(
            model,
            norm=cfg.norm,
            eps=cfg.epsilon,
            version=str(raw_config.get("version", "standard")),
            device=device,
        )
        super().__init__(model, attack)

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        x_adv, metrics = self.attack.run(x, y, verbose=False)
        preds = self._predict(x_adv)
        success = preds != y
        metadata = {"autoattack": metrics, "backend": "inrepo"}
        metadata.update({k: metrics[k] for k in (
            "aa_subattacks_skipped",
            "aa_skip_reasons",
            "gradient_unavailable",
            "aa_all_gradient_skipped",
        ) if k in metrics})
        return AttackResult(x_adv=x_adv, predictions=preds, success_mask=success, metadata=metadata)


class _OfficialAutoAttackRunner(_BaseRunner):
    def __init__(self, model, cfg: AttackConfig, device: str = "cpu", raw_config: Optional[Dict[str, Any]] = None):
        raw_config = raw_config or {}
        self.cfg = cfg
        self._raw_config = dict(raw_config)
        self._device = device
        adversary = build_official_autoattack(
            model,
            norm=cfg.norm,
            eps=cfg.epsilon,
            version=str(raw_config.get("version", "standard")),
            device=device,
            raw_config=raw_config,
        )
        super().__init__(model, adversary)
        self.chosen_attack = "aa_official"
        self.selected_attack_impl = "official.AutoAttack"

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        skipped = []
        try:
            x_adv = run_official_autoattack(self.attack, x, y, batch_size=self.cfg.batch_size)
        except RuntimeError as exc:
            if not is_gradient_unavailable_error(exc):
                raise
            logger.warning(
                "Official AutoAttack skipped: gradient unavailable on this defense (%s)",
                exc,
            )
            x_adv = x
            skipped = list(self._raw_config.get("attacks_to_run") or ["apgd-ce", "apgd-t", "fab", "square"])
        preds = self._predict(x_adv)
        success = preds != y
        return AttackResult(
            x_adv=x_adv,
            predictions=preds,
            success_mask=success,
            metadata={
                "backend": "fra31",
                "version": str(self._raw_config.get("version", "standard")),
                "chosen_attack": "aa_official",
                "selected_attack_impl": "official.AutoAttack",
                "aa_subattacks_skipped": skipped,
                "gradient_unavailable": bool(skipped),
                "aa_all_gradient_skipped": bool(skipped),
            },
        )


class _AABPDARunner(_BaseRunner):
    def __init__(
        self,
        model,
        cfg: AttackConfig,
        defense=None,
        device: str = "cpu",
        raw_config: Optional[Dict[str, Any]] = None,
    ):
        raw_config = raw_config or {}
        defense = _resolve_defense(model, defense) or IdentityDefenseAdapter(model)
        approx_fn = _resolve_bpda_approximation(defense, cfg)
        wrapped = BPDAWrappedModel(defense, approx_fn=approx_fn)
        wrapped.eval()
        self.cfg = cfg
        self._raw_config = dict(raw_config)
        self._device = device
        backend = str(raw_config.get("backend", "official")).lower()
        if backend in {"inrepo", "in-repo", "internal"}:
            adversary = AutoAttack(
                wrapped,
                norm=cfg.norm,
                eps=cfg.epsilon,
                version=str(raw_config.get("version", "standard")),
                device=device,
            )
            self._backend = "inrepo"
        else:
            adversary = build_official_autoattack(
                wrapped,
                norm=cfg.norm,
                eps=cfg.epsilon,
                version=str(raw_config.get("version", "standard")),
                device=device,
                raw_config=raw_config,
            )
            self._backend = "fra31"
        super().__init__(wrapped, adversary, eval_model=defense)
        self.chosen_attack = "aa_bpda"
        self.selected_attack_impl = f"{self._backend}.AutoAttack+BPDA"

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        if self._backend == "inrepo":
            x_adv, metrics = self.attack.run(x, y, verbose=False)
            metadata = {"autoattack": metrics, "backend": "inrepo", "bpda": True}
            metadata.update({k: metrics[k] for k in (
                "aa_subattacks_skipped",
                "aa_skip_reasons",
                "gradient_unavailable",
                "aa_all_gradient_skipped",
            ) if k in metrics})
        else:
            try:
                x_adv = run_official_autoattack(self.attack, x, y, batch_size=self.cfg.batch_size)
                metadata = {
                    "backend": "fra31",
                    "version": str(self._raw_config.get("version", "standard")),
                    "bpda": True,
                }
            except RuntimeError as exc:
                if not is_gradient_unavailable_error(exc):
                    raise
                logger.warning("AA+BPDA official AutoAttack skipped: gradient unavailable (%s)", exc)
                x_adv = x
                metadata = {
                    "backend": "fra31",
                    "version": str(self._raw_config.get("version", "standard")),
                    "bpda": True,
                    "aa_subattacks_skipped": list(self._raw_config.get("attacks_to_run") or ["apgd-ce"]),
                    "gradient_unavailable": True,
                    "aa_all_gradient_skipped": True,
                }
        preds = self._predict(x_adv)
        success = preds != y
        metadata["chosen_attack"] = "aa_bpda"
        metadata["selected_attack_impl"] = self.selected_attack_impl
        return AttackResult(x_adv=x_adv, predictions=preds, success_mask=success, metadata=metadata)


class _SquareAttackRunner(_BaseRunner):
    def __init__(self, model, cfg: AttackConfig, device: str = "cpu", raw_config: Optional[Dict[str, Any]] = None):
        raw_config = raw_config or {}
        n_queries = int(raw_config.get("n_queries", 5000))
        p_init = float(raw_config.get("p_init", 0.8))
        allow_short = bool(raw_config.get("allow_short_budget", False))
        access = str(raw_config.get("access") or "scores")
        if str(cfg.norm).lower().replace("_", "") in {"l2", "2"}:
            attack = SquareAttackL2(
                model,
                eps=cfg.epsilon,
                n_queries=n_queries,
                p_init=p_init,
                device=device,
            )
        else:
            attack = SquareAttack(
                model,
                eps=cfg.epsilon,
                n_queries=n_queries,
                p_init=p_init,
                loss_type=str(raw_config.get("loss_type", "margin")),
                device=device,
                allow_short_budget=allow_short,
            )
        super().__init__(model, attack)
        self.chosen_attack = "labels" if str(raw_config.get("loss_type", "")).lower() in {"label", "labels", "hard"} else "scores"
        self.selected_attack_impl = type(attack).__name__
        self._access = access
        self._allow_short_budget = allow_short

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        x_adv, stats = self.attack(x, y, targeted=False, verbose=False)
        preds = self._predict(x_adv)
        success = preds != y
        return AttackResult(
            x_adv=x_adv,
            predictions=preds,
            success_mask=success,
            metadata={
                "queries_used": stats.get("queries_used"),
                "square": stats,
                "access": self._access,
                "allow_short_budget": self._allow_short_budget,
                "query_floor_relaxed": bool(self._allow_short_budget and self.attack.n_queries < 1000),
                "chosen_attack": self.chosen_attack,
                "selected_attack_impl": self.selected_attack_impl,
            },
        )


class _FeatureSquareRunner(_BaseRunner):
    def __init__(self, model, cfg: AttackConfig, device: str = "cpu", raw_config: Optional[Dict[str, Any]] = None):
        raw_config = raw_config or {}
        n_queries = int(raw_config.get("n_queries", 5000))
        allow_short = bool(raw_config.get("allow_short_budget", False))
        attack = FeatureSquareAttack(
            model,
            eps=float(raw_config.get("epsilon", cfg.epsilon if cfg.epsilon and cfg.epsilon > 0.1 else 1.0)),
            n_queries=n_queries,
            p_init=float(raw_config.get("p_init", 0.1)),
            loss_type=str(raw_config.get("loss_type", "margin")),
            device=device,
            allow_short_budget=allow_short,
        )
        super().__init__(model, attack)
        self.chosen_attack = "feature_square"
        self.selected_attack_impl = "FeatureSquareAttack"
        self._allow_short_budget = allow_short
        self._access = str(raw_config.get("access") or "scores")

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        x_adv, stats = self.attack(x, y, targeted=False, verbose=False)
        preds = self._predict(x_adv)
        success = preds != y
        return AttackResult(
            x_adv=x_adv,
            predictions=preds,
            success_mask=success,
            metadata={
                "queries_used": stats.get("queries_used"),
                "feature_square": stats,
                "access": self._access,
                "allow_short_budget": self._allow_short_budget,
                "query_floor_relaxed": bool(self._allow_short_budget and self.attack.n_queries < 1000),
                "realizable": False,
                "space": "feature",
                "chosen_attack": self.chosen_attack,
                "selected_attack_impl": self.selected_attack_impl,
            },
        )


class _FABAttackRunner(_BaseRunner):
    def __init__(self, model, cfg: AttackConfig, device: str = "cpu", raw_config: Optional[Dict[str, Any]] = None):
        raw_config = raw_config or {}
        use_ensemble = bool(raw_config.get("ensemble", False))
        if use_ensemble:
            attack = FABEnsemble(model, norm=cfg.norm, device=device)
        else:
            attack = FAB(
                model,
                norm=cfg.norm,
                steps=int(raw_config.get("steps", cfg.n_iterations)),
                n_restarts=int(raw_config.get("n_restarts", cfg.n_restarts)),
                device=device,
            )
        super().__init__(model, attack)

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        x_adv = self.attack(x, y)
        preds = self._predict(x_adv)
        success = preds != y
        return AttackResult(x_adv=x_adv, predictions=preds, success_mask=success, metadata={})


class _MAPGDAttackRunner(_BaseRunner):
    def __init__(self, model, cfg: AttackConfig, device: str = "cpu", raw_config: Optional[Dict[str, Any]] = None):
        raw_config = raw_config or {}
        alpha = float(cfg.step_size) if cfg.step_size is not None else float(raw_config.get("alpha", 2 / 255))
        attack = MAPGD(
            model,
            eps=cfg.epsilon,
            alpha=alpha,
            steps=int(raw_config.get("steps", cfg.n_iterations)),
            norm=cfg.norm,
            alpha_volterra=raw_config.get("alpha_volterra"),
            memory_length=raw_config.get("memory_length"),
            kernel_type=str(raw_config.get("kernel", "power_law")),
            device=device,
        )
        super().__init__(model, attack)

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        x_adv = self.attack(x, y)
        preds = self._predict(x_adv)
        success = preds != y
        return AttackResult(x_adv=x_adv, predictions=preds, success_mask=success, metadata={})


class _BPDAAttackRunner(_BaseRunner):
    def __init__(self, model, cfg: AttackConfig, defense=None, device: str = "cpu"):
        defense = _resolve_defense(model, defense)
        if defense is None:
            raise ValueError("BPDA attack requires a defense wrapper with transform()")
        base_model = _resolve_base_model(model, defense)
        alpha = float(cfg.step_size) if cfg.step_size is not None else 2 / 255
        attack = BPDA(
            base_model,
            defense.transform,
            approx_fn=_resolve_bpda_approximation(defense, cfg),
            eps=cfg.epsilon,
            alpha=alpha,
            steps=cfg.n_iterations,
            norm=cfg.norm,
            device=device,
        )
        super().__init__(base_model, attack, eval_model=defense)

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        x_adv = self.attack(x, y)
        preds = self._predict(x_adv)
        success = preds != y
        return AttackResult(x_adv=x_adv, predictions=preds, success_mask=success, metadata={})


class _EOTAttackRunner(_BaseRunner):
    def __init__(self, model, cfg: AttackConfig, defense=None, device: str = "cpu", raw_config: Optional[Dict[str, Any]] = None):
        raw_config = raw_config or {}
        defense = _resolve_defense(model, defense)
        if defense is None:
            raise ValueError("EOT attack requires a defense wrapper with transform()")
        base_model = _resolve_base_model(model, defense)
        alpha = float(cfg.step_size) if cfg.step_size is not None else 2 / 255
        if bool(raw_config.get("adaptive", False)):
            attack = AdaptiveEOT(
                base_model,
                defense.transform,
                target_variance=float(raw_config.get("target_variance", 0.01)),
                min_samples=int(raw_config.get("min_samples", 10)),
                max_samples=int(raw_config.get("max_samples", 100)),
                confidence=float(raw_config.get("confidence", 0.95)),
                eps=cfg.epsilon,
                alpha=alpha,
                steps=cfg.n_iterations,
                norm=cfg.norm,
                device=device,
            )
        else:
            attack = EOT(
                base_model,
                defense.transform,
                num_samples=cfg.eot_samples,
                importance_sampling=cfg.eot_importance_weighted,
                eps=cfg.epsilon,
                alpha=alpha,
                steps=cfg.n_iterations,
                norm=cfg.norm,
                device=device,
            )
        super().__init__(base_model, attack, eval_model=defense)

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        x_adv = self.attack(x, y)
        preds = self._predict(x_adv)
        success = preds != y
        return AttackResult(x_adv=x_adv, predictions=preds, success_mask=success, metadata={})


class _HybridAttackRunner(_BaseRunner):
    def __init__(self, model, cfg: AttackConfig, defense=None, device: str = "cpu", raw_config: Optional[Dict[str, Any]] = None):
        raw_config = raw_config or {}
        defense = _resolve_defense(model, defense)
        if defense is None:
            raise ValueError("Hybrid attack requires a defense wrapper with transform()")
        base_model = _resolve_base_model(model, defense)
        alpha = float(cfg.step_size) if cfg.step_size is not None else 2 / 255
        attack = HybridBPDAEOT(
            base_model,
            defense,
            approx_fn=_resolve_bpda_approximation(defense, cfg),
            num_samples=int(raw_config.get("eot_samples", cfg.eot_samples)),
            importance_sampling=cfg.eot_importance_weighted,
            eps=cfg.epsilon,
            alpha=alpha,
            steps=cfg.n_iterations,
            norm=cfg.norm,
            device=device,
        )
        super().__init__(base_model, attack, eval_model=defense)

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        x_adv = self.attack(x, y)
        preds = self._predict(x_adv)
        success = preds != y
        return AttackResult(x_adv=x_adv, predictions=preds, success_mask=success, metadata={})


class _HybridVolterraAttackRunner(_BaseRunner):
    def __init__(
        self,
        model,
        cfg: AttackConfig,
        defense=None,
        device: str = "cpu",
        raw_config: Optional[Dict[str, Any]] = None,
    ):
        raw_config = raw_config or {}
        defense = _resolve_defense(model, defense)
        if defense is None:
            raise ValueError("Hybrid-Volterra attack requires a defense wrapper with transform()")
        base_model = _resolve_base_model(model, defense)

        alpha_volterra = raw_config.get("alpha_volterra")
        memory_length = raw_config.get("memory_length")
        if alpha_volterra is None or memory_length is None:
            raise ValueError(
                "hybrid-volterra requires --volterra-alpha and --volterra-memory-length "
                "(or config keys alpha_volterra/memory_length)."
            )

        kernel_type = raw_config.get("volterra_kernel", raw_config.get("kernel", "power_law"))
        alpha = float(cfg.step_size) if cfg.step_size is not None else 2 / 255
        attack = HybridBPDAEOTVolterra(
            base_model,
            defense,
            approx_fn=_resolve_bpda_approximation(defense, cfg),
            num_samples=int(raw_config.get("eot_samples", cfg.eot_samples)),
            importance_sampling=cfg.eot_importance_weighted,
            eps=cfg.epsilon,
            alpha=alpha,
            steps=cfg.n_iterations,
            norm=cfg.norm,
            alpha_volterra=float(alpha_volterra),
            memory_length=int(memory_length),
            kernel_type=str(kernel_type),
            device=device,
        )
        super().__init__(base_model, attack, eval_model=defense)

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        x_adv = self.attack(x, y)
        preds = self._predict(x_adv)
        success = preds != y
        return AttackResult(x_adv=x_adv, predictions=preds, success_mask=success, metadata={})


class _NeurInSpectreRunner(_BaseRunner):
    def __init__(
        self,
        model,
        cfg: AttackConfig,
        *,
        characterization=None,
        characterization_loader=None,
        defense=None,
        device: str = "cpu",
        raw_config: Optional[Dict[str, Any]] = None,
    ):
        raw_config = raw_config or {}
        defense = _resolve_defense(model, defense)
        eval_model = defense or model
        char = characterization
        if char is None and characterization_loader is not None:
            thresholds = raw_config.get("characterization_thresholds")
            if isinstance(thresholds, dict) and isinstance(
                thresholds.get("defense_analyzer_threshold_overrides"), dict
            ):
                thresholds = thresholds["defense_analyzer_threshold_overrides"]
            if thresholds is not None and not isinstance(thresholds, dict):
                thresholds = None
            analyzer = DefenseAnalyzer(
                eval_model,
                n_samples=int(raw_config.get("characterization_samples", 50)),
                device=device,
                verbose=False,
                thresholds=thresholds,
                volterra_gradient_source=str(raw_config.get("volterra_gradient_source", "pre_optimizer")),
                volterra_optimizer=str(raw_config.get("volterra_optimizer", "sgd")),
                volterra_optimizer_beta1=float(raw_config.get("volterra_optimizer_beta1", raw_config.get("volterra_optimizer_beta", 0.9))),
                volterra_optimizer_beta2=float(raw_config.get("volterra_optimizer_beta2", 0.999)),
                volterra_optimizer_eps=float(raw_config.get("volterra_optimizer_eps", 1e-8)),
            )
            char = analyzer.characterize(characterization_loader, eps=cfg.epsilon)

        attack, chosen_name = self._select_attack(model, defense, cfg, char, device, raw_config)
        super().__init__(_resolve_base_model(model, defense), attack, eval_model=eval_model)
        self.characterization = char
        self._model = model
        self._defense = defense
        self._device = device
        self._raw_config = dict(raw_config)
        self.chosen_attack = chosen_name
        self.selected_attack_impl = type(attack).__name__
        self._annotate_characterization(char, chosen_name, self.selected_attack_impl)

    def update_config(self, config: Dict[str, Any]) -> None:
        cfg = _to_attack_config(config)
        self._raw_config.update(dict(config))
        self.attack, chosen_name = self._select_attack(
            self._model, self._defense, cfg, self.characterization, self._device, self._raw_config
        )
        self.chosen_attack = chosen_name
        self.selected_attack_impl = type(self.attack).__name__
        self._annotate_characterization(self.characterization, chosen_name, self.selected_attack_impl)

    @staticmethod
    def _annotate_characterization(char, chosen_name: str, impl_name: str) -> None:
        if char is None:
            return
        char.chosen_attack = chosen_name
        char.selected_attack_impl = impl_name
        metadata = getattr(char, "metadata", None)
        if isinstance(metadata, dict):
            metadata["chosen_attack"] = chosen_name
            metadata["selected_attack_impl"] = impl_name

    def _select_attack(self, model, defense, cfg: AttackConfig, char, device: str, raw_config: Dict[str, Any]):
        obf_types = getattr(char, "obfuscation_types", []) if char is not None else []
        requires_bpda = getattr(char, "requires_bpda", False) if char is not None else False
        requires_eot = getattr(char, "requires_eot", False) if char is not None else False
        requires_mapgd = getattr(char, "requires_mapgd", False) if char is not None else False
        if bool(raw_config.get("always_bpda", cfg.use_bpda)):
            requires_bpda = True
        if bool(raw_config.get("always_eot", cfg.use_eot)):
            requires_eot = True

        # Volterra/memory gating:
        # - "auto": use when characterization recommends MA-PGD / memory.
        # - "on": force-enable memory (best-effort fallbacks if no characterization).
        # - "off": never use memory.
        volterra_mode = str(raw_config.get("volterra_mode", "auto")).lower().strip()
        if volterra_mode not in {"auto", "on", "off"}:
            volterra_mode = "auto"
        use_volterra = volterra_mode == "on" or (volterra_mode == "auto" and bool(requires_mapgd))

        if ObfuscationType.VANISHING in obf_types:
            cfg.loss = LossFunction.LOGIT_MARGIN

        if requires_bpda and requires_eot:
            if use_volterra and defense is not None:
                alpha_volterra = raw_config.get("alpha_volterra")
                if alpha_volterra is None and char is not None:
                    alpha_volterra = getattr(char, "alpha_volterra", None)
                if alpha_volterra is None:
                    alpha_volterra = 0.5

                memory_length = raw_config.get("memory_length")
                if memory_length is None and char is not None:
                    memory_length = getattr(char, "recommended_memory_length", None)
                if memory_length is None:
                    memory_length = 20

                kernel_type = raw_config.get("volterra_kernel", raw_config.get("kernel", "power_law"))
                logger.info(
                    "[NeurInSpectre] Selected attack: HybridBPDAEOTVolterra "
                    "(BPDA+EOT+Volterra) mode=%s alpha=%s k=%s kernel=%s",
                    volterra_mode,
                    f"{float(alpha_volterra):.3f}" if alpha_volterra is not None else "n/a",
                    int(memory_length),
                    str(kernel_type),
                )
                return HybridBPDAEOTVolterra(
                    _resolve_base_model(model, defense),
                    defense,
                    approx_fn=_resolve_bpda_approximation(defense, cfg),
                    num_samples=int(raw_config.get("eot_samples", cfg.eot_samples)),
                    importance_sampling=cfg.eot_importance_weighted,
                    eps=cfg.epsilon,
                    alpha=float(cfg.step_size) if cfg.step_size is not None else 2 / 255,
                    steps=cfg.n_iterations,
                    norm=cfg.norm,
                    alpha_volterra=float(alpha_volterra),
                    memory_length=int(memory_length),
                    kernel_type=str(kernel_type),
                    device=device,
                ), "hybrid_volterra"
            logger.info("[NeurInSpectre] Selected attack: HybridBPDAEOT (BPDA+EOT) mode=%s", volterra_mode)
            return HybridBPDAEOT(
                _resolve_base_model(model, defense),
                defense,
                approx_fn=_resolve_bpda_approximation(defense, cfg),
                num_samples=int(raw_config.get("eot_samples", cfg.eot_samples)),
                importance_sampling=cfg.eot_importance_weighted,
                eps=cfg.epsilon,
                alpha=float(cfg.step_size) if cfg.step_size is not None else 2 / 255,
                steps=cfg.n_iterations,
                norm=cfg.norm,
                device=device,
            ), "hybrid"
        if requires_bpda and defense is not None:
            return BPDA(
                _resolve_base_model(model, defense),
                defense.transform,
                approx_fn=_resolve_bpda_approximation(defense, cfg),
                eps=cfg.epsilon,
                alpha=float(cfg.step_size) if cfg.step_size is not None else 2 / 255,
                steps=cfg.n_iterations,
                norm=cfg.norm,
                device=device,
            ), "bpda"
        if requires_eot and defense is not None:
            return EOT(
                _resolve_base_model(model, defense),
                defense.transform,
                num_samples=int(raw_config.get("eot_samples", cfg.eot_samples)),
                importance_sampling=cfg.eot_importance_weighted,
                eps=cfg.epsilon,
                alpha=float(cfg.step_size) if cfg.step_size is not None else 2 / 255,
                steps=cfg.n_iterations,
                norm=cfg.norm,
                device=device,
            ), "eot"
        if requires_mapgd and use_volterra:
            logger.info("[NeurInSpectre] Selected attack: MA-PGD (Volterra memory) mode=%s", volterra_mode)
            return MAPGD(
                model,
                eps=cfg.epsilon,
                alpha=float(cfg.step_size) if cfg.step_size is not None else 2 / 255,
                steps=cfg.n_iterations,
                norm=cfg.norm,
                alpha_volterra=getattr(char, "alpha_volterra", None) if char is not None else None,
                memory_length=getattr(char, "recommended_memory_length", None) if char is not None else None,
                device=device,
            ), "mapgd"
        return APGD(
            model,
            eps=cfg.epsilon,
            steps=cfg.n_iterations,
            norm=cfg.norm,
            loss=cfg.loss.value,
            n_restarts=cfg.n_restarts,
            device=device,
        ), "apgd"

    def run(self, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        x_adv = self.attack(x, y)
        preds = self._predict(x_adv)
        success = preds != y
        metadata = {
            "characterization": self.characterization.to_dict() if self.characterization else {},
            "chosen_attack": getattr(self, "chosen_attack", None),
            "selected_attack_impl": getattr(self, "selected_attack_impl", None),
        }
        return AttackResult(x_adv=x_adv, predictions=preds, success_mask=success, metadata=metadata)
