"""
CLI utilities for NeurInSpectre.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

import yaml

from ..defenses.factory import DefenseFactory
from ..evaluation.datasets import DatasetFactory
from ..models.factory import ModelFactory
from ..models.loader import load_model as load_rb_model
from ..evaluation.metrics import compute_perturbation_metrics, compute_query_efficiency
from ..evaluation.problem_space import compute_asr_query_curve

logger = logging.getLogger(__name__)


def _supports_weights_only() -> bool:
    try:
        return "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        return False


def _is_weights_only_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "weights only load failed" in message or "weights_only" in message or "unsupported global" in message


def _build_safe_globals() -> list[type]:
    safe_names = [
        "Sequential",
        "Module",
        "ModuleList",
        "ModuleDict",
        "Flatten",
        "Linear",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "LayerNorm",
        "GroupNorm",
        "ReLU",
        "ReLU6",
        "LeakyReLU",
        "GELU",
        "SiLU",
        "Sigmoid",
        "Tanh",
        "MaxPool1d",
        "MaxPool2d",
        "MaxPool3d",
        "AvgPool1d",
        "AvgPool2d",
        "AvgPool3d",
        "AdaptiveAvgPool1d",
        "AdaptiveAvgPool2d",
        "AdaptiveAvgPool3d",
        "Dropout",
        "Dropout2d",
        "Dropout3d",
        "Identity",
    ]
    safe = []
    for name in safe_names:
        obj = getattr(nn, name, None)
        if obj is not None:
            safe.append(obj)
    return safe


_SAFE_GLOBALS = _build_safe_globals()


def _torch_load(path: Path, *, map_location: str, weights_only: Optional[bool] = None) -> Any:
    kwargs = {"map_location": map_location}
    if weights_only is not None and _supports_weights_only():
        kwargs["weights_only"] = weights_only
    return torch.load(path, **kwargs)


def _load_torch_checkpoint(path: Path) -> Any:
    if not _supports_weights_only():
        return torch.load(path, map_location="cpu")

    try:
        return _torch_load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not _is_weights_only_error(exc):
            raise

    if _SAFE_GLOBALS and hasattr(torch.serialization, "safe_globals"):
        try:
            with torch.serialization.safe_globals(_SAFE_GLOBALS):
                return _torch_load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            if not _is_weights_only_error(exc):
                raise

    logger.warning(
        "Weights-only load failed for %s; retrying with weights_only=False. "
        "Only do this for trusted checkpoints.",
        path,
    )
    return _torch_load(path, map_location="cpu", weights_only=False)


def resolve_device(device: str) -> str:
    device = str(device).lower()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU.")
        return "cpu"
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        logger.warning("MPS not available; falling back to CPU.")
        return "cpu"
    return device


def _should_pin_memory(device: Optional[str]) -> bool:
    if device is None:
        return torch.cuda.is_available()
    return str(device).lower() == "cuda"


def _format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _risk_label(asr: float, drop: float) -> str:
    if asr >= 0.5 or drop >= 0.4:
        return "HIGH"
    if asr >= 0.2 or drop >= 0.2:
        return "MEDIUM"
    return "LOW"


def _pick_norm_metrics(perturbation: Dict[str, Any], norm: str) -> Tuple[Optional[float], Optional[float]]:
    if not perturbation:
        return None, None
    norm_key = str(norm).lower()
    if norm_key in {"linf", "l_inf"}:
        return perturbation.get("linf_mean"), perturbation.get("linf_max")
    if norm_key == "l2":
        return perturbation.get("l2_mean"), perturbation.get("l2_max")
    if norm_key == "l1":
        return perturbation.get("l1_mean"), None
    return perturbation.get("primary_norm_mean"), perturbation.get("primary_norm_max")


def summarize_attack_findings(
    summary: Dict[str, Any],
    *,
    attack_type: str,
    defense: str,
    dataset: str,
    epsilon: float,
    norm: str,
    targeted: bool,
) -> Tuple[str, ...]:
    clean_acc = float(summary.get("clean_accuracy", 0.0))
    robust_acc = float(summary.get("robust_accuracy", 0.0))
    asr = float(summary.get("attack_success_rate", 0.0))
    drop = clean_acc - robust_acc
    risk = _risk_label(asr, drop)
    status = "COMPROMISE LIKELY" if asr >= 0.5 and clean_acc >= 0.5 else "ELEVATED RISK" if asr >= 0.2 else "LOW RISK"

    perturbation = summary.get("perturbation", {})
    norm_mean, norm_max = _pick_norm_metrics(perturbation, norm)

    query_eff = summary.get("query_efficiency", {}) or {}
    mean_queries = query_eff.get("mean_queries")
    median_queries = query_eff.get("median_queries")
    mean_iters = query_eff.get("mean_iterations")

    lines = [
        (
            "Status: "
            f"{status} (risk={risk}, ASR={_format_float(asr)}, "
            f"robust_acc={_format_float(robust_acc)}, clean_acc={_format_float(clean_acc)}, "
            f"drop={_format_float(drop)})"
        ),
        (
            "Attack surface: "
            f"attack={attack_type}, defense={defense}, dataset={dataset}, targeted={bool(targeted)}"
        ),
    ]

    if norm_mean is not None:
        if norm_max is not None:
            lines.append(
                f"Perturbation ({norm}): mean={_format_float(norm_mean)}, "
                f"max={_format_float(norm_max)} (epsilon={_format_float(epsilon)})"
            )
        else:
            lines.append(
                f"Perturbation ({norm}): mean={_format_float(norm_mean)} (epsilon={_format_float(epsilon)})"
            )
    if mean_queries is not None or mean_iters is not None:
        lines.append(
            "Query cost: "
            f"mean_queries={_format_float(mean_queries)}, "
            f"median_queries={_format_float(median_queries)}, "
            f"mean_iterations={_format_float(mean_iters)}"
        )
    return tuple(lines)


def summarize_characterization_findings(
    characterization: Dict[str, Any],
    *,
    defense: str,
    dataset: str,
) -> Tuple[str, ...]:
    obf_types = characterization.get("obfuscation_types") or []
    obf_str = ", ".join(obf_types) if obf_types else "none"
    confidence = characterization.get("confidence")
    etd = characterization.get("etd_score")
    alpha = characterization.get("alpha_volterra")
    grad_var = characterization.get("gradient_variance")
    rank = characterization.get("jacobian_rank")
    timescale = characterization.get("autocorr_timescale")

    requires_bpda = bool(characterization.get("requires_bpda"))
    requires_eot = bool(characterization.get("requires_eot"))
    requires_mapgd = bool(characterization.get("requires_mapgd"))
    eot_samples = characterization.get("recommended_eot_samples")
    memory_len = characterization.get("recommended_memory_length")

    bypasses = []
    if requires_bpda:
        bypasses.append("BPDA")
    if requires_eot:
        bypasses.append("EOT")
    if requires_mapgd:
        bypasses.append("MA-PGD")
    bypass_str = ", ".join(bypasses) if bypasses else "none"

    attack_surface = []
    if "shattered" in obf_types:
        attack_surface.append("shattered gradients -> BPDA")
    if "stochastic" in obf_types:
        attack_surface.append("stochastic defense -> EOT")
    if "vanishing" in obf_types:
        attack_surface.append("vanishing gradients -> logit-margin loss")
    if "rl_trained" in obf_types:
        attack_surface.append("temporal policy -> MA-PGD")
    if not attack_surface:
        attack_surface.append("no strong obfuscation signals detected")

    lines = [
        f"Obfuscation: {obf_str} (confidence={_format_float(confidence)})",
        (
            "Recommended bypass: "
            f"{bypass_str} (eot_samples={eot_samples}, memory_length={memory_len})"
        ),
        (
            "Signals: "
            f"ETD={_format_float(etd)}, alpha={_format_float(alpha)}, "
            f"grad_var={_format_float(grad_var)}, jacobian_rank={_format_float(rank)}, "
            f"autocorr_timescale={_format_float(timescale)}"
        ),
        f"Attack surface: defense={defense}, dataset={dataset}, focus={'; '.join(attack_surface)}",
    ]
    return tuple(lines)

def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def load_threshold_overrides(path: str | Path) -> Dict[str, Any]:
    """
    Load DefenseAnalyzer threshold overrides from JSON.

    Accepts either:
    - a direct overrides mapping, e.g. {"ETD_THRESHOLD_SEVERE": 0.65}
    - a full calibrate-thresholds payload containing "defense_analyzer_threshold_overrides"
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        obj: Any = json.load(handle)
    if isinstance(obj, dict) and isinstance(obj.get("defense_analyzer_threshold_overrides"), dict):
        obj = obj["defense_analyzer_threshold_overrides"]
    if not isinstance(obj, dict):
        raise ValueError(f"Threshold overrides JSON must be an object/dict: {path}")
    # Defensive copy: callers may mutate.
    return dict(obj)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)


def load_model(
    model_ref: str | Path | Dict[str, Any],
    *,
    dataset: str = "cifar10",
    device: str = "cpu",
    domain: str = "vision",
    training_type: str = "standard",
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    model_kwargs = model_kwargs or {}
    if isinstance(model_ref, (str, Path)):
        ref = str(model_ref)
        path = Path(ref)
        if path.exists():
            return _load_model_from_path(path, device=device, model_kwargs=model_kwargs)
        return ModelFactory.load_model(
            domain=domain,
            model_name=ref,
            training_type=training_type,
            dataset=dataset,
            device=device,
            **model_kwargs,
        )

    if isinstance(model_ref, dict):
        return _load_model_from_dict(model_ref, dataset=dataset, device=device)

    raise TypeError("model_ref must be a path, name, or config dict")


def _load_model_from_dict(
    cfg: Dict[str, Any],
    *,
    dataset: str,
    device: str,
) -> nn.Module:
    cfg = dict(cfg)
    if "path" in cfg or "checkpoint_path" in cfg or "weights_path" in cfg:
        path = cfg.get("path") or cfg.get("checkpoint_path") or cfg.get("weights_path")
        model_kwargs = dict(cfg.get("model_kwargs") or {})
        return _load_model_from_path(Path(path), device=device, model_kwargs=model_kwargs, cfg=cfg)

    if "model_factory" in cfg or "model_name" in cfg or "architecture" in cfg:
        model_name = cfg.get("model_name") or cfg.get("architecture") or cfg.get("name")
        domain = cfg.get("domain", "vision")
        training_type = cfg.get("training_type", "standard")
        model_kwargs = dict(cfg.get("model_kwargs") or {})
        if _looks_like_carmon(None, cfg, model_kwargs):
            default_carmon = Path("models/cifar10/Linf/Carmon2019Unlabeled.pt")
            return _load_carmon2019(default_carmon, device=device, cfg=cfg, model_kwargs=model_kwargs)
        # Pass through common RobustBench args if provided at top-level.
        for extra_key in ("threat_model", "model_dir"):
            if extra_key in cfg and extra_key not in model_kwargs:
                model_kwargs[extra_key] = cfg[extra_key]
        if cfg.get("model_factory"):
            return load_rb_model(
                model_name=model_name,
                dataset=cfg.get("dataset", dataset),
                threat_model=cfg.get("threat_model", "Linf"),
                device=device,
                model_dir=cfg.get("model_dir"),
                model_factory=cfg.get("model_factory"),
                model_kwargs=model_kwargs,
                checkpoint_path=cfg.get("checkpoint_path"),
            )
        return ModelFactory.load_model(
            domain=domain,
            model_name=model_name,
            training_type=training_type,
            dataset=cfg.get("dataset", dataset),
            device=device,
            **model_kwargs,
        )

    raise ValueError("Model config must include path or model_name/architecture.")


def _looks_like_ember_gbdt(
    path: Optional[Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> bool:
    cfg = cfg or {}
    model_kwargs = model_kwargs or {}
    haystacks = [
        str(path or ""),
        str(cfg.get("model_name") or ""),
        str(cfg.get("loader") or ""),
        str(cfg.get("training_type") or ""),
        str(model_kwargs.get("loader") or ""),
    ]
    blob = " ".join(haystacks).lower()
    if any(key in blob for key in ("ember_gbdt", "ember_model_2018", "lightgbm")):
        return True
    suffix = str(Path(path).suffix).lower() if path else ""
    return suffix in {".txt", ".model"} and "ember" in blob


def _looks_like_carmon(
    path: Optional[Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> bool:
    cfg = cfg or {}
    model_kwargs = model_kwargs or {}
    haystacks = [
        str(path or ""),
        str(cfg.get("model_name") or ""),
        str(cfg.get("architecture") or ""),
        str(cfg.get("loader") or ""),
        str(model_kwargs.get("model_name") or ""),
        str(model_kwargs.get("loader") or ""),
    ]
    return any("carmon2019" in str(item).lower() or str(item).lower() in {"carmon", "carmon2019"} for item in haystacks)


def _load_carmon2019(
    path: Path,
    *,
    device: str,
    cfg: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    from neurinspectre.models.wide_resnet_carmon import load_carmon2019_local

    cfg = cfg or {}
    model_kwargs = model_kwargs or {}
    assert_clean = bool(model_kwargs.get("assert_clean_accuracy", cfg.get("assert_clean_accuracy", False)))
    return load_carmon2019_local(str(path), device=device, assert_clean_accuracy=assert_clean)


def _load_model_from_path(
    path: Path,
    *,
    device: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    model_kwargs = model_kwargs or {}
    cfg = cfg or {}
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if _looks_like_ember_gbdt(path, cfg, model_kwargs):
        blob = " ".join(
            str(v).lower()
            for v in [
                path,
                cfg.get("model_name"),
                cfg.get("loader"),
                model_kwargs.get("loader"),
            ]
        )
        if "2024" in blob or "thrember" in blob or "ember2024" in blob:
            from neurinspectre.models.ember_gbdt import EmberGBDT2024

            return EmberGBDT2024.from_file(path)
        from neurinspectre.models.ember_gbdt import EmberGBDT

        return EmberGBDT.from_file(path)

    if _looks_like_carmon(path, cfg, model_kwargs):
        return _load_carmon2019(path, device=device, cfg=cfg, model_kwargs=model_kwargs)

    if path.suffix.lower() == ".onnx":
        return _load_onnx_model(path, device=device)

    try:
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        return model
    except Exception:
        pass

    obj = _load_torch_checkpoint(path)
    if isinstance(obj, nn.Module):
        model = obj.to(device)
        model.eval()
        return model

    if isinstance(obj, dict):
        if isinstance(obj.get("model"), nn.Module):
            model = obj["model"].to(device)
            model.eval()
            return model

        state = obj.get("state_dict") or obj.get("model_state_dict")
        # Support raw state-dict checkpoints saved as an OrderedDict
        # (common torchvision format with keys like "layer1.0.conv1.weight").
        if state is None:
            looks_like_state_dict = bool(obj) and all(
                isinstance(k, str) for k in obj.keys()
            ) and any(torch.is_tensor(v) for v in obj.values())
            if looks_like_state_dict:
                state = obj
        if state is not None:
            model_name = (
                obj.get("model_name")
                or obj.get("architecture")
                or cfg.get("model_name")
                or cfg.get("architecture")
                or model_kwargs.get("model_name")
            )
            domain = obj.get("domain", cfg.get("domain", model_kwargs.get("domain", "vision")))
            training_type = obj.get(
                "training_type",
                cfg.get("training_type", model_kwargs.get("training_type", "standard")),
            )
            dataset = obj.get("dataset", cfg.get("dataset", model_kwargs.get("dataset", "cifar10")))
            if not model_name:
                raise ValueError(
                    "Checkpoint is a state_dict without model_name. "
                    "Provide model_name via config or save a full module."
                )
            factory_kwargs = {
                k: v
                for k, v in model_kwargs.items()
                if k not in {"model_name", "domain", "training_type", "dataset"}
            }
            model = ModelFactory.load_model(
                domain=domain,
                model_name=model_name,
                training_type=training_type,
                dataset=dataset,
                device=device,
                **factory_kwargs,
            )
            model.load_state_dict(state, strict=False)
            model.eval()
            return model

    raise ValueError("Unsupported model checkpoint format.")


def _load_onnx_model(path: Path, *, device: str) -> nn.Module:
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "onnxruntime is required to load .onnx models. Install with: pip install onnxruntime"
        ) from exc

    available = set(ort.get_available_providers())
    providers = ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device in {"mps", "metal"} and "CoreMLExecutionProvider" in available:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    session = None
    session_errors: list[str] = []

    try:
        session = ort.InferenceSession(str(path), providers=providers)
    except Exception as exc:
        session_errors.append(f"default: {exc}")

        # Fallback #1: disable extended optimizations (SimplifiedLayerNormFusion)
        for level, label in (
            (ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, "basic"),
            (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "disable_all"),
        ):
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = level
            sess_options.enable_mem_pattern = False
            sess_options.enable_cpu_mem_arena = False
            sess_options.add_session_config_entry("session.disable_prepacking", "1")
            try:
                session = ort.InferenceSession(
                    str(path),
                    sess_options=sess_options,
                    providers=providers,
                )
                logger.warning(
                    "ONNX Runtime session created with graph optimizations set to %s "
                    "to avoid fusion/runtime incompatibilities.",
                    label,
                )
                break
            except Exception as exc2:
                session_errors.append(f"{label}: {exc2}")

    if session is None:
        error_text = "\n".join(session_errors)
        raise RuntimeError(
            "Failed to initialize ONNX Runtime session. "
            "This is often caused by graph-optimization fusions "
            "(e.g., SimplifiedLayerNormFusion) or FP16 kernels unsupported by CPU. "
            "Try exporting an FP32 model or using a GPU/CoreML provider. "
            f"Details:\n{error_text}"
        )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    class ONNXModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_np = x.detach().cpu().numpy()
            outputs = session.run([output_name], {input_name: x_np})
            return torch.from_numpy(outputs[0]).to(x.device)

    model = ONNXModel()
    model.eval()
    return model


def load_dataset(
    dataset_name: str,
    *,
    data_path: Optional[str] = None,
    labels_path: Optional[str] = None,
    nuscenes_version: Optional[str] = None,
    num_samples: int = 1000,
    batch_size: int = 128,
    seed: int = 42,
    num_workers: int = 4,
    split: str = "test",
    download: bool = True,
    device: Optional[str] = None,
    filter_label: Optional[int] = None,
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
    name = str(dataset_name).lower()
    pin_memory = _should_pin_memory(device)
    effective_workers = int(num_workers)
    if effective_workers > 0:
        # macOS + PyTorch shared-memory workers can fail in sandboxed/managed
        # environments when `torch_shm_manager` is not executable. Prefer a
        # stable default for AE/repro runs.
        if platform.system() == "Darwin":
            effective_workers = 0
        else:
            try:
                torch_dir = Path(torch.__file__).resolve().parent
                shm = torch_dir / "bin" / "torch_shm_manager"
                if shm.exists() and not os.access(shm, os.X_OK):
                    effective_workers = 0
            except Exception:
                pass
    if name in {"cifar10", "imagenet100", "ember"}:
        kwargs: Dict[str, Any] = {
            "n_samples": int(num_samples),
            "seed": int(seed),
            "batch_size": int(batch_size),
            "num_workers": int(effective_workers),
            "split": split,
            "pin_memory": bool(pin_memory),
        }
        if data_path:
            kwargs["root"] = data_path
        # Only CIFAR-10 supports implicit download via torchvision.
        if name == "cifar10":
            kwargs["download"] = bool(download)
        if name == "ember" and filter_label is not None:
            kwargs["filter_label"] = int(filter_label)
        return DatasetFactory.get_dataset(name, **kwargs)
    if name == "nuscenes":
        kwargs = {
            "n_samples": int(num_samples),
            "seed": int(seed),
            "batch_size": int(batch_size),
            "num_workers": int(effective_workers),
            "pin_memory": bool(pin_memory),
            "split": split,
        }
        if data_path:
            kwargs["root"] = data_path
        if labels_path:
            kwargs["labels_path"] = labels_path
        if nuscenes_version:
            kwargs["version"] = nuscenes_version
        return DatasetFactory.get_dataset(name, **kwargs)

    if name == "cifar100":
        return _load_cifar100(
            data_path=data_path,
            num_samples=num_samples,
            seed=seed,
            batch_size=batch_size,
            num_workers=effective_workers,
            split=split,
            download=download,
            pin_memory=pin_memory,
        )
    if name == "imagenet":
        return _load_imagenet(
            data_path=data_path,
            num_samples=num_samples,
            seed=seed,
            batch_size=batch_size,
            num_workers=effective_workers,
            split=split,
            pin_memory=pin_memory,
        )
    if name == "custom":
        return _load_custom_dataset(
            data_path=data_path,
            num_samples=num_samples,
            seed=seed,
            batch_size=batch_size,
            num_workers=effective_workers,
            pin_memory=pin_memory,
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")


def _sample_indices(total: int, num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if num_samples <= 0 or num_samples >= total:
        return np.arange(total)
    return rng.choice(total, size=int(num_samples), replace=False)


def _load_cifar100(
    *,
    data_path: Optional[str],
    num_samples: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    split: str,
    download: bool,
    pin_memory: bool,
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
    root = data_path or "./data/cifar100"
    train = str(split).lower() in {"train", "training"}
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR100(
        root=root,
        train=train,
        download=bool(download),
        transform=transform,
    )
    indices = _sample_indices(len(dataset), num_samples, seed)
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    x_tensor = torch.stack([dataset[i][0] for i in indices])
    y_tensor = torch.tensor([dataset[i][1] for i in indices])
    return loader, x_tensor, y_tensor


def _load_imagenet(
    *,
    data_path: Optional[str],
    num_samples: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    split: str,
    pin_memory: bool,
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
    if not data_path:
        raise ValueError("ImageNet requires --data-path pointing to imagenet root.")
    root = Path(data_path)
    split_dir = "val" if str(split).lower() in {"val", "valid", "validation"} else "train"
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    dataset = torchvision.datasets.ImageFolder(root=str(root / split_dir), transform=transform)
    indices = _sample_indices(len(dataset), num_samples, seed)
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    x_tensor = torch.stack([dataset[i][0] for i in indices])
    y_tensor = torch.tensor([dataset[i][1] for i in indices])
    return loader, x_tensor, y_tensor


def _load_custom_dataset(
    *,
    data_path: Optional[str],
    num_samples: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
    if not data_path:
        raise ValueError("Custom dataset requires --data-path.")
    path = Path(data_path)
    if path.is_dir():
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.ImageFolder(root=str(path), transform=transform)
        indices = _sample_indices(len(dataset), num_samples, seed)
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
        )
        x_tensor = torch.stack([dataset[i][0] for i in indices])
        y_tensor = torch.tensor([dataset[i][1] for i in indices])
        return loader, x_tensor, y_tensor

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        x = torch.tensor(data["x"]).float()
        y = torch.tensor(data["y"]).long()
    elif path.suffix == ".npy":
        raise ValueError("Custom .npy datasets must be packaged as .npz with x/y arrays.")
    elif path.suffix == ".pt":
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict) and "x" in data and "y" in data:
            x = data["x"].float()
            y = data["y"].long()
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            x = data[0].float()
            y = data[1].long()
        else:
            raise ValueError("Unsupported .pt format for custom dataset.")
    else:
        raise ValueError("Custom dataset must be a directory or .npz/.pt file.")

    indices = _sample_indices(len(x), num_samples, seed)
    x = x[indices]
    y = y[indices]
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    return loader, x, y


def build_defense(
    defense_name: str,
    base_model: nn.Module,
    params: Optional[Dict[str, Any]],
    *,
    device: str,
) -> Optional[nn.Module]:
    if defense_name in {None, "", "none", "identity", "id"}:
        return None
    params = params or {}
    key = str(defense_name).lower()
    mapping = {
        "jpeg": "jpeg_compression",
        "jpeg_compression": "jpeg_compression",
        "bitdepth": "bit_depth_reduction",
        "bit_depth": "bit_depth_reduction",
        "bit_depth_reduction": "bit_depth_reduction",
        "randsmooth": "randomized_smoothing",
        "randomized_smoothing": "randomized_smoothing",
        "thermometer": "thermometer_encoding",
        "thermometer_encoding": "thermometer_encoding",
        "distillation": "defensive_distillation",
        "defensive_distillation": "defensive_distillation",
        "ensemble": "ensemble_diversity",
        "ensemble_diversity": "ensemble_diversity",
        "feature_squeezing": "feature_squeezing",
        "gradient_regularization": "gradient_regularization",
        "at_transform": "at_transform",
        "spatial_smoothing": "spatial_smoothing",
        "certified_defense": "certified_defense",
        "random_pad_crop": "random_pad_crop",
        "random_padcrop": "random_pad_crop",
        "random_noise": "random_noise",
        "rl_obfuscation": "rl_obfuscation",
        "tent": "tent",
        "total_variation": "total_variation",
    }
    if key == "custom":
        return _load_custom_defense(base_model, params, device=device)

    defense_key = mapping.get(key)
    if defense_key is None:
        raise ValueError(f"Unknown defense: {defense_name}")
    return DefenseFactory.create_defense(defense_key, base_model, params)


def _load_custom_defense(base_model: nn.Module, params: Dict[str, Any], *, device: str) -> nn.Module:
    module_path = params.get("module_path") or params.get("module")
    class_name = params.get("class_name") or params.get("class")
    if not module_path or not class_name:
        raise ValueError("Custom defense requires module_path and class_name in config.")

    if str(module_path).endswith(".py"):
        path = Path(module_path)
        if not path.exists():
            raise FileNotFoundError(f"Custom defense module not found: {path}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import custom defense from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)

    cls = getattr(module, class_name)
    kwargs = {k: v for k, v in params.items() if k not in {"module_path", "module", "class_name", "class"}}
    try:
        return cls(base_model=base_model, device=device, **kwargs)
    except TypeError:
        return cls(base_model, **kwargs)


def select_target_labels(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(x)
    # Binary-logit models may return shape [B] or [B,1]. In that case the only
    # non-true class is the opposite label.
    if logits.ndim == 1 or (logits.ndim == 2 and int(logits.size(1)) == 1):
        return (1 - y).long()
    logits = logits.clone()
    logits[torch.arange(logits.size(0), device=logits.device), y] = -float("inf")
    return logits.argmax(dim=1)


def evaluate_attack_runner(
    runner: Any,
    eval_model: nn.Module,
    dataloader: DataLoader,
    *,
    num_samples: Optional[int],
    device: str,
    targeted: bool = False,
    save_dir: Optional[str] = None,
    norm: str = "Linf",
    progress_callback: Optional[Any] = None,
    query_budgets: Optional[Any] = None,
) -> Dict[str, Any]:
    eval_model.eval()
    total = 0
    clean_correct = 0
    adv_correct = 0
    success_total = 0
    num_classes: Optional[int] = None
    perturbation_accum: Dict[str, float] = {}
    perturbation_count = 0
    query_counts = []
    iteration_counts = []
    per_sample_queries: list[int] = []
    per_sample_success: list[bool] = []
    batch_index = 0
    last_meta: Dict[str, Any] = {}
    forward_passes = 0
    wall_t0 = time.perf_counter()
    peak_vram_bytes = None
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _count_forward(_mod, _inp, _out):
        nonlocal forward_passes
        forward_passes += 1

    hooks = []
    seen_ids = set()
    for candidate in (eval_model, getattr(runner, "model", None)):
        if candidate is None or id(candidate) in seen_ids:
            continue
        try:
            hooks.append(candidate.register_forward_hook(_count_forward))
            seen_ids.add(id(candidate))
        except Exception:
            continue

    save_path = Path(save_dir) if save_dir else None
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)

    for x, y in dataloader:
        if num_samples is not None and total >= num_samples:
            break

        x = x.to(device)
        y = y.to(device)
        batch_size = int(x.size(0))
        if num_samples is not None:
            remaining = int(num_samples - total)
            if remaining <= 0:
                break
            if batch_size > remaining:
                x = x[:remaining]
                y = y[:remaining]
                batch_size = int(x.size(0))

        def _preds_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, int]:
            # Binary-logit models: threshold at 0 -> {0,1}
            if logits.ndim == 1:
                return (logits > 0).long(), 2
            if logits.ndim == 2 and int(logits.size(1)) == 1:
                return (logits.squeeze(1) > 0).long(), 2
            if logits.ndim < 2:
                # Unrecognized output shape; fall back to a safe sentinel.
                b = int(logits.shape[0]) if logits.ndim >= 1 else 0
                return torch.zeros(b, device=logits.device, dtype=torch.long), 0
            return logits.argmax(dim=1), int(logits.size(1))

        with torch.no_grad():
            clean_logits = eval_model(x)
            clean_preds, inferred_classes = _preds_from_logits(clean_logits)
            if num_classes is None:
                num_classes = int(inferred_classes)
        correct_mask = clean_preds == y
        clean_correct += int(correct_mask.sum().item())
        total += batch_size

        if not correct_mask.any():
            batch_index += 1
            continue

        x_attack = x[correct_mask]
        y_attack = y[correct_mask]

        if targeted:
            if not hasattr(runner, "attack") or not hasattr(runner.attack, "run"):
                raise ValueError("Targeted attack requested but runner does not expose attack.run.")
            if hasattr(runner.attack, "supports_targeted") and not runner.attack.supports_targeted():
                raise ValueError("Selected attack does not support targeted mode.")
            target_labels = select_target_labels(eval_model, x_attack, y_attack)
            result = runner.attack.run(runner.model, x_attack, y_attack, targeted=True, target_labels=target_labels)
        else:
            result = runner.run(x_attack, y_attack)

        x_adv = result.x_adv if hasattr(result, "x_adv") else result
        if hasattr(result, "predictions") and result.predictions is not None:
            preds_adv = result.predictions
        else:
            with torch.no_grad():
                adv_logits = eval_model(x_adv)
                preds_adv, _ = _preds_from_logits(adv_logits)

        success = preds_adv != y_attack
        success_total += int(success.sum().item())
        adv_correct += int((preds_adv == y_attack).sum().item())

        # Query accounting (Tier 3 evidence): SquareAttack and other query-based
        # runners may expose per-sample counts via AttackResult.metadata["queries_used"].
        meta = getattr(result, "metadata", None)
        if isinstance(meta, dict):
            last_meta = dict(meta)
        if isinstance(meta, dict) and meta.get("queries_used") is not None:
            q = meta.get("queries_used")
            try:
                if isinstance(q, np.ndarray):
                    q_list = q.reshape(-1).tolist()
                elif isinstance(q, (list, tuple)):
                    q_list = list(q)
                else:
                    q_list = [q]
                if len(q_list) == int(x_attack.size(0)):
                    per_sample_queries.extend(int(float(v)) for v in q_list)
                    per_sample_success.extend([bool(v) for v in success.detach().cpu().tolist()])
            except Exception:
                # Best-effort: query accounting is optional and should not fail evaluation.
                pass

        pert_metrics = compute_perturbation_metrics(x_attack, x_adv, norm=norm)
        if pert_metrics:
            for k, v in pert_metrics.items():
                perturbation_accum[k] = perturbation_accum.get(k, 0.0) + float(v) * int(x_attack.size(0))
            perturbation_count += int(x_attack.size(0))

        if hasattr(result, "queries") and result.queries is not None:
            query_counts.append(int(result.queries))
        if hasattr(result, "iterations") and result.iterations is not None:
            iteration_counts.append(int(result.iterations))

        if save_path:
            indices = torch.nonzero(correct_mask).flatten().cpu().tolist()
            torch.save(
                {
                    "x_adv": x_adv.detach().cpu(),
                    "y": y_attack.detach().cpu(),
                    "indices": indices,
                },
                save_path / f"batch_{batch_index}.pt",
            )

        if progress_callback:
            try:
                if hasattr(result, "queries") and result.queries is not None:
                    progress_callback(batch_size, queries=int(result.queries))
                else:
                    progress_callback(batch_size)
            except TypeError:
                progress_callback(batch_size)

        batch_index += 1

    for handle in hooks:
        try:
            handle.remove()
        except Exception:
            pass

    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            peak_vram_bytes = int(torch.cuda.max_memory_allocated())
        except Exception:
            peak_vram_bytes = None

    clean_accuracy = clean_correct / total if total > 0 else 0.0
    robust_accuracy = adv_correct / total if total > 0 else 0.0
    attack_success_rate = success_total / clean_correct if clean_correct > 0 else 0.0
    # Some papers/reporting pipelines define "ASR" over *all* samples (not only the
    # clean-correct subset). We export both to make the definition audit-able.
    attack_success_rate_overall = success_total / total if total > 0 else 0.0
    # Overall misclassification rate after attack (includes samples already wrong at clean time).
    compromise_rate = 1.0 - float(robust_accuracy)

    perturbation_summary = {}
    if perturbation_count > 0:
        perturbation_summary = {k: v / perturbation_count for k, v in perturbation_accum.items()}

    query_summary = {}
    queries_mean = None
    iterations_mean = None
    if per_sample_queries and len(per_sample_queries) == len(per_sample_success):
        try:
            eff = compute_query_efficiency(per_sample_queries, torch.tensor(per_sample_success, dtype=torch.bool))
            query_summary = dict(eff)
            if "mean_queries" in eff:
                queries_mean = float(eff["mean_queries"])
        except Exception:
            pass
    elif query_counts:
        # Fallback: some runners only expose scalar query counts per batch.
        query_arr = np.array(query_counts, dtype=float)
        queries_mean = float(query_arr.mean())
        query_summary = {
            "total_queries": float(query_arr.sum()),
            "mean_queries": queries_mean,
            "median_queries": float(np.median(query_arr)),
        }
    if iteration_counts:
        iter_arr = np.array(iteration_counts, dtype=float)
        iterations_mean = float(iter_arr.mean())
        query_summary["mean_iterations"] = iterations_mean

    cost = {
        "wall_time_s": float(time.perf_counter() - wall_t0),
        "forward_passes": int(forward_passes),
        "queries": queries_mean,
        "peak_vram_bytes": peak_vram_bytes,
    }
    out = {
        "clean_accuracy": float(clean_accuracy),
        "robust_accuracy": float(robust_accuracy),
        "compromise_rate": float(compromise_rate),
        "attack_success_rate": float(attack_success_rate),
        "attack_success_rate_conditional": float(attack_success_rate),
        "attack_success_rate_overall": float(attack_success_rate_overall),
        "samples": int(total),
        "correct_samples": int(clean_correct),
        "success_samples": int(success_total),
        "num_classes": int(num_classes or 0),
        "perturbation": perturbation_summary,
        "query_efficiency": query_summary,
        "queries": queries_mean,
        "iterations": iterations_mean,
        "cost": cost,
    }
    if last_meta:
        out["attack_metadata"] = last_meta
        for key in (
            "aa_subattacks_skipped",
            "aa_skip_reasons",
            "gradient_unavailable",
            "aa_all_gradient_skipped",
            "chosen_attack",
            "selected_attack_impl",
            "backend",
            "access",
            "query_floor_relaxed",
            "realizable",
            "space",
        ):
            if key in last_meta:
                out[key] = last_meta[key]
    budgets = query_budgets
    if budgets is None:
        budgets = [100, 500, 2000, 5000]
    if per_sample_queries and per_sample_success:
        curve = compute_asr_query_curve(per_sample_queries, per_sample_success, list(budgets))
        if curve:
            out["query_curve"] = curve
    for key in ("chosen_attack", "selected_attack_impl"):
        if key not in out and getattr(runner, key, None):
            out[key] = getattr(runner, key)
    if "backend" not in out and hasattr(runner, "_backend"):
        out["backend"] = runner._backend
    return out
