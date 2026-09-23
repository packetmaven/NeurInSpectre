"""First-class security pipeline: ingest → sanitize → features → model → threshold.

Grosse et al. (USENIX Sec '24): industry models sit in pipelines, not isolation.
NeurInSpectre characterizes each stage and routes the attack at the stage that
actually obfuscates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class PipelineStage:
    name: str
    kind: str
    requires_bpda: bool = False
    requires_eot: bool = False
    requires_problem_space: bool = False
    obfuscation_types: List[str] = field(default_factory=list)
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    approx_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "requires_bpda": bool(self.requires_bpda),
            "requires_eot": bool(self.requires_eot),
            "requires_problem_space": bool(self.requires_problem_space),
            "obfuscation_types": list(self.obfuscation_types),
            "params": dict(self.params),
        }


class SecurityPipeline(nn.Module):
    """Ordered pipeline that still looks like a DefenseWrapper to existing attacks."""

    VALID_KINDS = ("ingest", "sanitize", "features", "classifier", "threshold")

    def __init__(
        self,
        classifier: nn.Module,
        stages: Optional[Sequence[PipelineStage]] = None,
        *,
        device: str = "cpu",
        name: str = "pipeline",
    ):
        super().__init__()
        self.classifier = classifier
        self.base_model = classifier
        self.device = device
        self.name = name
        self.stages: List[PipelineStage] = list(stages or [])
        if not any(s.kind == "classifier" for s in self.stages):
            self.stages.append(PipelineStage(name="classifier", kind="classifier"))

    def preprocess_stages(self) -> List[PipelineStage]:
        return [s for s in self.stages if s.kind != "classifier"]

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for stage in self.preprocess_stages():
            if stage.transform is None:
                continue
            out = stage.transform(out)
        return out

    def get_bpda_approximation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        approxs = [s.approx_fn for s in self.preprocess_stages() if s.approx_fn is not None]
        if not approxs:
            return lambda x: x

        def _compose(x: torch.Tensor) -> torch.Tensor:
            out = x
            for fn in approxs:
                out = fn(out)
            return out

        return _compose

    def forward(self, x: torch.Tensor, use_approximation: bool = False) -> torch.Tensor:
        if use_approximation:
            with torch.no_grad():
                x_actual = self.transform(x)
            x_approx = self.get_bpda_approximation()(x)
            x_defended = x_actual + (x_approx - x_approx.detach())
        else:
            x_defended = self.transform(x)
        return self.classifier(x_defended)

    def attack_surface(self) -> List[PipelineStage]:
        return [
            s
            for s in self.stages
            if s.requires_bpda or s.requires_eot or s.requires_problem_space
        ]

    def characterize(self) -> Dict[str, Any]:
        surface = self.attack_surface()
        recommended = "apgd"
        if any(s.requires_problem_space for s in surface):
            recommended = "problem_space"
        elif any(s.requires_bpda for s in surface) and any(s.requires_eot for s in surface):
            recommended = "hybrid"
        elif any(s.requires_bpda for s in surface):
            recommended = "bpda"
        elif any(s.requires_eot for s in surface):
            recommended = "eot"
        return {
            "name": self.name,
            "n_stages": len(self.stages),
            "stages": [s.to_dict() for s in self.stages],
            "attack_surface": [s.name for s in surface],
            "requires_bpda": any(s.requires_bpda for s in self.stages),
            "requires_eot": any(s.requires_eot for s in self.stages),
            "requires_problem_space": any(s.requires_problem_space for s in self.stages),
            "recommended_recipe": recommended,
        }

    @classmethod
    def from_defense(cls, defense, *, device: str = "cpu", name: Optional[str] = None) -> "SecurityPipeline":
        if defense is None:
            raise ValueError("from_defense requires a defense or model")
        if isinstance(defense, SecurityPipeline):
            return defense
        classifier = getattr(defense, "base_model", None) or getattr(defense, "model", None) or defense
        spec = getattr(defense, "spec", None)
        spec_name = getattr(spec, "name", None) or type(defense).__name__
        stages: List[PipelineStage] = []
        if hasattr(defense, "transform") and hasattr(defense, "get_bpda_approximation"):
            obf = getattr(defense, "obfuscation_types", []) or (getattr(spec, "obfuscation_types", []) if spec else [])
            obf_vals = [o.value if hasattr(o, "value") else str(o) for o in obf]
            requires_bpda = bool(getattr(defense, "requires_bpda", False))
            requires_eot = bool(getattr(defense, "requires_eot", False))
            kind = "ingest" if "jpeg" in str(spec_name).lower() else "sanitize"
            trivial = (
                not obf_vals
                and not requires_bpda
                and not requires_eot
                and str(spec_name).lower() in {"identitydefenseadapter", "none", "identity"}
            )
            if not trivial:
                stages.append(
                    PipelineStage(
                        name=str(spec_name),
                        kind=kind,
                        requires_bpda=requires_bpda,
                        requires_eot=requires_eot,
                        obfuscation_types=obf_vals,
                        transform=defense.transform,
                        approx_fn=defense.get_bpda_approximation(),
                        params=dict(getattr(spec, "params", {}) or {}),
                    )
                )
        stages.append(PipelineStage(name="classifier", kind="classifier"))
        return cls(classifier, stages, device=device, name=name or str(spec_name))

    @classmethod
    def identity(cls, classifier: nn.Module, *, device: str = "cpu") -> "SecurityPipeline":
        return cls(classifier, [PipelineStage(name="classifier", kind="classifier")], device=device, name="identity")

    @classmethod
    def from_ember_gbdt(cls, classifier: Optional[nn.Module] = None, *, device: str = "cpu") -> "SecurityPipeline":
        dummy = classifier if classifier is not None else nn.Identity()
        stages = [
            PipelineStage(
                name="pe_ingest",
                kind="ingest",
                requires_problem_space=True,
                params={"note": "EMBER2018 release has no PE binaries"},
            ),
            PipelineStage(
                name="ember_features",
                kind="features",
                requires_problem_space=True,
                params={"dim": 2381, "extractor": "PEFeatureExtractor(2)"},
            ),
            PipelineStage(name="ember_gbdt", kind="classifier"),
            PipelineStage(name="malware_threshold", kind="threshold", params={"threshold": 0.5, "positive_class": 1}),
        ]
        return cls(dummy, stages, device=device, name="ember_gbdt")

    @classmethod
    def from_ember2024_gbdt(
        cls,
        classifier: Optional[nn.Module] = None,
        *,
        device: str = "cpu",
        variant: str = "pe",
    ) -> "SecurityPipeline":
        """Build the EMBER 2024 pipeline.

        ``variant`` selects which classifier stage the pipeline advertises:
        ``"pe"`` (combined Win32 + Win64 + .NET), ``"win32"``, or ``"win64"``.
        All variants share the same v3 feature extractor (thrember, pefile).
        """
        variant = str(variant or "pe").lower()
        classifier_name = {
            "pe": "ember2024_gbdt",
            "win32": "ember2024_win32_gbdt",
            "win64": "ember2024_win64_gbdt",
            "apk": "ember2024_apk_gbdt",
            "elf": "ember2024_elf_gbdt",
            "pdf": "ember2024_pdf_gbdt",
            "dotnet": "ember2024_dotnet_gbdt",
            "all": "ember2024_all_gbdt",
        }.get(variant, "ember2024_gbdt")
        model_filename = {
            "pe": "EMBER2024_PE.model",
            "win32": "EMBER2024_Win32.model",
            "win64": "EMBER2024_Win64.model",
            "apk": "EMBER2024_APK.model",
            "elf": "EMBER2024_ELF.model",
            "pdf": "EMBER2024_PDF.model",
            "dotnet": "EMBER2024_Dot_Net.model",
            "all": "EMBER2024_all.model",
        }.get(variant, "EMBER2024_PE.model")

        dummy = classifier if classifier is not None else nn.Identity()
        stages = [
            PipelineStage(
                name="pe_ingest",
                kind="ingest",
                requires_problem_space=True,
                params={
                    "note": "EMBER2024 public split has no PE binaries; VT API required",
                    "extractor_library": "pefile",
                },
            ),
            PipelineStage(
                name="ember2024_features",
                kind="features",
                requires_problem_space=True,
                params={
                    "feature_version": 3,
                    "dim": 2568,
                    "extractor": "thrember.features.PEFeatureExtractor",
                },
            ),
            PipelineStage(
                name=classifier_name,
                kind="classifier",
                params={"variant": variant, "model_filename": model_filename},
            ),
            PipelineStage(
                name="malware_threshold",
                kind="threshold",
                params={"threshold": 0.5, "positive_class": 1},
            ),
        ]
        return cls(dummy, stages, device=device, name=classifier_name)
