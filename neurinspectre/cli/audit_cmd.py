"""
``neurinspectre audit`` — defense linter for a security pipeline.

Whitebox: official AutoAttack, AA+BPDA, NeurInSpectre, cheap PGD.
Practical modes: Square on scores or hard labels with ASR-vs-query curves.
Also records pipeline characterization and an optional PE parse stub.

Default targets:
  - carmon: Carmon2019Unlabeled with no preprocessor
  - jpeg-carmon: same backbone + real PIL JPEG q=75
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml

from .evaluate_cmd import run_evaluation
from .utils import save_json
from ..evaluation.problem_space import evaluate_pe_parse
from ..pipelines import SecurityPipeline

logger = logging.getLogger(__name__)

DEFAULT_CARMON_PATH = "models/cifar10/Linf/Carmon2019Unlabeled.pt"
DEFAULT_EMBER_GBDT_PATH = "data/ember/ember2018/ember_model_2018.txt"
DEFAULT_EMBER2024_GBDT_PATH = "data/ember/ember2024/EMBER2024_PE.model"
DEFAULT_EMBER2024_WIN32_PATH = "data/ember/ember2024/EMBER2024_Win32.model"
DEFAULT_EMBER2024_WIN64_PATH = "data/ember/ember2024/EMBER2024_Win64.model"
DEFAULT_EMBER2024_APK_PATH = "data/ember/ember2024/EMBER2024_APK.model"
DEFAULT_EMBER2024_ELF_PATH = "data/ember/ember2024/EMBER2024_ELF.model"
DEFAULT_EMBER2024_PDF_PATH = "data/ember/ember2024/EMBER2024_PDF.model"
DEFAULT_EMBER2024_DOTNET_PATH = "data/ember/ember2024/EMBER2024_Dot_Net.model"
DEFAULT_EMBER2024_ALL_PATH = "data/ember/ember2024/EMBER2024_all.model"
DEFAULT_EMBER_DATA = "./data/ember"
ROBUSTBENCH_CARMON_ROBUST_ACC = 0.5953
CIFAR10_LINF_EPS = 8 / 255
EMBER_FEATURE_EPS = 1.0

AUDIT_TARGETS = {
    "carmon": {
        "name": "cm_carmon2019",
        "type": "none",
        "params": {},
        "notes": [
            "Identity / no preprocessor on Carmon2019Unlabeled.",
            f"RobustBench official AutoAttack robust accuracy is {ROBUSTBENCH_CARMON_ROBUST_ACC:.2%} at n=10000.",
            "NeurInSpectre must not 'win' here via a larger budget than official AA.",
        ],
    },
    "jpeg-carmon": {
        "name": "cm_carmon2019_jpeg",
        "type": "jpeg_compression",
        "params": {"quality": 75},
        "notes": [
            "Same Carmon2019 backbone with real PIL JPEG q=75.",
            "aa_official attacks the non-differentiable defended model.",
            "aa_bpda uses true JPEG forward and the defense BPDA backward (identity at q>=75).",
        ],
    },
    "ember-gbdt": {
        "name": "md_ember2018_gbdt",
        "type": "none",
        "params": {},
        "notes": [
            "Official Elastic EMBER 2018 LightGBM (ember_model_2018.txt), not the Table 2 MLP.",
            "EMBER2018 public release is feature vectors; it does not include PE binaries.",
            "Same-sample table requires --pe-sample (file or directory of PE bytes).",
            "Feature-space FeatureSquare is unrealizable. Problem-space is Full DOS + padding.",
            "GAMMA-padding is used only if --benign-corpus is set. This is not GAMMA section injection.",
            "Evasion threat model: GBDT-detected malware → benign, PE-valid only.",
        ],
    },
    "ember2024-gbdt": {
        "name": "md_ember2024_pe_gbdt",
        "type": "none",
        "params": {},
        "notes": [
            "EMBER2024 LightGBM PE detector (thrember, FutureComputing4AI, KDD 2025).",
            "Feature version 3 uses pefile (Python-only) instead of LIEF 0.9.0.",
            "Extraction is OS-agnostic; no lief version pinning drama. Cross-platform reproducible.",
            "Default model: EMBER2024_PE.model (Win32 + Win64 + .NET). Feature dim = 2568.",
            "Same-sample table requires --pe-sample; same problem-space transforms as EMBER2018.",
            "Feature-space FeatureSquare is still unrealizable on mixed-scale features.",
            "EMBER2024 dataset also has no PE binaries; VirusTotal API required for the public split.",
        ],
    },
    "ember2024-win32-gbdt": {
        "name": "md_ember2024_win32_gbdt",
        "type": "none",
        "params": {},
        "notes": [
            "EMBER2024 LightGBM Win32-only detector (thrember, feature version 3, dim 2568).",
            "Specialist model; expect lower detection rate than the combined PE model on Win64-heavy corpora.",
        ],
    },
    "ember2024-win64-gbdt": {
        "name": "md_ember2024_win64_gbdt",
        "type": "none",
        "params": {},
        "notes": [
            "EMBER2024 LightGBM Win64-only detector (thrember, feature version 3, dim 2568).",
            "Specialist model; on the reference corpus outperforms Win32 (144/148 vs 98/148).",
        ],
    },
    "ember2024-apk-gbdt": {
        "name": "md_ember2024_apk_gbdt",
        "type": "none",
        "params": {},
        "notes": [
            "EMBER2024 LightGBM Android APK detector (thrember, feature version 3, dim 2568).",
            "Non-PE target: file loader accepts ZIP magic (PK\\x03\\x04). No Full DOS / overlay column.",
        ],
    },
    "ember2024-elf-gbdt": {
        "name": "md_ember2024_elf_gbdt",
        "type": "none",
        "params": {},
        "notes": [
            "EMBER2024 LightGBM ELF detector (thrember, feature version 3, dim 2568).",
            "Non-PE target: file loader accepts ELF magic (\\x7fELF). No Full DOS / overlay column.",
        ],
    },
    "ember2024-pdf-gbdt": {
        "name": "md_ember2024_pdf_gbdt",
        "type": "none",
        "params": {},
        "notes": [
            "EMBER2024 LightGBM PDF detector (thrember, feature version 3, dim 2568).",
            "Non-PE target: file loader accepts PDF magic (%PDF-). No Full DOS / overlay column.",
        ],
    },
    "ember2024-dotnet-gbdt": {
        "name": "md_ember2024_dotnet_gbdt",
        "type": "none",
        "params": {},
        "notes": [
            "EMBER2024 LightGBM .NET assembly detector (thrember, feature version 3, dim 2568).",
            ".NET assemblies are PE files (MZ header + CLR); routed through the PE evaluator so",
            "Full DOS + overlay/padding work in addition to feature-space FeatureSquare.",
        ],
    },
    "ember2024-all-gbdt": {
        "name": "md_ember2024_all_gbdt",
        "type": "none",
        "params": {},
        "notes": [
            "EMBER2024 LightGBM universal cross-format detector (thrember, feature version 3, dim 2568).",
            "Trained on all six file types (Win32/Win64/.NET/APK/ELF/PDF). Feature-space only lane:",
            "PE-specific problem-space transforms do not apply to non-PE inputs.",
        ],
    },
}


EMBER2024_PE_TARGETS = {
    "ember2024-gbdt", "ember2024-win32-gbdt", "ember2024-win64-gbdt",
    "ember2024-dotnet-gbdt",
}
EMBER2024_NONPE_TARGETS = {
    "ember2024-apk-gbdt", "ember2024-elf-gbdt", "ember2024-pdf-gbdt",
    "ember2024-all-gbdt",
}
EMBER2024_TARGETS = EMBER2024_PE_TARGETS | EMBER2024_NONPE_TARGETS
EMBER_TARGETS = {"ember-gbdt", *EMBER2024_TARGETS}
_EMBER_KEY_ALIASES = {
    "ember2024": "ember2024-gbdt",
    "ember-2024": "ember2024-gbdt",
    "ember_2024": "ember2024-gbdt",
    "ember2024-gbdt": "ember2024-gbdt",
    "ember2024-pe-gbdt": "ember2024-gbdt",
    "ember2024-pe": "ember2024-gbdt",
    "ember2024-win32": "ember2024-win32-gbdt",
    "ember2024-win32-gbdt": "ember2024-win32-gbdt",
    "ember2024-win64": "ember2024-win64-gbdt",
    "ember2024-win64-gbdt": "ember2024-win64-gbdt",
    "ember2024-apk": "ember2024-apk-gbdt",
    "ember2024-apk-gbdt": "ember2024-apk-gbdt",
    "ember2024-elf": "ember2024-elf-gbdt",
    "ember2024-elf-gbdt": "ember2024-elf-gbdt",
    "ember2024-pdf": "ember2024-pdf-gbdt",
    "ember2024-pdf-gbdt": "ember2024-pdf-gbdt",
    "ember2024-dotnet": "ember2024-dotnet-gbdt",
    "ember2024-dotnet-gbdt": "ember2024-dotnet-gbdt",
    "ember2024-dot-net": "ember2024-dotnet-gbdt",
    "ember2024-dot-net-gbdt": "ember2024-dotnet-gbdt",
    "ember2024-all": "ember2024-all-gbdt",
    "ember2024-all-gbdt": "ember2024-all-gbdt",
    "ember-gbdt": "ember-gbdt",
    "ember2018": "ember-gbdt",
    "ember-2018": "ember-gbdt",
}


_EMBER2024_TARGET_TO_FAMILY = {
    "ember2024-apk-gbdt": "APK",
    "ember2024-elf-gbdt": "ELF",
    "ember2024-pdf-gbdt": "PDF",
    "ember2024-all-gbdt": "ANY",  # universal detector; accept any bytes
}


def _is_ember2024_nonpe_target(target: str) -> bool:
    return _normalize_target(target) in EMBER2024_NONPE_TARGETS


def _normalize_target(target: str) -> str:
    key = str(target or "").lower().replace("_", "-").strip()
    return _EMBER_KEY_ALIASES.get(key, key)


def _is_ember_target(target: str) -> bool:
    return _normalize_target(target) in EMBER_TARGETS


def _is_ember2024_target(target: str) -> bool:
    return _normalize_target(target) in EMBER2024_TARGETS


def _ember2024_default_model_path(target: str) -> str:
    key = _normalize_target(target)
    return {
        "ember2024-gbdt": DEFAULT_EMBER2024_GBDT_PATH,
        "ember2024-win32-gbdt": DEFAULT_EMBER2024_WIN32_PATH,
        "ember2024-win64-gbdt": DEFAULT_EMBER2024_WIN64_PATH,
        "ember2024-apk-gbdt": DEFAULT_EMBER2024_APK_PATH,
        "ember2024-elf-gbdt": DEFAULT_EMBER2024_ELF_PATH,
        "ember2024-pdf-gbdt": DEFAULT_EMBER2024_PDF_PATH,
        "ember2024-dotnet-gbdt": DEFAULT_EMBER2024_DOTNET_PATH,
        "ember2024-all-gbdt": DEFAULT_EMBER2024_ALL_PATH,
    }.get(key, DEFAULT_EMBER2024_GBDT_PATH)


def default_carmon_model(
    *,
    model_path: Optional[str] = None,
    assert_clean_accuracy: bool = False,
) -> Dict[str, Any]:
    return {
        "path": str(model_path or DEFAULT_CARMON_PATH),
        "model_name": "Carmon2019Unlabeled",
        "training_type": "robustbench",
        "loader": "carmon2019",
        "dataset": "cifar10",
        "assert_clean_accuracy": bool(assert_clean_accuracy),
    }


DEFAULT_QUERY_BUDGETS = [10, 50, 100, 500, 5000]
SMOKE_QUERY_BUDGETS = [10, 25, 50]


def parse_audit_modes(mode: Optional[str], *, smoke: bool) -> List[str]:
    raw = str(mode or ("all" if smoke else "whitebox")).lower().replace("_", "-")
    if raw == "all":
        return ["whitebox", "scores", "labels"]
    if raw in {"whitebox", "scores", "labels", "feature", "problem"}:
        return [raw]
    raise click.ClickException(
        f"Unknown audit mode {mode!r}. Choose one of: whitebox, scores, labels, feature, problem, all"
    )


def parse_query_budgets(raw: Any, *, smoke: bool) -> List[int]:
    if raw is None:
        return list(SMOKE_QUERY_BUDGETS if smoke else DEFAULT_QUERY_BUDGETS)
    if isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = [part.strip() for part in str(raw).split(",") if part.strip()]
    budgets: List[int] = []
    for item in values:
        try:
            q = int(item)
        except (TypeError, ValueError) as exc:
            raise click.ClickException(f"Invalid query budget {item!r}") from exc
        if q > 0:
            budgets.append(q)
    if not budgets:
        raise click.ClickException("At least one positive --query-budgets value is required")
    return budgets


def _practical_square(
    *,
    name: str,
    attack_type: str,
    loss_type: str,
    smoke: bool,
    budgets: List[int],
) -> Dict[str, Any]:
    n_queries = max(budgets) if budgets else (50 if smoke else 5000)
    return {
        "name": name,
        "type": attack_type,
        "n_queries": int(n_queries),
        "allow_short_budget": bool(n_queries < 1000),
        "loss_type": loss_type,
        "access": attack_type,
        "query_budgets": list(budgets),
    }


def build_audit_attacks(
    *,
    smoke: bool,
    include_pgd: bool = True,
    modes: Optional[List[str]] = None,
    query_budgets: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    selected = list(modes or parse_audit_modes(None, smoke=smoke))
    budgets = list(query_budgets or parse_query_budgets(None, smoke=smoke))
    attacks: List[Dict[str, Any]] = []
    if "whitebox" in selected:
        if smoke:
            official = {
                "name": "aa_official",
                "type": "aa_official",
                "version": "custom",
                "attacks_to_run": ["apgd-ce"],
            }
            bpda = {
                "name": "aa_bpda",
                "type": "aa_bpda",
                "version": "custom",
                "attacks_to_run": ["apgd-ce"],
            }
            ni = {"name": "neurinspectre", "type": "neurinspectre", "characterization_samples": 8, "n_iterations": 10}
            pgd = {"name": "pgd", "type": "pgd", "steps": 5, "n_iterations": 5}
        else:
            official = {"name": "aa_official", "type": "aa_official", "version": "standard"}
            bpda = {"name": "aa_bpda", "type": "aa_bpda", "version": "standard"}
            ni = {"name": "neurinspectre", "type": "neurinspectre", "characterization_samples": 50}
            pgd = {"name": "pgd", "type": "pgd", "steps": 20, "n_iterations": 20}
        attacks.extend([official, bpda, ni])
        if include_pgd:
            attacks.append(pgd)
    if "scores" in selected:
        attacks.append(
            _practical_square(
                name="scores",
                attack_type="scores",
                loss_type="margin",
                smoke=smoke,
                budgets=budgets,
            )
        )
    if "labels" in selected:
        attacks.append(
            _practical_square(
                name="labels",
                attack_type="labels",
                loss_type="label",
                smoke=smoke,
                budgets=budgets,
            )
        )
    return attacks


def build_audit_config(
    *,
    target: str,
    n_examples: int,
    smoke: bool,
    model_path: Optional[str] = None,
    data_root: str = "./data/cifar10",
    batch_size: int = 16,
    seed: int = 42,
    assert_clean_accuracy: bool = False,
    include_pgd: bool = True,
    mode: Optional[str] = None,
    query_budgets: Optional[Any] = None,
    pe_sample: Optional[str] = None,
    benign_corpus: Optional[str] = None,
) -> Dict[str, Any]:
    key = _normalize_target(target)
    if key not in AUDIT_TARGETS:
        raise click.ClickException(
            f"Unknown audit target {target!r}. Choose one of: {', '.join(sorted(AUDIT_TARGETS))}"
        )
    spec = AUDIT_TARGETS[key]
    modes = parse_audit_modes(mode, smoke=smoke)
    budgets = parse_query_budgets(query_budgets, smoke=smoke)
    if _is_ember_target(key):
        default_path = (
            _ember2024_default_model_path(key)
            if _is_ember2024_target(key)
            else DEFAULT_EMBER_GBDT_PATH
        )
        return _build_ember_audit_config(
            spec=spec,
            target_key=key,
            n_examples=n_examples,
            smoke=smoke,
            model_path=model_path or default_path,
            data_root=data_root if data_root != "./data/cifar10" else DEFAULT_EMBER_DATA,
            batch_size=batch_size,
            seed=seed,
            include_pgd=include_pgd,
            modes=modes,
            budgets=budgets,
            pe_sample=pe_sample,
            benign_corpus=benign_corpus,
        )
    return {
        "seed": int(seed),
        "attack_batch_size": int(batch_size),
        "iterations": 10 if smoke else 100,
        "perturbation": {"epsilon": CIFAR10_LINF_EPS, "norm": "Linf"},
        "validity_gates": {"enabled": True, "strict": False},
        "query_budgets": list(budgets),
        "datasets": {
            "cifar10": {
                "path": data_root,
                "split": "test",
                "batch_size": int(batch_size),
                "num_workers": 0,
                "num_samples": int(n_examples),
            }
        },
        "defenses": [
            {
                "name": spec["name"],
                "type": spec["type"],
                "dataset": "cifar10",
                "model": default_carmon_model(
                    model_path=model_path,
                    assert_clean_accuracy=assert_clean_accuracy,
                ),
                "params": dict(spec["params"]),
            }
        ],
        "attacks": build_audit_attacks(
            smoke=smoke,
            include_pgd=include_pgd,
            modes=modes,
            query_budgets=budgets,
        ),
        "audit": {
            "target": key,
            "notes": list(spec["notes"]),
            "robustbench_carmon_robust_acc": ROBUSTBENCH_CARMON_ROBUST_ACC,
            "smoke": bool(smoke),
            "n_examples": int(n_examples),
            "modes": modes,
            "query_budgets": list(budgets),
            "pe_sample": pe_sample,
        },
    }


def _build_ember_audit_config(
    *,
    spec: Dict[str, Any],
    target_key: str,
    n_examples: int,
    smoke: bool,
    model_path: Optional[str],
    data_root: str,
    batch_size: int,
    seed: int,
    include_pgd: bool,
    modes: List[str],
    budgets: List[int],
    pe_sample: Optional[str],
    benign_corpus: Optional[str] = None,
) -> Dict[str, Any]:
    is_2024 = target_key in EMBER2024_TARGETS
    loader = "ember2024_gbdt" if is_2024 else "ember_gbdt"
    default_model = _ember2024_default_model_path(target_key) if is_2024 else DEFAULT_EMBER_GBDT_PATH
    n_queries = max(budgets) if budgets else (50 if smoke else 5000)
    attacks: List[Dict[str, Any]] = []
    # Same-sample PE evaluation owns both columns; do not mix in EMBER memmap vectors.
    # The on-disk memmap is 2381-d EMBER 2018 features; running FeatureSquare against
    # the 2024 booster (dim 2568) without --pe-sample would shape-mismatch. Refuse the
    # attack column in that case; the CLI later prints a hint pointing to --pe-sample.
    want_feature = (not pe_sample) and (not is_2024) and any(
        m in {"all", "scores", "labels", "feature", "whitebox"} for m in modes
    )
    if want_feature:
        attacks.append(
            {
                "name": "feature_square",
                "type": "feature_square",
                "n_queries": int(n_queries),
                "allow_short_budget": bool(n_queries < 1000),
                "loss_type": "margin",
                "epsilon": EMBER_FEATURE_EPS,
                "query_budgets": list(budgets),
            }
        )
        if include_pgd:
            attacks.append(
                {
                    "name": "pgd",
                    "type": "pgd",
                    "steps": 5 if smoke else 20,
                    "n_iterations": 5 if smoke else 20,
                    "epsilon": EMBER_FEATURE_EPS,
                }
            )
    return {
        "seed": int(seed),
        "attack_batch_size": int(batch_size),
        "iterations": 10 if smoke else 100,
        "perturbation": {"epsilon": EMBER_FEATURE_EPS, "norm": "Linf"},
        "validity_gates": {"enabled": True, "strict": False, "min_correct_samples": 1},
        "query_budgets": list(budgets),
        "datasets": {
            "ember": {
                "path": data_root,
                "split": "test",
                "batch_size": int(batch_size),
                "num_workers": 0,
                "num_samples": int(n_examples),
                "filter_label": 1,
            }
        },
        "defenses": [
            {
                "name": spec["name"],
                "type": spec["type"],
                "dataset": "ember",
                "model": {
                    "path": str(model_path or default_model),
                    "model_name": loader,
                    "loader": loader,
                    "dataset": "ember",
                    "domain": "malware_detection",
                    "ember_feature_version": 3 if is_2024 else 2,
                },
                "params": dict(spec["params"]),
            }
        ],
        "attacks": attacks,
        "audit": {
            "target": target_key,
            "notes": list(spec["notes"]),
            "smoke": bool(smoke),
            "n_examples": int(n_examples),
            "modes": modes,
            "query_budgets": list(budgets),
            "pe_sample": pe_sample,
            "benign_corpus": benign_corpus,
            "feature_epsilon": EMBER_FEATURE_EPS,
            "threat_model": "malware_evasion",
            "same_sample": bool(pe_sample),
            "ember_feature_version": 3 if is_2024 else 2,
            "pipeline": characterize_audit_pipeline(target_key),
            "measurement_scope": _measurement_scope_for_target(target_key),
        },
    }


def _measurement_scope_for_target(target_key: str) -> Dict[str, Any]:
    from ..malware.measurement_scope import build_measurement_scope

    return build_measurement_scope(target_key)


def _first_result(summary: Dict[str, Any]) -> Dict[str, Any]:
    results = summary.get("results") or []
    if not results:
        return {}
    return results[0] if isinstance(results[0], dict) else {}


def build_audit_report(summary: Dict[str, Any], *, config: Dict[str, Any]) -> Dict[str, Any]:
    audit_meta = dict(config.get("audit") or {})
    row = _first_result(summary)
    attacks = dict(row.get("attacks") or {})
    characterization = dict(row.get("characterization") or {})
    clean_acc = None
    validity = None
    for metrics in attacks.values():
        if isinstance(metrics, dict):
            if clean_acc is None and "clean_accuracy" in metrics:
                clean_acc = metrics.get("clean_accuracy")
            if validity is None and isinstance(metrics.get("validity"), dict):
                validity = metrics.get("validity")
    recommended = characterization.get("chosen_attack")
    if not recommended:
        ni = attacks.get("neurinspectre") or {}
        recommended = ni.get("chosen_attack")
    pipeline = audit_meta.get("pipeline") or characterize_audit_pipeline(str(audit_meta.get("target") or ""))
    if not recommended and isinstance(pipeline, dict):
        recommended = pipeline.get("recommended_recipe")
    problem_space = audit_meta.get("problem_space")
    if problem_space is None:
        problem_space = evaluate_pe_parse(audit_meta.get("pe_sample"))
    same = audit_meta.get("same_sample_result")
    same_detail = None
    if isinstance(same, dict) and isinstance(same.get("feature_vs_problem_space"), dict):
        comparison = dict(same["feature_vs_problem_space"])
        comparison.setdefault("same_sample", True)
        if isinstance(same.get("feature_space"), dict):
            attacks.setdefault("feature_square", same["feature_space"])
        if isinstance(same.get("problem_space"), dict):
            problem_space = same["problem_space"]
        same_detail = {
            "n_listed": same.get("n_listed"),
            "n_scanned": same.get("n_scanned"),
            "n_extracted": same.get("n_extracted"),
            "n_detected_malware": same.get("n_detected_malware"),
            "n_skipped": same.get("n_skipped"),
            "stopped_early": same.get("stopped_early"),
            "skip_reasons": same.get("skip_reasons"),
            "extractor": same.get("extractor"),
            "samples": same.get("samples"),
            # A1 — Capa-informed tag filter provenance
            "tag_filter": same.get("tag_filter"),
            "tag_filter_active": bool(same.get("tag_filter_active")),
            "n_filtered_out_by_tag": int(same.get("n_filtered_out_by_tag") or 0),
            "n_untagged_seen": int(same.get("n_untagged_seen") or 0),
            "filter_include_untagged": bool(same.get("filter_include_untagged")),
            # D9 — best-of-search bytes provenance (opt-in via --save-best-bytes)
            "best_bytes_manifest": same.get("best_bytes_manifest") or [],
            "best_bytes_rejected": same.get("best_bytes_rejected") or [],
            "best_bytes_dir": same.get("best_bytes_dir"),
        }
    else:
        feature_asr = None
        fs = attacks.get("feature_square") or {}
        if isinstance(fs, dict) and "attack_success_rate" in fs:
            feature_asr = fs.get("attack_success_rate")
        problem_asr = None
        if isinstance(problem_space, dict):
            problem_asr = problem_space.get("valid_success_rate")
            if problem_asr is None:
                problem_asr = problem_space.get("attack_success_rate")
        comparison = {
            "feature_space_asr": feature_asr,
            "problem_space_valid_asr": problem_asr,
            "feature_space_realizable": False,
            "same_sample": False,
            "note": (
                "Feature-space ASR is not a PE-valid finding. "
                "Pass --pe-sample with PE files for the same-sample table."
            ),
        }
    target_str = str(audit_meta.get("target") or "")
    measurement_scope = None
    if _is_ember_target(target_str):
        from ..malware.measurement_scope import build_measurement_scope

        measurement_scope = build_measurement_scope(target_str)
    return {
        "kind": "neurinspectre_audit",
        "target": audit_meta.get("target"),
        "measurement_scope": measurement_scope,
        "defense": row.get("defense"),
        "defense_type": row.get("type"),
        "dataset": row.get("dataset", "cifar10"),
        "n_examples": audit_meta.get("n_examples"),
        "smoke": bool(audit_meta.get("smoke")),
        "modes": list(audit_meta.get("modes") or []),
        "query_budgets": list(audit_meta.get("query_budgets") or []),
        "clean_accuracy": clean_acc,
        "validity": validity,
        "characterization": characterization,
        "pipeline": pipeline,
        "problem_space": problem_space,
        "feature_vs_problem_space": comparison,
        "same_sample": bool(audit_meta.get("same_sample") or isinstance(same, dict)),
        "same_sample_detail": same_detail,
        "chosen_attack": recommended,
        "attacks": attacks,
        "notes": list(audit_meta.get("notes") or []),
        "robustbench_carmon_robust_acc": audit_meta.get("robustbench_carmon_robust_acc"),
        "official_reproduction": _official_reproduction_from_report(same, same_detail),
        "quote_as_ember2018": _quote_as_ember2018(
            audit_meta.get("target"), same, same_detail
        ),
        "timing": summary.get("timing"),
    }


def _quote_as_ember2018(target, same, same_detail) -> bool:
    """True only for an Elastic-verified EMBER 2018 extractor.

    EMBER 2024 ``official_reproduction`` is cross-platform and must not set
    this flag. A 2024 number is quoted as EMBER 2024, not as EMBER 2018.
    """
    official = _official_reproduction_from_report(same, same_detail)
    if not official:
        return False
    return _normalize_target(str(target or "")) == "ember-gbdt"


def _official_reproduction_from_report(same, same_detail) -> Optional[bool]:
    ext = None
    if isinstance(same, dict):
        ext = same.get("extractor")
    if ext is None and isinstance(same_detail, dict):
        ext = same_detail.get("extractor")
    if isinstance(ext, dict) and "official_reproduction" in ext:
        return bool(ext.get("official_reproduction"))
    return None


def characterize_audit_pipeline(target: str, *, device: str = "cpu") -> Dict[str, Any]:
    """Describe the audit target as a SecurityPipeline without reloading Carmon."""
    import torch.nn as nn

    from ..malware.measurement_scope import enrich_gbdt_pipeline_characterization

    key = _normalize_target(target)
    dummy = nn.Identity()
    if key == "ember-gbdt":
        base = SecurityPipeline.from_ember_gbdt(dummy, device=device).characterize()
        return enrich_gbdt_pipeline_characterization(base, key)
    if key in EMBER2024_TARGETS:
        variant = key.replace("ember2024-", "").replace("-gbdt", "")
        base = SecurityPipeline.from_ember2024_gbdt(
            dummy, device=device, variant=variant
        ).characterize()
        return enrich_gbdt_pipeline_characterization(base, key)
    if key == "jpeg-carmon":
        from ..defenses.wrappers import JPEGCompressionDefense

        defense = JPEGCompressionDefense(dummy, quality=75, device=device)
        pipeline = SecurityPipeline.from_defense(defense, device=device, name="jpeg_compression")
    else:
        pipeline = SecurityPipeline.identity(dummy, device=device)
    return pipeline.characterize()


def run_audit(ctx: click.Context, **kwargs: Any) -> None:
    if bool(kwargs.get("crossing_matrix")) or bool(kwargs.get("capa_diff_best")):
        if not bool(kwargs.get("save_best_bytes")):
            raise click.ClickException(
                "--crossing-matrix and --capa-diff-best require --save-best-bytes"
            )
    target = str(kwargs.get("target") or "carmon")
    output_dir = Path(str(kwargs.get("output_dir") or "results/audit"))
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke = bool(kwargs.get("smoke") or kwargs.get("smoke_test"))
    n_examples = int(kwargs.get("n_examples") or (8 if smoke else 1000))
    if smoke:
        n_examples = min(n_examples, 16)

    config = build_audit_config(
        target=target,
        n_examples=n_examples,
        smoke=smoke,
        model_path=kwargs.get("model_path"),
        data_root=str(kwargs.get("data_root") or "./data/cifar10"),
        batch_size=int(kwargs.get("batch_size") or (4 if smoke else 16)),
        seed=int(kwargs.get("seed") or 42),
        assert_clean_accuracy=bool(kwargs.get("assert_clean_accuracy", not smoke)),
        include_pgd=not bool(kwargs.get("no_pgd", False)),
        mode=kwargs.get("mode"),
        query_budgets=kwargs.get("query_budgets"),
        pe_sample=kwargs.get("pe_sample"),
        benign_corpus=kwargs.get("benign_corpus"),
    )
    if "pipeline" not in (config.get("audit") or {}):
        config["audit"]["pipeline"] = characterize_audit_pipeline(target)
    pe_source = kwargs.get("pe_sample")
    if not pe_source:
        config["audit"]["problem_space"] = evaluate_pe_parse(None)
        if _is_ember_target(target):
            hint = (
                "EMBER2018 has no PE binaries."
                if not _is_ember2024_target(target)
                else "EMBER2024 also has no PE binaries in the public split (VT API required)."
            )
            config["audit"]["problem_space"]["hint"] = (
                f"{hint} Pass --pe-sample (file or directory) "
                "to run the same-sample Full DOS / padding table."
            )
    require_official = bool(kwargs.get("require_official_reproduction"))
    require_detected = bool(kwargs.get("require_detected"))
    is_ember = _is_ember_target(target)
    is_2024 = _is_ember2024_target(target)
    if require_official:
        if not is_ember:
            raise click.ClickException(
                "--require-official-reproduction applies only to --target ember-gbdt / ember2024-gbdt"
            )
        if is_2024:
            from neurinspectre.malware.ember2024_extract import extractor_status as _v3_status

            st = _v3_status()
            if not st.get("official_reproduction"):
                raise click.ClickException(
                    "EMBER 2024 (thrember) extractor is not available on this host "
                    f"(platform={st.get('platform')} thrember={st.get('thrember_version')}). "
                    f"{st.get('hint') or ''}"
                )
        else:
            from neurinspectre.malware.ember_extract import extractor_status

            st = extractor_status()
            if not st.get("official_reproduction"):
                raise click.ClickException(
                    "EMBER v2 extraction is not Elastic-verified on this host "
                    f"(platform={st.get('platform')} lief={st.get('lief_version')} "
                    f"official_reproduction={st.get('official_reproduction')}). "
                    "Elastic checked Windows/Linux with lief 0.9.0 or 0.10.1, not Mac. "
                    "Omit --require-official-reproduction for wiring smokes."
                )
    if require_detected and not pe_source:
        raise click.ClickException("--require-detected requires --pe-sample")

    cfg_path = output_dir / "audit_config.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    click.echo(f"[audit] wrote {cfg_path}")

    model_path = Path(str(config["defenses"][0]["model"]["path"]))
    if not model_path.exists():
        if is_ember:
            hint = (
                "Fetch with: python scripts/download_ember2024.py"
                if is_2024
                else "See scripts/download_ember2018.py"
            )
            raise click.ClickException(
                f"EMBER {'2024' if is_2024 else '2018'} GBDT not found at {model_path}. {hint}"
            )
        raise click.ClickException(f"Carmon2019 checkpoint not found at {model_path}.")

    if pe_source and is_ember:
        from ..attacks.problem_space_pe import load_benign_payloads
        from ..evaluation.ember_same_sample import evaluate_ember_same_sample

        is_nonpe = _is_ember2024_nonpe_target(target)
        if is_2024:
            from neurinspectre.models.ember_gbdt import EmberGBDT2024
            from neurinspectre.malware.ember2024_extract import (
                extract_ember2024_features,
                extractor_status as _ember2024_extractor_status,
            )

            gbdt = EmberGBDT2024.from_file(model_path)
            extractor_callable = extract_ember2024_features
            v3_status = _ember2024_extractor_status()
        else:
            from neurinspectre.models.ember_gbdt import EmberGBDT

            gbdt = EmberGBDT.from_file(model_path)
            extractor_callable = None
            v3_status = None
        n_queries = max(list(config.get("query_budgets") or [50]))

        # Optional Capa-informed filter (A1) — load sidecar tags and build a
        # TagFilter from the CLI flags. Missing sidecar or missing flags means
        # unfiltered (identical to prior behavior).
        from neurinspectre.malware.capa_filters import (
            TagFilter,
            _parse_csv as _capa_parse_csv,
            load_tags_sidecar,
        )

        def _flat(values):
            out = []
            for v in values or []:
                out.extend(_capa_parse_csv(v))
            return out

        tag_filter = TagFilter(
            file_type=_flat(kwargs.get("filter_file_type")),
            family=_flat(kwargs.get("filter_family")),
            tag=_flat(kwargs.get("filter_tag")),
            ttp=_flat(kwargs.get("filter_ttp")),
            mbc=_flat(kwargs.get("filter_mbc")),
            capability=_flat(kwargs.get("filter_capability")),
            min_vt_detected=kwargs.get("min_vt_detected"),
        )
        tags_by_sha256 = None
        sidecar_path = kwargs.get("filter_tags_json")
        if sidecar_path:
            tags_by_sha256 = load_tags_sidecar(sidecar_path)
            click.echo(
                f"[audit] loaded {len(tags_by_sha256)} tag rows from {sidecar_path}"
            )
        if tag_filter.is_active() and not tags_by_sha256:
            raise click.ClickException(
                "Capa filters set but no --filter-tags-json sidecar provided; "
                "run scripts/tag_pe_corpus_from_ember2024.py first."
            )

        if is_nonpe:
            from ..evaluation.ember_same_sample_nonpe import (
                evaluate_ember2024_nonpe_same_sample,
            )
            from ..malware.file_families import resolve_family

            family = resolve_family(_EMBER2024_TARGET_TO_FAMILY[target])
            same = evaluate_ember2024_nonpe_same_sample(
                pe_source,
                gbdt,
                family,
                n_queries=int(n_queries),
                feature_eps=float(config["audit"].get("feature_epsilon") or EMBER_FEATURE_EPS),
                query_budgets=config.get("query_budgets"),
                seed=int(config.get("seed") or 42),
                max_samples=int(config["audit"].get("n_examples") or n_examples),
                extractor=extractor_callable,
                tag_filter=tag_filter if tag_filter.is_active() else None,
                tags_by_sha256=tags_by_sha256,
                filter_include_untagged=bool(kwargs.get("filter_include_untagged")),
            )
        else:
            same = evaluate_ember_same_sample(
                pe_source,
                gbdt,
                n_queries=int(n_queries),
                feature_eps=float(config["audit"].get("feature_epsilon") or EMBER_FEATURE_EPS),
                query_budgets=config.get("query_budgets"),
                seed=int(config.get("seed") or 42),
                payload_size=256 if smoke else 4096,
                benign_payloads=load_benign_payloads(kwargs.get("benign_corpus")),
                max_samples=int(config["audit"].get("n_examples") or n_examples),
                extractor=extractor_callable,
                tag_filter=tag_filter if tag_filter.is_active() else None,
                tags_by_sha256=tags_by_sha256,
                filter_include_untagged=bool(kwargs.get("filter_include_untagged")),
                capa_preserve=bool(kwargs.get("capa_preserve")),
                capa_rules_dir=kwargs.get("capa_rules_dir"),
                capa_preserve_mode=str(kwargs.get("capa_preserve_mode") or "all"),
                enable_section_slack=bool(kwargs.get("enable_section_slack")),
                transform_set=str(kwargs.get("transform_set") or "default"),
                fulldos_quiet_only=bool(kwargs.get("fulldos_quiet_only")),
                best_bytes_dir=(Path(output_dir) / "best_bytes")
                    if bool(kwargs.get("save_best_bytes")) else None,
            )
        if v3_status is not None:
            # evaluate_ember_same_sample stubs `extractor` when a custom callable is
            # provided; overwrite with the real thrember/pefile status so that
            # official_reproduction and quote_as_ember2018 propagate to the report.
            same["extractor"] = v3_status
        config["audit"]["same_sample_result"] = same
        config["audit"]["problem_space"] = same.get("problem_space")
        if require_detected and int(same.get("n_detected_malware") or 0) == 0:
            raise click.ClickException(
                "Same-sample found no GBDT-detected malware "
                f"(scanned={same.get('n_scanned')} extracted={same.get('n_extracted')} "
                f"skip_reasons={same.get('skip_reasons')}). "
                "Pass PE files the official GBDT classifies as malware, "
                "or omit --require-detected."
            )

    eval_kwargs = dict(kwargs)
    eval_kwargs["config"] = str(cfg_path)
    eval_kwargs["output_dir"] = str(output_dir)
    eval_kwargs["smoke_test"] = False
    eval_kwargs.setdefault("report", True)
    eval_kwargs.setdefault("device", kwargs.get("device", "auto"))
    if config.get("attacks"):
        run_evaluation(ctx, **eval_kwargs)
        summary_path = output_dir / "summary.json"
        if not summary_path.exists():
            raise click.ClickException(f"Evaluation did not write {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "results": [
                {
                    "defense": config["defenses"][0]["name"],
                    "type": config["defenses"][0]["type"],
                    "dataset": config["defenses"][0].get("dataset"),
                    "attacks": {},
                    "characterization": {},
                }
            ],
            "timing": {},
        }
    report = build_audit_report(summary, config=config)
    if _is_ember_target(report.get("target") or ""):
        click.echo(
            f"[audit] official_reproduction={report.get('official_reproduction')} "
            f"quote_as_ember={report.get('quote_as_ember2018')}"
        )
    report_path = output_dir / "audit_report.json"
    save_json(report, report_path)
    click.echo(f"[audit] report written to {report_path}")

    if _is_ember_target(report.get("target") or "") and report.get("measurement_scope"):
        from ..malware.measurement_scope import not_measured_ids

        gap_ids = ",".join(not_measured_ids())
        click.echo(
            "[audit] measurement_scope: named LightGBM + parse gate + query budget "
            f"(engagement gaps not measured: {gap_ids})"
        )

    if bool(kwargs.get("crossing_matrix")):
        from ..evaluation.transferability import (
            default_crossing_model_paths,
            score_transferability,
        )

        models = [
            (name, p) for name, p in default_crossing_model_paths() if p.is_file()
        ]
        if models:
            crossing = score_transferability(report_path, models)
            crossing_path = output_dir / "crossing_matrix.json"
            save_json(crossing, crossing_path)
            click.echo(
                f"[audit] crossing_matrix -> {crossing_path} "
                f"n_samples={crossing.get('n_samples')}"
            )
        else:
            click.echo("[audit] crossing_matrix skipped (no default model files on disk)")

    if bool(kwargs.get("capa_diff_best")):
        from ..evaluation.capa_diff_audit import capa_diff_best_bytes_report

        capa_report = capa_diff_best_bytes_report(
            report_path,
            rules_dir=Path(kwargs["capa_rules_dir"])
            if kwargs.get("capa_rules_dir")
            else None,
            backend=str(kwargs.get("capa_diff_backend") or "full"),
        )
        capa_path = output_dir / "capa_diff_audit.json"
        save_json(capa_report, capa_path)
        click.echo(
            f"[audit] capa_diff_audit -> {capa_path} "
            f"n_scanned={capa_report.get('n_scanned')} errors={capa_report.get('n_errors')}"
        )

    if bool(kwargs.get("write_diagnosis")) and _is_ember_target(report.get("target") or ""):
        from ..evaluation.ember_audit_diagnosis import summarize_ember_audit_report

        diag = summarize_ember_audit_report(report)
        diag_path = output_dir / "ember_audit_diagnosis.json"
        save_json(diag, diag_path)
        click.echo(f"[audit] ember_audit_diagnosis -> {diag_path} n={diag.get('n')}")

    # A3 — always attempt the Capa-tagged claim ledger. Cheap; no-op when
    # there is no same_sample_detail or when the samples carry no tags.
    same_detail_out = report.get("same_sample_detail") or {}
    samples_out = same_detail_out.get("samples") or []
    if samples_out and any(isinstance(s, dict) and s.get("tags_full") for s in samples_out):
        from neurinspectre.malware.bypass_ledger import build_ledger

        ledger = build_ledger(report)
        ledger_path = output_dir / "ember_bypass_ledger.json"
        save_json(ledger, ledger_path)
        click.echo(
            f"[audit] bypass ledger -> {ledger_path}  "
            f"(n_flipped={ledger['n_flipped']}, n_close_calls={ledger['n_close_calls']})"
        )
    chosen = report.get("chosen_attack")
    if chosen:
        click.echo(f"[audit] chosen_attack={chosen}")
    for name, metrics in (report.get("attacks") or {}).items():
        if not isinstance(metrics, dict):
            continue
        asr = metrics.get("attack_success_rate")
        robust = metrics.get("robust_accuracy")
        cost = metrics.get("cost") or {}
        click.echo(
            f"[audit] {name}: ASR={asr} RA={robust} "
            f"wall={cost.get('wall_time_s')}s forwards={cost.get('forward_passes')}"
        )
        curve = metrics.get("query_curve") or []
        if curve:
            bits = ", ".join(
                f"@{int(c.get('query_budget', 0))}={float(c.get('asr', 0.0)):.2f}"
                for c in curve
                if isinstance(c, dict)
            )
            if bits:
                click.echo(f"[audit] {name} query_curve {bits}")
    cmp_ = report.get("feature_vs_problem_space") or {}
    if cmp_:
        click.echo(
            f"[audit] same_sample={report.get('same_sample')} "
            f"feature_asr={cmp_.get('feature_space_asr')} "
            f"problem_valid_asr={cmp_.get('problem_space_valid_asr')} "
            f"n={cmp_.get('n')}"
        )
        if _is_ember_target(report.get("target") or "") and not report.get("same_sample"):
            click.echo(
                "[audit] feature-space ASR is not PE-valid. "
                "Pass --pe-sample for the same-sample Full DOS / padding table."
            )
    detail = report.get("same_sample_detail") or {}
    if detail:
        click.echo(
            f"[audit] scanned={detail.get('n_scanned')} "
            f"extracted={detail.get('n_extracted')} "
            f"detected={detail.get('n_detected_malware')} "
            f"skipped={detail.get('n_skipped')} "
            f"skip_reasons={detail.get('skip_reasons')}"
        )
        ext = detail.get("extractor") or {}
        if ext and not ext.get("available", True):
            click.echo(f"[audit] extractor unavailable: {ext.get('reasons')} {ext.get('hint') or ''}")
