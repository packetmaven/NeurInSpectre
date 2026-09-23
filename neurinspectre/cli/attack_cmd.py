"""
Attack command implementation for NeurInSpectre CLI.

Implements Paper Algorithm 2 "NEURINSPECTRE Adaptive Attack" with full
Phase 1 -> Phase 2 -> Phase 3 pipeline wired through CLI.

FIXES APPLIED:
- Issue #3: EOT sample count now set from characterization (Paper Alg 2 line 6)
- Issue #4: BPDA approximation selection wired from characterization (Paper Alg 2 lines 3-5)
- Issue #5: DLR/logit-margin loss selection based on obfuscation type (Paper Eq 4, 15)
- Enhancement #2: Phase-aware progress tracking (Paper Section 3)
- Enhancement #9: Query count tracking in progress bar

Cross-ref: Paper Section 3.2 "Phase 2: Adaptive Attack Synthesis"
Cross-ref: Paper Section 4 "Implementation"
Cross-ref: Paper Table 1 "Attack success rate against evaluated defenses"
Cross-ref: Paper Algorithm 2 "NEURINSPECTRE Adaptive Attack"

Version: 2.0.1
"""

from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click

from ..attacks import AttackFactory
from .exporters import export_attack_json, export_attack_sarif
from .formatters import build_console, render_attack_report
from .progress import ProgressReporter
from .utils import (
    build_defense,
    evaluate_attack_runner,
    load_dataset,
    load_model,
    load_threshold_overrides,
    load_yaml,
    resolve_device,
    save_json,
    set_seed,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper-aligned constants
# ---------------------------------------------------------------------------

# Paper Section 4: "Krylov order k=20 provides sufficient spectral resolution"
DEFAULT_KRYLOV_ORDER = 20

# Paper Table 1 caption: "epsilon=8/255 for Linf attacks"
DEFAULT_EPSILON = 8 / 255

# Paper Section 4: "EOT sampling... typically N in [10,100]"
DEFAULT_EOT_SAMPLES = 50

# Paper Section 4: importance weighting reduces variance (vs uniform sampling).
EOT_IMPORTANCE_WEIGHTING_DEFAULT = True

# Paper Section 5.3: warns about too-small EOT sample counts on stochastic defenses.
MIN_EOT_SAMPLES_WARNING = 10

# Paper Section 3.2.3: "kappa is a confidence margin (typically 0)"
DEFAULT_LOGIT_MARGIN = 0.0


def _select_loss_function(
    characterization: Optional[Dict[str, Any]],
    attack_type: str,
) -> str:
    """
    Select loss function based on characterized obfuscation type.

    ISSUE #5 FIX: Paper Algorithm 2 line 8 specifies logit-based loss for
    vanishing gradients; Paper Section 2.1.3 recommends DLR for scale
    invariance against defenses that manipulate logit magnitudes.

    Decision logic:
        VANISHING / distillation  ->  logit_margin  (Paper Eq 15, Section 3.2.3)
        SHATTERED / STOCHASTIC    ->  dlr           (Paper Eq 4, Section 2.1.3)
        default                   ->  ce            (Paper Eq 1)

    Cross-ref: Paper Equation 4  (DLR loss)
    Cross-ref: Paper Equation 15 (L_logit loss for vanishing gradients)
    Cross-ref: Paper Section 5.4.1 Case Study - Defensive Distillation
    """
    if characterization is None or attack_type != "neurinspectre":
        return "ce"  # default cross-entropy

    obf_types = characterization.get("obfuscation_types") or []
    obf_lower = [t.lower() for t in obf_types]

    # Paper Algorithm 2 line 8: if VANISHING then L = L_logit
    # Paper Section 5.4.1: use logit-based loss to bypass softmax saturation.
    if "vanishing" in obf_lower or "distillation" in obf_lower:
        logger.info(
            "Vanishing/distillation gradients detected -> switching to logit-margin loss "
            "(Paper Eq 15, Section 3.2.3, Case Study 5.4.1)"
        )
        return "logit_margin"

    # Paper Section 2.1.3: "DLR... robust to defensive techniques that
    #   manipulate logit magnitudes."
    # DLR is preferred over CE when any obfuscation is present.
    if obf_types:
        logger.info(
            "Gradient obfuscation detected (%s) -> using DLR loss for scale invariance "
            "(Paper Eq 4, Section 2.1.3)",
            ", ".join(obf_types),
        )
        return "dlr"

    return "ce"


def _configure_eot(
    attack_config: Dict[str, Any],
    characterization: Optional[Dict[str, Any]],
) -> None:
    """
    Configure EOT sampling parameters from characterization output.

    ISSUE #3 FIX: Wires Phase 1 characterization into Phase 2 attack config.

    Paper Algorithm 2 line 6:
        N_EOT = EstimateSampleCount(sigma^2_g)

    Paper Section 4:
        "We adaptively select the number of EOT samples N based on estimated
         gradient variance. Importance weighting reduces estimator variance
         compared to uniform sampling."

    Paper Section 5.3:
        Warns that very small N can materially reduce ASR on stochastic defenses.

    Paper Section 5.4.2 Case Study - Randomized Smoothing:
        "Our characterization detects stochasticity (sigma^2_g > 0.1) and applies
         EOT with N=50 samples."

    Cross-ref: Paper Equation 7  (EOT gradient averaging)
    Cross-ref: Paper Equation 14 (importance-weighted EOT)
    """
    if characterization is None:
        return

    requires_eot = bool(characterization.get("requires_eot", False))
    if not requires_eot:
        return

    # Paper Alg 2 line 6: N_EOT = EstimateSampleCount(sigma^2_g)
    recommended_n = characterization.get("recommended_eot_samples")
    gradient_variance = characterization.get("gradient_variance")

    if recommended_n is not None and int(recommended_n) > 0:
        n_eot = int(recommended_n)
    elif gradient_variance is not None:
        # Heuristic aligned with paper case studies:
        # Paper Section 5.4.2: sigma^2_g > 0.1 -> N=50
        gv = float(gradient_variance)
        if gv > 1.0:
            n_eot = 100  # very high variance
        elif gv > 0.1:
            n_eot = 50   # moderate variance (Paper Case Study 2)
        else:
            n_eot = 20   # low but present variance
    else:
        n_eot = DEFAULT_EOT_SAMPLES

    # Paper Section 5.3: warn if too few samples
    if n_eot < MIN_EOT_SAMPLES_WARNING:
        logger.warning(
            "EOT sample count N=%d is below recommended minimum (%d). "
            "Paper Section 5.3: very small N can materially reduce ASR on stochastic defenses.",
            n_eot,
            MIN_EOT_SAMPLES_WARNING,
        )

    attack_config["n_eot_samples"] = n_eot

    # Paper Eq 14: importance weights w_i proportional to L(f(g_i(x)), y)
    # Paper Section 4: importance weighting reduces estimator variance.
    attack_config["eot_importance_weighting"] = EOT_IMPORTANCE_WEIGHTING_DEFAULT

    logger.info(
        "Stochastic defense detected -> EOT configured: N=%d samples, "
        "importance_weighting=%s (Paper Eq 14, Section 4). "
        "Gradient variance sigma^2_g=%s",
        n_eot,
        attack_config["eot_importance_weighting"],
        f"{gradient_variance:.4f}" if gradient_variance is not None else "n/a",
    )


def _configure_bpda(
    attack_config: Dict[str, Any],
    characterization: Optional[Dict[str, Any]],
    defense_name: str,
) -> None:
    """
    Configure BPDA approximation from characterization output.

    ISSUE #4 FIX: Wires Phase 1 characterization into Phase 2 BPDA config.

    Paper Algorithm 2 lines 3-5:
        if SHATTERED then
            g' = LearnApproximation(g) or g' = Identity

    Paper Section 3.2.1:
        "For defenses where g is approximately identity-preserving, we use
         g'(x) = x."

    Paper Equation 13:
        g' = argmin E_x[ ||g(x) - g'(x)||^2 + lambda ||nabla g'(x)||^2_F ]

    Paper Section 4:
        "For non-differentiable transformations, we train lightweight neural
         networks (~100K parameters) to approximate the transformation."

    Paper Section 5.4.3 Case Study - Thermometer Encoding:
        Trains a differentiable approximation network g' to model the
        thermometer transformation, then uses BPDA through g'.

    Cross-ref: Paper Equation 5  (BPDA gradient computation)
    Cross-ref: Paper Equation 13 (learned approximation objective)
    """
    if characterization is None:
        return

    requires_bpda = bool(characterization.get("requires_bpda", False))
    if not requires_bpda:
        return

    obf_types = characterization.get("obfuscation_types") or []
    obf_lower = [t.lower() for t in obf_types]

    # Defenses where identity approximation suffices (Paper Section 3.2.1):
    #   These are approximately identity-preserving: g(x) approx x
    #   JPEG compression, bit-depth reduction, spatial smoothing
    identity_defenses = {
        "jpeg", "jpegcompression", "jpeg_compression",
        "bitdepth", "bitdepthreduction", "bit_depth_reduction",
        "spatialsmoothing", "spatial_smoothing",
        "randompadcrop", "random_padcrop",
        "featuresqueezing", "feature_squeezing",
    }

    # Defenses requiring learned approximation (Paper Section 4, 5.4.3):
    #   Thermometer encoding creates non-trivial transformation
    learned_defenses = {
        "thermometer", "thermometerencoding", "thermometer_encoding",
    }

    defense_key = defense_name.lower().replace("-", "").replace("_", "")

    if defense_key in learned_defenses:
        attack_config["bpda_approximation"] = "learned"
        attack_config["bpda_network_params"] = 100_000  # Paper Section 4: "~100K params"
        attack_config["bpda_jacobian_reg"] = True  # Paper Eq 13: lambda||nabla g'||^2_F
        logger.info(
            "Shattered gradients (%s) -> BPDA with LEARNED approximation "
            "(Paper Section 4: ~100K params, Eq 13 Jacobian regularization). "
            "Case Study: Section 5.4.3 Thermometer Encoding.",
            defense_name,
        )
    elif defense_key in identity_defenses or "shattered" in obf_lower:
        attack_config["bpda_approximation"] = "identity"
        logger.info(
            "Shattered gradients (%s) -> BPDA with IDENTITY approximation "
            "(Paper Section 3.2.1: g is approximately identity-preserving, g'(x)=x).",
            defense_name,
        )
    else:
        # Unknown defense: default to identity, log for user awareness
        attack_config["bpda_approximation"] = "identity"
        logger.info(
            "Shattered gradients (%s) -> BPDA with identity approximation (default). "
            "If ASR is low, consider --bpda-approximation learned.",
            defense_name,
        )


def _configure_vanishing(
    attack_config: Dict[str, Any],
    characterization: Optional[Dict[str, Any]],
) -> None:
    """
    Configure vanishing gradient bypass from characterization.

    Paper Algorithm 2 line 8:
        L = L_logit  (use logit-based loss)

    Paper Section 3.2.3:
        "When vanishing gradients are detected, we employ gradient
         regularization and soft labels. For defensive distillation,
         we target the pre-softmax logits directly."

    Paper Equation 15:
        L_logit(x, y) = max_{i!=y} z_i - z_y + kappa
        where kappa is a confidence margin (typically 0).

    Paper Section 5.4.1 Case Study - Defensive Distillation:
        "NEURINSPECTRE's characterization identifies vanishing gradients
         with ||g||_2 < 10^-6. The synthesized attack uses logit-based
         loss L_logit, bypassing the softmax saturation."
    """
    if characterization is None:
        return

    obf_types = characterization.get("obfuscation_types") or []
    obf_lower = [t.lower() for t in obf_types]

    if "vanishing" not in obf_lower and "distillation" not in obf_lower:
        return

    # Paper Eq 15: confidence margin kappa (typically 0)
    attack_config["logit_margin"] = DEFAULT_LOGIT_MARGIN
    attack_config["use_logit_loss"] = True

    logger.info(
        "Vanishing gradients detected -> targeting pre-softmax logits directly "
        "(Paper Eq 15, margin kappa=%.1f). "
        "Paper Section 5.4.1: L_logit bypasses softmax saturation.",
        DEFAULT_LOGIT_MARGIN,
    )


def _configure_hybrid(
    attack_config: Dict[str, Any],
    characterization: Optional[Dict[str, Any]],
) -> None:
    """
    Configure hybrid attack when multiple obfuscation types detected.

    Paper Section 3.2.4 "Hybrid Attack Integration":
        "When multiple obfuscation types are present, we combine techniques."

    Paper Equation 16:
        nabla_hybrid = (1/N) sum nabla L(f(g'(g_i(x))), y)
        "This applies EOT over stochastic transformations while using BPDA
         through non-differentiable components."

    Paper Section 5.2.1:
        "The misclassified defense (Ensemble Diversity) exhibited
         characteristics of both stochastic and shattered gradients;
         applying the hybrid attack achieved success."
    """
    if characterization is None:
        return

    obf_types = characterization.get("obfuscation_types") or []
    if len(obf_types) < 2:
        return

    attack_config["hybrid_mode"] = True
    logger.info(
        "Multiple obfuscation types detected (%s) -> hybrid attack mode enabled "
        "(Paper Eq 16, Section 3.2.4). Combines BPDA + EOT as needed.",
        ", ".join(obf_types),
    )


def run_attack(ctx: click.Context, **kwargs: Any) -> None:
    """
    Execute NeurInSpectre attack command.

    Implements the full 3-phase pipeline:
        Phase 1: Defense Characterization (if attack_type == "neurinspectre")
        Phase 2: Adaptive Attack Synthesis (config from characterization)
        Phase 3: Attack Execution and Evaluation

    Cross-ref: Paper Section 3 "NEURINSPECTRE Framework"
    Cross-ref: Paper Algorithm 2 "NEURINSPECTRE Adaptive Attack"
    """
    # -------------------------------------------------------------------
    # 0. Parse and validate arguments
    # -------------------------------------------------------------------
    cmd_verbose = int(kwargs.get("verbose", 0) or 0)
    if ctx is not None:
        ctx.obj = ctx.obj or {}
        if cmd_verbose:
            ctx.obj["verbose"] = max(int(ctx.obj.get("verbose", 0)), cmd_verbose)

    device = resolve_device(kwargs.get("device", "cpu"))
    seed = int(kwargs.get("seed", 42))
    set_seed(seed)

    dataset_name = kwargs.get("dataset")
    data_path = kwargs.get("data_path")
    labels_path = kwargs.get("labels_path")
    nuscenes_version = kwargs.get("nuscenes_version")
    if dataset_name == "custom" and not data_path:
        raise click.ClickException("Custom dataset requires --data-path.")
    if dataset_name == "nuscenes" and not labels_path:
        raise click.ClickException(
            "nuScenes dataset requires --labels-path mapping sample_token -> class label."
        )

    batch_size = int(kwargs.get("batch_size", 128))
    num_samples = int(kwargs.get("num_samples", 1000))

    # -------------------------------------------------------------------
    # 1. Load data, model, defense
    # -------------------------------------------------------------------
    loader, _x, _y = load_dataset(
        dataset_name,
        data_path=data_path,
        labels_path=labels_path,
        nuscenes_version=nuscenes_version,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )

    model_path = kwargs.get("model")
    model = load_model(model_path, dataset=dataset_name, device=device)

    defense_name = kwargs.get("defense", "none")
    defense_cfg_path = kwargs.get("defense_config")
    defense_params: Dict[str, Any] = {}
    if defense_cfg_path:
        defense_params = load_yaml(defense_cfg_path)
        if "defense" in defense_params and isinstance(defense_params["defense"], dict):
            defense_params = defense_params["defense"]

    defense_model = build_defense(defense_name, model, defense_params, device=device)
    eval_model = defense_model or model

    # -------------------------------------------------------------------
    # 2. Attack configuration (base)
    # -------------------------------------------------------------------
    attack_type = kwargs.get("attack_type", "neurinspectre")
    epsilon = float(kwargs.get("epsilon", DEFAULT_EPSILON))
    norm = str(kwargs.get("norm", "Linf"))
    iterations = int(kwargs.get("iterations", 100))
    targeted = bool(kwargs.get("targeted", False))

    threshold_overrides: Optional[Dict[str, Any]] = None
    threshold_overrides_path = kwargs.get("threshold_overrides")
    if threshold_overrides_path:
        try:
            threshold_overrides = load_threshold_overrides(str(threshold_overrides_path))
        except Exception as exc:
            raise click.ClickException(f"Failed to load --threshold-overrides: {exc}") from exc

    if targeted and attack_type == "neurinspectre":
        click.echo(
            "Targeted mode not supported for adaptive attack; "
            "falling back to APGD."
        )
        attack_type = "apgd"

    # Paper Table 1 reproduction indicator
    is_paper_settings = (
        abs(epsilon - 8 / 255) < 1e-6
        and norm == "Linf"
        and iterations >= 100
    )
    if is_paper_settings:
        logger.info(
            "Using Paper Table 1 evaluation settings: eps=8/255, Linf, %d iterations",
            iterations,
        )

    attack_config: Dict[str, Any] = {
        "epsilon": epsilon,
        "norm": norm,
        "n_iterations": iterations,
        "batch_size": batch_size,
        "seed": seed,
        "targeted": targeted,
        # Volterra memory control (Issue 6: decisive experiment plumbing)
        "volterra_mode": str(kwargs.get("volterra_mode", "auto")),
        "volterra_kernel": str(kwargs.get("volterra_kernel", "power_law")),
        # Tier 2: optimizer-confound control for Volterra alpha fitting.
        "volterra_gradient_source": str(kwargs.get("volterra_gradient_source", "pre_optimizer")),
        "volterra_optimizer": str(kwargs.get("volterra_optimizer", "sgd")),
    }
    if threshold_overrides:
        # Passed through to DefenseAnalyzer during Phase 1 characterization.
        attack_config["characterization_thresholds"] = dict(threshold_overrides)
    if kwargs.get("volterra_alpha") is not None:
        attack_config["alpha_volterra"] = float(kwargs["volterra_alpha"])
    if kwargs.get("volterra_memory_length") is not None:
        attack_config["memory_length"] = int(kwargs["volterra_memory_length"])

    # -------------------------------------------------------------------
    # 3. Build console and progress
    # -------------------------------------------------------------------
    quiet = bool(ctx.obj.get("quiet", False)) if ctx and ctx.obj else False
    no_progress = bool(kwargs.get("no_progress", False))
    report_format = str(kwargs.get("report_format", "rich"))
    no_color = bool(kwargs.get("no_color", False))
    force_color = bool(kwargs.get("color", False))
    brief = bool(kwargs.get("brief", False))
    summary_only = bool(kwargs.get("summary_only", False))
    console = build_console(no_color=no_color, force_color=force_color)

    show_progress = not no_progress and not quiet
    total_samples = int(num_samples) if num_samples is not None else None

    # -------------------------------------------------------------------
    # 4. Create attack runner (Phase 1 may happen inside for neurinspectre)
    # -------------------------------------------------------------------
    start_time = time.time()

    with ProgressReporter(
        description="Phase 1/3: Initializing...",
        total=total_samples,
        enabled=show_progress,
        console=console,
        show_queries=(attack_type == "neurinspectre"),
    ) as progress:

        # -- Phase 1: Defense Characterization (Paper Algorithm 2 line 1) --
        if attack_type == "neurinspectre":
            progress.set_phase("Defense Characterization", 1, 3)

        runner = AttackFactory.create_attack(
            attack_type,
            model,
            config=attack_config,
            characterization_loader=loader if attack_type == "neurinspectre" else None,
            defense=defense_model,
            device=device,
        )

        # ==============================================================
        # ISSUE #3, #4, #5 FIXES:
        # Wire characterization output -> attack configuration
        #
        # After Phase 1 (characterization), we read the results and
        # configure the attack accordingly. This implements Paper
        # Algorithm 2 lines 2-9 in the CLI layer.
        # ==============================================================
        characterization_dict: Dict[str, Any] = {}

        if attack_type == "neurinspectre" and hasattr(runner, "characterization"):
            char_obj = runner.characterization
            if char_obj is not None:
                try:
                    characterization_dict = char_obj.to_dict()
                except Exception:
                    characterization_dict = {}

                if characterization_dict:
                    obf_types = characterization_dict.get("obfuscation_types") or []
                    logger.info(
                        "Phase 1 complete: obfuscation=%s, confidence=%.4f",
                        ", ".join(obf_types) if obf_types else "none",
                        float(characterization_dict.get("confidence", 0.0)),
                    )

                    # -- Phase 2: Adaptive Attack Synthesis --
                    # (Paper Algorithm 2 lines 2-9)
                    progress.set_phase("Adaptive Attack Synthesis", 2, 3)

                    # ISSUE #5 FIX: Select loss function
                    # Paper Eq 4 (DLR), Eq 15 (logit-margin)
                    loss_fn = _select_loss_function(
                        characterization_dict, attack_type
                    )
                    attack_config["loss"] = loss_fn

                    # ISSUE #3 FIX: Configure EOT
                    # Paper Alg 2 line 6, Eq 7, Eq 14
                    _configure_eot(attack_config, characterization_dict)

                    # ISSUE #4 FIX: Configure BPDA
                    # Paper Alg 2 lines 3-5, Eq 5, Eq 13
                    _configure_bpda(
                        attack_config, characterization_dict, defense_name
                    )

                    # Configure vanishing gradient bypass
                    # Paper Alg 2 line 8, Eq 15
                    _configure_vanishing(attack_config, characterization_dict)

                    # Configure hybrid mode if multiple obfuscation types
                    # Paper Section 3.2.4, Eq 16
                    _configure_hybrid(attack_config, characterization_dict)

                    # Apply updated config to runner if supported
                    if hasattr(runner, "update_config"):
                        runner.update_config(attack_config)
                    elif hasattr(runner, "attack") and hasattr(
                        runner.attack, "update_config"
                    ):
                        runner.attack.update_config(attack_config)

        # -- Phase 3: Attack Execution (Paper Algorithm 2 lines 10-16) --
        if attack_type == "neurinspectre":
            progress.set_phase("Attack Execution", 3, 3)
        else:
            progress.advance(0, description="Running attack...")

        summary = evaluate_attack_runner(
            runner,
            eval_model,
            loader,
            num_samples=num_samples,
            device=device,
            targeted=targeted,
            save_dir=kwargs.get("save_adversarials"),
            norm=norm,
            progress_callback=progress.advance if show_progress else None,
        )

    elapsed = time.time() - start_time

    selected_attack_impl = None
    try:
        selected_attack_impl = runner.attack.__class__.__name__  # type: ignore[attr-defined]
    except Exception:
        selected_attack_impl = None

    # -------------------------------------------------------------------
    # 5. Extract final characterization (if available)
    # -------------------------------------------------------------------
    if not characterization_dict:
        if hasattr(runner, "characterization") and runner.characterization is not None:
            try:
                characterization_dict = runner.characterization.to_dict()
            except Exception:
                characterization_dict = {}

    # -------------------------------------------------------------------
    # 6. Build output payload
    # -------------------------------------------------------------------
    attack_success_rate = summary.get("attack_success_rate", 0.0)

    output = {
        "attack": str(attack_type),
        "defense": str(defense_name),
        "dataset": str(dataset_name),
        "epsilon": float(epsilon),
        "norm": str(norm),
        "results": {
            "attack_success_rate": attack_success_rate,
            "robust_accuracy": summary.get("robust_accuracy", 0.0),
            "clean_accuracy": summary.get("clean_accuracy", 0.0),
            "selected_attack_impl": selected_attack_impl,
            "queries": summary.get("queries"),
            "iterations": summary.get("iterations"),
            "samples": summary.get("samples", 0),
            "correct_samples": summary.get("correct_samples", 0),
            "perturbation": summary.get("perturbation", {}),
            "query_efficiency": summary.get("query_efficiency", {}),
            "characterization": characterization_dict,
        },
        "attack_success_rate": attack_success_rate,
        # Record which paper-aligned configs were applied
        "attack_config_applied": {
            "loss": attack_config.get("loss", "ce"),
            "bpda_approximation": attack_config.get("bpda_approximation"),
            "n_eot_samples": attack_config.get("n_eot_samples"),
            "eot_importance_weighting": attack_config.get(
                "eot_importance_weighting"
            ),
            "hybrid_mode": attack_config.get("hybrid_mode", False),
            "use_logit_loss": attack_config.get("use_logit_loss", False),
            "logit_margin": attack_config.get("logit_margin"),
            "volterra_mode": attack_config.get("volterra_mode"),
            "volterra_kernel": attack_config.get("volterra_kernel"),
            "alpha_volterra": attack_config.get("alpha_volterra"),
            "memory_length": attack_config.get("memory_length"),
        },
        "timing": {
            "total_seconds": float(elapsed),
            "per_sample_ms": float(
                elapsed * 1000.0 / max(1, summary.get("samples", 1))
            ),
        },
    }

    # -------------------------------------------------------------------
    # 7. Save results
    # -------------------------------------------------------------------
    output_path = Path(kwargs.get("output", "attack_results.json"))
    save_json(output, output_path)
    click.echo(f"Attack results written to {output_path}")

    # Optional Tier 2 artifact: persist a compact feature row for calibration / audit.
    save_features_path = kwargs.get("save_features")
    if save_features_path:
        try:
            from datetime import datetime

            meta = characterization_dict.get("metadata") if isinstance(characterization_dict, dict) else {}
            meta = meta if isinstance(meta, dict) else {}
            feature_row = {
                "kind": "attack_features",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "attack": str(attack_type),
                "defense": str(defense_name),
                "dataset": str(dataset_name),
                "epsilon": float(epsilon),
                "norm": str(norm),
                "seed": int(seed),
                "iterations": int(iterations),
                "num_samples": int(num_samples) if num_samples is not None else None,
                "batch_size": int(batch_size),
                "results": {
                    "attack_success_rate": float(summary.get("attack_success_rate", 0.0)),
                    "robust_accuracy": float(summary.get("robust_accuracy", 0.0)),
                    "clean_accuracy": float(summary.get("clean_accuracy", 0.0)),
                    "samples": int(summary.get("samples", 0) or 0),
                    "correct_samples": int(summary.get("correct_samples", 0) or 0),
                    "queries_mean": summary.get("queries"),
                    "iterations_mean": summary.get("iterations"),
                },
                # Keep the full characterization for traceability.
                "characterization": characterization_dict,
                # Mirror a few scalar "detector" features at top-level for easy parsing.
                "detector_features": {
                    "etd_score": characterization_dict.get("etd_score") if isinstance(characterization_dict, dict) else None,
                    "alpha_volterra": characterization_dict.get("alpha_volterra") if isinstance(characterization_dict, dict) else None,
                    "jacobian_rank": characterization_dict.get("jacobian_rank") if isinstance(characterization_dict, dict) else None,
                    "gradient_variance": characterization_dict.get("gradient_variance") if isinstance(characterization_dict, dict) else None,
                    "spectral_entropy_norm": meta.get("spectral_entropy_norm"),
                    "high_freq_ratio": meta.get("high_freq_ratio"),
                    "volterra_rmse_scaled": meta.get("volterra_rmse_scaled"),
                    "confidence": characterization_dict.get("confidence") if isinstance(characterization_dict, dict) else None,
                },
            }

            p = Path(str(save_features_path))
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(feature_row, sort_keys=True) + "\n")
            click.echo(f"Features row appended to {p}")
        except Exception as exc:
            raise click.ClickException(f"Failed to write --save-features artifact: {exc}") from exc

    # -------------------------------------------------------------------
    # 8. Render report
    # -------------------------------------------------------------------
    report = bool(kwargs.get("report", True))
    if report and not quiet:
        render_attack_report(
            console,
            summary=summary,
            meta={
                "attack": str(attack_type),
                "defense": str(defense_name),
                "dataset": str(dataset_name),
                "epsilon": float(epsilon),
                "norm": str(norm),
                "iterations": int(iterations),
                "targeted": bool(targeted),
            },
            output_path=str(output_path),
            verbosity=(
                int(ctx.obj.get("verbose", 0)) if ctx and ctx.obj else cmd_verbose
            ),
            report_format=report_format,
            brief=brief,
            summary_only=summary_only,
        )

    # -------------------------------------------------------------------
    # 9. Optional exports (JSON executive report, SARIF for CI/CD)
    # -------------------------------------------------------------------
    json_output = kwargs.get("json_output")
    if json_output:
        export_attack_json(
            {
                "attack": str(attack_type),
                "defense": str(defense_name),
                "dataset": str(dataset_name),
                "attack_success_rate": float(
                    summary.get("attack_success_rate", 0.0)
                ),
                "robust_accuracy": float(
                    summary.get("robust_accuracy", 0.0)
                ),
                "clean_accuracy": float(
                    summary.get("clean_accuracy", 0.0)
                ),
                "epsilon": float(epsilon),
                "norm": str(norm),
                "targeted": bool(targeted),
                "characterization": characterization_dict,
                "attack_config_applied": output.get(
                    "attack_config_applied", {}
                ),
            },
            json_output,
        )
        click.echo(f"JSON report written to {json_output}")

    sarif_output = kwargs.get("sarif_output")
    if sarif_output:
        export_attack_sarif(
            {
                "attack": str(attack_type),
                "defense": str(defense_name),
                "dataset": str(dataset_name),
                "attack_success_rate": float(
                    summary.get("attack_success_rate", 0.0)
                ),
                "characterization": characterization_dict,
            },
            sarif_output,
        )
        click.echo(f"SARIF report written to {sarif_output}")
