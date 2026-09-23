"""
Defense characterization command implementation for NeurInSpectre CLI.

Implements Paper Algorithm 1 "Spectral Gradient Analysis" with full
Phase 1 characterization pipeline including:
  - Gradient quality analysis (Paper Section 3.1.1)
  - Spectral analysis via Krylov subspaces (Paper Section 3.1.2)
  - Stochasticity detection (Paper Section 3.1.3)
  - Non-differentiability detection (Paper Section 3.1.4)

PAPER ALIGNMENT:
- All characterization signals match Paper definitions exactly
- Krylov order k=20 per Paper Section 4
- ETD, alpha_Volterra, gradient_variance, jacobian_rank, autocorr_timescale
- Outputs feed directly into attack_cmd.py Phase 2 (Issues #3, #4, #5)

Cross-ref: Paper Section 3.1 "Phase 1: Defense Characterization"
Cross-ref: Paper Algorithm 1 "Spectral Gradient Analysis"
Cross-ref: Paper Equations 8-12 (characterization metrics)
Cross-ref: Paper Section 4 "Implementation"
Cross-ref: Paper Table 2 "Ablation study"

Version: 2.0.1
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click
import numpy as np
import torch.nn as nn

from ..characterization.defense_analyzer import DefenseAnalyzer
from ..characterization.layer3_etd import compute_etd_features
from .exporters import export_characterization_json, export_characterization_sarif
from .formatters import build_console, render_characterization_report
from .progress import status
from .utils import (
    build_defense,
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

# Paper Section 4: "Characterization phase completes in under 60 seconds"
DEFAULT_NUM_CHARACTERIZATION_SAMPLES = 100

# Paper Table 1 caption: "epsilon=8/255 for Linf attacks"
DEFAULT_EPSILON = 8 / 255


def run_characterization(ctx: click.Context, **kwargs: Any) -> None:
    """
    Execute NeurInSpectre characterization command.

    Implements Paper Algorithm 1 "Spectral Gradient Analysis" with full
    pipeline for automated defense obfuscation detection.

    Paper Algorithm 1 pseudocode:
        1. Σ ← {} ∪ Spectral signatures
        2. for x in {x_i} do
        3.     g ← ∇_x L(D(x), y)
        4.     Q, T ← Lanczos(H, g, k)  # Tridiagonal decomposition
        5.     λ ← eig(T)               # Ritz values
        6.     Σ ← Σ ∪ {||g||², λ, T}
        7. end for
        8. τ ← ClassifyObfuscation(Σ)
        9. return τ

    Cross-ref: Paper Section 3.1 "Phase 1: Defense Characterization"
    Cross-ref: Paper Algorithm 1 "Spectral Gradient Analysis"
    Cross-ref: Paper Equations 8, 9, 10, 11, 12
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

    num_samples = int(kwargs.get("num_samples", DEFAULT_NUM_CHARACTERIZATION_SAMPLES))
    batch_size = int(kwargs.get("batch_size", 128))

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

    use_bpda_approx = bool(kwargs.get("use_bpda_approx", False))
    if use_bpda_approx and defense_model is not None and hasattr(defense_model, "get_bpda_approximation"):
        class _BPDAApproxWrapper(nn.Module):
            def __init__(self, defense):
                super().__init__()
                self.defense = defense

            def forward(self, x):
                try:
                    return self.defense(x, use_approximation=True)
                except TypeError:
                    return self.defense(x)

        eval_model = _BPDAApproxWrapper(defense_model)
        logger.info("Using BPDA approximation for characterization.")
    elif use_bpda_approx:
        logger.warning("BPDA approximation requested but no compatible defense found.")

    # -------------------------------------------------------------------
    # 2. Configuration
    # -------------------------------------------------------------------
    krylov_order = int(kwargs.get("krylov_order", DEFAULT_KRYLOV_ORDER))
    epsilon = DEFAULT_EPSILON
    visualize = bool(kwargs.get("visualize", False))

    threshold_overrides: Optional[Dict[str, Any]] = None
    threshold_overrides_path = kwargs.get("threshold_overrides")
    if threshold_overrides_path:
        try:
            threshold_overrides = load_threshold_overrides(str(threshold_overrides_path))
        except Exception as exc:
            raise click.ClickException(f"Failed to load --threshold-overrides: {exc}") from exc

    # Paper Section 4: validate Krylov order
    if krylov_order < 10 or krylov_order > 50:
        logger.warning(
            "Krylov order k=%d is outside recommended range [10, 50]. "
            "Paper Section 5.3: use k in [10,50] for stable results.",
            krylov_order,
        )

    # Paper Section 4: "under 60 seconds for typical models"
    if num_samples > 200:
        logger.warning(
            "num_samples=%d may exceed 60-second characterization target "
            "(Paper Section 4). Recommended: <=200 samples.",
            num_samples,
        )

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

    # -------------------------------------------------------------------
    # 4. Run characterization (Paper Algorithm 1)
    # -------------------------------------------------------------------
    verbose = bool(ctx.obj.get("verbose", 0)) if ctx and ctx.obj else bool(cmd_verbose)

    # Initialize DefenseAnalyzer
    # Paper Section 4: "n_samples" for characterization
    analyzer = DefenseAnalyzer(
        eval_model,
        n_samples=num_samples,
        n_probe_images=num_samples,
        device=device,
        verbose=verbose,
        krylov_dim=krylov_order,
        thresholds=threshold_overrides,
        volterra_gradient_source=str(kwargs.get("volterra_gradient_source", "pre_optimizer")),
        volterra_optimizer=str(kwargs.get("volterra_optimizer", "sgd")),
    )

    start_time = time.time()

    with status(
        console,
        "Running spectral gradient analysis (Paper Algorithm 1)...",
        enabled=show_progress,
    ):
        # Paper Algorithm 1 line 1: CharacterizeDefense(D)
        # This executes lines 2-8: gradient collection, Lanczos decomposition,
        # eigenvalue extraction, obfuscation classification
        characterization = analyzer.characterize(loader, eps=epsilon)

    elapsed = time.time() - start_time

    # Paper Section 4: "under 60 seconds for typical models"
    if elapsed > 60.0:
        logger.warning(
            "Characterization took %.1fs (Paper Section 4 target: <60s). "
            "Consider reducing --num-samples.",
            elapsed,
        )

    # -------------------------------------------------------------------
    # 5. Collect gradients for optional Krylov features + visualization
    # -------------------------------------------------------------------
    # Paper Section 3.1.2: Krylov subspace analysis
    # Paper Equation 9: K_k(H, g) = span{g, Hg, H^2 g, ..., H^(k-1) g}
    gradients, images, labels = analyzer.collect_gradient_samples(loader, eps=epsilon)

    krylov_features: Dict[str, Any] = {}
    spectrum_path: Optional[Path] = None

    if gradients:
        # Paper Section 3.1.2: "eigenvalue distribution within K_k reveals
        # gradient obfuscation patterns"
        grad_matrix = np.stack(gradients, axis=0)
        krylov_features = compute_etd_features(grad_matrix, krylov_dim=krylov_order)

        # Optional visualization (Paper Extended Results Appendix A)
        if visualize:
            # Paper: "eigenvalue spectrum"
            # Compute covariance eigenvalues for visualization
            H_approx = np.cov(grad_matrix, rowvar=False)
            eigvals = np.linalg.eigvalsh(H_approx)
            eigvals = np.sort(eigvals)[::-1]  # descending order

            output_path = Path(kwargs.get("output", "characterization.json"))
            spectrum_path = output_path.with_suffix(".spectrum.png")
            _plot_spectrum(eigvals, spectrum_path)

    # -------------------------------------------------------------------
    # 6. Build output payload
    # -------------------------------------------------------------------
    output = {
        "defense": str(defense_name),
        "dataset": str(dataset_name),
        "characterization": characterization.to_dict(),
        "krylov_features": krylov_features,
        "timing": {
            "total_seconds": float(elapsed),
            "per_sample_ms": float(elapsed * 1000.0 / max(1, num_samples)),
        },
        "config": {
            "krylov_order": krylov_order,
            "num_samples": num_samples,
            "epsilon": epsilon,
        },
    }

    if spectrum_path:
        output["visualization"] = {"spectrum": str(spectrum_path)}

    # -------------------------------------------------------------------
    # 7. Save results
    # -------------------------------------------------------------------
    output_path = Path(kwargs.get("output", "characterization.json"))
    save_json(output, output_path)
    click.echo(f"Characterization written to {output_path}")

    if spectrum_path:
        click.echo(f"Spectrum plot saved to {spectrum_path}")

    # -------------------------------------------------------------------
    # 8. Render report
    # -------------------------------------------------------------------
    report = bool(kwargs.get("report", True))
    if report and not quiet:
        render_characterization_report(
            console,
            characterization=characterization.to_dict(),
            defense=str(defense_name),
            dataset=str(dataset_name),
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
        export_characterization_json(
            {
                "defense": str(defense_name),
                "dataset": str(dataset_name),
                **characterization.to_dict(),
            },
            json_output,
        )
        click.echo(f"JSON report written to {json_output}")

    sarif_output = kwargs.get("sarif_output")
    if sarif_output:
        export_characterization_sarif(characterization.to_dict(), sarif_output)
        click.echo(f"SARIF report written to {sarif_output}")

    # -------------------------------------------------------------------
    # 10. Log characterization accuracy indicator
    # -------------------------------------------------------------------
    # Paper Section 5.2.1 discusses identification accuracy; we report the
    # model-specific confidence score as a local proxy.
    obf_types = characterization.to_dict().get("obfuscation_types") or []
    confidence = float(characterization.to_dict().get("confidence", 0.0))

    if obf_types and confidence >= 0.7:
        logger.info(
            "High-confidence characterization (%.1f%%) achieved. "
            "Paper Section 5.2.1: see paper for aggregate study-level results.",
            confidence * 100,
        )
    elif obf_types and confidence < 0.7:
        logger.warning(
            "Moderate-confidence characterization (%.1f%%). "
            "Consider increasing --num-samples or --krylov-order for better resolution.",
            confidence * 100,
        )
    else:
        logger.info(
            "No strong obfuscation detected (confidence=%.1f%%). "
            "Defense may not employ gradient obfuscation.",
            confidence * 100,
        )


def _plot_spectrum(eigvals: np.ndarray, output_path: Path) -> None:
    """
    Plot eigenvalue spectrum for visualization.

    Paper Extended Results Appendix A: Eigenvalue spectrum plots
    show distinct patterns for shattered/vanishing/stochastic gradients.

    Patterns:
        Shattered    -> Large condition number, many near-zero eigenvalues
        Vanishing    -> All eigenvalues clustered near zero
        Stochastic   -> High variance in eigenvalue estimates

    Cross-ref: Paper Section 3.1.2 "Spectral Gradient Analysis"
    """
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eigvals, marker="o", markersize=2, linewidth=1)
    ax.set_title("Gradient Spectrum (Covariance Eigenvalues)")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.grid(True, alpha=0.3)

    # Add condition number annotation
    if len(eigvals) > 0 and eigvals[-1] != 0:
        cond_number = eigvals[0] / eigvals[-1]
        ax.text(
            0.02,
            0.98,
            f"Condition number kappa(H) = {cond_number:.2e}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
