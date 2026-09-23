"""
NeurInSpectre CLI - Command-line interface for adaptive adversarial evaluation.

Usage:
    neurinspectre attack --model resnet50.pth --defense jpeg --epsilon 0.03
    neurinspectre characterize --model resnet50.pth --defense jpeg
    neurinspectre evaluate --config eval.yaml

Version: 2.0
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

try:
    import rich_click as click
except Exception:  # pragma: no cover - fallback
    import click

logger = logging.getLogger(__name__)

_CLICK_COMMANDS = {
    "attack",
    "analyze",
    "baselines",
    "calibrate-thresholds",
    "characterize",
    "defense-analyzer",
    "doctor",
    "mitre-atlas",
    "drift-detect",
    "evaluate",
    "figures",
    "table2",
    "table2-smoke",
    "audit",
    "compare",
    "config",
    "score-ember2024-challenge",
    "missrate-report",
    "bypass-ledger",
    "run-capa",
    "transferability",
    "train-function-ml",
    "index-capa-supplement",
    "lookup-capa-functions",
    "download-ember2024",
    "download-ember2024-challenge",
    "download-ember2024-capa",
    "tag-pe-corpus",
    "diagnose-ember-audit",
    "scope-pe-corpus",
    "capa-diff-audit",
    "ember-pipeline-info",
    "engagement-gaps",
}


@click.group()
@click.version_option(version="2.0.0", prog_name="neurinspectre")
@click.option("--verbose", "-v", count=True, help="Increase verbosity (-v, -vv, -vvv)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output (errors only)")
@click.pass_context
def cli(ctx: click.Context, verbose: int, quiet: bool) -> None:
    """
    NeurInSpectre - Adaptive Adversarial Attack Framework

    Automatically characterizes gradient obfuscation defenses and
    synthesizes adaptive attacks to bypass them.

    \b
    Examples:
        # Run adaptive attack
        neurinspectre attack --model model.pth --defense jpeg --epsilon 0.03

        # Characterize defense
        neurinspectre characterize --model model.pth --defense jpeg

        # Full evaluation suite (Paper Section 5)
        neurinspectre evaluate --config evaluation.yaml

        # Audit a defense as a security pipeline
        neurinspectre audit --target jpeg-carmon --smoke
        neurinspectre audit --target ember-gbdt --smoke --pe-sample /path/to/pe_dir

    Cross-ref: Paper Section 3 "NEURINSPECTRE Framework"
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif verbose == 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose >= 3:
        logging.getLogger("neurinspectre").setLevel(logging.DEBUG)
        logging.getLogger("torch").setLevel(logging.DEBUG)

    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


@cli.command("attack")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to model file (.pth, .pt, or .onnx)",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity for this command",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    type=click.Choice(
        ["cifar10", "cifar100", "imagenet", "imagenet100", "ember", "nuscenes", "custom"]
    ),
    help="Dataset name",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    help="Path to custom dataset (required if dataset=custom)",
)
@click.option(
    "--labels-path",
    type=click.Path(exists=True),
    help="Path to nuScenes label map JSON (required for dataset=nuscenes)",
)
@click.option(
    "--nuscenes-version",
    type=str,
    default="v1.0-mini",
    show_default=True,
    help="nuScenes version string (e.g., v1.0-mini, v1.0-trainval)",
)
@click.option(
    "--defense",
    type=click.Choice(
        [
            "none",
            "jpeg",
            "bitdepth",
            "randsmooth",
            "thermometer",
            "distillation",
            "ensemble",
            "feature_squeezing",
            "gradient_regularization",
            "at_transform",
            "spatial_smoothing",
            "random_pad_crop",
            "rl_obfuscation",
            "tent",
            "certified_defense",
            "custom",
        ]
    ),
    default="none",
    help="Defense mechanism",
)
@click.option(
    "--defense-config",
    type=click.Path(exists=True),
    help="Path to defense configuration YAML",
)
@click.option(
    "--threshold-overrides",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON file with DefenseAnalyzer threshold overrides (e.g., from calibrate-thresholds)",
)
@click.option(
    "--attack-type",
    type=click.Choice(
        [
            "neurinspectre",
            "pgd",
            "apgd",
            "autoattack",
            "square",
            "fab",
            "bpda",
            "eot",
            "hybrid",
            "hybrid-volterra",
            "mapgd",
        ]
    ),
    default="neurinspectre",
    help="Attack algorithm (default: neurinspectre adaptive)",
)
@click.option(
    "--volterra-mode",
    type=click.Choice(["auto", "on", "off"]),
    default="auto",
    show_default=True,
    help="Volterra memory usage for adaptive attack (auto=characterization driven)",
)
@click.option(
    "--volterra-kernel",
    type=click.Choice(["power_law", "exponential", "uniform"]),
    default="power_law",
    show_default=True,
    help="Volterra kernel family (used when volterra-mode != off)",
)
@click.option(
    "--volterra-alpha",
    type=float,
    default=None,
    help="Override alpha_volterra (otherwise use characterization-recommended value)",
)
@click.option(
    "--volterra-memory-length",
    type=int,
    default=None,
    help="Override memory length k (otherwise use characterization-recommended value)",
)
@click.option(
    "--volterra-gradient-source",
    type=click.Choice(["pre_optimizer", "post_optimizer"]),
    default="pre_optimizer",
    show_default=True,
    help="Gradient source used when fitting Volterra alpha during Phase 1 characterization (confound control)",
)
@click.option(
    "--volterra-optimizer",
    type=click.Choice(["sgd", "sgd_momentum", "adam", "rmsprop"]),
    default="sgd",
    show_default=True,
    help="Optimizer model used only when --volterra-gradient-source=post_optimizer",
)
@click.option(
    "--epsilon",
    "-e",
    type=float,
    default=8 / 255,
    help="Perturbation budget (default: 8/255 for Linf)",
)
@click.option(
    "--norm",
    type=click.Choice(["Linf", "L2", "L1"]),
    default="Linf",
    help="Lp norm for perturbation budget",
)
@click.option(
    "--iterations",
    "-n",
    type=int,
    default=100,
    help="Number of attack iterations",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=128,
    help="Batch size for evaluation",
)
@click.option(
    "--num-samples",
    type=int,
    default=1000,
    help="Number of samples to attack",
)
@click.option(
    "--targeted/--untargeted",
    default=False,
    help="Targeted vs untargeted attack",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="attack_results.json",
    help="Output file for results",
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Export executive report JSON to path",
)
@click.option(
    "--sarif-output",
    type=click.Path(),
    help="Export SARIF report to path",
)
@click.option(
    "--report-format",
    type=click.Choice(["rich", "text"]),
    default="rich",
    help="Report output format",
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Print red-team findings to stdout",
)
@click.option(
    "--brief",
    is_flag=True,
    help="Show only critical metrics (scan-friendly output)",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Show executive summary only",
)
@click.option(
    "--color",
    is_flag=True,
    help="Force color output (overrides NO_COLOR)",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable color output",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress indicators",
)
@click.option(
    "--save-adversarials",
    type=click.Path(),
    help="Save adversarial examples to directory",
)
@click.option(
    "--save-features",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional features artifact path (JSONL). Writes one record with detector/characterization features + attack summary.",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps", "auto"]),
    default="cuda",
    help="Device for computation",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
@click.pass_context
def attack_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Run adaptive adversarial attack against defended model.

    This command implements Paper Algorithm 2 "NEURINSPECTRE Adaptive Attack".

    \b
    Examples:
        # Basic adaptive attack
        neurinspectre attack -m model.pth -d cifar10 --defense jpeg -e 0.03

        # Full configuration
        neurinspectre attack \\
            --model resnet50.pth \\
            --dataset imagenet \\
            --defense randsmooth \\
            --defense-config smooth_config.yaml \\
            --epsilon 0.5 --norm L2 \\
            --iterations 100 \\
            --output results.json

        # Compare multiple attacks
        for attack in neurinspectre pgd apgd autoattack; do
            neurinspectre attack -m model.pth -d cifar10 \\
                --attack-type $attack -o ${attack}_results.json
        done

    Cross-ref: Paper Section 3.2 "Phase 2: Adaptive Attack Synthesis"
    Cross-ref: Paper Table 1 "Attack success rate against evaluated defenses"
    """
    from .attack_cmd import run_attack

    run_attack(ctx, **kwargs)


@cli.command("analyze")
@click.option(
    "--model",
    "-m",
    required=False,
    type=str,
    help="Optional model path or name (defaults to a bundled CIFAR-10 TorchScript model when available)",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity for this command",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    type=click.Choice(
        ["cifar10", "cifar100", "imagenet", "imagenet100", "ember", "nuscenes", "custom"]
    ),
    help="Dataset name",
)
@click.option(
    "--data-path",
    type=click.Path(),
    help="Optional dataset root override (built-in datasets can auto-download if omitted)",
)
@click.option(
    "--labels-path",
    type=click.Path(exists=True),
    help="Path to nuScenes label map JSON (required for dataset=nuscenes)",
)
@click.option(
    "--nuscenes-version",
    type=str,
    default="v1.0-mini",
    show_default=True,
    help="nuScenes version string (e.g., v1.0-mini, v1.0-trainval)",
)
@click.option("--defense", required=True, type=str, help="Defense name (e.g., jpeg, bitdepth, randsmooth, ...)")
@click.option(
    "--epsilon",
    "-e",
    type=float,
    default=8 / 255,
    help="Perturbation budget (default: 8/255 for Linf)",
)
@click.option(
    "--norm",
    type=click.Choice(["Linf", "L2", "L1"]),
    default="Linf",
    help="Lp norm for perturbation budget",
)
@click.option(
    "--iterations",
    "-n",
    type=int,
    default=100,
    help="Number of attack iterations",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=128,
    help="Batch size for evaluation",
)
@click.option(
    "--num-samples",
    type=int,
    default=1000,
    help="Number of samples to attack",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results",
    show_default=True,
    help="Directory to write the default output JSON into",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file for results (overrides --output-dir default naming)",
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Print findings to stdout",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress indicators",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps", "auto"]),
    default="auto",
    help="Device for computation",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
@click.option(
    "--volterra-gradient-source",
    type=click.Choice(["pre_optimizer", "post_optimizer"]),
    default="pre_optimizer",
    show_default=True,
    help="Gradient source used when fitting Volterra alpha during Phase 1 characterization (confound control)",
)
@click.option(
    "--volterra-optimizer",
    type=click.Choice(["sgd", "sgd_momentum", "adam", "rmsprop"]),
    default="sgd",
    show_default=True,
    help="Optimizer model used only when --volterra-gradient-source=post_optimizer",
)
@click.option(
    "--save-features",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional features artifact path (JSONL). Writes one record with detector/characterization features + attack summary.",
)
@click.pass_context
def analyze_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Paper-style alias for running a single adaptive evasion evaluation.

    This wraps `neurinspectre attack` with `--attack-type neurinspectre` and a
    default output location under `results/`.
    """

    import os
    from pathlib import Path

    def _normalize_defense_name(raw: str) -> str:
        key = str(raw or "").strip().lower().replace("-", "_")
        key = key.replace(" ", "_")
        aliases = {
            "jpeg_compression": "jpeg",
            "jpegcompression": "jpeg",
            "bit_depth": "bitdepth",
            "bit_depth_reduction": "bitdepth",
            "randomized_smoothing": "randsmooth",
            "randsmoothing": "randsmooth",
            "featsqueeze": "feature_squeezing",
            "feature_squeeze": "feature_squeezing",
            "featuresqueezing": "feature_squeezing",
            "gradreg": "gradient_regularization",
            "gradient_reg": "gradient_regularization",
            "defensive_distillation": "distillation",
            "distill": "distillation",
            "at_transforms": "at_transform",
            "at_transformations": "at_transform",
            "spatial_smooth": "spatial_smoothing",
            "randpadcrop": "random_pad_crop",
            "randompadcrop": "random_pad_crop",
            "certified": "certified_defense",
        }
        canonical = {
            "none",
            "jpeg",
            "bitdepth",
            "randsmooth",
            "thermometer",
            "distillation",
            "ensemble",
            "feature_squeezing",
            "gradient_regularization",
            "at_transform",
            "spatial_smoothing",
            "random_pad_crop",
            "certified_defense",
        }
        key = aliases.get(key, key)
        if key not in canonical:
            raise click.ClickException(
                "Unknown defense name for `analyze`: "
                f"{raw!r}. Expected one of: {', '.join(sorted(canonical))}"
            )
        return key

    dataset = str(kwargs.get("dataset"))
    defense = _normalize_defense_name(str(kwargs.get("defense")))

    model = kwargs.get("model")
    if not model:
        # Default bundled models (best-effort). For other datasets, require explicit model.
        if dataset == "cifar10":
            for cand in (
                Path("models") / "cifar10_resnet20_norm_ts.pt",
                Path("models") / "cifar10_cnn_ts.pt",
            ):
                if cand.exists():
                    model = str(cand)
                    break
        if not model:
            raise click.ClickException(
                f"Missing --model and no default model found for dataset={dataset}. "
                "Provide --model pointing to a local model artifact."
            )

    # Default output path under results/
    output = kwargs.get("output")
    if not output:
        out_dir = Path(str(kwargs.get("output_dir") or "results"))
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"analyze_{dataset}_{defense}.json"
    output = str(output)

    # If user passed a data-path string that does not exist, do not fail at Click parsing time;
    # let dataset loaders auto-download for supported datasets.
    data_path = kwargs.get("data_path")
    if data_path:
        dp = Path(str(data_path))
        if not dp.exists():
            # Preserve the user intent but avoid a confusing "path must exist" error at CLI layer.
            os.makedirs(dp, exist_ok=True)

    from .attack_cmd import run_attack

    run_attack(
        ctx,
        model=str(model),
        verbose=kwargs.get("verbose", 0),
        dataset=dataset,
        data_path=data_path,
        labels_path=kwargs.get("labels_path"),
        nuscenes_version=kwargs.get("nuscenes_version"),
        defense=defense,
        defense_config=None,
        attack_type="neurinspectre",
        epsilon=float(kwargs.get("epsilon", 8 / 255)),
        norm=str(kwargs.get("norm", "Linf")),
        iterations=int(kwargs.get("iterations", 100)),
        batch_size=int(kwargs.get("batch_size", 128)),
        num_samples=int(kwargs.get("num_samples", 1000)),
        targeted=False,
        output=output,
        report=bool(kwargs.get("report", True)),
        report_format="text",
        brief=False,
        summary_only=False,
        color=False,
        no_color=False,
        no_progress=bool(kwargs.get("no_progress", False)),
        device=str(kwargs.get("device", "auto")),
        seed=int(kwargs.get("seed", 42)),
    )


@cli.command("characterize")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to model file",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity for this command",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    type=click.Choice(
        ["cifar10", "cifar100", "imagenet", "imagenet100", "ember", "nuscenes", "custom"]
    ),
    help="Dataset name",
)
@click.option("--data-path", type=click.Path(exists=True), help="Path to custom dataset")
@click.option(
    "--labels-path",
    type=click.Path(exists=True),
    help="Path to nuScenes label map JSON (required for dataset=nuscenes)",
)
@click.option(
    "--nuscenes-version",
    type=str,
    default="v1.0-mini",
    show_default=True,
    help="nuScenes version string (e.g., v1.0-mini, v1.0-trainval)",
)
@click.option(
    "--defense",
    type=click.Choice(
        [
            "none",
            "jpeg",
            "bitdepth",
            "randsmooth",
            "thermometer",
            "distillation",
            "ensemble",
            "feature_squeezing",
            "gradient_regularization",
            "at_transform",
            "spatial_smoothing",
            "random_pad_crop",
            "rl_obfuscation",
            "tent",
            "certified_defense",
            "custom",
        ]
    ),
    default="none",
    help="Defense mechanism to characterize",
)
@click.option(
    "--defense-config",
    type=click.Path(exists=True),
    help="Path to defense configuration",
)
@click.option(
    "--threshold-overrides",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON file with DefenseAnalyzer threshold overrides (e.g., from calibrate-thresholds)",
)
@click.option(
    "--use-bpda-approx",
    is_flag=True,
    help="Use BPDA approximation during characterization (for non-differentiable defenses)",
)
@click.option(
    "--krylov-order",
    "-k",
    type=int,
    default=20,
    help="Krylov subspace order (default: 20)",
)
@click.option(
    "--num-samples",
    type=int,
    default=100,
    help="Number of samples for characterization",
)
@click.option(
    "--volterra-gradient-source",
    type=click.Choice(["pre_optimizer", "post_optimizer"]),
    default="pre_optimizer",
    show_default=True,
    help="Gradient source used when fitting Volterra alpha (confound control)",
)
@click.option(
    "--volterra-optimizer",
    type=click.Choice(["sgd", "sgd_momentum", "adam", "rmsprop"]),
    default="sgd",
    show_default=True,
    help="Optimizer model used only when --volterra-gradient-source=post_optimizer",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="characterization.json",
    help="Output file for characterization results",
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Export executive report JSON to path",
)
@click.option(
    "--sarif-output",
    type=click.Path(),
    help="Export SARIF report to path",
)
@click.option(
    "--report-format",
    type=click.Choice(["rich", "text"]),
    default="rich",
    help="Report output format",
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Print red-team findings to stdout",
)
@click.option(
    "--brief",
    is_flag=True,
    help="Show only critical metrics",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Show executive summary only",
)
@click.option(
    "--color",
    is_flag=True,
    help="Force color output (overrides NO_COLOR)",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable color output",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress indicators",
)
@click.option(
    "--visualize",
    is_flag=True,
    help="Generate visualizations (eigenvalue spectrum, etc.)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps", "auto"]),
    default="cuda",
    help="Device for computation",
)
@click.pass_context
def characterize_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Characterize defense obfuscation type.

    Implements Paper Algorithm 1 "Spectral Gradient Analysis".
    Identifies shattered/stochastic/vanishing/exploding gradients.

    \b
    Examples:
        # Characterize defense
        neurinspectre characterize -m model.pth -d cifar10 --defense jpeg

        # With visualization
        neurinspectre characterize -m model.pth -d cifar10 \\
            --defense randsmooth --visualize

        # Custom Krylov order
        neurinspectre characterize -m model.pth -d cifar10 \\
            --defense thermometer -k 50

    Cross-ref: Paper Section 3.1 "Phase 1: Defense Characterization"
    Cross-ref: Paper Equation 9 "Krylov subspace"
    """
    from .characterize_cmd import run_characterization

    run_characterization(ctx, **kwargs)


cli.add_command(characterize_cmd, name="defense-analyzer")


@cli.command("compare")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["attacks", "defenses", "runs", "baseline", "characterization"]),
    default="attacks",
    help="Comparison mode",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity for this command",
)
@click.argument("input_files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON path",
)
@click.option(
    "--sort-by",
    default="asr",
    type=click.Choice(["asr", "confidence", "etd", "alpha", "variance"]),
    help="Sort key for comparison outputs",
)
@click.option(
    "--threshold",
    type=float,
    default=2.0,
    help="Significance threshold in percentage points",
)
@click.option(
    "--expected-asr-path",
    "--expected-asr",
    type=click.Path(exists=True),
    help="Expected ASR baseline YAML/JSON for --mode baseline (kept out of repo)",
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Export executive report JSON to path",
)
@click.option(
    "--sarif-output",
    type=click.Path(),
    help="Export SARIF report to path",
)
@click.option(
    "--report-format",
    type=click.Choice(["rich", "text"]),
    default="rich",
    help="Report output format",
)
@click.option(
    "--color",
    is_flag=True,
    help="Force color output (overrides NO_COLOR)",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable color output",
)
@click.pass_context
def compare_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Compare attacks, defenses, runs, or baselines side-by-side.

    \b
    Examples:
        # Attack comparison (Table 1 style)
        neurinspectre compare --mode attacks eval_results/summary.json

        # Defense ranking per attack
        neurinspectre compare --mode defenses eval_results/summary.json

        # Regression comparison between runs
        neurinspectre compare --mode runs run_a/summary.json run_b/summary.json --threshold 3.0

        # External baseline comparison (expected ASR kept out of repo)
        neurinspectre compare --mode baseline eval_results/summary.json \\
            --expected-asr-path /path/to/expected_asr.yaml

        # Characterization signal comparison
        neurinspectre compare --mode characterization char_results/*.json --sort-by alpha

    """
    from .compare_cmd import run_compare

    run_compare(ctx, **kwargs)


@cli.command("evaluate")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="YAML configuration file for evaluation",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity for this command",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="evaluation_results",
    help="Output directory for all results",
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Export executive report JSON to path",
)
@click.option(
    "--sarif-output",
    type=click.Path(),
    help="Export SARIF report to path",
)
@click.option(
    "--report-format",
    type=click.Choice(["rich", "text"]),
    default="rich",
    help="Report output format",
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Print red-team findings to stdout",
)
@click.option(
    "--brief",
    is_flag=True,
    help="Show only critical metrics",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Show executive summary only",
)
@click.option(
    "--color",
    is_flag=True,
    help="Force color output (overrides NO_COLOR)",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable color output",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress indicators",
)
@click.option(
    "--defenses",
    multiple=True,
    help="Specific defenses to evaluate (default: all in config)",
)
@click.option(
    "--attacks",
    multiple=True,
    help="Specific attacks to run (default: all)",
)
@click.option(
    "--parallel",
    "-j",
    type=int,
    default=1,
    help="Number of parallel workers",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume interrupted evaluation",
)
@click.option(
    "--smoke-test",
    is_flag=True,
    help="Run a minimal smoke subset (limits samples/iterations and selects 1 defense + 1 attack)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps", "auto"]),
    default="cuda",
    help="Device for computation",
)
@click.option(
    "--seeds",
    multiple=True,
    type=int,
    help="Run independent seeds and report mean ± std + 95% CI (repeat flag; recommend >=5 for paper tables)",
)
@click.option(
    "--num-seeds",
    type=int,
    default=1,
    show_default=True,
    help="If >1 and --seeds not set: use [seed, seed+1, ...] from config seed",
)
@click.pass_context
def evaluate_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Run full evaluation suite (Paper Section 5).

    Evaluates multiple defenses with ensemble of attacks:
    - APGD-CE
    - APGD-DLR
    - NEURINSPECTRE Adaptive
    - Square Attack (query-based)

    \b
    Example config (evaluation.yaml):
        defenses:
          - name: jpeg_compression
            type: jpeg
            quality: 75
          - name: randomized_smoothing
            type: randsmooth
            sigma: 0.25

        attacks:
          - neurinspectre
          - apgd
          - autoattack

        datasets:
          cifar10:
            path: ./data/cifar10
            num_samples: 1000

        perturbation:
          epsilon: 0.03
          norm: Linf

    \b
    Examples:
        # Full evaluation
        neurinspectre evaluate --config evaluation.yaml

        # Specific defenses only
        neurinspectre evaluate -c eval.yaml --defenses jpeg randsmooth

        # Parallel execution
        neurinspectre evaluate -c eval.yaml -j 4

        # Resume interrupted
        neurinspectre evaluate -c eval.yaml --resume

    Cross-ref: Paper Section 5 "Evaluation"
    Cross-ref: Paper Table 1 "Attack success rate against evaluated defenses"
    """
    from .evaluate_cmd import run_evaluation

    run_evaluation(ctx, **kwargs)


@cli.command("table2")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="YAML configuration file for Table 2 evaluation",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity for this command",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="results/table2",
    help="Output directory for all results",
)
@click.option(
    "--thresholds",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional JSON thresholds/overrides (from calibrate-thresholds) applied during Phase 1 characterization",
)
@click.option(
    "--strict-real-data/--no-strict-real-data",
    default=True,
    help="Require only real datasets and local assets",
)
@click.option(
    "--strict-dataset-budgets/--no-strict-dataset-budgets",
    default=True,
    help="Require and enforce dataset-specific attack budgets",
)
@click.option(
    "--allow-missing",
    is_flag=True,
    help="Allow missing local assets in strict checks",
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Export executive report JSON to path",
)
@click.option(
    "--sarif-output",
    type=click.Path(),
    help="Export SARIF report to path",
)
@click.option(
    "--report-format",
    type=click.Choice(["rich", "text"]),
    default="rich",
    help="Report output format",
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Print red-team findings to stdout",
)
@click.option(
    "--brief",
    is_flag=True,
    help="Show only critical metrics",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Show executive summary only",
)
@click.option(
    "--color",
    is_flag=True,
    help="Force color output (overrides NO_COLOR)",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable color output",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress indicators",
)
@click.option(
    "--defenses",
    multiple=True,
    help="Specific defenses to evaluate (default: all in config)",
)
@click.option(
    "--attacks",
    multiple=True,
    help="Specific attacks to run (default: all)",
)
@click.option(
    "--parallel",
    "-j",
    type=int,
    default=1,
    help="Number of parallel workers",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume interrupted evaluation",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps", "auto"]),
    default="auto",
    help="Device for computation",
)
@click.option(
    "--seeds",
    multiple=True,
    type=int,
    help="Run independent seeds and report mean ± std + 95% CI (repeat flag; recommend >=5 for paper tables)",
)
@click.option(
    "--num-seeds",
    type=int,
    default=1,
    show_default=True,
    help="If >1 and --seeds not set: use [seed, seed+1, ...] from config seed",
)
@click.pass_context
def table2_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Run Table 2 pipeline with strict real-data checks.

    This command normalizes Table 2 config variants, enforces real-dataset
    constraints, and then executes the standard evaluation matrix runner.
    """
    from .table2_cmd import run_table2

    run_table2(ctx, **kwargs)


@cli.command("table2-smoke")
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity for this command",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="results/table2_smoke_real",
    help="Output directory for all results",
)
@click.option(
    "--data-root",
    type=click.Path(),
    default="./data",
    help="Base directory used to discover datasets",
)
@click.option(
    "--models-root",
    type=click.Path(),
    default="./models",
    help="Base directory used to discover model artifacts",
)
@click.option(
    "--pgd-steps",
    type=int,
    default=10,
    show_default=True,
    help="PGD steps for smoke run",
)
@click.option(
    "--neurinspectre-steps",
    type=int,
    default=10,
    show_default=True,
    help="NeurInSpectre attack steps for smoke run",
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Export executive report JSON to path",
)
@click.option(
    "--sarif-output",
    type=click.Path(),
    help="Export SARIF report to path",
)
@click.option(
    "--report-format",
    type=click.Choice(["rich", "text"]),
    default="rich",
    help="Report output format",
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Print red-team findings to stdout",
)
@click.option(
    "--brief",
    is_flag=True,
    help="Show only critical metrics",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Show executive summary only",
)
@click.option(
    "--color",
    is_flag=True,
    help="Force color output (overrides NO_COLOR)",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable color output",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress indicators",
)
@click.option(
    "--defenses",
    multiple=True,
    help="Specific defenses to evaluate (default: all discovered)",
)
@click.option(
    "--attacks",
    multiple=True,
    help="Specific attacks to run (default: all)",
)
@click.option(
    "--parallel",
    "-j",
    type=int,
    default=1,
    help="Number of parallel workers",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume interrupted evaluation",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps", "auto"]),
    default="auto",
    help="Device for computation",
)
@click.pass_context
def table2_smoke_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Run a small, real-data Table2 smoke matrix.

    Discovers which datasets/models are available locally, generates a minimal
    config, then runs the standard `table2` pipeline with strict real-data
    checks (validity + integrity gates enabled).
    """

    from .table2_smoke_cmd import run_table2_smoke

    run_table2_smoke(ctx, **kwargs)


@cli.command("audit")
@click.option(
    "--target",
    type=click.Choice([
        "carmon",
        "jpeg-carmon",
        "ember-gbdt",
        "ember2024-gbdt",
        "ember2024-win32-gbdt",
        "ember2024-win64-gbdt",
        "ember2024-apk-gbdt",
        "ember2024-elf-gbdt",
        "ember2024-pdf-gbdt",
        "ember2024-dotnet-gbdt",
        "ember2024-all-gbdt",
    ]),
    default="carmon",
    show_default=True,
    help=(
        "Preset audit target: carmon, jpeg-carmon, ember-gbdt (EMBER2018), "
        "ember2024-gbdt (thrember v3 PE), ember2024-win32-gbdt, ember2024-win64-gbdt, "
        "ember2024-apk-gbdt, ember2024-elf-gbdt, ember2024-pdf-gbdt, "
        "ember2024-dotnet-gbdt, ember2024-all-gbdt"
    ),
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="results/audit",
    help="Output directory for audit_report.json and evaluate artifacts",
)
@click.option(
    "--model-path",
    type=click.Path(),
    default=None,
    help="Override Carmon2019Unlabeled.pt or ember_model_2018.txt",
)
@click.option(
    "--data-root",
    type=click.Path(),
    default="./data/cifar10",
    help="CIFAR-10 root, or EMBER feature root for --target ember-gbdt",
)
@click.option(
    "--n-examples",
    type=int,
    default=None,
    help="Evaluation samples (default 1000; smoke 8). ember-gbdt --pe-sample: detected-malware PEs",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Attack batch size",
)
@click.option(
    "--smoke",
    "--smoke-test",
    is_flag=True,
    help="Tiny official-AA subset (custom APGD-CE) to prove the linter wiring",
)
@click.option(
    "--assert-clean-accuracy/--no-assert-clean-accuracy",
    default=None,
    help="Run Carmon clean-acc sanity check on load (default: on unless --smoke)",
)
@click.option("--no-pgd", is_flag=True, help="Skip the cheap PGD column")
@click.option(
    "--mode",
    type=click.Choice(["whitebox", "scores", "labels", "feature", "problem", "all"]),
    default=None,
    help="Attack access: whitebox (AA/NI), scores, labels, or all (smoke default: all)",
)
@click.option(
    "--query-budgets",
    type=str,
    default=None,
    help="Comma-separated Square query budgets (default 10,50,100,500,5000; smoke 10,25,50)",
)
@click.option(
    "--pe-sample",
    "--pe-dir",
    type=click.Path(),
    default=None,
    help="PE file or directory for same-sample EMBER problem-space evaluation",
)
@click.option(
    "--benign-corpus",
    type=click.Path(),
    default=None,
    help="Optional benign PE file/dir whose bytes are used as GAMMA-padding payloads",
)
@click.option(
    "--require-official-reproduction",
    is_flag=True,
    help="Fail if EMBER v2 extraction is not Elastic-verified (Mac, or lief other than 0.9.0/0.10.1)",
)
@click.option(
    "--require-detected",
    is_flag=True,
    help="Fail if same-sample found no GBDT-detected malware PEs",
)
@click.option(
    "--filter-tags-json",
    type=click.Path(exists=True),
    default=None,
    help="Sidecar JSON/JSONL keyed by SHA-256 with per-file caps/ttps/mbc/family/file_type."
         " Build with neurinspectre run-capa <pe_dir> --sidecar tags.json, or"
         " scripts/tag_pe_corpus_from_ember2024.py for challenge-set tags.",
)
@click.option(
    "--filter-include-untagged/--filter-exclude-untagged",
    default=False,
    help="When Capa filters are active, whether to keep PE files whose SHA-256 is not in the sidecar",
)
@click.option("--filter-file-type", "filter_file_type", default=(), multiple=True,
              help="Capa filter (repeatable): Win32,Win64,Dot_Net,PDF,ELF,APK")
@click.option("--filter-family", "filter_family", default=(), multiple=True,
              help="Capa filter (repeatable): family substrings")
@click.option("--filter-tag", "filter_tag", default=(), multiple=True,
              help="Capa filter (repeatable): behavior/property/packer/exploit/group")
@click.option("--filter-ttp", "filter_ttp", default=(), multiple=True,
              help="Capa filter (repeatable): ATT&CK tactic/technique/ID")
@click.option("--filter-mbc", "filter_mbc", default=(), multiple=True,
              help="Capa filter (repeatable): MBC objective/behavior/ID")
@click.option("--filter-capability", "filter_capability", default=(), multiple=True,
              help="Capa filter (repeatable): Capa capability or namespace")
@click.option("--min-vt-detected", "min_vt_detected", type=int, default=None,
              help="Capa filter: minimum VirusTotal numerator (from detection_ratio)")
@click.option("--capa-preserve/--no-capa-preserve", default=False,
              help="Per-mutation capability-preservation gate (file-level Capa). "
                   "Rejects mutations that drop a capability the original file exhibited.")
@click.option("--capa-preserve-mode", type=click.Choice(["all", "ttps", "mbc"]), default="all",
              show_default=True,
              help="C8: strictness of the capa preservation gate. "
                   "all=every capability; ttps=only ATT&CK-tagged rules; "
                   "mbc=only Malware Behavior Catalog-tagged rules.")
@click.option("--capa-rules-dir", type=click.Path(), default=None,
              help="Override capa-rules directory (default: $CAPA_RULES or data/capa-rules)")
@click.option("--enable-section-slack/--no-section-slack", default=False,
              help="Add C7 section-slack padding to the problem-space transform mix. "
                   "Writes into file-alignment slack of the last section; preserves "
                   "function bytes and file length by construction.")
@click.option("--transform-set", type=click.Choice(["default", "combined"]), default="default",
              show_default=True,
              help="D10: 'default' mixes Full DOS + padding (+ section-slack when enabled); "
                   "'combined' applies Full DOS + section-slack + overlay padding in every "
                   "candidate (AdvMal-TF / PhantomCall multi-region envelope). Implies "
                   "--enable-section-slack.")
@click.option("--fulldos-quiet-only/--no-fulldos-quiet-only", default=False,
              help="E12: reject any candidate whose thrember pefilewarnings feature band "
                   "differs from the baseline. Makes the 'quiet' property a hard invariant "
                   "rather than an empirical accident.")
@click.option("--save-best-bytes/--no-save-best-bytes", default=False,
              help="D9: write each sample's best-of-search mutated PE to "
                   "<out>/best_bytes/<sha256>.mutated.bin so transferability "
                   "re-scoring against other detectors is possible offline.")
@click.option("--crossing-matrix/--no-crossing-matrix", default=False,
              help="After audit, score best_bytes on EMBER2018 + 2024 PE/Win32/Win64 "
                   "(requires --save-best-bytes). Crossing rule: clean p>=0.5, bytes "
                   "changed, mutated p<0.5.")
@click.option("--capa-diff-best/--no-capa-diff-best", default=False,
              help="One-shot Capa diff (vivisect/full) original vs each best_bytes "
                   "file (requires --save-best-bytes). Not a sandbox gate.")
@click.option("--capa-diff-backend", type=click.Choice(["file_level", "full"]),
              default="full", show_default=True,
              help="Capa backend for --capa-diff-best.")
@click.option("--write-diagnosis/--no-write-diagnosis", default=False,
              help="EMBER: write ember_audit_diagnosis.json beside audit_report.json.")
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps", "auto"]),
    default="auto",
    help="Device for computation",
)
@click.option("--seed", type=int, default=42, help="RNG seed")
@click.option("--verbose", "-v", count=True, help="Increase verbosity")
@click.option("--no-progress", is_flag=True, help="Disable progress bars")
@click.pass_context
def audit_cmd(ctx: click.Context, **kwargs) -> None:
    """
    Audit one defense+model as a security pipeline.

    Whitebox: official AA, AA+BPDA, NeurInSpectre. Practical modes: Square on
    scores or hard labels with ASR-vs-query curves. --smoke uses a custom
    APGD-CE subset plus short Square budgets.

    ember-gbdt: official Elastic LightGBM. Same-sample Full DOS / padding
    requires --pe-sample. Feature-space ASR is not PE-valid.
    """
    from .audit_cmd import run_audit

    if kwargs.get("assert_clean_accuracy") is None:
        kwargs["assert_clean_accuracy"] = not bool(kwargs.get("smoke"))
    run_audit(ctx, **kwargs)


@cli.command("score-ember2024-challenge")
@click.option(
    "--challenge-dir",
    type=click.Path(exists=True),
    default="data/ember/ember2024/dataset/challenge",
    show_default=True,
    help="Directory of unzipped EMBER2024 challenge JSONLs",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="results/ember2024/challenge_scoring.json",
    show_default=True,
    help="Where to write the per-model / per-file-type summary JSON",
)
@click.option(
    "--pe-model",
    type=click.Path(),
    default="data/ember/ember2024/EMBER2024_PE.model",
    show_default=True,
)
@click.option(
    "--win32-model",
    type=click.Path(),
    default="data/ember/ember2024/EMBER2024_Win32.model",
    show_default=True,
)
@click.option(
    "--win64-model",
    type=click.Path(),
    default="data/ember/ember2024/EMBER2024_Win64.model",
    show_default=True,
)
@click.option("--filter-file-type", "filter_file_type", default=(), multiple=True,
              help="Repeatable / comma list: Win32,Win64,Dot_Net,PDF,ELF,APK")
@click.option("--filter-family", "filter_family", default=(), multiple=True,
              help="Repeatable / comma list of family substrings")
@click.option("--filter-tag", "filter_tag", default=(), multiple=True,
              help="Repeatable / comma list matching behavior/property/packer/exploit/group")
@click.option("--filter-ttp", "filter_ttp", default=(), multiple=True,
              help="Repeatable / comma list matching ATT&CK tactic/technique/ID")
@click.option("--filter-mbc", "filter_mbc", default=(), multiple=True,
              help="Repeatable / comma list matching MBC objective/behavior/ID")
@click.option("--filter-capability", "filter_capability", default=(), multiple=True,
              help="Repeatable / comma list matching Capa capability or namespace")
@click.option("--min-vt-detected", "min_vt_detected", type=int, default=None,
              help="Minimum VirusTotal numerator (from detection_ratio 'X/Y')")
@click.option("--miss-min-support", "miss_min_support", type=int, default=20, show_default=True,
              help="Minimum #files carrying a label before it enters the miss scoreboard")
@click.option("--miss-limit-per-namespace", "miss_limit_per_namespace",
              type=int, default=25, show_default=True,
              help="How many top-miss-rate labels to keep per namespace in the JSON")
def score_ember2024_challenge_cmd(
    challenge_dir: str,
    output: str,
    pe_model: str,
    win32_model: str,
    win64_model: str,
    filter_file_type: tuple[str, ...],
    filter_family: tuple[str, ...],
    filter_tag: tuple[str, ...],
    filter_ttp: tuple[str, ...],
    filter_mbc: tuple[str, ...],
    filter_capability: tuple[str, ...],
    min_vt_detected: int | None,
    miss_min_support: int,
    miss_limit_per_namespace: int,
) -> None:
    """Score the EMBER 2024 challenge set with the three PE sub-models.

    Uses thrember's v3 raw features shipped inside each challenge JSONL
    record; no PE binaries required. Detection models are the shipped
    ``EMBER2024_PE.model`` / ``EMBER2024_Win32.model`` /
    ``EMBER2024_Win64.model``.

    Capa-informed filters compose as OR within a field, AND across fields;
    matching is case-insensitive substring and honors bracketed ATT&CK/MBC
    IDs (``T1055`` matches ``"Process Injection [T1055]"``).
    """
    from pathlib import Path

    # Import the standalone script's helper (single source of truth).
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "score_ember2024_challenge",
        Path(__file__).resolve().parent.parent.parent / "scripts" / "score_ember2024_challenge.py",
    )
    if spec is None or spec.loader is None:
        raise click.ClickException("Could not locate scripts/score_ember2024_challenge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from neurinspectre.malware.capa_filters import TagFilter, _parse_csv

    def _flatten(values: tuple[str, ...]) -> list[str]:
        """Repeated flag semantics: ``--filter-tag a --filter-tag b,c`` → [a, b, c]."""
        out: list[str] = []
        for v in values:
            out.extend(_parse_csv(v))
        return out

    flt = TagFilter(
        file_type=_flatten(filter_file_type),
        family=_flatten(filter_family),
        tag=_flatten(filter_tag),
        ttp=_flatten(filter_ttp),
        mbc=_flatten(filter_mbc),
        capability=_flatten(filter_capability),
        min_vt_detected=min_vt_detected,
    )
    summary = module._score(
        Path(challenge_dir),
        {"PE": pe_model, "Win32": win32_model, "Win64": win64_model},
        flt,
        miss_min_support=miss_min_support,
        miss_limit_per_namespace=miss_limit_per_namespace,
    )
    import json as _json

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps(summary, indent=2))
    click.echo(f"wrote {out_path}")
    if flt.is_active():
        click.echo(f"  filter: {flt.as_dict()}")
        click.echo(
            f"  seen={summary['n_records_total']}  "
            f"after_filter={summary['n_after_filter']}  "
            f"features_built={summary['n_features_built']}"
        )
    if summary["n_features_built"] == 0:
        click.echo("  no records survived filter+extract — nothing scored")
        return
    for name, cell in summary["per_model"].items():
        n = summary["n_features_built"]
        click.echo(
            f"  {name:5s}  det>=0.5={cell['detected_ge_0.5']}/{n} "
            f"({cell['detected_ge_0.5']/n:.3f})  median={cell['median_p']:.4f}"
        )
    click.echo(f"  n_miss_all_models: {summary['n_miss_all_models']}")


@cli.command("scope-pe-corpus")
@click.argument("pe_path", type=click.Path(exists=True))
@click.option("--challenge-dir", type=click.Path(exists=True), default=None,
              help="EMBER2024 challenge JSONL directory (SHA overlap preflight)")
@click.option("--supplement-index", type=click.Path(exists=True), default=None,
              help="EMBER2024 Capa supplement index JSON")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Write scope JSON (default: stdout)")
def scope_pe_corpus_cmd(pe_path, challenge_dir, supplement_index, output):
    """Hash a PE directory and report overlap with challenge + Capa supplement."""
    import json as _json
    from pathlib import Path
    from neurinspectre.malware.pe_scope import scope_pe_corpus

    pe_root = Path(pe_path)
    click.echo(f"Hashing MZ files under {pe_root} …", err=True)
    result = scope_pe_corpus(
        pe_root,
        challenge_dir=Path(challenge_dir) if challenge_dir else None,
        supplement_index=Path(supplement_index) if supplement_index else None,
    )
    click.echo(
        f"  {result['n_mz_files']} files; challenge overlap {result['n_overlap_challenge']}; "
        f"supplement overlap {result['n_overlap_capa_supplement']}",
        err=True,
    )
    text = _json.dumps(result, indent=2)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text)
        click.echo(f"wrote {output}")
    else:
        click.echo(text)


@cli.command("transferability")
@click.argument("report", type=click.Path(exists=True))
@click.option("--models", "-m", multiple=True, required=False,
              help="Repeatable: name=path pairs, e.g. -m PE=data/ember/ember2024/EMBER2024_PE.model")
@click.option("--default-crossing/--no-default-crossing", default=False,
              help="Use shipped EMBER2018 + PE + Win32 + Win64 checkpoints (skip missing).")
@click.option("--output", "-o", type=click.Path(),
              default=None, help="Write transferability JSON here (default: alongside report)")
def transferability_cmd(report, models, default_crossing, output):
    """D9: re-score an audit's best-of-search mutated PEs against additional
    EMBER 2024 sub-model detectors. Requires the audit to have been run
    with --save-best-bytes.
    """
    import json as _json
    from pathlib import Path
    from neurinspectre.evaluation.transferability import score_transferability
    from neurinspectre.evaluation.transferability import default_crossing_model_paths

    parsed: list[tuple[str, Path]] = []
    if default_crossing:
        parsed = [(n, p) for n, p in default_crossing_model_paths() if p.is_file()]
    for m in models or []:
        if "=" not in m:
            raise click.BadParameter(f"expected name=path, got {m!r}")
        name, path = m.split("=", 1)
        parsed.append((name.strip(), Path(path.strip())))
    if not parsed:
        raise click.ClickException("Pass -m name=path and/or --default-crossing")
    result = score_transferability(Path(report), parsed)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(_json.dumps(result, indent=2))
        click.echo(f"wrote {output}")
    else:
        click.echo(_json.dumps(result["summary"], indent=2))
        click.echo(f"n_samples: {result['n_samples']}")


@cli.command("train-function-ml")
@click.option("--supplement", type=click.Path(),
              default="data/ember/ember2024/capa", show_default=True)
@click.option("--index", type=click.Path(),
              default="data/ember/ember2024/capa_supplement_index.json", show_default=True)
@click.option("--output", "-o", type=click.Path(),
              default="results/ember2024/E11/function_ml_report.json", show_default=True)
@click.option("--top-k-labels", type=int, default=20, show_default=True)
@click.option("--target-per-capability", type=int, default=5000, show_default=True)
@click.option("--negative-pool-size", type=int, default=20000, show_default=True)
@click.option("--feature-dim", type=int, default=4096, show_default=True)
@click.option("--n-estimators", type=int, default=200, show_default=True)
@click.option("--max-records-scan", type=int, default=None,
              help="Cap total records read from shards (for smoke tests)")
@click.option("--seed", type=int, default=42, show_default=True)
def train_function_ml_cmd(supplement, index, output, top_k_labels,
                          target_per_capability, negative_pool_size,
                          feature_dim, n_estimators, max_records_scan, seed):
    """E11: train per-capability LightGBM classifiers on Capa supplement
    functions (opcode n-grams over disassembly)."""
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "train_function_ml",
        Path(__file__).resolve().parent.parent.parent / "scripts" / "train_function_ml.py",
    )
    if spec is None or spec.loader is None:
        raise click.ClickException("Could not locate scripts/train_function_ml.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [
        "--supplement", str(supplement),
        "--index", str(index),
        "--output", str(output),
        "--top-k-labels", str(top_k_labels),
        "--target-per-capability", str(target_per_capability),
        "--negative-pool-size", str(negative_pool_size),
        "--feature-dim", str(feature_dim),
        "--n-estimators", str(n_estimators),
        "--seed", str(seed),
    ]
    if max_records_scan is not None:
        argv += ["--max-records-scan", str(max_records_scan)]
    old = sys.argv[:]
    try:
        sys.argv = ["train_function_ml.py"] + argv
        exit_code = module.main()
    finally:
        sys.argv = old
    if exit_code:
        raise click.ClickException(f"training failed (exit {exit_code})")


@cli.command("index-capa-supplement")
@click.option("--supplement", type=click.Path(), default="data/ember/ember2024/capa",
              show_default=True, help="Root directory containing the Capa supplement shards")
@click.option("--output", "-o", type=click.Path(),
              default="data/ember/ember2024/capa_supplement_index.json", show_default=True)
@click.option("--limit-functions-per-file", type=int, default=None,
              help="Cap per-file function count (smoke tests)")
def index_capa_supplement_cmd(supplement, output, limit_functions_per_file):
    """Build a compact SHA-256 -> per-function Capa metadata index from the
    EMBER 2024 Capa supplement (~23.8 GB of shards).
    """
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "index_capa_supplement",
        Path(__file__).resolve().parent.parent.parent / "scripts" / "index_capa_supplement.py",
    )
    if spec is None or spec.loader is None:
        raise click.ClickException("Could not locate scripts/index_capa_supplement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ["--supplement", str(supplement), "--output", str(output)]
    if limit_functions_per_file:
        argv += ["--limit-functions-per-file", str(limit_functions_per_file)]
    import sys
    old = sys.argv[:]
    try:
        sys.argv = ["index_capa_supplement.py"] + argv
        exit_code = module.main()
    finally:
        sys.argv = old
    if exit_code:
        raise click.ClickException(f"indexing failed (exit {exit_code})")


@cli.command("lookup-capa-functions")
@click.argument("sha256")
@click.option("--index", type=click.Path(exists=True),
              default="data/ember/ember2024/capa_supplement_index.json", show_default=True,
              help="Capa supplement index JSON (build with index-capa-supplement)")
@click.option("--top-capabilities", type=int, default=15, show_default=True)
def lookup_capa_functions_cmd(sha256, index, top_capabilities):
    """Look up per-function Capa metadata for one SHA-256 in the supplement index.

    Reads only that hash out of the index. It does not load the 1.4 GB
    EMBER 2024 index into memory.
    """
    from pathlib import Path
    from neurinspectre.malware.capa_supplement_index import lookup_shas
    import json as _json
    funcs = lookup_shas(Path(index), [sha256]).get(sha256.lower(), [])
    if not funcs:
        click.echo(f"no supplement entries for sha256={sha256.lower()[:16]}...")
        return
    cap_counter: dict[str, int] = {}
    for f in funcs:
        for c in f.get("capa") or []:
            cap_counter[c] = cap_counter.get(c, 0) + 1
    top = sorted(cap_counter.items(), key=lambda kv: -kv[1])[:top_capabilities]
    click.echo(_json.dumps({
        "sha256": sha256.lower(),
        "n_functions": len(funcs),
        "n_unique_capabilities": len(cap_counter),
        "top_capabilities": top,
        "functions": funcs[:20],
    }, indent=2))


@cli.command("run-capa")
@click.argument("pe_path", type=click.Path(exists=True))
@click.option("--backend", type=click.Choice(["file_level", "full"]), default="file_level",
              show_default=True,
              help="file_level = pefile (fast, ~1s); full = vivisect (slow, ~minutes)")
@click.option("--rules-dir", type=click.Path(), default=None,
              help="capa-rules directory (default: $CAPA_RULES or data/capa-rules)")
@click.option("--diff-against", type=click.Path(exists=True), default=None,
              help="Second file: emit capa_diff (dropped/added/preserved) between the two")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Write result JSON to this path (default: stdout)")
@click.option("--sidecar", type=click.Path(), default=None,
              help="Also write a --filter-tags-json sidecar keyed by SHA-256")
@click.option("--supplement-index", type=click.Path(exists=True), default=None,
              help="EMBER 2024 Capa supplement index. Matching SHA-256s get their "
                   "function labels merged into the result. Hashes that are absent "
                   "stay file-level only.")
def run_capa_cmd(pe_path, backend, rules_dir, diff_against, output, sidecar, supplement_index):
    """Run Capa on one PE or every MZ file in a directory.

    Uses ``neurinspectre.malware.capa_scan``. The ``file_level`` backend
    matches only file-scoped rules (packer / property / compiler / string
    patterns) — fast enough for per-mutation gating. The ``full`` backend
    disassembles with vivisect and matches function-level rules too, but
    takes minutes per binary.

    ``--sidecar`` writes the tag file ``neurinspectre audit --filter-tags-json``
    consumes. ``--supplement-index`` adds EMBER 2024 function-level labels
    only for SHA-256s present in that index.
    """
    import hashlib as _hash
    import json as _json
    from pathlib import Path

    try:
        from neurinspectre.malware.capa_scan import (
            capabilities_file_level, capabilities_full, capa_diff as _diff,
            filter_sidecar_record, metadata_for_matches, CapaUnavailable,
        )
    except ImportError as exc:
        raise click.ClickException(f"capa wrapper import failed: {exc}")

    fn = capabilities_file_level if backend == "file_level" else capabilities_full
    rules_p = Path(rules_dir) if rules_dir else None
    root = Path(pe_path)
    if diff_against and root.is_dir():
        raise click.ClickException("--diff-against applies to one file, not a directory")

    def _mz_files(path: Path):
        if path.is_file():
            yield path
            return
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file():
                continue
            if any(part.startswith(".") for part in candidate.parts):
                continue
            try:
                with candidate.open("rb") as fh:
                    magic = fh.read(2)
            except OSError:
                continue
            if magic == b"MZ":
                yield candidate

    files = list(_mz_files(root))
    if root.is_dir() and not files:
        raise click.ClickException(f"no MZ files under {root}")

    supplement_by_sha = {}
    if supplement_index and not diff_against:
        from neurinspectre.malware.capa_supplement_index import lookup_shas
        digests = []
        for path in files:
            digests.append(_hash.sha256(path.read_bytes()).hexdigest())
        supplement_by_sha = lookup_shas(Path(supplement_index), digests)

    rows = []
    try:
        if diff_against:
            original = root.read_bytes()
            other = Path(diff_against).read_bytes()
            result = {"path": str(root), "backend": backend, "other_path": str(diff_against)}
            result.update(_diff(original, other, backend=backend, rules_dir=rules_p))
            text = _json.dumps(result, indent=2)
            if output:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_text(text)
                click.echo(f"wrote {output}")
            else:
                click.echo(text)
            return

        for index, path in enumerate(files, start=1):
            data = path.read_bytes()
            sha = _hash.sha256(data).hexdigest()
            err = None
            try:
                caps = fn(data, rules_dir=rules_p)
            except CapaUnavailable:
                raise
            except Exception as exc:
                caps = frozenset()
                err = f"{type(exc).__name__}: {exc}"
            if root.is_dir():
                click.echo(
                    f"[run-capa] {index}/{len(files)} {path.name} n={len(caps)}",
                    err=True,
                )
            rows.append({
                "path": str(path),
                "sha256": sha,
                "capabilities": caps,
                "error": err,
                "supplement": supplement_by_sha.get(sha) or [],
            })
    except CapaUnavailable as exc:
        raise click.ClickException(str(exc))

    matched = set()
    for row in rows:
        matched.update(row["capabilities"])
    meta = metadata_for_matches(frozenset(matched), rules_dir=rules_p) if matched else {}
    sidecar_obj = {}
    file_results = []
    for row in rows:
        record = filter_sidecar_record(
            sha256=row["sha256"],
            capabilities=row["capabilities"],
            metadata=meta,
            supplement_functions=row["supplement"],
            path=row["path"],
            error=row["error"],
        )
        sidecar_obj[row["sha256"]] = record
        file_results.append({
            "path": row["path"],
            "sha256": row["sha256"],
            "backend": backend,
            "n_capabilities": len(row["capabilities"]),
            "capabilities": sorted(row["capabilities"]),
            "n_attack": len(record["ttps"]),
            "n_mbc": len(record["mbc"]),
            "in_ember2024_capa_supplement": record["in_ember2024_capa_supplement"],
            "supplement_n_functions": record["supplement_n_functions"],
            "supplement_capabilities": record["supplement_capabilities"],
            "error": row["error"],
        })

    if len(file_results) == 1 and not root.is_dir():
        result = file_results[0]
    else:
        result = {
            "kind": "run-capa",
            "backend": backend,
            "n_files": len(file_results),
            "n_errors": sum(1 for row in file_results if row["error"]),
            "n_in_ember2024_capa_supplement": sum(
                1 for row in file_results if row["in_ember2024_capa_supplement"]
            ),
            "files": file_results,
        }
    text = _json.dumps(result, indent=2)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text)
        click.echo(f"wrote {output}")
    else:
        click.echo(text)
    if sidecar:
        side = Path(sidecar)
        side.parent.mkdir(parents=True, exist_ok=True)
        side.write_text(_json.dumps(sidecar_obj, indent=2) + "\n")
        click.echo(f"wrote sidecar {side} ({len(sidecar_obj)} files)", err=True)


@cli.command("bypass-ledger")
@click.argument("report", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Where to write the ledger JSON (default: sibling ember_bypass_ledger.json)")
@click.option("--markdown", "markdown_path", type=click.Path(), default=None,
              help="Also render a Markdown table to this path")
@click.option("--close-call-min-delta", type=float, default=0.05, show_default=True,
              help="Minimum clean_p-best_p drop for a non-flipped row to show up as a close call")
@click.option("--top-n-tags", type=int, default=8, show_default=True,
              help="Top-N ATT&CK / MBC / Capa labels per row")
@click.option("--limit-close-calls", type=int, default=25, show_default=True,
              help="Cap the close-calls list to N rows (0 = unlimited)")
def bypass_ledger_cmd(report, output, markdown_path, close_call_min_delta, top_n_tags,
                      limit_close_calls):
    """Build a Capa-tagged claim ledger from an audit_report.json.

    Emits one row per successful problem-space bypass (best_p < 0.5) plus
    one row per close call (score dropped by >=--close-call-min-delta but
    did not flip). Every row carries the original file's ATT&CK, MBC and
    Capa provenance so the "still malware" evidence rides alongside the
    parse-valid ASR.
    """
    import json as _json
    from pathlib import Path

    from neurinspectre.malware.bypass_ledger import (
        build_ledger,
        read_report,
        render_markdown,
    )

    audit = read_report(report)
    ledger = build_ledger(
        audit,
        close_call_min_delta=close_call_min_delta,
        top_n_tags=top_n_tags,
        limit=(limit_close_calls if limit_close_calls else None),
    )
    report_path = Path(report)
    if report_path.is_dir():
        base = report_path
    else:
        base = report_path.parent
    out = Path(output) if output else (base / "ember_bypass_ledger.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(ledger, indent=2))
    click.echo(f"wrote {out}")
    click.echo(
        f"  n_kept={ledger['n_kept']} n_flipped={ledger['n_flipped']} "
        f"n_close_calls={ledger['n_close_calls']}"
    )
    if markdown_path:
        md_path = Path(markdown_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(ledger))
        click.echo(f"  markdown -> {md_path}")


@cli.command("missrate-report")
@click.argument("scoring_json", type=click.Path(exists=True))
@click.option("--namespace", "namespaces", multiple=True, default=(),
              help="Repeatable: which namespaces to print (default: all present in JSON)")
@click.option("--top", "top_n", type=int, default=10, show_default=True,
              help="How many top rows per namespace to print")
def missrate_report_cmd(scoring_json, namespaces, top_n):
    """Pretty-print the per-label all-model-miss scoreboard from a scoring JSON.

    Input is any ``challenge_scoring.json`` produced by
    ``neurinspectre score-ember2024-challenge``. No models are re-run — this
    is a re-slicer on the already-computed ``miss_cohorts`` block.
    """
    import json as _json

    data = _json.loads(Path(scoring_json).read_text())
    cohorts = data.get("miss_cohorts") or {}
    if not cohorts:
        raise click.ClickException(
            f"{scoring_json} has no 'miss_cohorts' block. Re-run "
            "`neurinspectre score-ember2024-challenge` to add it."
        )
    from neurinspectre.malware.miss_cohorts import summarize_scoreboard

    if namespaces:
        cohorts = {k: v for k, v in cohorts.items() if k in namespaces}
    click.echo(summarize_scoreboard(cohorts, top_n=top_n))


@cli.command("download-ember2024")
@click.option("--dest", type=click.Path(), default="data/ember/ember2024", show_default=True,
              help="Destination directory for model files")
@click.option("--models", "-m", multiple=True, default=(),
              help="Repeatable: specific EMBER2024 model filenames (default: PE + Win32 + Win64)")
@click.option("--all", "all_", is_flag=True,
              help="Download every .model file in the HuggingFace repo")
@click.option("--manifest", type=click.Path(), default=None,
              help="Where to write the SHA-256 manifest (default: <dest>/download_manifest.json)")
def download_ember2024_cmd(dest, models, all_, manifest):
    """Fetch EMBER2024 LightGBM detection models with SHA-256 manifest.

    Wraps ``scripts/download_ember2024.py``.
    """
    from pathlib import Path
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "download_ember2024",
        Path(__file__).resolve().parent.parent.parent / "scripts" / "download_ember2024.py",
    )
    if spec is None or spec.loader is None:
        raise click.ClickException("Could not locate scripts/download_ember2024.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ["--dest", str(dest)]
    if manifest:
        argv += ["--manifest", str(manifest)]
    if all_:
        argv.append("--all")
    if models:
        argv += ["--models", *models]
    import sys
    old = sys.argv[:]
    try:
        sys.argv = ["download_ember2024.py"] + argv
        exit_code = module.main()
    finally:
        sys.argv = old
    if exit_code:
        raise click.ClickException(f"download failed (exit {exit_code})")


@cli.command("download-ember2024-challenge")
@click.option("--dest", type=click.Path(), default="data/ember/ember2024/dataset", show_default=True,
              help="Destination directory for the challenge archive + unzipped JSONLs")
@click.option("--manifest", type=click.Path(), default=None,
              help="Path for the SHA-256 manifest (default: <dest>/challenge_manifest.json)")
def download_ember2024_challenge_cmd(dest, manifest):
    """Fetch the 32 MB EMBER2024 challenge JSONLs (features only, no PE binaries)."""
    from pathlib import Path
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "download_ember2024_challenge",
        Path(__file__).resolve().parent.parent.parent / "scripts" / "download_ember2024_challenge.py",
    )
    if spec is None or spec.loader is None:
        raise click.ClickException("Could not locate scripts/download_ember2024_challenge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ["--dest", str(dest)]
    if manifest:
        argv += ["--manifest", str(manifest)]
    import sys
    old = sys.argv[:]
    try:
        sys.argv = ["download_ember2024_challenge.py"] + argv
        exit_code = module.main()
    finally:
        sys.argv = old
    if exit_code:
        raise click.ClickException(f"download failed (exit {exit_code})")


@cli.command("download-ember2024-capa")
@click.option("--dest", type=click.Path(), default="data/ember/ember2024/capa", show_default=True)
@click.option("--manifest", type=click.Path(), default=None)
@click.option("--all", "all_", is_flag=True, help="Grab every shard (23.8 GB)")
@click.option("--smallest", type=int, default=0,
              help="Grab N smallest shards (sniff a subset)")
@click.option("--split", type=click.Choice(["train", "test"]), default=None)
@click.option("--file-type", type=click.Choice(["Win32", "Win64"]), default=None)
def download_ember2024_capa_cmd(dest, manifest, all_, smallest, split, file_type):
    """Fetch shards of the EMBER 2024 Capa supplement (function-level, up to 23.8 GB)."""
    from pathlib import Path
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "download_ember2024_capa",
        Path(__file__).resolve().parent.parent.parent / "scripts" / "download_ember2024_capa.py",
    )
    if spec is None or spec.loader is None:
        raise click.ClickException("Could not locate scripts/download_ember2024_capa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ["--dest", str(dest)]
    if manifest:
        argv += ["--manifest", str(manifest)]
    if all_:
        argv.append("--all")
    if smallest:
        argv += ["--smallest", str(smallest)]
    if split:
        argv += ["--split", split]
    if file_type:
        argv += ["--file-type", file_type]
    import sys
    old = sys.argv[:]
    try:
        sys.argv = ["download_ember2024_capa.py"] + argv
        exit_code = module.main()
    finally:
        sys.argv = old
    if exit_code:
        raise click.ClickException(f"download failed (exit {exit_code})")


@cli.command("tag-pe-corpus")
@click.option("--pe-dir", type=click.Path(exists=True), required=True,
              help="Directory of PE files to hash and look up in EMBER2024")
@click.option("--dataset-dir", type=click.Path(exists=True),
              default="data/ember/ember2024/dataset/challenge", show_default=True,
              help="EMBER2024 dataset directory (JSONLs)")
@click.option("--output", "-o", type=click.Path(), required=True,
              help="Where to write the SHA-256 -> tag-record sidecar JSON")
@click.option("--keep-fields",
              default="sha256,file_type,family,behavior,file_property,packer,exploit,group,caps,ttps,mbc,detection_ratio",
              show_default=True,
              help="Comma list of dataset fields to copy into the sidecar")
def tag_pe_corpus_cmd(pe_dir, dataset_dir, output, keep_fields):
    """Build a Capa-tag sidecar for a PE corpus by SHA-256 lookup in EMBER2024.

    Output is compatible with ``neurinspectre audit --filter-tags-json``.
    """
    from pathlib import Path
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tag_pe_corpus_from_ember2024",
        Path(__file__).resolve().parent.parent.parent / "scripts" / "tag_pe_corpus_from_ember2024.py",
    )
    if spec is None or spec.loader is None:
        raise click.ClickException("Could not locate scripts/tag_pe_corpus_from_ember2024.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [
        "--pe-dir", str(pe_dir),
        "--dataset-dir", str(dataset_dir),
        "--output", str(output),
        "--keep-fields", keep_fields,
    ]
    import sys
    old = sys.argv[:]
    try:
        sys.argv = ["tag_pe_corpus_from_ember2024.py"] + argv
        exit_code = module.main()
    finally:
        sys.argv = old
    if exit_code:
        raise click.ClickException(f"tagging failed (exit {exit_code})")


@cli.command("diagnose-ember-audit")
@click.argument("report", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Diagnosis JSON path (default: sibling ember_audit_diagnosis.json)")
def diagnose_ember_audit_cmd(report, output):
    """Summarize an EMBER same-sample audit report into a compact diagnosis JSON."""
    import json as _json
    from pathlib import Path

    from neurinspectre.evaluation.ember_audit_diagnosis import (
        load_ember_audit_report,
        summarize_ember_audit_report,
    )

    report_path = Path(report)
    diag = summarize_ember_audit_report(load_ember_audit_report(report_path))
    out = Path(output) if output else None
    if out is None:
        base = report_path if report_path.is_dir() else report_path.parent
        out = base / "ember_audit_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(diag, indent=2, default=str), encoding="utf-8")
    click.echo(f"wrote {out}")
    if diag.get("n") == 0:
        raise click.ClickException("diagnosis: no GBDT-detected samples in report")


@cli.command("capa-diff-audit")
@click.argument("report", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Write capa_diff_audit.json (default: sibling of report)")
@click.option("--capa-rules-dir", type=click.Path(), default=None,
              help="Override capa-rules directory")
@click.option("--backend", "capa_diff_backend",
              type=click.Choice(["file_level", "full"]), default="full", show_default=True)
@click.option("--max-samples", type=int, default=None,
              help="Limit rows from best_bytes_manifest")
def capa_diff_audit_cmd(report, output, capa_rules_dir, capa_diff_backend, max_samples):
    """Capa diff original vs audit best_bytes PEs (post-hoc; requires --save-best-bytes audit)."""
    import json as _json
    from pathlib import Path

    from neurinspectre.evaluation.capa_diff_audit import capa_diff_best_bytes_report

    report_path = Path(report)
    if report_path.is_dir():
        report_path = report_path / "audit_report.json"
    capa_report = capa_diff_best_bytes_report(
        report_path,
        rules_dir=Path(capa_rules_dir) if capa_rules_dir else None,
        backend=str(capa_diff_backend),
        max_samples=max_samples,
    )
    out = Path(output) if output else report_path.parent / "capa_diff_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(capa_report, indent=2, default=str), encoding="utf-8")
    click.echo(
        f"wrote {out} n_scanned={capa_report.get('n_scanned')} "
        f"errors={capa_report.get('n_errors')}"
    )


@cli.command("engagement-gaps")
@click.option("--json/--no-json", "as_json", default=True, show_default=True,
              help="Print engagement gap catalog as JSON")
def engagement_gaps_cmd(as_json):
    """List SOW boundaries: sandbox, AV/EDR, GAMMA section injection, IAT/graph (not CLI attacks)."""
    import json as _json

    from neurinspectre.malware.measurement_scope import engagement_gaps_summary

    summary = engagement_gaps_summary()
    if as_json:
        click.echo(_json.dumps(summary, indent=2))
        return
    click.echo(summary["cli_policy"])
    for row in summary["not_measured"]:
        click.echo(f"  - {row['id']}: {row.get('client_question')}")


@cli.command("ember-pipeline-info")
@click.option(
    "--target",
    required=True,
    help="Audit target (ember-gbdt, ember2024-gbdt, jpeg-carmon, carmon, …)",
)
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps", "auto"]), default="cpu",
              show_default=True)
@click.option("--json/--no-json", "as_json", default=True, show_default=True,
              help="Print pipeline characterization JSON")
def ember_pipeline_info_cmd(target, device, as_json):
    """Print SecurityPipeline characterization + measurement_scope for an audit target."""
    import json as _json

    from neurinspectre.cli.audit_cmd import characterize_audit_pipeline

    info = characterize_audit_pipeline(target, device=device)
    if as_json:
        click.echo(_json.dumps(info, indent=2, default=str))
    else:
        click.echo(str(info))


@cli.command("doctor")
@click.option("--json-output", type=click.Path(), help="Write environment report JSON to path")
@click.option("--as-json", is_flag=True, help="Print environment report JSON to stdout")
@click.option(
    "--models-dir",
    type=click.Path(),
    default="models",
    show_default=True,
    help="Models directory to scan for stub metadata",
)
@click.option("--check-models/--no-check-models", default=True, help="Scan models dir for stub markers")
@click.pass_context
def doctor_cli_cmd(ctx: click.Context, **kwargs) -> None:
    """Environment + dependency sanity checks (no network)."""
    from .doctor_cmd import run_doctor

    run_doctor(ctx, **kwargs)


@cli.command("drift-detect")
@click.option("--reference", "-r", required=True, type=click.Path(exists=True), help="Reference data (.npy/.npz)")
@click.option("--current", "-c", required=True, type=click.Path(exists=True), help="Current data (.npy/.npz)")
@click.option(
    "--methods",
    default="hotelling,ks,bayesian",
    show_default=True,
    help="Comma-separated methods: hotelling, ks, mmd, ks_ad_cvm, bayesian",
)
@click.option("--confidence-level", type=float, default=0.95, show_default=True, help="Confidence level")
@click.option("--output", "-o", type=click.Path(), help="Write JSON results to this path (otherwise prints JSON)")
@click.option("--plot", type=click.Path(), help="Optional PNG plot path for a drift summary visualization")
@click.option("--plot-feature-index", type=int, default=0, show_default=True, help="Feature index to plot (default: 0)")
def drift_detect_cmd(
    reference: str,
    current: str,
    methods: str,
    confidence_level: float,
    output: str | None,
    plot: str | None,
    plot_feature_index: int,
) -> None:
    """Multivariate drift detection (Hotelling / KS / MMD / Bayesian CP)."""
    from .drift_detect_cmd import run_drift_detect

    payload, out_path = run_drift_detect(
        reference=reference,
        current=current,
        methods=methods,
        confidence_level=confidence_level,
        output=output,
        plot=plot,
        plot_feature_index=plot_feature_index,
    )
    if out_path:
        click.echo(str(out_path))
    else:
        click.echo(payload)


@cli.command("config")
@click.argument("config_type", type=click.Choice(["attack", "defense", "evaluation", "audit"]))
@click.option("--output", "-o", type=click.Path(), help="Output file (default: stdout)")
@click.option(
    "--target",
    type=click.Choice([
        "carmon",
        "jpeg-carmon",
        "ember-gbdt",
        "ember2024-gbdt",
        "ember2024-win32-gbdt",
        "ember2024-win64-gbdt",
        "ember2024-apk-gbdt",
        "ember2024-elf-gbdt",
        "ember2024-pdf-gbdt",
        "ember2024-dotnet-gbdt",
        "ember2024-all-gbdt",
    ]),
    default="carmon",
    show_default=True,
    help="Preset for `config audit`",
)
@click.option("--smoke/--full", default=True, show_default=True, help="Smoke vs full budgets for `config audit`")
@click.option("--pe-sample", "--pe-dir", type=click.Path(), default=None, help="PE file/dir for ember-gbdt same-sample")
@click.option("--benign-corpus", type=click.Path(), default=None, help="Benign PE file/dir for GAMMA-padding payloads")
def config_cmd(
    config_type: str,
    output: str | None,
    target: str,
    smoke: bool,
    pe_sample: str | None,
    benign_corpus: str | None,
) -> None:
    """
    Generate example configuration files.

    \b
    Examples:
        neurinspectre config attack > attack.yaml
        neurinspectre config evaluation -o evaluation.yaml
        neurinspectre config audit --target ember-gbdt --smoke --pe-sample ./pe_dir
    """
    if config_type == "audit":
        import yaml

        from .audit_cmd import build_audit_config

        payload = build_audit_config(
            target=target,
            n_examples=8 if smoke else 1000,
            smoke=smoke,
            pe_sample=pe_sample,
            benign_corpus=benign_corpus,
        )
        config_str = yaml.safe_dump(payload, sort_keys=False)
    else:
        from .config import generate_example_config

        config_str = generate_example_config(config_type)

    if output:
        Path(output).write_text(config_str)
        click.echo(f"Configuration written to {output}")
    else:
        click.echo(config_str)


# Baseline comparison harness (Issue 4).
from .baselines_cmd import baselines_cmd as baselines_cli_cmd  # noqa: E402

cli.add_command(baselines_cli_cmd)

# Tier 2: ROC/AUC threshold calibration.
from .calibrate_thresholds_cmd import calibrate_thresholds_cmd as calibrate_thresholds_cli_cmd  # noqa: E402

cli.add_command(calibrate_thresholds_cli_cmd)

# Paper figure generation (Issue: paper figures -> reproducible CLI).
from .figures_cmd import figures_cmd as figures_cli_cmd  # noqa: E402

cli.add_command(figures_cli_cmd)

# MITRE ATLAS validation/coverage (Tier 2 rigor; offline STIX).
from .mitre_atlas_cmd import mitre_atlas_cmd as mitre_atlas_cli_cmd  # noqa: E402

cli.add_command(mitre_atlas_cli_cmd)


def main() -> None:
    """Main CLI entry point"""
    try:
        argv = sys.argv[1:]
        if argv and argv[0] not in _CLICK_COMMANDS and argv[0] not in {"-h", "--help", "--version"}:
            # AE/ops ergonomics: NeurInSpectre ships a large legacy argparse CLI
            # (`neurinspectre.cli.__main__`) for several extended modules that are
            # not yet ported to Click. For a small, explicit allowlist we delegate
            # automatically (no env var required) so copy/paste reproduction commands
            # work out of the box.
            legacy_allowlist = {
                # Argument names as users actually type them
                "gradient-inversion",
                "gradient_inversion",
                "rl-obfuscation",
                "attention-security",
                "adversarial-ednn",
                "subnetwork_hijack",
                "activation_steganography",
                "activation-steganography",
                "statistical_evasion",
                "statistical-evasion",
                # Debug/infra helpers in legacy CLI
                "gpu",
                "obfuscated-gradient",
            }
            enable_legacy = str(os.environ.get("NEURINSPECTRE_ENABLE_LEGACY_CLI", "")).lower() in {
                "1",
                "true",
                "yes",
            }
            if enable_legacy or argv[0] in legacy_allowlist:
                from .__main__ import main as legacy_main

                legacy_main()
                return
            print(
                f"[ERROR] Unknown command: {argv[0]!r}\n"
                "Legacy CLI fallback is disabled by default.\n"
                "Run `neurinspectre --help` to see supported commands.",
                file=sys.stderr,
            )
            sys.exit(2)
        cli(obj={})
    except Exception as e:  # pragma: no cover - CLI error handling
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
