#!/usr/bin/env python3
"""
NeurInSpectre Attack Modules CLI Commands

Adds direct CLI entrypoints for the packaged attack modules so they can be invoked as:
  neurinspectre <attack> ...

These commands are wrappers around the existing `neurinspectre.attacks.*` implementations.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_attack_commands(subparsers) -> None:
    """Register attack subcommands with the main CLI."""
    # TS-Inverse (gradient inversion) attack
    ts = subparsers.add_parser(
        "ts-inverse",
        aliases=["ts_inverse", "ts-inverse-attack", "ts_inverse_attack"],
        help="🔴 TS-Inverse gradient inversion attack (wrapper)",
    )
    ts.add_argument("--target-gradients", dest="target_gradients", required=True, help="Input gradients .npy")
    ts.add_argument(
        "--model-factory",
        dest="model_factory",
        default=None,
        help="Python factory 'module:function' that returns a torch.nn.Module (optional)",
    )
    ts.add_argument(
        "--model-kwargs",
        dest="model_kwargs",
        default=None,
        help="JSON dict of kwargs to pass to the model factory (optional)",
    )
    ts.add_argument(
        "--allow-demo-model",
        dest="allow_demo_model",
        action="store_true",
        help="Allow using the built-in demo model (explicit demo mode)",
    )
    ts.add_argument(
        "--reconstruction-quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="Iteration budget preset (default: medium)",
    )
    ts.add_argument("--output-dir", default="_cli_runs/ts_inverse", help="Output directory (default: _cli_runs/ts_inverse)")
    ts.set_defaults(func=_handle_ts_inverse)

    # ConcreTizer (3D inversion) attack
    conc = subparsers.add_parser(
        "concretizer",
        aliases=["concreTizer", "concretizer-attack", "concretizer_attack"],
        help="🔴 ConcreTizer 3D model inversion attack (wrapper)",
    )
    conc.add_argument(
        "--target-model",
        dest="target_model",
        default=None,
        help="Target model callable spec 'module:function' OR 'dummy' (no default; required unless --target-model-file is set)",
    )
    conc.add_argument(
        "--target-model-file",
        dest="target_model_file",
        default=None,
        help="Path to a TorchScript model file to query (optional)",
    )
    conc.add_argument(
        "--allow-dummy",
        dest="allow_dummy",
        action="store_true",
        help="Allow using the built-in deterministic dummy target model (explicit demo mode)",
    )
    conc.add_argument("--voxel-resolution", dest="voxel_resolution", type=int, default=32, help="Voxel resolution (default: 32)")
    conc.add_argument("--max-queries", dest="max_queries", type=int, default=200, help="Max queries (default: 200)")
    conc.add_argument(
        "--refinement-iterations",
        dest="refinement_iterations",
        type=int,
        default=25,
        help="Refinement iterations (default: 25)",
    )
    conc.add_argument("--output-dir", default="_cli_runs/concretizer", help="Output directory (default: _cli_runs/concretizer)")
    conc.set_defaults(func=_handle_concretizer)

    # Latent-space jailbreak (requires HF model download unless already cached)
    lj = subparsers.add_parser(
        "latent-jailbreak",
        aliases=["latent_jailbreak", "latent-space-jailbreak", "latent_space_jailbreak"],
        help="🔴 Latent-space jailbreak attack (wrapper; requires HF model)",
    )
    lj.add_argument("--model", required=True, help="HuggingFace model id (e.g., gpt2)")
    lj.add_argument("--prompt", required=True, help="Prompt to jailbreak")
    lj.add_argument("--objective", default=None, help="Target behavior label (optional)")
    lj.add_argument("--start-layer", dest="start_layer", type=int, default=0, help="Start layer index (default: 0)")
    lj.add_argument("--end-layer", dest="end_layer", type=int, default=3, help="End layer index (default: 3)")
    lj.add_argument("--magnitude", type=float, default=1.0, help="Steering strength (default: 1.0)")
    lj.add_argument("--max-attempts", dest="max_attempts", type=int, default=20, help="Max attempts (default: 20)")
    lj.add_argument("--output-dir", dest="output_dir", default="_cli_runs/latent_jailbreak", help="Output directory")
    lj.set_defaults(func=_handle_latent_jailbreak)


def _handle_ts_inverse(args) -> int:
    try:
        from .adversarial_ts_inverse import run_ts_inverse

        return int(run_ts_inverse(args))
    except Exception as e:
        logger.error(f"TS-Inverse CLI failed: {e}")
        return 1


def _handle_concretizer(args) -> int:
    try:
        from .adversarial_concretizer import run_concretizer

        return int(run_concretizer(args))
    except Exception as e:
        logger.error(f"ConcreTizer CLI failed: {e}")
        return 1


def _handle_latent_jailbreak(args) -> int:
    try:
        from .latent_jailbreak import run_latent_jailbreak

        return int(run_latent_jailbreak(args))
    except Exception as e:
        logger.error(f"Latent jailbreak CLI failed: {e}")
        return 1


