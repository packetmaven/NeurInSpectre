"""
Evaluation command implementation for NeurInSpectre CLI.

Runs an evaluation matrix (defenses x attacks) over real datasets/models and
produces JSON summaries plus a terminal report.

This repo intentionally does not ship paper baseline numbers; any baseline/expected
values must be supplied via external files (see: `baseline_validation` config or
`neurinspectre compare --mode baseline --expected-asr-path ...`).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click

from ..attacks import AttackFactory
from ..evaluation.baseline_validation import (
    build_observed_asr_matrix,
    load_expected_asr,
    validate_asr_matrix,
)
from ..evaluation.artifact_integrity import nuscenes_label_map_hash_gate
from ..evaluation.budgets import get_attack_budgets, resolve_attack_config
from ..evaluation.datasets import DatasetFactory
from ..evaluation.validity_gates import evaluate_clean_validity, resolve_validity_gates
from .metadata import write_run_metadata
from .exporters import export_evaluation_json, export_evaluation_sarif
from .formatters import build_console, render_evaluation_report
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

# Paper Table 1: 12 defenses evaluated
DEFAULT_NUM_DEFENSES = 12

# Paper Section 3.3: "ensemble of attacks" (4 attacks per defense)
DEFAULT_ATTACKS_PER_DEFENSE = 4

# Paper Section 4: "full evaluation takes 2-8 hours depending on defense complexity"
EXPECTED_EVALUATION_HOURS_MIN = 2
EXPECTED_EVALUATION_HOURS_MAX = 8

# Paper Section 5.1: 1000 samples for evaluation
DEFAULT_EVAL_SAMPLES = 1000

# Enhancement #8: Parallelization
# Paper Section 4: "All attacks are fully batched and GPU-accelerated"
DEFAULT_PARALLEL_WORKERS = 1  # Conservative default (user must opt-in)
RECOMMENDED_MAX_WORKERS = 4   # Balances GPU memory vs throughput


def _resolve_seed_list(config: Dict[str, Any], *, cli_seeds: Any, num_seeds: Any) -> List[int]:
    """
    Statistical rigor (Issue 8): allow evaluation replication across multiple seeds.

    Priority:
      1) CLI --seeds (repeat flag)
      2) config.seeds: [..]
      3) config.seed (+ optional --num-seeds > 1)
    """
    if cli_seeds:
        return [int(s) for s in list(cli_seeds)]
    cfg_seeds = config.get("seeds")
    if isinstance(cfg_seeds, (list, tuple)) and cfg_seeds:
        return [int(s) for s in cfg_seeds]

    base = int(config.get("seed", 42))
    try:
        n = int(num_seeds or 1)
    except Exception:
        n = 1
    if n <= 1:
        return [base]
    return [base + i for i in range(n)]


def _mean_std(values: List[Any]) -> Tuple[float, float, int]:
    vals: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        vals.append(fv)
    n = int(len(vals))
    if n == 0:
        return 0.0, 0.0, 0
    mean = float(sum(vals) / n)
    if n == 1:
        return mean, 0.0, 1
    var = float(sum((x - mean) ** 2 for x in vals) / float(n - 1))
    return mean, float(math.sqrt(max(0.0, var))), n


def _t_critical_975(df: int) -> float:
    """
    Two-sided 95% CI uses t_{0.975, df}.

    We intentionally avoid a SciPy dependency here. This lookup is sufficient for
    the repo's intended usage (n_seeds typically 5-10 for paper tables).
    """
    try:
        dfi = int(df)
    except Exception:
        dfi = 0
    if dfi <= 0:
        return float("nan")
    # Source: standard t critical value table for p=0.975 (two-sided 95% CI).
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    if dfi in table:
        return float(table[dfi])
    # Large-df approximation: standard normal.
    return 1.96


def _mean_std_ci95(values: List[Any], *, clamp01: bool = False) -> Tuple[float, float, int, float, float]:
    """
    Return mean, std (sample), n, and 95% CI for the mean.
    """
    mean, std, n = _mean_std(values)
    if n <= 1:
        lo = float(mean)
        hi = float(mean)
    else:
        se = float(std) / float(math.sqrt(float(n)))
        tcrit = _t_critical_975(n - 1)
        half = float(tcrit) * float(se)
        lo = float(mean) - half
        hi = float(mean) + half
    if clamp01:
        lo = max(0.0, min(1.0, lo))
        hi = max(0.0, min(1.0, hi))
    return float(mean), float(std), int(n), float(lo), float(hi)


def _aggregate_pair_metrics(per_seed: List[Dict[str, Any]], *, seeds: List[int]) -> Dict[str, Any]:
    """
    Aggregate a single (defense, attack) metrics dict across seeds.

    We keep the mean in the original key (for compatibility with compare/ranking),
    and add *_std / *_n / *_ci95_low / *_ci95_high keys alongside it.
    """
    out: Dict[str, Any] = {}
    for key in ("attack_success_rate", "robust_accuracy", "clean_accuracy", "correct_samples", "samples"):
        vals = []
        for m in per_seed:
            if not isinstance(m, dict):
                continue
            if key in m:
                vals.append(m.get(key))
        clamp01 = key in {"attack_success_rate", "robust_accuracy", "clean_accuracy"}
        mean, std, n, lo, hi = _mean_std_ci95(vals, clamp01=clamp01)
        out[key] = float(mean)
        out[f"{key}_std"] = float(std)
        out[f"{key}_n"] = int(n)
        out[f"{key}_ci95_low"] = float(lo)
        out[f"{key}_ci95_high"] = float(hi)

    # For auditability: include the ASR per seed.
    seed_asr: Dict[str, float] = {}
    for seed, m in zip(seeds, per_seed):
        if isinstance(m, dict) and "attack_success_rate" in m:
            try:
                # Keep audit output stable/readable (avoid 0.30000000000000004 noise).
                seed_asr[str(seed)] = float(round(float(m.get("attack_success_rate", 0.0)), 12))
            except Exception:
                seed_asr[str(seed)] = 0.0
    out["attack_success_rate_by_seed"] = seed_asr
    return out


def _aggregate_results_across_seeds(
    seed_summaries: List[Dict[str, Any]],
    *,
    seeds: List[int],
) -> List[Dict[str, Any]]:
    """
    Aggregate evaluate/table2 results across multiple per-seed runs.
    """
    per_seed_maps: List[Dict[Tuple[str, str, str], Dict[str, Any]]] = []
    for summary in seed_summaries:
        m: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for res in summary.get("results", []) or []:
            if not isinstance(res, dict):
                continue
            key = (
                str(res.get("defense", res.get("type", "defense"))),
                str(res.get("type", res.get("defense", "defense"))),
                str(res.get("dataset", "")),
            )
            m[key] = res
        per_seed_maps.append(m)

    all_keys = sorted({k for m in per_seed_maps for k in m.keys()})
    aggregated: List[Dict[str, Any]] = []

    for key in all_keys:
        base = None
        for m in per_seed_maps:
            if key in m:
                base = m[key]
                break
        if base is None:
            continue

        defense_name = str(base.get("defense", base.get("type", "defense")))
        defense_type = str(base.get("type", defense_name))
        dataset_name = str(base.get("dataset", ""))

        attack_names = sorted(
            {
                str(a)
                for m in per_seed_maps
                for a in (m.get(key, {}).get("attacks", {}) or {}).keys()
            }
        )
        attacks_out: Dict[str, Any] = {}
        for attack in attack_names:
            per_seed_metrics: List[Dict[str, Any]] = []
            for m in per_seed_maps:
                metrics = (m.get(key, {}).get("attacks", {}) or {}).get(attack, {})
                per_seed_metrics.append(metrics if isinstance(metrics, dict) else {})
            attacks_out[attack] = _aggregate_pair_metrics(per_seed_metrics, seeds=seeds)

        characterization = {}
        for m in per_seed_maps:
            if key in m and isinstance(m[key].get("characterization"), dict) and m[key]["characterization"]:
                characterization = dict(m[key]["characterization"])
                break

        aggregated.append(
            {
                "defense": defense_name,
                "type": defense_type,
                "dataset": dataset_name,
                "attacks": attacks_out,
                "characterization": characterization,
                "multi_seed": {"enabled": True, "seeds": list(seeds), "n_seeds": int(len(seeds))},
            }
        )

    return aggregated


def run_evaluation(ctx: click.Context, **kwargs: Any) -> None:
    cmd_verbose = int(kwargs.get("verbose", 0) or 0)
    if ctx is not None:
        ctx.obj = ctx.obj or {}
        if cmd_verbose:
            ctx.obj["verbose"] = max(int(ctx.obj.get("verbose", 0)), cmd_verbose)
    device = resolve_device(kwargs.get("device", "cpu"))
    config_path = kwargs.get("config")
    config_base_dir = Path(str(config_path)).resolve().parent
    config = load_yaml(config_path)

    # -------------------------------------------------------------------
    # Tier 2: optional calibrated threshold injection for Phase 1
    # -------------------------------------------------------------------
    # Config accepts:
    #   characterization_thresholds: path/to/thresholds.json
    # or
    #   threshold_overrides: path/to/thresholds.json
    # or
    #   characterization_thresholds: { ... overrides dict ... }
    thr_cfg = (
        config.get("characterization_thresholds")
        or config.get("threshold_overrides")
        or config.get("thresholds")
    )
    if isinstance(thr_cfg, (str, Path)):
        p = Path(str(thr_cfg))
        if not p.is_absolute():
            cwd_candidate = Path.cwd() / p
            if cwd_candidate.exists():
                p = cwd_candidate
            else:
                p = config_base_dir / p
        if not p.exists():
            raise click.ClickException(f"characterization_thresholds file not found: {p}")
        try:
            config["characterization_thresholds"] = load_threshold_overrides(p)
            config["characterization_thresholds_path"] = str(p)
        except Exception as exc:
            raise click.ClickException(f"Failed to load characterization_thresholds: {exc}") from exc
    elif isinstance(thr_cfg, dict):
        # Ensure the key is present at the expected location (AttackFactory reads it).
        config["characterization_thresholds"] = dict(thr_cfg)

    quiet = bool(ctx.obj.get("quiet", False)) if ctx and ctx.obj else False
    no_progress = bool(kwargs.get("no_progress", False))
    report_format = str(kwargs.get("report_format", "rich"))
    no_color = bool(kwargs.get("no_color", False))
    force_color = bool(kwargs.get("color", False))
    console = build_console(no_color=no_color, force_color=force_color)

    out_dir = Path(kwargs.get("output_dir", "evaluation_results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_test = bool(kwargs.get("smoke_test", False))
    # Best-effort artifact metadata capture for AE/debugging.
    if not bool(kwargs.get("_skip_run_metadata", False)):
        try:
            write_run_metadata(
                out_dir,
                config_path=config_path,
                device=str(device),
                extra={"command": "evaluate", "smoke_test": smoke_test},
            )
        except Exception:
            logger.debug("Failed to write run metadata", exc_info=True)

    defense_filter = set(kwargs.get("defenses") or [])
    attack_filter = set(kwargs.get("attacks") or [])
    parallel = max(1, int(kwargs.get("parallel", DEFAULT_PARALLEL_WORKERS)))
    resume = bool(kwargs.get("resume", False))
    summary_only = bool(kwargs.get("summary_only", False))

    # -------------------------------------------------------------------
    # Issue 8 (+ WOOT revision): multi-seed replication (mean ± std + 95% CI)
    # -------------------------------------------------------------------
    seed_override = kwargs.get("_seed_override", None)
    if seed_override is not None:
        seeds = [int(seed_override)]
    else:
        seeds = _resolve_seed_list(config, cli_seeds=kwargs.get("seeds"), num_seeds=kwargs.get("num_seeds"))

    # Multi-seed orchestrator: run each seed in a subdirectory, then aggregate.
    if seed_override is None and len(seeds) > 1:
        seed_summaries: List[Dict[str, Any]] = []
        seed_paths: Dict[str, str] = {}

        # Keep per-seed runs quiet; we render a single aggregated report at the end.
        for s in seeds:
            seed_dir = out_dir / f"seed_{int(s)}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            seed_paths[str(int(s))] = str(seed_dir / "summary.json")

            child_kwargs = dict(kwargs)
            child_kwargs["_seed_override"] = int(s)
            child_kwargs["output_dir"] = str(seed_dir)
            child_kwargs["report"] = False
            child_kwargs["json_output"] = None
            child_kwargs["sarif_output"] = None
            child_kwargs["seeds"] = ()
            child_kwargs["num_seeds"] = 1
            run_evaluation(ctx, **child_kwargs)

            seed_summary = _load_json(seed_dir / "summary.json")
            seed_summaries.append(seed_summary)

        aggregated_results = _aggregate_results_across_seeds(seed_summaries, seeds=[int(s) for s in seeds])
        summary = {
            "config": {**copy.deepcopy(config), "seeds": [int(s) for s in seeds]},
            "results": aggregated_results,
            "multi_seed": {
                "enabled": True,
                "seeds": [int(s) for s in seeds],
                "num_seeds": int(len(seeds)),
                "seed_summaries": seed_paths,
            },
            "timing": {
                "total_seconds": float(sum(float(ss.get("timing", {}).get("total_seconds", 0.0)) for ss in seed_summaries)),
                "defenses": len(aggregated_results),
            },
        }
        summary["highlights"] = _summarize_evaluation_findings(aggregated_results, config=config)
        baseline_report = _run_baseline_validation(
            aggregated_results,
            config=config,
            config_path=config_path,
        )
        summary["baseline_validation"] = baseline_report
        save_json(summary, out_dir / "summary.json")
        click.echo(f"Evaluation summary written to {out_dir / 'summary.json'}")

        if baseline_report.get("enabled") and baseline_report.get("strict") and not baseline_report.get("passed", False):
            raise click.ClickException(
                "Baseline validation failed: "
                f"{baseline_report.get('failed_count', 0)} pairs out of tolerance, "
                f"{baseline_report.get('missing_expected_count', 0)} missing expected pairs."
            )

        report = bool(kwargs.get("report", True))
        brief = bool(kwargs.get("brief", False))
        if report and not quiet:
            render_evaluation_report(
                console,
                results=aggregated_results,
                config=config,
                output_dir=str(out_dir),
                report_format=report_format,
                brief=brief,
                summary_only=summary_only,
            )

        json_output = kwargs.get("json_output")
        if json_output:
            export_evaluation_json({"config": summary.get("config", {}), "results": aggregated_results}, json_output)
            click.echo(f"JSON report written to {json_output}")

        sarif_output = kwargs.get("sarif_output")
        if sarif_output:
            export_evaluation_sarif({"pairs": _evaluation_pairs(aggregated_results)}, sarif_output)
            click.echo(f"SARIF report written to {sarif_output}")
        return

    # Single-seed path (legacy behavior)
    seed = int(seeds[0]) if seeds else int(config.get("seed", 42))
    config["seed"] = seed
    config.pop("seeds", None)
    set_seed(seed)

    defenses = _normalize_defenses(config.get("defenses", []))
    attacks = _normalize_attacks(config.get("attacks", []))
    if defense_filter:
        defenses = [d for d in defenses if d["name"] in defense_filter or d["type"] in defense_filter]
    if attack_filter:
        attacks = [a for a in attacks if a["name"] in attack_filter]

    if smoke_test:
        # Smoke mode should be fast and self-contained; disable baseline validation
        # and clamp dataset sizes/iterations to keep runtime bounded.
        config["baseline_validation"] = {"enabled": False}
        try:
            config["iterations"] = min(int(config.get("iterations", config.get("attack_iterations", 100))), 10)
        except Exception:
            config["iterations"] = 10
        try:
            config["attack_batch_size"] = min(int(config.get("attack_batch_size", 128)), 64)
        except Exception:
            config["attack_batch_size"] = 64
        datasets_cfg = config.get("datasets") or {}
        if isinstance(datasets_cfg, dict):
            for _ds_name, ds in datasets_cfg.items():
                if not isinstance(ds, dict):
                    continue
                ns = ds.get("num_samples", ds.get("n_samples", DEFAULT_EVAL_SAMPLES))
                try:
                    ns_i = int(ns)
                except Exception:
                    ns_i = DEFAULT_EVAL_SAMPLES
                ds["num_samples"] = min(ns_i, 100)
                try:
                    ds["batch_size"] = min(int(ds.get("batch_size", config["attack_batch_size"])), 64)
                except Exception:
                    ds["batch_size"] = int(config["attack_batch_size"])
                ds.setdefault("num_workers", 0)

        # Prefer a "fast" defense/attack if present; otherwise take the first.
        def _pick_defense(entries: List[Dict[str, Any]]) -> Dict[str, Any] | None:
            prefer = ("none", "jpeg", "bitdepth")
            for want in prefer:
                for d in entries:
                    key = str(d.get("type", d.get("name", ""))).lower()
                    if key == want or str(d.get("name", "")).lower() == want:
                        return d
            return entries[0] if entries else None

        def _pick_attack(entries: List[Dict[str, Any]]) -> Dict[str, Any] | None:
            prefer = ("pgd", "apgd", "neurinspectre")
            for want in prefer:
                for a in entries:
                    if str(a.get("name", "")).lower() == want:
                        return a
            return entries[0] if entries else None

        dsel = _pick_defense(defenses)
        asel = _pick_attack(attacks)
        defenses = [dsel] if dsel else []
        attacks = [asel] if asel else []
        parallel = 1

        if not defenses or not attacks:
            raise click.ClickException("Smoke test requested but config contains no runnable defenses/attacks.")

    if parallel > RECOMMENDED_MAX_WORKERS:
        logger.warning(
            "Requested parallel workers=%d exceeds recommended maximum (%d). "
            "Paper Section 4: GPU memory contention may reduce throughput.",
            parallel,
            RECOMMENDED_MAX_WORKERS,
        )

    if len(defenses) == DEFAULT_NUM_DEFENSES and len(attacks) == DEFAULT_ATTACKS_PER_DEFENSE:
        logger.info(
            "Table 1 reproduction mode detected: %d defenses x %d attacks.",
            DEFAULT_NUM_DEFENSES,
            DEFAULT_ATTACKS_PER_DEFENSE,
        )
    else:
        logger.info(
            "Evaluation matrix: %d defenses x %d attacks (default: %d x %d).",
            len(defenses),
            len(attacks),
            DEFAULT_NUM_DEFENSES,
            DEFAULT_ATTACKS_PER_DEFENSE,
        )

    base_attack_cfg = _base_attack_config(config)
    validity_gates = resolve_validity_gates(config)
    strict_fail_fast = bool(validity_gates.get("enabled")) and bool(validity_gates.get("strict"))

    results: List[Dict[str, Any]] = []
    start_time = time.time()

    total_units = len(defenses) if parallel > 1 else len(defenses) * max(1, len(attacks))
    with ProgressReporter(
        description="Evaluation suite...",
        total=total_units,
        enabled=not no_progress and not quiet,
        console=console,
    ) as progress:
        if parallel > 1:
            logger.info(
                "Running parallel evaluation with %d workers (Paper Section 4).",
                parallel,
            )
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(
                        _evaluate_defense,
                        defense,
                        attacks,
                        config,
                        base_attack_cfg,
                        device,
                        out_dir,
                        resume,
                        seed,
                        config_base_dir,
                        None,
                    ): defense
                    for defense in defenses
                }
                for future in as_completed(futures):
                    defense_entry = futures[future]
                    defense_name = defense_entry.get("name", defense_entry.get("type", "defense"))
                    try:
                        result = future.result()
                        results.append(result)
                        avg_asr = _compute_avg_asr(result)
                        progress.advance(
                            1,
                            description=f"Evaluation suite... [{defense_name}: ASR={avg_asr:.1%}]",
                        )
                    except Exception as exc:
                        if strict_fail_fast:
                            # In strict validity mode, do not publish partial or meaningless results.
                            if isinstance(exc, click.ClickException):
                                raise
                            raise click.ClickException(str(exc)) from exc
                        logger.error(
                            "Defense %s failed: %s",
                            defense_name,
                            exc,
                            exc_info=True,
                        )
        else:
            for defense in defenses:
                results.append(
                    _evaluate_defense(
                        defense,
                        attacks,
                        config,
                        base_attack_cfg,
                        device,
                        out_dir,
                        resume,
                        seed,
                        config_base_dir,
                        progress.advance,
                    )
                )

    summary = {
        "config": config,
        "results": results,
        "timing": {
            "total_seconds": float(time.time() - start_time),
            "defenses": len(results),
        },
    }
    summary["highlights"] = _summarize_evaluation_findings(results, config=config)
    baseline_report = _run_baseline_validation(
        results,
        config=config,
        config_path=config_path,
    )
    summary["baseline_validation"] = baseline_report
    save_json(summary, out_dir / "summary.json")
    click.echo(f"Evaluation summary written to {out_dir / 'summary.json'}")

    if baseline_report.get("enabled") and baseline_report.get("strict") and not baseline_report.get("passed", False):
        raise click.ClickException(
            "Baseline validation failed: "
            f"{baseline_report.get('failed_count', 0)} pairs out of tolerance, "
            f"{baseline_report.get('missing_expected_count', 0)} missing expected pairs."
        )

    report = bool(kwargs.get("report", True))
    brief = bool(kwargs.get("brief", False))
    if report and not quiet:
        render_evaluation_report(
            console,
            results=results,
            config=config,
            output_dir=str(out_dir),
            report_format=report_format,
            brief=brief,
            summary_only=summary_only,
        )

    json_output = kwargs.get("json_output")
    if json_output:
        export_evaluation_json(
            {
                "config": config,
                "results": results,
            },
            json_output,
        )
        click.echo(f"JSON report written to {json_output}")

    sarif_output = kwargs.get("sarif_output")
    if sarif_output:
        export_evaluation_sarif(
            {
                "pairs": _evaluation_pairs(results),
            },
            sarif_output,
        )
        click.echo(f"SARIF report written to {sarif_output}")


def _normalize_defenses(defenses_cfg: Any) -> List[Dict[str, Any]]:
    if isinstance(defenses_cfg, dict):
        return [{"name": name, **(cfg or {})} for name, cfg in defenses_cfg.items()]
    if isinstance(defenses_cfg, list):
        entries = []
        for item in defenses_cfg:
            if isinstance(item, str):
                entries.append({"name": item, "type": item})
            elif isinstance(item, dict):
                entry = dict(item)
                entry.setdefault("name", entry.get("type", "defense"))
                entries.append(entry)
            else:
                raise ValueError("Defense entry must be a string or mapping.")
        return entries
    raise ValueError("defenses must be a list or mapping.")


def _normalize_attacks(attacks_cfg: Any) -> List[Dict[str, Any]]:
    if isinstance(attacks_cfg, dict):
        return [{"name": name, **(cfg or {})} for name, cfg in attacks_cfg.items()]
    if isinstance(attacks_cfg, list):
        entries = []
        for item in attacks_cfg:
            if isinstance(item, str):
                entries.append({"name": item})
            elif isinstance(item, dict):
                entry = dict(item)
                entry.setdefault("name", entry.get("type", "attack"))
                entries.append(entry)
            else:
                raise ValueError("Attack entry must be a string or mapping.")
        return entries
    raise ValueError("attacks must be a list or mapping.")


def _base_attack_config(config: Dict[str, Any]) -> Dict[str, Any]:
    perturbation = config.get("perturbation", {})
    return {
        "epsilon": float(perturbation.get("epsilon", 8 / 255)),
        "norm": str(perturbation.get("norm", "Linf")),
        "n_iterations": int(config.get("iterations", config.get("attack_iterations", 100))),
        "batch_size": int(config.get("attack_batch_size", 128)),
        "seed": int(config.get("seed", 42)),
    }


def _compute_avg_asr(result: Dict[str, Any]) -> float:
    attacks = result.get("attacks", {}) or {}
    if not attacks:
        return 0.0
    values = [float(v.get("attack_success_rate", 0.0)) for v in attacks.values()]
    return float(sum(values) / max(1, len(values)))


def _summarize_evaluation_findings(
    results: List[Dict[str, Any]],
    *,
    config: Dict[str, Any],
    top_n: int = 5,
) -> Tuple[str, ...]:
    entries: List[Tuple[float, float, str, str, float, float]] = []
    per_defense: List[Tuple[float, str, str, float]] = []

    for res in results:
        defense = str(res.get("defense", res.get("type", "defense")))
        attacks = res.get("attacks", {}) or {}
        best_asr = -1.0
        best_attack = "unknown"
        best_robust = 0.0
        for attack_name, metrics in attacks.items():
            asr = float(metrics.get("attack_success_rate", 0.0))
            robust = float(metrics.get("robust_accuracy", 0.0))
            clean = float(metrics.get("clean_accuracy", 0.0))
            drop = clean - robust
            entries.append((asr, drop, defense, str(attack_name), robust, clean))
            if asr > best_asr:
                best_asr = asr
                best_attack = str(attack_name)
                best_robust = robust
        if best_asr >= 0:
            per_defense.append((best_asr, defense, best_attack, best_robust))

    entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top_entries = entries[: max(1, int(top_n))]
    top_str = ", ".join(
        f"{defense}/{attack} ASR={asr:.3f} RA={robust:.3f}"
        for asr, _drop, defense, attack, robust, _clean in top_entries
    )

    per_defense.sort(key=lambda item: item[0], reverse=True)
    weak_str = ", ".join(
        f"{defense}({attack} ASR={asr:.3f} RA={robust:.3f})"
        for asr, defense, attack, robust in per_defense[: max(1, min(3, len(per_defense)))]
    )

    flagged = [
        entry
        for entry in entries
        if entry[0] >= 0.5 or entry[1] >= 0.3
    ]
    invalid = 0
    for res in results:
        attacks = res.get("attacks", {}) or {}
        for _attack, metrics in attacks.items():
            validity = (metrics or {}).get("validity") if isinstance(metrics, dict) else None
            if isinstance(validity, dict) and validity.get("enabled") and not validity.get("passed", True):
                invalid += 1
    # If dataset-specific budgets are configured, report them instead of the global
    # perturbation defaults (which can be misleading for multi-dataset Table2 runs).
    threat_line = None
    budgets = get_attack_budgets(config)
    datasets_seen = sorted({str(r.get("dataset")) for r in results if r.get("dataset")})
    if budgets and datasets_seen:
        parts: List[str] = []
        for dataset_name in datasets_seen:
            budget = dict(budgets.get(dataset_name, {}) or {})
            eps = budget.get("epsilon", budget.get("eps"))
            norm = budget.get("norm")
            if eps is None and norm is None:
                continue
            parts.append(f"{dataset_name} norm={norm} eps={eps}")
        if parts:
            threat_line = f"Threat model: {', '.join(parts)}"
    if not threat_line:
        epsilon = config.get("perturbation", {}).get("epsilon", 8 / 255)
        norm = config.get("perturbation", {}).get("norm", "Linf")
        threat_line = f"Threat model: norm={norm}, epsilon={epsilon}"

    lines = [
        threat_line,
        f"Top compromise pairs (ASR): {top_str}" if top_str else "Top compromise pairs (ASR): n/a",
        f"Weakest defenses (best attack ASR): {weak_str}" if weak_str else "Weakest defenses: n/a",
        f"Flagged pairs (ASR>=0.5 or drop>=0.3): {len(flagged)}/{len(entries)}",
    ]
    if invalid:
        lines.append(f"Validity warnings: {invalid} pairs have low clean accuracy / too few clean-correct samples")
    return tuple(lines)


def _evaluation_pairs(results: List[Dict[str, Any]]) -> List[Tuple[str, str, float, float]]:
    pairs: List[Tuple[str, str, float, float]] = []
    for res in results:
        defense = str(res.get("defense", res.get("type", "defense")))
        attacks = res.get("attacks", {}) or {}
        for attack_name, metrics in attacks.items():
            asr = float(metrics.get("attack_success_rate", 0.0))
            clean = float(metrics.get("clean_accuracy", 0.0))
            robust = float(metrics.get("robust_accuracy", 0.0))
            drop = clean - robust
            pairs.append((defense, str(attack_name), asr, drop))
    return pairs


def _run_baseline_validation(
    results: List[Dict[str, Any]],
    *,
    config: Dict[str, Any],
    config_path: str | None,
) -> Dict[str, Any]:
    validation_cfg = dict(config.get("baseline_validation", {}) or {})
    if not validation_cfg or not bool(validation_cfg.get("enabled", False)):
        return {"enabled": False}

    base_dir = Path(config_path).parent if config_path else None
    expected = load_expected_asr(validation_cfg, base_dir=base_dir)
    strict = bool(validation_cfg.get("strict", False))
    if not expected:
        if strict:
            return {
                "enabled": True,
                "strict": True,
                "passed": False,
                "reason": "missing_expected_asr",
                "failed_count": 0,
                "missing_expected_count": 0,
            }
        return {
            "enabled": True,
            "strict": False,
            "passed": False,
            "reason": "missing_expected_asr",
            "failed_count": 0,
            "missing_expected_count": 0,
        }

    observed_rows: List[Tuple[str, str, float]] = []
    for res in results:
        defense = str(res.get("defense", res.get("type", "defense")))
        attacks = res.get("attacks", {}) or {}
        for attack_name, metrics in attacks.items():
            observed_rows.append(
                (
                    defense,
                    str(attack_name),
                    float(metrics.get("attack_success_rate", 0.0)),
                )
            )
    tolerance = float(validation_cfg.get("tolerance", 0.02))
    require_all = bool(validation_cfg.get("require_all_expected", True))
    report = validate_asr_matrix(
        build_observed_asr_matrix(observed_rows),
        expected,
        tolerance=tolerance,
        require_all_expected=require_all,
    )
    report["enabled"] = True
    report["strict"] = strict
    return report


def _evaluate_defense(
    defense_entry: Dict[str, Any],
    attacks: List[Dict[str, Any]],
    config: Dict[str, Any],
    base_attack_cfg: Dict[str, Any],
    device: str,
    out_dir: Path,
    resume: bool,
    seed: int,
    config_base_dir: Path,
    progress_callback: Any,
) -> Dict[str, Any]:
    defense_name = defense_entry.get("name", defense_entry.get("type", "defense"))
    defense_type = defense_entry.get("type", defense_name)

    output_path = out_dir / f"{defense_name}.json"
    existing: Dict[str, Any] | None = None
    existing_attacks: Dict[str, Any] = {}
    existing_characterization: Dict[str, Any] = {}
    existing_integrity: Dict[str, Any] = {}
    if resume and output_path.exists():
        # Resume semantics: merge per-attack results.
        #
        # Previous behavior returned early and prevented "resume then run missing attacks"
        # workflows (common when AutoAttack is slow or a dataset fails mid-matrix).
        try:
            loaded = _load_json(output_path)
            if isinstance(loaded, dict):
                existing = loaded
                existing_attacks = dict(loaded.get("attacks", {}) or {})
                existing_characterization = dict(loaded.get("characterization", {}) or {})
                existing_integrity = dict(loaded.get("integrity", {}) or {})
        except Exception:
            existing = None

    datasets_cfg = config.get("datasets", {})
    dataset_name = _resolve_dataset(defense_entry, datasets_cfg)
    dataset_cfg = dict(datasets_cfg.get(dataset_name, {}))
    if "path" in dataset_cfg and "data_path" not in dataset_cfg:
        dataset_cfg["data_path"] = dataset_cfg["path"]

    num_samples = int(dataset_cfg.get("num_samples", dataset_cfg.get("n_samples", DEFAULT_EVAL_SAMPLES)))
    batch_size = int(dataset_cfg.get("batch_size", base_attack_cfg.get("batch_size", 128)))
    num_workers = int(dataset_cfg.get("num_workers", 4))
    split = str(dataset_cfg.get("split", "test"))
    labels_path = dataset_cfg.get("labels_path")
    nuscenes_version = dataset_cfg.get("version") or dataset_cfg.get("nuscenes_version")

    def _resolve_existing_path(raw_path: str) -> Path:
        p = Path(raw_path)
        if p.is_absolute():
            return p
        cwd_candidate = Path.cwd() / p
        if cwd_candidate.exists():
            return cwd_candidate
        cfg_candidate = config_base_dir / p
        if cfg_candidate.exists():
            return cfg_candidate
        return cfg_candidate

    loader, _x, _y = load_dataset(
        dataset_name,
        data_path=dataset_cfg.get("data_path") or dataset_cfg.get("path") or dataset_cfg.get("root"),
        labels_path=labels_path,
        nuscenes_version=nuscenes_version,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        split=split,
        download=bool(dataset_cfg.get("download", True)),
        device=device,
        filter_label=dataset_cfg.get("filter_label"),
    )

    model_ref = _resolve_model_ref(defense_entry, config, dataset_name)

    # Optional artifact integrity gate (AE-friendly): prevent silent nuScenes label-map mismatch.
    validity_gates = resolve_validity_gates(config)
    integrity: Dict[str, Any] = {}
    if (
        str(dataset_name).lower() == "nuscenes"
        and bool(validity_gates.get("enabled", False))
        and bool(validity_gates.get("require_label_map_sha256_match", False))
    ):
        if not labels_path:
            integrity_report: Dict[str, Any] = {
                "enabled": True,
                "passed": False,
                "reasons": ["missing_labels_path_config"],
            }
        else:
            resolved_labels_path = _resolve_existing_path(str(labels_path))

            model_path: Path | None = None
            if isinstance(model_ref, (str, Path)):
                candidate = _resolve_existing_path(str(model_ref))
                if candidate.exists():
                    model_path = candidate
            elif isinstance(model_ref, dict):
                path_val = (
                    model_ref.get("path")
                    or model_ref.get("checkpoint_path")
                    or model_ref.get("weights_path")
                )
                if path_val:
                    candidate = _resolve_existing_path(str(path_val))
                    if candidate.exists():
                        model_path = candidate

            if model_path is None:
                integrity_report = {
                    "enabled": True,
                    "passed": False,
                    "reasons": ["missing_model_path_for_hash_gate"],
                    "hint": "Provide an explicit nuScenes TorchScript path (and its .meta.json) to enable label-map hashing.",
                    "labels_path": str(resolved_labels_path),
                }
            else:
                integrity_report = nuscenes_label_map_hash_gate(
                    model_path=str(model_path),
                    labels_path=str(resolved_labels_path),
                )

        integrity["nuscenes_label_map_hash"] = integrity_report
        if bool(validity_gates.get("strict", False)) and not bool(integrity_report.get("passed", False)):
            raise click.ClickException(
                "nuScenes label-map hash gate failed (model/label_map mismatch or missing metadata). "
                f"defense={defense_name} reasons={','.join(integrity_report.get('reasons', []) or [])}"
            )

    model = load_model(model_ref, dataset=dataset_name, device=device)

    defense_params = {
        k: v
        for k, v in defense_entry.items()
        if k
        not in {
            "name",
            "type",
            "dataset",
            "model",
            "model_path",
            "model_name",
            "model_factory",
            "model_spec",
            "model_provenance",
            "domain",
            "claimed_robust_accuracy",
            "enabled",
            "id",
            "group",
        }
    }
    if "defense_config" in defense_params or "config_path" in defense_params:
        cfg_path = defense_params.pop("defense_config", None) or defense_params.pop("config_path", None)
        if cfg_path:
            loaded = load_yaml(cfg_path)
            if "defense" in loaded and isinstance(loaded["defense"], dict):
                loaded = loaded["defense"]
            defense_params.update(loaded)

    # Support configs that nest defense kwargs under a ``params`` dictionary.
    # Table2 spec-style YAMLs frequently use this convention.
    nested_params = defense_params.pop("params", None)
    if isinstance(nested_params, dict):
        # Prefer explicit top-level keys if both are provided.
        for k, v in nested_params.items():
            defense_params.setdefault(k, v)
    defense_model = build_defense(defense_type, model, defense_params, device=device)
    eval_model = defense_model or model

    attack_results: Dict[str, Any] = dict(existing_attacks)
    characterization: Dict[str, Any] = dict(existing_characterization)

    for attack_entry in attacks:
        # Support multiple variants of the same underlying attack by allowing:
        #   {"name": "neurinspectre_no_mem", "type": "neurinspectre", ...}
        #
        # Back-compat: if `type` is absent, `name` is treated as the implementation key.
        attack_name = attack_entry.get("name") or attack_entry.get("id") or attack_entry.get("type")
        attack_type = attack_entry.get("type") or attack_name
        if not attack_name or not attack_type:
            continue
        # If we are resuming and already have this attack in the per-defense JSON,
        # do not recompute it.
        if resume and attack_name in existing_attacks:
            continue
        attack_cfg = dict(base_attack_cfg)
        attack_cfg.update(
            {
                k: v
                for k, v in attack_entry.items()
                if k not in {"name", "id", "type"}
            }
        )
        # Allow per-attack ``steps`` to override the global ``iterations`` default.
        # ``AttackFactory._to_attack_config`` prefers ``n_iterations`` over ``steps``,
        # so we map it explicitly here.
        if "steps" in attack_entry and "n_iterations" not in attack_entry:
            try:
                attack_cfg["n_iterations"] = int(attack_entry.get("steps"))  # type: ignore[arg-type]
            except Exception:
                pass
        attack_cfg = resolve_attack_config(
            config,
            attack_cfg=attack_cfg,
            dataset_name=dataset_name,
            strict_budget=bool(config.get("strict_dataset_budgets", False)),
        )
        attack_key = str(attack_type).lower()
        # For baseline attacks (PGD/APGD/AutoAttack/...), the attack target is the
        # *defended* model. For BPDA/EOT/Hybrid/NeurInSpectre, we need both the
        # base model and the defense wrapper so the attack can use transform()
        # and/or BPDA approximations.
        if attack_key in {
            "bpda",
            "eot",
            "hybrid",
            "hybrid_volterra",
            "hybrid-volterra",
            "neurinspectre",
            "aa_bpda",
            "autoattack_bpda",
            "aa_official_bpda",
        }:
            attack_model = model
            attack_defense = defense_model
        else:
            attack_model = eval_model
            attack_defense = None

        # Tier 2: allow Table2 runs to apply calibrated thresholds to the Phase 1
        # DefenseAnalyzer characterization inside the NeurInSpectre runner.
        if attack_key == "neurinspectre":
            thresholds = config.get("characterization_thresholds")
            if isinstance(thresholds, dict) and thresholds:
                attack_cfg.setdefault("characterization_thresholds", dict(thresholds))

        runner = AttackFactory.create_attack(
            attack_key,
            attack_model,
            config=attack_cfg,
            characterization_loader=loader if attack_key == "neurinspectre" else None,
            defense=attack_defense,
            device=device,
        )
        query_budgets = attack_cfg.get("query_budgets", config.get("query_budgets"))
        summary = evaluate_attack_runner(
            runner,
            eval_model,
            loader,
            num_samples=num_samples,
            device=device,
            targeted=bool(attack_cfg.get("targeted", False)),
            norm=attack_cfg.get("norm", "Linf"),
            query_budgets=query_budgets,
        )
        gates = resolve_validity_gates(config)
        validity = evaluate_clean_validity(summary, gates)
        if validity.get("enabled"):
            summary["validity"] = validity
            if bool(gates.get("strict", False)) and not bool(validity.get("passed", False)):
                raise click.ClickException(
                    "Validity gate failed (clean accuracy too low for meaningful ASR). "
                    f"dataset={dataset_name} defense={defense_name} attack={attack_name} "
                    f"clean_acc={float(validity.get('observed', {}).get('clean_accuracy', 0.0)):.3f} "
                    f"correct={int(validity.get('observed', {}).get('correct_samples', 0))}/"
                    f"{int(validity.get('observed', {}).get('samples', 0))} "
                    f"reasons={','.join(validity.get('reasons', []) or [])}"
                )
        if hasattr(runner, "chosen_attack") and runner.chosen_attack:
            summary.setdefault("chosen_attack", runner.chosen_attack)
            summary.setdefault("selected_attack_impl", getattr(runner, "selected_attack_impl", None))
        attack_results[str(attack_name)] = summary

        if progress_callback:
            asr = float(summary.get("attack_success_rate", 0.0))
            robust = float(summary.get("robust_accuracy", 0.0))
            progress_callback(
                1,
                description=f"{defense_name} vs {attack_name} ASR={asr:.1%} RA={robust:.1%}",
            )

        # Record characterization once per defense (the output should be identical across
        # multiple neurinspectre variants; the `type` may differ only by routing toggles).
        if attack_key == "neurinspectre" and hasattr(runner, "characterization"):
            try:
                characterization = runner.characterization.to_dict()
            except Exception:
                characterization = {}

    result = {
        "defense": defense_name,
        "type": defense_type,
        "dataset": dataset_name,
        "attacks": attack_results,
        "characterization": characterization,
    }
    # Preserve integrity info if present from either the current run or prior resume file.
    integrity_out: Dict[str, Any] = {}
    if existing_integrity:
        integrity_out.update(existing_integrity)
    if integrity:
        integrity_out.update(integrity)
    if integrity_out:
        result["integrity"] = integrity_out
    save_json(result, output_path)
    return result


def _resolve_dataset(defense_entry: Dict[str, Any], datasets_cfg: Dict[str, Any]) -> str:
    if "dataset" in defense_entry:
        return str(defense_entry["dataset"])

    defense_key = str(defense_entry.get("type", defense_entry.get("name", ""))).lower()
    alias_map = {
        "jpeg": "jpeg_compression",
        "bitdepth": "bit_depth_reduction",
        "randsmooth": "randomized_smoothing",
        "thermometer": "thermometer_encoding",
        "distillation": "defensive_distillation",
        "ensemble": "ensemble_diversity",
    }
    defense_key = alias_map.get(defense_key, defense_key)
    mapped = DatasetFactory.DEFENSE_TO_DATASET.get(defense_key)
    if mapped:
        return mapped

    if len(datasets_cfg) == 1:
        return next(iter(datasets_cfg.keys()))

    raise ValueError("Dataset not specified for defense and could not be inferred.")


def _resolve_model_ref(defense_entry: Dict[str, Any], config: Dict[str, Any], dataset_name: str) -> Any:
    if "model" in defense_entry:
        return defense_entry["model"]
    if "model_path" in defense_entry:
        return defense_entry["model_path"]
    if "model_name" in defense_entry:
        return {"model_name": defense_entry["model_name"]}

    models_cfg = config.get("models") or {}
    if isinstance(models_cfg, dict):
        if defense_entry.get("name") in models_cfg:
            return models_cfg[defense_entry["name"]]
        if dataset_name in models_cfg:
            return models_cfg[dataset_name]

    datasets_cfg = config.get("datasets", {})
    dataset_cfg = datasets_cfg.get(dataset_name, {}) if isinstance(datasets_cfg, dict) else {}
    if isinstance(dataset_cfg, dict):
        if "model" in dataset_cfg:
            return dataset_cfg["model"]
        if "model_name" in dataset_cfg:
            return {"model_name": dataset_cfg["model_name"]}

    global_model = config.get("model") or config.get("model_name")
    if global_model:
        return global_model

    default_models = {
        "cifar10": "resnet18",
        "cifar100": "resnet18",
        "imagenet": "resnet50",
        "imagenet100": "resnet50",
    }
    if dataset_name in default_models:
        return {"model_name": default_models[dataset_name], "domain": "vision"}

    raise ValueError("Model configuration missing for evaluation.")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
