#!/usr/bin/env python3
"""
NeurInSpectre Mathematical Foundations CLI Commands
Advanced mathematical analysis tools for gradient obfuscation detection
"""

import logging
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def register_mathematical_commands(subparsers):
    """Register mathematical analysis commands"""
    
    # Main mathematical analysis command
    math_parser = subparsers.add_parser(
        'math',
        help='🧮 Advanced mathematical analysis tools',
        description='GPU-accelerated mathematical foundations for gradient obfuscation detection'
    )
    
    math_subparsers = math_parser.add_subparsers(dest='math_command', help='Mathematical analysis commands')
    
    # Spectral decomposition command
    spectral_parser = math_subparsers.add_parser(
        'spectral',
        help='🔬 Advanced spectral decomposition analysis',
        description='Multi-level spectral analysis for gradient obfuscation detection'
    )
    spectral_parser.add_argument('--input', '-i', required=True, help='Input gradient data file (.npy)')
    spectral_parser.add_argument('--output', '-o', help='Output analysis results (.json)')
    spectral_parser.add_argument('--levels', '-l', type=int, default=5, help='Number of decomposition levels')
    spectral_parser.add_argument('--device', '-d', choices=['auto', 'mps', 'cuda', 'cpu'], 
                                default='auto', help='Device preference for computation')
    spectral_parser.add_argument('--precision', '-p', choices=['float32', 'float64'], 
                                default='float32', help='Numerical precision')
    spectral_parser.add_argument('--plot', help='Save visualization plot to file')
    spectral_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Exponential integrator command
    integrator_parser = math_subparsers.add_parser(
        'integrate',
        help='⚡ Advanced exponential time differencing',
        description='ETD-RK4 integration for gradient evolution analysis'
    )
    integrator_parser.add_argument('--input', '-i', required=True, help='Input gradient data file (.npy)')
    integrator_parser.add_argument('--output', '-o', help='Output evolution results (.npy)')
    integrator_parser.add_argument('--steps', '-s', type=int, default=100, help='Number of integration steps')
    integrator_parser.add_argument('--dt', type=float, default=0.01, help='Time step size')
    integrator_parser.add_argument('--krylov-dim', type=int, default=30, help='Krylov subspace dimension')
    integrator_parser.add_argument('--device', '-d', choices=['auto', 'mps', 'cuda', 'cpu'], 
                                default='auto', help='Device preference for computation')
    integrator_parser.add_argument('--precision', '-p', choices=['float32', 'float64'], 
                                default='float32', help='Numerical precision')
    integrator_parser.add_argument('--plot', help='Save evolution plot to file')
    integrator_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Demonstration command
    demo_parser = math_subparsers.add_parser(
        'demo',
        help='🎯 Demonstrate mathematical capabilities',
        description='Run comprehensive demonstration of mathematical foundations'
    )
    demo_parser.add_argument('--device', '-d', choices=['auto', 'mps', 'cuda', 'cpu'], 
                            default='auto', help='Device preference for computation')
    demo_parser.add_argument('--precision', '-p', choices=['float32', 'float64'], 
                            default='float32', help='Numerical precision')
    demo_parser.add_argument('--save-results', help='Save demonstration results to file')
    demo_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Test command
    test_parser = math_subparsers.add_parser(
        'test',
        help='🧪 Run comprehensive test suite',
        description='Run complete mathematical foundations test suite'
    )
    test_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    test_parser.add_argument('--test-type', choices=['all', 'foundations', 'cli', 'spectral', 'integration', 'devices', 'performance'], 
                            default='all', help='Type of tests to run')
    
    # Set command handlers
    math_parser.set_defaults(func=handle_mathematical_command)

    # Lightweight plotting command to render saved evolution arrays
    plot_parser = math_subparsers.add_parser(
        'plot-evolution',
        help='🖼️ Plot evolution norms from a saved .npy file',
        description='Render a branded norm+guardrail triage chart from evol*.npy'
    )
    plot_parser.add_argument('--input', '-i', required=True, help='Input evolution file (.npy) [T,D]')
    plot_parser.add_argument('--output', '-o', required=True, help='Output plot (.png)')
    plot_parser.add_argument(
        '--baseline',
        default=None,
        help='Optional baseline evolution file (.npy) used to compute guardrail statistics (recommended).',
    )
    plot_parser.add_argument(
        '--guardrail-method',
        choices=['mu_sigma', 'median_mad', 'quantile'],
        default='mu_sigma',
        help='Guardrail statistic to compute from the baseline (default: mu_sigma).',
    )
    plot_parser.add_argument(
        '--k',
        type=float,
        default=2.0,
        help='Guardrail multiplier for mu_sigma/median_mad (guard = center + k*scale). Default: 2.0',
    )
    plot_parser.add_argument(
        '--q',
        type=float,
        default=0.99,
        help='Quantile for guardrail-method=quantile (default: 0.99).',
    )
    plot_parser.add_argument(
        '--title',
        default='NeurInSpectre — ETD-RK4 Norm Evolution (Guardrail + Breach Windows)',
        help='Plot title (NeurInSpectre branding is applied automatically if missing).',
    )
    plot_parser.set_defaults(plot_command='plot_evolution')

def handle_mathematical_command(args):
    """Handle mathematical analysis commands"""
    if args.math_command == 'spectral':
        return run_spectral_analysis(args)
    elif args.math_command == 'integrate':
        return run_integration_analysis(args)
    elif args.math_command == 'demo':
        return run_mathematical_demo(args)
    elif args.math_command == 'test':
        return run_test_suite(args)
    elif getattr(args, 'plot_command', None) == 'plot_evolution':
        return run_plot_evolution(args)
    else:
        logger.error("❌ Unknown mathematical command. Use 'spectral', 'integrate', 'demo', or 'test'")
        return 1

def run_spectral_analysis(args):
    """Run advanced spectral decomposition analysis"""
    try:
        # Import mathematical foundations
        from ..mathematical import GPUAcceleratedMathEngine
        
        logger.info("🔬 Starting Advanced Spectral Decomposition Analysis")
        logger.info(f"📊 Input: {args.input}")
        logger.info(f"📈 Levels: {args.levels}")
        logger.info(f"🖥️  Device: {args.device}")
        logger.info(f"🔢 Precision: {args.precision}")
        
        # Check input file
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"❌ Input file not found: {args.input}")
            return 1
        
        # Load gradient data
        logger.info("📥 Loading gradient data...")
        gradient_data = np.load(input_path)
        logger.info(f"📊 Data shape: {gradient_data.shape}")
        
        # Check for MPS and float64 incompatibility
        import torch
        precision_to_use = args.precision
        if args.device == 'auto' or args.device == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                if args.precision == 'float64':
                    logger.warning("⚠️  MPS (Apple Silicon) does not support float64")
                    logger.warning("   Automatically using float32 instead")
                    precision_to_use = 'float32'
        
        # Initialize mathematical engine
        logger.info("🚀 Initializing GPU Mathematical Engine...")
        math_engine = GPUAcceleratedMathEngine(
            precision=precision_to_use,
            device_preference=args.device
        )
        
        # Run spectral decomposition
        logger.info("🔍 Running spectral decomposition...")
        results = math_engine.advanced_spectral_decomposition(
            gradient_data, 
            decomposition_levels=args.levels
        )
        
        # Convert tensors to numpy for JSON serialization
        serializable_results = _convert_results_to_serializable(results)
        
        # Log key findings
        logger.info("📋 Analysis Results:")
        if 'summary_metrics' in results:
            summary = results['summary_metrics']
            if 'mean_entropy' in summary:
                entropy_val = summary['mean_entropy']
                try:
                    if hasattr(entropy_val, 'mean'):
                        val = entropy_val.mean().item()
                    else:
                        val = entropy_val.item()
                    if not np.isnan(val):
                        logger.info(f"  🔢 Mean Entropy: {val:.4f}")
                except Exception:
                    pass
                    
            if 'entropy_variance' in summary:
                variance_val = summary['entropy_variance']
                try:
                    if hasattr(variance_val, 'mean'):
                        val = variance_val.mean().item()
                    else:
                        val = variance_val.item()
                    if not np.isnan(val):
                        logger.info(f"  📊 Entropy Variance: {val:.4f}")
                except Exception:
                    pass
        
        if 'obfuscation_indicators' in results:
            indicators = results['obfuscation_indicators']
            if 'spectral_irregularity' in indicators:
                irregularity = indicators['spectral_irregularity']
                try:
                    val = irregularity.mean().item() if hasattr(irregularity, 'mean') else float(irregularity)
                    if not np.isnan(val):
                        logger.info(f"  ⚠️  Spectral Irregularity: {val:.4f}")
                except Exception:
                    pass
            if 'cross_level_consistency' in indicators:
                consistency = indicators['cross_level_consistency']
                logger.info(f"  🔗 Cross-level Consistency: {consistency.mean().item():.4f}")
        
        # Save results
        if args.output:
            logger.info(f"💾 Saving results to: {args.output}")
            with open(args.output, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        # Create visualization
        if args.plot:
            logger.info("🎨 Creating visualization...")
            _create_spectral_visualization(gradient_data, results, args.plot)
        
        logger.info("✅ Spectral analysis completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Spectral analysis failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def run_integration_analysis(args):
    """Run exponential time differencing integration analysis"""
    try:
        # Import mathematical foundations
        from ..mathematical import GPUAcceleratedMathEngine, AdvancedExponentialIntegrator
        
        logger.info("⚡ Starting Exponential Time Differencing Analysis")
        logger.info(f"📊 Input: {args.input}")
        logger.info(f"🔢 Steps: {args.steps}")
        logger.info(f"⏱️  Time step: {args.dt}")
        logger.info(f"🖥️  Device: {args.device}")
        
        # Check input file
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"❌ Input file not found: {args.input}")
            return 1
        
        # Load gradient data
        logger.info("📥 Loading gradient data...")
        gradient_data = np.load(input_path)
        logger.info(f"📊 Data shape: {gradient_data.shape}")
        
        # Check for MPS and float64 incompatibility
        import torch
        precision_to_use = args.precision
        if args.device == 'auto' or args.device == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                if args.precision == 'float64':
                    logger.warning("⚠️  MPS (Apple Silicon) does not support float64")
                    logger.warning("   Automatically using float32 instead")
                    precision_to_use = 'float32'
        
        # Initialize mathematical engine
        logger.info("🚀 Initializing GPU Mathematical Engine...")
        math_engine = GPUAcceleratedMathEngine(
            precision=precision_to_use,
            device_preference=args.device
        )
        
        # Initialize integrator
        integrator = AdvancedExponentialIntegrator(math_engine)
        
        # Define nonlinear function for gradient evolution
        def nonlinear_func(u):
            import torch
            return -0.1 * u**3 + 0.05 * torch.sin(u) + 0.01 * torch.cos(2 * u)
        
        # Run integration
        logger.info("🔄 Running ETD-RK4 integration...")
        
        # Use subset of data for integration (limit size for performance)
        max_size = min(gradient_data.size, 1000)
        u_initial = gradient_data.flatten()[:max_size]
        
        evolution_history = []
        # ensure 1D float32 numpy for consistency
        current_state = np.asarray(u_initial, dtype=np.float32).ravel()
        
        for step in range(args.steps):
            if args.verbose and step % 10 == 0:
                logger.info(f"  Step {step}/{args.steps}")
            
            # Perform integration step
            next_state = integrator.etd_rk4_step(
                current_state, 
                None,  # Use default linear operator
                nonlinear_func, 
                args.dt,
                args.krylov_dim
            )
            
            # Store evolution (as 1D float32 numpy)
            try:
                ns = next_state.detach().cpu().numpy().astype(np.float32).ravel()
            except Exception:
                # if already numpy-like
                ns = np.asarray(next_state, dtype=np.float32).ravel()
            evolution_history.append(ns)
            current_state = ns
        
        # Stack to strict 2D [T, D]
        try:
            evolution_array = np.stack(evolution_history, axis=0)
        except Exception:
            # fallback: coerce differing lengths by truncating to min length
            min_d = min(arr.shape[0] for arr in evolution_history)
            evolution_array = np.stack([arr[:min_d] for arr in evolution_history], axis=0)
        
        # Log evolution statistics
        initial_norm = np.linalg.norm(u_initial)
        final_norm = np.linalg.norm(evolution_array[-1])
        denom = float(initial_norm) if float(initial_norm) != 0.0 else 1.0
        rel_change = float((final_norm - initial_norm) / denom)
        logger.info("📊 Evolution Statistics:")
        logger.info(f"  Initial norm: {initial_norm:.6f}")
        logger.info(f"  Final norm: {final_norm:.6f}")
        logger.info(f"  Relative change: {rel_change:.6f}")
        
        # Save evolution results
        if args.output:
            logger.info(f"💾 Saving evolution to: {args.output}")
            np.save(args.output, evolution_array)
        
        # Create visualization
        if args.plot:
            logger.info("🎨 Creating evolution plot...")
            _create_evolution_visualization(evolution_array, args.plot)
        
        logger.info("✅ Integration analysis completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Integration analysis failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def run_mathematical_demo(args):
    """Run comprehensive mathematical demonstration"""
    try:
        # Import mathematical foundations
        from ..mathematical import demonstrate_advanced_mathematics
        
        logger.info("🎯 Starting Mathematical Foundations Demonstration")
        logger.info(f"🖥️  Device: {args.device}")
        logger.info(f"🔢 Precision: {args.precision}")
        
        # Run demonstration
        results = demonstrate_advanced_mathematics(
            device_preference=args.device,
            precision=args.precision
        )
        
        # Save results if requested
        if args.save_results:
            logger.info(f"💾 Saving demonstration results to: {args.save_results}")
            
            # Convert results to serializable format
            serializable_results = {}
            for key, value in results.items():
                if key in ['math_engine', 'integrator']:
                    # Skip non-serializable objects
                    continue
                else:
                    serializable_results[key] = _convert_results_to_serializable(value)
            
            with open(args.save_results, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        logger.info("✅ Mathematical demonstration completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Mathematical demonstration failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def run_test_suite(args):
    """Run comprehensive test suite"""
    try:
        # Import mathematical foundations test suite
        from ..mathematical import run_test_suite as run_math_tests
        
        logger.info("🧪 Starting Mathematical Foundations Test Suite")
        logger.info(f"📊 Test type: {args.test_type}")
        logger.info(f"🔢 Verbose: {args.verbose}")
        
        # Run test suite
        success = run_math_tests(verbose=args.verbose)
        
        if success:
            logger.info("✅ All tests passed!")
            return 0
        else:
            logger.error("❌ Some tests failed!")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def _convert_results_to_serializable(results):
    """Convert PyTorch tensors to serializable format"""
    if isinstance(results, dict):
        serializable = {}
        for key, value in results.items():
            serializable[key] = _convert_results_to_serializable(value)
        return serializable
    elif hasattr(results, 'cpu') and hasattr(results, 'numpy'):
        # PyTorch tensor
        return results.cpu().numpy().tolist()
    elif isinstance(results, np.ndarray):
        return results.tolist()
    elif isinstance(results, (list, tuple)):
        return [_convert_results_to_serializable(item) for item in results]
    else:
        return results

def run_plot_evolution(args):
    """Render a branded norm+guardrail triage plot from a saved evolution array."""
    try:
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from contextlib import nullcontext

        def _brand_title(title: str) -> str:
            t = str(title or '').strip()
            if not t:
                return "NeurInSpectre — ETD-RK4 Norm Evolution (Guardrail + Breach Windows)"
            if t.lower().startswith("neurinspectre"):
                return t
            # Strip common leading punctuation / separators.
            t2 = t.lstrip("—-• ").strip()
            return f"NeurInSpectre — {t2}" if t2 else "NeurInSpectre — ETD-RK4 Norm Evolution (Guardrail + Breach Windows)"

        def _to_norm_series(a: np.ndarray) -> np.ndarray:
            a = np.asarray(a)
            if a.ndim == 1:
                norms_ = np.abs(a)
            elif a.ndim == 2:
                norms_ = np.linalg.norm(a, axis=1)
            else:
                a2 = a.reshape(a.shape[0], -1)
                norms_ = np.linalg.norm(a2, axis=1)
            norms_ = np.asarray(norms_, dtype=np.float64).reshape(-1)
            norms_ = norms_[np.isfinite(norms_)]
            return norms_

        def _guardrail_stats(base: np.ndarray, method: str, k: float, q: float) -> tuple[float, dict]:
            b = np.asarray(base, dtype=np.float64).reshape(-1)
            b = b[np.isfinite(b)]
            if b.size < 2:
                return float("nan"), {"baseline_n": int(b.size)}

            mu = float(b.mean())
            sd = float(b.std(ddof=0) + 1e-12)
            med = float(np.median(b))
            mad = float(np.median(np.abs(b - med)) + 1e-12)
            _MAD_TO_SIGMA = 0.6744897501960817  # Normal: MAD ≈ 0.67449σ
            mad_sigma = float(mad / _MAD_TO_SIGMA)

            info = {
                "baseline_n": int(b.size),
                "baseline_mu": mu,
                "baseline_sigma": sd,
                "baseline_median": med,
                "baseline_mad": mad,
                "baseline_mad_sigma": mad_sigma,
            }

            m = str(method or "mu_sigma").strip().lower()
            if m == "median_mad":
                guard_ = float(med + float(k) * mad_sigma)
                info["guardrail_label"] = f"median+k·MADσ (k={float(k):g})"
                return guard_, info
            if m == "quantile":
                qq = float(q)
                qq = float(np.clip(qq, 0.5, 0.999))
                guard_ = float(np.quantile(b, qq))
                info["guardrail_label"] = f"quantile (q={qq:.3f})"
                return guard_, info

            # Default: mu_sigma
            guard_ = float(mu + float(k) * sd)
            info["guardrail_label"] = f"μ+kσ (k={float(k):g})"
            return guard_, info

        title = _brand_title(getattr(args, "title", None))

        # Load main series
        arr = np.load(args.input)
        norms = _to_norm_series(arr)
        if norms.size < 2:
            raise ValueError("Evolution series too small to plot.")

        # Guardrail baseline: default to self-series (less precise), but prefer explicit baseline.
        baseline_path = getattr(args, "baseline", None)
        if baseline_path:
            base_arr = np.load(str(baseline_path))
            base_norms = _to_norm_series(base_arr)
            guard_src = f"baseline={Path(str(baseline_path)).name}"
        else:
            base_norms = norms
            guard_src = "baseline=self (recommend: --baseline <clean.npy>)"

        guard_method = getattr(args, "guardrail_method", "mu_sigma")
        k = float(getattr(args, "k", 2.0))
        q = float(getattr(args, "q", 0.99))
        guard, stats = _guardrail_stats(base_norms, guard_method, k=k, q=q)
        if not np.isfinite(guard):
            raise ValueError("Guardrail could not be computed (baseline too small or non-finite).")

        t = np.arange(norms.size, dtype=int)
        breach = norms > guard
        near_margin = max(0.02 * guard, 0.10 * float(np.std(base_norms) + 1e-12))
        near = (norms >= (guard - near_margin)) & (~breach)

        # Contiguous breach windows
        breach_idx = np.where(breach)[0]
        windows: list[tuple[int, int]] = []
        if breach_idx.size:
            s = int(breach_idx[0])
            p = int(breach_idx[0])
            for ii in breach_idx[1:]:
                ii = int(ii)
                if ii == p + 1:
                    p = ii
                else:
                    windows.append((s, p))
                    s = p = ii
            windows.append((s, p))

        breach_steps = int(breach.sum())
        breach_frac = float(breach_steps / float(norms.size))
        max_norm = float(np.max(norms))
        min_norm = float(np.min(norms))
        max_t = int(np.argmax(norms))
        first_breach = int(breach_idx[0]) if breach_idx.size else None
        longest_window = int(max((b - a + 1) for (a, b) in windows)) if windows else 0

        # --- Styling (local-only; no global state mutation beyond this call)
        style_ctx = nullcontext()
        try:
            style_ctx = plt.style.context("seaborn-v0_8-whitegrid")
        except Exception:
            pass

        with style_ctx, mpl.rc_context(
            {
                "axes.titlesize": 14,
                "axes.labelsize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "figure.titlesize": 16,
                "font.family": "DejaVu Sans",
            }
        ):
            fig = plt.figure(figsize=(12.8, 6.9), dpi=180)
            gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.35], hspace=0.22, wspace=0.12)

            ax = fig.add_subplot(gs[0, :])
            ax_blue = fig.add_subplot(gs[1, 0])
            ax_red = fig.add_subplot(gs[1, 1])
            ax_blue.axis("off")
            ax_red.axis("off")

            # Main plot
            ax.plot(t, norms, color="#C0392B", lw=2.2, label="L2 norm")
            ax.axhline(
                guard,
                color="#2C3E50",
                lw=1.8,
                ls=(0, (6, 3)),
                label=f"Guardrail ({stats.get('guardrail_label','μ+kσ')} = {guard:.2f})",
            )
            # Shaded bands
            ax.axhspan(guard - near_margin, guard, color="#F39C12", alpha=0.08, lw=0, label="Near guardrail")
            if np.any(breach):
                ax.fill_between(t, guard, norms, where=breach, color="#E74C3C", alpha=0.14, interpolate=True, label="Breach")
                ax.scatter(t[breach], norms[breach], s=18, color="#E74C3C", edgecolor="white", linewidth=0.4, zorder=4)

            # Breach window shading (helps readability when breaches are contiguous)
            for (a, b) in windows[:80]:
                ax.axvspan(a - 0.5, b + 0.5, color="#E74C3C", alpha=0.06, lw=0, zorder=0)

            # Key annotations
            ax.scatter([max_t], [max_norm], s=36, color="#C0392B", edgecolor="white", linewidth=0.6, zorder=5)
            ax.annotate(
                f"max={max_norm:.2f}",
                xy=(max_t, max_norm),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D0D0D0", alpha=0.95),
            )
            if first_breach is not None:
                ax.annotate(
                    f"first breach @ t={first_breach}",
                    xy=(first_breach, float(norms[first_breach])),
                    xytext=(12, -18),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF5E6", edgecolor="#F39C12", alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color="#F39C12", lw=1.0),
                )

            ax.set_title(title, pad=12)
            ax.set_xlabel("Integration step")
            ax.set_ylabel("L2 norm")
            ax.grid(True, which="major", alpha=0.35)
            ax.set_xlim(int(t.min()), int(t.max()))

            # Summary box (top-right inside axes)
            sum_lines = [
                "Summary",
                f"steps={int(norms.size)}",
                f"guardrail={guard:.2f} ({guard_src})",
                f"min={min_norm:.2f}  max={max_norm:.2f}",
                f"breach_steps={breach_steps} ({breach_frac*100:.1f}%)",
                f"breach_windows={len(windows)}  longest={longest_window}",
                f"near_steps={int(near.sum())}",
            ]
            ax.text(
                0.995,
                0.98,
                "\n".join(sum_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D0D0D0", alpha=0.96),
            )

            leg = ax.legend(loc="upper left", frameon=True, framealpha=0.92)
            leg.get_frame().set_edgecolor("#D0D0D0")

            # Next steps panels (keep “red team” guidance test-focused; no evasion instructions)
            blue_text = (
                "Blue team — practical next steps\n"
                "• Calibrate guardrail on clean baseline: --baseline <clean.npy>\n"
                "• Alert on breach windows + persistence (contiguous runs > single spikes)\n"
                "• Correlate with other signals: spectral / anomaly / drift-detect\n"
                "• Respond: quarantine batch/run; tighten clipping/DP; retrain if recurrent"
            )
            ax_blue.text(
                0.0,
                1.0,
                blue_text,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.55", facecolor="#ECF3FF", edgecolor="#1F5FBF", alpha=0.95),
            )

            red_text = (
                "Red team — validation checklist (safe)\n"
                "• Run controlled adversarial scenarios; record if/when breaches occur\n"
                "• Measure lead time + false positives vs baseline runs\n"
                "• Stress-test patterns: spikes, slow drift, periodic artifacts\n"
                "• Report: save plot + thresholds used; attach breach windows"
            )
            ax_red.text(
                0.0,
                1.0,
                red_text,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.55", facecolor="#FFF1F2", edgecolor="#B91C1C", alpha=0.95),
            )

            # Footer note for interpretability
            fig.text(
                0.01,
                0.01,
                "Note: Guardrail is a heuristic control-limit. For precision, compute it on a clean baseline and re-validate after distribution shifts.",
                fontsize=8.5,
                color="#555",
            )

            out_path = Path(str(args.output))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
            plt.close(fig)

        print(str(Path(args.output)))
        return 0
    except Exception as e:
        logger.error(f"Failed to plot evolution: {e}")
        return 1

def _create_spectral_visualization(gradient_data, results, output_file):
    """Create INTERACTIVE spectral analysis visualization with Plotly"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create interactive figure with subplots
        # Increased vertical_spacing to prevent text overlap between rows
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '📊 Original Gradient Signal',
                '🎯 Spectral Magnitude (Level 0)',
                '⚠️ Obfuscation Indicators',
                '📈 Summary Metrics'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'bar'}]],
            horizontal_spacing=0.12,
            vertical_spacing=0.28,  # Increased from 0.20 to prevent text overlap
            row_heights=[0.42, 0.58]  # Give bottom row more space for angled labels
        )
        
        # Flatten gradient data
        if len(gradient_data.shape) > 1:
            grad_flat = gradient_data.reshape(-1)
        else:
            grad_flat = gradient_data
        
        # Sample for display (full data still zoomable)
        sample_size = min(len(grad_flat), 10000)  # Show up to 10k points
        grad_sample = grad_flat[:sample_size]
        
        # Calculate threat levels for each gradient point
        threat_levels = []
        red_team_actions = []
        blue_team_actions = []
        
        for g in grad_sample:
            abs_g = abs(g)
            if abs_g > 2.0:
                threat_levels.append("🔴 CRITICAL")
                red_team_actions.append("EXPLOIT: High gradient for data extraction")
                blue_team_actions.append("URGENT: Clip to ≤1.0, increase DP noise")
            elif abs_g > 1.0:
                threat_levels.append("🟠 HIGH")
                red_team_actions.append("TARGET: Moderate gradient for MI attacks")
                blue_team_actions.append("ACTION: Monitor and apply masking")
            elif abs_g > 0.5:
                threat_levels.append("🟡 MEDIUM")
                red_team_actions.append("PROBE: Useful for model probing")
                blue_team_actions.append("MONITOR: Track over time")
            else:
                threat_levels.append("🟢 LOW")
                red_team_actions.append("SKIP: Minimal value")
                blue_team_actions.append("BASELINE: Normal operation")
        
        # Plot 1: Original gradient signal with INTERACTIVE HOVER
        marker_colors = ['red' if abs(g) > 2.0 else 'orange' if abs(g) > 1.0 else 'yellow' if abs(g) > 0.5 else 'green' 
                        for g in grad_sample]
        
        fig.add_trace(go.Scatter(
            x=list(range(len(grad_sample))),
            y=grad_sample,
            mode='lines+markers',
            name='Gradient Signal',
            line=dict(color='blue', width=1),
            marker=dict(size=3, color=marker_colors),
            customdata=list(zip(threat_levels, red_team_actions, blue_team_actions)),
            hovertemplate=
                '<b>🎯 GRADIENT ANALYSIS</b><br>' +
                '━━━━━━━━━━━━━━━━━━━━━━<br>' +
                '<b>Index:</b> %{x}<br>' +
                '<b>Value:</b> %{y:.6f}<br>' +
                '<b>Absolute:</b> ' + '%{y}' + '<br>' +
                '<b>Threat:</b> %{customdata[0]}<br>' +
                '<b>🔴 Red Team:</b> %{customdata[1]}<br>' +
                '<b>🔵 Blue Team:</b> %{customdata[2]}<br>' +
                '<extra></extra>'
        ), row=1, col=1)
        
        # Plot 2: Spectral magnitude with INTERACTIVE HOVER
        if 'spectral_levels' in results and 'level_0' in results['spectral_levels']:
            magnitude = results['spectral_levels']['level_0']['magnitude']
            if hasattr(magnitude, 'cpu'):
                magnitude = magnitude.cpu().numpy()
            
            mag_data = np.abs(magnitude[0]) if magnitude.ndim > 1 else np.abs(magnitude)
            
            # Calculate frequency interpretations
            freq_info = []
            for i, mag_val in enumerate(mag_data):
                if mag_val > np.percentile(mag_data, 95):
                    freq_info.append("🚨 HIGH POWER - Obfuscation signature")
                elif mag_val > np.percentile(mag_data, 75):
                    freq_info.append("⚠️ ELEVATED - Periodic pattern detected")
                else:
                    freq_info.append("Normal frequency component")
            
            fig.add_trace(go.Scatter(
                x=list(range(len(mag_data))),
                y=mag_data,
                mode='lines',
                name='Spectral Magnitude',
                line=dict(color='red', width=2),
                customdata=freq_info,
                hovertemplate=
                    '<b>🎯 SPECTRAL ANALYSIS</b><br>' +
                    '━━━━━━━━━━━━━━━━━━━━━━<br>' +
                    '<b>Frequency Bin:</b> %{x}<br>' +
                    '<b>Magnitude:</b> %{y:.2e}<br>' +
                    '<b>Assessment:</b> %{customdata}<br>' +
                    '<b>🔴 Red Team:</b> High peaks = obfuscation signatures<br>' +
                    '<b>🔵 Blue Team:</b> Monitor dominant frequencies<br>' +
                    '<extra></extra>'
            ), row=1, col=2)
            
            # Set log scale for y-axis
            fig.update_yaxes(type="log", row=1, col=2)
        
        # Plot 3: Obfuscation indicators with INTERACTIVE HOVER
        if 'obfuscation_indicators' in results:
            indicators = results['obfuscation_indicators']
            indicator_names = []
            indicator_values = []
            mitre_mapping = []
            red_guidance = []
            blue_guidance = []
            
            for key, value in indicators.items():
                try:
                    # Handle different types safely
                    if hasattr(value, 'cpu'):
                        value = value.cpu().numpy()
                    
                    # Convert to scalar safely
                    if hasattr(value, 'mean'):
                        val = float(value.mean())
                    elif hasattr(value, 'item'):
                        val = float(value.item())
                    elif isinstance(value, (np.ndarray, list)):
                        val = float(np.mean(value)) if len(value) > 0 else 0.0
                    elif isinstance(value, (int, float)):
                        val = float(value)
                    else:
                        val = 0.0
                    
                    indicator_names.append(key.replace('_', ' ').title())
                    indicator_values.append(val)
                    
                    # Map to MITRE ATLAS and add guidance
                    if 'spectral' in key.lower():
                        mitre_mapping.append("AML.T0043 Craft Adversarial Data")
                        red_guidance.append("HIGH spectral anomaly = data extraction opportunity")
                        blue_guidance.append("Apply spectral filtering, increase noise")
                    elif 'cross' in key.lower():
                        mitre_mapping.append("AML.T0043 Craft Adversarial Data")
                        red_guidance.append("Cross-level correlation useful for attack")
                        blue_guidance.append("Monitor cross-layer dependencies")
                    elif 'correlation' in key.lower():
                        mitre_mapping.append("AML.T0024.001 Invert AI Model")
                        red_guidance.append("Correlation reveals model structure")
                        blue_guidance.append("Decorrelate gradients across batches")
                    elif 'progression' in key.lower():
                        mitre_mapping.append("AML.T0024.000 Infer Training Data Membership")
                        red_guidance.append("Progression patterns indicate training data")
                        blue_guidance.append("Randomize training order")
                    elif 'high_frequency' in key.lower():
                        mitre_mapping.append("AML.T0068 LLM Prompt Obfuscation")
                        red_guidance.append("HF noise used to evade detection")
                        blue_guidance.append("Apply low-pass filtering")
                    else:
                        mitre_mapping.append("General Indicator")
                        red_guidance.append("Analyze for exploitation potential")
                        blue_guidance.append("Monitor this metric")
                
                except Exception as e:
                    logger.warning(f"Skipping indicator {key}: {e}")
                    continue
            
            if indicator_names:
                bar_colors = ['red' if v > 100 else 'orange' if v > 20 else 'yellow' if v > 5 else 'green' 
                             for v in indicator_values]
                
                fig.add_trace(go.Bar(
                    x=indicator_names,
                    y=indicator_values,
                    name='Obfuscation Indicators',
                    marker_color=bar_colors,
                    customdata=list(zip(mitre_mapping, red_guidance, blue_guidance)),
                    hovertemplate=
                        '<b>⚠️ OBFUSCATION INDICATOR</b><br>' +
                        '━━━━━━━━━━━━━━━━━━━━━━<br>' +
                        '<b>Indicator:</b> %{x}<br>' +
                        '<b>Value:</b> %{y:.3f}<br>' +
                        '<b>🛡️ MITRE ATLAS:</b> %{customdata[0]}<br>' +
                        '<b>🔴 Red Team:</b> %{customdata[1]}<br>' +
                        '<b>🔵 Blue Team:</b> %{customdata[2]}<br>' +
                        '<extra></extra>'
                ), row=2, col=1)
        
        # Plot 4: Summary metrics with INTERACTIVE HOVER
        if 'summary_metrics' in results:
            summary = results['summary_metrics']
            metric_names = []
            metric_values = []
            interpretations = []
            
            # Debug: Print what we have
            logger.info(f"Summary metrics available: {list(summary.keys())}")
            
            for key, value in summary.items():
                # Convert tensor to numpy first
                if hasattr(value, 'cpu'):
                    value = value.cpu().numpy()
                
                # Convert to scalar - COMPREHENSIVE handling
                try:
                    # First try: Use squeeze to remove dimensions of size 1
                    if isinstance(value, np.ndarray):
                        value_squeezed = np.squeeze(value)  # Remove singleton dimensions
                        if value_squeezed.size == 1:
                            val = float(value_squeezed.flat[0])  # Use flat[0] instead of item()
                        elif value_squeezed.size > 1:
                            val = float(np.mean(value_squeezed))  # Multi-element: use mean
                        else:
                            val = 0.0
                    elif hasattr(value, 'item'):
                        val = float(value.item())
                    elif isinstance(value, (list, tuple)):
                        val = float(np.mean(value)) if len(value) > 0 else 0.0
                    elif isinstance(value, (int, float)):
                        val = float(value)
                    else:
                        val = 0.0
                except Exception as conv_error:
                    logger.warning(f"Conversion error for {key}: {conv_error}")
                    val = 0.0
                    
                metric_names.append(key.replace('_', ' ').title())
                
                # Handle NaN and inf values - SHOW THEM with interpretation
                if np.isnan(val):
                    logger.info(f"Metric {key} is NaN - showing as warning bar")
                    metric_values.append(0.1)  # Small visible value
                    interpretations.append("⚠️ NaN - Need more decomposition levels (increase data size)")
                elif np.isinf(val):
                    logger.info(f"Metric {key} is inf - showing as alert")
                    metric_values.append(1000)  # Visible capped value
                    interpretations.append("🚨 Inf - Extreme anomaly detected")
                else:
                    metric_values.append(val)
                    
                    # Add interpretations for VALID values
                    if 'entropy' in key.lower():
                        if val > 4.0:
                            interpretations.append("High entropy = complex/obfuscated gradients")
                        else:
                            interpretations.append("Normal entropy = clean gradients")
                    elif 'variance' in key.lower():
                        if val > 1.0:
                            interpretations.append("High variance = unstable/attacked")
                        else:
                            interpretations.append("Normal variance")
                    elif 'centroid' in key.lower():
                        interpretations.append("Spectral center of mass")
                    elif 'deviation' in key.lower():
                        if val > 100000:
                            interpretations.append("Extreme deviation = likely obfuscation")
                        else:
                            interpretations.append("Normal spectral deviation")
                    elif 'flatness' in key.lower():
                        interpretations.append("Spectral flatness metric")
                    else:
                        interpretations.append(f"{key} measurement")
            
            # Debug logging
            logger.info(f"Summary Metrics to display: {len(metric_names)} metrics")
            logger.info(f"Metric names: {metric_names}")
            logger.info(f"Metric values: {metric_values}")
            
            if metric_names:
                bar_colors = ['red' if v > 200000 else 'orange' if v > 100000 else 'lightblue' for v in metric_values]
                
                fig.add_trace(go.Bar(
                    x=metric_names,
                    y=metric_values,
                    name='Summary Metrics',
                    marker_color=bar_colors,
                    customdata=interpretations,
                    hovertemplate=
                        '<b>📈 SUMMARY METRIC</b><br>' +
                        '━━━━━━━━━━━━━━━━━━━━━━<br>' +
                        '<b>Metric:</b> %{x}<br>' +
                        '<b>Value:</b> %{y:.3f}<br>' +
                        '<b>Interpretation:</b> %{customdata}<br>' +
                        '<extra></extra>',
                    text=[f'{v:.3f}' for v in metric_values],
                    textposition='outside'
                ), row=2, col=2)
        
        # Update layout for PROFESSIONAL INTERACTIVE EXPERIENCE
        fig.update_layout(
            title=dict(
                text='NeurInSpectre - Advanced Spectral Analysis',
                font=dict(size=20),
                x=0.5,
                xanchor='center'
            ),
            template='plotly_dark',
            height=1300,
            width=1600,
            showlegend=False,  # Individual charts are self-explanatory
            margin=dict(l=80, r=80, t=100, b=160),  # Reasonable bottom margin
            hoverlabel=dict(
                bgcolor="rgba(0,0,0,0.95)",
                font_size=12,
                font_family="monospace",
                bordercolor="cyan"
            )
        )
        
        # Update subplot title positions to prevent overlap
        for annotation in fig['layout']['annotations']:
            annotation['y'] = annotation['y'] + 0.02  # Move titles up slightly
            annotation['font'] = dict(size=14, color='white')
        
        # Add clear axis labels with adjusted standoff
        fig.update_xaxes(title_text="Sample Index", title_standoff=5, row=1, col=1)
        fig.update_yaxes(title_text="Gradient Value", row=1, col=1)
        
        fig.update_xaxes(title_text="Frequency Bin", title_standoff=5, row=1, col=2)
        fig.update_yaxes(title_text="Magnitude (log scale)", row=1, col=2)
        
        fig.update_xaxes(title_text="Indicator Type", tickangle=45, title_standoff=20, row=2, col=1)
        fig.update_yaxes(title_text="Indicator Value", row=2, col=1)
        
        fig.update_xaxes(title_text="Metric", tickangle=45, title_standoff=20, row=2, col=2)
        fig.update_yaxes(title_text="Metric Value", row=2, col=2)
        
        # Save as INTERACTIVE HTML
        html_file = output_file.replace('.png', '_interactive.html')
        
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'scrollZoom': True,  # Enable scroll to zoom
            'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
            'responsive': True
        }
        
        
        # Build red/blue team guidance text (avoiding f-string emoji issues)
        red_text = (
            "<b>RED TEAM INTELLIGENCE</b><br>"
            "<b>Panel 2 (Spectral):</b> large peaks / low entropy can indicate structured artifacts (compare to baseline)<br>"
            "<b>Panel 3 (Indicators):</b> elevated indicators suggest higher anomaly/leakage proxy (validate on your data)<br>"
            "<b>Next Step (authorized):</b> neurinspectre gradient_inversion recover --gradients grads.npy --out-prefix _cli_runs/ginv_<br>"
            "<b>Note:</b> treat these as diagnostics, not proof of exploitability"
        )
        
        blue_text = (
            "<b>BLUE TEAM DEFENSE</b><br>"
            "<b>If peaks >2.5:</b> Apply spectral filtering NOW<br>"
            "<b>Defense:</b> Low-pass filter + DP-SGD (epsilon=0.5)<br>"
            "<b>Monitor:</b> Run spectral every 100 batches<br>"
            "<b>Alert:</b> New peaks or entropy >4.0<br>"
            "<b>Research:</b> USENIX 2024, CCS 2024, NDSS 2024"
        )
        
        # Add annotations
        fig.add_annotation(
            text=red_text,
            xref="paper", yref="paper",
            x=0.05, y=-0.95,
            xanchor='left', yanchor='top',
            showarrow=False,
            font=dict(size=10, color='white', family='monospace'),
            align='left',
            bgcolor='rgba(220, 53, 69, 0.2)',
            bordercolor='#dc3545',
            borderwidth=2,
            borderpad=10
        )
        
        fig.add_annotation(
            text=blue_text,
            xref="paper", yref="paper",
            x=0.95, y=-0.95,
            xanchor='right', yanchor='top',
            showarrow=False,
            font=dict(size=10, color='white', family='monospace'),
            align='left',
            bgcolor='rgba(0, 123, 255, 0.2)',
            bordercolor='#007bff',
            borderwidth=2,
            borderpad=10
        )
        
        fig.add_annotation(
            text='<b>📚 Spectral Analysis:</b> FFT-based detection of gradient obfuscation via frequency anomalies, spectral peaks, and entropy metrics',
            xref="paper", yref="paper",
            x=0.5, y=-0.85,
            showarrow=False,
            font=dict(size=10, color='#6c757d', family='monospace'),
            align='center',
            xanchor='center'
        )
        
        fig.write_html(html_file, config=config)
        
        logger.info(f"✅ Interactive spectral analysis saved to: {html_file}")
        logger.info("🔍 Features: Zoom (scroll wheel), Pan (shift+drag), Hover (detailed tooltips)")
        logger.info("📊 Red/Blue team guidance included in all hover tooltips")

        # Also save a static PNG for reports (no Plotly image backend required).
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            from pathlib import Path as _Path

            outp = _Path(str(output_file))
            try:
                outp.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            with mpl.rc_context({'font.size': 11}):
                fig_mpl, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
                fig_mpl.suptitle('NeurInSpectre — Advanced Spectral Analysis', fontsize=16, fontweight='bold')

                # Panel 1: gradient signal (sampled)
                axes[0, 0].plot(grad_sample, linewidth=1.2, color='#1f5fbf')
                axes[0, 0].set_title('Gradient Signal (sample)')
                axes[0, 0].grid(True, alpha=0.25)

                # Panel 2: spectral magnitude (level 0)
                if 'spectral_levels' in results and 'level_0' in results['spectral_levels']:
                    magnitude = results['spectral_levels']['level_0'].get('magnitude', None)
                    if hasattr(magnitude, 'cpu'):
                        magnitude = magnitude.cpu().numpy()
                    if magnitude is not None:
                        mag_data = np.abs(magnitude[0]) if getattr(magnitude, "ndim", 0) > 1 else np.abs(magnitude)
                        axes[0, 1].plot(mag_data, linewidth=1.2, color='#cc0000')
                        axes[0, 1].set_yscale('log')
                axes[0, 1].set_title('Spectral Magnitude (Level 0)')
                axes[0, 1].grid(True, alpha=0.25)

                # Panel 3: obfuscation indicators
                if 'obfuscation_indicators' in results and isinstance(results['obfuscation_indicators'], dict):
                    keys = list(results['obfuscation_indicators'].keys())
                    vals = []
                    for k in keys:
                        v = results['obfuscation_indicators'].get(k, 0.0)
                        if hasattr(v, 'cpu'):
                            v = v.cpu().numpy()
                        try:
                            if isinstance(v, np.ndarray):
                                v = float(np.mean(v)) if v.size else 0.0
                            elif hasattr(v, 'item'):
                                v = float(v.item())
                            else:
                                v = float(v)
                        except Exception:
                            v = 0.0
                        vals.append(v)
                    if keys:
                        axes[1, 0].bar(range(len(keys)), vals, color='#e38b29')
                        axes[1, 0].set_xticks(range(len(keys)))
                        axes[1, 0].set_xticklabels([k.replace('_', ' ') for k in keys], rotation=25, ha='right')
                axes[1, 0].set_title('Obfuscation Indicators')
                axes[1, 0].grid(True, alpha=0.25, axis='y')

                # Panel 4: summary metrics
                if 'summary_metrics' in results and isinstance(results['summary_metrics'], dict):
                    keys = list(results['summary_metrics'].keys())
                    vals = []
                    for k in keys:
                        v = results['summary_metrics'].get(k, 0.0)
                        if hasattr(v, 'cpu'):
                            v = v.cpu().numpy()
                        try:
                            if isinstance(v, np.ndarray):
                                v = float(np.mean(v)) if v.size else 0.0
                            elif hasattr(v, 'item'):
                                v = float(v.item())
                            else:
                                v = float(v)
                        except Exception:
                            v = 0.0
                        vals.append(v)
                    if keys:
                        axes[1, 1].bar(range(len(keys)), vals, color='#2A9D8F')
                        axes[1, 1].set_xticks(range(len(keys)))
                        axes[1, 1].set_xticklabels([k.replace('_', ' ') for k in keys], rotation=25, ha='right')
                axes[1, 1].set_title('Summary Metrics')
                axes[1, 1].grid(True, alpha=0.25, axis='y')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig_mpl.savefig(str(outp), dpi=300, bbox_inches='tight')
                plt.close(fig_mpl)

            logger.info(f"📊 Static PNG saved to: {output_file}")
        except Exception as png_err:
            logger.warning(f"Static PNG generation failed: {png_err}")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to create spectral visualization: {str(e)}")

def _create_evolution_visualization(evolution_array, output_file):
    """Create interactive Plotly evolution visualization with hover data"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        # Downsample if needed
        max_T = 4000
        arr = evolution_array
        if arr.shape[0] > max_T:
            step = int(np.ceil(arr.shape[0] / max_T))
            arr = arr[::step]
        
        # Select dimensions with highest temporal variance
        var_per_dim = np.var(arr, axis=0)
        order = np.argsort(var_per_dim)[::-1]
        n_dims_to_plot = int(min(5, arr.shape[1]))
        sel = order[:n_dims_to_plot]
        
        # Create 2x2 subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('📊 Evolution Over Time (top-variance dims)', '📈 Norm Evolution',
                           '🌀 Phase Space Density', '📊 Final State Distribution'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'histogram'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Plot 1: Evolution over time with hover data
        time_steps = np.arange(arr.shape[0])
        for idx in sel:
            dim_idx = int(idx)
            gradient_slope = np.gradient(arr[:, dim_idx])
            
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=arr[:, dim_idx],
                mode='lines',
                name=f'Dim {dim_idx+1}',
                line=dict(width=2),
                hovertemplate=(
                    '<b>Dimension %{fullData.name}</b><br>' +
                    'Time Step: %{x}<br>' +
                    'Value: %{y:.4f}<br>' +
                    'Slope: %{customdata:.4f}<br>' +
                    '<b>🔴 Red Team:</b> Target high-variance dims<br>' +
                    '<b>🔵 Blue Team:</b> Monitor for anomalies<br>' +
                    '<extra></extra>'
                ),
                customdata=gradient_slope
            ), row=1, col=1)
        
        # Add steep-slope shading as rectangles
        if arr.shape[0] > 3:
            slopes = np.abs(np.gradient(arr[:, sel], axis=0))
            step_slope = slopes.max(axis=1)
            thr = step_slope.mean() + 2.0 * step_slope.std()
            steep_regions = np.where(step_slope > thr)[0]
            for t in steep_regions:
                fig.add_vrect(
                    x0=max(0, t-0.5), x1=min(arr.shape[0]-1, t+0.5),
                    fillcolor="red", opacity=0.15, layer="below", line_width=0,
                    row=1, col=1
                )
        
        # Plot 2: Norm evolution with threat indicators
        norms = np.linalg.norm(arr, axis=1)
        norm_change = np.gradient(norms)
        threat_level = np.abs(norm_change) / (np.max(np.abs(norm_change)) + 1e-10)
        
        colors = ['rgba(255,0,0,0.7)' if t > 0.5 else 'rgba(0,128,255,0.7)' 
                  for t in threat_level]
        
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=norms,
            mode='lines+markers',
            name='L2 Norm',
            line=dict(color='red', width=2),
            marker=dict(size=4, color=colors),
            hovertemplate=(
                '<b>Norm Evolution</b><br>' +
                'Time Step: %{x}<br>' +
                'L2 Norm: %{y:.4f}<br>' +
                'Rate of Change: %{customdata:.4f}<br>' +
                '<b>🔴 Red Team:</b> Rapid norm changes = detectable<br>' +
                '<b>🔵 Blue Team:</b> Alert on sudden norm spikes<br>' +
                '<b>MITRE ATLAS:</b> AML.T0043 (Craft Adversarial Data)<br>' +
                '<extra></extra>'
            ),
            customdata=norm_change
        ), row=1, col=2)
        
        # Plot 3: Phase space density with start/end markers
        if arr.shape[1] >= 2 and len(sel) >= 2:
            d1, d2 = int(sel[0]), int(sel[1])
            x = arr[:, d1]
            y = arr[:, d2]
            
            # 2D histogram for density
            fig.add_trace(go.Histogram2d(
                x=x,
                y=y,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title='Count', x=0.46),
                hovertemplate=(
                    '<b>Phase Space</b><br>' +
                    f'Dim {d1+1}: %{{x:.4f}}<br>' +
                    f'Dim {d2+1}: %{{y:.4f}}<br>' +
                    'Count: %{z}<br>' +
                    '<extra></extra>'
                )
            ), row=2, col=1)
            
            # Start/End markers
            fig.add_trace(go.Scatter(
                x=[x[0], x[-1]],
                y=[y[0], y[-1]],
                mode='markers+text',
                marker=dict(size=12, color=['green', 'red'], line=dict(width=2, color='white')),
                text=['START', 'END'],
                textposition='top center',
                name='Trajectory',
                hovertemplate=(
                    '<b>%{text}</b><br>' +
                    f'Dim {d1+1}: %{{x:.4f}}<br>' +
                    f'Dim {d2+1}: %{{y:.4f}}<br>' +
                    '<extra></extra>'
                )
            ), row=2, col=1)
        
        # Plot 4: Final state distribution with statistics
        final_state = arr[-1]
        mean_val = np.mean(final_state)
        std_val = np.std(final_state)
        
        fig.add_trace(go.Histogram(
            x=final_state.ravel(),
            nbinsx=30,
            marker=dict(color='purple', line=dict(color='black', width=1)),
            name='Final State',
            hovertemplate=(
                '<b>Final State Distribution</b><br>' +
                'Value Range: %{x}<br>' +
                'Frequency: %{y}<br>' +
                f'Mean: {mean_val:.4f}<br>' +
                f'Std Dev: {std_val:.4f}<br>' +
                '<b>🔴 Red Team:</b> Uniform dist = less detectable<br>' +
                '<b>🔵 Blue Team:</b> Check for anomalous distributions<br>' +
                '<extra></extra>'
            )
        ), row=2, col=2)
        
        # Update layout
        fig.update_xaxes(title_text="Time Step", row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(title_text="State Value", row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_xaxes(title_text="Time Step", row=1, col=2, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(title_text="L2 Norm", row=1, col=2, gridcolor='rgba(128,128,128,0.2)')
        
        if arr.shape[1] >= 2 and len(sel) >= 2:
            d1, d2 = int(sel[0]), int(sel[1])
            fig.update_xaxes(title_text=f"Dim {d1+1}", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
            fig.update_yaxes(title_text=f"Dim {d2+1}", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
        
        fig.update_xaxes(title_text="State Value", row=2, col=2, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(title_text="Frequency", row=2, col=2, gridcolor='rgba(128,128,128,0.2)')
        
        fig.update_layout(
            title=dict(
                text='⚡ NeurInSpectre ETD-RK4 Evolution Analysis - Interactive Dashboard',
                x=0.5,
                xanchor='center',
                font=dict(size=22, color='white')
            ),
            showlegend=True,
            legend=dict(
                x=0.01, 
                y=0.99, 
                bgcolor='rgba(0,0,0,0.7)', 
                font=dict(color='white', size=11),
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1
            ),
            height=1250,  # Increased height for guidance boxes below
            width=1800,   # Increased width for better readability
            template='plotly_dark',
            hovermode='closest',
            uirevision='constant',  # Preserve zoom/pan state
            plot_bgcolor='rgba(20,20,20,0.95)',
            paper_bgcolor='rgba(10,10,10,1)',
            margin=dict(l=100, r=100, t=100, b=320)  # Increased bottom margin for guidance boxes
        )
        
        # Add annotations for Red/Blue team guidance BELOW the graphs
        fig.add_annotation(
            text=(
                '<b>🔵 Blue Team Guidance:</b><br>' +
                '• Monitor steep-slope intervals (red shaded regions in Evolution plot)<br>' +
                '• Set guardrails at norm spikes and establish baseline thresholds<br>' +
                '• Focus detection on top-variance dimensions (shown in legend)<br>' +
                '• Alert on sudden changes in phase space trajectory<br>' +
                '• MITRE ATLAS: AML.T0043 (Craft Adversarial Data)'
            ),
            xref="paper", yref="paper",
            x=0.25, y=-0.24,  # Moved significantly further down
            showarrow=False,
            font=dict(size=11, color='white'),
            align="left",
            bgcolor='rgba(31,95,191,0.85)',
            bordercolor='#1f5fbf',
            borderwidth=2,
            borderpad=12,
            xanchor='center',
            width=800  # Fixed width for better formatting
        )
        
        fig.add_annotation(
            text=(
                '<b>🔴 Red Team Guidance:</b><br>' +
                '• Avoid detectable spikes in norm evolution (smooth gradual changes)<br>' +
                '• Induce gradual drift in non-dominant dimensions to evade detection<br>' +
                '• Maintain uniform/normal distributions in final state to avoid anomalies<br>' +
                '• Keep phase space trajectory smooth without abrupt transitions<br>' +
                '• MITRE ATLAS: AML.T0043 (Craft Adversarial Data)'
            ),
            xref="paper", yref="paper",
            x=0.75, y=-0.24,  # Moved significantly further down
            showarrow=False,
            font=dict(size=11, color='white'),
            align="left",
            bgcolor='rgba(204,0,0,0.85)',
            bordercolor='#cc0000',
            borderwidth=2,
            borderpad=12,
            xanchor='center',
            width=800  # Fixed width for better formatting
        )
        
        # Save as interactive HTML
        html_file = output_file.replace('.png', '_interactive.html')
        fig.write_html(html_file)
        logger.info(f"📊 Interactive evolution visualization saved to: {html_file}")
        logger.info("🔍 Features: Zoom (scroll wheel), Pan (click+drag), Hover (detailed tooltips)")
        logger.info("📊 Red/Blue team guidance included in all hover tooltips")
        
        # Also save static PNG for reports
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        with mpl.rc_context({'font.size': 11}):
            fig_mpl, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
            fig_mpl.suptitle('⚡ NeurInSpectre ETD-RK4 Evolution Analysis', fontsize=16, fontweight='bold')
            
            # Simplified static version
            for idx in sel:
                axes[0, 0].plot(arr[:, int(idx)], alpha=0.9, linewidth=1.4)
            axes[0, 0].set_title('Evolution Over Time')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend([f'Dim {int(i)+1}' for i in sel], fontsize=8)
            
            axes[0, 1].plot(norms, 'r-', linewidth=2)
            axes[0, 1].set_title('Norm Evolution')
            axes[0, 1].grid(True, alpha=0.3)
            
            if arr.shape[1] >= 2 and len(sel) >= 2:
                d1, d2 = int(sel[0]), int(sel[1])
                axes[1, 0].hexbin(arr[:, d1], arr[:, d2], gridsize=35, cmap='Blues')
                axes[1, 0].set_title('Phase Space Density')
            
            axes[1, 1].hist(final_state.ravel(), bins=30, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].set_title('Final State Distribution')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        logger.info(f"📊 Static PNG saved to: {output_file}")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to create evolution visualization: {str(e)}")
        # Fallback 1: simple, robust PNG with clear security cue (norms + guardrail)
        try:
            import matplotlib.pyplot as _plt
            norms = np.linalg.norm(evolution_array, axis=1)
            mu, sd = float(norms.mean()), float(norms.std()+1e-8)
            guard = mu + 2.0*sd
            _plt.figure(figsize=(12,5), dpi=150)
            _plt.plot(norms, 'r-', linewidth=1.6, label='Norm')
            _plt.axhline(guard, color='black', ls='--', label=f'μ+2σ={guard:.2f}')
            _plt.title('⚡ ETD-RK4 Norm Evolution with Guardrail')
            _plt.xlabel('Time Step')
            _plt.ylabel('L2 Norm')
            _plt.legend(frameon=False)
            # Keys as caption
            _plt.figtext(0.02, 0.01, 'Blue: investigate breaches; baseline guardrails; correlate with drift windows.', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6f0ff', edgecolor='#1f5fbf', alpha=0.95))
            _plt.figtext(0.56, 0.01, 'Red: avoid spikes; distribute drift; stay below guardrail.', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe6e6', edgecolor='#cc0000', alpha=0.95))
            _plt.tight_layout(rect=[0, 0.06, 1, 1])
            _plt.savefig(output_file, dpi=300, bbox_inches='tight')
            _plt.close()
            logger.info(f"📊 Fallback evolution visualization saved to: {output_file}")
        except Exception as ee:
            logger.warning(f"⚠️ Fallback visualization also failed: {ee}")
        # Fallback 2: robust Plotly HTML 4-panel layout
        try:
            import numpy as _np
            from plotly.subplots import make_subplots as _mk
            import plotly.graph_objects as _go
            html_out = output_file.rsplit('.', 1)[0] + '.html'
            fig = _mk(rows=2, cols=2, subplot_titles=(
                'Evolution Over Time (top-variance dims)',
                'Norm Evolution',
                'Phase Space Density',
                'Final State Distribution'
            ))
            # Top-variance dims
            var = _np.var(evolution_array, axis=0)
            order = _np.argsort(var)[::-1]
            sel = order[:min(5, evolution_array.shape[1])]
            for idx in sel:
                fig.add_trace(_go.Scatter(y=_np.asarray(evolution_array[:, int(idx)], dtype=float), mode='lines', name=f'Dim {int(idx)+1}'), row=1, col=1)
            # Norms
            norms = _np.linalg.norm(evolution_array, axis=1)
            fig.add_trace(_go.Scatter(y=norms, mode='lines', name='Norm'), row=1, col=2)
            # Phase space density
            if evolution_array.shape[1] >= 2:
                d1, d2 = int(order[0]), int(order[1])
                x = _np.asarray(evolution_array[:, d1], dtype=float).ravel()
                y = _np.asarray(evolution_array[:, d2], dtype=float).ravel()
                H, xedges, yedges = _np.histogram2d(x, y, bins=35)
                fig.add_trace(_go.Heatmap(z=H.T, x=xedges, y=yedges, colorscale='Blues', colorbar=dict(title='Count')), row=2, col=1)
            # Final state histogram
            final_state = _np.asarray(evolution_array[-1], dtype=float).ravel()
            fig.add_trace(_go.Histogram(x=final_state, nbinsx=30, marker_color='purple'), row=2, col=2)
            # Add Blue/Red annotations
            fig.add_annotation(text='Blue: monitor steep-slope intervals; guardrails at spikes; focus on top-variance dims.',
                               xref='paper', yref='paper', x=0.01, y=-0.1, showarrow=False,
                               bgcolor='#e6f0ff', bordercolor='#1f5fbf')
            fig.add_annotation(text='Red: avoid spikes; induce gradual drift in non-dominant dims.',
                               xref='paper', yref='paper', x=0.55, y=-0.1, showarrow=False,
                               bgcolor='#ffe6e6', bordercolor='#cc0000')
            fig.update_layout(height=800, width=1200, title_text='⚡ NeurInSpectre ETD-RK4 Evolution Analysis (Interactive)')
            fig.write_html(html_out)
            logger.info(f"📄 Interactive evolution visualization saved to: {html_out}")
        except Exception as ee2:
            logger.warning(f"⚠️ Plotly HTML fallback failed: {ee2}")

if __name__ == "__main__":
    # For testing
    pass 