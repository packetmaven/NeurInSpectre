"""
NeurInSpectre CLI: Temporal Analysis Commands
Integrates temporal_evolution.py functionality into the CLI system
"""

import numpy as np
import json
import sys
import os
import logging
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Support running this module as a script from a source checkout.
# When imported as part of the package/CLI, avoid sys.path side effects.
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from ..core.corner_case_detection.temporal_evolution import TemporalEvolutionAnalyzer
    _TEMPORAL_IMPORT_ERROR = None
except ImportError as e:
    # No synthetic/demo fallback: if the core analyzer can't be imported, fail fast at runtime.
    TemporalEvolutionAnalyzer = None  # type: ignore[assignment]
    _TEMPORAL_IMPORT_ERROR = str(e)

def add_temporal_analysis_parser(subparsers):
    """Add temporal analysis command to CLI"""
    parser = subparsers.add_parser(
        'temporal-analysis',
        help='🕐 Temporal evolution analysis for gradient obfuscation detection'
    )
    
    # Subcommands for temporal analysis
    temporal_subparsers = parser.add_subparsers(dest='temporal_command', help='Temporal analysis commands')
    
    # Sequence analysis command
    sequence_parser = temporal_subparsers.add_parser(
        'sequence',
        help='Analyze temporal sequence of gradients'
    )
    sequence_parser.add_argument('--input-dir', required=True, help='Directory containing gradient sequence files')
    sequence_parser.add_argument('--pattern', default='*.npy', help='File pattern to match (default: *.npy)')
    sequence_parser.add_argument('--window-size', type=int, default=10, help='Sliding window size (default: 10)')
    sequence_parser.add_argument('--step', type=int, default=5, help='Step size for sliding window (default: 5)')
    sequence_parser.add_argument('--output-report', help='Output JSON report file')
    sequence_parser.add_argument('--output-plot', help='Output visualization file')
    sequence_parser.add_argument('--scalogram', choices=['off','stft','cwt'], default='off', help='Add time–frequency scalogram')
    sequence_parser.add_argument('--band-marks', default='', help='Comma-separated normalized freq marks (e.g., 0.05,0.1,0.2)')
    sequence_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Progressive obfuscation detection
    progressive_parser = temporal_subparsers.add_parser(
        'progressive',
        help='Detect progressive obfuscation patterns'
    )
    progressive_parser.add_argument('--input-dir', required=True, help='Directory containing gradient sequence files')
    progressive_parser.add_argument('--pattern', default='*.npy', help='File pattern to match (default: *.npy)')
    progressive_parser.add_argument('--output-report', help='Output JSON report file')
    progressive_parser.add_argument('--output-plot', help='Output visualization file')
    progressive_parser.add_argument('--scalogram', choices=['off','stft','cwt'], default='off', help='Add time–frequency scalogram')
    progressive_parser.add_argument('--band-marks', default='', help='Comma-separated normalized freq marks (e.g., 0.05,0.1,0.2)')
    progressive_parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold (default: 0.5)')
    progressive_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Real-time monitoring
    monitor_parser = temporal_subparsers.add_parser(
        'monitor',
        help='Real-time temporal evolution monitoring'
    )
    monitor_parser.add_argument('--input-stream', required=True, help='Input stream file or directory')
    monitor_parser.add_argument('--window-size', type=int, default=20, help='Monitoring window size (default: 20)')
    monitor_parser.add_argument('--alert-threshold', type=float, default=0.7, help='Alert threshold (default: 0.7)')
    monitor_parser.add_argument('--interval', type=float, default=1.0, help='Polling interval in seconds (default: 1.0)')
    monitor_parser.add_argument('--max-iterations', type=int, default=0, help='Max polling iterations (0 = run until Ctrl-C) (default: 0)')
    monitor_parser.add_argument('--pattern', default='*.npy', help='When input-stream is a directory: glob pattern (default: *.npy)')
    monitor_parser.add_argument('--output-log', help='Output monitoring log file')
    monitor_parser.add_argument('--output-plot', help='Optional HTML summary visualization')
    monitor_parser.add_argument('--dashboard', action='store_true', help='Launch monitoring dashboard')
    monitor_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Trend analysis
    trend_parser = temporal_subparsers.add_parser(
        'trend',
        help='Analyze temporal trends in gradient sequences'
    )
    trend_parser.add_argument('--input-dir', required=True, help='Directory containing gradient sequence files')
    trend_parser.add_argument('--pattern', default='*.npy', help='File pattern to match (default: *.npy)')
    trend_parser.add_argument('--analysis-type', choices=['linear', 'quadratic', 'spectral', 'all'], 
                            default='all', help='Type of trend analysis (default: all)')
    trend_parser.add_argument('--output-report', help='Output JSON report file')
    trend_parser.add_argument('--output-plot', help='Output visualization file')
    trend_parser.add_argument('--scalogram', choices=['off','stft','cwt'], default='off', help='Add time–frequency scalogram')
    trend_parser.add_argument('--band-marks', default='', help='Comma-separated normalized freq marks (e.g., 0.05,0.1,0.2)')
    trend_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    parser.set_defaults(func=handle_temporal_analysis)

def handle_temporal_analysis(args):
    """Handle temporal analysis commands"""
    if args.temporal_command == 'sequence':
        return handle_sequence_analysis(args)
    elif args.temporal_command == 'progressive':
        return handle_progressive_analysis(args)
    elif args.temporal_command == 'monitor':
        return handle_temporal_monitoring(args)
    elif args.temporal_command == 'trend':
        return handle_trend_analysis(args)
    else:
        print("❌ No temporal analysis command specified. Use --help for options.")
        return 1

def handle_sequence_analysis(args):
    """Handle temporal sequence analysis"""
    if TemporalEvolutionAnalyzer is None:
        print("❌ TemporalEvolutionAnalyzer unavailable (import failed).")
        print(f"   Error: {_TEMPORAL_IMPORT_ERROR}")
        return 1
    if args.verbose:
        print("🕐 Starting temporal sequence analysis...")
        print(f"📁 Input directory: {args.input_dir}")
        print(f"🔍 Pattern: {args.pattern}")
        print(f"🪟 Window size: {args.window_size}")
        print(f"👣 Step size: {args.step}")
    
    # Load gradient sequence
    gradient_sequence = load_gradient_sequence(args.input_dir, args.pattern, args.verbose)
    
    if not gradient_sequence:
        print("❌ No gradient files found or loaded")
        return 1
    
    # Initialize analyzer
    analyzer = TemporalEvolutionAnalyzer(window_size=args.window_size, step=args.step)
    
    # Analyze sequence
    if args.verbose:
        print("🔍 Analyzing temporal sequence...")
    
    result = analyzer.analyze_temporal_sequence(gradient_sequence)
    
    # Print results
    print("\n📊 Temporal Sequence Analysis Results")
    print("=" * 50)
    print(f"🕐 Temporal Evolution Detected: {result['is_temporal_evolution']}")
    print(f"🎯 Confidence: {result['confidence']:.3f}")
    print(f"📈 Trend Type: {result['trend_type']}")
    
    if 'metrics' in result:
        print(f"📊 Mean Norm: {result['metrics'].get('mean_norm', 'N/A')}")
        print(f"📊 Std Norm: {result['metrics'].get('std_norm', 'N/A')}")
        if 'spectral_entropy' in result['metrics']:
            print(f"🌊 Spectral Entropy: {result['metrics']['spectral_entropy']:.3f}")
    
    # Save report
    if args.output_report:
        save_analysis_report(result, args.output_report, 'temporal_sequence')
        if args.verbose:
            print(f"💾 Report saved to: {args.output_report}")
    
    # Generate visualization
    if args.output_plot:
        create_temporal_sequence_plot(gradient_sequence, result, args.output_plot)
        if args.verbose:
            print(f"📊 Visualization saved to: {args.output_plot}")
    
    return 0

def handle_progressive_analysis(args):
    """Handle progressive obfuscation analysis"""
    if TemporalEvolutionAnalyzer is None:
        print("❌ TemporalEvolutionAnalyzer unavailable (import failed).")
        print(f"   Error: {_TEMPORAL_IMPORT_ERROR}")
        return 1
    if args.verbose:
        print("🕐 Starting progressive obfuscation analysis...")
        print(f"📁 Input directory: {args.input_dir}")
        print(f"🎯 Threshold: {args.threshold}")
    
    # Load gradient sequence
    gradient_sequence = load_gradient_sequence(args.input_dir, args.pattern, args.verbose)
    
    if not gradient_sequence:
        print("❌ No gradient files found or loaded")
        return 1
    
    # Initialize analyzer
    analyzer = TemporalEvolutionAnalyzer()
    
    # Analyze progressive obfuscation
    if args.verbose:
        print("🔍 Analyzing progressive obfuscation patterns...")
    
    result = analyzer.detect_progressive_obfuscation(gradient_sequence)
    
    # Print results
    print("\n📊 Progressive Obfuscation Analysis Results")
    print("=" * 50)
    print(f"🚨 Progressive Obfuscation Detected: {result['is_progressive_obfuscation']}")
    print(f"🎯 Confidence: {result['confidence']:.3f}")
    print(f"📈 Slope: {result['slope']:.6f}")
    print(f"📊 P-value: {result['p_value']:.6f}")
    
    if 'obfuscation_scores' in result:
        scores = result['obfuscation_scores']
        print(f"📊 Obfuscation Score Range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"📊 Mean Obfuscation Score: {np.mean(scores):.3f}")
    
    # Threat assessment
    if result['is_progressive_obfuscation'] and result['confidence'] > args.threshold:
        print("\n🚨 THREAT DETECTED: Progressive obfuscation pattern identified!")
        print("🛡️  Recommended actions:")
        print("   • Increase monitoring frequency")
        print("   • Review security policies")
        print("   • Consider additional countermeasures")
    
    # Save report
    if args.output_report:
        save_analysis_report(result, args.output_report, 'progressive_obfuscation')
        if args.verbose:
            print(f"💾 Report saved to: {args.output_report}")
    
    # Generate visualization
    if args.output_plot:
        create_progressive_obfuscation_plot(gradient_sequence, result, args.output_plot)
        if args.verbose:
            print(f"📊 Visualization saved to: {args.output_plot}")
    
    return 0

def handle_temporal_monitoring(args):
    """Handle real-time temporal monitoring"""
    if TemporalEvolutionAnalyzer is None:
        print("❌ TemporalEvolutionAnalyzer unavailable (import failed).")
        print(f"   Error: {_TEMPORAL_IMPORT_ERROR}")
        return 1
    if args.verbose:
        print("🕐 Starting real-time temporal monitoring...")
        print(f"📡 Input stream: {args.input_stream}")
        print(f"🪟 Window size: {args.window_size}")
        print(f"🚨 Alert threshold: {args.alert_threshold}")
    
    # Initialize analyzer
    analyzer = TemporalEvolutionAnalyzer(window_size=args.window_size)
    
    # Monitoring loop (NO simulation / NO synthetic data)
    print("\n🔄 Monitoring temporal evolution patterns (real data only)...")
    print("=" * 60)
    monitoring_log = []

    import time
    import glob

    stream = str(args.input_stream)
    interval = float(getattr(args, "interval", 1.0))
    max_iter = int(getattr(args, "max_iterations", 0) or 0)
    pattern = str(getattr(args, "pattern", "*.npy"))

    last_seen_mtime = None
    last_seen_dir_state = None

    def _load_one(p: str) -> np.ndarray:
        arr = np.load(p, allow_pickle=True)
        if getattr(arr, "dtype", None) is object and getattr(arr, "shape", ()) == ():
            arr = arr.item()
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.number):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _sequence_from_array(arr: np.ndarray) -> list[np.ndarray]:
        # If user provides a 2D array, treat axis0 as time.
        if arr.ndim == 0:
            return [arr.reshape(1)]
        if arr.ndim == 1:
            return [arr.reshape(-1)]
        if arr.ndim == 2:
            return [np.asarray(arr[i, :]).reshape(-1) for i in range(arr.shape[0])]
        a = arr.reshape(arr.shape[0], -1)
        return [np.asarray(a[i, :]).reshape(-1) for i in range(a.shape[0])]

    it = 0
    while True:
        it += 1

        seq: list[np.ndarray] = []
        source_label = None

        if os.path.isfile(stream):
            mtime = os.path.getmtime(stream)
            if last_seen_mtime is None or mtime != last_seen_mtime:
                last_seen_mtime = mtime
                arr = _load_one(stream)
                seq_all = _sequence_from_array(arr)
                w = int(args.window_size)
                seq = seq_all[-w:] if len(seq_all) > w else seq_all
                source_label = os.path.basename(stream)
        elif os.path.isdir(stream):
            files = sorted(glob.glob(os.path.join(stream, pattern)), key=lambda p: os.path.getmtime(p))
            if files:
                w = int(args.window_size)
                win = files[-w:]
                state = tuple((p, os.path.getmtime(p)) for p in win)
                if state != last_seen_dir_state:
                    last_seen_dir_state = state
                    seq = [_load_one(p).reshape(-1) for p in win]
                    source_label = os.path.basename(win[-1])
        else:
            print(f"❌ Input stream path not found: {os.path.abspath(stream)}")
            return 1

        if seq:
            result = analyzer.analyze_temporal_sequence(seq)
            timestamp = datetime.now().strftime("%H:%M:%S")
            confidence = float(result.get("confidence", 0.0) or 0.0)
            trend = result.get("trend_type", "none")
            src = f" [{source_label}]" if source_label else ""
            print(f"[{timestamp}]{src} Confidence: {confidence:.3f} | Trend: {trend}")

            is_alert = confidence > float(args.alert_threshold)
            if is_alert:
                print(
                    f"🚨 ALERT: Temporal evolution detected (confidence={confidence:.3f} > {float(args.alert_threshold):.3f})"
                )

            log_entry = {
                "timestamp": timestamp,
                "source": source_label,
                "confidence": confidence,
                "trend_type": trend,
                "alert": is_alert,
            }
            monitoring_log.append(log_entry)
            if args.output_log:
                append_to_log(log_entry, args.output_log)
        else:
            if args.verbose:
                print("⚠️  No real data available yet for monitoring window.")

        if max_iter > 0 and it >= max_iter:
            break
        time.sleep(max(0.0, interval))

    print("\n✅ Monitoring session completed")
    # Optional summary visualization
    try:
        if getattr(args, 'output_plot', None):
            from plotly.subplots import make_subplots as _mk
            import plotly.graph_objects as _go
            iters = [i+1 for i,_ in enumerate(monitoring_log)]
            confs = [float(e.get('confidence',0.0)) for e in monitoring_log]
            alerts = [1 if e.get('alert', False) else 0 for e in monitoring_log]
            fig = _mk(rows=1, cols=2, column_widths=[0.7,0.3], subplot_titles=(
                'Confidence over time', 'Alert Count'
            ))
            fig.add_trace(_go.Scatter(x=iters, y=confs, mode='lines+markers', name='confidence'), row=1, col=1)
            if any(alerts):
                fig.add_trace(_go.Scatter(x=[i for i,a in zip(iters,alerts) if a], y=[c for c,a in zip(confs,alerts) if a],
                                          mode='markers', marker=dict(color='red', size=8), name='alerts'), row=1, col=1)
            fig.add_trace(_go.Bar(x=['alerts'], y=[sum(alerts)], name='alerts'), row=1, col=2)
            fig.update_layout(height=520, width=980, title_text='Temporal Monitor Summary', showlegend=False)
            fig.write_html(args.output_plot)
            print(f"📄 Summary visualization saved to: {args.output_plot}")
    except Exception:
        pass
    return 0

def handle_trend_analysis(args):
    """Handle trend analysis"""
    if TemporalEvolutionAnalyzer is None:
        print("❌ TemporalEvolutionAnalyzer unavailable (import failed).")
        print(f"   Error: {_TEMPORAL_IMPORT_ERROR}")
        return 1
    if args.verbose:
        print("🕐 Starting trend analysis...")
        print(f"📁 Input directory: {args.input_dir}")
        print(f"📊 Analysis type: {args.analysis_type}")
    
    # Load gradient sequence
    gradient_sequence = load_gradient_sequence(args.input_dir, args.pattern, args.verbose)
    
    if not gradient_sequence:
        print("❌ No gradient files found or loaded")
        return 1
    
    # Initialize analyzer
    analyzer = TemporalEvolutionAnalyzer()
    
    # Analyze trends
    if args.verbose:
        print("🔍 Analyzing temporal trends...")
    
    result = analyzer.analyze_temporal_sequence(gradient_sequence)
    
    # Print detailed trend analysis
    print("\n📊 Temporal Trend Analysis Results")
    print("=" * 50)
    
    if 'trend_analysis' in result:
        trend_data = result['trend_analysis']
        print(f"📈 Significant Trend: {trend_data.get('has_significant_trend', False)}")
        print(f"🎯 Trend Confidence: {trend_data.get('trend_confidence', 0.0):.3f}")
        print(f"📊 Trend Type: {trend_data.get('trend_type', 'none')}")
        print(f"📈 Trend Slope: {trend_data.get('trend_slope', 0.0):.6f}")
        print(f"📊 P-value: {trend_data.get('trend_pvalue', 1.0):.6f}")
        print(f"📊 Linear R²: {trend_data.get('linear_r2', 0.0):.3f}")
    
    # Additional metrics
    if 'metrics' in result:
        metrics = result['metrics']
        print("\n📊 Additional Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.3f}")
    
    # Save report
    if args.output_report:
        save_analysis_report(result, args.output_report, 'trend_analysis')
        if args.verbose:
            print(f"💾 Report saved to: {args.output_report}")
    
    # Generate visualization
    if args.output_plot:
        create_trend_analysis_plot(gradient_sequence, result, args.output_plot)
        if args.verbose:
            print(f"📊 Visualization saved to: {args.output_plot}")
    
    return 0

def load_gradient_sequence(input_dir, pattern, verbose=False):
    """Load gradient sequence from directory"""
    import glob
    
    if verbose:
        print(f"🔍 Loading gradient sequence from {input_dir}...")
    
    # Find matching files
    search_pattern = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"❌ No files found matching pattern: {search_pattern}")
        print("   No synthetic/demo fallback will be generated.")
        return None
    
    if verbose:
        print(f"📁 Found {len(files)} gradient files")
    
    # Load gradients
    gradients = []
    for file_path in files:
        try:
            gradient = np.load(file_path)
            # Robustness: sanitize non-finite numeric values early so downstream analysis is stable.
            gradient = np.asarray(gradient)
            if np.issubdtype(gradient.dtype, np.number):
                gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
            gradients.append(gradient)
            if verbose:
                print(f"   ✅ Loaded: {os.path.basename(file_path)} (shape: {gradient.shape})")
        except Exception as e:
            print(f"   ❌ Failed to load {file_path}: {e}")
    
    return gradients

def save_analysis_report(result, output_file, analysis_type):
    """Save analysis report to JSON file"""
    report = {
        'analysis_type': analysis_type,
        'timestamp': datetime.now().isoformat(),
        'results': result
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

def append_to_log(log_entry, log_file):
    """Append log entry to file"""
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry, default=str) + '\n')

def create_temporal_sequence_plot(gradient_sequence, result, output_file):
    """Create temporal sequence visualization"""
    fig = make_subplots(
        rows=2, cols=3,
        column_widths=[0.42, 0.29, 0.29],
        subplot_titles=['Gradient Norms Over Time', 'Trend Analysis', 'Spectral Analysis', 'Scalogram (time × freq)', 'Metrics Summary', ''],
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "heatmap"}],
               [{"type": "scatter"}, {"type": "bar"}, {"type": "domain"}]]
    )
    
    # Calculate gradient norms
    norms = [np.linalg.norm(g) for g in gradient_sequence]
    times = list(range(len(norms)))
    
    # 1. Gradient norms over time
    fig.add_trace(
        go.Scatter(x=times, y=norms, mode='lines+markers', name='Gradient Norms'),
        row=1, col=1
    )
    
    # 2. Trend analysis
    if 'trend_analysis' in result:
        trend_data = result['trend_analysis']
        if trend_data.get('has_significant_trend', False):
            # Add trend line
            slope = trend_data.get('trend_slope', 0)
            intercept = np.mean(norms)
            trend_line = [intercept + slope * t for t in times]
            fig.add_trace(
                go.Scatter(x=times, y=trend_line, mode='lines', name='Trend Line'),
                row=1, col=2
            )
    
    # 3. Spectral analysis (if available)
    if 'metrics' in result and 'spectral_entropy' in result['metrics']:
        # Create simple spectral visualization
        fft_data = np.abs(np.fft.fft(norms))[:len(norms)//2]
        freqs = np.fft.fftfreq(len(norms))[:len(norms)//2]
        fig.add_trace(
            go.Scatter(x=freqs, y=fft_data, mode='lines', name='Spectrum'),
            row=2, col=1
        )

    # 4. Scalogram (optional; STFT default)
    try:
        import numpy as _np
        from scipy import signal as _sig
        norms = _np.array(norms)
        f, t, Sxx = _sig.spectrogram(norms, nperseg=max(8, len(norms)//16))
        # Normalize for display
        S = _np.log10(Sxx + 1e-12)
        fig.add_trace(
            go.Heatmap(
                z=S, x=t, y=f, colorscale='Viridis',
                colorbar=dict(title='log10 power', thickness=14, x=1.03)
            ),
            row=1, col=3
        )
        # Optionally mark bands (use common marks)
        marks = [0.05, 0.1, 0.2]
        for m in marks:
            fig.add_hline(y=m, line_color='red', line_dash='dot', row=1, col=3)
    except Exception:
        pass
    
    # 5. Metrics summary
    if 'metrics' in result:
        metrics = result['metrics']
        metric_names = list(metrics.keys())[:5]  # Show first 5 metrics
        metric_values = [metrics[name] for name in metric_names]
        fig.add_trace(
            go.Bar(x=metric_names, y=metric_values, name='Metrics'),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Temporal Sequence Analysis",
        height=820,
        width=1200,
        showlegend=True,
        legend=dict(orientation='h', y=-0.14, x=0.5, xanchor='center'),
        margin=dict(l=60, r=140, t=70, b=120)
    )
    
    fig.write_html(output_file)

def create_progressive_obfuscation_plot(gradient_sequence, result, output_file):
    """Create progressive obfuscation visualization"""
    fig = make_subplots(
        rows=2, cols=3,
        column_widths=[0.42, 0.29, 0.29],
        subplot_titles=['Obfuscation Scores Over Time', 'Trend Analysis', 'Score Distribution', 'Scalogram (time × freq)', 'Detection Summary', ''],
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "heatmap"}],
               [{"type": "histogram"}, {"type": "indicator"}, {"type": "domain"}]]
    )
    
    # Get obfuscation scores
    if 'obfuscation_scores' in result:
        scores = result['obfuscation_scores']
        times = list(range(len(scores)))
        
        # 1. Obfuscation scores over time
        fig.add_trace(
            go.Scatter(x=times, y=scores, mode='lines+markers', name='Obfuscation Scores'),
            row=1, col=1
        )
        
        # 2. Trend line
        slope = result.get('slope', 0)
        intercept = np.mean(scores)
        trend_line = [intercept + slope * t for t in times]
        fig.add_trace(
            go.Scatter(x=times, y=trend_line, mode='lines', name='Trend Line'),
            row=1, col=2
        )
        
        # 3. Score distribution
        fig.add_trace(
            go.Histogram(x=scores, name='Score Distribution'),
            row=2, col=1
        )
        # Scalogram of scores (STFT)
        try:
            import numpy as _np
            from scipy import signal as _sig
            s = _np.array(scores)
            f, t, Sxx = _sig.spectrogram(s, nperseg=max(8, len(s)//16))
            Z = _np.log10(Sxx + 1e-12)
            fig.add_trace(
                go.Heatmap(
                    z=Z, x=t, y=f, colorscale='Viridis',
                    colorbar=dict(title='log10 power', thickness=14, x=1.03)
                ), row=1, col=3
            )
            for m in [0.05, 0.1, 0.2]:
                fig.add_hline(y=m, line_color='red', line_dash='dot', row=1, col=3)
        except Exception:
            pass
    
    # 4. Detection indicator
    confidence = result.get('confidence', 0.0)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Detection Confidence"},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 0.8], 'color': "yellow"},
                            {'range': [0.8, 1], 'color': "red"}]}
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Progressive Obfuscation Analysis",
        height=820,
        width=1200,
        showlegend=True,
        legend=dict(orientation='h', y=-0.14, x=0.5, xanchor='center'),
        margin=dict(l=60, r=140, t=70, b=120)
    )
    
    fig.write_html(output_file)

def create_trend_analysis_plot(gradient_sequence, result, output_file):
    """Create trend analysis visualization"""
    fig = make_subplots(
        rows=2, cols=3,
        column_widths=[0.42, 0.29, 0.29],
        subplot_titles=['Gradient Evolution', 'Trend Components', 'Autocorrelation', 'Scalogram (time × freq)', 'Spectral Features', ''],
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "heatmap"}],
               [{"type": "scatter"}, {"type": "bar"}, {"type": "domain"}]]
    )
    
    # Calculate gradient norms
    norms = [np.linalg.norm(g) for g in gradient_sequence]
    times = list(range(len(norms)))
    
    # 1. Gradient evolution
    fig.add_trace(
        go.Scatter(x=times, y=norms, mode='lines+markers', name='Gradient Norms'),
        row=1, col=1
    )
    
    # 2. Trend components
    if 'trend_analysis' in result:
        trend_data = result['trend_analysis']
        if trend_data.get('has_significant_trend', False):
            slope = trend_data.get('trend_slope', 0)
            intercept = np.mean(norms)
            trend_line = [intercept + slope * t for t in times]
            fig.add_trace(
                go.Scatter(x=times, y=trend_line, mode='lines', name='Linear Trend'),
                row=1, col=2
            )
    
    # 3. Autocorrelation (simplified)
    if len(norms) > 5:
        autocorr = np.correlate(norms, norms, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        lags = list(range(len(autocorr)))
        fig.add_trace(
            go.Scatter(x=lags, y=autocorr, mode='lines', name='Autocorrelation'),
            row=2, col=1
        )
    
    # 4. Scalogram of norms
    try:
        import numpy as _np
        from scipy import signal as _sig
        arr = _np.array(norms)
        f, t, Sxx = _sig.spectrogram(arr, nperseg=max(8, len(arr)//16))
        Z = _np.log10(Sxx + 1e-12)
        fig.add_trace(
            go.Heatmap(
                z=Z, x=t, y=f, colorscale='Viridis',
                colorbar=dict(title='log10 power', thickness=14, x=1.03)
            ), row=1, col=3
        )
        for m in [0.05, 0.1, 0.2]:
            fig.add_hline(y=m, line_color='red', line_dash='dot', row=1, col=3)
    except Exception:
        pass

    # 5. Spectral features
    if 'metrics' in result:
        metrics = result['metrics']
        spectral_metrics = {k: v for k, v in metrics.items() if 'spectral' in k.lower()}
        if spectral_metrics:
            metric_names = list(spectral_metrics.keys())
            metric_values = list(spectral_metrics.values())
            fig.add_trace(
                go.Bar(x=metric_names, y=metric_values, name='Spectral Metrics'),
                row=2, col=2
            )
    
    fig.update_layout(
        title="Temporal Trend Analysis",
        height=820,
        width=1200,
        showlegend=True,
        legend=dict(orientation='h', y=-0.14, x=0.5, xanchor='center'),
        margin=dict(l=60, r=140, t=70, b=120)
    )
    
    fig.write_html(output_file) 