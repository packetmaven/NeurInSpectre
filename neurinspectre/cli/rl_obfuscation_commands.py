"""
NeurInSpectre CLI: RL Obfuscation Detection Commands
Integrates critical_rl_obfuscation.py functionality into the CLI system
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

def _brand_title(title: str) -> str:
    """Ensure every visualization title starts with 'NeurInSpectre' (no leading emoji)."""
    t = str(title or "").strip()
    if not t:
        return "NeurInSpectre"
    if t.startswith("NeurInSpectre"):
        return t
    return f"NeurInSpectre — {t}"

# Support running this module as a script from a source checkout.
# When imported as part of the package/CLI, avoid sys.path side effects.
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    # Prefer relative import for package execution.
    from ..security.critical_rl_obfuscation import CriticalRLObfuscationDetector
    USING_REAL_DETECTOR = True
    _RL_IMPORT_ERROR = None
except ImportError as e:
    # No synthetic/demo fallback: fail fast at runtime if detector isn't available.
    USING_REAL_DETECTOR = False
    _RL_IMPORT_ERROR = str(e)
    CriticalRLObfuscationDetector = None  # type: ignore[assignment]

def add_rl_obfuscation_parser(subparsers):
    """Add RL obfuscation detection command to CLI"""
    parser = subparsers.add_parser(
        'rl-obfuscation',
        aliases=['critical-rl-obfuscation', 'critical_rl_obfuscation'],
        help='🚨 Critical RL-obfuscation detection and analysis'
    )
    
    # Subcommands for RL obfuscation detection
    rl_subparsers = parser.add_subparsers(dest='rl_command', help='RL obfuscation detection commands')
    
    # Single gradient analysis
    analyze_parser = rl_subparsers.add_parser(
        'analyze',
        help='Analyze single gradient for RL obfuscation patterns'
    )
    analyze_parser.add_argument('--input-file', required=True, help='Input gradient file (.npy)')
    analyze_parser.add_argument('--sensitivity', choices=['critical', 'high', 'medium'], 
                               default='high', help='Detection sensitivity level (default: high)')
    analyze_parser.add_argument('--output-report', help='Output JSON report file')
    analyze_parser.add_argument('--output-plot', help='Output visualization file')
    analyze_parser.add_argument('--metadata', help='Additional metadata JSON file')
    analyze_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Batch analysis
    batch_parser = rl_subparsers.add_parser(
        'batch',
        help='Batch analysis of multiple gradients'
    )
    batch_parser.add_argument('--input-dir', required=True, help='Directory containing gradient files')
    batch_parser.add_argument('--pattern', default='*.npy', help='File pattern to match (default: *.npy)')
    batch_parser.add_argument('--sensitivity', choices=['critical', 'high', 'medium'], 
                             default='high', help='Detection sensitivity level (default: high)')
    batch_parser.add_argument('--output-report', help='Output batch analysis report')
    batch_parser.add_argument('--output-summary', help='Output summary visualization')
    batch_parser.add_argument('--threshold', type=float, default=0.6, help='Threat threshold (default: 0.6)')
    batch_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Real-time monitoring
    monitor_parser = rl_subparsers.add_parser(
        'monitor',
        help='Real-time RL obfuscation monitoring'
    )
    monitor_parser.add_argument('--input-stream', required=True, help='Input stream file or directory')
    monitor_parser.add_argument('--sensitivity', choices=['critical', 'high', 'medium'], 
                               default='critical', help='Detection sensitivity level (default: critical)')
    monitor_parser.add_argument('--alert-threshold', type=float, default=0.8, help='Alert threshold (default: 0.8)')
    monitor_parser.add_argument('--interval', type=float, default=1.0, help='Polling interval in seconds (default: 1.0)')
    monitor_parser.add_argument('--max-iterations', type=int, default=0, help='Max polling iterations (0 = run until Ctrl-C) (default: 0)')
    monitor_parser.add_argument('--pattern', default='*.npy', help='When input-stream is a directory: glob pattern (default: *.npy)')
    monitor_parser.add_argument('--output-log', help='Output monitoring log file')
    monitor_parser.add_argument('--dashboard', action='store_true', help='Launch monitoring dashboard')
    monitor_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Threat intelligence
    intel_parser = rl_subparsers.add_parser(
        'intel',
        help='Generate threat intelligence report'
    )
    intel_parser.add_argument('--input-data', required=True, help='Input analysis data or directory')
    intel_parser.add_argument('--output-intel', help='Output intelligence report')
    intel_parser.add_argument('--format', choices=['json', 'html', 'pdf'], 
                             default='json', help='Output format (default: json)')
    intel_parser.add_argument('--include-recommendations', action='store_true', 
                             help='Include actionable recommendations')
    intel_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Signature analysis
    signature_parser = rl_subparsers.add_parser(
        'signature',
        help='Analyze RL training signatures and patterns'
    )
    signature_parser.add_argument('--input-file', required=True, help='Input gradient file (.npy)')
    signature_parser.add_argument('--analysis-type', choices=['policy', 'value', 'actor-critic', 'q-learning', 'all'], 
                                 default='all', help='Type of signature analysis (default: all)')
    signature_parser.add_argument('--output-report', help='Output signature analysis report')
    signature_parser.add_argument('--output-plot', help='Output signature visualization')
    signature_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    parser.set_defaults(func=handle_rl_obfuscation)

def handle_rl_obfuscation(args):
    """Handle RL obfuscation detection commands"""
    if args.rl_command == 'analyze':
        return handle_single_analysis(args)
    elif args.rl_command == 'batch':
        return handle_batch_analysis(args)
    elif args.rl_command == 'monitor':
        return handle_real_time_monitoring(args)
    elif args.rl_command == 'intel':
        return handle_threat_intelligence(args)
    elif args.rl_command == 'signature':
        return handle_signature_analysis(args)
    else:
        print("❌ No RL obfuscation command specified. Use --help for options.")
        return 1

def handle_single_analysis(args):
    """Handle single gradient analysis"""
    if CriticalRLObfuscationDetector is None:
        print("❌ CriticalRLObfuscationDetector unavailable (import failed).")
        print(f"   Error: {_RL_IMPORT_ERROR}")
        return 1
    if args.verbose:
        print("🚨 Starting critical RL obfuscation analysis...")
        print(f"📁 Input file: {args.input_file}")
        print(f"⚡ Sensitivity: {args.sensitivity}")
    
    # Load gradient data
    try:
        gradient_data = np.load(args.input_file)
        # Sanitize NaN/Inf and enforce 1D flatten for robust plotting
        import numpy as _np
        gradient_data = _np.nan_to_num(gradient_data, nan=0.0, posinf=0.0, neginf=0.0)
        if args.verbose:
            print(f"✅ Loaded gradient data: shape {gradient_data.shape}")
    except Exception as e:
        print(f"❌ Failed to load gradient file: {e}")
        return 1
    
    # Load metadata if provided
    metadata = None
    if args.metadata:
        try:
            with open(args.metadata, 'r') as f:
                metadata = json.load(f)
            if args.verbose:
                print(f"✅ Loaded metadata: {len(metadata)} entries")
        except Exception as e:
            print(f"⚠️  Warning: Failed to load metadata: {e}")
    
    # Initialize detector
    detector = CriticalRLObfuscationDetector(sensitivity_level=args.sensitivity)
    
    # Analyze gradient
    if args.verbose:
        print("🔍 Analyzing gradient for RL obfuscation patterns...")
    
    result = detector.detect_rl_obfuscation(gradient_data, metadata)
    
    # Print results
    print_analysis_results(result, args.verbose)
    
    # Save report (ALWAYS save) - FIX: Ensure directory exists
    from pathlib import Path
    
    output_report = args.output_report or '_cli_runs/rl_analysis.json'
    Path(output_report).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_report, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Report saved to: {output_report}")
    except Exception as e:
        print(f"⚠️  Failed to save report: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate visualization
    if args.output_plot:
        try:
            create_single_analysis_plot(gradient_data, result, args.output_plot)
        except Exception as e:
            if args.verbose:
                print(f"⚠️ Plotly export failed, using fallback: {e}")
        if args.verbose:
            from pathlib import Path as _P
            _out = _P(args.output_plot)
            alt = _out.with_suffix('.html')
            if _out.exists():
                print(f"📊 Visualization saved to: {args.output_plot}")
            elif alt.exists():
                print(f"📊 Visualization saved to: {alt}")
            else:
                print("⚠️ Visualization export attempted, but no file found; proceeding to ensure PNG fallback.")
    # Create PNG visualization (ALWAYS create) - FIX: Ensure directory exists
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    output_plot = args.output_plot or '_cli_runs/rl_analysis.png'
    Path(output_plot).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        comp = result.get('component_scores', {})
        if comp:
            keys = list(comp.keys())
            vals = [comp[k] for k in keys]
            
            plt.figure(figsize=(10, 6))
            plt.bar(keys, vals, color='#E74C3C')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Score (0-1)')
            plt.title(_brand_title(
                f"RL Obfuscation Detection - {result.get('threat_classification', 'UNKNOWN')} "
                f"(Threat: {result.get('overall_threat_level', 0):.2f})"
            ))
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(output_plot, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Visualization saved to: {output_plot}")
            
            # Verify file was created
            if Path(output_plot).exists():
                print(f"✅ Confirmed: {output_plot} exists ({Path(output_plot).stat().st_size} bytes)")
            else:
                print("❌ ERROR: File not created despite no exception")
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    return 0

def handle_batch_analysis(args):
    """Handle batch analysis of multiple gradients"""
    if CriticalRLObfuscationDetector is None:
        print("❌ CriticalRLObfuscationDetector unavailable (import failed).")
        print(f"   Error: {_RL_IMPORT_ERROR}")
        return 1
    if args.verbose:
        print("🚨 Starting batch RL obfuscation analysis...")
        print(f"📁 Input directory: {args.input_dir}")
        print(f"🔍 Pattern: {args.pattern}")
        print(f"⚡ Sensitivity: {args.sensitivity}")
        print(f"🎯 Threshold: {args.threshold}")
    
    # Load gradient files
    gradient_files = load_gradient_files(args.input_dir, args.pattern, args.verbose)
    
    if not gradient_files:
        print("❌ No gradient files found")
        return 1
    
    # Initialize detector
    detector = CriticalRLObfuscationDetector(sensitivity_level=args.sensitivity)
    
    # Analyze each gradient
    batch_results = []
    threat_count = 0
    
    print(f"\n🔍 Analyzing {len(gradient_files)} gradient files...")
    print("=" * 60)
    
    for i, file_path in enumerate(gradient_files):
        if args.verbose:
            print(f"📁 Processing: {os.path.basename(file_path)} ({i+1}/{len(gradient_files)})")
        
        try:
            gradient_data = np.load(file_path)
            import numpy as _np
            gradient_data = _np.nan_to_num(gradient_data, nan=0.0, posinf=0.0, neginf=0.0)
            result = detector.detect_rl_obfuscation(gradient_data)
            
            # Add file info to result
            result['file_path'] = file_path
            result['file_name'] = os.path.basename(file_path)
            
            batch_results.append(result)
            
            # Check threat level
            if result['overall_threat_level'] > args.threshold:
                threat_count += 1
                threat_status = "🚨 THREAT"
            else:
                threat_status = "✅ CLEAN"
            
            print(f"   {threat_status} | Threat: {result['overall_threat_level']:.3f} | Class: {result['threat_classification']}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    # Print batch summary
    print("\n📊 Batch Analysis Summary")
    print("=" * 50)
    print(f"📁 Files analyzed: {len(batch_results)}")
    print(f"🚨 Threats detected: {threat_count}")
    print(f"📊 Threat rate: {threat_count/len(batch_results)*100:.1f}%")
    
    # Calculate statistics
    threat_levels = [r['overall_threat_level'] for r in batch_results]
    confidences = [r['detection_confidence'] for r in batch_results]
    
    print(f"📊 Mean threat level: {np.mean(threat_levels):.3f}")
    print(f"📊 Max threat level: {np.max(threat_levels):.3f}")
    print(f"📊 Mean confidence: {np.mean(confidences):.3f}")
    
    # Save batch report
    if args.output_report:
        save_batch_report(batch_results, args.output_report, args)
        if args.verbose:
            print(f"💾 Batch report saved to: {args.output_report}")
    
    # Generate summary visualization
    if args.output_summary:
        create_batch_summary_plot(batch_results, args.output_summary)
        if args.verbose:
            print(f"📊 Summary visualization saved to: {args.output_summary}")
    
    return 0

def handle_real_time_monitoring(args):
    """Handle real-time RL obfuscation monitoring"""
    if CriticalRLObfuscationDetector is None:
        print("❌ CriticalRLObfuscationDetector unavailable (import failed).")
        print(f"   Error: {_RL_IMPORT_ERROR}")
        return 1
    if args.verbose:
        print("🚨 Starting real-time RL obfuscation monitoring...")
        print(f"📡 Input stream: {args.input_stream}")
        print(f"⚡ Sensitivity: {args.sensitivity}")
        print(f"🚨 Alert threshold: {args.alert_threshold}")
    
    # Initialize detector
    detector = CriticalRLObfuscationDetector(sensitivity_level=args.sensitivity)
    
    # Monitoring loop (NO simulation / NO synthetic data)
    print("\n🔄 Monitoring for RL obfuscation patterns (real data only)...")
    print("=" * 60)

    import time
    import glob

    stream = str(args.input_stream)
    interval = float(getattr(args, "interval", 1.0))
    max_iter = int(getattr(args, "max_iterations", 0) or 0)
    pattern = str(getattr(args, "pattern", "*.npy"))

    last_seen = None  # (path, mtime)
    alert_count = 0
    sample_count = 0
    it = 0

    def _load_gradient(p: str) -> np.ndarray:
        arr = np.load(p, allow_pickle=True)
        if getattr(arr, "dtype", None) is object and getattr(arr, "shape", ()) == ():
            arr = arr.item()
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.number):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.reshape(-1)

    try:
        while True:
            it += 1

            candidate = None
            cand_mtime = None

            if os.path.isfile(stream):
                candidate = stream
                cand_mtime = os.path.getmtime(stream)
            elif os.path.isdir(stream):
                files = sorted(glob.glob(os.path.join(stream, pattern)), key=lambda p: os.path.getmtime(p))
                if files:
                    candidate = files[-1]
                    cand_mtime = os.path.getmtime(candidate)
            else:
                print(f"❌ Input stream path not found: {os.path.abspath(stream)}")
                return 1

            if candidate is not None and cand_mtime is not None and last_seen != (candidate, cand_mtime):
                last_seen = (candidate, cand_mtime)
                grad = _load_gradient(candidate)

                result = detector.detect_rl_obfuscation(grad)
                sample_count += 1

                timestamp = datetime.now().strftime("%H:%M:%S")
                threat_level = float(result.get("overall_threat_level", 0.0) or 0.0)
                classification = str(result.get("threat_classification", "UNKNOWN"))
                src = f" [{os.path.basename(candidate)}]" if candidate else ""

                if threat_level > float(args.alert_threshold):
                    alert_count += 1
                    status = "🚨 ALERT"
                    print(f"[{timestamp}]{src} {status} | Threat: {threat_level:.3f} | Class: {classification}")
                    intel = result.get("actionable_intelligence") or {}
                    primary = intel.get("primary_threats") or []
                    if primary:
                        try:
                            print(f"   🎯 Primary threats: {', '.join([str(x) for x in primary])}")
                        except Exception:
                            pass
                else:
                    status = "✅ CLEAN"
                    if args.verbose:
                        print(f"[{timestamp}]{src} {status} | Threat: {threat_level:.3f} | Class: {classification}")

                if args.output_log:
                    log_entry = {
                        "timestamp": timestamp,
                        "source": candidate,
                        "threat_level": threat_level,
                        "classification": classification,
                        "alert": bool(threat_level > float(args.alert_threshold)),
                        "component_scores": result.get("component_scores"),
                    }
                    append_to_log(log_entry, args.output_log)
            else:
                if args.verbose:
                    print("⚠️  No new gradient sample detected yet.")

            if max_iter > 0 and it >= max_iter:
                break
            time.sleep(max(0.0, interval))
    except KeyboardInterrupt:
        pass

    print("\n📊 Monitoring Summary")
    print("=" * 50)
    print(f"🔄 Samples analyzed: {sample_count}")
    print(f"🚨 Alerts triggered: {alert_count}")
    if sample_count > 0:
        print(f"📊 Alert rate: {alert_count / sample_count * 100:.1f}%")

    return 0

def handle_threat_intelligence(args):
    """Handle threat intelligence generation"""
    if args.verbose:
        print("🚨 Generating threat intelligence report...")
        print(f"📁 Input data: {args.input_data}")
        print(f"📄 Format: {args.format}")
    
    # Load analysis data
    if os.path.isfile(args.input_data):
        # Single file
        try:
            with open(args.input_data, 'r') as f:
                analysis_data = json.load(f)
            if args.verbose:
                print("✅ Loaded analysis data from file")
        except Exception as e:
            print(f"❌ Failed to load analysis data: {e}")
            return 1
    else:
        # Directory with multiple analyses
        analysis_data = load_analysis_directory(args.input_data, args.verbose)
        if not analysis_data:
            print("❌ No analysis data found")
            return 1
    
    # Generate intelligence report
    intel_report = generate_intelligence_report(analysis_data, args.include_recommendations)
    
    # Print summary
    print("\n📊 Threat Intelligence Summary")
    print("=" * 50)
    print(f"🎯 Threat level: {intel_report['overall_assessment']['threat_level']}")
    print(f"📊 Confidence: {intel_report['overall_assessment']['confidence']:.3f}")
    print(f"🚨 Critical findings: {len(intel_report['critical_findings'])}")
    
    # Print key findings
    for finding in intel_report['critical_findings'][:3]:
        print(f"   • {finding}")
    
    # Save intelligence report
    if args.output_intel:
        save_intelligence_report(intel_report, args.output_intel, args.format)
        if args.verbose:
            print(f"💾 Intelligence report saved to: {args.output_intel}")
    # Lightweight summary HTML with radar-like bar and badges
    try:
        if args.format == 'html' and args.output_intel:
            import plotly.graph_objects as _go
            from plotly.subplots import make_subplots as _mk
            fig = _mk(rows=1, cols=2, column_widths=[0.6,0.4], subplot_titles=('Component Fingerprint','Overall'))
            comps = ['policy_fingerprint','semantic_consistency','conditional_triggers','periodic_patterns','evasion_signatures','reward_optimization','training_artifacts','adversarial_patterns']
            vals = [0.0]*len(comps)
            # try to backfill from a typical component set if present in analysis_data list
            try:
                # if list provided earlier, take first v if exists
                pass
            except Exception:
                pass
            fig.add_trace(_go.Bar(x=[c.replace('_',' ').title() for c in comps], y=vals), row=1, col=1)
            fig.add_trace(_go.Indicator(mode='gauge+number', value=float(intel_report['overall_assessment']['confidence']), title={'text':'Confidence'}), row=1, col=2)
            fig.update_layout(height=520, width=980, title_text=_brand_title('RL Obfuscation Intelligence Summary'), showlegend=False)
            html = args.output_intel  # overwrite with richer plot inline
            fig.write_html(html)
    except Exception:
        pass
    
    return 0

def handle_signature_analysis(args):
    """Handle RL signature analysis"""
    if CriticalRLObfuscationDetector is None:
        print("❌ CriticalRLObfuscationDetector unavailable (import failed).")
        print(f"   Error: {_RL_IMPORT_ERROR}")
        return 1
    if args.verbose:
        print("🚨 Starting RL signature analysis...")
        print(f"📁 Input file: {args.input_file}")
        print(f"📊 Analysis type: {args.analysis_type}")
    
    # Load gradient data
    try:
        gradient_data = np.load(args.input_file)
        if args.verbose:
            print(f"✅ Loaded gradient data: shape {gradient_data.shape}")
    except Exception as e:
        print(f"❌ Failed to load gradient file: {e}")
        return 1
    
    # Initialize detector
    detector = CriticalRLObfuscationDetector(sensitivity_level='high')
    
    # Perform signature analysis
    if args.verbose:
        print("🔍 Analyzing RL training signatures...")
    
    result = detector.detect_rl_obfuscation(gradient_data)
    
    # Print signature analysis results
    print("\n📊 RL Signature Analysis Results")
    print("=" * 50)
    
    component_scores = result['component_scores']
    
    if args.analysis_type in ['policy', 'all']:
        print(f"🎯 Policy Fingerprint: {component_scores.get('policy_fingerprint', 0.0):.3f}")
    
    if args.analysis_type in ['value', 'all']:
        print(f"💰 Value Function Patterns: {component_scores.get('reward_optimization', 0.0):.3f}")
    
    if args.analysis_type in ['actor-critic', 'all']:
        print(f"🎭 Actor-Critic Signatures: {component_scores.get('adversarial_patterns', 0.0):.3f}")
    
    if args.analysis_type in ['q-learning', 'all']:
        print(f"🧠 Q-Learning Patterns: {component_scores.get('training_artifacts', 0.0):.3f}")
    
    print(f"🔄 Periodic Patterns: {component_scores.get('periodic_patterns', 0.0):.3f}")
    print(f"🚨 Evasion Signatures: {component_scores.get('evasion_signatures', 0.0):.3f}")
    
    # Save signature report
    if args.output_report:
        save_signature_report(result, args.output_report, args.analysis_type)
        if args.verbose:
            print(f"💾 Signature report saved to: {args.output_report}")
    
    # Generate signature visualization
    if args.output_plot:
        create_signature_plot(gradient_data, result, args.output_plot)
        if args.verbose:
            print(f"📊 Signature visualization saved to: {args.output_plot}")
    
    return 0

def print_analysis_results(result, verbose=False):
    """Print analysis results in formatted way"""
    print("\n🚨 CRITICAL RL OBFUSCATION ANALYSIS RESULTS")
    print("=" * 60)
    print(f"🎯 Overall Threat Level: {result['overall_threat_level']:.3f}")
    print(f"⚠️  Threat Classification: {result['threat_classification']}")
    print(f"🎯 Detection Confidence: {result['detection_confidence']:.3f}")
    
    print("\n📊 Component Scores:")
    for component, score in result['component_scores'].items():
        print(f"   {component.replace('_', ' ').title()}: {score:.3f}")
    
    if verbose and 'actionable_intelligence' in result:
        intel = result['actionable_intelligence']
        
        if 'primary_threats' in intel:
            print("\n🎯 Primary Threats:")
            for threat in intel['primary_threats']:
                print(f"   • {threat}")
        
        if 'attack_vectors' in intel:
            print("\n🚨 Attack Vectors:")
            for vector in intel['attack_vectors']:
                print(f"   • {vector}")
    
    if 'recommended_actions' in result:
        print("\n🛡️  Recommended Actions:")
        for action in result['recommended_actions']:
            print(f"   • {action}")

def load_gradient_files(input_dir, pattern, verbose=False):
    """Load gradient files from directory"""
    import glob
    
    if verbose:
        print(f"🔍 Loading gradient files from {input_dir}...")
    
    # Find matching files
    search_pattern = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"❌ No files found matching pattern: {search_pattern}")
        return None
    
    if verbose:
        print(f"📁 Found {len(files)} gradient files")
    
    return files

def save_analysis_report(result, output_file, analysis_type, input_file=None):
    """Save analysis report to JSON file"""
    report = {
        'analysis_type': analysis_type,
        'timestamp': datetime.now().isoformat(),
        'input_file': input_file,
        'results': result
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

def save_batch_report(batch_results, output_file, args):
    """Save batch analysis report"""
    report = {
        'analysis_type': 'batch_analysis',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'input_dir': args.input_dir,
            'pattern': args.pattern,
            'sensitivity': args.sensitivity,
            'threshold': args.threshold
        },
        'summary': {
            'files_analyzed': len(batch_results),
            'threats_detected': sum(1 for r in batch_results if r['overall_threat_level'] > args.threshold),
            'mean_threat_level': np.mean([r['overall_threat_level'] for r in batch_results]),
            'max_threat_level': np.max([r['overall_threat_level'] for r in batch_results])
        },
        'results': batch_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

def save_signature_report(result, output_file, analysis_type):
    """Save signature analysis report"""
    report = {
        'analysis_type': 'signature_analysis',
        'signature_type': analysis_type,
        'timestamp': datetime.now().isoformat(),
        'results': result
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

def append_to_log(log_entry, log_file):
    """Append log entry to file"""
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry, default=str) + '\n')

def load_analysis_directory(input_dir, verbose=False):
    """Load analysis data from directory"""
    import glob
    
    analysis_files = glob.glob(os.path.join(input_dir, '*.json'))
    
    if not analysis_files:
        return None
    
    analysis_data = []
    for file_path in analysis_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            analysis_data.append(data)
        except Exception as e:
            if verbose:
                print(f"⚠️  Warning: Failed to load {file_path}: {e}")
    
    return analysis_data

def generate_intelligence_report(analysis_data, include_recommendations=False):
    """Generate threat intelligence report"""
    if isinstance(analysis_data, list):
        # Multiple analyses
        threat_levels = []
        confidences = []
        classifications = []
        
        for data in analysis_data:
            if 'results' in data:
                result = data['results']
                threat_levels.append(result.get('overall_threat_level', 0.0))
                confidences.append(result.get('detection_confidence', 0.0))
                classifications.append(result.get('threat_classification', 'LOW'))
        
        overall_threat = np.mean(threat_levels) if threat_levels else 0.0
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
    else:
        # Single analysis
        result = analysis_data.get('results', {})
        overall_threat = result.get('overall_threat_level', 0.0)
        overall_confidence = result.get('detection_confidence', 0.0)
        classifications = [result.get('threat_classification', 'LOW')]
    
    # Generate findings
    critical_findings = []
    
    if overall_threat > 0.8:
        critical_findings.append("High-confidence RL obfuscation patterns detected")
    
    if overall_threat > 0.6:
        critical_findings.append("Significant policy gradient signatures identified")
    
    if 'CRITICAL' in classifications:
        critical_findings.append("Critical threat classification assigned")
    
    # Generate recommendations
    recommendations = []
    if include_recommendations:
        if overall_threat > 0.7:
            recommendations.append("Implement immediate containment measures")
            recommendations.append("Increase monitoring frequency")
        
        if overall_threat > 0.5:
            recommendations.append("Review security policies")
            recommendations.append("Consider additional countermeasures")
    
    intel_report = {
        'overall_assessment': {
            'threat_level': 'CRITICAL' if overall_threat > 0.8 else 'HIGH' if overall_threat > 0.6 else 'MEDIUM' if overall_threat > 0.4 else 'LOW',
            'confidence': overall_confidence,
            'numeric_threat_level': overall_threat
        },
        'critical_findings': critical_findings,
        'recommendations': recommendations,
        'timestamp': datetime.now().isoformat()
    }
    
    return intel_report

def save_intelligence_report(intel_report, output_file, format_type):
    """Save intelligence report in specified format"""
    if format_type == 'json':
        with open(output_file, 'w') as f:
            json.dump(intel_report, f, indent=2, default=str)
    elif format_type == 'html':
        # Generate HTML report
        html_content = generate_html_report(intel_report)
        with open(output_file, 'w') as f:
            f.write(html_content)
    elif format_type == 'pdf':
        # For PDF, save as HTML first (PDF generation would require additional libraries)
        html_content = generate_html_report(intel_report)
        html_file = output_file.replace('.pdf', '.html')
        with open(html_file, 'w') as f:
            f.write(html_content)
        print(f"📄 HTML report saved to: {html_file} (PDF generation requires additional libraries)")

def generate_html_report(intel_report):
    """Generate HTML intelligence report"""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NeurInSpectre Threat Intelligence Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #d62728; color: white; padding: 20px; }}
            .section {{ margin: 20px 0; }}
            .finding {{ background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-left: 4px solid #d62728; }}
            .recommendation {{ background-color: #e8f5e8; padding: 10px; margin: 5px 0; border-left: 4px solid #28a745; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚨 NeurInSpectre Threat Intelligence Report</h1>
            <p>Generated: {intel_report['timestamp']}</p>
        </div>
        
        <div class="section">
            <h2>Overall Assessment</h2>
            <p><strong>Threat Level:</strong> {intel_report['overall_assessment']['threat_level']}</p>
            <p><strong>Confidence:</strong> {intel_report['overall_assessment']['confidence']:.3f}</p>
        </div>
        
        <div class="section">
            <h2>Critical Findings</h2>
            {''.join(f'<div class="finding">{finding}</div>' for finding in intel_report['critical_findings'])}
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            {''.join(f'<div class="recommendation">{rec}</div>' for rec in intel_report['recommendations'])}
        </div>
    </body>
    </html>
    """
    
    return html_template

def create_single_analysis_plot(gradient_data, result, output_file):
    """Create single analysis visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Gradient Distribution', 'Component Scores', 'Threat Assessment', 'Signature Analysis'],
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "indicator"}, {"type": "scatter"}]]
    )
    
    # 1. Gradient distribution
    flat_vals = gradient_data.flatten()
    fig.add_trace(
        go.Histogram(x=flat_vals, nbinsx=50, name='Gradient Distribution'),
        row=1, col=1
    )
    # Percentile bands (P90/P95/P99) to aid threshold selection
    try:
        import numpy as _np
        p90, p95, p99 = _np.percentile(flat_vals, [90, 95, 99])
        for p, c in [(p90, 'orange'), (p95, 'red'), (p99, 'purple')]:
            fig.add_vline(x=float(p), line_width=1.5, line_dash='dot', line_color=c, row=1, col=1)
        fig.add_annotation(xref='x1', yref='paper', x=float(p95), y=0.98,
                           text=f"p90={p90:.3g} | p95={p95:.3g} | p99={p99:.3g}",
                           showarrow=False, font=dict(size=10),
                           bgcolor='rgba(255,230,230,0.85)', bordercolor='rgba(204,0,0,0.9)')
    except Exception:
        pass
    
    # 2. Component scores
    components = list(result['component_scores'].keys())
    scores = list(result['component_scores'].values())
    
    fig.add_trace(
        go.Bar(x=components, y=scores, name='Component Scores'),
        row=1, col=2
    )
    
    # 3. Threat assessment gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=result['overall_threat_level'],
            title={'text': "Threat Level"},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "red"},
                   'steps': [{'range': [0, 0.3], 'color': "lightgray"},
                            {'range': [0.3, 0.6], 'color': "yellow"},
                            {'range': [0.6, 1], 'color': "red"}]}
        ),
        row=2, col=1
    )
    
    # 4. Signature analysis (spectral view with shaded bands and primary threats callout)
    flat = flat_vals
    fft_data = np.abs(np.fft.fft(flat))[:len(flat)//2]
    freqs = np.fft.fftfreq(len(flat))[:len(flat)//2]
    fig.add_trace(go.Scatter(x=freqs, y=fft_data, mode='lines', name='Spectral Signature'), row=2, col=2)
    # Shade low/high frequency bands for interpretability
    try:
        band_shapes = [
            dict(type='rect', xref='x4', yref='y4', x0=0.0, x1=0.05, y0=0, y1=float(max(fft_data)*1.05), fillcolor='rgba(0,128,0,0.08)', line=dict(width=0)),
            dict(type='rect', xref='x4', yref='y4', x0=0.3, x1=float(max(freqs)), y0=0, y1=float(max(fft_data)*1.05), fillcolor='rgba(255,0,0,0.08)', line=dict(width=0)),
        ]
        existing = list(fig.layout.shapes) if fig.layout.shapes else []
        fig.update_layout(shapes=existing + band_shapes)
    except Exception:
        pass
    # Primary threats callout
    prim = None
    if 'actionable_intelligence' in result and 'primary_threats' in result['actionable_intelligence']:
        # Handle both string and structured (dict) threat entries robustly.
        def _threat_to_str(x):
            if isinstance(x, str):
                return x
            if isinstance(x, dict):
                if x.get('description') is not None:
                    return str(x.get('description'))
                if x.get('component') is not None:
                    return str(x.get('component')).replace('_', ' ')
                return json.dumps(x, default=str)
            return str(x)

        prim_items = list(result['actionable_intelligence'].get('primary_threats', []))[:2]
        prim = ", ".join(_threat_to_str(x) for x in prim_items if x is not None)
    elif 'component_scores' in result:
        comp = max(result['component_scores'], key=lambda k: result['component_scores'][k])
        prim = comp.replace('_', ' ')
    if prim:
        fig.add_annotation(x=0.02, y=0.95, xref='paper', yref='paper', showarrow=False,
                           text=f"Primary threats: {prim}<br>Recommend: guardrails on top components",
                           bgcolor='rgba(255,230,230,0.9)', bordercolor='rgba(204,0,0,0.9)',
                           font=dict(size=10))
    
    fig.update_layout(
        title=_brand_title(f"RL Obfuscation Analysis - {result['threat_classification']} Threat"),
        height=800,
        showlegend=True
    )
    # Save as HTML (always). We intentionally avoid Plotly static image export here because
    # `plotly`/`kaleido` version mismatches are common and emit noisy warnings; we generate
    # a deterministic static PNG via Matplotlib below instead.
    try:
        from pathlib import Path as _P
        outp = _P(output_file)
        if outp.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            html_path = outp.with_suffix('.html')
            fig.write_html(str(html_path))
        else:
            fig.write_html(str(outp))
    except Exception:
        # Last-resort fallback
        fig.write_html(output_file)
    # Always write an HTML copy for maximum compatibility
    try:
        from pathlib import Path as _P
        html_fallback = _P(output_file).with_suffix('.html')
        fig.write_html(str(html_fallback))
    except Exception:
        pass
    # Additionally, generate a static PNG using matplotlib as a compatibility fallback
    try:
        from pathlib import Path as _P
        import matplotlib.pyplot as _plt
        outp = _P(output_file)
        png_path = outp if outp.suffix.lower() == '.png' else outp.with_suffix('.png')
        comp = result.get('component_scores', {})
        keys = list(comp.keys())
        vals = [comp[k] for k in keys]
        _plt.figure(figsize=(8,4))
        _plt.bar(range(len(vals)), vals, color='#9B59B6')
        _plt.xticks(range(len(vals)), [k.replace('_','\n') for k in keys], fontsize=8)
        _plt.ylim(0,1)
        _plt.title(_brand_title(
            f"RL Obfuscation Components — {result.get('threat_classification','')} "
            f"(threat {result.get('overall_threat_level',0):.2f})"
        ))
        _plt.ylabel('Score')
        _plt.tight_layout()
        _plt.savefig(str(png_path), dpi=160)
        _plt.close()
    except Exception:
        pass

def create_batch_summary_plot(batch_results, output_file):
    """Create batch analysis summary visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Threat Level Distribution', 'Classification Summary', 'Component Score Heatmap', 'Detection Timeline'],
        specs=[[{"type": "histogram"}, {"type": "pie"}],
               [{"type": "heatmap"}, {"type": "scatter"}]]
    )
    
    # 1. Threat level distribution
    threat_levels = [r['overall_threat_level'] for r in batch_results]
    fig.add_trace(
        go.Histogram(x=threat_levels, nbinsx=20, name='Threat Levels'),
        row=1, col=1
    )
    
    # 2. Classification summary
    classifications = [r['threat_classification'] for r in batch_results]
    class_counts = {}
    for cls in classifications:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    fig.add_trace(
        go.Pie(labels=list(class_counts.keys()), values=list(class_counts.values()), name='Classifications'),
        row=1, col=2
    )
    
    # 3. Component score heatmap
    components = list(batch_results[0]['component_scores'].keys())
    score_matrix = []
    
    for component in components:
        component_scores = [r['component_scores'][component] for r in batch_results]
        score_matrix.append(component_scores)
    # Explicitly position and title the heatmap colorbar so it doesn't clash with legends
    heatmap_trace = go.Heatmap(
        z=score_matrix,
        y=components,
        colorscale='Reds',
        name='Component Scores',
        colorbar=dict(
            title=dict(text='Component Score', side='right'),
            thickness=14,
            x=1.02  # place just outside subplot area
        )
    )
    fig.add_trace(heatmap_trace, row=2, col=1)
    
    # 4. Detection timeline with confidence ribbon (synthetic if not provided)
    indices = list(range(len(batch_results)))
    # Build a simple ±5% ribbon around levels to guide reading as uncertainty
    lower = [max(0.0, t - 0.02) for t in threat_levels]
    upper = [min(1.0, t + 0.02) for t in threat_levels]
    fig.add_trace(
        go.Scatter(
            x=indices, y=upper, mode='lines', line=dict(width=0), showlegend=False,
            hoverinfo='skip', name='upper'
        ), row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=indices, y=lower, mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor='rgba(155, 89, 182, 0.15)', showlegend=False, hoverinfo='skip', name='lower'
        ), row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=indices, y=threat_levels, mode='lines+markers', name='Threat Timeline',
                   line=dict(color='#9B59B6')), row=2, col=2
    )
    
    # Improve readability: move legend to bottom, increase margins to avoid clipping colorbar text
    fig.update_layout(
        title=_brand_title("Batch RL Obfuscation Analysis Summary"),
        height=820,
        width=1100,
        showlegend=True,
        legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center'),
        margin=dict(l=60, r=140, t=70, b=130)
    )
    
    fig.write_html(output_file)

def create_signature_plot(gradient_data, result, output_file):
    """Create signature analysis visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Policy Gradient Signatures', 'Periodic Patterns', 'Evasion Signatures', 'Overall Assessment'],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. Policy gradient signatures (first 200 elements)
    policy_section = gradient_data.flatten()[:200]
    fig.add_trace(
        go.Scatter(x=list(range(len(policy_section))), y=policy_section, mode='lines', name='Policy Signatures'),
        row=1, col=1
    )
    
    # 2. Periodic patterns analysis
    fft_data = np.abs(np.fft.fft(gradient_data.flatten()))
    freqs = np.fft.fftfreq(len(gradient_data.flatten()))
    
    # Focus on low frequencies for periodic patterns
    low_freq_mask = np.abs(freqs) < 0.1
    fig.add_trace(
        go.Scatter(x=freqs[low_freq_mask], y=fft_data[low_freq_mask], mode='lines', name='Periodic Patterns'),
        row=1, col=2
    )
    
    # 3. Evasion signatures (high frequency components)
    high_freq_mask = np.abs(freqs) > 0.1
    fig.add_trace(
        go.Scatter(x=freqs[high_freq_mask], y=fft_data[high_freq_mask], mode='lines', name='Evasion Signatures'),
        row=2, col=1
    )
    
    # 4. Overall assessment
    components = list(result['component_scores'].keys())
    scores = list(result['component_scores'].values())
    
    fig.add_trace(
        go.Bar(x=components, y=scores, name='Component Scores'),
        row=2, col=2
    )
    
    fig.update_layout(
        title=_brand_title("RL Signature Analysis"),
        height=800,
        showlegend=True
    )
    
    fig.write_html(output_file) 
def handle_rl_obfuscation_command(args):
    """
    Main handler for rl-obfuscation commands
    Routes to appropriate sub-handler based on args.rl_command
    """
    if args.rl_command == 'analyze':
        return handle_single_analysis(args)
    elif args.rl_command == 'batch':
        return handle_batch_analysis(args)
    elif args.rl_command == 'monitor':
        return handle_real_time_monitoring(args)
    elif args.rl_command == 'signature':
        return handle_signature_analysis(args)
    else:
        print(f"❌ Unknown rl-obfuscation command: {args.rl_command}")
        return 1

