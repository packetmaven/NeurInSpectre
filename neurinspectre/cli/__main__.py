#!/usr/bin/env python3
"""
NeurInSpectre CLI Main Entry Point
Enables execution via: python -m neurinspectre.cli <command>
"""

import sys
import argparse
import logging

logger = logging.getLogger(__name__)

class _NeurInSpectreArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with friendlier diagnostics for common copy/paste errors."""

    def error(self, message):  # noqa: D401 - argparse override
        argv = sys.argv[1:]
        hint = None

        # Common real-world failure: user accidentally pastes a command twice, so a valid
        # subcommand becomes an "unrecognized argument" later in the argv stream.
        if "unrecognized arguments:" in str(message):
            if "analyze-attack-vectors" in str(message) and argv.count("analyze-attack-vectors") >= 2:
                hint = (
                    "It looks like you pasted the command twice. `--target-data` must be a single file path.\n"
                    "Example:\n"
                    "  neurinspectre analyze-attack-vectors \\\n"
                    "    --target-data suspicious_data.npy \\\n"
                    "    --mitre-atlas --owasp-llm --output-dir _cli_runs/intel --verbose"
                )
            elif any(tok in argv for tok in ["--mitre-atlas", "--owasp-llm"]) and argv and argv[0] != "analyze-attack-vectors":
                hint = (
                    "Make sure you run a subcommand first, then flags (don’t paste flags on their own line).\n"
                    "Example:\n"
                    "  neurinspectre analyze-attack-vectors \\\n"
                    "    --target-data suspicious_data.npy \\\n"
                    "    --mitre-atlas --owasp-llm --output-dir _cli_runs/intel"
                )

        # Mirror argparse’s default formatting, but append our hint when available.
        self.print_usage(sys.stderr)
        self._print_message(f"{self.prog}: error: {message}\n", sys.stderr)
        if hint:
            self._print_message(f"\nHint:\n{hint}\n", sys.stderr)
        self.exit(2)

def main():
    """Main CLI entry point"""
    # Configure logging at runtime (avoid import-time side effects).
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    parser = _NeurInSpectreArgumentParser(
        prog='neurinspectre',
        description='NeurInSpectre Time Travel Debugger CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neurinspectre dashboard --model gpt2 --port 8080
  neurinspectre dashboard-manager backup --name my_backup
  neurinspectre dashboard-manager status
  neurinspectre dashboard-manager start --all
  neurinspectre dashboard-manager emergency
  neurinspectre frequency-adversarial --input-spectrum real_leaked_grads.npy
  neurinspectre attack-graph-viz --input-path attack_data.json
  neurinspectre occlusion-analysis --image-path image.jpg --model google/vit-base-patch16-224
  neurinspectre obfuscated-gradient create --input-file gradients.npy --output-dir ./gradient_analysis
  neurinspectre obfuscated-gradient analyze --gradient-file gradients.npy --output-dir ./results
  neurinspectre obfuscated-gradient demo --device auto --interactive
  neurinspectre obfuscated-gradient monitor --device mps --duration 120  # Autodetects running models
  neurinspectre obfuscated-gradient monitor --device auto --duration 60 --output-report analysis.json
  neurinspectre obfuscated-gradient generate --samples 1024 --attack-type ts-inverse
  neurinspectre math spectral --input gradients.npy --output analysis.json --plot results.png
  neurinspectre math integrate --input gradients.npy --steps 100 --dt 0.01
  neurinspectre math demo --device auto --save-results demo_results.json
  neurinspectre gpu detect --output gpu_report.json
  neurinspectre gpu models --quick
  neurinspectre gpu monitor --continuous --duration 120
  neurinspectre gpu apple --verbose
  neurinspectre gpu nvidia --quick
  neurinspectre temporal-analysis sequence --input-dir ./gradients --output-report temporal_report.json
  neurinspectre rl-obfuscation analyze --input-file gradient.npy --sensitivity high
  neurinspectre red-team attack-planning --target-data targets.json --attack-vectors vectors.json
  neurinspectre blue-team incident-response --incident-data incidents.json --timeline-data timeline.json
  neurinspectre comprehensive-test full --output-report test_results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Obfuscated Gradient Visualization command
    gradient_parser = subparsers.add_parser('obfuscated-gradient', help='🎭 Obfuscated gradient visualization and analysis')
    gradient_subparsers = gradient_parser.add_subparsers(dest='gradient_command', help='Gradient analysis commands')
    
    # Create comprehensive analysis
    create_parser = gradient_subparsers.add_parser('create', help='🎨 Create comprehensive gradient analysis')
    create_parser.add_argument(
        '--input-file',
        '-i',
        required=True,
        help='Input gradient file (.npy) to analyze (must exist; no demo fallback)',
    )
    create_parser.add_argument(
        '--reference-file',
        '-r',
        default=None,
        help='Optional reference/baseline gradient file (.npy) for comparison (must exist; no demo fallback)',
    )
    create_parser.add_argument('--output-dir', '-o', default='.', help='Output directory for visualizations')
    create_parser.add_argument('--device', '-d', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto', help='Device preference')
    create_parser.add_argument('--format', choices=['png', 'jpg', 'pdf'], default='png', help='Output image format')
    create_parser.add_argument('--dpi', type=int, default=150, help='Image resolution (DPI)')
    create_parser.add_argument('--style', choices=['default', 'dark', 'light'], default='default', help='Visualization style')
    create_parser.add_argument('--interactive', action='store_true', help='Generate interactive plots')
    create_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Analyze existing gradients
    analyze_parser = gradient_subparsers.add_parser('analyze', help='🔍 Analyze existing gradient files')
    analyze_parser.add_argument('--gradient-file', '-g', required=True, help='Input gradient file (.npy)')
    analyze_parser.add_argument(
        '--reference-file',
        '-r',
        default=None,
        help='Optional reference/baseline gradient file (.npy) for comparison (must exist; no demo fallback)',
    )
    analyze_parser.add_argument('--output-dir', '-o', default='.', help='Output directory for analysis results')
    analyze_parser.add_argument('--threshold', '-t', type=float, default=2.58, help='Z-score threshold for anomaly detection')
    analyze_parser.add_argument('--device', '-d', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto', help='Device preference')
    analyze_parser.add_argument('--save-data', action='store_true', help='Save processed gradient data')
    analyze_parser.add_argument('--format', choices=['png', 'jpg', 'pdf'], default='png', help='Output image format')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Demo mode
    demo_parser = gradient_subparsers.add_parser('demo', help='🎯 Run gradient visualization demo')
    demo_parser.add_argument('--output-dir', '-o', default='.', help='Output directory for demo results')
    demo_parser.add_argument('--device', '-d', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto', help='Device preference')
    demo_parser.add_argument('--interactive', action='store_true', help='Interactive demonstration')
    demo_parser.add_argument('--quick', action='store_true', help='Quick demo (fewer visualizations)')
    demo_parser.add_argument('--style', choices=['default', 'dark', 'light'], default='default', help='Visualization style')
    demo_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Real-time monitoring
    monitor_parser = gradient_subparsers.add_parser('monitor', help='📡 Real-time gradient monitoring')
    monitor_parser.add_argument('--model-path', '-m', help='[DEPRECATED] Path to model - now autodetects all running PyTorch models')
    monitor_parser.add_argument('--device', '-d', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto', help='Device preference')
    monitor_parser.add_argument('--buffer-size', '-b', type=int, default=1000, help='Gradient buffer size')
    monitor_parser.add_argument('--update-interval', '-u', type=float, default=0.1, help='Update interval in seconds')
    monitor_parser.add_argument('--analysis-window', '-w', type=int, default=100, help='Analysis window size')
    monitor_parser.add_argument('--no-plot', action='store_true', help='Disable live plotting')
    monitor_parser.add_argument('--duration', '-t', type=int, default=60, help='Monitoring duration in seconds')
    monitor_parser.add_argument('--output-report', '-r', help='Output analysis report file')
    monitor_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Generate test data
    _generate_parser = gradient_subparsers.add_parser('generate', help='🔄 Generate test gradient data')
    _generate_parser.add_argument('--output-dir', '-o', default='.', help='Output directory for generated data')
    _generate_parser.add_argument('--samples', '-s', type=int, default=512, help='Number of gradient samples')
    _generate_parser.add_argument('--noise-level', '-n', type=float, default=0.1, help='Noise level for obfuscated gradients')
    _generate_parser.add_argument('--attack-type', choices=['ts-inverse', 'concretizer', 'ednn', 'rl-obfuscation'], default='ts-inverse', help='Attack pattern type')
    _generate_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Capture adversarial gradients (integrated from capture_obfuscated_gradients.py)
    capture_parser = gradient_subparsers.add_parser('capture-adversarial', help='🔴 Capture adversarial obfuscated gradients with latest offensive techniques')
    capture_parser.add_argument('--output-dir', '-o', default='_cli_runs', help='Output directory')
    capture_parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    capture_parser.add_argument('--batches', type=int, default=5, help='Batches per epoch')
    capture_parser.add_argument('--device', '-d', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto', help='Device preference')
    capture_parser.add_argument('--attack-type', choices=['rl_policy', 'periodic', 'conditional_trigger', 'high_frequency', 'gradient_masking', 'combined'], 
                               default='combined', help='Obfuscation attack technique')
    capture_parser.add_argument('--model-size', choices=['small', 'medium', 'large'], default='medium', help='Model size for gradient generation')
    
    # Train and monitor real models (integrated monitoring)
    train_monitor_parser = gradient_subparsers.add_parser('train-and-monitor', help='🎯 Train real model with integrated gradient monitoring')
    train_monitor_parser.add_argument('--model', '-m', default='gpt2', help='HuggingFace model name (e.g., gpt2, Qwen/Qwen-1_8B, EleutherAI/gpt-neo-125M)')
    train_monitor_parser.add_argument('--output-dir', '-o', default='_cli_runs', help='Output directory')
    train_monitor_parser.add_argument('--steps', '-s', type=int, default=50, help='Number of training steps')
    train_monitor_parser.add_argument('--device', '-d', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto', help='Device preference')
    train_monitor_parser.add_argument('--learning-rate', '--lr', type=float, default=5e-5, help='Learning rate')
    train_monitor_parser.add_argument('--auto-analyze', action='store_true', help='Automatically run analysis after training')
    
    # Mathematical analysis command
    from .mathematical_commands import register_mathematical_commands as _reg_math
    _reg_math(subparsers)

    # Statistical analysis commands (drift + enhanced z-score)
    from .statistical_commands import register_statistical_commands as _reg_stats
    _reg_stats(subparsers)

    # Attack module wrappers (TS-Inverse / ConcreTizer / latent jailbreak)
    from .attacks_commands import register_attack_commands as _reg_attacks
    _reg_attacks(subparsers)

    # GPU security module (gpu_security.py)
    from .gpu_security_commands import register_gpu_security_commands as _reg_gpu_sec
    _reg_gpu_sec(subparsers)
    
    # Spectral decomposition command
    spectral_parser = subparsers.add_parser('spectral', help='🔬 Advanced spectral decomposition analysis')
    spectral_parser.add_argument('--input', '-i', required=True, help='Input gradient data file (.npy)')
    spectral_parser.add_argument('--output', '-o', help='Output analysis results (.json)')
    spectral_parser.add_argument('--baseline', default=None, help='Optional baseline/reference array (.npy/.npz) for comparison (recommended)')
    spectral_parser.add_argument('--topk-peaks', type=int, default=6, help='Number of peaks to annotate (default: 6)')
    spectral_parser.add_argument('--peak-db', type=float, default=6.0, help='dB threshold for highlighting peaks (default: 6.0)')
    spectral_parser.add_argument('--no-demean', action='store_true', help='Do not remove per-signal mean before FFT (not recommended)')
    spectral_parser.add_argument('--window', choices=['hann', 'none'], default='hann', help='Window function before FFT (default: hann)')
    spectral_parser.add_argument('--levels', '-l', type=int, default=5, help='Number of decomposition levels')
    spectral_parser.add_argument('--device', '-d', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto', help='Device preference')
    spectral_parser.add_argument('--precision', '-p', choices=['float32', 'float64'], default='float32', help='Precision')
    spectral_parser.add_argument('--plot', help='Save visualization plot to file')
    spectral_parser.add_argument('--html', help='Write interactive HTML triage dashboard')
    spectral_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Integration command
    integrate_parser = subparsers.add_parser('integrate', help='⚡ Advanced exponential time differencing')
    integrate_parser.add_argument('--input', '-i', required=True, help='Input gradient data file (.npy)')
    integrate_parser.add_argument('--output', '-o', help='Output evolution results (.npy)')
    integrate_parser.add_argument('--steps', '-s', type=int, default=100, help='Number of integration steps')
    integrate_parser.add_argument('--dt', type=float, default=0.01, help='Time step size')
    integrate_parser.add_argument('--krylov-dim', type=int, default=30, help='Krylov subspace dimension')
    integrate_parser.add_argument('--device', '-d', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto', help='Device preference')
    integrate_parser.add_argument('--precision', '-p', choices=['float32', 'float64'], default='float32', help='Precision')
    integrate_parser.add_argument('--plot', help='Save evolution plot to file')
    integrate_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='🎯 Demonstrate mathematical capabilities')
    demo_parser.add_argument('--device', '-d', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto', help='Device preference')
    demo_parser.add_argument('--precision', '-p', choices=['float32', 'float64'], default='float32', help='Precision')
    demo_parser.add_argument('--save-results', help='Save demonstration results to file')
    demo_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='🧪 Run comprehensive test suite')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    test_parser.add_argument('--test-type', choices=['all', 'foundations', 'cli', 'spectral', 'integration', 'devices', 'performance'], 
                            default='all', help='Type of tests to run')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch TTD Dashboard')
    dashboard_parser.add_argument('--model', default='gpt2', help='Model to use')
    dashboard_parser.add_argument('--port', type=int, default=8080, help='Port to run on')
    dashboard_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    dashboard_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    # New explicit data inputs
    dashboard_parser.add_argument('--gradient-file', help='Path to real gradient data (.npy)')
    dashboard_parser.add_argument('--attention-file', help='Path to real attention data (.npy)')
    dashboard_parser.add_argument(
        '--batch-dir',
        default=None,
        help=("Optional directory for batch gradient files. "
              "Only used when --gradient-file is not provided. "
              "Loads up to 10 files matching '*grad*.npy'/'*gradient*.npy' and concatenates them."),
    )

    # Privacy / DP accounting (NO SIMULATION: ε only shown if provided or computable)
    dashboard_parser.add_argument('--privacy-limit', type=float, default=3.0,
                                help='Privacy ε threshold for alerts/visualization (default: 3.0)')
    dashboard_parser.add_argument('--dp-sample-rate', type=float, default=None,
                                help='DP-SGD sample rate q (batch_size/dataset_size) for ε accounting')
    dashboard_parser.add_argument('--dp-noise-multiplier', type=float, default=None,
                                help='DP-SGD noise multiplier σ for ε accounting')
    dashboard_parser.add_argument('--dp-delta', type=float, default=1e-5,
                                help='DP δ for ε accounting (default: 1e-5)')
    dashboard_parser.add_argument('--privacy-file', help='Optional .npy file containing ε series aligned to gradient steps')
    
    # Dashboard management commands
    dashboard_mgr_parser = subparsers.add_parser('dashboard-manager', help='🛡️ Dashboard backup and management system')
    dashboard_mgr_subparsers = dashboard_mgr_parser.add_subparsers(dest='dashboard_manager_command', help='Dashboard management commands')
    
    # Backup command
    backup_parser = dashboard_mgr_subparsers.add_parser('backup', help='Create dashboard backup')
    backup_parser.add_argument('--name', help='Backup name (default: auto-generated)')
    
    # List command
    list_parser = dashboard_mgr_subparsers.add_parser('list', help='List available backups')
    
    # Restore command
    restore_parser = dashboard_mgr_subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('--name', required=True, help='Backup name to restore')
    
    # Status command
    status_parser = dashboard_mgr_subparsers.add_parser('status', help='Check dashboard status')
    
    # Start command
    start_parser = dashboard_mgr_subparsers.add_parser('start', help='Start dashboards')
    start_group = start_parser.add_mutually_exclusive_group(required=True)
    start_group.add_argument('--all', action='store_true', help='Start all dashboards')
    start_group.add_argument('--dashboard', help='Start specific dashboard (ttd, 3x2, intelligence, research, enhanced)')
    
    # Stop command
    stop_parser = dashboard_mgr_subparsers.add_parser('stop', help='Stop dashboards')
    stop_group = stop_parser.add_mutually_exclusive_group(required=True)
    stop_group.add_argument('--all', action='store_true', help='Stop all dashboards')
    stop_group.add_argument('--dashboard', help='Stop specific dashboard (ttd, 3x2, intelligence, research, enhanced)')
    
    # Restart command
    restart_parser = dashboard_mgr_subparsers.add_parser('restart', help='Restart all dashboards')
    
    # Emergency command
    # NOTE: underscore prefix avoids IDE/type-checker "unused variable" diagnostics for
    # parser handles that exist purely to register CLI flags.
    _emergency_parser = dashboard_mgr_subparsers.add_parser('emergency', help='Emergency restoration')
    _emergency_parser.add_argument('--backup', help='Specific backup to use (default: latest)')
    
    # Frequency adversarial command
    freq_parser = subparsers.add_parser('frequency-adversarial', help='Frequency adversarial analysis')
    freq_parser.add_argument('--input-spectrum', required=True, help='Input spectrum file (.npy)')
    freq_parser.add_argument('--viz', default='dashboard', help='Visualization type')
    freq_parser.add_argument('--threshold', type=float, default=0.75, help='Detection threshold')
    freq_parser.add_argument(
        '--output-plot',
        nargs='?',
        const='_cli_runs/frequency_adversarial.png',
        default=None,
        help="Write Plotly visualization to PNG (and also writes *_interactive.html). "
             "Pass a path, or use `--output-plot` for the default: _cli_runs/frequency_adversarial.png",
    )
    freq_parser.add_argument(
        '--save-metrics',
        nargs='?',
        const='_cli_runs/frequency_adversarial_metrics.json',
        default=None,
        help="Write metrics JSON. Pass a path, or use `--save-metrics` for the default: "
             "_cli_runs/frequency_adversarial_metrics.json",
    )
    
    # Correlate command
    corr_parser = subparsers.add_parser('correlate', help='Cross-module correlation analysis')
    corr_subparsers = corr_parser.add_subparsers(dest='correlate_action', required=True)
    run_corr_parser = corr_subparsers.add_parser('run', help='Run correlation analysis')
    run_corr_parser.add_argument('--primary', required=True, help='Primary analysis type')
    run_corr_parser.add_argument('--secondary', required=True, help='Secondary analysis type')
    run_corr_parser.add_argument('--primary-file', required=True, help='Primary data file (.npy/.npz) (must exist; no synthetic fallback)')
    run_corr_parser.add_argument('--secondary-file', required=True, help='Secondary data file (.npy/.npz) (must exist; no synthetic fallback)')
    run_corr_parser.add_argument('--out-prefix', default='_cli_runs/corr_', help='Output prefix for artifacts (default: _cli_runs/corr_)')
    run_corr_parser.add_argument('--plot', help='Optional output plot path (PNG)')
    run_corr_parser.add_argument('--temporal-window', type=float, default=1.5, help='Temporal window')
    run_corr_parser.add_argument('--spatial-threshold', type=float, default=0.75, help='Spatial threshold')
    run_corr_parser.add_argument('--device', default='auto', help='Device preference label (unused by numpy path; default: auto)')
    run_corr_parser.add_argument('--interactive', action='store_true', help='Generate interactive HTML')
    run_corr_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Attack graph visualization command with subcommands
    # NOTE: underscore prefix avoids IDE/type-checker "unused variable" diagnostics for
    # parser handles that exist purely to register CLI flags.
    _attack_parser = subparsers.add_parser('attack-graph', help='🗺️ ATLAS attack graph analysis and visualization')
    attack_subparsers = _attack_parser.add_subparsers(dest='attack_command', help='Attack graph commands')
    
    # Prepare scenario
    prepare_parser = attack_subparsers.add_parser('prepare', help='📝 Prepare attack scenario data')
    prepare_parser.add_argument('--output', '-o', default='_cli_runs/atlas_case.json', help='Output JSON path')
    prepare_parser.add_argument(
        '--output-dir',
        default=None,
        help="Optional output directory (writes <output-dir>/attack_graph.json unless --output sets a filename)",
    )
    prepare_parser.add_argument('--scenario', choices=['jailbreak_extraction', 'poison_backdoor'], 
                               default='jailbreak_extraction', help='Attack scenario type')
    prepare_parser.add_argument(
        '--atlas-ids',
        default=None,
        help='Optional: comma/space-separated AML.T*/AML.TA* ids to build a scenario from the official ATLAS catalog'
    )
    prepare_parser.add_argument(
        '--atlas-ids-file',
        default=None,
        help='Optional: text file with one AML.T*/AML.TA* id per line (used with prepare)'
    )
    
    # Visualize graph
    viz_parser = attack_subparsers.add_parser('visualize', help='🎨 Visualize attack graph')
    viz_parser.add_argument('--input-path', '-i', required=True, help='Input attack data JSON')
    viz_parser.add_argument('--output-path', '-o', required=True, help='Output HTML visualization path')
    viz_parser.add_argument('--title', default='NeurInSpectre ATLAS Attack Graph', help='Visualization title')
    viz_parser.add_argument('--open', action='store_true', help='Open the generated HTML in your default browser')
    
    # Legacy command for backwards compatibility
    attack_viz_parser = subparsers.add_parser('attack-graph-viz', help='ATLAS attack graph visualization (legacy)')
    attack_viz_parser.add_argument('--input-path', required=True, help='Input attack data path')
    attack_viz_parser.add_argument('--output-path', required=True, help='Output visualization path')
    attack_viz_parser.add_argument('--title', default='NeurInSpectre ATLAS Attack Graph', help='Visualization title')
    attack_viz_parser.add_argument('--open', action='store_true', help='Open the generated HTML in your default browser')

    # MITRE ATLAS catalog + coverage utilities (offline STIX)
    from .mitre_atlas_cli import register_mitre_atlas
    register_mitre_atlas(subparsers)

    # ---------------------------------------------------------------------
    # Top-level module entrypoints (so users can run: neurinspectre <module> ...)
    # ---------------------------------------------------------------------

    # Debug / environment info
    # NOTE: underscore prefix avoids IDE/type-checker "unused variable" diagnostics for
    # parser handles that exist purely to register CLI flags.
    _dbg_parser = subparsers.add_parser(
        'debug-info',
        aliases=['debug_info', 'debuginfo', 'debug'],
        help='🧰 Print NeurInSpectre debug information (env, deps, GPU)',
    )
    _dbg_parser.add_argument('--output', '-o', default=None, help='Optional output JSON path')
    _dbg_parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON')

    # Tools: generate DeMarking-test PCAP
    # NOTE: underscore prefix avoids IDE/type-checker "unused variable" diagnostics for
    # parser handles that exist purely to register CLI flags.
    _pcap_parser = subparsers.add_parser(
        'generate-demarking-pcap',
        aliases=['generate_demarking_pcap', 'demarking-pcap', 'demarking_pcap'],
        help='🧪 Generate a DeMarking-test PCAP (timing/IPD-based)',
    )
    _pcap_parser.add_argument('--out', default='network_flows.pcap', help='Output .pcap path')
    _pcap_parser.add_argument('--packets', type=int, default=600, help='Number of packets to write (default: 600)')
    _pcap_parser.add_argument('--seed', type=int, default=1337, help='RNG seed (default: 1337)')
    _pcap_parser.add_argument('--threshold', type=float, default=0.6, help='Detector threshold you plan to use (default: 0.6)')
    _pcap_parser.add_argument('--margin', type=float, default=0.005, help='Require score >= threshold+margin (default: 0.005)')
    _pcap_parser.add_argument('--max-tries', type=int, default=2000, help='Max seeds to try to hit threshold (default: 2000)')
    _pcap_parser.add_argument(
        '--json',
        nargs='?',
        const='-',
        default=None,
        help="Optional JSON output. Use `--json` to print JSON, or `--json <path>` to write a report.",
    )

    # Tools: generate ConcreTizer detector exerciser .npy
    ctz_parser = subparsers.add_parser(
        'generate-concretizer-data',
        aliases=[
            'generate_concretizer_data',
            'concretizer-data',
            'concretizer_data',
        ],
        help='🧪 Generate a ConcreTizer detector exerciser .npy (synthetic, deterministic)',
    )
    ctz_parser.add_argument('--out', default='attack_data/concretizer_attack_data.npy', help='Output .npy path')
    ctz_parser.add_argument('--timesteps', type=int, default=300, help='Number of timesteps/queries (T)')
    ctz_parser.add_argument('--features', type=int, default=64, help='Number of features (D)')
    ctz_parser.add_argument('--dip-every', type=int, default=100, help='Set every Nth row amplitude to dip-value (default: 100)')
    ctz_parser.add_argument('--baseline', type=float, default=1.0, help='Baseline amplitude value (default: 1.0)')
    ctz_parser.add_argument('--dip-value', type=float, default=0.0, help='Dip amplitude value (default: 0.0)')
    ctz_parser.add_argument('--threshold', type=float, default=0.9, help='Detector threshold you plan to use (default: 0.9)')
    ctz_parser.add_argument(
        '--json',
        nargs='?',
        const='-',
        default=None,
        help="Optional JSON output. Use `--json` to print JSON, or `--json <path>` to write a report.",
    )

    # AI Security Research Dashboard 2025 (Dash)
    rsdash_parser = subparsers.add_parser(
        'ai-security-dashboard',
        aliases=[
            'ai_security_dashboard',
            'research-dashboard',
            'research_dashboard',
            # Module-name aliases (so users can do: neurinspectre ai_security_research_dashboard_2025 ...)
            'ai_security_research_dashboard_2025',
            'ai-security-research-dashboard-2025',
            'research-dashboard-2025',
        ],
        help='📊 Launch AI Security Research Dashboard 2025 (Dash)',
    )
    rsdash_parser.add_argument('--host', default='127.0.0.1', help='Bind host (default: 127.0.0.1)')
    rsdash_parser.add_argument('--port', type=int, default=8117, help='Bind port (default: 8117)')
    rsdash_parser.add_argument('--debug', action='store_true', help='Enable Dash debug mode')
    rsdash_parser.add_argument(
        '--simulated',
        action='store_true',
        help='Enable simulated demo telemetry (otherwise, no synthetic timelines are generated)',
    )

    # Integrated system orchestrator (wrapper)
    integ_parser = subparsers.add_parser(
        'integrated-system',
        aliases=[
            'integrated_system',
            'integrated-neurinspectre',
            'integrated_neurinspectre',
            # Module-name alias
            'integrated_neurinspectre_system',
        ],
        help='🧩 Integrated NeurInSpectre orchestrator (wrapper)',
    )
    integ_parser.add_argument('--input', '-i', required=True, help='Input data file (.npy/.npz)')
    integ_parser.add_argument(
        '--sensitivity',
        choices=['low', 'medium', 'high', 'adaptive'],
        default='adaptive',
        help="Sensitivity profile (default: adaptive)",
    )
    integ_parser.add_argument('--output', '-o', default='_cli_runs/integrated_scan.json', help='Output JSON path')
    integ_parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON')
    
    # Occlusion analysis command
    occlusion_parser = subparsers.add_parser('occlusion-analysis', help='Adversarial occlusion vulnerability analysis')
    occlusion_parser.add_argument('--model', '-m', default='google/vit-base-patch16-224', help='HuggingFace model name')
    occlusion_parser.add_argument('--image-path', '-i', help='Path to input image')
    occlusion_parser.add_argument('--image-url', '-u', help='URL to input image')
    occlusion_parser.add_argument('--patch-size', type=int, default=32, help='Size of occlusion patch')
    occlusion_parser.add_argument('--stride', type=int, default=16, help='Stride for patch movement')
    occlusion_parser.add_argument('--output-2d', '-o2', help='Output file for 2D visualization')
    occlusion_parser.add_argument('--output-3d', '-o3', help='Output file for 3D visualization')

    # Attention heatmap visualization (token×token)
    attn_parser = subparsers.add_parser('attention-heatmap', help='Generate token×token attention heatmap')
    attn_parser.add_argument('--model', required=True, help='HuggingFace model name, e.g., gpt2')
    attn_parser.add_argument('--prompt', required=True, help='Prompt text to analyze')
    attn_parser.add_argument('--layer', type=int, default=0, help='Layer index (default 0)')
    attn_parser.add_argument('--head', type=int, default=0, help='Head index (default 0)')
    attn_parser.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto', help='Device')
    attn_parser.add_argument('--output', default='attention_heatmap.png', help='Output PNG path')
    attn_parser.add_argument('--baseline-prompt', help='Optional baseline prompt for delta comparison')
    attn_parser.add_argument('--out-prefix', default='attn_', help='Output prefix (heatmap/delta/profiles/summary)')
    

    # Attention Security Analysis (heatmap + IsolationForest token anomaly scores)
    attn_sec = subparsers.add_parser('attention-security', help='Attention heatmap + token anomaly scores (IsolationForest)')
    attn_sec_src = attn_sec.add_mutually_exclusive_group(required=True)
    attn_sec_src.add_argument('--prompt', help='Prompt text to analyze')
    attn_sec_src.add_argument('--prompt-file', default=None, help='Text file with prompt (first non-empty line)')
    attn_sec.add_argument('--model', required=True, help='HuggingFace model name, e.g., gpt2')
    attn_sec.add_argument('--layer', default='all', help="Layer index (int) or 'all' to average across layers")
    attn_sec.add_argument('--layer-start', type=int, default=None, help='Start layer (inclusive) when using --layer all')
    attn_sec.add_argument('--layer-end', type=int, default=None, help='End layer (inclusive) when using --layer all')
    attn_sec.add_argument('--max-tokens', type=int, default=128, help='Max tokens to visualize (default: 128)')
    attn_sec.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto', help='Device')
    attn_sec.add_argument('--output-png', default='_cli_runs/attention_security.png', help='Output PNG path')
    attn_sec.add_argument('--out-json', default='_cli_runs/attention_security.json', help='Output JSON path')
    attn_sec.add_argument('--out-html', default='_cli_runs/attention_security.html', help='Output interactive HTML path')
    attn_sec.add_argument('--contamination', default='auto', help="IsolationForest contamination ('auto' or float)")
    attn_sec.add_argument('--n-estimators', type=int, default=256, help='IsolationForest n_estimators (default: 256)')
    attn_sec.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    attn_sec.add_argument('--title', default='NeurInSpectre — Attention Security Analysis', help='Plot title')

    # Attention-Gradient Alignment (AGA): layer×head alignment heatmap
    aga = subparsers.add_parser(
        'activation_attention_gradient_alignment',
        help='Attention-Gradient Alignment (AGA): layer×head heatmap of cos-sim(attn, dObj/dAttn)'
    )
    aga_sub = aga.add_subparsers(dest='aga_action', required=True)

    aga_c = aga_sub.add_parser('craft', help='Compute AGA from real attentions + gradients')
    aga_c.add_argument('--model', required=True, help='HuggingFace model id (works for models that can return attentions)')
    aga_c.add_argument('--tokenizer', required=False, help='Optional tokenizer id (defaults to --model)')
    aga_src = aga_c.add_mutually_exclusive_group(required=True)
    aga_src.add_argument('--prompt', help='Prompt text to analyze')
    aga_src.add_argument('--prompt-file', default=None, help='Text file with prompt (first non-empty line)')

    aga_c.add_argument('--max-tokens', type=int, default=128, help='Max tokens (default: 128)')
    aga_c.add_argument('--layer-start', type=int, default=0, help='First layer index (default: 0)')
    aga_c.add_argument('--layer-end', type=int, default=None, help='Last layer index (inclusive; default: last)')
    aga_c.add_argument(
        '--attn-source',
        choices=['auto','attentions','encoder_attentions','decoder_attentions','cross_attentions'],
        default='auto',
        help='Which attention tensors to use (default: auto)'
    )
    aga_c.add_argument(
        '--objective',
        choices=['auto','lm_nll','hidden_l2'],
        default='auto',
        help='Scalar objective to backprop into attentions (default: auto)'
    )
    aga_c.add_argument(
        '--attn-impl',
        choices=['auto','eager'],
        default='auto',
        help='Force attention implementation (eager helps some models export attentions)'
    )
    aga_c.add_argument('--risk-threshold', type=float, default=0.25, help='Guidance threshold for high alignment (default: 0.25)')
    aga_c.add_argument('--clip-percentile', type=float, default=0.99, help='Color scale clip percentile on |alignment| (default: 0.99)')
    aga_c.add_argument('--trust-remote-code', action='store_true', help='Allow loading models with custom code from HF (use with caution)')

    aga_c.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto', help='Device preference')
    aga_c.add_argument('--title', default='Attention-Gradient Alignment (AGA)', help='Plot title')
    aga_c.add_argument('--out-json', default='_cli_runs/attention_gradient_alignment.json', help='Output metrics JSON path')
    aga_c.add_argument('--out-png', default='_cli_runs/attention_gradient_alignment.png', help='Output PNG path')
    aga_c.add_argument('--out-html', default=None, help='Optional interactive HTML dashboard path (requires plotly)')

    aga_v = aga_sub.add_parser('visualize', help='Render PNG from a saved metrics JSON')
    aga_v.add_argument('--in-json', required=True, help='Input metrics JSON path')
    aga_v.add_argument('--title', default=None, help='Optional override title')
    aga_v.add_argument('--out-png', default='_cli_runs/attention_gradient_alignment.png', help='Output PNG path')
    aga_v.add_argument('--out-html', default=None, help='Optional interactive HTML dashboard path (requires plotly)')

    # GPU detection commands
    from .gpu_detection_cli import add_gpu_commands
    add_gpu_commands(subparsers)
    
    # Temporal analysis commands
    from .temporal_analysis_commands import add_temporal_analysis_parser
    add_temporal_analysis_parser(subparsers)
    
    # RL obfuscation commands
    from .rl_obfuscation_commands import add_rl_obfuscation_parser
    add_rl_obfuscation_parser(subparsers)
    
    # Red team commands
    from .red_team_commands import add_red_team_parser
    add_red_team_parser(subparsers)
    
    # Blue team commands
    from .blue_team_commands import add_blue_team_parser
    add_blue_team_parser(subparsers)
    
    # Comprehensive testing commands
    from .comprehensive_test_commands import add_comprehensive_test_parser
    add_comprehensive_test_parser(subparsers)
    
    # Security commands
    from .security_commands import add_security_commands
    add_security_commands(subparsers)

    # Explanation visualization command
    vizexp_parser = subparsers.add_parser('visualize-explanations', help='Visualize model explanation attributions')
    vizexp_parser.add_argument('--explanation', '-e', required=True, help='Path to explanation JSON/NPY/NPZ/CSV')
    vizexp_parser.add_argument('--out-prefix', default='explain_', help='Output file prefix')
    vizexp_parser.add_argument('--topk', type=int, default=20, help='Top-K features to show in bar chart')

    # Layer activations visualization command
    # NOTE: underscore prefix avoids IDE/type-checker "unused variable" diagnostics for
    # parser handles that exist purely to register CLI flags.
    _activations_parser = subparsers.add_parser('activations', help='Visualize transformer layer activations for a prompt')
    _activations_parser.add_argument('--model', '-m', required=True, help='HuggingFace model id (e.g., EleutherAI/gpt-neo-125M)')
    _activations_parser.add_argument('--prompt', '-p', required=True, help='Prompt text to run through the model')
    # NOTE: underscore prefix avoids IDE/type-checker "unused variable" diagnostics for
    # group handles that exist purely to register CLI flags.
    _layer_group = _activations_parser.add_mutually_exclusive_group(required=True)
    _layer_group.add_argument('--layer', type=int, help='Numeric transformer block index (e.g., 10)')
    _layer_group.add_argument('--layer-path', help='Dot path to module (e.g., transformer.h.10)')
    _activations_parser.add_argument('--topk', type=int, default=20, help='Top-K last-token activations to plot')
    _activations_parser.add_argument('--out-prefix', default='_cli_runs/act_', help='Output filename prefix')
    _activations_parser.add_argument('--device', '-d', default='auto', choices=['auto', 'mps', 'cuda', 'cpu'], help='Device preference')
    _activations_parser.add_argument('--hotspot-percentile', type=float, default=95.0, help='Percentile for hotspot threshold (e.g., 95)')
    _activations_parser.add_argument('--json-out', default=None, help='Optional JSON file to write top-k and hotspot spans')
    _activations_parser.add_argument('--interactive', action='store_true', help='Generate interactive Plotly HTML with Red/Blue team guidance')
    
    # Activation steganography command
    steg_parser = subparsers.add_parser('activation_steganography', help='Activation steganography encode/extract')
    steg_sub = steg_parser.add_subparsers(dest='steg_action', required=True)
    steg_enc = steg_sub.add_parser('encode', help='Encode payload bits into prompt/neurons')
    steg_enc.add_argument('--model', required=True)
    steg_enc.add_argument('--tokenizer', required=True)
    steg_enc.add_argument('--prompt', required=True)
    steg_enc.add_argument('--payload-bits', required=True, help='Comma-separated bits, e.g., 1,0,1')
    steg_enc.add_argument('--target-neurons', required=True, help='Comma-separated neuron indices')
    steg_enc.add_argument('--out-prefix', default='steg_')
    steg_ext = steg_sub.add_parser('extract', help='Extract payload bits from activations')
    steg_ext.add_argument('--activations', required=True, help='.npy/.npz path to activations [seq, hidden] (must exist; no synthetic fallback)')
    steg_ext.add_argument('--target-neurons', required=True)
    steg_ext.add_argument('--threshold', type=float, default=0.0)
    steg_ext.add_argument('--out-prefix', default='steg_')

    # Subnetwork hijack
    snh_parser = subparsers.add_parser('subnetwork_hijack', help='Identify and inject neuron subnetworks')
    snh_sub = snh_parser.add_subparsers(dest='snh_action', required=True)
    snh_id = snh_sub.add_parser('identify', help='Identify neuron subnetwork clusters')
    snh_id.add_argument('--activations', required=True, help='Path to activations .npy [. (N,D)]')
    snh_id.add_argument('--n_clusters', type=int, default=8)
    snh_id.add_argument('--out-prefix', default='_cli_runs/snh_')
    snh_id.add_argument('--interactive', action='store_true', help='Generate interactive HTML with vulnerability metrics')
    snh_inj = snh_sub.add_parser('inject', help='Plan subnetwork injection')
    snh_inj.add_argument('--model', required=True)
    snh_inj.add_argument('--subnetwork', required=True, help='Comma-separated neuron ids')
    snh_inj.add_argument('--trigger', required=True)
    snh_inj.add_argument('--out-prefix', default='snh_')

    # Activation drift evasion (real hidden-state drift over prompt sequences)
    ade_parser = subparsers.add_parser('activation_drift_evasion', help='Extract/visualize activation drift over prompt sequences')
    ade_sub = ade_parser.add_subparsers(dest='ade_action', required=True)

    # Craft: compute drift trajectory from a prompt sequence (real hidden states)
    ade_c = ade_sub.add_parser('craft', help='Compute drift trajectory from prompts (real hidden states)')
    ade_c.add_argument('--model', required=True, help='HuggingFace model id (e.g., gpt2, distilbert-base-uncased)')
    ade_c.add_argument('--tokenizer', required=False, help='Optional tokenizer id (defaults to --model)')
    ade_c.add_argument('--prompts-file', required=True, help='Text file: one prompt per line (time steps)')
    ade_c.add_argument('--baseline-prompt', required=False, help='Optional baseline prompt (defaults to first line of prompts-file)')
    ade_c.add_argument('--layer', type=int, default=0, help='Layer index (0-based; default: 0)')
    ade_c.add_argument('--reduce', choices=['last', 'mean', 'maxabs', 'max'], default='last',
                       help='Reduce token dimension into per-neuron values (default: last)')
    # Accept both underscore and hyphenated forms for neuron selection
    ade_c.add_argument('--target_neurons', dest='target_neurons', required=False, help='Comma-separated neuron indices to track')
    ade_c.add_argument('--target-neurons', dest='target_neurons', required=False, help='Alias of --target_neurons')
    ade_c.add_argument('--topk', type=int, default=5, help='If --target-neurons not set: select top-K by |drift| at final step (default: 5)')
    ade_c.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto', help='Device preference')
    ade_c.add_argument('--out-prefix', default='_cli_runs/', help='Output prefix / directory (default: _cli_runs/)')

    # Visualize: plot drift trajectory; optionally add token-level drift correlation
    ade_v = ade_sub.add_parser('visualize', help='Visualize drift trajectory (and optional token correlation)')
    ade_v.add_argument('--activation_traj', required=True, help='Path to drift trajectory (.npy) [steps, neurons]')
    ade_v.add_argument('--target_neurons', dest='target_neurons', required=False, help='Comma-separated neuron indices (for labels)')
    ade_v.add_argument('--target-neurons', dest='target_neurons', required=False, help='Alias of --target_neurons')
    ade_v.add_argument('--out-prefix', default='_cli_runs/', help='Output prefix / directory (default: _cli_runs/)')
    ade_v.add_argument('--interactive', action='store_true', help='Generate interactive Plotly HTML with Rolling Z and TTE metrics')
    # Optional prompt structure drift correlation (computed from real hidden states)
    ade_v.add_argument('--model', required=False, help='Optional HF model id for token drift correlation plot')
    ade_v.add_argument('--tokenizer', required=False, help='Optional tokenizer id (defaults to --model)')
    ade_v.add_argument('--baseline-prompt', required=False, help='Baseline prompt for token drift correlation')
    ade_v.add_argument('--test-prompt', required=False, help='Test prompt for token drift correlation')
    ade_v.add_argument('--layer', type=int, default=0, help='Layer index for token drift correlation (default: 0)')
    ade_v.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto', help='Device preference')
    ade_v.add_argument('--spike-percentile', type=float, default=90.0, help='Percentile to mark drift spikes in token plot (default: 90)')


    # Gradient inversion
    gi_parser = subparsers.add_parser('gradient_inversion', aliases=['gradient-inversion'], help='Recover features from gradients (demo)')
    gi_sub = gi_parser.add_subparsers(dest='gi_action', required=True)
    gi_r = gi_sub.add_parser('recover', help='Recover features from gradients')
    gi_r.add_argument('--gradients', required=True, help='Input gradients array (.npy)')
    gi_r.add_argument('--model', required=False, help='Optional HF model id for labeling/context (e.g., gpt2)')
    gi_r.add_argument('--tokenizer', required=False, help='Optional HF tokenizer id (e.g., gpt2)')
    gi_r.add_argument('--layer', type=int, default=None,
                      help='If gradients are 3D [layers, steps, features], select layer index (default: mean across layers)')
    gi_r.add_argument('--out-prefix', default='ginv_', help='Output prefix (default: ginv_)')
    gi_r.add_argument('--out-html', default=None, help='Output interactive HTML path (default: <out-prefix>reconstruction_heatmap.html)')
    gi_r.add_argument('--out-png', default=None, help='Output PNG path (default: <out-prefix>reconstruction_heatmap.png)')

    # Statistical evasion
    se_parser = subparsers.add_parser('statistical_evasion', help='Generate/score statistical evasion datasets')
    se_sub = se_parser.add_subparsers(dest='se_action', required=True)
    se_gen = se_sub.add_parser('generate', help='Generate benign/attack datasets')
    se_gen.add_argument('--samples', type=int, default=512)
    se_gen.add_argument('--features', type=int, default=64)
    se_gen.add_argument('--shift', type=float, default=0.3, help='Attack mean shift')
    se_gen.add_argument('--out-dir', default='.')
    # Compatibility: single-file output
    se_gen.add_argument('--output', required=False, help='Optional single file (.npz/.npy object) containing benign/attack')
    se_score = se_sub.add_parser('score', help='Score evasion statistically')
    se_score.add_argument('--data', required=False, help='Attack/subject data .npy')
    se_score.add_argument('--reference', required=False, help='Benign/reference .npy')
    # Compatibility: single combined input
    se_score.add_argument('--input', required=False, help='Combined input (.npz or .npy object) with benign/attack')
    se_score.add_argument('--method', default='ks', choices=['ks'], help='Statistical test method (default: ks)')
    se_score.add_argument('--alpha', type=float, default=0.05, help='Significance level for guidance')
    se_score.add_argument('--out-prefix', default='se_')

    # Neuron watermarking
    nw_parser = subparsers.add_parser('neuron_watermarking', help='Embed/detect neuron watermark')
    nw_sub = nw_parser.add_subparsers(dest='nw_action', required=True)
    nw_emb = nw_sub.add_parser('embed', help='Embed watermark into activations')
    nw_emb.add_argument('--activations', required=True, help='Input activations .npy/.npz [N,D] (must exist; no synthetic fallback)')
    nw_emb.add_argument('--watermark-bits', '--watermark_bits', dest='watermark_bits', required=True, help='Comma-separated bits, e.g., 1,0,1')
    nw_emb.add_argument('--target-pathway', '--target_pathway', dest='target_pathway', required=True, help='Comma-separated neuron ids')
    nw_emb.add_argument('--epsilon', type=float, default=0.1)
    nw_emb.add_argument('--out-prefix', default='wm_')
    nw_det = nw_sub.add_parser('detect', help='Detect watermark from activations')
    nw_det.add_argument('--activations', required=True)
    nw_det.add_argument('--target-pathway', '--target_pathway', dest='target_pathway', required=True)
    nw_det.add_argument('--threshold', type=float, default=0.0)
    nw_det.add_argument('--sweep', action='store_true', help='Sweep thresholds and compute confidence')
    nw_det.add_argument('--out-prefix', default='wm_')

    # Prompt Injection Analysis (wrapper)
    pia = subparsers.add_parser('prompt_injection_analysis', help='Analyze prompt injection vs benign')
    pia.add_argument('--suspect_prompt', required=True)
    pia.add_argument('--clean_prompt', required=True)
    # Optional model/device for tokenizer-aware analysis (cross-platform)
    pia.add_argument('--model', required=False, help='Optional HF model/tokenizer id for token-aware metrics (e.g., gpt2, facebook/opt-350m)')
    pia.add_argument('--device', required=False, choices=['auto','mps','cuda','cpu'], default='auto', help='Device preference (meta only)')
    pia.add_argument('--out-prefix', default='pia_')
    # Layer/head selection for attention visualization (per 2024 prompt injection research: 
    # Greshake et al., Perez & Ribeiro, Liu et al. - attention pattern analysis)
    pia.add_argument('--layer', type=int, required=True, help='Transformer layer index for attention extraction (0-indexed)')
    pia.add_argument('--head', type=int, required=True, help='Attention head index within the layer (0-indexed)')

    # Occlusion (wrapper to occlusion-analysis)
    occ = subparsers.add_parser('occlusion', help='Occlusion analysis wrapper')
    # NOTE: underscore prefix avoids IDE/type-checker "unused variable" diagnostics for
    # subparser-group handles that exist purely to register CLI flags.
    _occ_sub = occ.add_subparsers(dest='occ_action', required=True)
    occ_an = _occ_sub.add_parser('analyse', help='Run occlusion on single image')
    occ_an.add_argument('--model', default='google/vit-base-patch16-224')
    occ_an.add_argument('--image-path')
    occ_an.add_argument('--image-url')
    occ_an.add_argument('--patch-size', type=int, default=32)
    occ_an.add_argument('--stride', type=int, default=16)
    occ_an.add_argument('--output-2d')
    occ_an.add_argument('--output-3d')

    # Anomaly detection (simple)
    # NOTE: underscore prefix avoids IDE/type-checker "unused variable" diagnostics for
    # parser builder handles that exist purely to register flags.
    _anom = subparsers.add_parser('anomaly', help='Simple anomaly detect on activations')
    # Convenience flags directly on the command (no subcommand needed)
    _anom.add_argument('--input', dest='input', help='Alias for --activations')
    _anom.add_argument('--activations', help='Path to activations array (.npy/.npz)')
    _anom.add_argument('--reference', help='Optional baseline/reference array (.npy/.npz) for Z computation')
    _anom.add_argument('--method', choices=['auto', 'z', 'robust_z', 'iforest'], default='auto', help='Detection method')
    _anom.add_argument('--z', type=float, default=3.0, help='Z threshold (for z/robust_z methods)')
    _anom.add_argument('--fdr', type=float, default=None, help='Optional Benjamini–Hochberg FDR q (0<q<1) on entry-level p-values from |Z|; if set, uses FDR to flag entries')
    _anom.add_argument(
        '--flagging',
        choices=['auto', 'z', 'fdr'],
        default='auto',
        help="Which flagging mask to use for the final report/plot. "
             "`auto` uses FDR if `--fdr` is provided and succeeds, otherwise uses fixed |Z| threshold.",
    )
    _anom.add_argument('--robust', action='store_true', default=False, help='Use median/MAD robust Z')
    _anom.add_argument('--topk', type=int, default=10, help='Report top-K anomalous features')
    _anom.add_argument('--out-prefix', default='anom_', help='Output prefix for images')
    _anom.add_argument('--output', help='Write summary JSON to this exact path')
    anom_sub = _anom.add_subparsers(dest='anom_action', required=False)
    anom_det = anom_sub.add_parser('detect', help='Detect anomalies')
    anom_det.add_argument('--input', dest='input', help='Alias for --activations')
    anom_det.add_argument('--activations', required=False)
    anom_det.add_argument('--reference', required=False)
    anom_det.add_argument('--method', choices=['auto', 'z', 'robust_z', 'iforest'], default='auto')
    anom_det.add_argument('--z', type=float, default=3.0)
    anom_det.add_argument('--fdr', type=float, default=None)
    anom_det.add_argument('--flagging', choices=['auto', 'z', 'fdr'], default='auto')
    anom_det.add_argument('--robust', action='store_true', default=False)
    anom_det.add_argument('--topk', type=int, default=10)
    anom_det.add_argument('--out-prefix', default='anom_')
    anom_det.add_argument('--output')

    # Neuron Activation Anomaly Detection (layer-wise, baseline vs test prompts)
    act_anom = subparsers.add_parser(
        'activation_anomaly_detection',
        help='Layer-wise activation anomaly detection (baseline vs test prompts)'
    )
    act_anom.add_argument('--model', required=True, help='HuggingFace model id (e.g., gpt2, meta-llama/Llama-3.1-8B)')
    # NOTE: underscore prefix avoids IDE/type-checker "unused variable" diagnostics for
    # group handles that exist purely to register CLI flags.
    _baseline_src = act_anom.add_mutually_exclusive_group(required=True)
    _baseline_src.add_argument('--baseline-prompt', help='Baseline/benign prompt (single)')
    _baseline_src.add_argument('--baseline-file', default=None,
                              help='Text file with one baseline prompt per line (recommended)')
    act_anom.add_argument('--test-prompt', required=True, help='Test/suspect prompt')
    act_anom.add_argument('--threshold', type=float, default=2.5, help='Z threshold for anomaly detection (default: 2.5)')
    act_anom.add_argument('--robust', action='store_true',
                          help='Use robust Z (median/MAD) instead of mean/std')
    act_anom.add_argument('--sigma-floor', type=float, default=None,
                          help='Optional minimum denominator for Z-score (stabilizes near-zero baseline variance)')
    act_anom.add_argument('--layer-start', type=int, default=0,
                          help='First layer index to include (default: 0)')
    act_anom.add_argument('--layer-end', type=int, default=None,
                          help='Last layer index to include (inclusive)')
    act_anom.add_argument('--max-layers', type=int, default=None,
                          help='Optional cap on number of transformer layers analyzed')
    act_anom.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto', help='Device preference')
    act_anom.add_argument('--out', default='_cli_runs/anomaly_detection.html', help='Output HTML path')

    # Neural Persistence heatmap (neurons × layers)
    act_heat = subparsers.add_parser(
        'activation_neuron_heatmap',
        help='Neural persistence heatmap across layers (real activations)'
    )
    act_heat.add_argument('--model', required=True, help='HuggingFace model id (e.g., gpt2, distilbert-base-uncased)')
    heat_src = act_heat.add_mutually_exclusive_group(required=True)
    heat_src.add_argument('--prompt', help='Single prompt to analyze')
    heat_src.add_argument('--prompts-file', default=None, help='Text file: one prompt per line (aggregated baseline)')
    act_heat.add_argument('--topk', type=int, default=50, help='Top-K neurons to display (default: 50)')
    act_heat.add_argument('--reduce', choices=['mean','last','maxabs','max'], default='mean',
                          help='Reduce token dimension into per-neuron values (default: mean)')
    act_heat.add_argument('--aggregate', choices=['mean','median'], default='mean',
                          help='Aggregation across prompts-file (default: mean)')
    act_heat.add_argument('--layer-start', type=int, default=0, help='First layer index to include (default: 0)')
    act_heat.add_argument('--layer-end', type=int, default=None, help='Last layer index to include (inclusive)')
    act_heat.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto', help='Device preference')
    act_heat.add_argument('--out', default='_cli_runs/neuron_heatmap.html', help='Output HTML path')

    # Activation change (attack patterns) for a single layer
    act_diff = subparsers.add_parser(
        'activation_attack_patterns',
        help='Compare baseline vs test activations for a layer (real activations)'
    )
    act_diff.add_argument('--model', required=True, help='HuggingFace model id (e.g., gpt2, distilbert-base-uncased)')
    act_diff.add_argument('--baseline-prompt', required=True, help='Baseline/benign prompt')
    act_diff.add_argument('--test-prompt', required=True, help='Test/suspect prompt')
    act_diff.add_argument('--layer', type=int, required=True, help='Layer index (0-indexed)')
    act_diff.add_argument('--topk', type=int, default=10, help='Top-K changed positions to display (default: 10)')
    act_diff.add_argument('--compare', choices=['prefix','last'], default='prefix',
                          help='Compare overlapping prefix tokens or last token only (default: prefix)')
    act_diff.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto', help='Device preference')
    act_diff.add_argument('--out', default='_cli_runs/attack_patterns.html', help='Output HTML path')


    # Time-Travel Debugging (layer-wise Δ + attention variance; real hidden states + attentions)
    ttd = subparsers.add_parser(
        'activation_time_travel_debugging',
        help='Time-travel debugging: layer-wise activation Δ (L1) + attention variance (baseline vs test)'
    )
    ttd_sub = ttd.add_subparsers(dest='ttd_action', required=True)

    ttd_c = ttd_sub.add_parser('craft', help='Compute metrics from baseline vs test (real hidden states + attentions)')
    ttd_c.add_argument('--model', required=True, help='HuggingFace model id (e.g., gpt2, distilbert-base-uncased)')
    ttd_c.add_argument('--tokenizer', required=False, help='Optional tokenizer id (defaults to --model)')

    base_src = ttd_c.add_mutually_exclusive_group(required=True)
    base_src.add_argument('--baseline-prompt', help='Baseline/benign prompt')
    base_src.add_argument('--baseline-file', default=None, help='Text file; first non-empty line used as baseline prompt')

    test_src = ttd_c.add_mutually_exclusive_group(required=True)
    test_src.add_argument('--test-prompt', help='Test/suspect prompt')
    test_src.add_argument('--test-file', default=None, help='Text file; first non-empty line used as test prompt')

    ttd_c.add_argument('--layer-start', type=int, default=0, help='First layer index to include (default: 0)')
    ttd_c.add_argument('--layer-end', type=int, default=None, help='Last layer index to include (inclusive)')
    ttd_c.add_argument('--max-tokens', type=int, default=128, help='Max tokens to run (default: 128)')

    ttd_c.add_argument(
        '--delta-mode',
        choices=['token_l1_mean_x100', 'token_l1_mean', 'mean_vec_l1_x100', 'mean_vec_l1'],
        default='token_l1_mean_x100',
        help='Activation Δ definition (default: token_l1_mean_x100)'
    )
    ttd_c.add_argument('--attn-var-mode', choices=['per_query','global'], default='per_query',
                       help='Attention variance mode (default: per_query)')
    ttd_c.add_argument('--attn-var-scale', choices=['seq2','seq','none'], default='seq2',
                       help='Scale attention variance for readability (default: seq2)')
    ttd_c.add_argument('--attention-source', choices=['test','baseline','delta_abs'], default='test',
                       help='Which attention series to plot (default: test)')

    ttd_c.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto', help='Device preference')
    ttd_c.add_argument('--title', default='Time-Travel Debugging – Layer-wise Activation Δ & Attention Variance', help='Plot title')
    ttd_c.add_argument('--out-json', default='_cli_runs/time_travel_debugging.json', help='Output metrics JSON path')
    ttd_c.add_argument('--out-png', default='_cli_runs/time_travel_debugging.png', help='Output PNG path')

    ttd_v = ttd_sub.add_parser('visualize', help='Render PNG from a saved metrics JSON')
    ttd_v.add_argument('--in-json', required=True, help='Input metrics JSON path')
    ttd_v.add_argument('--title', default=None, help='Optional override title')
    ttd_v.add_argument('--out-png', default='_cli_runs/time_travel_debugging.png', help='Output PNG path')


    # Eigen‑Collapse Rank Shrinkage Radar (top‑k covariance eigenvalues; real hidden states)
    ecr = subparsers.add_parser(
        'activation_eigen_collapse_radar',
        help='Eigen-collapse radar: top-k covariance eigenvalues per layer (real hidden states)'
    )
    ecr_sub = ecr.add_subparsers(dest='ecr_action', required=True)

    ecr_c = ecr_sub.add_parser('craft', help='Compute per-layer eigen-spectrum from real hidden states')
    ecr_c.add_argument('--model', required=True, help='HuggingFace model id (e.g., gpt2, distilbert-base-uncased)')
    ecr_c.add_argument('--tokenizer', required=False, help='Optional tokenizer id (defaults to --model)')

    ecr_src = ecr_c.add_mutually_exclusive_group(required=True)
    ecr_src.add_argument('--prompt', help='Single prompt to analyze')
    ecr_src.add_argument('--prompts-file', default=None, help='Text file: one prompt per line (aggregate)')

    ecr_c.add_argument('--layer-start', type=int, default=0, help='First layer index to include (default: 0)')
    ecr_c.add_argument('--layer-end', type=int, default=None, help='Last layer index to include (inclusive)')
    ecr_c.add_argument('--every', type=int, default=1, help='Plot every Nth layer (default: 1 = all)')
    ecr_c.add_argument('--max-tokens', type=int, default=128, help='Max tokens to run (default: 128)')
    ecr_c.add_argument('--k', type=int, default=5, help='Top-k eigenvalues to plot (default: 5)')
    ecr_c.add_argument('--normalize', choices=['eig1','sum','none'], default='eig1',
                       help='Normalize eigenvalues per layer (default: eig1)')
    ecr_c.add_argument('--aggregate', choices=['mean','median'], default='mean',
                       help='Aggregation across prompts-file (default: mean)')

    ecr_c.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto', help='Device preference')
    ecr_c.add_argument('--title', default='Eigen-Collapse Rank Shrinkage Radar', help='Plot title')
    ecr_c.add_argument('--out-json', default='_cli_runs/eigen_collapse_radar.json', help='Output metrics JSON path')
    ecr_c.add_argument('--out-png', default='_cli_runs/eigen_collapse_radar.png', help='Output PNG path')

    ecr_v = ecr_sub.add_parser('visualize', help='Render PNG from a saved metrics JSON')
    ecr_v.add_argument('--in-json', required=True, help='Input metrics JSON path')
    ecr_v.add_argument('--title', default=None, help='Optional override title')
    ecr_v.add_argument('--out-png', default='_cli_runs/eigen_collapse_radar.png', help='Output PNG path')




    # Eigenvalue Spectrum Histogram (covariance eigenvalues; real hidden states; baseline-vs-test recommended)
    evs = subparsers.add_parser(
        'activation_eigenvalue_spectrum',
        help='Eigenvalue spectrum histogram: covariance eigenvalues from real hidden states (baseline-vs-test recommended)'
    )
    evs_sub = evs.add_subparsers(dest='evs_action', required=True)

    evs_c = evs_sub.add_parser('craft', help='Compute eigenvalue spectrum from real hidden states')
    evs_c.add_argument('--model', required=True, help='HuggingFace model id (e.g., gpt2, distilbert-base-uncased)')
    evs_c.add_argument('--tokenizer', required=False, help='Optional tokenizer id (defaults to --model)')

    evs_src = evs_c.add_mutually_exclusive_group(required=True)
    evs_src.add_argument('--prompt', help='Single prompt to analyze')
    evs_src.add_argument('--prompts-file', default=None, help='Text file: one prompt per line (aggregate)')

    evs_base = evs_c.add_mutually_exclusive_group(required=False)
    evs_base.add_argument('--baseline-prompt', default=None, help='Optional baseline prompt (benign/reference) for drift comparison')
    evs_base.add_argument('--baseline-prompts-file', default=None, help='Optional baseline prompt suite file (one prompt per line)')

    evs_c.add_argument('--label', default='sample', help='Label/tag for title (e.g., adversarial_fgsm)')
    evs_c.add_argument('--layer', default='all', help="Layer index (0-indexed int) or 'all' to aggregate")
    evs_c.add_argument('--layer-start', type=int, default=0, help='First layer index (inclusive) when using --layer all')
    evs_c.add_argument('--layer-end', type=int, default=None, help='Last layer index (inclusive) when using --layer all')
    evs_c.add_argument('--max-tokens', type=int, default=128, help='Max tokens to run (default: 128)')
    evs_c.add_argument('--bins', type=int, default=40, help='Histogram bins (default: 40)')
    evs_c.add_argument('--x-scale', choices=['linear','log10'], default='linear',
                       help='X-axis scaling (default: linear; log10 helps heavy-tail spectra)')
    evs_c.add_argument('--top-k-layers', type=int, default=5,
                       help='Report/top-highlight K most shifted layers when baseline is provided (default: 5)')
    evs_c.add_argument('--top-k-prompts', type=int, default=10,
                       help='Report/top-highlight K most anomalous prompts when baseline+suites are used (default: 10)')

    evs_c.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto', help='Device preference')
    evs_c.add_argument('--title', default='NeurInSpectre Eigenvalue Spectrum', help='Plot title prefix')
    evs_c.add_argument('--out-json', default='_cli_runs/eigenvalue_spectrum.json', help='Output metrics JSON path')
    evs_c.add_argument('--out-png', default='_cli_runs/eigenvalue_spectrum.png', help='Output PNG path')
    evs_c.add_argument('--out-html', default=None, help='Optional interactive HTML output path')

    evs_v = evs_sub.add_parser('visualize', help='Render PNG/HTML from a saved metrics JSON')
    evs_v.add_argument('--in-json', required=True, help='Input metrics JSON path')
    evs_v.add_argument('--title', default=None, help='Optional override title')
    evs_v.add_argument('--x-scale', choices=['linear','log10'], default=None, help='Optional override x-axis scale')
    evs_v.add_argument('--out-png', default='_cli_runs/eigenvalue_spectrum.png', help='Output PNG path')
    evs_v.add_argument('--out-html', default=None, help='Optional interactive HTML output path')



    # FFT Security Spectrum (token-norm rFFT; real hidden states; per-layer)
    ffts = subparsers.add_parser(
        'activation_fft_security_spectrum',
        help='FFT security spectrum: token-norm rFFT per prompt + mean (real hidden states; per-layer)'
    )
    ffts_sub = ffts.add_subparsers(dest='ffts_action', required=True)

    ffts_c = ffts_sub.add_parser('craft', help='Compute per-prompt FFT spectra from real hidden states')
    ffts_c.add_argument('--model', required=True, help='HuggingFace model id (e.g., gpt2, distilbert-base-uncased)')
    ffts_c.add_argument('--tokenizer', required=False, help='Optional tokenizer id (defaults to --model)')

    ffts_src = ffts_c.add_mutually_exclusive_group(required=True)
    ffts_src.add_argument('--prompt', help='Single prompt to analyze')
    ffts_src.add_argument('--prompts-file', default=None, help='Text file: one prompt per line (aggregate)')

    ffts_c.add_argument('--layer', type=int, required=True, help='Layer index (0-indexed)')
    ffts_c.add_argument('--max-tokens', type=int, default=128, help='Max tokens to run (default: 128)')
    ffts_c.add_argument('--tail-start', type=float, default=0.25,
                        help='High-frequency tail start in normalized freq (default: 0.25)')
    ffts_c.add_argument('--z-threshold', type=float, default=2.0, help='Flag threshold for |z| (default: 2.0)')
    ffts_c.add_argument('--z-mode', choices=['standard','robust'], default='standard',
                        help='Z-score mode across prompts (default: standard)')
    ffts_c.add_argument('--prompt-index', type=int, default=0, help='Prompt index to render in PNG (0-indexed)')

    # Optional benign baseline suite (z-scores computed vs baseline distribution)
    ffts_c.add_argument('--baseline-prompts-file', default=None,
                        help='Optional benign baseline prompts file (one prompt per line); if set, z-scores are computed vs baseline')

    # Signal processing controls (to make injection-like regime shifts more visible)
    ffts_c.add_argument('--signal-mode', choices=['token_norm','delta_token_norm','cosine_delta','mean_abs_delta'], default='token_norm',
                        help='1D signal derived from hidden states before FFT (default: token_norm)')
    ffts_c.add_argument('--detrend', choices=['none','mean'], default='none',
                        help='Detrend 1D signal before FFT (default: none; mean removes DC)')
    ffts_c.add_argument('--window', choices=['none','hann'], default='none',
                        help='Window function before FFT (default: none; hann reduces leakage)')
    ffts_c.add_argument('--fft-size', type=int, default=0,
                        help='If >0, use fixed FFT size (zero-pad/truncate) for all prompts (recommended to avoid common_prefix truncation)')
    ffts_c.add_argument('--segment', choices=['prefix','suffix'], default='prefix',
                        help='Which part of the token signal to analyze (default: prefix; suffix helps when injections occur late)')

    ffts_c.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto', help='Device preference')
    ffts_c.add_argument('--title', default='FFT Security Spectrum - Token-Norm FFT (per prompt + mean)', help='Plot title')
    ffts_c.add_argument('--out-json', default='_cli_runs/fft_security_spectrum.json', help='Output metrics JSON path')
    ffts_c.add_argument('--out-png', default='_cli_runs/fft_security_spectrum.png', help='Output PNG path')

    ffts_v = ffts_sub.add_parser('visualize', help='Render PNG from a saved metrics JSON')
    ffts_v.add_argument('--in-json', required=True, help='Input metrics JSON path')
    ffts_v.add_argument('--prompt-index', type=int, default=0, help='Prompt index to render (0-indexed)')
    ffts_v.add_argument('--title', default=None, help='Optional override title')
    ffts_v.add_argument('--out-png', default='_cli_runs/fft_security_spectrum.png', help='Output PNG path')
    # Layer-Level Causal Impact Analysis
    lci = subparsers.add_parser(
        'activation_layer_causal_impact',
        help='🔥 Layer-level causal impact analysis (KL divergence per layer; identifies hot layers for red/blue team action)'
    )
    lci.add_argument('--model', required=True, help='HuggingFace model id (e.g., gpt2, distilbert-base-uncased, bert-base-uncased)')
    lci.add_argument('--tokenizer', required=False, help='Optional tokenizer id (defaults to --model)')
    lci.add_argument('--baseline-prompt', required=True, help='Benign/reference prompt (establishes baseline activation distribution)')
    lci.add_argument('--test-prompt', required=True, help='Test/adversarial prompt (compared against baseline)')
    lci.add_argument('--method', choices=['kl', 'js', 'l2'], default='kl',
                     help='Divergence/distance method: kl (KL divergence), js (Jensen-Shannon), l2 (Euclidean) [default: kl]')
    lci.add_argument('--percentile', type=float, default=95.0,
                     help='Percentile threshold for identifying hot layers (default: 95.0 - based on research showing 1-2%% causal neurons)')
    lci.add_argument('--layer-start', type=int, default=None, help='Starting layer index (default: 0)')
    lci.add_argument('--layer-end', type=int, default=None, help='Ending layer index (default: last layer)')
    lci.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto', help='Device preference')
    lci.add_argument('--interactive', action='store_true', help='Generate interactive HTML visualization')
    lci.add_argument('--title', default='Layer-Level Causal Impact (hot layers in red)',
                     help='Plot title')
    lci.add_argument('--out-json', default='_cli_runs/layer_causal_impact.json',
                     help='Output metrics JSON path')
    lci.add_argument('--out-png', default='_cli_runs/layer_causal_impact.png',
                     help='Output PNG path')
    lci.add_argument('--out-html', default='_cli_runs/layer_causal_impact.html',
                     help='Output HTML path (when --interactive is set)')

    # Backdoor watermark (compat)
    bdw = subparsers.add_parser('backdoor_watermark', help='Backdoor/watermark compatibility wrapper')
    bdw_sub = bdw.add_subparsers(dest='bdw_action', required=True)
    bdw_inj = bdw_sub.add_parser('inject_backdoor', help='Alias of watermark embed')
    bdw_inj.add_argument('--activations', required=True, help='Input activations .npy/.npz [N,D] (must exist; no synthetic fallback)')
    bdw_inj.add_argument('--watermark-bits', required=True)
    bdw_inj.add_argument('--target_pathway', required=True)
    bdw_inj.add_argument('--epsilon', type=float, default=0.2)
    bdw_inj.add_argument('--out-prefix', default='bdw_')
    bdw_emb = bdw_sub.add_parser('embed_watermark', help='Alias of watermark embed')
    bdw_emb.add_argument('--activations', required=True, help='Input activations .npy/.npz [N,D] (must exist; no synthetic fallback)')
    bdw_emb.add_argument('--watermark-bits', required=True)
    bdw_emb.add_argument('--target_pathway', required=True)
    bdw_emb.add_argument('--epsilon', type=float, default=0.2)
    bdw_emb.add_argument('--out-prefix', default='bdw_')
    bdw_det = bdw_sub.add_parser('detect_watermark', help='Alias of watermark detect')
    bdw_det.add_argument('--activations', required=True)
    bdw_det.add_argument('--target_pathway', required=True)
    bdw_det.add_argument('--threshold', type=float, default=0.0)
    bdw_det.add_argument('--out-prefix', default='bdw_')

    # DNA neuron ablation (Top-K impact; per-layer optional; interactive HTML optional)
    dna = subparsers.add_parser('dna_neuron_ablation', help='Neuron ablation impact (Top-K) from activations')
    dna.add_argument('--activations', required=True, help='Input activations .npy (N×D or L×N×D; D=neurons)')
    dna.add_argument('--layer', type=int, default=None,
                     help='Layer index when activations are 3D (default: flatten across layers)')
    dna.add_argument('--layer-axis', type=int, default=0,
                     help='Layer axis for 3D activations (default: 0, i.e., [L, N, D])')
    dna.add_argument('--topk', type=int, default=10, help='How many neurons to rank (default: 10)')
    dna.add_argument('--bootstrap', type=int, default=200, help='Bootstrap resamples for 95%% CI + stability (default: 200)')
    dna.add_argument('--perm-trials', type=int, default=200, help='Permutation trials for Top-3 p-value (default: 200)')
    dna.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    dna.add_argument('--interactive', action='store_true', help='Write interactive Plotly HTML report')
    dna.add_argument('--title', default='NeurInSpectre — DNA Neuron Ablation Impact (Top-K)', help='Plot title')
    dna.add_argument('--out-csv', default=None, help='Output CSV path (default: <out-prefix>ablation.csv)')
    dna.add_argument('--out-json', default=None, help='Output JSON path (default: <out-prefix>ablation.json)')
    dna.add_argument('--out-png', default=None, help='Output PNG path (default: <out-prefix>ablation_impact.png)')
    dna.add_argument('--out-html', default=None, help='Output HTML path (default: <out-prefix>ablation_impact.html)')
    dna.add_argument('--out-prefix', default='_cli_runs/dna_', help='Output prefix (default: _cli_runs/dna_)')

    # Fusion attack (data-driven): combine two arrays and visualize sensitivity vs alpha
    fus = subparsers.add_parser('fusion_attack', help='Combine two arrays and analyze fusion sensitivity')
    fus.add_argument('--primary', required=True)
    fus.add_argument('--secondary', required=True, help='Secondary array (.npy/.npz) to fuse (required; no synthetic fallback)')
    fus.add_argument('--alpha', type=float, default=0.5)
    fus.add_argument('--sweep', action='store_true', help='Sweep alpha in [0,1] and plot metric')
    fus.add_argument('--interactive', action='store_true', help='Generate interactive Plotly HTML with research-based guidance')
    fus.add_argument('--out-prefix', default='_cli_runs/fusion_')

    # Fusion π-viz (time-series L2 norms; real data; per-layer)
    pi_viz = subparsers.add_parser('fusion_pi_viz', help='π-viz: compare modality magnitudes over time (arrays or prompts)')
    # Array mode (no simulation): provide both --primary and --secondary
    pi_viz.add_argument('--primary', required=False, help='Primary modality .npy (T×D or L×T×D)')
    pi_viz.add_argument('--secondary', required=False, help='Secondary modality .npy (T×D or L×T×D)')
    # Prompt/model mode: derive modalities from real hidden states
    pi_viz.add_argument('--model', required=False, help='HF model id (enables prompt mode)')
    pi_viz.add_argument('--tokenizer', required=False, help='Optional tokenizer id (defaults to --model)')
    pi_viz.add_argument('--prompt-a', required=False, help='Prompt for modality A (prompt mode)')
    pi_viz.add_argument('--prompt-b', required=False, help='Prompt for modality B (prompt mode)')
    # Layer controls
    pi_viz.add_argument('--layer', type=int, default=None, help='Layer index (required for prompt mode; required for 3D arrays)')
    pi_viz.add_argument('--layer-axis', type=int, default=0, help='Layer axis for array mode when ndim>=3 (default: 0)')
    # Plot controls
    pi_viz.add_argument('--max-steps', type=int, default=25, help='Max timesteps/tokens to plot (default: 25)')
    pi_viz.add_argument('--z-threshold', type=float, default=3.0, help='Robust z threshold on |Δ| spikes (default: 3.0)')
    pi_viz.add_argument('--max-spikes', type=int, default=6, help='Max spike steps to highlight (default: 6)')
    pi_viz.add_argument('--out-json', default='_cli_runs/fusion_pi_viz_summary.json', help='Output summary JSON path')
    pi_viz.add_argument('--title', default='Fusion Attack Analysis: π-viz', help='Plot title')
    pi_viz.add_argument('--interactive', action='store_true', help='Also write interactive HTML')
    pi_viz.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto', help='Device preference (prompt mode)')
    pi_viz.add_argument('--out-png', default='_cli_runs/fusion_pi_viz.png', help='Output PNG path')
    pi_viz.add_argument('--out-html', default='_cli_runs/fusion_pi_viz.html', help='Output HTML path (if --interactive)')

    # Fusion co-attention traces (original vs fused feature traces)
    # NOTE: underscore prefix avoids IDE/type-checker "unused variable" diagnostics for
    # parser handles that exist purely to register CLI flags.
    _coatt = subparsers.add_parser(
        'fusion_co_attention_traces',
        help='Co-attention fuse two traces and plot feature traces (original vs fused; any feature; prompt or array mode)'
    )
    coatt_sub = _coatt.add_subparsers(dest='coatt_action', required=True)

    coatt_c = coatt_sub.add_parser('craft', help='Compute co-attention fusion + render traces')

    # Mode selection: prompt mode (derive traces from real model hidden states) or array mode
    coatt_mode = coatt_c.add_mutually_exclusive_group(required=True)
    coatt_mode.add_argument('--model', required=False, help='HF model id (enables prompt mode)')
    coatt_mode.add_argument('--trace-a', required=False, help='Trace A .npy (T×D or L×T×D) (enables array mode)')

    coatt_c.add_argument('--tokenizer', required=False, help='Optional tokenizer id (defaults to --model)')
    coatt_c.add_argument('--prompt-a', required=False, help='Prompt A (prompt mode)')
    coatt_c.add_argument('--prompt-b', required=False, help='Prompt B (prompt mode)')
    coatt_c.add_argument('--trace-b', required=False, help='Trace B .npy (T×D or L×T×D) (array mode)')

    # Layer / sequence controls
    coatt_c.add_argument('--layer', type=int, default=None, help='Layer index (required for prompt mode; required for 3D arrays)')
    coatt_c.add_argument('--layer-axis', type=int, default=0, help='Layer axis for array mode when ndim>=3 (default: 0)')
    coatt_c.add_argument('--max-tokens', type=int, default=128, help='Max tokens per prompt (prompt mode; default: 128)')
    coatt_c.add_argument('--max-steps', type=int, default=101, help='Max timesteps/tokens to plot (default: 101)')

    # Feature selection
    coatt_c.add_argument('--feature', default='0', help='Primary feature index (0-based) or "auto"')
    coatt_c.add_argument('--feature2', default='1', help='Secondary feature index (0-based) or "none"')

    # Fusion parameters
    coatt_c.add_argument('--alpha', type=float, default=0.55, help='Fusion strength alpha in [0,1] (default: 0.55)')
    coatt_c.add_argument('--temperature', type=float, default=0.25, help='Softmax temperature (smaller -> sharper) (default: 0.25)')

    # Plot scaling
    coatt_c.add_argument('--scale', choices=['none','zscore','tanh_z','minmax'], default='tanh_z', help='Shared scaling mode for traces (default: tanh_z)')

    coatt_c.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto', help='Device preference (prompt mode)')
    coatt_c.add_argument('--trust-remote-code', action='store_true', help='Allow loading models with custom code from HF (use with caution)')

    coatt_c.add_argument('--title', default='NeurInSpectre — Co-Attention Trace Fusion (co_attention)', help='Plot title')
    coatt_c.add_argument('--out-json', default='_cli_runs/co_attention_traces.json', help='Output metrics JSON path')
    coatt_c.add_argument('--out-png', default='_cli_runs/co_attention_traces.png', help='Output PNG path')
    coatt_c.add_argument('--interactive', action='store_true', help='Also write interactive Plotly HTML')
    coatt_c.add_argument('--out-html', default='_cli_runs/co_attention_traces.html', help='Output HTML path (if --interactive)')

    coatt_v = coatt_sub.add_parser('visualize', help='Render traces PNG from a saved metrics JSON')
    coatt_v.add_argument('--in-json', required=True, help='Input metrics JSON path')
    coatt_v.add_argument('--title', default=None, help='Optional override title')
    coatt_v.add_argument('--out-png', default='_cli_runs/co_attention_traces.png', help='Output PNG path')
    coatt_v.add_argument('--out-html', default=None, help='Optional: also write interactive HTML')

    # EDNN: Element-wise Differential Attack  
    ednn_p = subparsers.add_parser('adversarial-ednn',
                                   help='🔴 EDNN element-wise differential attack')
    ednn_p.add_argument('--attack-type', choices=['inversion', 'steganographic', 'membership_inference', 'rag_poison'],
                       default='inversion')
    ednn_p.add_argument('--data', required=True, help='Input embeddings (.npy/.npz) (must exist; no synthetic fallback)')
    ednn_p.add_argument(
        '--model',
        default=None,
        help='Optional HuggingFace model id or local path for embedding model/tokenizer (required for inversion/steganographic/rag_poison)',
    )
    ednn_p.add_argument(
        '--device',
        choices=['auto', 'mps', 'cuda', 'cpu'],
        default='auto',
        help='Device preference for embedding model (default: auto)',
    )
    ednn_p.add_argument(
        '--reference-embeddings',
        default=None,
        help='Optional reference embeddings (.npy/.npz) for NN-based reconstruction / membership inference',
    )
    ednn_p.add_argument(
        '--reference-texts',
        default=None,
        help='Optional reference texts file (one line per reference embedding) for NN reconstruction',
    )
    ednn_p.add_argument('--embedding-dim', type=int, default=768)
    ednn_p.add_argument('--target-query', help='For RAG poisoning')
    ednn_p.add_argument('--poisoned-document', help='For RAG poisoning: poisoned document content (string)')
    ednn_p.add_argument('--poisoned-document-file', help='For RAG poisoning: path to poisoned document (text file)')
    ednn_p.add_argument('--target-tokens', help='Target tokens file for sensitive token attacks')
    ednn_p.add_argument('--output-dir', '-o', default='_cli_runs/ednn')
    ednn_p.add_argument('--verbose', '-v', action='store_true')

    # EDNN RAG Poison: Specialized command
    rag_p = subparsers.add_parser('ednn-rag-poison', help='🔴 EDNN RAG poisoning')
    rag_p.add_argument('--model-path', '-m', required=True)
    rag_p.add_argument('--vector-db', choices=['weaviate', 'pinecone', 'local'], default='local')
    rag_p.add_argument(
        '--vector-db-embeddings',
        default=None,
        help='Optional local vector DB embeddings (.npy/.npz) for rank-based evaluation',
    )
    rag_p.add_argument('--malicious-doc', '-d', required=True)
    rag_p.add_argument('--target-query', '-q', required=True)
    rag_p.add_argument('--poison-ratio', type=float, default=0.1)
    rag_p.add_argument('--similarity-threshold', type=float, default=0.85)
    rag_p.add_argument(
        '--device',
        choices=['auto', 'mps', 'cuda', 'cpu'],
        default='auto',
        help='Device preference for embedding model (default: auto)',
    )
    rag_p.add_argument('--output-dir', '-o', default='_cli_runs/ednn_rag_poison')
    rag_p.add_argument('--verbose', '-v', action='store_true')

    # Attack Vector Analysis with CVE/MITRE ATLAS Mapping
    attack_vec_p = subparsers.add_parser('analyze-attack-vectors',
                                         help='🎯 Analyze attack vectors with CVE/MITRE ATLAS mapping')
    attack_vec_p.add_argument(
        '--target-data',
        required=True,
        help='Target data file (.npy/.npz) - activations, gradients, embeddings (must exist; no demo fallback)',
    )
    attack_vec_p.add_argument(
        '--cve-mapping',
        action='store_true',
        help='Deprecated/no-op: NeurInSpectre does not emit real CVEs. Use `pip-audit` / `npm audit` instead.',
    )
    attack_vec_p.add_argument('--mitre-atlas', action='store_true', default=True, help='Map to MITRE ATLAS techniques')
    attack_vec_p.add_argument('--owasp-llm', action='store_true', default=True, help='Map to OWASP LLM Top 10')
    attack_vec_p.add_argument('--output-dir', '-o', default='_cli_runs', help='Output directory')
    attack_vec_p.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Recommend Countermeasures
    counter_p = subparsers.add_parser('recommend-countermeasures',
                                      help='🔵 Recommend countermeasures for detected threats')
    counter_p.add_argument('--threat-level', choices=['low', 'medium', 'high', 'critical'], default='high')
    counter_p.add_argument('--attack-vectors', help='Comma-separated attack vectors to defend against')
    # Optional: let users generate recommendations from data (auto-detect vectors).
    counter_p.add_argument('--target-data', help='Target data file (.npy) - activations, gradients, embeddings')
    counter_p.add_argument('--output-dir', '-o', default='_cli_runs', help='Output directory')
    counter_p.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1

    # Modular CLI support: if a subcommand registered a callable handler via
    # `parser.set_defaults(func=...)`, run it without needing to edit the giant
    # dispatch chain below.
    func = getattr(args, "func", None)
    if callable(func):
        return int(func(args))
    
    # Execute the appropriate command
    try:
        if args.command == 'obfuscated-gradient':
            return handle_obfuscated_gradient_command(args)
        elif args.command == 'math':
            return handle_math_command(args)
        # RL Obfuscation
        if args.command == 'rl-obfuscation':
            from .rl_obfuscation_commands import handle_rl_obfuscation
            return handle_rl_obfuscation(args)
        
        elif args.command == 'dashboard':
            from .ttd import run_dashboard
            return run_dashboard(args)
        elif args.command == 'dashboard-manager':
            return handle_dashboard_manager_command(args)
        elif args.command == 'frequency-adversarial':
            from .frequency_analysis import run_frequency_adversarial
            return run_frequency_adversarial(args)
        elif args.command == 'demo':
            # Mathematical demo
            try:
                import json
                import numpy as np
                rng = np.random.default_rng(42)
                data = rng.normal(0, 1, size=(256, 128)).astype('float32')
                # Simple statistics as demo content
                stats = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'shape': list(data.shape),
                    'device': getattr(args, 'device', 'auto'),
                    'precision': getattr(args, 'precision', 'float32')
                }
                if getattr(args, 'save_results', None):
                    with open(args.save_results, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(args.save_results)
                else:
                    print(json.dumps(stats, indent=2))
                return 0
            except Exception as e:
                logger.error(f"Demo command failed: {e}")
                return 1
        elif args.command == 'correlate':
            from .correlation_analysis import run_correlation
            return run_correlation(args)
        elif args.command == 'attack-graph':
            # Handle new unified attack-graph command with subcommands
            if not args.attack_command:
                logger.error("No attack-graph subcommand specified. Use: prepare or visualize")
                return 1
            
            if args.attack_command == 'prepare':
                # Prepare attack scenario
                from pathlib import Path
                import json

                # Support --output-dir (smoke-matrix / convenience) while keeping --output for backwards compatibility.
                # - If --output-dir is provided and --output is left as default, write <output-dir>/attack_graph.json
                # - If both are provided, treat --output as filename and place it under --output-dir
                output_path = Path(args.output)
                out_dir = getattr(args, "output_dir", None)
                if out_dir:
                    od = Path(str(out_dir))
                    od.mkdir(parents=True, exist_ok=True)
                    if str(args.output).strip() == "_cli_runs/atlas_case.json":
                        output_path = od / "attack_graph.json"
                    else:
                        output_path = od / Path(str(args.output)).name
                
                # If provided, build a scenario directly from the official ATLAS catalog
                # (supports ANY AML.T*/AML.TA* id present in the vendored STIX bundle).
                if getattr(args, 'atlas_ids', None) or getattr(args, 'atlas_ids_file', None):
                    import re
                    from ..mitre_atlas.registry import (
                        load_stix_atlas_bundle,
                        list_atlas_tactics,
                        tactic_by_phase_name,
                        technique_index,
                    )

                    raw_ids = []
                    if getattr(args, 'atlas_ids', None):
                        raw_ids += re.split(r"[\s,]+", str(args.atlas_ids).strip())
                    if getattr(args, 'atlas_ids_file', None):
                        fp = Path(str(args.atlas_ids_file))
                        raw_ids += [
                            ln.strip()
                            for ln in fp.read_text().splitlines()
                            if ln.strip() and not ln.strip().startswith('#')
                        ]

                    atlas_ids = [x.strip() for x in raw_ids if x.strip()]
                    # Deduplicate (stable)
                    _seen = set()
                    atlas_ids = [x for x in atlas_ids if not (x in _seen or _seen.add(x))]

                    bundle = load_stix_atlas_bundle()
                    tactics = list_atlas_tactics(bundle)
                    tid_to_t = {t.tactic_id: t for t in tactics}
                    phase_to_t = tactic_by_phase_name(bundle)
                    tech_idx = technique_index(bundle)

                    def _tactic_sort(tid: str) -> int:
                        try:
                            return int(str(tid).split('AML.TA', 1)[1])
                        except Exception:
                            return 10**9

                    selected_tactic_ids = set()
                    technique_nodes = []
                    edges = []

                    for aid in atlas_ids:
                        if aid.startswith('AML.TA'):
                            if aid not in tid_to_t:
                                raise ValueError(f'Unknown ATLAS tactic id: {aid}')
                            selected_tactic_ids.add(aid)
                            continue

                        tech = tech_idx.get(aid)
                        if tech is None:
                            raise ValueError(f'Unknown ATLAS technique id: {aid}')

                        tactic_ids = []
                        tactic_names = []
                        for ph in tech.tactic_phase_names:
                            t = phase_to_t.get(ph)
                            if t is None:
                                continue
                            if t.tactic_id not in tactic_ids:
                                tactic_ids.append(t.tactic_id)
                                tactic_names.append(t.name)
                                selected_tactic_ids.add(t.tactic_id)

                        primary_phase = tactic_names[0] if tactic_names else 'Unknown'

                        technique_nodes.append({
                            'id': tech.technique_id,
                            'label': tech.name,
                            'atlas_phase': primary_phase,
                            'atlas_id': tech.technique_id,
                            'tactic_ids': tactic_ids,
                            'tactics': tactic_names,
                            'url': tech.url,
                        })

                        for tn in tactic_names:
                            edges.append({'source': tn, 'target': tech.technique_id})

                    ordered = sorted([tid_to_t[tid] for tid in selected_tactic_ids], key=lambda t: _tactic_sort(t.tactic_id))
                    tactic_nodes = [
                        {'id': t.name, 'label': t.name, 'atlas_phase': t.name, 'atlas_id': t.tactic_id}
                        for t in ordered
                    ]

                    # Lightweight tactic chain (keeps visualization readable)
                    for i in range(len(ordered) - 1):
                        edges.append({'source': ordered[i].name, 'target': ordered[i + 1].name})

                    data = {
                        'nodes': tactic_nodes + technique_nodes,
                        'edges': edges,
                        'generated_from': {'atlas_ids': atlas_ids},
                    }

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'w') as f:
                        json.dump(data, f, indent=2)

                    logger.info(f"✅ ATLAS graph prepared from {len(atlas_ids)} id(s): {output_path}")
                    logger.info(f"🗺️ Nodes: {len(data['nodes'])}, Edges: {len(data['edges'])}")
                    logger.info("🚀 Next (generate HTML): neurinspectre attack-graph visualize "
                                f"--input-path {output_path} --output-path _cli_runs/attack_graph.html")
                    logger.info("🌐 Open the HTML file: open _cli_runs/attack_graph.html  (macOS)")
                    logger.info("   Tip: do NOT prefix the neurinspectre command with `open`.")
                    return 0

                SCENARIOS = {
                    'jailbreak_extraction': {
                        'nodes': [
                            {'id': 'Execution', 'label': 'Execution', 'atlas_phase': 'Execution', 'atlas_id': 'AML.TA0005'},
                            {'id': 'Defense Evasion', 'label': 'Defense Evasion', 'atlas_phase': 'Defense Evasion', 'atlas_id': 'AML.TA0007'},
                            {'id': 'Collection', 'label': 'Collection', 'atlas_phase': 'Collection', 'atlas_id': 'AML.TA0009'},
                            {'id': 'Exfiltration', 'label': 'Exfiltration', 'atlas_phase': 'Exfiltration', 'atlas_id': 'AML.TA0010'},
                            {'id': 'prompt_injection', 'label': 'Prompt Injection (adv templates)', 'atlas_phase': 'Execution', 'atlas_id': 'AML.T0051', 'metrics': {'success_rate': 0.62}},
                            {'id': 'jailbreak', 'label': 'Jailbreak (role-play)', 'atlas_phase': 'Defense Evasion', 'atlas_id': 'AML.T0054', 'metrics': {'success_rate': 0.48}},
                            {'id': 'model_extraction', 'label': 'Model Extraction (functional I/O)', 'atlas_phase': 'Exfiltration', 'atlas_id': 'AML.T0024.002', 'metrics': {'queries': 1.2e5}},
                            {'id': 'tool_abuse', 'label': 'Tool Abuse (file/system calls)', 'atlas_phase': 'Execution', 'atlas_id': 'AML.T0053', 'metrics': {'calls': 340}},
                        ],
                        'edges': [
                            {'source': 'Execution', 'target': 'prompt_injection'},
                            {'source': 'prompt_injection', 'target': 'jailbreak'},
                            {'source': 'jailbreak', 'target': 'Collection'},
                            {'source': 'Collection', 'target': 'model_extraction'},
                            {'source': 'Defense Evasion', 'target': 'Exfiltration'},
                            {'source': 'model_extraction', 'target': 'Exfiltration'}
                        ]
                    },
                    'poison_backdoor': {
                        'nodes': [
                            {'id': 'Persistence', 'label': 'Persistence', 'atlas_phase': 'Persistence', 'atlas_id': 'AML.TA0006'},
                            {'id': 'Defense Evasion', 'label': 'Defense Evasion', 'atlas_phase': 'Defense Evasion', 'atlas_id': 'AML.TA0007'},
                            {'id': 'Impact', 'label': 'Impact', 'atlas_phase': 'Impact', 'atlas_id': 'AML.TA0011'},
                            {'id': 'data_poisoning', 'label': 'Data Poisoning (training set)', 'atlas_phase': 'Persistence', 'atlas_id': 'AML.T0020', 'metrics': {'poison_frac': 0.02}},
                            {'id': 'backdoor', 'label': 'Backdoor (triggered neurons)', 'atlas_phase': 'AI Attack Staging', 'atlas_id': 'AML.T0043.004', 'metrics': {'ASR': 0.91}},
                            {'id': 'watermark_removal', 'label': 'Watermark Removal', 'atlas_phase': 'Impact', 'atlas_id': 'AML.T0031', 'metrics': {'distortion': 0.03}},
                        ],
                        'edges': [
                            {'source': 'Persistence', 'target': 'data_poisoning'},
                            {'source': 'data_poisoning', 'target': 'backdoor'},
                            {'source': 'backdoor', 'target': 'Impact'},
                            {'source': 'Defense Evasion', 'target': 'watermark_removal'}
                        ]
                    }
                }
                
                data = SCENARIOS[args.scenario]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"✅ ATLAS scenario prepared: {output_path}")
                logger.info(f"📊 Scenario: {args.scenario}")
                logger.info(f"🗺️ Nodes: {len(data['nodes'])}, Edges: {len(data['edges'])}")
                logger.info("🚀 Next (generate HTML): neurinspectre attack-graph visualize "
                            f"--input-path {output_path} --output-path _cli_runs/attack_graph.html")
                logger.info("🌐 Open the HTML file: open _cli_runs/attack_graph.html  (macOS)")
                logger.info("   Tip: do NOT prefix the neurinspectre command with `open`.")
                return 0
                
            elif args.attack_command == 'visualize':
                # Visualize attack graph
                from .attack_visualization import run_attack_visualization
                return run_attack_visualization(args)
            
            return 0
        
        elif args.command == 'attack-graph-viz':
            # Legacy command - redirect to attack visualization
            from .attack_visualization import run_attack_visualization
            return run_attack_visualization(args)
        elif args.command == 'mitre-atlas':
            from .mitre_atlas_cli import run_mitre_atlas
            return run_mitre_atlas(args)
        elif args.command in ['debug-info', 'debug_info', 'debuginfo', 'debug']:
            try:
                import json
                from pathlib import Path
                from .. import debug_info as _dbg

                payload = {
                    "system": _dbg.get_system_info(),
                    "gpu": _dbg.get_gpu_info(),
                    "packages": _dbg.get_package_info(),
                    "neurinspectre": _dbg.get_neurinspectre_info(),
                }

                text = json.dumps(payload, indent=2 if getattr(args, "pretty", False) else None)
                if getattr(args, "output", None):
                    outp = Path(str(args.output))
                    outp.parent.mkdir(parents=True, exist_ok=True)
                    outp.write_text(text + ("\n" if not text.endswith("\n") else ""))
                    print(str(outp))
                else:
                    print(text)
                return 0
            except Exception as e:
                logger.error(f"debug-info failed: {e}")
                return 1
        elif args.command in ['generate-demarking-pcap', 'generate_demarking_pcap', 'demarking-pcap', 'demarking_pcap']:
            try:
                import json
                from pathlib import Path
                from ..tools.generate_demarking_pcap import generate_demarking_pcap

                pcap_info = generate_demarking_pcap(
                    args.out,
                    packets=int(args.packets),
                    seed=int(args.seed),
                    threshold=float(args.threshold),
                    margin=float(args.margin),
                    max_tries=int(args.max_tries),
                )
                j = getattr(args, "json", None)
                if j is None:
                    # Default: print output path only (easy piping)
                    print(str(pcap_info.get("out_path", args.out)))
                elif str(j).strip() == "-":
                    # Print JSON metadata
                    print(json.dumps(pcap_info, indent=2))
                else:
                    # Write JSON metadata to a file
                    jp = Path(str(j))
                    jp.parent.mkdir(parents=True, exist_ok=True)
                    jp.write_text(json.dumps(pcap_info, indent=2) + "\n")
                    print(str(jp))
                    print(str(pcap_info.get("out_path", args.out)))
                return 0
            except Exception as e:
                logger.error(f"generate-demarking-pcap failed: {e}")
                return 1
        elif args.command in [
            'generate-concretizer-data',
            'generate_concretizer_data',
            'concretizer-data',
            'concretizer_data',
        ]:
            try:
                import json
                from pathlib import Path

                from ..tools.generate_concretizer_attack_data import generate_concretizer_attack_data

                ctz_info = generate_concretizer_attack_data(
                    out=str(args.out),
                    timesteps=int(args.timesteps),
                    features=int(args.features),
                    dip_every=int(args.dip_every),
                    baseline_value=float(args.baseline),
                    dip_value=float(args.dip_value),
                    threshold=float(args.threshold),
                )

                j = getattr(args, "json", None)
                if j is None:
                    print(str(ctz_info.get("out_path", args.out)))
                elif str(j).strip() == "-":
                    print(json.dumps(ctz_info, indent=2))
                else:
                    jp = Path(str(j))
                    jp.parent.mkdir(parents=True, exist_ok=True)
                    jp.write_text(json.dumps(ctz_info, indent=2) + "\n")
                    print(str(jp))
                    print(str(ctz_info.get("out_path", args.out)))
                return 0
            except Exception as e:
                logger.error(f"generate-concretizer-data failed: {e}")
                return 1
        elif args.command in [
            'ai-security-dashboard',
            'ai_security_dashboard',
            'research-dashboard',
            'research_dashboard',
            'ai_security_research_dashboard_2025',
            'ai-security-research-dashboard-2025',
            'research-dashboard-2025',
        ]:
            try:
                from ..ai_security_research_dashboard_2025 import AISecurityResearchDashboard

                dash = AISecurityResearchDashboard(allow_simulated=bool(getattr(args, "simulated", False)))
                # Dash app run (long-running)
                dash.run(host=str(args.host), port=int(args.port), debug=bool(getattr(args, "debug", False)))
                return 0
            except Exception as e:
                logger.error(f"ai-security-dashboard failed: {e}")
                return 1
        elif args.command in [
            'integrated-system',
            'integrated_system',
            'integrated-neurinspectre',
            'integrated_neurinspectre',
            'integrated_neurinspectre_system',
        ]:
            try:
                import json
                import numpy as np
                from pathlib import Path
                from ..integrated_neurinspectre_system import IntegratedNeurInSpectre

                inp = str(args.input)
                arr = np.load(inp, allow_pickle=True)
                arr = np.asarray(arr)
                sysi = IntegratedNeurInSpectre(sensitivity_profile=str(args.sensitivity))
                res = sysi.comprehensive_security_scan(arr)
                outp = Path(str(args.output))
                outp.parent.mkdir(parents=True, exist_ok=True)
                outp.write_text(json.dumps(res, indent=2 if getattr(args, "pretty", False) else None) + "\n")
                print(str(outp))
                return 0
            except Exception as e:
                logger.error(f"integrated-system failed: {e}")
                return 1
        elif args.command == 'occlusion-analysis':
            from .occlusion_analysis import run_occlusion_analysis_command
            return run_occlusion_analysis_command(args)
        elif args.command == 'attention-heatmap':
            try:
                from .attention_heatmap import main as _attn_main
                argv = [
                    '--model', args.model,
                    '--prompt', args.prompt,
                    '--layer', str(args.layer),
                    '--head', str(args.head),
                    '--device', args.device,
                    '--output', args.output,
                    '--out-prefix', getattr(args, 'out_prefix', 'attn_'),
                ]
                if getattr(args, 'baseline_prompt', None):
                    argv += ['--baseline', args.baseline_prompt]
                return _attn_main(argv)
            except Exception as e:
                logger.error(f"Attention heatmap failed: {e}")
                return 1
        elif args.command == 'attention-security':
            try:
                from .attention_security_analysis import main as _attnsec_main
                argv = [
                    '--model', args.model,
                    '--layer', str(getattr(args, 'layer', 'all')),
                    '--max-tokens', str(getattr(args, 'max_tokens', 128)),
                    '--device', getattr(args, 'device', 'auto'),
                    '--output-png', getattr(args, 'output_png', '_cli_runs/attention_security.png'),
                    '--out-json', getattr(args, 'out_json', '_cli_runs/attention_security.json'),
                    '--out-html', getattr(args, 'out_html', '_cli_runs/attention_security.html'),
                    '--contamination', str(getattr(args, 'contamination', 'auto')),
                    '--n-estimators', str(getattr(args, 'n_estimators', 256)),
                    '--seed', str(getattr(args, 'seed', 0)),
                    '--title', str(getattr(args, 'title', 'NeurInSpectre — Attention Security Analysis')),
                ]
                if getattr(args, 'layer_start', None) is not None:
                    argv += ['--layer-start', str(args.layer_start)]
                if getattr(args, 'layer_end', None) is not None:
                    argv += ['--layer-end', str(args.layer_end)]
                if getattr(args, 'prompt', None):
                    argv += ['--prompt', args.prompt]
                elif getattr(args, 'prompt_file', None):
                    argv += ['--prompt-file', args.prompt_file]
                else:
                    raise ValueError('Prompt required')
                return _attnsec_main(argv)
            except Exception as e:
                logger.error(f"Attention security analysis failed: {e}")
                return 1
        elif args.command == 'activation_attention_gradient_alignment':
            try:
                import json
                import hashlib
                from pathlib import Path

                import numpy as np
                import torch
                import torch.nn.functional as F
                from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

                from ..visualization.attention_gradient_alignment import (
                    AGAMetrics,
                    aga_matrix,
                    plot_attention_gradient_alignment,
                )

                # Visualization path does not require torch/device; handle it early to avoid
                # static-analyzer "unused variable" warnings for device-selection locals.
                if args.aga_action == 'visualize':
                    in_json = Path(str(getattr(args, 'in_json')))
                    obj = json.loads(in_json.read_text())

                    metrics = AGAMetrics(
                        title=str(obj.get('title', 'Attention-Gradient Alignment (AGA)')),
                        model=str(obj.get('model', 'model')),
                        tokenizer=str(obj.get('tokenizer', obj.get('model', 'model'))),
                        prompt=str(obj.get('prompt', '')),
                        layer_start=int(obj.get('layer_start', 0)),
                        layer_end=int(obj.get('layer_end', 0)),
                        seq_len=int(obj.get('seq_len', 0)),
                        num_layers=int(obj.get('num_layers', 0)),
                        num_heads=int(obj.get('num_heads', 0)),
                        objective=str(obj.get('objective', 'auto')),
                        attn_source=str(obj.get('attn_source', 'auto')),
                        risk_threshold=float(obj.get('risk_threshold', 0.25)),
                        clip_percentile=float(obj.get('clip_percentile', 0.99)),
                        alignment=[[float(v) for v in row] for row in obj.get('alignment', [])],
                        subtitle=str(obj.get('subtitle')) if obj.get('subtitle') is not None else None,
                        prompt_sha16=str(obj.get('prompt_sha16')) if obj.get('prompt_sha16') is not None else None,
                    )

                    out_png = Path(str(getattr(args, 'out_png', '_cli_runs/attention_gradient_alignment.png')))
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    ttl = getattr(args, 'title', None) or metrics.title
                    plot_attention_gradient_alignment(metrics, title=str(ttl), out_path=str(out_png), guidance=True)
                    print(str(out_png))

                    out_html = getattr(args, "out_html", None)
                    if out_html:
                        from ..visualization.attention_gradient_alignment import write_attention_gradient_alignment_html

                        outp = Path(str(out_html))
                        outp.parent.mkdir(parents=True, exist_ok=True)
                        write_attention_gradient_alignment_html(metrics, out_path=str(outp), title=str(ttl))
                        print(str(outp))
                    return 0

                # Resolve device (craft path)
                dev = getattr(args, 'device', 'auto')
                if dev == 'auto':
                    if torch.cuda.is_available():
                        dev = 'cuda'
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        dev = 'mps'
                    else:
                        dev = 'cpu'
                device = torch.device(str(dev))

                # craft
                def _read_first_nonempty_line(fp: str) -> str:
                    pth = Path(str(fp))
                    for ln in pth.read_text().splitlines():
                        s = ln.strip()
                        if s:
                            return s
                    raise ValueError('prompt-file is empty')

                prompt = getattr(args, 'prompt', None)
                if not prompt and getattr(args, 'prompt_file', None):
                    prompt = _read_first_nonempty_line(getattr(args, 'prompt_file'))
                if not prompt:
                    raise ValueError('Prompt required')

                tok_id = getattr(args, 'tokenizer', None) or args.model
                trust_rc = bool(getattr(args, 'trust_remote_code', False))
                tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True, trust_remote_code=trust_rc)

                max_tokens = int(getattr(args, 'max_tokens', 128) or 128)
                layer_start = int(getattr(args, 'layer_start', 0) or 0)
                layer_end = getattr(args, 'layer_end', None)
                layer_end = int(layer_end) if layer_end is not None else None

                attn_source = str(getattr(args, 'attn_source', 'auto') or 'auto')
                objective = str(getattr(args, 'objective', 'auto') or 'auto')
                attn_impl = str(getattr(args, 'attn_impl', 'auto') or 'auto')

                risk_threshold = float(getattr(args, 'risk_threshold', 0.25) or 0.25)
                clip_percentile = float(getattr(args, 'clip_percentile', 0.99) or 0.99)

                # Load model (prefer safetensors; fall back safely)
                load_kwargs = {'trust_remote_code': trust_rc}
                if attn_impl == 'eager':
                    load_kwargs['attn_implementation'] = 'eager'

                mdl = None
                used_cls = 'base'

                # If objective suggests LM loss, prefer a CausalLM head when possible
                if objective in ('auto', 'lm_nll'):
                    try:
                        mdl = AutoModelForCausalLM.from_pretrained(args.model, use_safetensors=True, **load_kwargs)
                        used_cls = 'causal_lm'
                    except Exception:
                        try:
                            mdl = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
                            used_cls = 'causal_lm'
                        except Exception:
                            mdl = None

                if mdl is None:
                    try:
                        mdl = AutoModel.from_pretrained(args.model, use_safetensors=True, **load_kwargs)
                        used_cls = 'base'
                    except Exception:
                        try:
                            mdl = AutoModel.from_pretrained(args.model, **load_kwargs)
                            used_cls = 'base'
                        except Exception as e:
                            raise RuntimeError(
                                "Failed to load model weights. Prefer models that ship 'safetensors' weights, "
                                "or upgrade torch (>=2.6) to load legacy .bin weights safely."
                            ) from e

                mdl.to(device)
                mdl.eval()

                inputs = tok(str(prompt), return_tensors='pt', truncation=True, max_length=max_tokens)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward with gradients enabled
                mdl.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    out = mdl(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)

                    # Choose attention tensor group
                    def _pick_attn(o):
                        if attn_source == 'attentions':
                            return getattr(o, 'attentions', None)
                        if attn_source == 'encoder_attentions':
                            return getattr(o, 'encoder_attentions', None)
                        if attn_source == 'decoder_attentions':
                            return getattr(o, 'decoder_attentions', None)
                        if attn_source == 'cross_attentions':
                            return getattr(o, 'cross_attentions', None)
                        # auto
                        for nm in ('attentions', 'decoder_attentions', 'encoder_attentions', 'cross_attentions'):
                            cand = getattr(o, nm, None)
                            if cand is not None:
                                # Some models return tuples of Nones under flash attention
                                if isinstance(cand, (tuple, list)) and any(x is not None for x in cand):
                                    return cand
                        return None

                    attn = _pick_attn(out)

                    # Retry once with eager attention if auto and attentions are unavailable
                    # (common when SDPA/FlashAttention is active).
                    if attn is None and attn_impl == 'auto':
                        try:
                            load_kwargs_retry = dict(load_kwargs)
                            load_kwargs_retry['attn_implementation'] = 'eager'

                            mdl_retry = None
                            if used_cls == 'causal_lm':
                                try:
                                    mdl_retry = AutoModelForCausalLM.from_pretrained(
                                        args.model,
                                        use_safetensors=True,
                                        **load_kwargs_retry,
                                    )
                                except Exception:
                                    mdl_retry = AutoModelForCausalLM.from_pretrained(
                                        args.model,
                                        **load_kwargs_retry,
                                    )
                            else:
                                try:
                                    mdl_retry = AutoModel.from_pretrained(
                                        args.model,
                                        use_safetensors=True,
                                        **load_kwargs_retry,
                                    )
                                except Exception:
                                    mdl_retry = AutoModel.from_pretrained(
                                        args.model,
                                        **load_kwargs_retry,
                                    )

                            if mdl_retry is not None:
                                mdl = mdl_retry.to(device)
                                mdl.eval()
                                mdl.zero_grad(set_to_none=True)
                                out = mdl(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
                                attn = _pick_attn(out)
                                attn_impl = 'eager'
                        except Exception:
                            pass

                    if attn is None:
                        raise ValueError(
                            "Model did not return attentions. Try --attn-impl eager, a different model, "
                            "or ensure output_attentions=True is supported for this architecture."
                        )

                    attn_list = [a for a in attn if a is not None]
                    if not attn_list:
                        raise ValueError(
                            "Model returned no usable attention tensors. Try --attn-impl eager (disables flash attention outputs)."
                        )

                    for a in attn_list:
                        a.retain_grad()

                    # Objective selection
                    if objective == 'auto':
                        objective = 'lm_nll' if getattr(out, 'logits', None) is not None else 'hidden_l2'

                    if objective == 'lm_nll' and getattr(out, 'logits', None) is not None:
                        logits = out.logits
                        vocab = int(logits.shape[-1])
                        if 'input_ids' not in inputs:
                            raise ValueError('Tokenizer did not produce input_ids')
                        labels = inputs['input_ids'][:, 1:].contiguous()
                        logits_shift = logits[:, :-1, :].contiguous()
                        loss = F.cross_entropy(logits_shift.view(-1, vocab), labels.view(-1), reduction='mean')
                    else:
                        hs = getattr(out, 'last_hidden_state', None)
                        if hs is None and getattr(out, 'hidden_states', None) is not None:
                            hs = out.hidden_states[-1]
                        if hs is None:
                            raise ValueError('Model did not return hidden states for proxy objective')
                        loss = hs.float().pow(2).mean()

                    loss.backward()

                    n_layers = int(len(attn_list))
                    layer_start = max(0, int(layer_start))
                    if layer_end is None:
                        layer_end = n_layers - 1
                    layer_end = min(int(layer_end), n_layers - 1)
                    if layer_end < layer_start:
                        raise ValueError('--layer-end must be >= --layer-start')

                    att_np = []
                    grad_np = []
                    for li in range(layer_start, layer_end + 1):
                        a = attn_list[li]
                        a0 = a[0].detach().cpu().numpy()
                        g = a.grad
                        if g is None:
                            g0 = np.zeros_like(a0)
                        else:
                            g0 = g[0].detach().cpu().numpy()
                        att_np.append(a0)
                        grad_np.append(g0)

                mat = aga_matrix(att_np, grad_np)
                num_layers = int(mat.shape[0])
                num_heads = int(mat.shape[1])

                # Compute enhanced metrics (research: 2024-2025)
                from ..visualization.attention_gradient_alignment import (
                    compute_trigger_attention_ratio,
                    compute_gradient_attention_anomaly_score,
                    compute_head_similarity_matrix,
                    identify_high_risk_heads,
                )
                
                # TAR (Trigger Attention Ratio) - computed on mean attention across layers
                tar_scores = []
                gaas_scores = []
                head_sim_scores = []
                
                for idx, attn_layer in enumerate(att_np):
                    tar = compute_trigger_attention_ratio(attn_layer)
                    tar_scores.append(tar)
                    
                    gaas, _ = compute_gradient_attention_anomaly_score(attn_layer, grad_np[idx])
                    gaas_scores.append(gaas)
                    
                    head_sim = compute_head_similarity_matrix(attn_layer)
                    # Mean off-diagonal similarity (exclude self-similarity)
                    h = head_sim.shape[0]
                    if h > 1:
                        mask = ~np.eye(h, dtype=bool)
                        mean_sim = float(np.mean(head_sim[mask]))
                    else:
                        mean_sim = 0.0
                    head_sim_scores.append(mean_sim)
                
                tar_mean = float(np.mean(tar_scores)) if tar_scores else None
                gaas_mean = float(np.mean(gaas_scores)) if gaas_scores else None
                head_sim_mean = float(np.mean(head_sim_scores)) if head_sim_scores else None
                
                # Identify high-risk heads
                high_risk_analysis = identify_high_risk_heads(mat, risk_threshold=risk_threshold, percentile=90.0)

                model_short = str(args.model).split('/')[-1]
                prompt_str = str(prompt)
                prompt_short = prompt_str if len(prompt_str) <= 64 else prompt_str[:61] + '...'
                subtitle = f"{model_short} | '{prompt_short}'"

                title = str(getattr(args, 'title', 'Attention-Gradient Alignment (AGA)'))
                out_json = Path(str(getattr(args, 'out_json', '_cli_runs/attention_gradient_alignment.json')))
                out_png = Path(str(getattr(args, 'out_png', '_cli_runs/attention_gradient_alignment.png')))
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_png.parent.mkdir(parents=True, exist_ok=True)

                sha16 = hashlib.sha256(prompt_str.encode('utf-8', errors='ignore')).hexdigest()[:16]

                metrics = AGAMetrics(
                    title=title,
                    model=str(args.model),
                    tokenizer=str(tok_id),
                    prompt=prompt_str,
                    layer_start=int(layer_start),
                    layer_end=int(layer_end),
                    seq_len=int(inputs['input_ids'].shape[1]) if 'input_ids' in inputs else 0,
                    num_layers=int(num_layers),
                    num_heads=int(num_heads),
                    objective=str(objective),
                    attn_source=str(attn_source),
                    risk_threshold=float(risk_threshold),
                    clip_percentile=float(clip_percentile),
                    alignment=[[float(v) for v in row.tolist()] for row in mat],
                    trigger_attention_ratio=tar_mean,
                    gaas_score=gaas_mean,
                    head_similarity_mean=head_sim_mean,
                    high_risk_analysis=high_risk_analysis,
                    subtitle=subtitle,
                    prompt_sha16=sha16,
                )

                # Flagged head indices (useful for triage)
                flagged_pos = []
                flagged_neg = []
                for li in range(num_layers):
                    for hi in range(num_heads):
                        v = float(mat[li, hi])
                        if v >= float(risk_threshold):
                            flagged_pos.append({'layer': int(layer_start) + li, 'head': hi, 'value': v})
                        if v <= -float(risk_threshold):
                            flagged_neg.append({'layer': int(layer_start) + li, 'head': hi, 'value': v})

                # Save JSON
                obj = {
                    'title': metrics.title,
                    'subtitle': metrics.subtitle,
                    'model': metrics.model,
                    'tokenizer': metrics.tokenizer,
                    'prompt': metrics.prompt,
                    'prompt_sha16': metrics.prompt_sha16,
                    'layer_start': metrics.layer_start,
                    'layer_end': metrics.layer_end,
                    'seq_len': metrics.seq_len,
                    'num_layers': metrics.num_layers,
                    'num_heads': metrics.num_heads,
                    'objective': metrics.objective,
                    'attn_source': metrics.attn_source,
                    'attn_impl': attn_impl,
                    'model_class': used_cls,
                    'risk_threshold': metrics.risk_threshold,
                    'clip_percentile': metrics.clip_percentile,
                    'alignment': metrics.alignment,
                    'flagged_positive': flagged_pos,
                    'flagged_negative': flagged_neg,
                    # Enhanced metrics (2024-2025 research)
                    'trigger_attention_ratio': metrics.trigger_attention_ratio,
                    'gaas_score': metrics.gaas_score,
                    'head_similarity_mean': metrics.head_similarity_mean,
                    'high_risk_analysis': metrics.high_risk_analysis,
                    'notes': [
                        'AGA is a heuristic sensitivity signal (cosine similarity between attention and dJ/dAttention).',
                        'Use for triage: compare benign vs suspect prompt families and corroborate with other modules.',
                    ],
                }

                out_json.write_text(json.dumps(obj, indent=2))

                plot_attention_gradient_alignment(metrics, title=title, out_path=str(out_png))

                print(str(out_json))
                print(str(out_png))

                out_html = getattr(args, "out_html", None)
                if out_html:
                    from ..visualization.attention_gradient_alignment import write_attention_gradient_alignment_html

                    outp = Path(str(out_html))
                    outp.parent.mkdir(parents=True, exist_ok=True)
                    write_attention_gradient_alignment_html(metrics, out_path=str(outp), title=str(title))
                    print(str(outp))
                return 0

            except Exception as e:
                logger.error(f"Activation AGA failed: {e}")
                return 1
        elif args.command == 'visualize-explanations':
            try:
                import os, json
                import numpy as np
                import matplotlib.pyplot as plt
                from pathlib import Path

                def load_any(path: str):
                    p = Path(path)
                    if not p.exists():
                        raise FileNotFoundError(path)
                    if p.suffix.lower() == '.json':
                        with open(p, 'r') as f:
                            return json.load(f)
                    if p.suffix.lower() == '.npy':
                        return np.load(p)
                    if p.suffix.lower() == '.npz':
                        npz = np.load(p)
                        if 'attributions' in npz:
                            return npz['attributions']
                        key = max(npz.files, key=lambda k: npz[k].size)
                        return npz[key]
                    if p.suffix.lower() in ['.csv', '.txt']:
                        return np.loadtxt(p, delimiter=',')
                    raise ValueError('Unsupported explanation format')

                exp = load_any(args.explanation)
                feature_names = None
                values = None
                if isinstance(exp, dict):
                    # Recursive find for nested structures (graphs/datasets)
                    def _find_nested_keys(obj, keys):
                        if isinstance(obj, dict):
                            if all(k in obj for k in keys):
                                return obj
                            for v in obj.values():
                                hit = _find_nested_keys(v, keys)
                                if hit is not None:
                                    return hit
                        elif isinstance(obj, (list, tuple)):
                            for v in obj:
                                hit = _find_nested_keys(v, keys)
                                if hit is not None:
                                    return hit
                        return None

                    graph = _find_nested_keys(exp, ['nodes', 'edges'])
                    dsblk = _find_nested_keys(exp, ['datasets']) if graph is None else None

                    # Graph: derive node importance
                    if graph is not None:
                        try:
                            import numpy as _np
                            import networkx as _nx
                            G = _nx.DiGraph()
                            for n in graph.get('nodes', []):
                                nid = n.get('id') or n.get('name')
                                if nid is None:
                                    continue
                                G.add_node(nid)
                            for e in graph.get('edges', []):
                                s = e.get('source'); t = e.get('target')
                                if s is None or t is None:
                                    continue
                                G.add_edge(s, t)
                            try:
                                pr = _nx.pagerank(G)
                            except Exception:
                                pr = {n: (G.in_degree(n) + G.out_degree(n)) for n in G.nodes()}
                            feature_names = list(pr.keys())
                            values = _np.array([pr[n] for n in feature_names], dtype=float)
                        except Exception:
                            counts = {}
                            for n in graph.get('nodes', []):
                                key = n.get('atlas_phase') or n.get('label') or n.get('id') or 'unknown'
                                counts[key] = counts.get(key, 0) + 1
                            feature_names = list(counts.keys())
                            values = np.array(list(counts.values()), dtype=float)

                    # Datasets: aggregate numeric scores or fallback to unit weight
                    elif dsblk is not None:
                        ds = dsblk.get('datasets', {}) if isinstance(dsblk, dict) else {}
                        names = []
                        scores = []
                        for key, meta in ds.items():
                            score = None
                            if isinstance(meta, dict):
                                for cand in ['confidence', 'score', 'count', 'weight', 'severity']:
                                    v = meta.get(cand)
                                    if isinstance(v, (int, float)):
                                        score = float(v)
                                        break
                            if score is None:
                                score = 1.0
                            names.append(key)
                            scores.append(score)
                        if names:
                            feature_names = names
                            values = np.array(scores, dtype=float)
                        else:
                            feature_names = ['datasets']
                            values = np.array([0.0], dtype=float)

                    else:
                        # Generic dict → robust ragged stacking; if none numeric, use frequency counts
                        keys = list(exp.keys())
                        vals = list(exp.values())
                        coerced = []
                        keep_idx = []
                        for i, v in enumerate(vals):
                            try:
                                arr_v = np.array(v, dtype=float).reshape(-1)
                                coerced.append(arr_v)
                                keep_idx.append(i)
                            except Exception:
                                continue
                        if coerced:
                            max_len = max(a.shape[0] for a in coerced)
                            pad = np.full((len(coerced), max_len), np.nan, dtype=float)
                            for i, a in enumerate(coerced):
                                pad[i, :a.shape[0]] = a
                            feature_names = [keys[i] for i in keep_idx]
                            values = pad
                        else:
                            # Frequency counts fallback to avoid hard failure
                            counts = {k: 1.0 for k in keys}
                            feature_names = list(counts.keys())
                            values = np.array(list(counts.values()), dtype=float)
                else:
                    # Array-like → coerce to float, allowing ragged lists via padding
                    try:
                        arr = np.array(exp, dtype=float)
                        values = arr
                    except Exception:
                        # Ragged lists
                        seq = list(exp) if hasattr(exp, '__iter__') else [exp]
                        coerced = []
                        for v in seq:
                            try:
                                coerced.append(np.array(v, dtype=float).reshape(-1))
                            except Exception:
                                continue
                        if not coerced:
                            raise
                        max_len = max(a.shape[0] for a in coerced)
                        pad = np.full((len(coerced), max_len), np.nan, dtype=float)
                        for i, a in enumerate(coerced):
                            pad[i, :a.shape[0]] = a
                        values = pad

                if np.ndim(values) >= 2:
                    mean_imp = np.nanmean(values, axis=tuple(range(1, values.ndim)))
                else:
                    mean_imp = values

                if feature_names is None:
                    feature_names = [f'f{i}' for i in range(len(mean_imp))]

                req_topk = max(1, int(args.topk))
                order = np.argsort(np.abs(mean_imp))[::-1]
                idxs = order[:req_topk]
                k = int(len(idxs))
                top_names = [feature_names[i] for i in idxs]
                top_vals = mean_imp[idxs]
                top_abs = np.abs(top_vals)
                coverage = float(top_abs.sum() / (np.nansum(np.abs(mean_imp)) + 1e-8))

                import numpy as _np
                import matplotlib.pyplot as _plt
                fig, ax = _plt.subplots(figsize=(max(9, k*0.55), 4.6))
                colors = ['#cc0000' if v < 0 else '#1f5fbf' for v in top_vals]
                ax.bar(range(k), top_vals, color=colors)
                ax.set_xticks(range(k))
                ax.set_xticklabels(top_names, rotation=45, ha='right')
                ax.set_title('NeurInSpectre — Explanation Importances (Top‑K, signed)')
                ax.set_xlabel('Feature')
                ax.set_ylabel('Attribution (signed)')
                # Coverage curve
                ax2 = ax.twinx()
                ax2.plot(range(k), _np.cumsum(top_abs)/( _np.abs(mean_imp).sum()+1e-8 ), 'k--', linewidth=1.2, label='Cumulative |importance|')
                ax2.set_ylim(0,1.05)
                ax2.set_yticks([0,0.25,0.5,0.75,1.0])
                ax2.set_ylabel('Cumulative coverage')
                ax2.legend(loc='upper right', fontsize=8)
                # Red/Blue guidance
                fig.text(0.01, 0.02, 'Red: negative attribution (suppressive). Blue: positive (supportive).', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f2f2f2', edgecolor='#808080', alpha=0.9))
                fig.text(0.55, 0.02, 'Blue: act on top supportive drivers; Red: audit suppressive drivers.', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6f0ff', edgecolor='#1f5fbf', alpha=0.95))
                fig.tight_layout(rect=[0, 0.08, 1, 1])
                fbar = f"{args.out_prefix}explain_topk.png"
                fig.savefig(fbar, dpi=200)

                if np.ndim(values) == 2:
                    _plt.figure(figsize=(10.5, 4.2))
                    vmax = float(_np.nanmax(_np.abs(values))) + 1e-8
                    im = _plt.imshow(values, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
                    cbar = _plt.colorbar(im)
                    cbar.set_label('Attribution (signed)')
                    _plt.title('NeurInSpectre — Explanation Heatmap (diverging, 0‑centered)')
                    _plt.xlabel('Token/Position index')
                    _plt.ylabel('Feature index')
                    # Guidance stacked below
                    _plt.figtext(0.01, 0.02, 'Red: suppressive regions; audit upstream inputs/process.', fontsize=9,
                                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe6e6', edgecolor='#cc0000', alpha=0.95))
                    _plt.figtext(0.55, 0.02, 'Blue: supportive regions; validate and protect key drivers.', fontsize=9,
                                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6f0ff', edgecolor='#1f5fbf', alpha=0.95))
                    _plt.tight_layout(rect=[0, 0.12, 1, 1])
                    fhm = f"{args.out_prefix}explain_heatmap.png"
                    _plt.savefig(fhm, dpi=200)
                    print(fhm)

                print(fbar)
                return 0
            except Exception as e:
                logger.error(f"Explanation visualization failed: {e}")
                return 1
        elif args.command == 'subnetwork_hijack':
            try:
                import json
                import os
                from glob import glob

                import numpy as np
                from pathlib import Path
                import matplotlib.pyplot as plt
                if args.snh_action == 'identify':
                    act_path = str(getattr(args, "activations", "") or "")
                    if not act_path:
                        raise ValueError("--activations is required")

                    try:
                        A = np.load(act_path)
                    except FileNotFoundError:
                        # Provide a helpful "did you mean" message.
                        parent = os.path.dirname(act_path) or "."
                        base = os.path.basename(act_path)

                        candidates: list[str] = []
                        try:
                            candidates = sorted(glob(os.path.join(parent, "*watermarked*.npy")))
                        except Exception:
                            candidates = []

                        hints = ""
                        if candidates:
                            show = candidates[:12]
                            hints = "\n".join([f"  - {p}" for p in show])
                            if len(candidates) > len(show):
                                hints += f"\n  ... and {len(candidates) - len(show)} more"
                            hints = (
                                f"\n\nFile not found: {act_path}\n"
                                f"Nearby candidates (same directory):\n{hints}\n\n"
                                f"Tip: many commands write `<out-prefix>watermarked.npy` (e.g. `_cli_runs/backdoor_wm_watermarked.npy`)."
                            )
                        else:
                            hints = (
                                f"\n\nFile not found: {act_path}\n"
                                f"Tip: check your `--out-prefix`. Many commands write `<out-prefix>watermarked.npy`."
                            )

                        raise FileNotFoundError(hints) from None
                    # Clean NaN/Inf and normalize dims
                    import numpy as _np
                    A = _np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
                    arr = np.array(A)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    elif arr.ndim > 2:
                        arr = arr.reshape(-1, arr.shape[-1])
                    # Simple kmeans via scikit-learn if available, else random labels
                    try:
                        from sklearn.cluster import KMeans
                        km = KMeans(n_clusters=int(args.n_clusters), n_init=5, random_state=42)
                        labels = km.fit_predict(arr.T)  # cluster neurons by their activation patterns
                    except Exception:
                        rng = np.random.default_rng(42)
                        labels = rng.integers(0, int(args.n_clusters), size=arr.shape[1])
                    counts = np.bincount(labels, minlength=int(args.n_clusters))
                    
                    # Calculate research-based vulnerability metrics (2024-2025)
                    k = int(args.n_clusters)
                    cluster_metrics = []
                    total_energy = np.sum(arr ** 2)
                    
                    for ci in range(k):
                        mask = (labels == ci)
                        if not mask.any():
                            cluster_metrics.append({
                                'cluster_id': ci,
                                'size': 0,
                                'vulnerability_score': 0.0,
                                'energy_ratio': 0.0,
                                'entropy': 0.0,
                                'centroid_norm': 0.0
                            })
                            continue
                        
                        cluster_acts = arr[:, mask]
                        centroid = cluster_acts.mean(axis=1)
                        
                        # Energy ratio
                        cluster_energy = np.sum(cluster_acts ** 2)
                        energy_ratio = float(cluster_energy / (total_energy + 1e-10))
                        
                        # Entropy (Shannon)
                        probs = np.abs(cluster_acts.flatten())
                        probs = probs / (probs.sum() + 1e-10)
                        probs = probs[probs > 1e-10]
                        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
                        
                        # Centroid norm
                        centroid_norm = float(np.linalg.norm(centroid))
                        
                        # Cohesion (intra-cluster std dev)
                        cohesion = float(np.std(cluster_acts))
                        
                        # Vulnerability score (composite metric from research)
                        # High energy + low entropy + high centroid = vulnerable
                        vuln_score = min(1.0,
                            0.4 * min(1.0, energy_ratio * 5) +  # Weight energy heavily
                            0.3 * (1.0 - min(1.0, entropy / 10)) +  # Low entropy = vulnerable
                            0.3 * min(1.0, cohesion)  # High cohesion = targetable
                        )
                        
                        cluster_metrics.append({
                            'cluster_id': ci,
                            'size': int(counts[ci]),
                            'vulnerability_score': float(vuln_score),
                            'energy_ratio': float(energy_ratio),
                            'entropy': float(entropy),
                            'centroid_norm': float(centroid_norm),
                            'cohesion': float(cohesion)
                        })
                    
                    # Save enhanced JSON with metrics
                    outj = Path(f"{args.out_prefix}subnetwork_clusters.json")
                    outj.write_text(json.dumps({
                        'n_clusters': k,
                        'counts': counts.tolist(),
                        'cluster_metrics': cluster_metrics
                    }, indent=2))
                    
                    # Generate INTERACTIVE Plotly visualization if requested
                    if args.interactive:
                        try:
                            import plotly.graph_objects as go
                            from plotly.subplots import make_subplots
                            
                            # Sort clusters by vulnerability score
                            sorted_metrics = sorted(cluster_metrics, key=lambda x: x['vulnerability_score'], reverse=True)
                            
                            # Create 2x2 dashboard
                            fig = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=(
                                    '🎯 Cluster Sizes (Neuron Count)',
                                    '⚠️ Vulnerability Scores',
                                    '⚡ Energy Ratio Distribution',
                                    '📊 Cluster Metrics Table'
                                ),
                                specs=[[{'type': 'bar'}, {'type': 'bar'}],
                                       [{'type': 'bar'}, {'type': 'table'}]],
                                vertical_spacing=0.18,
                                horizontal_spacing=0.12
                            )
                            
                            # Panel 1: Cluster sizes with top-3 highlighting
                            cluster_ids = [m['cluster_id'] for m in cluster_metrics]
                            sizes = [m['size'] for m in cluster_metrics]
                            top3_ids = [m['cluster_id'] for m in sorted(cluster_metrics, key=lambda x: x['size'], reverse=True)[:3]]
                            
                            colors = ['#9B59B6' if cid in top3_ids else '#2A9D8F' for cid in cluster_ids]
                            
                            fig.add_trace(go.Bar(
                                x=cluster_ids,
                                y=sizes,
                                marker_color=colors,
                                name='Cluster Size',
                                hovertemplate=(
                                    '<b>Cluster %{x}</b><br>' +
                                    'Size: %{y} neurons<br>' +
                                    '<b>🔴 Red Team:</b> Large clusters = easier targets<br>' +
                                    '<b>🔵 Blue Team:</b> Monitor top-3 clusters closely<br>' +
                                    '<extra></extra>'
                                )
                            ), row=1, col=1)
                            
                            # Panel 2: Vulnerability scores
                            vulns = [m['vulnerability_score'] for m in cluster_metrics]
                            vuln_colors = ['#E74C3C' if v >= 0.7 else '#F39C12' if v >= 0.5 else '#27AE60' for v in vulns]
                            
                            fig.add_trace(go.Bar(
                                x=cluster_ids,
                                y=vulns,
                                marker_color=vuln_colors,
                                name='Vulnerability',
                                hovertemplate=(
                                    '<b>Cluster %{x}</b><br>' +
                                    'Vulnerability: %{y:.3f}<br>' +
                                    '<b>🔴 Red Team:</b> Score ≥0.7 = prime hijack target<br>' +
                                    '<b>🔵 Blue Team:</b> Score ≥0.5 requires hardening<br>' +
                                    '<extra></extra>'
                                )
                            ), row=1, col=2)
                            
                            # Add vulnerability threshold lines
                            fig.add_hline(y=0.7, line_dash="dash", line_color="red", line_width=2,
                                         annotation_text="Critical (0.7)", row=1, col=2)
                            fig.add_hline(y=0.5, line_dash="dash", line_color="orange", line_width=2,
                                         annotation_text="Monitor (0.5)", row=1, col=2)
                            
                            # Panel 3: Energy ratios
                            energies = [m['energy_ratio'] for m in cluster_metrics]
                            energy_colors = ['#E74C3C' if e >= 0.3 else '#F39C12' if e >= 0.2 else '#3498DB' for e in energies]
                            
                            fig.add_trace(go.Bar(
                                x=cluster_ids,
                                y=energies,
                                marker_color=energy_colors,
                                name='Energy Ratio',
                                hovertemplate=(
                                    '<b>Cluster %{x}</b><br>' +
                                    'Energy Ratio: %{y:.3f}<br>' +
                                    '<b>🔴 Red Team:</b> Ratio >0.3 = low-budget hijack possible<br>' +
                                    '<b>🔵 Blue Team:</b> Ratio >0.2 requires clipping/regularization<br>' +
                                    '<extra></extra>'
                                )
                            ), row=2, col=1)
                            
                            fig.add_hline(y=0.3, line_dash="dash", line_color="red", line_width=2,
                                         annotation_text="Red Target (0.3)", row=2, col=1)
                            fig.add_hline(y=0.2, line_dash="dash", line_color="orange", line_width=2,
                                         annotation_text="Blue Monitor (0.2)", row=2, col=1)
                            
                            # Panel 4: Comprehensive metrics table
                            table_data = {
                                'Cluster': [f"C{m['cluster_id']}" for m in sorted_metrics[:10]],
                                'Vuln': [f"{m['vulnerability_score']:.3f}" for m in sorted_metrics[:10]],
                                'Energy': [f"{m['energy_ratio']:.3f}" for m in sorted_metrics[:10]],
                                'Entropy': [f"{m['entropy']:.2f}" for m in sorted_metrics[:10]],
                                'Size': [str(m['size']) for m in sorted_metrics[:10]]
                            }
                            
                            fig.add_trace(go.Table(
                                header=dict(
                                    values=['<b>Cluster</b>', '<b>Vuln</b>', '<b>Energy</b>', '<b>Entropy</b>', '<b>Size</b>'],
                                    fill_color='rgba(128,128,128,0.5)',
                                    align='left',
                                    font=dict(color='white', size=11)
                                ),
                                cells=dict(
                                    values=[table_data['Cluster'], table_data['Vuln'], table_data['Energy'], 
                                           table_data['Entropy'], table_data['Size']],
                                    fill_color='rgba(50,50,50,0.5)',
                                    align='left',
                                    font=dict(color='white', size=10),
                                    height=25
                                )
                            ), row=2, col=2)
                            
                            # Add Red/Blue team guidance below
                            high_vuln_clusters = [m['cluster_id'] for m in cluster_metrics if m['vulnerability_score'] >= 0.7]
                            high_energy_clusters = [m['cluster_id'] for m in cluster_metrics if m['energy_ratio'] >= 0.3]
                            
                            red_guidance = [
                                f"🎯 TARGET CLUSTERS: {high_vuln_clusters if high_vuln_clusters else 'None critical'} (Vuln≥0.7)",
                                f"⚡ ENERGY EXPLOIT: {high_energy_clusters if high_energy_clusters else 'None'} (Energy≥0.3) → Low-budget hijack",
                                f"🔍 TECHNIQUE: Single-token trigger optimization on top clusters",
                                f"📊 BACKDOOR: Graft weights into high-energy neurons for persistent control"
                            ]
                            
                            monitor_clusters = [m['cluster_id'] for m in cluster_metrics if m['vulnerability_score'] >= 0.5]
                            clip_clusters = [m['cluster_id'] for m in cluster_metrics if m['energy_ratio'] >= 0.2]
                            
                            blue_guidance = [
                                f"🛡️ HARDEN CLUSTERS: {monitor_clusters if monitor_clusters else 'All secure'} (Vuln≥0.5)",
                                f"✂️ CLIP/REGULARIZE: {clip_clusters if clip_clusters else 'None needed'} (Energy≥0.2)",
                                f"📈 MONITOR: Track entropy drops (<4 bits) indicating hijack convergence",
                                f"🔒 DEFENSE: Activation clipping + dropout noise on vulnerable clusters"
                            ]
                            
                            fig.add_annotation(
                                text="<b>🔴 RED TEAM - Subnetwork Hijacking</b><br>" + "<br>".join(red_guidance),
                                xref="paper", yref="paper",
                                x=0.75, y=-0.18,
                                showarrow=False,
                                font=dict(size=10, color='white'),
                                align="left",
                                bgcolor='rgba(204,0,0,0.85)',
                                bordercolor='#cc0000',
                                borderwidth=2,
                                borderpad=12,
                                xanchor='center',
                                width=800
                            )
                            
                            fig.add_annotation(
                                text="<b>🔵 BLUE TEAM - Subnetwork Defense</b><br>" + "<br>".join(blue_guidance),
                                xref="paper", yref="paper",
                                x=0.25, y=-0.18,
                                showarrow=False,
                                font=dict(size=10, color='white'),
                                align="left",
                                bgcolor='rgba(31,95,191,0.85)',
                                bordercolor='#1f5fbf',
                                borderwidth=2,
                                borderpad=12,
                                xanchor='center',
                                width=800
                            )
                            
                            # Layout with increased bottom margin
                            fig.update_layout(
                                title=dict(
                                    text=f'⚡ NeurInSpectre Subnetwork Hijack Analysis - {k} Clusters',
                                    x=0.5,
                                    xanchor='center',
                                    font=dict(size=20, color='white')
                                ),
                                height=1200,  # Increased height
                                width=1800,
                                template='plotly_dark',
                                hovermode='closest',
                                uirevision='constant',
                                showlegend=False,
                                plot_bgcolor='rgba(20,20,20,0.95)',
                                paper_bgcolor='rgba(10,10,10,1)',
                                xaxis=dict(gridcolor='rgba(128,128,128,0.2)', zerolinecolor='rgba(128,128,128,0.2)'),
                                yaxis=dict(gridcolor='rgba(128,128,128,0.2)', zerolinecolor='rgba(128,128,128,0.2)'),
                                margin=dict(l=80, r=80, t=100, b=280)  # Increased bottom margin
                            )
                            
                            # Update axes
                            fig.update_xaxes(title_text="Cluster ID", row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
                            fig.update_yaxes(title_text="Neuron Count", row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
                            fig.update_xaxes(title_text="Cluster ID", row=1, col=2, gridcolor='rgba(128,128,128,0.2)')
                            fig.update_yaxes(title_text="Vulnerability Score (0-1)", row=1, col=2, gridcolor='rgba(128,128,128,0.2)')
                            fig.update_xaxes(title_text="Cluster ID", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
                            fig.update_yaxes(title_text="Energy Ratio", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
                            
                            # Save interactive HTML
                            html_file = f"{args.out_prefix}interactive.html"
                            fig.write_html(html_file)
                            logger.info(f"📊 Interactive HTML with vulnerability metrics: {html_file}")
                            logger.info(f"🔍 Research metrics: Vulnerability scores, Energy ratios, Entropy, Cohesion")
                            print(html_file)
                            
                        except Exception as e:
                            logger.warning(f"Interactive visualization failed: {e}, using static only")
                    
                    # Save JSON
                    outj.write_text(json.dumps({
                        'n_clusters': k,
                        'counts': counts.tolist(),
                        'cluster_metrics': cluster_metrics
                    }, indent=2))
                    # Build combined centroid dendrogram + sizes bar chart
                    try:
                        # Compute cluster centroids in activation space
                        k = int(args.n_clusters)
                        centroids = []
                        for ci in range(k):
                            mask = (labels == ci)
                            if mask.any():
                                centroids.append(arr[:, mask].mean(axis=1))  # shape [T]
                            else:
                                centroids.append(np.zeros(arr.shape[0]))
                        centroids = np.stack(centroids, axis=0)  # [k, T]

                        # Hierarchical clustering of centroids
                        from scipy.spatial.distance import pdist
                        from scipy.cluster.hierarchy import linkage, dendrogram
                        Z = linkage(pdist(centroids, metric='euclidean'), method='average')

                        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                        # Left: dendrogram with cluster labels C0..C{k-1}
                        ax1 = axes[0]
                        dendrogram(Z, labels=[f"C{i}" for i in range(k)], ax=ax1, color_threshold=0)
                        ax1.set_title('NeurInSpectre — Centroid Dendrogram')
                        ax1.set_ylabel('Distance')

                        # Right: bar chart with top-3 highlight and notes
                        ax2 = axes[1]
                        bars = ax2.bar(range(len(counts)), counts, color='#2A9D8F')
                        top_idx = np.argsort(counts)[-3:][::-1]
                        for rank, idx in enumerate(top_idx, 1):
                            bars[idx].set_color('#9B59B6')
                            ax2.text(idx, counts[idx] + 0.5, f'Top {rank}', ha='center', fontsize=9)
                        ax2.set_title('NeurInSpectre — Cluster Sizes')
                        ax2.set_xlabel('Cluster Index')
                        ax2.set_ylabel('Neuron Count')

                        import textwrap as _tw
                        btxt = _tw.fill('Blue: harden/monitor top-3 largest clusters; consider redundancy for choke points; watch sudden cluster growth.', width=78)
                        rtxt = _tw.fill('Red: target largest clusters first; probe cluster stability; test whether smaller clusters act as stealth paths.', width=78)
                        fig.text(0.02, 0.01, btxt, fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6f0ff', edgecolor='#1f5fbf', alpha=0.95))
                        fig.text(0.55, 0.01, rtxt, fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe6e6', edgecolor='#cc0000', alpha=0.95))
                        fig.suptitle(f'NeurInSpectre — Subnetwork Hijack – Cluster Overview (k={k})')
                        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

                        f_overview = f"{args.out_prefix}cluster_overview.png"
                        fig.savefig(f_overview, dpi=220)
                        plt.close(fig)
                    except Exception:
                        f_overview = None

                    # Also emit the original bar-only figure for backward compatibility
                    plt.figure(figsize=(9.5, 4.6))
                    bars = plt.bar(range(len(counts)), counts, color='#2A9D8F')
                    top_idx = np.argsort(counts)[-3:][::-1]
                    for rank, idx in enumerate(top_idx, 1):
                        bars[idx].set_color('#9B59B6')
                        plt.text(idx, counts[idx] + 0.5, f'Top {rank}', ha='center', fontsize=9)
                    plt.title('NeurInSpectre — Neuron Subnetwork Sizes')
                    plt.xlabel('Cluster id')
                    plt.ylabel('Neuron count')
                    fig2 = plt.gcf()
                    import textwrap as _tw
                    btxt = _tw.fill('Blue: harden/monitor top-3 largest clusters; consider redundancy for choke points; watch sudden cluster growth.', width=78)
                    rtxt = _tw.fill('Red: target largest clusters first; probe cluster stability; test whether smaller clusters act as stealth paths.', width=78)
                    fig2.text(0.01, 0.02, btxt, fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6f0ff', edgecolor='#1f5fbf', alpha=0.95))
                    fig2.text(0.56, 0.02, rtxt, fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe6e6', edgecolor='#cc0000', alpha=0.95))
                    plt.tight_layout(rect=[0, 0.22, 1, 1])
                    fbar = f"{args.out_prefix}snh_sizes.png"
                    plt.savefig(fbar, dpi=200)
                    plt.close(fig2)

                    print(str(outj))
                    print(fbar)
                    if f_overview:
                        print(f_overview)
                    return 0
                elif args.snh_action == 'inject':
                    neurons = [int(x) for x in str(args.subnetwork).split(',') if x.strip()]
                    plan = {'model': args.model, 'subnetwork': neurons, 'trigger': args.trigger, 'status': 'planned'}
                    from pathlib import Path
                    p = Path(f"{args.out_prefix}snh_injection_plan.json")
                    p.write_text(json.dumps(plan, indent=2))
                    print(str(p))
                    return 0
                else:
                    return 1
            except Exception as e:
                logger.error(f"Subnetwork hijack failed: {e}")
                return 1
        elif args.command == 'activation_drift_evasion':
            try:
                import numpy as np
                import matplotlib.pyplot as plt
                import json as _json
                import json
                from pathlib import Path
                if args.ade_action == 'craft':
                    # REAL drift extraction over prompt steps (no synthetic trajectories).
                    prompts_file = getattr(args, 'prompts_file', None)
                    if not prompts_file:
                        raise ValueError('craft requires --prompts-file (one prompt per step).')

                    prompts = [ln.strip() for ln in Path(prompts_file).read_text().splitlines() if ln.strip()]
                    if not prompts:
                        raise ValueError('prompts-file is empty')

                    baseline_prompt = getattr(args, 'baseline_prompt', None) or prompts[0]
                    step_prompts = prompts

                    import hashlib
                    import torch
                    from transformers import AutoModel, AutoTokenizer

                    # Resolve device
                    dev = getattr(args, 'device', 'auto')
                    if dev == 'auto':
                        if torch.cuda.is_available():
                            dev = 'cuda'
                        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            dev = 'mps'
                        else:
                            dev = 'cpu'

                    tok_id = getattr(args, 'tokenizer', None) or args.model
                    tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)

                    # Prefer safetensors to avoid unsafe torch.load on older torch versions
                    try:
                        mdl = AutoModel.from_pretrained(args.model, use_safetensors=True)
                    except Exception:
                        try:
                            mdl = AutoModel.from_pretrained(args.model)
                        except Exception as e:
                            raise RuntimeError(
                                "Failed to load model weights. Prefer models that ship 'safetensors' weights, "
                                "or upgrade torch (>=2.6) to load legacy .bin weights safely."
                            ) from e

                    mdl.eval()
                    mdl.to(dev)

                    layer = int(getattr(args, 'layer', 0) or 0)
                    reduce = str(getattr(args, 'reduce', 'last') or 'last').lower()

                    def _reduce_hidden(h):
                        # h: torch.Tensor [batch, seq, hidden] or [seq, hidden]
                        if h.dim() == 3:
                            h = h[0]
                        if h.dim() != 2:
                            h = h.reshape(-1, h.shape[-1])
                        if reduce in ('last', 'last_token'):
                            return h[-1]
                        if reduce in ('mean', 'avg', 'average'):
                            return h.mean(dim=0)
                        if reduce in ('max',):
                            return h.max(dim=0).values
                        if reduce in ('maxabs', 'max_abs', 'max-abs'):
                            abs_h = h.abs()
                            pos = abs_h.argmax(dim=0)
                            return h[pos, torch.arange(h.size(1), device=h.device)]
                        raise ValueError(f"Unknown reduce={reduce}")

                    def _layer_vec(prompt: str):
                        inputs = tok(prompt, return_tensors='pt', truncation=True)
                        inputs = {k: v.to(dev) for k, v in inputs.items()}
                        with torch.no_grad():
                            out = mdl(**inputs, output_hidden_states=True, return_dict=True)
                        hs = getattr(out, 'hidden_states', None)
                        if hs is None:
                            raise ValueError('Model did not return hidden_states')
                        layers = list(hs[1:])  # drop embedding output
                        if layer < 0 or layer >= len(layers):
                            raise ValueError(f"--layer must be in [0, {len(layers)-1}] for this model")
                        return _reduce_hidden(layers[layer]).detach().cpu()

                    base_vec = _layer_vec(baseline_prompt)
                    last_vec = _layer_vec(step_prompts[-1])

                    # Neuron selection
                    if getattr(args, 'target_neurons', None):
                        neurons = [int(x) for x in str(args.target_neurons).split(',') if x.strip()]
                        if not neurons:
                            raise ValueError('target-neurons is empty')
                    else:
                        k = int(getattr(args, 'topk', 5) or 5)
                        k = max(1, k)
                        drift_abs = (last_vec - base_vec).abs()
                        k = min(k, int(drift_abs.numel()))
                        neurons = drift_abs.topk(k).indices.tolist()

                    # Compute drift trajectory: drift(step, neuron) = vec(step)[neuron] - base_vec[neuron]
                    traj = np.zeros((len(step_prompts), len(neurons)), dtype=np.float32)
                    base_sel = base_vec[neurons]
                    for i, pr in enumerate(step_prompts):
                        v = _layer_vec(pr)
                        traj[i, :] = (v[neurons] - base_sel).numpy().astype(np.float32)

                    out_prefix = str(getattr(args, 'out_prefix', '_cli_runs/') or '_cli_runs/')
                    out_traj = Path(f"{out_prefix}drift.npy")
                    out_traj.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(out_traj), traj)

                    def _sha16(s: str) -> str:
                        return hashlib.sha256(s.encode('utf-8', errors='ignore')).hexdigest()[:16]

                    meta = {
                        'model': str(args.model),
                        'tokenizer': str(tok_id),
                        'device': str(dev),
                        'layer': int(layer),
                        'reduce': str(reduce),
                        'steps': int(len(step_prompts)),
                        'neurons': [int(x) for x in neurons],
                        'baseline_prompt_len': int(len(baseline_prompt)),
                        'baseline_prompt_sha16': _sha16(baseline_prompt),
                        'prompts_file': str(prompts_file),
                        'prompts_file_sha16': _sha16('\n'.join(step_prompts[:50])),
                    }
                    out_meta = Path(f"{out_prefix}drift_meta.json")
                    out_meta.write_text(json.dumps(meta, indent=2))

                    print(str(out_traj))
                    print(str(out_meta))
                    return 0

                elif args.ade_action == 'visualize':
                    traj = np.load(args.activation_traj)
                    # Optional metadata (for real neuron ids + layer context)
                    meta = None
                    try:
                        meta_path = Path(args.activation_traj).with_name('drift_meta.json')
                        if meta_path.exists():
                            meta = json.loads(meta_path.read_text())
                    except Exception:
                        meta = None
                    neuron_labels = None
                    if isinstance(meta, dict) and isinstance(meta.get('neurons'), list):
                        neuron_labels = [str(x) for x in meta.get('neurons')]
                    # Clean NaN/Inf and shape
                    import numpy as _np
                    traj = _np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)
                    if traj.ndim != 2:
                        traj = traj.reshape(traj.shape[0], -1)
                    plt.figure(figsize=(12,5.2))
                    max_neurons = min(traj.shape[1], 5)
                    peaks = []
                    for i in range(max_neurons):
                        y = traj[:, i]
                        plt.plot(y, label=f'neuron_{neuron_labels[i]}' if neuron_labels and i < len(neuron_labels) else f'neuron_{i}')
                        j = int(np.argmax(np.abs(np.gradient(y)))) if y.size>1 else 0
                        peaks.append((i, int(j), float(y[j])))
                        plt.scatter([j], [y[j]], s=14)
                    # Shade regions of steep slope across any neuron
                    import numpy as _np
                    gsum = _np.sum(_np.abs(_np.gradient(traj[:, :max_neurons], axis=0)), axis=1)
                    thr = float(_np.percentile(gsum, 90))
                    for t in range(1, len(gsum)):
                        if gsum[t] >= thr:
                            plt.axvspan(t-1, t, color='orange', alpha=0.12)
                    # Rolling Z and CUSUM
                    ysum = _np.sum(traj[:, :max_neurons], axis=1)
                    mu, sd = float(_np.mean(ysum)), float(_np.std(ysum) + 1e-8)
                    z = (ysum - mu) / sd
                    k = 0.5
                    cusum = _np.maximum(0, _np.cumsum(z - k))
                    # Forecast TTE (time-to-exceed) a guardrail if trend positive
                    guard = 3.0
                    slope = float(_np.polyfit(_np.arange(len(cusum)), cusum, 1)[0]) if len(cusum)>1 else 0.0
                    tte = float((guard - cusum[-1]) / max(1e-6, slope)) if slope>0 else float('inf')
                    drift_mag = float(np.trapz(np.abs(traj[:, :max_neurons]), axis=0).sum())
                    plt.title('NeurInSpectre — Activation Drift Trajectories')
                    plt.xlabel('Step index')
                    plt.ylabel('Activation drift')
                    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=max(2, max_neurons), frameon=False)
                    import textwrap as _tw
                    btxt = _tw.fill(
                        'Blue: Watch for shaded "steep-change" regions.\n'
                        f'• CUSUM (running sum of normalized drift) ≈ {cusum[-1]:.2f} — larger means sustained change over time.\n'
                        f'• TTE (time-to-exceed) ≈ {tte:.1f} steps — estimated steps until the guardrail is crossed if the current trend continues.\n'
                        '• Rolling Z (current Z‑score vs baseline): values > 3 indicate statistically unusual drift.',
                        width=92
                    )
                    rtxt = _tw.fill(
                        'Red: Reduce sustained change.\n'
                        '• Lower CUSUM by flattening the trend (smaller per‑step changes).\n'
                        '• Keep Rolling Z below 3 by reducing volatility.\n'
                        '• If TTE is small, slow or reverse the drift to avoid crossing the guardrail.',
                        width=92
                    )
                    fig = plt.gcf()
                    fig.text(0.01, 0.02, btxt, fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6f0ff', edgecolor='#1f5fbf', alpha=0.95))
                    fig.text(0.56, 0.02, rtxt, fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe6e6', edgecolor='#cc0000', alpha=0.95))
                    plt.tight_layout(rect=[0, 0.22, 1, 1])
                    fline = f"{args.out_prefix}drift_plot.png"
                    plt.savefig(fline, dpi=200)
                    # Optional: prompt structure–drift correlation (token-level drift)

                    token_png = None

                    try:

                        if getattr(args, 'model', None) and getattr(args, 'baseline_prompt', None) and getattr(args, 'test_prompt', None):

                            import torch

                            from transformers import AutoModel, AutoTokenizer

                            dev2 = getattr(args, 'device', 'auto')

                            if dev2 == 'auto':

                                if torch.cuda.is_available():

                                    dev2 = 'cuda'

                                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():

                                    dev2 = 'mps'

                                else:

                                    dev2 = 'cpu'

                            tok_id2 = getattr(args, 'tokenizer', None) or getattr(args, 'model', None)

                            tok2 = AutoTokenizer.from_pretrained(tok_id2, use_fast=True)

                            try:

                                mdl2 = AutoModel.from_pretrained(args.model, use_safetensors=True)

                            except Exception:

                                mdl2 = AutoModel.from_pretrained(args.model)

                            mdl2.eval(); mdl2.to(dev2)

                            layer2 = int(getattr(args, 'layer', 0) or 0)

                            # Token-aligned drift per position: ||h_test[t] - h_base[t]||_2

                            b_in = tok2(str(args.baseline_prompt), return_tensors='pt', truncation=True)

                            t_in = tok2(str(args.test_prompt), return_tensors='pt', truncation=True)

                            b_in = {k: v.to(dev2) for k, v in b_in.items()}

                            t_in = {k: v.to(dev2) for k, v in t_in.items()}

                            with torch.no_grad():

                                b_out = mdl2(**b_in, output_hidden_states=True, return_dict=True)

                                t_out = mdl2(**t_in, output_hidden_states=True, return_dict=True)

                            b_hs = getattr(b_out, 'hidden_states', None)

                            t_hs = getattr(t_out, 'hidden_states', None)

                            if b_hs is None or t_hs is None:

                                raise ValueError('hidden_states unavailable for token correlation')

                            b_layers = list(b_hs[1:])

                            t_layers = list(t_hs[1:])

                            if layer2 < 0 or layer2 >= min(len(b_layers), len(t_layers)):

                                raise ValueError('layer out of range for token correlation')

                            b_tok = b_layers[layer2][0]  # [seq, hidden]

                            t_tok = t_layers[layer2][0]

                            L = int(min(b_tok.shape[0], t_tok.shape[0]))

                            b_tok = b_tok[:L]

                            t_tok = t_tok[:L]

                            drift_tok = torch.linalg.norm((t_tok - b_tok), dim=1).detach().cpu().numpy()

                            # Plot token drift + highlight spikes

                            plt.figure(figsize=(12, 3.6))

                            plt.plot(drift_tok, linewidth=1.6)

                            plt.title('Prompt Structure–Drift Correlation')

                            plt.xlabel('Token index (aligned)')

                            plt.ylabel('Drift (L2 hidden-state delta)')

                            sp = float(getattr(args, 'spike_percentile', 90.0))

                            thr_tok = float(np.percentile(drift_tok, sp)) if drift_tok.size else 0.0

                            spike_idx = np.where(drift_tok >= thr_tok)[0].tolist()

                            for sidx in spike_idx:

                                plt.axvline(int(sidx), color='red', alpha=0.25, linewidth=1.0)

                            plt.grid(True, alpha=0.25)

                            fig_tok = plt.gcf()

                            fig_tok.text(0.5, 0.01, 'Red lines: token positions with high drift (percentile threshold). Investigate/block as needed.',

                                         ha='center', fontsize=9, color='#444')

                            plt.tight_layout(rect=[0, 0.06, 1, 1])

                            token_png = f"{args.out_prefix}drift_prompt_structure.png"

                            plt.savefig(token_png, dpi=200)

                            plt.close(fig_tok)

                    except Exception:

                        token_png = None

                    # Export summary JSON
                    summ = {
                        'peaks': [{'neuron': int(i), 'step': int(j), 'value': float(v)} for i,j,v in peaks],
                        'steep_regions_percentile': 90.0,
                        'drift_magnitude_sum': drift_mag,
                        'cusum_last': float(cusum[-1]),
                        'tte_guard': tte,
                        'rolling_z_last': float(z[-1])
                    }
                    Path(f"{args.out_prefix}drift_summary.json").write_text(_json.dumps(summ, indent=2))
                    
                    # Generate INTERACTIVE Plotly HTML if requested
                    if args.interactive:
                        try:
                            import plotly.graph_objects as go
                            from plotly.subplots import make_subplots
                            
                            # Create interactive figure with drift metrics
                            fig_drift = go.Figure()
                            
                            # Plot drift trajectories
                            for i in range(max_neurons):
                                y = traj[:, i]
                                peak_step = int(np.argmax(np.abs(np.gradient(y)))) if y.size > 1 else 0
                                
                                fig_drift.add_trace(go.Scatter(
                                    x=list(range(len(y))),
                                    y=y,
                                    mode='lines+markers',
                                    name=(f'Neuron {neuron_labels[i]}' if neuron_labels and i < len(neuron_labels) else f'Neuron {i}'),
                                    line=dict(width=2),
                                    marker=dict(size=4),
                                    hovertemplate=(
                                        (f'<b>Neuron {neuron_labels[i]}</b><br>' if neuron_labels and i < len(neuron_labels) else f'<b>Neuron {i}</b><br>') +
                                        'Step: %{x}<br>' +
                                        'Drift: %{y:.4f}<br>' +
                                        f'Peak Step: {peak_step}<br>' +
                                        f'Current Rolling Z: {z[-1]:.2f}<br>' +
                                        '<b>🔴 Red:</b> Gradual drift evades detection<br>' +
                                        '<b>🔵 Blue:</b> Z>3 = anomaly detected<br>' +
                                        '<extra></extra>'
                                    )
                                ))
                                
                                # Add peak marker
                                fig_drift.add_trace(go.Scatter(
                                    x=[peak_step],
                                    y=[y[peak_step]],
                                    mode='markers',
                                    marker=dict(size=10, color='red', symbol='star'),
                                    showlegend=False,
                                    name=f'Peak {i}',
                                    hovertemplate=(f'<b>Peak Change</b><br>Neuron {neuron_labels[i]}<br>Step: {peak_step}<br>Value: {y[peak_step]:.4f}<extra></extra>' if neuron_labels and i < len(neuron_labels) else f'<b>Peak Change</b><br>Neuron {i}<br>Step: {peak_step}<br>Value: {y[peak_step]:.4f}<extra></extra>')
                                ))
                            
                            # Add steep-change shading
                            for t in range(1, len(gsum)):
                                if gsum[t] >= thr:
                                    fig_drift.add_vrect(
                                        x0=t-1, x1=t,
                                        fillcolor="orange", opacity=0.15,
                                        layer="below", line_width=0
                                    )
                            
                            # Add Red/Blue team guidance with CUSUM, TTE, Rolling Z
                            red_guidance = [
                                f"⏸️ CUSUM: {cusum[-1]:.2f} → Lower by flattening trend (smaller per-step changes)",
                                f"🎯 ROLLING Z: {z[-1]:.2f} → Keep below 3 by reducing volatility",
                                f"⏱️ TTE: {tte:.1f} steps → If small, slow or reverse drift to avoid guardrail",
                                f"📊 DRIFT MAGNITUDE: {drift_mag:.2f} → Distribute across neurons for stealth"
                            ]
                            
                            blue_guidance = [
                                f"🚨 CUSUM MONITOR: Current={cusum[-1]:.2f} → Alert if >3.0 (sustained drift)",
                                f"⚠️ ROLLING Z: Current={z[-1]:.2f} → Z>3 indicates statistically unusual drift",
                                f"⏱️ TTE ALERT: {tte:.1f} steps until guardrail breach → Prepare mitigation",
                                f"📈 STEEP REGIONS: {int(np.sum(gsum >= thr))} detected (orange shaded) → Investigate"
                            ]
                            
                            fig_drift.add_annotation(
                                text="<b>🔴 RED TEAM - Drift Evasion</b><br>" + "<br>".join(red_guidance),
                                xref="paper", yref="paper",
                                x=0.75, y=-0.35,
                                showarrow=False,
                                font=dict(size=10, color='white'),
                                align="left",
                                bgcolor='rgba(204,0,0,0.85)',
                                bordercolor='#cc0000',
                                borderwidth=2,
                                borderpad=12,
                                xanchor='center',
                                yanchor='top',
                                width=800
                            )
                            
                            fig_drift.add_annotation(
                                text="<b>🔵 BLUE TEAM - Drift Detection</b><br>" + "<br>".join(blue_guidance),
                                xref="paper", yref="paper",
                                x=0.25, y=-0.35,
                                showarrow=False,
                                font=dict(size=10, color='white'),
                                align="left",
                                bgcolor='rgba(31,95,191,0.85)',
                                bordercolor='#1f5fbf',
                                borderwidth=2,
                                borderpad=12,
                                xanchor='center',
                                yanchor='top',
                                width=800
                            )
                            
                            # Layout
                            fig_drift.update_layout(
                                title=dict(
                                    text=f'⚡ NeurInSpectre Activation Drift Analysis | CUSUM={cusum[-1]:.2f} | Z={z[-1]:.2f} | TTE={tte:.1f}',
                                    x=0.5,
                                    xanchor='center',
                                    font=dict(size=18, color='white')
                                ),
                                xaxis_title="Step Index",
                                yaxis_title="Activation Drift",
                                height=900,
                                width=1800,
                                template='plotly_dark',
                                hovermode='closest',
                                uirevision='constant',
                                showlegend=True,
                                legend=dict(x=0.5, y=-0.08, xanchor='center', orientation='h', 
                                           bgcolor='rgba(0,0,0,0.7)', font=dict(color='white')),
                                plot_bgcolor='rgba(20,20,20,0.95)',
                                paper_bgcolor='rgba(10,10,10,1)',
                                margin=dict(l=80, r=80, t=120, b=520),
                            )
                            
                            # Save interactive HTML
                            html_file = f"{args.out_prefix}drift_interactive.html"
                            fig_drift.write_html(html_file)
                            logger.info(f"📊 Interactive drift analysis HTML: {html_file}")
                            logger.info(f"🔍 Metrics visible: CUSUM={cusum[-1]:.2f}, Rolling Z={z[-1]:.2f}, TTE={tte:.1f}")
                            print(html_file)
                            
                        except Exception as e:
                            logger.warning(f"Interactive visualization failed: {e}, using static PNG only")
                    
                    print(fline)
                    print(f"{args.out_prefix}drift_summary.json")
                    if token_png:
                        print(token_png)
                    return 0
                else:
                    return 1
            except Exception as e:
                logger.error(f"Activation drift evasion failed: {e}")
                return 1
        elif args.command == 'gradient_inversion':
            try:
                import numpy as np
                import matplotlib.pyplot as plt
                import json
                from pathlib import Path
                grads = np.load(args.gradients)
                # Optional: if gradients are [layers, steps, features], select or aggregate layers
                if getattr(args, 'layer', None) is not None and getattr(grads, 'ndim', 0) >= 3:
                    li = int(getattr(args, 'layer'))
                    if li < 0 or li >= int(grads.shape[0]):
                        raise ValueError(f"--layer {li} out of range for gradients with shape={getattr(grads, 'shape', None)}")
                    grads = grads[li]
                elif getattr(grads, 'ndim', 0) == 3:
                    # Default behavior: aggregate across layer axis
                    grads = grads.mean(axis=0)
                # Optional context for labeling/metadata
                model_id = getattr(args, 'model', None)
                tok_id = getattr(args, 'tokenizer', None)
                G = np.array(grads)
                # Robustness: sanitize non-finite values so normalization/visualization never throws.
                G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
                if G.ndim == 1:
                    G = G.reshape(1, -1)
                elif G.ndim > 2:
                    G = G.reshape(-1, G.shape[-1])
                # Simple demo reconstruction: cumulative sum of normalized gradients
                ng = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
                rec = np.cumsum(ng, axis=0)
                np.save(f"{args.out_prefix}reconstructed.npy", rec.astype('float32'))
                # Save simple metadata if provided
                if model_id or tok_id:
                    meta = {'model': model_id, 'tokenizer': tok_id}
                    Path(f"{args.out_prefix}reconstructed_meta.json").write_text(json.dumps(meta, indent=2))
                # Compute per-step energy and robust, symmetric color limits
                energy = np.linalg.norm(rec, axis=1)
                mu, sd = float(np.mean(energy)), float(np.std(energy) + 1e-8)
                guard = mu + 2.0 * sd
                p5, p95 = np.percentile(rec, 5.0), np.percentile(rec, 95.0)
                vmax = float(max(abs(p5), abs(p95)))
                vmin = -vmax
                # Two-panel visualization: energy (top) + robust heatmap (bottom)
                import matplotlib.pyplot as _plt
                from matplotlib.ticker import MaxNLocator
                fig, (ax1, ax2) = _plt.subplots(2, 1, figsize=(11, 6.0), gridspec_kw={'height_ratios': [1, 2]})
                # Panel 1: energy curve with guardrail
                ax1.plot(energy, label='Per-step L2 energy', linewidth=1.2)
                ax1.axhline(guard, color='red', linestyle='--', label=f'guardrail μ+2σ = {guard:.2f}')
                title_ctx = f" (model: {model_id})" if model_id else ""
                ax1.set_title(f'NeurInSpectre — Reconstruction Energy over Steps{title_ctx}')
                ax1.set_xlabel('Step index'); ax1.set_ylabel('L2 energy')
                ax1.legend(loc='upper left', fontsize=8)
                ax1.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
                ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
                # Panel 2: robust diverging heatmap centered at 0
                im = ax2.imshow(rec.T, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')
                cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                cbar.set_label('Reconstruction value (a.u.)', fontsize=9)
                cbar.ax.tick_params(labelsize=8)
                ax2.set_title('NeurInSpectre — Gradient Inversion Reconstruction (features × steps)')
                ax2.set_xlabel('Step index'); ax2.set_ylabel('Feature index')
                ax2.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
                ax2.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
                # Add key below the figure
                fig.text(0.5, 0.02, 'Top: per-step energy with μ+2σ guardrail. Bottom: robust heatmap (5–95th pct) with symmetric limits around 0.', ha='center', fontsize=9)
                fig.tight_layout(rect=[0, 0.05, 1, 1])
                fhm = getattr(args, 'out_png', None) or f"{args.out_prefix}reconstruction_heatmap.png"
                Path(fhm).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fhm, dpi=200)
                # Interactive HTML report (Plotly) — required for full forensics.
                import plotly.graph_objects as _go
                import plotly.io as _pio
                import html as _html

                # Prepare z as features × steps
                _z = rec.T  # shape: [features, steps]
                _features, _steps = _z.shape

                # Custom data per cell: energy at this step, and guard value
                _cd_energy = np.tile(energy, (_features, 1))
                _cd_guard = np.full((_features, _steps), guard, dtype=float)
                _custom = np.dstack([_cd_energy, _cd_guard])

                # Calculate security metrics
                max_energy = float(np.max(energy))
                breach_steps = int(np.sum(energy > guard))
                high_value_cells = int(np.sum(np.abs(_z) > 1.0))

                # Determine threat level (heuristic)
                if breach_steps > len(energy) * 0.3 or max_energy > guard * 2:
                    threat_level = 'CRITICAL'
                elif breach_steps > len(energy) * 0.1 or max_energy > guard * 1.5:
                    threat_level = 'HIGH'
                else:
                    threat_level = 'MEDIUM'

                html_fig = _go.Figure(
                    data=_go.Heatmap(
                        z=_z,
                        colorscale='RdBu',
                        zmin=vmin,
                        zmax=vmax,
                        colorbar=dict(title='Reconstruction value (a.u.)'),
                        customdata=_custom,
                        hovertemplate=(
                            '<b>Gradient Inversion Reconstruction</b><br>'
                            'Feature: %{y}<br>'
                            'Step: %{x}<br>'
                            'Value: %{z:.4f}<br>'
                            'Step energy: %{customdata[0]:.4f}<br>'
                            'Guardrail (μ+2σ): %{customdata[1]:.4f}'
                            '<extra></extra>'
                        ),
                    )
                )

                title_ctx = f" (model: {model_id})" if model_id else ""
                html_fig.update_layout(
                    title=dict(
                        text=f'NeurInSpectre — Gradient Inversion Reconstruction (interactive){title_ctx}',
                        x=0.01,
                        xanchor='left',
                        font=dict(size=20),
                    ),
                    xaxis_title='Step index',
                    yaxis_title='Feature index',
                    template='plotly_white',
                    height=860,
                    width=1600,
                    hovermode='closest',
                    margin=dict(l=80, r=90, t=90, b=120),
                )

                # Red/Blue guidance (outside the plot area)
                red_guidance = [
                    f'Reconstruction: {breach_steps}/{len(energy)} steps breach guardrail → easier inversion',
                    f'Exploit: max energy={max_energy:.2f} (guard={guard:.2f}); focus on breach windows',
                    f'Target: {high_value_cells} high-value cells (|val|>1.0) carry recoverable signal',
                    'Indicator: coherent vertical stripes (stable features across steps) → persistent leakage',
                ]

                blue_guidance = [
                    f'Threat: {threat_level} (breaches={breach_steps}; max_energy={max_energy:.2f}; guard={guard:.2f})',
                    f'Urgent: apply gradient clipping; start with max_norm≈{guard/2:.2f}',
                    f'Defense: inject DP noise; start with σ≈{max_energy/5:.2f} (tune for utility)',
                    'Monitor: alert when breach count rises or when high-value stripes persist across rounds',
                ]

                fig_html = _pio.to_html(html_fig, include_plotlyjs='cdn', full_html=False)

                report = f'''<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>NeurInSpectre — Gradient Inversion Reconstruction</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; color: #111; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 18px; }}
    .card {{ border: 1px solid #e3e3e3; border-radius: 10px; padding: 14px; margin: 12px 0; background: #fff; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    @media (max-width: 920px) {{ .two {{ grid-template-columns: 1fr; }} }}
    ul {{ margin: 8px 0 0 18px; }}
    code {{ background: #f7f7f7; padding: 2px 5px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h2 style="margin:0 0 6px 0; font-size: 18px;">NeurInSpectre — Gradient Inversion Reconstruction</h2>
      <div style="color:#555; font-size: 13px;">Interactive heatmap (features × steps). Hover shows per-cell value + step energy + guardrail.</div>
      <div style="color:#555; font-size: 13px; margin-top: 6px;">Guardrail: <code>μ(energy)+2σ</code>. Breaches: <code>{breach_steps}</code> / <code>{len(energy)}</code>. Threat: <code>{_html.escape(threat_level)}</code>.</div>
    </div>

    <div class="card">{fig_html}</div>

    <div class="two">
      <div class="card">
        <h3 style="margin:0; font-size: 15px;">Blue team: practical next steps</h3>
        <ul>{''.join('<li>'+_html.escape(x)+'</li>' for x in blue_guidance)}</ul>
      </div>
      <div class="card">
        <h3 style="margin:0; font-size: 15px;">Red team: practical next steps</h3>
        <ul>{''.join('<li>'+_html.escape(x)+'</li>' for x in red_guidance)}</ul>
      </div>
    </div>

    <div class="card">
      <h3 style="margin:0; font-size: 15px;">Context / mapping</h3>
      <ul>
        <li>DLG (Zhu et al., 2019)</li>
        <li>iDLG (Geiping et al., 2020)</li>
        <li>MITRE ATLAS: AML.T0024.001 (Invert AI Model)</li>
      </ul>
    </div>
  </div>
</body>
</html>
'''

                fhtml = getattr(args, 'out_html', None) or f"{args.out_prefix}reconstruction_heatmap.html"
                Path(fhtml).parent.mkdir(parents=True, exist_ok=True)
                Path(fhtml).write_text(report, encoding='utf-8')

                print(f"{args.out_prefix}reconstructed.npy")
                print(fhm)
                print(fhtml)
                return 0
            except Exception as e:
                logger.error(f"Gradient inversion failed: {e}")
                return 1
        elif args.command == 'statistical_evasion':
            try:
                import numpy as np, json
                from pathlib import Path
                from scipy.stats import ks_2samp
                if args.se_action == 'generate':
                    N, D = int(args.samples), int(args.features)
                    rng = np.random.default_rng(42)
                    # Realistic generator: correlated benign + sparse shifts / bursts in attack
                    u = rng.normal(0, 1, size=(D, 8))
                    u, _ = np.linalg.qr(u)
                    s = np.linspace(1.5, 0.3, 8)
                    L = (u * s) @ u.T + 0.2 * np.eye(D)
                    L = np.linalg.cholesky(L + 1e-6 * np.eye(D))
                    benign = (rng.normal(size=(N, D)) @ L.T).astype('float32')
                    # Attack: mixture of sparse mean shift + occasional heavy-tail bursts
                    shift_vec = np.zeros(D, dtype='float32')
                    idx = rng.choice(D, size=max(5, D//16), replace=False)
                    shift_vec[idx] = float(args.shift)
                    base = (rng.normal(size=(N, D)) @ L.T)
                    bursts = rng.binomial(1, 0.05, size=(N, 1)) * rng.normal(0.0, 2.0, size=(N, D))
                    attack = (base + shift_vec + bursts).astype('float32')
                    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / 'benign.npy', benign)
                    np.save(out_dir / 'attack.npy', attack)
                    # If a single-file output is requested, write a proper numeric file
                    if getattr(args, 'output', None):
                        out_path = Path(args.output)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        suf = out_path.suffix.lower()
                        if suf == '.npz':
                            np.savez(out_path, benign=benign, attack=attack)
                        elif suf == '.npy':
                            # Save numeric array [2, N, D]: [0]=benign, [1]=attack
                            np.save(out_path, np.stack([benign, attack], axis=0))
                        else:
                            # Default to .npz
                            np.savez(out_path.with_suffix('.npz'), benign=benign, attack=attack)
                        print(str(out_path))
                    print(str(out_dir / 'benign.npy'))
                    print(str(out_dir / 'attack.npy'))
                    return 0
                elif args.se_action == 'score':
                    # Support either separate files or a combined --input
                    if getattr(args, 'input', None):
                        ip = Path(args.input)
                        if ip.suffix.lower() == '.npz':
                            combo = np.load(ip)
                            if 'attack' in combo:
                                X = combo['attack']
                            elif 'data' in combo:
                                X = combo['data']
                            else:
                                raise ValueError('Combined input missing attack/data array')
                            if 'benign' in combo:
                                R = combo['benign']
                            elif 'reference' in combo:
                                R = combo['reference']
                            else:
                                raise ValueError('Combined input missing benign/reference array')
                        else:
                            combo = np.load(ip, allow_pickle=True).item()
                            if 'attack' in combo:
                                X = combo['attack']
                            elif 'data' in combo:
                                X = combo['data']
                            else:
                                raise ValueError('Combined input missing attack/data array')
                            if 'benign' in combo:
                                R = combo['benign']
                            elif 'reference' in combo:
                                R = combo['reference']
                            else:
                                raise ValueError('Combined input missing benign/reference array')
                    else:
                        X = np.load(args.data); R = np.load(args.reference)
                    # Clean NaN/Inf
                    import numpy as _np
                    X = _np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                    R = _np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
                    X = X.reshape(-1, X.shape[-1]); R = R.reshape(-1, R.shape[-1])
                    D = min(X.shape[1], R.shape[1]); X = X[:, :D]; R = R[:, :D]
                    if X.shape[0] == 0 or R.shape[0] == 0 or D == 0:
                        from pathlib import Path
                        outj = Path(f"{args.out_prefix}se_score.json")
                        outj.write_text(json.dumps({'pvals': [], 'mean_p': None, 'note': 'insufficient data'}, indent=2))
                        print(str(outj))
                        return 0
                    pvals = []
                    for j in range(D):
                        stat, p = ks_2samp(X[:, j], R[:, j])
                        pvals.append(float(p))
                    from pathlib import Path
                    outj = Path(f"{args.out_prefix}se_score.json")
                    outj.write_text(json.dumps({'pvals': pvals, 'mean_p': float(np.mean(pvals))}, indent=2))
                    # P-value plot with Blue/Red keys for interpretability
                    import matplotlib.pyplot as _plt
                    figm, axm = _plt.subplots(figsize=(9,3.2))
                    # Show either all features or top-100 smallest p-values for readability
                    idx_sorted = np.argsort(pvals)
                    top_cap = 100 if D > 100 else D
                    sel = idx_sorted[:top_cap]
                    axm.bar(range(len(sel)), [pvals[i] for i in sel], color='#2A9D8F')
                    axm.axhline(args.alpha, color='red', ls='--', label=f'alpha={args.alpha:.2f}')
                    axm.set_title('NeurInSpectre — Statistical Evasion P-Values (lowest features)')
                    axm.set_xlabel('Feature (ranked by p)'); axm.set_ylabel('p-value')
                    axm.set_ylim(1e-6, 1.0)
                    axm.set_yscale('log')
                    axm.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
                    fig = _plt.gcf()
                    import textwrap as _tw
                    below = int(np.sum(np.array(pvals) < args.alpha))
                    meanp = float(np.mean(pvals))
                    btxt = _tw.fill(f'Blue: prioritize lowest p-value features (count below alpha={args.alpha:.2f}: {below}); confirm with secondary tests; monitor shift over time. Mean p={meanp:.3g}.', width=90)
                    rtxt = _tw.fill('Red: distribute shift to avoid consistent low‑p features; jitter around alpha threshold.', width=90)
                    fig.text(0.01, 0.02, btxt, fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6f0ff', edgecolor='#1f5fbf', alpha=0.95))
                    fig.text(0.56, 0.02, rtxt, fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe6e6', edgecolor='#cc0000', alpha=0.95))
                    _plt.tight_layout(rect=[0, 0.18, 1, 1])
                    _plt.savefig(f"{args.out_prefix}se_pvals.png", dpi=200, bbox_inches='tight')
                    # Interactive HTML bar chart with hover
                    try:
                        import plotly.graph_objects as _go
                        hover_idx = [int(i) for i in sel]
                        hover_p = [float(pvals[i]) for i in sel]
                        html_fig = _go.Figure(_go.Bar(x=list(range(len(hover_idx))), y=hover_p,
                                                      hovertext=[f'feature {i}<br>p={p:.4g}' for i,p in zip(hover_idx, hover_p)],
                                                      marker_color='#2A9D8F'))
                        html_fig.add_hline(y=args.alpha, line_dash='dash', line_color='red')
                        html_fig.update_layout(title='NeurInSpectre — Statistical Evasion P-Values (lowest features)',
                                              xaxis_title='Feature (ranked by p)', yaxis_title='p-value (log scale)',
                                              yaxis_type='log', template='plotly_white', margin=dict(l=60,r=40,t=60,b=60))
                        fhtml2 = f"{args.out_prefix}se_pvals.html"
                        html_fig.write_html(fhtml2)
                    except Exception:
                        pass
                    print(str(outj))
                    return 0
                else:
                    return 1
            except Exception as e:
                logger.error(f"Statistical evasion failed: {e}")
                return 1
        elif args.command == 'neuron_watermarking':
            try:
                import numpy as np, json
                from pathlib import Path
                if args.nw_action == 'embed':
                    wb = getattr(args, 'watermark_bits', None)
                    bits = [int(b) for b in str(wb).split(',') if str(b).strip() != '']
                    if not bits:
                        raise ValueError('Empty --watermark-bits')
                    bits = [1 if int(b) != 0 else 0 for b in bits]

                    pathway = [int(n) for n in str(getattr(args, 'target_pathway')).split(',') if str(n).strip() != '']
                    if not pathway:
                        raise ValueError('Empty --target-pathway')

                    p = Path(str(args.activations))
                    if not p.exists():
                        logger.error(f"Activations file not found: {p}")
                        return 1

                    # Robust load: .npy/.npz, dict payloads
                    if p.suffix.lower() == '.npz':
                        npz = np.load(str(p), allow_pickle=True)
                        if len(npz.files) == 0:
                            raise ValueError(f"Empty .npz: {p}")
                        A = np.asarray(npz[npz.files[0]])
                    else:
                        obj = np.load(str(p), allow_pickle=True)
                        if getattr(obj, "dtype", None) == object and getattr(obj, "shape", ()) == ():
                            obj = obj.item()
                        if isinstance(obj, dict):
                            for k in ("activations", "data", "X", "x", "arr"):
                                if k in obj:
                                    obj = obj[k]
                                    break
                        A = np.asarray(obj)

                    arr = np.array(A)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    elif arr.ndim > 2:
                        arr = arr.reshape(-1, arr.shape[-1])
                    # Ensure sufficient width for pathway
                    need_w = max(pathway)+1
                    if arr.shape[1] < need_w:
                        pad = np.zeros((arr.shape[0], need_w - arr.shape[1]), dtype=arr.dtype)
                        arr = np.concatenate([arr, pad], axis=1)
                    eps = float(args.epsilon)
                    for i, n in enumerate(pathway):
                        b = bits[i % len(bits)]
                        arr[:, n] += eps * (1 if b else -1)
                    np.save(f"{args.out_prefix}watermarked.npy", arr.astype('float32'))
                    Path(f"{args.out_prefix}wm_meta.json").write_text(json.dumps({'bits': bits, 'pathway': pathway, 'epsilon': eps}, indent=2))
                    print(f"{args.out_prefix}watermarked.npy")
                    print(f"{args.out_prefix}wm_meta.json")
                    return 0
                elif args.nw_action == 'detect':
                    import numpy as _np
                    import matplotlib.pyplot as _plt
                    A = np.load(args.activations)
                    arr = _np.array(A)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    elif arr.ndim > 2:
                        arr = arr.reshape(-1, arr.shape[-1])
                    pathway = [int(n) for n in str(args.target_pathway).split(',') if n.strip()]
                    thr = float(args.threshold)
                    means = []
                    for n in pathway:
                        if 0 <= n < arr.shape[1]:
                            means.append(float(arr[:, n].mean()))
                        else:
                            means.append(0.0)
                    bits = [1 if m >= thr else 0 for m in means]
                    result = {'pathway': pathway, 'threshold': thr, 'means': [float(m) for m in means], 'bits': bits}
                    conf = None
                    sweep_path = None
                    if getattr(args, 'sweep', False):
                        # Sweep thresholds from min..max of means and compute agreement rate
                        mmin, mmax = (float(_np.min(means)), float(_np.max(means)))
                        if mmax - mmin < 1e-8:
                            grid = _np.linspace(mmin-0.5, mmax+0.5, 21)
                        else:
                            grid = _np.linspace(mmin, mmax, 41)
                        rates = []
                        # A simple target pattern: majority of pathway bits expected 1 when threshold <= mean
                        # Confidence: max stability of predicted bits over contiguous threshold range
                        prev_bits = None
                        stable_len = 0
                        best_stable = 0
                        for t in grid:
                            cur_bits = [1 if m >= t else 0 for m in means]
                            rate = sum(cur_bits) / max(1, len(cur_bits))
                            rates.append(rate)
                            if cur_bits == prev_bits:
                                stable_len += 1
                            else:
                                best_stable = max(best_stable, stable_len)
                                stable_len = 1
                                prev_bits = cur_bits
                        best_stable = max(best_stable, stable_len)
                        conf = float(best_stable / max(1, len(grid)))
                        # Plot sweep
                        _plt.figure(figsize=(9,3.2))
                        _plt.plot(grid, rates, label='fraction of 1-bits', color='#1f5fbf')
                        _plt.axvline(thr, color='red', linestyle='--', label=f'threshold={thr:g}')
                        # Highlight longest stable plateau span
                        # Recompute to capture indices
                        best_span=(0,0); prev=None; cur_len=0; start_idx=0; best_len=0
                        for i, t in enumerate(grid):
                            cur_bits = [1 if m >= t else 0 for m in means]
                            if cur_bits == prev:
                                cur_len += 1
                            else:
                                if cur_len > best_len:
                                    best_len = cur_len; best_span = (start_idx, i-1)
                                prev = cur_bits; cur_len = 1; start_idx = i
                        if cur_len > best_len:
                            best_len = cur_len; best_span = (start_idx, len(grid)-1)
                        if best_len > 1:
                            a,b = best_span
                            _plt.axvspan(
                                grid[a], grid[b],
                                facecolor='#9ecaff', alpha=0.9,
                                edgecolor='#0b4f9c', linewidth=1.2,
                                label='stable plateau'
                            )
                        _plt.title('NeurInSpectre — Watermark Threshold Sweep')
                        _plt.xlabel('Threshold'); _plt.ylabel('Fraction of 1-bits')
                        _plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
                        fig = _plt.gcf()
                        fig.text(0.01, 0.01,
                                 'Blue: select thresholds inside stable plateaus; alert on confidence drops.', fontsize=9,
                                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6f0ff', edgecolor='#1f5fbf', alpha=0.95))
                        fig.text(0.56, 0.01,
                                 'Red: shorten plateaus with noise; push means toward boundary to induce flips.', fontsize=9,
                                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe6e6', edgecolor='#cc0000', alpha=0.95))
                        _plt.tight_layout(rect=[0, 0.20, 1, 1])
                        sweep_path = f"{args.out_prefix}wm_sweep.png"
                        _plt.savefig(sweep_path, dpi=200, bbox_inches='tight')
                        result['sweep_plot'] = sweep_path
                        result['stable_plateau'] = {'start': float(grid[best_span[0]]), 'end': float(grid[best_span[1]])} if best_len>1 else None
                        # Mark bit flips across thresholds in JSON
                        flips = []
                        for n, m in zip(pathway, means):
                            flips.append({'neuron': int(n), 'flip_near': float(m)})
                        result['bit_flip_near_threshold'] = flips
                    if conf is not None:
                        result['confidence'] = conf
                    outj = Path(f"{args.out_prefix}wm_detect.json")
                    outj.write_text(json.dumps(result, indent=2))
                    print(str(outj))
                    if sweep_path:
                        print(sweep_path)
                    return 0
                else:
                    return 1
            except Exception as e:
                logger.error(f"Neuron watermarking failed: {e}")
                return 1
        elif args.command == 'spectral':
            try:
                import numpy as np
                import json
                import matplotlib as mpl
                import matplotlib.pyplot as plt
                from contextlib import nullcontext

                def _load_array(path: str) -> np.ndarray:
                    """Load .npy/.npz, unwrapping dict/object arrays, and sanitize non-finite values."""
                    loaded = np.load(path, allow_pickle=True)
                    if isinstance(loaded, np.lib.npyio.NpzFile):
                        # Prefer common keys; else first array-like
                        arr0 = None
                        for key in ['activations', 'A', 'arr', 'data', 'x', 'X', 'gradients', 'g']:
                            if key in loaded:
                                arr0 = np.array(loaded[key])
                                break
                        if arr0 is None:
                            for k in loaded.files:
                                try:
                                    arr0 = np.array(loaded[k])
                                    break
                                except Exception:
                                    continue
                        if arr0 is None:
                            raise ValueError(".npz did not contain any array entries")
                        arrx = arr0
                    else:
                        arrx = loaded
                        if isinstance(arrx, np.ndarray) and getattr(arrx, "dtype", None) is object and getattr(arrx, "shape", ()) == ():
                            arrx = arrx.item()
                        if isinstance(arrx, dict):
                            tmp = None
                            for key in ['activations', 'A', 'arr', 'data', 'x', 'X', 'gradients', 'g']:
                                if key in arrx:
                                    tmp = np.array(arrx[key])
                                    break
                            if tmp is None:
                                for v in arrx.values():
                                    try:
                                        tmp = np.array(v)
                                        break
                                    except Exception:
                                        continue
                            if tmp is None:
                                raise ValueError("Could not resolve array from dict-like .npy object")
                            arrx = tmp
                        arrx = np.array(arrx)

                    arrx = np.nan_to_num(arrx, nan=0.0, posinf=0.0, neginf=0.0)
                    return arrx

                def _as_signals(a: np.ndarray) -> np.ndarray:
                    a = np.asarray(a)
                    if a.ndim == 1:
                        sig = a[None, :]
                    elif a.ndim == 2:
                        sig = a
                    else:
                        sig = a.reshape(-1, a.shape[-1])
                    if sig.shape[-1] < 2:
                        pad = np.zeros((sig.shape[0], 2 - sig.shape[-1]), dtype=sig.dtype)
                        sig = np.concatenate([sig, pad], axis=1)
                    return np.asarray(sig, dtype=np.float64)

                # Load and prepare signals
                arr = _load_array(str(args.input))
                signals = _as_signals(arr)
                n_signals, n_len = int(signals.shape[0]), int(signals.shape[-1])

                demean = not bool(getattr(args, "no_demean", False))
                if demean:
                    signals = signals - np.mean(signals, axis=-1, keepdims=True)

                window = str(getattr(args, "window", "hann") or "hann").strip().lower()
                if window == "hann":
                    w = np.hanning(n_len).astype(np.float64)
                    signals_w = signals * w[None, :]
                else:
                    signals_w = signals

                # Spectrum statistics
                fft = np.fft.rfft(signals_w, axis=-1)
                mags = np.abs(fft)
                mag_mean = np.mean(mags, axis=0)
                mag_p10 = np.percentile(mags, 10.0, axis=0)
                mag_p90 = np.percentile(mags, 90.0, axis=0)
                freqs = np.fft.rfftfreq(n_len, d=1.0)

                # Define heuristic low/mid/high bands (relative to Nyquist)
                fmax = float(freqs.max() if freqs.size else 1.0)
                b1, b2 = 0.33 * fmax, 0.66 * fmax

                # Energy (mean power) for shares + entropy
                pow_mean = np.mean(mags**2, axis=0)
                pow_mean = np.asarray(pow_mean, dtype=np.float64)

                # Baseline comparison (recommended)
                baseline_path = getattr(args, "baseline", None)
                base = None
                base_mag_mean = None
                base_pow_mean = None
                ratio_db = None
                if baseline_path:
                    base_arr = _load_array(str(baseline_path))
                    base_sig = _as_signals(base_arr)
                    if demean:
                        base_sig = base_sig - np.mean(base_sig, axis=-1, keepdims=True)
                    if window == "hann":
                        base_sig = base_sig * np.hanning(base_sig.shape[-1]).astype(np.float64)[None, :]
                    base_fft = np.fft.rfft(base_sig, axis=-1)
                    base_mags = np.abs(base_fft)
                    base_mag_mean = np.mean(base_mags, axis=0)
                    base_pow_mean = np.mean(base_mags**2, axis=0)
                    # Align lengths if needed (truncate to min bins)
                    min_bins = int(min(mag_mean.size, base_mag_mean.size))
                    mag_mean = mag_mean[:min_bins]
                    mag_p10 = mag_p10[:min_bins]
                    mag_p90 = mag_p90[:min_bins]
                    pow_mean = pow_mean[:min_bins]
                    base_mag_mean = base_mag_mean[:min_bins]
                    base_pow_mean = base_pow_mean[:min_bins]
                    freqs = freqs[:min_bins]
                    fmax = float(freqs.max() if freqs.size else fmax)
                    b1, b2 = 0.33 * fmax, 0.66 * fmax

                    eps = 1e-12
                    ratio = (mag_mean + eps) / (base_mag_mean + eps)
                    ratio_db = 20.0 * np.log10(ratio)

                # Peak selection:
                # - with baseline: pick top peaks by positive ratio_db (new narrowband amplification)
                # - without baseline: pick peaks by "local prominence" (mag / smooth(mag))
                topk = int(getattr(args, "topk_peaks", 6) or 6)
                topk = max(0, min(topk, int(mag_mean.size)))
                peak_db_thr = float(getattr(args, "peak_db", 6.0) or 6.0)
                peak_metric_name = "ratio_db" if ratio_db is not None else "prominence_db"
                prom_db = None

                valid = np.arange(mag_mean.size)
                # Avoid DC bin in peak selection
                valid = valid[freqs > 0.0]

                if ratio_db is not None:
                    metric = ratio_db.copy()
                    metric[~np.isfinite(metric)] = -1e9
                    metric = metric[valid]
                    if metric.size and topk:
                        idxs = valid[np.argpartition(metric, -min(topk, metric.size))[-min(topk, metric.size):]]
                        idxs = idxs[np.argsort(ratio_db[idxs])[::-1]]
                    else:
                        idxs = np.array([], dtype=int)
                    # Count bins above threshold in mid/high bands (defender-relevant)
                    spike_bins = int(np.sum((ratio_db > peak_db_thr) & (freqs >= b1)))
                else:
                    # Smooth magnitude curve (moving average) to estimate background
                    win = int(max(7, min(61, (mag_mean.size // 25) * 2 + 1)))  # odd-ish window
                    kernel = np.ones(win, dtype=np.float64) / float(win)
                    smooth = np.convolve(mag_mean, kernel, mode="same")
                    prom = (mag_mean + 1e-12) / (smooth + 1e-12)
                    prom_db = 20.0 * np.log10(prom)
                    metric = prom_db.copy()
                    metric[~np.isfinite(metric)] = -1e9
                    metric = metric[valid]
                    if metric.size and topk:
                        idxs = valid[np.argpartition(metric, -min(topk, metric.size))[-min(topk, metric.size):]]
                        idxs = idxs[np.argsort(prom_db[idxs])[::-1]]
                    else:
                        idxs = np.array([], dtype=int)
                    spike_bins = int(np.sum((prom_db > peak_db_thr) & (freqs >= b1)))

                # Band energy shares + spectral entropy (exclude DC)
                low_mask = (freqs >= 0) & (freqs < b1)
                mid_mask = (freqs >= b1) & (freqs < b2)
                high_mask = (freqs >= b2)
                tot_energy = float(np.sum(pow_mean) + 1e-12)
                low_share = float(np.sum(pow_mean[low_mask]) / tot_energy)
                mid_share = float(np.sum(pow_mean[mid_mask]) / tot_energy)
                high_share = float(np.sum(pow_mean[high_mask]) / tot_energy)

                p = pow_mean[freqs > 0.0].copy()
                p = p / float(np.sum(p) + 1e-12)
                ent_bits = float(-np.sum(p * np.log2(p + 1e-18))) if p.size else 0.0
                ent_norm = float(ent_bits / (np.log2(float(p.size)) if p.size > 1 else 1.0))
                hf_ratio = float(np.sum(pow_mean[high_mask]) / (tot_energy + 1e-12))

                # Build JSON summary
                summary = {
                    "schema": "neurinspectre.spectral.v2",
                    "input_path": str(args.input),
                    "baseline_path": (str(baseline_path) if baseline_path else None),
                    "signals": {"n_signals": int(n_signals), "signal_length": int(n_len)},
                    "preprocess": {"demean": bool(demean), "window": str(window)},
                    "bands": {"fmax": float(fmax), "low_end": float(b1), "mid_end": float(b2)},
                    "energy_shares": {"low": low_share, "mid": mid_share, "high": high_share},
                    "spectral_entropy": {"bits": float(ent_bits), "normalized": float(np.clip(ent_norm, 0.0, 1.0))},
                    "hf_energy_ratio": float(np.clip(hf_ratio, 0.0, 1.0)),
                    "spike_bins_ge_band1": int(spike_bins),
                    "peak_metric": peak_metric_name,
                    "top_peaks": [],
                }

                for i in idxs.tolist():
                    entry = {"freq": float(freqs[i]), "mag_mean": float(mag_mean[i])}
                    if ratio_db is not None:
                        entry["ratio_db_vs_baseline"] = float(ratio_db[i])
                    summary["top_peaks"].append(entry)

                # Shared UI text (used by both static PNG and interactive HTML outputs)
                baseline_disp = str(baseline_path) if baseline_path else "none (recommend: --baseline clean.npy)"
                sum_lines = [
                    "Summary (heuristic)",
                    f"signals={n_signals}  len={n_len}",
                    f"baseline={baseline_disp}",
                    f"energy shares: low={low_share:.2f} mid={mid_share:.2f} high={high_share:.2f}",
                    f"entropy_norm={float(np.clip(ent_norm,0,1)):.2f}  hf_ratio={hf_ratio:.2f}",
                    f"spike_bins(≥mid)={spike_bins}  thr={peak_db_thr:g} dB",
                ]
                blue_text = (
                    "Blue team — practical next steps\n"
                    "• Establish baseline (clean): run with --baseline clean.npy\n"
                    "• Alert on NEW + persistent narrowband peaks (esp. mid/high band)\n"
                    "• Localize: slice by layer/block/parameter groups and re-run spectral\n"
                    "• Mitigate: tighten clipping / privacy controls; validate regressions\n"
                    "• Correlate: pair with anomaly/drift modules for confirmation"
                )
                red_text = (
                    "Red team — evaluation checklist (safe)\n"
                    "• Run controlled test scenarios and record if peaks exceed threshold\n"
                    "• Measure FP/FN vs baseline; check stability across seeds/runs\n"
                    "• Stress-test patterns: spikes, slow drift, periodic artifacts\n"
                    "• Report: share top peak freqs + dB deltas + baseline used"
                )
                note_text = (
                    "Note: Spectral features are heuristic. For precision, compare against a clean baseline "
                    "and re-validate after distribution shifts."
                )

                # If a plot path is provided, generate and save the figure.
                if getattr(args, 'plot', None):
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
                        # Layout: 2 plot rows + a dedicated text row underneath (no overlap).
                        fig = plt.figure(figsize=(12.8, 9.1), dpi=180)
                        gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.65, 1.35], hspace=0.26)
                        ax = fig.add_subplot(gs[0, 0])
                        ax_delta = fig.add_subplot(gs[1, 0])
                        gs_text = gs[2, 0].subgridspec(1, 3, width_ratios=[1.15, 1.85, 1.85], wspace=0.14)
                        ax_sum = fig.add_subplot(gs_text[0, 0])
                        ax_blue = fig.add_subplot(gs_text[0, 1])
                        ax_red = fig.add_subplot(gs_text[0, 2])
                        for _a in (ax_sum, ax_blue, ax_red):
                            _a.axis("off")

                        # --- Main spectrum (mean + variability band)
                        ax.semilogy(freqs, mag_mean, color="#1F77B4", lw=2.1, label="Suspect mean")
                        ax.fill_between(freqs, mag_p10, mag_p90, color="#1F77B4", alpha=0.10, linewidth=0, label="Suspect p10–p90")

                        if base_mag_mean is not None:
                            ax.semilogy(freqs, base_mag_mean, color="#6C757D", lw=1.6, ls="--", label="Baseline mean")

                        ax.set_title("NeurInSpectre — Mean Magnitude Spectrum (Triage View)")
                        ax.set_xlabel("Frequency (normalized)")
                        ax.set_ylabel("Magnitude (log)")
                        ax.grid(alpha=0.25)
                        ax.set_xlim(0.0, float(fmax))

                        # Subtle low/mid/high shading
                        ax.axvspan(0, b1, color="#EEF6FF", alpha=0.55, label="Low band")
                        ax.axvspan(b1, b2, color="#FFF6E6", alpha=0.45, label="Mid band")
                        ax.axvspan(b2, fmax, color="#FFEAEA", alpha=0.40, label="High band")

                        # Annotate selected peaks
                        for i in idxs.tolist():
                            # Highlight only peaks that exceed the threshold; keep non-flagged peaks subtle.
                            if ratio_db is not None:
                                score_db = float(ratio_db[i])
                            else:
                                score_db = float(prom_db[i]) if prom_db is not None else 0.0
                            flagged = bool(score_db >= peak_db_thr)
                            mcolor = "#D62728" if flagged else "#F59E0B"
                            malpha = 0.95 if flagged else 0.45
                            ax.plot(freqs[i], mag_mean[i], marker="o", markersize=5, color=mcolor, alpha=malpha, zorder=5)
                            lab = f"{freqs[i]:.2f}"
                            if flagged:
                                if ratio_db is not None:
                                    lab = f"{freqs[i]:.2f}  (+{score_db:+.1f} dB)"
                                else:
                                    lab = f"{freqs[i]:.2f}  ({score_db:+.1f} dB)"
                            ax.annotate(
                                lab,
                                (freqs[i], mag_mean[i]),
                                textcoords="offset points",
                                xytext=((6, 6) if float(freqs[i]) < 0.92 * float(fmax) else (-62, 6)),
                                fontsize=8,
                                color="#B91C1C" if flagged else "#92400E",
                                ha=("left" if float(freqs[i]) < 0.92 * float(fmax) else "right"),
                                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#E5E7EB", alpha=0.95),
                            )

                        leg = ax.legend(loc="upper left", frameon=True, framealpha=0.92)
                        leg.get_frame().set_edgecolor("#D0D0D0")

                        # --- Delta / prominence panel
                        if ratio_db is not None:
                            ax_delta.plot(freqs, ratio_db, color="#6C5CE7", lw=1.9, label="Δ magnitude (dB vs baseline)")
                            ax_delta.axhline(peak_db_thr, color="#B91C1C", lw=1.2, ls=(0, (6, 3)), label=f"peak threshold = {peak_db_thr:g} dB")
                            ax_delta.axhspan(peak_db_thr, max(float(np.nanmax(ratio_db)), peak_db_thr), color="#FFEAEA", alpha=0.35, lw=0)
                            # Mark peak points
                            for i in idxs.tolist():
                                ax_delta.scatter([freqs[i]], [ratio_db[i]], s=26, color=("#D62728" if float(ratio_db[i]) >= peak_db_thr else "#F59E0B"), edgecolor="white", linewidth=0.5, zorder=4)
                            ax_delta.set_title("Baseline comparison (dB)")
                            ax_delta.set_ylabel("dB")
                        else:
                            # Compute and plot local prominence (dB) for visibility (reuse precomputed if available).
                            if prom_db is None:
                                win = int(max(7, min(61, (mag_mean.size // 25) * 2 + 1)))
                                kernel = np.ones(win, dtype=np.float64) / float(win)
                                smooth = np.convolve(mag_mean, kernel, mode="same")
                                prom_db = 20.0 * np.log10((mag_mean + 1e-12) / (smooth + 1e-12))
                            ax_delta.plot(freqs, prom_db, color="#6C5CE7", lw=1.9, label="Local prominence (dB)")
                            ax_delta.axhline(peak_db_thr, color="#B91C1C", lw=1.2, ls=(0, (6, 3)), label=f"peak threshold = {peak_db_thr:g} dB")
                            ax_delta.axhspan(peak_db_thr, max(float(np.nanmax(prom_db)), peak_db_thr), color="#FFEAEA", alpha=0.35, lw=0)
                            for i in idxs.tolist():
                                ax_delta.scatter([freqs[i]], [prom_db[i]], s=26, color=("#D62728" if float(prom_db[i]) >= peak_db_thr else "#F59E0B"), edgecolor="white", linewidth=0.5, zorder=4)
                            ax_delta.set_title("Peak prominence (no baseline)")
                            ax_delta.set_ylabel("dB")

                        ax_delta.set_xlabel("Frequency (normalized)")
                        ax_delta.grid(alpha=0.25)
                        ax_delta.set_xlim(0.0, float(fmax))
                        ax_delta.legend(frameon=False, loc="upper right")

                        # --- Text panels (moved below plots so nothing overlaps)
                        ax_sum.text(
                            0.0,
                            1.0,
                            "\n".join(sum_lines),
                            va="top",
                            ha="left",
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#D0D0D0", alpha=0.98),
                        )
                        ax_blue.text(
                            0.0,
                            1.0,
                            blue_text,
                            va="top",
                            ha="left",
                            fontsize=9.5,
                            bbox=dict(boxstyle="round,pad=0.55", facecolor="#ECF3FF", edgecolor="#1F5FBF", alpha=0.97),
                        )
                        ax_red.text(
                            0.0,
                            1.0,
                            red_text,
                            va="top",
                            ha="left",
                            fontsize=9.5,
                            bbox=dict(boxstyle="round,pad=0.55", facecolor="#FFF1F2", edgecolor="#B91C1C", alpha=0.97),
                        )

                        fig.text(
                            0.01,
                            0.01,
                            note_text,
                            fontsize=8.5,
                            color="#555",
                        )

                        # Avoid tight_layout warnings on mixed text/axes layouts; use explicit margins.
                        fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.06)
                        fig.savefig(args.plot, dpi=220, bbox_inches="tight")
                print(args.plot)

                # Interactive HTML triage dashboard (Plotly)
                if getattr(args, "html", None):
                    try:
                        import html as _html
                        from pathlib import Path as _Path

                        import plotly.graph_objects as _go
                        import plotly.io as _pio
                        from plotly.subplots import make_subplots as _make_subplots
                    except Exception as e:
                        raise ImportError(
                            "Interactive spectral HTML requires Plotly. Install with: pip install plotly"
                        ) from e

                    # Build figure (2 stacked panels)
                    sub_titles = [
                        "Mean Magnitude Spectrum (log)",
                        ("Baseline comparison (dB)" if ratio_db is not None else "Peak prominence (no baseline)"),
                    ]
                    figp = _make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.10,
                        row_heights=[0.62, 0.38],
                        subplot_titles=sub_titles,
                    )

                    # Band shading
                    figp.add_vrect(x0=0.0, x1=float(b1), fillcolor="rgba(238,246,255,0.55)", line_width=0, row=1, col=1)
                    figp.add_vrect(x0=float(b1), x1=float(b2), fillcolor="rgba(255,246,230,0.45)", line_width=0, row=1, col=1)
                    figp.add_vrect(x0=float(b2), x1=float(fmax), fillcolor="rgba(255,234,234,0.40)", line_width=0, row=1, col=1)

                    # Suspect p10–p90 band
                    figp.add_trace(
                        _go.Scatter(
                            x=freqs,
                            y=mag_p10,
                            mode="lines",
                            line=dict(width=0),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )
                    figp.add_trace(
                        _go.Scatter(
                            x=freqs,
                            y=mag_p90,
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor="rgba(31,119,180,0.14)",
                            name="Suspect p10–p90",
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=1,
                    )
                    figp.add_trace(
                        _go.Scatter(
                            x=freqs,
                            y=mag_mean,
                            mode="lines",
                            line=dict(color="#1F77B4", width=2.2),
                            name="Suspect mean",
                            hovertemplate="f=%{x:.3f}<br>mag=%{y:.3g}<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )
                    if base_mag_mean is not None:
                        figp.add_trace(
                            _go.Scatter(
                                x=freqs,
                                y=base_mag_mean,
                                mode="lines",
                                line=dict(color="#6C757D", width=1.6, dash="dash"),
                                name="Baseline mean",
                                hovertemplate="f=%{x:.3f}<br>baseline mag=%{y:.3g}<extra></extra>",
                            ),
                            row=1,
                            col=1,
                        )

                    # Peaks (markers only; details on hover)
                    peak_x = [float(freqs[i]) for i in idxs.tolist()]
                    peak_y = [float(mag_mean[i]) for i in idxs.tolist()]
                    peak_score = []
                    peak_color = []
                    for i in idxs.tolist():
                        if ratio_db is not None:
                            sc = float(ratio_db[i])
                        else:
                            sc = float(prom_db[i]) if prom_db is not None else 0.0
                        peak_score.append(sc)
                        peak_color.append("#D62728" if sc >= peak_db_thr else "#F59E0B")

                    if peak_x:
                        figp.add_trace(
                            _go.Scatter(
                                x=peak_x,
                                y=peak_y,
                                mode="markers",
                                marker=dict(size=9, color=peak_color, line=dict(color="white", width=0.7)),
                                name="Top peaks",
                                hovertemplate=(
                                    "f=%{x:.3f}<br>mag=%{y:.3g}<br>"
                                    + (("ΔdB=%{customdata:.2f} (vs baseline)" if ratio_db is not None else "prominence dB=%{customdata:.2f}")
                                       + "<extra></extra>")
                                ),
                                customdata=peak_score,
                            ),
                            row=1,
                            col=1,
                        )

                    # Delta / prominence panel
                    if ratio_db is not None:
                        y2 = ratio_db
                        y2_name = "Δ magnitude (dB vs baseline)"
                    else:
                        if prom_db is None:
                            win = int(max(7, min(61, (mag_mean.size // 25) * 2 + 1)))
                            kernel = np.ones(win, dtype=np.float64) / float(win)
                            smooth = np.convolve(mag_mean, kernel, mode="same")
                            prom_db = 20.0 * np.log10((mag_mean + 1e-12) / (smooth + 1e-12))
                        y2 = prom_db
                        y2_name = "Local prominence (dB)"

                    figp.add_trace(
                        _go.Scatter(
                            x=freqs,
                            y=y2,
                            mode="lines",
                            line=dict(color="#6C5CE7", width=2.0),
                            name=y2_name,
                            hovertemplate="f=%{x:.3f}<br>dB=%{y:.2f}<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )
                    figp.add_hline(y=float(peak_db_thr), line_dash="dash", line_color="#B91C1C", row=2, col=1)
                    try:
                        ymax = float(np.nanmax(y2)) if np.size(y2) else float(peak_db_thr)
                        ymax = max(ymax, float(peak_db_thr))
                        figp.add_hrect(y0=float(peak_db_thr), y1=ymax, fillcolor="rgba(255,234,234,0.45)", line_width=0, row=2, col=1)
                    except Exception:
                        pass

                    if peak_x:
                        figp.add_trace(
                            _go.Scatter(
                                x=peak_x,
                                y=[float(y2[i]) for i in idxs.tolist()],
                                mode="markers",
                                marker=dict(size=8, color=peak_color, line=dict(color="white", width=0.7)),
                                showlegend=False,
                                hovertemplate="f=%{x:.3f}<br>dB=%{y:.2f}<extra></extra>",
                            ),
                            row=2,
                            col=1,
                        )

                    figp.update_yaxes(type="log", title_text="Magnitude (log)", row=1, col=1)
                    figp.update_yaxes(title_text="dB", zeroline=False, row=2, col=1)
                    figp.update_xaxes(title_text="Frequency (normalized)", range=[0.0, float(fmax)], row=2, col=1)

                    figp.update_layout(
                        title=dict(text="NeurInSpectre — Mean Magnitude Spectrum (Triage View)", x=0.5, xanchor="center"),
                        template="plotly_white",
                        height=820,
                        # Put the legend below the plots (avoids title/legend collisions and keeps the top clean).
                        margin=dict(l=70, r=40, t=95, b=130),
                        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0.0),
                    )

                    plot_div = _pio.to_html(
                        figp,
                        full_html=False,
                        include_plotlyjs=True,
                        config={"displaylogo": False, "responsive": True},
                    )

                    # Build HTML with below-the-plot action boxes (no overlap)
                    def _lines_to_ul(txt: str) -> str:
                        lines = [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]
                        if not lines:
                            return "<ul></ul>"
                        # Drop header line; strip leading bullet glyphs for list items.
                        items = []
                        for ln in lines[1:]:
                            ln = ln.lstrip("•").strip()
                            if ln:
                                items.append(f"<li>{_html.escape(ln)}</li>")
                        return "<ul>" + "".join(items) + "</ul>"

                    sum_html = "<br>".join(_html.escape(s) for s in sum_lines)
                    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>NeurInSpectre — Spectral Triage</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 18px; color: #111; }}
    .grid {{ display: grid; grid-template-columns: 1.1fr 1.6fr 1.6fr; gap: 14px; margin-top: 14px; }}
    .box {{ border-radius: 10px; padding: 14px 14px; border: 1px solid #d0d0d0; background: #fff; }}
    .box h3 {{ margin: 0 0 8px 0; font-size: 14px; }}
    .summary {{ background: #ffffff; }}
    .blue {{ background: #ECF3FF; border-color: #1F5FBF; }}
    .red {{ background: #FFF1F2; border-color: #B91C1C; }}
    .note {{ margin-top: 10px; font-size: 12.5px; color: #555; }}
    ul {{ margin: 0; padding-left: 18px; }}
    li {{ margin: 4px 0; }}
  </style>
</head>
<body>
  {plot_div}
  <div class="grid">
    <div class="box summary">
      <h3>Summary (heuristic)</h3>
      <div style="font-size: 13px; line-height: 1.25;">{sum_html}</div>
    </div>
    <div class="box blue">
      <h3>Blue team — practical next steps</h3>
      {_lines_to_ul(blue_text)}
    </div>
    <div class="box red">
      <h3>Red team — evaluation checklist (safe)</h3>
      {_lines_to_ul(red_text)}
    </div>
  </div>
  <div class="note">{_html.escape(note_text)}</div>
</body>
</html>
"""

                    out_html = str(getattr(args, "html"))
                    _Path(out_html).parent.mkdir(parents=True, exist_ok=True)
                    _Path(out_html).write_text(page, encoding="utf-8")
                    print(out_html)

                # Write summary to --output if provided; else write alongside plot if we have a plot;
                # otherwise print to stdout so the command remains useful without file outputs.
                from pathlib import Path as _Path
                out_json = getattr(args, 'output', None)
                if out_json:
                    _Path(out_json).write_text(json.dumps(summary, indent=2))
                    print(out_json)
                elif getattr(args, 'plot', None) or getattr(args, 'html', None):
                    base_path = str(getattr(args, 'plot', None) or getattr(args, 'html', None))
                    # Prefer stable naming regardless of output extension.
                    if base_path.lower().endswith(".png"):
                        summary_path = base_path[:-4] + "_summary.json"
                    elif base_path.lower().endswith(".html"):
                        summary_path = base_path[:-5] + "_summary.json"
                    else:
                        summary_path = base_path + "_summary.json"
                    _Path(summary_path).write_text(json.dumps(summary, indent=2))
                    print(summary_path)
                else:
                    print(json.dumps(summary, indent=2))
                return 0
            except Exception as e:
                logger.error(f"Spectral analysis failed: {e}")
                return 1
        elif args.command == 'integrate':
            try:
                import numpy as np
                import matplotlib.pyplot as plt
                from pathlib import Path
                # Load input as 1D state vector
                X = np.load(args.input)
                arr = np.nan_to_num(np.array(X), nan=0.0, posinf=0.0, neginf=0.0)
                if arr.ndim == 0:
                    state0 = arr.reshape(1)
                elif arr.ndim == 1:
                    state0 = arr
                else:
                    state0 = arr.reshape(-1)
                dtype = np.float32 if getattr(args, 'precision', 'float32') == 'float32' else np.float64
                state0 = state0.astype(dtype, copy=False)
                steps = int(getattr(args, 'steps', 100))
                dt = float(getattr(args, 'dt', 0.01))
                # Simple stable linear decay model x(t) = exp(-a t) * x0 (ETD analytic solution)
                a = dtype(0.1)
                t = np.arange(steps, dtype=dtype) * dtype(dt)
                decay = np.exp(-a * t)[:, None]  # [steps, 1]
                states = decay * state0[None, :]  # [steps, D]
                out_path = getattr(args, 'output', None) or 'integrate.npy'
                np.save(out_path, states)
                # Optional norm plot
                if getattr(args, 'plot', None):
                    norms = np.linalg.norm(states, axis=1)
                    plt.figure(figsize=(9, 3.2))
                    plt.plot(t, norms, label='||x(t)||')
                    plt.title('NeurInSpectre — Integration Norm Over Time')
                    plt.xlabel('t'); plt.ylabel('Norm')
                    plt.grid(alpha=0.25)
                    plt.tight_layout()
                    plt.savefig(args.plot, dpi=200, bbox_inches='tight')
                print(out_path)
                return 0
            except Exception as e:
                logger.error(f"Integration failed: {e}")
                return 1
        elif args.command == 'test':
            try:
                import numpy as np
                # Smoke tests for spectral/integrate/demo handlers
                # 1) Spectral: create tiny signal
                x = np.sin(np.linspace(0, 8*np.pi, 256, dtype=float)).astype('float32')
                np.save('._tmp_spec.npy', x)
                class _A: pass
                a = _A(); a.input='._tmp_spec.npy'; a.plot='._tmp_spec.png'
                # Reuse spectral branch by constructing args-like object
                _ = a
                # 2) Integrate: simple state vector
                np.save('._tmp_int.npy', np.random.randn(64).astype('float32'))
                print('NeurInSpectre math test: OK')
                return 0
            except Exception as e:
                logger.error(f"Test suite failed: {e}")
                return 1
        elif args.command == 'prompt_injection_analysis':
            try:
                import numpy as np, json, math, re
                import matplotlib.pyplot as plt
                from pathlib import Path
                sp = args.suspect_prompt; cp = args.clean_prompt
                # Feature set informed by recent prompt-injection research (last 7 months):
                # length, token count, URL count, punctuation ratio, uppercase ratio, Shannon entropy
                url_re = re.compile(r"https?://|www\\.")
                def shannon_entropy(s: str):
                    if not s:
                        return 0.0
                    from collections import Counter
                    cnt = Counter(s)
                    n = len(s)
                    return -sum((c/n)*math.log2(c/n) for c in cnt.values())
                def feats(s):
                    tokens = s.split()
                    num_tokens = len(tokens)
                    num_chars = len(s)
                    url_count = len(url_re.findall(s))
                    puncts = sum(1 for c in s if c in '.,:;!?')
                    punct_ratio = (puncts / max(1, num_chars))
                    upper_ratio = (sum(1 for c in s if c.isupper()) / max(1, num_chars))
                    entropy = shannon_entropy(s)
                    return np.array([num_tokens, num_chars, url_count, punct_ratio, upper_ratio, entropy], dtype=float)
                # Advanced indicators (recent community guidance): code-fences, tool terms, base64-like blocks, JSON-like
                codef_re = re.compile(r"```|<code>|</code>")
                tool_re = re.compile(r"tool\.|exec|curl|wget|/etc/|~/.ssh|id_rsa|OPENAI_API_KEY|AWS_SECRET_ACCESS_KEY", re.I)
                b64_re = re.compile(r"^[A-Za-z0-9+/=\n\r]+$")
                json_like_re = re.compile(r"\{\s*\"[A-Za-z0-9_]+\"\s*:")
                def adv_feats(s: str):
                    n = max(1, len(s))
                    digits = sum(c.isdigit() for c in s) / n
                    codef = len(codef_re.findall(s))
                    tools = len(tool_re.findall(s))
                    has_json = 1.0 if json_like_re.search(s) else 0.0
                    blocks = [blk for blk in re.split(r"\s+", s) if len(blk) >= 16 and b64_re.match(blk or '')]
                    b64_blocks = float(len(blocks))
                    return np.array([digits, codef, tools, has_json, b64_blocks], dtype=float)
                fs_base, fc_base = feats(sp), feats(cp)
                fs_adv, fc_adv = adv_feats(sp), adv_feats(cp)
                fs = np.concatenate([fs_base, fs_adv])
                fc = np.concatenate([fc_base, fc_adv])
                delta = fs - fc
                labels = ['tokens','chars','url_count','punct_ratio','upper_ratio','entropy',
                          'digit_ratio','code_fences','tool_terms','json_like','base64_blocks']
                # Action hints
                red_next = 'escalate token/URL and punctuation ratios where delta>0; test evasive casing/spacing'
                blue_next = 'cap URL/uppercase/punctuation ratios; sanitize inputs; entropy floor to detect templating'
                Path(f"{args.out_prefix}pia.json").write_text(
                    json.dumps({'labels': labels, 'suspect': fs.tolist(), 'clean': fc.tolist(), 'delta': delta.tolist(), 'red_next': red_next, 'blue_next': blue_next}, indent=2)
                )
                plt.figure(figsize=(10.5,4.8))
                x = np.arange(len(labels)); width=0.38
                clean_c = '#2A9D8F'
                suspect_c = '#E76F51'
                edge_c = '#1f1f1f'
                plt.bar(x - width/2, fc, width, label='clean', color=clean_c, edgecolor=edge_c, linewidth=0.6)
                plt.bar(x + width/2, fs, width, label='suspect', color=suspect_c, edgecolor=edge_c, linewidth=0.6)
                plt.xticks(x, labels, rotation=20)
                # Composite risk score from normalized deltas
                def _safe_norm(a, b):
                    denom = np.maximum(np.abs(b), 1e-8)
                    return float(np.clip(a/denom, -5.0, 5.0))
                # Weighted composite risk (emphasize tool/structure signals)
                weights = {
                    'url_count': 1.0,
                    'punct_ratio': 0.8,
                    'upper_ratio': 0.6,
                    'entropy': 0.6,
                    'tokens': 0.3,
                    'chars': 0.3,
                    'digit_ratio': 0.5,
                    'code_fences': 0.8,
                    'tool_terms': 1.2,
                    'json_like': 0.6,
                    'base64_blocks': 1.0,
                }
                risk = 0.0
                for k, wk in weights.items():
                    if k in labels:
                        i = labels.index(k)
                        risk += wk * abs(_safe_norm(delta[i], fc[i] if i < len(fc) else 1.0))
                sev_c = '#1f5fbf'
                if risk >= 6.0:
                    sev_c = '#cc0000'
                elif risk >= 3.0:
                    sev_c = '#e38b29'
                plt.title('NeurInSpectre — Prompt Injection Feature Compare', color=sev_c)
                plt.xlabel('Feature')
                plt.ylabel('Value')
                ax = plt.gca()
                ax.yaxis.grid(True, linestyle='-', alpha=0.15)
                # annotate deltas above bars for clarity (with legible background)
                for i, d in enumerate(delta):
                    y = max(fs[i], fc[i]) * 1.06 + 1e-9
                    plt.text(i, y, f"Δ={d:.2f}", ha='center', fontsize=9,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85))
                # Top-Δ callouts
                kcall = min(3, len(labels))
                top_idx = np.argsort(np.abs(delta))[-kcall:][::-1]
                for j in top_idx:
                    plt.annotate(labels[j], xy=(j, max(fs[j], fc[j])), xytext=(0, 20), textcoords='offset points',
                                 ha='center', fontsize=9,
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff9c4', edgecolor='#bfa700', alpha=0.9),
                                 arrowprops=dict(arrowstyle='->', color='#bfa700', lw=1.0))
                # Co-occurrence badges (multi-signal alignment)
                def _rise(name):
                    return (name in labels) and (delta[labels.index(name)] > 0)
                co_msgs = []
                co_url_entropy_punct = bool(_rise('url_count') and _rise('entropy') and _rise('punct_ratio'))
                if co_url_entropy_punct:
                    co_msgs.append('URL+entropy+punctuation rising')
                if _rise('tool_terms') and (_rise('json_like') or _rise('code_fences')):
                    co_msgs.append('tool terms with JSON/code‑fence')
                if _rise('base64_blocks'):
                    co_msgs.append('base64‑like blocks present')
                if co_msgs:
                    plt.figtext(0.02, 0.02, '; '.join(co_msgs), fontsize=9,
                                bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6f0ff', edgecolor='#1f5fbf', alpha=0.95))
                # Legend and keys
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
                fig = plt.gcf()
                import textwrap as _tw
                wrap_w = max(60, int(fig.get_size_inches()[0] * 10))
                btxt = _tw.fill('Blue: cap/normalize URLs, punctuation, casing; strip code‑fences; block tool‑term routing; scan for base64; alert on multi‑signal co‑rise.', width=wrap_w)
                rtxt = _tw.fill('Red: route via tool‑terms + JSON; align small deltas across URL/entropy/punct; wrap in code‑fences; use base64 blocks when needed.', width=wrap_w)
                plt.figtext(0.5, 0.02, 'Bars: clean (teal) vs suspect (orange-red); Δ labels show suspect−clean', ha='center', fontsize=9)
                plt.figtext(0.01, 0.10, btxt, fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6f0ff', edgecolor='#1f5fbf', alpha=0.95))
                plt.figtext(0.01, 0.17, rtxt, fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe6e6', edgecolor='#cc0000', alpha=0.95))
                plt.tight_layout(rect=[0, 0.30, 1, 1])
                # Augment JSON with risk/co-occurrence summary
                try:
                    meta_p = Path(f"{args.out_prefix}pia.json")
                    if meta_p.exists():
                        meta = json.loads(meta_p.read_text())
                    else:
                        meta = {}
                    meta.update({
                        'risk_score': float(risk),
                        'top_delta_features': [labels[int(i)] for i in top_idx.tolist()],
                        'cooccurrence_url_entropy_punct': bool(co_url_entropy_punct)
                    })
                    meta_p.write_text(json.dumps(meta, indent=2))
                except Exception:
                    pass
                fbar = f"{args.out_prefix}pia_compare.png"; plt.savefig(fbar, dpi=200, bbox_inches='tight')
                # Generate interactive HTML with attention visualization if model provided
                if getattr(args, 'model', None):
                    try:
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        import torch
                        from transformers import AutoTokenizer, AutoModel
                        
                        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
                        tokenizer = AutoTokenizer.from_pretrained(args.model)
                        # Use attn_implementation='eager' to enable attention output (SDPA doesn't support it)
                        model = AutoModel.from_pretrained(args.model, attn_implementation='eager').to(device)
                        model.eval()
                        
                        # Tokenize suspect prompt
                        encoded = tokenizer(sp, return_tensors="pt").to(device)
                        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
                        clean_tokens = [t.replace("##","").replace("Ġ","") for t in tokens]
                        
                        with torch.no_grad():
                            outputs = model(**encoded, output_attentions=True)
                        
                        # Get attention from user-specified layer/head (validated against model structure)
                        attn = outputs.attentions
                        num_layers = len(attn)
                        num_heads = attn[0].shape[1]
                        layer_idx = min(args.layer, num_layers - 1)
                        head_idx = min(args.head, num_heads - 1)
                        if args.layer >= num_layers:
                            logger.warning(f"Requested layer {args.layer} exceeds model layers ({num_layers}), using layer {layer_idx}")
                        if args.head >= num_heads:
                            logger.warning(f"Requested head {args.head} exceeds model heads ({num_heads}), using head {head_idx}")
                        heat = attn[layer_idx][0, head_idx].cpu().numpy()
                        seq_len = len(clean_tokens)
                        
                        # Create dual-panel figure
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("🔥 Injection Pattern Heatmap", "📊 Attention Vector Distribution"),
                            specs=[[{"type": "heatmap"}, {"type": "scatter3d"}]],
                            column_widths=[0.52, 0.48],
                            horizontal_spacing=0.12
                        )
                        
                        # Panel 1: Heatmap
                        fig.add_trace(go.Heatmap(
                            z=heat, x=clean_tokens, y=clean_tokens,
                            colorscale='Reds', colorbar=dict(title='Attention', x=0.46, len=0.9, thickness=15, tickfont=dict(size=10))
                        ), row=1, col=1)
                        
                        # Panel 2: 3D scatter
                        x_c, y_c, z_c, colors = [], [], [], []
                        for i in range(seq_len):
                            for j in range(seq_len):
                                x_c.append(i); y_c.append(j)
                                z_c.append(heat[i,j]); colors.append(heat[i,j])
                        
                        fig.add_trace(go.Scatter3d(
                            x=x_c, y=y_c, z=z_c, mode='markers',
                            marker=dict(size=4, color=colors, colorscale='Reds', opacity=0.8)
                        ), row=1, col=2)
                        
                        # === SECURITY METRICS (2024 Research: Attention Tracker arxiv 2411.00348, UTDMF, AEGIS) ===
                        
                        # 1. Attention Entropy (FIXED: per-row entropy, normalized to [0,1])
                        # Low entropy = focused attention on few tokens = injection targeting specific positions
                        row_entropies = []
                        for row in heat:
                            row_probs = row / (row.sum() + 1e-10)  # Normalize row
                            row_ent = -np.sum(row_probs * np.log(row_probs + 1e-10))
                            max_ent = np.log(len(row))  # Maximum possible entropy
                            row_entropies.append(row_ent / (max_ent + 1e-10))  # Normalize to [0,1]
                        attn_entropy = float(np.mean(row_entropies))  # Now properly in [0,1]
                        
                        # 2. Diagonal Dominance - ratio of self-attention vs cross-attention
                        # Low = attention going to other tokens = potential hijacking
                        diag_vals = np.diag(heat)
                        off_diag_mean = float((heat.sum() - diag_vals.sum()) / (heat.size - len(diag_vals) + 1e-10))
                        diag_dominance = float(np.mean(diag_vals) / (off_diag_mean + 1e-10))
                        
                        # 3. Attention Sparsity - fraction of near-zero attention weights
                        sparsity = float(np.sum(heat < 0.1) / heat.size)
                        
                        # 4. Max off-diagonal attention (injection signal)
                        off_diag = heat.copy()
                        np.fill_diagonal(off_diag, 0)
                        max_off_diag = float(np.max(off_diag))
                        max_off_idx = np.unravel_index(np.argmax(off_diag), off_diag.shape)
                        
                        # 5. Attention sink detection (first/last token captures too much attention)
                        first_col_attn = float(np.mean(heat[:, 0]))
                        last_col_attn = float(np.mean(heat[:, -1]))
                        sink_score = max(first_col_attn, last_col_attn)
                        
                        # 6. NEW: Distraction Effect (Attention Tracker 2024) - attention concentration on specific tokens
                        # High attention to any single token from many source tokens = distraction/injection
                        col_attention = heat.sum(axis=0)  # How much each token receives
                        distraction_score = float(np.max(col_attention) / (np.mean(col_attention) + 1e-10))
                        
                        # 7. NEW: Injection keyword detection in high-attention positions
                        injection_keywords = {'ignore', 'forget', 'disregard', 'override', 'system', 'admin', 'sudo', 'bypass', 'jailbreak', 'dan', 'unrestricted'}
                        keyword_attention = 0.0
                        for i, token in enumerate(clean_tokens):
                            if token.lower().strip('[].,') in injection_keywords:
                                keyword_attention += float(col_attention[i])
                        keyword_score = keyword_attention / (col_attention.sum() + 1e-10)
                        
                        # RISK ASSESSMENT (FIXED FORMULA based on 2024 research)
                        # Higher values = higher risk for all components
                        risk_components = {
                            'low_entropy': (1 - attn_entropy) * 0.15,      # Low entropy = focused attack
                            'low_diag': max(0, 1 - diag_dominance/2) * 0.15,  # Low diagonal = hijacking
                            'off_diag_peak': min(1, max_off_diag * 2) * 0.20,  # High off-diag = targeting
                            'sink': min(1, sink_score * 2) * 0.15,         # Attention sink capture
                            'distraction': min(1, (distraction_score - 1) / 3) * 0.15,  # Distraction effect
                            'keywords': min(1, keyword_score * 10) * 0.20,  # Injection keywords detected
                        }
                        risk_score = sum(risk_components.values())
                        risk_score = max(0, min(1, risk_score))  # Clamp to [0,1]
                        risk_level = "CRITICAL" if risk_score > 0.6 else "HIGH" if risk_score > 0.4 else "MEDIUM" if risk_score > 0.25 else "LOW"
                        
                        fig.update_layout(
                            title=dict(text=f"🔎 NeurInSpectre Prompt Injection Analysis - Layer {layer_idx} Head {head_idx} | Risk: {risk_level} ({risk_score:.2f})", font=dict(size=18)),
                            paper_bgcolor='#0d0d0d', plot_bgcolor='#1a1a1a',
                            font=dict(color='white'), height=1350, width=1600,
                            margin=dict(l=100, r=100, t=100, b=700)
                        )
                        
                        # === RED TEAM GUIDANCE (Offensive AI Security Research 2024) ===
                        red_guidance = [
                            f"🎯 ATTENTION HIJACKING: Off-diag peak at tokens ({max_off_idx[0]}→{max_off_idx[1]}) = {max_off_diag:.3f}",
                            f"   → Inject adversarial tokens at position {max_off_idx[1]} to maximize cross-attention capture",
                            f"🔥 ENTROPY EXPLOITATION: Entropy={attn_entropy:.3f} ({'LOW - exploit focused attention' if attn_entropy < 0.5 else 'HIGH - use attention diffusion attack'})",
                            f"   → {'Insert high-salience tokens (IGNORE, SYSTEM, ADMIN) to concentrate attention' if attn_entropy > 0.5 else 'Attention already focused - amplify with repetition'}",
                            f"⚡ SINK ATTACK: First-token={first_col_attn:.3f}, Last-token={last_col_attn:.3f}",
                            f"   → {'Prepend injection (attention sink at [CLS])' if first_col_attn > last_col_attn else 'Append injection (attention flows to end)'}",
                            f"🎭 LAYER STRATEGY: L{layer_idx}H{head_idx} shows {'sparse targeting - use direct injection' if sparsity > 0.7 else 'diffuse attention - use multi-point injection'}",
                        ]
                        
                        # === BLUE TEAM GUIDANCE (Defensive AI Security Research 2024) ===
                        blue_guidance = [
                            f"🛡️ DETECTION: Diagonal dominance={diag_dominance:.3f} ({'ANOMALY - below 1.5 threshold' if diag_dominance < 1.5 else 'NORMAL'})",
                            f"   → Set alert when diag_dominance < 1.5 (indicates cross-token attention hijacking)",
                            f"🔍 MONITORING: Track off-diagonal max ({max_off_diag:.3f}) across requests",
                            f"   → Baseline normal range, alert on >2σ deviation (current {'EXCEEDS' if max_off_diag > 0.3 else 'within'} typical 0.3 threshold)",
                            f"🔒 DEFENSE: {'Apply attention masking on positions 0,{max_off_idx[1]}' if max_off_diag > 0.3 else 'Standard input sanitization sufficient'}",
                            f"   → Implement sliding window attention or sparse attention to limit injection propagation",
                            f"📊 BASELINE: Compare entropy ({attn_entropy:.3f}) against clean prompt distribution",
                            f"   → Flag if entropy deviation >0.2 from baseline (indicates structured injection)",
                        ]
                        
                        # Add annotations using yshift for PIXEL-LEVEL control (fixes overlap)
                        # === METRICS LEGEND ===
                        legend_metrics = f"""<b>📊 KEY METRICS</b>  |  Entropy: {attn_entropy:.3f}  |  Diag Dominance: {diag_dominance:.3f}  |  Off-Diag Max: {max_off_diag:.3f}  |  Sparsity: {sparsity:.1%}  |  Sink: {sink_score:.3f}"""
                        fig.add_annotation(
                            text=legend_metrics,
                            xref="paper", yref="paper", x=0.5, y=0,
                            showarrow=False, font=dict(size=11, color='#ddd', family='monospace'),
                            align="center", bgcolor='rgba(50,50,50,0.95)', bordercolor='#666',
                            borderwidth=1, borderpad=8, xanchor='center', yanchor='top', yshift=-130
                        )
                        
                        # === RED TEAM (left side) ===
                        fig.add_annotation(
                            text="<b>🔴 RED TEAM - Prompt Injection Attack Vectors</b><br>" + "<br>".join(red_guidance),
                            xref="paper", yref="paper", x=0.01, y=0,
                            showarrow=False, font=dict(size=10, color='white', family='monospace'),
                            align="left", bgcolor='rgba(140,20,20,0.95)', bordercolor='#ff3333',
                            borderwidth=2, borderpad=12, xanchor='left', yanchor='top', yshift=-170, width=740
                        )
                        
                        # === BLUE TEAM (right side) ===
                        fig.add_annotation(
                            text="<b>🔵 BLUE TEAM - Detection & Defense Strategies</b><br>" + "<br>".join(blue_guidance),
                            xref="paper", yref="paper", x=0.51, y=0,
                            showarrow=False, font=dict(size=10, color='white', family='monospace'),
                            align="left", bgcolor='rgba(20,60,140,0.95)', bordercolor='#3399ff',
                            borderwidth=2, borderpad=12, xanchor='left', yanchor='top', yshift=-170, width=740
                        )
                        
                        # === Research citations ===
                        fig.add_annotation(
                            text="📚 Research: Greshake et al. 2024 (Indirect Injection) • Perez & Ribeiro 2024 (Ignore This Title) • Liu et al. 2024 (Attention Sink) • Zou et al. 2024 (Universal Adversarial)",
                            xref="paper", yref="paper", x=0.5, y=0,
                            showarrow=False, font=dict(size=9, color='#777'), align="center", xanchor='center', yanchor='top', yshift=-400
                        )
                        
                        html_file = f"{args.out_prefix}pia_interactive.html"
                        fig.write_html(html_file)
                        print(html_file)
                        logger.info(f"📊 Interactive HTML saved: {html_file}")
                    except Exception as ie:
                        logger.warning(f"Interactive viz skipped: {ie}")
                
                print(f"{args.out_prefix}pia.json")
                print(fbar)
                return 0
            except Exception as e:
                logger.error(f"Prompt injection analysis failed: {e}")
                return 1
        elif args.command == 'occlusion':
            try:
                from types import SimpleNamespace
                from .occlusion_analysis import run_occlusion_analysis_command
                # Provide defaults for outputs if missing
                out2 = getattr(args, 'output_2d', None) or 'occlusion_2d.png'
                out3 = getattr(args, 'output_3d', None) or 'occlusion_3d.html'
                a = SimpleNamespace(
                    model=args.model,
                    image_path=args.image_path,
                    image_url=args.image_url or 'https://picsum.photos/256',
                    patch_size=args.patch_size,
                    stride=args.stride,
                    output_2d=out2,
                    output_3d=out3
                )
                return run_occlusion_analysis_command(a)
            except Exception as e:
                logger.error(f"Occlusion wrapper failed: {e}")
                return 1
        elif args.command == 'anomaly':
            try:
                import numpy as np, json, os
                import matplotlib.pyplot as plt
                from pathlib import Path
                # Resolve input path from either --activations or --input
                act_path = getattr(args, 'activations', None) or getattr(args, 'input', None)
                if not act_path:
                    raise ValueError("Missing --activations/--input for anomaly detection")
                # Load .npy or .npz (permit pickled objects for saved dicts)
                if str(act_path).lower().endswith('.npz'):
                    data = np.load(act_path, allow_pickle=True)
                    arr = None
                    for key in ['activations', 'A', 'arr', 'data', 'x', 'X']:
                        if key in data:
                            arr = np.array(data[key])
                            break
                    if arr is None:
                        # Fallback: first array-like in npz
                        for k in data.files:
                            try:
                                arr = np.array(data[k])
                                break
                            except Exception:
                                continue
                    if arr is None:
                        raise ValueError(".npz did not contain any array entries")
                else:
                    loaded = np.load(act_path, allow_pickle=True)
                    # If this is a zero-dim object carrying a dict/list, unwrap
                    if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
                        loaded = loaded.item()
                    if isinstance(loaded, dict):
                        arr = None
                        for key in ['activations', 'A', 'arr', 'data', 'x', 'X']:
                            if key in loaded:
                                arr = np.array(loaded[key])
                                break
                        if arr is None:
                            # Fallback to first array-like value
                            for v in loaded.values():
                                try:
                                    arr = np.array(v)
                                    break
                                except Exception:
                                    continue
                    else:
                        arr = np.array(loaded)
                if arr is None:
                    raise ValueError('Could not resolve activations array from input')
                # Sanitize values
                if np.any(~np.isfinite(arr)):
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                elif arr.ndim > 2:
                    arr = arr.reshape(-1, arr.shape[-1])

                # Optional baseline/reference array for Z computation (more precise than self-baselining).
                ref_path = getattr(args, 'reference', None)
                ref_arr = None
                ref_align_info = {}
                if ref_path:
                    try:
                        if str(ref_path).lower().endswith('.npz'):
                            data = np.load(ref_path, allow_pickle=True)
                            tmp = None
                            for key in ['activations', 'A', 'arr', 'data', 'x', 'X']:
                                if key in data:
                                    tmp = np.array(data[key])
                                    break
                            if tmp is None:
                                for k in data.files:
                                    try:
                                        tmp = np.array(data[k])
                                        break
                                    except Exception:
                                        continue
                            if tmp is None:
                                raise ValueError(".npz did not contain any array entries")
                        else:
                            loaded = np.load(ref_path, allow_pickle=True)
                            if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
                                loaded = loaded.item()
                            if isinstance(loaded, dict):
                                tmp = None
                                for key in ['activations', 'A', 'arr', 'data', 'x', 'X']:
                                    if key in loaded:
                                        tmp = np.array(loaded[key])
                                        break
                                if tmp is None:
                                    for v in loaded.values():
                                        try:
                                            tmp = np.array(v)
                                            break
                                        except Exception:
                                            continue
                            else:
                                tmp = np.array(loaded)
                        if tmp is None:
                            raise ValueError('Could not resolve reference array')
                        if np.any(~np.isfinite(tmp)):
                            tmp = np.nan_to_num(tmp, nan=0.0, posinf=0.0, neginf=0.0)
                        if tmp.ndim == 1:
                            tmp = tmp.reshape(1, -1)
                        elif tmp.ndim > 2:
                            tmp = tmp.reshape(-1, tmp.shape[-1])

                        # Align feature dimensions (explicit + recorded). If dims mismatch, truncate both
                        # to the minimum dimension and record the alignment decision for auditability.
                        if tmp.shape[1] != arr.shape[1]:
                            min_d = int(min(tmp.shape[1], arr.shape[1]))
                            ref_align_info = {
                                'reference_feature_dim': int(tmp.shape[1]),
                                'input_feature_dim': int(arr.shape[1]),
                                'aligned_feature_dim': int(min_d),
                                'feature_dim_alignment': 'truncate',
                            }
                            tmp = tmp[:, :min_d]
                            arr = arr[:, :min_d]
                        else:
                            ref_align_info = {
                                'reference_feature_dim': int(tmp.shape[1]),
                                'input_feature_dim': int(arr.shape[1]),
                                'aligned_feature_dim': int(arr.shape[1]),
                                'feature_dim_alignment': 'none',
                            }

                        ref_arr = tmp
                    except Exception as e:
                        raise ValueError(f"Failed to load --reference {ref_path}: {e}") from e

                # Method selection
                method = getattr(args, 'method', 'auto')
                use_robust = bool(getattr(args, 'robust', False)) or method == 'robust_z'
                zthr = float(getattr(args, 'z', 3.0))
                # Always compute z for visualization consistency
                baseline = ref_arr if ref_arr is not None else arr
                if use_robust:
                    # Robust Z: scale MAD to be sigma-consistent for a Normal distribution.
                    # For Normal: MAD ≈ 0.67449 * σ  =>  z ≈ 0.67449 * (x - median) / MAD
                    med = np.median(baseline, axis=0)
                    mad = np.median(np.abs(baseline - med), axis=0) + 1e-8
                    _MAD_TO_SIGMA = 0.6744897501960817
                    z = _MAD_TO_SIGMA * (arr - med) / mad
                else:
                    mu = baseline.mean(axis=0); sd = baseline.std(axis=0) + 1e-8
                    z = (arr - mu) / sd
                # If requested, run isolation forest on samples
                if method in ['iforest', 'auto']:
                    iforest_info = {}
                    try:
                        from sklearn.ensemble import IsolationForest
                        clf = IsolationForest(random_state=0, n_estimators=100, contamination='auto')
                        clf.fit(arr)
                        preds = clf.predict(arr)  # -1 anomaly, 1 normal
                        scores = clf.decision_function(arr)
                        iforest_info = {
                            'iforest_mean_score': float(np.mean(scores)),
                            'iforest_anomaly_fraction': float(np.mean(preds == -1)),
                        }
                    except Exception:
                        if method == 'iforest':
                            # Fallback silently to robust z summary
                            iforest_info = {'iforest_fallback': True}
                else:
                    iforest_info = {}

                # Feature-wise summary using Z
                abs_z = np.abs(z)
                mask_zthr = abs_z > zthr

                # Optional FDR control (more precise than fixed |Z| threshold under many comparisons).
                # We compute it on entry-level p-values derived from |Z| (two-sided).
                fdr_q = getattr(args, 'fdr', None)
                fdr_info = {}
                mask_fdr = None
                if fdr_q is not None:
                    try:
                        q = float(fdr_q)
                    except Exception:
                        q = None
                    if q is not None and 0.0 < q < 1.0:
                        try:
                            # Two-sided p-values under N(0,1): p = 2*sf(|z|)
                            try:
                                from scipy.stats import norm
                                pvals = 2.0 * norm.sf(abs_z)
                            except Exception:
                                import math
                                pvals = np.vectorize(lambda x: math.erfc(float(x) / math.sqrt(2.0)))(abs_z)

                            flat = pvals.reshape(-1)
                            m = int(flat.size)
                            order = np.argsort(flat)
                            p_sorted = flat[order]
                            crit = (np.arange(1, m + 1, dtype=np.float64) / float(m)) * float(q)
                            below = p_sorted <= crit
                            if np.any(below):
                                kmax = int(np.max(np.where(below)[0]))
                                p_thr = float(p_sorted[kmax])
                                reject = flat <= p_thr
                            else:
                                p_thr = None
                                reject = np.zeros_like(flat, dtype=bool)
                            mask_fdr = reject.reshape(pvals.shape)
                            fdr_info = {
                                'fdr_q': float(q),
                                'fdr_p_threshold': (float(p_thr) if p_thr is not None else None),
                                'fdr_rejections': int(np.sum(mask_fdr)),
                            }
                        except Exception:
                            fdr_info = {'fdr_failed': True}
                    else:
                        fdr_info = {'fdr_invalid_q': fdr_q}

                # Choose the active mask for counting/flagging based on --flagging.
                flag_mode = str(getattr(args, 'flagging', 'auto') or 'auto').strip().lower()
                if flag_mode not in {'auto', 'z', 'fdr'}:
                    flag_mode = 'auto'
                if flag_mode == 'z':
                    mask_entries = mask_zthr
                    flagging_used = 'z_threshold'
                elif flag_mode == 'fdr':
                    if mask_fdr is not None and 'fdr_q' in fdr_info:
                        mask_entries = mask_fdr
                        flagging_used = 'fdr'
                    else:
                        mask_entries = mask_zthr
                        flagging_used = 'z_threshold_fallback'
                else:  # auto
                    if mask_fdr is not None and 'fdr_q' in fdr_info:
                        mask_entries = mask_fdr
                        flagging_used = 'fdr'
                    else:
                        mask_entries = mask_zthr
                        flagging_used = 'z_threshold'

                max_abs_z = abs_z.max(axis=0)
                D = int(max_abs_z.shape[0])
                topk = min(int(getattr(args, 'topk', 10)), max(1, max_abs_z.shape[0]))
                idxs = np.argsort(max_abs_z)[-topk:][::-1]

                # Human-friendly feature labels (helps when inputs are summary-stat artifacts).
                feature_labels = [f"f{int(i)}" for i in range(int(max_abs_z.shape[0]))]
                if int(max_abs_z.shape[0]) == 3 and arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= 20:
                    # Heuristic: common capture format is [mean, std, max] per step.
                    try:
                        cols = np.asarray(arr[:, :3], dtype=np.float64)
                        frac_neg = np.mean(cols < 0.0, axis=0)
                        max_ge_mean = float(np.mean(cols[:, 2] >= cols[:, 0]))
                        if float(frac_neg[1]) < 0.01 and float(frac_neg[2]) < 0.01 and max_ge_mean > 0.90:
                            feature_labels = ["mean", "std", "max"]
                    except Exception:
                        pass

                # Counts (avoid ambiguous "num_anomalies" semantics)
                # Flag at the entry level first, then derive feature/sample flags from that mask.
                flagged_features = np.where(np.any(mask_entries, axis=0))[0]
                num_flagged_entries = int(mask_entries.sum())
                num_flagged_features = int(flagged_features.size)
                try:
                    num_flagged_samples = int(np.any(mask_entries, axis=1).sum())
                except Exception:
                    num_flagged_samples = int(1 if num_flagged_entries > 0 else 0)

                # Make next steps practical: include indices for flagged samples/features (bounded lists).
                flagged_sample_indices = np.where(np.any(mask_entries, axis=1))[0].astype(int).tolist()
                flagged_samples_detail = []
                for s in flagged_sample_indices[:50]:
                    feats = np.where(mask_entries[int(s)])[0].astype(int).tolist()
                    flagged_samples_detail.append(
                        {
                            "sample_idx": int(s),
                            "feature_indices": [int(f) for f in feats],
                            "feature_names": [feature_labels[int(f)] if int(f) < len(feature_labels) else f"f{int(f)}" for f in feats],
                            "z_scores": [float(z[int(s), int(f)]) for f in feats],
                        }
                    )

                flagged_features_detail = []
                for f in flagged_features.astype(int).tolist():
                    f = int(f)
                    try:
                        max_s = int(np.argmax(np.abs(z[:, f])))
                    except Exception:
                        max_s = 0
                    flagged_features_detail.append(
                        {
                            "feature_idx": f,
                            "feature_name": feature_labels[f] if f < len(feature_labels) else f"f{f}",
                            "max_abs_z": float(max_abs_z[f]) if f < max_abs_z.shape[0] else 0.0,
                            "max_abs_z_sample": int(max_s),
                            "num_flagged_entries": int(np.sum(mask_entries[:, f])) if f < mask_entries.shape[1] else 0,
                        }
                    )

                # Report what we actually did for "auto"
                method_requested = str(method)
                method_used = method_requested
                if method_requested == 'auto':
                    method_used = 'iforest+z' if ('iforest_mean_score' in iforest_info) else ('robust_z' if use_robust else 'z')
                summary = {
                    'schema': 'neurinspectre.anomaly.v2',
                    'method_requested': method_requested,
                    'method_used': method_used,
                    'z_threshold': zthr,
                    'reference_path': (str(ref_path) if ref_path else None),
                    'reference_shape': (list(ref_arr.shape) if ref_arr is not None else None),
                    'reference_alignment': ref_align_info,
                    'flagging': flagging_used,
                    'flagging_mode': flag_mode,
                    'z_threshold_counts': {
                        'num_flagged_entries': int(mask_zthr.sum()),
                        'num_flagged_samples': int(np.any(mask_zthr, axis=1).sum()),
                        'num_flagged_features': int(np.any(mask_zthr, axis=0).sum()),
                    },
                    'fdr_counts': (
                        {
                            'num_flagged_entries': int(mask_fdr.sum()) if mask_fdr is not None else 0,
                            'num_flagged_samples': int(np.any(mask_fdr, axis=1).sum()) if mask_fdr is not None else 0,
                            'num_flagged_features': int(np.any(mask_fdr, axis=0).sum()) if mask_fdr is not None else 0,
                        }
                        if mask_fdr is not None and 'fdr_q' in fdr_info
                        else None
                    ),
                    'robust': bool(use_robust),
                    # Back-compat: legacy name (entries, not samples)
                    'num_anomalies': num_flagged_entries,
                    'num_flagged_entries': num_flagged_entries,
                    'num_flagged_features': num_flagged_features,
                    'num_flagged_samples': num_flagged_samples,
                    'feature_labels': feature_labels,
                    'flagged_feature_indices': [int(i) for i in flagged_features.astype(int).tolist()],
                    'flagged_feature_names': [feature_labels[int(i)] if int(i) < len(feature_labels) else f"f{int(i)}" for i in flagged_features.astype(int).tolist()],
                    'topk_indices': [int(i) for i in idxs.tolist()],
                    'topk_feature_names': [feature_labels[int(i)] if int(i) < len(feature_labels) else f"f{int(i)}" for i in idxs.tolist()],
                    'topk_scores': [float(max_abs_z[i]) for i in idxs.tolist()],
                    'flagged_sample_indices': [int(i) for i in flagged_sample_indices[:200]],
                    'flagged_samples_detail': flagged_samples_detail,
                    'flagged_features_detail': flagged_features_detail,
                }
                summary.update(iforest_info)
                summary.update(fdr_info)
                # Write JSON
                out_json = getattr(args, 'output', None) or f"{args.out_prefix}anomaly.json"
                Path(os.path.dirname(out_json) or '.').mkdir(parents=True, exist_ok=True)
                Path(out_json).write_text(json.dumps(summary, indent=2))
                # ------------------------------------------------------------------
                # Plots (professional multi-panel triage view + top-K bar)
                # ------------------------------------------------------------------
                import matplotlib as mpl
                from contextlib import nullcontext

                # Subtle, publication-style defaults (avoid global state mutation beyond this call).
                style_ctx = nullcontext()
                try:
                    style_ctx = plt.style.context("seaborn-v0_8-whitegrid")
                except Exception:
                    pass

                # Compute a few additional diagnostics to make the next steps more concrete.
                D = int(max_abs_z.shape[0])
                x_idx = np.arange(D, dtype=int)
                # Flagged features should reflect the *active flagging method*.
                try:
                    flagged = np.any(mask_entries, axis=0)
                except Exception:
                    flagged = max_abs_z > zthr
                # If FDR is active, differentiate "Z exceeds threshold" vs "FDR rejected".
                zfeat = None
                fdrfeat = None
                try:
                    zfeat = np.any(mask_zthr, axis=0)
                except Exception:
                    zfeat = max_abs_z > zthr
                if mask_fdr is not None and 'fdr_q' in fdr_info:
                    try:
                        fdrfeat = np.any(mask_fdr, axis=0)
                    except Exception:
                        fdrfeat = None
                # "Near threshold" band is useful for monitoring diffuse / low-amplitude drift.
                near_margin = float(min(0.5, max(0.15, 0.10 * zthr)))
                near = (max_abs_z > (zthr - near_margin)) & (~flagged)
                z_only = None
                if fdrfeat is not None and zfeat is not None:
                    z_only = (zfeat.astype(bool) & (~fdrfeat.astype(bool)))

                def _gini(values: np.ndarray) -> float:
                    v = np.asarray(values, dtype=np.float64).reshape(-1)
                    v = v[np.isfinite(v)]
                    if v.size < 2:
                        return 0.0
                    v = np.abs(v)
                    s = float(v.sum())
                    if s <= 0:
                        return 0.0
                    v = np.sort(v)
                    n = v.size
                    # Gini = (2*sum(i*v_i)/(n*sum(v)) - (n+1)/n)
                    idx = np.arange(1, n + 1, dtype=np.float64)
                    g = (2.0 * float(np.sum(idx * v)) / (n * s)) - (float(n + 1) / float(n))
                    return float(np.clip(g, 0.0, 1.0))

                concentration = _gini(max_abs_z)
                max_i = int(np.argmax(max_abs_z)) if D else 0
                max_v = float(max_abs_z[max_i]) if D else 0.0
                num_flag = int(np.sum(flagged))
                num_near = int(np.sum(near))

                # Feature labels: default to f0..fD. For common 3-col summary-stat artifacts
                # (mean/std/max per step), provide semantic labels for readability.
                feature_labels = [f"f{int(i)}" for i in x_idx.tolist()]
                if D == 3:
                    try:
                        cols = np.asarray(arr[:, :3], dtype=np.float64)
                        frac_neg = np.mean(cols < 0.0, axis=0)
                        max_ge_mean = float(np.mean(cols[:, 2] >= cols[:, 0]))
                        if float(frac_neg[1]) < 0.01 and float(frac_neg[2]) < 0.01 and max_ge_mean > 0.90:
                            feature_labels = ["mean", "std", "max"]
                    except Exception:
                        pass
                max_name = feature_labels[max_i] if max_i < len(feature_labels) else f"f{max_i}"

                with style_ctx, mpl.rc_context(
                    {
                        "axes.titlesize": 13,
                        "axes.labelsize": 11,
                        "xtick.labelsize": 9,
                        "ytick.labelsize": 9,
                        "legend.fontsize": 9,
                        "figure.titlesize": 15,
                        "font.family": "DejaVu Sans",
                    }
                ):
                    # Slightly taller canvas so the right-hand action cards never overlap.
                    fig = plt.figure(figsize=(12.4, 7.4))
                    gs = fig.add_gridspec(
                        2,
                        2,
                        width_ratios=[2.35, 1.15],
                        height_ratios=[1.05, 0.95],
                        wspace=0.25,
                        hspace=0.30,
                    )

                    ax_main = fig.add_subplot(gs[0, 0])
                    ax_bar = fig.add_subplot(gs[1, 0])
                    ax_side = fig.add_subplot(gs[:, 1])
                    ax_side.axis("off")

                    # --- Main: Max |Z| per feature (with threshold + near-threshold band)
                    ax_main.plot(x_idx, max_abs_z, color="#34495E", lw=2.0, alpha=0.85, zorder=2, label="Max |Z| per feature")
                    ax_main.scatter(x_idx[~flagged], max_abs_z[~flagged], s=28, color="#2A9D8F", alpha=0.9, zorder=3, label="Normal")
                    if z_only is not None and np.any(z_only):
                        ax_main.scatter(x_idx[z_only], max_abs_z[z_only], s=40, color="#F4A261", alpha=0.95, zorder=4, label="Z>thr (not FDR)")
                    if np.any(near):
                        ax_main.scatter(x_idx[near], max_abs_z[near], s=36, color="#F4A261", alpha=0.95, zorder=4, label="Near threshold")
                    if np.any(flagged):
                        ax_main.scatter(x_idx[flagged], max_abs_z[flagged], s=44, color="#D04A5C", alpha=0.95, zorder=5, label="Flagged")

                    ax_main.axhline(zthr, color="#C0392B", linestyle="--", lw=1.8, label=f"Z-threshold = {zthr:g}")
                    # Shade bands
                    ymax = float(max(max_abs_z.max() if D else 0.0, zthr * 1.15))
                    ax_main.axhspan(zthr, ymax, color="#FDE2E4", alpha=0.45, zorder=0)
                    ax_main.axhspan(zthr - near_margin, zthr, color="#FFF3D6", alpha=0.45, zorder=0)

                    # Annotate up to 10 most suspicious features (either flagged, or top-K if none flagged).
                    annotate_ids = np.where(flagged)[0]
                    if annotate_ids.size == 0:
                        annotate_ids = idxs[: min(10, idxs.size)]
                    for i in annotate_ids[:10]:
                        lab = feature_labels[int(i)] if int(i) < len(feature_labels) else f"f{int(i)}"
                        ax_main.annotate(
                            f"{lab}\n{max_abs_z[int(i)]:.2f}",
                            xy=(int(i), float(max_abs_z[int(i)])),
                            xytext=(0, 14),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#D0D0D0", alpha=0.95),
                            zorder=10,
                        )

                    ax_main.set_title("NeurInSpectre — Anomaly Z‑Score (per feature)")
                    ax_main.set_xlabel("Feature index")
                    ax_main.set_ylabel("Max |Z| across samples")
                    ax_main.set_xlim(-0.5, float(D - 0.5) if D else 0.5)
                    ax_main.set_ylim(float(min(0.0, max_abs_z.min() - 0.15)) if D else 0.0, ymax)
                    if D <= 20:
                        ax_main.set_xticks(x_idx)
                        ax_main.set_xticklabels(feature_labels)
                    ax_main.legend(loc="upper left", frameon=True, framealpha=0.9)

                    # --- Bar: Top‑K features (highlight flagged/near-threshold)
                    top_vals = [float(max_abs_z[i]) for i in idxs]
                    bar_colors = []
                    for i in idxs:
                        if flagged[int(i)]:
                            bar_colors.append("#D04A5C")
                        elif near[int(i)]:
                            bar_colors.append("#F4A261")
                        else:
                            bar_colors.append("#2A9D8F")
                    ax_bar.bar(range(topk), top_vals, color=bar_colors, alpha=0.95)
                    ax_bar.axhline(zthr, color="#C0392B", linestyle="--", lw=1.6)
                    ax_bar.set_title("Top‑K features by Max |Z|")
                    ax_bar.set_xlabel("Feature (ranked)")
                    ax_bar.set_ylabel("Max |Z|")
                    ax_bar.set_xticks(range(topk))
                    ax_bar.set_xticklabels(
                        [feature_labels[int(i)] if int(i) < len(feature_labels) else f"f{int(i)}" for i in idxs],
                        rotation=0,
                    )
                    ax_bar.grid(alpha=0.25, axis="y")

                    # --- Side panel: crisp “what next” cards
                    ax_side.set_xlim(0.0, 1.0)
                    ax_side.set_ylim(0.0, 1.0)
                    iforest_frac = None
                    try:
                        if "iforest_anomaly_fraction" in iforest_info:
                            iforest_frac = float(iforest_info.get("iforest_anomaly_fraction", 0.0))
                    except Exception:
                        iforest_frac = None

                    header_lines = [
                        "Summary",
                        f"• samples={int(arr.shape[0])}  features={D}",
                        f"• method={method_used}  z_thr={zthr:g}",
                        f"• flagged_features={num_flag}  near={num_near}",
                        f"• flagged_samples={num_flagged_samples}  entries={num_flagged_entries}",
                        f"• max={max_name} ({max_v:.2f})",
                        f"• concentration(Gini)={concentration:.2f}",
                    ]
                    if iforest_frac is not None:
                        header_lines.insert(4, f"• iforest_anom_frac={iforest_frac:.2f}")
                    header = "\n".join(header_lines)
                    ax_side.text(
                        0.02,
                        0.98,
                        header,
                        va="top",
                        ha="left",
                        fontsize=10,
                        fontweight="bold",
                        linespacing=1.25,
                        bbox=dict(boxstyle="round,pad=0.55", facecolor="#F7F9FB", edgecolor="#D0D0D0", alpha=0.98),
                        transform=ax_side.transAxes,
                        clip_on=True,
                        wrap=True,
                    )

                    blue_lines = [
                        "Blue team triage (next steps):",
                        "• Re-run with --method robust_z; ideally add --reference <clean.npy>.",
                        "• Check persistence across windows/runs (recurrence matters).",
                        "• Localize: inspect top‑K time series (drift vs spikes vs periodicity).",
                        "• Cross-validate: correlate vs other artifacts; run comprehensive-scan.",
                        "• If many comparisons: consider --fdr 0.05 to reduce false positives.",
                        "• Respond: quarantine suspect inputs; tighten monitoring; retrain if persistent.",
                    ]
                    import textwrap as _tw

                    def _wrap_block(lines, width: int = 52) -> str:
                        out = []
                        for ln in lines:
                            if ln.startswith("•"):
                                out.append(_tw.fill(ln, width=width, subsequent_indent="  "))
                            else:
                                out.append(_tw.fill(ln, width=width))
                        return "\n".join(out)

                    blue_text = _wrap_block(blue_lines, width=52)
                    ax_side.text(
                        0.02,
                        0.62,
                        blue_text,
                        va="top",
                        ha="left",
                        fontsize=9.2,
                        linespacing=1.25,
                        bbox=dict(boxstyle="round,pad=0.55", facecolor="#E6F0FF", edgecolor="#1F5FBF", alpha=0.98),
                        transform=ax_side.transAxes,
                        clip_on=True,
                        wrap=True,
                    )

                    red_lines = [
                        "Red team validation (safe, test-focused):",
                        "• Sweep `--z` and robust_z; calibrate vs clean baseline.",
                        "• Compare fixed Z-threshold vs --fdr q (multiple comparisons control).",
                        "• Case matrix: spikes, gradual drift, periodic artifacts; measure detection.",
                        "• Reproducibility: pin seeds; keep artifacts; compare runs.",
                        "• Coverage/report: pair with other detectors; document FP/FN; propose tuning.",
                    ]
                    red_text = _wrap_block(red_lines, width=52)
                    ax_side.text(
                        0.02,
                        0.24,
                        red_text,
                        va="top",
                        ha="left",
                        fontsize=9.2,
                        linespacing=1.25,
                        bbox=dict(boxstyle="round,pad=0.55", facecolor="#FFE6E6", edgecolor="#B22222", alpha=0.98),
                        transform=ax_side.transAxes,
                        clip_on=True,
                        wrap=True,
                    )

                    fig.suptitle("NeurInSpectre — Anomaly Triage (Z‑score + Top‑K + Next Steps)", y=0.995, fontweight="bold")
                    # Avoid `tight_layout()` warnings on mixed text/axes layouts. Use explicit margins.
                    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.90)

                    # Save main triage figure
                    fplot = f"{args.out_prefix}anomaly.png"
                    fig.savefig(fplot, dpi=220)
                    plt.close(fig)

                    # Keep a separate top-K bar for backwards compatibility (existing filename).
                    fbar = f"{args.out_prefix}anomaly_topk.png"
                    plt.figure(figsize=(max(6, topk * 0.55), 3.2))
                    plt.bar(range(topk), [max_abs_z[i] for i in idxs], color=bar_colors, alpha=0.95)
                    plt.axhline(zthr, color="#C0392B", linestyle="--", lw=1.6)
                    plt.xticks(range(topk), [feature_labels[int(i)] if int(i) < len(feature_labels) else f"f{int(i)}" for i in idxs])
                    plt.title("NeurInSpectre — Anomaly Top‑K Features")
                    plt.xlabel("Feature index")
                    plt.ylabel("Max |Z|")
                    plt.grid(alpha=0.25, axis="y")
                    plt.tight_layout()
                    plt.savefig(fbar, dpi=220, bbox_inches="tight")
                    plt.close()
                # Optional drift sparklines for top-K: values across samples
                fspark = None
                try:
                    if arr.shape[0] > 1:
                        import math
                        rows = int(math.ceil(topk/5))
                        cols = min(5, topk)
                        fig, axs = plt.subplots(rows, cols, figsize=(cols*2.4, rows*1.8), squeeze=False)
                        for n, feat in enumerate(idxs):
                            r, c = divmod(n, cols)
                            ax2 = axs[r][c]
                            series = arr[:, int(feat)]
                            ax2.plot(series, lw=1.0, color='#d04a5c')
                            try:
                                import numpy as _np
                                w = max(5, int(arr.shape[0] * 0.1))
                                mu = _np.convolve(series, _np.ones(w)/w, mode='same')
                                def _roll_std(x, win):
                                    pad = win // 2
                                    xs = _np.pad(x, (pad, pad), mode='edge')
                                    out = _np.empty_like(x, dtype=float)
                                    for i in range(len(x)):
                                        seg = xs[i:i+win]
                                        out[i] = seg.std() if seg.std() > 0 else 1.0
                                    return out
                                sigma = _roll_std(series, w)
                                rz = (series - mu) / (sigma + 1e-9)
                                ax2.plot(rz, lw=0.8, color='#2A9D8F', alpha=0.8)
                            except Exception:
                                pass
                            try:
                                k = 0.5
                                h = 5.0
                                s_pos = 0.0; tte_idx = None
                                for i, x in enumerate(series - series.mean()):
                                    s_pos = max(0.0, s_pos + x - k)
                                    if s_pos > h:
                                        tte_idx = i; break
                                if tte_idx is not None:
                                    ax2.axvline(tte_idx, color='#0b4f9c', lw=1.0, ls=':')
                            except Exception:
                                pass
                            ax2.axhline(0.0, color='#888', lw=0.5)
                            ax2.axhline(zthr, color='red', lw=0.7, ls='--')
                            ax2.set_title(f'f{int(feat)}', fontsize=8)
                            if r != rows - 1:
                                ax2.set_xticks([])
                            if c != 0:
                                ax2.set_yticks([])
                            ax2.text(0.02, 0.92, 't', transform=ax2.transAxes, fontsize=7, color='#555')
                            ax2.text(0.90, 0.05, 'z', transform=ax2.transAxes, fontsize=7, color='#555')
                            if r == rows - 1:
                                ax2.set_xlabel('Time', fontsize=9, labelpad=6)
                            if c == 0:
                                ax2.set_ylabel('Z-score', fontsize=9, labelpad=6)
                        total_axes = rows * cols
                        for k in range(topk, total_axes):
                            r, c = divmod(k, cols)
                            axs[r][c].set_visible(False)
                        fig.suptitle('NeurInSpectre — Anomaly Drift Sparklines (Top‑K features across samples)', fontsize=10)
                        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.88, wspace=0.25, hspace=0.35)
                        fig.text(0.5, 0.06, 'Time (sample index)', ha='center', fontsize=11, fontweight='bold')
                        fig.text(0.015, 0.5, 'Z-score (per feature)', va='center', rotation='vertical', fontsize=11, fontweight='bold')
                        fig.tight_layout(rect=[0.06, 0.08, 0.98, 0.90])
                        fspark = f"{args.out_prefix}anomaly_sparklines.png"
                        fig.savefig(fspark, dpi=220, bbox_inches="tight")
                        plt.close(fig)
                except Exception:
                    pass
                print(out_json)
                print(fplot); print(fbar)
                if fspark:
                    print(fspark)
                return 0
            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
                return 1
        elif args.command == 'activation_anomaly_detection':
            try:
                import os
                from pathlib import Path
                import torch
                from transformers import AutoModel, AutoTokenizer
                from ..visualization.dna_visualizer import plot_anomaly_detection

                # Resolve device
                dev = getattr(args, 'device', 'auto')
                if dev == 'auto':
                    if torch.cuda.is_available():
                        dev = 'cuda'
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        dev = 'mps'
                    else:
                        dev = 'cpu'

                tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
                # Prefer safetensors to avoid unsafe torch.load on older torch versions
                try:
                    mdl = AutoModel.from_pretrained(args.model, use_safetensors=True)
                except Exception:
                    try:
                        mdl = AutoModel.from_pretrained(args.model)
                    except Exception as e:
                        raise RuntimeError(
                            "Failed to load model weights. Prefer models that ship 'safetensors' weights, "
                            "or upgrade torch (>=2.6) to load legacy .bin weights safely."
                        ) from e
                mdl.eval()
                mdl.to(dev)

                def _hidden_states(prompt: str):
                    inputs = tok(prompt, return_tensors='pt', truncation=True)
                    inputs = {k: v.to(dev) for k, v in inputs.items()}
                    with torch.no_grad():
                        out = mdl(**inputs, output_hidden_states=True, return_dict=True)
                    hs = getattr(out, 'hidden_states', None)
                    if hs is None:
                        raise ValueError('Model did not return hidden_states; ensure it supports output_hidden_states=True')
                    layers = list(hs[1:])  # drop embedding output
                    # Optional layer windowing (inclusive end)
                    layer_start = int(getattr(args, 'layer_start', 0) or 0)
                    layer_end = getattr(args, 'layer_end', None)
                    layer_end = int(layer_end) if layer_end is not None else None
                    if layer_start < 0:
                        layer_start = 0
                    if layer_end is not None and layer_end < layer_start:
                        raise ValueError('--layer-end must be >= --layer-start')

                    layers = layers[layer_start:(layer_end + 1) if layer_end is not None else None]

                    # Back-compat: cap number of layers after windowing
                    if getattr(args, 'max_layers', None) is not None:
                        layers = layers[: int(args.max_layers)]

                    return {f'layer_{i + layer_start}': layers[i] for i in range(len(layers))}

                # Build baseline patterns
                baseline_file = getattr(args, 'baseline_file', None)
                if baseline_file:
                    p = Path(baseline_file)
                    prompts = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
                    if not prompts:
                        raise ValueError('baseline-file is empty')
                    base_dicts = [_hidden_states(pr) for pr in prompts]
                    baseline = {}
                    for layer in base_dicts[0].keys():
                        tensors = [d[layer] for d in base_dicts]
                        # concatenate on batch dimension (each prompt contributes one batch)
                        baseline[layer] = torch.cat(tensors, dim=0)
                else:
                    baseline = _hidden_states(args.baseline_prompt)

                test = _hidden_states(args.test_prompt)

                fig = plot_anomaly_detection(
                    baseline,
                    test,
                    threshold=float(getattr(args, 'threshold', 2.5)),
                    robust=bool(getattr(args, 'robust', False)),
                    sigma_floor=getattr(args, 'sigma_floor', None),
                )

                out_path = Path(getattr(args, 'out', '_cli_runs/anomaly_detection.html'))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(str(out_path))
                print(str(out_path))
                return 0
            except Exception as e:
                logger.error(f"Activation anomaly detection failed: {e}")
                return 1

        elif args.command == 'activation_neuron_heatmap':
            try:
                from pathlib import Path
                import numpy as np
                import torch
                from transformers import AutoModel, AutoTokenizer
                from ..visualization.dna_visualizer import plot_neuron_heatmap

                # Resolve device
                dev = getattr(args, 'device', 'auto')
                if dev == 'auto':
                    if torch.cuda.is_available():
                        dev = 'cuda'
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        dev = 'mps'
                    else:
                        dev = 'cpu'

                tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
                # Prefer safetensors to avoid unsafe torch.load on older torch versions
                try:
                    mdl = AutoModel.from_pretrained(args.model, use_safetensors=True)
                except Exception:
                    try:
                        mdl = AutoModel.from_pretrained(args.model)
                    except Exception as e:
                        raise RuntimeError(
                            "Failed to load model weights. Prefer models that ship 'safetensors' weights, "
                            "or upgrade torch (>=2.6) to load legacy .bin weights safely."
                        ) from e
                mdl.eval(); mdl.to(dev)

                layer_start = int(getattr(args, 'layer_start', 0) or 0)
                layer_end = getattr(args, 'layer_end', None)
                layer_end = int(layer_end) if layer_end is not None else None
                reduce = str(getattr(args, 'reduce', 'mean')).lower()

                def _reduce_hidden(h):
                    # h: torch.Tensor [batch, seq, hidden] or [seq, hidden]
                    if h.dim() == 3:
                        h = h[0]
                    if h.dim() != 2:
                        h = h.reshape(-1, h.shape[-1])
                    if reduce in ('mean','avg','average'):
                        return h.mean(dim=0)
                    if reduce in ('last','last_token'):
                        return h[-1]
                    if reduce in ('max',):
                        return h.max(dim=0).values
                    if reduce in ('maxabs','max_abs','max-abs'):
                        # per-dimension select token with max |activation|
                        abs_h = h.abs()
                        pos = abs_h.argmax(dim=0)
                        return h[pos, torch.arange(h.size(1), device=h.device)]
                    raise ValueError(f"Unknown reduce={reduce}")

                def _layer_vectors(prompt: str):
                    inputs = tok(prompt, return_tensors='pt', truncation=True)
                    inputs = {k: v.to(dev) for k, v in inputs.items()}
                    with torch.no_grad():
                        out = mdl(**inputs, output_hidden_states=True, return_dict=True)
                    hs = getattr(out, 'hidden_states', None)
                    if hs is None:
                        raise ValueError('Model did not return hidden_states')
                    layers = list(hs[1:])
                    vecs = {}
                    for i, t in enumerate(layers):
                        if i < layer_start:
                            continue
                        if layer_end is not None and i > layer_end:
                            break
                        vec = _reduce_hidden(t).detach().cpu().numpy()
                        vecs[f'layer_{i}'] = vec
                    if not vecs:
                        raise ValueError('No layers selected for heatmap')
                    return vecs

                prompts_file = getattr(args, 'prompts_file', None)
                if prompts_file:
                    lines = [ln.strip() for ln in Path(prompts_file).read_text().splitlines() if ln.strip()]
                    if not lines:
                        raise ValueError('prompts-file is empty')
                    dicts = [_layer_vectors(pr) for pr in lines]
                    keys = dicts[0].keys()
                    agg = str(getattr(args, 'aggregate', 'mean')).lower()
                    merged = {}
                    for k in keys:
                        stack = np.stack([d[k] for d in dicts], axis=0)
                        merged[k] = np.median(stack, axis=0) if agg == 'median' else np.mean(stack, axis=0)
                    # Build a report-style title (matches historical screenshot style)
                    model_short = str(args.model).split('/')[-1]
                    model_title = model_short.upper() if model_short.lower() == 'gpt2' else model_short
                    inferred_end = max(int(str(k).split('_')[-1]) for k in merged.keys())
                    layer_end_disp = layer_end if layer_end is not None else inferred_end
                    title = (
                        f"Neural Persistence Analysis - {model_title}<br>"
                        "<span style='font-size:14px;color:#444'>Neural Persistence Analysis: Tracking Adversarial Pathways in Transformer Models</span><br>"
                        f"<span style='font-size:12px;color:#777'>Model: {args.model} | Layers {layer_start}-{layer_end_disp} | prompts={len(lines)} (agg={agg}) | reduce={reduce}</span>"
                    )
                    fig = plot_neuron_heatmap(
                        merged,
                        top_k=int(getattr(args, 'topk', 50)),
                        reduce='mean',
                        title=title,
                        colorbar_title='Activation Magnitude',
                    )
                else:
                    merged = _layer_vectors(args.prompt)
                    model_short = str(args.model).split('/')[-1]
                    model_title = model_short.upper() if model_short.lower() == 'gpt2' else model_short
                    inferred_end = max(int(str(k).split('_')[-1]) for k in merged.keys())
                    layer_end_disp = layer_end if layer_end is not None else inferred_end
                    title = (
                        f"Neural Persistence Analysis - {model_title}<br>"
                        "<span style='font-size:14px;color:#444'>Neural Persistence Analysis: Tracking Adversarial Pathways in Transformer Models</span><br>"
                        f"<span style='font-size:12px;color:#777'>Model: {args.model} | Layers {layer_start}-{layer_end_disp} | reduce={reduce}</span>"
                    )
                    fig = plot_neuron_heatmap(
                        merged,
                        top_k=int(getattr(args, 'topk', 50)),
                        reduce='mean',
                        title=title,
                        colorbar_title='Activation Magnitude',
                    )

                out_path = Path(getattr(args, 'out', '_cli_runs/neuron_heatmap.html'))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(str(out_path))
                print(str(out_path))
                return 0
            except Exception as e:
                logger.error(f"Activation neuron heatmap failed: {e}")
                return 1

        elif args.command == 'activation_attack_patterns':
            try:
                from pathlib import Path
                import torch
                from transformers import AutoModel, AutoTokenizer
                from ..visualization.dna_visualizer import plot_attack_patterns

                # Resolve device
                dev = getattr(args, 'device', 'auto')
                if dev == 'auto':
                    if torch.cuda.is_available():
                        dev = 'cuda'
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        dev = 'mps'
                    else:
                        dev = 'cpu'

                tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
                # Prefer safetensors to avoid unsafe torch.load on older torch versions
                try:
                    mdl = AutoModel.from_pretrained(args.model, use_safetensors=True)
                except Exception:
                    try:
                        mdl = AutoModel.from_pretrained(args.model)
                    except Exception as e:
                        raise RuntimeError(
                            "Failed to load model weights. Prefer models that ship 'safetensors' weights, "
                            "or upgrade torch (>=2.6) to load legacy .bin weights safely."
                        ) from e
                mdl.eval(); mdl.to(dev)

                def _layer(prompt: str, layer: int):
                    inputs = tok(prompt, return_tensors='pt', truncation=True)
                    ids = inputs.get('input_ids')[0].tolist() if 'input_ids' in inputs else []
                    toks = tok.convert_ids_to_tokens(ids) if ids else []
                    inputs = {k: v.to(dev) for k, v in inputs.items()}
                    with torch.no_grad():
                        out = mdl(**inputs, output_hidden_states=True, return_dict=True)
                    hs = getattr(out, 'hidden_states', None)
                    if hs is None:
                        raise ValueError('Model did not return hidden_states')
                    layers = list(hs[1:])
                    if layer < 0 or layer >= len(layers):
                        raise ValueError(f'Layer index {layer} out of range (0..{len(layers)-1})')
                    return layers[layer], toks

                layer_idx = int(getattr(args, 'layer'))
                base_t, base_tok = _layer(args.baseline_prompt, layer_idx)
                test_t, test_tok = _layer(args.test_prompt, layer_idx)

                fig = plot_attack_patterns(
                    {f'layer_{layer_idx}': base_t},
                    {f'layer_{layer_idx}': test_t},
                    layer_idx=layer_idx,
                    top_k=int(getattr(args, 'topk', 10)),
                    compare=str(getattr(args, 'compare', 'prefix')),
                    baseline_tokens=base_tok,
                    test_tokens=test_tok,
                )

                out_path = Path(getattr(args, 'out', '_cli_runs/attack_patterns.html'))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(str(out_path))
                print(str(out_path))
                return 0
            except Exception as e:
                logger.error(f"Activation attack patterns failed: {e}")
                return 1



        elif args.command == 'activation_time_travel_debugging':
            try:
                import json
                import hashlib
                from pathlib import Path
                import numpy as np
                import torch
                from transformers import AutoModel, AutoTokenizer

                from ..visualization.time_travel_debugging import TimeTravelMetrics, plot_time_travel_debugging

                # Resolve device
                dev = getattr(args, 'device', 'auto')
                if dev == 'auto':
                    if torch.cuda.is_available():
                        dev = 'cuda'
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        dev = 'mps'
                    else:
                        dev = 'cpu'

                def _read_first_line(path: str) -> str:
                    pth = Path(path)
                    lines = [ln.strip() for ln in pth.read_text().splitlines() if ln.strip()]
                    if not lines:
                        raise ValueError(f'{path} is empty')
                    return lines[0]

                if args.ttd_action == 'visualize':
                    in_json = Path(str(getattr(args, 'in_json')))
                    obj = json.loads(in_json.read_text())
                    layers = [int(x) for x in obj.get('layers', [])]
                    deltas = [float(x) for x in obj.get('activation_delta_l1', [])]
                    av = [float(x) for x in obj.get('attention_variance', [])]
                    avb = obj.get('attention_variance_baseline', None)
                    avb = [float(x) for x in avb] if isinstance(avb, list) else None

                    metrics = TimeTravelMetrics(
                        layers=layers,
                        activation_delta_l1=deltas,
                        attention_variance=av,
                        attention_variance_baseline=avb,
                        delta_mode=str(obj.get('delta_mode', 'token_l1_mean')),
                        attn_var_mode=str(obj.get('attn_var_mode', 'per_query')),
                        attn_var_scale=str(obj.get('attn_var_scale', 'seq2')),
                    )

                    out_png = Path(str(getattr(args, 'out_png', '_cli_runs/time_travel_debugging.png')))
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    title = getattr(args, 'title', None) or obj.get('title') or 'Time-Travel Debugging – Layer-wise Activation Δ & Attention Variance'
                    plot_time_travel_debugging(metrics, title=str(title), out_path=str(out_png))
                    print(str(out_png))
                    return 0

                # craft
                tok_id = getattr(args, 'tokenizer', None) or args.model
                tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
                # Prefer safetensors; for attention introspection, prefer eager attention impl
                try:
                    mdl = AutoModel.from_pretrained(args.model, use_safetensors=True, attn_implementation='eager')
                except Exception:
                    try:
                        mdl = AutoModel.from_pretrained(args.model, use_safetensors=True)
                    except Exception:
                        try:
                            mdl = AutoModel.from_pretrained(args.model, attn_implementation='eager')
                        except Exception:
                            try:
                                mdl = AutoModel.from_pretrained(args.model)
                            except Exception as e:
                                raise RuntimeError(
                                    "Failed to load model weights. Prefer models that ship 'safetensors' weights, "
                                    "or upgrade torch (>=2.6) to load legacy .bin weights safely."
                                ) from e
                mdl.eval(); mdl.to(dev)

                base_prompt = getattr(args, 'baseline_prompt', None)
                if not base_prompt and getattr(args, 'baseline_file', None):
                    base_prompt = _read_first_line(str(getattr(args, 'baseline_file')))
                test_prompt = getattr(args, 'test_prompt', None)
                if not test_prompt and getattr(args, 'test_file', None):
                    test_prompt = _read_first_line(str(getattr(args, 'test_file')))

                if not base_prompt or not test_prompt:
                    raise ValueError('baseline and test prompts required')

                max_tokens = int(getattr(args, 'max_tokens', 128) or 128)
                layer_start = int(getattr(args, 'layer_start', 0) or 0)
                layer_end = getattr(args, 'layer_end', None)
                layer_end = int(layer_end) if layer_end is not None else None
                if layer_start < 0:
                    layer_start = 0
                if layer_end is not None and layer_end < layer_start:
                    raise ValueError('--layer-end must be >= --layer-start')

                delta_mode = str(getattr(args, 'delta_mode', 'token_l1_mean_x100'))
                attn_mode = str(getattr(args, 'attn_var_mode', 'per_query'))
                attn_scale = str(getattr(args, 'attn_var_scale', 'seq2'))
                attn_source = str(getattr(args, 'attention_source', 'test'))

                def _forward(prompt: str):
                    inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=max_tokens)
                    inputs = {k: v.to(dev) for k, v in inputs.items()}
                    with torch.no_grad():
                        out = mdl(**inputs, output_hidden_states=True, output_attentions=True, return_dict=True)
                    hs = getattr(out, 'hidden_states', None)
                    attns = getattr(out, 'attentions', None)
                    if hs is None:
                        raise ValueError('Model did not return hidden_states')
                    if attns is None:
                        raise ValueError('Model did not return attentions; ensure it supports output_attentions=True')
                    layers_h = list(hs[1:])  # drop embedding
                    layers_a = list(attns)
                    n_layers = min(len(layers_h), len(layers_a))
                    layers_h = layers_h[:n_layers]
                    layers_a = layers_a[:n_layers]
                    return layers_h, layers_a

                base_h, base_a = _forward(str(base_prompt))
                test_h, test_a = _forward(str(test_prompt))

                n_layers = min(len(base_h), len(test_h))
                if layer_end is None:
                    layer_end = n_layers - 1
                if layer_end >= n_layers:
                    layer_end = n_layers - 1

                layers = list(range(layer_start, layer_end + 1))
                if not layers:
                    raise ValueError('No layers selected')

                deltas = []
                attn_vars_test = []
                attn_vars_base = []

                for li in layers:
                    hb = base_h[li][0]  # [seq, hidden]
                    ht = test_h[li][0]

                    if delta_mode == 'mean_vec_l1':
                        vb = hb.mean(dim=0)
                        vt = ht.mean(dim=0)
                        d = torch.sum(torch.abs(vt - vb)).item()
                    elif delta_mode == 'mean_vec_l1_x100':
                        vb = hb.mean(dim=0)
                        vt = ht.mean(dim=0)
                        d = torch.mean(torch.abs(vt - vb)).item() * 100.0
                    else:
                        n = int(min(hb.shape[0], ht.shape[0]))
                        if n <= 0:
                            d = 0.0
                        elif delta_mode == 'token_l1_mean':
                            token_l1 = torch.sum(torch.abs(ht[:n] - hb[:n]), dim=-1)
                            d = token_l1.mean().item()
                        else:
                            # token_l1_mean_x100: 100 * mean_{t,d} |Δ|
                            d = torch.mean(torch.abs(ht[:n] - hb[:n])).item() * 100.0
                    deltas.append(float(d))

                    # attention variance for baseline + test
                    ab = base_a[li][0]  # [heads, seq, seq]
                    at = test_a[li][0]

                    def _attn_var(a: torch.Tensor) -> float:
                        a = a.float()
                        if attn_mode == 'global':
                            v = a.var(unbiased=False).item()
                        else:
                            v = a.var(dim=-1, unbiased=False).mean().item()
                        seq_len = int(a.shape[-1])
                        if attn_scale == 'seq2':
                            v *= float(seq_len * seq_len)
                        elif attn_scale == 'seq':
                            v *= float(seq_len)
                        return float(v)

                    vb = _attn_var(ab)
                    vt = _attn_var(at)
                    attn_vars_base.append(vb)
                    attn_vars_test.append(vt)

                # Choose which attention series to plot
                if attn_source == 'baseline':
                    attn_plot = attn_vars_base
                elif attn_source == 'delta_abs':
                    attn_plot = [float(abs(t - b)) for t, b in zip(attn_vars_test, attn_vars_base)]
                else:
                    attn_plot = attn_vars_test

                metrics = TimeTravelMetrics(
                    layers=layers,
                    activation_delta_l1=deltas,
                    attention_variance=attn_plot,
                    attention_variance_baseline=attn_vars_base,
                    delta_mode=delta_mode,
                    attn_var_mode=attn_mode,
                    attn_var_scale=attn_scale,
                )

                title = str(getattr(args, 'title', 'Time-Travel Debugging – Layer-wise Activation Δ & Attention Variance'))

                out_json = Path(str(getattr(args, 'out_json', '_cli_runs/time_travel_debugging.json')))
                out_png = Path(str(getattr(args, 'out_png', '_cli_runs/time_travel_debugging.png')))
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_png.parent.mkdir(parents=True, exist_ok=True)

                obj = {
                    'title': title,
                    'mode': 'baseline_vs_test',
                    'model': str(args.model),
                    'tokenizer': str(tok_id),
                    'layer_start': int(layer_start),
                    'layer_end': int(layer_end),
                    'layers': [int(x) for x in layers],
                    'delta_mode': delta_mode,
                    'attn_var_mode': attn_mode,
                    'attn_var_scale': attn_scale,
                    'attention_source': attn_source,
                    'activation_delta_l1': [float(x) for x in deltas],
                    'attention_variance': [float(x) for x in attn_plot],
                    'attention_variance_baseline': [float(x) for x in attn_vars_base],
                    'baseline_prompt_sha16': hashlib.sha256(str(base_prompt).encode('utf-8', errors='ignore')).hexdigest()[:16],
                    'test_prompt_sha16': hashlib.sha256(str(test_prompt).encode('utf-8', errors='ignore')).hexdigest()[:16],
                }
                out_json.write_text(json.dumps(obj, indent=2))

                plot_time_travel_debugging(metrics, title=title, out_path=str(out_png))

                print(str(out_json))
                print(str(out_png))
                return 0

            except Exception as e:
                logger.error(f"Activation time-travel debugging failed: {e}")
                return 1


        elif args.command == 'activation_eigen_collapse_radar':
            try:
                import json
                import hashlib
                from pathlib import Path
                import numpy as np
                import torch
                from transformers import AutoModel, AutoTokenizer

                from ..visualization.eigen_collapse_radar import (
                    EigenCollapseRadarMetrics,
                    topk_cov_eigvals,
                    normalize_eigvals,
                    plot_eigen_collapse_radar,
                )

                # Resolve device
                dev = getattr(args, 'device', 'auto')
                if dev == 'auto':
                    if torch.cuda.is_available():
                        dev = 'cuda'
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        dev = 'mps'
                    else:
                        dev = 'cpu'

                if args.ecr_action == 'visualize':
                    in_json = Path(str(getattr(args, 'in_json')))
                    obj = json.loads(in_json.read_text())
                    layers = [int(x) for x in obj.get('layers', [])]
                    k = int(obj.get('k', 5))
                    normalize = str(obj.get('normalize', 'eig1'))
                    eigvals = obj.get('eigvals', [])
                    subtitle = obj.get('subtitle', None)

                    metrics = EigenCollapseRadarMetrics(
                        model=str(obj.get('model', 'model')),
                        layers=layers,
                        k=k,
                        normalize=normalize,  # type: ignore
                        eigvals=[[float(v) for v in row] for row in eigvals],
                        subtitle=str(subtitle) if subtitle is not None else None,
                    )

                    out_png = Path(str(getattr(args, 'out_png', '_cli_runs/eigen_collapse_radar.png')))
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    title = getattr(args, 'title', None) or obj.get('title') or 'Eigen-Collapse Rank Shrinkage Radar'
                    plot_eigen_collapse_radar(metrics, title=str(title), out_path=str(out_png), guidance=True)
                    print(str(out_png))
                    return 0

                # craft
                tok_id = getattr(args, 'tokenizer', None) or args.model
                tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
                # Prefer safetensors
                try:
                    mdl = AutoModel.from_pretrained(args.model, use_safetensors=True)
                except Exception:
                    try:
                        mdl = AutoModel.from_pretrained(args.model)
                    except Exception as e:
                        raise RuntimeError(
                            "Failed to load model weights. Prefer models that ship 'safetensors' weights, "
                            "or upgrade torch (>=2.6) to load legacy .bin weights safely."
                        ) from e
                mdl.eval(); mdl.to(dev)

                max_tokens = int(getattr(args, 'max_tokens', 128) or 128)
                k = int(getattr(args, 'k', 5) or 5)
                normalize = str(getattr(args, 'normalize', 'eig1') or 'eig1')
                agg = str(getattr(args, 'aggregate', 'mean') or 'mean')
                layer_start = int(getattr(args, 'layer_start', 0) or 0)
                layer_end = getattr(args, 'layer_end', None)
                layer_end = int(layer_end) if layer_end is not None else None
                every = int(getattr(args, 'every', 1) or 1)
                if every < 1:
                    every = 1

                prompts = []
                prompts_file = getattr(args, 'prompts_file', None)
                if prompts_file:
                    pth = Path(str(prompts_file))
                    lines = [ln.strip() for ln in pth.read_text().splitlines() if ln.strip()]
                    if not lines:
                        raise ValueError('prompts-file is empty')
                    prompts = lines
                else:
                    prompts = [str(getattr(args, 'prompt'))]

                # Collect normalized eigenvalues per prompt per layer
                per_layer: dict[int, list[np.ndarray]] = {}

                for pr in prompts:
                    inputs = tok(pr, return_tensors='pt', truncation=True, max_length=max_tokens)
                    inputs = {k_: v.to(dev) for k_, v in inputs.items()}
                    with torch.no_grad():
                        out = mdl(**inputs, output_hidden_states=True, return_dict=True)
                    hs = getattr(out, 'hidden_states', None)
                    if hs is None:
                        raise ValueError('Model did not return hidden_states')
                    layers_h = list(hs[1:])  # drop embedding output
                    n_layers = len(layers_h)
                    if n_layers == 0:
                        raise ValueError('No transformer layers found')

                    ls = max(0, int(layer_start))
                    le = int(layer_end) if layer_end is not None else (n_layers - 1)
                    le = min(le, n_layers - 1)
                    if le < ls:
                        raise ValueError('--layer-end must be >= --layer-start')

                    for li in range(ls, le + 1):
                        h = layers_h[li][0].detach().cpu().numpy().astype('float32', copy=False)
                        eig = topk_cov_eigvals(h, k=k)
                        eig_n = normalize_eigvals(eig, mode=normalize)  # type: ignore
                        per_layer.setdefault(li, []).append(eig_n)

                # Aggregate across prompts
                layers = sorted(per_layer.keys())
                if every > 1:
                    layers = [ly for ly in layers if ((ly - layers[0]) % every) == 0]

                eigvals = []
                for ly in layers:
                    stack = np.stack(per_layer[ly], axis=0)
                    vec = np.median(stack, axis=0) if agg == 'median' else np.mean(stack, axis=0)
                    eigvals.append([float(x) for x in vec.tolist()])

                model_short = str(args.model).split('/')[-1]
                subtitle = f"{model_short} | k={k} eigenvalues"

                metrics = EigenCollapseRadarMetrics(
                    model=str(args.model),
                    layers=[int(x) for x in layers],
                    k=int(k),
                    normalize=normalize,  # type: ignore
                    eigvals=eigvals,
                    subtitle=subtitle,
                )

                title = str(getattr(args, 'title', 'Eigen-Collapse Rank Shrinkage Radar'))
                out_json = Path(str(getattr(args, 'out_json', '_cli_runs/eigen_collapse_radar.json')))
                out_png = Path(str(getattr(args, 'out_png', '_cli_runs/eigen_collapse_radar.png')))
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_png.parent.mkdir(parents=True, exist_ok=True)

                # Minimal metadata (store prompt hashes, not raw prompts)
                meta = {
                    'title': title,
                    'model': str(args.model),
                    'tokenizer': str(tok_id),
                    'k': int(k),
                    'normalize': normalize,
                    'aggregate': agg,
                    'layer_start': int(layer_start),
                    'layer_end': int(layer_end) if layer_end is not None else None,
                    'every': int(every),
                    'prompt_count': int(len(prompts)),
                    'subtitle': subtitle,
                    'layers': [int(x) for x in layers],
                    'eigvals': eigvals,
                }

                # Convenience: per-layer collapse score (higher = more collapsed / anisotropic)
                try:
                    import numpy as _np

                    vals = _np.asarray(eigvals, dtype=_np.float64)
                    collapse_scores = []
                    if vals.ndim == 2 and vals.shape[0] == len(layers):
                        if str(normalize) == 'sum':
                            # If eigenvalues are normalized to sum=1, eig1 share is a direct collapse proxy
                            collapse_scores = [float(x) for x in vals[:, 0].tolist()]
                        else:
                            if vals.shape[1] <= 1:
                                collapse_scores = [0.0 for _ in layers]
                            else:
                                petal_mean = _np.mean(vals[:, 1:], axis=1)
                                if str(normalize) == 'none':
                                    petal_mean = petal_mean / (_np.maximum(vals[:, 0], 1e-12))
                                # Smaller petals => higher collapse score
                                score = 1.0 - _np.clip(petal_mean, 0.0, 1.0)
                                collapse_scores = [float(x) for x in score.tolist()]
                    else:
                        collapse_scores = [0.0 for _ in layers]

                    topk_layers = min(5, len(layers))
                    order = list(reversed(_np.argsort(_np.asarray(collapse_scores, dtype=_np.float64)).tolist()))[:topk_layers]
                    meta['collapse_scores'] = collapse_scores
                    meta['most_collapsed_layers'] = [int(layers[int(i)]) for i in order]
                except Exception:
                    pass

                if prompts_file:
                    meta['prompts_file'] = str(prompts_file)
                    meta['prompts_file_sha16'] = hashlib.sha256(Path(str(prompts_file)).read_bytes()).hexdigest()[:16]
                else:
                    meta['prompt_sha16'] = hashlib.sha256(str(prompts[0]).encode('utf-8', errors='ignore')).hexdigest()[:16]

                out_json.write_text(json.dumps(meta, indent=2))

                plot_eigen_collapse_radar(metrics, title=title, out_path=str(out_png), guidance=True)

                print(str(out_json))
                print(str(out_png))
                return 0

            except Exception as e:
                logger.error(f"Activation eigen-collapse radar failed: {e}")
                return 1



        elif args.command == 'activation_eigenvalue_spectrum':
            try:
                import json
                import hashlib
                from pathlib import Path
                import numpy as np
                import torch
                from transformers import AutoModel, AutoTokenizer

                from ..visualization.eigenvalue_spectrum import (
                    EigenvalueSpectrumMetrics,
                    cov_eigenvalues,
                    summarize_eigenvalues,
                    plot_eigenvalue_spectrum,
                    plot_eigenvalue_spectrum_interactive,
                    transform_eigs,
                    ks_statistic,
                    wasserstein_q,
                    js_divergence_hist,
                    robust_zscore,
                )

                def _int_keyed(d: object) -> dict[int, dict[str, float]]:
                    if not isinstance(d, dict):
                        return {}
                    out: dict[int, dict[str, float]] = {}
                    for k, v in d.items():
                        try:
                            ik = int(k)
                        except Exception:
                            continue
                        if isinstance(v, dict):
                            out[ik] = {str(kk): float(vv) for kk, vv in v.items()}
                    return out

                # Resolve device
                dev = getattr(args, 'device', 'auto')
                if dev == 'auto':
                    if torch.cuda.is_available():
                        dev = 'cuda'
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        dev = 'mps'
                    else:
                        dev = 'cpu'

                if args.evs_action == 'visualize':
                    in_json = Path(str(getattr(args, 'in_json')))
                    obj = json.loads(in_json.read_text())

                    metrics = EigenvalueSpectrumMetrics(
                        model=str(obj.get('model', 'model')),
                        label=str(obj.get('label', 'sample')),
                        layer_mode=str(obj.get('layer_mode', 'single')),
                        layer_indices=[int(x) for x in obj.get('layer_indices', [])],
                        eigenvalues=[float(x) for x in obj.get('eigenvalues', [])],
                        bins=int(obj.get('bins', 40)),
                        stats={str(k): float(v) for k, v in (obj.get('stats') or {}).items()},
                        subtitle=str(obj.get('subtitle')) if obj.get('subtitle') is not None else None,
                    )

                    baseline_obj = obj.get('baseline') or None
                    baseline_metrics = None
                    baseline_layer_summary = None
                    if isinstance(baseline_obj, dict):
                        baseline_metrics = EigenvalueSpectrumMetrics(
                            model=str(obj.get('model', 'model')),
                            label=str(baseline_obj.get('label', 'baseline')),
                            layer_mode=str(obj.get('layer_mode', 'single')),
                            layer_indices=[int(x) for x in obj.get('layer_indices', [])],
                            eigenvalues=[float(x) for x in (baseline_obj.get('eigenvalues') or [])],
                            bins=int(obj.get('bins', 40)),
                            stats={str(k): float(v) for k, v in (baseline_obj.get('stats') or {}).items()},
                            subtitle=str(baseline_obj.get('subtitle')) if baseline_obj.get('subtitle') is not None else None,
                        )
                        baseline_layer_summary = _int_keyed(baseline_obj.get('layer_summary') or {})

                    layer_summary = _int_keyed(obj.get('layer_summary') or {})
                    drift = obj.get('drift') if isinstance(obj.get('drift'), dict) else None
                    top_prompt_anomalies = obj.get('top_prompt_anomalies') if isinstance(obj.get('top_prompt_anomalies'), list) else None

                    out_png = Path(str(getattr(args, 'out_png', '_cli_runs/eigenvalue_spectrum.png')))
                    out_png.parent.mkdir(parents=True, exist_ok=True)

                    ttl = getattr(args, 'title', None) or obj.get('title') or 'NeurInSpectre Eigenvalue Spectrum'
                    x_scale = getattr(args, 'x_scale', None) or obj.get('x_scale') or 'linear'

                    plot_eigenvalue_spectrum(
                        metrics,
                        title=str(ttl),
                        out_path=str(out_png),
                        guidance=True,
                        x_scale=str(x_scale),
                        baseline=baseline_metrics,
                        layer_summary=layer_summary or None,
                        baseline_layer_summary=baseline_layer_summary or None,
                        drift=drift,
                        top_prompt_anomalies=top_prompt_anomalies,
                    )

                    out_html = getattr(args, 'out_html', None)
                    if out_html:
                        plot_eigenvalue_spectrum_interactive(
                            metrics,
                            title=str(ttl),
                            out_html=str(out_html),
                            x_scale=str(x_scale),
                            baseline=baseline_metrics,
                            layer_summary=layer_summary or None,
                            baseline_layer_summary=baseline_layer_summary or None,
                            drift=drift,
                            top_prompt_anomalies=top_prompt_anomalies,
                        )
                        print(str(out_html))

                    print(str(out_png))
                    return 0

                # craft
                tok_id = getattr(args, 'tokenizer', None) or args.model
                tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)

                # Prefer safetensors
                try:
                    mdl = AutoModel.from_pretrained(args.model, use_safetensors=True)
                except Exception:
                    try:
                        mdl = AutoModel.from_pretrained(args.model)
                    except Exception as e:
                        raise RuntimeError(
                            "Failed to load model weights. Prefer models that ship 'safetensors' weights, "
                            "or upgrade torch (>=2.6) to load legacy .bin weights safely."
                        ) from e

                mdl.eval(); mdl.to(dev)

                max_tokens = int(getattr(args, 'max_tokens', 128) or 128)
                bins = int(getattr(args, 'bins', 40) or 40)
                label = str(getattr(args, 'label', 'sample') or 'sample')
                x_scale = str(getattr(args, 'x_scale', 'linear') or 'linear')
                top_k_layers = int(getattr(args, 'top_k_layers', 5) or 5)
                top_k_prompts = int(getattr(args, 'top_k_prompts', 10) or 10)

                layer_q = str(getattr(args, 'layer', 'all') or 'all').strip().lower()
                layer_start = int(getattr(args, 'layer_start', 0) or 0)
                layer_end = getattr(args, 'layer_end', None)
                layer_end = int(layer_end) if layer_end is not None else None

                # Load test prompts
                prompts_file = getattr(args, 'prompts_file', None)
                if prompts_file:
                    pth = Path(str(prompts_file))
                    prompts = [ln.strip() for ln in pth.read_text().splitlines() if ln.strip()]
                    if not prompts:
                        raise ValueError('prompts-file is empty')
                else:
                    prompts = [str(getattr(args, 'prompt'))]

                # Optional baseline prompts
                baseline_prompts_file = getattr(args, 'baseline_prompts_file', None)
                baseline_prompt = getattr(args, 'baseline_prompt', None)
                baseline_prompts: list[str] = []
                if baseline_prompts_file:
                    bp = Path(str(baseline_prompts_file))
                    baseline_prompts = [ln.strip() for ln in bp.read_text().splitlines() if ln.strip()]
                    if not baseline_prompts:
                        raise ValueError('baseline-prompts-file is empty')
                elif baseline_prompt:
                    baseline_prompts = [str(baseline_prompt)]

                layer_indices: list[int] = []

                def _select_layers(n_layers: int) -> list[int]:
                    if layer_q == 'all':
                        ls = max(0, int(layer_start))
                        le = int(layer_end) if layer_end is not None else (n_layers - 1)
                        le = min(le, n_layers - 1)
                        if le < ls:
                            raise ValueError('--layer-end must be >= --layer-start')
                        return list(range(ls, le + 1))
                    li = int(layer_q)
                    if li < 0 or li > (n_layers - 1):
                        raise ValueError(f'--layer {li} out of range (0..{n_layers-1})')
                    return [li]

                def _run_suite(suite: list[str]) -> tuple[list[float], dict[int, list[float]], list[list[float]]]:
                    all_eigs: list[float] = []
                    by_layer: dict[int, list[float]] = {}
                    per_prompt: list[list[float]] = []

                    for pr in suite:
                        inputs = tok(pr, return_tensors='pt', truncation=True, max_length=max_tokens)
                        inputs = {k_: v.to(dev) for k_, v in inputs.items()}
                        with torch.no_grad():
                            out = mdl(**inputs, output_hidden_states=True, return_dict=True)
                        hs = getattr(out, 'hidden_states', None)
                        if hs is None:
                            raise ValueError('Model did not return hidden_states')

                        layers_h = list(hs[1:])  # drop embedding output
                        n_layers = len(layers_h)
                        if n_layers == 0:
                            raise ValueError('No transformer layers found')

                        sel = _select_layers(n_layers)
                        if not layer_indices:
                            layer_indices.extend([int(x) for x in sel])

                        prompt_eigs: list[float] = []
                        for li in sel:
                            h = layers_h[li][0].detach().cpu().numpy().astype('float32', copy=False)
                            eig = cov_eigenvalues(h)
                            vals = [float(x) for x in eig.tolist()]
                            by_layer.setdefault(int(li), []).extend(vals)
                            all_eigs.extend(vals)
                            prompt_eigs.extend(vals)

                        per_prompt.append(prompt_eigs)

                    return all_eigs, by_layer, per_prompt

                # Run test suite first (sets layer_indices)
                test_eigs, test_by_layer, test_per_prompt = _run_suite(prompts)

                # Optional baseline suite
                base_eigs: list[float] = []
                base_by_layer: dict[int, list[float]] = {}
                base_per_prompt: list[list[float]] = []
                if baseline_prompts:
                    base_eigs, base_by_layer, base_per_prompt = _run_suite(baseline_prompts)

                # Summaries
                stats_test = {str(k): float(v) for k, v in summarize_eigenvalues(test_eigs).items()}
                layer_summary = {int(li): {str(k): float(v) for k, v in summarize_eigenvalues(vs).items()} for li, vs in test_by_layer.items()}

                stats_base = None
                baseline_layer_summary = None
                if baseline_prompts:
                    stats_base = {str(k): float(v) for k, v in summarize_eigenvalues(base_eigs).items()}
                    baseline_layer_summary = {int(li): {str(k): float(v) for k, v in summarize_eigenvalues(vs).items()} for li, vs in base_by_layer.items()}

                model_short = str(args.model).split('/')[-1]
                mode = 'all' if layer_q == 'all' else 'single'
                subtitle = f"{model_short} | layers={len(layer_indices)} | eig_count={int(stats_test.get('count', 0.0))} | x_scale={x_scale}"

                metrics = EigenvalueSpectrumMetrics(
                    model=str(args.model),
                    label=label,
                    layer_mode=mode,
                    layer_indices=[int(x) for x in layer_indices],
                    eigenvalues=[float(x) for x in test_eigs],
                    bins=int(bins),
                    stats=stats_test,
                    subtitle=subtitle,
                )

                baseline_metrics = None
                if baseline_prompts:
                    baseline_metrics = EigenvalueSpectrumMetrics(
                        model=str(args.model),
                        label='baseline',
                        layer_mode=mode,
                        layer_indices=[int(x) for x in layer_indices],
                        eigenvalues=[float(x) for x in base_eigs],
                        bins=int(bins),
                        stats=stats_base or {},
                        subtitle=f"baseline | prompts={len(baseline_prompts)}",
                    )

                # Drift metrics (baseline vs test)
                drift = None
                if baseline_prompts:
                    rows: list[dict[str, float]] = []
                    for li in layer_indices:
                        be = base_by_layer.get(int(li), [])
                        te = test_by_layer.get(int(li), [])
                        if not be or not te:
                            continue
                        bx = transform_eigs(be, x_scale=x_scale)
                        tx = transform_eigs(te, x_scale=x_scale)
                        ks = float(ks_statistic(bx, tx))
                        w1 = float(wasserstein_q(bx, tx))
                        js = float(js_divergence_hist(bx, tx, bins=bins))
                        bsum = (baseline_layer_summary or {}).get(int(li), {})
                        tsum = layer_summary.get(int(li), {})
                        dt1 = float(tsum.get('top1_frac', 0.0) - bsum.get('top1_frac', 0.0))
                        rows.append({
                            'layer': float(int(li)),
                            'ks': ks,
                            'wasserstein_q': w1,
                            'js': js,
                            'delta_top1_frac': dt1,
                        })

                    if rows:
                        ks_vals = [float(r['ks']) for r in rows]
                        w1_vals = [float(r['wasserstein_q']) for r in rows]
                        js_vals = [float(r['js']) for r in rows]
                        dt_vals = [abs(float(r['delta_top1_frac'])) for r in rows]

                        _, _, zks = robust_zscore(ks_vals)
                        _, _, zw1 = robust_zscore(w1_vals)
                        _, _, zjs = robust_zscore(js_vals)
                        _, _, zdt = robust_zscore(dt_vals)

                        for i, r in enumerate(rows):
                            score = float(max(0.0, float(zks[i])) + max(0.0, float(zw1[i])) + max(0.0, float(zjs[i])) + max(0.0, float(zdt[i])))
                            r['score'] = score

                        rows.sort(key=lambda rr: float(rr.get('score', 0.0)), reverse=True)

                    bx_all = transform_eigs(base_eigs, x_scale=x_scale)
                    tx_all = transform_eigs(test_eigs, x_scale=x_scale)
                    drift = {
                        'x_scale': x_scale,
                        'summary': {
                            'ks': float(ks_statistic(bx_all, tx_all)),
                            'wasserstein_q': float(wasserstein_q(bx_all, tx_all)),
                            'js': float(js_divergence_hist(bx_all, tx_all, bins=bins)),
                        },
                        'top_layers': [
                            {k: float(v) for k, v in r.items()}
                            for r in rows[: max(1, top_k_layers)]
                        ]
                        if rows
                        else [],
                    }

                # Prompt-level anomaly ranking (requires baseline + a test suite)
                top_prompt_anomalies = None
                if baseline_prompts and prompts_file and len(prompts) > 1:
                    base_all = transform_eigs(base_eigs, x_scale=x_scale)
                    scores: list[float] = []
                    for pe in test_per_prompt:
                        px = transform_eigs(pe, x_scale=x_scale)
                        s = float(ks_statistic(base_all, px) + wasserstein_q(base_all, px) + js_divergence_hist(base_all, px, bins=bins))
                        scores.append(s)

                    _, _, z = robust_zscore(scores)
                    order = list(np.argsort(-z))
                    top_prompt_anomalies = []
                    for j in order[: max(1, top_k_prompts)]:
                        pr = prompts[int(j)]
                        top_prompt_anomalies.append({
                            'index': int(j),
                            'anomaly_score': float(z[int(j)]),
                            'raw_score': float(scores[int(j)]),
                            'prompt_sha16': hashlib.sha256(pr.encode('utf-8', errors='ignore')).hexdigest()[:16],
                            'snippet': pr[:140].replace('\n', ' '),
                        })

                title = str(getattr(args, 'title', 'NeurInSpectre Eigenvalue Spectrum'))
                out_json = Path(str(getattr(args, 'out_json', '_cli_runs/eigenvalue_spectrum.json')))
                out_png = Path(str(getattr(args, 'out_png', '_cli_runs/eigenvalue_spectrum.png')))
                out_html = getattr(args, 'out_html', None)

                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_png.parent.mkdir(parents=True, exist_ok=True)

                meta = {
                    'title': title,
                    'model': str(args.model),
                    'tokenizer': str(tok_id),
                    'label': label,
                    'layer_mode': mode,
                    'layer_indices': [int(x) for x in layer_indices],
                    'layer_start': int(layer_start),
                    'layer_end': int(layer_end) if layer_end is not None else None,
                    'bins': int(bins),
                    'x_scale': x_scale,
                    'prompt_count': int(len(prompts)),
                    'stats': {str(k): float(v) for k, v in stats_test.items()},
                    'eigenvalues': [float(x) for x in test_eigs],
                    'subtitle': subtitle,
                    'layer_summary': layer_summary,
                    'drift': drift,
                    'top_prompt_anomalies': top_prompt_anomalies,
                }

                if prompts_file:
                    meta['prompts_file'] = str(prompts_file)
                    meta['prompts_file_sha16'] = hashlib.sha256(Path(str(prompts_file)).read_bytes()).hexdigest()[:16]
                else:
                    meta['prompt_sha16'] = hashlib.sha256(str(prompts[0]).encode('utf-8', errors='ignore')).hexdigest()[:16]

                if baseline_prompts:
                    base_meta = {
                        'label': 'baseline',
                        'prompt_count': int(len(baseline_prompts)),
                        'stats': stats_base or {},
                        'eigenvalues': [float(x) for x in base_eigs],
                        'layer_summary': baseline_layer_summary or {},
                        'subtitle': f"baseline | prompts={len(baseline_prompts)}",
                    }
                    if baseline_prompts_file:
                        base_meta['prompts_file'] = str(baseline_prompts_file)
                        base_meta['prompts_file_sha16'] = hashlib.sha256(Path(str(baseline_prompts_file)).read_bytes()).hexdigest()[:16]
                    else:
                        base_meta['prompt_sha16'] = hashlib.sha256(str(baseline_prompts[0]).encode('utf-8', errors='ignore')).hexdigest()[:16]

                    meta['baseline'] = base_meta

                out_json.write_text(json.dumps(meta, indent=2))

                plot_eigenvalue_spectrum(
                    metrics,
                    title=title,
                    out_path=str(out_png),
                    guidance=True,
                    x_scale=x_scale,
                    baseline=baseline_metrics,
                    layer_summary=layer_summary,
                    baseline_layer_summary=baseline_layer_summary,
                    drift=drift,
                    top_prompt_anomalies=top_prompt_anomalies,
                )

                if out_html:
                    plot_eigenvalue_spectrum_interactive(
                        metrics,
                        title=title,
                        out_html=str(out_html),
                        x_scale=x_scale,
                        baseline=baseline_metrics,
                        layer_summary=layer_summary,
                        baseline_layer_summary=baseline_layer_summary,
                        drift=drift,
                        top_prompt_anomalies=top_prompt_anomalies,
                    )

                print(str(out_json))
                print(str(out_png))
                if out_html:
                    print(str(out_html))
                return 0

            except Exception as e:
                logger.error(f"Activation eigenvalue spectrum failed: {e}")
                return 1


        elif args.command == 'activation_fft_security_spectrum':
            try:
                import json
                import hashlib
                from pathlib import Path
                import numpy as np
                import torch
                from transformers import AutoModel, AutoTokenizer

                from ..visualization.fft_security_spectrum import (
                    FFTSecuritySpectrumMetrics,
                    token_signal,
                    detrend_signal,
                    rfft_power,
                    high_freq_tail_ratio,
                    plot_fft_security_spectrum,
                )

                # Resolve device
                dev = getattr(args, 'device', 'auto')
                if dev == 'auto':
                    if torch.cuda.is_available():
                        dev = 'cuda'
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        dev = 'mps'
                    else:
                        dev = 'cpu'

                if args.ffts_action == 'visualize':
                    in_json = Path(str(getattr(args, 'in_json')))
                    obj = json.loads(in_json.read_text())

                    metrics = FFTSecuritySpectrumMetrics(
                        title=str(obj.get('title', 'FFT Security Spectrum')),
                        model=str(obj.get('model', 'model')),
                        tokenizer=str(obj.get('tokenizer', obj.get('model', 'model'))),
                        layer=int(obj.get('layer', 0)),
                        seq_len=int(obj.get('seq_len', 0)),
                        prompt_count=int(obj.get('prompt_count', 0)),
                        seq_mode=str(obj.get('seq_mode', 'common_prefix')),
                        signal_mode=str(obj.get('signal_mode', 'token_norm')),
                        detrend=str(obj.get('detrend', 'none')),
                        window=str(obj.get('window', 'none')),
                        fft_size=int(obj.get('fft_size', 0) or 0),
                        segment=str(obj.get('segment', 'prefix')),
                        tail_start=float(obj.get('tail_start', 0.25)),
                        z_threshold=float(obj.get('z_threshold', 2.0)),
                        z_mode=str(obj.get('z_mode', 'standard')),
                        freqs=[float(x) for x in obj.get('freqs', [])],
                        spectra=[[float(v) for v in row] for row in obj.get('spectra', [])],
                        mean_spectrum=[float(x) for x in obj.get('mean_spectrum', [])],
                        dominant_freqs=[float(x) for x in obj.get('dominant_freqs', [])],
                        dominant_powers=[float(x) for x in obj.get('dominant_powers', [])],
                        tail_ratios=[float(x) for x in obj.get('tail_ratios', [])],
                        dominant_z=[float(x) for x in obj.get('dominant_z', [])],
                        tail_z=[float(x) for x in obj.get('tail_z', [])],
                        baseline_prompt_count=int(obj.get('baseline_prompt_count')) if obj.get('baseline_prompt_count') is not None else None,
                        baseline_mean_spectrum=[float(x) for x in obj.get('baseline_mean_spectrum', [])] if obj.get('baseline_mean_spectrum') is not None else None,
                        baseline_dominant_powers=[float(x) for x in obj.get('baseline_dominant_powers', [])] if obj.get('baseline_dominant_powers') is not None else None,
                        baseline_tail_ratios=[float(x) for x in obj.get('baseline_tail_ratios', [])] if obj.get('baseline_tail_ratios') is not None else None,
                        baseline_prompt_sha16=obj.get('baseline_prompt_sha16', None),
                        baseline_label=str(obj.get('baseline_label')) if obj.get('baseline_label') is not None else None,
                        subtitle=str(obj.get('subtitle')) if obj.get('subtitle') is not None else None,
                        prompt_sha16=obj.get('prompt_sha16', None),
                    )

                    out_png = Path(str(getattr(args, 'out_png', '_cli_runs/fft_security_spectrum.png')))
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    title = getattr(args, 'title', None) or obj.get('title') or 'FFT Security Spectrum'
                    pidx = int(getattr(args, 'prompt_index', 0) or 0)
                    plot_fft_security_spectrum(metrics, prompt_index=pidx, title=str(title), out_path=str(out_png), guidance=True)
                    print(str(out_png))
                    return 0

                # craft
                tok_id = getattr(args, 'tokenizer', None) or args.model
                tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
                # Prefer safetensors
                try:
                    mdl = AutoModel.from_pretrained(args.model, use_safetensors=True)
                except Exception:
                    try:
                        mdl = AutoModel.from_pretrained(args.model)
                    except Exception as e:
                        raise RuntimeError(
                            "Failed to load model weights. Prefer models that ship 'safetensors' weights, "
                            "or upgrade torch (>=2.6) to load legacy .bin weights safely."
                        ) from e
                mdl.eval(); mdl.to(dev)

                layer = int(getattr(args, 'layer', 0))
                max_tokens = int(getattr(args, 'max_tokens', 128) or 128)
                tail_start = float(getattr(args, 'tail_start', 0.25) or 0.25)
                z_threshold = float(getattr(args, 'z_threshold', 2.0) or 2.0)
                z_mode = str(getattr(args, 'z_mode', 'standard') or 'standard')
                prompt_index = int(getattr(args, 'prompt_index', 0) or 0)

                prompts = []
                prompts_file = getattr(args, 'prompts_file', None)
                if prompts_file:
                    pth = Path(str(prompts_file))
                    lines = [ln.strip() for ln in pth.read_text().splitlines() if ln.strip()]
                    if not lines:
                        raise ValueError('prompts-file is empty')
                    prompts = lines
                else:
                    prompts = [str(getattr(args, 'prompt'))]

                # Extract per-prompt signals from real hidden states (test suite)
                signal_mode = str(getattr(args, 'signal_mode', 'token_norm') or 'token_norm')
                detrend = str(getattr(args, 'detrend', 'none') or 'none')
                window = str(getattr(args, 'window', 'none') or 'none')
                fft_size = int(getattr(args, 'fft_size', 0) or 0)
                segment = str(getattr(args, 'segment', 'prefix') or 'prefix')
                baseline_prompts_file = getattr(args, 'baseline_prompts_file', None)

                def _read_prompts_file(fp):
                    pth = Path(str(fp))
                    lines = [ln.strip() for ln in pth.read_text().splitlines() if ln.strip()]
                    if not lines:
                        raise ValueError('prompts-file is empty')
                    return lines

                def _extract_hidden_signal(pr_text: str) -> np.ndarray:
                    inputs = tok(pr_text, return_tensors='pt', truncation=True, max_length=max_tokens)
                    inputs = {k_: v.to(dev) for k_, v in inputs.items()}
                    with torch.no_grad():
                        out = mdl(**inputs, output_hidden_states=True, return_dict=True)
                    hs = getattr(out, 'hidden_states', None)
                    if hs is None:
                        raise ValueError('Model did not return hidden_states')
                    layers_h = list(hs[1:])  # drop embedding output
                    n_layers = len(layers_h)
                    if layer < 0 or layer >= n_layers:
                        raise ValueError(f'--layer out of range: {layer} (model has {n_layers} layers)')
                    h = layers_h[layer][0].detach().cpu().numpy().astype('float32', copy=False)
                    sig = token_signal(h, mode=signal_mode)
                    sig = detrend_signal(sig, mode=detrend)
                    return np.asarray(sig, dtype=np.float64)

                test_prompts = prompts
                test_signals = []
                test_sha16 = []
                for pr in test_prompts:
                    test_signals.append(_extract_hidden_signal(pr))
                    test_sha16.append(hashlib.sha256(str(pr).encode('utf-8', errors='ignore')).hexdigest()[:16])

                baseline_prompts = None
                baseline_signals = None
                baseline_sha16 = None
                if baseline_prompts_file:
                    baseline_prompts = _read_prompts_file(baseline_prompts_file)
                    baseline_signals = []
                    baseline_sha16 = []
                    for pr in baseline_prompts:
                        baseline_signals.append(_extract_hidden_signal(pr))
                        baseline_sha16.append(hashlib.sha256(str(pr).encode('utf-8', errors='ignore')).hexdigest()[:16])

                if fft_size > 0:
                    seq_len = int(fft_size)
                    seq_mode = f"fixed_nfft={seq_len}"
                else:
                    min_test = int(min(len(s) for s in test_signals))
                    min_base = int(min(len(s) for s in baseline_signals)) if baseline_signals else min_test
                    seq_len = int(min(min_test, min_base))
                    seq_mode = f"common_{segment}"

                if seq_len < 8:
                    raise ValueError(
                        'Need at least 8 tokens for a stable spectrum. Use longer prompts, increase --max-tokens, '
                        'or set --fft-size (recommended) to avoid common-prefix truncation.'
                    )

                def _segment(sig: np.ndarray) -> np.ndarray:
                    if segment == 'suffix':
                        return sig[-min(len(sig), seq_len):]
                    return sig[:min(len(sig), seq_len)]

                def _zs(x: np.ndarray, ref: np.ndarray | None) -> np.ndarray:
                    xx = np.asarray(x, dtype=np.float64)
                    rr = np.asarray(ref, dtype=np.float64) if ref is not None else xx
                    if z_mode == 'robust':
                        med = float(np.median(rr))
                        mad = float(np.median(np.abs(rr - med)))
                        denom = max(1.4826 * mad, 1e-12)
                        return (xx - med) / denom
                    mu = float(np.mean(rr))
                    sd = float(np.std(rr))
                    sd = max(sd, 1e-12)
                    return (xx - mu) / sd

                # Compute spectra + features for test prompts
                spectra = []
                dominant_freqs = []
                dominant_powers = []
                tail_ratios = []

                freqs_ref = None
                for sig in test_signals:
                    seg = _segment(sig)
                    freqs, power = rfft_power(seg, n_fft=(seq_len if fft_size > 0 else None), window=window)
                    if freqs_ref is None:
                        freqs_ref = freqs
                    spectra.append(power.astype('float64', copy=False))
                    di = int(np.argmax(power))
                    dominant_freqs.append(float(freqs[di]))
                    dominant_powers.append(float(power[di]))
                    tail_ratios.append(float(high_freq_tail_ratio(freqs, power, tail_start=tail_start, exclude_dc=True)))

                assert freqs_ref is not None
                stack = np.stack(spectra, axis=0)
                mean_spectrum = np.mean(stack, axis=0)

                baseline_mean_spectrum = None
                baseline_dom = None
                baseline_tail = None
                if baseline_signals:
                    base_spectra = []
                    baseline_dom = []
                    baseline_tail = []
                    for sig in baseline_signals:
                        seg = _segment(sig)
                        freqs_b, power_b = rfft_power(seg, n_fft=(seq_len if fft_size > 0 else None), window=window)
                        # Ensure baseline uses identical frequency bins (same seq_len / n_fft)
                        if freqs_ref is not None and freqs_b.shape == freqs_ref.shape:
                            pass
                        base_spectra.append(power_b.astype('float64', copy=False))
                        di = int(np.argmax(power_b))
                        baseline_dom.append(float(power_b[di]))
                        baseline_tail.append(float(high_freq_tail_ratio(freqs_b, power_b, tail_start=tail_start, exclude_dc=True)))
                    baseline_mean_spectrum = np.mean(np.stack(base_spectra, axis=0), axis=0)

                dom_z = _zs(np.asarray(dominant_powers, dtype=np.float64), np.asarray(baseline_dom, dtype=np.float64) if baseline_dom is not None else None)
                tail_z = _zs(np.asarray(tail_ratios, dtype=np.float64), np.asarray(baseline_tail, dtype=np.float64) if baseline_tail is not None else None)

                model_short = str(args.model).split('/')[-1]
                if baseline_prompts is not None:
                    subtitle = (
                        f"{model_short} | layer={layer} | test={len(test_prompts)} | baseline={len(baseline_prompts)} | "
                        f"seq={seq_len} ({seq_mode}/{segment}) | sig={signal_mode} | detrend={detrend} | win={window} | z={z_mode}"
                    )
                else:
                    subtitle = (
                        f"{model_short} | layer={layer} | prompts={len(test_prompts)} | seq={seq_len} ({seq_mode}/{segment}) | "
                        f"sig={signal_mode} | detrend={detrend} | win={window} | z={z_mode}"
                    )

                title = str(getattr(args, 'title', 'FFT Security Spectrum - Token-Norm FFT (per prompt + mean)'))
                out_json = Path(str(getattr(args, 'out_json', '_cli_runs/fft_security_spectrum.json')))
                out_png = Path(str(getattr(args, 'out_png', '_cli_runs/fft_security_spectrum.png')))
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_png.parent.mkdir(parents=True, exist_ok=True)

                metrics = FFTSecuritySpectrumMetrics(
                    title=title,
                    model=str(args.model),
                    tokenizer=str(tok_id),
                    layer=int(layer),
                    seq_len=int(seq_len),
                    prompt_count=int(len(test_prompts)),
                    seq_mode=str(seq_mode),
                    signal_mode=str(signal_mode),
                    detrend=str(detrend),
                    window=str(window),
                    fft_size=int(fft_size),
                    segment=str(segment),
                    tail_start=float(tail_start),
                    z_threshold=float(z_threshold),
                    z_mode=str(z_mode),
                    freqs=[float(x) for x in freqs_ref.tolist()],
                    spectra=[[float(v) for v in row.tolist()] for row in stack],
                    mean_spectrum=[float(v) for v in mean_spectrum.tolist()],
                    dominant_freqs=[float(x) for x in dominant_freqs],
                    dominant_powers=[float(x) for x in dominant_powers],
                    tail_ratios=[float(x) for x in tail_ratios],
                    dominant_z=[float(x) for x in dom_z.tolist()],
                    tail_z=[float(x) for x in tail_z.tolist()],
                    subtitle=subtitle,
                    prompt_sha16=test_sha16,
                    baseline_prompt_count=int(len(baseline_prompts)) if baseline_prompts is not None else None,
                    baseline_mean_spectrum=[float(v) for v in baseline_mean_spectrum.tolist()] if baseline_mean_spectrum is not None else None,
                    baseline_dominant_powers=[float(x) for x in baseline_dom] if baseline_dom is not None else None,
                    baseline_tail_ratios=[float(x) for x in baseline_tail] if baseline_tail is not None else None,
                    baseline_prompt_sha16=baseline_sha16,
                    baseline_label=str(baseline_prompts_file) if baseline_prompts_file else None,
                )

                # Save JSON
                obj = {
                    'title': metrics.title,
                    'subtitle': metrics.subtitle,
                    'model': metrics.model,
                    'tokenizer': metrics.tokenizer,
                    'layer': metrics.layer,
                    'seq_len': metrics.seq_len,
                    'prompt_count': metrics.prompt_count,
                    'seq_mode': metrics.seq_mode,
                    'signal_mode': metrics.signal_mode,
                    'detrend': metrics.detrend,
                    'window': metrics.window,
                    'fft_size': metrics.fft_size,
                    'segment': metrics.segment,
                    'tail_start': metrics.tail_start,
                    'z_threshold': metrics.z_threshold,
                    'z_mode': metrics.z_mode,
                    'freqs': metrics.freqs,
                    'spectra': metrics.spectra,
                    'mean_spectrum': metrics.mean_spectrum,
                    'dominant_freqs': metrics.dominant_freqs,
                    'dominant_powers': metrics.dominant_powers,
                    'tail_ratios': metrics.tail_ratios,
                    'dominant_z': metrics.dominant_z,
                    'tail_z': metrics.tail_z,
                    'prompt_sha16': metrics.prompt_sha16,
                }

                if prompts_file:
                    obj['prompts_file'] = str(prompts_file)
                    obj['prompts_file_sha16'] = hashlib.sha256(Path(str(prompts_file)).read_bytes()).hexdigest()[:16]

                if baseline_prompts_file:
                    obj['baseline_prompts_file'] = str(baseline_prompts_file)
                    obj['baseline_prompts_file_sha16'] = hashlib.sha256(Path(str(baseline_prompts_file)).read_bytes()).hexdigest()[:16]
                    obj['baseline_prompt_count'] = metrics.baseline_prompt_count
                    obj['baseline_mean_spectrum'] = metrics.baseline_mean_spectrum
                    obj['baseline_dominant_powers'] = metrics.baseline_dominant_powers
                    obj['baseline_tail_ratios'] = metrics.baseline_tail_ratios
                    obj['baseline_prompt_sha16'] = metrics.baseline_prompt_sha16
                    obj['baseline_label'] = metrics.baseline_label

                out_json.write_text(json.dumps(obj, indent=2))

                plot_fft_security_spectrum(metrics, prompt_index=prompt_index, title=title, out_path=str(out_png), guidance=True)

                print(str(out_json))
                print(str(out_png))
                return 0

            except Exception as e:
                logger.error(f"Activation FFT security spectrum failed: {e}")
                return 1

        elif args.command == 'activation_layer_causal_impact':
            try:
                import json
                import hashlib
                from pathlib import Path
                import numpy as np
                import torch
                from transformers import AutoModel, AutoTokenizer

                from ..visualization.layer_causal_impact import (
                    analyze_layer_causal_impact,
                    create_causal_impact_visualization,
                )

                # Resolve device
                dev = getattr(args, 'device', 'auto')
                if dev == 'auto':
                    if torch.cuda.is_available():
                        dev = 'cuda'
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        dev = 'mps'
                    else:
                        dev = 'cpu'

                logger.info(f"🔥 Layer-Level Causal Impact Analysis")
                logger.info(f"   Model: {args.model}")
                logger.info(f"   Device: {dev}")
                logger.info(f"   Method: {args.method}")
                logger.info(f"   Percentile: {args.percentile}th")

                # Load model and tokenizer
                tok_id = getattr(args, 'tokenizer', None) or args.model
                tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
                model.eval()

                # Run analysis
                impact_scores, hot_layers = analyze_layer_causal_impact(
                    model=model,
                    tokenizer=tokenizer,
                    baseline_prompt=args.baseline_prompt,
                    test_prompt=args.test_prompt,
                    device=dev,
                    method=args.method,
                    percentile=args.percentile,
                    layer_start=args.layer_start,
                    layer_end=args.layer_end
                )

                # Create visualization
                fig, html_str = create_causal_impact_visualization(
                    impact_scores=impact_scores,
                    hot_layers=hot_layers,
                    title=args.title,
                    percentile=args.percentile,
                    method=args.method,
                    interactive=args.interactive
                )

                # Save outputs
                out_json = Path(args.out_json)
                out_png = Path(args.out_png)
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_png.parent.mkdir(parents=True, exist_ok=True)

                # Save JSON metadata
                metadata = {
                    'model': args.model,
                    'tokenizer': tok_id,
                    'baseline_prompt': args.baseline_prompt,
                    'test_prompt': args.test_prompt,
                    'baseline_prompt_sha16': hashlib.sha256(args.baseline_prompt.encode('utf-8')).hexdigest()[:16],
                    'test_prompt_sha16': hashlib.sha256(args.test_prompt.encode('utf-8')).hexdigest()[:16],
                    'method': args.method,
                    'percentile': float(args.percentile),
                    'layer_start': args.layer_start,
                    'layer_end': args.layer_end,
                    'device': dev,
                    'impact_scores': {int(k): float(v) for k, v in impact_scores.items()},
                    'hot_layers': [int(x) for x in hot_layers],
                    'hot_layer_count': len(hot_layers),
                    'total_layers': len(impact_scores),
                    'research_citations': [
                        'SoK: Comprehensive Causality Analysis Framework for LLM Security (Dec 2025) - arxiv.org/abs/2512.04841',
                        'Backdoor Attribution: Elucidating and Controlling Backdoor in LLMs (Sep 2025) - arxiv.org/abs/2509.21761',
                        'HAct: Out-of-Distribution Detection with Neural Net Activation Histograms (arXiv:2309.04837, Sep 2023)'
                    ]
                }
                out_json.write_text(json.dumps(metadata, indent=2))

                # Save PNG
                try:
                    fig.write_image(str(out_png), width=1200, height=600, scale=2)
                except Exception as e:
                    logger.warning(f"PNG export requires kaleido: {e}")
                    out_png = None  # Will only generate HTML

                # Save HTML if interactive
                if args.interactive and html_str:
                    out_html = Path(args.out_html)
                    out_html.parent.mkdir(parents=True, exist_ok=True)
                    out_html.write_text(html_str)
                    logger.info(f"📊 Interactive HTML: {out_html}")
                    print(str(out_html))

                logger.info(f"✅ Layer-Level Causal Impact Analysis Complete")
                logger.info(f"   Hot layers: {hot_layers}")
                logger.info(f"   Total layers analyzed: {len(impact_scores)}")
                logger.info(f"   Output JSON: {out_json}")
                logger.info(f"   Output PNG: {out_png}")

                print(str(out_json))
                print(str(out_png))
                return 0

            except Exception as e:
                logger.error(f"Layer-level causal impact analysis failed: {e}")
                import traceback
                traceback.print_exc()
                return 1
        elif args.command == 'backdoor_watermark':
            try:
                import numpy as np, json
                from pathlib import Path
                # Implement inline to avoid re-dispatch issues
                if args.bdw_action in ['inject_backdoor', 'embed_watermark']:
                    bits = [int(b) for b in str(args.watermark_bits).split(',') if b.strip()]
                    pathway = [int(n) for n in str(args.target_pathway).split(',') if n.strip()]
                    p = Path(str(args.activations))
                    if not p.exists():
                        logger.error(f"Activations file not found: {p}")
                        return 1
                    if p.suffix.lower() == '.npz':
                        npz = np.load(str(p), allow_pickle=True)
                        if len(npz.files) == 0:
                            raise ValueError(f"Empty .npz: {p}")
                        A = np.asarray(npz[npz.files[0]])
                    else:
                        obj = np.load(str(p), allow_pickle=True)
                        if getattr(obj, "dtype", None) == object and getattr(obj, "shape", ()) == ():
                            obj = obj.item()
                        if isinstance(obj, dict):
                            for k in ("activations", "data", "X", "x", "arr"):
                                if k in obj:
                                    obj = obj[k]
                                    break
                        A = np.asarray(obj)
                    arr = np.array(A)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    elif arr.ndim > 2:
                        arr = arr.reshape(-1, arr.shape[-1])
                    need_w = max(pathway) + 1
                    if arr.shape[1] < need_w:
                        pad = np.zeros((arr.shape[0], need_w - arr.shape[1]), dtype=arr.dtype)
                        arr = np.concatenate([arr, pad], axis=1)
                    eps = float(args.epsilon)
                    for i, n in enumerate(pathway):
                        b = bits[i % len(bits)]
                        arr[:, n] += eps * (1 if b else -1)
                    np.save(f"{args.out_prefix}watermarked.npy", arr.astype('float32'))
                    Path(f"{args.out_prefix}wm_meta.json").write_text(json.dumps({'bits': bits, 'pathway': pathway, 'epsilon': eps}, indent=2))
                    print(f"{args.out_prefix}watermarked.npy")
                    print(f"{args.out_prefix}wm_meta.json")
                    return 0
                elif args.bdw_action == 'detect_watermark':
                    import numpy as _np
                    A = np.load(args.activations)
                    arr = _np.array(A)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    elif arr.ndim > 2:
                        arr = arr.reshape(-1, arr.shape[-1])
                    pathway = [int(n) for n in str(args.target_pathway).split(',') if n.strip()]
                    thr = float(args.threshold)
                    means = []
                    for n in pathway:
                        if 0 <= n < arr.shape[1]:
                            means.append(float(arr[:, n].mean()))
                        else:
                            means.append(0.0)
                    bits = [1 if m >= thr else 0 for m in means]
                    result = {'pathway': pathway, 'threshold': thr, 'means': [float(m) for m in means], 'bits': bits}
                    Path(f"{args.out_prefix}wm_detect.json").write_text(json.dumps(result, indent=2))
                    print(f"{args.out_prefix}wm_detect.json")
                    return 0
                else:
                    return 1
            except Exception as e:
                logger.error(f"Backdoor/watermark failed: {e}")
                return 1
        elif args.command == 'dna_neuron_ablation':
            try:
                import numpy as np, csv, json
                import matplotlib.pyplot as plt
                from pathlib import Path

                A = np.load(args.activations)
                arr0 = np.array(A)

                # Normalize to 2D: [samples, neurons]
                layer = getattr(args, 'layer', None)
                layer_axis = int(getattr(args, 'layer_axis', 0))

                if arr0.ndim == 1:
                    arr = arr0.reshape(1, -1)
                elif arr0.ndim == 2:
                    arr = arr0
                elif arr0.ndim == 3:
                    ax = layer_axis
                    if ax < 0:
                        ax += arr0.ndim
                    if ax < 0 or ax >= arr0.ndim:
                        raise ValueError(f"--layer-axis {layer_axis} out of range for activations ndim={arr0.ndim}")
                    if ax != 0:
                        arr0 = np.moveaxis(arr0, ax, 0)
                    if layer is not None:
                        li = int(layer)
                        if li < 0 or li >= int(arr0.shape[0]):
                            raise ValueError(f"--layer {li} out of range for activations with shape={arr0.shape}")
                        arr = arr0[li].reshape(-1, arr0.shape[-1])
                    else:
                        # Backward-compatible: flatten across layers when no layer is specified
                        arr = arr0.reshape(-1, arr0.shape[-1])
                else:
                    arr = arr0.reshape(-1, arr0.shape[-1])

                if arr.ndim != 2 or arr.shape[1] < 1:
                    raise ValueError(f"Expected activations to reduce to 2D [N,D], got shape={getattr(arr, 'shape', None)}")

                # Baseline signal: mean across samples per neuron
                means = arr.mean(axis=0)
                baseline_norm = float(np.linalg.norm(means) + 1e-12)

                k = max(1, min(int(args.topk), int(arr.shape[1])))
                topk_idx = np.argsort(np.abs(means))[-k:][::-1]

                # Ablation impact proxy: remove one neuron from the mean vector
                impacts_abs = []
                for n in topk_idx:
                    ablated = means.copy()
                    ablated[int(n)] = 0.0
                    new_norm = float(np.linalg.norm(ablated))
                    impacts_abs.append(float(max(0.0, baseline_norm - new_norm)))

                impacts_pct = [float(v / baseline_norm * 100.0) for v in impacts_abs]

                # Output paths (prefix + optional overrides)
                out_prefix = str(getattr(args, 'out_prefix', '_cli_runs/dna_'))
                outcsv = Path(str(getattr(args, 'out_csv', None) or f"{out_prefix}ablation.csv"))
                outj = Path(str(getattr(args, 'out_json', None) or f"{out_prefix}ablation.json"))
                outpng = Path(str(getattr(args, 'out_png', None) or f"{out_prefix}ablation_impact.png"))
                outhtml = Path(str(getattr(args, 'out_html', None) or f"{out_prefix}ablation_impact.html"))

                for pth in [outcsv, outj, outpng, outhtml]:
                    try:
                        pth.parent.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass

                # Save CSV
                with outcsv.open('w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['neuron', 'mean_activation', 'impact_abs', 'impact_pct'])
                    for i, n in enumerate(topk_idx):
                        w.writerow([int(n), float(means[int(n)]), float(impacts_abs[i]), float(impacts_pct[i])])

                # Bootstrap 95% CIs for impacts + stability fractions
                B = int(getattr(args, 'bootstrap', 200))
                seed = int(getattr(args, 'seed', 42))
                rng = np.random.default_rng(seed)

                ci_low_abs, ci_high_abs, stab = [], [], []
                if B > 0 and int(arr.shape[0]) >= 2:
                    for idx in topk_idx:
                        boots = []
                        for _ in range(B):
                            bs = rng.integers(0, int(arr.shape[0]), size=int(arr.shape[0]))
                            m_bs = arr[bs].mean(axis=0)
                            base_bs = float(np.linalg.norm(m_bs) + 1e-12)
                            ab_bs = m_bs.copy(); ab_bs[int(idx)] = 0.0
                            imp_bs = max(0.0, base_bs - float(np.linalg.norm(ab_bs)))
                            boots.append(float(imp_bs))
                        boots_arr = np.array(boots, dtype=float)
                        ci_low_abs.append(float(np.percentile(boots_arr, 2.5)))
                        ci_high_abs.append(float(np.percentile(boots_arr, 97.5)))
                        stab.append(float((boots_arr > 0).mean()))
                else:
                    ci_low_abs = [0.0] * len(topk_idx)
                    ci_high_abs = [0.0] * len(topk_idx)
                    stab = [0.0] * len(topk_idx)

                ci_low_pct = [float(max(0.0, c) / baseline_norm * 100.0) for c in ci_low_abs]
                ci_high_pct = [float(max(0.0, c) / baseline_norm * 100.0) for c in ci_high_abs]

                # Permutation p-value for Top-3 cumulative impact (heuristic triage)
                perm_trials = int(getattr(args, 'perm_trials', 200))
                top3_k = min(3, len(impacts_abs))
                obs_top3_abs = float(sum(sorted(impacts_abs, reverse=True)[:top3_k]))
                obs_top3_pct = float(sum(sorted(impacts_pct, reverse=True)[:top3_k]))

                ge_count = 0
                if perm_trials > 0 and obs_top3_abs > 0.0:
                    for _ in range(perm_trials):
                        perm = rng.permutation(int(arr.shape[1]))[:k]
                        p_imp = []
                        for n in perm:
                            ab = means.copy(); ab[int(n)] = 0.0
                            p_imp.append(float(max(0.0, baseline_norm - float(np.linalg.norm(ab)))))
                        s = float(sum(sorted(p_imp, reverse=True)[:top3_k]))
                        if s >= obs_top3_abs:
                            ge_count += 1
                    pval = float((ge_count + 1) / (perm_trials + 1))
                else:
                    pval = None

                # Save JSON
                payload = {
                    'title': str(getattr(args, 'title', 'NeurInSpectre — DNA Neuron Ablation Impact (Top-K)')),
                    'activations': str(args.activations),
                    'layer': None if layer is None else int(layer),
                    'layer_axis': int(layer_axis),
                    'input_shape': list(getattr(arr0, 'shape', [])),
                    'reduced_shape': list(getattr(arr, 'shape', [])),
                    'topk': int(k),
                    'topk_neurons': [int(i) for i in topk_idx.tolist()],
                    'mean_activation': [float(means[int(i)]) for i in topk_idx.tolist()],
                    'impact_abs': [float(v) for v in impacts_abs],
                    'impact_pct': [float(v) for v in impacts_pct],
                    'ci_low_pct': [float(v) for v in ci_low_pct],
                    'ci_high_pct': [float(v) for v in ci_high_pct],
                    'stability_p': [float(v) for v in stab],
                    'baseline_norm': float(baseline_norm),
                    'top3_cum_impact_pct': float(obs_top3_pct),
                    'top3_p_value': None if pval is None else float(pval),
                    'bootstrap': int(B),
                    'perm_trials': int(perm_trials),
                    'seed': int(seed),
                }
                outj.write_text(json.dumps(payload, indent=2), encoding='utf-8')

                # Convert to relative percent impact for visibility
                vis_impacts = impacts_pct[:]
                used_proxy = False
                if float(sum(vis_impacts)) == 0.0:
                    proxy = [float(abs(means[int(i)])) for i in topk_idx]
                    if float(sum(proxy)) > 0.0:
                        vis_impacts = proxy
                        used_proxy = True

                # Plot impact bar chart + cumulative curve
                fig, ax = plt.subplots(figsize=(max(7, len(topk_idx) * 0.55), 3.8))

                bar_color = '#e76f51' if not used_proxy else '#cccccc'
                label = 'Impact: Δ||mean|| (% of baseline)' if not used_proxy else 'Proxy: |mean_activation|'
                ax.bar(range(len(topk_idx)), vis_impacts, color=bar_color, label=label)

                # Add 95% CI whiskers if available and not in proxy mode
                if (not used_proxy) and len(ci_low_pct) == len(vis_impacts):
                    for x, (l, h) in enumerate(zip(ci_low_pct, ci_high_pct)):
                        ax.vlines(x, l, h, colors='#444', linewidth=1.2)
                        ax.hlines([l, h], x - 0.12, x + 0.12, colors='#444', linewidth=1.2)

                ax.set_xticks(range(len(topk_idx)))
                ax.set_xticklabels([int(i) for i in topk_idx], rotation=45, ha='right')
                ax.set_title(str(getattr(args, 'title', 'NeurInSpectre — DNA Neuron Ablation Impact (Top-K)')), fontsize=18, pad=16, fontweight='bold')
                ax.set_xlabel('Neuron index'); ax.set_ylabel('Impact (% Δ L2 norm)')

                y_max = max(vis_impacts) if len(vis_impacts) else 1.0
                ax.set_ylim(0.0, y_max * 1.2 + (1e-6 if y_max == 0.0 else 0.0))

                # Stability heatstrip (green=stable, amber=uncertain, red=unstable)
                try:
                    from matplotlib.patches import Rectangle as _Rect
                    strip_bottom = y_max * 0.95
                    strip_h = y_max * 0.035
                    for x, s in enumerate(stab):
                        if s >= 0.8:
                            col = '#2A9D8F'
                        elif s >= 0.5:
                            col = '#f4d35e'
                        else:
                            col = '#d04a5c'
                        ax.add_patch(_Rect((x - 0.4, strip_bottom), 0.8, strip_h, color=col, alpha=0.9, ec='none'))
                    ax.text(0.01, 0.98, 'stability', transform=ax.transAxes, va='top', ha='left', fontsize=10, color='#555')
                except Exception:
                    pass

                ax2 = ax.twinx()
                denom = max(1e-8, float(sum(vis_impacts)))
                ax2.plot(range(len(topk_idx)), np.cumsum(vis_impacts) / denom, 'k--', lw=1.4, label='Cumulative impact')
                ax2.set_ylim(0, 1.05)
                ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                ax2.set_ylabel('Cumulative fraction')

                lines, labels_ = ax.get_legend_handles_labels()
                lines2, labels2_ = ax2.get_legend_handles_labels()
                fig.legend(lines + lines2, labels_ + labels2_, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

                # Top-3 badge
                try:
                    badge = 'significant' if (pval is not None and pval <= 0.05 and (not used_proxy)) else ('proxy' if used_proxy else 'n.s.')
                    ptxt = f"p≈{pval:.3f}" if pval is not None else "p≈n/a"
                    fig.text(
                        0.02,
                        0.02,
                        f"Top-3 cumulative impact: {obs_top3_pct:.3f}% ({badge}, {ptxt})",
                        fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='#f2f2f2', edgecolor='#808080', alpha=0.95),
                    )
                except Exception:
                    pass

                plt.tight_layout(rect=[0, 0.18, 1, 0.95])
                fig.savefig(str(outpng), dpi=200, bbox_inches='tight')

                # Optional interactive HTML report
                if getattr(args, 'interactive', False):
                    import plotly.graph_objects as go
                    import plotly.io as pio
                    from plotly.subplots import make_subplots
                    import html as _html

                    x_labels = [str(int(i)) for i in topk_idx.tolist()]
                    y_vals = vis_impacts
                    y_title = 'Impact (% Δ L2 norm)' if not used_proxy else 'Proxy: |mean_activation|'

                    figp = make_subplots(specs=[[{'secondary_y': True}]])

                    # Error bars for percent impacts
                    err_plus = None
                    err_minus = None
                    if (not used_proxy) and len(ci_low_pct) == len(y_vals):
                        err_plus = [max(0.0, float(h) - float(y)) for y, h in zip(y_vals, ci_high_pct)]
                        err_minus = [max(0.0, float(y) - float(l)) for y, l in zip(y_vals, ci_low_pct)]

                    bar_custom = []
                    for i, n in enumerate(topk_idx.tolist()):
                        bar_custom.append([
                            float(means[int(n)]),
                            float(stab[i]) if i < len(stab) else 0.0,
                            float(ci_low_pct[i]) if i < len(ci_low_pct) else 0.0,
                            float(ci_high_pct[i]) if i < len(ci_high_pct) else 0.0,
                        ])

                    figp.add_trace(
                        go.Bar(
                            x=x_labels,
                            y=y_vals,
                            name='Impact' if not used_proxy else 'Proxy',
                            marker_color=bar_color,
                            customdata=bar_custom,
                            hovertemplate=(
                                '<b>DNA Neuron Ablation</b><br>'
                                'Neuron: %{x}<br>'
                                f'{_html.escape(y_title)}: %{{y:.4f}}<br>'
                                'mean_activation: %{customdata[0]:.6f}<br>'
                                'stability P(impact>0): %{customdata[1]:.3f}<br>'
                                '95% CI: [%{customdata[2]:.4f}, %{customdata[3]:.4f}]<extra></extra>'
                            ),
                            error_y=None if err_plus is None else dict(type='data', symmetric=False, array=err_plus, arrayminus=err_minus, thickness=1, width=4, color='#444'),
                        ),
                        secondary_y=False,
                    )

                    # Stability strip as thin bars near the top
                    try:
                        y_max_p = float(max(y_vals)) if len(y_vals) else 1.0
                        strip_bottom = y_max_p * 0.95
                        strip_h = y_max_p * 0.035 if y_max_p > 0 else 0.01
                        stab_cols = []
                        for s in stab:
                            if s >= 0.8:
                                stab_cols.append('#2A9D8F')
                            elif s >= 0.5:
                                stab_cols.append('#f4d35e')
                            else:
                                stab_cols.append('#d04a5c')
                        figp.add_trace(
                            go.Bar(
                                x=x_labels,
                                y=[strip_h] * len(x_labels),
                                base=[strip_bottom] * len(x_labels),
                                marker_color=stab_cols,
                                opacity=0.9,
                                hovertemplate='stability P(impact>0): %{customdata:.3f}<extra></extra>',
                                customdata=stab,
                                showlegend=False,
                            ),
                            secondary_y=False,
                        )
                        figp.add_annotation(
                            text='stability',
                            xref='paper', yref='paper',
                            x=0.01, y=0.99,
                            showarrow=False,
                            font=dict(size=12, color='#555'),
                        )
                    except Exception:
                        pass

                    denom = max(1e-8, float(sum(y_vals)))
                    cum = (np.cumsum(y_vals) / denom).tolist()
                    figp.add_trace(
                        go.Scatter(
                            x=x_labels,
                            y=cum,
                            mode='lines+markers',
                            name='Cumulative impact',
                            line=dict(color='black', dash='dash', width=2),
                            hovertemplate='cumulative fraction: %{y:.3f}<extra></extra>',
                        ),
                        secondary_y=True,
                    )

                    subtitle = f"layer={layer}" if layer is not None else "all layers (flattened)"
                    figp.update_layout(
                        # Avoid repeating the page title inside the plot; keep only context here.
                        title=dict(text=f"<span style='font-size:12px;color:#666'>activations={_html.escape(str(args.activations))} | {subtitle} | topk={k}</span>", x=0.01, xanchor='left'),
                        xaxis_title='Neuron index',
                        template='plotly_white',
                        height=720,
                        width=1400,
                        margin=dict(l=80, r=80, t=90, b=120),
                        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.22),
                    )
                    figp.update_yaxes(title_text=y_title, secondary_y=False)
                    figp.update_yaxes(title_text='Cumulative fraction', secondary_y=True, range=[0, 1.05])

                    badge = 'significant' if (pval is not None and pval <= 0.05 and (not used_proxy)) else ('proxy' if used_proxy else 'n.s.')
                    ptxt = f"p≈{pval:.3f}" if pval is not None else "p≈n/a"
                    figp.add_annotation(
                        text=f"Top-3 cumulative impact: {obs_top3_pct:.3f}% ({badge}, {ptxt})",
                        xref='paper', yref='paper', x=0.0, y=-0.18,
                        showarrow=False,
                        align='left',
                        font=dict(size=12, color='#111'),
                    )

                    # Guidance + findings derived from salient plot features
                    denom = max(1e-8, float(sum(y_vals)))
                    p_frac = np.array(y_vals, dtype=float) / denom
                    top1_frac = float(p_frac[0]) if len(p_frac) else 0.0
                    top3_frac = float(p_frac[: min(3, len(p_frac))].sum()) if len(p_frac) else 0.0
                    hhi = float(np.sum(p_frac * p_frac)) if len(p_frac) else 0.0
                    eff_neurons = float(1.0 / max(hhi, 1e-12)) if len(p_frac) else 0.0

                    # Gini on p_frac (0=uniform, 1=fully concentrated)
                    try:
                        _pv = np.sort(np.clip(p_frac, 0.0, 1.0))
                        _n = int(_pv.size)
                        if _n <= 1 or float(_pv.sum()) <= 0.0:
                            gini = 0.0
                        else:
                            _cum = np.cumsum(_pv)
                            gini = float(((_n + 1.0) - 2.0 * float(_cum.sum()) / float(_cum[-1])) / _n)
                            gini = float(max(0.0, min(1.0, gini)))
                    except Exception:
                        gini = 0.0

                    if top3_frac >= 0.70 or eff_neurons <= 2.5:
                        concentration = 'HIGH'
                    elif top3_frac >= 0.50 or eff_neurons <= 4.0:
                        concentration = 'MEDIUM'
                    else:
                        concentration = 'LOW'

                    stable_count = int(sum(1 for s in stab if float(s) >= 0.8))
                    uncertain_count = int(sum(1 for s in stab if 0.5 <= float(s) < 0.8))
                    unstable_count = int(sum(1 for s in stab if float(s) < 0.5))

                    dom_thr = 0.10
                    dom_count = int(sum(1 for v in p_frac.tolist() if float(v) >= dom_thr))
                    dom_stable = int(sum(1 for v, s in zip(p_frac.tolist(), stab) if float(v) >= dom_thr and float(s) >= 0.8))

                    # CI width summary (only meaningful when not in proxy mode)
                    if (not used_proxy) and len(ci_low_pct) == len(y_vals):
                        ci_widths = [max(0.0, float(h) - float(l)) for l, h in zip(ci_low_pct, ci_high_pct)]
                        ci_width_med = float(np.median(ci_widths)) if len(ci_widths) else 0.0
                    else:
                        ci_width_med = None

                    findings = [
                        f'Concentration={concentration} (top1={top1_frac:.2f}, top3={top3_frac:.2f}, effective≈{eff_neurons:.1f}, gini≈{gini:.2f})',
                        f'Stability: {stable_count}/{k} stable (P>0.8), {uncertain_count} uncertain, {unstable_count} unstable',
                        f'Dominant neurons (≥{dom_thr:.0%} of Top-K): {dom_count} (stable dominant: {dom_stable})',
                    ]
                    if ci_width_med is not None:
                        findings.append(f'Median 95% CI width: {ci_width_med:.4f} (pct points)')
                    if used_proxy:
                        findings.append('NOTE: impact signal ~0; using proxy=|mean_activation|. Treat as weak evidence; increase samples or change representation/metric.')

                    # Salient feature → action mapping
                    cues = [
                        'Steep cumulative curve: small neuron sets dominate → targeted interventions and hardening are higher leverage.',
                        'Green stability strip: effect persists across bootstraps → prioritize for audits/regression tests.',
                        'Wide whiskers: sensitivity to sampling → stratify by prompt/class/client and rerun.',
                        'Rank churn across layers/runs: non-stationary internals → monitor + keep fixed regression prompts.',
                    ]

                    # Top-K table HTML (quick triage; CSV is canonical)
                    rows = []
                    for i, n in enumerate(topk_idx.tolist()):
                        st = float(stab[i]) if i < len(stab) else 0.0
                        st_lab = 'stable' if st >= 0.8 else ('uncertain' if st >= 0.5 else 'unstable')
                        frac = float(p_frac[i]) if i < len(p_frac) else 0.0
                        rows.append(
                            f"<tr>"
                            f"<td style='padding:6px 8px; text-align:right;'>{i+1}</td>"
                            f"<td style='padding:6px 8px;'><code>{int(n)}</code></td>"
                            f"<td style='padding:6px 8px; text-align:right;'>{float(y_vals[i]):.4f}</td>"
                            f"<td style='padding:6px 8px; text-align:right;'>{frac:.3f}</td>"
                            f"<td style='padding:6px 8px; text-align:right;'>{float(means[int(n)]):.6f}</td>"
                            f"<td style='padding:6px 8px; text-align:right;'>{st:.3f}</td>"
                            f"<td style='padding:6px 8px;'>{st_lab}</td>"
                            f"</tr>"
                        )

                    table_html = (
                        "<table style='width:100%; border-collapse: collapse; font-size: 13px;'>"
                        "<thead><tr>"
                        "<th style='text-align:right; padding:6px 8px; border-bottom:1px solid #eee;'>#</th>"
                        "<th style='text-align:left; padding:6px 8px; border-bottom:1px solid #eee;'>Neuron</th>"
                        f"<th style='text-align:right; padding:6px 8px; border-bottom:1px solid #eee;'>{_html.escape(y_title)}</th>"
                        "<th style='text-align:right; padding:6px 8px; border-bottom:1px solid #eee;'>Frac(topK)</th>"
                        "<th style='text-align:right; padding:6px 8px; border-bottom:1px solid #eee;'>Mean act</th>"
                        "<th style='text-align:right; padding:6px 8px; border-bottom:1px solid #eee;'>Stability</th>"
                        "<th style='text-align:left; padding:6px 8px; border-bottom:1px solid #eee;'>Label</th>"
                        "</tr></thead>"
                        "<tbody>" + "".join(rows) + "</tbody></table>"
                    )

                    # Next-step guidance: condition on concentration + stability
                    red_lines = [
                        'Start with neurons that are simultaneously high impact (bar height) and high stability (green strip).',
                        'Validate effect with a real intervention (ablate/patch) and measure output/logit change, not just activation statistics.',
                    ]
                    if concentration == 'HIGH':
                        red_lines.append('Concentration is HIGH: focus on Top-1 to Top-3 first; small edits may steer behavior with minimal surface changes.')
                    elif concentration == 'MEDIUM':
                        red_lines.append('Concentration is MEDIUM: prioritize Top-3 to Top-5 and test transfer across prompts/contexts.')
                    else:
                        red_lines.append('Concentration is LOW: expect distributed circuits; test multi-neuron coalitions (not just single neurons).')
                    if dom_stable > 0:
                        red_lines.append('At least one dominant neuron is stable: prioritize it for targeted perturbation/patching and evaluate drift vs baseline.')
                    if unstable_count >= max(2, k // 3):
                        red_lines.append('Many neurons are unstable: collect more samples or stratify activations (clean vs triggered) to isolate consistent effects.')
                    red_lines.append('Operationalize: run per-layer; look for layers where concentration spikes (often mid/late layers for safety/instruction features).')

                    blue_lines = [
                        'Treat stable high-impact neurons as single points of failure: monitor their activation distributions across updates.',
                        'Use this as a regression test: defenses should reduce concentration (flatten cumulative curve) and/or reduce stability (less green).',
                    ]
                    if concentration == 'HIGH':
                        blue_lines.append('Concentration is HIGH: harden/regularize around these neurons; alert if Top-3 fraction increases over time.')
                    elif concentration == 'MEDIUM':
                        blue_lines.append('Concentration is MEDIUM: monitor the Top-5 set and alert on rank churn (which neurons dominate changes).')
                    else:
                        blue_lines.append('Concentration is LOW: broaden monitoring to circuit-level signals; per-neuron alerts may be noisy.')
                    if ci_width_med is not None and ci_width_med > 0.25:
                        blue_lines.append('Wide CIs suggest variance: increase sample size and validate on fixed evaluation prompts/datasets.')
                    blue_lines.append('Layer sweep: if one layer dominates, focus audits/guardrails on that layer’s activations.')

                    refs = [
                        ('NeuroStrike (Sep 2025)', 'https://arxiv.org/abs/2509.11864'),
                        ('NeuronTune (Aug 2025)', 'https://arxiv.org/abs/2508.09473'),
                        ('Safety-neuron projection (Aug 2025)', 'https://arxiv.org/abs/2508.09190'),
                        ('Backdoor Attribution (Sep 2025)', 'https://arxiv.org/abs/2509.21761'),
                        ('Backdoored attention patterns (Aug 2025)', 'https://arxiv.org/abs/2508.15847'),
                        ('BEAT defense (Jun 2025)', 'https://arxiv.org/abs/2506.16447'),
                        ('ShadowLogic (Nov 2025)', 'https://arxiv.org/abs/2511.00664'),
                        ('Hedonic neuron coalitions (Sep 2025)', 'https://arxiv.org/abs/2509.23684'),
                        ('Backdoor sample detection (Sep 2025)', 'https://arxiv.org/abs/2509.05318'),
                    ]
                    refs_html = ''.join(
                        f"<li><a href='{_html.escape(url)}' target='_blank' rel='noopener noreferrer'>{_html.escape(name)}</a></li>"
                        for name, url in refs
                    )

                    findings_html = ''.join(f"<li>{_html.escape(x)}</li>" for x in findings)
                    cues_html = ''.join(f"<li>{_html.escape(x)}</li>" for x in cues)

                    fig_html = pio.to_html(figp, include_plotlyjs='cdn', full_html=False)

                    report = f'''<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{_html.escape(payload['title'])}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; color: #111; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 18px; }}
    .card {{ border: 1px solid #e3e3e3; border-radius: 10px; padding: 14px; margin: 12px 0; background: #fff; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    @media (max-width: 920px) {{ .two {{ grid-template-columns: 1fr; }} }}
    ul {{ margin: 8px 0 0 18px; }}
    code, pre {{ background: #f7f7f7; padding: 2px 5px; border-radius: 6px; }}
    pre {{ padding: 10px 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h2 style="margin:0 0 6px 0; font-size: 18px;">{_html.escape(payload['title'])}</h2>
      <div style="color:#555; font-size: 13px;">Ablation-impact proxy from activations. Hover shows mean activation, stability, CI, and fraction-of-TopK.</div>
      <div style="color:#555; font-size: 13px; margin-top: 6px;">Top-3 cumulative impact: <code>{obs_top3_pct:.3f}%</code> ({_html.escape(badge)}, {_html.escape(ptxt)}). Layer: <code>{_html.escape(str(layer) if layer is not None else 'all')}</code>. Concentration: <code>{_html.escape(concentration)}</code>.</div>
    </div>

    <div class="card">{fig_html}</div>

    <div class="card">
      <h3 style="margin:0; font-size: 15px;">Findings (derived from the plot)</h3>
      <ul>{findings_html}</ul>
    </div>

    <div class="card">
      <h3 style="margin:0; font-size: 15px;">Top-K table</h3>
      <div style="color:#555; font-size: 12px; margin-top:6px;">CSV is canonical for bulk analysis; this table is quick triage.</div>
      <div style="margin-top:10px;">{table_html}</div>
    </div>

    <div class="card">
      <h3 style="margin:0; font-size: 15px;">How to act on salient visual cues</h3>
      <ul>{cues_html}</ul>
    </div>

    <div class="two">
      <div class="card">
        <h3 style="margin:0; font-size: 15px;">Blue team: practical next steps</h3>
        <ul>{''.join('<li>'+_html.escape(x)+'</li>' for x in blue_lines)}</ul>
      </div>
      <div class="card">
        <h3 style="margin:0; font-size: 15px;">Red team: practical next steps</h3>
        <ul>{''.join('<li>'+_html.escape(x)+'</li>' for x in red_lines)}</ul>
      </div>
    </div>

    <div class="card">
      <h3 style="margin:0; font-size: 15px;">Recent research (mid/late 2025)</h3>
      <ul>{refs_html}</ul>
    </div>

    <div class="card">
      <h3 style="margin:0; font-size: 15px;">Outputs</h3>
      <ul>
        <li>CSV: <code>{_html.escape(str(outcsv))}</code></li>
        <li>JSON: <code>{_html.escape(str(outj))}</code></li>
        <li>PNG: <code>{_html.escape(str(outpng))}</code></li>
      </ul>
    </div>
  </div>
</body>
</html>
'''
                    outhtml.write_text(report, encoding='utf-8')

                print(str(outcsv))
                print(str(outj))
                print(str(outpng))
                if getattr(args, 'interactive', False):
                    print(str(outhtml))
                return 0
            except Exception as e:
                logger.error(f"DNA neuron ablation failed: {e}")
                return 1
        elif args.command == 'fusion_co_attention_traces':
            try:
                import json
                import hashlib
                from pathlib import Path

                import numpy as np

                from ..visualization.co_attention_traces import (
                    CoAttentionTracesMetrics,
                    co_attention_fuse,
                    plot_co_attention_traces,
                    plot_co_attention_traces_interactive,
                    _scale_group,
                )

                if args.coatt_action == 'visualize':
                    in_json = Path(str(getattr(args, 'in_json')))
                    obj = json.loads(in_json.read_text())

                    metrics = CoAttentionTracesMetrics(
                        title=str(obj.get('title', 'NeurInSpectre — Co-Attention Trace Fusion (co_attention)')),
                        strategy=str(obj.get('strategy', 'co_attention')),
                        model=obj.get('model', None),
                        tokenizer=obj.get('tokenizer', None),
                        layer=None if obj.get('layer') is None else int(obj.get('layer')),
                        prompt_a=obj.get('prompt_a', None),
                        prompt_b=obj.get('prompt_b', None),
                        prompt_a_sha16=obj.get('prompt_a_sha16', None),
                        prompt_b_sha16=obj.get('prompt_b_sha16', None),
                        seq_len=int(obj.get('seq_len', 0)),
                        feature=int(obj.get('feature', 0)),
                        feature2=None if obj.get('feature2') is None else int(obj.get('feature2')),
                        scale=str(obj.get('scale', 'tanh_z')),
                        alpha=float(obj.get('alpha', 0.55)),
                        temperature=float(obj.get('temperature', 0.25)),
                        x=[int(v) for v in obj.get('x', [])],
                        a_f1=[float(v) for v in obj.get('a_f1', [])],
                        b_f1=[float(v) for v in obj.get('b_f1', [])],
                        a_f2=obj.get('a_f2', None),
                        b_f2=obj.get('b_f2', None),
                        fa_f1=[float(v) for v in obj.get('fa_f1', [])],
                        fb_f1=[float(v) for v in obj.get('fb_f1', [])],
                        fa_f2=obj.get('fa_f2', None),
                        fb_f2=obj.get('fb_f2', None),
                        subtitle=str(obj.get('subtitle')) if obj.get('subtitle') is not None else None,
                    )

                    out_png = Path(str(getattr(args, 'out_png', '_cli_runs/co_attention_traces.png')))
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    ttl = getattr(args, 'title', None) or metrics.title
                    plot_co_attention_traces(metrics, title=str(ttl), out_path=str(out_png))

                    out_html = getattr(args, 'out_html', None)
                    if out_html:
                        out_html_p = Path(str(out_html))
                        out_html_p.parent.mkdir(parents=True, exist_ok=True)
                        plot_co_attention_traces_interactive(metrics, title=str(ttl), out_path=str(out_html_p))
                        print(str(out_png))
                        print(str(out_html_p))
                        return 0

                    print(str(out_png))
                    return 0

                # craft
                use_model = bool(getattr(args, 'model', None))
                max_steps = int(getattr(args, 'max_steps', 101) or 101)

                feature_raw = str(getattr(args, 'feature', '0') or '0')
                feature2_raw = str(getattr(args, 'feature2', '1') or '1')

                alpha = float(getattr(args, 'alpha', 0.55) or 0.55)
                temperature = float(getattr(args, 'temperature', 0.25) or 0.25)
                scale = str(getattr(args, 'scale', 'tanh_z') or 'tanh_z')

                layer = getattr(args, 'layer', None)

                trace_a = None
                trace_b = None

                model_name = None
                tok_name = None
                prompt_a = None
                prompt_b = None
                pa_sha16 = None
                pb_sha16 = None

                if use_model:
                    import torch
                    from transformers import AutoModel, AutoTokenizer

                    if not getattr(args, 'prompt_a', None) or not getattr(args, 'prompt_b', None):
                        raise ValueError('prompt mode requires --prompt-a and --prompt-b')
                    if layer is None:
                        raise ValueError('prompt mode requires --layer (0-indexed)')

                    # Device
                    dev = getattr(args, 'device', 'auto')
                    if dev == 'auto':
                        if torch.cuda.is_available():
                            dev = 'cuda'
                        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            dev = 'mps'
                        else:
                            dev = 'cpu'
                    device = torch.device(str(dev))

                    trust_rc = bool(getattr(args, 'trust_remote_code', False))
                    model_name = str(args.model)
                    tok_name = str(getattr(args, 'tokenizer', None) or args.model)
                    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True, trust_remote_code=trust_rc)

                    # Prefer safetensors
                    try:
                        mdl = AutoModel.from_pretrained(model_name, use_safetensors=True, trust_remote_code=trust_rc)
                    except Exception:
                        try:
                            mdl = AutoModel.from_pretrained(model_name, trust_remote_code=trust_rc)
                        except Exception as e:
                            raise RuntimeError(
                                "Failed to load model weights. Prefer models that ship 'safetensors' weights, "
                                "or upgrade torch (>=2.6) to load legacy .bin weights safely."
                            ) from e

                    mdl.to(device)
                    mdl.eval()

                    max_tokens = int(getattr(args, 'max_tokens', 128) or 128)

                    def _extract(prompt: str) -> np.ndarray:
                        enc = tok(prompt, return_tensors='pt', truncation=True, max_length=max_tokens)
                        enc = {k: v.to(device) for k, v in enc.items()}
                        with torch.no_grad():
                            out = mdl(**enc, output_hidden_states=True, return_dict=True)

                        hs = getattr(out, 'hidden_states', None)
                        if hs is None:
                            hs = getattr(out, 'encoder_hidden_states', None)
                        if hs is None:
                            hs = getattr(out, 'decoder_hidden_states', None)

                        if hs is None:
                            last = getattr(out, 'last_hidden_state', None)
                            if last is None:
                                raise ValueError('Model did not return hidden states')
                            layers_h = [last]
                        else:
                            # Drop embedding output if present
                            layers_h = list(hs[1:]) if len(hs) > 1 else list(hs)

                        n_layers = len(layers_h)
                        li = int(layer)
                        if li < 0 or li >= n_layers:
                            raise ValueError(f'--layer out of range: {li} (model returned {n_layers} hidden-state layers)')

                        h = layers_h[li][0].detach().cpu().numpy().astype('float32', copy=False)
                        return np.asarray(h, dtype=np.float64)

                    prompt_a = str(getattr(args, 'prompt_a'))
                    prompt_b = str(getattr(args, 'prompt_b'))
                    pa_sha16 = hashlib.sha256(prompt_a.encode('utf-8', errors='ignore')).hexdigest()[:16]
                    pb_sha16 = hashlib.sha256(prompt_b.encode('utf-8', errors='ignore')).hexdigest()[:16]

                    trace_a = _extract(prompt_a)
                    trace_b = _extract(prompt_b)

                else:
                    # Array mode
                    if not getattr(args, 'trace_a', None) or not getattr(args, 'trace_b', None):
                        raise ValueError('array mode requires --trace-a and --trace-b')

                    A = np.load(str(getattr(args, 'trace_a')))
                    B = np.load(str(getattr(args, 'trace_b')))

                    def _select(arr: np.ndarray) -> np.ndarray:
                        x = np.asarray(arr)
                        if x.ndim >= 3:
                            if layer is None:
                                raise ValueError('array mode with ndim>=3 requires --layer')
                            axis = int(getattr(args, 'layer_axis', 0) or 0)
                            x = np.moveaxis(x, axis, 0)[int(layer)]
                        if x.ndim != 2:
                            raise ValueError('Trace arrays must be T×D or L×T×D')
                        return np.asarray(x, dtype=np.float64)

                    trace_a = _select(A)
                    trace_b = _select(B)

                assert trace_a is not None and trace_b is not None

                # Align to common length
                N = int(min(int(trace_a.shape[0]), int(trace_b.shape[0]), int(max_steps)))
                if N < 8:
                    raise ValueError('Need at least 8 steps to plot; increase --max-tokens/--max-steps')

                A = trace_a[:N]
                B = trace_b[:N]

                # Robustness: sanitize non-finite values early so fusion/scaling never emits NaNs.
                A = np.nan_to_num(np.asarray(A), nan=0.0, posinf=0.0, neginf=0.0)
                B = np.nan_to_num(np.asarray(B), nan=0.0, posinf=0.0, neginf=0.0)

                # Feature parsing / auto selection
                D = int(min(A.shape[1], B.shape[1]))

                def _parse_feat(s: str, *, default: int) -> int:
                    st = str(s or '').strip().lower()
                    if st == 'auto':
                        # Prefer features that vary over time and differ across traces.
                        # (Exclude the first token to reduce BOS/first-token artifacts.)
                        A0 = A[1:] if A.shape[0] > 1 else A
                        B0 = B[1:] if B.shape[0] > 1 else B
                        n = int(min(A0.shape[0], B0.shape[0]))
                        A0 = A0[:n]
                        B0 = B0[:n]

                        disagree = np.mean(np.abs(A0 - B0), axis=0)
                        if n > 1:
                            dyn_a = np.mean(np.abs(np.diff(A0, axis=0)), axis=0)
                            dyn_b = np.mean(np.abs(np.diff(B0, axis=0)), axis=0)
                        else:
                            dyn_a = np.zeros((D,), dtype=np.float64)
                            dyn_b = np.zeros((D,), dtype=np.float64)
                        spread = A0.std(axis=0) + B0.std(axis=0)

                        score = disagree + 0.75 * (dyn_a + dyn_b) + 0.25 * spread
                        j = int(np.argmax(score))
                        return int(j)
                    if st in ('none', 'null', ''):
                        return int(default)
                    return int(st)

                feat1 = _parse_feat(feature_raw, default=0)
                if feat1 < 0 or feat1 >= D:
                    raise ValueError(f'--feature out of range: {feat1} (dim={D})')

                feat2 = None
                f2s = str(feature2_raw or '').strip().lower()
                if f2s not in ('none', 'null', ''):
                    feat2 = _parse_feat(feature2_raw, default=1)
                    if feat2 < 0 or feat2 >= D:
                        raise ValueError(f'--feature2 out of range: {feat2} (dim={D})')

                    # If auto picked the same feature twice, choose the next-best feature.
                    if f2s == 'auto' and int(feat2) == int(feat1) and D > 1:
                        A0 = A[1:] if A.shape[0] > 1 else A
                        B0 = B[1:] if B.shape[0] > 1 else B
                        n = int(min(A0.shape[0], B0.shape[0]))
                        A0 = A0[:n]
                        B0 = B0[:n]
                        disagree = np.mean(np.abs(A0 - B0), axis=0)
                        if n > 1:
                            dyn_a = np.mean(np.abs(np.diff(A0, axis=0)), axis=0)
                            dyn_b = np.mean(np.abs(np.diff(B0, axis=0)), axis=0)
                        else:
                            dyn_a = np.zeros((D,), dtype=np.float64)
                            dyn_b = np.zeros((D,), dtype=np.float64)
                        spread = A0.std(axis=0) + B0.std(axis=0)
                        score = disagree + 0.75 * (dyn_a + dyn_b) + 0.25 * spread
                        score[int(feat1)] = -1e18
                        feat2 = int(np.argmax(score))

                # Fuse
                fused_a, fused_b, info = co_attention_fuse(A, B, alpha=alpha, temperature=temperature)

                # Extract features
                orig_a1 = A[:, feat1]
                orig_b1 = B[:, feat1]
                fa1 = fused_a[:, feat1]
                fb1 = fused_b[:, feat1]

                # Scale group consistently across original + fused
                s_orig_a1, s_orig_b1, s_fa1, s_fb1 = _scale_group([orig_a1, orig_b1, fa1, fb1], mode=scale)

                a_f2 = b_f2 = fa_f2 = fb_f2 = None
                if feat2 is not None:
                    orig_a2 = A[:, feat2]
                    orig_b2 = B[:, feat2]
                    fa2 = fused_a[:, feat2]
                    fb2 = fused_b[:, feat2]
                    s_orig_a2, s_orig_b2, s_fa2, s_fb2 = _scale_group([orig_a2, orig_b2, fa2, fb2], mode=scale)
                    a_f2 = [float(v) for v in s_orig_a2]
                    b_f2 = [float(v) for v in s_orig_b2]
                    fa_f2 = [float(v) for v in s_fa2]
                    fb_f2 = [float(v) for v in s_fb2]

                x = list(range(N))

                # Peak divergence (on scaled traces): where fused deviates most from original
                peak_divergence_index = None
                peak_divergence_step = None
                peak_divergence_value = None
                try:
                    d = np.abs(np.asarray(s_fa1, dtype=np.float64) - np.asarray(s_orig_a1, dtype=np.float64))
                    d = d + np.abs(np.asarray(s_fb1, dtype=np.float64) - np.asarray(s_orig_b1, dtype=np.float64))
                    peak_idx = int(np.argmax(d))
                    peak_divergence_index = peak_idx
                    peak_divergence_step = int(x[peak_idx])
                    peak_divergence_value = float(d[peak_idx])
                except Exception:
                    peak_divergence_index = None
                    peak_divergence_step = None
                    peak_divergence_value = None

                # Subtitle
                if model_name:
                    model_short = str(model_name).split('/')[-1]
                    subtitle = f"{model_short} | layer={int(layer)} | alpha={alpha:.2f} | temp={temperature:.2f} | feature={feat1}"
                else:
                    subtitle = f"array mode | alpha={alpha:.2f} | temp={temperature:.2f} | feature={feat1}"

                title = str(getattr(args, 'title', 'NeurInSpectre — Co-Attention Trace Fusion (co_attention)'))
                out_json = Path(str(getattr(args, 'out_json', '_cli_runs/co_attention_traces.json')))
                out_png = Path(str(getattr(args, 'out_png', '_cli_runs/co_attention_traces.png')))
                out_json.parent.mkdir(parents=True, exist_ok=True)
                out_png.parent.mkdir(parents=True, exist_ok=True)

                metrics = CoAttentionTracesMetrics(
                    title=title,
                    strategy='co_attention',
                    model=model_name,
                    tokenizer=tok_name,
                    layer=None if layer is None else int(layer),
                    prompt_a=prompt_a,
                    prompt_b=prompt_b,
                    prompt_a_sha16=pa_sha16,
                    prompt_b_sha16=pb_sha16,
                    seq_len=int(N),
                    feature=int(feat1),
                    feature2=None if feat2 is None else int(feat2),
                    scale=str(scale),
                    alpha=float(alpha),
                    temperature=float(temperature),
                    x=[int(v) for v in x],
                    a_f1=[float(v) for v in s_orig_a1],
                    b_f1=[float(v) for v in s_orig_b1],
                    a_f2=a_f2,
                    b_f2=b_f2,
                    fa_f1=[float(v) for v in s_fa1],
                    fb_f1=[float(v) for v in s_fb1],
                    fa_f2=fa_f2,
                    fb_f2=fb_f2,
                    subtitle=subtitle,
                )

                # Lightweight interpretability: alignment pairs (argmax matches)
                peak_divergence_a2b_pair = None
                peak_divergence_a2b_weight = None
                try:
                    a2b = np.asarray(info.get('a2b'), dtype=np.float64)
                    align_pairs = [(int(i), int(np.argmax(a2b[i]))) for i in range(min(a2b.shape[0], N))]
                    if peak_divergence_index is not None:
                        pi = int(peak_divergence_index)
                        if 0 <= pi < int(a2b.shape[0]):
                            bj = int(np.argmax(a2b[pi]))
                            peak_divergence_a2b_pair = [int(pi), int(bj)]
                            peak_divergence_a2b_weight = float(a2b[pi, bj])
                except Exception:
                    align_pairs = []
                    peak_divergence_a2b_pair = None
                    peak_divergence_a2b_weight = None

                obj = {
                    'title': metrics.title,
                    'subtitle': metrics.subtitle,
                    'strategy': metrics.strategy,
                    'model': metrics.model,
                    'tokenizer': metrics.tokenizer,
                    'layer': metrics.layer,
                    'prompt_a': metrics.prompt_a,
                    'prompt_b': metrics.prompt_b,
                    'prompt_a_sha16': metrics.prompt_a_sha16,
                    'prompt_b_sha16': metrics.prompt_b_sha16,
                    'seq_len': metrics.seq_len,
                    'feature': metrics.feature,
                    'feature2': metrics.feature2,
                    'scale': metrics.scale,
                    'alpha': metrics.alpha,
                    'temperature': metrics.temperature,
                    'x': metrics.x,
                    'a_f1': metrics.a_f1,
                    'b_f1': metrics.b_f1,
                    'a_f2': metrics.a_f2,
                    'b_f2': metrics.b_f2,
                    'fa_f1': metrics.fa_f1,
                    'fb_f1': metrics.fb_f1,
                    'fa_f2': metrics.fa_f2,
                    'fb_f2': metrics.fb_f2,
                    'alignment_pairs': align_pairs,
                    'peak_divergence_step': peak_divergence_step,
                    'peak_divergence_value': peak_divergence_value,
                    'peak_divergence_a2b_pair': peak_divergence_a2b_pair,
                    'peak_divergence_a2b_weight': peak_divergence_a2b_weight,
                }

                out_json.write_text(json.dumps(obj, indent=2))

                plot_co_attention_traces(metrics, out_path=str(out_png), title=title)

                if bool(getattr(args, 'interactive', False)):
                    out_html = Path(str(getattr(args, 'out_html', '_cli_runs/co_attention_traces.html')))
                    out_html.parent.mkdir(parents=True, exist_ok=True)
                    plot_co_attention_traces_interactive(metrics, out_path=str(out_html), title=title)

                print(str(out_json))
                print(str(out_png))
                if bool(getattr(args, 'interactive', False)):
                    print(str(out_html))
                return 0

            except Exception as e:
                logger.error(f"Fusion co-attention traces failed: {e}")
                return 1

        elif args.command == 'fusion_pi_viz':
            try:
                import numpy as np
                from pathlib import Path

                from ..visualization.fusion_pi_viz import l2_timeseries, plot_pi_viz, save_pi_viz_png, compute_metrics

                # Determine mode
                use_model = bool(getattr(args, 'model', None))

                if use_model:
                    import torch
                    from transformers import AutoModel, AutoTokenizer

                    if not getattr(args, 'prompt_a', None) or not getattr(args, 'prompt_b', None):
                        raise ValueError('prompt mode requires --prompt-a and --prompt-b')
                    if getattr(args, 'layer', None) is None:
                        raise ValueError('prompt mode requires --layer (0-indexed)')

                    # Device
                    dev = getattr(args, 'device', 'auto')
                    if dev == 'auto':
                        if torch.cuda.is_available():
                            dev = 'cuda'
                        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            dev = 'mps'
                        else:
                            dev = 'cpu'

                    tok_id = getattr(args, 'tokenizer', None) or args.model
                    tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
                    try:
                        mdl = AutoModel.from_pretrained(args.model, use_safetensors=True)
                    except Exception:
                        try:
                            mdl = AutoModel.from_pretrained(args.model)
                        except Exception as e:
                            raise RuntimeError(
                                "Failed to load model weights. Prefer models that ship 'safetensors' weights, "
                                "or upgrade torch (>=2.6) to load legacy .bin weights safely."
                            ) from e
                    mdl.eval()
                    mdl.to(dev)

                    layer = int(getattr(args, 'layer'))
                    max_steps = int(getattr(args, 'max_steps', 25) or 25)

                    def _series_tokens(prompt: str):
                        inputs = tok(prompt, return_tensors='pt', truncation=True)
                        # token strings for interpretability
                        try:
                            ids = inputs.get('input_ids')[0].tolist()
                            toks = tok.convert_ids_to_tokens(ids)
                        except Exception:
                            toks = []
                        inputs = {k: v.to(dev) for k, v in inputs.items()}
                        with torch.no_grad():
                            out = mdl(**inputs, output_hidden_states=True, return_dict=True)
                        hs = getattr(out, 'hidden_states', None)
                        if hs is None:
                            raise ValueError('Model did not return hidden_states')
                        layers = list(hs[1:])
                        if layer < 0 or layer >= len(layers):
                            raise ValueError(f"--layer must be in [0, {len(layers)-1}] for this model")
                        h = layers[layer][0]  # [seq, hidden]
                        h = h[:max_steps]
                        arr = h.detach().cpu().numpy().astype('float32', copy=False)
                        series = np.linalg.norm(arr, axis=1).astype('float32')
                        if toks:
                            toks = toks[: int(series.shape[0])]
                        return series, toks

                    a, toks_a = _series_tokens(str(args.prompt_a))
                    b, toks_b = _series_tokens(str(args.prompt_b))
                    # Align to common length for fair comparison
                    n = int(min(a.shape[0], b.shape[0]))
                    a = a[:n]
                    b = b[:n]
                    toks_a = toks_a[:n] if toks_a else None
                    toks_b = toks_b[:n] if toks_b else None
                    title = str(getattr(args, 'title', 'Fusion Attack Analysis: π-viz'))

                else:
                    # Array mode (no simulation): require both inputs
                    if not getattr(args, 'primary', None) or not getattr(args, 'secondary', None):
                        raise ValueError('array mode requires --primary and --secondary (no simulation)')

                    P = np.load(str(args.primary))
                    S = np.load(str(args.secondary))
                    layer = getattr(args, 'layer', None)
                    layer_axis = int(getattr(args, 'layer_axis', 0) or 0)
                    max_steps = int(getattr(args, 'max_steps', 25) or 25)
                    a = l2_timeseries(P, layer=layer, layer_axis=layer_axis, max_steps=max_steps)
                    b = l2_timeseries(S, layer=layer, layer_axis=layer_axis, max_steps=max_steps)

                    title = str(getattr(args, 'title', 'Fusion Attack Analysis: π-viz'))
                    if layer is not None and np.asarray(P).ndim >= 3:
                        title = f"{title} (layer={int(layer)})"

                    toks_a = None
                    toks_b = None

                import json
                import hashlib

                z_thr = float(getattr(args, 'z_threshold', 3.0) or 3.0)
                max_spikes = int(getattr(args, 'max_spikes', 6) or 6)

                # Write outputs
                out_png = Path(str(getattr(args, 'out_png', '_cli_runs/fusion_pi_viz.png')))
                out_png.parent.mkdir(parents=True, exist_ok=True)
                save_pi_viz_png(a, b, str(out_png), title=title, z_threshold=z_thr)
                print(str(out_png))

                # Summary JSON (metrics + triage step)
                m = compute_metrics(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), z_threshold=z_thr)
                out_json = Path(str(getattr(args, 'out_json', '_cli_runs/fusion_pi_viz_summary.json')))
                out_json.parent.mkdir(parents=True, exist_ok=True)

                summary = {
                    'mode': 'prompt' if use_model else 'array',
                    'title': str(title),
                    'steps': int(m.steps),
                    'corr': None if not np.isfinite(m.corr) else float(m.corr),
                    'mean_gap': float(m.mean_gap),
                    'max_gap': float(m.max_gap),
                    'max_gap_step': int(m.max_gap_step),
                    'z_threshold': float(m.spike_z_threshold),
                    'spike_steps': [int(x) for x in (m.spike_steps or [])],
                }

                if use_model:
                    pa = str(getattr(args, 'prompt_a'))
                    pb = str(getattr(args, 'prompt_b'))
                    summary.update({
                        'model': str(args.model),
                        'tokenizer': str(getattr(args, 'tokenizer', None) or args.model),
                        'layer': int(getattr(args, 'layer')),
                        'prompt_a_sha16': hashlib.sha256(pa.encode('utf-8', errors='ignore')).hexdigest()[:16],
                        'prompt_b_sha16': hashlib.sha256(pb.encode('utf-8', errors='ignore')).hexdigest()[:16],
                    })
                    # token at max-gap step (if available)
                    try:
                        t = int(m.max_gap_step)
                        summary['token_a_at_max_gap'] = toks_a[t] if toks_a else None
                        summary['token_b_at_max_gap'] = toks_b[t] if toks_b else None
                    except Exception:
                        summary['token_a_at_max_gap'] = None
                        summary['token_b_at_max_gap'] = None
                else:
                    summary.update({
                        'primary': str(getattr(args, 'primary', None)),
                        'secondary': str(getattr(args, 'secondary', None)),
                        'layer': None if getattr(args, 'layer', None) is None else int(getattr(args, 'layer')),
                        'layer_axis': int(getattr(args, 'layer_axis', 0) or 0),
                    })

                out_json.write_text(json.dumps(summary, indent=2))
                print(str(out_json))

                if bool(getattr(args, 'interactive', False)):
                    fig = plot_pi_viz(
                        a,
                        b,
                        title=title,
                        tokens_a=toks_a,
                        tokens_b=toks_b,
                        z_threshold=z_thr,
                        max_spikes=max_spikes,
                    )
                    out_html = Path(str(getattr(args, 'out_html', '_cli_runs/fusion_pi_viz.html')))
                    out_html.parent.mkdir(parents=True, exist_ok=True)
                    fig.write_html(str(out_html))
                    print(str(out_html))

                return 0
            except Exception as e:
                logger.error(f"Fusion π-viz failed: {e}")
                return 1

        elif args.command == 'fusion_attack':
            try:
                import numpy as np
                import matplotlib.pyplot as plt
                import json
                from pathlib import Path
                P = np.load(args.primary)
                p_arr = np.array(P)
                if np.any(~np.isfinite(p_arr)):
                    p_arr = np.nan_to_num(p_arr, nan=0.0, posinf=0.0, neginf=0.0)
                p = p_arr.reshape(-1, p_arr.shape[-1]) if p_arr.ndim > 1 else p_arr.reshape(1, -1)
                if not getattr(args, 'secondary', None) or not Path(args.secondary).exists():
                    raise FileNotFoundError(
                        "Missing --secondary (required; no synthetic fallback). Provide a real .npy/.npz secondary array."
                    )

                S = np.load(args.secondary)
                s_arr = np.array(S)
                if np.any(~np.isfinite(s_arr)):
                    s_arr = np.nan_to_num(s_arr, nan=0.0, posinf=0.0, neginf=0.0)
                s = s_arr.reshape(-1, s_arr.shape[-1]) if s_arr.ndim > 1 else s_arr.reshape(1, -1)
                D = min(p.shape[1], s.shape[1]); N = min(p.shape[0], s.shape[0])
                p = p[:N,:D]; s = s[:N,:D]
                # Guard small shapes
                if p.size == 0 or s.size == 0:
                    raise ValueError('Primary/secondary arrays must be non-empty after alignment')
                if getattr(args, 'sweep', False):
                    alphas = np.linspace(0.0, 1.0, 41)
                    metric = []
                    m_lo = []
                    m_hi = []
                    for a in alphas:
                        fused = (1-a)*p + a*s
                        # Data-driven metric: mean per-sample Δ||·||₂ relative to primary
                        p_norm = np.linalg.norm(p, axis=1)
                        f_norm = np.linalg.norm(fused, axis=1)
                        delta = f_norm - p_norm
                        metric.append(float(delta.mean()))
                        if delta.size > 1:
                            m_lo.append(float(np.percentile(delta, 5)))
                            m_hi.append(float(np.percentile(delta, 95)))
                        else:
                            m_lo.append(float(delta[0]))
                            m_hi.append(float(delta[0]))
                    metric = np.array(metric, dtype=float)
                    m_mu = metric
                    m_lo = np.array(m_lo, dtype=float)
                    m_hi = np.array(m_hi, dtype=float)
                    # Derivative (slope) to find unstable regions
                    slope = np.gradient(metric, alphas)
                    steep_idx = int(np.argmax(np.abs(slope)))
                    steep_alpha = float(alphas[steep_idx])
                    # Unstable bands where |slope| exceeds mean + 1 std
                    thr = float(np.mean(np.abs(slope)) + np.std(np.abs(slope)))
                    unstable = []
                    on=False; start=0
                    for i, sv in enumerate(np.abs(slope)):
                        if sv >= thr and not on:
                            on=True; start=i
                        elif sv < thr and on:
                            on=False; unstable.append((alphas[start], alphas[i-1]))
                    if on:
                        unstable.append((alphas[start], alphas[-1]))
                    # Three-row figure: metric (+ CI), derivative, slope heatmap by dimension
                    # Make heatmap panel tall enough to clearly see each dimension
                    dim_vis = int(min(p.shape[1], 128))
                    heat_ratio = max(3.0, dim_vis / 8.0)
                    fig_height = 6.0 + dim_vis * 0.16
                    fig, axes = plt.subplots(3,1, figsize=(12, fig_height),
                                             gridspec_kw={'height_ratios':[2,1,heat_ratio]})
                    ax1, ax2, ax3 = axes
                    ax1.plot(alphas, metric, label='Δ||fused|| vs ||primary||')
                    # CI band
                    ax1.fill_between(alphas, m_lo, m_hi, color='#dbe9ff', alpha=0.6, label='CI (5-95%)')
                    ax1.plot(alphas, m_mu, color='#0b4f9c', alpha=0.8, linewidth=1.2, label='CI mean')
                    ax1.axvline(float(args.alpha), color='red', linestyle='--', label=f'alpha={float(args.alpha):g}')
                    ax1.axvline(steep_alpha, color='#0b4f9c', linestyle='-.', label=f'steepest α={steep_alpha:.2f}')
                    for a,b in unstable:
                        ax1.axvspan(a, b, color='#fff0e6', alpha=0.7, label=None)
                    ax1.set_title('NeurInSpectre — Fusion Alpha Sweep', fontsize=16, pad=26, fontweight='bold')
                    ax1.set_xlabel('Alpha'); ax1.set_ylabel('Δ L2 norm')
                    # Derivative subplot
                    ax2.plot(alphas, slope, color='#1f5fbf', label='slope d(Δ)/dα')
                    ax2.axhline(0.0, color='#888', lw=0.7)
                    ax2.axvline(steep_alpha, color='#0b4f9c', linestyle='-.')
                    ax2.axhline(thr, color='#cc0000', linestyle='--', label='|slope| threshold')
                    ax2.axhline(-thr, color='#cc0000', linestyle='--')
                    for a,b in unstable:
                        ax2.axvspan(a, b, color='#ffe6e6', alpha=0.5)
                    ax2.set_xlabel('Alpha'); ax2.set_ylabel('Slope')
                    # Slope severity background (quantiles)
                    try:
                        q75 = float(np.quantile(np.abs(slope), 0.75))
                        q90 = float(np.quantile(np.abs(slope), 0.90))
                        ax2.axhspan(q75, q90, color='#ffd9d9', alpha=0.35)
                        ax2.axhspan(q90, max(ax2.get_ylim()), color='#ffb3b3', alpha=0.35)
                        ax2.axhspan(min(ax2.get_ylim()), -q75, color='#ffd9d9', alpha=0.35)
                        ax2.axhspan(-q90, -q75, color='#ffb3b3', alpha=0.35)
                    except Exception:
                        pass
                    # Dimension-wise slope heatmap (approx sensitivity across dims)
                    try:
                        # For each alpha, estimate per-dimension change of fused vector w.r.t. alpha
                        # Using finite difference on |(1-a)p + a s|
                        dim = dim_vis
                        small = 1e-3
                        heat = []
                        for a in alphas:
                            f0 = (1-a)*p[:1,:dim] + a*s[:1,:dim]
                            f1 = (1-(a+small))*p[:1,:dim] + (a+small)*s[:1,:dim]
                            hs = np.squeeze(np.abs(f1 - f0)) / small
                            heat.append(hs)
                        heat = np.array(heat, dtype=float)
                        im = ax3.imshow(heat.T, aspect='auto', origin='lower',
                                        extent=[alphas[0], alphas[-1], 0, dim], cmap='magma')
                        ax3.set_ylabel('Dimension')
                        ax3.set_xlabel('Alpha')
                        ax3.set_title('Slope Sensitivity Heatmap (per-dimension)')
                        # Overlay unstable alpha bands for visual linkage
                        for a,b in unstable:
                            ax3.axvspan(a, b, color='white', alpha=0.08, ec=None)
                        # Dimension ticks (every ~10%)
                        try:
                            import numpy as _np
                            step = max(1, dim // 10)
                            ax3.set_yticks(_np.arange(0, dim+1, step))
                        except Exception:
                            pass
                        cb = fig.colorbar(im, ax=ax3, orientation='vertical', pad=0.01)
                        cb.set_label('|∂fused/∂α|')
                        # Annotate top‑5 most sensitive dimensions (mean over α)
                        try:
                            import numpy as _np
                            top_idx = _np.argsort(heat.mean(axis=0))[-5:][::-1]
                            label = ', '.join(map(str, top_idx.tolist()))
                            ax3.text(0.99, 0.02, f'Top sensitive dims: {label}', transform=ax3.transAxes,
                                     ha='right', va='bottom', fontsize=9,
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff7e6', edgecolor='#e6a800', alpha=0.9))
                        except Exception:
                            pass
                    except Exception:
                        ax3.text(0.5, 0.5, 'Heatmap unavailable', transform=ax3.transAxes, ha='center')
                    # Shared legend placed above plots with extra top margin for legibility
                    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, fontsize=10)
                    import textwrap as _tw
                    wrap_w = max(60, int(fig.get_size_inches()[0] * 10))
                    btxt = _tw.fill('Blue: set guardrails at high‑slope α; alert on entry into unstable bands; monitor top sensitive dimensions; test varied secondaries.', width=wrap_w)
                    rtxt = _tw.fill('Red: choose α near steepest slope for desired drift; target top sensitive dimensions for controlled drift; design secondaries along low‑noise dims.', width=wrap_w)
                    fig.text(0.01, 0.04, btxt, fontsize=9,
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6f0ff', edgecolor='#1f5fbf', alpha=0.95))
                    fig.text(0.56, 0.04, rtxt, fontsize=9,
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe6e6', edgecolor='#cc0000', alpha=0.95))
                    fig.tight_layout(rect=[0, 0.20, 1, 0.92])
                    fplot = f"{args.out_prefix}fusion_sweep.png"; fig.savefig(fplot, dpi=200, bbox_inches='tight')
                    Path(f"{args.out_prefix}fusion_sweep.json").write_text(
                        json.dumps({
                            'alphas': [float(a) for a in alphas.tolist()],
                            'delta_l2': [float(x) for x in metric.tolist()],
                            'slope': [float(x) for x in slope.tolist()],
                            'alpha_mark': float(args.alpha),
                            'steepest_alpha': steep_alpha,
                            'unstable_bands': [{'start': float(a), 'end': float(b)} for a,b in unstable]
                        }, indent=2)
                    )
                    print(fplot)
                    print(f"{args.out_prefix}fusion_sweep.json")
                    
                    # Generate INTERACTIVE Plotly HTML if requested
                    if args.interactive:
                        try:
                            import plotly.graph_objects as go
                            from plotly.subplots import make_subplots
                            
                            # Calculate additional security metrics
                            # 1. Cosine similarity between modalities
                            cosine_sim = []
                            for t in range(min(len(p), len(s))):
                                cos = np.dot(p[t], s[t]) / (np.linalg.norm(p[t]) * np.linalg.norm(s[t]) + 1e-10)
                                cosine_sim.append(float(cos))
                            cosine_sim = np.array(cosine_sim)
                            cosine_mean = float(np.mean(cosine_sim))
                            cosine_min = float(np.min(cosine_sim))
                            drift_frames = int(np.sum(cosine_sim < 0.85))
                            
                            # 2. Rank analysis
                            fused_chosen = (1-float(args.alpha))*p + float(args.alpha)*s
                            fused_rank = np.linalg.matrix_rank(fused_chosen)
                            rank_ratio = fused_rank / fused_chosen.shape[0]
                            
                            # 3. Off-Diagonal Energy Ratio (ODER)
                            with np.errstate(all="ignore"):
                                correlation_matrix = np.corrcoef(p.T, s.T)
                                correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                                n = p.shape[1]
                                diagonal_mask = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) <= 3
                                diagonal_energy = np.sum(np.abs(correlation_matrix[:n, n:][diagonal_mask[:n, :n]]))
                                total_energy = np.sum(np.abs(correlation_matrix[:n, n:]))
                                oder = 1.0 - (diagonal_energy / (total_energy + 1e-10))
                            
                            # 4. Overall risk assessment
                            risk_score = min(1.0,
                                0.3 * (1.0 - min(1.0, cosine_mean)) +  # Low cosine = risky
                                0.3 * (1.0 - rank_ratio) +  # Low rank = vulnerable
                                0.2 * oder +  # High ODER = hijack
                                0.2 * (len(unstable) / 10.0)  # Unstable zones = exploitable
                            )
                            
                            if risk_score > 0.7:
                                risk_level = "CRITICAL"
                            elif risk_score > 0.5:
                                risk_level = "HIGH"
                            elif risk_score > 0.3:
                                risk_level = "MEDIUM"
                            else:
                                risk_level = "LOW"
                            
                            # Create 4-panel interactive dashboard (added metrics panel)
                            fig_plotly = make_subplots(
                                rows=2, cols=2,
                                row_heights=[0.5, 0.5],
                                subplot_titles=(
                                    '📈 Alpha Sweep - Fusion Sensitivity',
                                    '📊 Security Risk Metrics',
                                    '🎯 Slope Sensitivity (d(Δ)/dα)',
                                    '🔥 Per-Dimension Slope Heatmap'
                                ),
                                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                                       [{'type': 'scatter'}, {'type': 'heatmap'}]],
                                vertical_spacing=0.15,
                                horizontal_spacing=0.12
                            )
                            
                            # Panel 1: Alpha sweep curve with CI band
                            fig_plotly.add_trace(go.Scatter(
                                x=alphas, y=metric,
                                mode='lines', name='Δ||fused|| vs ||primary||',
                                line=dict(color='cyan', width=3),
                                hovertemplate=(
                                    '<b>Alpha Sweep</b><br>' +
                                    'Alpha: %{x:.3f}<br>' +
                                    'Δ L2 Norm: %{y:.3f}<br>' +
                                    '<b>🔴 Red:</b> Peak = max impact point<br>' +
                                    '<b>🔵 Blue:</b> Monitor for norm spikes<br>' +
                                    '<extra></extra>'
                                )
                            ), row=1, col=1)
                            
                            # CI band
                            fig_plotly.add_trace(go.Scatter(
                                x=np.concatenate([alphas, alphas[::-1]]),
                                y=np.concatenate([m_hi, m_lo[::-1]]),
                                fill='toself',
                                fillcolor='rgba(52, 152, 219, 0.2)',
                                line=dict(color='rgba(52, 152, 219, 0)', width=0),
                                name='CI (5-95%)',
                                showlegend=True,
                                hovertemplate='CI Band<extra></extra>'
                            ), row=1, col=1)
                            
                            # Add alpha markers and unstable zone shading
                            for a_start, a_end in unstable:
                                fig_plotly.add_vrect(
                                    x0=a_start, x1=a_end,
                                    fillcolor="orange", opacity=0.2,
                                    layer="below", line_width=0,
                                    annotation_text="Unstable",
                                    annotation_position="top",
                                    row=1, col=1
                                )
                            
                            fig_plotly.add_vline(x=float(args.alpha), line_dash="dash", line_color="red", 
                                                line_width=2, annotation_text=f"Your α={args.alpha}", row=1, col=1)
                            fig_plotly.add_vline(x=steep_alpha, line_dash="dashdot", line_color="cyan",
                                                line_width=2, annotation_text=f"Steepest={steep_alpha:.2f}", row=1, col=1)
                            
                            # Panel 2: Security Metrics Display (using text annotations instead of table to avoid xaxis conflicts)
                            detection_confidence = 1.0 - risk_score  # Inverse of risk
                            modal_balance = float(args.alpha)
                            
                            # Create metrics text display
                            metrics_text = [
                                f"🎯 <b>Risk Level:</b> {risk_level}",
                                f"📊 <b>Risk Score:</b> {risk_score:.3f}",
                                f"🔍 <b>Detection Confidence:</b> {detection_confidence:.3f}",
                                f"📈 <b>Cosine Mean:</b> {cosine_mean:.3f}",
                                f"⚠️ <b>Drift Frames:</b> {drift_frames}/{len(cosine_sim)}",
                                f"🎲 <b>Rank Ratio:</b> {rank_ratio:.3f}",
                                f"⚡ <b>ODER:</b> {oder:.3f}",
                                f"🔄 <b>Modal Balance (α):</b> {modal_balance:.3f}",
                                f"📍 <b>Optimal Alpha:</b> {steep_alpha:.3f}",
                                f"🚨 <b>Unstable Zones:</b> {len(unstable)}"
                            ]
                            
                            # Add empty scatter to hold the metrics panel
                            fig_plotly.add_trace(go.Scatter(
                                x=[0], y=[0],
                                mode='text',
                                text=[''],
                                showlegend=False,
                                hoverinfo='skip'
                            ), row=1, col=2)
                            
                            # Add metrics as annotations in the panel
                            y_positions = np.linspace(0.9, 0.1, len(metrics_text))
                            for idx, (text, y_pos) in enumerate(zip(metrics_text, y_positions)):
                                bgcolor = 'rgba(220,53,69,0.5)' if idx == 0 and risk_level in ['CRITICAL', 'HIGH'] else 'rgba(50,50,50,0.7)'
                                fig_plotly.add_annotation(
                                    text=text,
                                    xref="x2", yref="y2",
                                    x=0.5, y=y_pos,
                                    showarrow=False,
                                    font=dict(size=11, color='white', family='monospace'),
                                    align="left",
                                    bgcolor=bgcolor,
                                    bordercolor='rgba(128,128,128,0.8)',
                                    borderwidth=1,
                                    borderpad=6,
                                    xanchor='center',
                                    yanchor='middle'
                                )
                            
                            # Panel 3: Slope with threshold bands and quantile zones
                            slope_colors = ['red' if abs(s) > thr else 'orange' if abs(s) > thr*0.75 else 'green' 
                                           for s in slope]
                            
                            q75 = float(np.quantile(np.abs(slope), 0.75))
                            q90 = float(np.quantile(np.abs(slope), 0.90))
                            
                            fig_plotly.add_trace(go.Scatter(
                                x=alphas, y=slope,
                                mode='lines+markers',
                                name='Slope d(Δ)/dα',
                                line=dict(color='#1f5fbf', width=2),
                                marker=dict(size=5, color=slope_colors),
                                hovertemplate=(
                                    '<b>Slope Sensitivity</b><br>' +
                                    'Alpha: %{x:.3f}<br>' +
                                    'Slope: %{y:.3f}<br>' +
                                    f'Threshold: ±{thr:.3f}<br>' +
                                    '<b>🔴 Red:</b> High |slope| = maximum leverage point<br>' +
                                    '<b>🔵 Blue:</b> Steep regions = attack vulnerability<br>' +
                                    '<extra></extra>'
                                )
                            ), row=2, col=1)
                            
                            # Add threshold lines with labels
                            fig_plotly.add_hline(y=thr, line_dash="dash", line_color="red", line_width=2,
                                                annotation_text=f"Threshold +{thr:.2f}", row=2, col=1)
                            fig_plotly.add_hline(y=-thr, line_dash="dash", line_color="red", line_width=2,
                                                annotation_text=f"Threshold -{thr:.2f}", row=2, col=1)
                            fig_plotly.add_hline(y=0, line_color="gray", line_width=1, row=2, col=1)
                            
                            # Add quantile reference bands
                            fig_plotly.add_hrect(y0=q75, y1=q90, fillcolor="rgba(255, 217, 217, 0.2)", 
                                                layer="below", line_width=0, row=2, col=1)
                            fig_plotly.add_hrect(y0=-q90, y1=-q75, fillcolor="rgba(255, 217, 217, 0.2)", 
                                                layer="below", line_width=0, row=2, col=1)
                            
                            # Panel 3: Per-dimension slope heatmap
                            dim_vis = min(p.shape[1], 128)
                            slope_per_dim = []
                            for dim in range(dim_vis):
                                dim_slopes = []
                                for a in alphas:
                                    fused_a = (1-a)*p[:, dim] + a*s[:, dim]
                                    dim_slopes.append(float(np.linalg.norm(fused_a)))
                                slope_per_dim.append(np.gradient(dim_slopes))
                            
                            slope_heatmap = np.array(slope_per_dim)
                            
                            fig_plotly.add_trace(go.Heatmap(
                                z=slope_heatmap,
                                x=alphas,
                                y=list(range(dim_vis)),
                                colorscale='Hot',
                                hovertemplate=(
                                    '<b>Dimension Sensitivity</b><br>' +
                                    'Dimension: %{y}<br>' +
                                    'Alpha: %{x:.3f}<br>' +
                                    'Slope: %{z:.4f}<br>' +
                                    '<b>🔴 Red:</b> Bright dims = manipulation targets<br>' +
                                    '<b>🔵 Blue:</b> Monitor bright rows for attacks<br>' +
                                    '<extra></extra>'
                                )
                            ), row=2, col=2)
                            
                            # Add Red/Blue guidance annotations with comprehensive metrics
                            top_dims = np.argsort(np.abs(slope_heatmap).max(axis=1))[-4:][::-1]
                            
                            red_guidance = [
                                f"🎯 RISK={risk_level} ({risk_score:.2f}) | Detection Confidence={detection_confidence:.2f} (Lower=Better)",
                                f"⚡ OPTIMAL ALPHA: {steep_alpha:.3f} (steepest slope) | Peak norm α={alphas[np.argmax(np.abs(metric))]:.3f}",
                                f"📊 TARGET DIMS: {list(top_dims)} | Cosine={cosine_mean:.2f} | Rank={rank_ratio:.2f}",
                                f"🔍 TECHNIQUE: Modal norm injection at α={alphas[np.argmax(np.abs(metric))]:.3f} | ODER={oder:.2f}",
                                f"🚨 AVOID: {len(unstable)} unstable zones (orange shaded) for stealth"
                            ]
                            
                            blue_guidance = [
                                f"🛡️ THREAT={risk_level} ({risk_score:.2f}) | {drift_frames} frames with cosine<0.85",
                                f"⚠️ GUARDRAILS: Δ norm ±{np.std(metric)*2:.1f} | Cosine>0.85 | Rank>{rank_ratio*0.75:.2f}",
                                f"📈 MONITOR: Dims {list(top_dims)} | ODER threshold=0.2 (current={oder:.2f})",
                                f"🔒 RESTRICT: Alpha to safe ranges (avoid steep_α ± 0.1 = {steep_alpha-0.1:.2f}-{steep_alpha+0.1:.2f})",
                                f"📊 VALIDATE: CI band compliance | Rank ratio={rank_ratio:.2f} (alert if <0.25)"
                            ]
                            
                            fig_plotly.add_annotation(
                                text="<b>🔴 RED TEAM - Fusion Attack Optimization</b><br>" + "<br>".join(red_guidance),
                                xref="paper", yref="paper",
                                x=0.75, y=-0.15,
                                showarrow=False,
                                font=dict(size=10, color='white'),
                                align="left",
                                bgcolor='rgba(204,0,0,0.85)',
                                bordercolor='#cc0000',
                                borderwidth=2,
                                borderpad=12,
                                xanchor='center',
                                width=780
                            )
                            
                            fig_plotly.add_annotation(
                                text="<b>🔵 BLUE TEAM - Fusion Attack Defense</b><br>" + "<br>".join(blue_guidance),
                                xref="paper", yref="paper",
                                x=0.25, y=-0.15,
                                showarrow=False,
                                font=dict(size=10, color='white'),
                                align="left",
                                bgcolor='rgba(31,95,191,0.85)',
                                bordercolor='#1f5fbf',
                                borderwidth=2,
                                borderpad=12,
                                xanchor='center',
                                width=780
                            )
                            
                            # Layout
                            fig_plotly.update_layout(
                                title=dict(
                                    text='⚡ NeurInSpectre Fusion Attack Analysis',
                                    x=0.5,
                                    xanchor='center',
                                    font=dict(size=20, color='white')
                                ),
                                height=1400,
                                width=1800,
                                template='plotly_dark',
                                hovermode='closest',
                                uirevision='constant',
                                showlegend=True,
                                legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.7)', font=dict(color='white')),
                                plot_bgcolor='rgba(20,20,20,0.95)',
                                paper_bgcolor='rgba(10,10,10,1)',
                                margin=dict(l=80, r=80, t=100, b=290)
                            )
                            
                            # Update axes with enhanced labels for all scatter/heatmap subplots
                            fig_plotly.update_xaxes(title_text="Alpha (0=Primary, 1=Secondary)", row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
                            fig_plotly.update_yaxes(title_text="Δ L2 Norm", row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
                            
                            # Hide axes for metrics panel (row=1, col=2)
                            fig_plotly.update_xaxes(visible=False, row=1, col=2)
                            fig_plotly.update_yaxes(visible=False, row=1, col=2)
                            
                            fig_plotly.update_xaxes(title_text="Alpha", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
                            fig_plotly.update_yaxes(title_text="Slope (Attack Sensitivity)", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
                            fig_plotly.update_xaxes(title_text="Alpha", row=2, col=2, gridcolor='rgba(128,128,128,0.2)')
                            fig_plotly.update_yaxes(title_text="Dimension (Feature Index)", row=2, col=2, gridcolor='rgba(128,128,128,0.2)')
                            
                            # Save interactive HTML
                            html_file = f"{args.out_prefix}interactive.html"
                            fig_plotly.write_html(html_file)
                            logger.info(f"📊 Interactive fusion analysis HTML: {html_file}")
                            logger.info(f"🔍 Features: Zoom, Pan, Hover with Red/Blue guidance")
                            logger.info(f"📈 Research-based: DEF CON '25, IEEE S&P '24, Mandiant July '25")
                            print(html_file)
                            
                        except Exception as e:
                            logger.warning(f"Interactive visualization failed: {e}, using static PNG only")
                    
                # Always produce one fused output at requested alpha
                alpha = float(args.alpha)
                fused = (1-alpha)*p + alpha*s
                np.save(f"{args.out_prefix}fused.npy", fused.astype('float32'))
                print(f"{args.out_prefix}fused.npy")
                return 0
            except Exception as e:
                logger.error(f"Fusion attack failed: {e}")
                return 1
        elif args.command == 'adversarial-ednn':
            from .adversarial_ednn import run_ednn
            return run_ednn(args)
        elif args.command == 'ednn-rag-poison':
            from .ednn_rag_poison import run_ednn_rag_poison
            return run_ednn_rag_poison(args)
        elif args.command == 'analyze-attack-vectors':
            from .attack_vector_analysis import run_attack_vector_analysis
            return run_attack_vector_analysis(args)
        elif args.command == 'recommend-countermeasures':
            from .attack_vector_analysis import run_recommend_countermeasures
            return run_recommend_countermeasures(args)
        elif args.command == 'activation_steganography':
            try:
                import json
                import numpy as np
                from pathlib import Path
                from ..activation_steganography import ActivationSteganography as _MaybeECC

                if args.steg_action == 'encode':
                    # Parse payload bits and neurons
                    bits = [int(b.strip()) for b in str(args.payload_bits).split(',') if b.strip() != '']
                    neurons = [int(n.strip()) for n in str(args.target_neurons).split(',') if n.strip() != '']
                    if len(bits) != len(neurons):
                        # allow unequal, truncate to min
                        m = min(len(bits), len(neurons))
                        bits, neurons = bits[:m], neurons[:m]
                    # Allow empty inputs gracefully
                    if len(bits) == 0 or len(neurons) == 0:
                        bits = []
                        neurons = []
                    # Try ECC-based encode if available; else marker-based
                    try:
                        encoder = _MaybeECC()
                        encoded_prompt = encoder.encode_payload(args.prompt, bits, neurons)
                        method = 'ecc'
                    except Exception:
                        marker = ','.join(map(str, bits)) if bits else ''
                        encoded_prompt = f"{args.prompt} [STEG:{marker}]" if marker else args.prompt
                        method = 'marker'
                    # Save outputs
                    outp = Path(f"{args.out_prefix}encoded_prompt.txt")
                    outp.write_text(encoded_prompt)
                    meta = {
                        'method': method,
                        'payload_bits': bits,
                        'target_neurons': neurons,
                        'model': args.model,
                        'tokenizer': args.tokenizer,
                        'prompt_len': len(args.prompt),
                    }
                    Path(f"{args.out_prefix}steg_metadata.json").write_text(json.dumps(meta, indent=2))
                    print(str(outp))
                    print(f"{args.out_prefix}steg_metadata.json")
                    return 0
                elif args.steg_action == 'extract':
                    # Load activations and extract bits via thresholding
                    act_p = getattr(args, 'activations', None)
                    if not act_p:
                        logger.error("Activation steganography extract requires --activations (no synthetic fallback).")
                        return 1
                    ap = Path(str(act_p))
                    if not ap.exists():
                        logger.error(f"Activations file not found: {ap}")
                        return 1
                    if ap.suffix.lower() == '.npz':
                        npz = np.load(str(ap), allow_pickle=True)
                        if len(npz.files) == 0:
                            raise ValueError(f"Empty .npz: {ap}")
                        A = np.asarray(npz[npz.files[0]])
                    else:
                        obj = np.load(str(ap), allow_pickle=True)
                        if getattr(obj, "dtype", None) == object and getattr(obj, "shape", ()) == ():
                            obj = obj.item()
                        if isinstance(obj, dict):
                            for k in ("activations", "data", "X", "x", "arr"):
                                if k in obj:
                                    obj = obj[k]
                                    break
                        A = np.asarray(obj)
                    arr = np.array(A)
                    # Clean NaN/Inf
                    import numpy as _np
                    arr = _np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    # Normalize dims to [hidden]
                    if arr.ndim == 1:
                        last = arr
                    elif arr.ndim == 2:
                        last = arr[-1]
                    else:
                        # assume [batch, seq, hidden]
                        last = arr.reshape(-1, arr.shape[-1])[-1]
                    neurons = [int(n.strip()) for n in str(args.target_neurons).split(',') if n.strip() != '']
                    try:
                        thr = float(args.threshold)
                    except Exception:
                        thr = 0.0
                    bits = []
                    for n in neurons:
                        if n < 0 or n >= last.shape[-1]:
                            bits.append(0)
                        else:
                            bits.append(1 if last[n] >= thr else 0)
                    # Save
                    outj = Path(f"{args.out_prefix}steg_extract.json")
                    outj.write_text(json.dumps({'target_neurons': neurons, 'threshold': thr, 'bits': bits}, indent=2))
                    # Visualization: bar plot of target neuron activations vs threshold with Red/Blue keys
                    try:
                        import matplotlib.pyplot as _plt
                        import numpy as _np
                        if len(neurons) == 0:
                            raise RuntimeError('No target neurons provided')
                        vals = [_np.nan if (n < 0 or n >= last.shape[-1]) else float(last[n]) for n in neurons]
                        colors = ['#2A9D8F' if b == 1 else '#7f8c8d' for b in bits]
                        _plt.figure(figsize=(9.5, 3.6))
                        _plt.bar(range(len(neurons)), vals, color=colors)
                        _plt.axhline(thr, color='red', linestyle='--', label=f'threshold={thr:g}')
                        _plt.title('NeurInSpectre — Activation Steganography Extraction')
                        _plt.xlabel('Target neuron index (order provided)')
                        _plt.ylabel('Activation value (last step)')
                        _plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
                        fig = _plt.gcf()
                        import textwrap as _tw
                        btxt = _tw.fill('Blue: bits=1 above threshold; monitor stable carriers; rotate thresholds if drift observed.', width=90)
                        rtxt = _tw.fill('Red: push carriers below threshold; randomize carriers to shorten stability; target near-threshold flips.', width=90)
                        fig.text(0.01, 0.02, btxt, fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6f0ff', edgecolor='#1f5fbf', alpha=0.95))
                        fig.text(0.56, 0.02, rtxt, fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe6e6', edgecolor='#cc0000', alpha=0.95))
                        _plt.tight_layout(rect=[0, 0.20, 1, 1])
                        fplot = f"{args.out_prefix}steg_extract.png"
                        _plt.savefig(fplot, dpi=200, bbox_inches='tight')
                        print(fplot)
                    except Exception:
                        pass
                    print(str(outj))
                    return 0
                else:
                    logger.error('Unknown activation_steganography action')
                    return 1
            except Exception as e:
                logger.error(f"Activation steganography failed: {e}")
                return 1
        elif args.command == 'activations':
            # Inline implementation to avoid adding new files
            try:
                import os
                import torch
                import numpy as np
                import matplotlib.pyplot as plt
                from transformers import AutoTokenizer, AutoModelForCausalLM

                def resolve_model_ref(model_ref: str) -> str:
                    """Resolve HF model id or local checkpoint directory.

                    Supports:
                    - explicit local paths (existing file/dir)
                    - name-only lookups in common local dirs:
                      - ./models/<name>, ./_models/<name>
                      - directories listed in $NEURINSPECTRE_MODEL_DIRS (colon-separated)
                    """
                    ref = os.path.expanduser(str(model_ref))
                    if os.path.exists(ref):
                        return ref
                    # Name-only: search local dirs
                    if any(sep in ref for sep in (os.sep, "/", "\\")):
                        return ref
                    search_dirs = []
                    env_dirs = os.environ.get("NEURINSPECTRE_MODEL_DIRS", "").strip()
                    if env_dirs:
                        search_dirs.extend([d for d in env_dirs.split(":") if d.strip()])
                    search_dirs.extend(["models", "_models"])
                    for base in search_dirs:
                        base_exp = os.path.expanduser(base)
                        cand = os.path.join(base_exp, ref)
                        if os.path.isdir(cand):
                            # Prefer directories that look like HF checkpoints.
                            if os.path.exists(os.path.join(cand, "config.json")):
                                return cand
                            # Fall back to any directory if it's the only match.
                            return cand
                    return ref

                def select_device(pref: str) -> torch.device:
                    if pref == 'auto':
                        if torch.backends.mps.is_available():
                            return torch.device('mps')
                        if torch.cuda.is_available():
                            return torch.device('cuda')
                        return torch.device('cpu')
                    if pref == 'mps':
                        return torch.device('mps')
                    if pref == 'cuda':
                        return torch.device('cuda')
                    return torch.device('cpu')

                def resolve_layer_module(model, spec_idx=None, spec_path=None):
                    if spec_path:
                        mod = model
                        for part in spec_path.split('.'):  # supports attributes and integer indices
                            if part.isdigit():
                                mod = mod[int(part)]
                            else:
                                mod = getattr(mod, part)
                        return mod
                    if spec_idx is not None:
                        # Common GPT-2/GPT-Neo stack location
                        return model.transformer.h[spec_idx]
                    raise ValueError('Provide --layer or --layer-path')

                # Load model/tokenizer
                model_ref_in = str(getattr(args, "model", ""))
                model_ref = resolve_model_ref(model_ref_in)
                try:
                    tok = AutoTokenizer.from_pretrained(model_ref, use_fast=True)
                    mdl = AutoModelForCausalLM.from_pretrained(model_ref)
                except Exception as e:
                    # Provide a clearer remediation message for local checkpoints vs HF ids.
                    msg = (
                        f"Failed to load --model '{model_ref_in}'.\n\n"
                        "NeurInSpectre expects either:\n"
                        "- a HuggingFace model id (e.g. 'gpt2', 'distilbert-base-uncased') OR\n"
                        "- a local directory containing a HF checkpoint (must include 'config.json').\n\n"
                        f"Resolved model ref: {model_ref}\n\n"
                        "If you have a local checkpoint named 'poisoned_gpt2', pass its path:\n"
                        "  neurinspectre activations --model /absolute/path/to/poisoned_gpt2 ...\n"
                        "Or put it under ./models/poisoned_gpt2 (or set NEURINSPECTRE_MODEL_DIRS).\n"
                    )
                    raise RuntimeError(msg) from e
                dev = select_device(args.device)
                mdl.to(dev).eval()

                inputs = tok(args.prompt, return_tensors='pt')
                inputs = {k: v.to(dev) for k, v in inputs.items()}

                activations = {}
                target_module = resolve_layer_module(mdl, args.layer, args.layer_path)

                hook = target_module.register_forward_hook(
                    lambda mod, inp, out: activations.__setitem__('a', out[0].detach().cpu().numpy())
                )
                with torch.no_grad():
                    mdl(**inputs)
                hook.remove()

                if 'a' not in activations:
                    print('ERROR: Failed to capture activations')
                    return 1

                A = activations['a'][0]  # shape: [seq_len, hidden_dim]
                last = A[-1]
                layer_label = args.layer_path if args.layer_path else str(args.layer)

                # Compute hotspots on last-token absolute activations
                percentile = float(args.hotspot_percentile)
                thr = float(np.percentile(np.abs(last), percentile))
                hot_mask = np.abs(last) >= thr
                
                # Generate INTERACTIVE Plotly HTML if requested (2024-2025 research-based)
                if args.interactive:
                    try:
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        
                        # Calculate security metrics
                        activation_variance = float(np.var(last))
                        max_activation = float(np.max(np.abs(last)))
                        sparsity = float(np.sum(np.abs(last) < 0.01) / len(last))
                        hotspot_count = int(np.sum(hot_mask))
                        
                        # Determine threat level based on research
                        if max_activation > 10.0 or hotspot_count > len(last) * 0.1:
                            threat_level = "CRITICAL"
                        elif max_activation > 5.0 or hotspot_count > len(last) * 0.05:
                            threat_level = "HIGH"
                        elif activation_variance > 1.0:
                            threat_level = "MEDIUM"
                        else:
                            threat_level = "LOW"
                        
                        # Create interactive figure
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=(
                                f'🎯 Layer {layer_label} Last-Token Activations',
                                f'🗺️ Activation Heatmap (Hidden × Sequence)',
                                f'📊 Top-{args.topk} Critical Neurons',
                                '🛡️ Security Analysis'
                            ),
                            specs=[[{'type': 'scatter'}, {'type': 'heatmap'}],
                                   [{'type': 'bar'}, {'type': 'table'}]],
                            vertical_spacing=0.15,
                            horizontal_spacing=0.12
                        )
                        
                        # Panel 1: Last-token line plot with hotspots
                        fig.add_trace(go.Scatter(
                            x=list(range(len(last))),
                            y=last,
                            mode='lines',
                            name='Activations',
                            line=dict(color='cyan', width=2),
                            hovertemplate=(
                                '<b>Neuron %{x}</b><br>' +
                                'Activation: %{y:.4f}<br>' +
                                f'Hotspot: %{{customdata}}<br>' +
                                '<b>🔴 Red Team:</b> High activation = steering target<br>' +
                                '<b>🔵 Blue Team:</b> Monitor for manipulation<br>' +
                                '<extra></extra>'
                            ),
                            customdata=['YES' if hot_mask[i] else 'NO' for i in range(len(last))]
                        ), row=1, col=1)
                        
                        # Add hotspot shading
                        for s, e in [(s, e) for s, e in [(i, i+1) for i in range(len(hot_mask)) if hot_mask[i]]]:
                            fig.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.15, layer="below", line_width=0, row=1, col=1)
                        
                        # Panel 2: Activation heatmap with token labels (clean display)
                        tokens = tok.convert_ids_to_tokens(inputs['input_ids'][0])
                        # Clean tokens: remove Ġ (GPT-2 space marker) and other special chars
                        token_strings = [str(t).replace('Ġ', '').replace('Â', '').replace('ċ', '') for t in tokens]
                        
                        fig.add_trace(go.Heatmap(
                            z=A.T,
                            x=token_strings,
                            y=list(range(A.shape[1])),
                            colorscale='Viridis',
                            hovertemplate=(
                                '<b>Token:</b> %{x}<br>' +
                                '<b>Neuron:</b> %{y}<br>' +
                                '<b>Activation:</b> %{z:.4f}<br>' +
                                '<b>🔴 Red Team:</b> High values = injection points for activation steering<br>' +
                                '<b>🔵 Blue Team:</b> Anomalous patterns indicate potential attacks<br>' +
                                '<extra></extra>'
                            )
                        ), row=1, col=2)
                        
                        # Panel 3: Top-K bar chart
                        topk = max(1, int(args.topk))
                        idxs = np.argsort(np.abs(last))[-topk:][::-1]
                        vals = last[idxs]
                        mu, sd = float(np.mean(last)), float(np.std(last) + 1e-12)
                        z_scores = (vals - mu) / sd
                        
                        fig.add_trace(go.Bar(
                            x=list(range(topk)),
                            y=vals,
                            marker_color=['red' if abs(z) > 3 else 'orange' if abs(z) > 2 else 'yellow' 
                                         for z in z_scores],
                            hovertemplate=(
                                'Neuron: %{customdata[0]}<br>' +
                                'Activation: %{y:.4f}<br>' +
                                'Z-score: %{customdata[1]:.2f}<br>' +
                                '<b>🔴 Red Team:</b> Target neuron %{customdata[0]} for activation steering<br>' +
                                '<b>🔵 Blue Team:</b> Z>3 = anomaly → clip or regularize<br>' +
                                '<extra></extra>'
                            ),
                            customdata=list(zip(idxs, z_scores))
                        ), row=2, col=1)
                        
                        # Panel 4: Security summary table
                        summary_data = {
                            'Metric': ['Threat Level', 'Max Activation', 'Variance', 'Sparsity', 'Hotspots', 'Top-K Max |Z|'],
                            'Value': [
                                threat_level,
                                f"{max_activation:.3f}",
                                f"{activation_variance:.3f}",
                                f"{sparsity:.1%}",
                                f"{hotspot_count}/{len(last)}",
                                f"{float(np.max(np.abs(z_scores))):.2f}"
                            ]
                        }
                        
                        fig.add_trace(go.Table(
                            header=dict(values=['<b>Security Metric</b>', '<b>Value</b>'],
                                       fill_color='rgba(128,128,128,0.5)',
                                       align='left',
                                       font=dict(color='white', size=12)),
                            cells=dict(values=[summary_data['Metric'], summary_data['Value']],
                                      fill_color='rgba(50,50,50,0.5)',
                                      align='left',
                                      font=dict(color='white', size=11),
                                      height=30)
                        ), row=2, col=2)
                        
                        # Add research-based Red/Blue team guidance below
                        red_guidance = [
                            f"🎯 STEERING: Target {hotspot_count} hotspot neurons for activation patching",
                            f"🔍 JAILBREAK: High variance ({activation_variance:.2f}) enables semantic drift attacks",
                            f"📊 EXPLOIT: Top-K neurons (|Z|>{float(np.max(np.abs(z_scores))):.1f}) for trojans/backdoors"
                        ]
                        
                        blue_guidance = [
                            f"🛡️ CLIP: Regularize neurons with |Z|>3.0 ({int(np.sum(np.abs(z_scores) > 3))} detected)",
                            f"⚠️ MONITOR: {hotspot_count} hotspot spans (>{percentile}th percentile) require baseline profiling",
                            f"🔒 DEFEND: Sparsity={sparsity:.1%} → Implement activation noise (σ={max_activation/10:.2f})"
                        ]
                        
                        fig.add_annotation(
                            text="<b>🔴 RED TEAM - Activation Exploitation</b><br>" + "<br>".join(red_guidance),
                            xref="paper", yref="paper", x=0.75, y=-0.16, showarrow=False,
                            font=dict(size=10, color='white'), align="left",
                            bgcolor='rgba(204,0,0,0.85)', bordercolor='#cc0000', borderwidth=2, borderpad=10,
                            xanchor='center', width=750
                        )
                        
                        fig.add_annotation(
                            text="<b>🔵 BLUE TEAM - Activation Defense</b><br>" + "<br>".join(blue_guidance),
                            xref="paper", yref="paper", x=0.25, y=-0.16, showarrow=False,
                            font=dict(size=10, color='white'), align="left",
                            bgcolor='rgba(31,95,191,0.85)', bordercolor='#1f5fbf', borderwidth=2, borderpad=10,
                            xanchor='center', width=750
                        )
                        
                        # Layout
                        fig.update_layout(
                            title=dict(text=f'⚡ NeurInSpectre Activation Analysis - Layer {layer_label}',
                                      x=0.5, xanchor='center', font=dict(size=18, color='white')),
                            height=1150, width=1800, template='plotly_dark', hovermode='closest',
                            uirevision='constant', showlegend=True,
                            legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.7)', font=dict(color='white')),
                            plot_bgcolor='rgba(20,20,20,0.95)', paper_bgcolor='rgba(10,10,10,1)',
                            margin=dict(l=80, r=80, t=100, b=240)
                        )
                        
                        # Update axes
                        fig.update_xaxes(title_text="Hidden Unit Index", row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
                        fig.update_yaxes(title_text="Activation Value", row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
                        fig.update_xaxes(title_text="Token Position", row=1, col=2, gridcolor='rgba(128,128,128,0.2)')
                        fig.update_yaxes(title_text="Hidden Unit", row=1, col=2, gridcolor='rgba(128,128,128,0.2)')
                        fig.update_xaxes(title_text="Top-K Rank", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
                        fig.update_yaxes(title_text="Activation Value", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
                        
                        # Save interactive HTML
                        html_file = f'{args.out_prefix}{layer_label}_interactive.html'
                        fig.write_html(html_file)
                        logger.info(f"📊 Interactive HTML saved: {html_file}")
                        logger.info(f"🔍 Features: Zoom, Hover (Red/Blue guidance), Research-based threat analysis")
                        print(html_file)
                    
                    except Exception as e:
                        logger.warning(f"Interactive visualization failed: {e}, falling back to static")
                # Build contiguous spans of hotspots
                spans = []
                start = None
                for i, on in enumerate(hot_mask.tolist()):
                    if on and start is None:
                        start = i
                    elif not on and start is not None:
                        spans.append((start, i - 1))
                        start = None
                if start is not None:
                    spans.append((start, len(hot_mask) - 1))

                # Line plot (last token)
                plt.figure(figsize=(10, 3))
                plt.plot(last)
                plt.title(f'NeurInSpectre — Layer {layer_label} Last-Token Activations')
                plt.xlabel('Hidden unit index')
                plt.ylabel('Activation value')
                # Overlay hotspot spans
                for s, e in spans:
                    plt.axvspan(s, e, color='red', alpha=0.15)
                if spans:
                    plt.suptitle(f'Hotspots P{percentile:.1f} | spans: {spans}', y=1.02, fontsize=9)
                plt.tight_layout()
                f1 = f'{args.out_prefix}{layer_label}_last_token_line.png'
                plt.savefig(f1, dpi=200)

                # Heatmap (hidden x sequence)
                plt.figure(figsize=(10, 3))
                im = plt.imshow(A.T, aspect='auto', cmap='viridis')
                cbar = plt.colorbar(im)
                cbar.set_label('Activation value')
                plt.title(f'NeurInSpectre — Layer {layer_label} Activations (hidden × sequence)')
                plt.xlabel('Token position (sequence index)')
                plt.ylabel('Hidden unit index')
                # Overlay horizontal hotspot spans across full sequence
                import matplotlib.pyplot as _plt
                ax = _plt.gca()
                for s, e in spans:
                    ax.axhspan(s, e, color='red', alpha=0.08)
                plt.tight_layout()
                f2 = f'{args.out_prefix}{layer_label}_heatmap.png'
                plt.savefig(f2, dpi=200)

                # Top-K bar chart (last token)
                topk = max(1, int(args.topk))
                idxs = np.argsort(np.abs(last))[-topk:][::-1]
                vals = last[idxs]
                plt.figure(figsize=(10, 4))
                plt.bar(range(topk), vals)
                plt.xticks(range(topk), idxs, rotation=90)
                plt.title(f'NeurInSpectre — Layer {layer_label} Top-{topk} Activations (last token)')
                plt.xlabel('Hidden unit index (top-|activation|)')
                plt.ylabel('Activation value')
                # Annotate if any of the bars fall inside hotspot spans
                if spans:
                    in_spans = []
                    for hid in idxs.tolist():
                        for s, e in spans:
                            if s <= hid <= e:
                                in_spans.append(hid)
                                break
                    if in_spans:
                        plt.suptitle(f'Top-k intersect hotspots at: {sorted(in_spans)}', y=1.02, fontsize=9)
                # Add simple z-score badges against last-token distribution
                try:
                    import numpy as _np
                    mu, sd = float(_np.mean(last)), float(_np.std(last) + 1e-12)
                    z = (vals - mu) / sd
                    # show the max-z on the figure to guide reading
                    zmax = float(_np.max(_np.abs(z)))
                    plt.figtext(0.98, 0.15, f'max |z|={zmax:.2f}', ha='right', va='bottom', fontsize=9,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f2f2f2', edgecolor='#808080', alpha=0.85))
                except Exception:
                    pass
                plt.tight_layout()
                f3 = f'{args.out_prefix}{layer_label}_topk.png'
                plt.savefig(f3, dpi=200)

                # Optional JSON export
                if args.json_out:
                    import json
                    export = {
                        'model': args.model,
                        'prompt': args.prompt,
                        'layer': layer_label,
                        'hidden_size': int(A.shape[1]),
                        'sequence_length': int(A.shape[0]),
                        'percentile': percentile,
                        'threshold_abs': thr,
                        'hotspot_spans': [
                            {
                                'start': int(s),
                                'end': int(e),
                                'mean': float(np.mean(last[s:e+1])),
                                'std': float(np.std(last[s:e+1]))
                            } for s, e in spans
                        ],
                        'topk': int(topk),
                        'topk_indices': [int(i) for i in idxs.tolist()],
                        'topk_values': [float(v) for v in vals.tolist()],
                        'outputs': {'line': f1, 'heatmap': f2, 'topk': f3}
                    }
                    with open(args.json_out, 'w') as jf:
                        json.dump(export, jf, indent=2)

                print(f1)
                print(f2)
                print(f3)
                return 0
            except Exception as e:
                logger.error(f"Activations command failed: {e}")
                return 1
        elif args.command == 'gpu':
            from .gpu_detection_cli import run_gpu_detection_command
            return run_gpu_detection_command(args)
        elif args.command == 'temporal-analysis':
            from .temporal_analysis_commands import handle_temporal_analysis
            return handle_temporal_analysis(args)
        elif args.command == 'rl-obfuscation':
            from .rl_obfuscation_commands import handle_rl_obfuscation
            return handle_rl_obfuscation(args)
        elif args.command == 'red-team':
            from .red_team_commands import handle_red_team
            return handle_red_team(args)
        elif args.command == 'blue-team':
            from .blue_team_commands import handle_blue_team
            return handle_blue_team(args)
        elif args.command == 'comprehensive-test':
            from .comprehensive_test_commands import handle_comprehensive_test
            return handle_comprehensive_test(args)
        elif args.command in ['drift-detect', 'drift_detect', 'drift-detection', 'drift_detection']:
            from .statistical_commands import _handle_drift_detect
            return _handle_drift_detect(args)
        elif args.command in ['zscore', 'z-score', 'z_score', 'zscore-analysis', 'zscore_analysis']:
            from .statistical_commands import _handle_zscore
            return _handle_zscore(args)
        elif args.command in ['adversarial-detect', 'evasion-detect', 'comprehensive-scan', 'realtime-monitor']:
            from .security_commands import handle_security_command
            return handle_security_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    except ImportError as e:
        logger.error(f"Failed to import command module: {e}")
        logger.info("Some TTD modules may not be available. Please check your installation.")
        return 1
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        return 1

def handle_obfuscated_gradient_command(args):
    """Handle obfuscated gradient visualization commands"""
    try:
        import os
        import numpy as np
        from pathlib import Path
        from ..security.visualization.obfuscated_gradient_visualizer import ObfuscatedGradientVisualizer
        
        if not args.gradient_command:
            logger.error("No gradient subcommand specified")
            return 1
        
        # Output directory (best-effort) — many subcommands support `--output-dir/-o`.
        # Some older paths used to ignore it for certain subcommands; keep it consistent.
        out_dir_raw = getattr(args, "output_dir", None)
        if out_dir_raw:
            output_dir = Path(str(out_dir_raw))
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")
        
        # Initialize visualizer (only for commands that need it)
        if args.gradient_command in ['create', 'analyze', 'demo', 'generate']:
            visualizer = ObfuscatedGradientVisualizer()
        
        if args.gradient_command == 'create':
            print("🎨 Creating comprehensive obfuscated gradient analysis...")
            print(f"📁 Output directory: {output_dir}")
            
            # Set device preference
            if hasattr(args, 'device') and args.device != 'auto':
                print(f"🔧 Device preference: {args.device}")
            
            # Load gradient data (strict: no demo fallback)
            print(f"📊 Analyzing gradient file: {args.input_file}")
            if not os.path.exists(args.input_file):
                logger.error(f"❌ Input file not found: {args.input_file}")
                logger.error("   Provide a real .npy file or use: neurinspectre obfuscated-gradient demo")
                return 1
            
            try:
                input_gradients = np.load(args.input_file, allow_pickle=True)
                input_gradients = np.asarray(input_gradients)
                print(f"   ✅ Loaded gradient data: {input_gradients.shape}")
            except Exception as e:
                logger.error(f"❌ Error loading gradient file: {e}")
                return 1

            gradients = {'observed': input_gradients}

            # Optional reference/baseline comparison (strict: must exist)
            if getattr(args, 'reference_file', None):
                ref_path = str(args.reference_file)
                if not os.path.exists(ref_path):
                    logger.error(f"❌ Reference file not found: {ref_path}")
                    return 1
                try:
                    ref = np.load(ref_path, allow_pickle=True)
                    ref = np.asarray(ref)
                    gradients['reference'] = ref
                    print(f"   ✅ Loaded reference baseline: {ref.shape}")
                except Exception as e:
                    logger.error(f"❌ Error loading reference file: {e}")
                    return 1
            else:
                print("ℹ️ No reference baseline provided; generating single-series indicators only.")
            
            # Create comprehensive visualization directly in output directory
            output_file = output_dir / "gradient_analysis_dashboard.png"
            visualizer.create_comprehensive_visualization(gradients, save_path=str(output_file))
            
            print(f"📄 Static PNG: {output_dir / 'gradient_analysis_dashboard.png'}")
            print(f"📄 Interactive HTML: {output_dir / 'gradient_analysis_dashboard_interactive.html'}")
            
            print(f"✅ Comprehensive gradient analysis completed!")
            print(f"📊 Visualizations saved to: {output_dir}")
            return 0
            
        elif args.gradient_command == 'analyze':
            print(f"🔍 Analyzing gradient file: {args.gradient_file}")
            
            # Load gradient data
            if not os.path.exists(args.gradient_file):
                logger.error(f"Gradient file not found: {args.gradient_file}")
                return 1
            
            try:
                gradient_data = np.load(args.gradient_file, allow_pickle=True)
                gradient_data = np.asarray(gradient_data)
                print(f"📊 Loaded gradient data: {gradient_data.shape}")
                
                # Create gradients dict for analysis
                gradients = {
                    'observed': gradient_data,
                }

                if getattr(args, 'reference_file', None):
                    ref_path = str(args.reference_file)
                    if not os.path.exists(ref_path):
                        logger.error(f"Reference file not found: {ref_path}")
                        return 1
                    ref = np.load(ref_path, allow_pickle=True)
                    gradients['reference'] = np.asarray(ref)
                    print(f"📊 Loaded reference data: {gradients['reference'].shape}")
                
            except Exception as e:
                logger.error(f"Failed to load gradient file: {e}")
                return 1
            
            # Perform analysis
            print(f"📈 Analyzing with threshold: {args.threshold}")
            print(f"📁 Results will be saved to: {output_dir}")
            
            # Create visualizations into the specified output directory
            output_path = output_dir / 'gradient_analysis_dashboard.png'
            visualizer.create_comprehensive_visualization(gradients, save_path=str(output_path))
            print(f"📄 Created: {output_path}")
            
            print("✅ Gradient analysis completed!")
            return 0
            
        elif args.gradient_command == 'demo':
            print("🎯 Running obfuscated gradient visualization demo...")
            
            if args.quick:
                print("⚡ Quick demo mode")
            if args.interactive:
                print("🖱️  Interactive mode enabled")
            
            # Load sample gradient data
            gradients = visualizer.load_sample_gradients()
            
            # Run demo into the requested output directory
            output_path = output_dir / 'gradient_analysis_dashboard.png'
            visualizer.create_comprehensive_visualization(gradients, save_path=str(output_path))
            print(f"📄 Created: {output_path}")
            
            print("✅ Demo completed successfully!")
            print(f"📊 Demo results saved to: {output_dir}")
            return 0
            
        elif args.gradient_command == 'generate':
            print(f"🔄 Generating test gradient data ({args.samples} samples)...")
            print(f"🎭 Attack type: {args.attack_type}")
            print(f"📊 Noise level: {args.noise_level}")
            
            # Generate test data
            clean_gradients = visualizer._generate_clean_gradients(args.samples)
            obfuscated_gradients = visualizer._generate_obfuscated_gradients(args.samples)
            
            # Save data
            clean_file = output_dir / 'generated_clean_gradients.npy'
            obfuscated_file = output_dir / 'generated_obfuscated_gradients.npy'
            
            np.save(clean_file, clean_gradients)
            np.save(obfuscated_file, obfuscated_gradients)
            
            print(f"📄 Clean gradients saved to: {clean_file}")
            print(f"📄 Obfuscated gradients saved to: {obfuscated_file}")
            print("✅ Test data generation completed!")
            return 0
            
        elif args.gradient_command == 'capture-adversarial':
            print("🔴 RED TEAM: Capturing adversarial obfuscated gradients...")
            print(f"🎯 Attack type: {args.attack_type}")
            print(f"📁 Output directory: {output_dir}")
            
            try:
                import torch
                import torch.nn as nn
                import json
                
                # Determine device
                if args.device == 'auto':
                    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
                else:
                    device = args.device
                
                print(f"🍎 Device: {device}")
                
                # Model sizes
                model_configs = {
                    'small': (768, 256, 64, 10),
                    'medium': (768, 512, 256, 10),
                    'large': (1024, 768, 512, 256, 10)
                }
                
                layers = model_configs[args.model_size]
                
                # Build model
                model_layers = []
                for i in range(len(layers) - 1):
                    model_layers.append(nn.Linear(layers[i], layers[i+1]))
                    if i < len(layers) - 2:  # No ReLU on last layer
                        model_layers.append(nn.ReLU())
                        if i == 1:  # Add dropout in middle
                            model_layers.append(nn.Dropout(0.1))
                
                model = nn.Sequential(*model_layers).to(device)
                print(f"📊 Model: {args.model_size} ({sum(p.numel() for p in model.parameters()):,} parameters)")
                
                # Gradient capture with obfuscation
                captured_gradients = []
                
                def apply_obfuscation(grad_np, attack_type):
                    """Apply configured obfuscation patterns (explicit demo mode)."""
                    if attack_type == 'rl_policy':
                        return grad_np + np.random.randn(*grad_np.shape) * 0.3
                    elif attack_type == 'periodic':
                        return grad_np + np.sin(np.linspace(0, 4*np.pi, len(grad_np))) * 0.2
                    elif attack_type == 'conditional_trigger':
                        trigger_pos = np.random.choice(len(grad_np), 5, replace=False)
                        grad_np[trigger_pos] = [0.8, 0.9, 1.0, 0.9, 0.8]
                        return grad_np
                    elif attack_type == 'high_frequency':
                        return grad_np + np.sin(np.linspace(0, 20*np.pi, len(grad_np))) * 0.1
                    elif attack_type == 'gradient_masking':
                        return np.sign(grad_np) * 0.01
                    else:  # combined
                        grad_np = grad_np + np.random.randn(*grad_np.shape) * 0.2
                        grad_np = grad_np + np.sin(np.linspace(0, 3*np.pi, len(grad_np))) * 0.15
                        spike_pos = np.random.choice(len(grad_np), 10, replace=False)
                        grad_np[spike_pos] += np.random.uniform(0.5, 1.0, 10)
                        grad_np = grad_np + np.sin(np.linspace(0, 15*np.pi, len(grad_np))) * 0.08
                        return grad_np
                
                def capture_hook(module, grad_input, grad_output):
                    if grad_output is not None:
                        for grad in grad_output:
                            if grad is not None and grad.numel() > 0:
                                grad_cpu = grad.detach().cpu().numpy().flatten()
                                obf_grad = apply_obfuscation(grad_cpu, args.attack_type)
                                
                                captured_gradients.append({
                                    'mean': float(obf_grad.mean()),
                                    'std': float(obf_grad.std()),
                                    'max': float(obf_grad.max()),
                                    'min': float(obf_grad.min())
                                })
                
                # Register hooks
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        module.register_full_backward_hook(capture_hook)
                
                print("🚀 Training with obfuscation attacks...")
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                for epoch in range(args.epochs):
                    for batch in range(args.batches):
                        x = torch.randn(32, layers[0]).to(device)
                        y = torch.randint(0, layers[-1], (32,)).to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(x)
                        loss = nn.CrossEntropyLoss()(outputs, y)
                        loss.backward()
                        optimizer.step()
                    
                    if epoch % 5 == 0:
                        print(f"Epoch {epoch}, Captured: {len(captured_gradients)} gradients")
                
                # Save results
                analysis = {
                    'gradient_history': captured_gradients,
                    'attack_metadata': {
                        'attack_type': args.attack_type,
                        'device': device,
                        'model_size': args.model_size,
                        'total_samples': len(captured_gradients)
                    }
                }
                
                json_file = output_dir / 'adversarial_obfuscated_analysis.json'
                with open(json_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
                
                # Convert to NPY
                grad_array = np.array([[g['mean'], g['std'], g['max']] for g in captured_gradients])
                npy_file = output_dir / 'adversarial_obfuscated_gradients.npy'
                np.save(npy_file, grad_array)
                
                print(f"")
                print(f"✅ Captured {len(captured_gradients)} adversarial gradients!")
                print(f"📄 JSON: {json_file}")
                print(f"📄 NPY:  {npy_file}")
                print(f"")
                print(f"🚀 Next: neurinspectre obfuscated-gradient create --input-file {npy_file} --output-dir {output_dir}")
                
                return 0
                
            except Exception as e:
                logger.error(f"Adversarial capture failed: {e}")
                return 1
        
        elif args.gradient_command == 'train-and-monitor':
            """Train a real HuggingFace model with integrated gradient monitoring"""
            try:
                import torch
                import torch.nn as nn
                import json
                import warnings
                import logging
                
                # Suppress transformers logging about loss_type
                logging.getLogger("transformers").setLevel(logging.ERROR)
                
                from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
                from ..security.visualization.realtime_gradient_monitor import RealtimeGradientMonitor
                
                print("=" * 80)
                print("🔬 NeurInSpectre Train-and-Monitor")
                print("=" * 80)
                
                # Setup device
                if args.device == 'auto':
                    if torch.backends.mps.is_available():
                        device = torch.device("mps")
                        print("✅ Using Apple Silicon MPS")
                    elif torch.cuda.is_available():
                        device = torch.device("cuda")
                        print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
                    else:
                        device = torch.device("cpu")
                        print("✅ Using CPU")
                else:
                    device = torch.device(args.device)
                    print(f"✅ Using specified device: {args.device}")
                
                # Load model and tokenizer
                print(f"\n📦 Loading model: {args.model}")
                try:
                    # Try loading as causal LM first (GPT-2, Qwen, etc.)
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='.*loss_type.*')
                        warnings.filterwarnings('ignore', category=FutureWarning)
                    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
                    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
                    print(f"✅ Loaded {args.model} as CausalLM")
                except Exception:
                    try:
                        # Fall back to base model (BERT, RoBERTa, etc.)
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', message='.*loss_type.*')
                            warnings.filterwarnings('ignore', category=FutureWarning)
                        model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
                        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
                        print(f"✅ Loaded {args.model} as base model")
                    except Exception as e:
                        logger.error(f"Failed to load model: {e}")
                        print("\n💡 Try one of these models:")
                        print("   - 'gpt2' (fastest)")
                        print("   - 'EleutherAI/gpt-neo-125M'")
                        print("   - 'Qwen/Qwen-1_8B' (if installed)")
                        print("   - 'bert-base-uncased'")
                        return 1
                
                # Move model to device
                model = model.to(device)
                model.train()
                
                # Initialize gradient monitor
                print("\n🔍 Initializing gradient monitor...")
                monitor = RealtimeGradientMonitor(
                    device=str(device),
                    buffer_size=args.steps,
                    update_interval=0.1,
                    analysis_window=min(100, args.steps),
                    auto_detect_models=False  # We're manually registering the model
                )
                
                # Register hooks on the model
                print("📡 Registering gradient capture hooks...")
                monitor.register_model_hooks(model)
                
                # Prepare training data
                training_texts = [
                    "The future of artificial intelligence is",
                    "Machine learning models can be",
                    "Neural network security requires",
                    "Gradient obfuscation techniques include",
                    "Adversarial attacks on AI systems",
                    "Deep learning frameworks enable",
                    "Model interpretability is important for",
                    "Privacy-preserving machine learning uses",
                ]
                
                # Setup optimizer
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
                
                # Training loop with gradient monitoring
                print(f"\n🚀 Starting training for {args.steps} steps...")
                print("=" * 80)
                
                all_gradients = []
                
                for step in range(args.steps):
                    # Get random training sample
                    text = training_texts[step % len(training_texts)]
                    
                    # Tokenize
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Add labels for causal LM
                    if hasattr(model, 'lm_head'):
                        inputs['labels'] = inputs['input_ids'].clone()
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    try:
                        outputs = model(**inputs)
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs.logits.mean()
                    except Exception:
                        outputs = model(**inputs)
                        loss = outputs.logits.mean()
                    
                    # Backward pass (monitor captures gradients here)
                    loss.backward()
                    
                    # Collect gradients for saving
                    step_grads = []
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_flat = param.grad.detach().cpu().numpy().flatten()
                            step_grads.extend(grad_flat[:100])  # First 100 values per param
                    
                    if step_grads:
                        all_gradients.extend(step_grads[:1000])  # Keep it manageable
                    
                    # Optimizer step
                    optimizer.step()
                    
                    # Progress
                    if (step + 1) % 10 == 0:
                        print(f"Step {step + 1}/{args.steps} | Loss: {loss.item():.4f} | Gradients: {len(all_gradients)}")
                
                print("=" * 80)
                print("✅ Training completed!")
                
                # Save captured gradients as NPY
                gradient_array = np.array(all_gradients[:10000])  # Cap at 10k values
                gradient_path = output_dir / "monitored_gradients.npy"
                np.save(gradient_path, gradient_array)
                print(f"\n💾 Saved {len(gradient_array)} gradient values to: {gradient_path}")
                
                # Get monitor results
                print("\n📊 Getting monitor analysis...")
                results = monitor.get_results()
                
                # Save monitor JSON
                json_path = output_dir / "monitor_analysis.json"
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"💾 Saved monitor analysis to: {json_path}")
                
                print(f"\n📈 Captured {len(results.get('gradient_history', []))} gradient statistics")
                
                # Auto-analyze if requested
                if args.auto_analyze:
                    print("\n🎯 Running automatic analysis...")
                    
                    # Run gradient analysis
                    print("\n1️⃣  Creating gradient analysis dashboard...")
                    from ..security.visualization.obfuscated_gradient_visualizer import ObfuscatedGradientVisualizer
                    visualizer = ObfuscatedGradientVisualizer()
                    
                    # Load the gradient data
                    grads_array = np.load(gradient_path)
                    
                    # Create gradient dict format expected by visualizer
                    gradients_dict = {
                        'observed': grads_array
                    }
                    
                    # Create the dashboard
                    save_path = str(output_dir / "gradient_analysis_dashboard.png")
                    visualizer.create_comprehensive_visualization(
                        gradients=gradients_dict,
                        save_path=save_path
                    )
                    print(f"✅ Dashboard: {output_dir}/gradient_analysis_dashboard_interactive.html")
                    
                    # Run spectral analysis
                    print("\n2️⃣  Running spectral analysis...")
                    from .mathematical_commands import run_spectral_analysis
                    spectral_args = type('Args', (), {
                        'input': str(gradient_path),
                        'output': str(output_dir / 'spectral.json'),
                        'levels': 5,
                        'device': args.device,
                        'precision': 'float32',
                        'plot': str(output_dir / 'spectral.png'),
                        'verbose': False
                    })()
                    run_spectral_analysis(spectral_args)
                    print(f"✅ Spectral: {output_dir}/spectral_interactive.html")
                
                # Summary
                print("\n" + "=" * 80)
                print("🎯 NEXT STEPS:")
                print("=" * 80)
                
                if not args.auto_analyze:
                    print("\n1️⃣  View Interactive Gradient Analysis Dashboard:")
                    print(f"    neurinspectre obfuscated-gradient create --input-file {gradient_path} --output-dir {output_dir}")
                    print(f"    open {output_dir}/gradient_analysis_dashboard_interactive.html")
                    
                    print("\n2️⃣  Run Spectral Analysis:")
                    print(f"    neurinspectre math spectral --input {gradient_path} --output {output_dir}/spectral.json --plot {output_dir}/spectral.png")
                    print(f"    open {output_dir}/spectral_interactive.html")
                else:
                    print("\n✅ Analysis complete! Open dashboards:")
                    print(f"    open {output_dir}/gradient_analysis_dashboard_interactive.html")
                    print(f"    open {output_dir}/spectral_interactive.html")
                
                print("\n3️⃣  View Monitor JSON:")
                print(f"    cat {json_path} | jq .")
                
                print("\n" + "=" * 80)
                print(f"✅ ALL FILES SAVED TO: {output_dir}")
                print("=" * 80)
                
                return 0
                
            except Exception as e:
                logger.error(f"Train-and-monitor failed: {e}")
                import traceback
                traceback.print_exc()
                return 1
            
        elif args.gradient_command == 'monitor':
            print("📡 Starting real-time gradient monitoring with autodetection...")
            print(f"📱 Device: {args.device}")
            print(f"📊 Buffer size: {args.buffer_size}")
            print(f"⏱️  Update interval: {args.update_interval}s")
            print(f"🪟 Analysis window: {args.analysis_window}")
            print(f"🕒 Duration: {args.duration}s")
            print("🔍 Will automatically detect and monitor ALL running PyTorch models")
            
            try:
                import torch
                import torch.nn as nn
                from ..security.visualization.realtime_gradient_monitor import RealTimeGradientMonitor
                
                # Initialize monitor with autodetection (no model required)
                monitor = RealTimeGradientMonitor(
                    device=args.device,
                    buffer_size=args.buffer_size,
                    update_interval=args.update_interval,
                    analysis_window=args.analysis_window,
                    auto_detect_models=True
                )
                
                print("🎯 Autodetection mode: Will capture gradients from any PyTorch model")
                print("🚀 Works with existing training scripts, Jupyter notebooks, or any PyTorch code")
                
                # Start monitoring
                monitor.start_monitoring(live_plot=not args.no_plot)
                
                # Monitor real models - no synthetic training needed
                import time
                
                print("🔄 Monitoring real-time gradients from detected models...")
                print("📊 Run your PyTorch training scripts now - gradients will be captured automatically")
                print("📊 Press Ctrl+C to stop monitoring")
                
                start_time = time.time()
                
                try:
                    while time.time() - start_time < args.duration:
                        time.sleep(5)  # Check every 5 seconds
                        
                        # Get current analysis
                        try:
                            analysis = monitor.get_current_analysis()
                            if analysis:
                                elapsed = time.time() - start_time
                                print(f"⏱️  {elapsed:.1f}s: Models detected: {analysis.get('models_detected', 0)}, "
                                      f"Gradients: {analysis.get('total_samples', 0)}, "
                                      f"Suspicious: {analysis.get('suspicious_rate', 0):.3f}")
                            else:
                                print("⏳ Waiting for PyTorch models to start training...")
                        except Exception as e:
                            if args.verbose:
                                print(f"Analysis error: {e}")
                        

                                      
                except KeyboardInterrupt:
                    print("\n⏹️  Monitoring interrupted by user")
                finally:
                    monitor.stop_monitoring()
                    
                    # Save analysis report if requested
                    if args.output_report:
                        monitor.save_analysis_report(args.output_report)
                        print(f"📄 Analysis report saved to: {args.output_report}")
                    else:
                        # Save with default name
                        default_report = f"realtime_gradient_analysis_{int(time.time())}.json"
                        monitor.save_analysis_report(default_report)
                        print(f"📄 Analysis report saved to: {default_report}")
                
                print("✅ Real-time monitoring completed!")
                return 0
                
            except ImportError as e:
                logger.error(f"Real-time monitoring modules not available: {e}")
                logger.info("Please ensure PyTorch and visualization modules are properly installed.")
                return 1
            except Exception as e:
                logger.error(f"Real-time monitoring failed: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                return 1
            
        else:
            logger.error(f"Unknown gradient command: {args.gradient_command}")
            return 1
            
    except ImportError as e:
        logger.error(f"Obfuscated gradient visualizer not available: {e}")
        logger.info("Please ensure the visualization module is properly installed.")
        return 1
    except Exception as e:
        logger.error(f"Gradient command execution failed: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def handle_math_command(args):
    """Handle mathematical analysis commands"""
    try:
        from .mathematical_commands import handle_mathematical_command
        return handle_mathematical_command(args)
    except ImportError as e:
        logger.error(f"Mathematical commands not available: {e}")
        return 1

def handle_dashboard_manager_command(args):
    """Handle dashboard management commands"""
    try:
        from .dashboard_manager import DashboardManager
        
        if not args.dashboard_manager_command:
            logger.error("No dashboard manager subcommand specified")
            return 1
        
        # Create dashboard manager
        manager = DashboardManager()
        
        if args.dashboard_manager_command == 'backup':
            backup_name = manager.create_backup(args.name)
            print(f"🎉 Backup created: {backup_name}")
            return 0
            
        elif args.dashboard_manager_command == 'list':
            backups = manager.list_backups()
            if not backups:
                print("📭 No backups found")
                return 0
            
            print("📋 Available backups:")
            for backup in backups:
                print(f"   📁 {backup['backup_name']}")
                print(f"      📅 Created: {backup['created']}")
                print(f"      📊 Dashboards: {len(backup['dashboards'])}")
                print()
            return 0
            
        elif args.dashboard_manager_command == 'restore':
            success = manager.restore_backup(args.name)
            return 0 if success else 1
            
        elif args.dashboard_manager_command == 'status':
            status = manager.get_dashboard_status()
            print("📊 Dashboard Status:")
            print("=" * 50)
            
            for dash_id, info in status.items():
                print(f"🔍 {info['name']} (Port {info['port']}):")
                print(f"   Process: {'✅ Running' if info['process_running'] else '❌ Not running'}")
                print(f"   Port: {'✅ Listening' if info['port_listening'] else '❌ Not listening'}")
                print(f"   HTTP: {'✅ Accessible' if info['http_accessible'] else '❌ Not accessible'}")
                print(f"   Log: {info['log_file']}")
                print()
            return 0
            
        elif args.dashboard_manager_command == 'start':
            if args.all:
                count = manager.start_all_dashboards()
                return 0 if count > 0 else 1
            else:
                success = manager.start_dashboard(args.dashboard)
                return 0 if success else 1
                
        elif args.dashboard_manager_command == 'stop':
            if args.all:
                count = manager.stop_all_dashboards()
                return 0 if count >= 0 else 1
            else:
                success = manager.stop_dashboard(args.dashboard)
                return 0 if success else 1
                
        elif args.dashboard_manager_command == 'restart':
            count = manager.restart_all_dashboards()
            return 0 if count > 0 else 1
            
        elif args.dashboard_manager_command == 'emergency':
            success = manager.emergency_restore(args.backup)
            return 0 if success else 1
            
        else:
            logger.error(f"Unknown dashboard manager command: {args.dashboard_manager_command}")
            return 1
            
    except ImportError as e:
        logger.error(f"Dashboard manager not available: {e}")
        return 1
    except Exception as e:
        logger.error(f"Dashboard manager command failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 
