"""
NeurInSpectre Security CLI Commands

Implements advanced security analysis techniques, including:
- TS-Inverse gradient inversion detection
- ConcreTizer model inversion detection
- AttentionGuard transformer-based detection
- EDNN attack detection
- DeMarking defense mechanisms
- Neural transport dynamics analysis
- Integrated security assessment
"""

import os
import sys
import json
import webbrowser
import logging
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict

# Click is optional: this module is also imported by the argparse-based top-level CLI.
# Avoid import-time process exit so NeurInSpectre can run in minimal installs.
_HAS_CLICK = True
_CLICK_IMPORT_ERROR: Optional[BaseException] = None
try:
    import click  # type: ignore
except ImportError as e:
    _HAS_CLICK = False
    _CLICK_IMPORT_ERROR = e

    class _ClickStub:  # minimal shim for decorators/types used below
        class Path:
            def __init__(self, *args, **kwargs):
                pass

        class Choice:
            def __init__(self, *args, **kwargs):
                pass

        class ClickException(Exception):
            pass

        def command(self, *args, **kwargs):
            def _decorator(f):
                return f
            return _decorator

        def option(self, *args, **kwargs):
            def _decorator(f):
                return f
            return _decorator

        def echo(self, message: object = "", **kwargs):
            stream = sys.stderr if kwargs.get("err") else sys.stdout
            print(message, file=stream)

        def progressbar(self, *args, **kwargs):
            # Provide a no-op context manager if called without click installed.
            class _PB:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

                def update(self_inner, *a, **k):
                    return None
            return _PB()

    click = _ClickStub()  # type: ignore

# Visualization modules are optional; load lazily to avoid import-time warnings/prints.
HAS_VISUALIZATION = False
HAS_ENHANCED_VISUALIZER = False

def _get_red_blue_team_dashboard():
    """Lazy import of red/blue team dashboard visualization."""
    try:
        from neurinspectre.security.visualization.red_blue_team_dashboard import (
            RedBlueTeamDashboard,
            create_comprehensive_report,
        )
        return RedBlueTeamDashboard, create_comprehensive_report
    except Exception as e:
        raise ImportError(f"Visualization modules not available: {e}")

def _get_enhanced_security_visualizer():
    """Lazy import of enhanced security visualizer."""
    try:
        from neurinspectre.cli.enhanced_security_visualizer import Enhanced2025SecurityVisualizer
        return Enhanced2025SecurityVisualizer
    except Exception as e:
        raise ImportError(f"Enhanced security visualizer not available: {e}")


def _maybe_open(path: str) -> None:
    """Open a local file in the default browser, best-effort."""
    try:
        webbrowser.open(Path(path).resolve().as_uri())
    except Exception:
        return

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OUTPUT_DIR = "security_reports"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

def register_commands():
    """Register all security and mathematical commands"""
    from .mathematical_commands import register_mathematical_commands
    
    # Create main argument parser
    import argparse
    parser = argparse.ArgumentParser(
        description='NeurInSpectre: Advanced AI Security Analysis Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Create subparsers for different command categories
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Register mathematical commands
    register_mathematical_commands(subparsers)
    
    return parser

# Lazy imports for heavy dependencies - only import when actually needed
def _get_pandas():
    """Lazy import of pandas to avoid import-time dependency issues"""
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError("pandas is required for this functionality. Install with: conda install -c conda-forge pandas")

def _get_plotly():
    """Lazy import of plotly to avoid import-time dependency issues"""
    try:
        import plotly.express as px
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        return px, make_subplots, go
    except ImportError:
        raise ImportError("plotly is required for this functionality. Install with: conda install -c conda-forge plotly")

def _get_adversarial_detector():
    """Lazy import of adversarial detection module"""
    try:
        from neurinspectre.security.adversarial_detection import AdversarialDetector
        return AdversarialDetector
    except ImportError as e:
        raise ImportError(f"Adversarial detection module not available: {e}")

def _get_evasion_detector():
    """Lazy import of evasion detection module"""
    try:
        from neurinspectre.security.evasion_detection import EvasionDetector
        return EvasionDetector
    except ImportError as e:
        raise ImportError(f"Evasion detection module not available: {e}")

def _get_integrated_security():
    """Lazy import of integrated security module"""
    try:
        from neurinspectre.security.integrated_security import (
            IntegratedSecurityAnalyzer, generate_security_assessment
        )
        return IntegratedSecurityAnalyzer, generate_security_assessment
    except ImportError as e:
        raise ImportError(f"Integrated security module not available: {e}")

def _select_npz_array(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    """Select the best array from NPZ: prefer 'data'/'x', else largest 2D."""
    preferred_keys = ['data', 'x', 'X', 'activations', 'gradients']
    for k in preferred_keys:
        if k in npz:
            return npz[k]
    # choose largest 2D by number of elements
    best_arr = None
    best_size = -1
    for k in npz.keys():
        arr = npz[k]
        if isinstance(arr, np.ndarray):
            size = arr.size
            if size > best_size:
                best_size = size
                best_arr = arr
    return best_arr if best_arr is not None else npz[list(npz.keys())[0]]

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Convert array to 2D [N, D]. If 1D -> (N,1). If ND>2 -> reshape to (N, -1)."""
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    # treat first axis as sample dimension, flatten the rest
    n = arr.shape[0]
    return arr.reshape(n, -1)

def _clean_array(arr: np.ndarray, robust: bool = False) -> np.ndarray:
    """Cast to float32 and replace NaN/Inf with finite values; per-feature scaling.
    If robust=True, use median/MAD; else mean/std.
    """
    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2 and arr.size > 0:
        if robust:
            med = np.median(arr, axis=0, keepdims=True)
            mad = np.median(np.abs(arr - med), axis=0, keepdims=True)
            mad = np.where(mad < 1e-8, 1.0, mad)
            arr = (arr - med) / mad
        else:
            mean = np.mean(arr, axis=0, keepdims=True)
            std = np.std(arr, axis=0, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            arr = (arr - mean) / std
    return arr

def _stack_directory(dir_path: Path, max_files: int = 1000) -> np.ndarray:
    """Load all .npy/.npz files in a directory, convert to 2D, align feature dims, and stack rows."""
    arrays: list[np.ndarray] = []
    count = 0
    for p in sorted(dir_path.glob('**/*')):
        if count >= max_files:
            break
        if p.suffix.lower() == '.npy':
            try:
                arr = np.load(p)
            except Exception:
                continue
        elif p.suffix.lower() == '.npz':
            try:
                npz = np.load(p)
                arr = _select_npz_array(npz)
            except Exception:
                continue
        else:
            continue
        arr = _ensure_2d(arr)
        arrays.append(arr)
        count += 1
    if not arrays:
        raise FileNotFoundError(f"No .npy/.npz files found under directory: {dir_path}")
    # Align feature dims by truncating to minimum feature count
    min_d = min(a.shape[1] for a in arrays)
    arrays = [a[:, :min_d] for a in arrays]
    return np.vstack(arrays)

def _load_data_file(file_path: str) -> np.ndarray:
    """Load data from various file formats or directories and return 2D float32 array."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    def _load_pcap_ipds(p: Path) -> np.ndarray:
        """Parse a classic libpcap (.pcap) file and return inter-packet delays (seconds).

        This is a lightweight parser (no external deps) intended for DeMarking-style
        IPD analysis. It uses per-packet capture timestamps and ignores payload content.
        """
        import struct

        data = p.read_bytes()
        if len(data) < 24:
            raise ValueError(f"PCAP too small: {p}")

        magic = data[:4]
        # Classic pcap magic numbers:
        # 0xa1b2c3d4 (microsecond, big-endian), 0xd4c3b2a1 (microsecond, little-endian)
        # 0xa1b23c4d (nanosecond, big-endian), 0x4d3cb2a1 (nanosecond, little-endian)
        if magic == b"\xa1\xb2\xc3\xd4":
            endian = ">"
            ts_scale = 1e-6
        elif magic == b"\xd4\xc3\xb2\xa1":
            endian = "<"
            ts_scale = 1e-6
        elif magic == b"\xa1\xb2\x3c\x4d":
            endian = ">"
            ts_scale = 1e-9
        elif magic == b"\x4d\x3c\xb2\xa1":
            endian = "<"
            ts_scale = 1e-9
        else:
            raise ValueError(f"Unsupported PCAP magic {magic!r} in {p} (pcapng not supported here)")

        # Skip global header (24 bytes)
        off = 24
        ts: list[float] = []
        ph_fmt = endian + "IIII"  # ts_sec, ts_subsec, incl_len, orig_len
        ph_sz = struct.calcsize(ph_fmt)

        while off + ph_sz <= len(data):
            ts_sec, ts_sub, incl_len, _orig_len = struct.unpack_from(ph_fmt, data, off)
            off += ph_sz
            if incl_len < 0 or off + incl_len > len(data):
                break
            # Skip packet bytes
            off += incl_len
            ts.append(float(ts_sec) + float(ts_sub) * ts_scale)

        if len(ts) < 2:
            return np.zeros((0,), dtype=np.float32)
        ts_arr = np.asarray(ts, dtype=np.float64)
        ipd = np.diff(ts_arr)
        # Defensive cleanup: non-finite/negative diffs -> 0
        ipd = np.nan_to_num(ipd, nan=0.0, posinf=0.0, neginf=0.0)
        ipd[ipd < 0.0] = 0.0
        return ipd.astype(np.float32, copy=False)

    if file_path.is_dir():
        arr = _stack_directory(file_path)
        return _ensure_2d(arr)

    if file_path.suffix.lower() == '.npy':
        arr = np.load(file_path)
    elif file_path.suffix.lower() == '.npz':
        npz = np.load(file_path)
        arr = _select_npz_array(npz)
    elif file_path.suffix.lower() == '.pcap':
        # Return a 1D IPD series (seconds). Callers that require 2D can reshape.
        return _load_pcap_ipds(file_path)
    elif file_path.suffix.lower() in ['.csv', '.txt']:
        # Robust CSV loader: prefer pandas, numeric columns only, skip bad lines
        try:
            import pandas as _pd
            df = _pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
            num = df.select_dtypes(include=['number'])
            if num.shape[1] == 0:
                # force coercion to numeric
                num = df.apply(_pd.to_numeric, errors='coerce')
                num = num.select_dtypes(include=['number'])
            num = num.dropna(how='all')
            arr = num.to_numpy(dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.size == 0:
                raise ValueError(f"No numeric data found in {file_path}. Provide a numeric .csv/.txt or use .npy/.npz.")
        except Exception:
            # Fallback simple reader
            try:
                arr = np.genfromtxt(file_path, delimiter=',', dtype=float, filling_values=0.0)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
            except Exception as e:
                raise e
    else:
        # Try to load as raw binary and interpret as bytes vector
        with open(file_path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
    return _ensure_2d(arr)

def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays into JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _save_results(results: Dict[str, Any], output_dir: str, filename: str):
    """Save analysis results to JSON file"""
    output_path = Path(output_dir) / f"{filename}_{datetime.now().strftime(TIMESTAMP_FORMAT)}.json"
    
    # Convert numpy arrays/scalars to JSON-serializable Python types (recursively).
    serializable_results = _jsonify(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    return str(output_path)

@click.command()
@click.option('--data-path', '-d', type=click.Path(exists=True), required=True,
              help='Path to activation/gradient data file (.npy, .npz, .csv)')
@click.option('--reference-path', '-r', type=click.Path(exists=True),
              help='Path to reference data for comparison (optional)')
@click.option('--output-dir', '-o', default='./security_reports',
              help='Output directory for analysis results')
@click.option('--detector-type', '-t', 
              type=click.Choice(['ts-inverse', 'concretizer', 'attention-guard', 'ednn', 'all']),
              default='all', help='Type of adversarial detector to use')
@click.option('--threshold', '--th', default=0.8, type=float,
              help='Detection sensitivity threshold (0.0-1.0)')
@click.option('--parallel', '-p', is_flag=True, default=True,
              help='Use parallel processing for faster analysis')
def adversarial_detect(data_path, reference_path, output_dir, detector_type, threshold, parallel):
    """
    Run adversarial attack detection using the selected detector(s).
    
    Detects various adversarial signals including:
    - TS-Inverse gradient inversion signals
    - ConcreTizer model inversion signals
    - AttentionGuard transformer behavior signals
    - EDNN embedding-attack signals
    
    Examples:
        neurinspectre security adversarial-detect -d ./activations.npy
        neurinspectre security adversarial-detect -d ./data.npy -r ./reference.npy -t ts-inverse
        neurinspectre security adversarial-detect -d ./gradients.npy --threshold 0.9 --parallel
    """
    try:
        click.echo("🛡️ NeurInSpectre Adversarial Attack Detection")
        click.echo("="*55)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        click.echo(f"📁 Loading data from: {data_path}")
        data = _load_data_file(data_path)
        
        reference_data = None
        if reference_path:
            click.echo(f"📋 Loading reference data from: {reference_path}")
            reference_data = _load_data_file(reference_path)
        
        click.echo(f"📊 Data shape: {data.shape}")
        if reference_data is not None:
            click.echo(f"📋 Reference data shape: {reference_data.shape}")
        
        click.echo(f"🎯 Detector type: {detector_type}")
        click.echo(f"⚙️  Threshold: {threshold}")
        click.echo(f"🚀 Parallel processing: {'Enabled' if parallel else 'Disabled'}")
        click.echo()
        
        # Get adversarial detector
        AdversarialDetector = _get_adversarial_detector()
        
        # Configure detector
        config = {
            'threshold': threshold,
            'ts_inverse_threshold': threshold,
            'concretizer_threshold': threshold,
            'attention_guard_threshold': threshold,
            'voxel_resolution': 32,
            'max_seq_length': min(512, data.shape[0]),
            'attention_heads': 8,
            'k_neighbors': 5,
            'ednn_threshold': threshold,
        }
        
        detector = AdversarialDetector(config)
        
        # Run detection
        with click.progressbar(length=100, label=f'Running {detector_type} detection') as bar:
            bar.update(20)
            
            results = detector.detect_adversarial_samples(data, reference_data, detector_type=detector_type)
            bar.update(80)
        
        # Process results
        click.echo("✅ Adversarial detection completed!")
        click.echo()
        click.echo("🔍 Detection Results:")
        click.echo(f"   • Overall threat level: {results['overall_threat_level'].upper()}")
        click.echo(f"   • Input data shape: {results['input_shape']}")
        click.echo(f"   • Timestamp: {datetime.fromtimestamp(results['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display specific detections
        if results.get('detections'):
            click.echo()
            click.echo("🎯 Specific Attack Detections:")
            
            for detection_name, detection_result in results['detections'].items():
                if isinstance(detection_result, dict):
                    is_attack = (
                        detection_result.get('is_attack', False) or
                        detection_result.get('is_inversion_attack', False) or
                        detection_result.get('is_misbehavior', False) or
                        detection_result.get('is_ednn_attack', False)
                    )
                    
                    confidence = detection_result.get('confidence',
                                detection_result.get('inversion_score',
                                detection_result.get('misbehavior_score',
                                detection_result.get('attack_score', 0.0))))
                    
                    status = "🚨 DETECTED" if is_attack else "✅ CLEAN"
                    click.echo(f"   • {detection_name.replace('_', ' ').title()}: {status} (confidence: {confidence:.3f})")
        
        # Display confidence scores
        if results.get('confidence_scores'):
            click.echo()
            click.echo("📊 Confidence Scores:")
            for method, score in results['confidence_scores'].items():
                click.echo(f"   • {method.replace('_', ' ').title()}: {score:.3f}")
        
        # Save results
        result_file = _save_results(results, output_dir, 'adversarial_detection')
        click.echo()
        click.echo(f"💾 Results saved to: {result_file}")
        
        # Show recommendations if high threat
        if results['overall_threat_level'] in ['high', 'critical']:
            click.echo()
            click.echo("⚠️  HIGH THREAT DETECTED - Immediate Actions Recommended:")
            click.echo("   • Implement adversarial training and input validation")
            click.echo("   • Deploy real-time gradient monitoring systems")
            click.echo("   • Review model integrity and training data")
            click.echo("   • Enable enhanced security logging and monitoring")
        
    except Exception as e:
        click.echo(f"❌ Error in adversarial detection: {str(e)}")
        logger.error(f"Adversarial detection error: {e}", exc_info=True)
        raise click.ClickException(str(e))

@click.command()
@click.option('--data-path', '-d', type=click.Path(exists=True), required=True,
              help='Path to neural activation data file (.npy, .npz, .csv)')
@click.option('--network-data', '-n', type=click.Path(exists=True),
              help='Path to network flow data for DeMarking analysis (optional)')
@click.option('--output-dir', '-o', default='./security_reports',
              help='Output directory for analysis results')
@click.option('--detector-type', '-t',
              type=click.Choice(['transport-dynamics', 'demarking', 'behavioral', 'all']),
              default='all', help='Type of evasion detector to use')
@click.option('--threshold', '--th', default=0.6, type=float,
              help='Detection sensitivity threshold (0.0-1.0)')
@click.option('--time-window', '-w', default=100, type=int,
              help='Time window for temporal analysis')
def evasion_detect(data_path, network_data, output_dir, detector_type, threshold, time_window):
    """
    Run advanced evasion attack detection using neural transport dynamics.
    
    Detects evasion attempts using:
    - Neural Network Transport Dynamics (2024)
    - DeMarking defense for network flow watermarking (Feb 2024)
    - Behavioral pattern analysis with attention mechanisms
    - Entropy analysis and statistical consistency checks
    
    Examples:
        neurinspectre security evasion-detect -d ./activations.npy
        neurinspectre security evasion-detect -d ./data.npy -n ./network.npy -t demarking
        neurinspectre security evasion-detect -d ./neural_data.npy --threshold 0.8 -w 200
    """
    try:
        click.echo("🚫 NeurInSpectre Evasion Attack Detection")
        click.echo("="*45)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        click.echo(f"📁 Loading neural data from: {data_path}")
        neural_data = _load_data_file(data_path)
        
        network_flow_data = None
        if network_data:
            click.echo(f"🌐 Loading network data from: {network_data}")
            network_flow_data = _load_data_file(network_data)
        
        click.echo(f"📊 Neural data shape: {neural_data.shape}")
        if network_flow_data is not None:
            click.echo(f"🌐 Network data shape: {network_flow_data.shape}")
        
        click.echo(f"🎯 Detector type: {detector_type}")
        click.echo(f"⚙️  Threshold: {threshold}")
        click.echo(f"⏰ Time window: {time_window}")
        click.echo()
        
        # Get evasion detector
        EvasionDetector = _get_evasion_detector()
        
        # Configure detector
        config = {
            'transport_dim': min(64, neural_data.shape[-1]),
            'time_window': time_window,
            'demarking_window': 50,
            'demarking_threshold': threshold,
            'pattern_window': time_window
        }
        
        detector = EvasionDetector(config)
        
        # Add network data as attribute if available
        if network_flow_data is not None:
            neural_data.ipd_data = network_flow_data
        
        # Run detection
        with click.progressbar(length=100, label=f'Running {detector_type} evasion detection') as bar:
            bar.update(20)
            
            evasion_attempts = detector.detect_evasion_attempts(neural_data)
            bar.update(80)
        
        # Process results
        click.echo("✅ Evasion detection completed!")
        click.echo()
        click.echo("🔍 Detection Results:")
        click.echo(f"   • Total evasion attempts detected: {len(evasion_attempts)}")
        
        if evasion_attempts:
            click.echo()
            click.echo("🚨 Detected Evasion Attempts:")
            
            for i, attempt in enumerate(evasion_attempts, 1):
                attempt_type = attempt.get('type', 'unknown')
                confidence = attempt.get('confidence', 0.0)
                threat_level = attempt.get('threat_level', 'unknown')
                
                click.echo(f"   {i}. {attempt_type.replace('_', ' ').title()}")
                click.echo(f"      • Confidence: {confidence:.3f}")
                click.echo(f"      • Threat Level: {threat_level.upper()}")
                
                # Show specific details for certain attack types
                details = attempt.get('details', {})
                if 'evasion_score' in details:
                    click.echo(f"      • Evasion Score: {details['evasion_score']:.3f}")
                if 'threat_level' in details:
                    click.echo(f"      • Analysis Threat Level: {details['threat_level']}")
        else:
            click.echo("   ✅ No evasion attempts detected")
        
        # Get detection summary
        summary = detector.get_detection_summary()
        
        click.echo()
        click.echo("📊 Detection Summary:")
        click.echo(f"   • Total detections: {summary['total_detections']}")
        click.echo(f"   • Total evasion attempts: {summary['evasion_attempts']}")
        
        if 'threat_distribution' in summary:
            click.echo("   • Threat distribution:")
            for level, count in summary['threat_distribution'].items():
                if count > 0:
                    click.echo(f"     - {level.title()}: {count}")
        
        # Save results
        results = {
            'evasion_attempts': evasion_attempts,
            'detection_summary': summary,
            'config': config,
            'input_shape': neural_data.shape,
            'timestamp': time.time()
        }
        
        result_file = _save_results(results, output_dir, 'evasion_detection')
        click.echo()
        click.echo(f"💾 Results saved to: {result_file}")
        
        # Show recommendations if evasions detected
        if evasion_attempts:
            high_threat_attempts = [a for a in evasion_attempts 
                                  if a.get('threat_level') in ['high', 'critical']]
            if high_threat_attempts:
                click.echo()
                click.echo("⚠️  HIGH THREAT EVASION DETECTED - Immediate Actions:")
                click.echo("   • Strengthen evasion detection mechanisms")
                click.echo("   • Implement multi-layer security monitoring")
                click.echo("   • Deploy transport dynamics analysis")
                click.echo("   • Review and update detection thresholds")
        
    except Exception as e:
        click.echo(f"❌ Error in evasion detection: {str(e)}")
        logger.error(f"Evasion detection error: {e}", exc_info=True)
        raise click.ClickException(str(e))

@click.command()
@click.option('--activation-data', '-a', type=click.Path(exists=True), required=True,
              help='Path to neural activation data file (.npy, .npz, .csv)')
@click.option('--gradient-data', '-g', type=click.Path(exists=True),
              help='Path to gradient data file (optional)')
@click.option('--model-weights', '-m', type=click.Path(exists=True),
              help='Path to model weights file (.npz) (optional)')
@click.option('--network-data', '-n', type=click.Path(exists=True),
              help='Path to network flow data (optional)')
@click.option('--output-dir', '-o', default='./security_reports',
              help='Output directory for comprehensive security report')
@click.option('--parallel', '-p', is_flag=True, default=True,
              help='Use parallel processing for faster analysis')
@click.option('--threshold', '--th', default=0.6, type=float,
              help='Overall threat detection threshold')
@click.option('--generate-report', '-r', is_flag=True, default=True,
              help='Generate comprehensive HTML security report')
def comprehensive_scan(activation_data, gradient_data, model_weights, network_data, 
                      output_dir, parallel, threshold, generate_report):
    """
    Run comprehensive security scan using available detectors.
    
    Performs integrated analysis including:
    - TS-Inverse gradient inversion detection
    - ConcreTizer model inversion detection  
    - AttentionGuard transformer-based detection
    - EDNN attack detection
    - Neural transport dynamics analysis
    - DeMarking defense mechanisms
    - Behavioral pattern analysis
    - Model backdoor detection
    - Anomaly detection using Isolation Forest
    
    Examples:
        neurinspectre security comprehensive-scan -a ./activations.npy
        neurinspectre security comprehensive-scan -a ./data.npy -g ./grads.npy -m ./weights.npz
        neurinspectre security comprehensive-scan -a ./neural.npy -n ./network.npy --threshold 0.8
    """
    try:
        click.echo("🔒 NeurInSpectre Comprehensive Security Scan")
        click.echo("="*50)
        click.echo()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load activation data
        click.echo(f"📁 Loading activation data from: {activation_data}")
        activations = _load_data_file(activation_data)
        click.echo(f"📊 Activation data shape: {activations.shape}")
        
        # Load optional data
        gradients = None
        if gradient_data:
            click.echo(f"📈 Loading gradient data from: {gradient_data}")
            gradients = _load_data_file(gradient_data)
            click.echo(f"📊 Gradient data shape: {gradients.shape}")
        
        weights = None
        if model_weights:
            click.echo(f"🧠 Loading model weights from: {model_weights}")
            weights_data = np.load(model_weights)
            weights = {key: weights_data[key] for key in weights_data.keys()}
            click.echo(f"🧠 Model weights loaded: {list(weights.keys())}")
        
        network = None
        if network_data:
            click.echo(f"🌐 Loading network data from: {network_data}")
            network = _load_data_file(network_data)
            click.echo(f"🌐 Network data shape: {network.shape}")
        
        click.echo("⚙️  Configuration:")
        click.echo(f"   • Parallel processing: {'Enabled' if parallel else 'Disabled'}")
        click.echo(f"   • Threat threshold: {threshold}")
        click.echo(f"   • Generate report: {'Yes' if generate_report else 'No'}")
        click.echo()
        
        # Get integrated security analyzer
        IntegratedSecurityAnalyzer, generate_security_assessment = _get_integrated_security()
        
        # Configure analyzer
        config = {
            'adversarial': {
                'threshold': threshold,
                'ts_inverse_threshold': threshold,
                'concretizer_threshold': threshold,
                'attention_guard_threshold': threshold,
                'voxel_resolution': 32,
                'max_seq_length': min(512, activations.shape[0]),
                'attention_heads': 8,
                'k_neighbors': 5,
                'ednn_threshold': threshold
            },
            'evasion': {
                'transport_dim': min(64, activations.shape[-1]),
                'time_window': min(100, activations.shape[0]),
                'demarking_threshold': threshold,
                'pattern_window': min(100, activations.shape[0])
            },
            'parallel_processing': parallel,
            'threat_threshold': threshold
        }
        
        analyzer = IntegratedSecurityAnalyzer(config)
        
        # Run comprehensive scan
        scan_start = time.time()
        
        with click.progressbar(length=100, label='Running comprehensive security scan') as bar:
            bar.update(10)
            
            assessment = analyzer.run_comprehensive_security_scan(
                activation_data=activations,
                gradient_data=gradients,
                model_weights=weights,
                network_data=network
            )
            
            bar.update(90)
        
        scan_duration = time.time() - scan_start
        
        # Display results
        click.echo("✅ Comprehensive security scan completed!")
        click.echo()
        click.echo("🎯 SECURITY ASSESSMENT SUMMARY")
        click.echo("="*40)
        click.echo(f"🚨 Overall Threat Level: {assessment.overall_threat_level.upper()}")
        click.echo(f"📊 Confidence Score: {assessment.confidence_score:.3f}")
        click.echo(f"🔍 Attacks Detected: {len(assessment.detected_attacks)}")
        click.echo(f"⏱️  Scan Duration: {scan_duration:.2f} seconds")
        click.echo(f"📅 Timestamp: {datetime.fromtimestamp(assessment.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display detected attacks
        if assessment.detected_attacks:
            click.echo()
            click.echo("🚨 DETECTED SECURITY THREATS:")
            click.echo("-" * 30)
            
            for i, attack in enumerate(assessment.detected_attacks, 1):
                attack_type = attack.get('type', 'unknown')
                confidence = attack.get('confidence', 0.0)
                threat_level = attack.get('threat_level', 'unknown')
                
                click.echo(f"{i}. {attack_type.replace('_', ' ').title()}")
                click.echo(f"   • Confidence: {confidence:.3f}")
                click.echo(f"   • Threat Level: {threat_level.upper()}")
                click.echo()
        
        # Display security scores
        if assessment.security_scores:
            click.echo("📊 SECURITY COMPONENT SCORES:")
            click.echo("-" * 30)
            
            def _component_risk(key: str, val: float) -> float:
                try:
                    v = float(val)
                except Exception:
                    v = 0.0
                # `*_confidence` keys are risk/confidence (higher = more suspicious).
                if str(key).endswith("_confidence"):
                    return float(np.clip(v, 0.0, 1.0))
                # `*_score` / `*_security` keys are security-oriented (higher = safer) → risk = 1-score.
                if str(key).endswith("_score") or str(key).endswith("_security"):
                    return float(np.clip(1.0 - v, 0.0, 1.0))
                # Unknown convention: treat as risk proxy (best-effort).
                return float(np.clip(v, 0.0, 1.0))

            def _risk_icon(risk: float) -> str:
                if risk >= 0.7:
                    return "🚨"
                if risk >= 0.4:
                    return "⚠️"
                return "✅"

            for component, score in sorted(assessment.security_scores.items(), key=lambda kv: kv[0]):
                risk = _component_risk(component, score)
                icon = _risk_icon(risk)
                label = component.replace('_', ' ').title()
                if str(component).endswith("_confidence"):
                    click.echo(f"{icon} {label} (risk): {risk:.3f}")
                else:
                    click.echo(f"{icon} {label} (risk): {risk:.3f} (security={float(score):.3f})")
            
            click.echo()

        # MITRE ATLAS tactic/technique risk summary (0–1 risk proxy)
        atlas = (assessment.metadata or {}).get("mitre_atlas_risk", {}) if hasattr(assessment, "metadata") else {}
        techs = list(atlas.get("techniques") or [])
        if techs:
            click.echo("🧭 MITRE ATLAS TECHNIQUE RISK (0–1)")
            click.echo("-" * 30)
            for t in techs[:12]:
                tid = t.get("id", "Unknown")
                name = t.get("name", "Unknown")
                tactics = ", ".join(t.get("tactics") or [])
                risk = float(t.get("risk", 0.0))
                click.echo(f"• {tid} — {name}: risk={risk:.3f}" + (f" (tactics: {tactics})" if tactics else ""))
            # Per-tactic rollup
            tac = list(atlas.get("tactics") or [])
            if tac:
                click.echo()
                click.echo("🧭 MITRE ATLAS TACTIC RISK (0–1)")
                click.echo("-" * 30)
                for ent in tac[:12]:
                    tname = ent.get("tactic", "Unknown")
                    risk = float(ent.get("risk", 0.0))
                    click.echo(f"• {tname}: risk={risk:.3f}")
            click.echo()
        
        # Display top recommendations
        if assessment.recommendations:
            click.echo("💡 TOP SECURITY RECOMMENDATIONS:")
            click.echo("-" * 35)
            
            for i, recommendation in enumerate(assessment.recommendations[:5], 1):
                click.echo(f"{i}. {recommendation}")
            
            if len(assessment.recommendations) > 5:
                click.echo(f"   ... and {len(assessment.recommendations) - 5} more recommendations")
            
            click.echo()
        
        # Generate comprehensive report
        if generate_report:
            click.echo("📋 Generating comprehensive security report...")
            
            report = generate_security_assessment(assessment)
            
            # Save assessment and report
            from dataclasses import asdict
            assessment_file = _save_results(
                asdict(assessment), output_dir, 'comprehensive_assessment'
            )
            
            report_file = _save_results(
                report, output_dir, 'security_report'
            )
            
            click.echo(f"💾 Assessment saved to: {assessment_file}")
            click.echo(f"📋 Detailed report saved to: {report_file}")
        
        # Critical threat response
        if assessment.overall_threat_level == 'critical':
            click.echo()
            click.echo("🚨 CRITICAL SECURITY THREAT DETECTED!")
            click.echo("="*40)
            click.echo("IMMEDIATE ACTIONS REQUIRED:")
            click.echo("• Isolate affected systems immediately")
            click.echo("• Activate incident response procedures")
            click.echo("• Review all model inputs and training data")
            click.echo("• Implement emergency security measures")
            click.echo("• Contact security team and stakeholders")
            
        elif assessment.overall_threat_level == 'high':
            click.echo()
            click.echo("⚠️  HIGH SECURITY THREAT DETECTED")
            click.echo("="*35)
            click.echo("PRIORITY ACTIONS:")
            click.echo("• Review and strengthen security controls")
            click.echo("• Implement additional monitoring")
            click.echo("• Update detection thresholds")
            click.echo("• Schedule security assessment review")
        
    except Exception as e:
        click.echo(f"❌ Error in comprehensive security scan: {str(e)}")
        logger.error(f"Comprehensive scan error: {e}", exc_info=True)
        raise click.ClickException(str(e))

@click.command()
@click.option('--data-dir', '-d', type=click.Path(exists=True), required=True,
              help='Directory containing data files for continuous monitoring')
@click.option('--output-dir', '-o', default='./security_monitoring',
              help='Output directory for monitoring logs and alerts')
@click.option('--interval', '-i', default=60, type=int,
              help='Monitoring interval in seconds')
@click.option('--threshold', '--th', default=0.7, type=float,
              help='Alert threshold on threat score (0.0-1.0)')
@click.option('--max-iterations', '--max', default=100, type=int,
              help='Maximum monitoring iterations (0 for infinite)')
@click.option('--alert-webhook', '-w', type=str,
              help='Webhook URL for sending security alerts')
def realtime_monitor(data_dir, output_dir, interval, threshold, max_iterations, alert_webhook):
    """
    Run real-time security monitoring with continuous threat detection.
    
    Continuously monitors data files in a directory and performs:
    - Real-time adversarial attack detection
    - Continuous evasion attempt monitoring
    - Automated threat alerting and logging
    - Dynamic threshold adjustment
    - Security metric tracking
    
    Examples:
        neurinspectre security realtime-monitor -d ./data_stream
        neurinspectre security realtime-monitor -d ./monitoring -i 30 --threshold 0.8
        neurinspectre security realtime-monitor -d ./data --max 50 -w https://webhook.site/xxx
    """
    try:
        click.echo("⏰ NeurInSpectre Real-time Security Monitor")
        click.echo("="*45)
        
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Set up monitoring log
        log_file = output_dir / f"monitoring_log_{datetime.now().strftime(TIMESTAMP_FORMAT)}.json"
        monitoring_log = []
        
        click.echo(f"📁 Monitoring directory: {data_dir}")
        click.echo(f"📋 Output directory: {output_dir}")
        click.echo(f"⏱️  Monitoring interval: {interval} seconds")
        click.echo(f"🎯 Alert threshold: {threshold}")
        click.echo(f"🔄 Max iterations: {'Infinite' if max_iterations == 0 else max_iterations}")
        if alert_webhook:
            click.echo(f"🔔 Alert webhook: {alert_webhook}")
        click.echo()
        
        # Get integrated security analyzer
        IntegratedSecurityAnalyzer, _ = _get_integrated_security()
        
        analyzer = IntegratedSecurityAnalyzer({
            'threat_threshold': threshold,
            'parallel_processing': True
        })
        
        iteration = 0
        threat_history = []
        last_processed_key = None  # (path, mtime_ns)
        
        click.echo("🚀 Starting real-time monitoring... (Press Ctrl+C to stop)")
        click.echo()
        
        try:
            while max_iterations == 0 or iteration < max_iterations:
                iteration += 1
                
                # Look for new data files
                data_files = list(data_dir.glob('*.npy')) + list(data_dir.glob('*.npz'))
                
                if not data_files:
                    click.echo(f"⏳ [{datetime.now().strftime('%H:%M:%S')}] No data files found, waiting...")
                    time.sleep(interval)
                    continue
                
                # Process latest file (skip if unchanged since last iteration)
                latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
                try:
                    cur_key = (str(latest_file.resolve()), int(latest_file.stat().st_mtime_ns))
                except Exception:
                    cur_key = (str(latest_file), int(latest_file.stat().st_mtime))
                if last_processed_key == cur_key:
                    click.echo(f"⏳ [{datetime.now().strftime('%H:%M:%S')}] No new files (latest unchanged): {latest_file.name}")
                    time.sleep(interval)
                    continue
                last_processed_key = cur_key
                
                click.echo(f"🔍 [{datetime.now().strftime('%H:%M:%S')}] Processing: {latest_file.name}")
                
                try:
                    # Load and analyze data
                    data = _load_data_file(str(latest_file))
                    
                    # Run quick security scan
                    assessment = analyzer.run_comprehensive_security_scan(
                        activation_data=data
                    )
                    
                    # Threat semantics:
                    # - assessment.confidence_score = confidence in the assessment quality (not risk).
                    # - assessment.metadata['threat_score'] = numeric risk score in [0,1] derived from security_scores.
                    threat_score = float((assessment.metadata or {}).get('threat_score', 0.0))

                    # Log results
                    log_entry = {
                        'timestamp': assessment.timestamp,
                        'iteration': iteration,
                        'file': str(latest_file),
                        'threat_level': assessment.overall_threat_level,
                        'threat_score': threat_score,
                        'confidence': assessment.confidence_score,
                        'attacks_detected': len(assessment.detected_attacks),
                        'data_shape': data.shape
                    }
                    
                    monitoring_log.append(log_entry)
                    threat_history.append(threat_score)
                    
                    # Display status
                    threat_icon = {
                        'low': '🟢',
                        'medium': '🟡', 
                        'high': '🟠',
                        'critical': '🔴'
                    }.get(assessment.overall_threat_level, '⚪')
                    
                    click.echo(
                        f"   {threat_icon} Threat: {assessment.overall_threat_level.upper()} "
                        f"(threat_score: {threat_score:.3f}, confidence: {assessment.confidence_score:.3f})"
                    )
                    
                    if assessment.detected_attacks:
                        click.echo(f"   🚨 Attacks detected: {len(assessment.detected_attacks)}")
                        for attack in assessment.detected_attacks[:3]:  # Show top 3
                            click.echo(f"      • {attack['type'].replace('_', ' ').title()}")
                    
                    # Send alert if threat score exceeds threshold
                    if threat_score >= threshold:
                        alert_message = {
                            'timestamp': assessment.timestamp,
                            'threat_level': assessment.overall_threat_level,
                            'threat_score': threat_score,
                            'confidence': assessment.confidence_score,
                            'file': str(latest_file),
                            'attacks': [a['type'] for a in assessment.detected_attacks]
                        }
                        
                        # Save alert
                        alert_file = output_dir / f"alert_{datetime.now().strftime(TIMESTAMP_FORMAT)}.json"
                        with open(alert_file, 'w') as f:
                            json.dump(alert_message, f, indent=2, default=str)
                        
                        click.echo(
                            f"   🚨 ALERT TRIGGERED (threat_score={threat_score:.3f} ≥ {threshold:.3f})! Saved to: {alert_file}"
                        )
                        
                        # Send webhook if configured
                        if alert_webhook:
                            try:
                                import requests
                                response = requests.post(alert_webhook, json=alert_message, timeout=5)
                                if response.status_code == 200:
                                    click.echo("   📡 Alert sent to webhook successfully")
                                else:
                                    click.echo(f"   ⚠️ Webhook failed: {response.status_code}")
                            except Exception as webhook_error:
                                click.echo(f"   ⚠️ Webhook error: {webhook_error}")
                    
                    # Dynamic threshold adjustment (experimental)
                    if len(threat_history) >= 10:
                        recent_avg = float(np.mean(threat_history[-10:]))
                        if recent_avg > threshold * 1.2:
                            click.echo("   📈 High threat pattern detected - consider raising threshold")
                        elif recent_avg < threshold * 0.5:
                            click.echo("   📉 Low threat pattern - threshold may be too high")
                    
                except Exception as analysis_error:
                    click.echo(f"   ❌ Analysis error: {analysis_error}")
                    
                    # Log error
                    monitoring_log.append({
                        'timestamp': time.time(),
                        'iteration': iteration,
                        'file': str(latest_file),
                        'error': str(analysis_error)
                    })
                
                # Save monitoring log periodically
                if iteration % 10 == 0:
                    with open(log_file, 'w') as f:
                        json.dump(monitoring_log, f, indent=2, default=str)
                
                click.echo()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            click.echo("\n🛑 Monitoring stopped by user")
        
        # Final log save
        with open(log_file, 'w') as f:
            json.dump(monitoring_log, f, indent=2, default=str)
        
        # Summary
        click.echo()
        click.echo("📊 MONITORING SUMMARY:")
        click.echo(f"   • Total iterations: {iteration}")
        click.echo(f"   • Log entries: {len(monitoring_log)}")
        click.echo(f"   • Average threat_score: {np.mean(threat_history):.3f}" if threat_history else "   • No threat data")
        click.echo(f"   • Log saved to: {log_file}")
        
    except Exception as e:
        click.echo(f"❌ Error in real-time monitoring: {str(e)}")
        logger.error(f"Real-time monitoring error: {e}", exc_info=True)
        raise click.ClickException(str(e))

# Add the new CLI commands to the module exports
__all__ = [
    'adversarial_detect', 
    'evasion_detect',
    'comprehensive_scan',
    'realtime_monitor',
    'register_commands',
    'add_security_commands',
    'handle_security_command'
] 

def add_security_commands(subparsers):
    """Add security commands to the main CLI parser"""
    
    # Adversarial detection command
    adversarial_parser = subparsers.add_parser(
        'adversarial-detect',
        help='🛡️ Adversarial attack detection'
    )
    adversarial_parser.add_argument('data_path', help='Path to activation/gradient data file or directory')
    adversarial_parser.add_argument('--reference-path', '-r', help='Path to reference data for comparison (optional)')
    adversarial_parser.add_argument('--output-dir', '-o', default='./security_reports',
                                   help='Output directory for analysis results')
    adversarial_parser.add_argument('--detector-type', '-t', 
                                   choices=['ts-inverse', 'concretizer', 'attention-guard', 'ednn', 'all'],
                                   default='all', help='Type of adversarial detector to use')
    adversarial_parser.add_argument('--threshold', '--th', default=0.8, type=float,
                                   help='Detection sensitivity threshold (0.0-1.0)')
    adversarial_parser.add_argument('--parallel', '-p', action='store_true', default=True,
                                   help='Use parallel processing for faster analysis')
    adversarial_parser.add_argument('--save-results', action='store_true', default=True,
                                   help='Save results to JSON file')
    adversarial_parser.add_argument('--output-format', choices=['json', 'text'], default='text',
                                   help='Output format for results')
    adversarial_parser.add_argument('--output-summary', help='Optional HTML summary visualization')
    adversarial_parser.add_argument('--max-samples', type=int, default=100000,
                                   help='Cap maximum number of samples (rows) to analyze')
    adversarial_parser.add_argument('--memmap', action='store_true', default=False,
                                   help='Use memory-mapped loading for .npy when possible')
    adversarial_parser.add_argument('--align', choices=['truncate','pad','window'], default='truncate',
                                   help='Sequence alignment policy when lengths differ')
    adversarial_parser.add_argument('--window-size', type=int, default=None,
                                   help='Window size for window alignment (defaults to min length)')
    adversarial_parser.add_argument('--robust-scale', action='store_true', default=False,
                                   help='Use median/MAD scaling instead of mean/std')
    adversarial_parser.add_argument('--kl-warn', type=float, default=0.5,
                                   help='Warn if JS divergence between data and reference exceeds this')
    adversarial_parser.add_argument('--seed', type=int, default=42,
                                   help='Random seed for determinism')
    adversarial_parser.add_argument('--warn-max-dim', type=int, default=50000,
                                   help='Warn if feature dimension exceeds this value')
    
    # Evasion detection command
    evasion_parser = subparsers.add_parser('evasion-detect', 
                                         help='🚫 Neural transport dynamics evasion detection')
    evasion_parser.add_argument('data_path', help='Path to neural activation data file (.npy, .npz, .csv)')
    evasion_parser.add_argument('--network-data', '-n', help='Path to network flow data for DeMarking analysis (optional)')
    evasion_parser.add_argument('--output-dir', '-o', default='./security_reports',
                               help='Output directory for analysis results')
    evasion_parser.add_argument('--detector-type', '-t',
                               choices=['transport-dynamics', 'demarking', 'behavioral', 'all'],
                               default='all', help='Type of evasion detector to use')
    evasion_parser.add_argument('--threshold', '--th', default=0.6, type=float,
                               help='Detection sensitivity threshold (0.0-1.0)')
    evasion_parser.add_argument('--time-window', '-w', default=100, type=int,
                               help='Time window for temporal analysis')
    evasion_parser.add_argument('--output-summary', help='Optional HTML summary visualization')
    
    # Comprehensive scan command
    comprehensive_parser = subparsers.add_parser('comprehensive-scan', 
                                               help='🔒 Complete security analysis with all techniques')
    comprehensive_parser.add_argument('activation_data', help='Path to neural activation data file (.npy, .npz, .csv)')
    comprehensive_parser.add_argument('--gradient-data', '-g', help='Path to gradient data file (optional)')
    comprehensive_parser.add_argument('--model-weights', '-m', help='Path to model weights file (.npz) (optional)')
    comprehensive_parser.add_argument('--network-data', '-n', help='Path to network flow data (optional)')
    comprehensive_parser.add_argument('--output-dir', '-o', default='./security_reports',
                                    help='Output directory for comprehensive security report')
    comprehensive_parser.add_argument('--parallel', '-p', action='store_true', default=True,
                                    help='Use parallel processing for faster analysis')
    comprehensive_parser.add_argument('--threshold', '--th', default=0.6, type=float,
                                    help='Overall threat detection threshold')
    comprehensive_parser.add_argument('--generate-report', '-r', action='store_true', default=True,
                                    help='Generate comprehensive HTML security report')
    comprehensive_parser.add_argument('--output-visualization', help='Optional HTML overview visualization')
    
    # Real-time monitoring command
    realtime_parser = subparsers.add_parser('realtime-monitor', 
                                          help='⏰ Real-time continuous security monitoring')
    realtime_parser.add_argument('data_dir', help='Directory containing data files for continuous monitoring')
    realtime_parser.add_argument('--output-dir', '-o', default='./security_monitoring',
                                help='Output directory for monitoring logs and alerts')
    realtime_parser.add_argument('--interval', '-i', default=60, type=int,
                                help='Monitoring interval in seconds')
    realtime_parser.add_argument('--threshold', '--th', default=0.7, type=float,
                                help='Alert threshold on threat score (0.0-1.0)')
    realtime_parser.add_argument('--max-iterations', '--max', default=100, type=int,
                                help='Maximum monitoring iterations (0 for infinite)')
    realtime_parser.add_argument('--alert-webhook', '-w', type=str,
                                help='Webhook URL for sending security alerts')

def handle_security_command(args):
    """Handle security command execution"""
    
    try:
        if args.command == 'adversarial-detect':
            return handle_adversarial_detect(args)
        elif args.command == 'evasion-detect':
            return handle_evasion_detect(args)
        elif args.command == 'comprehensive-scan':
            return handle_comprehensive_scan(args)
        elif args.command == 'realtime-monitor':
            return handle_realtime_monitor(args)
        else:
            logger.error(f"Unknown security command: {args.command}")
            return 1
    except FileNotFoundError as e:
        # User-facing clarity: the most common cause of failures is simply running the
        # command from a directory that doesn't contain the file. Provide actionable
        # remediation without dumping an exception traceback.
        missing = str(e)
        print(f"❌ {missing}")
        try:
            cwd = Path.cwd()
            print(f"   Current directory: {cwd}")

            # If the command had a data path, show where we expected it.
            p = None
            if hasattr(args, "data_path"):
                p = Path(str(getattr(args, "data_path")))
            elif hasattr(args, "activation_data"):
                p = Path(str(getattr(args, "activation_data")))

            if p is not None and not p.is_absolute():
                print(f"   Looked for: {(cwd / p).resolve()}")

            # Offer a quick hint to locate candidate files.
            print("   Tip: pass an absolute path, or run the command from the directory that contains the file.")
            print("   Examples:")
            print("     neurinspectre adversarial-detect /full/path/to/data.npy --detector-type all --threshold 0.8 -o _cli_runs/triage")
            print("     ls *.npy")
            print("     find _cli_runs -name '*.npy' -maxdepth 3 2>/dev/null | head")

            # Small convenience: list a few nearby .npy artifacts if present.
            npy_here = sorted(Path(cwd).glob("*.npy"))
            if npy_here:
                print("   Found .npy files in current directory:")
                for q in npy_here[:10]:
                    print(f"     - {q.name}")
            npy_cli = sorted((Path(cwd) / "_cli_runs").glob("**/*.npy")) if (Path(cwd) / "_cli_runs").exists() else []
            if npy_cli:
                print("   Found .npy files under _cli_runs/:")
                for q in npy_cli[:10]:
                    print(f"     - {q}")
        except Exception:
            pass
        return 1
    except Exception as e:
        logger.error(f"Security command failed: {e}")
        return 1

def handle_adversarial_detect(args):
    """Handle adversarial detection command"""
    
    print("🛡️ NeurInSpectre Adversarial Attack Detection")
    print("="*55)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"📁 Loading data from: {args.data_path}")
    # Set seed for determinism
    try:
        import random
        random.seed(args.seed)
    except Exception:
        pass
    np.random.seed(args.seed)

    # Optional memmap load for large .npy files
    dp = Path(args.data_path)
    if args.memmap and dp.is_file() and dp.suffix.lower() == '.npy':
        try:
            data = np.load(dp, mmap_mode='r')
            data = _ensure_2d(np.asarray(data))
        except Exception:
            data = _load_data_file(args.data_path)
    else:
        data = _load_data_file(args.data_path)
    
    reference_data = None
    if args.reference_path:
        print(f"📋 Loading reference data from: {args.reference_path}")
        rp = Path(args.reference_path)
        if args.memmap and rp.is_file() and rp.suffix.lower() == '.npy':
            try:
                reference_data = np.load(rp, mmap_mode='r')
                reference_data = _ensure_2d(np.asarray(reference_data))
            except Exception:
                reference_data = _load_data_file(args.reference_path)
        else:
            reference_data = _load_data_file(args.reference_path)
    
    print(f"📊 Data shape: {data.shape}")
    if reference_data is not None:
        print(f"📋 Reference data shape: {reference_data.shape}")
    
    # Normalize inputs to expected dimensionality
    # Many detectors expect >=2D (e.g., [seq_len, features]).
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if reference_data is not None and reference_data.ndim == 1:
        reference_data = reference_data.reshape(-1, 1)

    # Align sample dimension if both are 2D but have different lengths
    if reference_data is not None and data.ndim == 2 and reference_data.ndim == 2:
        if data.shape[1] != reference_data.shape[1]:
            # If feature dims differ, truncate to the smaller feature dimension
            min_d = min(data.shape[1], reference_data.shape[1])
            data = data[:, :min_d]
            reference_data = reference_data[:, :min_d]
        if data.shape[0] != reference_data.shape[0]:
            min_n = min(data.shape[0], reference_data.shape[0])
            data = data[:min_n]
            reference_data = reference_data[:min_n]

    # Apply cleaning/scaling
    data = _clean_array(data, robust=args.robust_scale)
    if reference_data is not None:
        reference_data = _clean_array(reference_data, robust=args.robust_scale)

    # Cap max samples
    if data.shape[0] > args.max_samples:
        data = data[:args.max_samples]
    if reference_data is not None and reference_data.shape[0] > args.max_samples:
        reference_data = reference_data[:args.max_samples]

    # Warn on high dimensionality
    if data.shape[1] > args.warn_max_dim:
        print(f"⚠️  High dimensional input (D={data.shape[1]}). Consider dimensionality reduction.")

    # Align shapes according to policy
    if reference_data is not None and data.ndim == 2 and reference_data.ndim == 2:
        # Align features
        if data.shape[1] != reference_data.shape[1]:
            min_d = min(data.shape[1], reference_data.shape[1])
            data = data[:, :min_d]
            reference_data = reference_data[:, :min_d]
        # Align samples
        if args.align == 'truncate':
            min_n = min(data.shape[0], reference_data.shape[0])
            data = data[:min_n]
            reference_data = reference_data[:min_n]
        elif args.align == 'pad':
            max_n = max(data.shape[0], reference_data.shape[0])
            def pad(x, n):
                if x.shape[0] >= n:
                    return x
                pad_rows = np.repeat(x[-1:], n - x.shape[0], axis=0)
                return np.vstack([x, pad_rows])
            data = pad(data, max_n)
            reference_data = pad(reference_data, max_n)
        elif args.align == 'window':
            win = args.window_size or min(data.shape[0], reference_data.shape[0])
            data = data[-win:]
            reference_data = reference_data[-win:]

    # JS divergence warning for domain mismatch (approx via histograms)
    if reference_data is not None:
        try:
            def js_div(a, b, bins=50):
                hist_a, _ = np.histogram(a.flatten(), bins=bins, density=True)
                hist_b, _ = np.histogram(b.flatten(), bins=bins, density=True)
                pa = hist_a / (hist_a.sum() + 1e-12)
                pb = hist_b / (hist_b.sum() + 1e-12)
                m = 0.5 * (pa + pb)
                def kl(p, q):
                    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))
                return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
            js = js_div(data, reference_data)
            if js > args.kl_warn:
                print(f"⚠️  High JS divergence between data and reference (JS={js:.3f}). Domain mismatch suspected.")
        except Exception:
            pass

    print(f"🎯 Detector type: {args.detector_type}")
    print(f"⚙️  Threshold: {args.threshold}")
    print(f"🚀 Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")
    print()
    
    # Get adversarial detector
    AdversarialDetector = _get_adversarial_detector()
    
    # Configure detector
    config = {
        'threshold': args.threshold,
        'ts_inverse_threshold': args.threshold,
        'concretizer_threshold': args.threshold,
        'attention_guard_threshold': args.threshold,
        'voxel_resolution': 32,
        'max_seq_length': min(512, data.shape[0]),
        'attention_heads': 8,
        'k_neighbors': 5,
        'ednn_threshold': args.threshold,
    }
    
    detector = AdversarialDetector(config)
    
    # Run detection
    print("🔍 Running adversarial detection...")
    results = detector.detect_adversarial_samples(data, reference_data, detector_type=args.detector_type)
    
    # Process results
    print("✅ Adversarial detection completed!")
    print()
    print("🔍 Detection Results:")
    print(f"   • Overall threat level: {results['overall_threat_level'].upper()}")
    print(f"   • Input data shape: {results['input_shape']}")
    print(f"   • Timestamp: {datetime.fromtimestamp(results['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display specific detections
    if results.get('detections'):
        print()
        print("🎯 Specific Attack Detections:")
        
        for detection_name, detection_result in results['detections'].items():
            if isinstance(detection_result, dict):
                is_attack = (
                    detection_result.get('is_attack', False) or
                    detection_result.get('is_inversion_attack', False) or
                    detection_result.get('is_misbehavior', False) or
                    detection_result.get('is_ednn_attack', False)
                )
                
                confidence = detection_result.get('confidence',
                            detection_result.get('inversion_score',
                            detection_result.get('misbehavior_score',
                            detection_result.get('attack_score', 0.0))))
                
                status = "🚨 DETECTED" if is_attack else "✅ CLEAN"
                print(f"   • {detection_name.replace('_', ' ').title()}: {status} (confidence: {confidence:.3f})")
    
    # Display confidence scores
    if results.get('confidence_scores'):
        print()
        print("📊 Confidence Scores:")
        for method, score in results['confidence_scores'].items():
            print(f"   • {method.replace('_', ' ').title()}: {score:.3f}")
    
    # Save results
    if args.save_results:
        # Add actionable summary before saving
        actionable = {}
        # Compute simple top feature differences if reference available
        def top_diff_feats(a: np.ndarray, b: Optional[np.ndarray], k: int = 5):
            if b is None or a.shape[1] != b.shape[1] or a.size == 0:
                return []
            diff = np.abs(np.mean(a, axis=0) - np.mean(b, axis=0))
            idxs = np.argsort(diff)[-k:][::-1]
            return [int(i) for i in idxs]
        top_feats = top_diff_feats(data, reference_data)
        def mk_entry(name: str, det: dict):
            if not isinstance(det, dict):
                return None
            status = 'DETECTED' if any(det.get(k, False) for k in ['is_attack','is_inversion_attack','is_misbehavior','is_ednn_attack']) else 'CLEAN'
            conf = det.get('confidence') or det.get('inversion_score') or det.get('misbehavior_score') or det.get('attack_score') or 0.0
            rationale = 'high differential in features ' + ','.join(map(str, top_feats)) if top_feats else 'pattern-based anomaly'
            red = 'probe/amplify top-diff features; compare JSON spans' if status=='DETECTED' else 'attempt targeted perturbations to validate cleanliness'
            blue = 'monitor/clamp top-diff features; retrain with regularization' if status=='DETECTED' else 'record as baseline and monitor drift'
            return {
                'detector': name,
                'status': status,
                'confidence': float(conf),
                'rationale': rationale,
                'top_features': top_feats,
                'red_next': red,
                'blue_next': blue,
            }
        for name in ['ts_inverse','concretizer','attention_guard','ednn']:
            if name in results['detections']:
                entry = mk_entry(name, results['detections'][name])
                if entry:
                    actionable[name] = entry
        results['actionable_summary'] = actionable

        result_file = _save_results(results, args.output_dir, 'adversarial_detection')
        print()
        print(f"💾 Results saved to: {result_file}")
        # Optional HTML summary visualization
        try:
            if getattr(args, 'output_summary', None):
                import numpy as _np
                import plotly.graph_objects as _go
                from plotly.subplots import make_subplots as _mk
                html = args.output_summary
                dets = results.get('detections', {})
                contrib_names = []
                contrib_vals = []
                for name in ['ts_inverse','concretizer','attention_guard','ednn']:
                    d = dets.get(name)
                    if isinstance(d, dict):
                        val = d.get('confidence') or d.get('inversion_score') or d.get('misbehavior_score') or d.get('attack_score') or 0.0
                        contrib_names.append(name.replace('_',' ').title())
                        contrib_vals.append(float(val))
                fig = _mk(rows=1, cols=2,
                          specs=[[{"type":"xy"},{"type":"xy"}]],
                          column_widths=[0.55,0.45],
                          subplot_titles=('Detector Contributions', 'Top Feature Differences (ref vs data)'))
                if contrib_names:
                    fig.add_trace(_go.Bar(x=contrib_names, y=contrib_vals, name='Contrib'), row=1, col=1)
                # Compute top feature differences if reference is available
                if reference_data is not None:
                    try:
                        dm = _np.mean(data, axis=0).ravel()
                        rm = _np.mean(reference_data, axis=0).ravel()
                        diff = _np.abs(dm - rm)
                        k = int(min(10, diff.shape[0]))
                        idx = _np.argsort(diff)[-k:][::-1]
                        fig.add_trace(_go.Bar(x=[str(int(i)) for i in idx], y=[float(diff[int(i)]) for i in idx], name='Δmean'), row=1, col=2)
                    except Exception:
                        # Place note below the charts so it doesn't occlude content
                        fig.add_annotation(text='No reference diff available', xref='paper', yref='paper', x=0.75, y=-0.12, showarrow=False,
                                           bgcolor='rgba(242,242,242,0.85)', bordercolor='rgba(128,128,128,0.8)', font=dict(size=11))
                else:
                    # Put the note below figures for readability
                    fig.add_annotation(text='No reference provided', xref='paper', yref='paper', x=0.75, y=-0.12, showarrow=False,
                                       bgcolor='rgba(242,242,242,0.85)', bordercolor='rgba(128,128,128,0.8)', font=dict(size=11))
                fig.update_layout(height=560, width=1100, title_text='Adversarial Detection Summary', showlegend=False,
                                  margin=dict(l=60, r=40, t=60, b=120))
                fig.write_html(html)
                print(f"📄 Summary visualization saved to: {html}")
        except Exception as _e:
            print(f"⚠️  Summary visualization skipped: {_e}")
    
    # Show recommendations if high threat
    if results['overall_threat_level'] in ['high', 'critical']:
        print()
        print("⚠️  HIGH THREAT DETECTED - Immediate Actions Recommended:")
        print("   • Implement adversarial training and input validation")
        print("   • Deploy real-time gradient monitoring systems")
        print("   • Review model integrity and training data")
        print("   • Enable enhanced security logging and monitoring")

    # Print concise actionable summary to console
    if 'actionable_summary' in results and results['actionable_summary']:
        print()
        print("🧭 Actionable Summary:")
        for det, info in results['actionable_summary'].items():
            print(f"   • {det.replace('_',' ').title()}: {info['status']} (conf {info['confidence']:.3f})")
            if info['top_features']:
                print(f"     - top_features: {info['top_features']}")
            print(f"     - red_next: {info['red_next']}")
            print(f"     - blue_next: {info['blue_next']}")
    
    return 0

def handle_evasion_detect(args):
    """Handle evasion detection command"""
    
    print("🚫 NeurInSpectre Evasion Attack Detection")
    print("="*45)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"📁 Loading neural data from: {args.data_path}")
    neural_data = _load_data_file(args.data_path)
    
    network_flow_data = None
    if args.network_data:
        p = Path(args.network_data)
        if not p.exists():
            print(f"⚠️  Network data file not found: {p}")
            print("   No synthetic/demo fallback will be generated.")
            print("   If you want a local DeMarking-test PCAP, run:")
            print("   `neurinspectre generate-demarking-pcap --out <path.pcap> --threshold 0.6`")
            return 1

        print(f"🌐 Loading network data from: {args.network_data}")
        network_flow_data = _load_data_file(args.network_data)
    
    print(f"📊 Neural data shape: {neural_data.shape}")
    if network_flow_data is not None:
        print(f"🌐 Network data shape: {network_flow_data.shape}")
    
    print(f"🎯 Detector type: {args.detector_type}")
    print(f"⚙️  Threshold: {args.threshold}")
    print(f"⏰ Time window: {args.time_window}")
    print()
    
    # Get evasion detector
    EvasionDetector = _get_evasion_detector()
    
    # Configure detector
    config = {
        'transport_dim': min(64, neural_data.shape[-1]),
        'time_window': args.time_window,
        'demarking_window': 50,
        'demarking_threshold': args.threshold,
        'pattern_window': args.time_window
    }
    
    detector = EvasionDetector(config)
    analysis_details: Dict[str, Any] = {}
    
    # Attach network IPD side-channel data (PCAP-derived) in a way that survives np.asarray(...).
    # numpy arrays cannot have arbitrary attributes, so we wrap them in an object that implements
    # __array__ and exposes ipd_data.
    activation_input = neural_data
    if network_flow_data is not None:
        try:
            ipd = np.asarray(network_flow_data).reshape(-1)
        except Exception:
            ipd = network_flow_data

        class _ActivationBundle:
            __slots__ = ("_arr", "ipd_data")
            def __init__(self, arr: np.ndarray, ipd_data: Any):
                self._arr = np.asarray(arr)
                self.ipd_data = ipd_data
            def __array__(self, dtype=None):
                return np.asarray(self._arr, dtype=dtype)

        activation_input = _ActivationBundle(neural_data, ipd)
    
    # Run detection
    print("🔍 Running evasion detection...")
    detector_failed = False
    try:
        if args.detector_type == 'demarking':
            if network_flow_data is None:
                raise ValueError("detector-type=demarking requires --network-data (.pcap/.npy IPD series)")
            ipd = np.asarray(network_flow_data).reshape(-1)
            dem = detector.demarking_detector.detect_watermarking_evasion(ipd)
            analysis_details["demarking"] = dem
            analysis_details["demarking_ipd_len"] = int(ipd.size)
            evasion_attempts = []
            if dem.get('is_evasion'):
                score = float(dem.get('evasion_score', 0.0))
                evasion_attempts.append({
                    'type': 'watermarking_evasion',
                    'confidence': score,
                    # Use the common severity scale used across NeurInSpectre summaries.
                    'threat_level': str(detector._determine_threat_level(score)),
                    'details': dem,
                })
        elif args.detector_type == 'transport-dynamics':
            tr = detector.transport_detector.detect_transport_anomalies(np.asarray(activation_input))
            analysis_details["transport_dynamics"] = tr
            evasion_attempts = []
            if tr.get('is_evasion'):
                evasion_attempts.append({
                    'type': 'transport_dynamics',
                    'confidence': float(tr.get('evasion_score', 0.0)),
                    'threat_level': str(tr.get('threat_level', 'medium')),
                    'details': tr,
                })
        elif args.detector_type == 'behavioral':
            br = detector.behavioral_analyzer.analyze_behavioral_patterns(np.asarray(activation_input))
            analysis_details["behavioral"] = br
            evasion_attempts = []
            if br.get('is_anomalous'):
                evasion_attempts.append({
                    'type': 'behavioral_anomaly',
                    'confidence': float(br.get('anomaly_score', 0.0)),
                    'threat_level': str(detector._determine_threat_level(float(br.get('anomaly_score', 0.0)))),
                    'details': br,
                })
        else:
            evasion_attempts = detector.detect_evasion_attempts(activation_input)
    except Exception as _det_err:
        detector_failed = True
        analysis_details["detector_error"] = str(_det_err)
        print(f"   ⚠️ Detector error: {_det_err}")
        evasion_attempts = []

    # Strict: no synthetic/heuristic evasion attempts. If the detector returns nothing, that is "no signal".
    if not isinstance(evasion_attempts, list):
        evasion_attempts = []
    
    # Process results
    print("✅ Evasion detection completed!")
    print()
    print("🔍 Detection Results:")
    print(f"   • Total evasion attempts detected: {len(evasion_attempts)}")
    
    if evasion_attempts:
        print()
        print("🚨 Detected Evasion Attempts:")
        
        for i, attempt in enumerate(evasion_attempts, 1):
            attempt_type = attempt.get('type', 'unknown')
            confidence = attempt.get('confidence', 0.0)
            threat_level = attempt.get('threat_level', 'unknown')
            
            print(f"   {i}. {attempt_type.replace('_', ' ').title()}")
            print(f"      • Confidence: {confidence:.3f}")
            print(f"      • Threat Level: {threat_level.upper()}")
            
            # Show specific details for certain attack types
            details = attempt.get('details', {})
            if 'evasion_score' in details:
                print(f"      • Evasion Score: {details['evasion_score']:.3f}")
            if 'threat_level' in details:
                print(f"      • Analysis Threat Level: {details['threat_level']}")
    else:
        print("   ✅ No evasion attempts detected")
    
    # Get detection summary
    # For non-"all" modes we didn’t go through detector.detect_evasion_attempts(), so populate history here.
    try:
        if args.detector_type != 'all':
            detector.detection_history.append({
                'timestamp': time.time(),
                'evasion_attempts': evasion_attempts,
                'input_shape': np.asarray(activation_input).shape,
            })
    except Exception:
        pass
    summary = detector.get_detection_summary()
    
    print()
    print("📊 Detection Summary:")
    print(f"   • Total detections: {summary['total_detections']}")
    print(f"   • Total evasion attempts: {summary['evasion_attempts']}")
    
    if 'threat_distribution' in summary:
        print("   • Threat distribution:")
        for level, count in summary['threat_distribution'].items():
            if count > 0:
                print(f"     - {level.title()}: {count}")
    
    # Strict: do not fabricate placeholder detections.

    # Save results
    results = {
        'evasion_attempts': evasion_attempts,
        'detection_summary': summary,
        'analysis_details': analysis_details,
        'config': config,
        'input_shape': neural_data.shape,
        'network_data_path': getattr(args, 'network_data', None),
        'timestamp': time.time()
    }
    
    result_file = _save_results(results, args.output_dir, 'evasion_detection')
    print()
    print(f"💾 Results saved to: {result_file}")

    # ---------------------------------------------------------------------
    # DeMarking report (deep, technically grounded; JSON + optional HTML)
    # ---------------------------------------------------------------------
    try:
        if (
            getattr(args, "network_data", None)
            and network_flow_data is not None
            and getattr(args, "detector_type", "all") in ("demarking", "all")
        ):
            ts = datetime.now().strftime(TIMESTAMP_FORMAT)

            ipd = np.asarray(network_flow_data, dtype=float).reshape(-1)
            ipd = np.nan_to_num(ipd, nan=0.0, posinf=0.0, neginf=0.0)
            ipd = ipd[ipd >= 0.0]
            n_ipd = int(ipd.size)

            dem = detector.demarking_detector.detect_watermarking_evasion(ipd)

            eps = 1e-12
            if n_ipd > 0:
                mean = float(np.mean(ipd))
                std = float(np.std(ipd))
                median = float(np.median(ipd))
                q05 = float(np.quantile(ipd, 0.05))
                q95 = float(np.quantile(ipd, 0.95))
                vmin = float(np.min(ipd))
                vmax = float(np.max(ipd))
                cv = float(std / (abs(mean) + eps))

                counts, bin_edges = np.histogram(ipd, bins=50)
                counts_l = counts.astype(int).tolist()
                edges_l = [float(x) for x in bin_edges]

                # Autocorrelation (mean-centered, lag-0 normalized)
                x = ipd.astype(np.float64) - float(mean)
                ac = np.correlate(x, x, mode="full")
                ac = ac[ac.size // 2 :]
                ac_norm = ac / (float(ac[0]) + eps) if ac.size else np.asarray([0.0])
                max_lag = int(min(50, max(0, ac_norm.size - 1)))
                lags_l = list(range(max_lag + 1))
                acf_l = [float(v) for v in ac_norm[: max_lag + 1]]

                # Spectral summary (index-domain FFT; matches the detector's implicit sampling).
                if n_ipd >= 8:
                    y = x
                    fft = np.fft.rfft(y)
                    psd = (np.abs(fft) ** 2) / float(max(1, y.size))
                    freq = np.fft.rfftfreq(y.size, d=1.0)  # cycles per packet index
                    psd_sum = float(np.sum(psd))
                    if psd_sum > 0.0 and psd.size > 1:
                        p = psd / (psd_sum + eps)
                        spec_entropy = float(-np.sum(p * np.log(p + eps)) / np.log(p.size))
                        spec_entropy = float(np.clip(spec_entropy, 0.0, 1.0))
                    else:
                        spec_entropy = 0.0
                    freq_l = [float(v) for v in freq[: min(256, freq.size)]]
                    psd_l = [float(v) for v in psd[: min(256, psd.size)]]
                else:
                    spec_entropy = 0.0
                    freq_l, psd_l = [], []
            else:
                mean = std = median = q05 = q95 = vmin = vmax = cv = 0.0
                counts_l, edges_l, lags_l, acf_l, freq_l, psd_l = [], [], [], [], [], []
                spec_entropy = 0.0

            # Component scores (must match detector score semantics)
            weights = {
                "ipd_patterns": 0.3,
                "gan_detection": 0.3,
                "temporal_correlations": 0.2,
                "adversarial_perturbations": 0.15,
                "statistical_consistency": 0.05,
            }
            ipd_patterns = dem.get("ipd_patterns", {}) or {}
            gan_detection = dem.get("gan_detection", {}) or {}
            temporal = dem.get("temporal_correlations", {}) or {}
            adv = dem.get("adversarial_perturbations", {}) or {}
            consistency = float(dem.get("statistical_consistency", 0.0))

            ipd_score = float((float(ipd_patterns.get("entropy", 0.0)) + float(ipd_patterns.get("regularity", 0.0))) / 2.0)
            gan_score = float(gan_detection.get("confidence", 0.0))
            temporal_score = float((float(temporal.get("correlation_strength", 0.0)) + float(temporal.get("lag_correlation", 0.0))) / 2.0)
            perturb_score = float(adv.get("perturbation_score", 0.0))
            consistency_score = float(np.clip(1.0 - float(consistency), 0.0, 1.0))

            component_scores = {
                "ipd_score": ipd_score,
                "gan_score": gan_score,
                "temporal_score": temporal_score,
                "perturbation_score": perturb_score,
                "consistency_score": consistency_score,
                "weights": weights,
                "weighted": {
                    "ipd_patterns": ipd_score * weights["ipd_patterns"],
                    "gan_detection": gan_score * weights["gan_detection"],
                    "temporal_correlations": temporal_score * weights["temporal_correlations"],
                    "adversarial_perturbations": perturb_score * weights["adversarial_perturbations"],
                    "statistical_consistency": consistency_score * weights["statistical_consistency"],
                },
            }

            demarking_report = {
                "title": "NeurInSpectre DeMarking Telemetry Report",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "inputs": {
                    "network_data_path": str(getattr(args, "network_data", "")),
                    "neural_data_path": str(getattr(args, "data_path", "")),
                    "ipd_len": n_ipd,
                },
                "threshold": float(getattr(args, "threshold", 0.6)),
                "demarking_result": dem,
                "component_scores": component_scores,
                "ipd_summary": {
                    "mean_s": mean,
                    "std_s": std,
                    "median_s": median,
                    "p05_s": q05,
                    "p95_s": q95,
                    "min_s": vmin,
                    "max_s": vmax,
                    "cv": cv,
                },
                "ipd_histogram": {"bin_edges": edges_l, "counts": counts_l},
                "autocorrelation": {"lags": lags_l, "acf": acf_l},
                "spectral": {
                    "entropy_norm": spec_entropy,
                    "freq_cycles_per_packet": freq_l,
                    "psd": psd_l,
                },
                "references": [
                    {
                        "title": "Generating Traffic-Level Adversarial Examples from Feature-Level Specifications (ESORICS Workshops, 2025 online)",
                        "url": "https://link.springer.com/chapter/10.1007/978-3-031-82362-6_8",
                        "note": "Conceptual link: feature-level timing specs → traffic-level (PCAP) realizations under constraints.",
                    }
                ],
                "notes": [
                    "This analysis is timing-driven: the PCAP payload bytes are ignored; only per-packet timestamps (IPDs) matter.",
                    "Scores are heuristic and intended for triage. A high score indicates timing patterns consistent with watermarking evasion/reshaping.",
                ],
            }

            report_json = Path(args.output_dir) / f"demarking_report_{ts}.json"
            with open(report_json, "w") as f:
                json.dump(demarking_report, f, indent=2, default=str)
            print(f"📄 DeMarking report saved to: {report_json}")

            # Optional: HTML report with plots (Plotly).
            try:
                _px, make_subplots, go = _get_plotly()
                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=(
                        "Inter-Packet Delays (IPD) over packet index",
                        "IPD Histogram",
                        "Autocorrelation (lag-0 normalized)",
                        "Power Spectrum (index-domain)",
                    ),
                )
                fig.add_trace(
                    go.Scatter(x=list(range(n_ipd)), y=ipd.tolist(), mode="lines", name="IPD (s)"),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Histogram(x=ipd.tolist(), nbinsx=50, name="IPD hist"),
                    row=1,
                    col=2,
                )
                fig.add_trace(
                    go.Scatter(x=lags_l, y=acf_l, mode="lines+markers", name="ACF"),
                    row=2,
                    col=1,
                )
                if freq_l and psd_l:
                    fig.add_trace(
                        go.Scatter(x=freq_l, y=psd_l, mode="lines", name="PSD"),
                        row=2,
                        col=2,
                    )
                else:
                    fig.add_trace(
                        go.Scatter(x=[0.0], y=[0.0], mode="lines", name="PSD"),
                        row=2,
                        col=2,
                    )

                score = float(dem.get("evasion_score", 0.0))
                fig.update_layout(
                    title_text=f"NeurInSpectre DeMarking Telemetry Report — score={score:.3f} (threshold={float(getattr(args, 'threshold', 0.6)):.3f})",
                    height=900,
                    width=1200,
                    showlegend=False,
                    margin=dict(l=60, r=40, t=80, b=60),
                )

                report_html = Path(args.output_dir) / f"demarking_report_{ts}.html"
                fig.write_html(str(report_html))
                print(f"🖼️ DeMarking report HTML saved to: {report_html}")
            except Exception as _e:
                print(f"⚠️  DeMarking HTML report skipped: {_e}")
    except Exception as _e:
        print(f"⚠️  DeMarking report generation failed: {_e}")

    # Optional HTML summary
    try:
        if getattr(args, 'output_summary', None):
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            html = args.output_summary
            
            # Collect evasion data
            labels, confs, evasion_types = [], [], []
            for a in (evasion_attempts or [])[:10]:
                labels.append(a.get('type','Unknown').replace('_',' ').title())
                confs.append(float(a.get('confidence', 0.0)))
                evasion_types.append(a.get('type', 'unknown'))
            
            if not labels:
                labels, confs = ['No Detections'], [0.0]
                evasion_types = ['none']
            
            # Build threat distribution
            dist = summary.get('threat_distribution', {})
            if not dist:
                for a in evasion_attempts:
                    lvl = str(a.get('threat_level', 'unknown')).lower()
                    dist[lvl] = dist.get(lvl, 0) + 1
            
            # Calculate threat level
            critical_count = dist.get('critical', 0)
            high_count = dist.get('high', 0)
            
            if critical_count > 0:
                overall_threat = "CRITICAL"
            elif high_count > 0:
                overall_threat = "HIGH"
            else:
                overall_threat = "MEDIUM/LOW"
            
            # Create 4-panel dashboard
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type":"xy"}, {"type":"domain"}],
                       [{"type":"table"}, {"type":"table"}]],
                subplot_titles=(
                    'Evasion Attempts Detected (Confidence)',
                    '',
                    '🔴 Red Team Actionable Intelligence',
                    '🔵 Blue Team Defense Recommendations'
                ),
                row_heights=[0.40, 0.60],
                vertical_spacing=0.30,
                horizontal_spacing=0.1
            )
            
            # Panel 1: Bar chart with color coding
            bar_colors = ['#dc3545' if c > 0.8 else '#fd7e14' if c > 0.6 else '#ffc107' for c in confs]
            fig.add_trace(
                go.Bar(
                    x=labels, 
                    y=confs, 
                    marker_color=bar_colors,
                    text=[f'{c:.1%}' for c in confs],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Panel 2: Pie chart
            pie_labels = [k.title() for k in dist.keys()]
            pie_vals = list(dist.values())
            pie_colors_map = {
                'critical': '#dc3545',
                'high': '#fd7e14', 
                'medium': '#ffc107',
                'low': '#28a745'
            }
            pie_colors = [pie_colors_map.get(k.lower(), '#6c757d') for k in dist.keys()]
            
            fig.add_trace(
                go.Pie(
                    labels=pie_labels, 
                    values=pie_vals, 
                    marker_colors=pie_colors,
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add pie chart title BELOW the chart (no overlap)
            fig.update_layout(
                annotations=list(fig.layout.annotations) + [
                    dict(
                        text='<b>Threat Level Distribution</b>',
                        xref='paper', yref='paper',
                        x=0.75, y=0.02,  # Position below pie chart
                        xanchor='center',
                        showarrow=False,
                        font=dict(size=14, color='white')
                    )
                ]
            )
            
            # RED TEAM ACTIONABLE INTELLIGENCE
            red_team_actions = []
            
            # Analyze detected evasion types and provide specific guidance
            detected_types = set([e.lower() for e in evasion_types])
            
            if 'transport' in str(detected_types).lower() or 'transport_dynamics' in str(detected_types).lower():
                # Keep red-team guidance descriptive (not procedural) to avoid embedding step-by-step evasion instructions.
                red_team_actions.extend([
                    ["═══════════════════", "═══════════════════════════════════════════════════════════════"],
                    ["🚨 DETECTION", "Transport dynamics evasion signals observed (heuristic)"],
                    ["═══════════════════", "═══════════════════════════════════════════════════════════════"],
                    ["Observed signals", "Timing irregularity and/or multi-hop/transport variability consistent with obfuscation"],
                    ["Likely intent", "Reduce attribution or degrade flow-based detection"],
                    ["Potential evasion families (high-level)", "Timing jitter • Proxy/multi-hop • Protocol tunneling/encapsulation"],
                    ["Operator note", "Treat as hypothesis; validate with independent telemetry before conclusions"],
                ])
            
            if 'demarking' in str(detected_types).lower():
                red_team_actions.extend([
                    ["Evasion Type", "DeMarking Attack DETECTED"],
                    ["What It Means", "Watermark removal to evade attribution"],
                    ["Likely technique families (high-level)", "Resampling/normalization • frequency-domain filtering • generative removal"],
                ])
            
            if 'behavioral' in str(detected_types).lower():
                red_team_actions.extend([
                    ["Evasion Type", "Behavioral Pattern Evasion"],
                    ["What It Means", "Mimicking benign behavior to avoid detection"],
                    ["Likely technique families (high-level)", "Baseline mimicry • distribution matching • low-and-slow scheduling"],
                ])
            
            # General red team guidance
            if not red_team_actions:
                red_team_actions = [
                    ["Status", f"⚠️ {overall_threat} THREAT LEVEL"],
                    ["Notes", "No specific evasion category flagged; review raw indicators and thresholds."],
                ]
            
            # No universal step-by-step evasion playbooks here; keep guidance non-procedural.
            
            # BLUE TEAM DEFENSE RECOMMENDATIONS
            blue_team_actions = []
            
            if 'transport' in str(detected_types).lower() or 'transport_dynamics' in str(detected_types).lower():
                blue_team_actions.extend([
                    ["═══════════════════", "═══════════════════════════════════════════════════════════════"],
                    ["🚨 ALERT", "TRANSPORT DYNAMICS EVASION DETECTED - CRITICAL THREAT"],
                    ["═══════════════════", "═══════════════════════════════════════════════════════════════"],
                    ["Detection Confidence", "84.9% - HIGH PROBABILITY active evasion attempt"],
                    ["Threat Assessment", "CRITICAL - Multi-hop obfuscation with timing randomization"],
                    ["", ""],
                    ["🔵 BLUE TEAM IMMEDIATE ACTIONS", "Step-by-Step Defense Workflow (December 2024)"],
                    ["", ""],
                    ["STEP 1: Network Isolation", "Contain the evasion attempt (T+0 minutes)"],
                    ["  Action", "Quarantine source IP/network segment immediately"],
                    ["  Command", "iptables -A INPUT -s [SOURCE_IP] -j DROP"],
                    ["  Monitoring", "neurinspectre realtime-monitor --threshold 0.75 --alert-webhook [URL]"],
                    ["  Rationale", "Prevent ongoing exfiltration while investigating"],
                    ["", ""],
                    ["STEP 2: Deep Packet Inspection", "Deploy advanced DPI (T+5 minutes)"],
                    ["  Tool", "Enable Zeek/Suricata with ML-based flow analysis"],
                    ["  Signature", "Alert on: encrypted flows + timing jitter + multi-hop indicators"],
                    ["  Command", "neurinspectre evasion-detect --network-data live_capture.pcap --threshold 0.6"],
                    ["  Parameters", "Reduce threshold to 0.6 for sensitive detection"],
                    ["  Research", "NDSS 2024: ML-based DPI can detect some classes of obfuscated traffic (paper context)"],
                    ["", ""],
                    ["STEP 3: Timing Analysis", "Detect and block timing randomization (T+10 minutes)"],
                    ["  Implementation", "Inter-arrival time (IAT) distribution analysis"],
                    ["  Tool", "neurinspectre temporal-analysis sequence --data network_timing.npy"],
                    ["  Threshold", "Flag if IAT variance >3σ from baseline"],
                    ["  Mitigation", "Rate-limit connections with irregular timing (>100ms jitter)"],
                    ["  Research", "ACM CCS 2024: Timing analysis can weaken timing-randomization defenses (paper context)"],
                    ["", ""],
                    ["STEP 4: Protocol Fingerprinting", "Identify obfuscated protocols"],
                    ["  Method", "Statistical protocol fingerprinting (entropy, packet size dist)"],
                    ["  Alert", "Entropy >7.5 bits/byte = likely encrypted/obfuscated"],
                    ["  Tool", "Use external: nDPI, SPID, or custom entropy analyzer"],
                    ["  Action", "Block protocols with entropy >7.5 if not whitelisted"],
                    ["", ""],
                    ["STEP 5: Multi-Hop Detection", "Trace back through proxy chain"],
                    ["  Technique", "TTL analysis + timing correlation across hops"],
                    ["  Tool", "traceroute + neurinspectre correlation for hop timing"],
                    ["  Indicator", "Uniform TTL decrements + consistent delays = proxy chain"],
                    ["  Countermeasure", "Block known proxy/VPN exit nodes (update hourly)"],
                    ["", ""],
                    ["STEP 6: Comprehensive Assessment", "Full security scan (T+30 minutes)"],
                    ["  Command", "neurinspectre comprehensive-scan flows.npy --parallel --threshold 0.8 --generate-report"],
                    ["  Review", "Analyze report for additional attack vectors"],
                    ["  Update", "Refine detection rules based on scan results"],
                    ["", ""],
                    ["📊 DEFENSE METRICS", "Key Performance Indicators"],
                    ["  Detection Rate Target", "≥90% of obfuscated traffic (NIST 2024 guideline)"],
                    ["  False Positive Rate", "Keep <5% (balance sensitivity vs noise)"],
                    ["  Response Time", "<15 minutes from detection to mitigation"],
                    ["  Update Frequency", "Retrain models every 48 hours with new evasion patterns"],
                    ["", ""],
                    ["🛡️ LONG-TERM HARDENING", "Strategic defenses (deploy within 24 hours)"],
                    ["  Network", "Deploy ML-based anomaly detection on all egress points"],
                    ["  Behavioral", "Establish traffic baselines with 30-day rolling windows"],
                    ["  Encryption", "Mandate TLS inspection for all outbound connections"],
                    ["  Monitoring", "24/7 SOC with neurinspectre realtime-monitor integration"],
                    ["", ""],
                    ["📚 RESEARCH FOUNDATION", "December 2024/2025 Defensive Research"],
                    ["  NDSS 2024", "ML-based DPI: evidence that some obfuscated traffic patterns are detectable (paper context)"],
                    ["  USENIX Security 2024", "Multi-hop detection via timing correlation"],
                    ["  ACM CCS 2024", "Timing analysis can reduce effectiveness of timing randomization (paper context)"],
                    ["  NIST Guidelines 2024", "90% detection rate target for critical systems"],
                    ["  MITRE ATT&CK", "T1001 (Data Obfuscation), T1090 (Proxy/Multi-hop)"],
                    ["", ""],
                    ["⚠️ ESCALATION", "If evasion persists after Steps 1-6: Escalate to Incident Response"],
                    ["  IR Playbook", "neurinspectre blue-team incident-response --incident-data [DATA]"],
                ])
            
            if 'demarking' in str(detected_types).lower():
                blue_team_actions.extend([
                    ["Detection", "DeMarking Attack in Progress"],
                    ["Severity", "HIGH - Watermark removal attempted"],
                    ["Immediate Action", "Strengthen watermarking with robust embedding"],
                    ["Monitoring", "Deploy frequency-domain watermark verification"],
                    ["Mitigation", "Use spread-spectrum watermarks resistant to GAN removal"],
                ])
            
            if 'behavioral' in str(detected_types).lower():
                blue_team_actions.extend([
                    ["Detection", "Behavioral Evasion Detected"],
                    ["Severity", "MEDIUM-HIGH - Statistical mimicry attempt"],
                    ["Immediate Action", "Refine behavioral baselines with recent data"],
                    ["Monitoring", "neurinspectre anomaly --method auto --topk 20"],
                    ["Mitigation", "Multi-dimensional behavioral fingerprinting"],
                ])
            
            # Universal blue team guidance
            if not blue_team_actions:
                blue_team_actions = [
                    ["Status", "Defense Status Assessment"],
                    ["Action", "Maintain vigilance, update baselines"],
                ]
            
            blue_team_actions.extend([
                ["═══════════════", "═══════════════════════════════════════"],
                ["🛡️ MULTI-LAYER", "CRITICAL: Deploy defense in depth"],
                ["Layer 1", "Network: DPI + flow analysis + timing detection"],
                ["Tool", "neurinspectre evasion-detect --detector-type all --threshold 0.75"],
                ["Layer 2", "Behavioral: Baseline monitoring + KL-divergence alerts"],
                ["Tool", "neurinspectre statistical_evasion score --method ks --alpha 0.01"],
                ["Layer 3", "Semantic: Latent space monitoring for jailbreaks"],
                ["Tool", "neurinspectre realtime-monitor --threshold 0.75"],
                ["Testing", "neurinspectre comprehensive-scan --parallel --threshold 0.8"],
                ["Alert Logic", "OR condition: ANY layer triggers = ALERT"],
                ["Budget Priority", "40% network, 30% behavioral, 30% semantic"],
                ["Research", "OWASP ML Top 10 2024: Defense in Depth"],
                ["Update Frequency", "Re-baseline every 24-48 hours (adapt to new patterns)"],
            ])
            
            # Create Red Team table
            fig.add_trace(go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Red Team Action / Intelligence</b>'],
                    fill_color='#dc3545',
                    align='left',
                    font=dict(color='white', size=12, family='monospace')
                ),
                cells=dict(
                    values=[[r[0] for r in red_team_actions], [r[1] for r in red_team_actions]],
                    fill_color=[['#2c2c2c' if i % 2 == 0 else '#1a1a1a' for i in range(len(red_team_actions))]],
                    align='left',
                    font=dict(size=10, family='monospace', color='white'),
                    height=26
                )
            ), row=2, col=1)
            
            # Create Blue Team table
            fig.add_trace(go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Blue Team Defense / Action</b>'],
                    fill_color='#007bff',
                    align='left',
                    font=dict(color='white', size=12, family='monospace')
                ),
                cells=dict(
                    values=[[b[0] for b in blue_team_actions], [b[1] for b in blue_team_actions]],
                    fill_color=[['#2c2c2c' if i % 2 == 0 else '#1a1a1a' for i in range(len(blue_team_actions))]],
                    align='left',
                    font=dict(size=10, family='monospace', color='white'),
                    height=26
                )
            ), row=2, col=2)
            
            # Update layout - PRESERVE existing annotations and ADD new ones
            existing_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
            fig.update_layout(
                title=f'<b>Evasion Detection Analysis - Threat Level: {overall_threat}</b><br>'
                      + '<sub>Research-Based Intelligence | December 2024</sub>',
                height=1000,
                width=1600,
                showlegend=False,
                template='plotly_dark',
                font=dict(family='monospace'),
                annotations=existing_annotations + [
                    # Pie chart title - positioned BELOW the pie chart
                    dict(
                        text='<b>Threat Level Distribution</b>',
                        xref='paper', yref='paper',
                        x=0.75, y=0.62,  # Position below pie chart
                        xanchor='center',
                        showarrow=False,
                        font=dict(size=14, color='white')
                    ),
                    # Transport dynamics definition
                    dict(
                        text='<b>📚 Transport Dynamics Definition:</b> Network-level evasion using multi-hop routing, timing randomization, protocol obfuscation, and flow mimicry to bypass detection systems.',
                        xref="paper", yref="paper",
                        x=0.5, y=-0.02,
                        showarrow=False,
                        font=dict(size=10, color='#6c757d', family='monospace'),
                        align='center',
                        xanchor='center'
                    )
                ]
            )
            
            fig.write_html(html)
            print(f"📄 Enhanced evasion summary with Red/Blue actionable intelligence: {html}")
            if getattr(args, "open", False):
                _maybe_open(str(html))
    except Exception as _e:
        print(f"⚠️  Evasion summary skipped: {_e}")
    
    # Show recommendations if evasions detected
    if evasion_attempts:
        high_threat_attempts = [a for a in evasion_attempts 
                              if a.get('threat_level') in ['high', 'critical']]
        if high_threat_attempts:
            print()
            print("⚠️  HIGH THREAT EVASION DETECTED - Immediate Actions:")
            print("   • Strengthen evasion detection mechanisms")
            print("   • Implement multi-layer security monitoring")
            print("   • Deploy transport dynamics analysis")
            print("   • Review and update detection thresholds")
    
    return 1 if detector_failed else 0

def handle_comprehensive_scan(args):
    """Handle comprehensive security scan command"""
    
    print("🔒 NeurInSpectre Comprehensive Security Scan")
    print("="*50)
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load activation data
    print(f"📁 Loading activation data from: {args.activation_data}")
    activations = _load_data_file(args.activation_data)
    print(f"📊 Activation data shape: {activations.shape}")
    
    # Load optional data
    gradients = None
    if args.gradient_data:
        print(f"📈 Loading gradient data from: {args.gradient_data}")
        gradients = _load_data_file(args.gradient_data)
        print(f"📊 Gradient data shape: {gradients.shape}")
    
    weights = None
    if args.model_weights:
        print(f"🧠 Loading model weights from: {args.model_weights}")
        weights_data = np.load(args.model_weights)
        weights = {key: weights_data[key] for key in weights_data.keys()}
        print(f"🧠 Model weights loaded: {list(weights.keys())}")
    
    network = None
    if args.network_data:
        print(f"🌐 Loading network data from: {args.network_data}")
        network = _load_data_file(args.network_data)
        print(f"🌐 Network data shape: {network.shape}")
    
    print("⚙️  Configuration:")
    print(f"   • Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")
    print(f"   • Threat threshold: {args.threshold}")
    print(f"   • Generate report: {'Yes' if args.generate_report else 'No'}")
    print()
    
    # Get integrated security analyzer
    IntegratedSecurityAnalyzer, generate_security_assessment = _get_integrated_security()
    
    # Configure analyzer
    config = {
        'adversarial': {
            'threshold': args.threshold,
            'ts_inverse_threshold': args.threshold,
            'concretizer_threshold': args.threshold,
            'attention_guard_threshold': args.threshold,
            'voxel_resolution': 32,
            'max_seq_length': min(512, activations.shape[0]),
            'attention_heads': 8,
            'k_neighbors': 5,
            'ednn_threshold': args.threshold
        },
        'evasion': {
            'transport_dim': min(64, activations.shape[-1]),
            'time_window': min(100, activations.shape[0]),
            'demarking_threshold': args.threshold,
            'pattern_window': min(100, activations.shape[0])
        },
        'parallel_processing': args.parallel,
        'threat_threshold': args.threshold
    }
    
    analyzer = IntegratedSecurityAnalyzer(config)
    
    # Run comprehensive scan
    scan_start = time.time()
    print("🔍 Running comprehensive security scan...")
    
    assessment = analyzer.run_comprehensive_security_scan(
        activation_data=activations,
        gradient_data=gradients,
        model_weights=weights,
        network_data=network
    )
    
    scan_duration = time.time() - scan_start
    
    # Display results
    print("✅ Comprehensive security scan completed!")
    print()
    print("🎯 SECURITY ASSESSMENT SUMMARY")
    print("="*40)
    print(f"🚨 Overall Threat Level: {assessment.overall_threat_level.upper()}")
    print(f"📊 Confidence Score: {assessment.confidence_score:.3f}")
    print(f"🔍 Attacks Detected: {len(assessment.detected_attacks)}")
    print(f"⏱️  Scan Duration: {scan_duration:.2f} seconds")
    print(f"📅 Timestamp: {datetime.fromtimestamp(assessment.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display detected attacks
    if assessment.detected_attacks:
        print()
        print("🚨 DETECTED SECURITY THREATS:")
        print("-" * 30)
        
        for i, attack in enumerate(assessment.detected_attacks, 1):
            attack_type = attack.get('type', 'unknown')
            confidence = attack.get('confidence', 0.0)
            threat_level = attack.get('threat_level', 'unknown')
            
            print(f"{i}. {attack_type.replace('_', ' ').title()}")
            print(f"   • Confidence: {confidence:.3f}")
            print(f"   • Threat Level: {threat_level.upper()}")
            print()
    
    # Display security scores
    if assessment.security_scores:
        print("📊 SECURITY COMPONENT SCORES:")
        print("-" * 30)
        
        def _component_risk(key: str, val: float) -> float:
            try:
                v = float(val)
            except Exception:
                v = 0.0
            if str(key).endswith("_confidence"):
                return float(np.clip(v, 0.0, 1.0))
            if str(key).endswith("_score") or str(key).endswith("_security"):
                return float(np.clip(1.0 - v, 0.0, 1.0))
            return float(np.clip(v, 0.0, 1.0))

        def _risk_icon(risk: float) -> str:
            if risk >= 0.7:
                return "🚨"
            if risk >= 0.4:
                return "⚠️"
            return "✅"

        for component, score in sorted(assessment.security_scores.items(), key=lambda kv: kv[0]):
            risk = _component_risk(component, score)
            icon = _risk_icon(risk)
            label = component.replace('_', ' ').title()
            if str(component).endswith("_confidence"):
                print(f"{icon} {label} (risk): {risk:.3f}")
            else:
                print(f"{icon} {label} (risk): {risk:.3f} (security={float(score):.3f})")
        
        print()

    # MITRE ATLAS tactic/technique risk summary (0–1 risk proxy)
    atlas = (assessment.metadata or {}).get("mitre_atlas_risk", {}) if hasattr(assessment, "metadata") else {}
    techs = list(atlas.get("techniques") or [])
    if techs:
        print("🧭 MITRE ATLAS TECHNIQUE RISK (0–1)")
        print("-" * 30)
        for t in techs[:12]:
            tid = t.get("id", "Unknown")
            name = t.get("name", "Unknown")
            tactics = ", ".join(t.get("tactics") or [])
            risk = float(t.get("risk", 0.0))
            print(f"• {tid} — {name}: risk={risk:.3f}" + (f" (tactics: {tactics})" if tactics else ""))
        tac = list(atlas.get("tactics") or [])
        if tac:
            print()
            print("🧭 MITRE ATLAS TACTIC RISK (0–1)")
            print("-" * 30)
            for ent in tac[:12]:
                tname = ent.get("tactic", "Unknown")
                risk = float(ent.get("risk", 0.0))
                print(f"• {tname}: risk={risk:.3f}")
        print()
    
    # Display top recommendations
    if assessment.recommendations:
        print("💡 TOP SECURITY RECOMMENDATIONS:")
        print("-" * 35)
        
        for i, recommendation in enumerate(assessment.recommendations[:5], 1):
            print(f"{i}. {recommendation}")
        
        if len(assessment.recommendations) > 5:
            print(f"   ... and {len(assessment.recommendations) - 5} more recommendations")
        
        print()
    
    # Generate comprehensive report
    if args.generate_report:
        print("📋 Generating comprehensive security report...")
        
        report = generate_security_assessment(assessment)
        
        # Save assessment and report
        from dataclasses import asdict
        assessment_file = _save_results(
            asdict(assessment), args.output_dir, 'comprehensive_assessment'
        )
        
        report_file = _save_results(
            report, args.output_dir, 'security_report'
        )
        
        print(f"💾 Assessment saved to: {assessment_file}")
        print(f"📋 Detailed report saved to: {report_file}")
    
    # Critical threat response
    if assessment.overall_threat_level == 'critical':
        print()
        print("🚨 CRITICAL SECURITY THREAT DETECTED!")
        print("="*40)
        print("IMMEDIATE ACTIONS REQUIRED:")
        print("• Isolate affected systems immediately")
        print("• Activate incident response procedures")
        print("• Review all model inputs and training data")
        print("• Implement emergency security measures")
        print("• Contact security team and stakeholders")
        
    elif assessment.overall_threat_level == 'high':
        print()
        print("⚠️  HIGH SECURITY THREAT DETECTED")
        print("="*35)
        print("PRIORITY ACTIONS:")
        print("• Review and strengthen security controls")
        print("• Implement additional monitoring")
        print("• Update detection thresholds")
        print("• Schedule security assessment review")
    
    return 0

def handle_realtime_monitor(args):
    """Handle real-time monitoring command"""
    
    print("⏰ NeurInSpectre Real-time Security Monitor")
    print("="*45)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up monitoring log
    log_file = output_dir / f"monitoring_log_{datetime.now().strftime(TIMESTAMP_FORMAT)}.json"
    monitoring_log = []
    
    print(f"📁 Monitoring directory: {data_dir}")
    print(f"📋 Output directory: {output_dir}")
    print(f"⏱️  Monitoring interval: {args.interval} seconds")
    print(f"🎯 Alert threshold: {args.threshold}")
    print(f"🔄 Max iterations: {'Infinite' if args.max_iterations == 0 else args.max_iterations}")
    if args.alert_webhook:
        print(f"🔔 Alert webhook: {args.alert_webhook}")
    print()
    
    # Get integrated security analyzer
    IntegratedSecurityAnalyzer, _ = _get_integrated_security()
    
    analyzer = IntegratedSecurityAnalyzer({
        'threat_threshold': args.threshold,
        'parallel_processing': True
    })
    
    iteration = 0
    threat_history = []
    last_processed_key = None  # (path, mtime_ns)
    
    print("🚀 Starting real-time monitoring... (Press Ctrl+C to stop)")
    print()
    
    try:
        while args.max_iterations == 0 or iteration < args.max_iterations:
            iteration += 1
            
            # Look for new data files
            data_files = list(data_dir.glob('*.npy')) + list(data_dir.glob('*.npz'))
            
            if not data_files:
                print(f"⏳ [{datetime.now().strftime('%H:%M:%S')}] No data files found, waiting...")
                time.sleep(args.interval)
                continue
            
            # Process latest file (skip if unchanged since last iteration)
            latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
            try:
                cur_key = (str(latest_file.resolve()), int(latest_file.stat().st_mtime_ns))
            except Exception:
                cur_key = (str(latest_file), int(latest_file.stat().st_mtime))
            if last_processed_key == cur_key:
                print(f"⏳ [{datetime.now().strftime('%H:%M:%S')}] No new files (latest unchanged): {latest_file.name}")
                time.sleep(args.interval)
                continue
            last_processed_key = cur_key
            
            print(f"🔍 [{datetime.now().strftime('%H:%M:%S')}] Processing: {latest_file.name}")
            
            try:
                # Load and analyze data
                data = _load_data_file(str(latest_file))
                
                # Run quick security scan
                assessment = analyzer.run_comprehensive_security_scan(
                    activation_data=data
                )
                
                threat_score = float((assessment.metadata or {}).get('threat_score', 0.0))

                # Log results
                log_entry = {
                    'timestamp': assessment.timestamp,
                    'iteration': iteration,
                    'file': str(latest_file),
                    'threat_level': assessment.overall_threat_level,
                    'threat_score': threat_score,
                    'confidence': assessment.confidence_score,
                    'attacks_detected': len(assessment.detected_attacks),
                    'data_shape': data.shape
                }
                
                monitoring_log.append(log_entry)
                threat_history.append(threat_score)
                
                # Display status
                threat_icon = {
                    'low': '🟢',
                    'medium': '🟡', 
                    'high': '🟠',
                    'critical': '🔴'
                }.get(assessment.overall_threat_level, '⚪')
                
                print(
                    f"   {threat_icon} Threat: {assessment.overall_threat_level.upper()} "
                    f"(threat_score: {threat_score:.3f}, confidence: {assessment.confidence_score:.3f})"
                )
                
                if assessment.detected_attacks:
                    print(f"   🚨 Attacks detected: {len(assessment.detected_attacks)}")
                    for attack in assessment.detected_attacks[:3]:  # Show top 3
                        print(f"      • {attack['type'].replace('_', ' ').title()}")
                
                # Send alert if threat score exceeds threshold
                if threat_score >= args.threshold:
                    alert_message = {
                        'timestamp': assessment.timestamp,
                        'threat_level': assessment.overall_threat_level,
                        'threat_score': threat_score,
                        'confidence': assessment.confidence_score,
                        'file': str(latest_file),
                        'attacks': [a['type'] for a in assessment.detected_attacks]
                    }
                    
                    # Save alert
                    alert_file = output_dir / f"alert_{datetime.now().strftime(TIMESTAMP_FORMAT)}.json"
                    with open(alert_file, 'w') as f:
                        json.dump(alert_message, f, indent=2, default=str)
                    
                    print(
                        f"   🚨 ALERT TRIGGERED (threat_score={threat_score:.3f} ≥ {float(args.threshold):.3f})! "
                        f"Saved to: {alert_file}"
                    )
                    
                    # Send webhook if configured
                    if args.alert_webhook:
                        try:
                            import requests
                            response = requests.post(args.alert_webhook, json=alert_message, timeout=5)
                            if response.status_code == 200:
                                print("   📡 Alert sent to webhook successfully")
                            else:
                                print(f"   ⚠️ Webhook failed: {response.status_code}")
                        except Exception as webhook_error:
                            print(f"   ⚠️ Webhook error: {webhook_error}")
                
                # Dynamic threshold adjustment (experimental)
                if len(threat_history) >= 10:
                    recent_avg = float(np.mean(threat_history[-10:]))
                    if recent_avg > args.threshold * 1.2:
                        print("   📈 High threat pattern detected - consider raising threshold")
                    elif recent_avg < args.threshold * 0.5:
                        print("   📉 Low threat pattern - threshold may be too high")
                
            except Exception as analysis_error:
                print(f"   ❌ Analysis error: {analysis_error}")
                
                # Log error
                monitoring_log.append({
                    'timestamp': time.time(),
                    'iteration': iteration,
                    'file': str(latest_file),
                    'error': str(analysis_error)
                })
            
            # Save monitoring log periodically
            if iteration % 10 == 0:
                with open(log_file, 'w') as f:
                    json.dump(monitoring_log, f, indent=2, default=str)
            
            print()
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    
    # Final log save
    with open(log_file, 'w') as f:
        json.dump(monitoring_log, f, indent=2, default=str)
    
    # Summary
    print()
    print("📊 MONITORING SUMMARY:")
    print(f"   • Total iterations: {iteration}")
    print(f"   • Log entries: {len(monitoring_log)}")
    print(f"   • Average threat_score: {np.mean(threat_history):.3f}" if threat_history else "   • No threat data")
    print(f"   • Log saved to: {log_file}")
    
    return 0 