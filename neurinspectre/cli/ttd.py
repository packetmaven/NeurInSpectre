#!/usr/bin/env python3
"""
TTD (Time to Detection) Dashboard for NeurInSpectre
Real-time security monitoring for ML model attacks
"""

import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Lazy loading functions for heavy dependencies
def _get_plotly():
    """Lazy import plotly to avoid import errors if not installed"""
    try:
        import plotly.graph_objects as go
        from plotly import subplots as sp
        from plotly.offline import plot
        return go, sp, plot
    except ImportError as e:
        raise ImportError(
            "plotly is required for the TTD dashboard. Install with: pip install plotly"
        ) from e

def _get_pandas():
    """Lazy import pandas to avoid import errors if not installed"""
    try:
        import pandas as pd
        return pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for the TTD dashboard. Install with: pip install pandas"
        ) from e

def _get_numpy():
    """Lazy import numpy"""
    try:
        import numpy as np
        return np
    except ImportError as e:
        raise ImportError(
            "numpy is required. Install with: pip install numpy"
        ) from e

def _get_dash():
    """Lazy import dash components for interactive dashboard"""
    try:
        import dash
        from dash import dcc, html, Input, Output
        import dash_bootstrap_components as dbc
        return dash, dcc, html, Input, Output, dbc
    except ImportError:
        print("Error: dash is required for interactive TTD dashboard. Install with: pip install dash dash-bootstrap-components")
        return None, None, None, None, None, None

class TTDDashboard:
    def __init__(self, model_name="distilbert-base-uncased",
                 privacy_budget_limit=3.0, gradient_norm_threshold=5.0,
                 mi_threshold=0.8, reconstruction_threshold=0.7,
                 dp_noise_multiplier=None, dp_sample_rate=None, dp_delta=1e-5,
                 allow_simulated=False):
        self.go, self.sp, self.plot = _get_plotly()
        self.pd = _get_pandas()
        import numpy as np
        self.np = np
        
        # Model configuration
        self.model_name = model_name
        self.model_layers = self._get_model_layers(model_name)
        
        # Configurable security thresholds
        self.PRIVACY_BUDGET_LIMIT = privacy_budget_limit
        self.CRITICAL_GRADIENT_NORM = gradient_norm_threshold
        self.MEMBERSHIP_INFERENCE_THRESHOLD = mi_threshold
        self.RECONSTRUCTION_RISK_THRESHOLD = reconstruction_threshold

        # Differential privacy (DP-SGD) accounting (NO SIMULATION unless parameters provided)
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_sample_rate = dp_sample_rate
        self.dp_delta = dp_delta
        self.allow_simulated = allow_simulated

        # Optional precomputed ε series loaded from file (must align with gradient steps)
        self.real_privacy_epsilon = None
        self.privacy_metric = "unavailable"  # dp_epsilon | file_epsilon | unavailable
        # Caches for expensive computations (invalidated on new data load)
        self._cached_gradient_norms_full = None
        self._cached_gradient_norms_shape = None
        self._cached_privacy_series = None
        self._cached_privacy_key = None

        # Active dataset source for UI + reporting
        self.active_data_source = 'real'  # real | uploaded | live_model | simulated
        self._data_version = 0

        # Cache the MITRE ATLAS bubble timeline (expensive to compute)
        self._timeline_cache_key = None
        self._timeline_cache_fig = None
        self._timeline_cache_at = 0.0

        
        # Attack timing windows
        self.current_time = datetime.now()
        self.attack_start = self.current_time - timedelta(hours=2)
        
        # Real data storage
        self.real_gradient_data = None
        self.real_attention_data = None
        self.real_token_data = None
        
    def _get_model_layers(self, model_name):
        """Get layer configuration based on model type - INSTANT STARTUP"""
        # Use static config only for instant startup - no network calls
        model_configs = {
            "distilbert-base-uncased": {"num_layers": 6, "attention_heads": 12, "hidden_size": 768},
            "bert-base-uncased": {"num_layers": 12, "attention_heads": 12, "hidden_size": 768},
            "gpt2": {"num_layers": 12, "attention_heads": 12, "hidden_size": 768},
            "roberta-base": {"num_layers": 12, "attention_heads": 12, "hidden_size": 768},
            "t5-base": {"num_layers": 12, "attention_heads": 12, "hidden_size": 768}
        }
        return model_configs.get(model_name, model_configs["distilbert-base-uncased"])
    
    def _load_actual_model_config(self, model_name):
        """Load actual model configuration from HuggingFace - ASYNC SAFE VERSION"""
        try:
            # Quick check if model is in local cache first (fast)
            from transformers import AutoConfig
            
            # Try to load config without downloading (cache_dir check)
            try:
                config = AutoConfig.from_pretrained(model_name, local_files_only=True)
                print(f"✅ CACHED {model_name} config loaded (fast)")
            except Exception:
                # If not in cache, load normally but with minimal output
                config = AutoConfig.from_pretrained(model_name)
                print(f"✅ {model_name} config downloaded")
            
            # Extract actual configuration based on model type
            actual_config = {
                "num_layers": getattr(config, 'num_hidden_layers', None) or 
                             getattr(config, 'num_layers', None) or 
                             getattr(config, 'n_layer', None) or 12,
                "attention_heads": getattr(config, 'num_attention_heads', None) or 
                                 getattr(config, 'num_heads', None) or 
                                 getattr(config, 'n_head', None) or 12,
                "hidden_size": getattr(config, 'hidden_size', None) or 
                              getattr(config, 'd_model', None) or 
                              getattr(config, 'n_embd', None) or 768,
                "model_type": getattr(config, 'model_type', 'unknown'),
                "vocab_size": getattr(config, 'vocab_size', 'unknown'),
                "is_encoder_decoder": getattr(config, 'is_encoder_decoder', False)
            }
            
            return actual_config
                
        except Exception as e:
            print(f"⚠️ Could not load config for {model_name}: {e}")
            raise e
    
    def _get_static_model_config(self, model_name):
        """Fallback static configuration"""
        model_configs = {
            "distilbert-base-uncased": {
                "num_layers": 6,
                "attention_heads": 12,
                "hidden_size": 768
            },
            "bert-base-uncased": {
                "num_layers": 12,
                "attention_heads": 12,
                "hidden_size": 768
            },
            "gpt2": {
                "num_layers": 12,
                "attention_heads": 12,
                "hidden_size": 768
            },
            "roberta-base": {
                "num_layers": 12,
                "attention_heads": 12,
                "hidden_size": 768
            },
            "t5-base": {
                "num_layers": 12,
                "attention_heads": 12,
                "hidden_size": 768
            }
        }
        return model_configs.get(model_name, model_configs["distilbert-base-uncased"])
    
    def load_real_data(self, gradient_file=None, attention_file=None, token_file=None, batch_dir=None, privacy_file=None):
        """Load real gradient, attention, and token data"""
        print(f"🔍 Loading real data for model: {self.model_name}")
        
        import numpy as np

        # Invalidate caches (new data sources may change shapes/lengths)
        self._cached_gradient_norms_full = None
        self._cached_gradient_norms_shape = None
        self._cached_privacy_series = None
        self._cached_privacy_key = None

        # Active dataset source for UI + reporting
        self.active_data_source = 'real'  # real | uploaded | live_model | simulated
        self._data_version = 0

        # Cache the MITRE ATLAS bubble timeline (expensive to compute)
        self._timeline_cache_key = None
        self._timeline_cache_fig = None
        self._timeline_cache_at = 0.0
        
        # Data source precedence:
        # - If an explicit gradient file is provided and exists, ALWAYS use it.
        # - Otherwise, optionally load a batch directory (useful for multi-file captures).
        have_gradient_file = bool(gradient_file and os.path.exists(gradient_file))
        batch_data_loaded = False
        if (not have_gradient_file) and batch_dir and os.path.exists(batch_dir):
            print(f"📂 Loading batch data from: {batch_dir}")
            self._load_batch_data(batch_dir)
            if self.real_gradient_data is not None:
                batch_data_loaded = True
                print(f"   ✅ Batch gradient data loaded successfully")
        
        # Load gradient data with enhanced processing (only if batch didn't load)
        if gradient_file and os.path.exists(gradient_file) and not batch_data_loaded:
            print(f"📊 Loading gradient data: {gradient_file}")
            try:
                loaded_data = np.load(gradient_file)
                loaded_data = np.array(loaded_data)
                print(f"   🔍 Raw loaded data shape: {loaded_data.shape}")
                print(f"   🔍 Raw loaded data type: {type(loaded_data)}")

                # Handle different data formats (NO SIMULATION)
                # - Empty/scalar inputs are rejected
                # - 1D arrays are interpreted as a per-step scalar series (N×1)
                if loaded_data.size == 0 or loaded_data.ndim == 0:
                    raise ValueError(
                        "Gradient file is empty or scalar; provide a non-empty array (1D series or 2D+ gradients)."
                    )

                if loaded_data.ndim == 1:
                    print(
                        f"   🔧 1D array detected ({loaded_data.shape[0]} elements) - interpreting as per-step scalar series"
                    )
                    self.real_gradient_data = loaded_data.reshape(-1, 1)
                elif loaded_data.ndim == 2:
                    # Each row is one step/sample
                    self.real_gradient_data = loaded_data
                else:
                    # Flatten to [N, D] for downstream norm computations
                    self.real_gradient_data = loaded_data.reshape(loaded_data.shape[0], -1)

                print(f"   ✅ Final gradient data shape: {self.real_gradient_data.shape}")
                print(f"   📊 Total samples available: {self.real_gradient_data.shape[0]}")

                
            except Exception as e:
                print(f"   ❌ Error loading gradient data: {e}")
                self.real_gradient_data = None
        
        # Load attention data  
        if attention_file and os.path.exists(attention_file):
            print(f"🎯 Loading attention data: {attention_file}")
            try:
                self.real_attention_data = np.load(attention_file)
                print(f"   ✅ Loaded attention data shape: {self.real_attention_data.shape}")
            except Exception as e:
                print(f"   ❌ Error loading attention data: {e}")
                self.real_attention_data = None
        
        # Load token data
        if token_file and os.path.exists(token_file):
            print(f"�� Loading token data: {token_file}")
            try:
                with open(token_file, 'r') as f:
                    self.real_token_data = f.read().strip().split()
                print(f"   ✅ Loaded {len(self.real_token_data)} tokens")
            except Exception as e:
                print(f"   ❌ Error loading token data: {e}")
                self.real_token_data = None
    

        # Load privacy epsilon series (optional). This should be a 1D array of ε values aligned to gradient steps.
        if privacy_file and os.path.exists(privacy_file):
            try:
                arr = np.load(privacy_file)
                arr = np.array(arr)
                if arr.ndim == 0:
                    self.real_privacy_epsilon = float(arr)
                    print(f"   🔒 Loaded scalar privacy ε from {privacy_file}")
                else:
                    self.real_privacy_epsilon = arr.reshape(-1)
                    print(f"   🔒 Loaded privacy ε series: {self.real_privacy_epsilon.shape} from {privacy_file}")
                self.privacy_metric = 'file_epsilon'
            except Exception as e:
                print(f"   ❌ Error loading privacy ε file: {e}")

    def _load_batch_data(self, batch_dir):
        """Load batch data from directory containing multiple files"""
        batch_path = Path(batch_dir)
        
        import numpy as np
        
        # Find gradient files
        gradient_files = list(batch_path.glob("*grad*.npy")) + list(batch_path.glob("*gradient*.npy"))
        if gradient_files and self.real_gradient_data is None:
            try:
                # Load and robustly concatenate multiple gradient files with mixed shapes
                grads_list = []
                min_feat = None
                loaded = 0
                for grad_file in gradient_files[:10]:  # Limit to 10 files for performance
                    arr = np.load(grad_file)
                    arr = np.array(arr)
                    if arr.size == 0:
                        continue
                    # Normalize to 2D [N, D]
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    elif arr.ndim > 2:
                        arr = arr.reshape(-1, arr.shape[-1])
                    # Track minimum feature width for alignment
                    min_feat = arr.shape[1] if min_feat is None else min(min_feat, arr.shape[1])
                    grads_list.append(arr)
                    loaded += 1
                if grads_list and min_feat and min_feat > 0:
                    # Align all arrays to same feature width
                    grads_aligned = [g[:, :min_feat] for g in grads_list]
                    self.real_gradient_data = np.concatenate(grads_aligned, axis=0)
                    print(f"   📊 Loaded {loaded} gradient files (aligned to D={min_feat})")
                else:
                    self.real_gradient_data = None
            except Exception as e:
                print(f"   ❌ Error loading batch gradient data: {e}")
        
        # Find attention files
        attention_files = list(batch_path.glob("*attention*.npy")) + list(batch_path.glob("*attn*.npy"))
        if attention_files and self.real_attention_data is None:
            try:
                # Load first attention file
                self.real_attention_data = np.load(attention_files[0])
                print(f"   🎯 Loaded attention data from {attention_files[0].name}")
            except Exception as e:
                print(f"   ❌ Error loading batch attention data: {e}")

        # Find privacy epsilon series files (optional)
        privacy_files = (list(batch_path.glob("*privacy*.npy")) +
                         list(batch_path.glob("*epsilon*.npy")) +
                         list(batch_path.glob("*dp*epsilon*.npy")))
        if privacy_files and self.real_privacy_epsilon is None:
            try:
                arr = np.load(privacy_files[0])
                arr = np.array(arr)
                if arr.ndim == 0:
                    self.real_privacy_epsilon = float(arr)
                    print(f"   🔒 Loaded scalar privacy ε from {privacy_files[0].name}")
                else:
                    self.real_privacy_epsilon = arr.reshape(-1)
                    print(f"   🔒 Loaded privacy ε series {self.real_privacy_epsilon.shape} from {privacy_files[0].name}")
                self.privacy_metric = 'file_epsilon'
            except Exception as e:
                print(f"   ❌ Error loading batch privacy ε data: {e}")


    def apply_uploaded_dataset(self, file_paths, *, source_label='uploaded'):
        """Activate uploaded artifacts as the current dashboard dataset.

        Supports:
        - Gradients: .npy/.npz (any filename). Will be normalized to 2D [steps, features].
        - Attention (optional): filename contains 'attn'/'attention' OR 4D tensor.
        - Privacy ε (optional): filename contains 'epsilon'/'privacy'/'eps' OR 1D tensor.

        Notes:
        - For multi-file uploads, all gradient-like tensors are concatenated along steps after
          aligning to the minimum feature width.
        - ε alignment is handled downstream (resampled if needed).
        """
        import numpy as np
        from pathlib import Path

        # Invalidate caches and bump data version
        self._cached_gradient_norms_full = None
        self._cached_gradient_norms_shape = None
        self._cached_privacy_series = None
        self._cached_privacy_key = None
        self._timeline_cache_key = None
        self._timeline_cache_fig = None
        self._timeline_cache_at = 0.0

        self.active_data_source = str(source_label or 'uploaded')
        self._data_version = int(getattr(self, '_data_version', 0)) + 1

        paths = [Path(str(p)) for p in (file_paths or []) if p]
        paths = [p for p in paths if p.exists()]

        def _load_array(fp: Path):
            suf = fp.suffix.lower()
            if suf == '.npy':
                return np.array(np.load(fp, allow_pickle=False))
            if suf == '.npz':
                npz = np.load(fp, allow_pickle=False)
                # Prefer common keys
                for k in ('gradients', 'grads', 'gradient', 'G', 'arr', 'data', 'x', 'X'):
                    if k in npz.files:
                        return np.array(npz[k])
                # Else pick largest
                key = max(npz.files, key=lambda kk: npz[kk].size)
                return np.array(npz[key])
            raise ValueError(f'Unsupported upload type: {fp}')

        attention_paths = []
        epsilon_paths = []
        gradient_paths = []

        for p in paths:
            name = p.name.lower()
            if any(k in name for k in ('attn', 'attention')):
                attention_paths.append(p)
                continue
            if any(k in name for k in ('epsilon', 'privacy', 'eps')):
                epsilon_paths.append(p)
                continue
            gradient_paths.append(p)

        # Load attention (optional)
        used_attention = None
        attention_updated = False
        if attention_paths:
            try:
                used_attention = attention_paths[0]
                self.real_attention_data = _load_array(used_attention)
                attention_updated = True
            except Exception as e:
                print(f"   ❌ Uploaded attention load failed: {e}")

        # Load epsilon (optional)
        used_epsilon = None
        epsilon_updated = False
        if epsilon_paths:
            try:
                used_epsilon = epsilon_paths[0]
                eps_arr = _load_array(used_epsilon)
                if eps_arr.ndim == 0:
                    self.real_privacy_epsilon = float(eps_arr)
                else:
                    self.real_privacy_epsilon = np.array(eps_arr, dtype=float).reshape(-1)
                self.privacy_metric = 'file_epsilon'
                epsilon_updated = True
            except Exception as e:
                print(f"   ❌ Uploaded ε load failed: {e}")

        # Load gradients (required)
        grads = []
        used_grad_files = []
        for p in gradient_paths:
            try:
                arr = _load_array(p)
            except Exception as e:
                print(f"   ❌ Skipping unreadable upload {p.name}: {e}")
                continue
            if arr.size == 0 or arr.ndim == 0:
                continue
            # Heuristic: move largest axis to the front (treat as steps)
            if arr.ndim == 1:
                arr2 = arr.reshape(-1, 1)
            elif arr.ndim == 2:
                arr2 = arr
            else:
                axis = int(np.argmax(arr.shape))
                if axis != 0:
                    arr = np.moveaxis(arr, axis, 0)
                arr2 = arr.reshape(arr.shape[0], -1)
            if arr2.size == 0:
                continue
            grads.append(np.array(arr2))
            used_grad_files.append(p)

        # If we failed to classify, attempt fallback by shape
        if not grads and paths:
            for p in paths:
                if p in attention_paths or p in epsilon_paths:
                    continue
                try:
                    arr = _load_array(p)
                except Exception:
                    continue
                if arr.size == 0 or arr.ndim == 0:
                    continue
                if arr.ndim == 4 and self.real_attention_data is None:
                    self.real_attention_data = arr
                    used_attention = p
                    continue
                if arr.ndim in (0, 1) and self.real_privacy_epsilon is None:
                    self.real_privacy_epsilon = float(arr) if arr.ndim == 0 else np.array(arr).reshape(-1)
                    self.privacy_metric = 'file_epsilon'
                    epsilon_updated = True
                    used_epsilon = p
                    continue
                # Otherwise treat as gradient
                if arr.ndim == 1:
                    arr2 = arr.reshape(-1, 1)
                elif arr.ndim == 2:
                    arr2 = arr
                else:
                    axis = int(np.argmax(arr.shape))
                    if axis != 0:
                        arr = np.moveaxis(arr, axis, 0)
                    arr2 = arr.reshape(arr.shape[0], -1)
                grads.append(np.array(arr2))
                used_grad_files.append(p)

        gradient_updated = False
        if grads:
            min_feat = min(g.shape[1] for g in grads)
            grads_aligned = [g[:, :min_feat] for g in grads]
            self.real_gradient_data = np.concatenate(grads_aligned, axis=0)
            gradient_updated = True
        # If the upload did not include any gradient-like tensors, preserve the existing gradient dataset.

        return {
            'source': self.active_data_source,
            'files': [str(p) for p in paths],
            'gradient_files': [str(p) for p in used_grad_files],
            'attention_file': str(used_attention) if used_attention else None,
            'epsilon_file': str(used_epsilon) if used_epsilon else None,
            'gradient_shape': list(self.real_gradient_data.shape) if self.real_gradient_data is not None else None,
            'attention_shape': list(self.real_attention_data.shape) if getattr(self, 'real_attention_data', None) is not None else None,
            'epsilon_len': (int(np.array(self.real_privacy_epsilon).size) if self.real_privacy_epsilon is not None and not np.isscalar(self.real_privacy_epsilon) else (1 if self.real_privacy_epsilon is not None else None)),
            'gradient_updated': gradient_updated,
            'attention_updated': bool(attention_updated),
            'epsilon_updated': bool(epsilon_updated),
        }

    def generate_gradient_leakage_data(self, use_real_data=True):
        """Generate realistic gradient leakage security data"""
        
        # NO SIMULATION by default: real data is required unless allow_simulated=True.
        if use_real_data:
            if self.real_gradient_data is None:
                raise ValueError("No real gradient data loaded (simulation disabled).")
            return self._generate_real_data_analysis()

        if self.allow_simulated:
            return self._generate_simulated_data()

        raise ValueError("Simulated data disabled (set allow_simulated=True to permit).")
    

    def _get_privacy_budget_series(self, total_steps: int):
        """Return a length=total_steps privacy budget series.

        NO SIMULATION:
        - If a real ε series is loaded from file (`self.real_privacy_epsilon`), use it.
        - Else if DP-SGD accounting parameters are provided (`dp_sample_rate`, `dp_noise_multiplier`, `dp_delta`), compute ε via an RDP accountant.
        - Else return None (privacy accounting unavailable).
        """
        import numpy as np

        if total_steps <= 0:
            return None

        # Cache key depends on privacy source + parameters
        if self.real_privacy_epsilon is not None:
            key = ('file', int(total_steps), int(getattr(self.real_privacy_epsilon, 'size', 1)))
        elif self.dp_noise_multiplier is not None and self.dp_sample_rate is not None:
            key = ('dp', int(total_steps), float(self.dp_sample_rate), float(self.dp_noise_multiplier), float(self.dp_delta))
        else:
            key = None

        if key is not None and getattr(self, '_cached_privacy_key', None) == key and getattr(self, '_cached_privacy_series', None) is not None:
            return self._cached_privacy_series

        if self.real_privacy_epsilon is not None:
            if np.isscalar(self.real_privacy_epsilon):
                self.privacy_metric = 'file_epsilon'
                series = np.full(total_steps, float(self.real_privacy_epsilon), dtype=float)
                self._cached_privacy_key = key
                self._cached_privacy_series = series
                return series

            arr = np.array(self.real_privacy_epsilon, dtype=float).reshape(-1)
            if arr.size == 1:
                self.privacy_metric = 'file_epsilon'
                series = np.full(total_steps, float(arr[0]), dtype=float)
                self._cached_privacy_key = key
                self._cached_privacy_series = series
                return series

            metric = 'file_epsilon'
            if arr.size != total_steps:
                # Be resilient: resample ε series to dashboard steps (visual alignment).
                # For exact step mapping, provide an aligned ε series.
                if arr.size < 2:
                    val = float(arr[-1]) if arr.size else float('nan')
                    self.privacy_metric = 'file_epsilon'
                    series = np.full(total_steps, val, dtype=float)
                    self._cached_privacy_key = key
                    self._cached_privacy_series = series
                    return series
                try:
                    x_old = np.linspace(0.0, 1.0, arr.size)
                    x_new = np.linspace(0.0, 1.0, total_steps)
                    arr = np.interp(x_new, x_old, arr).astype(float)
                except Exception:
                    if arr.size < total_steps:
                        pad = np.full(total_steps - arr.size, float(arr[-1]), dtype=float)
                        arr = np.concatenate([arr, pad], axis=0)
                    else:
                        arr = arr[:total_steps]
                metric = 'file_epsilon_resampled'

            self.privacy_metric = metric
            self._cached_privacy_key = key
            self._cached_privacy_series = arr
            return arr

        if self.dp_noise_multiplier is None or self.dp_sample_rate is None:
            return None

        series = self._compute_dp_epsilon_series(total_steps)
        self._cached_privacy_key = key
        self._cached_privacy_series = series
        return series

    def _compute_dp_epsilon_series(self, total_steps: int):
        """Compute ε(t) for t=1..total_steps for DP-SGD using an RDP accountant.

        This implements a Poisson-subsampled Gaussian mechanism accountant using integer Rényi orders.

        References (foundational):
        - Mironov (2017): Rényi Differential Privacy
        - Wang et al. (2019): Subsampled Rényi DP / Analytical Moments Accountant

        NOTE: ε depends on DP parameters (q, σ, δ) and the number of steps, not on gradient norms.
        """
        import math
        import numpy as np
        from scipy.special import gammaln, logsumexp

        q = float(self.dp_sample_rate)
        sigma = float(self.dp_noise_multiplier)
        delta = float(self.dp_delta)

        if not (0.0 < q <= 1.0):
            raise ValueError('dp_sample_rate must be in (0, 1]')
        if sigma <= 0.0:
            # No noise => no DP guarantee
            return np.full(total_steps, float('inf'), dtype=float)
        if not (0.0 < delta < 1.0):
            raise ValueError('dp_delta must be in (0, 1)')

        orders = np.arange(2, 65, dtype=int)
        log_q = math.log(q)
        log_1_q = math.log(1.0 - q) if q < 1.0 else float('-inf')

        # Per-step RDP for each order
        rdp_per_step = np.zeros_like(orders, dtype=float)
        for j, a in enumerate(orders):
            ks = np.arange(0, a + 1)
            log_binom = gammaln(a + 1) - gammaln(ks + 1) - gammaln(a - ks + 1)

            log_terms = log_binom + ks * log_q
            if q < 1.0:
                log_terms = log_terms + (a - ks) * log_1_q

            # Gaussian mechanism term
            log_terms = log_terms + (ks * (ks - 1)) / (2.0 * sigma * sigma)

            log_a = logsumexp(log_terms)
            rdp_per_step[j] = log_a / (a - 1.0)

        log_delta_inv = math.log(1.0 / delta)

        eps_series = np.empty(total_steps, dtype=float)
        for t in range(1, total_steps + 1):
            rdp_t = t * rdp_per_step
            eps_t = np.min(rdp_t + log_delta_inv / (orders - 1.0))
            eps_series[t - 1] = float(eps_t)

        self.privacy_metric = 'dp_epsilon'
        return eps_series


    def _generate_real_data_analysis(self):
        """Analyze real gradient data for security metrics (NO SIMULATION).

        IMPORTANT:
        - This function does **not** fabricate ε or add random noise.
        - If ε is required, provide either:
          - a real ε series via `privacy_file`/batch discovery, OR
          - DP accounting params: `dp_sample_rate`, `dp_noise_multiplier`, `dp_delta`.

        Note: ε depends on DP parameters and steps, not on gradient norms.
        """
        import time
        import numpy as np

        if self.real_gradient_data is None:
            raise ValueError('No real gradient data loaded (simulation disabled).')

        grad_arr = np.array(self.real_gradient_data)
        if grad_arr.ndim == 0 or grad_arr.size == 0:
            raise ValueError('Real gradient data is empty.')

        # Normalize shape to [N, D]
        if grad_arr.ndim == 1:
            grad_arr = grad_arr.reshape(-1, 1)
        elif grad_arr.ndim > 2:
            grad_arr = grad_arr.reshape(grad_arr.shape[0], -1)

        # Per-step gradient norm (cache for large arrays)
        if getattr(self, "_cached_gradient_norms_full", None) is not None and getattr(self, "_cached_gradient_norms_shape", None) == grad_arr.shape:
            gradient_norms_full = self._cached_gradient_norms_full
        else:
            gradient_norms_full = np.linalg.norm(grad_arr, axis=1).astype(float)
            self._cached_gradient_norms_full = gradient_norms_full
            self._cached_gradient_norms_shape = grad_arr.shape
        if gradient_norms_full.size == 0:
            raise ValueError('No gradient norms computed from real data.')

        total_samples = int(gradient_norms_full.shape[0])

        # Rolling window over real samples (no synthetic expansion)
        window_size = min(150, total_samples)
        scroll_speed = 5.0
        scroll_position = int((time.time() * scroll_speed) % total_samples) if total_samples > 0 else 0
        scrolling_indices = [(scroll_position + i) % total_samples for i in range(window_size)]

        gradient_norms = [float(gradient_norms_full[i]) for i in scrolling_indices]
        steps = [int(i) for i in scrolling_indices]  # sample indices (not wall-clock timestamps)

        # Privacy budget ε series (from file or DP accountant) — never fabricated
        privacy_series_full = self._get_privacy_budget_series(total_samples)
        privacy_available = privacy_series_full is not None
        if privacy_available:
            privacy_budget = [float(privacy_series_full[i]) for i in scrolling_indices]
        else:
            privacy_budget = [float('nan')] * window_size

        # Vulnerability scores from gradient norms (deterministic proxy)
        vulnerability_scores = []
        for g in gradient_norms:
            if g > self.CRITICAL_GRADIENT_NORM:
                vuln = min(0.95, 0.6 + (g / self.CRITICAL_GRADIENT_NORM) * 0.35)
            else:
                vuln = min(0.9, 0.1 + (g / self.CRITICAL_GRADIENT_NORM) * 0.6)
            vulnerability_scores.append(float(vuln))

        # Layer-wise vulnerability (only real if D is large enough; otherwise a global proxy)
        num_layers = int(self.model_layers.get('num_layers', 12))
        layer_vulnerabilities = {}
        if grad_arr.ndim == 2 and grad_arr.shape[1] >= num_layers:
            width = int(grad_arr.shape[1])
            chunk = max(1, width // num_layers)
            for li in range(num_layers):
                start_i = li * chunk
                end_i = (li + 1) * chunk if li < num_layers - 1 else width
                layer_slice = grad_arr[:, start_i:end_i]
                layer_norm = float(np.mean(np.linalg.norm(layer_slice, axis=1)))
                layer_vulnerabilities[f'Layer_{li+1}'] = float(min(0.95, layer_norm / self.CRITICAL_GRADIENT_NORM))
        else:
            global_vuln = float(np.mean(vulnerability_scores)) if vulnerability_scores else 0.5
            for li in range(num_layers):
                layer_vulnerabilities[f'Layer_{li+1}'] = global_vuln

        # Risk proxies (deterministic) — these are *not* measured attack success rates
        membership_inference_success = []
        reconstruction_success = []
        for i, g in enumerate(gradient_norms):
            grad_factor = min(g / self.CRITICAL_GRADIENT_NORM, 2.0)
            eps_factor = 0.0
            if privacy_available:
                eps = privacy_budget[i]
                if np.isfinite(eps) and self.PRIVACY_BUDGET_LIMIT > 0:
                    eps_factor = min(eps / self.PRIVACY_BUDGET_LIMIT, 2.0)

            mi = 0.5 + 0.25 * min(grad_factor, 1.0) + 0.15 * min(eps_factor, 1.0)
            membership_inference_success.append(float(min(max(mi, 0.0), 0.95)))

            recon = 0.15 + 0.6 * min(grad_factor, 1.0) + 0.15 * min(eps_factor, 1.0)
            reconstruction_success.append(float(min(max(recon, 0.0), 0.95)))

        atlas_data = self._load_atlas_attack_chains()

        return {
            'steps': steps,
            'privacy_budget': privacy_budget,
            'privacy_available': privacy_available,
            'privacy_metric': self.privacy_metric,
            'gradient_norms': gradient_norms,
            'vulnerability_scores': vulnerability_scores,
            'layer_vulnerabilities': layer_vulnerabilities,
            'membership_inference_success': membership_inference_success,
            'reconstruction_success': reconstruction_success,
            'data_source': getattr(self, 'active_data_source', 'real'),
            'model_name': self.model_name,
            'atlas_attacks': atlas_data,
            'rolling_timestamp': time.time(),
        }
    def _load_atlas_attack_chains(self):
        """Load MITRE ATLAS catalog + example attack chains for the dashboard.

        Important:
        - NeurInSpectre vendors the official MITRE ATLAS STIX bundle (v5.1.1) in-repo.
        - The *catalog* below is authoritative (16 tactics / 140 techniques).
        - The *attack_chains* below are illustrative (heuristics), not MITRE-authored.
        """
        # Keep numpy import (used elsewhere in this module); avoid hard-failing if absent.
        import numpy as np  # noqa: F401

        # -----------------------------------------------------------------
        # Load FULL official catalog (tactics + techniques)
        # -----------------------------------------------------------------
        atlas_tactics = {}
        atlas_techniques = {}
        try:
            from ..mitre_atlas.registry import (
                load_stix_atlas_bundle,
                list_atlas_tactics,
                list_atlas_techniques,
                tactic_by_phase_name,
            )

            bundle = load_stix_atlas_bundle()
            tactics = list_atlas_tactics(bundle)
            phase_to_tactic = tactic_by_phase_name(bundle)

            atlas_tactics = {
                t.tactic_id: {
                    'id': t.tactic_id,
                    'name': t.name,
                    'phase_name': t.phase_name,
                    'description': t.description,
                    'source': 'MITRE ATLAS STIX (atlas-navigator-data)',
                }
                for t in tactics
            }

            for tech in list_atlas_techniques(bundle):
                tactic_ids = []
                tactic_names = []
                for ph in tech.tactic_phase_names:
                    t = phase_to_tactic.get(ph)
                    if t is None:
                        continue
                    if t.tactic_id not in tactic_ids:
                        tactic_ids.append(t.tactic_id)
                        tactic_names.append(t.name)

                atlas_techniques[tech.technique_id] = {
                    'id': tech.technique_id,
                    'name': tech.name,
                    'description': tech.description,
                    'tactic_ids': tactic_ids,
                    'tactics': tactic_names,
                    'tactic_phase_names': tech.tactic_phase_names,
                    'url': tech.url,
                    'is_subtechnique': tech.is_subtechnique,
                    'source': 'MITRE ATLAS STIX (atlas-navigator-data)',
                }
        except Exception:
            # Catalog load is non-fatal: the dashboard can still run without ATLAS enrichment.
            atlas_tactics = {}
            atlas_techniques = {}

        # -----------------------------------------------------------------
        # Example chains (heuristic): keep IDs valid per official catalog
        # -----------------------------------------------------------------
        attack_chains_2025 = {
            'training_data_poisoning_chain': {
                'techniques': ['AML.T0020', 'AML.T0043', 'AML.T0042'],
                'severity': 'CRITICAL',
                'description': 'Poison training data → craft adversarial data → verify attack',
                'timeline': 'Example chain (NeurInSpectre heuristics)',
            },
            'llm_prompt_injection_chain': {
                'techniques': ['AML.T0051', 'AML.T0056', 'AML.T0057'],
                'severity': 'CRITICAL',
                'description': 'LLM prompt injection → extract system prompt → data leakage',
                'timeline': 'Example chain (NeurInSpectre heuristics)',
            },
            'membership_inference_chain': {
                'techniques': ['AML.T0024.000', 'AML.T0024.001', 'AML.T0085'],
                'severity': 'HIGH',
                'description': 'Infer training data membership → invert AI model → data from AI services',
                'timeline': 'Example chain (NeurInSpectre heuristics)',
            },
            'adversarial_evasion_chain': {
                'techniques': ['AML.T0043', 'AML.T0042', 'AML.T0015'],
                'severity': 'HIGH',
                'description': 'Craft adversarial data → verify attack → evade AI model',
                'timeline': 'Example chain (NeurInSpectre heuristics)',
            },
        }

        # Try to load additional real attack data from files (optional)
        atlas_files = [
            'realistic_atlas_attack_chain_critical_analysis.json',
            'real_malicious_timeline_events.json',
            'real_time_adversarial_events.json',
        ]

        atlas_data = {
            'tactics': atlas_tactics,
            'techniques': atlas_techniques,
            'attack_chains': attack_chains_2025,
            'catalog': {
                'vendor_bundle_path': 'neurinspectre/mitre_atlas/stix-atlas.json',
                'tactic_count': int(len(atlas_tactics)) if atlas_tactics else None,
                'technique_count': int(len(atlas_techniques)) if atlas_techniques else None,
            },
            'research_validation': 'MITRE ATLAS STIX (atlas-navigator-data)',
            'sources': [
                'MITRE ATLAS taxonomy (v5.1.1): https://github.com/mitre-atlas/atlas-data',
                'MITRE ATLAS STIX bundle: https://github.com/mitre-atlas/atlas-navigator-data',
            ],
        }

        # Attach optional external intel blobs if present
        for atlas_file in atlas_files:
            if os.path.exists(atlas_file):
                try:
                    with open(atlas_file, 'r') as f:
                        file_data = json.load(f)
                    atlas_data[atlas_file.replace('.json', '')] = file_data
                except Exception:
                    pass

        # Lightweight metrics for UI/telemetry
        try:
            chain_techs = sorted({tid for c in attack_chains_2025.values() for tid in (c.get('techniques') or [])})
        except Exception:
            chain_techs = []

        atlas_data['threat_metrics'] = {
            'active_campaigns': int(len(attack_chains_2025)),
            'referenced_chain_techniques': int(len(chain_techs)),
            'catalog_techniques': int(len(atlas_techniques)) if atlas_techniques else None,
            'catalog_tactics': int(len(atlas_tactics)) if atlas_tactics else None,
        }

        return atlas_data

    def _generate_simulated_data(self):

        """Generate simulated gradient leakage security data"""
        print("🎲 Generating simulated security data...")
        
        import numpy as np
        
        # Privacy budget exhaustion over training steps
        steps = np.arange(0, 1000, 10)
        privacy_budget = []
        current_budget = 0.1
        
        # Simulate privacy budget depletion with attack events
        for step in steps:
            # Normal privacy consumption
            current_budget += np.random.exponential(0.05)
            
            # Simulate gradient leakage attacks at specific intervals
            if step in [200, 350, 600, 850]:  # Attack windows
                current_budget += np.random.uniform(1.5, 3.0)  # Major privacy loss
            elif step % 100 == 0:  # Regular vulnerability windows
                current_budget += np.random.uniform(0.3, 0.8)
                
            privacy_budget.append(current_budget)
        
        # Gradient norm spikes (indicators of data extraction opportunities)
        gradient_norms = []
        vulnerability_scores = []
        
        for step in steps:
            base_norm = np.random.lognormal(0, 0.5)  # Normal gradient behavior
            
            # Attack windows show large gradient norms
            if step in [200, 350, 600, 850]:
                attack_norm = base_norm * np.random.uniform(8, 15)  # Severe leakage
                vuln_score = np.random.uniform(0.8, 0.95)  # High vulnerability
            elif step % 100 == 0:
                attack_norm = base_norm * np.random.uniform(3, 6)  # Moderate leakage
                vuln_score = np.random.uniform(0.6, 0.8)  # Medium vulnerability
            else:
                attack_norm = base_norm
                vuln_score = np.random.uniform(0.1, 0.4)  # Low vulnerability
                
            gradient_norms.append(attack_norm)
            vulnerability_scores.append(vuln_score)
        
        # Layer-wise vulnerability assessment for the specific model
        num_layers = self.model_layers["num_layers"]
        layer_vulnerabilities = {}
        
        for i in range(num_layers):
            if i < 2:  # Input layers
                vulnerability = np.random.uniform(0.1, 0.4)
            elif i < num_layers - 2:  # Middle/attention layers  
                vulnerability = np.random.uniform(0.5, 0.9)
            else:  # Output layers
                vulnerability = np.random.uniform(0.2, 0.5)
            
            layer_vulnerabilities[f"Layer_{i+1}"] = vulnerability
        
        # Risk proxy series over time (simulated demo only)
        membership_inference_success = []
        reconstruction_success = []
        
        for step in steps:
            # Membership inference risk proxy
            base_mi = 0.5  # Random guessing baseline
            if step in [200, 350, 600, 850]:
                mi_success = np.random.uniform(0.85, 0.95)  # High success during attacks
            elif privacy_budget[step//10] > self.PRIVACY_BUDGET_LIMIT:
                mi_success = np.random.uniform(0.7, 0.85)  # High success when budget exhausted
            else:
                mi_success = base_mi + np.random.uniform(0, 0.2)
            
            membership_inference_success.append(mi_success)
            
            # Reconstruction risk proxy
            if gradient_norms[step//10] > self.CRITICAL_GRADIENT_NORM:
                recon_success = np.random.uniform(0.75, 0.9)  # High success with large gradients
            elif step in [200, 350, 600, 850]:
                recon_success = np.random.uniform(0.6, 0.8)
            else:
                recon_success = np.random.uniform(0.1, 0.4)
                
            reconstruction_success.append(recon_success)
        
        return {
            'steps': steps,
            'privacy_budget': privacy_budget,
            'gradient_norms': gradient_norms,
            'vulnerability_scores': vulnerability_scores,
            'layer_vulnerabilities': layer_vulnerabilities,
            'membership_inference_success': membership_inference_success,
            'reconstruction_success': reconstruction_success,
            'data_source': 'simulated',
            'model_name': self.model_name
        }

    def _detect_atlas_techniques_from_gradients(self, gradient_data):
        """Heuristic MITRE ATLAS v5.1.1 technique tagging from gradient-derived signals.

Notes:
- NeurInSpectre vendors the official ATLAS STIX catalog (16 tactics / 140 techniques).
- This function tags a subset of techniques that are plausibly gradient-observable.
- Returned names/tactics are normalized against the official STIX bundle.

        """
        import numpy as np
        
        if gradient_data is None or len(gradient_data) == 0:
            return []
        
        # Calculate gradient statistics (robust + internally consistent).
        # Note: "mean" here is mean absolute magnitude since most thresholds are magnitude-driven.
        grad_array = np.array(gradient_data, dtype=np.float64)
        grad_array = np.nan_to_num(grad_array, nan=0.0, posinf=0.0, neginf=0.0)
        abs_grad = np.abs(grad_array)

        grad_norm = float(np.linalg.norm(grad_array))
        grad_mean = float(np.mean(abs_grad))
        grad_std = float(np.std(grad_array))
        grad_max = float(np.max(abs_grad))
        grad_min = float(np.min(abs_grad))
        grad_sparsity = float(np.mean(abs_grad < 0.01))

        # Kurtosis should be based on central moments of the signed gradients, not |g|.
        if grad_std > 1e-12 and np.isfinite(grad_std):
            grad_mu = float(np.mean(grad_array))
            grad_kurtosis = float(np.mean(((grad_array - grad_mu) / grad_std) ** 4))
        else:
            grad_kurtosis = 0.0

        grad_range = float(grad_max - grad_min)
        
        detected = []
        
        # =================================================================
        # RECONNAISSANCE (AML.TA0002)
        # =================================================================
        
        if grad_mean > 0.1:
            detected.append({'id': 'AML.T0000', 'name': 'Search Open Technical Databases', 
                           'tactic': 'Reconnaissance', 'tactic_id': 'AML.TA0002',
                           'severity': 1.0, 'confidence': 0.30,
                           'indicator': 'Gradient observation', 
                           'description': 'Search technical databases for AI information'})
        
        # SUB-TECHNIQUES OF AML.T0000:
        
        # AML.T0000.000: Journals and Conference Proceedings
        if grad_mean > 0.15:
            detected.append({'id': 'AML.T0000.000', 'name': 'Journals and Conference Proceedings',
                           'tactic': 'Reconnaissance', 'tactic_id': 'AML.TA0002',
                           'severity': 1.2, 'confidence': 0.32,
                           'indicator': f'Academic research pattern μ={grad_mean:.2f}',
                           'description': 'Search journals/conferences for AI research'})
        
        # AML.T0000.001: Pre-Print Repositories
        if grad_mean > 0.12:
            detected.append({'id': 'AML.T0000.001', 'name': 'Pre-Print Repositories',
                           'tactic': 'Reconnaissance', 'tactic_id': 'AML.TA0002',
                           'severity': 1.1, 'confidence': 0.30,
                           'indicator': f'Pre-print search μ={grad_mean:.2f}',
                           'description': 'Search arXiv and pre-print repos'})
        
        # AML.T0000.002: Technical Blogs
        if grad_mean > 0.10:
            detected.append({'id': 'AML.T0000.002', 'name': 'Technical Blogs',
                           'tactic': 'Reconnaissance', 'tactic_id': 'AML.TA0002',
                           'severity': 1.0, 'confidence': 0.28,
                           'indicator': f'Blog research μ={grad_mean:.2f}',
                           'description': 'Search technical blogs for AI info'})
        
        if grad_std > 0.5:
            detected.append({'id': 'AML.T0001', 'name': 'Search Open AI Vulnerability Analysis',
                           'tactic': 'Reconnaissance', 'tactic_id': 'AML.TA0002',
                           'severity': 1.5, 'confidence': 0.35,
                           'indicator': f'Variance σ={grad_std:.2f} suggests testing',
                           'description': 'Search for AI vulnerabilities and attacks'})
        
        # =================================================================
        # RESOURCE DEVELOPMENT (AML.TA0003)
        # =================================================================
        
        # AML.T0017.000: Adversarial AI Attacks
        if grad_norm > 3.0 and grad_std > 0.4:
            detected.append({'id': 'AML.T0017.000', 'name': 'Adversarial AI Attacks',
                           'tactic': 'Resource Development', 'tactic_id': 'AML.TA0003',
                           'severity': 3.5, 'confidence': 0.68,
                           'indicator': f'Attack development ||∇||={grad_norm:.2f}',
                           'description': 'Develop custom adversarial attack capabilities'})
        
        # AML.T0016.000: Adversarial AI Attack Implementations
        if grad_norm > 2.5 and grad_mean > 0.3:
            detected.append({'id': 'AML.T0016.000', 'name': 'Adversarial AI Attack Implementations',
                           'tactic': 'Resource Development', 'tactic_id': 'AML.TA0003',
                           'severity': 3.2, 'confidence': 0.62,
                           'indicator': f'Attack impl ||∇||={grad_norm:.2f}',
                           'description': 'Obtain adversarial attack implementation code'})
        
        if grad_mean > 0.5:
            detected.append({'id': 'AML.T0002', 'name': 'Acquire Public AI Artifacts',
                           'tactic': 'Resource Development', 'tactic_id': 'AML.TA0003',
                           'severity': 2.0, 'confidence': 0.40,
                           'indicator': f'Gradients μ={grad_mean:.2f}',
                           'description': 'Acquire public datasets or models'})
        
        # SUB-TECHNIQUES OF AML.T0002:
        
        # AML.T0002.000: Datasets
        if grad_mean > 0.3 and grad_norm > 1.0:
            detected.append({'id': 'AML.T0002.000', 'name': 'Datasets',
                           'tactic': 'Resource Development', 'tactic_id': 'AML.TA0003',
                           'severity': 2.2, 'confidence': 0.45,
                           'indicator': f'Dataset acquisition μ={grad_mean:.2f}',
                           'description': 'Acquire public datasets for attack preparation'})
        
        # AML.T0002.001: Models
        if grad_mean > 0.4 and grad_norm > 1.5:
            detected.append({'id': 'AML.T0002.001', 'name': 'Models',
                           'tactic': 'Resource Development', 'tactic_id': 'AML.TA0003',
                           'severity': 2.5, 'confidence': 0.50,
                           'indicator': f'Model acquisition ||∇||={grad_norm:.2f}',
                           'description': 'Acquire public models for attack foundation'})
        
        if grad_std > 0.6 and grad_max > 1.2:
            detected.append({'id': 'AML.T0019', 'name': 'Publish Poisoned Datasets',
                           'tactic': 'Resource Development', 'tactic_id': 'AML.TA0003',
                           'severity': 4.5, 'confidence': min(0.90, 0.6 + (grad_std / 6.0)),
                           'indicator': f'Extreme variance σ={grad_std:.2f}',
                           'description': 'Publish poisoned datasets for supply chain attacks'})
        
        if grad_std > 0.4 or grad_max > 1.0:
            detected.append({'id': 'AML.T0020', 'name': 'Poison Training Data',
                           'tactic': 'Resource Development', 'tactic_id': 'AML.TA0003',
                           'severity': min(5.0, 3.0 + (grad_std / 3.0)), 
                           'confidence': min(0.95, 0.6 + (grad_std / 5.0)),
                           'indicator': f'High variance σ={grad_std:.2f}',
                           'description': 'Embed vulnerabilities via poisoned training data'})
        
        # =================================================================
        # AI MODEL ACCESS (AML.TA0000)
        # =================================================================
        
        if grad_norm > 0.5:
            detected.append({'id': 'AML.T0040', 'name': 'AI Model Inference API Access',
                           'tactic': 'AI Model Access', 'tactic_id': 'AML.TA0000',
                           'severity': 1.5, 'confidence': 0.70,
                           'indicator': f'API access ||∇||={grad_norm:.2f}',
                           'description': 'Access model via legitimate inference API'})
        
        if grad_mean > 0.5:
            detected.append({'id': 'AML.T0044', 'name': 'Full AI Model Access',
                           'tactic': 'AI Model Access', 'tactic_id': 'AML.TA0000',
                           'severity': min(4.5, 2.0 + grad_mean), 'confidence': 0.85,
                           'indicator': f'White-box access μ={grad_mean:.2f}',
                           'description': 'Complete knowledge of model parameters'})
        
        if grad_norm > 1.5:
            detected.append({'id': 'AML.T0047', 'name': 'AI-Enabled Product or Service',
                           'tactic': 'AI Model Access', 'tactic_id': 'AML.TA0000',
                           'severity': 2.5, 'confidence': 0.50,
                           'indicator': f'Service access ||∇||={grad_norm:.2f}',
                           'description': 'Access AI through product/service interface'})
        
        # =================================================================
        # AI ATTACK STAGING (AML.TA0001)
        # =================================================================
        
        if grad_norm > 1.5 and grad_std > 0.4:
            detected.append({'id': 'AML.T0005', 'name': 'Create Proxy AI Model',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': 3.0, 'confidence': 0.65,
                           'indicator': f'Training patterns ||∇||={grad_norm:.2f}',
                           'description': 'Train surrogate model to mimic target'})
        
        # SUB-TECHNIQUES OF AML.T0005:
        
        # AML.T0005.000: Train Proxy via Gathered AI Artifacts
        if grad_norm > 3.0 and grad_std > 1.5:
            detected.append({'id': 'AML.T0005.000', 'name': 'Train Proxy via Gathered AI Artifacts',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': 3.2, 'confidence': 0.68,
                           'indicator': f'Proxy training with artifacts ||∇||={grad_norm:.2f}',
                           'description': 'Train proxy using gathered datasets/models'})
        
        # AML.T0005.001: Train Proxy via Replication  
        if grad_norm > 2.5 and grad_std > 1.2:
            detected.append({'id': 'AML.T0005.001', 'name': 'Train Proxy via Replication',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': 3.0, 'confidence': 0.65,
                           'indicator': f'Replication training σ={grad_std:.2f}',
                           'description': 'Train proxy by querying and replicating target'})
        
        # AML.T0005.002: Use Pre-Trained Model
        if grad_mean > 0.4 and grad_norm > 1.0:
            detected.append({'id': 'AML.T0005.002', 'name': 'Use Pre-Trained Model',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': 2.5, 'confidence': 0.55,
                           'indicator': f'Pre-trained model usage μ={grad_mean:.2f}',
                           'description': 'Use pre-trained model as attack foundation'})
        
        if grad_max > 1.5 or (grad_std > 0.5 and grad_norm > 3.0):
            detected.append({'id': 'AML.T0018', 'name': 'Manipulate AI Model',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': 4.8, 'confidence': min(0.92, 0.7 + (grad_max / 30.0)),
                           'indicator': f'Extreme gradients max={grad_max:.2f}',
                           'description': 'Directly manipulate model parameters'})
        
        # SUB-TECHNIQUES OF AML.T0018:
        
        # AML.T0018.000: Poison AI Model
        if grad_std > 2.5 and grad_max > 12.0:
            detected.append({'id': 'AML.T0018.000', 'name': 'Poison AI Model',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': 4.9, 'confidence': 0.90,
                           'indicator': f'Model poisoning σ={grad_std:.2f}',
                           'description': 'Directly poison model weights or structure'})
        
        # AML.T0018.001: Modify AI Model Architecture
        if grad_max > 18.0:
            detected.append({'id': 'AML.T0018.001', 'name': 'Modify AI Model Architecture',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': 4.7, 'confidence': 0.82,
                           'indicator': f'Architecture modification max={grad_max:.2f}',
                           'description': 'Alter model architecture for malicious purposes'})
        
        if 1.0 < grad_norm < 3.0:
            detected.append({'id': 'AML.T0042', 'name': 'Verify Attack',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': 2.5, 'confidence': 0.65,
                           'indicator': f'Moderate grads ||∇||={grad_norm:.2f}',
                           'description': 'Verify attack effectiveness'})
        
        if grad_norm > 2.0:
            detected.append({'id': 'AML.T0043', 'name': 'Craft Adversarial Data',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': min(5.0, 2.5 + (grad_norm / 10.0)),
                           'confidence': min(0.90, 0.5 + (grad_norm / 15.0)),
                           'indicator': f'Large norm ||∇||={grad_norm:.2f}',
                           'description': 'Modified inputs for adversarial effect'})
        
        # SUB-TECHNIQUES OF AML.T0043:
        
        # AML.T0043.000: White-Box Optimization  
        if grad_norm > 3.5 and grad_mean > 0.4:
            detected.append({'id': 'AML.T0043.000', 'name': 'White-Box Optimization',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': 4.8, 'confidence': 0.88,
                           'indicator': f'White-box crafting ||∇||={grad_norm:.2f}, μ={grad_mean:.2f}',
                           'description': 'Gradient-based adversarial crafting with full model access'})
        
        # AML.T0043.001: Black-Box Optimization
        if grad_norm > 3.0 and grad_std > 0.5:
            detected.append({'id': 'AML.T0043.001', 'name': 'Black-Box Optimization',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': 4.3, 'confidence': 0.75,
                           'indicator': f'Black-box crafting ||∇||={grad_norm:.2f}',
                           'description': 'Query-based adversarial optimization without gradients'})
        
        # AML.T0043.002: Black-Box Transfer
        if grad_norm > 2.5 and grad_std > 0.4:
            detected.append({'id': 'AML.T0043.002', 'name': 'Black-Box Transfer',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': 3.8, 'confidence': 0.70,
                           'indicator': f'Transfer attack σ={grad_std:.2f}',
                           'description': 'Transfer adversarial examples from surrogate model'})
        
        # AML.T0043.003: Manual Modification
        if grad_mean > 0.3 and grad_std < 3.0:
            detected.append({'id': 'AML.T0043.003', 'name': 'Manual Modification',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': 3.2, 'confidence': 0.60,
                           'indicator': f'Manual crafting μ={grad_mean:.2f}',
                           'description': 'Manually modify inputs to evade model'})
        
        # AML.T0043.004: Insert Backdoor Trigger
        if grad_std > 0.5 or grad_sparsity > 0.15:  # OR condition, lower thresholds
            detected.append({'id': 'AML.T0043.004', 'name': 'Insert Backdoor Trigger',
                           'tactic': 'AI Attack Staging', 'tactic_id': 'AML.TA0001',
                           'severity': 4.7, 'confidence': 0.78,
                           'indicator': f'Backdoor trigger σ={grad_std:.2f}, sparsity={grad_sparsity:.2%}',
                           'description': 'Insert trigger pattern for backdoor activation'})
        
        # =================================================================
        # EXECUTION (AML.TA0005)
        # =================================================================
        
        if grad_std > 2.0:  # Lowered threshold to ensure detection
            detected.append({'id': 'AML.T0051', 'name': 'LLM Prompt Injection',
                           'tactic': 'Execution', 'tactic_id': 'AML.TA0005',
                           'severity': 4.2, 'confidence': 0.75,
                           'indicator': f'Prompt injection σ={grad_std:.2f}',
                           'description': 'Malicious prompts for LLM manipulation'})
        
        if grad_std > 2.5 or grad_max > 8.0:  # OR condition for better coverage
            detected.append({'id': 'AML.T0054', 'name': 'LLM Jailbreak',
                           'tactic': 'Privilege Escalation', 'tactic_id': 'AML.TA0012',
                           'severity': 4.5, 'confidence': 0.80,
                           'indicator': f'Jailbreak pattern σ={grad_std:.2f}, max={grad_max:.2f}',
                           'description': 'Bypass LLM safety measures'})
        
        # =================================================================
        # EXFILTRATION (AML.TA0010)
        # =================================================================
        
        if grad_norm > 3.5:
            detected.append({'id': 'AML.T0024', 'name': 'Exfiltration via AI Inference API',
                           'tactic': 'Exfiltration', 'tactic_id': 'AML.TA0010',
                           'severity': 4.0, 'confidence': 0.75,
                           'indicator': f'Gradient pattern ||∇||={grad_norm:.2f}',
                           'description': 'Exfiltrate info via inference API'})
        
        if grad_norm > 3.0 and grad_std > 1.0:
            detected.append({'id': 'AML.T0024.000', 'name': 'Infer Training Data Membership',
                           'tactic': 'Exfiltration', 'tactic_id': 'AML.TA0010',
                           'severity': min(4.8, 3.5 + (grad_norm / 8.0)),
                           'confidence': min(0.88, 0.6 + (grad_std / 4.0)),
                           'indicator': f'Membership inference ||∇||={grad_norm:.2f}',
                           'description': 'Infer if data in training set'})
        if grad_sparsity < 0.3 and grad_norm > 4.0:
            detected.append({'id': 'AML.T0024.001', 'name': 'Invert AI Model',
                           'tactic': 'Exfiltration', 'tactic_id': 'AML.TA0010',
                           'severity': min(5.0, 4.0 + (grad_norm / 20.0)),
                           'confidence': min(0.80, 0.5 + ((1.0 - grad_sparsity) / 2.0)),
                           'indicator': f'Dense grads sparsity={grad_sparsity:.2%}',
                           'description': 'Reconstruct training data from model'})
        
        # AML.T0024.002: Extract AI Model
        if grad_norm > 6.0 and grad_sparsity < 0.4:
            detected.append({'id': 'AML.T0024.002', 'name': 'Extract AI Model',
                           'tactic': 'Exfiltration', 'tactic_id': 'AML.TA0010',
                           'severity': 4.8, 'confidence': 0.75,
                           'indicator': f'Model extraction ||∇||={grad_norm:.2f}',
                           'description': 'Extract model parameters through queries'})
        
        if grad_norm > 6.0:
            detected.append({'id': 'AML.T0025', 'name': 'Exfiltration via Cyber Means',
                           'tactic': 'Exfiltration', 'tactic_id': 'AML.TA0010',
                           'severity': 4.3, 'confidence': 0.70,
                           'indicator': f'Large gradient ||∇||={grad_norm:.2f}',
                           'description': 'Exfiltrate artifacts via cyber means'})
        
        # =================================================================
        # IMPACT (AML.TA0011)
        # =================================================================
        
        if grad_norm > 10.0:
            detected.append({'id': 'AML.T0034', 'name': 'Cost Harvesting',
                           'tactic': 'Impact', 'tactic_id': 'AML.TA0011',
                           'severity': 3.8, 'confidence': 0.60,
                           'indicator': f'High compute ||∇||={grad_norm:.2f}',
                           'description': 'Force target to expend resources'})
        
        if grad_std > 4.0 and grad_kurtosis > 8.0:
            detected.append({'id': 'AML.T0046', 'name': 'Spamming AI System with Chaff Data',
                           'tactic': 'Impact', 'tactic_id': 'AML.TA0011',
                           'severity': 3.5, 'confidence': 0.55,
                           'indicator': f'Chaotic σ={grad_std:.2f}',
                           'description': 'Overwhelm system with chaff data'})
        
        if grad_max > 15.0:
            detected.append({'id': 'AML.T0048', 'name': 'External Harms',
                           'tactic': 'Impact', 'tactic_id': 'AML.TA0011',
                           'severity': 5.0, 'confidence': min(0.92, 0.7 + (grad_max / 30.0)),
                           'indicator': f'Extreme max={grad_max:.2f}',
                           'description': 'AI resources abused for external harms'})
        
        if grad_norm > 12.0 and grad_std > 5.0:
            detected.append({'id': 'AML.T0029', 'name': 'Denial of AI Service',
                           'tactic': 'Impact', 'tactic_id': 'AML.TA0011',
                           'severity': 4.5, 'confidence': 0.70,
                           'indicator': f'Overload ||∇||={grad_norm:.2f}',
                           'description': 'Degrade or deny AI service'})
        
        if grad_std > 3.5 and grad_mean > 2.0:
            detected.append({'id': 'AML.T0031', 'name': 'Erode AI Model Integrity',
                           'tactic': 'Impact', 'tactic_id': 'AML.TA0011',
                           'severity': 4.2, 'confidence': 0.68,
                           'indicator': f'Erosion σ={grad_std:.2f}',
                           'description': 'Gradually degrade model performance'})
        
        # =================================================================
        # INITIAL ACCESS (AML.TA0004)
        # =================================================================
        
        if grad_std > 2.0:
            detected.append({'id': 'AML.T0010', 'name': 'AI Supply Chain Compromise',
                           'tactic': 'Initial Access', 'tactic_id': 'AML.TA0004',
                           'severity': 4.0, 'confidence': 0.55,
                           'indicator': f'Supply chain pattern σ={grad_std:.2f}',
                           'description': 'Compromise AI supply chain components'})
        
        if grad_norm > 4.0 and grad_sparsity > 0.4:
            detected.append({'id': 'AML.T0015', 'name': 'Evade AI Model',
                           'tactic': 'Initial Access', 'tactic_id': 'AML.TA0004',
                           'severity': 3.5, 'confidence': 0.65,
                           'indicator': f'Evasion pattern ||∇||={grad_norm:.2f}',
                           'description': 'Craft inputs to evade detection'})
        
        # =================================================================
        # DEFENSE EVASION (AML.TA0007)
        # =================================================================
        
        if grad_sparsity > 0.55:
            detected.append({'id': 'AML.T0008.003', 'name': 'Physical Countermeasures',
                           'tactic': 'Defense Evasion', 'tactic_id': 'AML.TA0007',
                           'severity': 3.0, 'confidence': 0.50,
                           'indicator': f'Evasion sparsity={grad_sparsity:.2%}',
                           'description': 'Physical methods to evade detection'})
        
        if grad_std > 1.0:  # Lowered to ensure detection
            detected.append({'id': 'AML.T0068', 'name': 'LLM Prompt Obfuscation',
                           'tactic': 'Defense Evasion', 'tactic_id': 'AML.TA0007',
                           'severity': 2.8, 'confidence': 0.55,
                           'indicator': f'Obfuscation σ={grad_std:.2f}',
                           'description': 'Obfuscate prompts to evade detection'})
        
        if grad_norm > 3.0 and grad_sparsity > 0.4:
            detected.append({'id': 'AML.T0015', 'name': 'Evade AI Model',
                           'tactic': 'Defense Evasion', 'tactic_id': 'AML.TA0007',
                           'severity': 3.2, 'confidence': 0.60,
                           'indicator': f'Evasion ||∇||={grad_norm:.2f}',
                           'description': 'Craft inputs to evade AI model detection'})
        
        # =================================================================
        # DISCOVERY (AML.TA0008)
        # =================================================================
        
        if grad_mean > 0.5 and grad_std < 2.0:
            detected.append({'id': 'AML.T0007', 'name': 'Discover AI Artifacts',
                           'tactic': 'Discovery', 'tactic_id': 'AML.TA0008',
                           'severity': 2.0, 'confidence': 0.45,
                           'indicator': f'Discovery pattern μ={grad_mean:.2f}',
                           'description': 'Discover ML artifacts in victim system'})
        
        if grad_norm > 2.0:
            detected.append({'id': 'AML.T0013', 'name': 'Discover AI Model Ontology',
                           'tactic': 'Discovery', 'tactic_id': 'AML.TA0008',
                           'severity': 2.5, 'confidence': 0.50,
                           'indicator': f'Model probing ||∇||={grad_norm:.2f}',
                           'description': 'Determine model class structure'})
        
        # =================================================================
        # COLLECTION (AML.TA0009)
        # =================================================================
        
        if grad_norm > 3.0:
            detected.append({'id': 'AML.T0035', 'name': 'AI Artifact Collection',
                           'tactic': 'Collection', 'tactic_id': 'AML.TA0009',
                           'severity': 3.5, 'confidence': 0.60,
                           'indicator': f'Collection ||∇||={grad_norm:.2f}',
                           'description': 'Collect AI artifacts from system'})
        
        # =================================================================
        # CREDENTIAL ACCESS (AML.TA0013)
        # =================================================================
        
        if grad_max > 8.0 and grad_sparsity > 0.5:
            detected.append({'id': 'AML.T0082', 'name': 'RAG Credential Harvesting',
                           'tactic': 'Credential Access', 'tactic_id': 'AML.TA0013',
                           'severity': 4.0, 'confidence': 0.55,
                           'indicator': f'Credential pattern max={grad_max:.2f}',
                           'description': 'Harvest credentials from RAG systems'})
        
        if grad_norm > 4.0 and grad_std > 2.0:
            detected.append({'id': 'AML.T0083', 'name': 'Credentials from AI Agent Configuration',
                           'tactic': 'Credential Access', 'tactic_id': 'AML.TA0013',
                           'severity': 3.8, 'confidence': 0.60,
                           'indicator': f'Config access ||∇||={grad_norm:.2f}',
                           'description': 'Extract credentials from AI agent config'})
        
        # =================================================================
        # PRIVILEGE ESCALATION (AML.TA0012)
        # =================================================================
        
        if grad_norm > 7.0 and grad_std > 3.0:
            detected.append({'id': 'AML.T0012', 'name': 'Valid Accounts',
                           'tactic': 'Initial Access', 'tactic_id': 'AML.TA0004',
                           'severity': 3.8, 'confidence': 0.50,
                           'indicator': f'Account access ||∇||={grad_norm:.2f}',
                           'description': 'Use valid accounts to gain initial access'})
        
        # =================================================================
        # LATERAL MOVEMENT (AML.TA0015)
        # =================================================================
        
        if grad_kurtosis > 8.0:  # Lowered threshold
            detected.append({'id': 'AML.T0092', 'name': 'Lateral Movement via AI Agent',
                           'tactic': 'Defense Evasion', 'tactic_id': 'AML.TA0007',
                           'severity': 3.5, 'confidence': 0.40,
                           'indicator': f'Lateral evasion kurtosis={grad_kurtosis:.2f}',
                           'description': 'Move laterally through AI agent capabilities'})
        
        if grad_range > 10.0:  # Wide range suggests credential reuse
            detected.append({'id': 'AML.T0091', 'name': 'Use Alternate Authentication Material',
                           'tactic': 'Lateral Movement', 'tactic_id': 'AML.TA0015',
                           'severity': 3.3, 'confidence': 0.45,
                           'indicator': f'Auth pattern range={grad_range:.2f}',
                           'description': 'Use alternate credentials for lateral movement'})
        
        # =================================================================
        # COMMAND AND CONTROL (AML.TA0014)
        # =================================================================
        
        if grad_std > 2.5 and grad_kurtosis > 3.0:  # Lowered for real gradient data
            detected.append({'id': 'AML.T0050', 'name': 'Command and Scripting Interpreter',
                           'tactic': 'Execution', 'tactic_id': 'AML.TA0005',
                           'severity': 3.8, 'confidence': 0.45,
                           'indicator': f'Execution pattern σ={grad_std:.2f}, kurt={grad_kurtosis:.2f}',
                           'description': 'Use command/script interpreter for execution'})
        
        # REMAINING SUB-TECHNIQUES (Complete 56 Implementation)
        sub_techniques_batch = [
            ('AML.T0016.001', 'Software Tools', 'AML.TA0003', grad_std > 1.0, 2.8, 0.45),
            ('AML.T0008.001', 'Consumer Hardware', 'AML.TA0008', grad_kurtosis > 8.0, 2.2, 0.38),
            ('AML.T0010.000', 'Hardware', 'AML.TA0004', grad_kurtosis > 9.0, 3.0, 0.42),
            ('AML.T0010.001', 'AI Software', 'AML.TA0004', grad_std > 1.2, 3.2, 0.48),
            ('AML.T0010.004', 'Container Registry', 'AML.TA0004', grad_std > 1.1, 2.8, 0.44),
            ('AML.T0011.000', 'Unsafe AI Artifacts', 'AML.TA0005', grad_std > 1.3, 3.3, 0.50),
            ('AML.T0011.001', 'Malicious Package', 'AML.TA0005', grad_std > 1.4, 3.5, 0.52),
            ('AML.T0043.003', 'Manual Modification', 'AML.TA0001', grad_mean > 1.2 and grad_std < 3.0, 3.0, 0.58),
            ('AML.T0048.000', 'Financial Harm', 'AML.TA0011', grad_max > 12.0, 4.5, 0.65),
            ('AML.T0048.001', 'Reputational Harm', 'AML.TA0011', grad_std > 3.0, 3.8, 0.60),
            ('AML.T0048.002', 'Societal Harm', 'AML.TA0011', grad_max > 14.0, 4.8, 0.68),
            ('AML.T0048.003', 'User Harm', 'AML.TA0011', grad_std > 2.8, 4.2, 0.62),
            ('AML.T0048.004', 'AI Intellectual Property Theft', 'AML.TA0011', grad_norm > 5.0, 4.0, 0.58),
            ('AML.T0051.000', 'Direct', 'AML.TA0005', grad_std > 2.0 or grad_sparsity > 0.25, 4.0, 0.70),
            ('AML.T0051.001', 'Indirect', 'AML.TA0005', grad_std > 1.5 or grad_sparsity > 0.20, 3.8, 0.65),
            ('AML.T0051.002', 'Triggered', 'AML.TA0005', grad_sparsity > 0.15, 4.3, 0.68),
            ('AML.T0052.000', 'Spearphishing via Social Engineering LLM', 'AML.TA0004', grad_std > 2.2, 3.5, 0.55),
            ('AML.T0008.002', 'Domains', 'AML.TA0008', grad_mean > 0.4, 2.0, 0.38),
            ('AML.T0016.002', 'Generative AI', 'AML.TA0003', grad_norm > 2.0, 2.8, 0.48),
            ('AML.T0069.000', 'Special Character Sets', 'AML.TA0008', grad_std > 1.8, 2.5, 0.45),
            ('AML.T0069.001', 'System Instruction Keywords', 'AML.TA0008', grad_std > 1.6, 2.6, 0.46),
            ('AML.T0069.002', 'System Prompt', 'AML.TA0008', grad_std > 2.2, 3.0, 0.52),
            ('AML.T0067.000', 'Citations', 'AML.TA0007', grad_mean > 0.6, 2.4, 0.42),
            ('AML.T0018.002', 'Embed Malware', 'AML.TA0001', grad_std > 2.0 and grad_max > 8.0, 4.5, 0.75),
            ('AML.T0008.000', 'AI Development Workspaces', 'AML.TA0008', grad_mean > 0.3, 2.0, 0.40),
            ('AML.T0008.004', 'Serverless', 'AML.TA0008', grad_mean > 0.5, 2.3, 0.40),
            ('AML.T0080.000', 'Memory', 'AML.TA0006', grad_kurtosis > 7.0, 3.0, 0.48),
            ('AML.T0080.001', 'Thread', 'AML.TA0006', grad_kurtosis > 6.5, 2.9, 0.46),
            ('AML.T0084.000', 'Embedded Knowledge', 'AML.TA0006', grad_std > 1.5, 3.2, 0.50),
            ('AML.T0084.001', 'Tool Definitions', 'AML.TA0006', grad_std > 1.6, 3.3, 0.52),
            ('AML.T0084.002', 'Activation Triggers', 'AML.TA0006', grad_sparsity > 0.6, 4.0, 0.65),
            ('AML.T0085.000', 'RAG Databases', 'AML.TA0006', grad_std > 1.4, 3.4, 0.54),
            ('AML.T0085.001', 'AI Agent Tools', 'AML.TA0006', grad_std > 1.5, 3.5, 0.56),
            ('AML.T0091.000', 'Application Access Token', 'AML.TA0015', grad_norm > 4.0, 3.2, 0.50),
        ]
        
        for tech_id, name, tactic_id, condition, severity, confidence in sub_techniques_batch:
            if condition:
                detected.append({'id': tech_id, 'name': name,
                               'tactic': 'Sub-Technique', 'tactic_id': tactic_id,
                               'severity': severity, 'confidence': confidence,
                               'indicator': f'{name} detected',
                               'description': name})
        
        # FINAL 10 SUB-TECHNIQUES TO COMPLETE 56/56
        final_batch = [
            ('AML.T0008.001', 'Consumer Hardware', 'AML.TA0008', grad_kurtosis > 6.0, 2.2, 0.38),
            ('AML.T0008.003', 'Physical Countermeasures', 'AML.TA0007', grad_sparsity > 0.10, 2.5, 0.40),
            ('AML.T0010.000', 'Hardware', 'AML.TA0004', grad_kurtosis > 5.0, 3.0, 0.42),
            ('AML.T0010.002', 'Data', 'AML.TA0004', grad_std > 1.3, 3.5, 0.52),
            ('AML.T0010.003', 'Model', 'AML.TA0004', grad_norm > 2.2, 3.8, 0.58),
            ('AML.T0043.003', 'Manual Modification', 'AML.TA0001', grad_mean > 0.8, 3.0, 0.58),
            ('AML.T0043.004', 'Insert Backdoor Trigger', 'AML.TA0001', grad_std > 1.8 and grad_sparsity > 0.50, 4.7, 0.78),
            ('AML.T0051.000', 'Direct', 'AML.TA0005', grad_std > 2.0 and grad_sparsity > 0.40, 4.0, 0.70),
            ('AML.T0051.001', 'Indirect', 'AML.TA0005', grad_std > 1.5 and grad_sparsity > 0.30, 3.8, 0.65),
            ('AML.T0051.002', 'Triggered', 'AML.TA0005', grad_sparsity > 0.58, 4.3, 0.68),
            ('AML.T0080.000', 'Memory', 'AML.TA0006', grad_kurtosis > 6.0, 3.0, 0.48),
            ('AML.T0080.001', 'Thread', 'AML.TA0006', grad_kurtosis > 5.5, 2.9, 0.46),
            ('AML.T0084.000', 'Embedded Knowledge', 'AML.TA0006', grad_std > 1.3, 3.2, 0.50),
            ('AML.T0084.001', 'Tool Definitions', 'AML.TA0006', grad_std > 1.4, 3.3, 0.52),
            ('AML.T0084.002', 'Activation Triggers', 'AML.TA0006', grad_sparsity > 0.10, 4.0, 0.65),
            ('AML.T0085.000', 'RAG Databases', 'AML.TA0006', grad_std > 1.2, 3.4, 0.54),
            ('AML.T0085.001', 'AI Agent Tools', 'AML.TA0006', grad_std > 1.3, 3.5, 0.56),
            ('AML.T0091.000', 'Application Access Token', 'AML.TA0015', grad_norm > 3.5, 3.2, 0.50),
        ]
        
        for tech_id, name, tactic_id, condition, severity, confidence in final_batch:
            if condition:
                detected.append({'id': tech_id, 'name': name,
                               'tactic': 'Sub-Technique', 'tactic_id': tactic_id,
                               'severity': severity, 'confidence': confidence,
                               'indicator': f'{name} pattern detected',
                               'description': name})
        

        # Normalize against official ATLAS STIX: drop unknown IDs and overwrite names/tactics.
        try:
            from ..mitre_atlas.registry import load_stix_atlas_bundle, technique_index, tactic_by_phase_name

            bundle = load_stix_atlas_bundle()
            idx_map = technique_index(bundle)
            phase_to_tactic = tactic_by_phase_name(bundle)

            normalized = []
            for d in detected:
                tid = d.get('id')
                tech = idx_map.get(tid)
                if tech is None:
                    continue

                d2 = dict(d)
                d2['name'] = tech.name
                if tech.description:
                    d2['description'] = tech.description.splitlines()[0].strip()

                tactic_ids = []
                tactic_names = []
                for ph in tech.tactic_phase_names:
                    t = phase_to_tactic.get(ph)
                    if t is None:
                        continue
                    if t.tactic_id not in tactic_ids:
                        tactic_ids.append(t.tactic_id)
                        tactic_names.append(t.name)

                if tactic_ids:
                    # Back-compat: single tactic fields
                    d2['tactic_id'] = tactic_ids[0]
                    d2['tactic'] = tactic_names[0]
                    # New: full tactic list
                    d2['tactic_ids'] = tactic_ids
                    d2['tactics'] = tactic_names

                d2['atlas_url'] = tech.url
                d2['atlas_is_subtechnique'] = tech.is_subtechnique
                normalized.append(d2)

            # De-duplicate by technique id (keep highest severity/confidence)
            best = {}
            for d in normalized:
                tid = d.get('id')
                if not tid:
                    continue
                score = (float(d.get('severity', 0.0)), float(d.get('confidence', 0.0)))
                prev = best.get(tid)
                if prev is None:
                    best[tid] = d
                    continue
                prev_score = (float(prev.get('severity', 0.0)), float(prev.get('confidence', 0.0)))
                if score > prev_score:
                    best[tid] = d

            detected = sorted(
                best.values(),
                key=lambda x: (-float(x.get('severity', 0.0)), -float(x.get('confidence', 0.0)), str(x.get('id', ''))),
            )
        except Exception:
            pass

        return detected
    
    def create_attack_timeline(self, data=None):
        """
        Create enhanced attack timeline with MITRE ATLAS technique bubbles
        
        Args:
            data: Optional data dict (not used - detection from self.real_gradient_data)
        """
        import numpy as np
        import datetime
        import plotly.express as px
        import plotly.graph_objects as go
        
        current_time = datetime.datetime.now()
        attack_events = []

        # Cache bubble timeline to avoid recomputing every refresh
        import time as _time
        cache_key = (int(getattr(self, '_data_version', 0)), str(getattr(self, 'active_data_source', 'real')), str(getattr(self, 'model_name', '')))
        if (getattr(self, '_timeline_cache_key', None) == cache_key and
                getattr(self, '_timeline_cache_fig', None) is not None and
                (_time.time() - float(getattr(self, '_timeline_cache_at', 0.0))) < 60.0):
            return self._timeline_cache_fig
        
        # DEBUG: Log function call
        sys.stdout.flush()
        sys.stderr.flush()
        
        # ANALYZE REAL GRADIENT DATA for MITRE ATLAS technique detection
        if self.real_gradient_data is not None:
            print("🔍 Analyzing gradient data for MITRE ATLAS technique detection...", flush=True)
            sys.stdout.flush()
            
            # Analyze gradients in rolling windows
            window_size = 10
            num_windows = min(200, len(self.real_gradient_data) // window_size)
            
            for i in range(num_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                
                if end_idx <= len(self.real_gradient_data):
                    gradient_window = self.real_gradient_data[start_idx:end_idx]
                    
                    # Calculate gradient norm FIRST (before using it)
                    grad_norm = float(np.linalg.norm(gradient_window))
                    
                    # Detect ATLAS techniques from this gradient window
                    techniques = self._detect_atlas_techniques_from_gradients(gradient_window)

                    # Keep only the top-K detections per window to keep the bubble chart responsive
                    try:
                        techniques = sorted(
                            techniques,
                            key=lambda d: (float(d.get('severity', 0.0)), float(d.get('confidence', 0.0))),
                            reverse=True,
                        )[:4]
                    except Exception:
                        techniques = techniques[:4]
                    
                    MAX_EVENTS = 350
                    # Create timeline events for each detected technique with unique timestamps
                    for tech_idx, tech in enumerate(techniques):
                        if len(attack_events) >= MAX_EVENTS:
                            break
                        # Spread timestamps: each window gets its own time range, techniques within window are offset
                        time_offset = datetime.timedelta(seconds=i * 36 + tech_idx * 6)  # 36 sec per window, 6 sec per technique
                        # Use second-resolution timestamps to avoid Plotly/Pandas nanosecond warnings
                        event_time = (current_time - time_offset).replace(microsecond=0)
                        
                        attack_events.append({
                            'timestamp': event_time,
                            'event_type': f"{tech['id']}: {tech['name']}",
                            'severity': float(tech['severity']),
                            'component': f"Gradient_Window_{i}",
                            # Keep raw fields separated for hover rendering (we wrap later)
                            'atlas_id': str(tech.get('id', '')),
                            'atlas_name': str(tech.get('name', '')),
                            'atlas_tactic': str(tech.get('tactic', '')),
                            'atlas_tactic_id': str(tech.get('tactic_id', '')),
                            'atlas_description': str(tech.get('description', '')),
                            'atlas_indicator': str(tech.get('indicator', '')),
                            'atlas_url': str(tech.get('atlas_url', '')) if tech.get('atlas_url') else '',
                            'attribution': f"ATLAS_{tech['id']}_Window_{i}_{tech_idx}",
                            'success_rate': float(tech['confidence']),
                            'data_extracted_mb': float(grad_norm / 100.0 if 'Exfiltrat' in tech['name'] or 'Invert' in tech['name'] else 0),
                            # Back-compat fields
                            'description': str(tech.get('description', '')),
                        })
            
            print(f"✅ Detected {len(attack_events)} MITRE ATLAS detections (events) from REAL gradient data", flush=True)
            sys.stdout.flush()
        else:
            # No real gradient data available
            print("⚠️  No real gradient data loaded - cannot detect MITRE ATLAS techniques")
            print("   Load a .npy gradient file to see technique detections")
        
        # Create MITRE ATLAS bubble visualization from REAL gradient analysis ONLY
        if attack_events and len(attack_events) > 0:
                import pandas as pd
                df = pd.DataFrame(attack_events)
                # Convert timestamp to datetime for proper sorting
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.floor('s')
                df = df.sort_values('timestamp', ascending=True)
                
                print(f"🎯 MITRE ATLAS: {len(attack_events)} detections (events) from gradient analysis")

                # Wrap long hover fields so tooltips never render off-screen.
                import textwrap

                def _wrap_hover(s: object, *, width: int = 72, max_lines: int = 8) -> str:
                    s2 = '' if s is None else str(s)
                    s2 = ' '.join(s2.split())
                    if not s2:
                        return ''
                    lines = textwrap.wrap(
                        s2,
                        width=width,
                        break_long_words=False,
                        break_on_hyphens=False,
                        replace_whitespace=False,
                    )
                    if max_lines and len(lines) > max_lines:
                        lines = lines[:max_lines]
                        lines[-1] = lines[-1].rstrip() + '…'
                    return '<br>'.join(lines)

                if 'atlas_indicator' in df.columns:
                    df['hover_indicator'] = df['atlas_indicator'].apply(lambda s: _wrap_hover(s, width=68, max_lines=3))
                else:
                    df['hover_indicator'] = ''
                if 'atlas_description' in df.columns:
                    df['hover_description'] = df['atlas_description'].apply(lambda s: _wrap_hover(s, width=76, max_lines=8))
                else:
                    df['hover_description'] = ''

                # Professional scatter plot with explicit hover template (consistent + legible).
                fig = px.scatter(
                    df,
                    x='timestamp',
                    y='severity',
                    color='event_type',        # Color by technique (legend hidden)
                    size='success_rate',       # Bubble size = confidence
                    custom_data=[
                        'atlas_id',
                        'atlas_name',
                        'atlas_tactic',
                        'atlas_tactic_id',
                        'success_rate',
                        'hover_indicator',
                        'hover_description',
                        'component',
                        'atlas_url',
                    ],
                    title=f"<b>🛡️ MITRE ATLAS v5.1.1 Technique Tagging (Heuristic)</b><br>"
                          f"<sub>Gradient-derived signals — {len(attack_events)} detections | Source: atlas.mitre.org</sub>",
                    labels={
                        'severity': 'Threat Severity (0–5)',
                        'timestamp': 'Detection Time',
                        'success_rate': 'Confidence',
                        'event_type': 'MITRE ATLAS Technique',
                    },
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )

                fig.update_traces(
                    hovertemplate=(
                        "<b>%{customdata[0]} — %{customdata[1]}</b><br>"
                        "<b>Tactic:</b> %{customdata[2]} (%{customdata[3]})<br>"
                        "<b>Time:</b> %{x}<br>"
                        "<b>Severity:</b> %{y:.2f}/5<br>"
                        "<b>Confidence:</b> %{customdata[4]:.1%}<br>"
                        "<b>Signal:</b><br>%{customdata[5]}<br>"
                        "<b>Description:</b><br>%{customdata[6]}<br>"
                        "<extra></extra>"
                    ),
                    hoverlabel=dict(
                        bgcolor="rgba(5, 10, 20, 0.96)",
                        font_size=13,
                        font_family="Arial",
                        font_color="white",
                        bordercolor="rgba(0, 255, 255, 0.65)",
                        align="left",
                        namelength=-1,
                    ),
                )
            
                # MITRE ATLAS enhanced layout with technique annotations
                # Ensure full timestamp range is visible
                try:
                    time_range = [df['timestamp'].min(), df['timestamp'].max()]
                except Exception:
                    time_range = None
                
                fig.update_layout(
                    xaxis=dict(
                        title="Detection Timeline (Gradient Analysis Window)",
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.3)',
                        type='date',
                        tickformat='%H:%M:%S',
                        range=time_range,  # Ensure all bubbles are visible
                        autorange=True  # Auto-fit to show all data
                    ),
                    yaxis=dict(
                        title="Threat Severity (MITRE ATLAS)",
                        range=[0, 5.5],
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.3)',
                        autorange=False  # Keep fixed 0-5.5 range
                    ),
                    plot_bgcolor='rgba(0,0,0,0.05)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    showlegend=False,  # Disable overwhelming legend - techniques shown in hover
                    # Legend disabled for cleaner view - all info available in hover tooltips
                    height=600,
                    annotations=[
                        # MITRE ATLAS info box - COMPACT, NO OVERLAP
                        dict(
                            text='<b>🛡️ MITRE ATLAS v5.1.1</b><br>' +
                                 f'<b>{len(attack_events)} Techniques</b> | Hover for details',
                            xref="paper", yref="paper",
                            x=0.01, y=0.75,
                            xanchor='left', yanchor='top',
                            showarrow=False,
                            bgcolor="rgba(0,120,180,0.88)",
                            bordercolor="cyan",
                            borderwidth=2,
                            borderpad=8,
                            font=dict(size=9, color='white', family='Arial')
                        )
                    ]
                )
            
                # MITRE ATLAS v5.1.1 Severity Zones - Simplified Labels
                fig.add_hrect(y0=4.5, y1=5.5, fillcolor="rgba(139,0,0,0.15)", 
                             layer="below", line_width=0,
                             annotation_text="CRITICAL", 
                             annotation_position="top right", annotation_font_size=10)
                fig.add_hrect(y0=3.5, y1=4.5, fillcolor="rgba(255,69,0,0.1)", 
                             layer="below", line_width=0,
                             annotation_text="HIGH", 
                             annotation_position="top right", annotation_font_size=10)
                fig.add_hrect(y0=2.5, y1=3.5, fillcolor="rgba(255,165,0,0.08)", 
                             layer="below", line_width=0,
                             annotation_text="MEDIUM", 
                             annotation_position="top right", annotation_font_size=10)
                fig.add_hrect(y0=0, y1=2.5, fillcolor="rgba(50,205,50,0.05)", 
                             layer="below", line_width=0,
                             annotation_text="LOW", 
                             annotation_position="top right", annotation_font_size=10)
            
                # Add MITRE ATLAS tactic annotations for detected technique clusters
                if 'atlas_tactic' in df.columns:
                    unique_tactics = df['atlas_tactic'].unique()
                    # Official MITRE ATLAS v5.1.1 Tactic Colors
                    tactic_colors = {
                        'AI Attack Staging': 'orange',        # AML.TA0001
                        'Resource Development': 'darkorange', # AML.TA0003
                        'AI Model Access': 'yellow',          # AML.TA0000
                        'Reconnaissance': 'cyan',             # AML.TA0002
                        'Exfiltration': 'red',                # AML.TA0010
                        'Execution': 'crimson',               # AML.TA0005
                        'Impact': 'darkred'                   # AML.TA0011
                    }
                    
                    for tactic in unique_tactics[:3]:  # Show only top 3 tactics to avoid clutter
                        tactic_data = df[df['atlas_tactic'] == tactic]
                        if not tactic_data.empty and len(tactic_data) >= 5:  # Only show if 5+ detections
                            # NOTE: Do NOT take the arithmetic mean of timestamps.
                            # Pandas computes means in nanoseconds and can yield fractional seconds, which triggers:
                            #   "Discarding nonzero nanoseconds in conversion." in Plotly JSON serialization.
                            # Use a representative timestamp (middle point in sorted order) at second resolution.
                            tactic_times = tactic_data['timestamp'].sort_values()
                            avg_time = tactic_times.iloc[len(tactic_times) // 2]
                            try:
                                avg_time = avg_time.to_pydatetime().replace(microsecond=0)
                            except Exception:
                                pass
                            max_severity = tactic_data['severity'].max()
                            color = tactic_colors.get(tactic, 'white')
                            
                            # Professional tactic cluster annotation
                            fig.add_annotation(
                                x=avg_time,
                                y=max_severity + 0.35,
                                text=f"<b>{tactic}</b><br><i>({len(tactic_data)} detections)</i>",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor=color,
                                arrowwidth=2,
                                ax=0,
                                ay=-40,
                                font=dict(size=11, color='white', family='Arial', weight='bold'),
                                bgcolor='rgba(0,0,0,0.85)',
                                bordercolor=color,
                                borderwidth=2,
                                borderpad=8
                            )
            
        else:
            # No synthetic/demo fallback: show an empty timeline with guidance.
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.update_layout(
                title="⚠️ Attack Timeline Unavailable (no gradient data loaded)",
                xaxis_title="Attack Time",
                yaxis_title="Threat Level",
                plot_bgcolor='rgba(0,0,0,0.05)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=600
            )
            fig.add_annotation(
                x=0.5, y=0.5, xref='paper', yref='paper',
                text="No real data loaded. Provide --gradient-file / --attention-file (or --batch-dir) to analyze real captures.",
                showarrow=False,
                font=dict(size=14, color='white'),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='rgba(255,255,255,0.35)',
                borderwidth=1,
                borderpad=10,
            )
        
        # Store cache
        try:
            self._timeline_cache_key = cache_key
            self._timeline_cache_fig = fig
            self._timeline_cache_at = __import__('time').time()
        except Exception:
            pass
        return fig

    def create_gradient_leakage_dashboard(self, use_real_data=True):
        """Create comprehensive gradient leakage security dashboard"""
        data = self.generate_gradient_leakage_data(use_real_data)
        
        # Create subplots
        fig = self.sp.make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                f'🚨 Privacy Budget Exhaustion (ε-DP) - {data["model_name"]}',
                f'⚠️ Gradient Norm Spikes - {data["data_source"].title()} Data',
                '📈 Leakage Risk Proxies Over Time',
                f'🔍 Layer-wise Vulnerability - {data["model_name"]}',
                '📊 Real-time Threat Score',
                '🛡️ Defense Recommendations'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"type": "indicator"}, {"secondary_y": False}]],
            vertical_spacing=0.12
        )
        
        # 1. Privacy Budget Exhaustion
        privacy_color = ['red' if x > self.PRIVACY_BUDGET_LIMIT else 'orange' if x > 5.0 else 'green' 
                        for x in data['privacy_budget']]
        
        # Convert timestamps to datetime objects for better display
        from datetime import datetime
        time_labels = [datetime.fromtimestamp(ts).strftime('%H:%M:%S') if isinstance(ts, (int, float)) else str(ts) 
                      for ts in data['steps']]
        
        fig.add_trace(
            self.go.Scatter(
                x=data['steps'],
                y=data['privacy_budget'],
                mode='lines+markers',
                name='Privacy Budget (ε)',
                line=dict(color='blue', width=2),
                marker=dict(color=privacy_color, size=4),
                hovertemplate='Time: %{customdata}<br>Privacy Budget: %{y:.2f}ε<br><extra></extra>',
                customdata=[datetime.fromtimestamp(ts).strftime('%H:%M:%S') for ts in data['steps']]
            ),
            row=1, col=1
        )
        
        # Critical threshold line
        fig.add_hline(y=self.PRIVACY_BUDGET_LIMIT, line_dash="dash", line_color="red",
                     annotation_text=f"CRITICAL: Privacy Exhausted (>{self.PRIVACY_BUDGET_LIMIT}ε)", row=1, col=1)
        
        # 2. Gradient Norm Spikes
        grad_color = ['red' if x > self.CRITICAL_GRADIENT_NORM else 'orange' if x > 2.0 else 'green' 
                     for x in data['gradient_norms']]
        
        fig.add_trace(
            self.go.Scatter(
                x=data['steps'],
                y=data['gradient_norms'],
                mode='lines+markers',
                name='Gradient L2 Norm',
                line=dict(color='cyan', width=2),
                marker=dict(color=grad_color, size=4),
                hovertemplate='Step: %{x}<br>Gradient Norm: %{y:.2f}<br><extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.add_hline(y=self.CRITICAL_GRADIENT_NORM, line_dash="dash", line_color="red",
                     annotation_text=f"CRITICAL: Data Extraction Risk (>{self.CRITICAL_GRADIENT_NORM})", row=1, col=2)
        
        # 3. Leakage risk proxies (0–1). These are heuristic indicators, not measured attack success rates.
        fig.add_trace(
            self.go.Scatter(
                x=data['steps'],
                y=data['membership_inference_success'],
                mode='lines',
                name='Membership Inference Risk Proxy',
                line=dict(color='red', width=2),
                hovertemplate='Step: %{x}<br>MI Risk Proxy: %{y:.1%}<br><extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            self.go.Scatter(
                x=data['steps'],
                y=data['reconstruction_success'],
                mode='lines',
                name='Reconstruction Risk Proxy',
                line=dict(color='orange', width=2),
                hovertemplate='Step: %{x}<br>Reconstruction Risk Proxy: %{y:.1%}<br><extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_hline(
            y=self.MEMBERSHIP_INFERENCE_THRESHOLD,
            line_dash="dash",
            line_color="red",
            annotation_text=f"MI Risk Threshold ({self.MEMBERSHIP_INFERENCE_THRESHOLD:.0%})",
            row=2,
            col=1,
        )
        fig.add_hline(y=self.RECONSTRUCTION_RISK_THRESHOLD, line_dash="dash", line_color="orange",
                     annotation_text=f"Reconstruction Threshold ({self.RECONSTRUCTION_RISK_THRESHOLD:.0%})", row=2, col=1)
        
        # 4. Layer-wise Vulnerability
        layers = list(data['layer_vulnerabilities'].keys())
        vulnerabilities = list(data['layer_vulnerabilities'].values())
        vuln_colors = ['red' if v > 0.7 else 'orange' if v > 0.5 else 'green' for v in vulnerabilities]
        
        fig.add_trace(
            self.go.Bar(
                x=layers,
                y=vulnerabilities,
                name='Layer Vulnerability',
                marker_color=vuln_colors,
                hovertemplate='%{x}<br>Vulnerability: %{y:.1%}<br><extra></extra>'
            ),
            row=2, col=2
        )
        
        # 5. Real-time Threat Score
        current_privacy = data['privacy_budget'][-1]
        current_gradient = data['gradient_norms'][-1]
        current_mi = data['membership_inference_success'][-1]
        
        # Calculate composite threat score
        privacy_threat = min(current_privacy / self.PRIVACY_BUDGET_LIMIT, 1.0)
        gradient_threat = min(current_gradient / self.CRITICAL_GRADIENT_NORM, 1.0)
        attack_threat = current_mi
        
        composite_threat = (privacy_threat * 0.4 + gradient_threat * 0.4 + attack_threat * 0.2)
        
        threat_color = 'red' if composite_threat > 0.8 else 'orange' if composite_threat > 0.5 else 'green'
        threat_level = 'CRITICAL' if composite_threat > 0.8 else 'HIGH' if composite_threat > 0.5 else 'MEDIUM'
        
        fig.add_trace(
            self.go.Indicator(
                mode="gauge+number+delta",
                value=composite_threat * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Threat Level: {threat_level}"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': threat_color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ),
            row=3, col=1
        )
        
        # 6. Defense Recommendations with Red/Blue Team Guidance
        recommendations = []
        red_team_actions = []
        blue_team_actions = []
        
        # Critical security findings
        if current_privacy > self.PRIVACY_BUDGET_LIMIT:
            recommendations.append(f"🚨 URGENT: Privacy budget exhausted ({current_privacy:.1f}ε > {self.PRIVACY_BUDGET_LIMIT}ε)")
            blue_team_actions.append("🛡️ BLUE: Stop training immediately, implement DP-SGD")
            red_team_actions.append("🔴 Verification: Privacy exhausted — run calibrated membership-inference evaluation (authorized)")
            
        if current_gradient > self.CRITICAL_GRADIENT_NORM:
            recommendations.append(f"⚠️ HIGH: Large gradient norms ({current_gradient:.2f} > {self.CRITICAL_GRADIENT_NORM})")
            blue_team_actions.append("🛡️ BLUE: Enable gradient clipping, reduce learning rate")
            red_team_actions.append("🔴 Verification: Large gradients — evaluate inversion susceptibility on the same gradients (authorized)")
            
        if current_mi > self.MEMBERSHIP_INFERENCE_THRESHOLD:
            recommendations.append(f"🎯 CRITICAL: High MI risk proxy ({current_mi:.1%} > {self.MEMBERSHIP_INFERENCE_THRESHOLD:.0%})")
            blue_team_actions.append("🛡️ BLUE: Increase DP noise, implement membership defense")
            red_team_actions.append("🔴 Verification: MI proxy elevated — evaluate reconstruction risk using your approved benchmark (authorized)")
            
        if max(vulnerabilities) > 0.7:
            vulnerable_layers = [layer for layer, vuln in data['layer_vulnerabilities'].items() if vuln > 0.7]
            recommendations.append(f"🔍 Vulnerable layers: {', '.join(vulnerable_layers)}")
            blue_team_actions.append(f"🛡️ BLUE: Secure {', '.join(vulnerable_layers)} with layer-wise DP")
            red_team_actions.append(f"🔴 Verification: Inspect {', '.join(vulnerable_layers)} for concentrated leakage signals (authorized)")
        
        # Model-specific intelligence
        if data['data_source'] == 'real':
            recommendations.append(f"📊 Real {data['model_name']} data analysis")
            blue_team_actions.append("🛡️ BLUE: Monitor real-time gradient leakage patterns")
            red_team_actions.append("🔴 Verification: Real gradients available — run your approved evaluation matrix (authorized)")
        
        # Threat level specific guidance
        if composite_threat > 0.8:
            blue_team_actions.append("🛡️ BLUE: CRITICAL - Activate incident response, halt training")
            red_team_actions.append("🔴 Verification: CRITICAL window — prioritize leakage evaluation and incident triage (authorized)")
        elif composite_threat > 0.5:
            blue_team_actions.append("🛡️ BLUE: HIGH - Implement emergency defenses, increase monitoring")
            red_team_actions.append("🔴 Verification: HIGH window — run focused validation on MI/inversion proxies (authorized)")
        else:
            blue_team_actions.append("🛡️ BLUE: MEDIUM - Maintain baseline defenses, routine monitoring")
            red_team_actions.append("🔴 Verification: MEDIUM window — continue measurement collection (authorized)")
        
        # Combine all recommendations
        all_recommendations = recommendations + [""] + blue_team_actions + [""] + red_team_actions
        
        if not recommendations:
            all_recommendations = ["✅ All metrics within acceptable ranges",
                                 "🛡️ BLUE: Maintain current security posture",
                                 "🔴 RED: No immediate attack vectors identified"]
        
        # Create text trace for recommendations with proper formatting
        # Split recommendations into multiple lines to prevent truncation
        max_chars_per_line = 45
        formatted_recommendations = []
        
        for i, rec in enumerate(all_recommendations):
            if not rec:  # Empty line for spacing
                formatted_recommendations.append("")
                continue
                
            if len(rec) > max_chars_per_line:
                # Split long recommendations into multiple lines
                words = rec.split()
                current_line = ""
                for word in words:
                    if len(current_line + word + " ") <= max_chars_per_line:
                        current_line += word + " "
                    else:
                        if current_line:
                            formatted_recommendations.append(current_line.strip())
                        current_line = word + " "
                if current_line.strip():
                    formatted_recommendations.append(current_line.strip())
            else:
                formatted_recommendations.append(rec)
        
        rec_text = "<br>".join(formatted_recommendations)
        
        fig.add_trace(
            self.go.Scatter(
                x=[0.5],
                y=[0.5],
                mode='text',
                text=[rec_text],
                textposition='middle center',
                textfont=dict(size=9, family="Courier New"),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'🛡️ NeurInSpectre: TTD Security Analysis - {data["model_name"]} ({data["data_source"].title()} Data)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=1200,
            showlegend=True,
            template='plotly_dark'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Training Steps", row=1, col=1)
        fig.update_yaxes(title_text="Privacy Budget (ε)", row=1, col=1)
        
        fig.update_xaxes(title_text="Training Steps", row=1, col=2)
        fig.update_yaxes(title_text="Gradient L2 Norm", row=1, col=2)
        
        fig.update_xaxes(title_text="Training Steps", row=2, col=1)
        fig.update_yaxes(title_text="Risk Proxy (0–1)", row=2, col=1)
        
        fig.update_xaxes(title_text="Model Layers", row=2, col=2)
        fig.update_yaxes(title_text="Vulnerability Score", row=2, col=2)
        
        # Hide axes for text subplot
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=3, col=2)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=3, col=2)
        
        return fig
    
    def run_ttd_analysis(self, output_file="ttd_gradient_leakage_dashboard.html", use_real_data=True):
        """Run TTD analysis and generate dashboard"""
        print("🚀 Starting TTD (Time to Detection) Analysis...")
        print(f"📊 Analyzing gradient leakage security metrics for {self.model_name}...")
        print(f"🎯 Using {'real' if use_real_data and self.real_gradient_data is not None else 'simulated'} data")
        
        # Generate dashboard
        fig = self.create_gradient_leakage_dashboard(use_real_data)
        
        # Save dashboard
        self.plot(fig, filename=output_file, auto_open=False)
        
        print(f"✅ TTD Dashboard generated: {output_file}")
        print("\n📋 Security Summary:")
        
        # Generate summary report
        data = self.generate_gradient_leakage_data(use_real_data)
        current_privacy = data['privacy_budget'][-1]
        current_gradient = data['gradient_norms'][-1]
        current_mi = data['membership_inference_success'][-1]
        max_vuln_layer = max(data['layer_vulnerabilities'], key=data['layer_vulnerabilities'].get)
        max_vuln_score = data['layer_vulnerabilities'][max_vuln_layer]
        
        print(f"   🤖 Model: {self.model_name} ({data['data_source']} data)")
        print(f"   🔒 Current Privacy Budget: {current_privacy:.2f}ε (Limit: {self.PRIVACY_BUDGET_LIMIT}ε)")
        print(f"   📈 Current Gradient Norm: {current_gradient:.2f} (Critical: {self.CRITICAL_GRADIENT_NORM})")
        print(f"   🎯 MI Risk Proxy: {current_mi:.1%} (Threshold: {self.MEMBERSHIP_INFERENCE_THRESHOLD:.0%})")
        print(f"   🔍 Most Vulnerable Layer: {max_vuln_layer} ({max_vuln_score:.1%})")
        
        if current_privacy > self.PRIVACY_BUDGET_LIMIT:
            print("   ⚠️  WARNING: Privacy budget exhausted!")
        if current_gradient > self.CRITICAL_GRADIENT_NORM:
            print("   ⚠️  WARNING: Critical gradient leakage detected!")
        if current_mi > self.MEMBERSHIP_INFERENCE_THRESHOLD:
            print("   ⚠️  WARNING: MI risk proxy above threshold!")
        
        return output_file

    def _generate_dynamic_atlas_advisories(self, data, threat_score, current_gradient, current_privacy, current_mi):
        """Generate dynamic ATLAS-based advisories based on real-time graph analysis"""
        
        # Analyze current threat patterns from graph data
        gradient_trend = self._analyze_gradient_trend(data.get('gradient_norms', []))
        privacy_status = self._analyze_privacy_budget(data.get('privacy_budget', []))
        attack_pattern = self._detect_attack_pattern(data, threat_score)
        
        # Dynamic Red Team Advisories based on graph analysis
        red_team_advisories = self._generate_red_team_actions(
            gradient_trend, privacy_status, attack_pattern, current_gradient, current_privacy
        )
        
        # Dynamic Blue Team Advisories based on threat landscape
        blue_team_advisories = self._generate_blue_team_actions(
            gradient_trend, privacy_status, attack_pattern, threat_score, current_mi
        )
        
        return red_team_advisories, blue_team_advisories
    
    def _analyze_gradient_trend(self, gradient_norms):
        """Analyze gradient trend patterns for tactical insights"""
        import numpy as np
        
        if len(gradient_norms) < 10:
            return {"trend": "insufficient_data", "volatility": 0.0, "peak_detected": False}
        
        recent_grads = gradient_norms[-20:]  # Last 20 data points
        older_grads = gradient_norms[-40:-20] if len(gradient_norms) >= 40 else gradient_norms[:-20]
        
        recent_avg = np.mean(recent_grads)
        older_avg = np.mean(older_grads) if older_grads else recent_avg
        
        volatility = np.std(recent_grads)
        peak_detected = any(g > self.CRITICAL_GRADIENT_NORM for g in recent_grads[-5:])
        
        if recent_avg > older_avg * 1.5:
            trend = "escalating"
        elif recent_avg < older_avg * 0.7:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "volatility": volatility,
            "peak_detected": peak_detected,
            "recent_avg": recent_avg,
            "change_rate": (recent_avg - older_avg) / older_avg if older_avg > 0 else 0.0
        }
    
    def _analyze_privacy_budget(self, privacy_budget):
        """Analyze privacy budget consumption patterns"""
        if len(privacy_budget) < 5:
            return {"status": "insufficient_data", "consumption_rate": 0.0, "critical_threshold": False}
        
        recent_budget = privacy_budget[-10:]
        consumption_rate = (recent_budget[-1] - recent_budget[0]) / len(recent_budget)
        
        status = "critical" if recent_budget[-1] > self.PRIVACY_BUDGET_LIMIT * 0.8 else \
                "warning" if recent_budget[-1] > self.PRIVACY_BUDGET_LIMIT * 0.6 else "normal"
        
        return {
            "status": status,
            "consumption_rate": consumption_rate,
            "critical_threshold": recent_budget[-1] > self.PRIVACY_BUDGET_LIMIT * 0.9,
            "current_level": recent_budget[-1]
        }
    
    def _detect_attack_pattern(self, data, threat_score):
        """Detect specific attack patterns from combined metrics"""
        import numpy as np
        
        gradient_norms = data.get('gradient_norms', [])
        privacy_budget = data.get('privacy_budget', [])
        vulnerability_scores = data.get('vulnerability_scores', [])
        
        # Pattern detection logic
        if len(gradient_norms) < 5:
            return {"pattern": "insufficient_data", "confidence": 0.0, "atlas_technique": None}
        
        recent_grads = gradient_norms[-5:]
        recent_privacy = privacy_budget[-5:] if len(privacy_budget) >= 5 else [5.0] * 5
        recent_vuln = vulnerability_scores[-5:] if len(vulnerability_scores) >= 5 else [0.5] * 5
        
        # Gradient Leakage Pattern (AML.T0020)
        if np.mean(recent_grads) > 3.0 and np.mean(recent_privacy) > 7.0:
            return {
                "pattern": "gradient_leakage",
                "confidence": min(0.95, threat_score + 0.1),
                "atlas_technique": "AML.T0020",
                "technique_name": "Gradient Leakage",
                "severity": "CRITICAL"
            }
        
        # Membership Inference Pattern (AML.T0048)
        elif threat_score > 0.7 and np.std(recent_grads) > 1.5:
            return {
                "pattern": "membership_inference",
                "confidence": threat_score,
                "atlas_technique": "AML.T0048",
                "technique_name": "Membership Inference",
                "severity": "HIGH"
            }
        
        # Model Inversion Pattern (AML.T0054)
        elif np.mean(recent_vuln) > 0.8:
            return {
                "pattern": "model_inversion",
                "confidence": np.mean(recent_vuln),
                "atlas_technique": "AML.T0054",
                "technique_name": "Model Inversion",
                "severity": "HIGH"
            }
        
        # Adversarial Examples Pattern (AML.T0043)
        elif np.mean(recent_grads) > 2.0 and threat_score > 0.6:
            return {
                "pattern": "adversarial_examples",
                "confidence": threat_score,
                "atlas_technique": "AML.T0043",
                "technique_name": "Adversarial Examples",
                "severity": "MEDIUM"
            }
        
        else:
            return {
                "pattern": "baseline_monitoring",
                "confidence": 0.3,
                "atlas_technique": "AML.T0018",
                "technique_name": "Property Inference",
                "severity": "LOW"
            }
    
    def _generate_red_team_actions(self, gradient_trend, privacy_status, attack_pattern, current_gradient, current_privacy):
        """Generate dynamic red team actions based on current threat landscape"""
        
        pattern_type = attack_pattern.get("pattern", "baseline_monitoring")
        atlas_technique = attack_pattern.get("atlas_technique", "AML.T0018")
        confidence = attack_pattern.get("confidence", 0.3)
        
        base_actions = {
            "gradient_leakage": {
                "immediate_actions": [
                    "🎯 Exploit high gradient variance for data reconstruction",
                    "⚡ Launch federated learning gradient inversion attacks",
                    "🔥 Target model parameters through gradient analysis",
                    "🎪 Implement batch reconstruction techniques"
                ],
                "cve_refs": ["CVE-2024-47792", "CVE-2025-0234"],
                "mitre_techniques": ["AML.T0020", "AML.T0054"],
                "success_probability": f"{confidence*100:.1f}%"
            },
            "membership_inference": {
                "immediate_actions": [
                    "🕵️ Deploy membership inference attacks on training data",
                    "📊 Analyze model confidence patterns for inference",
                    "🎯 Target specific user data through model queries",
                    "🔍 Exploit gradient instability for membership detection"
                ],
                "cve_refs": ["CVE-2024-45891"],
                "mitre_techniques": ["AML.T0048", "AML.T0018"],
                "success_probability": f"{confidence*100:.1f}%"
            },
            "model_inversion": {
                "immediate_actions": [
                    "🔄 Launch model inversion attacks for data reconstruction",
                    "🎨 Generate synthetic training data from model behavior",
                    "🎯 Target high-confidence predictions for inversion",
                    "📈 Exploit model overconfidence for data extraction"
                ],
                "cve_refs": ["CVE-2024-45891"],
                "mitre_techniques": ["AML.T0054", "AML.T0048"],
                "success_probability": f"{confidence*100:.1f}%"
            },
            "adversarial_examples": {
                "immediate_actions": [
                    "🎭 Deploy adversarial examples for model evasion",
                    "🔧 Craft input perturbations for bypass attacks",
                    "🎯 Target model decision boundaries",
                    "⚡ Implement iterative adversarial optimization"
                ],
                "cve_refs": ["CVE-2025-0156"],
                "mitre_techniques": ["AML.T0043", "AML.T0051"],
                "success_probability": f"{confidence*100:.1f}%"
            }
        }
        
        # Get base actions for detected pattern
        actions = base_actions.get(pattern_type, {
            "immediate_actions": [
                "📊 Monitor for attack opportunities",
                "🔍 Analyze model behavior patterns",
                "🎯 Prepare for future attack vectors",
                "⚡ Maintain surveillance on model updates"
            ],
            "cve_refs": [],
            "mitre_techniques": [atlas_technique],
            "success_probability": f"{confidence*100:.1f}%"
        })
        
        # Add dynamic context based on current metrics
        if gradient_trend["trend"] == "escalating":
            actions["immediate_actions"].insert(0, "🚨 URGENT: Exploit escalating gradient leakage window")
        
        if privacy_status["status"] == "critical":
            actions["immediate_actions"].insert(0, "💥 CRITICAL: Privacy budget near exhaustion - immediate attack window")
        
        if current_gradient > self.CRITICAL_GRADIENT_NORM:
            actions["immediate_actions"].append("🔥 High gradient norm detected - prime for extraction attacks")
        
        return {
            "pattern_detected": pattern_type.replace("_", " ").title(),
            "atlas_technique": atlas_technique,
            "confidence": f"{confidence*100:.1f}%",
            "actions": actions["immediate_actions"][:5],  # Limit to 5 actions
            "cve_refs": actions["cve_refs"],
            "mitre_techniques": actions["mitre_techniques"],
            "success_probability": actions["success_probability"],
            "threat_level": attack_pattern.get("severity", "MEDIUM")
        }
    
    def _generate_blue_team_actions(self, gradient_trend, privacy_status, attack_pattern, threat_score, current_mi):
        """Generate dynamic blue team defense actions based on threat analysis"""
        
        pattern_type = attack_pattern.get("pattern", "baseline_monitoring")
        severity = attack_pattern.get("severity", "LOW")
        
        defense_strategies = {
            "gradient_leakage": {
                "immediate_actions": [
                    "🛡️ URGENT: Implement gradient clipping (norm ≤ 1.0)",
                    "🔒 Deploy differential privacy with ε ≤ 0.1",
                    "🎯 Enable secure aggregation protocols",
                    "📊 Monitor gradient variance patterns",
                    "⚡ Implement noise injection in gradients"
                ],
                "response_time": "IMMEDIATE",
                "priority": "CRITICAL"
            },
            "membership_inference": {
                "immediate_actions": [
                    "🔍 Deploy membership inference detection systems",
                    "🛡️ Implement output perturbation mechanisms",
                    "📊 Monitor query patterns for inference attacks",
                    "🎯 Enable confidence score masking",
                    "⚡ Deploy adversarial training defenses"
                ],
                "response_time": "WITHIN 1 HOUR",
                "priority": "HIGH"
            },
            "model_inversion": {
                "immediate_actions": [
                    "🔄 Implement model inversion detection",
                    "🛡️ Deploy output sanitization systems",
                    "📈 Monitor for synthetic data generation attempts",
                    "🎯 Enable prediction confidence limiting",
                    "⚡ Implement query rate limiting"
                ],
                "response_time": "WITHIN 2 HOURS",
                "priority": "HIGH"
            },
            "adversarial_examples": {
                "immediate_actions": [
                    "🎭 Deploy adversarial detection systems",
                    "🔧 Implement input validation and sanitization",
                    "🎯 Enable adversarial training protocols",
                    "⚡ Monitor for input perturbation patterns",
                    "🛡️ Deploy ensemble defense mechanisms"
                ],
                "response_time": "WITHIN 4 HOURS",
                "priority": "MEDIUM"
            }
        }
        
        # Get base defense for detected pattern
        defense = defense_strategies.get(pattern_type, {
            "immediate_actions": [
                "📊 Maintain baseline monitoring",
                "🔍 Monitor for anomalous patterns",
                "🎯 Update threat detection rules",
                "⚡ Maintain defense readiness",
                "🛡️ Review security configurations"
            ],
            "response_time": "ROUTINE",
            "priority": "LOW"
        })
        
        # Add dynamic context based on current threat landscape
        if gradient_trend["trend"] == "escalating":
            defense["immediate_actions"].insert(0, "🚨 ESCALATION: Increase gradient monitoring frequency")
        
        if privacy_status["status"] == "critical":
            defense["immediate_actions"].insert(0, "💥 CRITICAL: Implement emergency privacy protection")
        
        if threat_score > 0.8:
            defense["immediate_actions"].insert(0, "🔴 HIGH THREAT: Activate incident response protocols")
        
        if current_mi > self.MEMBERSHIP_INFERENCE_THRESHOLD:
            defense["immediate_actions"].append("🕵️ High MI risk - deploy counter-inference measures")
        
        return {
            "pattern_detected": pattern_type.replace("_", " ").title(),
            "threat_level": severity,
            "priority": defense.get("priority", "MEDIUM"),
            "response_time": defense.get("response_time", "ROUTINE"),
            "actions": defense["immediate_actions"][:5],  # Limit to 5 actions
            "escalation_required": threat_score > 0.8 or severity == "CRITICAL",
            "monitoring_frequency": "CONTINUOUS" if severity == "CRITICAL" else "EVERY 5 MINUTES"
        }

def run_dashboard(args):
    """Run interactive TTD dashboard (called from __main__.py)"""
    print(f"🚀 Starting Interactive TTD Dashboard...")
    print(f"🤖 Model: {args.model}")
    print(f"🌐 Host: {args.host}:{args.port}")
    print(f"🔧 Debug: {args.debug}")
    
    # Import required components
    from datetime import datetime
    
    # Import dash components
    dash, dcc, html, Input, Output, dbc = _get_dash()
    if dash is None:
        print("❌ Could not load dash components. Please install with: pip install dash dash-bootstrap-components")
        return 1
    
    print("✅ dash available")
    
    # Check other dependencies
    try:
        import dash_bootstrap_components as dbc
        print("✅ dash_bootstrap_components available")
    except ImportError:
        print("❌ dash_bootstrap_components not available")
        return 1
    
    try:
        import plotly.graph_objects as go
        print("✅ plotly available")
    except ImportError:
        print("❌ plotly not available")
        return 1
    
    try:
        import pandas as pd
        print("✅ pandas available")
    except ImportError:
        print("❌ pandas not available")
        return 1
    
    try:
        import numpy as np
        print("✅ numpy available")
    except ImportError:
        print("❌ numpy not available")
        return 1
    
    # Create TTD dashboard instance
    ttd = TTDDashboard(
        model_name=args.model,
        privacy_budget_limit=getattr(args, "privacy_limit", 3.0),
        dp_noise_multiplier=getattr(args, "dp_noise_multiplier", None),
        dp_sample_rate=getattr(args, "dp_sample_rate", None),
        dp_delta=getattr(args, "dp_delta", 1e-5),
        allow_simulated=False,
    )
    
    # Resolve data inputs (no implicit "backup" datasets; only use explicitly provided paths)
    current_dir = os.getcwd()
    print("🔍 Loading dashboard data inputs")
    print(f"📁 Directory: {current_dir}")
    
    # Resolve gradient file (CLI-specified only)
    gradient_file = None
    if getattr(args, 'gradient_file', None):
        gf = os.path.abspath(args.gradient_file)
        if os.path.exists(gf):
            gradient_file = gf
            print(f"   ✅ Using CLI-specified: {gradient_file}")
        else:
            print(f"   ⚠️  Gradient file not found: {gf}")
    
    # Resolve attention file
    attention_file = None
    if getattr(args, 'attention_file', None):
        af = os.path.abspath(args.attention_file)
        if os.path.exists(af):
            attention_file = af
            print(f"   ✅ Using attention file from CLI: {attention_file}")
        else:
            print(f"   ⚠️  Attention file not found: {af}")
    
    # FORCE load real data if found (no fallback to simulated)
    batch_dir = getattr(args, 'batch_dir', None)
    privacy_file = getattr(args, 'privacy_file', None)
    if batch_dir and not os.path.exists(batch_dir):
        print(f"   ⚠️  Batch directory not found: {os.path.abspath(batch_dir)}")
        batch_dir = None
    if privacy_file and not os.path.exists(privacy_file):
        print(f"   ⚠️  Privacy file not found: {os.path.abspath(privacy_file)}")
        privacy_file = None

    if gradient_file or attention_file or batch_dir or privacy_file:
        print(f"🚀 FORCE LOADING REAL ADVERSARIAL ATTACK DATA:")
        if gradient_file:
            print(f"   ⚡ Gradient data: {gradient_file} ({os.path.getsize(gradient_file) / (1024*1024):.1f}MB)")
        if attention_file:
            print(f"   ⚡ Attention data: {attention_file} ({os.path.getsize(attention_file) / (1024):.1f}KB)")
        
        # Force load the real data
        ttd.load_real_data(
            gradient_file=gradient_file,
            attention_file=attention_file,
            batch_dir=batch_dir,
            privacy_file=privacy_file,
        )
        
        # Calculate total data size
        total_size = 0
        if gradient_file and os.path.exists(gradient_file):
            total_size += os.path.getsize(gradient_file)
        if attention_file and os.path.exists(attention_file):
            total_size += os.path.getsize(attention_file)
        
        print(f"✅ SUCCESSFULLY LOADED {total_size / (1024*1024):.1f}MB of REAL ADVERSARIAL ATTACK DATA")
        print(f"🔄 READY FOR ROLLING/SCROLLING REAL ATTACK DATA ANALYSIS")
    else:
        print("⚠️  No real data was loaded for the dashboard.")
        print("   Provide --gradient-file / --attention-file (or --batch-dir) to analyze real captures.")
    
    # Initialize Dash app with Apple Silicon stability settings
    app = dash.Dash(__name__, 
                    external_stylesheets=[dbc.themes.CYBORG],
                    suppress_callback_exceptions=True,  # Critical for Apple Silicon stability
                    update_title=None)  # Prevent title update crashes
    
    # Helper: build dynamic model dropdown options including CLI model
    def _pretty_label(model_id: str) -> str:
        if not model_id:
            return "Select"
        base = model_id.split('/')[-1]
        pretty = base.replace('-', ' ')
        # Common prettifications
        pretty = pretty.replace('gpt neo', 'GPT‑Neo').replace('gpt2', 'GPT‑2')
        pretty = pretty.replace('roberta', 'RoBERTa').replace('bert', 'BERT').replace('distil', 'Distil')
        return pretty.title().replace('Gpt‑Neo', 'GPT‑Neo').replace('Gpt‑2', 'GPT‑2').replace('Roberta', 'RoBERTa')
    
    def _scan_local_models() -> list:
        """Scan for locally cached HuggingFace models - Multi-method detection"""
        detected_models = []
        
        # Method 1: Use huggingface_hub scan_cache_dir (most reliable)
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            
            for repo in cache_info.repos:
                model_id = repo.repo_id
                if repo.repo_type == "model":  # Only models, not datasets
                    size_mb = repo.size_on_disk / (1024 * 1024)
                    detected_models.append({
                        'label': f'📦 {_pretty_label(model_id)} ({size_mb:.0f}MB)',
                        'value': model_id,
                        'size': repo.size_on_disk
                    })
            
            print(f"✅ Method 1: Detected {len(detected_models)} models via huggingface_hub.scan_cache_dir()")
        
        except Exception as e:
            print(f"⚠️ Method 1 failed ({e}), trying fallback...")
            
            # Method 2: Manual scan of cache directory
            try:
                cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
                if cache_dir.exists():
                    for model_dir in cache_dir.iterdir():
                        if model_dir.is_dir() and model_dir.name.startswith('models--'):
                            # Extract model name: models--org--modelname -> org/modelname
                            parts = model_dir.name.replace('models--', '').split('--')
                            if len(parts) == 2:
                                model_id = f"{parts[0]}/{parts[1]}"
                            elif len(parts) == 1:
                                model_id = parts[0]
                            else:
                                continue
                            
                            # Check if model has snapshots (actual downloaded files)
                            snapshot_dir = model_dir / 'snapshots'
                            if snapshot_dir.exists():
                                # Get latest snapshot
                                snapshots = list(snapshot_dir.iterdir())
                                if snapshots:
                                    # Check for model files
                                    latest_snapshot = snapshots[0]
                                    has_model = any(
                                        (latest_snapshot / f).exists() 
                                        for f in ['pytorch_model.bin', 'model.safetensors', 'config.json']
                                    )
                                    
                                    if has_model:
                                        detected_models.append({
                                            'label': f'💾 {_pretty_label(model_id)} (local)',
                                            'value': model_id
                                        })
                    
                    print(f"✅ Method 2: Detected {len(detected_models)} models via cache directory scan")
                else:
                    print(f"ℹ️ HuggingFace cache not found at: {cache_dir}")
            
            except Exception as e2:
                print(f"⚠️ Method 2 failed: {e2}")
        
        # Method 3: Check for Ollama models (Mac-specific)
        try:
            ollama_dir = Path.home() / '.ollama' / 'models'
            if ollama_dir.exists():
                ollama_models = []
                for manifest in ollama_dir.glob('*/manifest.json'):
                    model_name = manifest.parent.name
                    ollama_models.append({
                        'label': f'🦙 {model_name} (Ollama)',
                        'value': f'ollama:{model_name}'
                    })
                if ollama_models:
                    detected_models.extend(ollama_models)
                    print(f"✅ Method 3: Detected {len(ollama_models)} Ollama models")
        except Exception:
            pass
        
        # Sort by size if available, otherwise alphabetically
        if detected_models and 'size' in detected_models[0]:
            detected_models.sort(key=lambda x: x.get('size', 0), reverse=True)
        else:
            detected_models.sort(key=lambda x: x['label'])
        
        return detected_models

    # Default fallback models (always available for download)
    base_model_options = [
        {'label': 'DistilBERT', 'value': 'distilbert-base-uncased'},
        {'label': 'BERT', 'value': 'bert-base-uncased'},
        {'label': 'GPT-2', 'value': 'gpt2'},
        {'label': 'RoBERTa', 'value': 'roberta-base'},
        {'label': 'T5', 'value': 't5-base'}
    ]
    
    # Auto-detect locally cached models
    print("🔍 Scanning for locally cached models (HuggingFace + Ollama)...")
    local_models = _scan_local_models()
    
    # Combine: local models first (with icons), then separator, then fallback defaults
    if local_models:
        all_model_options = local_models + [
            {'label': '─────── Available for Download ───────', 'value': '', 'disabled': True}
        ] + base_model_options
        print(f"✅ Dropdown populated with {len(local_models)} local + {len(base_model_options)} downloadable models")
    else:
        all_model_options = base_model_options
        print("ℹ️ No local models detected, using downloadable models list")
    
    # Ensure CLI model appears in dropdown with a friendly label
    if args.model and not any(opt.get('value') == args.model for opt in all_model_options):
        all_model_options.insert(0, {'label': f'⭐ {_pretty_label(args.model)} (CLI selected)', 'value': args.model})

    # Inject custom CSS for dropdown styling
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                /* STANDARD WEB FONT SIZES - Default browser sizes */
                * {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif !important;
                }
                
                html {
                    font-size: 14px !important;
                }
                
                body {
                    font-size: 14px !important;
                    line-height: 1.5 !important;
                }
                
                /* Headings - Standard Web Sizes */
                h1, .h1 {
                    font-size: 24px !important;
                    font-weight: 700 !important;
                    margin-bottom: 16px !important;
                }
                
                h2, .h2 {
                    font-size: 20px !important;
                    font-weight: 600 !important;
                    margin-bottom: 14px !important;
                }
                
                h3, .h3 {
                    font-size: 18px !important;
                    font-weight: 600 !important;
                    margin-bottom: 12px !important;
                }
                
                h4, .h4 {
                    font-size: 16px !important;
                    font-weight: 600 !important;
                    margin-bottom: 10px !important;
                }
                
                h5, .h5 {
                    font-size: 14px !important;
                    font-weight: 600 !important;
                    margin-bottom: 8px !important;
                }
                
                h6, .h6 {
                    font-size: 13px !important;
                    font-weight: 600 !important;
                }
                
                /* All text elements - Standard web size */
                p, div, span, li, td, th, a {
                    font-size: 14px !important;
                    line-height: 1.5 !important;
                }
                
                /* Labels - Standard */
                label, .form-label {
                    font-size: 14px !important;
                    font-weight: 600 !important;
                    margin-bottom: 6px !important;
                }
                
                /* Buttons - Standard */
                .btn, button {
                    font-size: 14px !important;
                    padding: 10px 20px !important;
                    font-weight: 600 !important;
                    min-height: 38px !important;
                }
                
                /* Cards and alerts - Standard */
                .card-body, .alert {
                    font-size: 14px !important;
                    line-height: 1.5 !important;
                    padding: 16px !important;
                }
                
                .card-header {
                    font-size: 16px !important;
                    font-weight: 600 !important;
                    padding: 12px !important;
                }
                
                /* Tables - Standard */
                table, th, td {
                    font-size: 13px !important;
                    padding: 10px !important;
                }
                
                /* SCROLLABLE GRAPH CONTAINERS */
                .js-plotly-plot {
                    overflow-x: auto !important;
                    overflow-y: auto !important;
                    max-height: 600px !important;
                }
                
                /* Make graph containers scrollable */
                div[id*="timeline"], 
                div[id*="bubble"],
                div[id*="graph"],
                div[id*="plot"] {
                    overflow-x: auto !important;
                    overflow-y: auto !important;
                    max-height: 600px !important;
                    border: 2px solid #3498DB !important;
                    border-radius: 8px !important;
                    padding: 10px !important;
                }
                
                /* Container for all graphs */
                .graph-container {
                    overflow-x: auto !important;
                    overflow-y: auto !important;
                    max-height: 700px !important;
                    margin-bottom: 20px !important;
                }
                
                /* Custom dropdown styling - Standard web size */
                .Select-control {
                    background-color: #2C3E50 !important;
                    border: 2px solid #3498DB !important;
                    min-height: 38px !important;
                    font-size: 14px !important;
                }
                
                .Select-value-label,
                .Select-placeholder {
                    color: #FFFFFF !important;
                    font-weight: 600 !important;
                    font-size: 14px !important;
                    line-height: 38px !important;
                }
                
                /* Dropdown menu items - Standard */
                .Select-menu-outer {
                    background-color: #34495E !important;
                    border: 2px solid #3498DB !important;
                    z-index: 9999 !important;
                    max-height: 350px !important;
                    overflow-y: auto !important;
                }
                
                .Select-option {
                    background-color: #34495E !important;
                    color: #FFFFFF !important;
                    font-weight: 500 !important;
                    font-size: 13px !important;
                    padding: 10px !important;
                    line-height: 1.4 !important;
                }
                
                .Select-option:hover {
                    background-color: #2980B9 !important;
                    color: #FFFFFF !important;
                }
                
                .Select-option.is-selected {
                    background-color: #3498DB !important;
                    color: #FFFFFF !important;
                    font-weight: 700 !important;
                }
                
                .Select-option.is-disabled {
                    background-color: #1C2833 !important;
                    color: #95A5A6 !important;
                    font-style: italic !important;
                    text-align: center !important;
                }
                
                /* For VirtualizedSelect and newer components */
                div[class*="dropdown"] input {
                    color: #FFFFFF !important;
                    font-size: 14px !important;
                }
                
                /* Input fields - Standard web size */
                input, textarea, select {
                    font-size: 14px !important;
                    padding: 8px 12px !important;
                    min-height: 36px !important;
                }
                
                /* Graph and plot text - Standard size */
                .plotly text {
                    font-size: 11px !important;
                    font-weight: 400 !important;
                }
                
                /* Plotly axis labels - Standard */
                .plotly .xtick text,
                .plotly .ytick text {
                    font-size: 10px !important;
                    font-weight: 400 !important;
                }
                
                /* Plotly legend - Standard */
                .plotly .legend text {
                    font-size: 10px !important;
                    font-weight: 400 !important;
                }
                
                /* Plotly titles - Standard prominent */
                .plotly .gtitle {
                    font-size: 13px !important;
                    font-weight: 600 !important;
                }
                
                /* Plotly hover labels - Standard */
                .plotly .hoverlayer .hovertext {
                    font-size: 11px !important;
                }
                
                /* Plotly annotations - Standard */
                .plotly .annotation-text {
                    font-size: 10px !important;
                }
                
                /* Bootstrap overrides - Standard web */
                .container, .container-fluid {
                    font-size: 14px !important;
                }
                
                .row {
                    font-size: 14px !important;
                }
                
                .col, .col-1, .col-2, .col-3, .col-4, .col-5, .col-6,
                .col-7, .col-8, .col-9, .col-10, .col-11, .col-12 {
                    font-size: 14px !important;
                }
                
                /* Dash specific overrides */
                #react-entry-point, #react-entry-point * {
                    font-size: inherit !important;
                }
                
                ._dash-loading {
                    font-size: 14px !important;
                }
                
                /* Alert and info boxes - Standard */
                .alert, .info-box {
                    font-size: 14px !important;
                    padding: 12px !important;
                }
                
                /* List items - Standard */
                ul, ol {
                    font-size: 14px !important;
                }
                
                ul li, ol li {
                    font-size: 14px !important;
                    margin-bottom: 6px !important;
                    line-height: 1.5 !important;
                }
                
                /* SVG text elements in Plotly - Standard */
                svg text {
                    font-size: 10px !important;
                }
                
                /* Force font size on all SVG elements */
                svg tspan {
                    font-size: 10px !important;
                }
                
                /* Plotly axis tick labels */
                g.xtick text, g.ytick text {
                    font-size: 10px !important;
                }
                
                /* Plotly axis titles */
                g.g-xtitle text, g.g-ytitle text {
                    font-size: 11px !important;
                }
                
                /* Scrollbar styling */
                ::-webkit-scrollbar {
                    width: 14px;
                    height: 14px;
                }
                
                ::-webkit-scrollbar-track {
                    background: #2C3E50;
                    border-radius: 8px;
                }
                
                ::-webkit-scrollbar-thumb {
                    background: #3498DB;
                    border-radius: 8px;
                }
                
                ::-webkit-scrollbar-thumb:hover {
                    background: #2980B9;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Build the official MITRE ATLAS matrix panel (full catalog)
    atlas_catalog = ttd._load_atlas_attack_chains()

    def _sort_tactic_id(tid: str) -> int:
        try:
            return int(str(tid).split('AML.TA', 1)[1])
        except Exception:
            return 10**9

    def _sort_tech_id(tid: str):
        # sort AML.T0000 < AML.T0000.000 < ...
        try:
            body = str(tid).split('AML.T', 1)[1]
            if '.' in body:
                a, b = body.split('.', 1)
                return (int(a), int(b), str(tid))
            return (int(body), -1, str(tid))
        except Exception:
            return (10**9, 10**9, str(tid))

    def _build_atlas_panel_children(atlas_data):
        tactics = (atlas_data or {}).get('tactics') or {}
        techniques = (atlas_data or {}).get('techniques') or {}

        # tactic_id -> [(tech_id, tech_name)]
        by = {tid: [] for tid in tactics.keys()}
        for tech_id, tech in techniques.items():
            for tactic_id in (tech.get('tactic_ids') or []):
                by.setdefault(tactic_id, []).append((tech_id, tech.get('name') or ''))

        children = []
        for tactic_id in sorted(tactics.keys(), key=_sort_tactic_id):
            t = tactics.get(tactic_id) or {}
            tname = t.get('name') or str(tactic_id)
            bucket = by.get(tactic_id, [])
            bucket.sort(key=lambda x: _sort_tech_id(x[0]))

            children.append(
                html.Div([
                    html.H6(
                        f"{tname} ({tactic_id})",
                        style={
                            'marginBottom': '4px',
                            'borderBottom': '2px solid #444',
                            'paddingBottom': '2px',
                            'fontSize': '12px',
                            'color': '#ddd',
                        },
                    ),
                    html.Ul(
                        [
                            html.Li(
                                f"{tid} {name}",
                                style={'fontSize': '10px', 'marginBottom': '1px'},
                            )
                            for tid, name in bucket
                        ],
                        style={'marginBottom': '6px', 'paddingLeft': '12px', 'listStyle': 'none'},
                    ),
                ])
            )

        catalog = (atlas_data or {}).get('catalog') or {}
        tcount = catalog.get('tactic_count') or len(tactics) or None
        techcount = catalog.get('technique_count') or len(techniques) or None

        children.extend([
            html.Hr(style={'margin': '8px 0', 'borderColor': '#333'}),
            html.Div([
                html.Span('Source: ', style={'fontSize': '9px', 'color': '#666'}),
                html.A(
                    'mitre-atlas/atlas-navigator-data',
                    href='https://github.com/mitre-atlas/atlas-navigator-data',
                    target='_blank',
                    style={'fontSize': '9px', 'color': '#17a2b8'},
                ),
                html.Br(),
                html.Span(
                    f"{techcount} Techniques | {tcount} Tactics | v5.1.1",
                    style={'fontSize': '9px', 'color': '#666'},
                ),
            ], style={'textAlign': 'center'}),
        ])

        return children

    atlas_panel_children = _build_atlas_panel_children(atlas_catalog)

    # App layout
    app.layout = html.Div([
        dbc.Container([
            html.H1("🛡️ NeurInSpectre TTD Dashboard", className="text-center mb-4"),
            html.Hr(),
            
            # Controls - ALWAYS VISIBLE
            dbc.Row([
                dbc.Col([
                    html.Label("Model:", className="form-label", style={'color': 'white'}),
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=all_model_options,
                        value=args.model,
                        clearable=False,
                        style={
                            'marginBottom': '10px',
                            'color': '#FFFFFF',  # White text
                            'backgroundColor': '#2C3E50',  # Dark blue-gray background
                            'borderColor': '#3498DB',  # Bright blue border
                            'fontWeight': '500'
                        },
                        className='custom-dropdown'
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("Data Source:", className="form-label", style={'color': 'white'}),
                    dcc.Dropdown(
                        id='data-source-dropdown',
                        options=[
                            {'label': 'Real Adversarial Data (on-disk)', 'value': 'real'},
                            {'label': 'Uploaded Attack File', 'value': 'uploaded'},
                            {'label': 'Live Model (train + capture)', 'value': 'live_model'},
                            {'label': 'Simulated Data', 'value': 'simulated'}
                        ],
                        value='real',  # Default: on-disk real data
                        clearable=False,
                        style={'marginBottom': '10px'}
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("Bulk Upload Real Attack Data:", className="form-label"),
                    dcc.Upload(
                        id='bulk-upload',
                        children=html.Div([
                            '📁 Drag & Drop or ',
                            html.A('Select Files', style={'color': '#007bff', 'textDecoration': 'underline'})
                        ]),
                        style={
                            'width': '100%', 'height': '40px', 'lineHeight': '40px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'borderRadius': '5px', 'textAlign': 'center',
                            'backgroundColor': '#f8f9fa', 'cursor': 'pointer'
                        },
                        multiple=True,
                        accept='.npy,.npz,.json'
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("Update Interval (seconds):", className="form-label"),
                    dcc.Slider(
                        id='update-interval-slider',
                        min=5,
                        max=60,
                        step=5,
                        value=10,
                        marks={i: str(i) for i in range(5, 65, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=6)
            ], className="mb-4"),
            
            # Auto-refresh interval for REAL-TIME SCROLLING
            dcc.Interval(
                id='interval-component',
                interval=2*1000,  # 2 seconds for fast scrolling real attack data
                n_intervals=0
            ),

            dcc.Store(id='upload-token', data=0),
            
            # Dashboard content
            html.Div(id='dashboard-content')
            
        ], fluid=True)
    ])
    
    # Callback for bulk upload of real attack data
    @app.callback(
        [Output('bulk-upload', 'children'),
         Output('data-source-dropdown', 'value'),
         Output('upload-token', 'data')],
        [Input('bulk-upload', 'contents')],
        [dash.dependencies.State('bulk-upload', 'filename')],
    )
    def handle_bulk_upload(contents, filenames):
        import time as _time
        import base64
        from pathlib import Path

        # No upload yet
        if contents is None or filenames is None:
            return (
                html.Div([
                    '📁 Drag & Drop or ',
                    html.A('Select Files', style={'color': '#007bff', 'textDecoration': 'underline'})
                ]),
                dash.no_update,
                dash.no_update,
            )

        upload_dir = Path('_cli_runs/uploads')
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for content, filename in zip(contents, filenames):
            if not content or not filename:
                continue
            # Decode and save uploaded files
            try:
                _, content_string = content.split(',', 1)
            except ValueError:
                continue
            decoded = base64.b64decode(content_string)
            safe_name = Path(str(filename)).name
            out_path = upload_dir / safe_name
            out_path.write_bytes(decoded)
            saved_paths.append(str(out_path))
            print(f"📁 Bulk uploaded: {out_path}")

        if not saved_paths:
            return (
                html.Div(['❌ Upload failed: no files decoded'], style={'color': 'red', 'fontSize': '12px'}),
                dash.no_update,
                dash.no_update,
            )

        # Apply uploaded dataset immediately (do not rely on filename patterns)
        meta = ttd.apply_uploaded_dataset(saved_paths, source_label='uploaded')
        print(f"🔄 Activated uploaded dataset: {meta.get('gradient_shape')} from {len(meta.get('gradient_files') or [])} gradient files")

        # Force the dashboard to reflect the new source + refresh
        updated_bits = []
        if meta.get('gradient_updated'):
            updated_bits.append(f"gradients={meta.get('gradient_shape')}")
        else:
            updated_bits.append("gradients=unchanged" if meta.get('gradient_shape') else "gradients=missing")
        if meta.get('attention_updated'):
            updated_bits.append(f"attention={meta.get('attention_shape')}")
        if meta.get('epsilon_updated'):
            updated_bits.append(f"ε=len({meta.get('epsilon_len')})")

        summary = html.Div([
            f"✅ Uploaded {len(saved_paths)} files",
            html.Br(),
            "📁 Active source: Uploaded File",
            html.Br(),
            f"Applied: {', '.join(updated_bits)}",
        ], style={'color': 'green', 'fontSize': '12px'})

        return summary, 'uploaded', _time.time()

    # Initialize live model monitor for ACTUAL model execution
    live_monitor = None
    try:
        from neurinspectre.cli.real_time_model_monitor import LiveModelMonitor
        live_monitor = LiveModelMonitor(model_name=args.model)
        print(f"✅ Live model monitor initialized for {args.model}")
        print(f"   Device: {live_monitor.device}")
        print(f"   Ready to run model and capture gradients")
    except Exception as e:
        print(f"⚠️  Live monitor not available: {e}")
        print("   Using saved gradient analysis mode")
        live_monitor = None

    # Enhanced callback for updating dashboard with REAL LIVE MODEL DATA
    @app.callback(
        Output('dashboard-content', 'children'),
        [Input('interval-component', 'n_intervals'),
         Input('model-dropdown', 'value'),
         Input('data-source-dropdown', 'value'),
         Input('upload-token', 'data')],
        prevent_initial_call=False,
        suppress_callback_exceptions=True  # Critical for Apple Silicon
    )
    def update_dashboard_with_live_model(n, selected_model, data_source, upload_token):
        """Dashboard callback - WITH LIVE MODEL EXECUTION"""
        
        # Validate inputs to prevent crashes
        if n is None:
            n = 0
        if selected_model is None:
            selected_model = args.model or 'gpt2'
        if data_source is None:
            data_source = 'real'


        # Keep TTDDashboard source label in sync with UI selection
        try:
            if data_source == 'uploaded':
                ttd.active_data_source = 'uploaded'
            elif data_source == 'real':
                # If we were previously in uploaded mode, reload the original on-disk inputs
                if getattr(ttd, 'active_data_source', 'real') != 'real':
                    ttd.active_data_source = 'real'
                    try:
                        ttd.load_real_data(
                            gradient_file=gradient_file,
                            attention_file=attention_file,
                            batch_dir=getattr(args, 'batch_dir', None),
                            privacy_file=getattr(args, 'privacy_file', None),
                        )
                    except Exception as _e:
                        print(f'⚠️ Reload real data failed: {_e}')
                else:
                    ttd.active_data_source = 'real'
            elif data_source == 'live_model':
                ttd.active_data_source = 'live_model'
            elif data_source == 'simulated':
                ttd.active_data_source = 'simulated'
        except Exception as _e:
            print(f'⚠️ data source sync failed: {_e}')
        
        # Detect model change and update model metadata (config-driven when available)
        model_changed = False
        try:
            if ttd.model_name != selected_model:
                print(f"\n🔄 MODEL CHANGE: {ttd.model_name} → {selected_model}")
                ttd.model_name = selected_model

                # Start with static fallback (fast, no network)
                ttd.model_layers = ttd._get_model_layers(selected_model)

                # Try to load cached HF config (local only) so the selected model is actually reflected
                try:
                    from transformers import AutoConfig
                    cfg = AutoConfig.from_pretrained(selected_model, local_files_only=True)
                    actual = {
                        'num_layers': getattr(cfg, 'num_hidden_layers', None)
                                    or getattr(cfg, 'num_layers', None)
                                    or getattr(cfg, 'n_layer', None)
                                    or ttd.model_layers.get('num_layers', 12),
                        'attention_heads': getattr(cfg, 'num_attention_heads', None)
                                          or getattr(cfg, 'num_heads', None)
                                          or getattr(cfg, 'n_head', None)
                                          or ttd.model_layers.get('attention_heads', 12),
                        'hidden_size': getattr(cfg, 'hidden_size', None)
                                      or getattr(cfg, 'd_model', None)
                                      or getattr(cfg, 'n_embd', None)
                                      or ttd.model_layers.get('hidden_size', 768),
                        'model_type': getattr(cfg, 'model_type', None) or ttd.model_layers.get('model_type', 'unknown'),
                        'is_encoder_decoder': bool(getattr(cfg, 'is_encoder_decoder', False)),
                    }
                    ttd.model_layers.update(actual)
                    print(f"✅ Using cached model config: layers={actual['num_layers']} heads={actual['attention_heads']} hidden={actual['hidden_size']} type={actual['model_type']}")

                    # Non-fatal sanity check: attention tensor shape vs selected model
                    try:
                        import numpy as _np
                        attn = getattr(ttd, 'real_attention_data', None)
                        if isinstance(attn, _np.ndarray) and attn.ndim == 4:
                            L, H = int(attn.shape[0]), int(attn.shape[1])
                            if int(actual['num_layers']) != L or int(actual['attention_heads']) != H:
                                setattr(ttd, '_last_good_error',
                                        f"Attention artifact shape is L={L}, H={H} but selected model config is "
                                        f"L={actual['num_layers']}, H={actual['attention_heads']}. "
                                        "If the attention file comes from a different model, select that model for correct interpretation.")
                    except Exception:
                        pass

                except Exception:
                    # If transformers isn't installed or config not cached, keep static fallback.
                    pass

                model_changed = True
        except Exception as model_error:
            print(f"⚠️ Model update error (handled): {model_error}")
            selected_model = ttd.model_name or 'gpt2'
            model_changed = False

        # If Live Model mode is selected, ensure we actually start/refresh live monitoring for this model
        if data_source == 'live_model' and live_monitor is not None:
            if getattr(ttd, '_live_requested_model', None) != selected_model:
                setattr(ttd, '_live_requested_model', selected_model)
                model_changed = True
        
        # ENABLE LIVE MODEL EXECUTION when dropdown changes
        if model_changed and live_monitor and data_source == 'live_model':
            import threading
            import time
            
            def load_and_run_model():
                """Load model and run brief training to capture real gradients"""
                try:
                    print(f"\n🚀 LOADING MODEL LIVE: {selected_model}")
                    
                    # Switch or initialize model
                    if live_monitor.model is None:
                        live_monitor.model_name = selected_model
                        success = live_monitor.initialize_model()
                    else:
                        success = live_monitor.switch_model(selected_model)
                    
                    if success:
                        print(f"✅ Model loaded: {selected_model}")
                        
                        # Run brief training to capture real gradients
                        print(f"🏃 Running brief training on {selected_model} to capture gradients...")
                        live_monitor.start_live_training(num_epochs=1)
                        
                        # Wait for some gradient data
                        time.sleep(3)
                        
                        # Get real-time data and update TTD
                        if live_monitor.gradient_norms:
                            print(f"✅ Captured {len(live_monitor.gradient_norms)} real gradients from {selected_model}")
                            # Update TTD with live gradient data
                            import numpy as np
                            ttd.real_gradient_data = np.array(live_monitor.gradient_norms).reshape(-1, 1)
                        
                        # Update config
                        if hasattr(live_monitor, 'model_config'):
                            ttd.model_layers.update(live_monitor.model_config)
                    else:
                        print(f"❌ Failed to load {selected_model}")
                        
                except Exception as e:
                    print(f"⚠️ Model execution error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Start model loading in background
            model_thread = threading.Thread(target=load_and_run_model, daemon=True)
            model_thread.start()
            print(f"🔄 Started background execution for {selected_model}")
        elif model_changed:
            if data_source != 'live_model' and live_monitor is not None:
                print(f"ℹ️ Model selection updated (labels/config). Set Data Source to 'Live Model' to run {selected_model}.")
            else:
                print(f"📋 Live monitor not available - using saved gradients for {selected_model}")
        # Generate dashboard content - WITH LIVE MODEL EXECUTION
        try:
            # ENABLE live monitor if available
            live_monitor_active = live_monitor is not None
            if live_monitor and live_monitor.is_training:
                # Get REAL-TIME data from live model training
                print(f"📡 USING LIVE MODEL DATA from {live_monitor.model_name}")
                live_data = live_monitor.get_real_time_data()
                
                # Convert live model data to TTD format
                data = {
                    'steps': list(range(max(1, live_data['current_step'] - 99), live_data['current_step'] + 1)),
                    'gradient_norms': live_data['gradient_norms'][-100:] if live_data['gradient_norms'] else [1.0],
                    'privacy_budget': [live_data['privacy_budget']] * min(100, live_data['current_step']),
                    'vulnerability_scores': [0.5] * min(100, live_data['current_step']),
                    'layer_vulnerabilities': {f"Layer_{i+1}": 0.5 for i in range(6)},
                    'membership_inference_success': [0.6] * min(100, live_data['current_step']),
                    'reconstruction_success': [0.4] * min(100, live_data['current_step']),
                    'data_source': 'live_model',
                    'model_name': live_data['model_name'],
                    'live_training': True,
                    'current_step': live_data['current_step'],
                    'current_epoch': live_data['current_epoch'],
                    'is_training': live_data['is_training'],
                    'device': live_data['device'],
                    'security_status': live_data['security_status'],
                    'attack_events': live_data['attack_events']
                }
                
                use_real_data = True
                print(f"📡 LIVE MODEL DATA: Step {live_data['current_step']}, Device: {live_data['device']}, Status: {live_data['security_status']}")
                
            else:

                # Use static real-data analysis (NO SIMULATION)
                try:
                    if data_source == 'simulated':
                        # Simulation is disabled by default; only run if explicitly allowed
                        if getattr(ttd, 'allow_simulated', False):
                            data = ttd.generate_gradient_leakage_data(False)
                        else:
                            raise ValueError('Simulated data mode is disabled. Upload a .npy/.npz dataset or use on-disk real gradients.')
                    else:
                        data = ttd.generate_gradient_leakage_data(True)
                    use_real_data = True
                    setattr(ttd, '_last_good_data', data)
                    setattr(ttd, '_last_good_error', None)
                except Exception as e:
                    # Keep the full dashboard visible even if the current source lacks gradients.
                    # This is especially important for incident response workflows.
                    warn_msg = str(e)
                    setattr(ttd, '_last_good_error', warn_msg)
                    data = getattr(ttd, '_last_good_data', None)
                    if data is None:
                        import time as _time
                        # Minimal placeholder dataset
                        nl = int(getattr(ttd, 'model_layers', {}).get('num_layers', 6))
                        data = {
                            'steps': [0],
                            'gradient_norms': [],
                            'privacy_budget': [],
                            'vulnerability_scores': [],
                            'layer_vulnerabilities': {f'Layer_{i+1}': 0.0 for i in range(nl)},
                            'membership_inference_success': [],
                            'reconstruction_success': [],
                            'data_source': getattr(ttd, 'active_data_source', data_source),
                            'model_name': selected_model,
                            'rolling_timestamp': _time.time(),
                        }
                    use_real_data = bool(getattr(ttd, 'real_gradient_data', None) is not None)

            
            # Live model status indicator
            if data.get('data_source') == 'live_model':
                real_data_status = f"🔴 LIVE MODEL TRAINING - Step {data['current_step']}, Epoch {data['current_epoch']}"
                data_size_mb = 0.0  # Live data size varies
                live_status = f"Device: {data['device']}, Security: {data['security_status']}"
            else:
                if use_real_data and (ttd.real_gradient_data is not None):
                    real_data_status = "✅ REAL ADVERSARIAL DATA LOADED"
                elif getattr(ttd, 'allow_simulated', False) and str(data.get('data_source')) == 'simulated':
                    real_data_status = "⚠️ SIMULATED DATA (demo)"
                else:
                    real_data_status = "⚠️ NO REAL DATA LOADED"
                data_size_mb = ttd.real_gradient_data.nbytes / (1024*1024) if ttd.real_gradient_data is not None else 0.0
                live_status = "Static data analysis"
            # Calculate threat metrics safely (NO simulation defaults)
            import math
            current_privacy = data['privacy_budget'][-1] if data.get('privacy_budget') else float('nan')
            current_gradient = data['gradient_norms'][-1] if data.get('gradient_norms') else 0.0
            current_mi = data['membership_inference_success'][-1] if data.get('membership_inference_success') else 0.0

            # Normalize components to 0..1 where possible
            privacy_component = 0.0
            if isinstance(current_privacy, (int, float)) and math.isfinite(current_privacy) and ttd.PRIVACY_BUDGET_LIMIT > 0:
                privacy_component = min(current_privacy / ttd.PRIVACY_BUDGET_LIMIT, 1.0)

            gradient_component = 0.0
            if isinstance(current_gradient, (int, float)) and math.isfinite(current_gradient) and ttd.CRITICAL_GRADIENT_NORM > 0:
                gradient_component = min(current_gradient / ttd.CRITICAL_GRADIENT_NORM, 1.0)

            mi_component = 0.0
            if isinstance(current_mi, (int, float)) and math.isfinite(current_mi):
                mi_component = min(max(current_mi, 0.0), 1.0)

            # Composite threat score (bounded)
            threat_score = min((privacy_component + gradient_component + mi_component) / 3.0, 1.0)

            
            threat_level = "CRITICAL" if threat_score > 0.8 else "HIGH" if threat_score > 0.6 else "MEDIUM"
            threat_color = "danger" if threat_score > 0.8 else "warning" if threat_score > 0.6 else "info"
            
            # Generate dynamic ATLAS advisories based on current graph data
            red_team_advisory, blue_team_advisory = ttd._generate_dynamic_atlas_advisories(
                data, threat_score, current_gradient, current_privacy, current_mi
            )
            
            # Format dynamic red team actions
            red_team_actions = [html.Li(action) for action in red_team_advisory["actions"]]
            red_team_actions.extend([
                html.Hr(),
                html.H6("🎯 Technical Intelligence:", style={'color': '#ff6b6b', 'margin-top': '10px'}),
                html.P(f"Pattern Detected: {red_team_advisory['pattern_detected']}", style={'margin-bottom': '5px'}),
                html.P(f"ATLAS Technique: {red_team_advisory['atlas_technique']}", style={'margin-bottom': '5px'}),
                html.P(f"Success Probability: {red_team_advisory['success_probability']}", style={'margin-bottom': '5px'}),
                html.P(f"CVE References: {', '.join(red_team_advisory['cve_refs']) if red_team_advisory['cve_refs'] else 'None'}", style={'margin-bottom': '5px'}),
                html.P(f"Threat Level: {red_team_advisory['threat_level']}", style={'margin-bottom': '0px'})
            ])
            
            # Format dynamic blue team actions
            blue_team_actions = [html.Li(action) for action in blue_team_advisory["actions"]]
            blue_team_actions.extend([
                html.Hr(),
                html.H6("🛡️ Implementation Priority:", style={'color': '#4ecdc4', 'margin-top': '10px'}),
                html.P(f"Priority: {blue_team_advisory['priority']}", style={'margin-bottom': '5px'}),
                html.P(f"Response Time: {blue_team_advisory['response_time']}", style={'margin-bottom': '5px'}),
                html.P(f"Monitoring: {blue_team_advisory['monitoring_frequency']}", style={'margin-bottom': '5px'}),
                html.P(f"Escalation Required: {'YES' if blue_team_advisory['escalation_required'] else 'NO'}", style={'margin-bottom': '0px'})
            ])
            
            # Create WORKING, ERROR-FREE visualizations
            import plotly.graph_objects as go
            
            # 1. RICH SCROLLING REAL ADVERSARIAL ATTACK TIMELINE
            # Use the enhanced attack timeline method for consistency
            timeline_fig = ttd.create_attack_timeline(data)
            
            # 2. REAL-TIME THREAT GAUGE
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=threat_score * 100,
                title={'text': f"🎯 Threat Level: {threat_level}", 'font': {'size': 16}},
                delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickcolor': "white"},
                    'bar': {'color': "red" if threat_score > 0.8 else "orange" if threat_score > 0.6 else "green"},
                    'steps': [
                        {'range': [0, 50], 'color': 'lightgray'},
                        {'range': [50, 80], 'color': 'yellow'},
                        {'range': [80, 100], 'color': 'red'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            gauge_fig.update_layout(
                height=300,
                template='plotly_dark',
                font={'color': "white", 'size': 12}
            )
            
            # 3. GRADIENT LEAKAGE VISUALIZATION (use enhanced view whenever data exists)
            if data.get('gradient_norms') and len(data['gradient_norms']) > 0:
                # Create rolling window through available gradient norms (works for any length/source)
                total_points = len(data['gradient_norms'])
                window_size = min(50, total_points)
                if total_points <= window_size:
                    start_idx = 0
                else:
                    start_idx = (n * 2) % (total_points - window_size)
                end_idx = start_idx + window_size
                
                gradient_window = data['gradient_norms'][start_idx:end_idx]
                pb = data.get('privacy_budget') or []
                if len(pb) >= end_idx:
                    privacy_window = pb[start_idx:end_idx]
                else:
                    privacy_window = [float('nan')] * window_size
                
                # Extract timestamps/steps for the rolling window
                if len(data.get('steps') or []) >= end_idx:
                    time_window = (data.get('steps') or [])[start_idx:end_idx]
                else:
                    time_window = list(range(start_idx, end_idx))
                
                # Prefer real timestamps; otherwise render a GPU-like timeline for readability
                from datetime import datetime
                import time as _time
                time_labels = []
                numeric = all(isinstance(ts, (int, float)) for ts in time_window) and len(time_window) > 0
                max_ts = max(time_window) if numeric else None
                if numeric and max_ts is not None and float(max_ts) > 1e9:
                    for ts in time_window:
                        tsv = float(ts)
                        if tsv > 1e12:  # ms timestamps
                            tsv = tsv / 1000.0
                        try:
                            time_labels.append(datetime.fromtimestamp(tsv).strftime('%H:%M:%S'))
                        except Exception:
                            time_labels.append(str(int(tsv)))
                else:
                    base = float(data.get('rolling_timestamp') or _time.time())
                    step_sec = 2.0
                    npts = len(time_window)
                    for i in range(npts):
                        tsv = base - (npts - 1 - i) * step_sec
                        time_labels.append(datetime.fromtimestamp(tsv).strftime('%H:%M:%S'))
                
                grad_fig = go.Figure()
                
                # Advanced threat analysis with 2024-2025 research-based categorization
                threat_info = []
                mitre_techniques = []
                red_team_actions = []
                blue_team_actions = []
                
                critical_threshold = ttd.CRITICAL_GRADIENT_NORM
                privacy_limit = ttd.PRIVACY_BUDGET_LIMIT
                
                # Convert to numpy array for vectorized operations
                gradient_array = np.array(gradient_window)
                
                # Calculate aggregate window statistics for strategic guidance
                avg_grad = np.mean(gradient_array)
                max_grad = np.max(gradient_array)
                std_grad = np.std(gradient_array)
                sparsity = np.sum(gradient_array < 0.01) / len(gradient_array) if len(gradient_array) > 0 else 0
                high_grad_count = int(np.sum(gradient_array > 3.0))
                
                # Generate strategic guidance based on AGGREGATE WINDOW ANALYSIS
                # This provides actionable intelligence instead of per-sample noise
                
                critical_count = int(np.sum(gradient_array > critical_threshold))
                high_count = int(np.sum((gradient_array > 3.0) & (gradient_array <= critical_threshold)))
                medium_count = int(np.sum((gradient_array > 1.0) & (gradient_array <= 3.0)))
                low_count = int(np.sum(gradient_array <= 1.0))
                
                # Build per-point threat info and hover guidance (for tooltips)
                hover_red_actions = []
                hover_blue_actions = []
                
                for grad_norm in gradient_window:
                    if grad_norm > critical_threshold:
                        threat_info.append("🔴 CRITICAL")
                        mitre_techniques.append("AML.T0043.001")
                        hover_red_actions.append(
                            f"LEAKAGE RISK: ∇={grad_norm:.3f} high; validate reconstruction susceptibility on your benchmark"
                        )
                        hover_blue_actions.append(
                            f"CLIP: max_norm=1.0 | DP σ={max(2.0, grad_norm/2):.1f}"
                        )
                    elif grad_norm > 3.0:
                        threat_info.append("🟠 HIGH")
                        mitre_techniques.append("AML.T0048")
                        hover_red_actions.append(
                            f"MEMBERSHIP RISK: ∇={grad_norm:.3f} elevated; validate with calibrated membership inference evaluation"
                        )
                        hover_blue_actions.append(
                            f"NOISE: σ={grad_norm/3:.1f} | Rate limit ≤100/hr"
                        )
                    elif grad_norm > 1.0:
                        threat_info.append("🟡 MEDIUM")
                        mitre_techniques.append("AML.T0054")
                        hover_red_actions.append(
                            f"INFO LEAKAGE: ∇={grad_norm:.3f} moderate; may support property inference depending on context"
                        )
                        hover_blue_actions.append(
                            f"QUANTIZE: 8-bit | DP ε≤1.0"
                        )
                    else:
                        threat_info.append("🟢 LOW")
                        mitre_techniques.append("Baseline")
                        hover_red_actions.append(
                            f"LOW SIGNAL: ∇={grad_norm:.3f} below alert threshold"
                        )
                        hover_blue_actions.append(
                            f"OK: ∇={grad_norm:.3f} within baseline range"
                        )
                
                # STRATEGIC GUIDANCE: Based on AGGREGATE window analysis (for action cards)
                # This creates concise, actionable intelligence
                
                # Add critical threats if any exist
                if critical_count > 0:
                    red_team_actions.append(
                        f"🎯 CRITICAL ({critical_count} samples): ∇max={max_grad:.3f} → "
                        f"High leakage surface. Validate reconstruction susceptibility with your approved benchmark and record quality metrics."
                    )
                    blue_team_actions.append(
                        f"🚨 URGENT ({critical_count} high grads): Clip to max_norm=1.0 | "
                        f"DP noise σ={max(2.0, max_grad/2):.1f} | Gradient compression required"
                    )
                
                # Add high threats if any exist
                if high_count > 0:
                    red_team_actions.append(
                        f"🔍 HIGH ({high_count} samples): ∇avg={avg_grad:.3f} → "
                        f"Membership-inference risk proxy elevated. Validate with a calibrated evaluation (holdout + shadow model) before concluding."
                    )
                    blue_team_actions.append(
                        f"⚠️ DEFENSE ({high_count} targets): Noise σ={avg_grad/3:.1f} | "
                        f"Query limit ≤100/hr | Gradient masking on high-variance features"
                    )
                
                # Add medium threats if any exist
                if medium_count > 0:
                    red_team_actions.append(
                        f"📊 MEDIUM ({medium_count} samples): ∇std={std_grad:.3f} → "
                        f"Moderate leakage proxy. Consider property-inference risk assessment on your heldout evaluation set."
                    )
                    blue_team_actions.append(
                        f"🛡️ MITIGATE ({medium_count} features): Gradient quantization (8-bit) | "
                        f"Output perturbation | DP ε≤1.0"
                    )
                
                # Only show low status if it's the ONLY category
                if critical_count == 0 and high_count == 0 and medium_count == 0:
                    red_team_actions.append(
                        f"⏸️ WINDOW INSUFFICIENT: All {low_count} gradients below threshold "
                        f"(∇max={max_grad:.3f} < 1.0). Aggregate across batches or wait for training phase."
                    )
                    blue_team_actions.append(
                        f"✅ LOW-RISK WINDOW: {low_count} gradients within baseline range (∇max={max_grad:.3f}). "
                        f"Continue monitoring; if DP/clipping is enabled, confirm guardrails remain enforced."
                    )
                
                # Real gradient norms with ENHANCED HOVER showing security intelligence
                grad_fig.add_trace(go.Scatter(
                    x=time_labels,
                    y=gradient_window,
                    mode='lines+markers',
                    name='Gradient L2 Norm (Real Data)',
                    line=dict(color='cyan', width=3),
                    marker=dict(size=6, color=[
                        'red' if g > critical_threshold else 
                        'orange' if g > 3.0 else 
                        'yellow' if g > 1.0 else 
                        'green' for g in gradient_window
                    ]),
                    customdata=list(zip(threat_info, mitre_techniques, hover_red_actions, hover_blue_actions)),
                    hovertemplate=
                        '<b>🎯 Gradient Security Analysis</b><br>' +
                        '━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>' +
                        '<b>🕐 Attack Time:</b> %{x}<br>' +
                        '<b>📊 Gradient Norm:</b> %{y:.4f}<br>' +
                        '<b>⚠️ Threat Level:</b> %{customdata[0]}<br>' +
                        '<b>🛡️ MITRE ATLAS:</b> %{customdata[1]}<br>' +
                        '━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>' +
                        '<b>🔴 RED TEAM:</b><br>' +
                        '   %{customdata[2]}<br>' +
                        '<b>🔵 BLUE TEAM:</b><br>' +
                        '   %{customdata[3]}<br>' +
                        '<extra></extra>',
                    hoverlabel=dict(
                        bgcolor="rgba(0,0,0,0.95)",
                        font_size=13,
                        font_family="monospace",
                        bordercolor="cyan"
                    )
                ))
                
                # Elegant Privacy Budget Line - Dynamic Scrolling with Time Labels
                grad_fig.add_trace(go.Scatter(
                    x=time_labels,  # Use actual time labels, not indices
                    y=privacy_window,
                    mode='lines+markers',
                    name='Privacy Budget (ε-DP)',
                    line=dict(
                        color='rgba(255,165,0,0.9)',  # Elegant orange
                        width=3,  # Thicker for visibility
                        dash='dash',
                        shape='spline'  # Smooth elegant curves
                    ),
                    marker=dict(
                        size=6,
                        color=privacy_window,
                        colorscale=[[0, '#44ff44'], [0.6, '#ffaa00'], [1, '#ff4444']],  # Green→Orange→Red
                        showscale=False,
                        line=dict(color='white', width=1)
                    ),
                    yaxis='y2',
                    hovertemplate='<b>🕐 Time:</b> %{x}<br><b>🔒 Privacy Budget:</b> %{y:.2f}ε<br>' +
                                  '<b>Status:</b> %{customdata}<br><extra></extra>',
                    customdata=[('CRITICAL' if (np.isfinite(p) and p > privacy_limit) else ('SAFE' if np.isfinite(p) else 'N/A')) for p in privacy_window]
                ))
                
                grad_fig.update_layout(
                    title=dict(
                        text=f"📊 Real Gradient Leakage Analysis - Rolling Window ({data_size_mb:.1f}MB dataset)<br>" +
                             f"<sub>🛡️ MITRE ATLAS Mapping | Model: {selected_model} | 🍎 Apple Silicon MPS</sub>",
                        font=dict(size=16)
                    ),
                    xaxis_title="Attack Time (GPU Timeline)",
                    yaxis_title="Gradient Norm",
                    yaxis2=dict(
                        title=dict(
                            text="<b>Privacy Budget (ε-DP)</b>",
                            font=dict(size=14, color='orange')
                        ),
                        overlaying='y',
                        side='right',
                        showgrid=True,
                        gridcolor='rgba(255,165,0,0.2)',
                        gridwidth=1,
                        tickfont=dict(size=12, color='orange'),
                        range=[0, max(6, max(privacy_window) * 1.2)]  # Dynamic range
                    ),
                    height=400,
                    shapes=[
                        # Critical threshold line at ε=3.0
                        dict(
                            type='line',
                            yref='y2',
                            y0=privacy_limit, y1=privacy_limit,
                            xref='paper',
                            x0=0, x1=1,
                            line=dict(color='rgba(255,0,0,0.8)', width=3, dash='dot'),
                        ),
                        # Danger zone (above ε=3.0)
                        dict(
                            type='rect',
                            yref='y2',
                            y0=privacy_limit, y1=max(6, max(privacy_window) * 1.2) if privacy_window else 6,
                            xref='paper',
                            x0=0, x1=1,
                            fillcolor='rgba(255,0,0,0.12)',
                            line=dict(width=0),
                            layer='below'
                        ),
                        # Safe zone (below ε=1.0)
                        dict(
                            type='rect',
                            yref='y2',
                            y0=0, y1=1.0,
                            xref='paper',
                            x0=0, x1=1,
                            fillcolor='rgba(0,255,0,0.08)',
                            line=dict(width=0),
                            layer='below'
                        )
                    ],
                    template='plotly_dark',
                    xaxis=dict(
                        tickangle=45,
                        nticks=10
                    ),
                    hoverlabel=dict(
                        bgcolor="rgba(0,0,0,0.95)",
                        font_size=13,
                        font_family="monospace",
                        bordercolor="cyan"
                    ),
                    annotations=[
                        # RED TEAM - Title
                        dict(text='🔴 RED TEAM', xref="paper", yref="paper", x=0.01, y=0.35,
                            xanchor='left', yanchor='bottom', showarrow=False,
                            bgcolor="#330000", bordercolor="lime", borderwidth=2, borderpad=6,
                            font=dict(size=14, color='lime', family='Arial Black')),
                        # RED - Critical
                        dict(text='CRITICAL (>5.0): Exploit', xref="paper", yref="paper", x=0.01, y=0.28,
                            xanchor='left', yanchor='bottom', showarrow=False,
                            bgcolor="#330000", borderpad=4,
                            font=dict(size=11, color='yellow', family='Arial')),
                        # RED - High
                        dict(text='HIGH (>3.0): Target', xref="paper", yref="paper", x=0.01, y=0.21,
                            xanchor='left', yanchor='bottom', showarrow=False,
                            bgcolor="#330000", borderpad=4,
                            font=dict(size=11, color='yellow', family='Arial')),
                        # RED - Medium
                        dict(text='MEDIUM (>1.0): Probe', xref="paper", yref="paper", x=0.01, y=0.14,
                            xanchor='left', yanchor='bottom', showarrow=False,
                            bgcolor="#330000", borderpad=4,
                            font=dict(size=11, color='yellow', family='Arial')),
                        # BLUE TEAM - Title
                        dict(text='🔵 BLUE TEAM', xref="paper", yref="paper", x=0.99, y=0.35,
                            xanchor='right', yanchor='bottom', showarrow=False,
                            bgcolor="#000033", bordercolor="cyan", borderwidth=2, borderpad=6,
                            font=dict(size=14, color='cyan', family='Arial Black')),
                        # BLUE - Critical
                        dict(text='CRITICAL: Clip gradients', xref="paper", yref="paper", x=0.99, y=0.28,
                            xanchor='right', yanchor='bottom', showarrow=False,
                            bgcolor="#000033", borderpad=4,
                            font=dict(size=16, color='#ffff00', family='Arial Black')),
                        # BLUE - High
                        dict(text='HIGH: Increase DP noise', xref="paper", yref="paper", x=0.99, y=0.21,
                            xanchor='right', yanchor='bottom', showarrow=False,
                            bgcolor="#000033", borderpad=4,
                            font=dict(size=16, color='#ffff00', family='Arial Black')),
                        # BLUE - Medium
                        dict(text='MEDIUM: Monitor & mask', xref="paper", yref="paper", x=0.99, y=0.14,
                            xanchor='right', yanchor='bottom', showarrow=False,
                            bgcolor="#000033", borderpad=4,
                            font=dict(size=16, color='#ffff00', family='Arial Black')),
                        # Privacy Budget Threshold Annotation
                        dict(
                            text=f'⚠️ ε={privacy_limit:.1f} CRITICAL',
                            xref="paper", yref="y2",
                            x=0.5, y=privacy_limit,
                            xanchor='center', yanchor='bottom',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor='red',
                            ax=0, ay=-40,
                            bgcolor="rgba(0,0,0,0.95)",
                            bordercolor="#ff4444",
                            borderwidth=2,
                            borderpad=8,
                            font=dict(size=16, color='#ffff00', family='Arial Black')
                        )
                    ]
                )
            else:
                # ENHANCED gradient visualization - check if real or simulated data
                data_type = "Real Adversarial" if (use_real_data and data.get('data_source') != 'simulated') else "Simulated"
                print(f"🔍 DEBUG: Using {data_type.lower()} data path - adding timestamp extraction")
                
                # Extract or generate timestamps
                if len(data['steps']) >= len(data['gradient_norms']):
                    time_window = data['steps'][:len(data['gradient_norms'])]
                else:
                    # Generate timestamps
                    import time
                    current_time = time.time()
                    time_window = [current_time - (len(data['gradient_norms']) - i) * 2.0 for i in range(len(data['gradient_norms']))]
                
                # Format timestamps for display - APPLE SILICON MPS COMPATIBLE
                from datetime import datetime
                time_labels = []
                for ts in time_window:
                    try:
                        if isinstance(ts, (int, float)) and ts > 0:
                            if ts > 1e10:  # Handle millisecond timestamps
                                ts = ts / 1000.0
                            dt_obj = datetime.fromtimestamp(float(ts))
                            time_labels.append(dt_obj.strftime("%H:%M:%S"))
                        else:
                            time_labels.append(f"T{len(time_labels):02d}")
                    except (ValueError, OSError, OverflowError):
                        time_labels.append(f"T{len(time_labels):02d}")
                
                grad_fig = go.Figure()
                grad_fig.add_trace(go.Scatter(
                    x=time_labels,
                    y=data['gradient_norms'],
                    mode='lines+markers',
                    name=f'Gradient Norms ({data_type})',
                    line=dict(color='cyan', width=2),
                    hovertemplate='<b>🕐 Attack Time:</b> %{x}<br><b>📈 Gradient Norm:</b> %{y:.4f}<br><b>📍 Data Point:</b> %{pointNumber}<br><extra></extra>',
                    hoverlabel=dict(bgcolor="rgba(0,255,255,0.9)", font_size=16, font_color="black", bordercolor="cyan")
                ))
                grad_fig.update_layout(
                    title=f"📊 Gradient Analysis ({data_type} Data with GPU Timeline)",
                    xaxis_title="Attack Time (GPU Timeline)",
                    yaxis_title="Gradient Norm",
                    height=350,
                    template='plotly_dark',
                    hoverlabel=dict(
                        bgcolor="rgba(0,0,0,0.95)",
                        bordercolor="cyan",
                        font_size=16,
                        font_family="Arial",
                        font_color="white",
                        namelength=-1,
                        align="left"
                    ),
                    hovermode='x unified',
                    xaxis=dict(
                        tickangle=45,
                        nticks=8,
                        tickfont=dict(size=12),
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.3)'
                    ),
                    annotations=[
                        # Privacy Budget in Center of Graph - FIXED TEXT COLOR
                        dict(
                            text=f'Privacy Budget (ε-DP)<br>{(f"{current_privacy:.2f}ε" if np.isfinite(current_privacy) else "N/A")}<br>{("⚠️ EXHAUSTED" if (np.isfinite(current_privacy) and current_privacy > privacy_limit) else ("✅ SAFE" if np.isfinite(current_privacy) else "ℹ️ ε unavailable"))}',
                            xref="paper", yref="paper",
                            x=0.5, y=0.5,
                            xanchor='center', yanchor='middle',
                            showarrow=False,
                            bgcolor="rgba(0,0,0,0.95)",
                            bordercolor=("#ff4444" if (np.isfinite(current_privacy) and current_privacy > privacy_limit) else ("#00ff00" if np.isfinite(current_privacy) else "#ffaa00")), 
                            borderwidth=4,
                            borderpad=25,
                            font=dict(size=18, color='#ffff00', family='Arial Black')
                        )
                    ]
                )
            
            # Determine status color for display
            status_color = "success" if data.get("data_source") == "live_model" else "danger"
            # MITRE ATLAS catalog coverage (from vendored STIX bundle)
            atlas_catalog = (data.get("atlas_attacks") or {}).get("catalog") or {}
            atlas_tactics_n = atlas_catalog.get("tactic_count") or "?"
            atlas_techniques_n = atlas_catalog.get("technique_count") or "?"
            
            # Create WORKING dashboard layout with no DataFrame errors
            return html.Div([
                # DATA STATUS HEADER
                dbc.Alert([
                    html.H1("🛡️ NeurInSpectre Time Travel Debugger", className="text-center mb-2"),
                    html.H4(f"AI Security Analysis Platform - {'Real Adversarial Data' if use_real_data else 'Simulated Data'}", className="text-center", style={'color': 'white'}),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Badge(f"{'🔴 REAL ATTACK DATA ACTIVE' if use_real_data else '⚠️ SIMULATED DATA MODE'}", color="danger" if use_real_data else "warning", className="me-2"),
                            dbc.Badge(
                                f"MITRE ATLAS coverage: {atlas_tactics_n} tactics / {atlas_techniques_n} techniques | {data_size_mb:.2f}MB of {'real adversarial' if use_real_data else 'simulated'} data",
                                color="warning" if use_real_data else "info",
                            )
                        ], className="text-center")
                    ])
                ], color="dark", className="mb-4"),
                
                # STATUS BAR - ENHANCED FOR LIVE MODEL
                dbc.Card([
                    dbc.CardBody([
                        html.P([
                            "TTD enabled for gradient leakage, attention hijacking, and ATLAS attack chains",
                            html.Br(),
                            f"Status: {real_data_status}",
                            html.Br(),
                            f"Live Status: {live_status} | Model: {selected_model} | Updates: {n}"
                        ], className="mb-0 text-center", style={'color': 'white', 'font-weight': 'bold', 'font-size': '14px'})
                    ])
                ], color=status_color, className="mb-4"),

                # Non-fatal data source warning (keeps dashboard visible)
                (dbc.Alert(
                    [
                        html.H6('⚠️ Data Source Warning', className='mb-1'),
                        html.P(getattr(ttd, '_last_good_error', ''), className='mb-0'),
                    ],
                    color='warning',
                    className='mb-4',
                    is_open=bool(getattr(ttd, '_last_good_error', None)),
                ) if getattr(ttd, '_last_good_error', None) else html.Div()),
                
                # MITRE ATLAS INTELLIGENCE PANELS (2024-2025 RESEARCH VALIDATED)
                dbc.Row([
                    dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("🔴 Red Team Intelligence - MITRE ATLAS v5.1.1 (Official)"),
                                dbc.CardBody([
                                    html.Ul([
                                        html.Li("🎯 AML.T0020: Poison Training Data (Embed vulnerabilities)"),
                                        html.Li("⚡ AML.T0043: Craft Adversarial Data (Modified inputs)"), 
                                        html.Li("🔗 AML.T0051: LLM Prompt Injection (Malicious prompts)"),
                                        html.Li("🔍 AML.T0024.000: Infer Training Data Membership (Privacy leak)"),
                                        html.Li("📊 AML.T0024.001: Invert AI Model (Reconstruct training data)"),
                                        html.Li("🛡️ AML.T0044: Full AI Model Access (White-box gradients)"),
                                        html.Li("✅ AML.T0042: Verify Attack (Test effectiveness)"),
                                        html.Li("⚠️ AML.T0048: External Harms (Downstream impact)"),
                                    ], className="mb-0", style={'fontSize': '13px'})
                                ])
                            ], color="dark", outline=True)
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("🔵 Blue Team Defense - Research-Based"),
                            dbc.CardBody([
                                html.Ul([
                                    html.Li("🛡️ Gradient Analysis: Differential privacy (ε≤1.0)"),
                                    html.Li("⚙️ Multi-Modal Detection: Cross-modal consistency"),
                                    html.Li("⏱️ Adaptive Thresholds: Dynamic detection systems"),
                                    html.Li("🏃 TTD Integration: Real-time ATLAS mapping"),
                                    html.Li("📊 Technique tagging: STIX-normalized ATLAS IDs (heuristic)"),
                                ], className="mb-0")
                            ])
                        ], color="dark", outline=True)
                    ], width=6)
                ], className="mb-4"),
                
                # MAIN VISUALIZATION - ATTACK TIMELINE + SCROLLING MITRE ATLAS PANEL
                dbc.Row([
                    # Left: Attack Timeline (8 columns)
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H3("⚡ Real-Time Attack Analysis", style={'margin': '0', 'padding': '10px'})),
                            dbc.CardBody([
                                html.Div([
                                    dcc.Graph(
                                        figure=timeline_fig,
                                        style={'height': '700px', 'width': '100%'},
                                        config={'scrollZoom': True, 'displayModeBar': True}
                                    )
                                ], className="graph-container")
                            ])
                        ])
                    ], width=8),
                    
                                        # Right: Scrolling MITRE ATLAS Tactics & Techniques Panel (4 columns)
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("🛡️ MITRE ATLAS v5.1.1", style={'margin': '0', 'display': 'inline-block'}),
                                html.Span(" Complete Matrix", style={'fontSize': '11px', 'color': '#888'})
                            ], style={'padding': '8px', 'backgroundColor': '#1a1a2e'}),
                            dbc.CardBody([
                                html.Div(atlas_panel_children, style={
                                    'height': '680px',
                                    'overflowY': 'scroll',
                                    'overflowX': 'hidden',
                                    'paddingRight': '5px',
                                    'fontSize': '11px'
                                })
                            ], style={'padding': '8px', 'backgroundColor': '#0d1117'})
                        ], color="dark", outline=True, style={'height': '760px', 'backgroundColor': '#0d1117'})
                    ], width=4)
                ], className="mb-4"),
                
                # SECONDARY VISUALIZATIONS
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4(f"🔍 Gradient Leakage Analysis ({'Real Adversarial Data' if use_real_data else 'Simulated Data'})", style={'margin': '0', 'padding': '10px'})),
                            dbc.CardBody([
                                html.Div([
                                    dcc.Graph(
                                        figure=grad_fig,
                                        style={'height': '500px', 'width': '100%'},
                                        config={'scrollZoom': True, 'displayModeBar': True}
                                    )
                                ], className="graph-container")
                            ])
                        ])
                    ], width=8),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("🎯 Attack Metrics", style={'margin': '0', 'padding': '10px'})),
                            dbc.CardBody([
                                html.Div([
                                    dcc.Graph(
                                        figure=gauge_fig,
                                        style={'height': '450px', 'width': '100%'},
                                        config={'displayModeBar': True}
                                    )
                                ], className="graph-container")
                            ])
                        ])
                    ], width=4)
                ], className="mb-4"),
                
                # REAL DATA METRICS
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("📊 Real Attack Data", className="card-title"),
                                html.P(f"✅ Privacy Budget: {current_privacy:.2f}ε"),
                                html.P(f"✅ Gradient Norm: {current_gradient:.2f}"),
                                html.P(f"✅ Model: {selected_model}"),
                                html.P(f"✅ Data Source: {'⚠️ SIMULATED DATA' if not use_real_data else f'✅ REAL ADVERSARIAL DATA LOADED'}"),
                            ])
                        ], color="success" if use_real_data else "warning", outline=True)
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("🎯 Security Metrics", className="card-title"),
                                html.P(f"⚠️ Privacy Budget: {'HIGH USAGE' if current_privacy > 10 else 'NORMAL'}"),
                                html.P(f"✅ Gradient Leakage: {'LOW' if current_gradient < 5 else 'HIGH'}"),
                                html.P(f"⚠️ Attack Surface: {'EXPOSED' if threat_score > 0.6 else 'SECURE'}"),
                                html.P(f"🔄 Updates: {n}"),
                            ])
                        ], color=threat_color, outline=True)
                    ], width=6)
                ], className="mb-4"),
                
                # DYNAMIC RED TEAM / BLUE TEAM ACTION CARDS (Updates based on graph analysis)
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("🔴 Red Team Operations", className="text-danger"),
                                html.H6("Immediate Actions:", style={'margin-top': '10px'}),
                                html.Ul([html.Li(action) for action in dict.fromkeys(red_team_actions)], style={'margin-bottom': '0px'})
                            ])
                        ], color="danger", outline=True)
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("🔵 Blue Team Defense", className="text-primary"),
                                html.H6("Defensive Measures:", style={'margin-top': '10px'}),
                                html.Ul([html.Li(action) for action in dict.fromkeys(blue_team_actions)], style={'margin-bottom': '0px'})
                            ])
                        ], color="primary", outline=True)
                    ], width=6)
                ])
            ])
            
        except Exception as e:
            print(f"❌ TTD Callback error (crash prevented): {e}")
            
            # Always return valid, working content to prevent dashboard crash
            try:
                # Try to get basic model info safely
                model_name = selected_model or ttd.model_name or 'gpt2'
                
                return html.Div([
                    # Header
                    html.Div([
                        html.H1("🛡️ NeurInSpectre TTD Dashboard", 
                               style={'color': 'white', 'textAlign': 'center', 'marginBottom': '20px'})
                    ]),
                    
                    # Status bar - ALWAYS WORKING
                    dbc.Card([
                        dbc.CardBody([
                            html.P([
                                "TTD enabled for gradient leakage, attention hijacking, and ATLAS attack chains",
                                html.Br(),
                                f"Status: Dashboard running in safe mode",
                                html.Br(),
                                f"Model: {model_name} | Device: mps | Updates: {n}"
                            ], className="mb-0 text-center", style={'color': 'white', 'font-weight': 'bold'})
                        ])
                    ], color="success", className="mb-4"),
                    
                    # MITRE ATLAS panels - STATIC VERSION
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("🔴 Red Team Intelligence - MITRE ATLAS v5.1.1"),
                                dbc.CardBody([
                                    html.Ul([
                                        html.Li("🎯 AML.T0020: Poison Training Data"),
                                        html.Li("⚡ AML.T0043: Craft Adversarial Data"), 
                                        html.Li("🔗 AML.T0051: LLM Prompt Injection"),
                                        html.Li("🔍 AML.T0024.000: Infer Training Data Membership"),
                                        html.Li("📊 AML.T0044: Full AI Model Access"),
                                    ])
                                ])
                            ], color="dark", outline=True)
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("🔵 Blue Team Defense - Research-Based"),
                                dbc.CardBody([
                                    html.Ul([
                                        html.Li("🛡️ Gradient Analysis: Differential privacy"),
                                        html.Li("⚙️ Multi-Modal Detection: Cross-modal consistency"),
                                        html.Li("⏱️ Adaptive Thresholds: Dynamic detection"),
                                        html.Li(f"🎯 Current Model: {model_name}"),
                                    ])
                                ])
                            ], color="dark", outline=True)
                        ], width=6)
                    ], className="mb-4"),
                    
                    # Basic status info
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📊 Dashboard Status"),
                            html.P(f"✅ Model: {model_name}"),
                            html.P("✅ Device: Apple Silicon MPS"),
                            html.P("✅ Status: Running safely"),
                            html.P(f"✅ Updates: {n}")
                        ])
                    ])
                ])
                
            except Exception as fallback_error:
                print(f"❌ Even fallback failed: {fallback_error}")
                # Absolute minimal fallback
                return html.Div([
                    html.H1("🛡️ NeurInSpectre TTD", style={'color': 'white', 'textAlign': 'center'}),
                    html.P(f"Model: {selected_model or 'gpt2'}", style={'color': 'white', 'textAlign': 'center'}),
                    html.P("Dashboard running in minimal safe mode", style={'color': 'lightgray', 'textAlign': 'center'})
                ], style={'padding': '50px'})
    
    # Callback for updating interval - CRASH-SAFE
    @app.callback(
        Output('interval-component', 'interval'),
        [Input('update-interval-slider', 'value')],
        prevent_initial_call=False,
        suppress_callback_exceptions=True
    )
    def update_interval(value):
        """Update interval callback - Apple Silicon safe"""
        try:
            if value is None:
                value = 10
            return max(value * 1000, 1000)  # Convert to milliseconds, minimum 1 second
        except Exception as e:
            print(f"⚠️ Interval update error (handled): {e}")
            return 10000  # Default 10 seconds
    
    # Print startup information with real data status
    real_data_loaded = ttd.real_gradient_data is not None or ttd.real_attention_data is not None
    data_count = 0
    total_mb = 0.0
    
    if ttd.real_gradient_data is not None:
        data_count += 1
        total_mb += ttd.real_gradient_data.nbytes / (1024*1024)
    if ttd.real_attention_data is not None:
        data_count += 1
        total_mb += ttd.real_attention_data.nbytes / (1024*1024)
    
    print("============================================================")
    print("🔍 NeurInSpectre Time Travel Debugger")
    print("============================================================")
    print(f"🌐 Dashboard: http://{args.host}:{args.port}")
    print(f"📊 Attack Data: {data_count} datasets loaded")
    print(f"🎯 Real Data: {total_mb:.1f}MB total")
    if real_data_loaded:
        print("✅ LIVE STREAMING: Real adversarial attack data ACTIVE")
    else:
        print("⚠️ SIMULATED: Using generated security data")
    print("============================================================")
    
    # Run the app with Apple Silicon stability settings
    try:
        print("🔧 Starting TTD with Apple Silicon optimizations...")
        app.run(
            host=args.host, 
            port=args.port, 
            debug=False,  # Disable debug for Apple Silicon stability
            threaded=True,  # Enable threading
            use_reloader=False,  # Disable reloader (causes crashes on Apple Silicon)
            processes=1  # Single process for stability
        )
        return 0
    except Exception as e:
        print(f"❌ Failed to start TTD dashboard on {args.host}:{args.port}")
        print(f"Error: {e}")
        print("💡 Try using a different port: --port 8082")
        print("💡 Or check if another process is using the port")
        return 1

def main():
    """Main entry point for TTD dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TTD (Time to Detection) Dashboard for Gradient Leakage Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic TTD dashboard with simulated data
  python neurinspectre/cli/ttd.py --output security_dashboard.html
  
  # TTD with real gradient data
  python neurinspectre/cli/ttd.py --gradient-file real_leaked_grads.npy --output real_ttd.html
  
  # TTD with specific model and custom thresholds
  python neurinspectre/cli/ttd.py --model bert-base-uncased --privacy-limit 5.0 --gradient-threshold 3.0
  
  # TTD with batch data directory
  python neurinspectre/cli/ttd.py --batch-dir ./attack_data --model distilbert-base-uncased
  
  # TTD with all real data sources
  python neurinspectre/cli/ttd.py --gradient-file real_leaked_grads.npy --attention-file real_attention.npy --token-file real_tokens.txt
        """
    )
    
    # Data input arguments
    parser.add_argument("--gradient-file", "-g", 
                       help="Path to real gradient data file (.npy)")
    parser.add_argument("--attention-file", "-a", 
                       help="Path to real attention data file (.npy)")
    parser.add_argument("--token-file", "-t", 
                       help="Path to real token data file (.txt)")
    parser.add_argument("--batch-dir", "-b", 
                       help="Directory containing batch data files")
    
    # Model configuration
    parser.add_argument("--model", "-m", default="distilbert-base-uncased",
                       choices=["distilbert-base-uncased", "bert-base-uncased", "gpt2", "roberta-base", "t5-base"],
                       help="Model to analyze (default: distilbert-base-uncased)")
    
    # Security thresholds
    parser.add_argument("--privacy-limit", type=float, default=3.0,
                       help="Privacy budget limit for ε-differential privacy (default: 10.0)")
    parser.add_argument("--gradient-threshold", type=float, default=5.0,
                       help="Critical gradient norm threshold for data extraction (default: 5.0)")
    parser.add_argument("--mi-threshold", type=float, default=0.8,
                       help="Membership inference risk threshold (default: 0.8)")
    parser.add_argument("--reconstruction-threshold", type=float, default=0.7,
                       help="Data reconstruction attack threshold (default: 0.7)")
    
    # Output options
    parser.add_argument("--output", "-o", default="ttd_gradient_leakage_dashboard.html",
                       help="Output HTML file for dashboard")
    parser.add_argument("--use-simulated", action="store_true",
                       help="Force use of simulated data even if real data is available")
    parser.add_argument("--real-time", action="store_true",
                       help="Enable real-time monitoring mode (updates dashboard periodically)")
    parser.add_argument("--update-interval", type=int, default=30,
                       help="Update interval in seconds for real-time mode (default: 30)")
    
    args = parser.parse_args()
    
    # Create TTD dashboard with custom configuration
    ttd = TTDDashboard(
        model_name=args.model,
        privacy_budget_limit=args.privacy_limit,
        gradient_norm_threshold=args.gradient_threshold,
        mi_threshold=args.mi_threshold,
        reconstruction_threshold=args.reconstruction_threshold,
        allow_simulated=args.use_simulated,
    )
    
    # Load real data if specified
    if args.gradient_file or args.attention_file or args.token_file or args.batch_dir:
        ttd.load_real_data(
            gradient_file=args.gradient_file,
            attention_file=args.attention_file,
            token_file=args.token_file,
            batch_dir=args.batch_dir
        )
    
    # Real-time monitoring mode
    if args.real_time:
        print(f"🔄 Starting real-time monitoring mode (updates every {args.update_interval}s)")
        print("Press Ctrl+C to stop...")
        
        try:
            iteration = 0
            while True:
                iteration += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"ttd_realtime_{timestamp}_iter{iteration}.html"
                
                ttd.run_ttd_analysis(output_file, use_real_data=not args.use_simulated)
                print(f"🔄 Real-time update {iteration} completed: {output_file}")
                
                time.sleep(args.update_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ Real-time monitoring stopped by user")
    else:
        # Single analysis run
        ttd.run_ttd_analysis(args.output, use_real_data=not args.use_simulated)
        print(f"\n🌐 Open {args.output} in your browser to view the security dashboard")

if __name__ == "__main__":
    main() 