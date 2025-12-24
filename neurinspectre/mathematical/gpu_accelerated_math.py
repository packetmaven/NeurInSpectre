"""
NeurInSpectre: Advanced Mathematical Foundations with GPU Acceleration

This module implements the core mathematical foundations with support for:
- Mac Silicon MPS acceleration
- NVIDIA GPU CUDA acceleration
- Advanced spectral analysis techniques
- Sophisticated gradient obfuscation detection
"""

import numpy as np
import torch
import torch.nn.functional as F

class GPUAcceleratedMathEngine:
    """
    Advanced mathematical engine with GPU acceleration for both MPS and CUDA
    Implements world-class algorithms for gradient obfuscation detection
    """
    
    def __init__(self, precision='float32', device_preference='auto'):
        """
        Initialize GPU-accelerated mathematical engine
        
        Args:
            precision: 'float32' or 'float64' for numerical precision
            device_preference: 'auto', 'mps', 'cuda', or 'cpu'
        """
        self.precision = torch.float32 if precision == 'float32' else torch.float64
        self.device = self._initialize_device(device_preference)
        self.eps = torch.finfo(self.precision).eps
        
        # Advanced mathematical constants
        self.pi = torch.tensor(np.pi, dtype=self.precision, device=self.device)
        self.euler_gamma = torch.tensor(0.5772156649015329, dtype=self.precision, device=self.device)
        
        # Precomputed matrices for efficiency
        self._precomputed_matrices = {}
        self._krylov_cache = {}
        
        print(f"🚀 GPU Mathematical Engine initialized on {self.device}")
        print(f"   Precision: {precision}")
        print(f"   Device capabilities: {self._get_device_info()}")
    
    def _initialize_device(self, preference):
        """Initialize the best available device"""
        if preference == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        elif preference == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        elif preference == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif preference == 'cpu':
            return torch.device('cpu')
        else:
            return torch.device('cpu')
    
    def _get_device_info(self):
        """Get detailed device information"""
        if self.device.type == 'mps':
            return "Mac Silicon MPS (Metal Performance Shaders)"
        elif self.device.type == 'cuda':
            return f"NVIDIA GPU: {torch.cuda.get_device_name()}"
        else:
            return "CPU (Fallback)"
    
    def advanced_spectral_decomposition(self, gradient_tensor, decomposition_levels=5):
        """
        Advanced multi-level spectral decomposition with GPU acceleration
        
        Implements sophisticated spectral analysis for gradient obfuscation detection:
        - Multi-resolution spectral decomposition
        - Wavelet-like analysis in frequency domain
        - Advanced filtering for obfuscation patterns
        
        Args:
            gradient_tensor: Input gradient data
            decomposition_levels: Number of decomposition levels
            
        Returns:
            dict: Comprehensive spectral analysis results
        """
        # Convert to GPU tensor
        if not isinstance(gradient_tensor, torch.Tensor):
            gradient_tensor = torch.tensor(gradient_tensor, dtype=self.precision, device=self.device)
        else:
            gradient_tensor = gradient_tensor.to(device=self.device, dtype=self.precision)

        # Robustness: sanitize non-finite values early so downstream FFT/entropy math is stable.
        gradient_tensor = torch.nan_to_num(gradient_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure proper shape for analysis
        if gradient_tensor.dim() == 1:
            gradient_tensor = gradient_tensor.unsqueeze(0)
        
        batch_size, seq_len = gradient_tensor.shape
        
        # 1. Multi-level FFT decomposition
        spectral_levels = {}
        current_signal = gradient_tensor.clone()
        
        for level in range(decomposition_levels):
            # Compute FFT at current level
            fft_result = torch.fft.fft(current_signal, dim=-1)
            
            # Extract magnitude and phase
            magnitude = torch.abs(fft_result)
            # Phase: avoid MPS CPU-fallback warnings by using atan2 on real/imag.
            if str(getattr(fft_result.device, "type", "")) == "mps":
                phase = torch.atan2(fft_result.imag, fft_result.real)
            else:
                phase = torch.angle(fft_result)
            
            # Compute power spectral density
            psd = magnitude ** 2 / seq_len
            
            # Advanced spectral features
            spectral_centroid = self._compute_spectral_centroid(magnitude)
            spectral_rolloff = self._compute_spectral_rolloff(magnitude)
            spectral_flux = self._compute_spectral_flux(magnitude)
            spectral_entropy = self._compute_spectral_entropy(psd)
            
            spectral_levels[f'level_{level}'] = {
                'magnitude': magnitude,
                'phase': phase,
                'psd': psd,
                'centroid': spectral_centroid,
                'rolloff': spectral_rolloff,
                'flux': spectral_flux,
                'entropy': spectral_entropy
            }
            
            # Downsample for next level (if possible)
            if current_signal.shape[-1] > 4:
                current_signal = F.avg_pool1d(current_signal.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
            else:
                break
        
        # 2. Cross-level spectral correlation analysis
        cross_correlations = self._compute_cross_level_correlations(spectral_levels)
        
        # 3. Advanced obfuscation indicators
        obfuscation_indicators = self._compute_obfuscation_indicators(spectral_levels, cross_correlations)
        
        return {
            'spectral_levels': spectral_levels,
            'cross_correlations': cross_correlations,
            'obfuscation_indicators': obfuscation_indicators,
            'summary_metrics': self._compute_summary_metrics(spectral_levels)
        }
    
    def _compute_spectral_centroid(self, magnitude):
        """Compute spectral centroid (center of mass of spectrum)"""
        freqs = torch.arange(magnitude.shape[-1], dtype=self.precision, device=self.device)
        denom = torch.sum(magnitude, dim=-1) + self.eps
        return torch.sum(freqs * magnitude, dim=-1) / denom
    
    def _compute_spectral_rolloff(self, magnitude, rolloff_threshold=0.85):
        """Compute spectral rolloff (frequency below which X% of energy is contained)"""
        # "Energy" rolloff should be computed on power (magnitude^2), not raw magnitude.
        power = magnitude ** 2
        cumsum = torch.cumsum(power, dim=-1)
        total_energy = cumsum[..., -1:]
        threshold = rolloff_threshold * total_energy
        
        # Find rolloff frequency
        rolloff_indices = torch.argmax((cumsum >= threshold).float(), dim=-1)
        return rolloff_indices.float()
    
    def _compute_spectral_flux(self, magnitude):
        """Compute spectral flux (measure of spectral change)"""
        if magnitude.shape[0] > 1:
            diff = torch.diff(magnitude, dim=0)
            return torch.mean(torch.abs(diff), dim=-1)
        else:
            return torch.zeros(magnitude.shape[0], device=self.device, dtype=self.precision)
    
    def _compute_spectral_entropy(self, psd):
        """Compute normalized spectral entropy in [0, 1] (measure of spectral complexity).

        Notes:
        - We treat the PSD as a discrete distribution after normalization.
        - Entropy is normalized by the maximum entropy log(N_bins) so values are comparable
          across different decomposition levels (which have different spectrum lengths).
        """
        psd = torch.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize PSD to probability distribution
        psd_sum = torch.sum(psd, dim=-1, keepdim=True)
        psd_norm = psd / (psd_sum + self.eps)

        # Compute entropy (avoiding log(0))
        log_psd = torch.log(psd_norm + self.eps)
        entropy = -torch.sum(psd_norm * log_psd, dim=-1)

        n_bins = int(psd.shape[-1])
        if n_bins < 2:
            return torch.zeros_like(entropy)

        max_entropy = torch.log(torch.tensor(float(n_bins), device=self.device, dtype=self.precision))
        ent_norm = entropy / (max_entropy + self.eps)
        return torch.clamp(ent_norm, 0.0, 1.0)
    
    def _compute_cross_level_correlations(self, spectral_levels):
        """Compute correlations between different spectral levels"""
        correlations = {}
        level_keys = list(spectral_levels.keys())
        
        for i, level1 in enumerate(level_keys):
            for j, level2 in enumerate(level_keys[i+1:], i+1):
                # Compute correlation between magnitude spectra
                mag1 = spectral_levels[level1]['magnitude']
                mag2 = spectral_levels[level2]['magnitude']
                
                # Resize to same length for correlation
                min_len = min(mag1.shape[-1], mag2.shape[-1])
                mag1_resized = mag1[..., :min_len]
                mag2_resized = mag2[..., :min_len]
                
                # Compute Pearson correlation
                correlation = self._pearson_correlation(mag1_resized, mag2_resized)
                correlations[f'{level1}_{level2}'] = correlation
        
        return correlations
    
    def _pearson_correlation(self, x, y):
        """Compute Pearson correlation coefficient"""
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        y_mean = torch.mean(y, dim=-1, keepdim=True)
        
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        numerator = torch.sum(x_centered * y_centered, dim=-1)
        denominator = torch.sqrt(torch.sum(x_centered**2, dim=-1) * torch.sum(y_centered**2, dim=-1))

        corr = numerator / (denominator + self.eps)
        return torch.clamp(corr, -1.0, 1.0)
    
    def _compute_obfuscation_indicators(self, spectral_levels, cross_correlations):
        """Compute advanced obfuscation indicators"""
        indicators = {}
        
        # 1. Spectral irregularity (indicates artificial manipulation)
        spectral_irregularity = []
        for level_data in spectral_levels.values():
            magnitude = level_data['magnitude']
            # Compute second derivative to detect irregularities
            if magnitude.shape[-1] > 2:
                second_deriv = torch.diff(magnitude, n=2, dim=-1)
                irregularity = torch.std(second_deriv, dim=-1, unbiased=False)
                spectral_irregularity.append(irregularity)
        
        if spectral_irregularity:
            indicators['spectral_irregularity'] = torch.stack(spectral_irregularity, dim=-1)
        
        # 2. Cross-level consistency (obfuscated gradients show inconsistency)
        correlation_values = list(cross_correlations.values())
        if correlation_values:
            correlation_tensor = torch.stack(correlation_values, dim=-1)
            indicators['cross_level_consistency'] = torch.mean(correlation_tensor, dim=-1)
            if correlation_tensor.shape[-1] < 2:
                indicators['correlation_variance'] = torch.zeros_like(indicators['cross_level_consistency'])
            else:
                indicators['correlation_variance'] = torch.var(correlation_tensor, dim=-1, unbiased=False)
        
        # 3. Entropy progression (natural gradients show smooth entropy progression)
        entropy_values = []
        for level_data in spectral_levels.values():
            entropy_values.append(level_data['entropy'])
        
        if len(entropy_values) > 1:
            entropy_tensor = torch.stack(entropy_values, dim=-1)
            entropy_diff = torch.diff(entropy_tensor, dim=-1)
            if entropy_diff.shape[-1] < 2:
                indicators['entropy_progression_smoothness'] = torch.zeros_like(entropy_tensor[..., 0])
            else:
                indicators['entropy_progression_smoothness'] = torch.std(entropy_diff, dim=-1, unbiased=False)
        
        # 4. Frequency domain artifacts (specific to obfuscation techniques)
        indicators['high_frequency_energy'] = self._compute_high_frequency_energy(spectral_levels)
        indicators['spectral_peaks_anomaly'] = self._detect_spectral_peaks_anomaly(spectral_levels)
        
        return indicators
    
    def _compute_high_frequency_energy(self, spectral_levels):
        """Compute energy in high-frequency components (often elevated in obfuscated gradients)"""
        high_freq_energy = []
        
        for level_data in spectral_levels.values():
            psd = level_data['psd']
            # Consider top 25% frequencies as "high frequency"
            high_freq_start = int(0.75 * psd.shape[-1])
            high_freq_psd = psd[..., high_freq_start:]
            energy = torch.sum(high_freq_psd, dim=-1)
            high_freq_energy.append(energy)
        
        if high_freq_energy:
            return torch.stack(high_freq_energy, dim=-1)
        else:
            return torch.zeros(1, device=self.device, dtype=self.precision)
    
    def _detect_spectral_peaks_anomaly(self, spectral_levels):
        """Detect anomalous spectral peaks that indicate obfuscation"""
        peak_anomalies = []
        
        for level_data in spectral_levels.values():
            magnitude = level_data['magnitude']
            
            # Find peaks using simple local maxima detection
            if magnitude.shape[-1] > 2:
                # Compute local maxima
                left_shift = F.pad(magnitude[..., 1:], (0, 1), value=0)
                right_shift = F.pad(magnitude[..., :-1], (1, 0), value=0)
                
                peaks = (magnitude > left_shift) & (magnitude > right_shift)
                peak_heights = magnitude * peaks.float()
                
                # Compute peak anomaly score
                mean_peak_height = torch.mean(peak_heights, dim=-1)
                max_peak_height = torch.max(peak_heights, dim=-1)[0]
                
                # Anomaly score based on peak height distribution
                anomaly_score = max_peak_height / (mean_peak_height + self.eps)
                peak_anomalies.append(anomaly_score)
        
        if peak_anomalies:
            return torch.stack(peak_anomalies, dim=-1)
        else:
            return torch.zeros(1, device=self.device, dtype=self.precision)
    
    def _compute_summary_metrics(self, spectral_levels):
        """Compute summary metrics for the entire spectral analysis"""
        summary = {}
        
        # Overall spectral complexity
        all_entropies = []
        all_centroids = []
        all_rolloffs = []
        
        for level_data in spectral_levels.values():
            all_entropies.append(level_data['entropy'])
            all_centroids.append(level_data['centroid'])
            all_rolloffs.append(level_data['rolloff'])
        
        if all_entropies:
            summary['mean_entropy'] = torch.mean(torch.stack(all_entropies, dim=-1), dim=-1)
            summary['entropy_variance'] = torch.var(torch.stack(all_entropies, dim=-1), dim=-1, unbiased=False)
            
            summary['mean_centroid'] = torch.mean(torch.stack(all_centroids, dim=-1), dim=-1)
            summary['centroid_variance'] = torch.var(torch.stack(all_centroids, dim=-1), dim=-1, unbiased=False)
            
            summary['mean_rolloff'] = torch.mean(torch.stack(all_rolloffs, dim=-1), dim=-1)
            summary['rolloff_variance'] = torch.var(torch.stack(all_rolloffs, dim=-1), dim=-1, unbiased=False)
        
        return summary

class AdvancedExponentialIntegrator:
    """
    Advanced exponential time differencing schemes with GPU acceleration
    Implements sophisticated ETD methods for gradient evolution analysis
    """
    
    def __init__(self, math_engine):
        """
        Initialize with GPU-accelerated mathematical engine
        
        Args:
            math_engine: GPUAcceleratedMathEngine instance
        """
        self.math_engine = math_engine
        self.device = math_engine.device
        self.precision = math_engine.precision
        self.eps = math_engine.eps
        
        # Precomputed phi functions for efficiency
        self._phi_cache = {}
        
    def etd_rk4_step(self, u, L, N, dt, krylov_dim=30):
        """
        Fourth-order Exponential Time Differencing Runge-Kutta step
        
        Implements the ETD-RK4 scheme with Krylov subspace approximation:
        u_{n+1} = e^{L*dt} u_n + dt * phi_1(L*dt) N(u_n) + ...
        
        Args:
            u: Current state vector (gradient data)
            L: Linear operator (differential operator)
            N: Nonlinear function
            dt: Time step
            krylov_dim: Krylov subspace dimension for matrix exponential
            
        Returns:
            torch.Tensor: Next state vector
        """
        input_was_vector = False

        # Convert inputs to GPU tensors
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=self.precision, device=self.device)
        else:
            u = u.to(device=self.device, dtype=self.precision)
        
        # Ensure proper shape
        if u.dim() == 1:
            input_was_vector = True
            u = u.unsqueeze(0)
        
        batch_size, n = u.shape
        
        # Create linear operator matrix if not provided
        if L is None:
            L = self._create_default_linear_operator(n)
        elif not isinstance(L, torch.Tensor):
            L = torch.tensor(L, dtype=self.precision, device=self.device)
        else:
            L = L.to(device=self.device, dtype=self.precision)

        def _as_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=self.device, dtype=self.precision)
            return torch.tensor(x, dtype=self.precision, device=self.device)

        def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
            # Ensure x is (batch, n)
            if x.dim() == 1:
                return x.unsqueeze(0)
            return x

        def _matmul(M: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            # M: (n,n), X: (batch,n) -> (batch,n)
            return torch.matmul(M, X.T).T

        A = L * dt  # scaled linear operator
        A2 = 0.5 * A

        # Stage 1 nonlinear term
        N1 = _ensure_batch(_as_tensor(N(u) if callable(N) else N))

        # For moderate n, compute dense phi matrices and use standard ETD-RK4.
        # For larger n, compute phi(A)·v via Krylov/Arnoldi (matvec) to avoid O(n^2) storage.
        if n <= 256:
            # Dense phi matrices for A and A/2
            E, phi1, phi2, phi3 = self._direct_phi_computation(A)
            E2, phi1h, _phi2h, _phi3h = self._direct_phi_computation(A2)

            Q = 0.5 * dt * phi1h
            # Cox–Matthews / Kassam–Trefethen ETD-RK4 coefficients.
            # These reduce to classical RK4 weights when L=0 (phi1(0)=1, phi2(0)=1/2, phi3(0)=1/6).
            f1 = dt * (phi1 - 3.0 * phi2 + 4.0 * phi3)
            f2 = dt * (2.0 * phi2 - 4.0 * phi3)
            f3 = dt * (4.0 * phi3 - phi2)

            a = _matmul(E2, u) + _matmul(Q, N1)
            N2 = _ensure_batch(_as_tensor(N(a) if callable(N) else N1))

            b = _matmul(E2, u) + _matmul(Q, N2)
            N3 = _ensure_batch(_as_tensor(N(b) if callable(N) else N1))

            c = _matmul(E2, a) + _matmul(Q, (2.0 * N3 - N1))
            N4 = _ensure_batch(_as_tensor(N(c) if callable(N) else N1))

            u_new = _matmul(E, u) + _matmul(f1, N1) + _matmul(f2, (N2 + N3)) + _matmul(f3, N4)

            return u_new.squeeze(0) if input_was_vector else u_new

        # Large-n path: Krylov φ(A)·v products.
        out = torch.zeros_like(u)
        for b_idx in range(batch_size):
            u_b = u[b_idx].reshape(-1)
            N1_b = N1[b_idx].reshape(-1)

            E_u, _, _, _ = self._krylov_phi_apply_all(A, u_b, m=krylov_dim)
            E2_u, _, _, _ = self._krylov_phi_apply_all(A2, u_b, m=krylov_dim)

            _, phi1h_N1, _, _ = self._krylov_phi_apply_all(A2, N1_b, m=krylov_dim)
            Q_N1 = 0.5 * dt * phi1h_N1
            a_b = E2_u + Q_N1
            N2_b = _as_tensor(N(a_b) if callable(N) else N1_b).reshape(-1)

            _, phi1h_N2, _, _ = self._krylov_phi_apply_all(A2, N2_b, m=krylov_dim)
            Q_N2 = 0.5 * dt * phi1h_N2
            b_b = E2_u + Q_N2
            N3_b = _as_tensor(N(b_b) if callable(N) else N1_b).reshape(-1)

            E2_a, _, _, _ = self._krylov_phi_apply_all(A2, a_b, m=krylov_dim)
            _, phi1h_combo, _, _ = self._krylov_phi_apply_all(A2, (2.0 * N3_b - N1_b), m=krylov_dim)
            Q_combo = 0.5 * dt * phi1h_combo
            c_b = E2_a + Q_combo
            N4_b = _as_tensor(N(c_b) if callable(N) else N1_b).reshape(-1)

            # ETD-RK4 coefficients (same as dense path) applied via Krylov φ(A)·v products.
            _, phi1_N1, phi2_N1, phi3_N1 = self._krylov_phi_apply_all(A, N1_b, m=krylov_dim)
            _, _phi1_c, phi2_c, phi3_c = self._krylov_phi_apply_all(A, (N2_b + N3_b), m=krylov_dim)
            _, _phi1_N4, phi2_N4, phi3_N4 = self._krylov_phi_apply_all(A, N4_b, m=krylov_dim)

            term1 = dt * (phi1_N1 - 3.0 * phi2_N1 + 4.0 * phi3_N1)
            term2 = dt * (2.0 * phi2_c - 4.0 * phi3_c)
            term3 = dt * (4.0 * phi3_N4 - phi2_N4)

            u_new_b = E_u + term1 + term2 + term3
            out[b_idx] = u_new_b

        return out.squeeze(0) if input_was_vector else out
    
    def _create_default_linear_operator(self, n):
        """Create default linear operator for gradient evolution"""
        # Create a sophisticated differential operator
        # This represents the linear part of gradient dynamics
        
        # Second-order finite difference operator (Laplacian-like)
        main_diag = -2 * torch.ones(n, dtype=self.precision, device=self.device)
        off_diag = torch.ones(n-1, dtype=self.precision, device=self.device)
        
        L = torch.diag(main_diag) + torch.diag(off_diag, 1) + torch.diag(off_diag, -1)
        
        # Add damping term
        damping = -0.1 * torch.eye(n, dtype=self.precision, device=self.device)
        L = L + damping
        
        return L
    
    def _compute_phi_functions_krylov(self, A, m):
        """
        Compute phi functions using Krylov subspace approximation
        
        phi_0(A) = exp(A)
        phi_1(A) = (exp(A) - I) / A
        phi_2(A) = (exp(A) - I - A) / A^2
        phi_3(A) = (exp(A) - I - A - A^2/2) / A^3
        
        Args:
            A: Matrix argument
            m: Krylov subspace dimension
            
        Returns:
            tuple: (phi_0, phi_1, phi_2, phi_3) matrices
        """
        n = A.shape[0]
        
        # NOTE: This method materializes dense (n×n) phi matrices and is only practical
        # for moderate n. For large systems, `etd_rk4_step` uses `_krylov_phi_apply`
        # to compute phi(A)·v products directly.
        if n > 256:
            raise ValueError(
                f"Refusing to materialize phi matrices for n={int(n)}. "
                "Use `_krylov_phi_apply` for phi(A)·v products."
            )
        return self._direct_phi_computation(A)
    
    def _direct_phi_computation(self, A):
        """Direct computation of phi functions for small matrices"""
        # Compute matrix exponential.
        #
        # NOTE: `torch.matrix_exp` is still not implemented on MPS in some PyTorch builds and will
        # silently fall back to CPU with a warning. We avoid that by using an MPS-safe Pade(13)
        # scaling+squaring implementation when running on MPS.
        phi_0 = self._matrix_exp(A)
        
        # Compute phi_1 = (exp(A) - I) / A
        # Use series expansion for numerical stability
        phi_1 = self._phi_1_series(A)
        
        # Compute phi_2 = (exp(A) - I - A) / A^2
        phi_2 = self._phi_2_series(A)
        
        # Compute phi_3 = (exp(A) - I - A - A^2/2) / A^3
        phi_3 = self._phi_3_series(A)
        
        return phi_0, phi_1, phi_2, phi_3

    def _matrix_exp(self, A: torch.Tensor) -> torch.Tensor:
        """
        Matrix exponential with an MPS-safe path.

        PyTorch may fall back to CPU for `torch.matrix_exp` on MPS (emitting a warning). To keep
        computation on-GPU on Apple Silicon, we implement a standard Higham-style scaling+squaring
        Pade(13) approximation using matmul + solve (both supported on MPS).
        """
        A = torch.as_tensor(A)
        if getattr(A, "device", None) is not None and str(A.device.type) == "mps":
            return self._matrix_exp_pade13(A)
        try:
            return torch.matrix_exp(A)
        except NotImplementedError:
            # Explicit fallback for older builds / edge backends
            return torch.matrix_exp(A.cpu()).to(A.device)

    def _matrix_exp_pade13(self, A: torch.Tensor) -> torch.Tensor:
        """Compute exp(A) via scaling+squaring with a (13,13) Pade approximant."""
        if A.dim() != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("matrix_exp expects a square 2D tensor")
        n = int(A.shape[0])

        # 1-norm (max column sum) — avoids relying on backend-specific linalg.norm kernels.
        norm1 = torch.sum(torch.abs(A), dim=0).max()

        # Pade(13) theta bound (Higham 2005 / Al-Mohy & Higham 2009).
        theta13 = A.new_tensor(5.371920351148152)
        # Scaling s = max(0, ceil(log2(norm1/theta13)))
        if float(norm1) <= float(theta13):
            s = 0
        else:
            s = int(torch.ceil(torch.log2(norm1 / theta13)).item())
            s = max(0, s)

        # Scale A
        A_scaled = A / float(2**s)

        eye = torch.eye(n, dtype=A.dtype, device=A.device)
        A2 = A_scaled @ A_scaled
        A4 = A2 @ A2
        A6 = A2 @ A4

        # Pade(13) coefficients
        b = [
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ]
        b = [A.new_tensor(x) for x in b]

        # Following Higham's formulation for (13,13) Pade:
        U = A_scaled @ (
            A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2) + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * eye
        )
        V = A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * eye

        # exp(A) ≈ (V - U)^{-1} (V + U)
        P = V + U
        Q = V - U
        # On MPS, torch.linalg.solve currently falls back to CPU (warns).
        # Use an explicit inverse instead to keep the computation on-device.
        if str(getattr(A.device, "type", "")) == "mps":
            R = torch.linalg.inv(Q) @ P
        else:
            R = torch.linalg.solve(Q, P)

        # Squaring
        for _ in range(s):
            R = R @ R
        return R
    
    def _phi_1_series(self, A, max_terms=20):
        """Compute phi_1(A) = Σ_{j>=0} A^j/(j+1)! via a stable recurrence."""
        n = A.shape[0]
        eye = torch.eye(n, dtype=self.precision, device=self.device)
        phi = eye.clone()
        term = eye.clone()  # A^0 / 1!

        for j in range(1, max_terms + 1):
            term = torch.matmul(term, A) / float(j + 1)  # A^j/(j+1)!
            phi = phi + term
            if torch.norm(term) < self.eps:
                break

        return phi
    
    def _phi_2_series(self, A, max_terms=20):
        """Compute phi_2(A) = Σ_{j>=0} A^j/(j+2)! via a stable recurrence."""
        n = A.shape[0]
        eye = torch.eye(n, dtype=self.precision, device=self.device)
        phi = 0.5 * eye
        term = 0.5 * eye  # A^0 / 2!

        for j in range(1, max_terms + 1):
            term = torch.matmul(term, A) / float(j + 2)  # A^j/(j+2)!
            phi = phi + term
            if torch.norm(term) < self.eps:
                break

        return phi
    
    def _phi_3_series(self, A, max_terms=20):
        """Compute phi_3(A) = Σ_{j>=0} A^j/(j+3)! via a stable recurrence."""
        n = A.shape[0]
        eye = torch.eye(n, dtype=self.precision, device=self.device)
        phi = (1.0 / 6.0) * eye
        term = (1.0 / 6.0) * eye  # A^0 / 3!

        for j in range(1, max_terms + 1):
            term = torch.matmul(term, A) / float(j + 3)  # A^j/(j+3)!
            phi = phi + term
            if torch.norm(term) < self.eps:
                break

        return phi
    
    def _krylov_phi_approximation(self, A, m):
        """
        Deprecated: this implementation previously returned placeholder identities.

        For large systems, compute phi(A)·v products via `_krylov_phi_apply(...)`.
        Materializing dense phi matrices is generally not practical.
        """
        raise NotImplementedError(
            "Use `_krylov_phi_apply(A, v, order, m)` for Krylov phi(A)·v products."
        )

    def _krylov_phi_apply_all(self, A: torch.Tensor, v: torch.Tensor, *, m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute exp(A)@v and phi_k(A)@v for k=1..3 via a single Arnoldi run + block exp.

        Returns:
            (expAv, phi1Av, phi2Av, phi3Av) each shaped (n,).
        """
        v = v.reshape(-1)
        n = int(v.numel())

        v_norm = torch.norm(v)
        if float(v_norm) < 1e-14:
            z = torch.zeros_like(v)
            return z, z, z, z

        m_eff = int(min(max(m, 1), n))
        V = torch.zeros((n, m_eff), dtype=self.precision, device=self.device)
        H = torch.zeros((m_eff + 1, m_eff), dtype=self.precision, device=self.device)

        V[:, 0] = v / v_norm

        # Arnoldi with modified Gram-Schmidt
        for j in range(m_eff):
            w = torch.matmul(A, V[:, j])

            for i in range(j + 1):
                hij = torch.dot(V[:, i], w)
                H[i, j] = hij
                w = w - hij * V[:, i]

            if j < m_eff - 1:
                hj1 = torch.norm(w)
                H[j + 1, j] = hj1
                if float(hj1) < 1e-14:
                    m_eff = j + 1
                    V = V[:, :m_eff]
                    H = H[: m_eff + 1, :m_eff]
                    break
                V[:, j + 1] = w / hj1

        Hm = H[:m_eff, :m_eff]

        # Compute exp(Hm) and phi_k(Hm) via a single block-matrix exponential:
        # exp([[Hm, I, 0, 0],
        #      [0,  0, I, 0],
        #      [0,  0, 0, I],
        #      [0,  0, 0, 0]]) has top row blocks [exp(Hm), phi1(Hm), phi2(Hm), phi3(Hm)].
        eye = torch.eye(m_eff, dtype=self.precision, device=self.device)
        Z = torch.zeros((m_eff, m_eff), dtype=self.precision, device=self.device)
        M = torch.cat(
            [
                torch.cat([Hm, eye, Z, Z], dim=1),
                torch.cat([Z, Z, eye, Z], dim=1),
                torch.cat([Z, Z, Z, eye], dim=1),
                torch.cat([Z, Z, Z, Z], dim=1),
            ],
            dim=0,
        )

        expM = self._matrix_exp(M)

        expH = expM[:m_eff, :m_eff]
        phi1 = expM[:m_eff, m_eff : 2 * m_eff]
        phi2 = expM[:m_eff, 2 * m_eff : 3 * m_eff]
        phi3 = expM[:m_eff, 3 * m_eff : 4 * m_eff]

        e1 = torch.zeros((m_eff,), dtype=self.precision, device=self.device)
        e1[0] = 1.0

        y0 = torch.matmul(expH, e1)
        y1 = torch.matmul(phi1, e1)
        y2 = torch.matmul(phi2, e1)
        y3 = torch.matmul(phi3, e1)

        expAv = (v_norm * torch.matmul(V, y0)).reshape(-1)
        phi1Av = (v_norm * torch.matmul(V, y1)).reshape(-1)
        phi2Av = (v_norm * torch.matmul(V, y2)).reshape(-1)
        phi3Av = (v_norm * torch.matmul(V, y3)).reshape(-1)
        return expAv, phi1Av, phi2Av, phi3Av

    def _krylov_phi_apply(self, A: torch.Tensor, v: torch.Tensor, *, order: int, m: int) -> torch.Tensor:
        """
        Compute phi_order(A) @ v using an Arnoldi/Krylov approximation.

        Args:
            A: (n,n) matrix (already scaled, e.g., A = L*dt)
            v: (n,) vector
            order: 0 -> exp(A), 1 -> phi1(A), 2 -> phi2(A), 3 -> phi3(A)
            m: Krylov subspace dimension
        """
        expAv, phi1Av, phi2Av, phi3Av = self._krylov_phi_apply_all(A, v, m=m)
        if order == 0:
            return expAv
        if order == 1:
            return phi1Av
        if order == 2:
            return phi2Av
        if order == 3:
            return phi3Av
        raise ValueError(f"Unsupported phi order: {order}")

def get_engine_info():
    """Get information about the math engine"""
    return "GPU Accelerated Math Engine (MPS/CUDA/CPU)"

def get_precision():
    """Get default precision"""
    return "float32"

def get_device():
    """Get default device"""
    if torch.backends.mps.is_available():
        return "Mac Silicon MPS"
    elif torch.cuda.is_available():
        return f"NVIDIA {torch.cuda.get_device_name()}"
    else:
        return "CPU"

def demonstrate_advanced_mathematics(device_preference='auto', precision='float32'):
    """Demonstrate the advanced mathematical capabilities"""
    print("🧮 Demonstrating Advanced Mathematical Foundations")
    print("=" * 60)
    
    # Initialize GPU-accelerated engine
    math_engine = GPUAcceleratedMathEngine(precision=precision, device_preference=device_preference)
    
    # Generate test gradient data with various obfuscation patterns
    print("\n📊 Generating test gradient data...")
    
    # Clean gradient
    clean_gradient = np.random.randn(512) * 0.1
    
    # Obfuscated gradient (with artificial high-frequency components)
    obfuscated_gradient = clean_gradient.copy()
    obfuscated_gradient += 0.05 * np.sin(np.arange(512) * 0.5)  # Artificial periodicity
    obfuscated_gradient[::50] += 0.2 * np.random.randn(len(obfuscated_gradient[::50]))  # Spikes
    
    # Test advanced spectral decomposition
    print("\n🔬 Testing Advanced Spectral Decomposition...")
    
    clean_results = math_engine.advanced_spectral_decomposition(clean_gradient)
    obfuscated_results = math_engine.advanced_spectral_decomposition(obfuscated_gradient)
    
    print("Clean gradient analysis:")
    print(f"  Mean entropy: {clean_results['summary_metrics']['mean_entropy'].item():.4f}")
    print(f"  Spectral irregularity: {torch.mean(clean_results['obfuscation_indicators']['spectral_irregularity']).item():.4f}")
    
    print("Obfuscated gradient analysis:")
    print(f"  Mean entropy: {obfuscated_results['summary_metrics']['mean_entropy'].item():.4f}")
    print(f"  Spectral irregularity: {torch.mean(obfuscated_results['obfuscation_indicators']['spectral_irregularity']).item():.4f}")
    
    # Test exponential integrator
    print("\n⚡ Testing Advanced Exponential Integrator...")
    
    integrator = AdvancedExponentialIntegrator(math_engine)
    
    # Simple nonlinear function for testing
    def nonlinear_func(u):
        return -0.1 * u**3 + 0.05 * torch.sin(u)
    
    # Test ETD-RK4 step
    u_initial = torch.tensor(clean_gradient[:100], dtype=math_engine.precision, device=math_engine.device)
    u_next = integrator.etd_rk4_step(u_initial, None, nonlinear_func, 0.01)
    
    print(f"Initial state norm: {torch.norm(u_initial).item():.6f}")
    print(f"Next state norm: {torch.norm(u_next).item():.6f}")
    print(f"Evolution magnitude: {torch.norm(u_next - u_initial).item():.6f}")
    
    print("\n✅ Advanced Mathematical Foundations Test Complete")
    
    return {
        'math_engine': math_engine,
        'integrator': integrator,
        'clean_results': clean_results,
        'obfuscated_results': obfuscated_results
    }

if __name__ == "__main__":
    results = demonstrate_advanced_mathematics() 