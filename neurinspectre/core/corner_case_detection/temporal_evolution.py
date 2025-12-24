"""
Temporal Evolution Analysis Module

Implements detection of temporal patterns in gradient obfuscation,
including progressive obfuscation and training phase correlation.
"""

import numpy as np
from typing import Dict, List
from scipy import stats
from scipy.fft import fft, fftfreq
import warnings

class TemporalEvolutionAnalyzer:
    """
    Analyzes temporal evolution of gradient obfuscation patterns.
    
    This detector identifies patterns that emerge over time, such as progressive
    obfuscation or training phase-specific evasion techniques.
    """
    
    def __init__(self, window_size: int = 10, step: int = 5):
        """
        Initialize the Temporal Evolution Analyzer.
        
        Args:
            window_size: Size of the sliding window for temporal analysis
            step: Step size for the sliding window
        """
        self.window_size = window_size
        self.step = step
        self.trend_models = {}
        
    def analyze_temporal_sequence(self, gradient_sequence: List[np.ndarray]) -> Dict:
        """
        Analyze a sequence of gradients for temporal evolution patterns.
        
        Args:
            gradient_sequence: List of gradient arrays over time
            
        Returns:
            Dictionary containing temporal analysis results
        """
        if not gradient_sequence or len(gradient_sequence) < 2:
            return {
                'is_temporal_evolution': False,
                'confidence': 0.0,
                'trend_type': 'insufficient_data',
                'metrics': {}
            }
        
        # Flatten gradients if needed
        flat_sequence = [g.reshape(-1) for g in gradient_sequence]
        
        # Compute temporal features
        temporal_metrics = self._compute_temporal_metrics(flat_sequence)
        
        # Detect trends and patterns
        trend_analysis = self._analyze_trends(flat_sequence)
        
        # Combine results
        result = {
            'is_temporal_evolution': trend_analysis['has_significant_trend'],
            'confidence': float(trend_analysis['trend_confidence']),
            'trend_type': trend_analysis['trend_type'],
            'metrics': temporal_metrics,
            'trend_analysis': trend_analysis
        }
        
        return result
    
    def _compute_temporal_metrics(self, sequence: List[np.ndarray]) -> Dict[str, float]:
        """Compute various temporal metrics on the gradient sequence."""
        if len(sequence) < 2:
            return {}
        
        # Compute basic statistics over time
        norms = [np.linalg.norm(g) for g in sequence]
        means = [np.mean(np.abs(g)) for g in sequence]
        stds = [np.std(g) for g in sequence]
        
        # Compute changes between consecutive gradients
        deltas = [np.linalg.norm(sequence[i+1] - sequence[i]) 
                 for i in range(len(sequence)-1)]
        
        # Time series features
        norm_ts = np.array(norms)
        mean_ts = np.array(means)
        std_ts = np.array(stds)
        delta_ts = np.array(deltas)
        
        # Basic statistics
        metrics = {
            'mean_norm': float(np.mean(norm_ts)),
            'std_norm': float(np.std(norm_ts)),
            'mean_abs': float(np.mean(mean_ts)),
            'std_abs': float(np.std(mean_ts)),
            'mean_grad_std': float(np.mean(std_ts)),
            'std_grad_std': float(np.std(std_ts)),
            'mean_delta': float(np.mean(delta_ts)) if len(delta_ts) > 0 else 0.0,
            'std_delta': float(np.std(delta_ts)) if len(delta_ts) > 0 else 0.0,
            'max_delta': float(np.max(delta_ts)) if len(delta_ts) > 0 else 0.0,
            'min_delta': float(np.min(delta_ts)) if len(delta_ts) > 0 else 0.0,
        }
        
        # Spectral analysis of time series
        if len(norm_ts) > 4:  # Need enough points for FFT
            try:
                # Normalize
                norm_ts = (norm_ts - np.mean(norm_ts)) / (np.std(norm_ts) + 1e-10)
                
                # Compute power spectrum
                fft_vals = np.abs(fft(norm_ts)[1:len(norm_ts)//2])  # Only positive frequencies
                psd = fft_vals ** 2
                
                # Spectral features
                total_power = np.sum(psd)
                if total_power > 0:
                    psd_norm = psd / total_power
                    p = psd_norm[psd_norm > 0]
                    spectral_entropy_bits = float(-np.sum(p * np.log2(p + 1e-10))) if p.size else 0.0
                    spectral_entropy = (
                        spectral_entropy_bits / np.log2(len(psd_norm))
                        if len(psd_norm) > 1
                        else 0.0
                    )

                    # Frequency bins correspond to k / N (cycles per sample) for k>=1.
                    freqs = fftfreq(len(norm_ts), d=1.0)[1:len(norm_ts)//2]
                    dominant_frequency = float(freqs[int(np.argmax(psd))]) if freqs.size else 0.0
                    metrics.update({
                        # Normalized entropy in [0, 1] plus an unnormalized bits value for analysis.
                        'spectral_entropy': float(spectral_entropy),
                        'spectral_entropy_bits': float(spectral_entropy_bits),
                        'dominant_frequency': float(dominant_frequency),
                        'spectral_flatness': float(np.exp(np.mean(np.log(psd + 1e-10))) / np.mean(psd))
                    })
            except Exception as e:
                warnings.warn(f"Spectral analysis failed: {str(e)}")
        
        # Autocorrelation
        if len(norm_ts) > 5:
            try:
                autocorr = np.correlate(norm_ts, norm_ts, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / (autocorr[0] + 1e-10)
                
                # Find first zero crossing
                zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
                first_zero = zero_crossings[0] if len(zero_crossings) > 0 else len(autocorr)
                
                metrics.update({
                    'autocorr_first_zero': float(first_zero),
                    'autocorr_decay': float(autocorr[1] if len(autocorr) > 1 else 0.0)
                })
            except Exception as e:
                warnings.warn(f"Autocorrelation analysis failed: {str(e)}")
        
        return metrics
    
    def _analyze_trends(self, sequence: List[np.ndarray]) -> Dict:
        """Analyze temporal trends in the gradient sequence."""
        if len(sequence) < 3:
            return {
                'has_significant_trend': False,
                'trend_confidence': 0.0,
                'trend_type': 'insufficient_data',
                'trend_slope': 0.0,
                'trend_pvalue': 1.0
            }
        
        # Compute gradient norms over time
        norms = np.array([np.linalg.norm(g) for g in sequence])
        times = np.arange(len(norms))
        
        # Remove any inf/nan values
        valid_idx = np.isfinite(norms)
        if not np.any(valid_idx):
            return {
                'has_significant_trend': False,
                'trend_confidence': 0.0,
                'trend_type': 'invalid_data',
                'trend_slope': 0.0,
                'trend_pvalue': 1.0
            }
            
        times = times[valid_idx]
        norms = norms[valid_idx]
        
        if len(times) < 2:
            return {
                'has_significant_trend': False,
                'trend_confidence': 0.0,
                'trend_type': 'insufficient_valid_data',
                'trend_slope': 0.0,
                'trend_pvalue': 1.0
            }
        
        # Degenerate case: if the norm time-series is (near) constant, there is no meaningful trend.
        if float(np.std(norms)) < 1e-12:
            return {
                'has_significant_trend': False,
                'trend_confidence': 0.0,
                'trend_type': 'none',
                'trend_slope': 0.0,
                'trend_pvalue': 1.0,
                'linear_r2': 0.0,
            }

        # Normalize time and values
        times_norm = (times - np.min(times)) / (np.max(times) - np.min(times) + 1e-10)
        norms_norm = (norms - np.mean(norms)) / (np.std(norms) + 1e-10)
        
        # Linear regression for trend
        try:
            slope, _intercept, r_value, p_value, _std_err = stats.linregress(times_norm, norms_norm)
        except Exception:
            slope, r_value, p_value = 0.0, 0.0, 1.0
        if not np.isfinite(p_value):
            p_value = 1.0
        if not np.isfinite(r_value):
            r_value = 0.0
        
        # Determine trend type
        trend_type = 'none'
        confidence = 0.0
        
        if p_value < 0.05:  # Statistically significant trend
            if slope > 0:
                trend_type = 'increasing'
                confidence = min(1.0, abs(slope) * 2)  # Scale slope to 0-1 range
            else:
                trend_type = 'decreasing'
                confidence = min(1.0, abs(slope) * 2)
        
        # Check for non-linear patterns
        if len(norms_norm) > 5:
            # Check for U-shaped or inverted U patterns using quadratic fit
            # If variance is ~0, skip quadratic fit (R^2 is undefined / spuriously 1.0).
            if float(np.var(norms_norm)) > 1e-12:
                coeffs = np.polyfit(times_norm, norms_norm, 2)
                quadratic_fit = np.poly1d(coeffs)(times_norm)
            
                # Calculate R-squared for quadratic fit
                ss_res = np.sum((norms_norm - quadratic_fit) ** 2)
                ss_tot = np.sum((norms_norm - np.mean(norms_norm)) ** 2)
                if float(ss_tot) > 1e-12:
                    r2_quad = 1.0 - (ss_res / ss_tot)
                else:
                    r2_quad = 0.0
            
                # If quadratic fit explains more variance than linear
                if r2_quad > float(r_value) ** 2 + 0.1:  # Threshold for preferring quadratic
                    if coeffs[0] > 0:
                        trend_type = 'u_shaped'
                    else:
                        trend_type = 'inverted_u'
                    confidence = min(1.0, float(r2_quad))
        
        # Check for step changes
        if len(norms_norm) > 3:
            # Simple step detection using median absolute deviation
            med = np.median(norms_norm)
            mad = np.median(np.abs(norms_norm - med))
            
            # Count significant steps
            # If MAD is ~0, treat as no-step (avoids spurious detection on constant series).
            if float(mad) < 1e-12:
                steps = np.zeros_like(np.diff(norms_norm), dtype=bool)
            else:
                steps = np.abs(np.diff(norms_norm)) > (3 * mad)
            step_count = np.sum(steps)
            
            if step_count > 0 and step_count / len(steps) > 0.2:  # At least 20% steps
                trend_type = 'step_changes'
                confidence = min(1.0, step_count / len(steps) * 2)
        
        return {
            'has_significant_trend': trend_type != 'none',
            'trend_confidence': float(confidence),
            'trend_type': trend_type,
            'trend_slope': float(slope),
            'trend_pvalue': float(p_value),
            'linear_r2': float(r_value**2)
        }
    
    def detect_progressive_obfuscation(self, sequence: List[np.ndarray]) -> Dict:
        """
        Specifically detect progressive obfuscation patterns.
        
        Progressive obfuscation shows increasing obfuscation strength over time.
        """
        if len(sequence) < 3:
            return {
                'is_progressive_obfuscation': False,
                'confidence': 0.0,
                'metrics': {}
            }
        
        # Compute obfuscation scores over time
        obf_scores = [self._compute_obfuscation_score(g) for g in sequence]
        times = np.arange(len(obf_scores))
        
        # Check for increasing trend
        slope, _, _, p_value, _ = stats.linregress(times, obf_scores)
        
        is_progressive = (slope > 0) and (p_value < 0.05)
        # Confidence is monotone in positive slope; clamp to [0,1] for stability.
        confidence = float(np.clip(float(slope) * 5.0, 0.0, 1.0))
        
        return {
            'is_progressive_obfuscation': bool(is_progressive),
            'confidence': float(confidence),
            'slope': float(slope),
            'p_value': float(p_value),
            'obfuscation_scores': [float(s) for s in obf_scores]
        }
    
    def _compute_obfuscation_score(self, gradient: np.ndarray) -> float:
        """Compute a score indicating the likelihood of obfuscation."""
        if gradient.size == 0:
            return 0.0
            
        # Simple heuristic: higher kurtosis and lower entropy suggest obfuscation
        flat_grad = gradient.reshape(-1)

        # Robustness: ignore non-finite values.
        finite = np.isfinite(flat_grad)
        if not np.any(finite):
            return 0.0
        flat_grad = flat_grad[finite]
        if flat_grad.size == 0:
            return 0.0
        
        # Handle constant gradients
        if np.all(flat_grad == flat_grad[0]):
            return 1.0 if flat_grad[0] != 0 else 0.0
        
        # Normalize
        grad_norm = (flat_grad - np.mean(flat_grad)) / (np.std(flat_grad) + 1e-10)
        
        # Compute metrics
        kurt = float(stats.kurtosis(grad_norm, fisher=False))  # Normal kurtosis (3 for normal)

        # IMPORTANT: entropy must be computed on a proper probability mass function (PMF),
        # not a density histogram. Use counts and normalize to sum=1, then normalize
        # entropy by ln(N_bins) so the score is in [0, 1].
        counts, _ = np.histogram(grad_norm, bins=10)
        p = counts.astype(np.float64)
        p_sum = float(np.sum(p))
        if p_sum <= 0.0 or len(p) < 2:
            entropy_norm = 0.0
        else:
            p = p / (p_sum + 1e-12)
            ent = -float(np.sum(p * np.log(p + 1e-12)))  # nats
            entropy_norm = float(ent / np.log(len(p)))
            entropy_norm = float(np.clip(entropy_norm, 0.0, 1.0))
        
        # Combine into a score (0-1 range)
        kurt_score = float(np.clip((kurt - 1.5) / 5.0, 0.0, 1.0))  # Map kurtosis to [0,1]
        entropy_score = 1.0 - float(entropy_norm)  # Lower entropy => higher obfuscation score
        
        # Weighted combination; clamp to [0,1] for stability.
        return float(np.clip(0.6 * kurt_score + 0.4 * entropy_score, 0.0, 1.0))
