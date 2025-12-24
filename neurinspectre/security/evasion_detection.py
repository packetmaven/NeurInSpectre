"""
Advanced Evasion Detection Module
Implements evasion detection primitives including:
- Neural transport dynamics style indicators
- Network flow watermarking / DeMarking indicators
- Entropy and spectral indicators
- Behavioral pattern indicators
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
import logging
from collections import deque

import torch

try:
    from scipy import stats
    from scipy.signal import welch
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


@dataclass
class EvasionPattern:
    """Represents a detected evasion pattern"""
    pattern_type: str
    confidence: float
    features: Dict[str, float]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehavioralSignature:
    """Represents a behavioral signature for evasion detection"""
    signature_id: str
    pattern_vector: np.ndarray
    confidence_threshold: float
    temporal_consistency: float
    spatial_consistency: float


class NeuralTransportDynamicsDetector:
    """
    Neural Network Transport Dynamics detector for evasion attacks
    Based on neural network transport dynamics research (2024)
    """
    
    def __init__(self, transport_dim: int = 64, time_window: int = 100):
        self.transport_dim = transport_dim
        self.time_window = time_window
        self.dynamics_history = deque(maxlen=time_window)
        self.reference_dynamics = None
        self.transport_threshold = 0.7
        
    def detect_transport_anomalies(self, neural_activations: np.ndarray, 
                                  reference_activations: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect evasion attempts using neural transport dynamics
        
        Args:
            neural_activations: Current neural network activations
            reference_activations: Reference activations for comparison
            
        Returns:
            Transport dynamics analysis results
        """
        
        # Compute transport dynamics
        transport_dynamics = self._compute_transport_dynamics(neural_activations)
        
        # Analyze temporal evolution
        temporal_evolution = self._analyze_temporal_evolution(transport_dynamics)
        
        # Detect phase transitions
        phase_transitions = self._detect_phase_transitions(transport_dynamics)
        
        # Compute flow stability
        flow_stability = self._compute_flow_stability(transport_dynamics)
        
        # Analyze information flow
        information_flow = self._analyze_information_flow(neural_activations)
        
        # Detect critical points
        critical_points = self._detect_critical_points(transport_dynamics)
        
        # Compute overall evasion score
        evasion_score = self._compute_evasion_score(
            temporal_evolution, phase_transitions, flow_stability, 
            information_flow, critical_points
        )
        
        is_evasion = evasion_score > self.transport_threshold
        
        return {
            'is_evasion': is_evasion,
            'evasion_score': evasion_score,
            'transport_dynamics': transport_dynamics,
            'temporal_evolution': temporal_evolution,
            'phase_transitions': phase_transitions,
            'flow_stability': flow_stability,
            'information_flow': information_flow,
            'critical_points': critical_points,
            'threat_level': self._determine_threat_level(evasion_score)
        }
    
    def _compute_transport_dynamics(self, activations: np.ndarray) -> np.ndarray:
        """Compute neural transport dynamics"""
        # Convert to tensor
        activation_tensor = torch.from_numpy(activations).float()
        
        # Compute transport operator
        transport_operator = self._compute_transport_operator(activation_tensor)
        
        # Apply transport dynamics
        dynamics = torch.matmul(transport_operator, activation_tensor.T).T
        
        return dynamics.detach().numpy()
    
    def _compute_transport_operator(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute transport operator for neural dynamics"""
        batch_size, feature_dim = activations.shape
        
        # Compute covariance matrix
        centered_activations = activations - torch.mean(activations, dim=0)
        # Guard small batch sizes to avoid division by zero
        denom = max(1, int(batch_size - 1))
        covariance = torch.matmul(centered_activations.T, centered_activations) / denom
        
        # Compute transport operator (simplified)
        transport_operator = torch.inverse(covariance + 1e-6 * torch.eye(feature_dim))
        
        return transport_operator
    
    def _analyze_temporal_evolution(self, dynamics: np.ndarray) -> float:
        """Analyze temporal evolution of transport dynamics"""
        if len(dynamics) < 2:
            return 0.0
        
        # Compute temporal derivatives
        temporal_derivatives = np.diff(dynamics, axis=0)
        
        # Analyze derivative patterns
        derivative_variance = np.var(temporal_derivatives, axis=0)
        derivative_mean = np.mean(derivative_variance)
        
        # Normalize temporal evolution score
        temporal_score = min(1.0, derivative_mean * 10)
        
        return temporal_score
    
    def _detect_phase_transitions(self, dynamics: np.ndarray) -> List[Dict[str, Any]]:
        """Detect phase transitions in neural dynamics"""
        if len(dynamics) < 3:
            return []
        
        phase_transitions = []
        
        # Compute second derivatives
        second_derivatives = np.diff(dynamics, n=2, axis=0)
        
        # Detect abrupt changes
        for i, deriv in enumerate(second_derivatives):
            deriv_magnitude = np.linalg.norm(deriv)
            
            if deriv_magnitude > np.std(second_derivatives) * 2:
                phase_transitions.append({
                    'location': i,
                    'magnitude': deriv_magnitude,
                    'type': 'abrupt_change'
                })
        
        return phase_transitions
    
    def _compute_flow_stability(self, dynamics: np.ndarray) -> float:
        """Compute flow stability measure"""
        if len(dynamics) < 2:
            # With no temporal evolution we cannot estimate divergence; treat as maximally stable.
            return 1.0
        
        # Compute flow divergence
        flow_divergence = np.trace(np.gradient(dynamics, axis=0))
        
        # Compute stability measure
        # Guard numerical issues
        denom = 1.0 + float(abs(flow_divergence))
        if not np.isfinite(denom) or denom <= 0.0:
            denom = 1.0
        stability = 1.0 / denom
        
        return stability
    
    def _analyze_information_flow(self, activations: np.ndarray) -> float:
        """Analyze information flow through neural network"""
        activations = np.asarray(activations)
        activations = np.nan_to_num(activations, nan=0.0, posinf=0.0, neginf=0.0)
        if activations.ndim != 2 or activations.shape[0] < 2 or activations.shape[1] < 2:
            return 0.0
        
        # Compute mutual information between layers
        mutual_info = 0.0
        
        for i in range(activations.shape[1] - 1):
            with np.errstate(all="ignore"):
                correlation = np.corrcoef(activations[:, i], activations[:, i + 1])[0, 1]
            if not np.isfinite(correlation):
                continue
            # Clamp to avoid log of a negative due to numerical issues.
            rho2 = min(0.999999999, float(correlation) ** 2)
            mutual_info += -0.5 * np.log(1.0 - rho2 + 1e-10)
        
        # Normalize information flow
        denom = max(1, activations.shape[1] - 1)
        info_flow = mutual_info / float(denom)

        # Clamp to [0, 1] for stability.
        return float(np.clip(info_flow, 0.0, 1.0))
    
    def _detect_critical_points(self, dynamics: np.ndarray) -> List[Dict[str, Any]]:
        """Detect critical points in transport dynamics"""
        if len(dynamics) < 2:
            return []
        
        critical_points = []
        
        # Compute gradient magnitude
        gradient_magnitude = np.linalg.norm(np.gradient(dynamics, axis=0), axis=1)
        
        # Find local minima (critical points)
        for i in range(1, len(gradient_magnitude) - 1):
            if (gradient_magnitude[i] < gradient_magnitude[i - 1] and 
                gradient_magnitude[i] < gradient_magnitude[i + 1]):
                critical_points.append({
                    'location': i,
                    'magnitude': gradient_magnitude[i],
                    'type': 'local_minimum'
                })
        
        return critical_points
    
    def _compute_evasion_score(self, temporal_evolution: float, 
                             phase_transitions: List[Dict[str, Any]],
                             flow_stability: float, 
                             information_flow: float,
                             critical_points: List[Dict[str, Any]]) -> float:
        """Compute overall evasion score"""
        
        # Weight different components
        temporal_weight = 0.3
        phase_weight = 0.2
        stability_weight = 0.2
        info_weight = 0.2
        critical_weight = 0.1
        
        # Compute component scores
        phase_score = min(1.0, len(phase_transitions) * 0.2)
        critical_score = min(1.0, len(critical_points) * 0.1)
        
        # Combine scores
        evasion_score = (
            temporal_evolution * temporal_weight +
            phase_score * phase_weight +
            (1.0 - flow_stability) * stability_weight +
            information_flow * info_weight +
            critical_score * critical_weight
        )

        # Scores are interpreted as probabilities; clamp to [0, 1] for stability.
        return float(np.clip(evasion_score, 0.0, 1.0))
    
    def _determine_threat_level(self, evasion_score: float) -> str:
        """Determine threat level based on evasion score"""
        if evasion_score > 0.8:
            return 'critical'
        elif evasion_score > 0.6:
            return 'high'
        elif evasion_score > 0.4:
            return 'medium'
        else:
            return 'low'


class DeMarkingDefenseDetector:
    """
    DeMarking defense detector for network flow watermarking
    Based on "DeMarking: A Defense for Network Flow Watermarking in Real-Time" (Feb 2024)
    """
    
    def __init__(self, window_size: int = 50, threshold: float = 0.6):
        self.window_size = window_size
        self.threshold = threshold
        self.ipd_history = deque(maxlen=window_size)
        
    def detect_watermarking_evasion(self, inter_packet_delays: np.ndarray,
                                   reference_ipds: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect watermarking evasion attempts using DeMarking techniques
        
        Args:
            inter_packet_delays: Inter-packet delays to analyze
            reference_ipds: Reference IPDs for comparison
            
        Returns:
            Watermarking evasion detection results
        """
        
        # Analyze IPD patterns
        ipd_patterns = self._analyze_ipd_patterns(inter_packet_delays)
        
        # Detect GAN-generated IPDs
        gan_detection = self._detect_gan_generated_ipds(inter_packet_delays)
        
        # Analyze temporal correlations
        temporal_correlations = self._analyze_temporal_correlations(inter_packet_delays)
        
        # Detect adversarial perturbations
        adversarial_perturbations = self._detect_adversarial_perturbations(
            inter_packet_delays, reference_ipds
        )
        
        # Compute statistical consistency
        statistical_consistency = self._compute_statistical_consistency(inter_packet_delays)
        
        # Combine scores
        evasion_score = self._compute_watermarking_evasion_score(
            ipd_patterns, gan_detection, temporal_correlations,
            adversarial_perturbations, statistical_consistency
        )
        
        is_evasion = evasion_score > self.threshold
        
        return {
            'is_evasion': is_evasion,
            'evasion_score': evasion_score,
            'ipd_patterns': ipd_patterns,
            'gan_detection': gan_detection,
            'temporal_correlations': temporal_correlations,
            'adversarial_perturbations': adversarial_perturbations,
            'statistical_consistency': statistical_consistency,
            'threat_type': self._classify_threat_type(evasion_score, gan_detection)
        }
    
    def _analyze_ipd_patterns(self, ipds: np.ndarray) -> Dict[str, float]:
        """Analyze Inter-Packet Delay patterns"""
        if len(ipds) < 2:
            return {'entropy': 0.0, 'regularity': 0.0, 'variance': 0.0}
        
        # Compute *discrete* entropy on a proper probability mass function (PMF),
        # then normalize by ln(N_bins) so entropy ∈ [0, 1].
        counts, _ = np.histogram(ipds, bins=50)
        p = counts.astype(np.float64)
        p_sum = float(np.sum(p))
        if p_sum <= 0.0 or len(p) < 2:
            normalized_entropy = 0.0
        else:
            p = p / (p_sum + 1e-12)
            ent = -float(np.sum(p * np.log(p + 1e-12)))  # nats
            normalized_entropy = float(ent / np.log(len(p)))
            normalized_entropy = float(np.clip(normalized_entropy, 0.0, 1.0))
        
        # Compute regularity from normalized autocorrelation (lag-0 normalized).
        # Use mean-centered signal to avoid bias from a non-zero mean.
        x = ipds.astype(np.float64) - float(np.mean(ipds))
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[autocorr.size // 2 :]  # non-negative lags
        if len(autocorr) < 2 or not np.isfinite(autocorr[0]) or abs(autocorr[0]) < 1e-12:
            regularity = 0.0
        else:
            ac_norm = autocorr / (autocorr[0] + 1e-12)
            regularity = float(np.max(np.abs(ac_norm[1:])))
            regularity = float(np.clip(regularity, 0.0, 1.0))
        
        # Compute variance (cast to Python float for JSON safety)
        variance = float(np.var(ipds))
        
        return {
            'entropy': normalized_entropy,
            'regularity': regularity,
            'variance': variance
        }
    
    def _detect_gan_generated_ipds(self, ipds: np.ndarray) -> Dict[str, Any]:
        """Detect GAN-generated IPDs using statistical analysis"""
        if len(ipds) < 10:
            return {'is_gan': False, 'confidence': 0.0}
        
        # Analyze distribution characteristics
        distribution_analysis = self._analyze_distribution_characteristics(ipds)
        
        # Detect mode collapse
        mode_collapse = self._detect_mode_collapse(ipds)
        
        # Analyze spectral properties
        spectral_analysis = self._analyze_spectral_properties(ipds)
        
        # Compute GAN detection score (cast to Python floats for stability/JSON safety)
        gan_score = float(
            float(distribution_analysis.get('anomaly_score', 0.0)) * 0.4 +
            float(mode_collapse) * 0.3 +
            float(spectral_analysis.get('spectral_anomaly', 0.0)) * 0.3
        )
        
        is_gan = bool(gan_score > 0.7)
        
        return {
            'is_gan': is_gan,
            'confidence': float(gan_score),
            'distribution_analysis': distribution_analysis,
            'mode_collapse': float(mode_collapse),
            'spectral_analysis': spectral_analysis
        }
    
    def _analyze_distribution_characteristics(self, ipds: np.ndarray) -> Dict[str, float]:
        """Analyze distribution characteristics of IPDs"""
        # IMPORTANT: avoid SciPy skew/kurtosis warnings on near-constant data.
        # We compute stable standardized moments directly, and use the Jarque–Bera
        # statistic (chi-square with 2 dof) for a normality proxy:
        #   JB = n/6 * (S^2 + (K^2)/4),  p = exp(-JB/2)  (df=2)
        x = np.asarray(ipds, dtype=np.float64).reshape(-1)
        x = x[np.isfinite(x)]
        n = int(x.size)
        if n < 8:
            return {
                'anomaly_score': 0.5,
                'kurtosis': 0.0,
                'skewness': 0.0,
                'jb_stat': 0.0,
                'jb_p_value': 1.0,
            }

        mu = float(np.mean(x))
        sigma = float(np.std(x))
        if (not np.isfinite(sigma)) or sigma < 1e-12:
            # Degenerate / near-constant timing is highly suspicious but moments are undefined.
            return {
                'anomaly_score': 1.0,
                'kurtosis': 0.0,
                'skewness': 0.0,
                'jb_stat': float('inf'),
                'jb_p_value': 0.0,
            }

        z = (x - mu) / (sigma + 1e-12)
        skewness = float(np.mean(z ** 3))
        kurtosis = float(np.mean(z ** 4) - 3.0)  # excess kurtosis

        jb = float((n / 6.0) * (skewness ** 2 + (kurtosis ** 2) / 4.0))
        # For chi-square(2), survival function is exp(-x/2).
        jb_p = float(np.exp(-0.5 * jb)) if np.isfinite(jb) else 0.0
        jb_p = float(np.clip(jb_p, 0.0, 1.0))
        normality_deviation = 1.0 - jb_p

        anomaly_score = (abs(kurtosis) + abs(skewness) + float(normality_deviation)) / 3.0
        anomaly_score = float(min(1.0, anomaly_score))

        return {
            'anomaly_score': anomaly_score,
            'kurtosis': float(kurtosis),
            'skewness': float(skewness),
            'jb_stat': float(jb),
            'jb_p_value': float(jb_p),
        }
    
    def _detect_mode_collapse(self, ipds: np.ndarray) -> float:
        """Detect mode collapse in IPD distribution"""
        if len(ipds) < 5:
            return 0.0
        
        # Compute histogram
        hist, bin_edges = np.histogram(ipds, bins=20)
        
        # Detect excessive concentration in few bins
        non_zero_bins = int(np.sum(hist > 0))
        concentration_ratio = float(non_zero_bins) / float(len(hist) or 1)
        
        # Mode collapse score (lower concentration = higher score)
        mode_collapse_score = float(1.0 - concentration_ratio)
        
        return float(np.clip(mode_collapse_score, 0.0, 1.0))
    
    def _analyze_spectral_properties(self, ipds: np.ndarray) -> Dict[str, float]:
        """Analyze spectral properties of IPDs"""
        if not HAS_SCIPY or len(ipds) < 8:
            return {'spectral_anomaly': 0.0, 'dominant_freq': 0.0}
        
        # Compute power spectral density
        frequencies, psd = welch(ipds, nperseg=min(len(ipds), 256))
        
        # Analyze spectral characteristics
        dominant_freq = float(frequencies[np.argmax(psd)]) if len(psd) else 0.0

        # Spectral entropy MUST be computed on a normalized distribution.
        psd_sum = float(np.sum(psd))
        if psd_sum <= 0.0 or len(psd) < 2:
            spectral_anomaly = 0.0
        else:
            p = psd / (psd_sum + 1e-10)
            spectral_entropy = -float(np.sum(p * np.log(p + 1e-10)))
            # Normalize by maximum entropy (ln N)
            spectral_anomaly = float(min(1.0, spectral_entropy / np.log(len(psd))))
        
        return {
            'spectral_anomaly': spectral_anomaly,
            'dominant_freq': dominant_freq
        }
    
    def _analyze_temporal_correlations(self, ipds: np.ndarray) -> Dict[str, float]:
        """Analyze temporal correlations in IPDs"""
        if len(ipds) < 3:
            return {'correlation_strength': 0.0, 'lag_correlation': 0.0}
        
        # Compute autocorrelation
        autocorr = np.correlate(ipds, ipds, mode='full')
        denom = float(np.max(autocorr)) if autocorr.size else 0.0
        if (not np.isfinite(denom)) or abs(denom) < 1e-12:
            autocorr_normalized = np.zeros_like(autocorr, dtype=np.float64)
        else:
            autocorr_normalized = autocorr.astype(np.float64) / denom
        
        # Find peak correlations
        peak_indices = np.where(autocorr_normalized > 0.5)[0]
        correlation_strength = float(len(peak_indices)) / float(len(autocorr_normalized) or 1)
        
        # Compute lag correlation
        if len(ipds) > 1:
            with np.errstate(all="ignore"):
                lag_correlation = np.corrcoef(ipds[:-1], ipds[1:])[0, 1]
            if not np.isfinite(lag_correlation):
                lag_correlation = 0.0
        else:
            lag_correlation = 0.0
        
        return {
            'correlation_strength': float(correlation_strength),
            'lag_correlation': float(abs(lag_correlation))
        }
    
    def _detect_adversarial_perturbations(self, ipds: np.ndarray,
                                        reference_ipds: Optional[np.ndarray]) -> Dict[str, float]:
        """Detect adversarial perturbations in IPDs"""
        if reference_ipds is None:
            return {'perturbation_score': 0.0, 'statistical_distance': 0.0}
        
        # Compute statistical distance
        if HAS_SCIPY:
            ks_statistic, _ = stats.ks_2samp(ipds, reference_ipds)
            statistical_distance = ks_statistic
        else:
            # Fallback: compute basic distance
            statistical_distance = abs(np.mean(ipds) - np.mean(reference_ipds))
        
        # Analyze perturbation patterns
        perturbation_score = float(min(1.0, float(statistical_distance) * 5.0))
        
        return {
            'perturbation_score': float(perturbation_score),
            'statistical_distance': float(statistical_distance)
        }
    
    def _compute_statistical_consistency(self, ipds: np.ndarray) -> float:
        """Compute statistical consistency of IPDs"""
        if len(ipds) < 10:
            # Not enough samples to compare chunks; assume consistent (do not add risk).
            return 1.0
        
        # Split into chunks and compare
        chunk_size = len(ipds) // 4
        chunks = [ipds[i:i+chunk_size] for i in range(0, len(ipds), chunk_size)]
        
        # Compute consistency across chunks
        chunk_means = [np.mean(chunk) for chunk in chunks if len(chunk) > 0]
        chunk_stds = [np.std(chunk) for chunk in chunks if len(chunk) > 0]
        
        if len(chunk_means) < 2:
            return 1.0
        
        # Consistency score (higher = more consistent). Clamp to [0, 1] for stability.
        eps = 1e-12
        mean_consistency = 1.0 - float(np.std(chunk_means)) / (float(np.mean(np.abs(chunk_means))) + eps)
        std_consistency = 1.0 - float(np.std(chunk_stds)) / (float(np.mean(np.abs(chunk_stds))) + eps)
        consistency = 0.5 * (mean_consistency + std_consistency)
        return float(np.clip(consistency, 0.0, 1.0))
    
    def _compute_watermarking_evasion_score(self, ipd_patterns: Dict[str, float],
                                          gan_detection: Dict[str, Any],
                                          temporal_correlations: Dict[str, float],
                                          adversarial_perturbations: Dict[str, float],
                                          statistical_consistency: float) -> float:
        """Compute overall watermarking evasion score"""
        
        # Weight different components
        weights = {
            'ipd_patterns': 0.3,
            'gan_detection': 0.3,
            'temporal_correlations': 0.2,
            'adversarial_perturbations': 0.15,
            'statistical_consistency': 0.05
        }
        
        # Compute component scores
        ipd_score = (ipd_patterns['entropy'] + ipd_patterns['regularity']) / 2.0
        gan_score = gan_detection['confidence']
        temporal_score = (temporal_correlations['correlation_strength'] + 
                         temporal_correlations['lag_correlation']) / 2.0
        perturbation_score = adversarial_perturbations['perturbation_score']
        consistency_score = 1.0 - statistical_consistency
        
        # Combine scores
        evasion_score = (
            ipd_score * weights['ipd_patterns'] +
            gan_score * weights['gan_detection'] +
            temporal_score * weights['temporal_correlations'] +
            perturbation_score * weights['adversarial_perturbations'] +
            consistency_score * weights['statistical_consistency']
        )

        # Scores are interpreted as probabilities; clamp to [0, 1] for stability.
        return float(np.clip(evasion_score, 0.0, 1.0))
    
    def _classify_threat_type(self, evasion_score: float, gan_detection: Dict[str, Any]) -> str:
        """Classify the type of threat"""
        if evasion_score > 0.8:
            if gan_detection['is_gan']:
                return 'sophisticated_gan_evasion'
            else:
                return 'advanced_statistical_evasion'
        elif evasion_score > 0.6:
            return 'moderate_evasion'
        else:
            return 'low_risk'


class BehavioralPatternAnalyzer:
    """
    Behavioral pattern analyzer for evasion detection
    """
    
    def __init__(self, pattern_window: int = 100):
        self.pattern_window = pattern_window
        self.behavioral_signatures = {}
        self.pattern_history = deque(maxlen=pattern_window)
        
    def analyze_behavioral_patterns(self, activation_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze behavioral patterns in activation data
        
        Args:
            activation_data: Neural network activation data
            
        Returns:
            Behavioral pattern analysis results
        """
        
        # Extract behavioral features
        behavioral_features = self._extract_behavioral_features(activation_data)
        
        # Detect anomalous behaviors
        anomalous_behaviors = self._detect_anomalous_behaviors(behavioral_features)
        
        # Analyze pattern consistency
        pattern_consistency = self._analyze_pattern_consistency(behavioral_features)
        
        # Detect evasion signatures
        evasion_signatures = self._detect_evasion_signatures(behavioral_features)
        
        # Compute behavioral stability
        behavioral_stability = self._compute_behavioral_stability(activation_data)
        
        # Overall behavioral anomaly score
        anomaly_score = self._compute_behavioral_anomaly_score(
            anomalous_behaviors, pattern_consistency, evasion_signatures, behavioral_stability
        )
        
        return {
            'behavioral_features': behavioral_features,
            'anomalous_behaviors': anomalous_behaviors,
            'pattern_consistency': pattern_consistency,
            'evasion_signatures': evasion_signatures,
            'behavioral_stability': behavioral_stability,
            'anomaly_score': anomaly_score,
            'is_anomalous': anomaly_score > 0.6
        }
    
    def _extract_behavioral_features(self, activation_data: np.ndarray) -> Dict[str, float]:
        """Extract behavioral features from activation data"""
        features = {}
        
        # Basic statistical features
        features['mean_activation'] = np.mean(activation_data)
        features['std_activation'] = np.std(activation_data)
        features['skewness'] = self._compute_skewness(activation_data.flatten())
        features['kurtosis'] = self._compute_kurtosis(activation_data.flatten())
        
        # Spatial features
        if len(activation_data.shape) > 1:
            features['spatial_variance'] = np.var(np.mean(activation_data, axis=0))
            # Correlation across feature dimensions (rowvar=False). Use mean absolute off-diagonal correlation.
            try:
                with np.errstate(all="ignore"):
                    corr = np.corrcoef(activation_data, rowvar=False)
                if corr.ndim == 2 and corr.shape[0] > 1:
                    mask = ~np.eye(corr.shape[0], dtype=bool)
                    spatial_corr = float(np.mean(np.abs(corr[mask])))
                else:
                    spatial_corr = 0.0
            except Exception:
                spatial_corr = 0.0
            if not np.isfinite(spatial_corr):
                spatial_corr = 0.0
            features['spatial_correlation'] = float(np.clip(spatial_corr, 0.0, 1.0))
        
        # Temporal features
        if len(activation_data) > 1:
            features['temporal_smoothness'] = self._compute_temporal_smoothness(activation_data)
            features['temporal_complexity'] = self._compute_temporal_complexity(activation_data)
        
        # Entropy features
        features['entropy'] = self._compute_entropy(activation_data)
        
        return features
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _compute_temporal_smoothness(self, data: np.ndarray) -> float:
        """Compute temporal smoothness"""
        if len(data) < 2:
            return 0.0
        
        derivatives = np.diff(data, axis=0)
        var_mean = float(np.mean(np.var(derivatives, axis=0))) if derivatives.size else 0.0
        denom = 1.0 + max(0.0, var_mean)
        smoothness = 1.0 / denom
        
        return smoothness
    
    def _compute_temporal_complexity(self, data: np.ndarray) -> float:
        """Compute temporal complexity"""
        if len(data) < 3:
            return 0.0
        
        # Compute second derivatives
        second_derivatives = np.diff(data, n=2, axis=0)
        complexity = np.mean(np.abs(second_derivatives))
        
        return min(1.0, complexity)
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute entropy of data"""
        counts, _ = np.histogram(data.flatten(), bins=50)
        p = counts.astype(np.float64)
        p_sum = float(np.sum(p))
        if p_sum <= 0.0 or len(p) < 2:
            return 0.0
        p = p / (p_sum + 1e-12)
        ent = -float(np.sum(p * np.log(p + 1e-12)))  # nats
        ent_norm = ent / float(np.log(len(p)))
        return float(np.clip(ent_norm, 0.0, 1.0))
    
    def _detect_anomalous_behaviors(self, features: Dict[str, float]) -> List[str]:
        """Detect anomalous behaviors based on features"""
        anomalies = []
        
        # Define thresholds for anomaly detection
        thresholds = {
            'high_variance': 0.8,
            'low_entropy': 0.2,
            'high_skewness': 2.0,
            'high_kurtosis': 3.0,
            'low_smoothness': 0.3
        }
        
        # Check for anomalies
        if features.get('std_activation', 0) > thresholds['high_variance']:
            anomalies.append('high_variance')
        
        if features.get('entropy', 1) < thresholds['low_entropy']:
            anomalies.append('low_entropy')
        
        if abs(features.get('skewness', 0)) > thresholds['high_skewness']:
            anomalies.append('high_skewness')
        
        if abs(features.get('kurtosis', 0)) > thresholds['high_kurtosis']:
            anomalies.append('high_kurtosis')
        
        if features.get('temporal_smoothness', 1) < thresholds['low_smoothness']:
            anomalies.append('low_smoothness')
        
        return anomalies
    
    def _analyze_pattern_consistency(self, features: Dict[str, float]) -> float:
        """Analyze pattern consistency"""
        # Store features in history
        self.pattern_history.append(features)
        
        if len(self.pattern_history) < 2:
            # Not enough history to compare; assume consistent (do not add risk).
            return 1.0
        
        # Compute consistency across recent patterns
        recent_patterns = list(self.pattern_history)[-10:]  # Last 10 patterns
        
        consistency_scores = []
        for feature_name in features.keys():
            if feature_name in recent_patterns[0]:
                feature_values = [p[feature_name] for p in recent_patterns]
                denom = float(np.mean(np.abs(feature_values))) + 1e-12
                consistency = 1.0 - float(np.std(feature_values)) / denom
                consistency_scores.append(float(np.clip(consistency, 0.0, 1.0)))
        
        if not consistency_scores:
            return 1.0
        return float(np.clip(float(np.mean(consistency_scores)), 0.0, 1.0))
    
    def _detect_evasion_signatures(self, features: Dict[str, float]) -> List[str]:
        """Detect evasion signatures in behavioral patterns"""
        signatures = []
        
        # Define evasion signature patterns
        if (features.get('entropy', 0) < 0.3 and 
            features.get('std_activation', 0) > 0.7):
            signatures.append('entropy_manipulation')
        
        if (features.get('temporal_smoothness', 1) < 0.2 and 
            features.get('temporal_complexity', 0) > 0.8):
            signatures.append('temporal_disruption')
        
        if (features.get('spatial_variance', 0) > 0.8 and 
            features.get('spatial_correlation', 0) < 0.3):
            signatures.append('spatial_disruption')
        
        return signatures
    
    def _compute_behavioral_stability(self, activation_data: np.ndarray) -> float:
        """Compute behavioral stability"""
        if len(activation_data) < 2:
            # No temporal variation available; treat as stable (do not add risk).
            return 1.0
        
        # Compute stability as inverse of variation
        mean_activation = np.mean(activation_data, axis=0)
        variations = np.var(activation_data - mean_activation, axis=0)
        var_mean = float(np.mean(variations)) if len(variations) else 0.0
        denom = 1.0 + max(0.0, var_mean)
        stability = 1.0 / denom

        return float(np.clip(stability, 0.0, 1.0))
    
    def _compute_behavioral_anomaly_score(self, anomalous_behaviors: List[str],
                                        pattern_consistency: float,
                                        evasion_signatures: List[str],
                                        behavioral_stability: float) -> float:
        """Compute overall behavioral anomaly score"""
        
        # Weight different components
        anomaly_weight = 0.3
        consistency_weight = 0.2
        signature_weight = 0.3
        stability_weight = 0.2
        
        # Compute component scores
        anomaly_score = min(1.0, len(anomalous_behaviors) * 0.2)
        consistency_score = float(np.clip(1.0 - float(pattern_consistency), 0.0, 1.0))
        signature_score = min(1.0, len(evasion_signatures) * 0.3)
        stability_score = float(np.clip(1.0 - float(behavioral_stability), 0.0, 1.0))
        
        # Combine scores
        overall_score = (
            anomaly_score * anomaly_weight +
            consistency_score * consistency_weight +
            signature_score * signature_weight +
            stability_score * stability_weight
        )

        return float(np.clip(overall_score, 0.0, 1.0))


class EvasionDetector:
    """
    Main evasion detection system integrating all detection mechanisms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize specialized detectors
        self.transport_detector = NeuralTransportDynamicsDetector(
            transport_dim=self.config.get('transport_dim', 64),
            time_window=self.config.get('time_window', 100)
        )
        
        self.demarking_detector = DeMarkingDefenseDetector(
            window_size=self.config.get('demarking_window', 50),
            threshold=self.config.get('demarking_threshold', 0.6)
        )
        
        self.behavioral_analyzer = BehavioralPatternAnalyzer(
            pattern_window=self.config.get('pattern_window', 100)
        )
        
        self.detection_history = []
        
    def detect_evasion_attempts(self, activation_data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect evasion attempts in activation data
        
        Args:
            activation_data: Neural network activation data
            
        Returns:
            List of detected evasion attempts
        """
        
        # Preserve optional side-channel data (e.g., IPD series) while sanitizing activations.
        ipd_data = getattr(activation_data, 'ipd_data', None)
        activation_arr = np.asarray(activation_data)
        activation_arr = np.nan_to_num(activation_arr, nan=0.0, posinf=0.0, neginf=0.0)

        evasion_attempts = []

        # Degenerate activation inputs can trigger mean/std warnings (treated as errors in strict mode).
        # For empty/near-empty activations, skip activation-based detectors but still allow
        # watermarking (DeMarking) analysis if IPD side-channel data is provided.
        if activation_arr.size == 0 or (activation_arr.ndim >= 1 and activation_arr.shape[0] < 2):
            if ipd_data is not None:
                try:
                    ipd_arr = np.asarray(ipd_data)
                    ipd_arr = np.nan_to_num(ipd_arr, nan=0.0, posinf=0.0, neginf=0.0)
                    demarking_result = self.demarking_detector.detect_watermarking_evasion(ipd_arr)
                    if demarking_result.get('is_evasion'):
                        score = float(demarking_result.get('evasion_score', 0.0))
                        evasion_attempts.append({
                            'type': 'watermarking_evasion',
                            'confidence': score,
                            # Use the common severity scale across NeurInSpectre summaries.
                            'threat_level': self._determine_threat_level(score),
                            'details': demarking_result,
                        })
                except Exception as e:
                    logger.error(f"Error in evasion detection (watermarking only): {e}")
                    evasion_attempts.append({
                        'type': 'detection_error',
                        'confidence': 0.0,
                        'threat_level': 'unknown',
                        'error': str(e),
                    })

            self.detection_history.append({
                'timestamp': time.time(),
                'evasion_attempts': evasion_attempts,
                'input_shape': activation_arr.shape,
            })
            return evasion_attempts
        
        try:
            # Transport dynamics detection
            transport_result = self.transport_detector.detect_transport_anomalies(activation_arr)
            if transport_result['is_evasion']:
                evasion_attempts.append({
                    'type': 'transport_dynamics',
                    'confidence': transport_result['evasion_score'],
                    'threat_level': transport_result['threat_level'],
                    'details': transport_result
                })
            
            # Behavioral pattern analysis
            behavioral_result = self.behavioral_analyzer.analyze_behavioral_patterns(activation_arr)
            if behavioral_result['is_anomalous']:
                evasion_attempts.append({
                    'type': 'behavioral_anomaly',
                    'confidence': behavioral_result['anomaly_score'],
                    'threat_level': self._determine_threat_level(behavioral_result['anomaly_score']),
                    'details': behavioral_result
                })
            
            # DeMarking detection (if IPD data available)
            if ipd_data is not None:
                ipd_arr = np.asarray(ipd_data)
                ipd_arr = np.nan_to_num(ipd_arr, nan=0.0, posinf=0.0, neginf=0.0)
                demarking_result = self.demarking_detector.detect_watermarking_evasion(
                    ipd_arr
                )
                if demarking_result['is_evasion']:
                    score = float(demarking_result.get('evasion_score', 0.0))
                    evasion_attempts.append({
                        'type': 'watermarking_evasion',
                        'confidence': score,
                        'threat_level': self._determine_threat_level(score),
                        'details': demarking_result
                    })
            
        except Exception as e:
            logger.error(f"Error in evasion detection: {e}")
            evasion_attempts.append({
                'type': 'detection_error',
                'confidence': 0.0,
                'threat_level': 'unknown',
                'error': str(e)
            })
        
        # Store in history
        self.detection_history.append({
            'timestamp': time.time(),
            'evasion_attempts': evasion_attempts,
            'input_shape': activation_arr.shape
        })
        
        return evasion_attempts
    
    def _determine_threat_level(self, score: float) -> str:
        """Determine threat level based on score"""
        if score > 0.8:
            return 'critical'
        elif score > 0.6:
            return 'high'
        elif score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of all detections"""
        if not self.detection_history:
            return {'total_detections': 0, 'evasion_attempts': 0}
        
        total_detections = len(self.detection_history)
        total_evasion_attempts = sum(len(d['evasion_attempts']) for d in self.detection_history)
        
        # Analyze threat levels
        threat_levels = []
        for detection in self.detection_history:
            for attempt in detection['evasion_attempts']:
                threat_levels.append(attempt['threat_level'])
        
        threat_distribution = {
            level: threat_levels.count(level) 
            for level in ['low', 'medium', 'high', 'critical']
        }
        
        return {
            'total_detections': total_detections,
            'evasion_attempts': total_evasion_attempts,
            'threat_distribution': threat_distribution,
            'latest_detection': self.detection_history[-1] if self.detection_history else None
        }


# Helper functions for compatibility with test suite
def analyze_behavioral_patterns(activation_data: np.ndarray) -> Dict[str, Any]:
    """
    Analyze behavioral patterns using the behavioral analyzer
    
    Args:
        activation_data: Neural network activation data
        
    Returns:
        Behavioral pattern analysis results
    """
    analyzer = BehavioralPatternAnalyzer()
    return analyzer.analyze_behavioral_patterns(activation_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
    print("🛡️ Advanced Evasion Detection System")
    print("=" * 50)
    
    # Create detector
    detector = EvasionDetector()
    
    # Test with sample data
    test_data = np.random.randn(100, 64)
    
    # Run detection
    results = detector.detect_evasion_attempts(test_data)
    
    print(f"Detected {len(results)} evasion attempts")
    
    # Get summary
    summary = detector.get_detection_summary()
    print(f"Detection summary: {summary}") 