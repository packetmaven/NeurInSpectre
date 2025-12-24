"""
Advanced Adversarial Detection Module
Implements adversarial attack detection primitives including:
- Gradient inversion signals (TS-Inverse-style indicators)
- Model inversion signals (ConcreTizer-style indicators)
- Transformer behavior signals (AttentionGuard-style indicators)
- Embedding attack signals (EDNN-style indicators)
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import logging

import torch
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


@dataclass
class AttackSignature:
    """Represents an adversarial attack signature"""
    attack_type: str
    confidence: float
    features: Dict[str, float]
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class GradientInversionResult:
    """Results from gradient inversion attack detection"""
    is_inverted: bool
    inversion_confidence: float
    reconstructed_features: Optional[np.ndarray]
    privacy_leakage_score: float
    attack_complexity: str  # 'low', 'medium', 'high'


class TSInverseDetector:
    """
    TS-Inverse gradient inversion attack detector

    IMPORTANT:
    - This is a **heuristic signal detector** for gradient inversion / reconstruction risk on
      sequential or time-series-like updates.
    - It is **not** a faithful implementation of a single named paper.
    - Scores are **risk proxies** derived from statistics + spectrum indicators and should be
      interpreted as "signals consistent with inversion-style behavior", not attribution.
    """
    
    def __init__(self, sensitivity_threshold: float = 0.8):
        self.sensitivity_threshold = sensitivity_threshold
        self.gradient_history = []
        self.attack_patterns = {}
        
    def detect_gradient_inversion(self, gradients: np.ndarray, 
                                 reference_gradients: Optional[np.ndarray] = None) -> GradientInversionResult:
        """
        Detect TS-Inverse gradient inversion attacks
        
        Args:
            gradients: Input gradients to analyze
            reference_gradients: Clean reference gradients for comparison
            
        Returns:
            GradientInversionResult with detection results
        """
        
        # Analyze gradient magnitude patterns
        gradient_magnitude = np.linalg.norm(gradients, axis=-1)
        
        # Detect unusual gradient patterns characteristic of inversion attacks
        magnitude_variance = np.var(gradient_magnitude)
        magnitude_skewness = self._compute_skewness(gradient_magnitude)

        # Squash unbounded moments into [0, 1) so score thresholds are meaningful.
        var_score = float(magnitude_variance / (magnitude_variance + 1.0))
        skew_score = float(abs(magnitude_skewness) / (abs(magnitude_skewness) + 1.0))
        
        # Frequency domain analysis for temporal patterns
        try:
            with torch.no_grad():
                grad_tensor = torch.from_numpy(gradients).float()
                fft_result = torch.fft.fft(grad_tensor, dim=0)
                spectral_density = torch.abs(fft_result).cpu().numpy()
            spectral_entropy = self._compute_spectral_entropy(spectral_density)
        except Exception:
            spectral_entropy = 0.0
        
        # Compute privacy leakage score
        privacy_leakage = self._compute_privacy_leakage(gradients, reference_gradients)
        
        # Determine attack complexity (all components are ~[0,1])
        complexity_score = (
            0.35 * var_score
            + 0.25 * skew_score
            + 0.20 * spectral_entropy
            + 0.20 * privacy_leakage
        )
        
        if complexity_score > 0.8:
            attack_complexity = 'high'
        elif complexity_score > 0.5:
            attack_complexity = 'medium'
        else:
            attack_complexity = 'low'
        
        # Overall inversion confidence (normalized to [0,1])
        inversion_confidence = float(min(1.0, (var_score + skew_score + spectral_entropy + privacy_leakage) / 4.0))
        is_inverted = inversion_confidence > self.sensitivity_threshold
        
        # Attempt gradient reconstruction if attack detected
        reconstructed_features = None
        if is_inverted:
            reconstructed_features = self._reconstruct_from_gradients(gradients)
        
        return GradientInversionResult(
            is_inverted=is_inverted,
            inversion_confidence=inversion_confidence,
            reconstructed_features=reconstructed_features,
            privacy_leakage_score=privacy_leakage,
            attack_complexity=attack_complexity
        )
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_spectral_entropy(self, spectral_density: np.ndarray) -> float:
        """Compute normalized spectral entropy in [0, 1].

        Notes:
        - We treat the (magnitude) spectrum as a discrete distribution after normalization.
        - Entropy is normalized by the maximum possible entropy log(N_bins).
        """
        x = np.asarray(spectral_density, dtype=np.float64).reshape(-1)
        s = float(np.sum(x))
        if s <= 0.0 or x.size < 2:
            return 0.0

        p = x / (s + 1e-10)
        p = p[p > 0]
        if p.size < 2:
            return 0.0

        ent = -float(np.sum(p * np.log(p + 1e-10)))
        return float(ent / np.log(float(x.size)))
    
    def _compute_privacy_leakage(self, gradients: np.ndarray, 
                               reference_gradients: Optional[np.ndarray]) -> float:
        """Compute privacy leakage score"""
        if reference_gradients is None:
            # Use gradient magnitude distribution as proxy
            return min(1.0, np.var(np.linalg.norm(gradients, axis=-1)) * 10)

        # If shapes mismatch, fall back to a stable proxy (avoid correlation shape errors).
        if np.size(gradients) != np.size(reference_gradients):
            return min(1.0, np.var(np.linalg.norm(gradients, axis=-1)) * 10)
        
        # Compute correlation between gradients and reference (guard degenerate sizes to avoid numpy warnings)
        try:
            if np.size(gradients) < 2 or np.size(reference_gradients) < 2:
                return 0.0
            with np.errstate(all="ignore"):
                correlation = float(np.corrcoef(gradients.flatten(), reference_gradients.flatten())[0, 1])
            if not np.isfinite(correlation):
                return 0.0
            return max(0.0, 1.0 - abs(correlation))
        except Exception:
            return 0.0
    
    def _reconstruct_from_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Attempt to reconstruct original features from gradients"""
        # Simplified reconstruction using gradient descent
        reconstructed = torch.zeros_like(torch.from_numpy(gradients))
        reconstructed.requires_grad_(True)
        
        optimizer = torch.optim.Adam([reconstructed], lr=0.01)
        
        for _ in range(10):  # Limited iterations for demo
            optimizer.zero_grad()
            loss = F.mse_loss(reconstructed, torch.from_numpy(gradients))
            loss.backward()
            optimizer.step()
        
        return reconstructed.detach().numpy()


class ConcreTizerDetector:
    """
    ConcreTizer model inversion attack detector

    IMPORTANT:
    - This is a **heuristic signal detector** for model inversion / extraction-style behavior
      over numeric model outputs / activations.
    - It is **not** a faithful implementation of a single named paper.
    """
    
    def __init__(self, voxel_resolution: int = 32, inversion_threshold: float = 0.7):
        self.voxel_resolution = voxel_resolution
        self.inversion_threshold = float(inversion_threshold)
        
    def detect_model_inversion(self, model_outputs: np.ndarray, 
                             query_patterns: np.ndarray) -> Dict[str, Any]:
        """
        Detect ConcreTizer-style model inversion attacks
        
        Args:
            model_outputs: Model output activations
            query_patterns: Query patterns used to extract information
            
        Returns:
            Detection results with confidence scores
        """
        
        # Analyze voxel occupancy patterns
        voxel_occupancy = self._analyze_voxel_occupancy(model_outputs)
        
        # Detect systematic querying patterns
        query_systematicity = self._detect_systematic_queries(query_patterns)
        
        # Compute reconstruction confidence
        reconstruction_confidence = self._compute_reconstruction_confidence(
            model_outputs, voxel_occupancy
        )
        
        # Information leakage analysis
        information_leakage = self._analyze_information_leakage(model_outputs)
        
        # Overall inversion score
        inversion_score = (
            voxel_occupancy * 0.3 +
            query_systematicity * 0.3 +
            reconstruction_confidence * 0.2 +
            information_leakage * 0.2
        )
        
        inversion_score = float(max(0.0, min(1.0, float(inversion_score))))
        is_inversion_attack = bool(inversion_score > float(self.inversion_threshold))
        
        return {
            'is_inversion_attack': is_inversion_attack,
            'inversion_score': float(inversion_score),
            'voxel_occupancy': float(voxel_occupancy),
            'query_systematicity': float(query_systematicity),
            'reconstruction_confidence': float(reconstruction_confidence),
            'information_leakage': float(information_leakage),
            'attack_complexity': self._determine_attack_complexity(inversion_score)
        }
    
    def _analyze_voxel_occupancy(self, model_outputs: np.ndarray) -> float:
        """Analyze voxel occupancy patterns characteristic of 3D reconstruction"""
        x = np.asarray(model_outputs)

        # Compute correlation structure across output dimensions (when available).
        corr_score = 0.0
        # Need at least 2 observations for correlation to be well-defined; otherwise numpy emits warnings.
        if x.ndim >= 2 and x.shape[0] >= 2 and x.shape[1] >= 2:
            with np.errstate(all="ignore"):
                corr = np.corrcoef(x, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            off = corr[np.triu_indices(corr.shape[0], k=1)]
            corr_score = float(np.mean(np.abs(off))) if off.size else 0.0

        # Look for grid-like patterns
        grid_score = self._detect_grid_patterns(x)

        return float(min(1.0, (corr_score + grid_score) / 2.0))
    
    def _detect_grid_patterns(self, data: np.ndarray) -> float:
        """Detect grid-like patterns in data"""
        if len(data.shape) < 2:
            return 0.0
        
        # Compute autocorrelation
        autocorr = np.correlate(data.flatten(), data.flatten(), mode='full')
        peak_indices = np.where(autocorr > 0.8 * np.max(autocorr))[0]
        
        # Regular spacing indicates grid patterns
        if len(peak_indices) > 2:
            spacings = np.diff(peak_indices)
            spacing_variance = np.var(spacings)
            denom = float(np.mean(spacings)) + 1e-12
            score = 1.0 - float(spacing_variance) / denom
            return float(max(0.0, min(1.0, score)))
        
        return 0.0
    
    def _detect_systematic_queries(self, query_patterns: np.ndarray) -> float:
        """Detect systematic query patterns"""
        qp = np.asarray(query_patterns)
        if qp.ndim < 2 or qp.shape[0] < 2 or qp.shape[1] < 2:
            return 0.0
        
        # Compute query similarity
        with np.errstate(all="ignore"):
            query_similarity = np.corrcoef(qp)
        query_similarity = np.nan_to_num(query_similarity, nan=0.0, posinf=0.0, neginf=0.0)
        
        # High similarity indicates systematic querying
        upper = query_similarity[np.triu_indices_from(query_similarity, k=1)]
        similarity_score = float(np.mean(upper)) if upper.size else 0.0
        
        return float(max(0.0, min(1.0, similarity_score)))
    
    def _compute_reconstruction_confidence(self, model_outputs: np.ndarray, 
                                         voxel_occupancy: float) -> float:
        """Compute confidence in reconstruction capability"""
        # Analyze output entropy
        output_entropy = self._compute_entropy(model_outputs)
        
        # Lower entropy + high voxel occupancy = higher reconstruction confidence
        reconstruction_confidence = voxel_occupancy * (1.0 - output_entropy)
        
        return max(0.0, min(1.0, reconstruction_confidence))
    
    def _analyze_information_leakage(self, model_outputs: np.ndarray) -> float:
        """Analyze potential information leakage from model outputs"""
        # Compute mutual information between different output dimensions
        if model_outputs.ndim < 2 or model_outputs.shape[0] < 2 or model_outputs.shape[1] < 2:
            return 0.0
        
        # Simplified mutual information estimation
        with np.errstate(all="ignore"):
            correlation_matrix = np.corrcoef(model_outputs.T)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        mutual_info_proxy = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
        
        return float(max(0.0, min(1.0, float(mutual_info_proxy))))
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute normalized entropy of data"""
        # Discretize data for entropy calculation
        hist, _ = np.histogram(data.flatten(), bins=50)
        denom = float(np.sum(hist))
        if denom <= 0.0:
            return 0.0
        hist = hist / denom
        
        # Compute entropy
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        ent_norm = float(entropy / np.log(len(hist)))  # Normalize
        return float(max(0.0, min(1.0, ent_norm)))
    
    def _determine_attack_complexity(self, inversion_score: float) -> str:
        """Determine attack complexity level"""
        if inversion_score > 0.8:
            return 'high'
        elif inversion_score > 0.5:
            return 'medium'
        else:
            return 'low'


class AttentionGuardDetector:
    """
    AttentionGuard transformer-based misbehavior detector

    IMPORTANT:
    - This is a **heuristic signal detector** that looks for behavioral inconsistency,
      anomaly-like sequence/attention patterns, and distribution shifts.
    - It is **not** a faithful implementation of a single named paper.
    """
    
    def __init__(self, max_seq_length: int = 512, attention_heads: int = 8, misbehavior_threshold: float = 0.6):
        self.max_seq_length = max_seq_length
        self.attention_heads = attention_heads
        self.misbehavior_threshold = float(misbehavior_threshold)
        
        if HAS_TRANSFORMERS:
            self.tokenizer = None
            self.model = None
            self._init_attention_model()
    
    def _init_attention_model(self):
        """Initialize attention-based detection model"""
        if not HAS_TRANSFORMERS:
            return
        
        try:
            # Use a lightweight transformer for demonstration
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = AutoModel.from_pretrained('distilbert-base-uncased')
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
    
    def detect_misbehavior(self, sequence_data: np.ndarray, 
                          context_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect misbehavior using attention mechanisms
        
        Args:
            sequence_data: Sequential data to analyze
            context_data: Additional context information
            
        Returns:
            Misbehavior detection results
        """
        
        # Bound work to max_seq_length to avoid O(N^2) attention matrices on huge arrays.
        seq = np.asarray(sequence_data)
        if seq.size > 0 and seq.shape[0] > int(self.max_seq_length):
            seq = seq[-int(self.max_seq_length):]
        ctx = None
        if context_data is not None:
            ctx = np.asarray(context_data)
            if ctx.size > 0 and ctx.shape[0] > int(self.max_seq_length):
                ctx = ctx[-int(self.max_seq_length):]

        # Analyze attention patterns
        attention_analysis = self._analyze_attention_patterns(seq)
        
        # Detect anomalous sequences
        sequence_anomalies = self._detect_sequence_anomalies(seq)
        
        # Temporal consistency analysis (consistency ∈ [0,1]).
        # For a *risk* score we use temporal_anomaly = 1 - consistency.
        temporal_consistency = self._analyze_temporal_consistency(seq)
        temporal_anomaly = 0.0
        if temporal_consistency is not None:
            temporal_anomaly = float(np.clip(1.0 - float(temporal_consistency), 0.0, 1.0))

        # Context-aware analysis.
        # We compute context_consistency ∈ [0,1] (higher = more similar); for risk we use
        # context_anomaly = 1 - consistency. If no context is provided, we do not invent a score.
        context_consistency = 0.0
        context_anomaly = 0.0
        if ctx is not None:
            context_consistency = self._analyze_context_consistency(seq, ctx)
            context_anomaly = float(np.clip(1.0 - float(context_consistency), 0.0, 1.0))

        # Combine scores (higher = more anomalous / higher risk)
        misbehavior_score = (
            attention_analysis * 0.3 +
            sequence_anomalies * 0.3 +
            temporal_anomaly * 0.2 +
            context_anomaly * 0.2
        )
        misbehavior_score = float(np.clip(float(misbehavior_score), 0.0, 1.0))
        is_misbehavior = bool(misbehavior_score > float(self.misbehavior_threshold))
        
        return {
            'is_misbehavior': is_misbehavior,
            'misbehavior_score': float(misbehavior_score),
            'attention_analysis': float(attention_analysis),
            'sequence_anomalies': float(sequence_anomalies),
            'temporal_consistency': float(temporal_consistency) if temporal_consistency is not None else 0.0,
            'temporal_anomaly': float(temporal_anomaly),
            'context_consistency': float(context_consistency),
            'context_score': float(context_anomaly),
            'threat_level': self._determine_threat_level(misbehavior_score)
        }
    
    def _analyze_attention_patterns(self, sequence_data: np.ndarray) -> float:
        """Analyze attention patterns for anomalies"""
        seq = np.asarray(sequence_data)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        if seq.size == 0:
            return 0.0
        # Normalize to 2D [N, D] for attention proxy. If input is 1D -> [N,1].
        if seq.ndim == 1:
            seq = seq.reshape(-1, 1)
        elif seq.ndim != 2:
            seq = seq.reshape(-1, int(seq.shape[-1]))

        # Compute self-attention weights
        seq_tensor = torch.from_numpy(seq).float()
        if seq_tensor.shape[0] < 2 or seq_tensor.shape[1] < 1:
            return 0.0
        
        # Simplified attention computation
        attention_weights = torch.softmax(torch.matmul(seq_tensor, seq_tensor.T) / np.sqrt(seq_tensor.shape[-1]), dim=-1)
        
        # Analyze attention distribution
        attention_entropy = self._compute_attention_entropy(attention_weights)
        attention_concentration = self._compute_attention_concentration(attention_weights)
        
        # Anomalous patterns: very low or very high entropy
        anomaly_score = abs(attention_entropy - 0.5) + attention_concentration
        
        return max(0.0, min(1.0, anomaly_score))
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution"""
        # Normalize attention weights
        attention_flat = attention_weights.flatten()
        attention_norm = attention_flat / torch.sum(attention_flat)
        
        # Compute entropy
        entropy = -torch.sum(attention_norm * torch.log(attention_norm + 1e-10))
        return entropy.item() / np.log(len(attention_norm))
    
    def _compute_attention_concentration(self, attention_weights: torch.Tensor) -> float:
        """Compute concentration of attention"""
        # Gini coefficient as concentration measure
        attention_sorted = torch.sort(attention_weights.flatten())[0]
        n = len(attention_sorted)
        
        cumsum = torch.cumsum(attention_sorted, dim=0)
        denom = cumsum[-1]
        if float(denom.item()) == 0.0:
            return 0.0
        gini = (n + 1 - 2 * torch.sum(cumsum) / denom) / n
        
        return gini.item()
    
    def _detect_sequence_anomalies(self, sequence_data: np.ndarray) -> float:
        """Detect anomalies in sequential patterns"""
        if len(sequence_data) < 2:
            return 0.0
        
        # Compute sequence derivatives
        derivatives = np.diff(sequence_data, axis=0)
        
        # Analyze derivative patterns
        derivative_variance = np.var(derivatives, axis=0)
        derivative_skewness = np.mean([self._compute_skewness(deriv) for deriv in derivatives.T])
        
        # High variance or skewness indicates anomalies
        anomaly_score = np.mean(derivative_variance) + abs(derivative_skewness)
        
        return max(0.0, min(1.0, anomaly_score))
    
    def _analyze_temporal_consistency(self, sequence_data: np.ndarray) -> float:
        """Analyze temporal consistency of sequence"""
        if len(sequence_data) < 3:
            # Not enough data to estimate; return "unknown-but-not-anomalous" baseline.
            return 0.0
        
        # Compute autocorrelation
        autocorr = np.correlate(sequence_data.flatten(), sequence_data.flatten(), mode='full')
        
        # Analyze periodicity
        peak_indices = np.where(autocorr > 0.7 * np.max(autocorr))[0]
        
        if len(peak_indices) > 2:
            # Regular peaks indicate consistency
            peak_spacings = np.diff(peak_indices)
            denom = float(np.mean(peak_spacings)) + 1e-12
            consistency_score = 1.0 - float(np.var(peak_spacings)) / denom
        else:
            consistency_score = 0.0
        
        return max(0.0, min(1.0, consistency_score))
    
    def _analyze_context_consistency(self, sequence_data: np.ndarray, 
                                   context_data: np.ndarray) -> float:
        """Analyze consistency between sequence and context"""
        # Compute cross-correlation robustly.
        # Note: callers may supply different-length sequences (e.g., live window vs reference window).
        seq = np.asarray(sequence_data).flatten()
        ctx = np.asarray(context_data).flatten()
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        ctx = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)

        n = int(min(seq.size, ctx.size))
        if n < 2:
            return 0.0
        if seq.size != ctx.size:
            seq = seq[:n]
            ctx = ctx[:n]

        # Degenerate constant series => undefined correlation.
        if float(np.std(seq)) <= 1e-12 or float(np.std(ctx)) <= 1e-12:
            return 0.0

        with np.errstate(all="ignore"):
            cross_corr = float(np.corrcoef(seq, ctx)[0, 1])
        if not np.isfinite(cross_corr):
            cross_corr = 0.0

        # High positive correlation indicates consistency
        return float(max(0.0, cross_corr))
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data distribution"""
        if len(data) < 3:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        # Compute skewness
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        
        return skewness
    
    def _determine_threat_level(self, misbehavior_score: float) -> str:
        """Determine threat level based on misbehavior score"""
        if misbehavior_score > 0.8:
            return 'critical'
        elif misbehavior_score > 0.6:
            return 'high'
        elif misbehavior_score > 0.4:
            return 'medium'
        else:
            return 'low'


class EDNNAttackDetector:
    """
    EDNN (Element-wise Differential Nearest Neighbor) attack detector

    IMPORTANT:
    - This is a **heuristic signal detector** for embedding-space attacks (nearest-neighbor
      leakage / membership-style indicators).
    - It is **not** a faithful implementation of a single named paper.
    """
    
    def __init__(self, k_neighbors: int = 5, threshold: float = 0.7):
        self.k_neighbors = k_neighbors
        self.threshold = threshold
        self.embedding_history = []
        
    def detect_ednn_attack(self, embeddings: np.ndarray, 
                          reference_embeddings: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect EDNN attacks on embedding matrices
        
        Args:
            embeddings: Current embedding matrix
            reference_embeddings: Reference embeddings for comparison
            
        Returns:
            EDNN attack detection results
        """
        
        embeddings = np.asarray(embeddings)
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        if reference_embeddings is not None:
            reference_embeddings = np.asarray(reference_embeddings)
            reference_embeddings = np.nan_to_num(reference_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # Degenerate inputs: avoid mean/std/corr warnings (strict mode) and return a safe baseline.
        if embeddings.size == 0 or embeddings.ndim < 2 or embeddings.shape[0] < 2:
            return {
                'is_ednn_attack': False,
                'attack_score': 0.0,
                'element_patterns': 0.0,
                'nn_manipulation': 0.0,
                'differential_score': 0.0,
                'obfuscation_score': 0.0,
                'attack_type': 'none',
            }

        # Analyze element-wise patterns
        element_patterns = self._analyze_element_patterns(embeddings)
        
        # Detect nearest neighbor manipulation
        nn_manipulation = self._detect_nn_manipulation(embeddings, reference_embeddings)
        
        # Differential analysis
        differential_score = self._compute_differential_score(embeddings)
        
        # Obfuscation detection
        obfuscation_score = self._detect_obfuscation(embeddings)
        
        # Combine scores
        attack_score = (
            element_patterns * 0.3 +
            nn_manipulation * 0.3 +
            differential_score * 0.2 +
            obfuscation_score * 0.2
        )
        attack_score = float(np.clip(float(attack_score), 0.0, 1.0))
        is_ednn_attack = bool(attack_score > float(self.threshold))
        
        return {
            'is_ednn_attack': is_ednn_attack,
            'attack_score': float(attack_score),
            'element_patterns': float(element_patterns),
            'nn_manipulation': float(nn_manipulation),
            'differential_score': float(differential_score),
            'obfuscation_score': float(obfuscation_score),
            'attack_type': self._classify_attack_type(attack_score, element_patterns)
        }
    
    def _analyze_element_patterns(self, embeddings: np.ndarray) -> float:
        """Analyze element-wise patterns in embeddings"""
        embeddings = np.asarray(embeddings)
        if embeddings.size == 0 or embeddings.ndim < 2 or embeddings.shape[0] < 2:
            return 0.0
        # Compute element-wise statistics
        element_means = np.mean(embeddings, axis=0)
        element_stds = np.std(embeddings, axis=0)
        
        # Detect uniform patterns (suspicious)
        denom = float(np.mean(np.abs(element_stds))) + 1e-12
        uniformity_score = 1.0 - float(np.var(element_stds)) / denom
        uniformity_score = float(max(0.0, min(1.0, uniformity_score)))
        
        # Detect outlier elements
        outlier_score = np.mean(np.abs(element_means - np.median(element_means)) > 2 * np.std(element_means))
        
        return max(0.0, min(1.0, (uniformity_score + outlier_score) / 2.0))
    
    def _detect_nn_manipulation(self, embeddings: np.ndarray, 
                               reference_embeddings: Optional[np.ndarray]) -> float:
        """Detect manipulation of nearest neighbor relationships"""
        if reference_embeddings is None:
            return 0.0

        embeddings = np.asarray(embeddings)
        reference_embeddings = np.asarray(reference_embeddings)
        if embeddings.size == 0 or reference_embeddings.size == 0:
            return 0.0
        if embeddings.ndim < 2 or reference_embeddings.ndim < 2:
            return 0.0
        if embeddings.shape[0] < 2 or reference_embeddings.shape[0] < 2:
            return 0.0

        # Compute pairwise distance vectors (upper triangle) so we can compare distributions
        # even when the number of samples differs.
        current_distances = self._compute_pairwise_distances(embeddings)
        reference_distances = self._compute_pairwise_distances(reference_embeddings)

        cur = current_distances[np.triu_indices_from(current_distances, k=1)]
        ref = reference_distances[np.triu_indices_from(reference_distances, k=1)]
        if cur.size < 2 or ref.size < 2:
            return 0.0

        # Compare distance distributions via histogram correlation on common bins.
        vmin = float(min(np.min(cur), np.min(ref)))
        vmax = float(max(np.max(cur), np.max(ref)))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return 0.0

        bins = np.linspace(vmin, vmax + 1e-12, 32)
        h_cur, _ = np.histogram(cur, bins=bins, density=True)
        h_ref, _ = np.histogram(ref, bins=bins, density=True)

        # Degenerate distributions => no meaningful comparison.
        if float(np.std(h_cur)) <= 1e-12 or float(np.std(h_ref)) <= 1e-12:
            distance_correlation = 0.0
        else:
            with np.errstate(all="ignore"):
                distance_correlation = float(np.corrcoef(h_cur, h_ref)[0, 1])
            if not np.isfinite(distance_correlation):
                distance_correlation = 0.0

        # Low correlation indicates manipulation
        manipulation_score = 1.0 - abs(distance_correlation)
        return float(max(0.0, min(1.0, manipulation_score)))
    
    def _compute_pairwise_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between embeddings"""
        n = len(embeddings)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(embeddings[i] - embeddings[j])
                distances[i, j] = distance
                distances[j, i] = distance
        
        return distances
    
    def _compute_differential_score(self, embeddings: np.ndarray) -> float:
        """Compute differential score for attack detection"""
        # Compute first and second derivatives
        if len(embeddings) < 3:
            return 0.0
        
        first_diff = np.diff(embeddings, axis=0)
        second_diff = np.diff(first_diff, axis=0)
        
        # Analyze differential patterns
        first_diff_variance = np.var(first_diff, axis=0)
        second_diff_variance = np.var(second_diff, axis=0)
        
        # Unusual differential patterns indicate attacks
        differential_score = np.mean(first_diff_variance) + np.mean(second_diff_variance)
        
        return max(0.0, min(1.0, differential_score))
    
    def _detect_obfuscation(self, embeddings: np.ndarray) -> float:
        """Detect obfuscation in embeddings"""
        # Compute embedding entropy
        embedding_entropy = self._compute_entropy(embeddings)
        
        # Analyze value distribution
        value_distribution = np.histogram(embeddings.flatten(), bins=50)[0]
        distribution_entropy = self._compute_entropy(value_distribution)
        
        # Unusual entropy patterns indicate obfuscation
        obfuscation_score = abs(embedding_entropy - 0.5) + abs(distribution_entropy - 0.5)
        
        return max(0.0, min(1.0, obfuscation_score))
    
    def _classify_attack_type(self, attack_score: float, element_patterns: float) -> str:
        """Classify the type of EDNN attack"""
        if attack_score > 0.8:
            if element_patterns > 0.7:
                return 'sophisticated_obfuscation'
            else:
                return 'direct_manipulation'
        elif attack_score > 0.6:
            return 'moderate_attack'
        else:
            return 'low_risk'
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute normalized entropy of data"""
        if len(data) == 0:
            return 0.0
        
        hist, _ = np.histogram(data.flatten(), bins=50)
        hist = hist.astype(np.float64)
        s = float(np.sum(hist))
        if not np.isfinite(s) or s <= 0.0:
            return 0.0
        hist = hist / (s + 1e-10)

        # Remove zero probabilities
        hist = hist[hist > 1e-10]
        if len(hist) < 2:
            return 0.0

        # Compute entropy (nats) and normalize by ln(K)
        entropy = -float(np.sum(hist * np.log(hist + 1e-10)))
        return float(entropy / np.log(float(len(hist))))


class AdversarialDetector:
    """
    Main adversarial detection system integrating all detection mechanisms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize specialized detectors
        self.ts_inverse_detector = TSInverseDetector(
            sensitivity_threshold=self.config.get('ts_inverse_threshold', 0.8)
        )
        
        self.concretizer_detector = ConcreTizerDetector(
            voxel_resolution=self.config.get('voxel_resolution', 32),
            inversion_threshold=self.config.get('concretizer_threshold', self.config.get('threshold', 0.7)),
        )
        
        self.attention_guard = AttentionGuardDetector(
            max_seq_length=self.config.get('max_seq_length', 512),
            attention_heads=self.config.get('attention_heads', 8),
            misbehavior_threshold=self.config.get('attention_guard_threshold', self.config.get('threshold', 0.6)),
        )
        
        self.ednn_detector = EDNNAttackDetector(
            k_neighbors=self.config.get('k_neighbors', 5),
            threshold=self.config.get('ednn_threshold', 0.7)
        )
        
        self.detection_history = []
        
    def detect_adversarial_samples(
        self,
        samples: np.ndarray,
        reference_samples: Optional[np.ndarray] = None,
        detector_type: str = "all",
    ) -> Dict[str, Any]:
        """
        Comprehensive adversarial sample detection
        
        Args:
            samples: Input samples to analyze
            reference_samples: Reference samples for comparison
            
        Returns:
            Comprehensive detection results
        """
        
        # Robustness: ensure computations don't crash on non-finite inputs.
        samples = np.asarray(samples)
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        if reference_samples is not None:
            reference_samples = np.asarray(reference_samples)
            reference_samples = np.nan_to_num(reference_samples, nan=0.0, posinf=0.0, neginf=0.0)

        results = {
            'timestamp': time.time(),
            'input_shape': samples.shape,
            'detections': {},
            'overall_threat_level': 'low',
            'confidence_scores': {}
        }

        # Degenerate inputs: avoid warnings-as-errors from downstream stats/corr on empty arrays.
        if samples.size == 0 or samples.ndim < 2 or samples.shape[0] < 2:
            self.detection_history.append(results)
            return results
        
        # Normalize detector selection to the CLI vocabulary.
        dt = str(detector_type or "all").strip().lower().replace("_", "-")
        valid = {"all", "ts-inverse", "concretizer", "attention-guard", "ednn"}
        if dt not in valid:
            dt = "all"

        run_ts = dt in {"all", "ts-inverse"}
        run_conc = dt in {"all", "concretizer"}
        run_attn = dt in {"all", "attention-guard"}
        run_ednn = dt in {"all", "ednn"}

        # Run selected detectors
        try:
            # TS-Inverse detection
            if run_ts and len(samples.shape) > 1:
                ts_result = self.ts_inverse_detector.detect_gradient_inversion(samples, reference_samples)
                results['detections']['ts_inverse'] = {
                    'is_attack': ts_result.is_inverted,
                    # Keep `confidence` for downstream consumers, and add explicit key for internal aggregators.
                    'confidence': ts_result.inversion_confidence,
                    'inversion_confidence': ts_result.inversion_confidence,
                    'privacy_leakage': ts_result.privacy_leakage_score,
                    'complexity': ts_result.attack_complexity
                }
            
            # ConcreTizer detection
            if run_conc:
                # If a reference is provided, treat it as the best available proxy for a query/baseline series.
                query_patterns = reference_samples if reference_samples is not None else samples
                concretizer_result = self.concretizer_detector.detect_model_inversion(samples, query_patterns)
                results['detections']['concretizer'] = concretizer_result
            
            # AttentionGuard detection
            if run_attn:
                attention_result = self.attention_guard.detect_misbehavior(samples, reference_samples)
                results['detections']['attention_guard'] = attention_result
            
            # EDNN detection
            if run_ednn:
                ednn_result = self.ednn_detector.detect_ednn_attack(samples, reference_samples)
                results['detections']['ednn'] = ednn_result
            
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            results['error'] = str(e)
        
        # Compute overall threat assessment
        results['overall_threat_level'] = self._compute_overall_threat_level(results['detections'])
        results['confidence_scores'] = self._compute_confidence_scores(results['detections'])
        
        # Store in history
        self.detection_history.append(results)
        
        return results
    
    def _compute_overall_threat_level(self, detections: Dict[str, Any]) -> str:
        """Compute overall threat level from all detections"""
        threat_scores = []
        
        for detection_type, detection_result in detections.items():
            if isinstance(detection_result, dict):
                # Extract threat score based on detection type
                if 'inversion_confidence' in detection_result:
                    threat_scores.append(detection_result['inversion_confidence'])
                elif 'inversion_score' in detection_result:
                    threat_scores.append(detection_result['inversion_score'])
                elif 'misbehavior_score' in detection_result:
                    threat_scores.append(detection_result['misbehavior_score'])
                elif 'attack_score' in detection_result:
                    threat_scores.append(detection_result['attack_score'])
        
        if not threat_scores:
            return 'low'
        
        avg_threat = np.mean(threat_scores)
        
        if avg_threat > 0.8:
            return 'critical'
        elif avg_threat > 0.6:
            return 'high'
        elif avg_threat > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _compute_confidence_scores(self, detections: Dict[str, Any]) -> Dict[str, float]:
        """Compute confidence scores for each detection method"""
        confidence_scores = {}
        
        for detection_type, detection_result in detections.items():
            if isinstance(detection_result, dict):
                # Extract confidence based on detection type
                if 'inversion_confidence' in detection_result:
                    confidence_scores[detection_type] = detection_result['inversion_confidence']
                elif 'inversion_score' in detection_result:
                    confidence_scores[detection_type] = detection_result['inversion_score']
                elif 'misbehavior_score' in detection_result:
                    confidence_scores[detection_type] = detection_result['misbehavior_score']
                elif 'attack_score' in detection_result:
                    confidence_scores[detection_type] = detection_result['attack_score']
                else:
                    # No explicit score found; do not invent a confidence value.
                    confidence_scores[detection_type] = 0.0
        
        return confidence_scores
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of all detections"""
        if not self.detection_history:
            return {'total_detections': 0, 'threat_levels': {}}
        
        threat_levels = [d['overall_threat_level'] for d in self.detection_history]
        threat_distribution = {
            level: threat_levels.count(level) for level in ['low', 'medium', 'high', 'critical']
        }
        
        return {
            'total_detections': len(self.detection_history),
            'threat_levels': threat_distribution,
            'latest_detection': self.detection_history[-1],
            'average_confidence': np.mean([
                np.mean(list(d.get('confidence_scores', {}).values()) or [0.0])
                for d in self.detection_history
            ])
        }


# Helper functions for compatibility with test suite
def detect_gradient_anomalies(gradients: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect gradient anomalies using TS-Inverse detector
    
    Args:
        gradients: Input gradients to analyze
        
    Returns:
        List of detected anomalies
    """
    detector = TSInverseDetector()
    result = detector.detect_gradient_inversion(gradients)
    
    if result.is_inverted:
        return [{
            'type': 'gradient_inversion',
            'confidence': result.inversion_confidence,
            'privacy_leakage': result.privacy_leakage_score,
            'complexity': result.attack_complexity
        }]
    
    return []


def analyze_activation_patterns(activations: np.ndarray) -> Dict[str, Any]:
    """
    Analyze activation patterns using AttentionGuard
    
    Args:
        activations: Input activations to analyze
        
    Returns:
        Analysis results
    """
    detector = AttentionGuardDetector()
    result = detector.detect_misbehavior(activations)
    
    return {
        'is_anomalous': result['is_misbehavior'],
        'anomaly_score': result['misbehavior_score'],
        'pattern_type': result['threat_level'],
        'attention_analysis': result['attention_analysis'],
        'temporal_consistency': result['temporal_consistency']
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
    print("🔒 Advanced Adversarial Detection System")
    print("=" * 50)
    
    # Create detector
    detector = AdversarialDetector()
    
    # Test with sample data
    test_samples = np.random.randn(100, 64)
    reference_samples = np.random.randn(100, 64)
    
    # Run detection
    results = detector.detect_adversarial_samples(test_samples, reference_samples)
    
    print(f"Overall threat level: {results['overall_threat_level']}")
    print(f"Confidence scores: {results['confidence_scores']}")
    
    # Get summary
    summary = detector.get_detection_summary()
    print(f"Detection summary: {summary}") 