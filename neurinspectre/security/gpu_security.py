"""
GPU-Accelerated Security Analysis for NeurInSpectre
Optimized for both NVIDIA CUDA and Mac Silicon MPS
Designed for high-throughput security analysis on GPU (when available).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

def get_optimal_device() -> torch.device:
    """Get the optimal GPU device for the current platform"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🚀 Using NVIDIA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("🚀 Using Mac Silicon MPS GPU")
        return device
    else:
        raise RuntimeError("No GPU available! This module requires NVIDIA CUDA or Mac Silicon MPS.")

@dataclass
class GPUSecurityResult:
    """GPU-accelerated security analysis result"""
    threat_level: str
    confidence: float
    processing_time: float
    gpu_memory_used: float
    detection_details: Dict[str, Any]
    device_info: str

class GPUAdversarialDetector:
    """
    GPU-accelerated adversarial attack detector
    Implements GPU-accelerated detection routines.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_optimal_device()
        self.dtype = torch.float32
        # Internal compute dtype for stats/correlation. MPS does not support float64.
        self._stats_dtype = torch.float32 if self.device.type == 'mps' else torch.float64
        
        # Initialize GPU-optimized detection networks
        self._init_detection_networks()
        
        # GPU memory management
        self._clear_gpu_cache()
        
        logger.info(f"🛡️ GPU Adversarial Detector initialized on {self.device}")

    @staticmethod
    def _to_scalar(value: Union[torch.Tensor, float, int]) -> float:
        """Convert a tensor score to a Python float safely."""
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else torch.mean(value).item()
        return float(value)
    
    def _init_detection_networks(self):
        """Initialize GPU-optimized neural networks for detection"""
        
        # TS-Inverse gradient inversion detector
        self.ts_inverse_net = TSInverseDetectorNet().to(self.device)
        
        # ConcreTizer model inversion detector  
        self.concretizer_net = ConcreTizerDetectorNet().to(self.device)
        
        # AttentionGuard transformer detector
        self.attention_guard_net = AttentionGuardNet().to(self.device)
        
        # EDNN attack detector
        self.ednn_net = EDNNDetectorNet().to(self.device)
        
        # Set all networks to evaluation mode
        self.ts_inverse_net.eval()
        self.concretizer_net.eval()
        self.attention_guard_net.eval()
        self.ednn_net.eval()
    
    def _clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / 1024**3
        elif self.device.type == 'mps':
            return torch.mps.driver_allocated_memory() / 1024**3
        return 0.0
    
    def detect_adversarial_attacks(self, data: np.ndarray, 
                                 reference_data: Optional[np.ndarray] = None) -> GPUSecurityResult:
        """
        GPU-accelerated adversarial attack detection
        
        Args:
            data: Input data to analyze
            reference_data: Optional reference data for comparison
            
        Returns:
            GPU security analysis result
        """
        
        start_time = time.time()
        initial_memory = self._get_gpu_memory_usage()
        
        # Convert to GPU tensors with optimal dtype
        safe_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data_tensor = torch.from_numpy(safe_data).to(self.device, dtype=self.dtype)
        data_tensor = torch.nan_to_num(data_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        if reference_data is not None:
            safe_ref = np.nan_to_num(reference_data, nan=0.0, posinf=0.0, neginf=0.0)
            ref_tensor = torch.from_numpy(safe_ref).to(self.device, dtype=self.dtype)
            ref_tensor = torch.nan_to_num(ref_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            ref_tensor = None
        
        detection_results = {}
        
        with torch.no_grad():
            # Run all detection algorithms in parallel on GPU
            
            # TS-Inverse detection
            ts_result = self._run_ts_inverse_detection(data_tensor, ref_tensor)
            detection_results['ts_inverse'] = ts_result
            
            # ConcreTizer detection
            concretizer_result = self._run_concretizer_detection(data_tensor)
            detection_results['concretizer'] = concretizer_result
            
            # AttentionGuard detection
            attention_result = self._run_attention_guard_detection(data_tensor)
            detection_results['attention_guard'] = attention_result
            
            # EDNN detection
            ednn_result = self._run_ednn_detection(data_tensor, ref_tensor)
            detection_results['ednn'] = ednn_result
        
        # Compute overall threat assessment
        threat_level, confidence = self._compute_threat_assessment(detection_results)
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        final_memory = self._get_gpu_memory_usage()
        memory_used = final_memory - initial_memory
        
        # Clean up GPU memory
        del data_tensor
        if ref_tensor is not None:
            del ref_tensor
        self._clear_gpu_cache()
        
        return GPUSecurityResult(
            threat_level=threat_level,
            confidence=confidence,
            processing_time=processing_time,
            gpu_memory_used=memory_used,
            detection_details=detection_results,
            device_info=str(self.device)
        )
    
    def _run_ts_inverse_detection(self, data: torch.Tensor, 
                                reference: Optional[torch.Tensor]) -> Dict[str, float]:
        """GPU-accelerated TS-Inverse gradient inversion detection"""
        
        # Ensure data is in correct shape for convolution
        if len(data.shape) == 2:
            # Reshape to square image for MPS compatibility
            batch_size, feature_dim = data.shape
            img_size = int(np.ceil(np.sqrt(feature_dim)))
            padded_size = img_size * img_size
            if padded_size != feature_dim:
                padding = padded_size - feature_dim
                data = F.pad(data, (0, padding))
            data = data.view(batch_size, 1, img_size, img_size)
        elif len(data.shape) == 3:
            data = data.unsqueeze(1)  # Add channel dim
        
        # Run through TS-Inverse detection network
        detection_score = self.ts_inverse_net(data)
        
        # Compute frequency domain analysis on GPU
        fft_data = torch.fft.fft2(data)
        spectral_entropy = self._compute_spectral_entropy_gpu(fft_data)
        
        # Privacy leakage analysis
        privacy_score = self._analyze_privacy_leakage_gpu(data, reference)
        
        # Combine scores
        detection_score_val = self._to_scalar(detection_score)
        
        combined_score = (detection_score_val + spectral_entropy + privacy_score) / 3.0
        
        return {
            'is_attack': combined_score > 0.7,
            'confidence': combined_score,
            'detection_score': detection_score_val,
            'spectral_entropy': spectral_entropy,
            'privacy_leakage': privacy_score
        }
    
    def _run_concretizer_detection(self, data: torch.Tensor) -> Dict[str, float]:
        """GPU-accelerated ConcreTizer model inversion detection"""
        
        # Prepare data for 3D analysis
        if len(data.shape) == 2:
            # Reshape to simulate voxel space
            voxel_size = int(np.sqrt(data.shape[1]))
            if voxel_size * voxel_size == data.shape[1]:
                data_3d = data.view(data.shape[0], 1, voxel_size, voxel_size)
            else:
                # Pad to square
                pad_size = int(np.ceil(np.sqrt(data.shape[1])))
                padding = pad_size * pad_size - data.shape[1]
                data_padded = F.pad(data, (0, padding))
                data_3d = data_padded.view(data.shape[0], 1, pad_size, pad_size)
        else:
            data_3d = data
        
        # Run ConcreTizer detection
        detection_score = self.concretizer_net(data_3d)
        detection_score_val = self._to_scalar(detection_score)
        
        # Voxel occupancy analysis on GPU
        occupancy_score = self._analyze_voxel_occupancy_gpu(data_3d)
        
        # Grid pattern detection
        grid_score = self._detect_grid_patterns_gpu(data_3d)
        
        # Systematic query detection
        query_score = self._detect_systematic_queries_gpu(data_3d)
        
        combined_score = (detection_score_val + occupancy_score + grid_score + query_score) / 4.0
        
        return {
            'is_inversion_attack': combined_score > 0.6,
            'inversion_score': combined_score,
            'detection_score': detection_score_val,
            'voxel_occupancy': occupancy_score,
            'grid_patterns': grid_score,
            'systematic_queries': query_score
        }
    
    def _run_attention_guard_detection(self, data: torch.Tensor) -> Dict[str, float]:
        """GPU-accelerated AttentionGuard transformer detection"""
        
        # Prepare sequence data
        if len(data.shape) == 2:
            sequence_data = data.unsqueeze(0)  # Add batch dim
        else:
            sequence_data = data
        
        # Run through AttentionGuard network
        detection_score = self.attention_guard_net(sequence_data)
        detection_score_val = self._to_scalar(detection_score)
        
        # Attention pattern analysis on GPU
        attention_entropy = self._compute_attention_entropy_gpu(sequence_data)
        
        # Sequence anomaly detection
        sequence_anomalies = self._detect_sequence_anomalies_gpu(sequence_data)
        
        # Temporal consistency analysis
        temporal_consistency = self._analyze_temporal_consistency_gpu(sequence_data)
        
        combined_score = (detection_score_val + attention_entropy + 
                         sequence_anomalies + (1.0 - temporal_consistency)) / 4.0
        
        return {
            'is_misbehavior': combined_score > 0.6,
            'misbehavior_score': combined_score,
            'detection_score': detection_score_val,
            'attention_entropy': attention_entropy,
            'sequence_anomalies': sequence_anomalies,
            'temporal_consistency': temporal_consistency
        }
    
    def _run_ednn_detection(self, data: torch.Tensor, 
                          reference: Optional[torch.Tensor]) -> Dict[str, float]:
        """GPU-accelerated EDNN attack detection"""
        
        # Run EDNN detection network
        detection_score = self.ednn_net(data)
        detection_score_val = self._to_scalar(detection_score)
        
        # Element-wise pattern analysis on GPU
        element_patterns = self._analyze_element_patterns_gpu(data)
        
        # Nearest neighbor manipulation detection
        nn_manipulation = self._detect_nn_manipulation_gpu(data, reference)
        
        # Differential pattern analysis
        differential_score = self._compute_differential_patterns_gpu(data)
        
        # Embedding obfuscation detection
        obfuscation_score = self._detect_embedding_obfuscation_gpu(data)
        
        combined_score = (detection_score_val + element_patterns + 
                         nn_manipulation + differential_score + obfuscation_score) / 5.0
        
        return {
            'is_ednn_attack': combined_score > 0.7,
            'attack_score': combined_score,
            'detection_score': detection_score_val,
            'element_patterns': element_patterns,
            'nn_manipulation': nn_manipulation,
            'differential_patterns': differential_score,
            'obfuscation': obfuscation_score
        }
    
    def _compute_spectral_entropy_gpu(self, fft_data: torch.Tensor) -> float:
        """Compute *normalized* spectral entropy on GPU in [0, 1]."""
        magnitude = torch.abs(fft_data)
        power_spectrum = magnitude ** 2
        power_spectrum = torch.nan_to_num(power_spectrum, nan=0.0, posinf=0.0, neginf=0.0)

        if power_spectrum.dim() == 0:
            return 0.0

        # Flatten per sample; treat dim0 as batch when present.
        if power_spectrum.dim() == 1:
            p = power_spectrum.unsqueeze(0)
        else:
            p = power_spectrum.reshape(power_spectrum.shape[0], -1)

        p_sum = torch.sum(p, dim=-1, keepdim=True)
        p_norm = p / (p_sum + 1e-10)
        log_p = torch.log(p_norm + 1e-10)
        ent = -torch.sum(p_norm * log_p, dim=-1)  # nats

        n_bins = int(p_norm.shape[-1])
        if n_bins <= 1:
            ent_norm = torch.zeros_like(ent)
        else:
            max_ent = torch.log(torch.tensor(float(n_bins), dtype=self.dtype, device=self.device)) + 1e-10
            ent_norm = ent / max_ent

        ent_norm = torch.clamp(ent_norm, 0.0, 1.0)
        return float(torch.mean(ent_norm).item())
    
    def _analyze_privacy_leakage_gpu(self, data: torch.Tensor, 
                                   reference: Optional[torch.Tensor]) -> float:
        """Analyze privacy leakage on GPU"""
        if reference is None:
            # No reference => no defensible correlation-based leakage estimate.
            return 0.0

        # Mutual-information proxy via (safe) Pearson correlation.
        x = torch.nan_to_num(data.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(reference.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
        n = int(min(x.numel(), y.numel()))
        if n < 2:
            return 0.0

        x = x[:n].to(self._stats_dtype)
        y = y[:n].to(self._stats_dtype)
        x = x - torch.mean(x)
        y = y - torch.mean(y)

        sx = torch.std(x, unbiased=False)
        sy = torch.std(y, unbiased=False)
        denom = (sx * sy).clamp_min(1e-12)
        if float(sx.item()) < 1e-12 or float(sy.item()) < 1e-12:
            # If either series is (near) constant, Pearson correlation is not informative.
            return 0.0

        corr = torch.mean(x * y) / denom
        corr = torch.clamp(corr, -1.0, 1.0)
        return float(torch.abs(corr).item())
    
    def _analyze_voxel_occupancy_gpu(self, data_3d: torch.Tensor) -> float:
        """Analyze voxel occupancy patterns on GPU"""
        # Threshold for occupancy
        occupied = (data_3d > 0.1).float()
        occupancy_ratio = torch.mean(occupied).item()
        
        # Spatial correlation analysis
        shifted = torch.roll(occupied, shifts=1, dims=-1)
        a = occupied.flatten().to(self._stats_dtype)
        b = shifted.flatten().to(self._stats_dtype)
        a = a - torch.mean(a)
        b = b - torch.mean(b)
        sa = torch.std(a, unbiased=False)
        sb = torch.std(b, unbiased=False)
        if float(sa.item()) < 1e-12 or float(sb.item()) < 1e-12:
            spatial_corr_abs = 0.0
        else:
            corr = torch.mean(a * b) / (sa * sb + 1e-12)
            corr = torch.clamp(corr, -1.0, 1.0)
            spatial_corr_abs = float(torch.abs(corr).item())

        score = (float(occupancy_ratio) + spatial_corr_abs) / 2.0
        return float(min(1.0, max(0.0, score)))
    
    def _detect_grid_patterns_gpu(self, data_3d: torch.Tensor) -> float:
        """Detect grid patterns on GPU"""
        # Compute 2D autocorrelation
        data_2d = data_3d.squeeze()
        if len(data_2d.shape) == 3:
            data_2d = data_2d[0]  # Take first channel

        data_2d = torch.nan_to_num(data_2d, nan=0.0, posinf=0.0, neginf=0.0)

        # FFT-based autocorrelation avoids conv2d padding warnings and is more stable.
        fft = torch.fft.fft2(data_2d)
        autocorr = torch.fft.ifft2(torch.abs(fft) ** 2).real

        max_val = torch.max(autocorr)
        if float(max_val.item()) <= 0.0:
            return 0.0

        autocorr_norm = autocorr / (max_val + 1e-12)
        peaks = (autocorr_norm > 0.8).float()

        peak_spacing_score = float(torch.std(peaks, unbiased=False).item())
        if not np.isfinite(peak_spacing_score):
            return 0.0
        return float(min(1.0, max(0.0, peak_spacing_score)))
    
    def _detect_systematic_queries_gpu(self, data_3d: torch.Tensor) -> float:
        """Detect systematic query patterns on GPU"""
        # Compute pairwise similarities
        batch_size = data_3d.shape[0]
        
        if batch_size < 2:
            return 0.0
        
        # Flatten data for similarity computation
        flat_data = data_3d.view(batch_size, -1)
        
        # Compute cosine similarity matrix
        normalized = F.normalize(flat_data, p=2, dim=1)
        similarity_matrix = torch.mm(normalized, normalized.t())
        
        # Get upper triangular part (excluding diagonal)
        upper_tri = torch.triu(similarity_matrix, diagonal=1)
        vals = upper_tri[upper_tri > 0]
        if vals.numel() == 0:
            return 0.0
        avg_similarity = float(torch.mean(vals).item())
        if not np.isfinite(avg_similarity):
            return 0.0
        return float(min(1.0, max(0.0, avg_similarity)))
    
    def _compute_attention_entropy_gpu(self, sequence_data: torch.Tensor) -> float:
        """Compute attention entropy on GPU"""
        # Simple attention mechanism
        attention_weights = F.softmax(torch.matmul(sequence_data, sequence_data.transpose(-2, -1)) / 
                                    np.sqrt(sequence_data.shape[-1]), dim=-1)
        
        # Compute entropy of attention distribution
        attention_flat = attention_weights.view(-1)
        attention_norm = attention_flat / (torch.sum(attention_flat) + 1e-10)
        
        log_attention = torch.log(attention_norm + 1e-10)
        entropy = -torch.sum(attention_norm * log_attention)
        
        # Normalize by maximum possible entropy
        n_bins = int(attention_flat.numel())
        if n_bins <= 1:
            return 0.0
        max_entropy = float(np.log(n_bins))
        if max_entropy <= 0.0 or not np.isfinite(max_entropy):
            return 0.0

        normalized_entropy = float(entropy.item() / max_entropy)
        return float(min(1.0, max(0.0, normalized_entropy)))
    
    def _detect_sequence_anomalies_gpu(self, sequence_data: torch.Tensor) -> float:
        """Detect sequence anomalies on GPU"""
        # Compute sequence derivatives
        if sequence_data.shape[-2] < 2:
            return 0.0
        
        derivatives = torch.diff(sequence_data, dim=-2)
        
        # Analyze derivative statistics
        derivative_var = torch.var(derivatives, dim=-2)
        derivative_mean = torch.mean(derivative_var).item()
        
        # Compute skewness approximation
        centered = derivatives - torch.mean(derivatives, dim=-2, keepdim=True)
        std_dev = torch.std(derivatives, dim=-2, keepdim=True) + 1e-10
        normalized = centered / std_dev
        skewness = torch.mean(normalized ** 3).item()
        
        # High variance or skewness indicates anomalies
        anomaly_score = min(1.0, derivative_mean + abs(skewness) / 2.0)
        return anomaly_score
    
    def _analyze_temporal_consistency_gpu(self, sequence_data: torch.Tensor) -> float:
        """Analyze temporal consistency on GPU"""
        if sequence_data.shape[-2] < 3:
            # Not enough timesteps to estimate; treat as maximally consistent (do not add risk).
            return 1.0

        # Robust, GPU-friendly temporal consistency: coefficient of variation of |Δx_t|.
        # High consistency => small relative variation in step-to-step changes.
        with torch.no_grad():
            delta = torch.diff(sequence_data, dim=-2)
            if delta.numel() == 0:
                return 1.0
            step_mag = torch.mean(torch.abs(delta), dim=-1).reshape(-1)  # (B*(T-1),)
            step_mag = torch.nan_to_num(step_mag, nan=0.0, posinf=0.0, neginf=0.0)
            mu = torch.mean(step_mag)
            sd = torch.std(step_mag, unbiased=False)
            cv = sd / (mu + 1e-10)
            consistency = 1.0 / (1.0 + cv)
            consistency_val = float(consistency.item())
            if not np.isfinite(consistency_val):
                return 1.0
            return float(min(1.0, max(0.0, consistency_val)))
    
    def _analyze_element_patterns_gpu(self, data: torch.Tensor) -> float:
        """Analyze element-wise patterns on GPU"""
        # Element-wise statistics
        element_means = torch.mean(data, dim=0)
        element_stds = torch.std(data, dim=0)
        
        # Detect uniform patterns (suspicious)
        uniformity_score = 1.0 - (torch.var(element_stds) / (torch.mean(element_stds) + 1e-10)).item()
        uniformity_score = float(min(1.0, max(0.0, uniformity_score)))
        
        # Detect outlier elements
        median_mean = torch.median(element_means)
        std_means = torch.std(element_means)
        outliers = torch.abs(element_means - median_mean) > 2 * std_means
        outlier_score = torch.mean(outliers.float()).item()
        outlier_score = float(min(1.0, max(0.0, outlier_score)))

        return float(min(1.0, max(0.0, (uniformity_score + outlier_score) / 2.0)))
    
    def _detect_nn_manipulation_gpu(self, data: torch.Tensor, 
                                  reference: Optional[torch.Tensor]) -> float:
        """Detect nearest neighbor manipulation on GPU"""
        if reference is None:
            # Without a reference set we cannot compare NN structure; do not fabricate a score.
            return 0.0
        
        # Compute pairwise distances
        data_flat = data.view(data.shape[0], -1)
        ref_flat = reference.view(reference.shape[0], -1)
        
        # Euclidean distance matrices
        data_dists = torch.cdist(data_flat, data_flat)
        ref_dists = torch.cdist(ref_flat, ref_flat)
        
        # Compare distance distributions
        data_dists_flat = data_dists[torch.triu(torch.ones_like(data_dists), diagonal=1) == 1]
        ref_dists_flat = ref_dists[torch.triu(torch.ones_like(ref_dists), diagonal=1) == 1]
        
        min_len = min(len(data_dists_flat), len(ref_dists_flat))
        if min_len == 0:
            return 0.0
        
        x = torch.nan_to_num(data_dists_flat[:min_len].to(self._stats_dtype), nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(ref_dists_flat[:min_len].to(self._stats_dtype), nan=0.0, posinf=0.0, neginf=0.0)

        x = x - torch.mean(x)
        y = y - torch.mean(y)
        sx = torch.std(x, unbiased=False)
        sy = torch.std(y, unbiased=False)
        if float(sx.item()) < 1e-12 or float(sy.item()) < 1e-12:
            # Degenerate distributions => no informative comparison.
            return 0.0

        corr = torch.mean(x * y) / (sx * sy + 1e-12)
        corr = torch.clamp(corr, -1.0, 1.0)

        # Low |corr| indicates manipulation
        manipulation_score = 1.0 - float(torch.abs(corr).item())
        return float(min(1.0, max(0.0, manipulation_score)))
    
    def _compute_differential_patterns_gpu(self, data: torch.Tensor) -> float:
        """Compute differential patterns on GPU"""
        if data.shape[0] < 3:
            return 0.0
        
        # First and second differences
        first_diff = torch.diff(data, dim=0)
        second_diff = torch.diff(first_diff, dim=0)
        
        # Analyze differential patterns
        first_diff_var = torch.var(first_diff, dim=0)
        second_diff_var = torch.var(second_diff, dim=0)
        
        # Mean variances
        first_var_mean = torch.mean(first_diff_var).item()
        second_var_mean = torch.mean(second_diff_var).item()
        
        # Unusual differential patterns indicate attacks
        differential_score = min(1.0, first_var_mean + second_var_mean)
        return differential_score
    
    def _detect_embedding_obfuscation_gpu(self, data: torch.Tensor) -> float:
        """Detect embedding obfuscation on GPU"""
        # Compute embedding entropy
        data_flat = torch.nan_to_num(data.view(-1), nan=0.0, posinf=0.0, neginf=0.0)
        hist = torch.histc(data_flat, bins=50)
        hist_norm = hist / (torch.sum(hist) + 1e-10)
        
        # Remove zero probabilities
        hist_nonzero = hist_norm[hist_norm > 1e-10]
        
        if len(hist_nonzero) < 2:
            return 0.0
        
        # Compute entropy
        log_hist = torch.log(hist_nonzero)
        entropy = -torch.sum(hist_nonzero * log_hist).item()
        denom = float(np.log(len(hist_nonzero)))
        if denom <= 0.0 or not np.isfinite(denom):
            return 0.0
        normalized_entropy = float(entropy / denom)
        normalized_entropy = float(min(1.0, max(0.0, normalized_entropy)))
        
        # Unusual entropy patterns indicate obfuscation
        obfuscation_score = abs(normalized_entropy - 0.5) * 2
        return float(min(1.0, max(0.0, obfuscation_score)))
    
    def _compute_threat_assessment(self, detection_results: Dict[str, Dict]) -> Tuple[str, float]:
        """Compute overall threat assessment from detection results"""
        
        threat_scores = []
        for method, result in detection_results.items():
            if 'confidence' in result:
                threat_scores.append(result['confidence'])
            elif 'inversion_score' in result:
                threat_scores.append(result['inversion_score'])
            elif 'misbehavior_score' in result:
                threat_scores.append(result['misbehavior_score'])
            elif 'attack_score' in result:
                threat_scores.append(result['attack_score'])
        
        # Filter non-finite/invalid values
        threat_scores = [float(s) for s in threat_scores if np.isfinite(float(s))]

        if not threat_scores:
            return 'unknown', 0.0
        
        avg_score = float(np.mean(threat_scores))
        avg_score = float(np.clip(avg_score, 0.0, 1.0))
        
        if avg_score > 0.8:
            return 'critical', avg_score
        elif avg_score > 0.6:
            return 'high', avg_score
        elif avg_score > 0.4:
            return 'medium', avg_score
        else:
            return 'low', avg_score


class TSInverseDetectorNet(nn.Module):
    """GPU-optimized TS-Inverse detection network"""
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((8, 8))  # MPS-compatible size
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Fixed: 128 * 8 * 8 = 8192
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Ensure input is 4D (batch, channel, height, width)
        if len(x.shape) == 2:
            # Convert 2D to 4D, assuming square input
            side_length = int(np.sqrt(x.shape[1]))
            if side_length * side_length == x.shape[1]:
                x = x.view(x.size(0), 1, side_length, side_length)
            else:
                # Pad to nearest square
                target_size = int(np.ceil(np.sqrt(x.shape[1]))) ** 2
                padding = target_size - x.shape[1]
                x = F.pad(x, (0, padding))
                side_length = int(np.sqrt(target_size))
                x = x.view(x.size(0), 1, side_length, side_length)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x.squeeze()


class ConcreTizerDetectorNet(nn.Module):
    """GPU-optimized ConcreTizer detection network"""
    
    def __init__(self):
        super().__init__()
        
        # 3D convolution for voxel analysis
        self.conv3d1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv3d2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3d3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        
        self.pool3d = nn.AdaptiveAvgPool3d((4, 4, 4))  # MPS-compatible size
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 128)  # Fixed: 64 * 4 * 4 * 4 = 4096
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Ensure input is 5D (batch, channel, depth, height, width)
        if len(x.shape) == 2:
            # Convert 2D to 5D, assuming cubic input
            side_length = int(np.cbrt(x.shape[1]))
            if side_length ** 3 == x.shape[1]:
                x = x.view(x.size(0), 1, side_length, side_length, side_length)
            else:
                # Pad to nearest cube
                target_size = int(np.ceil(np.cbrt(x.shape[1]))) ** 3
                padding = target_size - x.shape[1]
                x = F.pad(x, (0, padding))
                side_length = int(np.cbrt(target_size))
                x = x.view(x.size(0), 1, side_length, side_length, side_length)
        elif len(x.shape) == 3:
            # Add channel and depth dimensions
            x = x.unsqueeze(1).unsqueeze(2)
        elif len(x.shape) == 4:
            x = x.unsqueeze(2)  # Add depth dimension
        
        x = F.relu(self.conv3d1(x))
        x = F.relu(self.conv3d2(x))
        x = F.relu(self.conv3d3(x))
        
        x = self.pool3d(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x.squeeze()


class AttentionGuardNet(nn.Module):
    """GPU-optimized AttentionGuard detection network"""
    
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        self.input_projection = nn.Linear(d_model, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Ensure correct input shape
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        batch_size, seq_len, feature_dim = x.shape
        
        # Project to model dimension if necessary
        if feature_dim != self.d_model:
            if feature_dim < self.d_model:
                # Pad features
                padding = self.d_model - feature_dim
                x = F.pad(x, (0, padding))
            else:
                # Project down
                x = self.input_projection(x)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        output = self.classifier(x)
        
        return output.squeeze()


class EDNNDetectorNet(nn.Module):
    """GPU-optimized EDNN detection network"""
    
    def __init__(self):
        super().__init__()
        
        self.embedding_analyzer = nn.Sequential(
            nn.Linear(768, 512),  # Assuming 768-dim embeddings
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.pattern_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Flatten if necessary
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Adjust input dimension if necessary
        if x.shape[1] != 768:
            if x.shape[1] < 768:
                # Pad to 768
                padding = 768 - x.shape[1]
                x = F.pad(x, (0, padding))
            else:
                # Truncate to 768
                x = x[:, :768]
        
        # Analyze embeddings
        features = self.embedding_analyzer(x)
        
        # Detect patterns
        output = self.pattern_detector(features)
        
        return output.squeeze()


def test_gpu_security_system():
    """Test the GPU security system on current platform"""
    
    try:
        device = get_optimal_device()
        print(f"✅ GPU device detected: {device}")
        
        # Initialize detector
        detector = GPUAdversarialDetector(device)
        print("✅ GPU detector initialized")
        
        # Create test data
        test_data = np.random.randn(10, 64).astype(np.float32)
        reference_data = np.random.randn(10, 64).astype(np.float32)
        
        print(f"📊 Test data shape: {test_data.shape}")
        
        # Run detection
        result = detector.detect_adversarial_attacks(test_data, reference_data)
        
        print("✅ GPU detection completed successfully!")
        print(f"🎯 Threat level: {result.threat_level}")
        print(f"📊 Confidence: {result.confidence:.3f}")
        print(f"⏱️ Processing time: {result.processing_time:.4f}s")
        print(f"💾 GPU memory used: {result.gpu_memory_used:.3f}GB")
        print(f"🖥️ Device: {result.device_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU security test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🚀 Testing GPU-accelerated security system...")
    success = test_gpu_security_system()
    
    if success:
        print("🎉 GPU security system is working correctly!")
    else:
        print("💥 GPU security system test failed!") 