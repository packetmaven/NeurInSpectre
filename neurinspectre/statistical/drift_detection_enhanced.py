"""
NeurInSpectre Enhanced Drift Detection Module
Implements advanced statistical drift detection methods
Intended for drift detection on real analysis inputs.

Key Features:
- Hotelling's T² test for multivariate drift detection
- Kolmogorov-Smirnov tests for distribution drift
- Bayesian change point detection with posterior probabilities
- Temporal drift analysis with sliding windows
- Statistical significance testing with Z-score analysis
"""

import numpy as np
from scipy import stats
from scipy.linalg import inv, LinAlgError
from sklearn.covariance import MinCovDet
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import warnings
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class DriftDetectionResults:
    """Container for drift detection results"""
    drift_detected: bool
    drift_score: float
    p_value: float
    confidence_interval: Tuple[float, float]
    change_points: List[int]
    drift_magnitude: float
    statistical_significance: Dict[str, float]
    feature_drift_scores: np.ndarray

class BaseDriftDetector(ABC):
    """Base class for drift detectors"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    @abstractmethod
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftDetectionResults:
        """Detect drift between reference and current data"""
        pass

class HotellingT2DriftDetector(BaseDriftDetector):
    """
    Hotelling's T² test for multivariate drift detection
    Based on 2024 research: "Temporal cross-validation impacts multivariate time series"
    """
    
    def __init__(self, confidence_level: float = 0.95, use_robust_covariance: bool = True):
        super().__init__(confidence_level)
        self.use_robust_covariance = use_robust_covariance
        self._last_p_value_method: str = "f"
        
    def compute_hotelling_t2(
        self, 
        X1: np.ndarray, 
        X2: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute Hotelling's T² statistic
        
        Args:
            X1: Reference data matrix (n1, p)
            X2: Current data matrix (n2, p)
            
        Returns:
            T² statistic, F-statistic, p-value
        """
        n1, p = X1.shape
        n2, _ = X2.shape

        # Need at least 2 samples per group to form a covariance estimate.
        if n1 < 2 or n2 < 2 or (n1 + n2 - 2) <= 0:
            self._last_p_value_method = "insufficient_samples"
            return 0.0, 0.0, 1.0
        
        # Compute sample means
        mean1 = np.mean(X1, axis=0)
        mean2 = np.mean(X2, axis=0)
        
        # Compute pooled covariance matrix
        if self.use_robust_covariance and min(n1, n2) > p:
            try:
                # Use robust covariance estimation
                combined_data = np.vstack([X1, X2])
                cov_estimator = MinCovDet(random_state=42)
                # Some MinCovDet runs can emit RuntimeWarnings (non-monotone determinant);
                # treat that as an unstable fit and fall back to the classical pooled covariance.
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    cov_estimator.fit(combined_data)
                S_pooled = cov_estimator.covariance_
            except Exception as e:
                logger.warning(f"Robust covariance estimation failed: {e}. Using empirical.")
                S1 = np.cov(X1, rowvar=False, bias=False)
                S2 = np.cov(X2, rowvar=False, bias=False)
                S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
        else:
            # Standard pooled covariance
            S1 = np.cov(X1, rowvar=False, bias=False)
            S2 = np.cov(X2, rowvar=False, bias=False)
            S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
        
        # Compute T² statistic
        try:
            S_inv = inv(S_pooled)
            diff = mean1 - mean2
            T2 = (n1 * n2) / (n1 + n2) * np.dot(np.dot(diff, S_inv), diff)
        except LinAlgError:
            # Regularize covariance matrix if singular
            S_regularized = S_pooled + np.eye(p) * 1e-6
            S_inv = inv(S_regularized)
            diff = mean1 - mean2
            T2 = (n1 * n2) / (n1 + n2) * np.dot(np.dot(diff, S_inv), diff)
        
        # Convert to an F-statistic when degrees of freedom are valid; otherwise use a
        # chi-square approximation (T² ~ χ²_p) to avoid invalid/NaN p-values for small samples.
        df1 = p
        df2 = n1 + n2 - p - 1
        denom = float((n1 + n2 - 2) * p)
        if df2 > 0 and denom > 0.0:
            F_stat = float(T2) * float(df2) / denom
            p_value = float(1.0 - stats.f.cdf(F_stat, df1, df2))
            self._last_p_value_method = "f"
        else:
            F_stat = 0.0
            p_value = float(1.0 - stats.chi2.cdf(float(T2), df=p))
            self._last_p_value_method = "chi2_approx"

        p_value = float(np.clip(p_value, 0.0, 1.0))
        
        return T2, F_stat, p_value
    
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftDetectionResults:
        """
        Detect drift using Hotelling's T² test
        
        Args:
            reference_data: Reference data matrix
            current_data: Current data matrix
            
        Returns:
            DriftDetectionResults object
        """
        logger.info(f"Hotelling's T² drift detection: ref={reference_data.shape}, cur={current_data.shape}")
        
        # Compute Hotelling's T² test
        T2, F_stat, p_value = self.compute_hotelling_t2(reference_data, current_data)
        
        # Determine if drift is detected
        drift_detected = p_value < self.alpha
        
        # Acceptance region under the null (at the configured confidence level):
        # drift indicated when T² > T²_critical.
        n1, p = reference_data.shape
        n2, _ = current_data.shape
        df1 = p
        df2 = n1 + n2 - p - 1
        
        # Critical value under the same p-value method used above.
        if getattr(self, "_last_p_value_method", "f") == "f" and df2 > 0:
            f_critical = stats.f.ppf(1 - self.alpha, df1, df2)
            T2_critical = f_critical * (n1 + n2 - 2) * p / (n1 + n2 - p - 1)
        else:
            # Chi-square approximation for small-sample/invalid df2 regime.
            T2_critical = float(stats.chi2.ppf(1 - self.alpha, df=p))
        
        ci_lower = 0.0
        ci_upper = float(T2_critical)
        
        # Feature-wise drift scores: standardized mean differences (robust, deterministic).
        feature_drift_scores = []
        for i in range(reference_data.shape[1]):
            x1 = reference_data[:, i]
            x2 = current_data[:, i]
            m1 = float(np.mean(x1))
            m2 = float(np.mean(x2))
            # Pooled std (ddof=1). If too few samples, fall back to a safe denominator.
            s1 = float(np.std(x1, ddof=1)) if x1.size >= 2 else 0.0
            s2 = float(np.std(x2, ddof=1)) if x2.size >= 2 else 0.0
            denom = 0.0
            if x1.size >= 2 and x2.size >= 2 and (x1.size + x2.size - 2) > 0:
                sp2 = (((x1.size - 1) * (s1 ** 2)) + ((x2.size - 1) * (s2 ** 2))) / float(x1.size + x2.size - 2)
                denom = float(np.sqrt(max(0.0, sp2)))
            else:
                # With tiny samples, use the combined (population) std as a reasonable scale.
                combined = np.concatenate([x1, x2], axis=0)
                denom = float(np.std(combined, ddof=0))
            denom = denom if denom > 1e-12 else 1.0
            feature_drift_scores.append(abs(m1 - m2) / denom)
        feature_drift_scores = np.array(feature_drift_scores)
        
        # Statistical significance measures
        statistical_significance = {
            'hotelling_t2': T2,
            'f_statistic': F_stat,
            'p_value': p_value,
            'p_value_method': getattr(self, "_last_p_value_method", "f"),
            'degrees_freedom_1': df1,
            'degrees_freedom_2': df2,
            't2_critical': float(T2_critical),
        }
        
        return DriftDetectionResults(
            drift_detected=drift_detected,
            drift_score=T2,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            change_points=[],  # Not applicable for two-sample test
            drift_magnitude=T2,
            statistical_significance=statistical_significance,
            feature_drift_scores=feature_drift_scores
        )

class KolmogorovSmirnovDriftDetector(BaseDriftDetector):
    """
    Kolmogorov-Smirnov test for distribution drift detection
    """
    
    def __init__(self, confidence_level: float = 0.95, multivariate_method: str = 'energy'):
        super().__init__(confidence_level)
        self.multivariate_method = multivariate_method
        
    def ks_test_multivariate(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        *,
        return_details: bool = False,
        n_permutations: int = 1000,
        random_state: Optional[int] = 42,
    ) -> Union[Tuple[float, float], Tuple[float, float, float]]:
        """
        Multivariate two-sample test using energy statistics (energy distance).
        (Note: this is not a literal multivariate KS test; it is an energy-based
        two-sample test commonly used for distribution drift detection.)
        
        Args:
            X1: Reference data
            X2: Current data
            
        Returns:
            (statistic, p_value) or (statistic, p_value, critical_value) if return_details=True.
        """
        from scipy.spatial.distance import pdist, squareform
        
        n1, n2 = len(X1), len(X2)
        
        if self.multivariate_method == 'energy':
            # Energy statistics approach
            # Compute pairwise distances
            combined = np.vstack([X1, X2])
            distances = squareform(pdist(combined))
            
            # Energy statistic
            def energy_stat(data1_idx, data2_idx):
                d11 = distances[np.ix_(data1_idx, data1_idx)]
                d22 = distances[np.ix_(data2_idx, data2_idx)]
                d12 = distances[np.ix_(data1_idx, data2_idx)]

                def mean_offdiag(d: np.ndarray) -> float:
                    n = d.shape[0]
                    if n <= 1:
                        return 0.0
                    return float((np.sum(d) - np.trace(d)) / (n * (n - 1)))

                term1 = mean_offdiag(d11)
                term2 = mean_offdiag(d22)
                term3 = float(np.mean(d12))

                stat = 2.0 * term3 - term1 - term2
                return float(max(0.0, stat))
            
            idx1 = np.arange(n1)
            idx2 = np.arange(n1, n1 + n2)
            
            observed_stat = energy_stat(idx1, idx2)
            
            # Permutation test for p-value / critical value
            rng = np.random.default_rng(random_state)
            permutation_stats = []
            
            for _ in range(n_permutations):
                combined_idx = rng.permutation(n1 + n2)
                perm_idx1 = combined_idx[:n1]
                perm_idx2 = combined_idx[n1:]
                perm_stat = energy_stat(perm_idx1, perm_idx2)
                permutation_stats.append(perm_stat)

            perm = np.asarray(permutation_stats, dtype=np.float64)
            p_value = float((1.0 + np.sum(perm >= observed_stat)) / (n_permutations + 1.0))
            critical_value = float(np.quantile(perm, 1.0 - self.alpha))

            if return_details:
                return observed_stat, p_value, critical_value
            return observed_stat, p_value
        else:
            # Fallback to univariate KS test on principal component
            from sklearn.decomposition import PCA
            
            # Project to first principal component
            pca = PCA(n_components=1)
            X1_proj = pca.fit_transform(X1).flatten()
            X2_proj = pca.transform(X2).flatten()
            
            # Standard KS test
            ks_stat, p_value = stats.ks_2samp(X1_proj, X2_proj)
            # Approximate two-sample KS critical value (two-sided).
            c_alpha = np.sqrt(-0.5 * np.log(self.alpha / 2.0))
            critical_value = float(c_alpha * np.sqrt((n1 + n2) / (n1 * n2)))

            if return_details:
                return float(ks_stat), float(p_value), critical_value
            return float(ks_stat), float(p_value)
    
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftDetectionResults:
        """
        Detect drift using Kolmogorov-Smirnov test
        
        Args:
            reference_data: Reference data matrix
            current_data: Current data matrix
            
        Returns:
            DriftDetectionResults object
        """
        logger.info(f"KS drift detection: ref={reference_data.shape}, cur={current_data.shape}")
        
        # Multivariate test + critical value for the configured method
        ks_stat, p_value, critical_value = self.ks_test_multivariate(
            reference_data,
            current_data,
            return_details=True,
        )
        
        # Determine if drift is detected
        drift_detected = p_value < self.alpha
        
        # Feature-wise KS tests
        feature_drift_scores = []
        for i in range(reference_data.shape[1]):
            feature_ks_stat, _ = stats.ks_2samp(reference_data[:, i], current_data[:, i])
            feature_drift_scores.append(feature_ks_stat)
        feature_drift_scores = np.array(feature_drift_scores)
        
        # Acceptance region under the null (at the configured confidence level).
        # Drift indicated when statistic > critical_value.
        n1, n2 = len(reference_data), len(current_data)
        ci_lower = 0.0
        ci_upper = float(critical_value)
        
        # Statistical significance measures
        statistical_significance = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'critical_value': critical_value,
            'sample_sizes': (n1, n2)
        }
        
        return DriftDetectionResults(
            drift_detected=drift_detected,
            drift_score=ks_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            change_points=[],
            drift_magnitude=ks_stat,
            statistical_significance=statistical_significance,
            feature_drift_scores=feature_drift_scores
        )

class BayesianChangePointDetector(BaseDriftDetector):
    """
    Bayesian change point detection with posterior probabilities
    """
    
    def __init__(self, confidence_level: float = 0.95, prior_strength: float = 1.0):
        super().__init__(confidence_level)
        self.prior_strength = prior_strength
        
    def compute_posterior_probabilities(self, data: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Compute posterior probabilities of change points
        
        Args:
            data: Time series data
            
        Returns:
            Posterior probabilities and detected change points
        """
        n_samples, n_features = data.shape

        posterior_probs = np.zeros(n_samples, dtype=np.float64)

        # Sliding window approach for change point detection
        window_size = min(50, max(2, n_samples // 4))
        if n_samples < (2 * window_size + 1):
            return posterior_probs, []

        # Work in log-space for numerical stability; final normalization is a softmax.
        log_scores = np.full(n_samples, -np.inf, dtype=np.float64)
        log_prior = -np.log(n_samples)

        def log_likelihood(data_points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
            diff = data_points - mean
            cov_inv = inv(cov)
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                return float("-inf")

            quad = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
            return float(-0.5 * (np.sum(quad) + diff.shape[0] * (log_det + n_features * np.log(2.0 * np.pi))))

        for t in range(window_size, n_samples - window_size):
            before = data[t - window_size : t]
            after = data[t : t + window_size]

            try:
                mean_before = np.mean(before, axis=0)
                cov_before = np.cov(before, rowvar=False) + np.eye(n_features) * 1e-6

                mean_after = np.mean(after, axis=0)
                cov_after = np.cov(after, rowvar=False) + np.eye(n_features) * 1e-6

                combined = np.vstack([before, after])
                mean_combined = np.mean(combined, axis=0)
                cov_combined = np.cov(combined, rowvar=False) + np.eye(n_features) * 1e-6

                ll_before = log_likelihood(before, mean_before, cov_before)
                ll_after = log_likelihood(after, mean_after, cov_after)
                ll_change = ll_before + ll_after

                ll_null = log_likelihood(combined, mean_combined, cov_combined)
                log_bayes_factor = ll_change - ll_null

                log_scores[t] = float(self.prior_strength) * float(log_bayes_factor) + log_prior
            except Exception:
                # Fallback to simple variance-based score (still produces a reasonable peak score).
                var_before = np.var(before, axis=0)
                var_after = np.var(after, axis=0)
                var_ratio = float(np.mean(var_after / (var_before + 1e-8)))
                log_scores[t] = float(self.prior_strength) * float(abs(np.log(var_ratio))) + log_prior

        finite = np.isfinite(log_scores)
        if not np.any(finite):
            return posterior_probs, []

        max_log = float(np.max(log_scores[finite]))
        weights = np.zeros(n_samples, dtype=np.float64)
        weights[finite] = np.exp(log_scores[finite] - max_log)

        total = float(np.sum(weights))
        if total > 0:
            posterior_probs = weights / total

        threshold = float(np.percentile(posterior_probs, 95))
        change_points = np.where(posterior_probs > threshold)[0].tolist()
        return posterior_probs, change_points
    
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftDetectionResults:
        """
        Detect drift using Bayesian change point detection
        
        Args:
            reference_data: Reference data matrix
            current_data: Current data matrix
            
        Returns:
            DriftDetectionResults object
        """
        logger.info(f"Bayesian change point detection: ref={reference_data.shape}, cur={current_data.shape}")
        
        # Combine data for sequential analysis
        combined_data = np.vstack([reference_data, current_data])
        
        # Compute posterior probabilities
        posterior_probs, change_points = self.compute_posterior_probabilities(combined_data)
        
        # Check if change points occur in the transition region
        transition_start = max(0, len(reference_data) - 10)
        transition_end = min(len(combined_data) - 1, len(reference_data) + 10)
        
        transition_change_points = [
            cp for cp in change_points 
            if transition_start <= cp <= transition_end
        ]
        
        # Drift detected if change points found in transition region
        drift_detected = len(transition_change_points) > 0
        
        # Drift score as maximum posterior probability in transition region
        if drift_detected:
            transition_probs = posterior_probs[transition_start:transition_end + 1]
            drift_score = float(np.max(transition_probs))
        else:
            drift_score = 0.0
        
        # Approximate p-value based on posterior probability
        p_value = 1 - drift_score if drift_detected else 1.0
        
        # Feature-wise analysis (simplified)
        feature_drift_scores = []
        for i in range(reference_data.shape[1]):
            ref_mean = np.mean(reference_data[:, i])
            cur_mean = np.mean(current_data[:, i])
            ref_std = np.std(reference_data[:, i])
            cur_std = np.std(current_data[:, i])
            
            # Normalized difference
            mean_diff = abs(cur_mean - ref_mean) / (ref_std + 1e-8)
            std_ratio = abs(np.log((cur_std + 1e-8) / (ref_std + 1e-8)))
            
            feature_drift_scores.append(mean_diff + std_ratio)
        feature_drift_scores = np.array(feature_drift_scores)
        
        # Acceptance region under the null (heuristic): posterior mass above the 95th percentile
        # is considered a change-point candidate.
        threshold = float(np.percentile(posterior_probs, 95))
        ci_lower = 0.0
        ci_upper = threshold
        
        # Statistical significance measures
        statistical_significance = {
            'max_posterior_prob': drift_score,
            'n_change_points': len(change_points),
            'transition_change_points': len(transition_change_points),
            'mean_posterior_prob': float(np.mean(posterior_probs)),
            'cp_threshold_p95': threshold,
            'transition_window': (transition_start, transition_end),
        }
        
        return DriftDetectionResults(
            drift_detected=drift_detected,
            drift_score=drift_score,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            change_points=change_points,
            drift_magnitude=drift_score,
            statistical_significance=statistical_significance,
            feature_drift_scores=feature_drift_scores
        )

class EnhancedDriftDetector:
    """
    Enhanced drift detector combining multiple methods
    """
    
    def __init__(self, confidence_level: float = 0.95, methods: List[str] = None):
        """
        Initialize enhanced drift detector
        
        Args:
            confidence_level: Statistical confidence level
            methods: List of methods to use ['hotelling', 'ks', 'bayesian']
        """
        self.confidence_level = confidence_level
        
        if methods is None:
            methods = ['hotelling', 'ks', 'bayesian']
        
        self.detectors = {}
        
        if 'hotelling' in methods:
            self.detectors['hotelling'] = HotellingT2DriftDetector(confidence_level)
        if 'ks' in methods:
            self.detectors['ks'] = KolmogorovSmirnovDriftDetector(confidence_level)
        if 'bayesian' in methods:
            self.detectors['bayesian'] = BayesianChangePointDetector(confidence_level)
    
    def detect_drift_ensemble(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray
    ) -> Dict[str, DriftDetectionResults]:
        """
        Detect drift using ensemble of methods
        
        Args:
            reference_data: Reference data matrix
            current_data: Current data matrix
            
        Returns:
            Dictionary of results from each method
        """
        results = {}
        
        for name, detector in self.detectors.items():
            try:
                result = detector.detect_drift(reference_data, current_data)
                results[name] = result
                logger.info(f"{name} drift detection: {result.drift_detected} (score: {result.drift_score:.4f})")
            except Exception as e:
                logger.error(f"Error in {name} drift detection: {e}")
                # Create dummy result
                results[name] = DriftDetectionResults(
                    drift_detected=False,
                    drift_score=0.0,
                    p_value=1.0,
                    confidence_interval=(0.0, 0.0),
                    change_points=[],
                    drift_magnitude=0.0,
                    statistical_significance={},
                    feature_drift_scores=np.zeros(reference_data.shape[1])
                )
        
        return results
    
    def get_consensus_result(self, results: Dict[str, DriftDetectionResults]) -> DriftDetectionResults:
        """
        Get consensus result from multiple detectors
        
        Args:
            results: Dictionary of results from different methods
            
        Returns:
            Consensus drift detection result
        """
        # Count number of methods detecting drift
        drift_votes = sum(1 for result in results.values() if result.drift_detected)
        total_methods = len(results)
        
        # Consensus: drift detected if majority of methods agree
        consensus_drift = drift_votes > total_methods / 2
        
        # Combine evidence via p-values (more comparable than raw drift scores across methods).
        p_values: List[float] = []
        for r in results.values():
            p = float(r.p_value)
            if not np.isfinite(p):
                p = 1.0
            p_values.append(float(np.clip(p, 0.0, 1.0)))

        consensus_p_value = float(np.mean(p_values)) if p_values else 1.0
        consensus_score = float(np.mean([1.0 - p for p in p_values])) if p_values else 0.0

        # Consensus score lives in [0,1] by construction.
        consensus_ci = (0.0, 1.0)
        
        # Combine change points
        all_change_points = []
        for result in results.values():
            all_change_points.extend(result.change_points)
        unique_change_points = sorted(list(set(all_change_points)))
        
        # Average feature drift scores
        feature_scores = np.array([result.feature_drift_scores for result in results.values()])
        consensus_feature_scores = np.mean(feature_scores, axis=0)
        
        # Combined statistical significance
        consensus_stats = {
            'drift_votes': drift_votes,
            'total_methods': total_methods,
            'consensus_strength': drift_votes / total_methods,
            'individual_results': {name: result.statistical_significance for name, result in results.items()}
        }
        
        return DriftDetectionResults(
            drift_detected=consensus_drift,
            drift_score=consensus_score,
            p_value=consensus_p_value,
            confidence_interval=consensus_ci,
            change_points=unique_change_points,
            drift_magnitude=consensus_score,
            statistical_significance=consensus_stats,
            feature_drift_scores=consensus_feature_scores
        )


def create_enhanced_drift_detector(
    confidence_level: float = 0.95,
    methods: List[str] = None
) -> EnhancedDriftDetector:
    """
    Factory function to create enhanced drift detector
    
    Args:
        confidence_level: Statistical confidence level
        methods: List of detection methods to use
        
    Returns:
        Configured drift detector
    """
    return EnhancedDriftDetector(confidence_level, methods)


# Example usage and testing
def demonstrate_enhanced_drift_detection():
    """Demonstrate enhanced drift detection capabilities"""
    print("=== NeurInSpectre Enhanced Drift Detection Demo ===")
    
    # Generate sample data with drift
    np.random.seed(42)
    n_samples, n_features = 500, 4
    
    # Reference data (stable)
    reference_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_samples
    )
    
    # Current data with drift (shifted mean and different covariance)
    drift_mean = np.array([1.5, -1.0, 0.5, -0.8])
    drift_cov = np.array([
        [1.5, 0.3, 0.1, 0.2],
        [0.3, 2.0, 0.2, 0.1],
        [0.1, 0.2, 0.8, 0.3],
        [0.2, 0.1, 0.3, 1.2]
    ])
    
    current_data = np.random.multivariate_normal(
        mean=drift_mean,
        cov=drift_cov,
        size=n_samples
    )
    
    print(f"Reference data: {reference_data.shape}")
    print(f"Current data: {current_data.shape}")
    print(f"True drift: Mean shift = {drift_mean}")
    
    # Create enhanced drift detector
    detector = create_enhanced_drift_detector(
        confidence_level=0.95,
        methods=['hotelling', 'ks', 'bayesian']
    )
    
    # Detect drift using all methods
    results = detector.detect_drift_ensemble(reference_data, current_data)
    
    # Get consensus result
    consensus = detector.get_consensus_result(results)
    
    print("\n=== Individual Method Results ===")
    for method, result in results.items():
        print(f"{method.upper()}:")
        print(f"  Drift Detected: {result.drift_detected}")
        print(f"  Drift Score: {result.drift_score:.4f}")
        print(f"  P-value: {result.p_value:.6f}")
        print(f"  Confidence Interval: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        print(f"  Feature Drift Scores: {result.feature_drift_scores}")
        print()
    
    print("=== Consensus Result ===")
    print(f"Consensus Drift Detected: {consensus.drift_detected}")
    print(f"Consensus Score: {consensus.drift_score:.4f}")
    print(f"Consensus P-value: {consensus.p_value:.6f}")
    print(f"Consensus Strength: {consensus.statistical_significance['consensus_strength']:.3f}")
    print(f"Feature Drift Scores: {consensus.feature_drift_scores}")
    
    if consensus.change_points:
        print(f"Detected Change Points: {consensus.change_points}")


if __name__ == "__main__":
    demonstrate_enhanced_drift_detection() 