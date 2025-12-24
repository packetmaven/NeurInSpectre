"""
NeurInSpectre Enhanced Z-Score Analysis Module
Implements multivariate, temporal, and robust Z-score analysis.

Key Features:
- Multivariate Z-scores using Mahalanobis distance
- Temporal Z-score analysis with sliding windows  
- Robust Z-scores using Median Absolute Deviation (MAD)
- Statistical significance testing with confidence intervals
- Cross-validation temporal analysis
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv, LinAlgError
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import deque
import logging
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ZScoreResults:
    """Container for Z-score analysis results"""
    univariate_zscores: np.ndarray
    multivariate_zscores: np.ndarray
    robust_zscores: np.ndarray
    temporal_zscores: np.ndarray
    anomaly_flags: np.ndarray
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, float]
    feature_contributions: np.ndarray

class EnhancedZScoreAnalyzer:
    """
    Enhanced Z-Score Analysis for NeurInSpectre
    
    Implements advanced statistical methods based on recent research:
    - Mahalanobis distance for multivariate anomaly detection
    - Temporal sliding window analysis 
    - Robust statistics using MAD
    - Statistical significance testing
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        window_size: int = 100,
        overlap_ratio: float = 0.5,
        robust_threshold: float = 3.0,
        use_robust_covariance: bool = True
    ):
        """
        Initialize Enhanced Z-Score Analyzer
        
        Args:
            confidence_level: Statistical confidence level (default: 0.95)
            window_size: Size of sliding window for temporal analysis
            overlap_ratio: Overlap ratio between consecutive windows
            robust_threshold: Threshold for robust outlier detection
            use_robust_covariance: Whether to use robust covariance estimation
        """
        self.confidence_level = confidence_level
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.robust_threshold = robust_threshold
        self.use_robust_covariance = use_robust_covariance
        
        # Initialize statistical components
        self.scaler = StandardScaler()
        self.covariance_estimator = None
        self.temporal_buffer = deque(maxlen=window_size * 3)
        
        # Statistical parameters
        self.alpha = 1 - confidence_level
        self.z_critical = stats.norm.ppf(1 - self.alpha/2)
        
        logger.info(f"Enhanced Z-Score Analyzer initialized with confidence level: {confidence_level}")
    
    def fit(self, X: np.ndarray) -> 'EnhancedZScoreAnalyzer':
        """
        Fit the analyzer to training data
        
        Args:
            X: Training data matrix (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting Enhanced Z-Score Analyzer on data shape: {X.shape}")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize covariance estimator
        if self.use_robust_covariance:
            self.covariance_estimator = MinCovDet(random_state=42)
        else:
            self.covariance_estimator = EmpiricalCovariance()
        
        try:
            # Robust covariance can emit RuntimeWarnings on ill-conditioned inputs.
            # Treat warnings as instability and fall back to empirical covariance.
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                self.covariance_estimator.fit(X_scaled)
            logger.info("Covariance estimation completed successfully")
        except Exception as e:
            logger.warning(f"Robust covariance estimation failed: {e}. Falling back to empirical.")
            self.covariance_estimator = EmpiricalCovariance()
            self.covariance_estimator.fit(X_scaled)
        
        return self

    def _needs_fit(self) -> bool:
        """Return True if scaler/covariance state isn't fitted yet."""
        scaler_fitted = hasattr(self.scaler, "mean_") and hasattr(self.scaler, "scale_")
        cov_fitted = (
            self.covariance_estimator is not None
            and hasattr(self.covariance_estimator, "covariance_")
            and hasattr(self.covariance_estimator, "location_")
        )
        return not (scaler_fitted and cov_fitted)
    
    def compute_univariate_zscores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute standard univariate Z-scores
        
        Args:
            X: Input data matrix
            
        Returns:
            Z-scores for each feature
        """
        # If user didn't call fit(), fall back to fitting on X (one-shot analysis mode).
        if self._needs_fit():
            self.fit(X)
        X_scaled = self.scaler.transform(X)
        z_scores = np.abs(X_scaled)
        return z_scores
    
    def compute_multivariate_zscores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute multivariate Z-scores using Mahalanobis distance
        Based on 2025 research: "Mahalanobis++: Improving OOD Detection via Feature Normalization"
        
        Args:
            X: Input data matrix
            
        Returns:
            Signed multivariate Z-scores derived from the chi-square CDF of squared
            Mahalanobis distance. Under the null (multivariate normal), these are
            approximately standard normal; large positive values correspond to
            right-tail (outlier) behavior.
        """
        # If user didn't call fit(), fall back to fitting on X (one-shot analysis mode).
        if self._needs_fit():
            self.fit(X)
        X_scaled = self.scaler.transform(X)
        
        try:
            # Get mean and covariance from fitted estimator
            mean = self.covariance_estimator.location_
            cov_matrix = self.covariance_estimator.covariance_
            
            # Compute Mahalanobis distances
            mahal_distances = []
            n_features = X_scaled.shape[1]
            try:
                cov_inv = inv(cov_matrix)
            except LinAlgError:
                cov_inv = inv(cov_matrix + np.eye(n_features) * 1e-6)
            
            for i, sample in enumerate(X_scaled):
                try:
                    distance = mahalanobis(sample, mean, cov_inv)
                    mahal_distances.append(distance)
                except Exception:
                    # Fallback to Euclidean distance if Mahalanobis fails
                    distance = np.linalg.norm(sample - mean)
                    mahal_distances.append(distance)
            
            # For multivariate normal data, squared Mahalanobis distance follows chi-square(df=n_features).
            mahal_squared = np.asarray(mahal_distances, dtype=np.float64) ** 2

            # Map chi-square CDF to a (signed) standard normal score.
            # U = CDF_chi2(D^2) ~ Uniform(0,1) under null => Z = Phi^{-1}(U) ~ Normal(0,1).
            cdf_vals = stats.chi2.cdf(mahal_squared, df=n_features)
            cdf_vals = np.clip(cdf_vals, 1e-12, 1.0 - 1e-12)
            z_scores = stats.norm.ppf(cdf_vals)
            return z_scores
            
        except Exception as e:
            logger.warning(f"Multivariate Z-score computation failed: {e}. Using fallback method.")
            # Fallback to mean of absolute univariate Z-scores
            univariate_z = self.compute_univariate_zscores(X)
            return np.mean(np.abs(univariate_z), axis=1)
    
    def compute_robust_zscores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute robust Z-scores using Median Absolute Deviation (MAD)
        Based on 2024 research: "Median Absolute Deviation for BGP Anomaly Detection"
        
        Args:
            X: Input data matrix
            
        Returns:
            Robust Z-scores using MAD
        """
        # Compute MAD for each feature
        medians = np.median(X, axis=0)
        mad_values = np.median(np.abs(X - medians), axis=0)
        
        # Avoid division by zero
        mad_values = np.where(mad_values == 0, np.finfo(float).eps, mad_values)
        
        # Scale factor for consistency with normal distribution
        # MAD * 1.4826 approximates standard deviation for normal data
        mad_scaled = mad_values * 1.4826
        
        # Compute robust Z-scores
        robust_z_scores = np.abs((X - medians) / mad_scaled)
        
        return robust_z_scores
    
    def compute_temporal_zscores(self, X: np.ndarray, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute temporal Z-scores using sliding window analysis
        Based on 2024 research: "Temporal cross-validation impacts multivariate time series"
        
        Args:
            X: Input data matrix
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            Temporal Z-scores with sliding window analysis
        """
        n_samples, n_features = X.shape
        temporal_z_scores = np.zeros((n_samples, n_features))
        
        # Calculate step size based on overlap ratio
        overlap = float(np.clip(self.overlap_ratio, 0.0, 0.99))
        step_size = max(1, int(self.window_size * (1.0 - overlap)))
        
        for i in range(0, n_samples - self.window_size + 1, step_size):
            # Extract window
            window_end = min(i + self.window_size, n_samples)
            window_data = X[i:window_end]
            
            # Compute local statistics for window
            window_mean = np.mean(window_data, axis=0)
            window_std = np.std(window_data, axis=0)
            
            # Avoid division by zero
            window_std = np.where(window_std == 0, np.finfo(float).eps, window_std)
            
            # Compute Z-scores for points in window
            for j in range(i, window_end):
                if j < n_samples:
                    temporal_z_scores[j] = np.abs((X[j] - window_mean) / window_std)
        
        # Handle remaining samples with last computed statistics
        if n_samples > self.window_size:
            last_window = X[-self.window_size:]
            last_mean = np.mean(last_window, axis=0)
            last_std = np.std(last_window, axis=0)
            last_std = np.where(last_std == 0, np.finfo(float).eps, last_std)
            
            # Fill remaining samples
            remaining_start = ((n_samples - self.window_size) // step_size) * step_size + self.window_size
            for j in range(remaining_start, n_samples):
                temporal_z_scores[j] = np.abs((X[j] - last_mean) / last_std)
        
        return temporal_z_scores
    
    def detect_anomalies(
        self, 
        X: np.ndarray, 
        method: str = 'hybrid',
        threshold: float = 3.0
    ) -> np.ndarray:
        """
        Detect anomalies using enhanced Z-score analysis
        
        Args:
            X: Input data matrix
            method: Detection method ('univariate', 'multivariate', 'robust', 'temporal', 'hybrid')
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            Boolean array indicating anomalies
        """
        if method == 'univariate':
            z_scores = self.compute_univariate_zscores(X)
            anomalies = np.any(z_scores > threshold, axis=1)
        elif method == 'multivariate':
            z_scores = self.compute_multivariate_zscores(X)
            anomalies = z_scores > threshold
        elif method == 'robust':
            z_scores = self.compute_robust_zscores(X)
            anomalies = np.any(z_scores > threshold, axis=1)
        elif method == 'temporal':
            z_scores = self.compute_temporal_zscores(X)
            anomalies = np.any(z_scores > threshold, axis=1)
        elif method == 'hybrid':
            # Combine multiple methods for enhanced detection
            uni_z = self.compute_univariate_zscores(X)
            multi_z = self.compute_multivariate_zscores(X)
            robust_z = self.compute_robust_zscores(X)
            temporal_z = self.compute_temporal_zscores(X)
            
            # Anomaly if any method exceeds threshold
            uni_anomalies = np.any(uni_z > threshold, axis=1)
            multi_anomalies = multi_z > threshold
            robust_anomalies = np.any(robust_z > threshold, axis=1)
            temporal_anomalies = np.any(temporal_z > threshold, axis=1)
            
            # Combine using logical OR (any method detects anomaly)
            anomalies = uni_anomalies | multi_anomalies | robust_anomalies | temporal_anomalies
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return anomalies
    
    def compute_confidence_intervals(self, z_scores: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Compute confidence intervals for Z-scores
        
        Args:
            z_scores: Array of Z-scores
            
        Returns:
            Dictionary with confidence intervals
        """
        mean_z = np.mean(z_scores)
        std_z = np.std(z_scores)
        n = len(z_scores)
        
        # Standard error
        se = std_z / np.sqrt(n)
        
        # Confidence interval
        margin_error = self.z_critical * se
        ci_lower = mean_z - margin_error
        ci_upper = mean_z + margin_error
        
        return {
            'mean': (ci_lower, ci_upper),
            'std': (std_z, std_z),  # Standard deviation doesn't have CI in same sense
            'range': (np.min(z_scores), np.max(z_scores))
        }
    
    def test_statistical_significance(self, z_scores: np.ndarray, null_mean: float = 0.0) -> Dict[str, float]:
        """
        Test statistical significance of Z-scores
        
        Args:
            z_scores: Array of Z-scores
            null_mean: Null hypothesis mean
            
        Returns:
            Dictionary with test statistics
        """
        z = np.asarray(z_scores, dtype=np.float64)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        if z.size < 2 or float(np.std(z)) < 1e-12:
            # Degenerate input: tests are not informative; report a safe, non-NaN structure.
            return {
                't_statistic': 0.0,
                'p_value': 1.0,
                'ks_statistic': 0.0,
                'ks_p_value': 1.0,
                'shapiro_statistic': 0.0,
                'shapiro_p_value': 1.0,
            }

        with warnings.catch_warnings():
            # Some SciPy stats can emit RuntimeWarnings on near-constant data.
            warnings.simplefilter("error", RuntimeWarning)
            try:
                # One-sample t-test
                t_stat, p_value = stats.ttest_1samp(z, null_mean)
            except Exception:
                t_stat, p_value = 0.0, 1.0

            try:
                # Kolmogorov-Smirnov test for normality
                ks_stat, ks_p_value = stats.kstest(z, 'norm')
            except Exception:
                ks_stat, ks_p_value = 0.0, 1.0

            # Shapiro-Wilk test for normality (if sample size allows)
            if len(z) <= 5000:
                try:
                    sw_stat, sw_p_value = stats.shapiro(z)
                except Exception:
                    sw_stat, sw_p_value = 0.0, 1.0
            else:
                # Avoid returning NaN (JSON-unsafe); Shapiro-Wilk is not applicable at this size.
                sw_stat, sw_p_value = 0.0, 1.0
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'ks_statistic': float(ks_stat),
            'ks_p_value': float(ks_p_value),
            'shapiro_statistic': float(sw_stat),
            'shapiro_p_value': float(sw_p_value),
        }
    
    def analyze(
        self, 
        X: np.ndarray, 
        timestamps: Optional[np.ndarray] = None,
        return_detailed: bool = True
    ) -> Union[ZScoreResults, np.ndarray]:
        """
        Comprehensive Z-score analysis
        
        Args:
            X: Input data matrix
            timestamps: Optional timestamps
            return_detailed: Whether to return detailed results
            
        Returns:
            ZScoreResults object or simple anomaly flags
        """
        logger.info(f"Starting comprehensive Z-score analysis on data shape: {X.shape}")
        
        # Compute all types of Z-scores
        univariate_z = self.compute_univariate_zscores(X)
        multivariate_z = self.compute_multivariate_zscores(X)
        robust_z = self.compute_robust_zscores(X)
        temporal_z = self.compute_temporal_zscores(X, timestamps)
        
        # Detect anomalies using hybrid method
        anomaly_flags = self.detect_anomalies(X, method='hybrid', threshold=self.robust_threshold)
        
        if not return_detailed:
            return anomaly_flags
        
        # Compute detailed statistics
        # Use multivariate Z-scores for overall analysis
        confidence_intervals = self.compute_confidence_intervals(multivariate_z)
        statistical_significance = self.test_statistical_significance(multivariate_z)
        
        # Feature contributions (importance of each feature in anomaly detection)
        feature_contributions = np.mean(np.abs(univariate_z), axis=0)
        feature_contributions = feature_contributions / (np.sum(feature_contributions) + 1e-12)  # Normalize
        
        results = ZScoreResults(
            univariate_zscores=univariate_z,
            multivariate_zscores=multivariate_z,
            robust_zscores=robust_z,
            temporal_zscores=temporal_z,
            anomaly_flags=anomaly_flags,
            confidence_intervals=confidence_intervals,
            statistical_significance=statistical_significance,
            feature_contributions=feature_contributions
        )
        
        logger.info(f"Analysis complete. Detected {np.sum(anomaly_flags)} anomalies out of {len(anomaly_flags)} samples")
        
        return results
    
    def get_anomaly_scores(self, X: np.ndarray, method: str = 'multivariate') -> np.ndarray:
        """
        Get anomaly scores for ranking/thresholding
        
        Args:
            X: Input data matrix
            method: Scoring method
            
        Returns:
            Anomaly scores
        """
        if method == 'multivariate':
            return self.compute_multivariate_zscores(X)
        elif method == 'robust':
            robust_z = self.compute_robust_zscores(X)
            return np.mean(robust_z, axis=1)  # Average across features
        elif method == 'temporal':
            temporal_z = self.compute_temporal_zscores(X)
            return np.mean(temporal_z, axis=1)  # Average across features
        else:
            univariate_z = self.compute_univariate_zscores(X)
            return np.mean(univariate_z, axis=1)  # Average across features


class AdaptiveZScoreAnalyzer(EnhancedZScoreAnalyzer):
    """
    Adaptive Z-Score Analyzer with dynamic thresholding
    Extends EnhancedZScoreAnalyzer with adaptive capabilities
    """
    
    def __init__(self, adaptation_rate: float = 0.1, **kwargs):
        """
        Initialize Adaptive Z-Score Analyzer
        
        Args:
            adaptation_rate: Rate of adaptation for dynamic thresholding
            **kwargs: Arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.adaptation_rate = adaptation_rate
        self.adaptive_threshold = self.robust_threshold
        self.threshold_history = deque(maxlen=100)
    
    def update_threshold(self, z_scores: np.ndarray, target_anomaly_rate: float = 0.05):
        """
        Update adaptive threshold based on observed Z-scores
        
        Args:
            z_scores: Recent Z-scores
            target_anomaly_rate: Target anomaly detection rate
        """
        # Calculate threshold that would yield target anomaly rate
        percentile = (1 - target_anomaly_rate) * 100
        new_threshold = np.percentile(z_scores, percentile)
        
        # Adaptive update
        self.adaptive_threshold = (
            (1 - self.adaptation_rate) * self.adaptive_threshold + 
            self.adaptation_rate * new_threshold
        )
        
        self.threshold_history.append(self.adaptive_threshold)
    
    def detect_anomalies_adaptive(self, X: np.ndarray, method: str = 'multivariate') -> np.ndarray:
        """
        Detect anomalies using adaptive threshold
        
        Args:
            X: Input data matrix
            method: Detection method
            
        Returns:
            Boolean array indicating anomalies
        """
        # Get anomaly scores
        scores = self.get_anomaly_scores(X, method)
        
        # Update adaptive threshold
        self.update_threshold(scores)
        
        # Detect anomalies using adaptive threshold
        anomalies = scores > self.adaptive_threshold
        
        return anomalies


def create_enhanced_zscore_analyzer(
    confidence_level: float = 0.95,
    window_size: int = 100,
    robust_threshold: float = 3.0,
    adaptive: bool = False
) -> Union[EnhancedZScoreAnalyzer, AdaptiveZScoreAnalyzer]:
    """
    Factory function to create enhanced Z-score analyzer
    
    Args:
        confidence_level: Statistical confidence level
        window_size: Window size for temporal analysis
        robust_threshold: Threshold for anomaly detection
        adaptive: Whether to use adaptive analyzer
        
    Returns:
        Configured analyzer instance
    """
    if adaptive:
        return AdaptiveZScoreAnalyzer(
            confidence_level=confidence_level,
            window_size=window_size,
            robust_threshold=robust_threshold
        )
    else:
        return EnhancedZScoreAnalyzer(
            confidence_level=confidence_level,
            window_size=window_size,
            robust_threshold=robust_threshold
        )


# Example usage and testing functions
def demonstrate_enhanced_zscore_analysis():
    """Demonstrate the enhanced Z-score analysis capabilities"""
    print("=== NeurInSpectre Enhanced Z-Score Analysis Demo ===")
    
    # Generate sample data with anomalies
    np.random.seed(42)
    n_samples, n_features = 1000, 5
    
    # Normal data
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=int(n_samples * 0.9)
    )
    
    # Anomalous data
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 3,  # Shifted mean
        cov=np.eye(n_features) * 2,    # Different covariance
        size=int(n_samples * 0.1)
    )
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    true_labels = np.hstack([
        np.zeros(len(normal_data)),
        np.ones(len(anomaly_data))
    ])
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    true_labels = true_labels[indices]
    
    print(f"Generated dataset: {X.shape} with {np.sum(true_labels)} true anomalies")
    
    # Create and fit analyzer
    analyzer = create_enhanced_zscore_analyzer(
        confidence_level=0.95,
        window_size=50,
        robust_threshold=2.5
    )
    
    # Split data for training and testing
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_test = true_labels[train_size:]
    
    # Fit analyzer
    analyzer.fit(X_train)
    
    # Analyze test data
    results = analyzer.analyze(X_test, return_detailed=True)
    
    # Print results
    print(f"\nDetected {np.sum(results.anomaly_flags)} anomalies out of {len(results.anomaly_flags)} test samples")
    print(f"True anomalies in test set: {np.sum(y_test)}")
    
    # Calculate performance metrics
    tp = np.sum((results.anomaly_flags == 1) & (y_test == 1))
    fp = np.sum((results.anomaly_flags == 1) & (y_test == 0))
    tn = np.sum((results.anomaly_flags == 0) & (y_test == 0))
    fn = np.sum((results.anomaly_flags == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print("\nPerformance Metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1_score:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    
    print("\nStatistical Significance:")
    for key, value in results.statistical_significance.items():
        print(f"{key}: {value:.6f}")
    
    print("\nConfidence Intervals:")
    for key, (lower, upper) in results.confidence_intervals.items():
        print(f"{key}: [{lower:.3f}, {upper:.3f}]")
    
    print("\nFeature Contributions:")
    for i, contrib in enumerate(results.feature_contributions):
        print(f"Feature {i}: {contrib:.3f}")


if __name__ == "__main__":
    demonstrate_enhanced_zscore_analysis() 