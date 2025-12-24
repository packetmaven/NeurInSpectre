"""
Integrated Security Analysis System
Combines multiple security analysis techniques, including:
- TS-Inverse gradient inversion detection
- ConcreTizer model inversion detection  
- AttentionGuard transformer-based detection
- EDNN attack detection
- DeMarking defense mechanisms
- Neural transport dynamics analysis
- Behavioral pattern analysis
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Import security modules (optional)
try:
    from .adversarial_detection import (
        AdversarialDetector,
    )
    from .evasion_detection import (
        EvasionDetector,
        DeMarkingDefenseDetector,
    )
    HAS_SECURITY_MODULES = True
except ImportError as e:
    logger.debug("Security modules not fully available: %s", e)
    HAS_SECURITY_MODULES = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _to_unit_interval(x: float) -> float:
    """Best-effort clamp to [0,1] for downstream aggregation stability."""
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except Exception:
        return 0.0


def _score_as_security_equivalent(key: str, score: float) -> float:
    """Normalize heterogeneous component scores onto a 'security' axis.

    Convention in this module:
    - `*_confidence` values are *attack/evasion confidence* (higher => more suspicious).
      We convert them to a security-equivalent score via `1 - confidence`.
    - `*_security` / `*_score` values are already security-oriented (higher => safer).

    Returning security-oriented scores ensures threat level / risk math is consistent.
    """
    s = _to_unit_interval(score)
    if str(key).endswith('_confidence'):
        return 1.0 - s
    return s


# ---------------------------------------------------------------------------
# MITRE ATLAS risk mapping (tactics/techniques) for comprehensive scan reports
# ---------------------------------------------------------------------------
_ATLAS_TECH_IDX = None
_ATLAS_PHASE_TO_TACTIC = None


def _get_atlas_indexes():
    """Lazy-load MITRE ATLAS technique index + phase->tactic mapping."""
    global _ATLAS_TECH_IDX, _ATLAS_PHASE_TO_TACTIC
    if _ATLAS_TECH_IDX is not None and _ATLAS_PHASE_TO_TACTIC is not None:
        return _ATLAS_TECH_IDX, _ATLAS_PHASE_TO_TACTIC

    try:
        from ..mitre_atlas.registry import load_stix_atlas_bundle, technique_index, tactic_by_phase_name

        bundle = load_stix_atlas_bundle()
        _ATLAS_TECH_IDX = technique_index(bundle)
        _ATLAS_PHASE_TO_TACTIC = tactic_by_phase_name(bundle)
    except Exception:
        _ATLAS_TECH_IDX = {}
        _ATLAS_PHASE_TO_TACTIC = {}

    return _ATLAS_TECH_IDX, _ATLAS_PHASE_TO_TACTIC


def _atlas_meta(tech_id: str) -> Dict[str, Any]:
    idx, phase_to_tactic = _get_atlas_indexes()
    tech = idx.get(str(tech_id))
    if tech is None:
        return {'id': str(tech_id), 'name': 'Unknown', 'tactics': [], 'url': None}

    tactics: List[str] = []
    for ph in (getattr(tech, "tactic_phase_names", None) or []):
        t = phase_to_tactic.get(ph)
        if t and t.name not in tactics:
            tactics.append(t.name)

    return {'id': str(tech_id), 'name': tech.name, 'tactics': tactics, 'url': getattr(tech, "url", None)}


def _compute_mitre_atlas_risk(
    activation_data: np.ndarray,
    gradient_data: Optional[np.ndarray],
    security_scores: Dict[str, float],
) -> Dict[str, Any]:
    """Compute per-technique and per-tactic risk (0–1) for MITRE ATLAS.

    Notes:
    - This is a **risk proxy** derived from (a) detector confidences and (b) signal-based heuristics.
    - It does NOT prove technique execution; it provides a consistent way to summarize observed signals.
    """
    # 1) Signals-based ATLAS mapping (heuristic) using the same analyzer as `analyze-attack-vectors`.
    signal_risk: Dict[str, float] = {}
    signal_detections: List[Dict[str, Any]] = []
    try:
        from ..cli.attack_vector_analysis import AttackVectorAnalyzer

        data = gradient_data if gradient_data is not None else activation_data
        ava = AttackVectorAnalyzer(verbose=False)
        _ = ava.analyze_data(np.asarray(data))
        rep = ava.generate_report()
        for tid, info in (rep.get("mitre_atlas_coverage") or {}).items():
            try:
                signal_risk[str(tid)] = _to_unit_interval(float(info.get("max_confidence", 0.0)))
            except Exception:
                signal_risk[str(tid)] = 0.0
        signal_detections = list(rep.get("detections") or [])
    except Exception:
        # If the heuristic mapper fails for any reason, we still report detector-derived mapping below.
        signal_risk = {}
        signal_detections = []

    # 2) Detector-to-ATLAS mapping (explicit, limited to techniques we can defensibly map).
    #    These are *signals consistent with* the technique, not definitive attribution.
    detector_map: Dict[str, List[str]] = {
        # Gradient/model inversion family
        "adversarial_ts_inverse_confidence": ["AML.T0024.001"],  # Invert AI Model
        "adversarial_concretizer_confidence": ["AML.T0024.001"],  # Invert AI Model
        # Adversarial example crafting (proxy)
        "adversarial_attention_guard_confidence": ["AML.T0043"],  # Craft Adversarial Data
        # Embedding / privacy leakage indicators (proxy)
        "adversarial_ednn_confidence": ["AML.T0024.000"],  # Infer Training Data Membership
    }

    # Merge into per-technique objects with transparent sources.
    techniques: Dict[str, Dict[str, Any]] = {}

    def _ensure_tid(tid: str) -> Dict[str, Any]:
        tid = str(tid)
        if tid not in techniques:
            meta = _atlas_meta(tid)
            techniques[tid] = {
                "id": tid,
                "name": meta.get("name", "Unknown"),
                "tactics": meta.get("tactics", []),
                "url": meta.get("url"),
                "risk": 0.0,
                "sources": {},  # source_name -> risk (0–1)
            }
        return techniques[tid]

    # Add heuristic signal risks.
    for tid, r in signal_risk.items():
        t = _ensure_tid(tid)
        t["sources"]["signals"] = _to_unit_interval(r)

    # Add detector-derived risks.
    for score_key, atlas_ids in detector_map.items():
        if score_key not in security_scores:
            continue
        v = _to_unit_interval(float(security_scores.get(score_key, 0.0)))
        for tid in atlas_ids:
            t = _ensure_tid(tid)
            t["sources"][score_key] = max(_to_unit_interval(t["sources"].get(score_key, 0.0)), v)

    # Finalize per-technique risk as max of sources.
    for t in techniques.values():
        src = t.get("sources") or {}
        t["risk"] = _to_unit_interval(max([0.0] + [float(x) for x in src.values() if isinstance(x, (int, float))]))

    # Compute per-tactic rollups.
    tactics: Dict[str, Dict[str, Any]] = {}
    for t in techniques.values():
        for tactic in (t.get("tactics") or []):
            entry = tactics.setdefault(
                str(tactic),
                {"tactic": str(tactic), "risk": 0.0, "techniques": []},
            )
            entry["techniques"].append(t["id"])
            entry["risk"] = float(max(float(entry["risk"]), float(t["risk"])))

    # Stable ordering for UI/reporting.
    tech_list = sorted(list(techniques.values()), key=lambda x: (-float(x.get("risk", 0.0)), str(x.get("id"))))
    tactic_list = sorted(list(tactics.values()), key=lambda x: (-float(x.get("risk", 0.0)), str(x.get("tactic"))))

    return {
        "notes": "Risk is a 0–1 proxy derived from detector confidences and signal heuristics (not definitive attribution).",
        "techniques": tech_list,
        "tactics": tactic_list,
        "signal_detections": signal_detections,
    }

@dataclass
class SecurityAssessment:
    """Comprehensive security assessment results"""
    timestamp: float
    overall_threat_level: str
    confidence_score: float
    detected_attacks: List[Dict[str, Any]]
    security_scores: Dict[str, float]
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class ThreatIntelligence:
    """Threat intelligence data structure"""
    threat_type: str
    severity: str
    indicators: List[str]
    mitigation_strategies: List[str]
    affected_components: List[str]


class IntegratedSecurityAnalyzer:
    """
    Integrated Security Analyzer combining available detection techniques.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.detection_history = []
        self.threat_intelligence = []
        
        # Initialize security modules if available
        if HAS_SECURITY_MODULES:
            self.adversarial_detector = AdversarialDetector(self.config.get('adversarial', {}))
            self.evasion_detector = EvasionDetector(self.config.get('evasion', {}))
        else:
            self.adversarial_detector = None
            self.evasion_detector = None
            logger.warning("Security modules not available - using fallback implementations")
        
        # Initialize threat detection components
        self.isolation_forest = None
        if HAS_SKLEARN:
            self.isolation_forest = IsolationForest(
                contamination=self.config.get('contamination', 0.1),
                random_state=42
            )
        
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
        self._anomaly_feature_dim: Optional[int] = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for security analysis"""
        return {
            'adversarial': {
                'ts_inverse_threshold': 0.8,
                'voxel_resolution': 32,
                'max_seq_length': 512,
                'attention_heads': 8,
                'k_neighbors': 5,
                'ednn_threshold': 0.7
            },
            'evasion': {
                'transport_dim': 64,
                'time_window': 100,
                'demarking_window': 50,
                'demarking_threshold': 0.6,
                'pattern_window': 100
            },
            'contamination': 0.1,
            'parallel_processing': True,
            'max_workers': 4,
            'threat_threshold': 0.6
        }
    
    def run_comprehensive_security_scan(self, 
                                      activation_data: np.ndarray,
                                      gradient_data: Optional[np.ndarray] = None,
                                      model_weights: Optional[Dict[str, np.ndarray]] = None,
                                      network_data: Optional[np.ndarray] = None) -> SecurityAssessment:
        """
        Run comprehensive security scan using all available detection methods
        
        Args:
            activation_data: Neural network activation data
            gradient_data: Gradient data for analysis
            model_weights: Model weights for backdoor detection
            network_data: Network flow data for watermarking detection
            
        Returns:
            Comprehensive security assessment
        """
        # Robustness: sanitize non-finite values early so downstream sklearn/numpy
        # operations don't error (especially under warnings-as-errors strict runs).
        activation_data = np.asarray(activation_data)
        if activation_data.ndim == 1:
            activation_data = activation_data.reshape(-1, 1)
        activation_data = np.nan_to_num(activation_data, nan=0.0, posinf=0.0, neginf=0.0)

        if gradient_data is not None:
            gradient_data = np.asarray(gradient_data)
            if gradient_data.ndim == 1:
                gradient_data = gradient_data.reshape(-1, 1)
            gradient_data = np.nan_to_num(gradient_data, nan=0.0, posinf=0.0, neginf=0.0)

        if network_data is not None:
            network_data = np.asarray(network_data)
            network_data = np.nan_to_num(network_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Empty activations: return a safe, low-confidence assessment (no crash, no warnings).
        if activation_data.size == 0 or activation_data.shape[0] == 0:
            return SecurityAssessment(
                timestamp=time.time(),
                overall_threat_level='low',
                confidence_score=0.0,
                detected_attacks=[],
                security_scores={},
                recommendations=["No activation data provided (empty input)."],
                metadata={
                    'scan_duration': 0.0,
                    'threat_score': 0.0,
                    'avg_security_score': None,
                    'input_shapes': {
                        'activation_data': activation_data.shape,
                        'gradient_data': gradient_data.shape if gradient_data is not None else None,
                        'model_weights': {k: v.shape for k, v in model_weights.items()} if model_weights else None,
                        'network_data': network_data.shape if network_data is not None else None,
                    },
                    'config': self.config,
                    'note': 'empty_activation_input',
                },
            )

        logger.info("🔒 Starting comprehensive security scan...")
        
        scan_start_time = time.time()
        detected_attacks = []
        security_scores = {}
        
        # Parallel processing if enabled
        if self.config.get('parallel_processing', True):
            detected_attacks, security_scores = self._run_parallel_detection(
                activation_data, gradient_data, model_weights, network_data
            )
        else:
            detected_attacks, security_scores = self._run_sequential_detection(
                activation_data, gradient_data, model_weights, network_data
            )
        
        # Compute overall assessment
        overall_threat_level = self._compute_overall_threat_level(security_scores)
        threat_score, avg_security_score = self._compute_threat_score_details(security_scores)
        confidence_score = self._compute_confidence_score(security_scores)
        recommendations = self._generate_recommendations(detected_attacks, security_scores)

        # MITRE ATLAS per-technique / per-tactic risk (0–1 risk proxy).
        mitre_atlas_risk = _compute_mitre_atlas_risk(
            activation_data=activation_data,
            gradient_data=gradient_data,
            security_scores=security_scores,
        )
        
        # Create assessment
        assessment = SecurityAssessment(
            timestamp=time.time(),
            overall_threat_level=overall_threat_level,
            confidence_score=confidence_score,
            detected_attacks=detected_attacks,
            security_scores=security_scores,
            recommendations=recommendations,
            metadata={
                'scan_duration': time.time() - scan_start_time,
                'threat_score': float(threat_score),
                'avg_security_score': (float(avg_security_score) if avg_security_score is not None else None),
                'input_shapes': {
                    'activation_data': activation_data.shape,
                    'gradient_data': gradient_data.shape if gradient_data is not None else None,
                    'model_weights': {k: v.shape for k, v in model_weights.items()} if model_weights else None,
                    'network_data': network_data.shape if network_data is not None else None
                },
                'config': self.config,
                'mitre_atlas_risk': mitre_atlas_risk,
            }
        )
        
        # Store in history
        self.detection_history.append(assessment)
        
        logger.info(f"🔒 Security scan completed in {assessment.metadata['scan_duration']:.2f}s")
        logger.info(f"🎯 Overall threat level: {overall_threat_level}")
        logger.info(f"📊 Confidence score: {confidence_score:.3f}")
        
        return assessment
    
    def _run_parallel_detection(self, activation_data: np.ndarray,
                              gradient_data: Optional[np.ndarray],
                              model_weights: Optional[Dict[str, np.ndarray]],
                              network_data: Optional[np.ndarray]) -> Tuple[List[Dict], Dict[str, float]]:
        """Run detection methods in parallel"""
        
        detected_attacks = []
        security_scores = {}
        
        with ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4)) as executor:
            futures = {}
            
            # Submit adversarial detection
            if HAS_SECURITY_MODULES and self.adversarial_detector:
                futures['adversarial'] = executor.submit(
                    self._run_adversarial_detection,
                    activation_data, gradient_data
                )
            
            # Submit evasion detection
            if HAS_SECURITY_MODULES and self.evasion_detector:
                futures['evasion'] = executor.submit(
                    self._run_evasion_detection,
                    activation_data
                )
            
            # Submit model security analysis
            if model_weights:
                futures['model_security'] = executor.submit(
                    self._run_model_security_analysis,
                    model_weights
                )
            
            # Submit network analysis
            if network_data is not None:
                futures['network_security'] = executor.submit(
                    self._run_network_security_analysis,
                    network_data
                )
            
            # Submit anomaly detection
            futures['anomaly'] = executor.submit(
                self._run_anomaly_detection,
                activation_data
            )
            
            # Collect results
            for detection_type, future in futures.items():
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    if result:
                        attacks, scores = result
                        detected_attacks.extend(attacks)
                        security_scores.update(scores)
                except Exception as e:
                    logger.error(f"Detection method {detection_type} failed: {e}")
                    security_scores[f"{detection_type}_error"] = 1.0
        
        return detected_attacks, security_scores
    
    def _run_sequential_detection(self, activation_data: np.ndarray,
                                gradient_data: Optional[np.ndarray],
                                model_weights: Optional[Dict[str, np.ndarray]],
                                network_data: Optional[np.ndarray]) -> Tuple[List[Dict], Dict[str, float]]:
        """Run detection methods sequentially"""
        
        detected_attacks = []
        security_scores = {}
        
        # Adversarial detection
        if HAS_SECURITY_MODULES and self.adversarial_detector:
            try:
                attacks, scores = self._run_adversarial_detection(activation_data, gradient_data)
                detected_attacks.extend(attacks)
                security_scores.update(scores)
            except Exception as e:
                logger.error(f"Adversarial detection failed: {e}")
                security_scores['adversarial_error'] = 1.0
        
        # Evasion detection
        if HAS_SECURITY_MODULES and self.evasion_detector:
            try:
                attacks, scores = self._run_evasion_detection(activation_data)
                detected_attacks.extend(attacks)
                security_scores.update(scores)
            except Exception as e:
                logger.error(f"Evasion detection failed: {e}")
                security_scores['evasion_error'] = 1.0
        
        # Model security analysis
        if model_weights:
            try:
                attacks, scores = self._run_model_security_analysis(model_weights)
                detected_attacks.extend(attacks)
                security_scores.update(scores)
            except Exception as e:
                logger.error(f"Model security analysis failed: {e}")
                security_scores['model_security_error'] = 1.0
        
        # Network security analysis
        if network_data is not None:
            try:
                attacks, scores = self._run_network_security_analysis(network_data)
                detected_attacks.extend(attacks)
                security_scores.update(scores)
            except Exception as e:
                logger.error(f"Network security analysis failed: {e}")
                security_scores['network_security_error'] = 1.0
        
        # Anomaly detection
        try:
            attacks, scores = self._run_anomaly_detection(activation_data)
            detected_attacks.extend(attacks)
            security_scores.update(scores)
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            security_scores['anomaly_error'] = 1.0
        
        return detected_attacks, security_scores
    
    def _run_adversarial_detection(self, activation_data: np.ndarray,
                                 gradient_data: Optional[np.ndarray]) -> Tuple[List[Dict], Dict[str, float]]:
        """Run adversarial detection analysis"""
        
        if not self.adversarial_detector:
            return [], {}
        
        # Use gradient data if available, otherwise use activation data
        analysis_data = gradient_data if gradient_data is not None else activation_data
        
        # Run adversarial detection
        results = self.adversarial_detector.detect_adversarial_samples(analysis_data)
        
        detected_attacks = []
        security_scores = {}
        
        # Process detection results
        for detection_type, detection_result in results.get('detections', {}).items():
            if isinstance(detection_result, dict):
                # Check if attack detected
                is_attack = (
                    detection_result.get('is_attack', False) or
                    detection_result.get('is_inversion_attack', False) or
                    detection_result.get('is_misbehavior', False) or
                    detection_result.get('is_ednn_attack', False)
                )
                
                if is_attack:
                    detected_attacks.append({
                        'type': f'adversarial_{detection_type}',
                        'confidence': detection_result.get('confidence', 
                                     detection_result.get('inversion_score',
                                     detection_result.get('misbehavior_score',
                                     detection_result.get('attack_score', 0.5)))),
                        'threat_level': detection_result.get('threat_level', 'medium'),
                        'details': detection_result
                    })
                
                # Extract security scores
                confidence_key = f'adversarial_{detection_type}_confidence'
                security_scores[confidence_key] = detection_result.get('confidence',
                                                 detection_result.get('inversion_score',
                                                 detection_result.get('misbehavior_score',
                                                 detection_result.get('attack_score', 0.0))))
        
        return detected_attacks, security_scores
    
    def _run_evasion_detection(self, activation_data: np.ndarray) -> Tuple[List[Dict], Dict[str, float]]:
        """Run evasion detection analysis"""
        
        if not self.evasion_detector:
            return [], {}
        
        # Run evasion detection
        evasion_attempts = self.evasion_detector.detect_evasion_attempts(activation_data)
        
        detected_attacks = []
        security_scores = {}
        
        # Process evasion attempts
        for attempt in evasion_attempts:
            detected_attacks.append({
                'type': f"evasion_{attempt.get('type', 'unknown')}",
                'confidence': attempt.get('confidence', 0.0),
                'threat_level': attempt.get('threat_level', 'medium'),
                'details': attempt.get('details', {})
            })
            
            # Add to security scores
            score_key = f"evasion_{attempt.get('type', 'unknown')}_confidence"
            security_scores[score_key] = attempt.get('confidence', 0.0)
        
        return detected_attacks, security_scores
    
    def _run_model_security_analysis(self, model_weights: Dict[str, np.ndarray]) -> Tuple[List[Dict], Dict[str, float]]:
        """Run model security analysis"""
        
        detected_attacks = []
        security_scores = {}
        
        # Analyze model weights for backdoors and vulnerabilities
        for layer_name, weights in model_weights.items():
            # Basic statistical analysis
            weight_stats = self._analyze_weight_statistics(weights)
            
            # Check for suspicious patterns
            if weight_stats['suspicious_patterns'] > 0.7:
                detected_attacks.append({
                    'type': 'model_backdoor',
                    'confidence': weight_stats['suspicious_patterns'],
                    'threat_level': 'high',
                    'details': {
                        'layer': layer_name,
                        'statistics': weight_stats
                    }
                })
            
            security_scores[f'model_{layer_name}_security'] = 1.0 - weight_stats['suspicious_patterns']
        
        return detected_attacks, security_scores
    
    def _analyze_weight_statistics(self, weights: np.ndarray) -> Dict[str, float]:
        """Analyze weight statistics for suspicious patterns"""
        
        # Compute basic statistics
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        skewness = self._compute_skewness(weights.flatten())
        kurtosis = self._compute_kurtosis(weights.flatten())
        
        # Detect suspicious patterns
        suspicious_patterns = 0.0
        
        # Unusual distribution characteristics
        if abs(skewness) > 2.0:
            suspicious_patterns += 0.3
        
        if abs(kurtosis) > 3.0:
            suspicious_patterns += 0.3
        
        # Extreme values
        if std_weight > 2 * abs(mean_weight):
            suspicious_patterns += 0.2
        
        # Regular patterns (potential backdoors)
        if self._detect_regular_patterns(weights):
            suspicious_patterns += 0.2
        
        return {
            'mean': mean_weight,
            'std': std_weight,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'suspicious_patterns': min(1.0, suspicious_patterns)
        }
    
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
    
    def _detect_regular_patterns(self, weights: np.ndarray) -> bool:
        """Detect regular patterns in weights that might indicate backdoors"""
        if len(weights.shape) < 2:
            return False
        
        # Check for repeating patterns
        flattened = weights.flatten()
        
        # Simple pattern detection: check for repeated subsequences
        pattern_length = min(10, len(flattened) // 4)
        
        for i in range(len(flattened) - 2 * pattern_length):
            pattern = flattened[i:i + pattern_length]
            next_pattern = flattened[i + pattern_length:i + 2 * pattern_length]
            
            # Check similarity
            if np.allclose(pattern, next_pattern, rtol=0.1):
                return True
        
        return False
    
    def _run_network_security_analysis(self, network_data: np.ndarray) -> Tuple[List[Dict], Dict[str, float]]:
        """Run network security analysis"""
        
        detected_attacks = []
        security_scores = {}
        
        # Treat network data as inter-packet delays for DeMarking analysis
        if HAS_SECURITY_MODULES and self.evasion_detector:
            try:
                demarking_detector = DeMarkingDefenseDetector()
                result = demarking_detector.detect_watermarking_evasion(network_data)
                
                if result['is_evasion']:
                    detected_attacks.append({
                        'type': 'network_watermarking_evasion',
                        'confidence': result['evasion_score'],
                        'threat_level': result.get('threat_type', 'medium'),
                        'details': result
                    })
                
                security_scores['network_watermarking_security'] = 1.0 - result['evasion_score']
                
            except Exception as e:
                logger.warning(f"Network security analysis failed: {e}")
                security_scores['network_security_error'] = 1.0
        
        return detected_attacks, security_scores
    
    def _run_anomaly_detection(self, activation_data: np.ndarray) -> Tuple[List[Dict], Dict[str, float]]:
        """Run general anomaly detection"""
        
        detected_attacks = []
        security_scores = {}
        
        if not HAS_SKLEARN or self.isolation_forest is None:
            # Fallback: simple statistical anomaly detection
            mean_activation = np.mean(activation_data, axis=0)
            std_activation = np.std(activation_data, axis=0)
            
            # Detect outliers using 3-sigma rule
            outliers = np.abs(activation_data - mean_activation) > 3 * std_activation
            outlier_ratio = np.mean(outliers)
            
            if outlier_ratio > 0.1:  # More than 10% outliers
                detected_attacks.append({
                    'type': 'statistical_anomaly',
                    'confidence': min(1.0, outlier_ratio * 5),
                    'threat_level': 'medium',
                    'details': {'outlier_ratio': outlier_ratio}
                })
            
            security_scores['statistical_anomaly_score'] = 1.0 - outlier_ratio
            
        else:
            # Use Isolation Forest
            try:
                # If the feature dimensionality changes between calls, the cached
                # scaler/model are no longer valid. Reset and retrain on the new shape.
                feature_dim = int(activation_data.shape[1]) if activation_data.ndim > 1 else 1
                if self._anomaly_feature_dim is None:
                    self._anomaly_feature_dim = feature_dim
                elif self._anomaly_feature_dim != feature_dim:
                    logger.info(
                        "Anomaly detector feature_dim changed (%s -> %s); resetting baseline.",
                        self._anomaly_feature_dim,
                        feature_dim,
                    )
                    # Reinitialize components for the new feature dimension.
                    self.scaler = StandardScaler()
                    self.isolation_forest = IsolationForest(
                        contamination=self.config.get('contamination', 0.1),
                        random_state=42,
                    )
                    self.is_trained = False
                    self._anomaly_feature_dim = feature_dim

                # Prepare data
                if not self.is_trained:
                    scaled_data = self.scaler.fit_transform(activation_data)
                    self.isolation_forest.fit(scaled_data)
                    self.is_trained = True
                else:
                    scaled_data = self.scaler.transform(activation_data)
                
                # IMPORTANT: Do not use `predict()` here.
                # In sklearn, `predict()` uses a threshold derived from `contamination`, which can
                # force a fixed outlier fraction even on the same data the model was fit on.
                # Instead, derive outliers from the distribution of `score_samples` using a robust
                # threshold so the outlier ratio is data-driven (not parameter-driven).
                scores = self.isolation_forest.score_samples(scaled_data)  # higher = more normal
                scores = np.asarray(scores, dtype=np.float64).reshape(-1)

                if scores.size < 2 or not np.all(np.isfinite(scores)):
                    outlier_ratio = 0.0
                    max_robust_z = 0.0
                    median_score = float(np.median(scores)) if scores.size else 0.0
                    mad_score = 0.0
                else:
                    median_score = float(np.median(scores))
                    mad_score = float(np.median(np.abs(scores - median_score)))
                    denom = 1.4826 * mad_score + 1e-12
                    robust_z = (median_score - scores) / denom  # higher => more anomalous
                    robust_z = np.nan_to_num(robust_z, nan=0.0, posinf=0.0, neginf=0.0)
                    max_robust_z = float(np.max(robust_z)) if robust_z.size else 0.0

                    z_thresh = float(self.config.get("anomaly_z_threshold", 6.0))
                    outliers = robust_z > z_thresh
                    outlier_ratio = float(np.mean(outliers)) if outliers.size else 0.0

                outlier_ratio = float(np.clip(outlier_ratio, 0.0, 1.0))

                # Report as a security score (higher = safer).
                security_scores['isolation_forest_score'] = float(np.clip(1.0 - outlier_ratio, 0.0, 1.0))

                # Emit an attack only if the outlier fraction is meaningfully high.
                ratio_thresh = float(self.config.get("anomaly_outlier_ratio_threshold", 0.01))
                if outlier_ratio >= ratio_thresh:
                    # Confidence proxy: scale so ~10% outliers => 1.0
                    confidence = float(np.clip(outlier_ratio / 0.10, 0.0, 1.0))
                    detected_attacks.append({
                        'type': 'isolation_forest_anomaly',
                        'confidence': confidence,
                        'threat_level': 'medium',
                        'details': {
                            'outlier_ratio': outlier_ratio,
                            'score_median': median_score,
                            'score_mad': mad_score,
                            'max_robust_z': max_robust_z,
                            'z_threshold': float(self.config.get("anomaly_z_threshold", 6.0)),
                            'ratio_threshold': ratio_thresh,
                        }
                    })
                
            except Exception as e:
                logger.warning(f"Isolation Forest anomaly detection failed: {e}")
                security_scores['anomaly_detection_error'] = 1.0
        
        return detected_attacks, security_scores
    
    def _compute_overall_threat_level(self, security_scores: Dict[str, float]) -> str:
        """Compute overall threat level from security scores"""

        if not security_scores:
            return 'unknown'

        threat_score, avg_security = self._compute_threat_score_details(security_scores)
        if avg_security is None:
            return 'unknown'
        if not np.isfinite(threat_score):
            threat_score = 0.0

        if threat_score > 0.8:
            return 'critical'
        elif threat_score > 0.6:
            return 'high'
        elif threat_score > 0.4:
            return 'medium'
        else:
            return 'low'

    def _compute_threat_score_details(self, security_scores: Dict[str, float]) -> Tuple[float, Optional[float]]:
        """Compute a numeric threat score in [0,1] from per-module security_scores.

        Conventions:
        - For `*_confidence` keys, higher means "more suspicious", so we invert to a security-equivalent
          score with `_score_as_security_equivalent`.
        - For `*_security` / `*_score` keys, higher already means "more secure".

        We aggregate in security space (higher => safer), then return threat_score = 1 - avg_security_score.
        """
        if not security_scores:
            return 0.0, None

        valid_scores = [
            _score_as_security_equivalent(key, score)
            for key, score in security_scores.items()
            if not str(key).endswith('_error')
        ]
        if not valid_scores:
            return 0.0, None

        avg_security_score = float(np.mean(valid_scores))
        threat_score = float(np.clip(1.0 - avg_security_score, 0.0, 1.0))
        return threat_score, avg_security_score
    
    def _compute_confidence_score(self, security_scores: Dict[str, float]) -> float:
        """Compute confidence score for the assessment"""
        
        if not security_scores:
            return 0.0
        
        # Count valid detections vs errors
        total_detections = len(security_scores)
        error_detections = len([key for key in security_scores.keys() 
                              if key.endswith('_error')])
        
        valid_ratio = 1.0 - (error_detections / total_detections)
        
        # Factor in score variance (lower variance = higher confidence)
        valid_scores = [
            _score_as_security_equivalent(key, score)
            for key, score in security_scores.items()
            if not key.endswith('_error')
        ]
        
        if valid_scores:
            score_variance = np.var(valid_scores)
            variance_factor = 1.0 / (1.0 + score_variance)
        else:
            variance_factor = 0.5
        
        confidence = valid_ratio * variance_factor
        return min(1.0, confidence)
    
    def _generate_recommendations(self, detected_attacks: List[Dict], 
                                security_scores: Dict[str, float]) -> List[str]:
        """Generate security recommendations based on analysis"""
        
        recommendations = []
        
        # General recommendations based on detected attacks
        attack_types = set(attack['type'] for attack in detected_attacks)
        
        if any('adversarial' in attack_type for attack_type in attack_types):
            recommendations.append(
                "🛡️ Implement adversarial training and input validation to counter adversarial attacks"
            )
            recommendations.append(
                "🔍 Deploy real-time gradient monitoring and anomaly detection systems"
            )
        
        if any('evasion' in attack_type for attack_type in attack_types):
            recommendations.append(
                "🚫 Strengthen evasion detection mechanisms and behavioral analysis"
            )
            recommendations.append(
                "📊 Implement multi-layer security monitoring and transport dynamics analysis"
            )
        
        if any('model' in attack_type for attack_type in attack_types):
            recommendations.append(
                "🔒 Implement model integrity verification and backdoor detection"
            )
            recommendations.append(
                "🏗️ Use secure model training pipelines and weight validation"
            )
        
        if any('network' in attack_type for attack_type in attack_types):
            recommendations.append(
                "🌐 Deploy network traffic analysis and watermarking defenses"
            )
            recommendations.append(
                "🔐 Implement encrypted communication and flow obfuscation"
            )
        
        if any('anomaly' in attack_type for attack_type in attack_types):
            recommendations.append(
                "📈 Enhance anomaly detection thresholds and monitoring systems"
            )
            recommendations.append(
                "🎯 Implement multi-modal anomaly detection and correlation analysis"
            )
        
        # Security score based recommendations
        low_security_components = [
            key.replace('_confidence', '').replace('_score', '').replace('_security', '')
            for key, score in security_scores.items()
            if (not key.endswith('_error')) and (_score_as_security_equivalent(key, score) < 0.5)
        ]
        
        if low_security_components:
            recommendations.append(
                f"⚠️ Focus security improvements on: {', '.join(set(low_security_components))}"
            )
        
        # Error handling recommendations
        error_components = [
            key.replace('_error', '') 
            for key in security_scores.keys() 
            if key.endswith('_error')
        ]
        
        if error_components:
            recommendations.append(
                f"🔧 Address detection system errors in: {', '.join(error_components)}"
            )
        
        # General security hygiene
        recommendations.extend([
            "📋 Regularly update security detection models and threat intelligence",
            "🔄 Implement continuous security monitoring and logging",
            "👥 Train security team on latest AI threat vectors and mitigation strategies",
            "🧪 Conduct regular security assessments and penetration testing"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations


def generate_security_assessment(scan_result: SecurityAssessment) -> Dict[str, Any]:
    """
    Generate a security assessment report from scan results
    
    Args:
        scan_result: Security assessment from comprehensive scan
        
    Returns:
        Formatted security assessment report
    """
    
    if not isinstance(scan_result, SecurityAssessment):
        return {
            'error': 'Invalid scan result format',
            'status': 'failed'
        }
    
    # Create assessment report
    assessment_report = {
        'executive_summary': {
            'threat_level': scan_result.overall_threat_level,
            'confidence': scan_result.confidence_score,
            'total_attacks_detected': len(scan_result.detected_attacks),
            'scan_timestamp': scan_result.timestamp,
            'scan_duration': scan_result.metadata.get('scan_duration', 0)
        },
        
        'threat_analysis': {
            'detected_attacks': scan_result.detected_attacks,
            'attack_distribution': _analyze_attack_distribution(scan_result.detected_attacks),
            'threat_timeline': _create_threat_timeline(scan_result.detected_attacks)
        },
        
        'security_metrics': {
            'security_scores': scan_result.security_scores,
            'component_analysis': _analyze_security_components(scan_result.security_scores),
            'risk_assessment': _compute_risk_assessment(scan_result.security_scores),
            # MITRE ATLAS (tactics/techniques) risk proxies derived from the scan inputs.
            'mitre_atlas_risk': scan_result.metadata.get('mitre_atlas_risk', {}),
        },
        
        'recommendations': {
            'immediate_actions': [r for r in scan_result.recommendations if '🚨' in r or '⚠️' in r],
            'short_term_improvements': [r for r in scan_result.recommendations if '🛡️' in r or '🔍' in r],
            'long_term_strategy': [r for r in scan_result.recommendations if '📋' in r or '👥' in r]
        },
        
        'technical_details': {
            'scan_configuration': scan_result.metadata.get('config', {}),
            'input_data_analysis': scan_result.metadata.get('input_shapes', {}),
            'detection_methods_used': list(scan_result.security_scores.keys())
        }
    }
    
    return assessment_report


def _analyze_attack_distribution(detected_attacks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze distribution of detected attacks"""
    
    if not detected_attacks:
        return {'total': 0, 'by_type': {}, 'by_threat_level': {}}
    
    # Group by attack type
    attack_types = {}
    threat_levels = {}
    
    for attack in detected_attacks:
        attack_type = attack.get('type', 'unknown')
        threat_level = attack.get('threat_level', 'unknown')
        
        attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
        threat_levels[threat_level] = threat_levels.get(threat_level, 0) + 1
    
    return {
        'total': len(detected_attacks),
        'by_type': attack_types,
        'by_threat_level': threat_levels,
        'avg_confidence': np.mean([a.get('confidence', 0) for a in detected_attacks])
    }


def _create_threat_timeline(detected_attacks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create threat timeline from detected attacks"""
    
    timeline = []
    
    for i, attack in enumerate(detected_attacks):
        timeline.append({
            'sequence': i + 1,
            'attack_type': attack.get('type', 'unknown'),
            'threat_level': attack.get('threat_level', 'unknown'),
            'confidence': attack.get('confidence', 0),
            'relative_time': f"T+{i * 0.1:.1f}s"  # Simulated timeline
        })
    
    return timeline


def _analyze_security_components(security_scores: Dict[str, float]) -> Dict[str, Any]:
    """Analyze security components performance"""
    
    if not security_scores:
        return {'total_components': 0, 'avg_score': 0, 'weakest_components': []}
    
    # Filter out error scores
    valid_scores = {
        k: _score_as_security_equivalent(k, v)
        for k, v in security_scores.items()
        if not k.endswith('_error')
    }
    error_scores = {k: v for k, v in security_scores.items() if k.endswith('_error')}
    
    # Find weakest components
    sorted_scores = sorted(valid_scores.items(), key=lambda x: x[1])
    weakest_components = sorted_scores[:3]  # Top 3 weakest
    
    return {
        'total_components': len(valid_scores),
        'avg_score': np.mean(list(valid_scores.values())) if valid_scores else 0,
        'weakest_components': weakest_components,
        'error_count': len(error_scores),
        'score_distribution': {
            'high_security': len([s for s in valid_scores.values() if s > 0.8]),
            'medium_security': len([s for s in valid_scores.values() if 0.5 <= s <= 0.8]),
            'low_security': len([s for s in valid_scores.values() if s < 0.5])
        }
    }


def _compute_risk_assessment(security_scores: Dict[str, float]) -> Dict[str, Any]:
    """Compute overall risk assessment"""
    
    if not security_scores:
        return {'risk_level': 'unknown', 'risk_score': 0}
    
    # Filter out error scores
    valid_scores = [
        _score_as_security_equivalent(key, score)
        for key, score in security_scores.items()
        if not key.endswith('_error')
    ]
    
    if not valid_scores:
        return {'risk_level': 'unknown', 'risk_score': 0}
    
    # Compute risk score (inverse of security score)
    avg_security = np.mean(valid_scores)
    risk_score = 1.0 - avg_security
    
    # Determine risk level
    if risk_score > 0.8:
        risk_level = 'critical'
    elif risk_score > 0.6:
        risk_level = 'high'
    elif risk_score > 0.4:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'confidence_interval': {
            'lower': max(0, risk_score - np.std(valid_scores)),
            'upper': min(1, risk_score + np.std(valid_scores))
        }
    }


def run_comprehensive_security_scan(activation_data: np.ndarray,
                                   gradient_data: Optional[np.ndarray] = None,
                                   model_weights: Optional[Dict[str, np.ndarray]] = None,
                                   network_data: Optional[np.ndarray] = None,
                                   config: Optional[Dict[str, Any]] = None) -> SecurityAssessment:
    """
    Convenience function for running comprehensive security scan
    
    Args:
        activation_data: Neural network activation data
        gradient_data: Gradient data for analysis
        model_weights: Model weights for backdoor detection
        network_data: Network flow data for watermarking detection
        config: Optional configuration dictionary
        
    Returns:
        Comprehensive security assessment
    """
    analyzer = IntegratedSecurityAnalyzer(config)
    return analyzer.run_comprehensive_security_scan(
        activation_data, gradient_data, model_weights, network_data
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
    print("🔒 Integrated Security Analysis System")
    print("=" * 50)
    
    # Create analyzer
    analyzer = IntegratedSecurityAnalyzer()
    
    # Test with sample data
    test_activations = np.random.randn(100, 64)
    test_gradients = np.random.randn(50, 64)
    test_weights = {'layer1': np.random.randn(64, 32), 'layer2': np.random.randn(32, 10)}
    
    # Run comprehensive scan
    assessment = analyzer.run_comprehensive_security_scan(
        activation_data=test_activations,
        gradient_data=test_gradients,
        model_weights=test_weights
    )
    
    # Generate assessment report
    report = generate_security_assessment(assessment)
    
    print(f"🎯 Threat Level: {assessment.overall_threat_level}")
    print(f"📊 Confidence: {assessment.confidence_score:.3f}")
    print(f"🔍 Attacks Detected: {len(assessment.detected_attacks)}")
    print(f"💡 Recommendations: {len(assessment.recommendations)}") 