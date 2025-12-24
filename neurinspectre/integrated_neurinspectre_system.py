#!/usr/bin/env python3
"""
Integrated NeurInSpectre System
Complete integration module for all NeurInSpectre security components
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class IntegratedNeurInSpectre:
    """
    Integrated NeurInSpectre System - Main orchestrator for all security modules
    """
    
    def __init__(self, sensitivity_profile: str = 'adaptive'):
        """
        Initialize the integrated system
        
        Args:
            sensitivity_profile: Detection sensitivity ('low', 'medium', 'high', 'adaptive')
        """
        self.sensitivity_profile = sensitivity_profile
        self.device = self._detect_device()
        self.initialized = False
        
        # Initialize components
        self.adversarial_detector = None
        self.gradient_analyzer = None
        self.obfuscation_detector = None
        self.attack_patterns = {}
        self._integrated_security: Optional[object] = None
        
        logger.info(f"Initialized IntegratedNeurInSpectre with {sensitivity_profile} sensitivity")
        
    def _detect_device(self) -> torch.device:
        """Detect available compute device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def initialize_components(self):
        """Initialize all security components"""
        try:
            # Prefer the real integrated security pipeline if available.
            # This keeps behavior consistent with the rest of the package while
            # retaining local fallbacks for environments missing optional deps.
            try:
                from .security.integrated_security import IntegratedSecurityAnalyzer

                self._integrated_security = IntegratedSecurityAnalyzer(config={
                    'parallel_processing': False,
                })
                self.adversarial_detector = None
                self.gradient_analyzer = None
                self.obfuscation_detector = None
                logger.info("Initialized IntegratedSecurityAnalyzer backend")
            except Exception as e:
                logger.debug("IntegratedSecurityAnalyzer unavailable (%s); using local fallbacks", e)

                # Initialize adversarial detection (fallback)
                self.adversarial_detector = AdversarialDetector(
                    device=self.device,
                    sensitivity=self.sensitivity_profile
                )
                
                # Initialize gradient analysis (fallback)
                self.gradient_analyzer = GradientAnalyzer(
                    device=self.device
                )
                
                # Initialize obfuscation detection (fallback)
                self.obfuscation_detector = ObfuscationDetector(
                    device=self.device
                )
            
            self.initialized = True
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            self.initialized = False
    
    def detect_adversarial_examples(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect adversarial examples in input data
        
        Args:
            data: Input data array
            
        Returns:
            Detection results dictionary
        """
        if not self.initialized:
            self.initialize_components()
            
        # If we have the real integrated backend, use it.
        if self._integrated_security is not None:
            try:
                assessment = self._integrated_security.run_comprehensive_security_scan(
                    activation_data=np.asarray(data),
                    gradient_data=np.asarray(data) if getattr(data, "ndim", 0) >= 2 else None,
                )
                adv_attacks = [
                    a for a in (assessment.detected_attacks or [])
                    if str(a.get('type', '')).startswith('adversarial_')
                ]
                is_adv = len(adv_attacks) > 0
                adv_conf = float(max((a.get('confidence', 0.0) for a in adv_attacks), default=0.0))
                # Best-effort dominant type (highest confidence)
                adv_type = 'none'
                if is_adv:
                    top = max(adv_attacks, key=lambda a: float(a.get('confidence', 0.0)))
                    adv_type = str(top.get('type', 'mixed')).replace('adversarial_', '') or 'mixed'
                return {
                    'is_adversarial': bool(is_adv),
                    # For callers, confidence is "attack confidence" (not scan reliability)
                    'confidence': float(adv_conf),
                    'attack_type': adv_type if is_adv else 'none',
                    'severity': str(assessment.overall_threat_level),
                }
            except Exception as e:
                logger.error("Integrated adversarial scan failed: %s", e)
                # Fall through to local fallback if present.

        if self.adversarial_detector is None:
            return {'error': 'Adversarial detector not initialized'}
        
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data_tensor = torch.from_numpy(data).float().to(self.device)
        else:
            data_tensor = data.to(self.device)
        
        # Perform detection
        try:
            detection_results = self.adversarial_detector.detect(data_tensor)
            return {
                'is_adversarial': detection_results['is_adversarial'],
                'confidence': detection_results['confidence'],
                'attack_type': detection_results.get('attack_type', 'unknown'),
                'severity': detection_results.get('severity', 'medium')
            }
        except Exception as e:
            logger.error(f"Adversarial detection failed: {e}")
            return {'error': str(e)}
    
    def analyze_gradients(self, gradients: np.ndarray) -> Dict[str, Any]:
        """
        Analyze gradients for obfuscation patterns
        
        Args:
            gradients: Gradient data
            
        Returns:
            Analysis results
        """
        if not self.initialized:
            self.initialize_components()
            
        if self._integrated_security is not None:
            try:
                assessment = self._integrated_security.run_comprehensive_security_scan(
                    activation_data=np.asarray(gradients),
                    gradient_data=np.asarray(gradients),
                )
                # Consider any evasion_* findings as obfuscation proxies.
                evasion_attacks = [
                    a for a in (assessment.detected_attacks or [])
                    if str(a.get('type', '')).startswith('evasion_')
                ]
                obf = len(evasion_attacks) > 0
                obf_strength = float(max((a.get('confidence', 0.0) for a in evasion_attacks), default=0.0))
                return {
                    'obfuscation_detected': bool(obf),
                    # For callers, strength is "evasion/obfuscation confidence"
                    'obfuscation_strength': float(obf_strength),
                    'patterns': [a.get('type') for a in evasion_attacks],
                    'recommendations': assessment.recommendations,
                }
            except Exception as e:
                logger.error("Integrated gradient scan failed: %s", e)
                # Fall through to local fallback.

        if self.gradient_analyzer is None:
            return {'error': 'Gradient analyzer not initialized'}
        
        try:
            analysis_results = self.gradient_analyzer.analyze(gradients)
            return {
                'obfuscation_detected': analysis_results['obfuscation_detected'],
                'obfuscation_strength': analysis_results['obfuscation_strength'],
                'patterns': analysis_results.get('patterns', []),
                'recommendations': analysis_results.get('recommendations', [])
            }
        except Exception as e:
            logger.error(f"Gradient analysis failed: {e}")
            return {'error': str(e)}
    
    def comprehensive_security_scan(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive security scan
        
        Args:
            data: Input data for scanning
            
        Returns:
            Comprehensive security results
        """
        if not self.initialized:
            self.initialize_components()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'sensitivity_profile': self.sensitivity_profile,
            'scan_results': {}
        }
        
        # Adversarial detection
        adv_results = self.detect_adversarial_examples(data)
        results['scan_results']['adversarial_detection'] = adv_results
        
        # Gradient analysis if gradients provided
        if data.ndim >= 2:
            grad_results = self.analyze_gradients(data)
            results['scan_results']['gradient_analysis'] = grad_results
        
        # Overall assessment
        is_threat = (
            adv_results.get('is_adversarial', False) or
            results['scan_results'].get('gradient_analysis', {}).get('obfuscation_detected', False)
        )
        
        results['overall_assessment'] = {
            'threat_detected': is_threat,
            'risk_level': self._calculate_risk_level(results['scan_results']),
            'recommendations': self._generate_recommendations(results['scan_results'])
        }
        
        return results
    
    def _calculate_risk_level(self, scan_results: Dict[str, Any]) -> str:
        """Calculate overall risk level"""
        adv_results = scan_results.get('adversarial_detection', {})
        grad_results = scan_results.get('gradient_analysis', {})
        
        # High risk conditions
        if adv_results.get('is_adversarial') and adv_results.get('confidence', 0) > 0.8:
            return 'high'
        if grad_results.get('obfuscation_detected') and grad_results.get('obfuscation_strength', 0) > 0.7:
            return 'high'
        
        # Medium risk conditions
        if adv_results.get('is_adversarial') or grad_results.get('obfuscation_detected'):
            return 'medium'
        
        return 'low'
    
    def _generate_recommendations(self, scan_results: Dict[str, Any]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        adv_results = scan_results.get('adversarial_detection', {})
        grad_results = scan_results.get('gradient_analysis', {})
        
        if adv_results.get('is_adversarial'):
            recommendations.append("Implement adversarial training")
            recommendations.append("Apply input preprocessing")
            recommendations.append("Monitor for attack patterns")
        
        if grad_results.get('obfuscation_detected'):
            recommendations.append("Implement gradient clipping")
            recommendations.append("Add noise to gradients")
            recommendations.append("Use federated learning defenses")
        
        if not recommendations:
            recommendations.append("Continue monitoring")
            recommendations.append("Maintain current security posture")
        
        return recommendations

class AdversarialDetector:
    """Mock adversarial detector for testing"""
    
    def __init__(self, device: torch.device, sensitivity: str = 'adaptive'):
        self.device = device
        self.sensitivity = sensitivity
    
    def detect(self, data: torch.Tensor) -> Dict[str, Any]:
        """Mock detection method"""
        # Simulate detection logic
        mean_val = torch.mean(data).item()
        std_val = torch.std(data).item()
        
        # Simple heuristic for demonstration
        is_adversarial = abs(mean_val) > 2.0 or std_val > 3.0
        confidence = min(abs(mean_val) * 0.3 + std_val * 0.2, 1.0)
        
        return {
            'is_adversarial': is_adversarial,
            'confidence': confidence,
            'attack_type': 'fgsm' if is_adversarial else 'none',
            'severity': 'high' if confidence > 0.7 else 'medium'
        }

class GradientAnalyzer:
    """Mock gradient analyzer for testing"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def analyze(self, gradients: np.ndarray) -> Dict[str, Any]:
        """Mock gradient analysis"""
        # Simulate obfuscation detection
        grad_std = np.std(gradients)
        grad_mean = np.mean(np.abs(gradients))
        
        obfuscation_detected = grad_std < 0.1 and grad_mean > 1.0
        obfuscation_strength = min(grad_mean * 0.5, 1.0)
        
        return {
            'obfuscation_detected': obfuscation_detected,
            'obfuscation_strength': obfuscation_strength,
            'patterns': ['gradient_clipping', 'noise_injection'] if obfuscation_detected else [],
            'recommendations': ['increase_learning_rate', 'add_regularization'] if obfuscation_detected else []
        }

class ObfuscationDetector:
    """Mock obfuscation detector for testing"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def detect(self, data: torch.Tensor) -> Dict[str, Any]:
        """Mock obfuscation detection"""
        return {
            'obfuscation_detected': False,
            'confidence': 0.5,
            'obfuscation_type': 'none'
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test the integrated system
    print("🧠 Testing NeurInSpectre Integrated System...")
    
    system = IntegratedNeurInSpectre(sensitivity_profile='adaptive')
    
    # Test with sample data
    test_data = np.random.randn(100, 50)
    results = system.comprehensive_security_scan(test_data)
    
    print("✅ Comprehensive scan completed")
    print(f"   Threat detected: {results['overall_assessment']['threat_detected']}")
    print(f"   Risk level: {results['overall_assessment']['risk_level']}")
    print(f"   Device: {results['device']}") 