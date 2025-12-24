#!/usr/bin/env python3
"""
Security Results Validation and Intelligence Report Generator
Generates comprehensive actionable intelligence reports for red and blue teams
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import traceback

# Add the current directory to Python path
sys.path.insert(0, '.')

def load_attack_metadata(metadata_file: str) -> Dict[str, Any]:
    """Load attack metadata from JSON file"""
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metadata from {metadata_file}: {e}")
        return {}

def generate_threat_intelligence_report(attack_type: str, results: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive threat intelligence report"""
    
    # Extract key metrics
    adversarial_results = results.get('adversarial_detection', {})
    evasion_results = results.get('evasion_detection', {})
    comprehensive_results = results.get('comprehensive_scan', {})
    
    # Create threat intelligence report
    report = {
        'attack_type': attack_type,
        'timestamp': datetime.now().isoformat(),
        'attack_metadata': metadata,
        'detection_summary': {
            'adversarial_detection': {
                'status': 'PASSED' if adversarial_results else 'FAILED',
                'threat_level': adversarial_results.get('threat_level', 'unknown'),
                'confidence': adversarial_results.get('confidence', 0.0),
                'detections': adversarial_results.get('detections', 0)
            },
            'evasion_detection': {
                'status': 'PASSED' if evasion_results else 'FAILED',
                'evasion_attempts': evasion_results.get('evasion_attempts', 0),
                'high_threat_attempts': evasion_results.get('high_threat_attempts', 0)
            },
            'comprehensive_scan': {
                'status': 'PASSED' if comprehensive_results else 'FAILED',
                'overall_threat_level': comprehensive_results.get('overall_threat_level', 'unknown'),
                'confidence_score': comprehensive_results.get('confidence_score', 0.0),
                'attacks_detected': comprehensive_results.get('attacks_detected', 0)
            }
        },
        'red_team_intelligence': generate_red_team_intelligence(attack_type, results, metadata),
        'blue_team_intelligence': generate_blue_team_intelligence(attack_type, results, metadata),
        'mitre_atlas_mapping': generate_mitre_atlas_mapping(attack_type, metadata),
        'actionable_recommendations': generate_actionable_recommendations(attack_type, results)
    }
    
    return report

def generate_red_team_intelligence(attack_type: str, results: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate red team operational intelligence"""
    
    red_team_intel = {
        'attack_effectiveness': {},
        'evasion_success_rate': 0.0,
        'detection_bypass_methods': [],
        'recommended_attack_vectors': [],
        'exploitation_opportunities': []
    }
    
    # Analyze attack effectiveness
    adversarial_results = results.get('adversarial_detection', {})
    evasion_results = results.get('evasion_detection', {})
    
    if adversarial_results:
        threat_level = adversarial_results.get('threat_level', '').lower()
        if threat_level in ['low', 'medium']:
            red_team_intel['attack_effectiveness']['adversarial'] = 'HIGH_STEALTH'
            red_team_intel['detection_bypass_methods'].append('Adversarial pattern masking')
        else:
            red_team_intel['attack_effectiveness']['adversarial'] = 'DETECTED'
    
    if evasion_results:
        evasion_attempts = evasion_results.get('evasion_attempts', 0)
        if evasion_attempts > 0:
            red_team_intel['evasion_success_rate'] = min(1.0, evasion_attempts / 10.0)
            red_team_intel['detection_bypass_methods'].append('Transport dynamics evasion')
    
    # Attack-specific recommendations
    if attack_type == 'TS-Inverse Attack':
        red_team_intel['recommended_attack_vectors'] = [
            'Gradient inversion against federated learning systems',
            'Privacy budget exhaustion attacks',
            'Differential privacy bypass techniques'
        ]
        red_team_intel['exploitation_opportunities'] = [
            'Target FL systems with epsilon-DP > 1.0',
            'Exploit gradient aggregation vulnerabilities',
            'Reconstruct training data from gradient updates'
        ]
    
    elif attack_type == 'ConcreTizer Attack':
        red_team_intel['recommended_attack_vectors'] = [
            '3D model inversion attacks',
            'Voxel-based data extraction',
            'Point cloud reconstruction'
        ]
        red_team_intel['exploitation_opportunities'] = [
            'Target 3D model APIs and services',
            'Extract proprietary model architectures',
            'Reconstruct training datasets'
        ]
    
    elif attack_type == 'EDNN Attack':
        red_team_intel['recommended_attack_vectors'] = [
            'Element-wise differential attacks',
            'Transformer embedding manipulation',
            'Nearest neighbor poisoning'
        ]
        red_team_intel['exploitation_opportunities'] = [
            'Target transformer-based NLP models',
            'Manipulate embedding spaces',
            'Poison similarity calculations'
        ]
    
    elif attack_type == 'Transport Evasion':
        red_team_intel['recommended_attack_vectors'] = [
            'Neural transport dynamics exploitation',
            'Information flow manipulation',
            'Gradient flow evasion'
        ]
        red_team_intel['exploitation_opportunities'] = [
            'Bypass neural network detection systems',
            'Manipulate information flow patterns',
            'Evade gradient-based defenses'
        ]
    
    elif attack_type == 'DeMarking Attack':
        red_team_intel['recommended_attack_vectors'] = [
            'Network flow watermark removal',
            'DeMarking defense circumvention',
            'Flow-based evasion techniques'
        ]
        red_team_intel['exploitation_opportunities'] = [
            'Remove network flow watermarks',
            'Bypass flow-based detection systems',
            'Maintain persistent network presence'
        ]
    
    return red_team_intel

def generate_blue_team_intelligence(attack_type: str, results: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate blue team defensive intelligence"""
    
    blue_team_intel = {
        'threat_assessment': {},
        'detection_coverage': {},
        'recommended_countermeasures': [],
        'monitoring_priorities': [],
        'incident_response_actions': []
    }
    
    # Analyze threat assessment
    comprehensive_results = results.get('comprehensive_scan', {})
    if comprehensive_results:
        threat_level = comprehensive_results.get('overall_threat_level', '').lower()
        confidence = comprehensive_results.get('confidence_score', 0.0)
        
        blue_team_intel['threat_assessment'] = {
            'threat_level': threat_level.upper(),
            'confidence': confidence,
            'risk_score': min(100, confidence * 100),
            'detection_accuracy': confidence
        }
    
    # Detection coverage analysis
    adversarial_results = results.get('adversarial_detection', {})
    evasion_results = results.get('evasion_detection', {})
    
    blue_team_intel['detection_coverage'] = {
        'adversarial_detection': 'ACTIVE' if adversarial_results else 'INACTIVE',
        'evasion_detection': 'ACTIVE' if evasion_results else 'INACTIVE',
        'comprehensive_scanning': 'ACTIVE' if comprehensive_results else 'INACTIVE'
    }
    
    # Attack-specific countermeasures
    if attack_type == 'TS-Inverse Attack':
        blue_team_intel['recommended_countermeasures'] = [
            'Implement enhanced differential privacy (ε < 1.0)',
            'Deploy gradient clipping and noise injection',
            'Monitor privacy budget exhaustion',
            'Implement secure aggregation protocols'
        ]
        blue_team_intel['monitoring_priorities'] = [
            'Federated learning gradient updates',
            'Privacy budget tracking',
            'Differential privacy noise levels',
            'Gradient reconstruction attempts'
        ]
    
    elif attack_type == 'ConcreTizer Attack':
        blue_team_intel['recommended_countermeasures'] = [
            'Implement 3D model access controls',
            'Deploy voxel-based input validation',
            'Monitor model inversion attempts',
            'Implement output perturbation'
        ]
        blue_team_intel['monitoring_priorities'] = [
            '3D model API access patterns',
            'Voxel reconstruction attempts',
            'Model query frequencies',
            'Point cloud extraction patterns'
        ]
    
    elif attack_type == 'EDNN Attack':
        blue_team_intel['recommended_countermeasures'] = [
            'Implement embedding space monitoring',
            'Deploy transformer input validation',
            'Monitor nearest neighbor calculations',
            'Implement adversarial training'
        ]
        blue_team_intel['monitoring_priorities'] = [
            'Embedding space anomalies',
            'Transformer attention patterns',
            'Nearest neighbor queries',
            'Element-wise differential patterns'
        ]
    
    elif attack_type == 'Transport Evasion':
        blue_team_intel['recommended_countermeasures'] = [
            'Implement neural transport monitoring',
            'Deploy gradient flow analysis',
            'Monitor information flow patterns',
            'Implement multi-layer detection'
        ]
        blue_team_intel['monitoring_priorities'] = [
            'Neural transport dynamics',
            'Information flow anomalies',
            'Gradient flow patterns',
            'Evasion attempt signatures'
        ]
    
    elif attack_type == 'DeMarking Attack':
        blue_team_intel['recommended_countermeasures'] = [
            'Implement robust watermarking',
            'Deploy multi-layer flow analysis',
            'Monitor watermark integrity',
            'Implement backup detection methods'
        ]
        blue_team_intel['monitoring_priorities'] = [
            'Network flow watermarks',
            'Watermark removal attempts',
            'Flow-based anomalies',
            'DeMarking attack signatures'
        ]
    
    # Incident response actions
    threat_level = blue_team_intel['threat_assessment'].get('threat_level', 'LOW')
    
    if threat_level == 'CRITICAL':
        blue_team_intel['incident_response_actions'] = [
            'Activate emergency incident response team',
            'Isolate affected systems immediately',
            'Preserve forensic evidence',
            'Notify stakeholders and management',
            'Implement emergency countermeasures'
        ]
    elif threat_level == 'HIGH':
        blue_team_intel['incident_response_actions'] = [
            'Escalate to security team',
            'Implement enhanced monitoring',
            'Review security controls',
            'Update detection thresholds',
            'Prepare incident response plan'
        ]
    else:
        blue_team_intel['incident_response_actions'] = [
            'Log security event',
            'Continue monitoring',
            'Review detection efficacy',
            'Update security metrics'
        ]
    
    return blue_team_intel

def generate_mitre_atlas_mapping(attack_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate MITRE ATLAS technique mapping"""
    
    atlas_mapping = {
        'techniques': [],
        'tactics': [],
        'mitigations': []
    }
    
    # Map attack types to MITRE ATLAS techniques
    if attack_type == 'TS-Inverse Attack':
        atlas_mapping['techniques'] = [
            'AML.T0043 - Inference API',
            'AML.T0044 - Full ML Model Access',
            'AML.T0047 - Poisoning via Gradient Computation'
        ]
        atlas_mapping['tactics'] = [
            'AML.TA0006 - ML Model Access',
            'AML.TA0003 - Persistence',
            'AML.TA0004 - Privilege Escalation'
        ]
    
    elif attack_type == 'ConcreTizer Attack':
        atlas_mapping['techniques'] = [
            'AML.T0043 - Inference API',
            'AML.T0051 - LLM Jailbreak',
            'AML.T0048 - Denial of ML Service'
        ]
        atlas_mapping['tactics'] = [
            'AML.TA0006 - ML Model Access',
            'AML.TA0009 - Exfiltration',
            'AML.TA0005 - Defense Evasion'
        ]
    
    elif attack_type == 'EDNN Attack':
        atlas_mapping['techniques'] = [
            'AML.T0043 - Inference API',
            'AML.T0051 - LLM Jailbreak',
            'AML.T0047 - Poisoning via Gradient Computation'
        ]
        atlas_mapping['tactics'] = [
            'AML.TA0006 - ML Model Access',
            'AML.TA0002 - ML Attack Staging',
            'AML.TA0005 - Defense Evasion'
        ]
    
    elif attack_type == 'Transport Evasion':
        atlas_mapping['techniques'] = [
            'AML.T0054 - Adversarial Example',
            'AML.T0055 - Membership Inference',
            'AML.T0056 - Model Extraction'
        ]
        atlas_mapping['tactics'] = [
            'AML.TA0005 - Defense Evasion',
            'AML.TA0007 - Discovery',
            'AML.TA0009 - Exfiltration'
        ]
    
    elif attack_type == 'DeMarking Attack':
        atlas_mapping['techniques'] = [
            'AML.T0054 - Adversarial Example',
            'AML.T0048 - Denial of ML Service',
            'AML.T0055 - Membership Inference'
        ]
        atlas_mapping['tactics'] = [
            'AML.TA0005 - Defense Evasion',
            'AML.TA0008 - Impact',
            'AML.TA0009 - Exfiltration'
        ]
    
    return atlas_mapping

def generate_actionable_recommendations(attack_type: str, results: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate actionable recommendations for both teams"""
    
    recommendations = {
        'immediate_actions': [],
        'short_term_improvements': [],
        'long_term_strategy': []
    }
    
    # Analyze results to determine urgency
    comprehensive_results = results.get('comprehensive_scan', {})
    threat_level = comprehensive_results.get('overall_threat_level', 'low').lower()
    
    if threat_level == 'critical':
        recommendations['immediate_actions'] = [
            'Activate emergency incident response procedures',
            'Isolate affected AI/ML systems immediately',
            'Implement emergency detection thresholds',
            'Review and update security controls',
            'Contact security team and stakeholders'
        ]
    elif threat_level == 'high':
        recommendations['immediate_actions'] = [
            'Escalate to security team for investigation',
            'Implement enhanced monitoring protocols',
            'Review current detection capabilities',
            'Update security alert thresholds'
        ]
    else:
        recommendations['immediate_actions'] = [
            'Log security event for trend analysis',
            'Continue routine monitoring',
            'Review detection accuracy metrics'
        ]
    
    # Short-term improvements
    recommendations['short_term_improvements'] = [
        'Implement automated threat detection',
        'Deploy real-time monitoring dashboards',
        'Enhance security team training',
        'Update incident response procedures',
        'Improve detection algorithm accuracy'
    ]
    
    # Long-term strategy
    recommendations['long_term_strategy'] = [
        'Develop advanced AI security capabilities',
        'Implement zero-trust AI architecture',
        'Create comprehensive threat intelligence',
        'Establish continuous security monitoring',
        'Build proactive defense mechanisms'
    ]
    
    return recommendations

def validate_security_results() -> Dict[str, Any]:
    """Validate security results and generate comprehensive intelligence reports"""
    
    print("🔍 NeurInSpectre Security Results Validation")
    print("="*60)
    print(f"Validation timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test data files and their metadata
    test_cases = [
        {
            'data_file': 'attack_data/ts_inverse_attack_data.npy',
            'metadata_file': 'attack_data/ts_inverse_metadata.json',
            'attack_type': 'TS-Inverse Attack'
        },
        {
            'data_file': 'attack_data/concretizer_attack_data.npy',
            'metadata_file': 'attack_data/concretizer_metadata.json',
            'attack_type': 'ConcreTizer Attack'
        },
        {
            'data_file': 'attack_data/ednn_attack_data.npy',
            'metadata_file': 'attack_data/ednn_metadata.json',
            'attack_type': 'EDNN Attack'
        },
        {
            'data_file': 'attack_data/transport_evasion_attack_data.npy',
            'metadata_file': 'attack_data/transport_evasion_metadata.json',
            'attack_type': 'Transport Evasion'
        },
        {
            'data_file': 'attack_data/demarking_attack_data.npy',
            'metadata_file': 'attack_data/demarking_metadata.json',
            'attack_type': 'DeMarking Attack'
        }
    ]
    
    validation_results = {}
    intelligence_reports = []
    
    for test_case in test_cases:
        if not Path(test_case['data_file']).exists():
            print(f"⚠️  Skipping {test_case['attack_type']}: Data file not found")
            continue
        
        print(f"🎯 Validating {test_case['attack_type']}")
        print("-" * 40)
        
        # Load metadata
        metadata = load_attack_metadata(test_case['metadata_file'])
        
        # Run validation tests
        try:
            from neurinspectre.security.adversarial_detection import AdversarialDetector
            from neurinspectre.security.evasion_detection import EvasionDetector
            from neurinspectre.security.integrated_security import IntegratedSecurityAnalyzer
            
            # Load attack data
            data = np.load(test_case['data_file'])
            print(f"✅ Loaded attack data: {data.shape}")
            
            # Test results container
            test_results = {}
            
            # Adversarial detection validation
            try:
                config = {
                    'ts_inverse_threshold': 0.8,
                    'voxel_resolution': 32,
                    'max_seq_length': min(512, data.shape[0]),
                    'attention_heads': 8,
                    'k_neighbors': 5,
                    'ednn_threshold': 0.8
                }
                
                detector = AdversarialDetector(config)
                results = detector.detect_adversarial_samples(data)
                
                test_results['adversarial_detection'] = {
                    'threat_level': results.get('overall_threat_level', 'unknown'),
                    'confidence': results.get('confidence_score', 0.0),
                    'detections': len(results.get('detections', {}))
                }
                
                print(f"   ✅ Adversarial detection: {test_results['adversarial_detection']['threat_level']}")
                
            except Exception as e:
                print(f"   ❌ Adversarial detection failed: {e}")
                test_results['adversarial_detection'] = None
            
            # Evasion detection validation
            try:
                config = {
                    'transport_dim': min(64, data.shape[-1]),
                    'time_window': min(100, data.shape[0]),
                    'demarking_threshold': 0.6,
                    'pattern_window': min(100, data.shape[0])
                }
                
                detector = EvasionDetector(config)
                evasion_attempts = detector.detect_evasion_attempts(data)
                
                test_results['evasion_detection'] = {
                    'evasion_attempts': len(evasion_attempts),
                    'high_threat_attempts': len([a for a in evasion_attempts 
                                                if a.get('threat_level') in ['high', 'critical']])
                }
                
                print(f"   ✅ Evasion detection: {test_results['evasion_detection']['evasion_attempts']} attempts")
                
            except Exception as e:
                print(f"   ❌ Evasion detection failed: {e}")
                test_results['evasion_detection'] = None
            
            # Comprehensive scan validation
            try:
                config = {
                    'adversarial': {
                        'ts_inverse_threshold': 0.6,
                        'voxel_resolution': 32,
                        'max_seq_length': min(512, data.shape[0]),
                        'attention_heads': 8,
                        'k_neighbors': 5,
                        'ednn_threshold': 0.6
                    },
                    'evasion': {
                        'transport_dim': min(64, data.shape[-1]),
                        'time_window': min(100, data.shape[0]),
                        'demarking_threshold': 0.6,
                        'pattern_window': min(100, data.shape[0])
                    },
                    'parallel_processing': True,
                    'threat_threshold': 0.6
                }
                
                analyzer = IntegratedSecurityAnalyzer(config)
                assessment = analyzer.run_comprehensive_security_scan(activation_data=data)
                
                test_results['comprehensive_scan'] = {
                    'overall_threat_level': assessment.overall_threat_level,
                    'confidence_score': assessment.confidence_score,
                    'attacks_detected': len(assessment.detected_attacks)
                }
                
                print(f"   ✅ Comprehensive scan: {test_results['comprehensive_scan']['overall_threat_level']}")
                
            except Exception as e:
                print(f"   ❌ Comprehensive scan failed: {e}")
                test_results['comprehensive_scan'] = None
            
            # Store validation results
            validation_results[test_case['attack_type']] = test_results
            
            # Generate intelligence report
            intelligence_report = generate_threat_intelligence_report(
                test_case['attack_type'], test_results, metadata
            )
            intelligence_reports.append(intelligence_report)
            
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            validation_results[test_case['attack_type']] = None
        
        print()
    
    # Create comprehensive validation report
    validation_report = {
        'validation_timestamp': datetime.now().isoformat(),
        'validation_results': validation_results,
        'intelligence_reports': intelligence_reports,
        'summary': {
            'total_attacks_tested': len(test_cases),
            'successful_validations': len([r for r in validation_results.values() if r is not None]),
            'validation_success_rate': len([r for r in validation_results.values() if r is not None]) / len(test_cases) * 100
        }
    }
    
    return validation_report

def save_validation_report(report: Dict[str, Any], output_file: str = 'security_validation_report.json'):
    """Save validation report to file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"💾 Validation report saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save report: {e}")

def print_validation_summary(report: Dict[str, Any]):
    """Print validation summary"""
    print("📊 VALIDATION SUMMARY")
    print("="*30)
    
    summary = report['summary']
    print(f"🎯 Total attacks tested: {summary['total_attacks_tested']}")
    print(f"✅ Successful validations: {summary['successful_validations']}")
    print(f"📈 Validation success rate: {summary['validation_success_rate']:.1f}%")
    print()
    
    print("🔍 INTELLIGENCE REPORTS GENERATED:")
    print("-" * 40)
    
    for intel_report in report['intelligence_reports']:
        attack_type = intel_report['attack_type']
        threat_level = intel_report['detection_summary']['comprehensive_scan']['overall_threat_level']
        
        print(f"📋 {attack_type}")
        print(f"   • Threat Level: {threat_level.upper()}")
        print(f"   • Red Team Intelligence: {len(intel_report['red_team_intelligence']['recommended_attack_vectors'])} vectors")
        print(f"   • Blue Team Intelligence: {len(intel_report['blue_team_intelligence']['recommended_countermeasures'])} countermeasures")
        print(f"   • MITRE ATLAS: {len(intel_report['mitre_atlas_mapping']['techniques'])} techniques")
        print()

def main():
    """Main validation entry point"""
    try:
        # Run validation
        validation_report = validate_security_results()
        
        # Save report
        save_validation_report(validation_report)
        
        # Print summary
        print_validation_summary(validation_report)
        
        # Success message
        success_rate = validation_report['summary']['validation_success_rate']
        if success_rate >= 90:
            print("🎉 VALIDATION SUCCESSFUL: Security modules are performing excellently!")
        elif success_rate >= 70:
            print("✅ VALIDATION GOOD: Security modules are performing well!")
        else:
            print("⚠️  VALIDATION NEEDS ATTENTION: Some security modules need improvement!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main()) 