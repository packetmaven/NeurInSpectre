"""
Defensive security shims (non-destructive).

These re-export existing implementations so downstream users can adopt the
defensive import path without breaking current code.

Usage (new preferred path):
    from neurinspectre.security.defensive import EvasionDetector
"""

from ..adversarial_detection import AttentionGuardDetector  # noqa: F401
from ..evasion_detection import (
    EvasionDetector,
    NeuralTransportDynamicsDetector,
    DeMarkingDefenseDetector,
    BehavioralPatternAnalyzer,
)
from ..integrated_security import (
    IntegratedSecurityAnalyzer,
    generate_security_assessment,
    run_comprehensive_security_scan,
)

__all__ = [
    "AttentionGuardDetector",
    "EvasionDetector",
    "NeuralTransportDynamicsDetector",
    "DeMarkingDefenseDetector",
    "BehavioralPatternAnalyzer",
    "IntegratedSecurityAnalyzer",
    "generate_security_assessment",
    "run_comprehensive_security_scan",
]


