"""
Security pipelines shims (non-destructive).

Provides stable orchestration import paths for integrated scans and reporting.
"""

from ..integrated_security import (
    IntegratedSecurityAnalyzer,
    SecurityAssessment,
    generate_security_assessment,
    run_comprehensive_security_scan,
)

__all__ = [
    "IntegratedSecurityAnalyzer",
    "SecurityAssessment",
    "generate_security_assessment",
    "run_comprehensive_security_scan",
]


