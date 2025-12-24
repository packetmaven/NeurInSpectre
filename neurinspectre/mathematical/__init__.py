"""
NeurInSpectre Mathematical Foundations Module
Advanced mathematical foundations with GPU acceleration for gradient obfuscation detection
"""

from __future__ import annotations

import importlib
from typing import Any

from .gpu_accelerated_math import (
    GPUAcceleratedMathEngine,
    AdvancedExponentialIntegrator,
    demonstrate_advanced_mathematics
)

# Add functions for direct import
from .gpu_accelerated_math import (
    get_engine_info,
    get_precision,
    get_device
)

_LAZY_EXPORTS = {
    # Avoid importing test modules (and their side effects) at package import time.
    'MathematicalFoundationsTestSuite': ('.tests', 'MathematicalFoundationsTestSuite'),
    'run_test_suite': ('.tests', 'run_test_suite'),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_name, attr = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_name, __name__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'GPUAcceleratedMathEngine',
    'AdvancedExponentialIntegrator', 
    'demonstrate_advanced_mathematics',
    'get_engine_info',
    'get_precision',
    'get_device',
    'MathematicalFoundationsTestSuite',
    'run_test_suite'
] 