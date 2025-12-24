"""
Offensive security shims (non-destructive).

These re-export existing implementations to provide a stable, future-friendly
import path without moving any files yet.

Usage (new preferred path):
    from neurinspectre.security.offensive import TSInverseDetector
"""

# Re-exports from existing modules (no file moves)
from ..adversarial_detection import (
    TSInverseDetector,
    ConcreTizerDetector,
    EDNNAttackDetector,
    AdversarialDetector,
)

try:
    # Optional offensive utility if available in package
    from ...activation_steganography import ActivationSteganography  # type: ignore
except Exception:  # pragma: no cover
    ActivationSteganography = None  # type: ignore

__all__ = [
    "TSInverseDetector",
    "ConcreTizerDetector",
    "EDNNAttackDetector",
    "AdversarialDetector",
    "ActivationSteganography",
]


