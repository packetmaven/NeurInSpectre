"""
NeurInSpectre - AI Security Research Platform
"""

from importlib import metadata as _importlib_metadata

try:
    __version__ = _importlib_metadata.version("neurinspectre")
except Exception:  # pragma: no cover
    __version__ = "0+unknown"
__author__ = "NeurInSpectre Team"

from typing import Any

# Import mathematical module (kept for backward compatibility)
from .mathematical import *  # noqa: F403

__all__ = ['mathematical'] + [  # type: ignore
    "GPUAcceleratedMathEngine",  # noqa: F405
    "AdvancedExponentialIntegrator",  # noqa: F405
    "demonstrate_advanced_mathematics",  # noqa: F405
    "get_engine_info",  # noqa: F405
    "get_precision",  # noqa: F405
    "get_device",  # noqa: F405
    "IntegratedNeurInSpectre",
    "BlueTeamIntelligenceEngine",
    "RedTeamIntelligenceEngine",
    "CriticalRLObfuscationDetector",
    "ObfuscatedGradientVisualizer",
]


def __getattr__(name: str) -> Any:  # PEP 562
    """Lazy-load heavy/optional submodules on demand.

    This keeps `import neurinspectre` lightweight and avoids import-time failures
    when optional visualization/dashboard dependencies are not installed.
    """
    if name == "IntegratedNeurInSpectre":
        from .integrated_neurinspectre_system import IntegratedNeurInSpectre as _T

        return _T

    if name == "CriticalRLObfuscationDetector":
        from .security.critical_rl_obfuscation import CriticalRLObfuscationDetector as _T

        return _T

    if name == "BlueTeamIntelligenceEngine":
        try:
            from .security.blue_team_intelligence import BlueTeamIntelligenceEngine as _T
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "BlueTeamIntelligenceEngine requires optional dashboard dependencies "
                "(e.g. pandas/plotly/sklearn)."
            ) from e
        return _T

    if name == "RedTeamIntelligenceEngine":
        try:
            from .security.red_team_intelligence import RedTeamIntelligenceEngine as _T
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "RedTeamIntelligenceEngine requires optional dashboard dependencies "
                "(e.g. pandas/plotly/networkx/sklearn)."
            ) from e
        return _T

    if name == "ObfuscatedGradientVisualizer":
        try:
            from .security.visualization.obfuscated_gradient_visualizer import ObfuscatedGradientVisualizer as _T
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "ObfuscatedGradientVisualizer requires optional visualization dependencies "
                "(e.g. matplotlib/seaborn)."
            ) from e
        return _T

    raise AttributeError(f"module 'neurinspectre' has no attribute {name!r}")