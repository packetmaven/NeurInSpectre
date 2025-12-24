"""Visualization utilities for NeurInSpectre.

This package contains lightweight Plotly-based visualization helpers that are
consumed by CLI commands and research materials.
"""

from .dna_visualizer import (
    plot_anomaly_detection,
    plot_attack_patterns,
    plot_neuron_heatmap,
)

from .fusion_pi_viz import plot_pi_viz

from .time_travel_debugging import plot_time_travel_debugging

from .eigen_collapse_radar import plot_eigen_collapse_radar

from .eigenvalue_spectrum import plot_eigenvalue_spectrum, plot_eigenvalue_spectrum_interactive

from .fft_security_spectrum import plot_fft_security_spectrum

from .attention_gradient_alignment import plot_attention_gradient_alignment

from .co_attention_traces import plot_co_attention_traces

__all__ = [
    "plot_anomaly_detection",
    "plot_attack_patterns",
    "plot_neuron_heatmap",
    "plot_pi_viz",
    "plot_time_travel_debugging",
    "plot_eigen_collapse_radar",
    "plot_eigenvalue_spectrum",
    "plot_eigenvalue_spectrum_interactive",
    "plot_fft_security_spectrum",
    "plot_attention_gradient_alignment",
    "plot_co_attention_traces",
]
