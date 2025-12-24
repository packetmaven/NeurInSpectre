"""
NeurInSpectre Attacks Module
Advanced offensive security attack implementations
"""

from .ednn_attack import EDNNAttack, EDNNConfig, load_embedding_model

__all__ = ['EDNNAttack', 'EDNNConfig', 'load_embedding_model']

