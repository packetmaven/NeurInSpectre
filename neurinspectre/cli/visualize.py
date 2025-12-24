import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from string import Template
import numpy as np
# import matplotlib.pyplot as plt  # FIXED: Made lazy import
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Lazy matplotlib import function
def _get_matplotlib():
    """Lazy import of matplotlib to avoid import-time dependency issues"""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization functionality. Install with: conda install -c conda-forge matplotlib")

try:
    from transformers import PreTrainedTokenizerBase as PreTrainedTokenizer
except ImportError:
    raise ImportError("transformers is required for tokenization functionality. Install with: pip install transformers")