"""
CLI for Structural Role Analysis in Attack Graphs using GraphWave & Astropy
Validated against MITRE ATT&CK Framework (v12.1)
"""
import argparse
import networkx as nx
import numpy as np
# from astropy.table import Table  # FIXED: Made lazy import
from sklearn.cluster import KMeans
import sys

# Lazy astropy import function
def _get_astropy_table():
    """Lazy import of astropy.table.Table to avoid import-time dependency issues"""
    try:
        from astropy.table import Table
        return Table
    except ImportError as e:
        error_msg = str(e)
        if "erfa" in error_msg.lower():
            raise ImportError(
                "astropy is missing the 'erfa' dependency. Install with:\n"
                "  pip install pyerfa astropy-iers-data\n"
                "  OR\n"
                "  conda install -c conda-forge astropy\n"
                f"Original error: {error_msg}"
            )
        elif "astropy-iers-data" in error_msg.lower():
            raise ImportError(
                "astropy is missing the 'astropy-iers-data' dependency. Install with:\n"
                "  pip install astropy-iers-data\n"
                "  OR\n"
                "  conda install -c conda-forge astropy\n"
                f"Original error: {error_msg}"
            )
        else:
            raise ImportError(
                "astropy is required for attack graph analysis. Install with:\n"
                "  pip install astropy pyerfa astropy-iers-data\n"
                "  OR\n"
                "  conda install -c conda-forge astropy\n"
                f"Original error: {error_msg}"
            )

try:
    import plotly.graph_objs as go
except ImportError:
    go = None 