"""Legacy CLI glue module.

The supported CLI entrypoint is `neurinspectre` (see `neurinspectre/cli/__main__.py`).
This module is retained for compatibility with older scripts that imported it.

It intentionally avoids importing optional submodules at import-time.
"""


# Lazy import functions for heavy CLI modules
def _get_attack_graph_cli():
    """Lazy import of attack_graph_cli to avoid optional dependency issues."""
    try:
        from neurinspectre.cli import attack_graph_cli as _attack_graph_cli
        return _attack_graph_cli
    except ImportError as e:
        from neurinspectre.utils.logging import logger
        error_msg = str(e)
        if 'erfa' in error_msg.lower() or 'astropy' in error_msg.lower():
            logger.warning(f'Attack graph CLI not available - astropy dependency issue: {e}')
            logger.info('Fix with: pip install pyerfa astropy-iers-data')
        else:
            logger.warning(f'Attack graph CLI not available: {e}')
        return None


def _get_attack_graph_analyzer():
    """Lazy import of attack_graph_analyzer."""
    try:
        from neurinspectre.cli import attack_graph_analyzer as _attack_graph_analyzer
        return _attack_graph_analyzer
    except ImportError as e:
        from neurinspectre.utils.logging import logger
        error_msg = str(e)
        if 'erfa' in error_msg.lower() or 'astropy' in error_msg.lower():
            logger.warning(f'Attack graph analyzer not available - astropy dependency issue: {e}')
            logger.info('Fix with: pip install pyerfa astropy-iers-data')
        else:
            logger.warning(f'Attack graph analyzer not available: {e}')
        return None


def _get_ai_attack_graph_viz():
    """Lazy import of ai_attack_graph_viz."""
    try:
        from neurinspectre.cli import ai_attack_graph_viz as _ai_attack_graph_viz
        return _ai_attack_graph_viz
    except ImportError as e:
        from neurinspectre.utils.logging import logger
        error_msg = str(e)
        if 'erfa' in error_msg.lower() or 'astropy' in error_msg.lower():
            logger.warning(f'AI attack graph viz not available - astropy dependency issue: {e}')
            logger.info('Fix with: pip install pyerfa astropy-iers-data')
        else:
            logger.warning(f'AI attack graph viz not available: {e}')
        return None
