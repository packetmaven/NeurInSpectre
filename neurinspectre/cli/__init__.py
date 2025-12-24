"""
Command-line interface for NeurInSpectre.
"""

def register_commands():
    """Lazy import of register_commands to avoid import-time dependencies"""
    from .security_commands import register_commands as _register_commands
    return _register_commands

def ttd():
    """Run the Time Travel Debugger CLI."""
    from .ttd import main
    return main()

# Remove immediate import: from .security_commands import register_commands
__all__ = ['ttd', 'register_commands'] 