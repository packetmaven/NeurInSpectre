#!/usr/bin/env python3
"""
NeurInSpectre Package Main Entry Point
Enables execution via: python -m neurinspectre <command>
"""

import sys

def main():
    """Main entry point for neurinspectre package"""
    # Delegate to CLI main
    from neurinspectre.cli.__main__ import main as cli_main
    return cli_main()

if __name__ == '__main__':
    sys.exit(main()) 