#!/usr/bin/env python3
"""
Standalone Test Script for NeurInSpectre Mathematical Foundations
Can be run independently to test all mathematical capabilities
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import neurinspectre
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Main test function"""
    try:
        from neurinspectre.mathematical import run_test_suite
        
        print("🧪 NeurInSpectre Mathematical Foundations Standalone Test")
        print("=" * 70)
        print()
        
        # Run the comprehensive test suite
        success = run_test_suite(verbose=True)
        
        if success:
            print("\n🎉 All tests passed! NeurInSpectre Mathematical Foundations are fully integrated and working!")
            return 0
        else:
            print("\n❌ Some tests failed. Please check the installation.")
            return 1
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Mathematical foundations may not be properly installed.")
        print("   Make sure you're running this from the correct directory.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 