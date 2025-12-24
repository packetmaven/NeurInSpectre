#!/usr/bin/env python3
"""
Test script for NeurInSpectre Mathematical Foundations
Demonstrates the advanced mathematical capabilities
"""

import numpy as np
import sys
from pathlib import Path

def test_mathematical_foundations():
    """Test the mathematical foundations integration"""
    print("🧮 Testing NeurInSpectre Mathematical Foundations")
    print("=" * 60)
    
    try:
        # Import the mathematical foundations
        from neurinspectre.mathematical import GPUAcceleratedMathEngine, AdvancedExponentialIntegrator, demonstrate_advanced_mathematics
        
        print("✅ Successfully imported mathematical foundations!")
        print()
        
        # Test 1: Basic engine initialization
        print("🚀 Test 1: GPU Mathematical Engine Initialization")
        math_engine = GPUAcceleratedMathEngine(precision='float32', device_preference='auto')
        print(f"   Device: {math_engine.device}")
        print(f"   Precision: {math_engine.precision}")
        print()
        
        # Test 2: Spectral decomposition
        print("🔬 Test 2: Advanced Spectral Decomposition")
        test_gradient = np.random.randn(256) * 0.1
        results = math_engine.advanced_spectral_decomposition(test_gradient, decomposition_levels=3)
        
        print(f"   Spectral levels: {len(results['spectral_levels'])}")
        print(f"   Cross-correlations: {len(results['cross_correlations'])}")
        print(f"   Obfuscation indicators: {len(results['obfuscation_indicators'])}")
        
        if 'summary_metrics' in results and 'mean_entropy' in results['summary_metrics']:
            entropy = results['summary_metrics']['mean_entropy']
            print(f"   Mean entropy: {entropy.item():.4f}")
        print()
        
        # Test 3: Exponential integrator
        print("⚡ Test 3: Advanced Exponential Integrator")
        integrator = AdvancedExponentialIntegrator(math_engine)
        
        # Simple test function
        def test_nonlinear(u):
            import torch
            return -0.1 * u**2 + 0.05 * torch.sin(u)
        
        u_test = test_gradient[:50]  # Use smaller subset
        u_next = integrator.etd_rk4_step(u_test, None, test_nonlinear, 0.01)
        
        print(f"   Initial norm: {np.linalg.norm(u_test):.6f}")
        print(f"   Final norm: {u_next.cpu().numpy() if hasattr(u_next, 'cpu') else u_next}")
        print()
        
        # Test 4: Full demonstration
        print("🎯 Test 4: Full Mathematical Demonstration")
        demo_results = demonstrate_advanced_mathematics()
        print("   Demonstration completed successfully!")
        print()
        
        print("✅ All mathematical foundation tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Mathematical foundations may not be properly installed.")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_integration():
    """Test CLI integration"""
    print("🖥️  Testing CLI Integration")
    print("=" * 60)
    
    try:
        # Test importing CLI components
        from neurinspectre.cli.mathematical_commands import register_mathematical_commands
        print("✅ Successfully imported CLI mathematical commands!")
        
        # Test argument parser creation
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        
        register_mathematical_commands(subparsers)
        print("✅ Successfully registered mathematical commands!")
        
        # Test help output
        help_output = parser.format_help()
        if 'math' in help_output:
            print("✅ Mathematical commands appear in help!")
        else:
            print("⚠️  Mathematical commands not found in help")
        
        return True
        
    except ImportError as e:
        print(f"❌ CLI import error: {e}")
        return False
    except Exception as e:
        print(f"❌ CLI test error: {e}")
        return False

def create_sample_data():
    """Create sample gradient data for testing"""
    print("📊 Creating sample gradient data for testing...")
    
    # Create realistic gradient data with different patterns
    n_samples = 512
    
    # Clean gradients (normal distribution)
    clean_gradients = np.random.normal(0, 0.1, n_samples)
    
    # Obfuscated gradients (with artificial patterns)
    obfuscated_gradients = clean_gradients.copy()
    obfuscated_gradients += 0.05 * np.sin(np.arange(n_samples) * 0.1)  # Periodic component
    obfuscated_gradients[::50] += 0.3 * np.random.randn(len(obfuscated_gradients[::50]))  # Spikes
    
    # Save sample data
    np.save('sample_clean_gradients.npy', clean_gradients)
    np.save('sample_obfuscated_gradients.npy', obfuscated_gradients)
    
    print("✅ Sample data created:")
    print("   • sample_clean_gradients.npy")
    print("   • sample_obfuscated_gradients.npy")
    print()

def main():
    """Main test function"""
    print("🧪 NeurInSpectre Mathematical Foundations Test Suite")
    print("=" * 70)
    print()
    
    # Create sample data
    create_sample_data()
    
    # Test mathematical foundations
    math_success = test_mathematical_foundations()
    print()
    
    # Test CLI integration
    cli_success = test_cli_integration()
    print()
    
    # Overall results
    print("📋 Test Results Summary")
    print("=" * 70)
    print(f"Mathematical Foundations: {'✅ PASS' if math_success else '❌ FAIL'}")
    print(f"CLI Integration:          {'✅ PASS' if cli_success else '❌ FAIL'}")
    print()
    
    if math_success and cli_success:
        print("🎉 All tests passed! Mathematical foundations are ready to use.")
        print()
        print("🚀 Try these commands:")
        print("   python -m neurinspectre.cli math demo")
        print("   python -m neurinspectre.cli math spectral --input sample_clean_gradients.npy")
        print("   python -m neurinspectre.cli math integrate --input sample_obfuscated_gradients.npy --steps 50")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 