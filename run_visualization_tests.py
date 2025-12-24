#!/usr/bin/env python3
"""
NeurInSpectre Visualization Test Runner
Runs all tested and verified CLI visualization commands
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(cmd, description, timeout=60):
    """Run a command and return the result"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({end_time - start_time:.2f}s)")
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ FAILED ({end_time - start_time:.2f}s)")
            if result.stderr:
                print(f"Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False

def main():
    """Run all visualization tests"""
    print("🎨 NeurInSpectre Visualization Test Suite")
    print("=" * 60)
    print(f"Directory: {os.getcwd()}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Core module imports
    test_imports = [
        ("from neurinspectre.security.blue_team_intelligence import BlueTeamIntelligenceEngine; print('BlueTeamIntelligenceEngine imported')", "Blue Team Intelligence Import"),
        ("from neurinspectre.security.red_team_intelligence import RedTeamIntelligenceEngine; print('RedTeamIntelligenceEngine imported')", "Red Team Intelligence Import"),
        ("from neurinspectre.security.critical_rl_obfuscation import CriticalRLObfuscationDetector; print('CriticalRLObfuscationDetector imported')", "Critical RL Obfuscation Import"),
        ("from neurinspectre.integrated_neurinspectre_system import IntegratedNeurInSpectre; print('IntegratedNeurInSpectre imported')", "Integrated System Import"),
        ("from neurinspectre.security.visualization.obfuscated_gradient_visualizer import ObfuscatedGradientVisualizer; print('ObfuscatedGradientVisualizer imported')", "Obfuscated Gradient Visualizer Import")
    ]
    
    print("\n📦 Testing Core Module Imports...")
    for cmd, desc in test_imports:
        results[desc] = run_command(f'python -c "{cmd}"', desc, timeout=30)
    
    # Test 2: GPU Detection
    print("\n🖥️ Testing GPU Detection...")
    gpu_cmd = """
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if hasattr(torch.backends, 'mps'):
    print(f'MPS Available: {torch.backends.mps.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
print(f'Selected Device: {device}')
"""
    results["GPU Detection"] = run_command(f'python -c "{gpu_cmd}"', "GPU Detection Test", timeout=15)
    
    # Test 3: Mathematical foundations
    print("\n🧮 Testing Mathematical Foundations...")
    math_cmd = "import neurinspectre.mathematical.advanced_mathematical_foundations as math_foundations; print('Mathematical foundations loaded successfully')"
    results["Mathematical Foundations"] = run_command(f'python -c "{math_cmd}"', "Mathematical Foundations Test", timeout=15)
    
    # Test 4: Data file verification
    print("\n📊 Testing Data Files...")
    data_files = [
        "testing_suite/test_data/sample_clean_gradients.npy",
        "testing_suite/test_data/sample_obfuscated_gradients.npy"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - EXISTS")
            results[f"Data File: {file_path}"] = True
        else:
            print(f"❌ {file_path} - MISSING")
            results[f"Data File: {file_path}"] = False
    
    # Test 5: Obfuscated Gradient Visualizer (main test)
    print("\n🎯 Testing Obfuscated Gradient Visualizer...")
    viz_cmd = "python neurinspectre/security/visualization/obfuscated_gradient_visualizer.py"
    results["Obfuscated Gradient Visualizer"] = run_command(viz_cmd, "Obfuscated Gradient Visualizer", timeout=120)
    
    # Test 6: Demo Dashboard
    print("\n🎨 Testing Demo Dashboard...")
    dashboard_cmd = "python research_materials/examples/demo_red_blue_dashboard.py"
    results["Demo Red Blue Dashboard"] = run_command(dashboard_cmd, "Demo Red Blue Dashboard", timeout=120)
    
    # Test 7: CLI Access
    print("\n🖥️ Testing CLI Access...")
    cli_cmd = "python -c \"import neurinspectre.cli; print('CLI module accessible')\""
    results["CLI Access"] = run_command(cli_cmd, "CLI Access Test", timeout=30)
    
    # Test 8: System Integration
    print("\n🔧 Testing System Integration...")
    integration_cmd = """
from neurinspectre.integrated_neurinspectre_system import IntegratedNeurInSpectre
system = IntegratedNeurInSpectre(sensitivity_profile='adaptive')
print('Integrated system initialized successfully')
print(f'Device: {system.device}')
print(f'Sensitivity: {system.sensitivity_profile}')
"""
    results["System Integration"] = run_command(f'python -c "{integration_cmd}"', "System Integration Test", timeout=45)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    success_rate = (passed / total) * 100
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} | {test_name}")
    
    print("-" * 60)
    print(f"📈 Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 NeurInSpectre visualizations are working correctly!")
        return 0
    elif success_rate >= 60:
        print("⚠️ Most visualizations working, some issues detected")
        return 1
    else:
        print("🚨 Multiple visualization issues detected")
        return 2

if __name__ == "__main__":
    exit_code = main()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Check generated visualization files (*.png)")
    print("2. Review any failed tests above")
    print("3. Run individual commands from CLI_VISUALIZATION_COMMANDS.md")
    print("4. Check requirements.txt for missing dependencies")
    print("=" * 60)
    
    sys.exit(exit_code) 