#!/usr/bin/env python3
"""
Mac Silicon (Apple MPS) Detection Verification Script
Verifies that NeurInSpectre can detect and use Apple Silicon GPU
"""

import sys
import platform
import subprocess
import torch
import numpy as np
from pathlib import Path

def check_system_info():
    """Check basic system information"""
    print("🖥️  System Information:")
    print(f"   Platform: {platform.platform()}")
    print(f"   Machine: {platform.machine()}")
    print(f"   Processor: {platform.processor()}")
    print(f"   Python: {sys.version}")
    
    # Check if this is Apple Silicon
    is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
    print(f"   Apple Silicon: {'✅ YES' if is_apple_silicon else '❌ NO'}")
    return is_apple_silicon

def check_pytorch_mps():
    """Check PyTorch MPS (Metal Performance Shaders) support"""
    print("\n🔥 PyTorch MPS Detection:")
    print(f"   PyTorch Version: {torch.__version__}")
    
    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    print(f"   MPS Available: {'✅ YES' if mps_available else '❌ NO'}")
    
    if mps_available:
        mps_built = torch.backends.mps.is_built()
        print(f"   MPS Built: {'✅ YES' if mps_built else '❌ NO'}")
        
        # Test MPS device creation
        try:
            device = torch.device("mps")
            print(f"   MPS Device: ✅ {device}")
            
            # Test tensor operations on MPS
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            z = torch.matmul(x, y)
            print(f"   MPS Tensor Ops: ✅ Working (result shape: {z.shape})")
            
            return True, device
        except Exception as e:
            print(f"   MPS Device Error: ❌ {e}")
            return False, None
    
    return False, None

def check_metal_support():
    """Check Metal framework support"""
    print("\n⚡ Metal Framework Detection:")
    
    try:
        # Try to run system_profiler to get GPU info
        result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout
            if 'Apple' in output and ('M1' in output or 'M2' in output or 'M3' in output):
                print("   ✅ Apple Silicon GPU detected via system_profiler")
                
                # Extract GPU info
                lines = output.split('\n')
                for i, line in enumerate(lines):
                    if 'Chipset Model:' in line:
                        gpu_model = line.split(':')[1].strip()
                        print(f"   GPU Model: {gpu_model}")
                    elif 'Total Number of Cores:' in line:
                        cores = line.split(':')[1].strip()
                        print(f"   GPU Cores: {cores}")
                
                return True
            else:
                print("   ⚠️ No Apple Silicon GPU found in system_profiler")
        else:
            print(f"   ❌ system_profiler failed: {result.stderr}")
    
    except Exception as e:
        print(f"   ❌ Error checking Metal: {e}")
    
    return False

def test_neurinspectre_gpu_detection():
    """Test NeurInSpectre's GPU detection"""
    print("\n🧠 NeurInSpectre GPU Detection:")
    
    try:
        # Import NeurInSpectre's mathematical module
        from neurinspectre.mathematical.gpu_accelerated_math import GPUAcceleratedMathEngine
        
        gpu_math = GPUAcceleratedMathEngine()
        
        print(f"   Device Selected: {gpu_math.device}")
        print(f"   Device Type: {gpu_math.device.type}")
        
        if gpu_math.device.type == 'mps':
            print("   ✅ NeurInSpectre successfully detected Apple MPS!")
            
            # Test actual computation with spectral decomposition
            import numpy as np
            test_data = np.random.randn(100, 256).astype('float32')
            result = gpu_math.advanced_spectral_decomposition(test_data, decomposition_levels=3)
            print(f"   ✅ GPU computation successful: spectral analysis completed")
            print(f"   ✅ Device info: {gpu_math._get_device_info()}")
            
            return True
        else:
            print(f"   ⚠️ NeurInSpectre using {gpu_math.device.type} instead of MPS")
            return False
            
    except ImportError as e:
        print(f"   ❌ Cannot import NeurInSpectre GPU module: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error testing NeurInSpectre GPU: {e}")
        return False

def test_dashboard_gpu_detection():
    """Test dashboard's GPU detection"""
    print("\n📊 Dashboard GPU Detection:")
    
    dashboard_file = Path("research_materials/dashboards/enhanced_mps_atlas_agent_dashboard_COMPLETE_WORKING.py.backup")
    
    if not dashboard_file.exists():
        print("   ❌ Dashboard file not found")
        return False
    
    # Read dashboard file to check GPU detection code
    with open(dashboard_file, 'r') as f:
        content = f.read()
    
    if 'torch.backends.mps.is_available()' in content:
        print("   ✅ Dashboard has MPS detection code")
    else:
        print("   ⚠️ Dashboard may not have MPS detection")
    
    if 'mps' in content.lower():
        print("   ✅ Dashboard references MPS")
        return True
    else:
        print("   ❌ Dashboard does not reference MPS")
        return False

def run_comprehensive_verification():
    """Run all verification checks"""
    print("🔍 Mac Silicon (Apple MPS) Detection Verification")
    print("=" * 60)
    
    results = {}
    
    # System check
    results['apple_silicon'] = check_system_info()
    
    # PyTorch MPS check
    results['pytorch_mps'], mps_device = check_pytorch_mps()
    
    # Metal framework check
    results['metal'] = check_metal_support()
    
    # NeurInSpectre GPU detection
    results['neurinspectre'] = test_neurinspectre_gpu_detection()
    
    # Dashboard GPU detection
    results['dashboard'] = test_dashboard_gpu_detection()
    
    # Summary
    print("\n📋 Verification Summary:")
    print("=" * 30)
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check.replace('_', ' ').title()}: {status}")
    
    overall_success = all(results.values())
    print(f"\n🎯 Overall Status: {'✅ ALL SYSTEMS GO' if overall_success else '⚠️ ISSUES DETECTED'}")
    
    if overall_success:
        print("\n🚀 Your Mac Silicon GPU is properly detected and ready for use!")
        print("   The dashboard should automatically use Apple MPS for acceleration.")
    else:
        print("\n🔧 Recommendations:")
        if not results['apple_silicon']:
            print("   - This system is not Apple Silicon")
        if not results['pytorch_mps']:
            print("   - Update PyTorch: pip install torch torchvision torchaudio")
        if not results['neurinspectre']:
            print("   - Check NeurInSpectre installation")
        if not results['dashboard']:
            print("   - Dashboard may need MPS detection updates")
    
    return results

def get_gpu_info_commands():
    """Provide commands to check GPU info"""
    print("\n💻 Commands to Check Mac Silicon GPU:")
    print("=" * 40)
    
    commands = [
        ("System GPU Info", "system_profiler SPDisplaysDataType"),
        ("Hardware Overview", "system_profiler SPHardwareDataType"),
        ("PyTorch MPS Test", "python -c \"import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')\""),
        ("Metal Support", "python -c \"import platform; print(f'Apple Silicon: {platform.machine() == \"arm64\"}')\")"),
        ("NeurInSpectre GPU", "python -c \"from neurinspectre.mathematical.gpu_accelerated_math import GPUAcceleratedMathEngine; print(GPUAcceleratedMathEngine().device)\""),
    ]
    
    for name, cmd in commands:
        print(f"\n🔍 {name}:")
        print(f"   {cmd}")
    
    print(f"\n📊 Dashboard GPU Detection:")
    print(f"   python research_materials/dashboards/enhanced_mps_atlas_agent_dashboard_COMPLETE_WORKING.py.backup --port 8899")
    print(f"   # Look for 'GPU Type: MPS' in the dashboard output")

if __name__ == "__main__":
    # Run verification
    results = run_comprehensive_verification()
    
    # Show commands
    get_gpu_info_commands()
    
    print(f"\n🎯 Quick Test Command:")
    print(f"python -c \"import torch; print('MPS:', torch.backends.mps.is_available(), torch.device('mps') if torch.backends.mps.is_available() else 'Not available')\"") 