#!/usr/bin/env python3
"""
Quick NVIDIA GPU Check - Fast detection of NVIDIA hardware and CUDA support
"""

import subprocess
import sys
import platform
import importlib.util

def check_nvidia_smi():
    """Quick check for nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_info = result.stdout.strip()
            driver_version = version_info.split('Driver Version: ')[1].split()[0] if 'Driver Version:' in version_info else 'Unknown'
            print(f"✅ NVIDIA SMI: Available (Driver: {driver_version})")
            return True
        else:
            print("❌ NVIDIA SMI: Not available")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ NVIDIA SMI: Not found")
        return False

def quick_gpu_info():
    """Get basic GPU information"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=name,memory.total,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print(f"\n🎯 Found {len(lines)} NVIDIA GPU(s):")
            
            for i, line in enumerate(lines):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        name = parts[0]
                        memory = parts[1]
                        usage = parts[2]
                        temp = parts[3]
                        print(f"   GPU {i}: {name}")
                        print(f"      Memory: {memory} MB total")
                        print(f"      Usage: {usage}% | Temp: {temp}°C")
            return True
        else:
            print("❌ Could not get GPU information")
            return False
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        print("❌ Could not get GPU information")
        return False

def check_cuda_frameworks():
    """Check CUDA support in AI frameworks"""
    print("\n🔥 CUDA Framework Support:")
    print("=" * 30)
    
    frameworks_checked = 0
    frameworks_available = 0
    
    # PyTorch CUDA
    try:
        import torch
        frameworks_checked += 1
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            cuda_version = torch.version.cuda
            
            print(f"✅ PyTorch CUDA: {device_count} device(s) (v{cuda_version})")
            print(f"   Current: GPU {current_device} ({device_name})")
            frameworks_available += 1
        else:
            print("❌ PyTorch CUDA: Not available")
    except ImportError:
        print("❌ PyTorch: Not installed")
    
    # TensorFlow GPU
    try:
        import tensorflow as tf
        frameworks_checked += 1
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow GPU: {len(gpus)} device(s)")
            frameworks_available += 1
        else:
            print("❌ TensorFlow GPU: No devices found")
    except ImportError:
        print("❌ TensorFlow: Not installed")
    
    # JAX GPU
    try:
        import jax
        frameworks_checked += 1
        devices = jax.devices('gpu')
        if devices:
            print(f"✅ JAX GPU: {len(devices)} device(s)")
            frameworks_available += 1
        else:
            print("❌ JAX GPU: No devices found")
    except ImportError:
        print("❌ JAX: Not installed")
    
    return frameworks_checked, frameworks_available

def check_running_processes():
    """Check for running GPU processes"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            
            if lines:
                print(f"\n🔄 Running GPU Processes ({len(lines)}):")
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        pid = parts[0]
                        name = parts[1]
                        memory = parts[2]
                        print(f"   🔹 {name} (PID: {pid}) - {memory} MB")
                return len(lines)
            else:
                print("\n🔄 No GPU processes currently running")
                return 0
        else:
            print("\n❌ Could not check running processes")
            return 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        print("\n❌ Could not check running processes")
        return 0

def system_info():
    """Show basic system information"""
    print("💻 System Information:")
    print("=" * 25)
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {platform.python_version()}")

def main():
    print("⚡ Quick NVIDIA GPU Check")
    print("=" * 30)
    
    # System info
    system_info()
    
    print()
    
    # Check nvidia-smi
    nvidia_available = check_nvidia_smi()
    
    if nvidia_available:
        # Get GPU info
        gpu_detected = quick_gpu_info()
        
        # Check CUDA frameworks
        frameworks_checked, frameworks_available = check_cuda_frameworks()
        
        # Check running processes
        running_processes = check_running_processes()
        
        # Summary
        print(f"\n🎉 Quick Summary:")
        print(f"   NVIDIA Driver: Available")
        print(f"   GPUs Detected: {'Yes' if gpu_detected else 'No'}")
        print(f"   CUDA Frameworks: {frameworks_available}/{frameworks_checked} available")
        print(f"   Running Processes: {running_processes}")
        
        if frameworks_available > 0:
            print(f"\n✅ NVIDIA GPU setup is working!")
        else:
            print(f"\n⚠️  NVIDIA GPU detected but no CUDA frameworks available")
    else:
        print(f"\n❌ No NVIDIA GPU or drivers detected")
        print(f"   This system may not have NVIDIA hardware")
        print(f"   Or NVIDIA drivers may not be installed")

if __name__ == "__main__":
    main() 