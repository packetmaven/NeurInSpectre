#!/usr/bin/env python3
"""
NeurInSpectre Debug Information Module
Provides system information for troubleshooting
"""

import sys
import os
import platform
import json
from importlib import metadata as importlib_metadata

def get_system_info():
    """Get basic system information"""
    return {
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "working_directory": os.getcwd(),
        "user": os.environ.get("USER", "unknown"),
        "shell": os.environ.get("SHELL", "unknown")
    }

def get_gpu_info():
    """Get GPU information"""
    gpu_info = {"mps_available": False, "cuda_available": False}

    import torch
    gpu_info["torch_version"] = torch.__version__
    gpu_info["mps_available"] = torch.backends.mps.is_available()
    gpu_info["cuda_available"] = torch.cuda.is_available()
    if gpu_info["cuda_available"]:
        gpu_info["cuda_version"] = torch.version.cuda
        gpu_info["cuda_device_count"] = torch.cuda.device_count()
    
    return gpu_info

def get_package_info():
    """Get installed package information"""
    packages: dict[str, str] = {}

    # Avoid pkg_resources (deprecated); prefer stdlib importlib.metadata.
    keys = [
        "neurinspectre",
        "torch",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "plotly",
        "dash",
        "transformers",
    ]
    for k in keys:
        try:
            packages[k] = importlib_metadata.version(k)
        except Exception:
            continue

    return packages

def get_neurinspectre_info():
    """Get NeurInSpectre specific information"""
    info = {}
    
    try:
        import neurinspectre
        info["version"] = getattr(neurinspectre, '__version__', 'unknown')
        info["path"] = neurinspectre.__file__
        
        # Check for key modules
        modules = ['cli', 'mathematical', 'security', 'statistical']
        for module in modules:
            try:
                __import__(f'neurinspectre.{module}')
                info[f"{module}_available"] = True
            except ImportError as e:
                info[f"{module}_available"] = False
                info[f"{module}_error"] = str(e)
                
    except ImportError as e:
        info["error"] = f"NeurInSpectre not properly installed: {e}"
    
    return info

def main():
    """Main debug info function"""
    print("🔍 NeurInSpectre Debug Information")
    print("=" * 50)
    
    # System Information
    print("\n💻 System Information:")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # GPU Information
    print("\n🎮 GPU Information:")
    gpu_info = get_gpu_info()
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    
    # Package Information
    print("\n📦 Key Packages:")
    packages = get_package_info()
    for package, version in packages.items():
        print(f"  {package}: {version}")
    
    # NeurInSpectre Information
    print("\n🧠 NeurInSpectre Information:")
    neurinspectre_info = get_neurinspectre_info()
    for key, value in neurinspectre_info.items():
        print(f"  {key}: {value}")
    
    # Create JSON report
    debug_data = {
        "system": system_info,
        "gpu": gpu_info,
        "packages": packages,
        "neurinspectre": neurinspectre_info
    }
    
    print("\n💾 Debug report also available as JSON:")
    print(json.dumps(debug_data, indent=2))

if __name__ == '__main__':
    main() 