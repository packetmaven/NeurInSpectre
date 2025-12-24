#!/usr/bin/env python3
"""
Quick AI Model Listing - Shows installed frameworks and basic model info
"""

import os
import sys
from pathlib import Path
import importlib.util
import platform

def check_frameworks():
    """Check which AI frameworks are installed"""
    print("🔧 Installed AI Frameworks:")
    print("=" * 30)
    
    frameworks = {
        'torch': 'PyTorch',
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'sklearn': 'Scikit-Learn',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'transformers': 'Hugging Face Transformers',
        'onnx': 'ONNX',
        'onnxruntime': 'ONNX Runtime',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'spacy': 'spaCy',
        'nltk': 'NLTK'
    }
    
    installed = []
    
    for module, name in frameworks.items():
        try:
            spec = importlib.util.find_spec(module)
            if spec is not None:
                try:
                    mod = importlib.import_module(module)
                    version = getattr(mod, '__version__', 'Unknown')
                    print(f"✅ {name}: {version}")
                    installed.append((name, version))
                except:
                    print(f"✅ {name}: Available (version unknown)")
                    installed.append((name, 'Unknown'))
            else:
                print(f"❌ {name}: Not installed")
        except:
            print(f"❌ {name}: Not installed")
    
    return installed

def quick_model_scan():
    """Quick scan for model files in current directory"""
    print("\n📦 Model Files in Current Directory:")
    print("=" * 40)
    
    current_dir = Path.cwd()
    model_extensions = ['.pt', '.pth', '.ckpt', '.h5', '.hdf5', '.pkl', '.joblib', '.onnx', '.npy']
    
    found_models = []
    
    for ext in model_extensions:
        for model_file in current_dir.rglob(f'*{ext}'):
            if model_file.is_file():
                size_mb = model_file.stat().st_size / 1024 / 1024
                if size_mb > 0.1:  # Skip very small files
                    found_models.append({
                        'name': model_file.name,
                        'size_mb': round(size_mb, 2),
                        'path': str(model_file.relative_to(current_dir)),
                        'type': ext
                    })
    
    if found_models:
        # Sort by size
        found_models.sort(key=lambda x: x['size_mb'], reverse=True)
        
        for model in found_models:
            print(f"📄 {model['name']} ({model['size_mb']} MB)")
            print(f"   📁 {model['path']}")
        
        print(f"\n🎯 Found {len(found_models)} model files")
        total_size = sum(m['size_mb'] for m in found_models)
        print(f"📊 Total size: {total_size:.1f} MB")
    else:
        print("No model files found in current directory")
    
    return found_models

def check_huggingface_cache():
    """Check Hugging Face cache"""
    print("\n🤗 Hugging Face Cache:")
    print("=" * 25)
    
    hf_cache = Path.home() / '.cache' / 'huggingface'
    
    if hf_cache.exists():
        # Check hub cache
        hub_cache = hf_cache / 'hub'
        if hub_cache.exists():
            models = [d for d in hub_cache.iterdir() if d.is_dir() and d.name.startswith('models--')]
            print(f"📦 Hub cache: {len(models)} models")
            
            # Show a few examples
            for model_dir in models[:3]:
                model_name = model_dir.name.replace('models--', '').replace('--', '/')
                print(f"   🤗 {model_name}")
            
            if len(models) > 3:
                print(f"   ... and {len(models) - 3} more")
        
        # Check transformers cache
        transformers_cache = hf_cache / 'transformers'
        if transformers_cache.exists():
            models = [d for d in transformers_cache.iterdir() if d.is_dir()]
            print(f"📦 Transformers cache: {len(models)} models")
    else:
        print("No Hugging Face cache found")

def check_system_info():
    """Show system information"""
    print("💻 System Information:")
    print("=" * 25)
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    
    # Check GPU availability
    try:
        import torch
        if torch.backends.mps.is_available():
            print("🔥 PyTorch MPS (Mac GPU): Available")
        else:
            print("❌ PyTorch MPS: Not available")
    except:
        print("❌ PyTorch: Not installed")

def main():
    print("🚀 Quick AI Model Inventory")
    print("=" * 35)
    
    # System info
    check_system_info()
    
    print()
    
    # Check frameworks
    installed = check_frameworks()
    
    # Quick model scan
    models = quick_model_scan()
    
    # Check HF cache
    check_huggingface_cache()
    
    print(f"\n🎉 Summary:")
    print(f"   Frameworks installed: {len(installed)}")
    print(f"   Model files found: {len(models) if models else 0}")
    
    if models:
        total_size = sum(m['size_mb'] for m in models)
        print(f"   Total model size: {total_size:.1f} MB")

if __name__ == "__main__":
    main() 