#!/usr/bin/env python3
"""
Comprehensive AI Model Inventory for Mac Silicon
Lists all installed AI models, frameworks, and related files
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
import importlib.util
from collections import defaultdict
import hashlib

class AIModelInventory:
    def __init__(self):
        self.home_dir = Path.home()
        self.current_dir = Path.cwd()
        self.models_found = defaultdict(list)
        
    def scan_pytorch_models(self):
        """Scan for PyTorch model files"""
        print("🔥 Scanning PyTorch Models...")
        
        pytorch_extensions = ['.pt', '.pth', '.ckpt', '.safetensors']
        search_paths = [
            self.current_dir,
            self.home_dir / '.cache' / 'torch',
            self.home_dir / '.cache' / 'huggingface',
            self.home_dir / 'Downloads',
            self.home_dir / 'models',
            Path('/opt/homebrew/lib/python*/site-packages') if Path('/opt/homebrew').exists() else None,
        ]
        
        # Add conda/pip site-packages
        try:
            import torch
            torch_path = Path(torch.__file__).parent.parent
            search_paths.append(torch_path)
        except:
            pass
        
        pytorch_models = []
        
        for search_path in search_paths:
            if search_path and search_path.exists():
                try:
                    for ext in pytorch_extensions:
                        for model_file in search_path.rglob(f'*{ext}'):
                            if model_file.is_file():
                                size_mb = model_file.stat().st_size / 1024 / 1024
                                
                                # Skip very small files (likely not actual models)
                                if size_mb > 0.1:
                                    pytorch_models.append({
                                        'name': model_file.name,
                                        'path': str(model_file),
                                        'size_mb': round(size_mb, 2),
                                        'type': 'PyTorch',
                                        'extension': ext,
                                        'location': str(model_file.parent)
                                    })
                except (PermissionError, OSError):
                    continue
        
        # Sort by size (largest first)
        pytorch_models.sort(key=lambda x: x['size_mb'], reverse=True)
        
        print(f"   Found {len(pytorch_models)} PyTorch models:")
        for model in pytorch_models[:10]:  # Show top 10
            print(f"      📦 {model['name']} ({model['size_mb']} MB)")
            print(f"         📁 {model['location']}")
        
        if len(pytorch_models) > 10:
            print(f"      ... and {len(pytorch_models) - 10} more")
        
        self.models_found['pytorch'] = pytorch_models
        return pytorch_models
    
    def scan_tensorflow_models(self):
        """Scan for TensorFlow/Keras models"""
        print("\n🧠 Scanning TensorFlow/Keras Models...")
        
        tf_extensions = ['.h5', '.hdf5', '.keras', '.pb', '.tflite']
        tf_models = []
        
        search_paths = [
            self.current_dir,
            self.home_dir / '.keras',
            self.home_dir / 'Downloads',
            self.home_dir / 'models',
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                try:
                    for ext in tf_extensions:
                        for model_file in search_path.rglob(f'*{ext}'):
                            if model_file.is_file():
                                size_mb = model_file.stat().st_size / 1024 / 1024
                                
                                if size_mb > 0.1:
                                    tf_models.append({
                                        'name': model_file.name,
                                        'path': str(model_file),
                                        'size_mb': round(size_mb, 2),
                                        'type': 'TensorFlow/Keras',
                                        'extension': ext,
                                        'location': str(model_file.parent)
                                    })
                except (PermissionError, OSError):
                    continue
        
        tf_models.sort(key=lambda x: x['size_mb'], reverse=True)
        
        print(f"   Found {len(tf_models)} TensorFlow/Keras models:")
        for model in tf_models[:5]:
            print(f"      📦 {model['name']} ({model['size_mb']} MB)")
            print(f"         📁 {model['location']}")
        
        self.models_found['tensorflow'] = tf_models
        return tf_models
    
    def scan_huggingface_models(self):
        """Scan for Hugging Face models"""
        print("\n🤗 Scanning Hugging Face Models...")
        
        hf_cache_dir = self.home_dir / '.cache' / 'huggingface'
        hf_models = []
        
        if hf_cache_dir.exists():
            try:
                # Look in transformers cache
                transformers_cache = hf_cache_dir / 'transformers'
                if transformers_cache.exists():
                    for model_dir in transformers_cache.iterdir():
                        if model_dir.is_dir():
                            total_size = 0
                            file_count = 0
                            
                            for file_path in model_dir.rglob('*'):
                                if file_path.is_file():
                                    total_size += file_path.stat().st_size
                                    file_count += 1
                            
                            if total_size > 1024 * 1024:  # > 1MB
                                size_mb = total_size / 1024 / 1024
                                hf_models.append({
                                    'name': model_dir.name,
                                    'path': str(model_dir),
                                    'size_mb': round(size_mb, 2),
                                    'type': 'Hugging Face',
                                    'files': file_count,
                                    'location': str(model_dir.parent)
                                })
                
                # Look in hub cache (newer format)
                hub_cache = hf_cache_dir / 'hub'
                if hub_cache.exists():
                    for model_dir in hub_cache.iterdir():
                        if model_dir.is_dir() and model_dir.name.startswith('models--'):
                            total_size = 0
                            file_count = 0
                            
                            for file_path in model_dir.rglob('*'):
                                if file_path.is_file():
                                    total_size += file_path.stat().st_size
                                    file_count += 1
                            
                            if total_size > 1024 * 1024:  # > 1MB
                                size_mb = total_size / 1024 / 1024
                                model_name = model_dir.name.replace('models--', '').replace('--', '/')
                                hf_models.append({
                                    'name': model_name,
                                    'path': str(model_dir),
                                    'size_mb': round(size_mb, 2),
                                    'type': 'Hugging Face Hub',
                                    'files': file_count,
                                    'location': str(model_dir.parent)
                                })
                
            except (PermissionError, OSError):
                pass
        
        hf_models.sort(key=lambda x: x['size_mb'], reverse=True)
        
        print(f"   Found {len(hf_models)} Hugging Face models:")
        for model in hf_models[:5]:
            print(f"      🤗 {model['name']} ({model['size_mb']} MB)")
            print(f"         📁 {model['location']}")
        
        self.models_found['huggingface'] = hf_models
        return hf_models
    
    def scan_scikit_learn_models(self):
        """Scan for scikit-learn models"""
        print("\n📊 Scanning Scikit-Learn Models...")
        
        sklearn_extensions = ['.pkl', '.joblib', '.pickle']
        sklearn_models = []
        
        search_paths = [
            self.current_dir,
            self.home_dir / 'Downloads',
            self.home_dir / 'models',
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                try:
                    for ext in sklearn_extensions:
                        for model_file in search_path.rglob(f'*{ext}'):
                            if model_file.is_file():
                                size_mb = model_file.stat().st_size / 1024 / 1024
                                
                                # Check if it might be a model file
                                if size_mb > 0.01 and any(keyword in model_file.name.lower() for keyword in 
                                    ['model', 'classifier', 'regressor', 'pipeline', 'estimator']):
                                    sklearn_models.append({
                                        'name': model_file.name,
                                        'path': str(model_file),
                                        'size_mb': round(size_mb, 2),
                                        'type': 'Scikit-Learn',
                                        'extension': ext,
                                        'location': str(model_file.parent)
                                    })
                except (PermissionError, OSError):
                    continue
        
        sklearn_models.sort(key=lambda x: x['size_mb'], reverse=True)
        
        print(f"   Found {len(sklearn_models)} Scikit-Learn models:")
        for model in sklearn_models:
            print(f"      📊 {model['name']} ({model['size_mb']} MB)")
            print(f"         📁 {model['location']}")
        
        self.models_found['sklearn'] = sklearn_models
        return sklearn_models
    
    def scan_onnx_models(self):
        """Scan for ONNX models"""
        print("\n⚙️ Scanning ONNX Models...")
        
        onnx_models = []
        search_paths = [
            self.current_dir,
            self.home_dir / 'Downloads',
            self.home_dir / 'models',
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                try:
                    for model_file in search_path.rglob('*.onnx'):
                        if model_file.is_file():
                            size_mb = model_file.stat().st_size / 1024 / 1024
                            
                            onnx_models.append({
                                'name': model_file.name,
                                'path': str(model_file),
                                'size_mb': round(size_mb, 2),
                                'type': 'ONNX',
                                'extension': '.onnx',
                                'location': str(model_file.parent)
                            })
                except (PermissionError, OSError):
                    continue
        
        onnx_models.sort(key=lambda x: x['size_mb'], reverse=True)
        
        print(f"   Found {len(onnx_models)} ONNX models:")
        for model in onnx_models:
            print(f"      ⚙️ {model['name']} ({model['size_mb']} MB)")
            print(f"         📁 {model['location']}")
        
        self.models_found['onnx'] = onnx_models
        return onnx_models
    
    def scan_neurinspectre_models(self):
        """Scan for NeurInSpectre-specific models"""
        print("\n🧠 Scanning NeurInSpectre Models...")
        
        neurinspectre_models = []
        
        # Check dataset cache
        dataset_cache = self.current_dir / 'neurinspectre' / 'dataset_cache'
        if dataset_cache.exists():
            for dataset_dir in dataset_cache.iterdir():
                if dataset_dir.is_dir():
                    total_size = 0
                    file_count = 0
                    
                    for file_path in dataset_dir.rglob('*'):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                            file_count += 1
                    
                    if total_size > 1024:  # > 1KB
                        size_mb = total_size / 1024 / 1024
                        neurinspectre_models.append({
                            'name': f"Dataset: {dataset_dir.name}",
                            'path': str(dataset_dir),
                            'size_mb': round(size_mb, 2),
                            'type': 'NeurInSpectre Dataset',
                            'files': file_count,
                            'location': str(dataset_dir.parent)
                        })
        
        # Check for other NeurInSpectre files
        neurinspectre_dir = self.current_dir / 'neurinspectre'
        if neurinspectre_dir.exists():
            for model_file in neurinspectre_dir.rglob('*.npy'):
                if model_file.is_file():
                    size_mb = model_file.stat().st_size / 1024 / 1024
                    if size_mb > 0.01:
                        neurinspectre_models.append({
                            'name': model_file.name,
                            'path': str(model_file),
                            'size_mb': round(size_mb, 2),
                            'type': 'NeurInSpectre Data',
                            'extension': '.npy',
                            'location': str(model_file.parent)
                        })
        
        neurinspectre_models.sort(key=lambda x: x['size_mb'], reverse=True)
        
        print(f"   Found {len(neurinspectre_models)} NeurInSpectre models/datasets:")
        for model in neurinspectre_models:
            print(f"      🧠 {model['name']} ({model['size_mb']} MB)")
            print(f"         📁 {model['location']}")
        
        self.models_found['neurinspectre'] = neurinspectre_models
        return neurinspectre_models
    
    def check_installed_frameworks(self):
        """Check which AI frameworks are installed"""
        print("\n🔧 Checking Installed AI Frameworks...")
        
        frameworks = {
            'torch': 'PyTorch',
            'tensorflow': 'TensorFlow',
            'keras': 'Keras',
            'sklearn': 'Scikit-Learn',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM',
            'catboost': 'CatBoost',
            'transformers': 'Hugging Face Transformers',
            'onnx': 'ONNX',
            'onnxruntime': 'ONNX Runtime',
            'jax': 'JAX',
            'flax': 'Flax',
            'optax': 'Optax',
            'cv2': 'OpenCV',
            'PIL': 'Pillow',
            'spacy': 'spaCy',
            'nltk': 'NLTK',
            'gensim': 'Gensim',
            'gym': 'OpenAI Gym',
            'stable_baselines3': 'Stable Baselines3'
        }
        
        installed = {}
        
        for module, name in frameworks.items():
            try:
                spec = importlib.util.find_spec(module)
                if spec is not None:
                    # Try to get version
                    try:
                        mod = importlib.import_module(module)
                        version = getattr(mod, '__version__', 'Unknown')
                    except:
                        version = 'Unknown'
                    
                    installed[name] = version
                    print(f"   ✅ {name}: {version}")
                else:
                    print(f"   ❌ {name}: Not installed")
            except:
                print(f"   ❌ {name}: Not installed")
        
        return installed
    
    def generate_summary(self):
        """Generate a comprehensive summary"""
        print("\n📋 Model Inventory Summary:")
        print("=" * 50)
        
        total_models = 0
        total_size_mb = 0
        
        for framework, models in self.models_found.items():
            count = len(models)
            size = sum(model.get('size_mb', 0) for model in models)
            total_models += count
            total_size_mb += size
            
            if count > 0:
                print(f"   {framework.title()}: {count} models ({size:.1f} MB)")
        
        print(f"\n🎯 Total: {total_models} models ({total_size_mb:.1f} MB)")
        
        # Top 10 largest models
        all_models = []
        for models in self.models_found.values():
            all_models.extend(models)
        
        all_models.sort(key=lambda x: x.get('size_mb', 0), reverse=True)
        
        if all_models:
            print(f"\n🏆 Top 10 Largest Models:")
            for i, model in enumerate(all_models[:10], 1):
                print(f"   {i}. {model['name']} ({model['size_mb']} MB) - {model['type']}")
        
        return {
            'total_models': total_models,
            'total_size_mb': round(total_size_mb, 2),
            'by_framework': {k: len(v) for k, v in self.models_found.items()},
            'largest_models': all_models[:10]
        }
    
    def save_inventory(self):
        """Save complete inventory to file"""
        inventory_data = {
            'system_info': {
                'platform': platform.platform(),
                'machine': platform.machine(),
                'python_version': platform.python_version()
            },
            'models_by_framework': dict(self.models_found),
            'summary': self.generate_summary()
        }
        
        with open('ai_model_inventory.json', 'w') as f:
            json.dump(inventory_data, f, indent=2, default=str)
        
        print(f"\n💾 Complete inventory saved to: ai_model_inventory.json")
    
    def run_full_scan(self):
        """Run complete model inventory scan"""
        print("🤖 AI Model Inventory Scanner")
        print("=" * 40)
        print(f"📁 Scanning from: {self.current_dir}")
        print(f"🏠 Home directory: {self.home_dir}")
        
        # Run all scans
        self.scan_pytorch_models()
        self.scan_tensorflow_models()
        self.scan_huggingface_models()
        self.scan_scikit_learn_models()
        self.scan_onnx_models()
        self.scan_neurinspectre_models()
        
        # Check frameworks
        installed_frameworks = self.check_installed_frameworks()
        
        # Generate summary
        summary = self.generate_summary()
        
        # Save to file
        self.save_inventory()
        
        return summary, installed_frameworks

def main():
    scanner = AIModelInventory()
    summary, frameworks = scanner.run_full_scan()
    
    print(f"\n🎉 Scan Complete!")
    print(f"   Found {summary['total_models']} AI models")
    print(f"   Total size: {summary['total_size_mb']} MB")
    print(f"   Installed frameworks: {len(frameworks)}")

if __name__ == "__main__":
    main() 