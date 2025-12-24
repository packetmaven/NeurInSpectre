#!/usr/bin/env python3
"""
NVIDIA GPU Detection and Model Inventory
Detects NVIDIA hardware, drivers, and running AI models
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
import importlib.util
from collections import defaultdict
import time
import psutil

class NVIDIADetector:
    def __init__(self):
        self.gpu_info = {}
        self.running_processes = []
        self.cuda_available = False
        self.nvidia_smi_available = False
        
    def check_nvidia_smi(self):
        """Check if nvidia-smi is available"""
        try:
            result = subprocess.run(['nvidia-smi', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.nvidia_smi_available = True
                print("✅ nvidia-smi: Available")
                print(f"   {result.stdout.strip().split('NVIDIA-SMI')[1].split('Driver')[0].strip()}")
                return True
            else:
                print("❌ nvidia-smi: Not available")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            print("❌ nvidia-smi: Not found")
            return False
    
    def get_gpu_info(self):
        """Get detailed GPU information"""
        if not self.nvidia_smi_available:
            return {}
        
        try:
            # Get GPU info in XML format for easier parsing
            result = subprocess.run([
                'nvidia-smi', '-q', '-x'
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                # Parse basic info with nvidia-smi
                basic_result = subprocess.run([
                    'nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=10)
                
                if basic_result.returncode == 0:
                    gpus = []
                    for line in basic_result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 10:
                                gpu = {
                                    'index': int(parts[0]),
                                    'name': parts[1],
                                    'memory_total_mb': int(parts[2]),
                                    'memory_used_mb': int(parts[3]),
                                    'memory_free_mb': int(parts[4]),
                                    'gpu_utilization_percent': parts[5],
                                    'memory_utilization_percent': parts[6],
                                    'temperature_c': parts[7],
                                    'power_draw_w': parts[8],
                                    'power_limit_w': parts[9]
                                }
                                gpus.append(gpu)
                    
                    self.gpu_info = {'gpus': gpus}
                    return self.gpu_info
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
        
        return {}
    
    def check_cuda_availability(self):
        """Check CUDA availability in various frameworks"""
        print("\n🔥 CUDA Framework Support:")
        print("=" * 30)
        
        # PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                print(f"✅ PyTorch CUDA: Available")
                print(f"   CUDA Version: {cuda_version}")
                print(f"   Device Count: {device_count}")
                print(f"   Current Device: {current_device} ({device_name})")
                
                self.cuda_available = True
                
                # Check memory
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / 1024**3
                    print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
            else:
                print("❌ PyTorch CUDA: Not available")
        except ImportError:
            print("❌ PyTorch: Not installed")
        except Exception as e:
            print(f"❌ PyTorch CUDA: Error - {e}")
        
        # TensorFlow GPU
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                print(f"✅ TensorFlow GPU: {len(gpus)} device(s)")
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
            else:
                print("❌ TensorFlow GPU: No devices found")
        except ImportError:
            print("❌ TensorFlow: Not installed")
        except Exception as e:
            print(f"❌ TensorFlow GPU: Error - {e}")
        
        # JAX GPU
        try:
            import jax
            devices = jax.devices('gpu')
            if devices:
                print(f"✅ JAX GPU: {len(devices)} device(s)")
                for i, device in enumerate(devices):
                    print(f"   GPU {i}: {device}")
            else:
                print("❌ JAX GPU: No devices found")
        except ImportError:
            print("❌ JAX: Not installed")
        except Exception as e:
            print(f"❌ JAX GPU: Error - {e}")
    
    def get_running_processes(self):
        """Get processes currently using GPU"""
        if not self.nvidia_smi_available:
            return []
        
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            processes = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            pid = int(parts[0])
                            
                            # Get additional process info
                            try:
                                proc = psutil.Process(pid)
                                process_info = {
                                    'pid': pid,
                                    'name': parts[1],
                                    'gpu_uuid': parts[2],
                                    'gpu_memory_mb': int(parts[3]),
                                    'command': ' '.join(proc.cmdline()[:3]),  # First 3 args
                                    'cpu_percent': proc.cpu_percent(),
                                    'memory_mb': proc.memory_info().rss / 1024 / 1024,
                                    'status': proc.status()
                                }
                                processes.append(process_info)
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                # Process might have ended or no permission
                                process_info = {
                                    'pid': pid,
                                    'name': parts[1],
                                    'gpu_uuid': parts[2],
                                    'gpu_memory_mb': int(parts[3]),
                                    'command': 'Unknown',
                                    'cpu_percent': 0,
                                    'memory_mb': 0,
                                    'status': 'Unknown'
                                }
                                processes.append(process_info)
            
            self.running_processes = processes
            return processes
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return []
    
    def scan_nvidia_models(self):
        """Scan for NVIDIA-specific model files"""
        print("\n🎯 NVIDIA Model Files:")
        print("=" * 25)
        
        # Look for TensorRT models
        tensorrt_extensions = ['.trt', '.engine', '.plan']
        # Look for CUDA-specific models
        cuda_extensions = ['.cubin', '.ptx']
        # Look for models that might be optimized for NVIDIA
        nvidia_keywords = ['tensorrt', 'cuda', 'nvidia', 'gpu', 'trt']
        
        current_dir = Path.cwd()
        home_dir = Path.home()
        
        search_paths = [
            current_dir,
            home_dir / 'Downloads',
            home_dir / 'models',
            home_dir / '.cache'
        ]
        
        nvidia_models = []
        
        for search_path in search_paths:
            if search_path.exists():
                try:
                    # Look for TensorRT files
                    for ext in tensorrt_extensions + cuda_extensions:
                        for model_file in search_path.rglob(f'*{ext}'):
                            if model_file.is_file():
                                size_mb = model_file.stat().st_size / 1024 / 1024
                                nvidia_models.append({
                                    'name': model_file.name,
                                    'path': str(model_file),
                                    'size_mb': round(size_mb, 2),
                                    'type': f'NVIDIA {ext.upper()}',
                                    'location': str(model_file.parent)
                                })
                    
                    # Look for files with NVIDIA keywords
                    for model_file in search_path.rglob('*'):
                        if model_file.is_file() and any(keyword in model_file.name.lower() for keyword in nvidia_keywords):
                            if model_file.suffix in ['.pt', '.pth', '.onnx', '.pb']:
                                size_mb = model_file.stat().st_size / 1024 / 1024
                                if size_mb > 0.1:
                                    nvidia_models.append({
                                        'name': model_file.name,
                                        'path': str(model_file),
                                        'size_mb': round(size_mb, 2),
                                        'type': 'NVIDIA-optimized',
                                        'location': str(model_file.parent)
                                    })
                except (PermissionError, OSError):
                    continue
        
        # Remove duplicates
        seen = set()
        unique_models = []
        for model in nvidia_models:
            if model['path'] not in seen:
                seen.add(model['path'])
                unique_models.append(model)
        
        nvidia_models = sorted(unique_models, key=lambda x: x['size_mb'], reverse=True)
        
        if nvidia_models:
            print(f"Found {len(nvidia_models)} NVIDIA-specific models:")
            for model in nvidia_models:
                print(f"   ⚡ {model['name']} ({model['size_mb']} MB) - {model['type']}")
                print(f"      📁 {model['location']}")
        else:
            print("No NVIDIA-specific model files found")
        
        return nvidia_models
    
    def monitor_gpu_usage(self, duration=5):
        """Monitor GPU usage for a short period"""
        if not self.nvidia_smi_available:
            return
        
        print(f"\n📊 GPU Usage Monitor ({duration}s):")
        print("=" * 30)
        
        for i in range(duration):
            try:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for gpu_idx, line in enumerate(lines):
                        if line.strip():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 4:
                                gpu_util = parts[0]
                                mem_used = parts[1]
                                mem_total = parts[2]
                                temp = parts[3]
                                
                                print(f"   GPU {gpu_idx}: {gpu_util}% GPU, {mem_used}/{mem_total} MB, {temp}°C")
                
                if i < duration - 1:
                    time.sleep(1)
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                break
    
    def generate_report(self):
        """Generate comprehensive NVIDIA report"""
        print("\n📋 NVIDIA GPU Report:")
        print("=" * 25)
        
        if self.gpu_info and 'gpus' in self.gpu_info:
            print(f"🎯 Found {len(self.gpu_info['gpus'])} NVIDIA GPU(s):")
            
            for gpu in self.gpu_info['gpus']:
                print(f"\n   GPU {gpu['index']}: {gpu['name']}")
                print(f"      Memory: {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB ({gpu['memory_utilization_percent']}%)")
                print(f"      GPU Usage: {gpu['gpu_utilization_percent']}%")
                print(f"      Temperature: {gpu['temperature_c']}°C")
                print(f"      Power: {gpu['power_draw_w']}/{gpu['power_limit_w']} W")
        else:
            print("❌ No NVIDIA GPUs detected")
        
        if self.running_processes:
            print(f"\n🔄 Running GPU Processes ({len(self.running_processes)}):")
            for proc in self.running_processes:
                print(f"   PID {proc['pid']}: {proc['name']}")
                print(f"      Command: {proc['command']}")
                print(f"      GPU Memory: {proc['gpu_memory_mb']} MB")
                print(f"      Status: {proc['status']}")
        else:
            print("\n🔄 No GPU processes currently running")
    
    def save_report(self):
        """Save detailed report to file"""
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'platform': platform.platform(),
                'machine': platform.machine(),
                'python_version': platform.python_version()
            },
            'nvidia_smi_available': self.nvidia_smi_available,
            'cuda_available': self.cuda_available,
            'gpu_info': self.gpu_info,
            'running_processes': self.running_processes
        }
        
        with open('nvidia_gpu_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: nvidia_gpu_report.json")
    
    def run_full_detection(self):
        """Run complete NVIDIA detection"""
        print("🚀 NVIDIA GPU Detection & Model Inventory")
        print("=" * 45)
        
        # Check nvidia-smi
        self.check_nvidia_smi()
        
        # Get GPU info
        if self.nvidia_smi_available:
            print("\n💻 GPU Hardware Information:")
            print("=" * 35)
            self.get_gpu_info()
            
            if self.gpu_info:
                for gpu in self.gpu_info.get('gpus', []):
                    print(f"🎯 GPU {gpu['index']}: {gpu['name']}")
                    print(f"   Memory: {gpu['memory_total_mb']} MB total")
                    print(f"   Current Usage: {gpu['gpu_utilization_percent']}% GPU, {gpu['memory_utilization_percent']}% Memory")
        
        # Check CUDA support
        self.check_cuda_availability()
        
        # Get running processes
        print("\n🔄 GPU Process Monitor:")
        print("=" * 25)
        processes = self.get_running_processes()
        
        if processes:
            print(f"Found {len(processes)} GPU processes:")
            for proc in processes:
                print(f"   🔹 {proc['name']} (PID: {proc['pid']})")
                print(f"      GPU Memory: {proc['gpu_memory_mb']} MB")
                print(f"      Command: {proc['command']}")
        else:
            print("No GPU processes currently running")
        
        # Scan for NVIDIA models
        nvidia_models = self.scan_nvidia_models()
        
        # Monitor usage briefly
        if self.nvidia_smi_available:
            self.monitor_gpu_usage(3)
        
        # Generate report
        self.generate_report()
        
        # Save report
        self.save_report()
        
        return {
            'nvidia_smi_available': self.nvidia_smi_available,
            'cuda_available': self.cuda_available,
            'gpu_count': len(self.gpu_info.get('gpus', [])),
            'running_processes': len(self.running_processes),
            'nvidia_models': len(nvidia_models)
        }

def main():
    detector = NVIDIADetector()
    summary = detector.run_full_detection()
    
    print(f"\n🎉 Detection Complete!")
    print(f"   NVIDIA SMI: {'Available' if summary['nvidia_smi_available'] else 'Not Found'}")
    print(f"   CUDA Support: {'Available' if summary['cuda_available'] else 'Not Available'}")
    print(f"   GPU Count: {summary['gpu_count']}")
    print(f"   Running Processes: {summary['running_processes']}")
    print(f"   NVIDIA Models: {summary['nvidia_models']}")

if __name__ == "__main__":
    main() 