#!/usr/bin/env python3
"""
Universal GPU Detection Script
Detects and analyzes GPU hardware across Mac Silicon (MPS), NVIDIA (CUDA), and AMD (ROCm)
"""

import subprocess
import platform
import sys
import json
import tempfile
import time
from pathlib import Path
import importlib.util

class UniversalGPUDetector:
    def __init__(self):
        self.system_info = {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'python_version': platform.python_version()
        }
        self.gpu_info = {
            'apple_silicon': {},
            'nvidia': {},
            'amd': {},
            'intel': {}
        }
        
    def detect_apple_silicon(self):
        """Detect Apple Silicon GPU (MPS)"""
        print("🍎 Apple Silicon GPU Detection:")
        print("=" * 35)
        
        if platform.machine() != 'arm64':
            print("❌ Not an Apple Silicon system")
            return False
        
        try:
            # Check system_profiler for GPU info
            result = subprocess.run([
                'system_profiler', 'SPDisplaysDataType', '-json'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                displays = data.get('SPDisplaysDataType', [])
                for display in displays:
                    chipset = display.get('sppci_model', 'Unknown')
                    vram = display.get('spdisplays_vram', 'Unknown')
                    
                    if 'Apple' in chipset or 'M1' in chipset or 'M2' in chipset or 'M3' in chipset:
                        print(f"✅ Apple GPU: {chipset}")
                        if vram != 'Unknown':
                            print(f"   VRAM: {vram}")
                        
                        self.gpu_info['apple_silicon'] = {
                            'chipset': chipset,
                            'vram': vram,
                            'detected': True
                        }
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        
        # Check PyTorch MPS
        try:
            import torch
            if torch.backends.mps.is_available():
                print("✅ PyTorch MPS: Available")
                print("   Metal Performance Shaders: Ready")
                
                # Test MPS device
                device = torch.device('mps')
                test_tensor = torch.randn(10, 10, device=device)
                print("   MPS Device Test: Passed")
                
                self.gpu_info['apple_silicon']['pytorch_mps'] = True
                return True
            else:
                print("❌ PyTorch MPS: Not available")
        except ImportError:
            print("❌ PyTorch: Not installed")
        except Exception as e:
            print(f"❌ PyTorch MPS: Error - {e}")
        
        return False
    
    def detect_nvidia(self):
        """Detect NVIDIA GPUs"""
        print("\n⚡ NVIDIA GPU Detection:")
        print("=" * 25)
        
        # Check nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ NVIDIA SMI: Available")
                
                # Get GPU info
                gpu_result = subprocess.run([
                    'nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if gpu_result.returncode == 0:
                    gpus = []
                    for line in gpu_result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 3:
                                gpu_info = {
                                    'name': parts[0],
                                    'memory_mb': int(parts[1]),
                                    'driver_version': parts[2]
                                }
                                gpus.append(gpu_info)
                                print(f"   GPU: {gpu_info['name']} ({gpu_info['memory_mb']} MB)")
                    
                    self.gpu_info['nvidia'] = {
                        'gpus': gpus,
                        'detected': True
                    }
                    
                    # Check CUDA frameworks
                    self.check_cuda_frameworks()
                    return True
            else:
                print("❌ NVIDIA SMI: Not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ NVIDIA SMI: Not found")
        
        return False
    
    def check_cuda_frameworks(self):
        """Check CUDA support in frameworks"""
        print("   🔥 CUDA Framework Support:")
        
        # PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                print(f"      ✅ PyTorch CUDA: {device_count} device(s)")
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    print(f"         GPU {i}: {props.name}")
            else:
                print("      ❌ PyTorch CUDA: Not available")
        except ImportError:
            print("      ❌ PyTorch: Not installed")
        
        # TensorFlow GPU
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                print(f"      ✅ TensorFlow GPU: {len(gpus)} device(s)")
            else:
                print("      ❌ TensorFlow GPU: No devices")
        except ImportError:
            print("      ❌ TensorFlow: Not installed")
    
    def detect_amd(self):
        """Detect AMD GPUs (ROCm)"""
        print("\n🔴 AMD GPU Detection:")
        print("=" * 20)
        
        # Check rocm-smi
        try:
            result = subprocess.run(['rocm-smi', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ ROCm SMI: Available")
                
                # Get GPU info
                gpu_result = subprocess.run(['rocm-smi', '--showproductname'], 
                                          capture_output=True, text=True, timeout=5)
                
                if gpu_result.returncode == 0:
                    print("   AMD GPUs detected:")
                    for line in gpu_result.stdout.strip().split('\n'):
                        if 'GPU' in line:
                            print(f"      {line.strip()}")
                    
                    self.gpu_info['amd']['detected'] = True
                    return True
            else:
                print("❌ ROCm SMI: Not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ ROCm SMI: Not found")
        
        # Check PyTorch ROCm
        try:
            import torch
            if torch.cuda.is_available() and 'rocm' in torch.version.hip:
                print("✅ PyTorch ROCm: Available")
                return True
        except (ImportError, AttributeError):
            pass
        
        print("❌ No AMD GPU support detected")
        return False
    
    def detect_intel(self):
        """Detect Intel GPUs"""
        print("\n🔵 Intel GPU Detection:")
        print("=" * 22)
        
        # Check for Intel GPU on different platforms
        if platform.system() == 'Linux':
            try:
                result = subprocess.run(['lspci', '-nn'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    intel_gpus = []
                    for line in result.stdout.split('\n'):
                        if 'VGA' in line and 'Intel' in line:
                            intel_gpus.append(line.strip())
                    
                    if intel_gpus:
                        print(f"✅ Intel GPUs detected: {len(intel_gpus)}")
                        for gpu in intel_gpus:
                            print(f"   {gpu}")
                        self.gpu_info['intel']['detected'] = True
                        return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        elif platform.system() == 'Darwin':  # macOS
            try:
                result = subprocess.run([
                    'system_profiler', 'SPDisplaysDataType'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and 'Intel' in result.stdout:
                    print("✅ Intel GPU detected in system")
                    self.gpu_info['intel']['detected'] = True
                    return True
            except subprocess.TimeoutExpired:
                pass
        
        print("❌ No Intel GPU detected")
        return False
    
    def get_gpu_memory_usage(self):
        """Get GPU memory usage across different platforms"""
        print("\n📊 GPU Memory Usage:")
        print("=" * 20)
        
        # NVIDIA
        if self.gpu_info['nvidia'].get('detected'):
            try:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=name,memory.used,memory.total',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 3:
                                name = parts[0]
                                used = int(parts[1])
                                total = int(parts[2])
                                usage_percent = (used / total) * 100
                                print(f"   ⚡ {name}: {used}/{total} MB ({usage_percent:.1f}%)")
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass
        
        # Apple Silicon (approximation)
        if self.gpu_info['apple_silicon'].get('detected'):
            try:
                # Get system memory as approximation (unified memory)
                result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print("   🍎 Apple Silicon: Unified Memory Architecture")
                    print("      (GPU shares system memory)")
            except subprocess.TimeoutExpired:
                pass
    
    def detect_running_models(self):
        """Detect running AI models across platforms"""
        print("\n🧠 Running AI Models:")
        print("=" * 20)
        
        # Look for common AI processes
        ai_processes = []
        
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    
                    # Check for AI/ML keywords
                    ai_keywords = ['torch', 'tensorflow', 'keras', 'transformers', 'train', 'model', 'gpu', 'cuda', 'mps']
                    
                    if any(keyword in cmdline.lower() for keyword in ai_keywords):
                        ai_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'command': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            print("   ❌ psutil not available for process detection")
            return
        
        if ai_processes:
            print(f"   Found {len(ai_processes)} potential AI processes:")
            for proc in ai_processes[:5]:  # Show first 5
                print(f"      PID {proc['pid']}: {proc['name']}")
                print(f"         {proc['command']}")
        else:
            print("   No AI processes currently detected")
    
    def generate_summary(self):
        """Generate comprehensive GPU summary"""
        print("\n🎯 GPU Detection Summary:")
        print("=" * 30)
        
        detected_gpus = []
        
        if self.gpu_info['apple_silicon'].get('detected'):
            detected_gpus.append("Apple Silicon (MPS)")
        
        if self.gpu_info['nvidia'].get('detected'):
            nvidia_count = len(self.gpu_info['nvidia'].get('gpus', []))
            detected_gpus.append(f"NVIDIA ({nvidia_count} GPU(s))")
        
        if self.gpu_info['amd'].get('detected'):
            detected_gpus.append("AMD (ROCm)")
        
        if self.gpu_info['intel'].get('detected'):
            detected_gpus.append("Intel")
        
        if detected_gpus:
            print(f"✅ Detected GPU Types: {', '.join(detected_gpus)}")
        else:
            print("❌ No GPU acceleration detected")
        
        # Framework support
        frameworks = []
        try:
            import torch
            if torch.cuda.is_available():
                frameworks.append("PyTorch CUDA")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                frameworks.append("PyTorch MPS")
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            if tf.config.experimental.list_physical_devices('GPU'):
                frameworks.append("TensorFlow GPU")
        except ImportError:
            pass
        
        if frameworks:
            print(f"🔧 AI Framework Support: {', '.join(frameworks)}")
        else:
            print("❌ No GPU-accelerated AI frameworks detected")
        
        return {
            'detected_gpus': detected_gpus,
            'supported_frameworks': frameworks,
            'gpu_info': self.gpu_info
        }
    
    def save_report(self):
        """Save detailed report to file"""
        summary = self.generate_summary()
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': self.system_info,
            'gpu_detection': self.gpu_info,
            'summary': summary
        }
        # Cache last report for callers
        try:
            self.last_report = report_data
        except Exception:
            pass
        # Prefer writing in CWD; if not writable, fall back to temp dir
        try:
            with open('universal_gpu_report.json', 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"\n💾 Detailed report saved to: universal_gpu_report.json")
        except Exception:
            try:
                fallback = Path(tempfile.gettempdir()) / 'universal_gpu_report.json'
                with open(fallback, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                print(f"\n💾 Detailed report saved to: {fallback}")
            except Exception:
                # Silent fallback; report already available via self.last_report
                pass
    
    def run_full_detection(self):
        """Run complete GPU detection across all platforms"""
        print("🌍 Universal GPU Detection")
        print("=" * 30)
        print(f"System: {self.system_info['platform']}")
        print(f"Architecture: {self.system_info['machine']}")
        print(f"Python: {self.system_info['python_version']}")
        
        # Detect all GPU types
        apple_detected = self.detect_apple_silicon()
        nvidia_detected = self.detect_nvidia()
        amd_detected = self.detect_amd()
        intel_detected = self.detect_intel()
        
        # Get memory usage
        self.get_gpu_memory_usage()
        
        # Detect running models
        self.detect_running_models()
        
        # Generate summary
        summary = self.generate_summary()
        
        # Save report (non-fatal on failure)
        try:
            self.save_report()
        except Exception:
            pass
        
        return summary

def main():
    detector = UniversalGPUDetector()
    summary = detector.run_full_detection()
    
    print(f"\n🎉 Detection Complete!")
    if summary['detected_gpus']:
        print(f"   GPU Types: {', '.join(summary['detected_gpus'])}")
        print(f"   Frameworks: {', '.join(summary['supported_frameworks'])}")
    else:
        print("   No GPU acceleration available")

if __name__ == "__main__":
    main() 