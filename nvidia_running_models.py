#!/usr/bin/env python3
"""
NVIDIA Running Models Detector
Identifies AI models currently running on NVIDIA GPUs
"""

import subprocess
import psutil
import json
import time
import re
from pathlib import Path
from collections import defaultdict

class NVIDIARunningModels:
    def __init__(self):
        self.gpu_processes = []
        self.model_signatures = self.load_model_signatures()
        
    def load_model_signatures(self):
        """Load signatures to identify different AI frameworks and models"""
        return {
            'pytorch': {
                'processes': ['python', 'python3', 'pytorch', 'torchrun'],
                'memory_patterns': ['torch', 'cuda', 'gpu'],
                'typical_memory_ranges': {
                    'small_model': (100, 1000),      # 100MB - 1GB
                    'medium_model': (1000, 4000),    # 1GB - 4GB  
                    'large_model': (4000, 12000),    # 4GB - 12GB
                    'xlarge_model': (12000, 48000)   # 12GB - 48GB
                }
            },
            'tensorflow': {
                'processes': ['python', 'python3', 'tensorflow', 'tf'],
                'memory_patterns': ['tensorflow', 'tf', 'gpu'],
                'typical_memory_ranges': {
                    'small_model': (200, 1500),
                    'medium_model': (1500, 6000),
                    'large_model': (6000, 16000),
                    'xlarge_model': (16000, 48000)
                }
            },
            'huggingface': {
                'processes': ['python', 'python3', 'transformers'],
                'memory_patterns': ['transformers', 'huggingface', 'hf'],
                'typical_memory_ranges': {
                    'small_model': (500, 2000),      # BERT-base, etc.
                    'medium_model': (2000, 8000),    # BERT-large, T5-base
                    'large_model': (8000, 24000),    # GPT-3.5, T5-large
                    'xlarge_model': (24000, 80000)   # GPT-4, Llama-70B
                }
            }
        }
    
    def get_gpu_processes(self):
        """Get detailed information about processes using GPU"""
        try:
            # Get GPU process info
            result = subprocess.run([
                'nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return []
            
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        pid = int(parts[0])
                        
                        try:
                            proc = psutil.Process(pid)
                            
                            # Get full command line
                            cmdline = proc.cmdline()
                            full_command = ' '.join(cmdline) if cmdline else 'Unknown'
                            
                            # Get working directory
                            try:
                                cwd = proc.cwd()
                            except (psutil.AccessDenied, psutil.NoSuchProcess):
                                cwd = 'Unknown'
                            
                            # Get environment variables (if accessible)
                            try:
                                env = proc.environ()
                                python_path = env.get('PYTHONPATH', '')
                                cuda_visible = env.get('CUDA_VISIBLE_DEVICES', '')
                            except (psutil.AccessDenied, psutil.NoSuchProcess):
                                python_path = ''
                                cuda_visible = ''
                            
                            process_info = {
                                'pid': pid,
                                'name': parts[1],
                                'gpu_uuid': parts[2],
                                'gpu_memory_mb': int(parts[3]),
                                'full_command': full_command,
                                'working_directory': cwd,
                                'python_path': python_path,
                                'cuda_devices': cuda_visible,
                                'cpu_percent': proc.cpu_percent(interval=0.1),
                                'memory_mb': proc.memory_info().rss / 1024 / 1024,
                                'create_time': proc.create_time(),
                                'status': proc.status()
                            }
                            
                            processes.append(process_info)
                            
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            # Process ended or no permission
                            process_info = {
                                'pid': pid,
                                'name': parts[1],
                                'gpu_uuid': parts[2],
                                'gpu_memory_mb': int(parts[3]),
                                'full_command': 'Access Denied',
                                'working_directory': 'Unknown',
                                'status': 'Unknown'
                            }
                            processes.append(process_info)
            
            self.gpu_processes = processes
            return processes
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return []
    
    def analyze_process_for_model(self, process):
        """Analyze a process to identify what AI model it might be running"""
        analysis = {
            'framework': 'Unknown',
            'model_type': 'Unknown',
            'model_size_category': 'Unknown',
            'confidence': 0,
            'indicators': []
        }
        
        command = process.get('full_command', '').lower()
        name = process.get('name', '').lower()
        gpu_memory = process.get('gpu_memory_mb', 0)
        working_dir = process.get('working_directory', '').lower()
        
        # Check for framework indicators
        framework_scores = {}
        
        for framework, signatures in self.model_signatures.items():
            score = 0
            indicators = []
            
            # Check process name
            if any(proc_name in name for proc_name in signatures['processes']):
                score += 2
                indicators.append(f"Process name matches {framework}")
            
            # Check command line
            if any(pattern in command for pattern in signatures['memory_patterns']):
                score += 3
                indicators.append(f"Command contains {framework} patterns")
            
            # Check working directory
            if framework in working_dir:
                score += 1
                indicators.append(f"Working directory suggests {framework}")
            
            # Memory-based model size estimation
            memory_ranges = signatures['typical_memory_ranges']
            for size_category, (min_mem, max_mem) in memory_ranges.items():
                if min_mem <= gpu_memory <= max_mem:
                    score += 1
                    indicators.append(f"Memory usage ({gpu_memory}MB) typical for {size_category}")
                    analysis['model_size_category'] = size_category
                    break
            
            if score > 0:
                framework_scores[framework] = {
                    'score': score,
                    'indicators': indicators
                }
        
        # Determine most likely framework
        if framework_scores:
            best_framework = max(framework_scores.keys(), key=lambda x: framework_scores[x]['score'])
            analysis['framework'] = best_framework
            analysis['confidence'] = framework_scores[best_framework]['score']
            analysis['indicators'] = framework_scores[best_framework]['indicators']
        
        # Additional model type detection
        if 'train' in command:
            analysis['model_type'] = 'Training'
        elif 'eval' in command or 'test' in command:
            analysis['model_type'] = 'Evaluation'
        elif 'infer' in command or 'predict' in command:
            analysis['model_type'] = 'Inference'
        elif 'serve' in command or 'server' in command:
            analysis['model_type'] = 'Serving'
        
        # Specific model detection
        model_keywords = {
            'bert': 'BERT',
            'gpt': 'GPT',
            'llama': 'LLaMA',
            'resnet': 'ResNet',
            'transformer': 'Transformer',
            'diffusion': 'Diffusion',
            'stable': 'Stable Diffusion',
            'yolo': 'YOLO',
            'clip': 'CLIP'
        }
        
        for keyword, model_name in model_keywords.items():
            if keyword in command:
                analysis['model_type'] = model_name
                analysis['confidence'] += 1
                analysis['indicators'].append(f"Command contains '{keyword}' keyword")
                break
        
        return analysis
    
    def get_gpu_utilization(self):
        """Get current GPU utilization"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            gpu_info = {
                                'index': int(parts[0]),
                                'name': parts[1],
                                'utilization_percent': int(parts[2]),
                                'memory_used_mb': int(parts[3]),
                                'memory_total_mb': int(parts[4]),
                                'temperature_c': int(parts[5])
                            }
                            gpus.append(gpu_info)
                return gpus
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
        
        return []
    
    def generate_report(self):
        """Generate comprehensive report of running models"""
        print("🔍 NVIDIA Running Models Analysis")
        print("=" * 40)
        
        # Get GPU utilization
        gpu_info = self.get_gpu_utilization()
        
        if gpu_info:
            print("\n💻 GPU Status:")
            for gpu in gpu_info:
                print(f"   GPU {gpu['index']}: {gpu['name']}")
                print(f"      Utilization: {gpu['utilization_percent']}%")
                print(f"      Memory: {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB")
                print(f"      Temperature: {gpu['temperature_c']}°C")
        
        # Get and analyze processes
        processes = self.get_gpu_processes()
        
        if not processes:
            print("\n❌ No GPU processes currently running")
            return
        
        print(f"\n🔄 Found {len(processes)} GPU processes:")
        print("=" * 40)
        
        for i, process in enumerate(processes, 1):
            analysis = self.analyze_process_for_model(process)
            
            print(f"\n📊 Process {i}:")
            print(f"   PID: {process['pid']}")
            print(f"   Name: {process['name']}")
            print(f"   GPU Memory: {process['gpu_memory_mb']} MB")
            print(f"   Working Directory: {process.get('working_directory', 'Unknown')}")
            
            print(f"\n   🧠 Model Analysis:")
            print(f"      Framework: {analysis['framework']}")
            print(f"      Model Type: {analysis['model_type']}")
            print(f"      Size Category: {analysis['model_size_category']}")
            print(f"      Confidence: {analysis['confidence']}/10")
            
            if analysis['indicators']:
                print(f"      Indicators:")
                for indicator in analysis['indicators']:
                    print(f"         • {indicator}")
            
            # Show command (truncated)
            command = process.get('full_command', 'Unknown')
            if len(command) > 100:
                command = command[:100] + "..."
            print(f"   Command: {command}")
        
        # Summary
        frameworks = [self.analyze_process_for_model(p)['framework'] for p in processes]
        framework_counts = {}
        for fw in frameworks:
            framework_counts[fw] = framework_counts.get(fw, 0) + 1
        
        total_gpu_memory = sum(p['gpu_memory_mb'] for p in processes)
        
        print(f"\n📋 Summary:")
        print(f"   Total GPU Memory Used: {total_gpu_memory} MB")
        print(f"   Frameworks Detected:")
        for framework, count in framework_counts.items():
            print(f"      {framework}: {count} process(es)")
    
    def save_report(self):
        """Save detailed report to JSON file"""
        processes = self.get_gpu_processes()
        gpu_info = self.get_gpu_utilization()
        
        analyzed_processes = []
        for process in processes:
            analysis = self.analyze_process_for_model(process)
            analyzed_processes.append({
                'process_info': process,
                'model_analysis': analysis
            })
        
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_info': gpu_info,
            'analyzed_processes': analyzed_processes,
            'summary': {
                'total_processes': len(processes),
                'total_gpu_memory_mb': sum(p['gpu_memory_mb'] for p in processes),
                'frameworks_detected': list(set(self.analyze_process_for_model(p)['framework'] for p in processes))
            }
        }
        
        with open('nvidia_running_models_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: nvidia_running_models_report.json")
    
    def monitor_continuously(self, duration=30, interval=5):
        """Monitor GPU processes continuously"""
        print(f"🔄 Monitoring GPU processes for {duration} seconds...")
        print(f"   Update interval: {interval} seconds")
        print("   Press Ctrl+C to stop early")
        
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                print(f"\n⏱️  Update at {time.strftime('%H:%M:%S')}")
                print("-" * 30)
                
                processes = self.get_gpu_processes()
                if processes:
                    for process in processes:
                        analysis = self.analyze_process_for_model(process)
                        print(f"PID {process['pid']}: {analysis['framework']} - {process['gpu_memory_mb']} MB")
                else:
                    print("No GPU processes running")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")

def main():
    detector = NVIDIARunningModels()
    
    # Generate main report
    detector.generate_report()
    
    # Save detailed report
    detector.save_report()
    
    # Ask if user wants continuous monitoring
    print(f"\n🔄 Would you like to monitor continuously? (y/n)")
    try:
        response = input().lower().strip()
        if response in ['y', 'yes']:
            detector.monitor_continuously(duration=60, interval=3)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main() 