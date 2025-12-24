#!/usr/bin/env python3
"""
Real-Time Gradient Monitor for NeurInSpectre
Autodetects and captures gradients from running models with GPU acceleration
Supports both Mac Silicon MPS and NVIDIA CUDA - no model specification required
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import time
import queue
import gc
import psutil
from typing import Dict, List, Tuple, Optional, Callable, Any
import warnings

# Avoid process-global warning suppression at import time. Silence warnings in your app entry point if desired.

from .obfuscated_gradient_visualizer import ObfuscatedGradientVisualizer

class RealTimeGradientMonitor:
    """
    Real-time gradient monitoring and analysis system
    Autodetects running models and captures gradients from any PyTorch model during backpropagation
    """
    
    def __init__(self, 
                 device: str = 'auto',
                 buffer_size: int = 1000,
                 update_interval: float = 0.1,
                 analysis_window: int = 100,
                 auto_detect_models: bool = True):
        """
        Initialize real-time gradient monitor with model autodetection
        
        Args:
            device: Device to use ('auto', 'mps', 'cuda', 'cpu')
            buffer_size: Maximum number of gradient samples to keep
            update_interval: Update frequency in seconds
            analysis_window: Number of samples for rolling analysis
            auto_detect_models: Whether to automatically detect and hook running models
        """
        self.device = self._detect_device(device)
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.analysis_window = analysis_window
        self.auto_detect_models = auto_detect_models
        
        # Initialize buffers
        self.gradient_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)
        self.analysis_buffer = deque(maxlen=buffer_size)
        
        # Model tracking
        self.detected_models = {}
        self.model_hooks = {}
        self.global_hooks = []
        
        # Analysis components
        self.visualizer = ObfuscatedGradientVisualizer()
        self.is_monitoring = False
        self.data_queue = queue.Queue()
        
        # Real-time plotting
        self.fig = None
        self.axes = None
        self.lines = {}
        self.animation = None
        
        # Performance metrics
        self.metrics = {
            'total_gradients': 0,
            'models_detected': 0,
            'suspicious_patterns': 0,
            'processing_time': 0,
            'gpu_utilization': 0
        }
        
        print(f"🚀 Real-time gradient monitor initialized (AutoDetect Mode)")
        print(f"📱 Device: {self.device}")
        print(f"📊 Buffer size: {buffer_size}")
        print(f"🔍 Model autodetection: {'Enabled' if auto_detect_models else 'Disabled'}")
        
    def _detect_device(self, device: str) -> torch.device:
        """Detect optimal device for gradient processing"""
        if device == 'auto':
            if torch.backends.mps.is_available():
                detected_device = torch.device('mps')
                print("🍎 Mac Silicon MPS detected and enabled")
            elif torch.cuda.is_available():
                detected_device = torch.device('cuda')
                print(f"🟢 NVIDIA CUDA detected: {torch.cuda.get_device_name(0)}")
            else:
                detected_device = torch.device('cpu')
                print("💻 Using CPU (GPU not available)")
        else:
            detected_device = torch.device(device)
            
        return detected_device
    
    def autodetect_running_models(self) -> List[nn.Module]:
        """
        Autodetect PyTorch models currently in memory
        """
        detected_models = []
        
        # Scan all objects in memory for PyTorch models
        for obj in gc.get_objects():
            if isinstance(obj, nn.Module):
                # Check if model has parameters and is potentially active
                param_count = sum(p.numel() for p in obj.parameters())
                if param_count > 0:
                    model_id = id(obj)
                    model_info = {
                        'model': obj,
                        'id': model_id,
                        'parameters': param_count,
                        'device': next(obj.parameters()).device if param_count > 0 else 'unknown',
                        'class_name': obj.__class__.__name__
                    }
                    
                    if model_id not in self.detected_models:
                        self.detected_models[model_id] = model_info
                        detected_models.append(obj)
                        print(f"🧠 Detected model: {model_info['class_name']} ({param_count:,} params) on {model_info['device']}")
        
        self.metrics['models_detected'] = len(self.detected_models)
        return detected_models
    
    def register_global_gradient_hooks(self):
        """Register global hooks to capture gradients from any model"""
        
        def global_backward_hook(grad_input, grad_output):
            """Global hook that captures gradients during any backward pass"""
            if grad_output is not None:
                for i, grad in enumerate(grad_output):
                    if grad is not None and grad.numel() > 0:
                        # Move gradient to our device and capture
                        grad_data = grad.detach().clone()
                        timestamp = time.time()
                        
                        # Add to queue for processing
                        self.data_queue.put({
                            'gradient': grad_data,
                            'timestamp': timestamp,
                            'layer_name': f'global_hook_{i}',
                            'shape': grad_data.shape,
                            'device': str(grad_data.device),
                            'source': 'global_hook'
                        })
        
        def tensor_backward_hook(grad):
            """Hook that attaches to tensors with requires_grad=True"""
            if grad is not None and grad.numel() > 0:
                grad_data = grad.detach().clone()
                timestamp = time.time()
                
                self.data_queue.put({
                    'gradient': grad_data,
                    'timestamp': timestamp,
                    'layer_name': 'tensor_grad',
                    'shape': grad_data.shape,
                    'device': str(grad_data.device),
                    'source': 'tensor_hook'
                })
            return grad
        
        # Register global hooks using PyTorch's hook system
        # This is a more aggressive approach - hook into ALL backward operations
        
        # Monkey patch tensor backward hooks
        original_backward = torch.Tensor.backward
        
        def hooked_backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
            # Call original backward
            result = original_backward(self, gradient, retain_graph, create_graph, inputs)
            
            # Capture gradient if available
            if self.grad is not None:
                grad_data = self.grad.detach().clone()
                timestamp = time.time()
                
                self.data_queue.put({
                    'gradient': grad_data,
                    'timestamp': timestamp,
                    'layer_name': 'backward_capture',
                    'shape': grad_data.shape,
                    'device': str(grad_data.device),
                    'source': 'backward_hook'
                })
            
            return result
        
        # Apply the hook
        torch.Tensor.backward = hooked_backward
        self.global_hooks.append(('tensor_backward', original_backward))
        
        print("📌 Global gradient capture hooks registered")
    
    def register_model_specific_hooks(self, models: List[nn.Module]):
        """Register hooks for specific detected models"""
        for model in models:
            model_id = id(model)
            if model_id not in self.model_hooks:
                hooks = []
                
                def create_gradient_hook(model_name, param_name):
                    def hook(grad):
                        if grad is not None:
                            grad_data = grad.detach().clone()
                            timestamp = time.time()
                            
                            self.data_queue.put({
                                'gradient': grad_data,
                                'timestamp': timestamp,
                                'layer_name': f'{model_name}.{param_name}',
                                'shape': grad_data.shape,
                                'device': str(grad_data.device),
                                'source': 'model_hook',
                                'model_id': model_id
                            })
                        return grad
                    return hook
                
                # Register hooks for all parameters
                model_name = model.__class__.__name__
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        handle = param.register_hook(create_gradient_hook(model_name, name))
                        hooks.append(handle)
                
                self.model_hooks[model_id] = hooks
                print(f"📌 Registered {len(hooks)} hooks for {model_name}")
    
    def start_monitoring(self, live_plot: bool = True):
        """Start real-time gradient monitoring with autodetection"""
        self.is_monitoring = True
        
        # Autodetect running models
        if self.auto_detect_models:
            print("🔍 Scanning for running PyTorch models...")
            detected_models = self.autodetect_running_models()
            
            if detected_models:
                print(f"✅ Found {len(detected_models)} models, registering hooks...")
                self.register_model_specific_hooks(detected_models)
            else:
                print("⚠️  No models detected, using global gradient capture")
        
        # Always register global hooks for comprehensive coverage
        self.register_global_gradient_hooks()
        
        # Start data processing thread
        self.processing_thread = threading.Thread(target=self._process_gradients, daemon=True)
        self.processing_thread.start()
        
        # Start model detection thread (continues to look for new models)
        if self.auto_detect_models:
            self.detection_thread = threading.Thread(target=self._continuous_model_detection, daemon=True)
            self.detection_thread.start()
        
        if live_plot:
            self._setup_live_plot()
            
        print("🎯 Real-time gradient monitoring started")
        print("📊 Monitoring ALL PyTorch models and gradients")
        print("📊 Use Ctrl+C to stop monitoring")
        
    def _continuous_model_detection(self):
        """Continuously scan for new models being created"""
        while self.is_monitoring:
            try:
                # Scan for new models every 5 seconds
                time.sleep(5)
                new_models = []
                
                for obj in gc.get_objects():
                    if isinstance(obj, nn.Module):
                        model_id = id(obj)
                        if model_id not in self.detected_models:
                            param_count = sum(p.numel() for p in obj.parameters())
                            if param_count > 0:
                                new_models.append(obj)
                
                if new_models:
                    print(f"🆕 Detected {len(new_models)} new models")
                    self.autodetect_running_models()  # Update our registry
                    self.register_model_specific_hooks(new_models)
                    
            except Exception as e:
                print(f"⚠️  Model detection error: {e}")
                
    def stop_monitoring(self):
        """Stop gradient monitoring and clean up hooks"""
        self.is_monitoring = False
        
        # Remove model-specific hooks
        for model_id, hooks in self.model_hooks.items():
            for handle in hooks:
                try:
                    handle.remove()
                except:
                    pass
        
        # Restore global hooks
        for hook_name, original_func in self.global_hooks:
            if hook_name == 'tensor_backward':
                torch.Tensor.backward = original_func
        
    def _process_gradients(self):
        """Background thread for processing captured gradients"""
        while self.is_monitoring:
            try:
                # Get gradient data with timeout
                data = self.data_queue.get(timeout=0.1)
                
                # Process gradient
                processed_data = self._analyze_gradient(data)
                
                # Add to buffers
                self.gradient_buffer.append(processed_data)
                self.timestamp_buffer.append(data['timestamp'])
                
                # Update metrics
                self.metrics['total_gradients'] += 1
                
                # Check for suspicious patterns
                if self._detect_suspicious_pattern(processed_data):
                    self.metrics['suspicious_patterns'] += 1
                    print(f"⚠️  Suspicious gradient pattern detected at {time.strftime('%H:%M:%S')}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error processing gradient: {e}")
                
    def _analyze_gradient(self, data: Dict) -> Dict:
        """Analyze captured gradient data"""
        start_time = time.time()
        
        gradient = data['gradient']
        
        # Flatten gradient for analysis
        grad_flat = gradient.flatten()
        
        # Compute statistics
        analysis = {
            'timestamp': data['timestamp'],
            'layer_name': data['layer_name'],
            'shape': data['shape'],
            'device': data['device'],
            'mean': float(torch.mean(grad_flat)),
            'std': float(torch.std(grad_flat)),
            'min': float(torch.min(grad_flat)),
            'max': float(torch.max(grad_flat)),
            'norm': float(torch.norm(grad_flat)),
            'sparsity': float(torch.sum(torch.abs(grad_flat) < 1e-6) / len(grad_flat)),
        }
        
        # GPU-accelerated spectral analysis
        if len(grad_flat) >= 64:  # Only for sufficiently large gradients
            try:
                # FFT analysis on GPU
                fft_result = torch.fft.fft(grad_flat.to(dtype=torch.complex64))
                power_spectrum = torch.abs(fft_result) ** 2
                
                analysis.update({
                    'spectral_entropy': float(self._compute_spectral_entropy(power_spectrum)),
                    'high_freq_energy': float(torch.mean(power_spectrum[len(power_spectrum)//2:])),
                    'dominant_frequency': int(torch.argmax(power_spectrum[:len(power_spectrum)//2])),
                })
            except Exception as e:
                print(f"Warning: Spectral analysis failed: {e}")
        
        # Update processing time metric
        self.metrics['processing_time'] = time.time() - start_time
        
        return analysis
    
    def _compute_spectral_entropy(self, power_spectrum: torch.Tensor) -> torch.Tensor:
        """Compute spectral entropy of power spectrum"""
        # Normalize power spectrum
        normalized = power_spectrum / torch.sum(power_spectrum)
        
        # Add small epsilon to avoid log(0)
        normalized = normalized + 1e-12
        
        # Compute entropy
        entropy = -torch.sum(normalized * torch.log(normalized))
        return entropy
    
    def _detect_suspicious_pattern(self, analysis: Dict) -> bool:
        """Detect suspicious gradient patterns that might indicate attacks"""
        # Check for various anomaly indicators
        suspicious_indicators = 0
        
        # 1. Unusual magnitude
        if abs(analysis['mean']) > 1.0 or analysis['std'] > 5.0:
            suspicious_indicators += 1
            
        # 2. Extreme sparsity (potential gradient masking)
        if analysis['sparsity'] > 0.95 or analysis['sparsity'] < 0.01:
            suspicious_indicators += 1
            
        # 3. High spectral entropy (potential obfuscation)
        if 'spectral_entropy' in analysis and analysis['spectral_entropy'] > 8.0:
            suspicious_indicators += 1
            
        # 4. Unusual frequency content
        if 'high_freq_energy' in analysis and analysis['high_freq_energy'] > analysis.get('spectral_entropy', 0) * 2:
            suspicious_indicators += 1
            
        # Threshold for suspicious pattern
        return suspicious_indicators >= 2
    
    def _setup_live_plot(self):
        """Setup live plotting for real-time visualization"""
        # Avoid process-global matplotlib style mutation. Apply style only for this figure creation.
        with plt.style.context('dark_background'):
            self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
            self.fig.suptitle('NeurInSpectre Real-Time Gradient Monitor', fontsize=16, color='white')
        
        # Initialize empty plots
        self._init_plots()
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self._update_plots, interval=int(self.update_interval * 1000),
            blit=False, cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show(block=False)
        
    def _init_plots(self):
        """Initialize all subplot components"""
        # Plot 1: Gradient magnitude over time
        ax = self.axes[0, 0]
        ax.set_title('Gradient Magnitude', color='white')
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('L2 Norm', color='white')
        self.lines['magnitude'] = ax.plot([], [], 'g-', linewidth=2)[0]
        
        # Plot 2: Mean gradient values
        ax = self.axes[0, 1]
        ax.set_title('Mean Gradient Values', color='white')
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Mean Value', color='white')
        self.lines['mean'] = ax.plot([], [], 'b-', linewidth=2)[0]
        
        # Plot 3: Gradient standard deviation
        ax = self.axes[0, 2]
        ax.set_title('Gradient Std Deviation', color='white')
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Std Dev', color='white')
        self.lines['std'] = ax.plot([], [], 'r-', linewidth=2)[0]
        
        # Plot 4: Spectral entropy
        ax = self.axes[1, 0]
        ax.set_title('Spectral Entropy', color='white')
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Entropy', color='white')
        self.lines['entropy'] = ax.plot([], [], 'orange', linewidth=2)[0]
        
        # Plot 5: Sparsity
        ax = self.axes[1, 1]
        ax.set_title('Gradient Sparsity', color='white')
        ax.set_xlabel('Time (s)', color='white')
        ax.set_ylabel('Sparsity Ratio', color='white')
        self.lines['sparsity'] = ax.plot([], [], 'purple', linewidth=2)[0]
        
        # Plot 6: Performance metrics
        ax = self.axes[1, 2]
        ax.set_title('Performance Metrics', color='white')
        ax.text(0.1, 0.8, 'Total Gradients: 0', transform=ax.transAxes, color='white')
        ax.text(0.1, 0.6, 'Suspicious: 0', transform=ax.transAxes, color='red')
        ax.text(0.1, 0.4, 'Processing Time: 0ms', transform=ax.transAxes, color='cyan')
        ax.text(0.1, 0.2, 'Device: ' + str(self.device), transform=ax.transAxes, color='green')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
    def _update_plots(self, frame):
        """Update live plots with new data"""
        if len(self.gradient_buffer) < 2:
            return
            
        # Get recent data
        recent_data = list(self.gradient_buffer)[-self.analysis_window:]
        recent_times = list(self.timestamp_buffer)[-self.analysis_window:]
        
        if not recent_data:
            return
            
        # Convert timestamps to relative times
        base_time = recent_times[0] if recent_times else time.time()
        times = [(t - base_time) for t in recent_times]
        
        # Extract data for plotting
        magnitudes = [d['norm'] for d in recent_data]
        means = [d['mean'] for d in recent_data]
        stds = [d['std'] for d in recent_data]
        entropies = [d.get('spectral_entropy', 0) for d in recent_data]
        sparsities = [d['sparsity'] for d in recent_data]
        
        # Update plots
        self.lines['magnitude'].set_data(times, magnitudes)
        self.lines['mean'].set_data(times, means)
        self.lines['std'].set_data(times, stds)
        self.lines['entropy'].set_data(times, entropies)
        self.lines['sparsity'].set_data(times, sparsities)
        
        # Auto-scale axes
        for i, ax in enumerate(self.axes.flat[:-1]):  # Skip metrics plot
            ax.relim()
            ax.autoscale_view()
            
        # Update metrics display
        ax = self.axes[1, 2]
        ax.clear()
        ax.set_title('Performance Metrics', color='white')
        ax.text(0.1, 0.8, f'Total Gradients: {self.metrics["total_gradients"]}', 
                transform=ax.transAxes, color='white', fontsize=12)
        ax.text(0.1, 0.6, f'Suspicious: {self.metrics["suspicious_patterns"]}', 
                transform=ax.transAxes, color='red', fontsize=12)
        ax.text(0.1, 0.4, f'Processing Time: {self.metrics["processing_time"]*1000:.1f}ms', 
                transform=ax.transAxes, color='cyan', fontsize=12)
        ax.text(0.1, 0.2, f'Device: {self.device}', 
                transform=ax.transAxes, color='green', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        self.fig.canvas.draw()
    
    def get_current_analysis(self) -> Dict:
        """Get current analysis summary"""
        if not self.gradient_buffer:
            return {}
            
        recent_data = list(self.gradient_buffer)[-10:]  # Last 10 samples
        
        summary = {
            'timestamp': time.time(),
            'device': str(self.device),
            'total_samples': len(self.gradient_buffer),
            'recent_samples': len(recent_data),
            'avg_magnitude': np.mean([d['norm'] for d in recent_data]),
            'avg_sparsity': np.mean([d['sparsity'] for d in recent_data]),
            'suspicious_rate': self.metrics['suspicious_patterns'] / max(1, self.metrics['total_gradients']),
            'processing_time_ms': self.metrics['processing_time'] * 1000,
            'metrics': self.metrics.copy()
        }
        
        return summary
    
    def save_analysis_report(self, filepath: str):
        """Save comprehensive analysis report"""
        analysis = self.get_current_analysis()
        
        # Add gradient buffer data
        analysis['gradient_history'] = [
            {k: v for k, v in d.items() if k != 'gradient'} 
            for d in list(self.gradient_buffer)
        ]
        
        import json
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
            
        print(f"📄 Analysis report saved to: {filepath}")
    
    def register_model_hooks(self, model: nn.Module):
        """Register hooks on a specific model (convenience method for single model)"""
        self.register_model_specific_hooks([model])
    
    def get_results(self) -> Dict:
        """Get results in a format compatible with the CLI"""
        gradient_history = []
        
        for data in list(self.gradient_buffer):
            gradient_history.append({
                'mean': data.get('mean', 0),
                'std': data.get('std', 0),
                'max': data.get('max', 0),
                'min': data.get('min', 0),
                'norm': data.get('norm', 0),
                'sparsity': data.get('sparsity', 0),
                'timestamp': data.get('timestamp', 0)
            })
        
        return {
            'gradient_history': gradient_history,
            'metrics': self.metrics.copy(),
            'device': str(self.device),
            'total_samples': len(self.gradient_buffer)
        }

def create_demo_model(device: torch.device) -> nn.Module:
    """Create a demo model for testing"""
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 10)
    ).to(device)
    
    return model

def demo_real_time_monitoring():
    """Demonstration of real-time gradient monitoring"""
    print("🎯 NeurInSpectre Real-Time Gradient Monitor Demo")
    print("=" * 60)
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Using Mac Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🟢 Using NVIDIA CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    # Create demo model
    model = create_demo_model(device)
    print(f"🧠 Created demo model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize monitor
    monitor = RealTimeGradientMonitor(
        device=device,
        buffer_size=1000,
        update_interval=0.1
    )
    
    # Start monitoring
    monitor.start_monitoring(live_plot=True)
    
    # Simulate training with synthetic data
    print("🔄 Starting synthetic training simulation...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    try:
        for epoch in range(100):  # Run for 100 steps
            # Generate synthetic batch
            batch_size = 32
            x = torch.randn(batch_size, 784, device=device)
            y = torch.randint(0, 10, (batch_size,), device=device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Backward pass (this triggers gradient hooks)
            loss.backward()
            optimizer.step()
            
            # Add some variation to create interesting patterns
            if epoch % 20 == 0:
                # Simulate adversarial gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += torch.randn_like(param.grad) * 0.1
            
            time.sleep(0.1)  # Control simulation speed
            
            if epoch % 10 == 0:
                analysis = monitor.get_current_analysis()
                print(f"Epoch {epoch}: Processed {analysis.get('total_samples', 0)} gradients, "
                      f"Suspicious rate: {analysis.get('suspicious_rate', 0):.3f}")
                
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    finally:
        monitor.stop_monitoring()
        
        # Save final report
        monitor.save_analysis_report('realtime_gradient_analysis.json')
        print("✅ Demo completed!")

# Alias for backwards compatibility
RealtimeGradientMonitor = RealTimeGradientMonitor

if __name__ == "__main__":
    demo_real_time_monitoring() 