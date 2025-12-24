#!/usr/bin/env python3
"""
Real-Time Model Monitor for NeurInSpectre TTD
Connects to actual model training and captures live gradients/metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import threading
import queue
import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, AutoConfig
import logging

class LiveModelMonitor:
    """Monitor a live training model and capture real-time security metrics"""
    
    def __init__(self, model_name="distilbert-base-uncased", device=None):
        # Set up device (prefer MPS on Mac, then CUDA, then CPU)
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("🍎 Using Apple MPS (Metal Performance Shaders)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("🚀 Using CUDA GPU")
            else:
                self.device = torch.device("cpu")
                print("💻 Using CPU")
        else:
            self.device = torch.device(device)
        
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # Real-time monitoring data
        self.gradient_queue = queue.Queue(maxsize=1000)
        self.loss_queue = queue.Queue(maxsize=1000)
        self.attention_queue = queue.Queue(maxsize=100)
        
        # Security metrics
        self.privacy_budget = 0.1
        self.gradient_norms = []
        self.attack_events = []
        
        # Training state
        self.is_training = False
        self.current_epoch = 0
        self.current_step = 0
        
        # Monitoring thread
        self.monitor_thread = None
        self.stop_monitoring = False
        
        # Security thresholds
        self.CRITICAL_GRADIENT_NORM = 5.0
        self.PRIVACY_BUDGET_LIMIT = 10.0
        self.ATTACK_THRESHOLD = 0.8
        
        # Model switching state
        self.model_config = None
        self.is_switching_model = False
        
    def switch_model(self, new_model_name):
        """Switch to a different model during runtime - ENHANCED FOR ACCURACY"""
        if new_model_name == self.model_name:
            print(f"📋 Model {new_model_name} already loaded, no switch needed")
            return True
            
        print(f"🔄 SWITCHING MODEL: {self.model_name} → {new_model_name}")
        
        # Set switching flag to pause monitoring
        self.is_switching_model = True
        was_training = self.is_training
        
        if was_training:
            print("⏸️ Pausing training for model switch...")
            self.is_training = False
            time.sleep(0.5)  # Allow current operations to complete
        
        try:
            # Clean up old model to free memory
            if hasattr(self, 'model') and self.model is not None:
                print(f"🗑️ Cleaning up old model: {self.model_name}")
                del self.model
                del self.tokenizer
                if hasattr(self, 'classifier'):
                    del self.classifier
                    
                # Clear GPU memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device.type == 'mps':
                    torch.mps.empty_cache()
                
                print("✅ Memory cleared")
            
            # Update model name
            old_model = self.model_name
            self.model_name = new_model_name
            
            # Load new model
            print(f"🤖 Loading new model: {new_model_name}")
            success = self.initialize_model()
            
            if success:
                # Reset monitoring state for new model
                self.current_step = 0
                self.current_epoch = 0
                self.gradient_norms = []
                self.privacy_budget = 0.1
                self.attack_events = []
                
                # Clear queues
                while not self.gradient_queue.empty():
                    try:
                        self.gradient_queue.get_nowait()
                    except queue.Empty:
                        break
                while not self.loss_queue.empty():
                    try:
                        self.loss_queue.get_nowait()
                    except queue.Empty:
                        break
                while not self.attention_queue.empty():
                    try:
                        self.attention_queue.get_nowait()
                    except queue.Empty:
                        break
                
                print(f"✅ MODEL SWITCH SUCCESSFUL: {old_model} → {new_model_name}")
                print(f"📊 New model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
                print(f"🔧 Model config: {self.model_config}")
                
                # Resume training if it was active
                if was_training:
                    print("▶️ Resuming training with new model...")
                    self.is_training = True
                
                self.is_switching_model = False
                return True
                
            else:
                print(f"❌ Model switch failed, reverting to {old_model}")
                self.model_name = old_model
                self.initialize_model()
                self.is_switching_model = False
                return False
                
        except Exception as e:
            print(f"❌ Error during model switch: {e}")
            print(f"🔄 Attempting to revert to {old_model}")
            self.model_name = old_model
            try:
                self.initialize_model()
            except:
                print(f"❌ Could not revert to {old_model}")
            self.is_switching_model = False
            return False
        
    def initialize_model(self):
        """Initialize the model for monitoring - ENHANCED WITH CONFIG CAPTURE"""
        print(f"🤖 Initializing {self.model_name} for live monitoring...")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            config = AutoConfig.from_pretrained(self.model_name)
            
            # Use eager attention for BERT models to avoid deprecation warning
            if 'bert' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    attn_implementation="eager"
                ).to(self.device)
            else:
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            
            # Capture actual model configuration for TTD dashboard
            self.model_config = {
                "num_layers": getattr(config, 'num_hidden_layers', None) or 
                             getattr(config, 'num_layers', None) or 
                             getattr(config, 'n_layer', None) or 12,
                "attention_heads": getattr(config, 'num_attention_heads', None) or 
                                 getattr(config, 'num_heads', None) or 
                                 getattr(config, 'n_head', None) or 12,
                "hidden_size": getattr(config, 'hidden_size', None) or 
                              getattr(config, 'd_model', None) or 
                              getattr(config, 'n_embd', None) or 768,
                "model_type": getattr(config, 'model_type', 'unknown'),
                "vocab_size": getattr(config, 'vocab_size', 'unknown'),
                "is_encoder_decoder": getattr(config, 'is_encoder_decoder', False),
                "total_parameters": 0  # Will be set below
            }
            
            # Add a simple classification head for training
            hidden_size = self.model_config["hidden_size"]
            self.classifier = nn.Linear(int(hidden_size), 2).to(self.device)  # Binary classification
            
            # Calculate total parameters including classifier
            total_params = sum(p.numel() for p in self.model.parameters()) + sum(p.numel() for p in self.classifier.parameters())
            self.model_config["total_parameters"] = total_params
            
            print(f"✅ Model loaded on {self.device}")
            print(f"📊 Model parameters: {total_params:,}")
            print(f"🔧 Model type: {self.model_config['model_type']}")
            print(f"📝 Vocab size: {self.model_config['vocab_size']:,}")
            print(f"🏗️ Architecture: {'Encoder-Decoder' if self.model_config['is_encoder_decoder'] else 'Encoder-Only'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            self.model_config = None
            return False
    
    def create_dummy_data(self, batch_size=8, seq_length=128, num_batches=100):
        """Create dummy training data for real model training"""
        print(f"📊 Creating dummy training data: {num_batches} batches of {batch_size} samples")
        
        # Create random token sequences
        vocab_size = self.tokenizer.vocab_size
        input_ids = torch.randint(0, vocab_size, (num_batches * batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, 2, (num_batches * batch_size,))  # Binary labels
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    def start_live_training(self, num_epochs=5, learning_rate=2e-5):
        """Start actual model training with live monitoring"""
        if not self.model or not self.tokenizer:
            if not self.initialize_model():
                return False
        
        print(f"🚀 Starting live model training monitoring...")
        print(f"⚙️ Epochs: {num_epochs}, Learning Rate: {learning_rate}")
        
        # Create training data
        dataloader = self.create_dummy_data()
        
        # Setup optimizer
        optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.classifier.parameters()),
            lr=learning_rate
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Start monitoring thread
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._security_monitor_loop)
        self.monitor_thread.start()
        
        self.is_training = True
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                epoch_loss = 0.0
                
                print(f"\n🔄 Epoch {epoch+1}/{num_epochs}")
                
                for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
                    self.current_step += 1
                    
                    # Move to device
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    
                    # Get model outputs with attention
                    outputs = self.model(input_ids, attention_mask=attention_mask, output_attentions=True)
                    
                    # Classification (robust to model output structure)
                    last_hidden = getattr(outputs, 'last_hidden_state', None)
                    if last_hidden is None:
                        # Some HF models return tuple with hidden states first
                        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                            last_hidden = outputs[0]
                    if last_hidden is None:
                        raise RuntimeError('Model did not return last_hidden_state')
                    if last_hidden.dim() == 3 and last_hidden.size(1) > 1:
                        pooled = last_hidden[:, 0, :]
                    else:
                        pooled = last_hidden.mean(dim=1) if last_hidden.dim() >= 2 else last_hidden
                    logits = self.classifier(pooled)
                    loss = criterion(logits, labels)
                    
                    # Backward pass
                    loss.backward()
                    
                    # CAPTURE REAL GRADIENTS FOR SECURITY ANALYSIS
                    self._capture_gradient_metrics()
                    
                    # CAPTURE ATTENTION PATTERNS
                    attn = getattr(outputs, 'attentions', None)
                    if attn is not None and len(attn) > 0:
                        self._capture_attention_patterns(attn)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    # Real-time updates every 10 steps
                    if batch_idx % 10 == 0:
                        current_loss = loss.item()
                        print(f"  Step {self.current_step}: Loss = {current_loss:.4f}, Gradient Norm = {self._get_current_gradient_norm():.4f}")
                        
                        # Add to loss queue for real-time monitoring
                        if not self.loss_queue.full():
                            self.loss_queue.put({
                                'step': self.current_step,
                                'loss': current_loss,
                                'timestamp': datetime.now().isoformat()
                            })
                    
                    # Simulate some training delay
                    time.sleep(0.1)
                
                avg_loss = epoch_loss / len(dataloader)
                print(f"✅ Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
                
        except Exception as e:
            print(f"❌ Training error: {e}")
        finally:
            self.is_training = False
            self.stop_monitoring = True
            if self.monitor_thread:
                self.monitor_thread.join()
    
    def _capture_gradient_metrics(self):
        """Capture gradients from the live model for security analysis"""
        total_norm = 0.0
        param_count = 0
        
        gradients = []
        
        # Collect gradients from all parameters
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Store gradient data for security analysis
                grad_data = param.grad.data.detach().cpu().numpy().flatten()
                gradients.extend(grad_data[:100])  # Sample first 100 values
        
        total_norm = total_norm ** (1. / 2)
        
        # Update gradient norms list
        self.gradient_norms.append(total_norm)
        if len(self.gradient_norms) > 1000:  # Keep only last 1000 values
            self.gradient_norms.pop(0)
        
        # Update privacy budget (simplified differential privacy simulation)
        privacy_loss = min(total_norm / 10.0, 0.5)  # Scale gradient norm to privacy loss
        self.privacy_budget += privacy_loss
        
        # Add to gradient queue for real-time monitoring
        if not self.gradient_queue.full():
            gradient_data = {
                'step': self.current_step,
                'norm': total_norm,
                'privacy_budget': self.privacy_budget,
                'gradients': gradients[:50],  # Sample for analysis
                'timestamp': datetime.now().isoformat()
            }
            self.gradient_queue.put(gradient_data)
        
        # Check for security threats
        self._check_security_threats(total_norm)
    
    def _capture_attention_patterns(self, attentions):
        """Capture attention patterns for adversarial analysis.
        Be resilient to models that return different attention shapes or None values.
        """
        try:
            if not attentions or self.attention_queue.full():
                return

            # Standard HF: tuple per layer -> tensor (B, H, S, S)
            attn_layer0 = None
            if isinstance(attentions, (list, tuple)) and len(attentions) > 0:
                attn_layer0 = attentions[0]

            # Some models may return a list with None placeholders
            if attn_layer0 is None:
                return

            # Handle possible shapes
            if hasattr(attn_layer0, 'detach'):
                t = attn_layer0
            else:
                # Unexpected type
                return

            # Expected shapes: (B,H,S,S) or (B,S,S)
            if t.dim() == 4 and t.size(0) > 0 and t.size(1) > 0:
                attn2d = t[0, 0, :, :]
            elif t.dim() == 3 and t.size(0) > 0:
                attn2d = t[0, :, :]
            else:
                return

            attention_sample = attn2d.detach().float().cpu().numpy()

            # Numerical safety for entropy
            eps = 1e-8
            attention_sample = np.clip(attention_sample, eps, 1.0)
            entropy = float(np.mean(-attention_sample * np.log(attention_sample)))

            attention_data = {
                'step': self.current_step,
                'attention_matrix': attention_sample.tolist(),
                'attention_entropy': entropy,
                'timestamp': datetime.now().isoformat()
            }

            self.attention_queue.put(attention_data)
        except Exception:
            # Swallow attention capture issues so training continues
            return
    
    def _check_security_threats(self, gradient_norm):
        """Real-time security threat detection"""
        current_time = datetime.now()
        
        # Large gradient norm - potential data extraction risk
        if gradient_norm > self.CRITICAL_GRADIENT_NORM:
            threat = {
                'type': 'gradient_leakage',
                'severity': min(gradient_norm / self.CRITICAL_GRADIENT_NORM, 3.0),
                'step': self.current_step,
                'norm': gradient_norm,
                'timestamp': current_time.isoformat(),
                'description': f'Critical gradient norm detected: {gradient_norm:.4f}'
            }
            self.attack_events.append(threat)
            print(f"🚨 SECURITY ALERT: Large gradient norm {gradient_norm:.4f} at step {self.current_step}")
        
        # Privacy budget exhaustion
        if self.privacy_budget > self.PRIVACY_BUDGET_LIMIT:
            threat = {
                'type': 'privacy_exhaustion',
                'severity': 2.0,
                'step': self.current_step,
                'budget': self.privacy_budget,
                'timestamp': current_time.isoformat(),
                'description': f'Privacy budget exhausted: {self.privacy_budget:.2f}ε'
            }
            self.attack_events.append(threat)
            print(f"⚠️ PRIVACY ALERT: Budget exhausted {self.privacy_budget:.2f}ε at step {self.current_step}")
    
    def _get_current_gradient_norm(self):
        """Get the most recent gradient norm"""
        return self.gradient_norms[-1] if self.gradient_norms else 0.0
    
    def _security_monitor_loop(self):
        """Background security monitoring loop"""
        print("🛡️ Security monitoring thread started")
        
        while not self.stop_monitoring:
            try:
                # Periodic security checks
                current_norm = self._get_current_gradient_norm()
                
                # Simulate membership inference attacks
                if len(self.gradient_norms) > 10:
                    recent_norms = self.gradient_norms[-10:]
                    variance = np.var(recent_norms)
                    
                    if variance > 2.0:  # High variance indicates potential attack
                        threat = {
                            'type': 'membership_inference',
                            'severity': min(variance / 2.0, 2.0),
                            'step': self.current_step,
                            'variance': variance,
                            'timestamp': datetime.now().isoformat(),
                            'description': f'High gradient variance detected: {variance:.4f}'
                        }
                        self.attack_events.append(threat)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"❌ Security monitor error: {e}")
                time.sleep(1)
        
        print("🛡️ Security monitoring thread stopped")
    
    def get_real_time_data(self):
        """Get real-time data for the dashboard"""
        # Collect all data from queues
        gradients = []
        losses = []
        attentions = []
        
        # Drain gradient queue
        while not self.gradient_queue.empty():
            try:
                gradients.append(self.gradient_queue.get_nowait())
            except queue.Empty:
                break
        
        # Drain loss queue  
        while not self.loss_queue.empty():
            try:
                losses.append(self.loss_queue.get_nowait())
            except queue.Empty:
                break
        
        # Drain attention queue
        while not self.attention_queue.empty():
            try:
                attentions.append(self.attention_queue.get_nowait())
            except queue.Empty:
                break
        
        return {
            'is_training': self.is_training,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'model_name': self.model_name,
            'device': str(self.device),
            'gradients': gradients,
            'losses': losses,
            'attentions': attentions,
            'attack_events': self.attack_events[-50:],  # Last 50 events
            'gradient_norms': self.gradient_norms[-100:],  # Last 100 norms
            'privacy_budget': self.privacy_budget,
            'security_status': self._get_security_status(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_security_status(self):
        """Get current security status"""
        current_norm = self._get_current_gradient_norm()
        recent_attacks = len([e for e in self.attack_events[-10:] if e['severity'] > 1.0])
        
        if recent_attacks > 3 or current_norm > self.CRITICAL_GRADIENT_NORM * 1.5:
            return 'CRITICAL'
        elif recent_attacks > 1 or current_norm > self.CRITICAL_GRADIENT_NORM:
            return 'HIGH'
        elif self.privacy_budget > self.PRIVACY_BUDGET_LIMIT * 0.8:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def save_monitoring_data(self, filename=None):
        """Save monitoring data to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"live_monitoring_data_{timestamp}.json"
        
        data = self.get_real_time_data()
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 Monitoring data saved to {filename}")
        return filename

def main():
    """Test the live model monitor"""
    print("🚀 Testing Live Model Monitor")
    
    # Initialize monitor
    monitor = LiveModelMonitor("distilbert-base-uncased")
    
    # Start live training
    try:
        monitor.start_live_training(num_epochs=2)
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user")
    
    # Save monitoring data
    filename = monitor.save_monitoring_data()
    
    # Print summary
    data = monitor.get_real_time_data()
    print(f"\n📊 Training Summary:")
    print(f"   Steps completed: {data['current_step']}")
    print(f"   Attack events: {len(data['attack_events'])}")
    print(f"   Final privacy budget: {data['privacy_budget']:.2f}ε")
    print(f"   Security status: {data['security_status']}")
    print(f"   Data saved to: {filename}")

if __name__ == "__main__":
    main() 