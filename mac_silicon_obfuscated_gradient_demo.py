#!/usr/bin/env python3
"""
Mac Silicon Step-by-Step Obfuscated Gradient Demo
Shows how to create and capture obfuscated gradients with NeurInSpectre autodetection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
import os

def create_obfuscated_gradients_demo():
    """
    Step-by-step demo of obfuscated gradient generation and capture on Mac Silicon
    """
    print("🍎 MAC SILICON OBFUSCATED GRADIENT DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Verify Mac Silicon MPS
    print("\n📱 STEP 1: Verify Mac Silicon MPS Support")
    print("-" * 40)
    
    if not torch.backends.mps.is_available():
        print("❌ Mac Silicon MPS not available! Using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device('mps')
        print("✅ Mac Silicon MPS detected and available")
        print(f"📱 Using device: {device}")
    
    # Step 2: Create a model with obfuscation capabilities
    print("\n🧠 STEP 2: Create Neural Network with Obfuscation Layers")
    print("-" * 40)
    
    class ObfuscatedModel(nn.Module):
        """Model designed to generate obfuscated gradients"""
        def __init__(self, input_size=784, hidden_size=256, output_size=10):
            super().__init__()
            # Main network
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU()
            )
            
            # Obfuscation layers - these will generate unusual gradient patterns
            self.obfuscation_layer = nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.Tanh(),  # Non-linear activation for gradient complexity
                nn.Linear(hidden_size // 4, hidden_size // 2),
                nn.Sigmoid()  # Another non-linear activation
            )
            
            # Output layer
            self.classifier = nn.Linear(hidden_size // 2, output_size)
            
            # Gradient noise injection parameters
            self.noise_scale = nn.Parameter(torch.tensor(0.1))
            
        def forward(self, x):
            # Standard forward pass
            encoded = self.encoder(x)
            
            # Apply obfuscation - this creates complex gradient patterns
            obfuscated = self.obfuscation_layer(encoded)
            
            # Add noise to create suspicious gradient patterns
            if self.training:
                noise = torch.randn_like(obfuscated) * self.noise_scale
                obfuscated = obfuscated + noise
            
            # Final classification
            output = self.classifier(obfuscated)
            return output
        
        def inject_gradient_obfuscation(self):
            """Manually inject obfuscated gradient patterns"""
            for param in self.parameters():
                if param.grad is not None:
                    # Add adversarial noise to gradients
                    adversarial_noise = torch.randn_like(param.grad) * 0.05
                    param.grad += adversarial_noise
                    
                    # Apply gradient clipping with unusual patterns
                    param.grad = torch.clamp(param.grad, -0.5, 0.5)
                    
                    # Introduce gradient sparsity (suspicious pattern)
                    mask = torch.rand_like(param.grad) > 0.3
                    param.grad *= mask.float()
    
    # Create the model
    model = ObfuscatedModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Created ObfuscatedModel with {param_count:,} parameters")
    print(f"🎯 Model includes obfuscation layers and gradient injection capabilities")
    print(f"📱 Model moved to device: {device}")
    
    # Step 3: Set up training with obfuscated gradients
    print("\n⚙️ STEP 3: Configure Training with Gradient Obfuscation")
    print("-" * 40)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Custom loss function that encourages obfuscated gradients
    def obfuscated_loss(outputs, targets, model):
        # Standard classification loss
        base_loss = criterion(outputs, targets)
        
        # Add regularization that creates unusual gradient patterns
        reg_loss = 0
        for param in model.parameters():
            # L1 regularization with complex patterns
            reg_loss += torch.sum(torch.abs(param) * torch.sin(param * 10))
        
        total_loss = base_loss + 0.001 * reg_loss
        return total_loss
    
    print("✅ Configured optimizer: Adam (lr=0.001)")
    print("✅ Loss function: Classification + Obfuscation regularization")
    print("✅ Gradient injection: Manual obfuscation post-backward")
    
    # Step 4: Show what patterns we'll generate
    print("\n🎭 STEP 4: Obfuscated Gradient Patterns to Generate")
    print("-" * 40)
    print("🔸 Adversarial noise injection")
    print("🔸 Gradient clipping with unusual bounds")
    print("🔸 Sparse gradient patterns")
    print("🔸 Non-linear activation interactions")
    print("🔸 Complex regularization gradients")
    print("🔸 Temporal gradient variations")
    
    return model, optimizer, obfuscated_loss, device

def run_obfuscated_training_demo(model, optimizer, loss_fn, device, duration=20):
    """
    Run training that generates obfuscated gradients for the monitor to capture
    """
    print(f"\n🔄 STEP 5: Generate Obfuscated Gradients ({duration} seconds)")
    print("-" * 40)
    
    model.train()
    start_time = time.time()
    epoch = 0
    
    # Statistics tracking
    gradient_stats = {
        'total_steps': 0,
        'obfuscation_applied': 0,
        'avg_gradient_norm': 0,
        'suspicious_patterns': 0
    }
    
    try:
        while time.time() - start_time < duration:
            # Generate synthetic batch
            batch_size = 64
            x = torch.randn(batch_size, 784, device=device)
            y = torch.randint(0, 10, (batch_size,), device=device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x)
            
            # Compute obfuscated loss
            loss = loss_fn(outputs, y, model)
            
            # Backward pass - this generates gradients that the monitor captures
            loss.backward()
            
            # STEP 5A: Apply gradient obfuscation
            model.inject_gradient_obfuscation()
            gradient_stats['obfuscation_applied'] += 1
            
            # STEP 5B: Calculate gradient statistics
            total_grad_norm = 0
            suspicious_count = 0
            
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    total_grad_norm += grad_norm
                    
                    # Check for suspicious patterns
                    sparsity = (param.grad == 0).float().mean().item()
                    if sparsity > 0.5:  # High sparsity is suspicious
                        suspicious_count += 1
            
            gradient_stats['avg_gradient_norm'] += total_grad_norm
            gradient_stats['suspicious_patterns'] += suspicious_count
            gradient_stats['total_steps'] += 1
            
            # Optimizer step
            optimizer.step()
            
            # STEP 5C: Periodic gradient manipulation for variety
            if epoch % 15 == 0:
                # Apply different obfuscation patterns
                for param in model.parameters():
                    if param.grad is not None:
                        # Pattern 1: Gradient reversal attack simulation
                        if epoch % 30 == 0:
                            param.grad *= -0.1
                        
                        # Pattern 2: Gradient scaling attack
                        elif epoch % 45 == 0:
                            param.grad *= 10.0
                            param.grad = torch.clamp(param.grad, -1.0, 1.0)
            
            # Progress reporting
            if epoch % 20 == 0:
                elapsed = time.time() - start_time
                avg_grad_norm = gradient_stats['avg_gradient_norm'] / max(1, gradient_stats['total_steps'])
                
                print(f"⏱️  {elapsed:.1f}s - Epoch {epoch}: Loss={loss.item():.4f}")
                print(f"   📊 Avg gradient norm: {avg_grad_norm:.6f}")
                print(f"   🎭 Obfuscations applied: {gradient_stats['obfuscation_applied']}")
                print(f"   ⚠️  Suspicious patterns: {gradient_stats['suspicious_patterns']}")
            
            epoch += 1
            time.sleep(0.05)  # Small delay for observable gradients
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    
    # Final statistics
    elapsed = time.time() - start_time
    avg_grad_norm = gradient_stats['avg_gradient_norm'] / max(1, gradient_stats['total_steps'])
    
    print(f"\n✅ OBFUSCATED GRADIENT GENERATION COMPLETED")
    print(f"   ⏱️  Duration: {elapsed:.1f} seconds")
    print(f"   🔄 Total training steps: {gradient_stats['total_steps']}")
    print(f"   🎭 Obfuscations applied: {gradient_stats['obfuscation_applied']}")
    print(f"   📊 Average gradient norm: {avg_grad_norm:.6f}")
    print(f"   ⚠️  Suspicious patterns generated: {gradient_stats['suspicious_patterns']}")
    
    return gradient_stats

def main():
    """
    Complete step-by-step demonstration
    """
    print("🚀 Starting complete Mac Silicon obfuscated gradient demonstration...")
    print("💡 This will create a model and generate obfuscated gradients")
    print("📡 Run the gradient monitor in another terminal to capture them!")
    print("\n" + "="*60)
    
    # Create the model and setup
    model, optimizer, loss_fn, device = create_obfuscated_gradients_demo()
    
    # Wait for user to start monitor
    print(f"\n⏳ READY TO GENERATE OBFUSCATED GRADIENTS")
    print("=" * 50)
    print("📡 In another terminal, run:")
    print("   neurinspectre obfuscated-gradient monitor --device mps --duration 30")
    print("")
    print("⌨️  Press Enter when monitor is running to start generating gradients...")
    input()
    
    # Run the obfuscated training
    stats = run_obfuscated_training_demo(model, optimizer, loss_fn, device, duration=25)
    
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print("📊 Check the gradient monitor output for captured obfuscated patterns!")
    print("📄 Analysis reports should show the suspicious gradient patterns we generated.")

if __name__ == "__main__":
    main() 