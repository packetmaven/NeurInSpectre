#!/usr/bin/env python3
def run_gradient_inversion(args):
    import numpy as np
    import torch
    import torch.nn as nn
    from pathlib import Path
    from neurinspectre.attacks.gradient_inversion_attack import GradientInversionAttack, GradientInversionConfig
    
    print(f"\n🔴 Gradient Inversion Attack")
    print(f"   Gradients: {args.gradients}")
    
    gradients = np.load(args.gradients)
    print(f"   ✅ Loaded: {gradients.shape}")
    
    # Infer dimensions from gradient shape
    if gradients.ndim == 2:
        input_dim, num_classes = gradients.shape
    else:
        input_dim, num_classes = 784, 10
    
    print(f"   📐 Model: input={input_dim}, classes={num_classes}")
    
    # SimpleMLP matching gradient shape
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_classes)
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    
    model = SimpleMLP()
    
    config = GradientInversionConfig(
        method=args.method,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        input_shape=(1, input_dim),
        num_classes=num_classes,
        device='cpu'
    )
    
    attack = GradientInversionAttack(model=model, config=config)
    
    # Use correct parameter name from model
    grad_dict = {'fc.weight': torch.from_numpy(gradients).float()}
    
    print(f"\n⚡ Reconstructing ({args.method}, {config.max_iterations} iters)...")
    
    result = attack.reconstruct(grad_dict)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'reconstructed_data.npy', result['reconstructed_data'])
    
    print(f"\n✅ Complete")
    print(f"   Success: {result['success']}")
    print(f"   Loss: {result['final_loss']:.6f}")
    print(f"   Iters: {result['iterations']}")
    print(f"   MITRE: {result['mitre_atlas']['technique']}")
    print(f"   💾 {output_dir}/reconstructed_data.npy\n")
    return 0
