# working_gradient_monitor.py
import torch
import torch.nn as nn
import json

class WorkingGradientCapture:
    def __init__(self):
        self.gradients = []
    
    def capture_hook(self, module, grad_input, grad_output):
        """This hook DOES work on Mac Silicon MPS"""
        if grad_output is not None:
            for grad in grad_output:
                if grad is not None:
                    # Move to CPU first (critical for MPS)
                    grad_cpu = grad.detach().cpu().numpy()
                    self.gradients.append({
                        'shape': list(grad_cpu.shape),
                        'mean': float(grad_cpu.mean()),
                        'std': float(grad_cpu.std()),
                        'max': float(grad_cpu.max())
                    })
                    print(f" Captured gradient: shape={grad_cpu.shape}, mean={grad_cpu.mean():.6f}")

# Create model
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = model.to(device)

# Create capture
capture = WorkingGradientCapture()

# Register hooks on ALL modules (this WORKS on MPS)
for module in model.modules():
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM)):
        module.register_full_backward_hook(capture.capture_hook)

print(f" Training on {device} with working gradient capture...")

optimizer = torch.optim.Adam(model.parameters())

for i in range(50):
    x = torch.randn(32, 100).to(device)
    y = torch.randint(0, 10, (32,)).to(device)
    
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()  # Hooks fire HERE
    optimizer.step()
    
    if i % 10 == 0:
        print(f"Step {i}, Loss: {loss.item():.4f}, Captured: {len(capture.gradients)}")

# Save results
with open('_cli_runs/working_analysis.json', 'w') as f:
    json.dump({'gradient_history': capture.gradients}, f, indent=2)

print(f" Captured {len(capture.gradients)} gradient samples!")
print(f" Saved to: _cli_runs/working_analysis.json")
