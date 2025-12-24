import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("🤖 Loading real model (GPT-2)...")
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Move to MPS if available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = model.to(device)
print(f"✅ Model loaded on {device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

print("🚀 Starting fine-tuning - Monitor should detect gradients...")

for step in range(50):
    # Create dummy training data
    inputs = tokenizer("This is a test sentence for gradient monitoring.", return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    
    # Backward pass - MONITOR CAPTURES HERE
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 5 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
    
    time.sleep(0.5)  # Slow enough for monitoring

print("✅ Fine-tuning complete!")
