#!/usr/bin/env python3
"""
Generate a REAL, reproducible PGD-like gradient sequence from a cached HuggingFace model.

Why this exists:
- The paper's CLI examples reference `pgd_gradient_sequence.npy`.
- Users often don't have a ready-made gradient trajectory file.
- This script produces an authentic gradient trajectory (computed by autograd) without requiring
  network access, as long as the model is cached locally.

Method:
- Treat input embeddings as the continuous attack variable.
- Perform a PGD-style gradient-ascent loop to increase the LM loss.
- Save the gradient w.r.t. the embedding perturbation at each step.

Output:
- A NumPy array of shape (steps, D), where D = seq_len * hidden_size.

Example:
  python tools/generate_real_pgd_gradient_sequence.py \\
    --model gpt2 \\
    --prompt "The quick brown fox" \\
    --steps 100 \\
    --eps 0.1 \\
    --step-size 0.01 \\
    --output pgd_gradient_sequence.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a real PGD-like gradient sequence (.npy).")
    ap.add_argument("--model", default="gpt2", help="HuggingFace model id (default: gpt2)")
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.", help="Prompt text")
    ap.add_argument("--steps", type=int, default=100, help="Number of PGD steps (default: 100)")
    ap.add_argument("--eps", type=float, default=0.1, help="L_inf PGD radius in embedding space (default: 0.1)")
    ap.add_argument("--step-size", type=float, default=0.01, help="PGD step size (default: 0.01)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--max-length", type=int, default=24, help="Max tokens (default: 24)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"], help="Device (default: cpu)")
    ap.add_argument(
        "--output",
        default="pgd_gradient_sequence.npy",
        help="Output path (.npy) (default: pgd_gradient_sequence.npy)",
    )
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(str(args.device))

    # Force offline operation: only use local cache (matches “no network required” claim).
    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True)
    model.to(device)
    model.eval()

    # Tokenize prompt (small seq_len keeps it fast).
    enc = tok(
        str(args.prompt),
        return_tensors="pt",
        truncation=True,
        max_length=int(args.max_length),
    )
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("Prompt must tokenize to at least 2 tokens to form a next-token loss.")

    # Base embeddings are fixed; delta is the attack variable.
    with torch.no_grad():
        base = model.get_input_embeddings()(input_ids).detach()
    delta = torch.zeros_like(base, device=device)

    eps = float(args.eps)
    step_size = float(args.step_size)
    steps = int(args.steps)

    grads = []
    losses = []

    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        out = model(inputs_embeds=(base + delta))
        logits = out.logits  # [B, T, V]

        # Teacher-forcing next-token loss; PGD maximizes this loss.
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()

        g = delta.grad.detach()
        grads.append(g.reshape(-1).float().cpu().numpy())
        losses.append(float(loss.detach().cpu().item()))

        with torch.no_grad():
            # PGD ascent step in L_inf ball.
            delta = delta + step_size * torch.sign(g)
            delta.clamp_(-eps, eps)

    arr = np.asarray(grads, dtype=np.float32)

    out_path = Path(str(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)

    # Also write a tiny sidecar JSON for provenance.
    meta = {
        "model": str(args.model),
        "prompt": str(args.prompt),
        "steps": int(steps),
        "eps": float(eps),
        "step_size": float(step_size),
        "max_length": int(args.max_length),
        "device": str(device),
        "shape": list(arr.shape),
        "loss_first": float(losses[0]) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
        "seed": int(args.seed),
    }
    try:
        import json

        with open(out_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


