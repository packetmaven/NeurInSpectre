#!/usr/bin/env python3
"""TS-Inverse Gradient Inversion Attack CLI"""

def run_ts_inverse(args):
    """Execute TS-Inverse attack"""
    import numpy as np
    import torch
    import importlib
    import json
    from pathlib import Path
    from neurinspectre.attacks.ts_inverse_attack import TSInverseAttack, TSInverseConfig, create_simple_time_series_model
    
    print("\n🔴 TS-Inverse: Gradient Inversion Attack")
    print(f"   Target gradients: {args.target_gradients}")
    print(f"   Quality: {args.reconstruction_quality}")
    
    # Load gradients robustly
    obj = np.load(args.target_gradients, allow_pickle=True)
    if getattr(obj, "dtype", None) is object and getattr(obj, "shape", ()) == ():
        obj = obj.item()
    if isinstance(obj, dict):
        for k in ("gradients", "target_gradients", "data", "X", "x", "arr"):
            if k in obj:
                obj = obj[k]
                break
    gradients = np.asarray(obj)
    if np.issubdtype(gradients.dtype, np.number):
        gradients = np.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)
    if gradients.ndim == 0:
        gradients = gradients.reshape(1, 1)
    elif gradients.ndim == 1:
        gradients = gradients.reshape(1, -1)
    elif gradients.ndim > 2:
        gradients = gradients.reshape(gradients.shape[0], -1)
    print(f"   ✅ Loaded gradients: {gradients.shape}")

    # Best-effort shape inference for dummy-data initialization
    seq_len_guess = int(gradients.shape[0])
    num_features_guess = int(gradients.shape[1]) if gradients.ndim == 2 else int(gradients.size)
    
    # Model selection:
    # - Prefer explicit model factory if provided.
    # - Otherwise require explicit --allow-demo-model for the built-in demo.
    model = None
    if getattr(args, "model_factory", None):
        spec = str(args.model_factory)
        if ":" not in spec:
            print("❌ Invalid --model-factory. Expected 'module:function'.")
            return 1
        mod_name, fn_name = spec.split(":", 1)
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
        except Exception as e:
            print(f"❌ Failed to import model factory {spec}: {e}")
            return 1
        if not callable(fn):
            print(f"❌ Model factory is not callable: {spec}")
            return 1
        kwargs = {}
        if getattr(args, "model_kwargs", None):
            try:
                kwargs = json.loads(str(args.model_kwargs))
                if not isinstance(kwargs, dict):
                    raise ValueError("model_kwargs must be a JSON dict")
            except Exception as e:
                print(f"❌ Invalid --model-kwargs JSON: {e}")
                return 1
        try:
            model = fn(**kwargs)
        except Exception as e:
            print(f"❌ Model factory failed: {e}")
            return 1
    else:
        if not getattr(args, "allow_demo_model", False):
            print("❌ Refusing to use demo model without --allow-demo-model.")
            print("   Provide --model-factory module:function, or pass --allow-demo-model for the built-in scaffold.")
            return 1
        model = create_simple_time_series_model(
            input_dim=num_features_guess,
            sequence_length=max(1, seq_len_guess),
        )
    
    # Config
    iters = {'low': 100, 'medium': 500, 'high': 1000}[args.reconstruction_quality]
    config = TSInverseConfig(max_iterations=iters, learning_rate=0.1)
    # Ensure dummy-data shapes are consistent with the (guessed) model input.
    config.batch_size = 1
    config.sequence_length = max(1, seq_len_guess)
    config.num_features = max(1, num_features_guess)
    
    # Run attack using CORRECT method: .attack()
    attack = TSInverseAttack(model=model, config=config)
    grad_dict = {'weight': torch.from_numpy(gradients).float()}
    result = attack.attack(grad_dict)
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'reconstructed.npy', result.reconstructed_data)
    
    # Use CORRECT attributes from TSInverseResult
    print(f"   ✅ Success: {result.success}")
    print(f"   ✅ Loss: {result.reconstruction_loss:.6f}")
    print(f"   ✅ Temporal Coherence: {result.temporal_coherence:.4f}")
    print(f"   ✅ Iterations: {result.iterations_used}")
    print(f"   💾 Saved: {output_dir}/reconstructed.npy\n")
    return 0
