#!/usr/bin/env python3
def run_concretizer(args):
    import numpy as np
    import torch
    import importlib
    from pathlib import Path
    from neurinspectre.attacks.concretizer_attack import ConcreTizerAttack, ConcreTizerConfig
    
    print("\n🔴 ConcreTizer: 3D Model Inversion Attack")
    print(f"   Resolution: {args.voxel_resolution}³")

    target_model = None
    target_desc = None

    # 1) TorchScript model path
    target_model_file = getattr(args, "target_model_file", None)
    if target_model_file:
        p = Path(str(target_model_file)).expanduser()
        if not p.exists():
            print(f"❌ Target model file not found: {p}")
            return 1
        try:
            ts = torch.jit.load(str(p), map_location="cpu")
            ts.eval()
        except Exception as e:
            print(f"❌ Failed to load TorchScript model: {e}")
            return 1

        def _ts_model(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                out = ts(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

        target_model = _ts_model
        target_desc = f"TorchScript:{p.name}"

    # 2) Python callable spec: module:function
    if target_model is None:
        spec = getattr(args, "target_model", None)
        if not spec:
            print("❌ Missing --target-model or --target-model-file.")
            print("   Provide --target-model module:function, or --target-model-file path/to/model.ts")
            return 1

        if str(spec).lower() == "dummy":
            if not getattr(args, "allow_dummy", False):
                print("❌ Refusing to use dummy target model without --allow-dummy.")
                return 1
            # Deterministic dummy model (explicit demo mode). Not a real target.
            g = torch.Generator(device="cpu")
            g.manual_seed(0)
            W = torch.randn(16, 10, generator=g) * 0.2
            b = torch.randn(10, generator=g) * 0.05

            def _dummy(x: torch.Tensor) -> torch.Tensor:
                x2 = x.reshape(x.shape[0], -1)
                # Pad/trim to 16 dims deterministically
                if x2.shape[1] < 16:
                    pad = torch.zeros((x2.shape[0], 16 - x2.shape[1]), dtype=x2.dtype, device=x2.device)
                    x2 = torch.cat([x2, pad], dim=1)
                elif x2.shape[1] > 16:
                    x2 = x2[:, :16]
                return x2 @ W.to(x2.dtype) + b.to(x2.dtype)

            target_model = _dummy
            target_desc = "dummy (deterministic demo)"
        else:
            if ":" not in str(spec):
                print("❌ Invalid --target-model. Expected 'module:function' or 'dummy'.")
                return 1
            mod_name, fn_name = str(spec).split(":", 1)
            try:
                mod = importlib.import_module(mod_name)
                fn = getattr(mod, fn_name)
            except Exception as e:
                print(f"❌ Failed to import target callable {spec}: {e}")
                return 1
            if not callable(fn):
                print(f"❌ Target callable is not callable: {spec}")
                return 1

            def _py_model(x: torch.Tensor) -> torch.Tensor:
                out = fn(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if not isinstance(out, torch.Tensor):
                    out = torch.as_tensor(out)
                return out

            target_model = _py_model
            target_desc = f"python:{spec}"

    print(f"   Target: {target_desc}")
    
    config = ConcreTizerConfig(
        voxel_resolution=args.voxel_resolution,
        max_queries=args.max_queries,
        refinement_iterations=args.refinement_iterations
    )
    
    concretizer = ConcreTizerAttack(target_model=target_model, config=config)
    result = concretizer.attack()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use CORRECT attributes from ConcreTizerResult
    np.save(output_dir / 'reconstructed_voxels.npy', result.reconstructed_voxels)
    np.save(output_dir / 'voxel_confidences.npy', result.voxel_confidences)
    
    print("\n✅ Complete")
    print(f"   Queries: {result.num_queries_used}")
    print(f"   Quality: {result.reconstruction_quality:.2%}")
    print(f"   Leakage: {result.information_leakage_score:.4f}")
    print(f"   💾 {output_dir}/reconstructed_voxels.npy\n")
    return 0
