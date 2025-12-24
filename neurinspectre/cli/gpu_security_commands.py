#!/usr/bin/env python3
"""
NeurInSpectre GPU Security CLI Commands

Integrates `neurinspectre.security.gpu_security` into the top-level CLI:
  neurinspectre gpu-security detect ...
  neurinspectre gpu-security test
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_array(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if getattr(arr, "dtype", None) is object and getattr(arr, "shape", ()) == ():
        arr = arr.item()
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.number):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def register_gpu_security_commands(subparsers) -> None:
    p = subparsers.add_parser(
        "gpu-security",
        aliases=["gpu_security", "gpu-sec", "gpu_sec"],
        help="🎮 GPU-accelerated security analysis (adversarial detection)",
    )
    sp = p.add_subparsers(dest="gpu_security_command", help="GPU security commands")

    det = sp.add_parser("detect", help="Run GPU security detector on array inputs")
    det.add_argument("--data", required=True, help="Input data (.npy/.npz/.npy object)")
    det.add_argument("--reference", help="Optional reference data (.npy/.npz)")
    det.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device preference (default: auto)",
    )
    det.add_argument("--output", "-o", help="Write JSON result to path (otherwise prints JSON)")
    det.set_defaults(func=_handle_gpu_security_detect)

    tst = sp.add_parser("test", help="Run built-in GPU security self-test")
    tst.set_defaults(func=_handle_gpu_security_test)


def _handle_gpu_security_detect(args) -> int:
    try:
        from ..security.gpu_security import GPUAdversarialDetector, get_optimal_device

        dev = str(getattr(args, "device", "auto") or "auto")
        if dev == "auto":
            device = get_optimal_device()
        else:
            import torch

            device = torch.device(dev)

        data = _load_array(str(args.data))
        ref = _load_array(str(args.reference)) if getattr(args, "reference", None) else None

        detector = GPUAdversarialDetector(device)
        res = detector.detect_adversarial_attacks(data, ref)

        payload = {
            "device": str(device),
            "data_shape": list(np.asarray(data).shape),
            "reference_shape": None if ref is None else list(np.asarray(ref).shape),
            "threat_level": getattr(res, "threat_level", None),
            "confidence": float(getattr(res, "confidence", 0.0)),
            "processing_time": float(getattr(res, "processing_time", 0.0)),
            "gpu_memory_used": float(getattr(res, "gpu_memory_used", 0.0)),
            "device_info": getattr(res, "device_info", None),
            "detected_attacks": getattr(res, "detected_attacks", None),
            "recommended_actions": getattr(res, "recommended_actions", None),
        }

        out = getattr(args, "output", None)
        if out:
            outp = Path(str(out))
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(payload, indent=2))
            print(str(outp))
        else:
            print(json.dumps(payload, indent=2))
        return 0
    except Exception as e:
        logger.error(f"GPU security detect failed: {e}")
        return 1


def _handle_gpu_security_test(args) -> int:
    try:
        from ..security.gpu_security import test_gpu_security_system

        ok = bool(test_gpu_security_system())
        return 0 if ok else 1
    except Exception as e:
        logger.error(f"GPU security test failed: {e}")
        return 1


