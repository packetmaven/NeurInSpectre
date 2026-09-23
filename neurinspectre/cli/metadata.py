"""
Run/environment metadata helpers.

These utilities exist to make evaluation runs self-describing for artifact
evaluation and debugging. They intentionally do not encode any paper baselines
or expected performance numbers.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..evaluation.artifact_integrity import sha256_file


def _pkg_version(dist_name: str) -> Optional[str]:
    try:
        from importlib import metadata as importlib_metadata  # py>=3.8 stdlib

        return str(importlib_metadata.version(dist_name))
    except Exception:
        return None


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def collect_env_metadata() -> Dict[str, Any]:
    """
    Best-effort snapshot of environment information.

    Keep this light-weight: avoid any network access or large directory scans.
    """

    env: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "process": {
            "cwd": os.getcwd(),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
        },
        "packages": {},
        "torch": {},
        "optional_modules": {},
    }

    # Common packages (best-effort).
    for dist in (
        "neurinspectre",
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-learn",
        "rich",
        "click",
        "rich-click",
        "onnxruntime",
        "nuscenes-devkit",
    ):
        v = _pkg_version(dist)
        if v:
            env["packages"][dist] = v

    # Optional modules by import name (best-effort).
    for mod in (
        "ember",
        "lief",
        "lightgbm",
        "pefile",
        "nuscenes",
        "onnxruntime",
        "robustbench",
    ):
        env["optional_modules"][mod] = bool(_module_available(mod))

    # Torch hardware capabilities (best-effort).
    try:
        import torch

        torch_info: Dict[str, Any] = {
            "version": str(getattr(torch, "__version__", "unknown")),
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        }
        if torch_info["cuda_available"]:
            try:
                torch_info["cuda_device_count"] = int(torch.cuda.device_count())
                if torch.cuda.device_count() > 0:
                    torch_info["cuda_device_0_name"] = str(torch.cuda.get_device_name(0))
            except Exception:
                pass
        env["torch"] = torch_info
    except Exception:
        env["torch"] = {"available": False}

    return env


def _git(cmd: list[str], *, cwd: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *cmd],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return (proc.stdout or "").strip()
    except Exception:
        return None


def collect_git_metadata(start_dir: str | Path | None = None) -> Dict[str, Any]:
    """
    Best-effort git metadata snapshot.

    This should never fail a run; it is purely informational.
    """

    cwd = Path(start_dir) if start_dir is not None else Path.cwd()
    top = _git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if not top:
        return {"available": False}

    top_p = Path(top)
    commit = _git(["rev-parse", "HEAD"], cwd=top_p)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=top_p)
    status = _git(["status", "--porcelain"], cwd=top_p)
    dirty = bool(status)

    return {
        "available": True,
        "toplevel": str(top_p),
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }


def write_run_metadata(
    out_dir: str | Path,
    *,
    config_path: str | Path | None,
    device: str | None = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Write a `run_metadata.json` file into `out_dir`.
    """

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    cfg_record: Dict[str, Any] = {"path": None, "sha256": None, "copied_to": None}
    if config_path:
        try:
            cfg_p = Path(config_path).resolve()
            cfg_record["path"] = str(cfg_p)
            cfg_record["sha256"] = sha256_file(cfg_p)
            try:
                copied = out_p / "config_used.yaml"
                copied.write_text(cfg_p.read_text(encoding="utf-8"), encoding="utf-8")
                cfg_record["copied_to"] = str(copied)
            except Exception:
                pass
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "tool": {"name": "NeurInSpectre"},
        "argv": list(sys.argv),
        "device": device,
        "config": cfg_record,
        "env": collect_env_metadata(),
        "git": collect_git_metadata(start_dir=Path.cwd()),
        "extra": dict(extra or {}),
    }

    meta_path = out_p / "run_metadata.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    return payload

