"""100 random Full DOS payloads × the four E12 smoke PEs.

Writes ``results/ember2024/E12/fulldos_pefw_probe.json``. The L1 is the
sum of absolute differences on the thrember PEFormatWarnings band from
``pefilewarnings_offset_dim()``. Quote a perturbation count only from
that file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from neurinspectre.malware.ember2024_extract import (
    extract_ember2024_features,
    pefilewarnings_offset_dim,
)
from neurinspectre.malware.pe_transforms import apply_fulldos, fulldos_capacity

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PE_DIR = Path("/Users/seren3/mwb/pe_only")
DEFAULT_OUT = ROOT / "results/ember2024/E12/fulldos_pefw_probe.json"
SMOKE_NAMES = (
    "001816f728d96a823a88e8243cba32486ff0207c123628fa767073bb4a49cd6e.exe",
    "006bdbc34e844d18a837989fca82ab15552a26733268f54b1a832af30ad6a008.dll",
    "04168e3872a815ace4f4c59a787ef0d10782df4b2111d46d899b10eade942f80.exe",
    "0977370b032f5af5f92e7f93c0917b5c27e78d7a68a0bb526e653591599311f0.dll",
)


def _band(features: np.ndarray, offset: int, dim: int) -> np.ndarray:
    return np.asarray(features, dtype=np.float64)[offset : offset + dim]


def probe_file(path: Path, *, n: int, rng: np.random.Generator, offset: int, dim: int) -> dict:
    raw = path.read_bytes()
    capacity = int(fulldos_capacity(raw))
    base = extract_ember2024_features(raw)
    row = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "fulldos_capacity": capacity,
        "n_attempted": n,
        "n_extracted": 0,
        "n_extract_failed": 0,
        "n_perturbed": 0,
        "max_l1": None,
        "baseline_extracted": base.get("features") is not None,
    }
    if base.get("features") is None or capacity <= 0:
        row["n_extract_failed"] = n
        row["reason"] = "baseline_extract_failed" if capacity > 0 else "no_fulldos_capacity"
        return row
    baseline = _band(base["features"], offset, dim)
    max_l1 = 0.0
    for _ in range(n):
        payload = rng.integers(0, 256, size=capacity, dtype=np.uint8).tobytes()
        mutated = apply_fulldos(raw, payload)
        extracted = extract_ember2024_features(mutated)
        feats = extracted.get("features")
        if feats is None:
            row["n_extract_failed"] += 1
            continue
        row["n_extracted"] += 1
        l1 = float(np.abs(_band(feats, offset, dim) - baseline).sum())
        if l1 > max_l1:
            max_l1 = l1
        if l1 != 0.0:
            row["n_perturbed"] += 1
    row["max_l1"] = max_l1
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pe-dir", type=Path, default=DEFAULT_PE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    offset, dim = pefilewarnings_offset_dim()
    rng = np.random.default_rng(args.seed)
    files = []
    for name in SMOKE_NAMES:
        path = args.pe_dir / name
        if not path.is_file():
            raise SystemExit(f"missing smoke PE: {path}")
        files.append(probe_file(path, n=args.n, rng=rng, offset=offset, dim=dim))
        print(
            f"{name[:16]} cap={files[-1]['fulldos_capacity']} "
            f"perturbed={files[-1]['n_perturbed']}/{files[-1]['n_extracted']} "
            f"max_l1={files[-1]['max_l1']}",
            flush=True,
        )
    n_attempted = sum(r["n_attempted"] for r in files)
    n_extracted = sum(r["n_extracted"] for r in files)
    n_perturbed = sum(r["n_perturbed"] for r in files)
    maxes = [r["max_l1"] for r in files if r["max_l1"] is not None]
    report = {
        "kind": "fulldos_pefw_probe",
        "pefilewarnings_offset": offset,
        "pefilewarnings_dim": dim,
        "n_per_file": args.n,
        "seed": args.seed,
        "n_files": len(files),
        "n_attempted": n_attempted,
        "n_extracted": n_extracted,
        "n_perturbed": n_perturbed,
        "max_l1": max(maxes) if maxes else None,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} perturbed={n_perturbed}/{n_extracted}", flush=True)


if __name__ == "__main__":
    main()
