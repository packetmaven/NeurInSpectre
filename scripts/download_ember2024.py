#!/usr/bin/env python3
"""Download EMBER 2024 (thrember) LightGBM detection models with SHA-256 audit.

Pulls the three PE-focused models (EMBER2024_PE, EMBER2024_Win32,
EMBER2024_Win64) by default; pass ``--all`` to grab every classifier in the
HuggingFace repo (family, behavior, packer, group, exploit, file_property, etc.).

The extractor library (``thrember``, feature version 3) is pip-installed
separately; see the module docstring in ``neurinspectre/malware/ember2024_extract.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ID = "joyce8/EMBER2024-benchmark-models"
DEFAULT_MODELS = (
    "EMBER2024_PE.model",
    "EMBER2024_Win32.model",
    "EMBER2024_Win64.model",
)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        default="data/ember/ember2024",
        help="Destination directory for model files",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific model filenames to download (default: PE + Win32 + Win64)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every .model file in the HuggingFace repo",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Where to write the SHA-256 manifest (default: <dest>/download_manifest.json)",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download, list_repo_files

    if args.all:
        wanted = [f for f in list_repo_files(REPO_ID) if f.endswith(".model")]
    else:
        wanted = list(args.models or DEFAULT_MODELS)

    manifest = []
    for name in wanted:
        local = Path(
            hf_hub_download(repo_id=REPO_ID, filename=name, local_dir=str(dest))
        )
        digest = _sha256(local)
        size = local.stat().st_size
        print(f"{name}: {size:,} bytes  sha256={digest}")
        manifest.append(
            {
                "name": name,
                "path": str(local),
                "size": size,
                "sha256": digest,
                "repo_id": REPO_ID,
            }
        )

    manifest_path = (
        Path(args.manifest) if args.manifest else (dest / "download_manifest.json")
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
