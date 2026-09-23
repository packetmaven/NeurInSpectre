#!/usr/bin/env python3
"""Download the EMBER 2024 challenge set (features only, no PE binaries).

The challenge set is 6,315 malicious files that were initially undetected by
every VirusTotal AV product and only labeled malicious after being re-scanned
at least 30 days later. This script fetches ``challenge.zip`` from the
official HuggingFace dataset repo (``joyce8/EMBER2024``), records a SHA-256
manifest, and unzips into ``data/ember/ember2024/dataset/challenge/``.

The unzipped JSONL files carry the full v3 raw features per record, so
downstream scripts (e.g. ``scripts/score_ember2024_challenge.py``) can run
end-to-end without touching a single PE binary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

REPO_ID = "joyce8/EMBER2024"
FILENAME = "challenge.zip"


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
        default="data/ember/ember2024/dataset",
        help="Destination directory for the challenge archive + unzipped JSONLs",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path for the SHA-256 manifest (default: <dest>/challenge_manifest.json)",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    local = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=FILENAME,
            local_dir=str(dest),
        )
    )
    size = local.stat().st_size
    digest = _sha256(local)
    print(f"{FILENAME}: {size:,} bytes  sha256={digest}")

    unpacked = dest / "challenge"
    unpacked.mkdir(exist_ok=True)
    with zipfile.ZipFile(local) as zf:
        zf.extractall(unpacked)
    entries = sorted(unpacked.rglob("*.jsonl"))
    print(f"unpacked into {unpacked}  ({len(entries)} jsonl files)")

    manifest_path = Path(args.manifest) if args.manifest else (dest / "challenge_manifest.json")
    manifest = {
        "repo_id": REPO_ID,
        "filename": FILENAME,
        "path": str(local),
        "size": size,
        "sha256": digest,
        "unpacked_dir": str(unpacked),
        "jsonl_files": [str(p.relative_to(dest)) for p in entries],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
