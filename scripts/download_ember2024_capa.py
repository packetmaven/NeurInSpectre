#!/usr/bin/env python3
"""Download shards of the EMBER 2024 Capa supplement (function-level).

The supplement (``joyce8/EMBER2024-capa`` on HuggingFace) is
**~23.8 GB across 128 weekly zips** covering malicious Win32 and Win64
files only. Each zip holds JSON objects for individual functions with raw
bytes, disassembly, and Capa capability labels (16,356,790 functions
total).

Do **not** pull the whole set unless you actually need it. Common flags:

  python scripts/download_ember2024_capa.py --smallest 1
  python scripts/download_ember2024_capa.py --split test --file-type Win32
  python scripts/download_ember2024_capa.py --all      # 23.8 GB

The script always writes a SHA-256 manifest under
``data/ember/ember2024/capa/download_manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ID = "joyce8/EMBER2024-capa"


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _list_shards():
    from huggingface_hub import get_paths_info, list_repo_files

    files = [f for f in list_repo_files(REPO_ID, repo_type="dataset") if f.startswith("data/")]
    infos = get_paths_info(REPO_ID, paths=files, repo_type="dataset")
    return [(pi.path, int(getattr(pi, "size", None) or 0)) for pi in infos]


def _filter(shards, args):
    picked = list(shards)
    if args.split:
        picked = [(p, s) for p, s in picked if f"/{args.split}/" in p]
    if args.file_type:
        picked = [(p, s) for p, s in picked if f"_{args.file_type}_" in p]
    picked.sort(key=lambda ps: ps[1])
    if args.smallest and args.smallest > 0:
        picked = picked[: args.smallest]
    elif args.all:
        pass
    elif not (args.split or args.file_type):
        raise SystemExit(
            "Refusing to download all 23.8 GB by default. Pass --all, --smallest N, "
            "or --split/--file-type filters."
        )
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="data/ember/ember2024/capa")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--all", action="store_true", help="Grab every shard (23.8 GB)")
    parser.add_argument(
        "--smallest", type=int, default=0, help="Grab N smallest shards (sniff a subset)"
    )
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--file-type", choices=["Win32", "Win64"], default=None)
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    picked = _filter(_list_shards(), args)
    total = sum(sz for _, sz in picked)
    print(f"selected {len(picked)} shards, ~{total / 1e9:.2f} GB")

    manifest = []
    for name, expected in picked:
        local = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=name,
                local_dir=str(dest),
            )
        )
        sha = _sha256(local)
        size = local.stat().st_size
        print(f"  {name}: {size / 1e6:6.1f} MB  sha256={sha[:16]}…")
        manifest.append(
            {
                "name": name,
                "path": str(local),
                "size": size,
                "sha256": sha,
                "repo_id": REPO_ID,
            }
        )

    manifest_path = Path(args.manifest) if args.manifest else (dest / "download_manifest.json")
    # Merge with any prior manifest so incremental downloads accumulate.
    prior = []
    if manifest_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text())
        except Exception:
            prior = []
    known = {row["sha256"] for row in prior if "sha256" in row}
    merged = prior + [row for row in manifest if row["sha256"] not in known]
    manifest_path.write_text(json.dumps(merged, indent=2, sort_keys=True))
    print(f"wrote {manifest_path}  ({len(merged)} shards indexed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
