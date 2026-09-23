#!/usr/bin/env python3
"""Build a Capa-tag sidecar for a PE corpus from the EMBER 2024 challenge JSONLs.

For each file in a PE directory, compute SHA-256 and look it up in the
EMBER 2024 challenge JSONLs (or any JSONL-with-sha256 source). Files that
match get their per-file tag record copied; files that don't match are
reported for --filter-include-untagged decision-making.

Output is a plain JSON dict keyed by lowercased SHA-256, compatible with
``neurinspectre audit --filter-tags-json``.

    python scripts/tag_pe_corpus_from_ember2024.py \
        --pe-dir /path/to/pe --dataset-dir data/ember/ember2024/dataset/challenge \
        --output pe_tags.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable


def _iter_pe_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        yield path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _index_dataset(dataset_dir: Path) -> Dict[str, Dict]:
    """Return {sha256_lower: record} for every JSONL row under dataset_dir."""
    idx: Dict[str, Dict] = {}
    for jsonl in sorted(dataset_dir.rglob("*.jsonl")):
        with jsonl.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sha = row.get("sha256")
                if sha:
                    idx[str(sha).lower()] = row
    return idx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pe-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/ember/ember2024/dataset/challenge"),
    )
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--keep-fields",
        default="sha256,file_type,family,behavior,file_property,packer,exploit,group,caps,ttps,mbc,detection_ratio",
        help="Comma list of dataset fields to copy into the sidecar",
    )
    args = parser.parse_args()

    if not args.pe_dir.is_dir():
        raise SystemExit(f"PE dir not found: {args.pe_dir}")
    if not args.dataset_dir.is_dir():
        raise SystemExit(f"Dataset dir not found: {args.dataset_dir}")

    fields = [x.strip() for x in args.keep_fields.split(",") if x.strip()]
    print(f"indexing {args.dataset_dir} ...")
    idx = _index_dataset(args.dataset_dir)
    print(f"  {len(idx):,} unique sha256 records indexed")

    tags: Dict[str, Dict] = {}
    n_seen = 0
    n_matched = 0
    n_missing = 0
    missing_examples = []
    for path in _iter_pe_paths(args.pe_dir):
        n_seen += 1
        sha = _sha256_file(path)
        row = idx.get(sha)
        if row is None:
            n_missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(path.name)
            continue
        n_matched += 1
        tags[sha] = {k: row.get(k) for k in fields if k in row}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tags, indent=2))
    print(
        f"scanned={n_seen}  matched={n_matched}  missing={n_missing}  "
        f"wrote {args.output}"
    )
    if missing_examples:
        print("  first missing:", ", ".join(missing_examples))
    if n_matched == 0:
        print("  no files matched the dataset — nothing to filter by")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
