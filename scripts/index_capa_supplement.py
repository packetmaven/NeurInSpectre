#!/usr/bin/env python3
"""Build a compact SHA-256 → per-function Capa metadata index from the
EMBER 2024 Capa supplement (23.8 GB across 128 shards).

Emits a single JSON keyed by lowercased SHA-256; each value is a list of
``{func_addr, capa, byte_len}``. The full supplement is not needed at
query time — downstream tooling reads only this index.

Runtime: on a warm SSD, ~2–4 minutes on the full supplement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurinspectre.malware.capa_supplement_index import build_index, summarize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supplement",
        type=Path,
        default=Path("data/ember/ember2024/capa"),
        help="Root directory containing the Capa supplement shards (.zip)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/ember/ember2024/capa_supplement_index.json"),
    )
    parser.add_argument(
        "--limit-functions-per-file",
        type=int,
        default=None,
        help="Cap per-file function count (for smoke tests)",
    )
    args = parser.parse_args()

    if not args.supplement.is_dir():
        raise SystemExit(f"Capa supplement not found at {args.supplement}. Run: neurinspectre download-ember2024-capa --all")

    print(f"indexing {args.supplement} ...")
    index = build_index(args.supplement, limit_functions_per_file=args.limit_functions_per_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, indent=None, separators=(",", ":")))
    summ = summarize(index)
    print(
        f"wrote {args.output}\n"
        f"  n_files={summ['n_files']:,} n_functions={summ['n_functions']:,} "
        f"n_unique_capabilities={summ['n_unique_capabilities']:,}"
    )
    print("  top-10 capabilities:")
    for name, n in summ["top_capabilities"][:10]:
        print(f"    {n:8d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
