#!/usr/bin/env python
"""EMBER 2018 same-sample audit inside the LIEF 0.9.0 Linux image.

Python 3.6 only. Do not import neurinspectre (3.10 syntax). This process is
the one that imports lief, so official_reproduction is computed here.

Fails closed unless platform is Linux and lief.__version__ starts with 0.9.0.
Does not execute PE files. Problem-space transforms are Full DOS (Demetrio)
and overlay/padding (Kolosnjaji). No GAMMA section injection.
"""
from __future__ import print_function

import argparse
import json
import os
import platform
import struct
import sys
import time

import numpy as np


def _official(lief_version):
    if platform.system() != "Linux":
        return False
    return str(lief_version or "").startswith("0.9.0")


def _e_lfanew(pe):
    if len(pe) < 64 or pe[:2] != b"MZ":
        raise ValueError("not a DOS/PE image")
    return int(struct.unpack_from("<I", pe, 0x3C)[0])


def _fulldos_ranges(pe):
    try:
        e = _e_lfanew(pe)
    except ValueError:
        return []
    if e < 64 or e > len(pe):
        return []
    ranges = []
    if 0x3C > 2:
        ranges.append((2, 0x3C))
    if e > 0x40:
        ranges.append((0x40, e))
    return [(a, b) for a, b in ranges if b > a]


def _apply_fulldos(pe, payload):
    ranges = _fulldos_ranges(pe)
    if not ranges:
        raise ValueError("no fulldos capacity")
    out = bytearray(pe)
    src = bytes(payload or b"")
    offset = 0
    for start, end in ranges:
        n = end - start
        chunk = src[offset:offset + n]
        if len(chunk) < n:
            chunk = chunk + bytes(n - len(chunk))
        out[start:end] = chunk[:n]
        offset += n
    return bytes(out)


def _parse(data):
    import pefile
    try:
        pe = pefile.PE(data=data)
        n_sections = len(getattr(pe, "sections", []) or [])
        entry = int(getattr(getattr(pe, "OPTIONAL_HEADER", None), "AddressOfEntryPoint", 0) or 0)
        passed = bool(data[:2] == b"MZ" and n_sections >= 1)
        return {"passed": passed, "entry_point": entry, "n_sections": n_sections}
    except Exception as exc:
        return {"passed": False, "error": str(exc), "entry_point": None, "n_sections": None}


def _valid(original, mutated, kind):
    before = _parse(original)
    after = _parse(mutated)
    reasons = []
    if before.get("passed") is not True:
        reasons.append("original_pe_invalid")
    if after.get("passed") is not True:
        reasons.append("mutated_pe_invalid")
    try:
        e0 = _e_lfanew(original)
        e1 = _e_lfanew(mutated)
    except ValueError:
        reasons.append("e_lfanew_unreadable")
        e0 = e1 = None
    if original[:2] != b"MZ" or mutated[:2] != b"MZ":
        reasons.append("mz_missing")
    if e0 is not None and e1 is not None and e0 != e1:
        reasons.append("e_lfanew_changed")
    if before.get("entry_point") != after.get("entry_point"):
        reasons.append("entry_point_changed")
    if before.get("n_sections") != after.get("n_sections"):
        reasons.append("section_count_changed")
    if kind == "fulldos" and e0 is not None and mutated[e0:] != original[e0:]:
        reasons.append("pe_image_rewritten")
    if kind == "padding":
        if not mutated.startswith(original):
            reasons.append("original_bytes_rewritten")
        if len(mutated) < len(original):
            reasons.append("file_shrunk")
    return len(reasons) == 0, reasons


def _extract(extractor, data):
    vec = np.asarray(extractor.feature_vector(data), dtype=np.float32).reshape(-1)
    if vec.size == 0 or not bool(np.isfinite(vec).all()):
        return None
    return vec


def _score(booster, vec):
    raw = float(np.asarray(booster.predict(vec.reshape(1, -1))).reshape(-1)[0])
    return raw


def _search(pe, extractor, booster, n_queries, payload_size, rng):
    vec0 = _extract(extractor, pe)
    if vec0 is None:
        return {"success": False, "scored": False, "queries_used": 0, "reasons": ["extract_failed"]}
    p0 = _score(booster, vec0)
    queries = 1
    if p0 < 0.5:
        return {
            "chosen_attack": "clean_already_evading",
            "success": True,
            "realizable": True,
            "queries_used": 1,
            "clean_p_malware": p0,
            "best_p_malware": p0,
        }
    ranges = _fulldos_ranges(pe)
    capacity = sum(b - a for a, b in ranges)
    best_p = p0
    best_kind = "clean"
    for _step in range(max(n_queries - 1, 0)):
        use_dos = capacity > 0 and bool(rng.rand() < 0.5)
        if use_dos:
            kind = "fulldos"
            payload = rng.randint(0, 256, size=capacity).astype(np.uint8).tobytes()
            try:
                cand = _apply_fulldos(pe, payload)
            except ValueError:
                queries += 1
                continue
        else:
            kind = "padding"
            payload = rng.randint(0, 256, size=payload_size).astype(np.uint8).tobytes()
            cand = pe + payload
        queries += 1
        ok, _reasons = _valid(pe, cand, kind)
        if not ok:
            continue
        vec = _extract(extractor, cand)
        if vec is None:
            continue
        p_mal = _score(booster, vec)
        if p_mal < best_p:
            best_p = p_mal
            best_kind = kind
        if p_mal < 0.5:
            return {
                "chosen_attack": kind,
                "success": True,
                "realizable": True,
                "queries_used": queries,
                "clean_p_malware": p0,
                "best_p_malware": p_mal,
            }
    return {
        "chosen_attack": best_kind,
        "success": False,
        "realizable": True,
        "queries_used": queries,
        "clean_p_malware": p0,
        "best_p_malware": best_p,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pe-dir", required=True)
    parser.add_argument("--model-path", default="/work/data/ember/ember2018/ember_model_2018.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--payload-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=0, help="0 = scan every file")
    parser.add_argument("--max-detected", type=int, default=0, help="0 = attack every detected PE")
    args = parser.parse_args()

    import lief
    import lightgbm as lgb
    from ember.features import PEFeatureExtractor

    lief_version = str(lief.__version__)
    official = _official(lief_version)
    status = {
        "available": True,
        "lief_version": lief_version,
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "official_reproduction": official,
        "quote_as_ember2018": official,
        "runner": "scripts/run_ember2018_lief090.py",
        "note": "Python 3.6 + lief 0.9.0 linux egg. Not the Python 3.10 CLI.",
    }
    print(json.dumps({"extractor": status}, indent=2), flush=True)
    if not official:
        print("refusing: official_reproduction is false", file=sys.stderr, flush=True)
        return 2

    if not os.path.isfile(args.model_path):
        print("model missing: %s" % args.model_path, file=sys.stderr)
        return 2
    if not os.path.isdir(args.pe_dir):
        print("pe dir missing: %s" % args.pe_dir, file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)
    booster = lgb.Booster(model_file=args.model_path)
    extractor = PEFeatureExtractor(2, print_feature_warning=False)

    names = sorted(
        n for n in os.listdir(args.pe_dir)
        if os.path.isfile(os.path.join(args.pe_dir, n)) and not n.startswith(".")
    )
    if args.max_files:
        names = names[: int(args.max_files)]

    samples = []
    detected = []
    t0 = time.time()
    for i, name in enumerate(names, 1):
        path = os.path.join(args.pe_dir, name)
        data = open(path, "rb").read()
        row = {"path": path, "name": name, "size": len(data)}
        try:
            _e_lfanew(data)
        except ValueError:
            row.update({"kept": False, "reason": "not_a_valid_pe"})
            samples.append(row)
            continue
        parsed = _parse(data)
        if not parsed.get("passed"):
            row.update({"kept": False, "reason": "not_a_valid_pe"})
            samples.append(row)
            continue
        vec = _extract(extractor, data)
        if vec is None:
            row.update({"kept": False, "reason": "extract_failed"})
            samples.append(row)
            continue
        if int(vec.size) != 2381:
            row.update({"kept": False, "reason": "dim_%d" % int(vec.size)})
            samples.append(row)
            continue
        p_mal = _score(booster, vec)
        row.update({
            "kept": p_mal >= 0.5,
            "reason": None if p_mal >= 0.5 else "not_detected_as_malware",
            "p_malware": p_mal,
            "feat_dim": 2381,
        })
        samples.append(row)
        if row["kept"]:
            detected.append((path, data, p_mal))
        if i % 10 == 0 or i == len(names):
            print("[inventory] %d/%d detected=%d" % (i, len(names), len(detected)), flush=True)
        if args.max_detected and len(detected) >= int(args.max_detected):
            break

    inventory = {
        "summary": {
            "n_listed": len(names),
            "n_scanned": len(samples),
            "n_detected_malware": len(detected),
            "elapsed_sec": time.time() - t0,
            "official_reproduction": True,
            "lief_version": lief_version,
        },
        "samples": samples,
    }
    inv_path = os.path.join(args.output_dir, "linux_pe_inventory.json")
    with open(inv_path, "w") as fh:
        json.dump(inventory, fh, indent=2)
    print(json.dumps(inventory["summary"], indent=2), flush=True)
    if not detected:
        print("no detected malware; refusing to invent a table", file=sys.stderr)
        return 2

    rng = np.random.RandomState(int(args.seed))
    rows = []
    n_flip = 0
    for idx, (path, data, p_clean) in enumerate(detected, 1):
        print("[attack] %d/%d %s" % (idx, len(detected), os.path.basename(path)), flush=True)
        result = _search(data, extractor, booster, int(args.n_queries), int(args.payload_size), rng)
        result["path"] = path
        result["clean_p_malware"] = result.get("clean_p_malware", p_clean)
        rows.append(result)
        if result.get("success") and result.get("realizable"):
            n_flip += 1
        ck = {
            "n_done": idx,
            "n_detected": len(detected),
            "problem_valid_asr": float(n_flip) / float(idx),
            "official_reproduction": True,
            "rows": rows,
        }
        with open(os.path.join(args.output_dir, "checkpoint.json"), "w") as fh:
            json.dump(ck, fh)

    n = len(rows)
    report = {
        "kind": "ember2018_lief090_same_sample",
        "official_reproduction": True,
        "quote_as_ember2018": True,
        "extractor": status,
        "n_examples": n,
        "n_detected_malware": n,
        "problem_space": {
            "attack_success_rate": float(n_flip) / float(n),
            "valid_success_rate": float(n_flip) / float(n),
            "transforms": [r.get("chosen_attack") for r in rows],
            "gamma_padding": False,
        },
        "feature_vs_problem_space": {
            "n": n,
            "same_sample": True,
            "feature_space_asr": None,
            "problem_space_valid_asr": float(n_flip) / float(n),
            "note": "FeatureSquare not re-run in the py3.6 image. Problem-space only.",
        },
        "samples": rows,
        "notes": [
            "Linux + Python 3.6 + lief 0.9.0 egg. official_reproduction true.",
            "Not the Python 3.10 neurinspectre CLI (that binary cannot import lief 0.9.0).",
            "Full DOS + overlay/padding. No GAMMA. Parse gate is pefile + e_lfanew + EP.",
        ],
    }
    out = os.path.join(args.output_dir, "audit_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(
        "[lief090] wrote %s n=%d problem_valid_asr=%.4f official_reproduction=True"
        % (out, n, report["feature_vs_problem_space"]["problem_space_valid_asr"]),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
