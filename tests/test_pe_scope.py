"""PE corpus scope preflight (challenge + supplement overlap)."""

import json
import struct
from pathlib import Path

from neurinspectre.malware.pe_scope import challenge_sha_set, scope_pe_corpus


def _tiny_pe() -> bytes:
    dos = bytearray(64)
    dos[0:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 64)
    pe = bytearray()
    pe += b"PE\x00\x00"
    pe += struct.pack("<HHIIIHH", 0x14C, 1, 0, 0, 0, 0xE0, 0x0102)
    opt = bytearray(224)
    struct.pack_into("<H", opt, 0, 0x10B)
    struct.pack_into("<I", opt, 16, 0x1000)
    struct.pack_into("<I", opt, 28, 0x400000)
    struct.pack_into("<I", opt, 32, 0x1000)
    struct.pack_into("<I", opt, 36, 0x200)
    struct.pack_into("<H", opt, 40, 4)
    struct.pack_into("<H", opt, 42, 0)
    struct.pack_into("<H", opt, 48, 4)
    struct.pack_into("<I", opt, 56, 0x2000)
    struct.pack_into("<I", opt, 60, 0x200)
    struct.pack_into("<H", opt, 68, 3)
    struct.pack_into("<H", opt, 92, 16)
    pe += opt
    sec = bytearray(40)
    sec[0:5] = b".text"
    struct.pack_into("<I", sec, 8, 0x50)
    struct.pack_into("<I", sec, 12, 0x1000)
    struct.pack_into("<I", sec, 16, 0x200)
    struct.pack_into("<I", sec, 20, 0x200)
    struct.pack_into("<I", sec, 36, 0x60000020)
    pe += sec
    return (bytes(dos) + bytes(pe)).ljust(0x400, b"\x00")


def test_challenge_sha_set_reads_jsonl(tmp_path):
    sha = "a" * 64
    (tmp_path / "train.jsonl").write_text(json.dumps({"sha256": sha}) + "\n")
    assert challenge_sha_set(tmp_path) == {sha}


def test_scope_pe_corpus_counts_mz(tmp_path):
    pe = tmp_path / "sample.exe"
    pe.write_bytes(_tiny_pe())
    (tmp_path / "not_pe.txt").write_text("hello")
    out = scope_pe_corpus(tmp_path)
    assert out["n_mz_files"] == 1
    assert len(out["files"]) == 1
    assert len(out["files"][0]["sha256"]) == 64
    pre = out.get("measurement_scope_preflight") or {}
    assert pre.get("not_measured_ids")
    assert "sandbox_execution" in pre["not_measured_ids"]
