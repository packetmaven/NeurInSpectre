"""Tests for C7: section-slack padding + Capa supplement index + CLI.

Coverage:

- ``_pe_section_slack`` / ``section_slack_capacity`` / ``apply_section_slack_pad``
  on the minimal-PE fixture: correct slack window, payload truncation to
  fit, file length preserved, non-slack bytes untouched.
- ``evaluate_transform_validity(kind="section_slack")`` accepts a slack
  write and rejects a file-length change.
- ``ProblemSpacePESearch`` with ``enable_section_slack=True`` records
  ``n_section_slack_attempts`` and ``section_slack_capacity`` in its
  result. Gate remains no-op-safe when the fixture has no slack.
- ``capa_supplement_index.build_index`` and ``lookup`` on a synthetic
  one-shard zip.
- Click subcommands ``index-capa-supplement`` and ``lookup-capa-functions``
  are registered and produce expected output.
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from neurinspectre.attacks.problem_space_pe import ProblemSpacePESearch
from neurinspectre.malware.pe_transforms import (
    apply_section_slack_pad,
    evaluate_transform_validity,
    _pe_section_slack,
    section_slack_capacity,
)
from neurinspectre.malware.capa_supplement_index import (
    build_index,
    load_index,
    lookup,
    summarize,
)


def _pe_with_slack(virtual_size=0x50, size_of_raw=0x200) -> bytes:
    """Build a fixture PE whose last section has (size_of_raw - virtual_size)
    bytes of file-alignment slack."""
    dos = bytearray(64); dos[0:2] = b"MZ"; struct.pack_into("<I", dos, 0x3C, 64)
    pe = bytearray(); pe += b"PE\x00\x00"
    pe += struct.pack("<HHIIIHH", 0x14C, 1, 0, 0, 0, 0xE0, 0x0102)
    opt = bytearray(224)
    struct.pack_into("<H", opt, 0, 0x10B); struct.pack_into("<I", opt, 16, 0x1000)
    struct.pack_into("<I", opt, 28, 0x400000); struct.pack_into("<I", opt, 32, 0x1000)
    struct.pack_into("<I", opt, 36, 0x200); struct.pack_into("<H", opt, 40, 4)
    struct.pack_into("<H", opt, 42, 0); struct.pack_into("<H", opt, 48, 4)
    struct.pack_into("<I", opt, 56, 0x2000); struct.pack_into("<I", opt, 60, 0x200)
    struct.pack_into("<H", opt, 68, 3); struct.pack_into("<H", opt, 92, 16); pe += opt
    # Section header
    sec = bytearray(40); sec[0:5] = b".text"
    struct.pack_into("<I", sec, 8, virtual_size)                 # VirtualSize
    struct.pack_into("<I", sec, 12, 0x1000)                      # VirtualAddress
    struct.pack_into("<I", sec, 16, size_of_raw)                 # SizeOfRawData
    struct.pack_into("<I", sec, 20, 0x200)                       # PointerToRawData
    struct.pack_into("<I", sec, 36, 0x60000020)
    pe += sec
    return (bytes(dos) + bytes(pe)).ljust(0x200 + size_of_raw, b"\x00")


# ---------------------------------------------------------------------------
# section_slack transform
# ---------------------------------------------------------------------------


def test_pe_section_slack_reports_window():
    b = _pe_with_slack(virtual_size=0x50, size_of_raw=0x200)
    start, content_end, file_end = _pe_section_slack(b, -1)
    assert start == 0x200
    assert content_end == 0x250
    assert file_end == 0x400
    assert section_slack_capacity(b, -1) == 0x1B0


def test_pe_section_slack_virtual_size_zero_has_no_slack():
    b = _pe_with_slack(virtual_size=0, size_of_raw=0x200)
    with pytest.raises(ValueError, match="VirtualSize=0"):
        _pe_section_slack(b, -1)
    assert section_slack_capacity(b, -1) == 0
    with pytest.raises(ValueError):
        apply_section_slack_pad(b, b"abc")


def test_evaluate_transform_validity_rejects_write_outside_slack_window():
    original = _pe_with_slack(virtual_size=0x50, size_of_raw=0x200)
    mutated = bytearray(original)
    mutated[0x200] = 0x41  # inside mapped content, before the slack window
    gate = evaluate_transform_validity(original, bytes(mutated), kind="section_slack")
    assert gate["passed"] is False
    assert "bytes_outside_slack_rewritten" in gate["reasons"]


def test_pe_section_slack_no_slack_raises():
    b = _pe_with_slack(virtual_size=0x200, size_of_raw=0x200)  # content fills the section
    with pytest.raises(ValueError):
        _pe_section_slack(b, -1)
    assert section_slack_capacity(b, -1) == 0


def test_apply_section_slack_pad_writes_only_into_slack():
    b = _pe_with_slack(virtual_size=0x50, size_of_raw=0x200)
    payload = b"X" * 32
    mutated = apply_section_slack_pad(b, payload)
    # File length unchanged
    assert len(mutated) == len(b)
    # Content window unchanged
    assert mutated[:0x250] == b[:0x250]
    # Slack head now equals our payload
    assert mutated[0x250 : 0x250 + 32] == payload
    # Rest of slack still zeros
    assert mutated[0x250 + 32 : 0x400] == b"\x00" * (0x400 - 0x250 - 32)


def test_apply_section_slack_pad_truncates_to_slack():
    b = _pe_with_slack(virtual_size=0x50, size_of_raw=0x200)
    payload = b"Y" * 0x10000  # far bigger than slack
    mutated = apply_section_slack_pad(b, payload)
    assert len(mutated) == len(b)
    # Whole slack window is Ys
    assert mutated[0x250 : 0x400] == b"Y" * (0x400 - 0x250)


def test_apply_section_slack_pad_raises_when_no_slack():
    b = _pe_with_slack(virtual_size=0x200, size_of_raw=0x200)
    with pytest.raises(ValueError):
        apply_section_slack_pad(b, b"abc")


def test_evaluate_transform_validity_accepts_section_slack_write():
    original = _pe_with_slack(virtual_size=0x50, size_of_raw=0x200)
    mutated = apply_section_slack_pad(original, b"Z" * 8)
    gate = evaluate_transform_validity(original, mutated, kind="section_slack")
    assert gate["passed"] is True, gate
    assert gate["size_delta"] == 0
    assert gate["reasons"] == []


def test_evaluate_transform_validity_rejects_length_change_for_section_slack():
    original = _pe_with_slack(virtual_size=0x50, size_of_raw=0x200)
    mutated = original + b"\x00" * 16  # length changed = disallowed for section_slack
    gate = evaluate_transform_validity(original, mutated, kind="section_slack")
    assert gate["passed"] is False
    assert "file_length_changed" in gate["reasons"]


# ---------------------------------------------------------------------------
# ProblemSpacePESearch integration
# ---------------------------------------------------------------------------


class _FakeGBDT(nn.Module):
    feature_dim = 2568

    def __init__(self, p=0.9):
        super().__init__()
        self._p = float(p)

    def predict_proba(self, x):
        n = x.shape[0]
        p1 = np.full(n, self._p, dtype=np.float32)
        return np.stack([1 - p1, p1], axis=1)

    def forward(self, x):
        probs = self.predict_proba(x.detach().cpu().numpy())
        s = np.log(probs[:, 1]) - np.log(probs[:, 0])
        return torch.as_tensor(np.stack([-s, s], axis=1), device=x.device, dtype=x.dtype)


def _fake_extract(_bytes):
    vec = np.zeros(2568, dtype=np.float32)
    return {"available": True, "features": vec, "reasons": [], "dim": 2568,
            "extractor": {"available": True, "reasons": []}}


def test_problem_space_search_reports_section_slack_provenance():
    b = _pe_with_slack(virtual_size=0x50, size_of_raw=0x200)
    search = ProblemSpacePESearch(
        _FakeGBDT(), n_queries=20, payload_size=32, seed=1,
        extractor=_fake_extract, enable_section_slack=True,
    )
    result = search.run_bytes(b, y=1)
    assert result["enable_section_slack"] is True
    assert result["section_slack_capacity"] == 0x1B0
    # With enable_section_slack + slack_cap>0, at least one attempt should have happened.
    assert result["n_section_slack_attempts"] > 0
    assert result["n_section_slack_no_capacity"] == 0


def test_problem_space_search_disabled_by_default():
    b = _pe_with_slack()
    search = ProblemSpacePESearch(
        _FakeGBDT(), n_queries=10, payload_size=32, seed=2,
        extractor=_fake_extract,
    )
    result = search.run_bytes(b, y=1)
    assert result["enable_section_slack"] is False
    assert result["n_section_slack_attempts"] == 0


# ---------------------------------------------------------------------------
# Capa supplement index
# ---------------------------------------------------------------------------


def _make_synthetic_shard(tmp_path: Path) -> Path:
    """Build a tiny 1-shard supplement zip mirroring the real schema."""
    shard = tmp_path / "shard.zip"
    records = [
        {"sha256": "aa" * 32, "func_addr": "0x1000", "capa": ["Read file on windows"],
         "bytes": "cc" * 12, "disasm": ["ret"]},
        {"sha256": "aa" * 32, "func_addr": "0x2000", "capa": ["Terminate process"],
         "bytes": "cc" * 20, "disasm": ["ret"]},
        {"sha256": "bb" * 32, "func_addr": "0x1500",
         "capa": ["Read file on windows", "Encode data using xor"],
         "bytes": "cc" * 8, "disasm": ["ret"]},
    ]
    with zipfile.ZipFile(shard, "w") as zf:
        with zf.open("records.json", "w") as fh:
            for r in records:
                fh.write((json.dumps(r) + "\n").encode())
    return shard


def test_build_index_and_lookup(tmp_path):
    _make_synthetic_shard(tmp_path)
    idx = build_index(tmp_path)
    assert set(idx) == {"aa" * 32, "bb" * 32}
    funcs_a = lookup(idx, "AA" * 32)  # case-insensitive
    assert len(funcs_a) == 2
    assert funcs_a[0]["capa"] == ["Read file on windows"]
    assert funcs_a[0]["byte_len"] == 12
    summ = summarize(idx)
    assert summ["n_files"] == 2
    assert summ["n_functions"] == 3
    assert summ["n_unique_capabilities"] == 3
    caps = dict(summ["top_capabilities"])
    assert caps["Read file on windows"] == 2


def test_load_index_roundtrip(tmp_path):
    _make_synthetic_shard(tmp_path)
    idx = build_index(tmp_path)
    p = tmp_path / "index.json"
    p.write_text(json.dumps(idx))
    idx2 = load_index(p)
    assert idx == idx2


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


def test_index_and_lookup_cli(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from neurinspectre.cli.main import cli, _CLICK_COMMANDS

    assert "index-capa-supplement" in _CLICK_COMMANDS
    assert "lookup-capa-functions" in _CLICK_COMMANDS

    supplement = tmp_path / "supplement"
    supplement.mkdir()
    _make_synthetic_shard(supplement)
    out = tmp_path / "idx.json"

    runner = CliRunner()
    res = runner.invoke(cli, ["index-capa-supplement",
                              "--supplement", str(supplement),
                              "--output", str(out)])
    assert res.exit_code == 0, res.output
    assert out.is_file()

    res = runner.invoke(cli, ["lookup-capa-functions", "aa" * 32,
                              "--index", str(out)])
    assert res.exit_code == 0, res.output
    assert "n_functions" in res.output
    assert "Read file on windows" in res.output

    # non-existent sha -> friendly not-found message
    res = runner.invoke(cli, ["lookup-capa-functions", "00" * 32,
                              "--index", str(out)])
    assert res.exit_code == 0
    assert "no supplement entries" in res.output
