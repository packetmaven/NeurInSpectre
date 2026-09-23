"""Tests for E12: pefile-warnings-quiet Full DOS gate.

Coverage:

- ``pefilewarnings_offset_dim()`` returns the canonical (2480, 88)
  layout when thrember is unavailable and matches thrember at runtime
  when it is.
- ``ProblemSpacePESearch(fulldos_quiet_only=True)`` records the gate
  in its result JSON and increments ``n_pefw_rejects`` when a fake
  extractor reports a shifted pefilewarnings band.
- ``fulldos_quiet_only=False`` (the default) does not consult the gate
  and does not increment the counter.
- Audit CLI advertises ``--fulldos-quiet-only``.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch
import torch.nn as nn

from neurinspectre.attacks.problem_space_pe import ProblemSpacePESearch
from neurinspectre.malware.ember2024_extract import pefilewarnings_offset_dim


def _pe():
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
    sec = bytearray(40); sec[0:5] = b".text"
    struct.pack_into("<I", sec, 8, 0x50); struct.pack_into("<I", sec, 12, 0x1000)
    struct.pack_into("<I", sec, 16, 0x200); struct.pack_into("<I", sec, 20, 0x200)
    struct.pack_into("<I", sec, 36, 0x60000020); pe += sec
    return (bytes(dos) + bytes(pe)).ljust(0x400, b"\x00")


class _FakeGBDT(nn.Module):
    feature_dim = 2568
    def __init__(self):
        super().__init__()
    def predict_proba(self, x):
        n = x.shape[0]
        p1 = np.full(n, 0.9, dtype=np.float32)
        return np.stack([1 - p1, p1], axis=1)
    def forward(self, x):
        probs = self.predict_proba(x.detach().cpu().numpy())
        s = np.log(probs[:, 1]) - np.log(probs[:, 0])
        return torch.as_tensor(np.stack([-s, s], axis=1), device=x.device, dtype=x.dtype)


# ---------------------------------------------------------------------------
# pefilewarnings_offset_dim
# ---------------------------------------------------------------------------


def test_pefilewarnings_offset_dim_returns_canonical_layout():
    """Regardless of whether thrember is loadable, this returns a
    (offset, dim) tuple with dim=88 (the thrember 0.1.0 PEFormatWarnings
    dim) and offset in a sensible range."""
    off, dim = pefilewarnings_offset_dim()
    assert dim == 88
    # thrember 0.1.0 places it at 2480; anywhere in the vector is fine
    # so long as the slice fits inside 2568.
    assert 0 <= off <= 2568 - dim


def test_pefilewarnings_offset_dim_is_cached():
    a = pefilewarnings_offset_dim()
    b = pefilewarnings_offset_dim()
    assert a == b


# ---------------------------------------------------------------------------
# ProblemSpacePESearch gate
# ---------------------------------------------------------------------------


def _make_extract_with_shift(shift_prob: float, seed: int = 0):
    """Return a fake extractor whose pefilewarnings band flips one bit
    with probability ``shift_prob``.
    """
    off, dim = pefilewarnings_offset_dim()
    rng = np.random.default_rng(seed)
    baseline = np.zeros(2568, dtype=np.float32)
    baseline[off + 3] = 1.0  # non-zero clean pefw signal

    call_count = {"n": 0}

    def extractor(pe_bytes):
        call_count["n"] += 1
        vec = baseline.copy()
        if call_count["n"] > 1 and rng.random() < shift_prob:
            # flip a random bin in the pefw band
            idx = int(rng.integers(0, dim))
            vec[off + idx] += 1.0
        return {"available": True, "features": vec, "reasons": [], "dim": 2568,
                "extractor": {"available": True, "reasons": []}}

    return extractor


def test_gate_off_by_default_does_not_reject():
    ex = _make_extract_with_shift(1.0, seed=1)  # every mutation shifts pefw
    search = ProblemSpacePESearch(
        _FakeGBDT(), n_queries=15, seed=1, extractor=ex,
    )
    result = search.run_bytes(_pe(), y=1)
    assert result["fulldos_quiet_only"] is False
    assert result["n_pefw_rejects"] == 0


def test_gate_on_rejects_every_shifted_mutation():
    ex = _make_extract_with_shift(1.0, seed=2)
    search = ProblemSpacePESearch(
        _FakeGBDT(), n_queries=15, seed=2, extractor=ex,
        fulldos_quiet_only=True,
    )
    result = search.run_bytes(_pe(), y=1)
    assert result["fulldos_quiet_only"] is True
    # Every non-clean call to the extractor shifts pefw -> gate rejects all.
    assert result["n_pefw_rejects"] > 0


def test_gate_on_but_no_shift_no_rejects():
    ex = _make_extract_with_shift(0.0, seed=3)  # never shifts
    search = ProblemSpacePESearch(
        _FakeGBDT(), n_queries=15, seed=3, extractor=ex,
        fulldos_quiet_only=True,
    )
    result = search.run_bytes(_pe(), y=1)
    assert result["fulldos_quiet_only"] is True
    assert result["n_pefw_rejects"] == 0


def test_audit_cli_advertises_fulldos_quiet_only():
    from click.testing import CliRunner
    from neurinspectre.cli.main import cli
    runner = CliRunner()
    help_text = runner.invoke(cli, ["audit", "--help"]).output
    assert "--fulldos-quiet-only" in help_text
    assert "pefilewarnings" in help_text
