"""Tests for C8: TTP-aware capability preservation modes.

Coverage:

- ``ProblemSpacePESearch(capa_preserve_mode=...)`` accepts all/ttps/mbc,
  rejects other values.
- The ``capa_preserve_mode`` value is stamped into the search result JSON.
- ``rules_with_attack_metadata`` / ``rules_with_mbc_metadata`` /
  ``restrict_to_ttps`` / ``restrict_to_mbc`` return sensible sets when
  called with a live capa-rules directory (skipped if unavailable).
- Audit CLI accepts ``--capa-preserve-mode`` and propagates the value.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from neurinspectre.attacks.problem_space_pe import ProblemSpacePESearch


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


def _fake_extract(_bytes):
    vec = np.zeros(2568, dtype=np.float32)
    return {"available": True, "features": vec, "reasons": [], "dim": 2568,
            "extractor": {"available": True, "reasons": []}}


def test_search_accepts_valid_modes():
    for mode in ("all", "ttps", "mbc"):
        s = ProblemSpacePESearch(_FakeGBDT(), extractor=_fake_extract,
                                 capa_preserve=False, capa_preserve_mode=mode)
        assert s.capa_preserve_mode == mode


def test_search_rejects_invalid_mode():
    with pytest.raises(ValueError, match="capa_preserve_mode"):
        ProblemSpacePESearch(_FakeGBDT(), extractor=_fake_extract,
                             capa_preserve_mode="mitre")


def test_search_result_records_mode_when_enabled(monkeypatch, tmp_path):
    """With capa_preserve=True + a bad rules dir, mode still stamped."""
    monkeypatch.setenv("CAPA_RULES", str(tmp_path / "nope"))
    search = ProblemSpacePESearch(
        _FakeGBDT(), n_queries=5, seed=1, extractor=_fake_extract,
        capa_preserve=True, capa_preserve_mode="ttps",
        capa_rules_dir=tmp_path / "nope",
    )
    result = search.run_bytes(_pe(), y=1)
    assert result["capa_preserve"] is True
    assert result["capa_preserve_mode"] == "ttps"
    assert result["capa_baseline_available"] is False


def test_search_mode_is_none_when_gate_off():
    search = ProblemSpacePESearch(
        _FakeGBDT(), n_queries=5, seed=1, extractor=_fake_extract,
    )
    result = search.run_bytes(_pe(), y=1)
    assert result["capa_preserve"] is False
    assert result["capa_preserve_mode"] is None


# ---------------------------------------------------------------------------
# rule-metadata helpers (require capa-rules on disk)
# ---------------------------------------------------------------------------


def _rules_available():
    from neurinspectre.malware.capa_scan import DEFAULT_RULES_DIR
    return DEFAULT_RULES_DIR.is_dir()


@pytest.mark.skipif(not _rules_available(), reason="capa-rules not present")
def test_rules_with_attack_metadata_is_nonempty_and_smaller_than_all():
    from neurinspectre.malware.capa_scan import (
        rules_with_attack_metadata, rules_with_mbc_metadata,
    )
    ttps = rules_with_attack_metadata()
    mbc = rules_with_mbc_metadata()
    assert 0 < len(ttps) < 5000
    assert 0 < len(mbc) < 5000


@pytest.mark.skipif(not _rules_available(), reason="capa-rules not present")
def test_restrict_helpers_are_intersections():
    from neurinspectre.malware.capa_scan import (
        rules_with_attack_metadata, rules_with_mbc_metadata,
        restrict_to_ttps, restrict_to_mbc,
    )
    ttps = rules_with_attack_metadata()
    mbc = rules_with_mbc_metadata()
    # Restricting an arbitrary set should equal the plain set intersection.
    caps = frozenset(list(ttps)[:3] + ["nonexistent rule"])
    r = restrict_to_ttps(caps)
    assert r == caps & ttps
    r = restrict_to_mbc(caps)
    assert r == caps & mbc


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_audit_cli_accepts_capa_preserve_mode():
    from click.testing import CliRunner
    from neurinspectre.cli.main import cli
    runner = CliRunner()
    help_text = runner.invoke(cli, ["audit", "--help"]).output
    assert "--capa-preserve-mode" in help_text
    assert "ttps" in help_text
    assert "mbc" in help_text
