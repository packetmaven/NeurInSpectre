"""Tests for D10: combined multi-region transform (AdvMal-TF / PhantomCall
byte-space envelope cross-check).

Coverage:

- ``apply_combined_multi_region`` produces a PE that:
    * grows by the overlay portion (length delta > 0),
    * still starts with MZ,
    * still has an unchanged ``e_lfanew``.
- ``evaluate_transform_validity(kind="combined_multi_region")`` accepts
  a legitimate combined mutation and rejects a shrunk one.
- ``ProblemSpacePESearch(transform_set="combined")`` implies
  ``enable_section_slack=True``, every attempt records
  ``combined_multi_region``, and the result stamps ``transform_set`` +
  ``n_combined_attempts``.
- ``ProblemSpacePESearch`` raises ``ValueError`` on invalid
  ``transform_set``.
- Audit CLI advertises ``--transform-set`` with correct choices.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch
import torch.nn as nn

from neurinspectre.attacks.problem_space_pe import ProblemSpacePESearch
from neurinspectre.malware.pe_transforms import (
    apply_combined_multi_region,
    evaluate_transform_validity,
    read_e_lfanew,
)


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


def _pe_no_slack():
    """Same fixture with VirtualSize filling SizeOfRawData, so slack is empty."""
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
    struct.pack_into("<I", sec, 8, 0x200); struct.pack_into("<I", sec, 12, 0x1000)
    struct.pack_into("<I", sec, 16, 0x200); struct.pack_into("<I", sec, 20, 0x200)
    struct.pack_into("<I", sec, 36, 0x60000020); pe += sec
    return (bytes(dos) + bytes(pe)).ljust(0x400, b"\x00")


class _FakeGBDT(nn.Module):
    feature_dim = 2568
    def __init__(self): super().__init__()
    def predict_proba(self, x):
        n = x.shape[0]
        p1 = np.full(n, 0.9, dtype=np.float32)
        return np.stack([1 - p1, p1], axis=1)
    def forward(self, x):
        probs = self.predict_proba(x.detach().cpu().numpy())
        s = np.log(probs[:, 1]) - np.log(probs[:, 0])
        return torch.as_tensor(np.stack([-s, s], axis=1), device=x.device, dtype=x.dtype)


def _fake_extract(_bytes):
    return {"available": True, "features": np.zeros(2568, dtype=np.float32),
            "reasons": [], "dim": 2568, "extractor": {"available": True, "reasons": []}}


# ---------------------------------------------------------------------------
# apply_combined_multi_region
# ---------------------------------------------------------------------------


def test_combined_grows_by_overlay_and_preserves_mz_and_elfanew():
    b = _pe()
    mut = apply_combined_multi_region(b, b"A" * 32, b"B" * 32, b"C" * 128)
    assert set(mut.regions) == {"section_slack", "fulldos", "overlay"}
    assert mut.data[:2] == b"MZ"
    assert read_e_lfanew(mut.data) == read_e_lfanew(b)
    assert len(mut.data) == len(b) + 128


def test_combined_writes_into_all_three_regions():
    b = _pe()
    mut = apply_combined_multi_region(b, b"A" * 32, b"B" * 32, b"C" * 16)
    assert set(mut.regions) == {"section_slack", "fulldos", "overlay"}
    # Full DOS region [2, 0x3C) should now begin with A (as much as fits)
    assert mut.data[2:34].startswith(b"A" * 32)
    # Overlay is at end of file
    assert mut.data[-16:] == b"C" * 16
    # Section slack should contain some B's (starts at 0x250 per fixture)
    assert b"B" * 16 in mut.data[0x250:0x400]


def test_combined_omits_slack_when_virtual_size_fills_the_section():
    b = _pe_no_slack()
    mut = apply_combined_multi_region(b, b"A" * 32, b"B" * 32, b"C" * 16)
    assert "section_slack" not in mut.regions
    assert "fulldos" in mut.regions
    assert "overlay" in mut.regions


def test_evaluate_transform_validity_accepts_combined():
    b = _pe()
    mut = apply_combined_multi_region(b, b"A" * 32, b"B" * 32, b"C" * 128)
    gate = evaluate_transform_validity(b, mut.data, kind="combined_multi_region")
    assert gate["passed"] is True, gate
    assert gate["reasons"] == []
    assert gate["size_delta"] == 128


def test_evaluate_transform_validity_rejects_shrunk_combined():
    b = _pe()
    # Fake a shrunk mutation (should never happen legitimately)
    mut = b[:-16]
    gate = evaluate_transform_validity(b, mut, kind="combined_multi_region")
    assert gate["passed"] is False
    assert "file_shrunk" in gate["reasons"]


# ---------------------------------------------------------------------------
# ProblemSpacePESearch integration
# ---------------------------------------------------------------------------


def test_combined_transform_set_implies_section_slack_and_records_provenance():
    search = ProblemSpacePESearch(
        _FakeGBDT(), n_queries=20, seed=1, extractor=_fake_extract,
        transform_set="combined",
    )
    # combined should force section slack on
    assert search.enable_section_slack is True
    assert search.transform_set == "combined"
    result = search.run_bytes(_pe(), y=1)
    assert result["transform_set"] == "combined"
    assert result["n_combined_attempts"] > 0
    # Every candidate should be combined; section-slack single-mode counter
    # is not incremented in combined mode.
    assert result["n_section_slack_attempts"] == 0


def _extract_drop_on_change(original: bytes):
    def extract(blob: bytes):
        p = 0.1 if blob != original else 0.9
        feat = np.zeros(2568, dtype=np.float32)
        feat[0] = p
        return {
            "available": True,
            "features": feat,
            "reasons": [],
            "dim": 2568,
            "extractor": {"available": True, "reasons": []},
        }
    return extract


class _ScoreFeature0(_FakeGBDT):
    def predict_proba(self, x):
        p1 = np.clip(np.asarray(x[:, 0], dtype=np.float32), 1e-6, 1 - 1e-6)
        return np.stack([1 - p1, p1], axis=1)


def test_complete_combined_is_the_chosen_attack():
    pe = _pe()
    search = ProblemSpacePESearch(
        _ScoreFeature0(), n_queries=6, seed=1, extractor=_extract_drop_on_change(pe),
        transform_set="combined",
    )
    result = search.run_bytes(pe, y=1)
    assert result["chosen_attack"] == "combined_multi_region"
    assert result["success"] is True


def test_incomplete_combined_is_not_the_chosen_attack():
    pe = _pe_no_slack()
    search = ProblemSpacePESearch(
        _ScoreFeature0(), n_queries=8, seed=3, extractor=_extract_drop_on_change(pe),
        transform_set="combined",
    )
    result = search.run_bytes(pe, y=1)
    assert result["n_combined_attempts"] > 0
    assert result["chosen_attack"] == "clean"
    assert result["success"] is False


def test_run_bytes_counters_are_per_file_not_a_running_total():
    search = ProblemSpacePESearch(
        _FakeGBDT(), n_queries=6, seed=1, extractor=_fake_extract,
        transform_set="combined",
    )
    first = search.run_bytes(_pe(), y=1)
    search.n_capa_calls = 1000
    search.n_capa_rejects = 1000
    search.n_pefw_rejects = 1000
    search.n_section_slack_attempts = 1000
    search.n_combined_attempts = 1000
    second = search.run_bytes(_pe(), y=1)
    assert first["n_combined_attempts"] > 0
    assert second["n_combined_attempts"] == first["n_combined_attempts"]
    assert second["n_capa_calls"] == 0
    assert second["n_capa_rejects"] == 0
    assert second["n_pefw_rejects"] == 0
    assert second["n_section_slack_attempts"] == 0


def test_default_transform_set_does_not_use_combined():
    search = ProblemSpacePESearch(
        _FakeGBDT(), n_queries=10, seed=2, extractor=_fake_extract,
    )
    result = search.run_bytes(_pe(), y=1)
    assert result["transform_set"] == "default"
    assert result["n_combined_attempts"] == 0


def test_invalid_transform_set_raises():
    with pytest.raises(ValueError, match="transform_set"):
        ProblemSpacePESearch(_FakeGBDT(), transform_set="phantomcall")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_audit_cli_advertises_transform_set():
    from click.testing import CliRunner
    from neurinspectre.cli.main import cli
    runner = CliRunner()
    help_text = runner.invoke(cli, ["audit", "--help"]).output
    assert "--transform-set" in help_text
    assert "combined" in help_text
