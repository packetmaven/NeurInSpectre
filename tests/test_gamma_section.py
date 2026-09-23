"""GAMMA section injection (secml-malware optional extra)."""

import struct

import pytest

from neurinspectre.malware.gamma_env import gamma_readiness, lief_policy_report
from neurinspectre.malware.gamma_section import gamma_secml_status
from neurinspectre.malware.pe_transforms import evaluate_transform_validity


def _minimal_pe() -> bytes:
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


def test_gamma_validity_gate_allows_section_growth():
    orig = _minimal_pe()
    overlay = evaluate_transform_validity(orig, orig + b"\x00" * 64, kind="padding")
    assert overlay.get("passed") is True
    gamma_kind = evaluate_transform_validity(orig, orig + b"X" * 128, kind="gamma_section")
    assert "section_count_changed" not in (gamma_kind.get("reasons") or [])


def test_lief_policy_documents_conflict():
    rep = lief_policy_report()
    assert rep.get("conflict_note")
    assert rep.get("recommended_gamma_audit_target") == "ember2024-gbdt"


def test_measurement_scope_moves_gamma_when_enabled():
    from neurinspectre.malware.measurement_scope import build_measurement_scope

    off = build_measurement_scope("ember2024-gbdt", gamma_sections_enabled=False)
    on = build_measurement_scope("ember2024-gbdt", gamma_sections_enabled=True)
    assert "gamma_section_injection" in off["not_measured_ids"]
    assert "gamma_section_injection" not in on["not_measured_ids"]
    assert "gamma_section_injection_secml" in on["measured"]


@pytest.mark.skipif(
    not gamma_secml_status().get("available"),
    reason="secml-malware not installed (pip install -e '.[gamma]')",
)
def test_secml_smoke_inject_and_validity():
    from neurinspectre.malware.gamma_section import smoke_gamma_section_inject

    rep = smoke_gamma_section_inject()
    assert rep.get("ok") is True, rep
    assert rep.get("size_after", 0) > rep.get("size_before", 0)


@pytest.mark.skipif(
    not gamma_secml_status().get("available"),
    reason="secml-malware not installed",
)
def test_gamma_readiness_reports_ready():
    rep = gamma_readiness(run_smoke_inject=True)
    assert rep.get("ready") is True, rep


def test_audit_gamma_flag_requires_secml():
    from click.testing import CliRunner
    from neurinspectre.cli.main import cli

    if gamma_secml_status().get("available"):
        pytest.skip("secml present; use test_audit_gamma_flag_wiring instead")
    r = CliRunner().invoke(
        cli,
        [
            "audit",
            "--target",
            "ember2024-gbdt",
            "--pe-sample",
            "/tmp/pe",
            "--enable-gamma-sections",
            "--smoke",
        ],
    )
    assert r.exit_code != 0
    assert "gamma" in (r.output or "").lower() or "secml" in (r.output or "").lower()


@pytest.mark.skipif(
    not gamma_secml_status().get("available"),
    reason="secml-malware not installed",
)
def test_problem_space_search_gamma_mode_runs():
    import numpy as np
    import torch
    import torch.nn as nn

    from neurinspectre.attacks.problem_space_pe import ProblemSpacePESearch

    class _Scorer(nn.Module):
        def forward(self, x):
            return torch.stack([torch.zeros(x.size(0)), torch.ones(x.size(0))], dim=1)

    pe = _minimal_pe()
    # Fabricate extractor that returns constant features
    def _fake_extract(_b):
        return {"features": np.zeros(16, dtype=np.float32), "reasons": []}

    search = ProblemSpacePESearch(
        _Scorer(),
        n_queries=5,
        seed=1,
        extractor=_fake_extract,
        enable_gamma_sections=True,
    )
    out = search.run_bytes(pe, y=1)
    assert out.get("gamma_section_enabled") is True
    assert int(out.get("n_gamma_attempts") or 0) >= 0
