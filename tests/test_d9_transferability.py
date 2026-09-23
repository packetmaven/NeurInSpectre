"""Tests for D9: cross-model transferability re-scoring.

Coverage:

- ``ProblemSpacePESearch`` includes ``best_bytes`` in its result on the
  non-flip path (record the best-of-search candidate for downstream
  transfer).
- ``score_transferability`` on a synthetic report + fake models produces
  the expected per-sample rows and summary metrics (per-model flip rate,
  any-model flip rate, all-model flip rate).
- ``neurinspectre transferability`` CLI subcommand accepts repeated
  ``-m name=path`` pairs and writes the JSON to ``--output`` when given.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from neurinspectre.attacks.problem_space_pe import ProblemSpacePESearch
from neurinspectre.evaluation.transferability import score_transferability


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
    return {"available": True, "features": np.zeros(2568, dtype=np.float32),
            "reasons": [], "dim": 2568, "extractor": {"available": True, "reasons": []}}


def test_search_returns_best_bytes_key():
    search = ProblemSpacePESearch(_FakeGBDT(), n_queries=6, seed=1,
                                  extractor=_fake_extract)
    result = search.run_bytes(_pe(), y=1)
    assert "best_bytes" in result
    assert isinstance(result["best_bytes"], (bytes, bytearray))
    assert len(result["best_bytes"]) > 0


# ---------------------------------------------------------------------------
# score_transferability
# ---------------------------------------------------------------------------


class _FakeSubModel:
    """Model that reports p_malware = self._p on any input."""
    def __init__(self, p, feature_dim=2568):
        self._p = float(p)
        self.feature_dim = int(feature_dim)
        self.classifier_name = ""
    def predict_proba(self, x):
        n = x.shape[0]
        p1 = np.full(n, self._p, dtype=np.float32)
        return np.stack([1 - p1, p1], axis=1)


def _fake_extract_2024(_bytes):
    """Match extract_ember2024_features's return contract."""
    return {"available": True, "features": np.zeros(2568, dtype=np.float32),
            "reasons": [], "dim": 2568}


def _write_pair(directory: Path, name: str, clean: bytes, mutated: bytes):
    clean_path = directory / f"{name}.clean"
    mut_path = directory / f"{name}.mutated.bin"
    clean_path.write_bytes(clean)
    mut_path.write_bytes(mutated)
    return clean_path, mut_path


def test_score_transferability_counts_only_threshold_crossings(tmp_path, monkeypatch):
    """Transfer requires clean p >= 0.5, changed bytes, and mutated p < 0.5.

    A model that scores every input at 0.4 is a baseline miss, not a finding.
    A model that scores clean bytes high and mutated bytes low is a transfer.
    Feature[0] carries that signal: 0 on clean bytes, 1 on mutated bytes.
    """
    best_bytes_dir = tmp_path / "best_bytes"
    best_bytes_dir.mkdir()
    clean_a, mut_a = _write_pair(best_bytes_dir, "aa", b"MZ" + b"\x00" * 62, b"MZ" + b"\x01" * 62)
    clean_b, mut_b = _write_pair(best_bytes_dir, "bb", b"MZ" + b"\x00" * 62, b"MZ" + b"\x02" * 62)
    report = {
        "same_sample_detail": {
            "best_bytes_manifest": [
                {"sample_path": str(clean_a), "sha256_original": "aa", "sha256_mutated": "aa1",
                 "path": str(mut_a), "size": 64,
                 "chosen_attack": "fulldos", "best_p_malware": 0.4,
                 "identical_to_original": False},
                {"sample_path": str(clean_b), "sha256_original": "bb", "sha256_mutated": "bb1",
                 "path": str(mut_b), "size": 64,
                 "chosen_attack": "padding", "best_p_malware": 0.99,
                 "identical_to_original": False},
            ],
        }
    }
    report_path = tmp_path / "audit_report.json"
    report_path.write_text(json.dumps(report))

    from neurinspectre.evaluation import transferability

    def _extract(data):
        flag = 1.0 if bytes(data)[2:3] != b"\x00" else 0.0
        vec = np.zeros(2568, dtype=np.float32)
        vec[0] = flag
        return {"available": True, "features": vec, "reasons": [], "dim": 2568}

    class _CrossingModel:
        feature_dim = 2568
        def __init__(self, mode):
            self.mode = mode
        def predict_proba(self, x):
            n = x.shape[0]
            if self.mode == "crosses":
                p1 = np.where(x[:, 0] > 0, 0.2, 0.9).astype(np.float32)
            else:
                p1 = np.full(n, 0.8, dtype=np.float32)
            return np.stack([1 - p1, p1], axis=1)

    monkeypatch.setattr(transferability, "extract_ember2024_features", _extract)
    monkeypatch.setattr(
        transferability, "_load_model",
        lambda p, name: _CrossingModel("crosses" if name == "crosses" else "never"),
    )
    result = score_transferability(report_path, [
        ("crosses", tmp_path / "crosses.model"),
        ("never", tmp_path / "never.model"),
    ])
    assert result["n_samples"] == 2
    assert result["n_excluded"] == 0
    assert result["summary"]["transfer_rate_by_model"]["crosses"] == 1.0
    assert result["summary"]["transfer_rate_by_model"]["never"] == 0.0
    assert result["summary"]["flip_rate_by_model"]["crosses"] == 1.0
    assert result["summary"]["any_model_flip_rate"] == 1.0
    assert result["summary"]["all_model_flip_rate"] == 0.0
    assert result["per_sample"][0]["clean_p_by_model"]["crosses"] == pytest.approx(0.9)
    assert result["per_sample"][0]["best_p_by_model"]["crosses"] == pytest.approx(0.2)


def test_null_chosen_attack_is_excluded_even_when_bytes_exist(tmp_path, monkeypatch):
    best = tmp_path / "best_bytes"
    best.mkdir()
    clean, mut = _write_pair(best, "orphan", b"MZ" + b"\x00" * 62, b"MZ" + b"\x03" * 62)
    good_clean, good_mut = _write_pair(best, "good", b"MZ" + b"\x00" * 62, b"MZ" + b"\x04" * 62)
    report = {
        "same_sample_detail": {
            "best_bytes_manifest": [
                {"sample_path": None, "sha256_original": "orphan",
                 "path": str(mut), "size": 64,
                 "chosen_attack": None, "best_p_malware": None},
                {"sample_path": str(good_clean), "sha256_original": "good",
                 "path": str(good_mut), "size": 64,
                 "chosen_attack": "fulldos", "best_p_malware": 0.8,
                 "identical_to_original": False},
            ],
        }
    }
    report_path = tmp_path / "audit_report.json"
    report_path.write_text(json.dumps(report))
    from neurinspectre.evaluation import transferability
    monkeypatch.setattr(transferability, "extract_ember2024_features", _fake_extract_2024)
    monkeypatch.setattr(transferability, "_load_model", lambda p, name: _FakeSubModel(0.2))
    result = score_transferability(report_path, [("m", tmp_path / "m.model")])
    assert result["n_manifest"] == 2
    assert result["n_excluded"] == 1
    assert "chosen_attack_missing" in result["excluded"][0]["problems"]
    assert result["n_samples"] == 1
    # Constant p=0.2 on clean and mutated is a baseline miss, not a transfer.
    assert result["summary"]["transfer_rate_by_model"]["m"] == 0.0
    assert result["summary"]["baseline_miss_rate_by_model"]["m"] == 1.0
    assert result["summary"]["score_below_0.5_rate_by_model"]["m"] == 1.0


def test_identical_bytes_are_not_a_transfer(tmp_path, monkeypatch):
    best = tmp_path / "best_bytes"
    best.mkdir()
    raw = b"MZ" + b"\x00" * 62
    clean, mut = _write_pair(best, "same", raw, raw)
    report = {"same_sample_detail": {"best_bytes_manifest": [{
        "sample_path": str(clean), "sha256_original": "same",
        "path": str(mut), "size": 64,
        "chosen_attack": "clean", "best_p_malware": 0.2,
        "identical_to_original": True,
    }]}}
    report_path = tmp_path / "audit_report.json"
    report_path.write_text(json.dumps(report))
    from neurinspectre.evaluation import transferability
    monkeypatch.setattr(transferability, "extract_ember2024_features", _fake_extract_2024)
    monkeypatch.setattr(transferability, "_load_model", lambda p, name: _FakeSubModel(0.2))
    result = score_transferability(report_path, [("m", tmp_path / "m.model")])
    assert result["n_samples"] == 1
    assert result["summary"]["transfer_rate_by_model"]["m"] == 0.0
    assert result["per_sample"][0]["identical_to_original"] is True


def test_score_transferability_skips_missing_files(tmp_path, monkeypatch):
    report = {
        "same_sample_detail": {
            "best_bytes_manifest": [
                {"sample_path": str(tmp_path / "missing.exe"), "sha256_original": "xx",
                 "path": str(tmp_path / "does_not_exist.bin"), "size": 0,
                 "chosen_attack": "fulldos", "best_p_malware": 0.9},
            ]
        }
    }
    report_path = tmp_path / "audit_report.json"
    report_path.write_text(json.dumps(report))
    from neurinspectre.evaluation import transferability
    monkeypatch.setattr(transferability, "_load_model",
                        lambda p, name: _FakeSubModel(0.6))
    result = score_transferability(report_path, [("m", tmp_path / "m.model")])
    assert result["n_samples"] == 0
    assert result["n_excluded"] == 1
    assert result["excluded"][0]["problems"] == ["mutated_bytes_missing"]


def test_transferability_cli_registered():
    from click.testing import CliRunner
    from neurinspectre.cli.main import cli, _CLICK_COMMANDS
    assert "transferability" in _CLICK_COMMANDS
    runner = CliRunner()
    res = runner.invoke(cli, ["transferability", "--help"])
    assert res.exit_code == 0
    assert "-m" in res.output or "--models" in res.output


def test_transferability_cli_writes_output_and_parses_models(tmp_path, monkeypatch):
    """End-to-end run: build a synthetic report + best-bytes, invoke CLI,
    verify output JSON matches direct call.
    """
    best_dir = tmp_path / "best_bytes"; best_dir.mkdir()
    clean = tmp_path / "a.exe"
    clean.write_bytes(b"MZ" + b"\x00" * 62)
    (best_dir / "a.bin").write_bytes(b"MZ" + b"\x01" * 62)
    report = {"same_sample_detail": {"best_bytes_manifest": [
        {"sample_path": str(clean), "sha256_original": "a",
         "path": str(best_dir / "a.bin"), "size": 64,
         "chosen_attack": "fulldos", "best_p_malware": 0.7,
         "identical_to_original": False},
    ]}}
    rp = tmp_path / "rep.json"; rp.write_text(json.dumps(report))
    out = tmp_path / "xfer.json"

    from neurinspectre.evaluation import transferability
    monkeypatch.setattr(transferability, "extract_ember2024_features", _fake_extract_2024)
    monkeypatch.setattr(transferability, "_load_model",
                        lambda p, name: _FakeSubModel(0.3))

    from click.testing import CliRunner
    from neurinspectre.cli.main import cli
    runner = CliRunner()
    res = runner.invoke(cli, [
        "transferability", str(rp),
        "-m", "PE=" + str(tmp_path / "pe.model"),
        "-m", "Win64=" + str(tmp_path / "win64.model"),
        "-o", str(out),
    ])
    assert res.exit_code == 0, res.output
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["n_samples"] == 1
    # Constant p=0.3 on clean and mutated never crosses 0.5 from above.
    assert data["summary"]["flip_rate_by_model"]["PE"] == 0.0
    assert data["summary"]["flip_rate_by_model"]["Win64"] == 0.0
    assert data["summary"]["baseline_miss_rate_by_model"]["PE"] == 1.0
