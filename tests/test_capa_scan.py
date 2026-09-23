"""Tests for neurinspectre.malware.capa_scan and the capa gate in
ProblemSpacePESearch.

Coverage:

- ``CapaUnavailable`` is raised cleanly when capa is not importable or the
  rules dir is missing (must not require capa+rules to run this test).
- ``_find_rules_dir`` honours the ``CAPA_RULES`` env var.
- The gate in ``ProblemSpacePESearch`` is a **no-op when capa_preserve is
  False**: existing search shape unchanged, no capa calls, no rejects.
- With ``capa_preserve=True`` but ``CAPA_RULES`` pointing at nothing, the
  search **still runs** (the baseline scan fails once; every mutation is
  then accepted since baseline=None), and the report records the failure.
- The Click ``run-capa`` subcommand is discoverable and fails cleanly
  when the rules dir is missing.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from neurinspectre.attacks.problem_space_pe import ProblemSpacePESearch
from neurinspectre.malware.capa_scan import (
    CapaUnavailable,
    DEFAULT_RULES_DIR,
    _find_rules_dir,
)


def _minimal_pe() -> bytes:
    """Same PE fixture used by the other integration tests."""
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

    def __init__(self):
        super().__init__()

    def predict_proba(self, x):
        n = x.shape[0]
        p1 = np.full(n, 0.9, dtype=np.float32)  # never flips below 0.5
        return np.stack([1 - p1, p1], axis=1)

    def forward(self, x):
        probs = self.predict_proba(x.detach().cpu().numpy())
        s = np.log(probs[:, 1]) - np.log(probs[:, 0])
        return torch.as_tensor(np.stack([-s, s], axis=1), device=x.device, dtype=x.dtype)


def _fake_extract(_bytes):
    vec = np.zeros(2568, dtype=np.float32)
    return {"available": True, "features": vec, "reasons": [], "dim": 2568,
            "extractor": {"available": True, "reasons": []}}


# ---------------------------------------------------------------------------
# capa_scan low-level
# ---------------------------------------------------------------------------


def test_find_rules_dir_env_var_takes_priority(tmp_path, monkeypatch):
    # empty tmp dir counts as a valid directory for the env-var test
    monkeypatch.setenv("CAPA_RULES", str(tmp_path))
    assert _find_rules_dir() == tmp_path


def test_find_rules_dir_raises_when_nothing_available(tmp_path, monkeypatch):
    monkeypatch.delenv("CAPA_RULES", raising=False)
    # temporarily hide the on-disk default by pointing DEFAULT_RULES_DIR at a
    # dir that does not exist. We do that by passing explicit=path/nope which
    # exercises the same error path.
    with pytest.raises(CapaUnavailable):
        _find_rules_dir(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# ProblemSpacePESearch gate no-op behavior
# ---------------------------------------------------------------------------


def test_search_gate_is_noop_when_capa_preserve_false():
    search = ProblemSpacePESearch(
        _FakeGBDT(), n_queries=5, payload_size=32, seed=1,
        extractor=_fake_extract,
    )
    result = search.run_bytes(_minimal_pe(), y=1)
    # capa fields present but disabled
    assert result["capa_preserve"] is False
    assert result["n_capa_calls"] == 0
    assert result["n_capa_rejects"] == 0
    assert result["capa_baseline_available"] is None
    assert search._capa_error is None


def test_search_reports_capa_error_when_rules_missing(monkeypatch):
    """capa_preserve=True with a bad rules dir should not crash the search."""
    monkeypatch.setenv("CAPA_RULES", "/nonexistent/capa-rules-nope")
    search = ProblemSpacePESearch(
        _FakeGBDT(), n_queries=5, payload_size=32, seed=1,
        extractor=_fake_extract,
        capa_preserve=True,
        capa_rules_dir=Path("/nonexistent/capa-rules-nope"),
    )
    result = search.run_bytes(_minimal_pe(), y=1)
    assert result["capa_preserve"] is True
    assert result["capa_baseline_available"] is False
    assert "not found" in (result["capa_baseline_error"] or "")
    # baseline was None -> every mutation accepted -> zero rejects
    assert result["n_capa_rejects"] == 0


# ---------------------------------------------------------------------------
# Click subcommand registration
# ---------------------------------------------------------------------------


def test_run_capa_cli_registered():
    from click.testing import CliRunner
    from neurinspectre.cli.main import cli, _CLICK_COMMANDS
    assert "run-capa" in _CLICK_COMMANDS
    runner = CliRunner()
    res = runner.invoke(cli, ["run-capa", "--help"])
    assert res.exit_code == 0
    assert "capa" in res.output.lower()


def test_lookup_shas_reads_only_requested_hashes(tmp_path):
    from neurinspectre.malware.capa_supplement_index import lookup_shas

    hit = "ab" * 32
    miss = "cd" * 32
    # A `]` inside a capability name must not terminate the array.
    index = {
        hit: [
            {"func_addr": "0x1000", "capa": ["use ] in a name", "Read file on windows"], "byte_len": 4},
            {"func_addr": "0x2000", "capa": ["Read file on windows"], "byte_len": 8},
        ],
        "ee" * 32: [{"func_addr": "0x1", "capa": ["other"], "byte_len": 1}],
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index, separators=(",", ":")))
    found = lookup_shas(path, [hit.upper(), miss])
    assert set(found) == {hit}
    labels = [c for func in found[hit] for c in func["capa"]]
    assert labels.count("Read file on windows") == 2
    assert "use ] in a name" in labels


def test_filter_sidecar_keeps_file_level_and_supplement_apart(tmp_path):
    from neurinspectre.malware.capa_filters import TagFilter, load_tags_sidecar
    from neurinspectre.malware.capa_scan import filter_sidecar_record

    sha = "12" * 32
    record = filter_sidecar_record(
        sha256=sha,
        capabilities={"linked against OpenSSL"},
        metadata={
            "linked against OpenSSL": {
                "namespace": "linking/static/openssl",
                "attack": [],
                "mbc": ["Cryptography::Crypto Library [C0059]"],
            }
        },
        supplement_functions=[
            {"func_addr": "0x10", "capa": ["Encode data using xor"], "byte_len": 3},
        ],
        path="/tmp/sample.exe",
    )
    side = tmp_path / "tags.json"
    side.write_text(json.dumps({sha: record}))
    loaded = load_tags_sidecar(side)
    assert loaded[sha]["in_ember2024_capa_supplement"] is True
    assert TagFilter(capability=["openssl"]).match(loaded[sha])
    assert TagFilter(capability=["Encode data using xor"]).match(loaded[sha])
    assert TagFilter(mbc=["C0059"]).match(loaded[sha])
    sources = {c["source"] for c in loaded[sha]["caps"]}
    assert sources == {"file_level", "ember2024_capa_supplement"}


def test_run_capa_directory_writes_audit_sidecar(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from neurinspectre.cli.main import cli
    from neurinspectre.malware import capa_scan

    sha_bytes = b"MZ" + b"\x00" * 30
    import hashlib
    sha = hashlib.sha256(sha_bytes).hexdigest()
    pe_dir = tmp_path / "pe"
    pe_dir.mkdir()
    (pe_dir / "sample.exe").write_bytes(sha_bytes)
    (pe_dir / "not_a_pe.txt").write_text("hello")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({
        sha: [{"func_addr": "0x401000", "capa": ["Encode data using xor"], "byte_len": 4}],
    }))

    monkeypatch.setattr(
        capa_scan, "capabilities_file_level",
        lambda data, rules_dir=None: frozenset({"compiled with Go"}),
    )
    monkeypatch.setattr(
        capa_scan, "metadata_for_matches",
        lambda names, rules_dir=None: {
            "compiled with Go": {"namespace": "compiler/go", "attack": [], "mbc": []},
        },
    )
    sidecar = tmp_path / "tags.json"
    out = tmp_path / "scan.json"
    runner = CliRunner()
    res = runner.invoke(cli, [
        "run-capa", str(pe_dir),
        "--output", str(out),
        "--sidecar", str(sidecar),
        "--supplement-index", str(index),
    ])
    assert res.exit_code == 0, res.output
    summary = json.loads(out.read_text())
    assert summary["n_files"] == 1
    assert summary["n_in_ember2024_capa_supplement"] == 1
    assert summary["files"][0]["supplement_capabilities"] == ["Encode data using xor"]
    tags = json.loads(sidecar.read_text())
    assert tags[sha]["file_level_capabilities"] == ["compiled with Go"]
    assert "not_a_pe.txt" not in res.output


def test_run_capa_cli_fails_cleanly_when_rules_missing(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from neurinspectre.cli.main import cli

    monkeypatch.setenv("CAPA_RULES", str(tmp_path / "no_such_rules"))
    pe = tmp_path / "sample.exe"
    pe.write_bytes(_minimal_pe())
    runner = CliRunner()
    res = runner.invoke(cli, ["run-capa", str(pe), "--rules-dir", str(tmp_path / "no_such_rules")])
    assert res.exit_code != 0
    assert "capa" in res.output.lower() or "rules" in res.output.lower()
