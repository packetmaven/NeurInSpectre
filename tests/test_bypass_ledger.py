"""Tests for neurinspectre.malware.bypass_ledger.

Coverage:

- ``_sample_to_row`` per-row projection: numerical fields, flipped flag,
  top-N ATT&CK/MBC/Capa aggregation, family/file_type/behavior/etc.
- ``build_ledger`` splits kept samples into flipped vs close-call and
  respects close_call_min_delta / limit
- Sorting: flipped ascending by best_p, close_calls descending by p_delta
- Markdown rendering does not crash on empty sections
- Click ``bypass-ledger`` subcommand is registered and end-to-end runs on
  a synthetic audit_report.json
"""

from __future__ import annotations

import json

import pytest

from neurinspectre.malware.bypass_ledger import (
    BypassRow,
    build_ledger,
    read_report,
    render_markdown,
)


def _sample(
    *,
    sha256="a" * 64,
    kept=True,
    clean_p=0.998,
    best_p=0.4,
    chosen="fulldos",
    queries=100,
    realizable=True,
    parse_valid=True,
    tags_full=None,
):
    return {
        "kept": kept,
        "sha256": sha256,
        "path": f"/mwb/{sha256[:8]}.exe",
        "pe_parse": {"passed": parse_valid},
        "problem": {
            "clean_p_malware": clean_p,
            "best_p_malware": best_p,
            "chosen_attack": chosen,
            "queries_used": queries,
            "realizable": realizable,
            "success": bool(best_p is not None and best_p < 0.5),
        },
        "tags_full": tags_full or {},
        "tags": {"family": (tags_full or {}).get("family"),
                 "file_type": (tags_full or {}).get("file_type")},
    }


def _ttp(t): return {"Tactic": "DISCOVERY", "Technique": t}
def _mbc(b): return {"Objective": "FILE SYSTEM", "Behavior": b}
def _cap(c): return {"Capability": c, "Namespace": "ns/a"}


# ---------------------------------------------------------------------------
# _sample_to_row
# ---------------------------------------------------------------------------


def test_row_flipped_and_delta():
    s = _sample(clean_p=0.9, best_p=0.4)
    from neurinspectre.malware.bypass_ledger import _sample_to_row
    row = _sample_to_row(s, top_n_tags=5)
    assert row.flipped is True
    assert row.clean_p == 0.9
    assert row.best_p == 0.4
    assert row.p_delta == pytest.approx(0.5)
    assert row.chosen == "fulldos"


def test_row_close_call_not_flipped():
    s = _sample(clean_p=0.9, best_p=0.6)
    from neurinspectre.malware.bypass_ledger import _sample_to_row
    row = _sample_to_row(s, top_n_tags=5)
    assert row.flipped is False
    assert row.p_delta == pytest.approx(0.3)


def test_row_tag_aggregation_top_n():
    tags = {
        "file_type": "Win32", "family": "rugmi",
        "behavior": ["downloader", "spyware"],
        "packer": ["nsis"],
        "ttps": [_ttp("File Discovery [T1083]"), _ttp("Process Injection [T1055]"),
                 _ttp("File Discovery [T1083]"), _ttp("System Info [T1082]")],
        "mbc": [_mbc("Create File [C0016]"), _mbc("Create File [C0016]"),
                _mbc("Read File [C0051]")],
        "caps": [_cap("Encode xor"), _cap("Encode xor"), _cap("Read file")],
    }
    s = _sample(tags_full=tags)
    from neurinspectre.malware.bypass_ledger import _sample_to_row
    row = _sample_to_row(s, top_n_tags=2)
    assert row.family == "rugmi"
    assert row.file_type == "Win32"
    assert set(row.behavior) == {"downloader", "spyware"}
    assert row.packer == ["nsis"]
    # top-2 by frequency
    assert row.ttps[0] == "File Discovery [T1083]"
    assert len(row.ttps) == 2
    assert row.mbc[0] == "Create File [C0016]"
    assert row.capa[0] == "Encode xor"


# ---------------------------------------------------------------------------
# build_ledger
# ---------------------------------------------------------------------------


def test_build_ledger_splits_flipped_and_close_calls():
    samples = [
        _sample(sha256="A" * 64, clean_p=0.99, best_p=0.4),   # flipped
        _sample(sha256="B" * 64, clean_p=0.99, best_p=0.6),   # close call (delta 0.39)
        _sample(sha256="C" * 64, clean_p=0.99, best_p=0.98),  # nothing (delta 0.01)
        _sample(sha256="D" * 64, clean_p=0.99, best_p=0.99, kept=False),  # skipped
    ]
    report = {"same_sample_detail": {"samples": samples},
              "target": "ember2024-gbdt", "official_reproduction": True}
    ledger = build_ledger(report, close_call_min_delta=0.05)
    assert ledger["n_kept"] == 3
    assert ledger["n_flipped"] == 1
    assert ledger["n_close_calls"] == 1
    assert ledger["flipped_rows"][0]["sha256"] == "A" * 64
    assert ledger["close_calls"][0]["sha256"] == "B" * 64


def test_build_ledger_sorts_correctly():
    samples = [
        _sample(sha256="A" * 64, clean_p=0.99, best_p=0.45),   # flip
        _sample(sha256="B" * 64, clean_p=0.99, best_p=0.20),   # flip, harder
        _sample(sha256="C" * 64, clean_p=0.99, best_p=0.60),   # close call, delta 0.39
        _sample(sha256="D" * 64, clean_p=0.99, best_p=0.55),   # close call, delta 0.44
    ]
    report = {"same_sample_detail": {"samples": samples}}
    ledger = build_ledger(report, close_call_min_delta=0.05)
    # flipped sorted ascending by best_p
    assert [r["sha256"] for r in ledger["flipped_rows"]] == ["B" * 64, "A" * 64]
    # close calls sorted descending by p_delta
    assert [r["sha256"] for r in ledger["close_calls"]] == ["D" * 64, "C" * 64]


def test_build_ledger_respects_limit():
    samples = [
        _sample(sha256=chr(ord("A") + i) * 64, clean_p=0.99, best_p=0.55 + 0.01 * i)
        for i in range(30)
    ]
    report = {"same_sample_detail": {"samples": samples}}
    ledger = build_ledger(report, close_call_min_delta=0.0, limit=5)
    assert len(ledger["close_calls"]) == 5


def test_build_ledger_handles_empty_samples():
    ledger = build_ledger({"same_sample_detail": {"samples": []}})
    assert ledger["n_kept"] == 0
    assert ledger["flipped_rows"] == []
    assert ledger["close_calls"] == []


def test_render_markdown_handles_empty_sections():
    ledger = {"target": "ember2024-gbdt", "n_kept": 0, "n_flipped": 0,
              "n_close_calls": 0, "close_call_min_delta": 0.05,
              "official_reproduction": True, "quote_as_ember2018": True,
              "flipped_rows": [], "close_calls": [], "note": "n/a"}
    md = render_markdown(ledger)
    assert "EMBER bypass ledger" in md
    assert "ember2024-gbdt" in md


# ---------------------------------------------------------------------------
# Click subcommand
# ---------------------------------------------------------------------------


def test_bypass_ledger_cli_registered_and_runs(tmp_path):
    from click.testing import CliRunner

    from neurinspectre.cli.main import cli, _CLICK_COMMANDS

    assert "bypass-ledger" in _CLICK_COMMANDS
    audit = {
        "target": "ember2024-gbdt", "official_reproduction": True,
        "quote_as_ember2018": True,
        "same_sample_detail": {
            "samples": [
                _sample(sha256="A" * 64, clean_p=0.9, best_p=0.3,
                        tags_full={"file_type": "Win32", "family": "rugmi",
                                   "ttps": [_ttp("File Discovery [T1083]")]}),
                _sample(sha256="B" * 64, clean_p=0.9, best_p=0.6,
                        tags_full={"file_type": "Win64", "family": "wacatac"}),
            ],
        },
    }
    p = tmp_path / "audit_report.json"
    p.write_text(json.dumps(audit))
    md = tmp_path / "ledger.md"
    runner = CliRunner()
    result = runner.invoke(cli, ["bypass-ledger", str(p),
                                 "--markdown", str(md),
                                 "--close-call-min-delta", "0.05"])
    assert result.exit_code == 0, result.output

    out_path = tmp_path / "ember_bypass_ledger.json"
    assert out_path.is_file()
    ledger = json.loads(out_path.read_text())
    assert ledger["n_kept"] == 2
    assert ledger["n_flipped"] == 1
    assert ledger["n_close_calls"] == 1
    md_text = md.read_text()
    assert "EMBER bypass ledger" in md_text
    assert "T1083" in md_text
