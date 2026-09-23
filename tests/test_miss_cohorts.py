"""Tests for neurinspectre.malware.miss_cohorts.

Coverage:

- ``_extract`` per-namespace projection (singular vs list vs nested dict)
- ``score_cohort`` counts, sorting, min-support filter, tie-breaking
- ``score_all_namespaces`` returns every ``NAMESPACES`` key
- ``summarize_scoreboard`` renders lines in the expected order
- Click subcommand ``missrate-report`` reads a JSON and echoes the table
"""

from __future__ import annotations

import json

import pytest

from neurinspectre.malware.miss_cohorts import (
    NAMESPACES,
    _extract,
    score_all_namespaces,
    score_cohort,
    summarize_scoreboard,
)


def _row(**kw):
    return dict(kw)


def _ttp(tactic, technique):
    return {"Tactic": tactic, "Technique": technique}


def _mbc(objective, behavior):
    return {"Objective": objective, "Behavior": behavior}


def _cap(cap, ns):
    return {"Capability": cap, "Namespace": ns}


# ---------------------------------------------------------------------------
# _extract per-namespace projection
# ---------------------------------------------------------------------------


def test_extract_file_type_and_family():
    row = _row(file_type="Win32", family="rugmi")
    assert _extract(row, "file_type") == ["Win32"]
    assert _extract(row, "family") == ["rugmi"]


def test_extract_list_namespaces_are_deduped_within_a_row():
    row = _row(behavior=["downloader", "downloader", "spyware"], packer=[], group=None)
    assert set(_extract(row, "behavior")) == {"downloader", "spyware"}
    assert _extract(row, "packer") == []
    assert _extract(row, "group") == []


def test_extract_ttp_tactic_and_technique():
    row = _row(ttps=[_ttp("DISCOVERY", "T1083"), _ttp("EXECUTION", "T1129"), _ttp("DISCOVERY", "T1082")])
    assert set(_extract(row, "ttps.tactic")) == {"DISCOVERY", "EXECUTION"}
    assert set(_extract(row, "ttps.technique")) == {"T1083", "T1129", "T1082"}


def test_extract_mbc():
    row = _row(mbc=[_mbc("FILE SYSTEM", "C0016"), _mbc("DATA", "C0026.002")])
    assert set(_extract(row, "mbc.objective")) == {"FILE SYSTEM", "DATA"}
    assert set(_extract(row, "mbc.behavior")) == {"C0016", "C0026.002"}


def test_extract_capa():
    row = _row(caps=[_cap("Read file on windows", "host-interaction/file-system/read")])
    assert _extract(row, "caps.capability") == ["Read file on windows"]
    assert _extract(row, "caps.namespace") == ["host-interaction/file-system/read"]


def test_extract_empty_and_none_are_dropped():
    row = _row(file_type=None, family="", behavior=[None, "", "adware"])
    assert _extract(row, "file_type") == []
    assert _extract(row, "family") == []
    assert _extract(row, "behavior") == ["adware"]


# ---------------------------------------------------------------------------
# score_cohort
# ---------------------------------------------------------------------------


def test_score_cohort_counts_and_sorting():
    records = [
        _row(family="rugmi"),
        _row(family="rugmi"),
        _row(family="rugmi"),  # 3x rugmi
        _row(family="wacatac"),
        _row(family="wacatac"),  # 2x wacatac
    ]
    # rugmi missed 1/3, wacatac missed 2/2 -> wacatac ranks first
    missed = [False, True, False, True, True]
    rows = score_cohort(records, missed, "family", min_support=2)
    assert len(rows) == 2
    assert rows[0].label == "wacatac"
    assert rows[0].n_total == 2
    assert rows[0].n_missed == 2
    assert rows[0].miss_rate == 1.0
    assert rows[1].label == "rugmi"
    assert pytest.approx(rows[1].miss_rate) == 1 / 3


def test_score_cohort_min_support_filters():
    records = [_row(family="one")] * 5 + [_row(family="two")] * 25
    missed = [False] * 5 + [True] * 25
    rows = score_cohort(records, missed, "family", min_support=20)
    labels = [r.label for r in rows]
    assert "two" in labels
    assert "one" not in labels


def test_score_cohort_length_mismatch_raises():
    with pytest.raises(ValueError):
        score_cohort([_row()], [True, False], "family")


def test_score_cohort_tie_breaks_by_larger_n_then_alphabetical():
    records = [_row(family="a")] * 3 + [_row(family="b")] * 3 + [_row(family="c")] * 3
    # all 100% miss
    missed = [True] * 9
    rows = score_cohort(records, missed, "family", min_support=1)
    # tie on miss_rate=1.0, tie on n_total=3 -> label ascending
    assert [r.label for r in rows] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# score_all_namespaces + summarize_scoreboard
# ---------------------------------------------------------------------------


def test_score_all_namespaces_covers_every_namespace():
    records = [_row(file_type="Win32", family="rugmi", behavior=["downloader"],
                    mbc=[_mbc("FILE SYSTEM", "C0016")],
                    ttps=[_ttp("DISCOVERY", "T1083")],
                    caps=[_cap("Read file", "host-interaction/file-system/read")])]
    missed = [True]
    board = score_all_namespaces(records, missed, min_support=1, limit_per_namespace=5)
    assert set(board.keys()) == set(NAMESPACES)


def test_summarize_scoreboard_prints_top_rows():
    board = {
        "family": [{"label": "wacatac", "n_total": 3, "n_missed": 3, "miss_rate": 1.0},
                   {"label": "rugmi", "n_total": 4, "n_missed": 1, "miss_rate": 0.25}],
        "empty_ns": [],
    }
    text = summarize_scoreboard(board, top_n=5)
    assert "family" in text
    assert "wacatac" in text
    assert "rugmi" in text
    # empty namespaces are skipped entirely
    assert "empty_ns" not in text


# ---------------------------------------------------------------------------
# Click missrate-report subcommand
# ---------------------------------------------------------------------------


def test_missrate_report_cli(tmp_path):
    from click.testing import CliRunner

    from neurinspectre.cli.main import cli, _CLICK_COMMANDS

    assert "missrate-report" in _CLICK_COMMANDS
    p = tmp_path / "scoring.json"
    p.write_text(json.dumps({
        "miss_cohorts": {
            "family": [
                {"label": "wacatac", "n_total": 3, "n_missed": 3, "miss_rate": 1.0},
                {"label": "rugmi", "n_total": 4, "n_missed": 1, "miss_rate": 0.25},
            ],
        },
    }))
    runner = CliRunner()
    result = runner.invoke(cli, ["missrate-report", str(p)])
    assert result.exit_code == 0, result.output
    assert "wacatac" in result.output
    assert "rugmi" in result.output


def test_missrate_report_cli_rejects_json_without_miss_cohorts(tmp_path):
    from click.testing import CliRunner

    from neurinspectre.cli.main import cli

    p = tmp_path / "nocoh.json"
    p.write_text(json.dumps({"per_model": {}}))
    runner = CliRunner()
    result = runner.invoke(cli, ["missrate-report", str(p)])
    assert result.exit_code != 0
    assert "miss_cohorts" in result.output


def test_missrate_report_cli_namespace_filter(tmp_path):
    from click.testing import CliRunner

    from neurinspectre.cli.main import cli

    p = tmp_path / "scoring.json"
    p.write_text(json.dumps({
        "miss_cohorts": {
            "family": [{"label": "wacatac", "n_total": 3, "n_missed": 3, "miss_rate": 1.0}],
            "ttps.technique": [{"label": "T1489", "n_total": 29, "n_missed": 19, "miss_rate": 0.655}],
        },
    }))
    runner = CliRunner()
    result = runner.invoke(cli, ["missrate-report", str(p), "--namespace", "family"])
    assert result.exit_code == 0
    assert "wacatac" in result.output
    assert "T1489" not in result.output
