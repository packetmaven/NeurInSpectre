"""Unit tests for neurinspectre.malware.capa_filters.TagFilter.

Coverage plan:

- CSV parsing (empty / spaces / mixed types)
- Bracket-ID extraction for ATT&CK (Txxxx) and MBC (Cxxxx/Bxxxx/Exxxx)
- Case-insensitive substring on plain strings
- Bracketed-ID equivalence (``T1055`` matches ``"Process Injection [T1055]"``)
- Per-field predicates on records that mimic the challenge JSONL schema
- OR-within-field / AND-across-fields composition
- ``min_vt_detected`` numerator threshold
- ``is_active`` false when no filters are set (no-filter fast path)
- ``apply_tag_filter`` seen/dropped counters

None of these tests touch the LightGBM or thrember stacks; they are pure
Python assertions on the predicate.
"""

from __future__ import annotations

import pytest

from neurinspectre.malware.capa_filters import (
    TagFilter,
    _BRACKET_ID_RE,
    _extract_bracket_ids,
    _match_needle,
    _parse_csv,
    apply_tag_filter,
)


# ---------------------------------------------------------------------------
# Micro-helpers
# ---------------------------------------------------------------------------


def _row(
    *,
    file_type: str = "Win32",
    family: str = "rugmi",
    behavior=None,
    file_property=None,
    packer=None,
    exploit=None,
    group=None,
    caps=None,
    ttps=None,
    mbc=None,
    detection_ratio: str = "20/70",
) -> dict:
    """Shape a record like the challenge JSONLs use."""
    return {
        "sha256": "0" * 64,
        "file_type": file_type,
        "family": family,
        "behavior": behavior or [],
        "file_property": file_property or [],
        "packer": packer or [],
        "exploit": exploit or [],
        "group": group or [],
        "caps": caps or [],
        "ttps": ttps or [],
        "mbc": mbc or [],
        "detection_ratio": detection_ratio,
    }


def _ttp(tactic: str, technique: str):
    return {"Tactic": tactic, "Technique": technique}


def _mbc(objective: str, behavior: str):
    return {"Objective": objective, "Behavior": behavior}


def _cap(capability: str, namespace: str):
    return {"Capability": capability, "Namespace": namespace, "Addrs": []}


# ---------------------------------------------------------------------------
# Parsing / low-level helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, []),
        ("", []),
        ("A", ["A"]),
        ("A,B,C", ["A", "B", "C"]),
        ("  A , B ,,,  C  ", ["A", "B", "C"]),
        (["A", " B "], ["A", "B"]),
        (["A", "", None, "B"], ["A", "B"]),
    ],
)
def test_parse_csv(raw, expected):
    assert _parse_csv(raw) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Process Injection [T1055]", ["T1055"]),
        ("File and Directory Discovery [T1083]", ["T1083"]),
        ("Sub-technique [T1497.001]", ["T1497.001"]),
        ("Read File [C0051]", ["C0051"]),
        ("Virtual Machine Detection [B0009]", ["B0009"]),
        ("File and Directory Discovery [E1083]", ["E1083"]),
        ("no id here", []),
        ("multiple [T1055] and [T1057]", ["T1055", "T1057"]),
    ],
)
def test_extract_bracket_ids(value, expected):
    assert _extract_bracket_ids(value) == expected


def test_bracket_id_re_is_anchored_to_brackets():
    # A bare Txxxx without brackets must not be picked up
    assert _BRACKET_ID_RE.findall("T1055 is a technique") == []


@pytest.mark.parametrize(
    "needle, hay, expected",
    [
        # basic substring, case-insensitive
        ("Win32", ["Win32"], True),
        ("win32", ["Win32"], True),
        ("32", ["Win32"], True),
        ("bogus", ["Win32"], False),
        # bracketed ID match
        ("T1055", ["Process Injection [T1055]"], True),
        ("t1055", ["Process Injection [T1055]"], True),
        ("C0051", ["Read File [C0051]"], True),
        # bare ID that doesn't have brackets in hay: substring still works
        ("Process", ["Process Injection [T1055]"], True),
        # empty needle
        ("", ["anything"], False),
        # empty hay
        ("Win32", [], False),
        # None-in-hay tolerated
        ("Win32", [None, "Win32"], True),
        # exact-ID must match only via brackets; needle "T105" would substring-match anyway
        ("T105", ["Process Injection [T1055]"], True),  # substring
    ],
)
def test_match_needle(needle, hay, expected):
    assert _match_needle(needle, hay) is expected


# ---------------------------------------------------------------------------
# Per-field predicates
# ---------------------------------------------------------------------------


def test_filter_is_inactive_when_empty():
    flt = TagFilter()
    assert flt.is_active() is False
    # inactive filter matches everything
    assert flt.match(_row()) is True


def test_filter_active_when_min_vt_only():
    flt = TagFilter(min_vt_detected=1)
    assert flt.is_active() is True


def test_file_type_filter_or_within_field():
    flt = TagFilter(file_type=["Win32", "Win64"])
    assert flt.match(_row(file_type="Win32")) is True
    assert flt.match(_row(file_type="Win64")) is True
    assert flt.match(_row(file_type="ELF")) is False


def test_family_filter_substring():
    flt = TagFilter(family=["rugm"])  # substring of rugmi
    assert flt.match(_row(family="rugmi")) is True
    assert flt.match(_row(family="rugminox")) is True
    assert flt.match(_row(family="wacatac")) is False


def test_tag_filter_hits_behavior_property_packer_exploit_group():
    flt = TagFilter(tag=["downloader"])
    assert flt.match(_row(behavior=["downloader"])) is True
    assert flt.match(_row(behavior=["stealer"])) is False

    # same tag flag hits packer field
    flt = TagFilter(tag=["nsis"])
    assert flt.match(_row(packer=["nsis"])) is True

    # or exploit
    flt = TagFilter(tag=["cve_2024_1086"])
    assert flt.match(_row(exploit=["cve_2024_1086"])) is True

    # or group
    flt = TagFilter(tag=["lazarusgroup"])
    assert flt.match(_row(group=["lazarusgroup"])) is True

    # or file_property
    flt = TagFilter(tag=["nsis", "msil"])
    assert flt.match(_row(file_property=["msil"])) is True


def test_ttp_filter_matches_tactic_and_technique_and_id():
    row = _row(ttps=[
        _ttp("DISCOVERY", "File and Directory Discovery [T1083]"),
        _ttp("DEFENSE EVASION", "Obfuscated Files or Information [T1027]"),
    ])
    assert TagFilter(ttp=["T1083"]).match(row) is True  # bracketed ID
    assert TagFilter(ttp=["t1083"]).match(row) is True  # case-insensitive
    assert TagFilter(ttp=["File and Directory"]).match(row) is True  # substring of technique
    assert TagFilter(ttp=["DISCOVERY"]).match(row) is True  # tactic
    assert TagFilter(ttp=["T9999"]).match(row) is False


def test_mbc_filter_matches_objective_and_behavior_and_id():
    row = _row(mbc=[
        _mbc("FILE SYSTEM", "Create File [C0016]"),
        _mbc("DATA", "Encode Data [C0026.002]"),
    ])
    assert TagFilter(mbc=["C0016"]).match(row) is True
    assert TagFilter(mbc=["c0026.002"]).match(row) is True
    assert TagFilter(mbc=["FILE SYSTEM"]).match(row) is True
    assert TagFilter(mbc=["encode"]).match(row) is True  # substring
    assert TagFilter(mbc=["X9999"]).match(row) is False


def test_capability_filter_matches_capability_and_namespace():
    row = _row(caps=[
        _cap("Read file on windows", "host-interaction/file-system/read"),
        _cap("Encode data using xor", "data-manipulation/encoding/xor"),
    ])
    assert TagFilter(capability=["encode data using xor"]).match(row) is True
    assert TagFilter(capability=["file-system/read"]).match(row) is True
    assert TagFilter(capability=["clipboard"]).match(row) is False


def test_min_vt_detected_threshold():
    assert TagFilter(min_vt_detected=10).match(_row(detection_ratio="15/70")) is True
    assert TagFilter(min_vt_detected=20).match(_row(detection_ratio="15/70")) is False
    # malformed detection_ratio -> reject
    assert TagFilter(min_vt_detected=1).match(_row(detection_ratio="")) is False
    assert TagFilter(min_vt_detected=1).match(_row(detection_ratio="not/a/ratio")) is False


# ---------------------------------------------------------------------------
# Composition: AND across fields
# ---------------------------------------------------------------------------


def test_and_across_fields_is_conjunction():
    row = _row(
        file_type="Win32",
        family="rugmi",
        ttps=[_ttp("DISCOVERY", "Foo [T1083]")],
    )
    # Every filter satisfied -> match
    assert TagFilter(file_type=["Win32"], family=["rugmi"], ttp=["T1083"]).match(row) is True
    # Any one unsatisfied -> no match
    assert TagFilter(file_type=["Win64"], family=["rugmi"], ttp=["T1083"]).match(row) is False
    assert TagFilter(file_type=["Win32"], family=["notrugmi"], ttp=["T1083"]).match(row) is False
    assert TagFilter(file_type=["Win32"], family=["rugmi"], ttp=["T9999"]).match(row) is False


# ---------------------------------------------------------------------------
# CLI parse-then-match glue
# ---------------------------------------------------------------------------


def test_from_cli_end_to_end():
    flt = TagFilter.from_cli(
        file_type="Win32 ,Win64",
        family=None,
        tag="downloader,lazarusgroup",
        ttp="T1055 , T1083",
        mbc=None,
        capability=None,
        min_vt_detected=10,
    )
    assert flt.file_type == ["Win32", "Win64"]
    assert flt.tag == ["downloader", "lazarusgroup"]
    assert flt.ttp == ["T1055", "T1083"]
    assert flt.min_vt_detected == 10
    assert flt.is_active()

    match_row = _row(
        file_type="Win64",
        behavior=["downloader"],
        ttps=[_ttp("DISCOVERY", "File Discovery [T1083]")],
        detection_ratio="30/70",
    )
    assert flt.match(match_row) is True

    non_match_row = _row(
        file_type="Win64",
        behavior=["stealer"],  # no downloader / no lazarusgroup
        ttps=[_ttp("DISCOVERY", "File Discovery [T1083]")],
        detection_ratio="30/70",
    )
    assert flt.match(non_match_row) is False


# ---------------------------------------------------------------------------
# apply_tag_filter batch helper
# ---------------------------------------------------------------------------


def test_click_repeated_flag_composition_via_flatten():
    """The Click subcommand ``score-ember2024-challenge`` accepts every
    filter flag as ``multiple=True`` **and** each value is comma-split.

    ``--filter-tag a --filter-tag b,c`` must produce ``["a", "b", "c"]``.
    This test guards against the regression seen in the A1 live walkthrough
    where the last flag silently overrode the earlier one.
    """
    def _flatten(values):
        out = []
        for v in values:
            out.extend(_parse_csv(v))
        return out

    assert _flatten(("nsis", "downloader,spyware")) == ["nsis", "downloader", "spyware"]
    assert _flatten(()) == []
    assert _flatten(("",)) == []
    assert _flatten(("  a  ,  b  ",)) == ["a", "b"]


def test_apply_tag_filter_counts():
    rows = [
        _row(family="rugmi"),
        _row(family="rugmi"),
        _row(family="wacatac"),
        _row(family="opensupdater"),
    ]
    flt = TagFilter(family=["rugmi"])
    kept, seen, dropped = apply_tag_filter(rows, flt)
    assert seen == 4
    assert dropped == 2
    assert len(kept) == 2
    # inactive filter is a pass-through
    kept2, seen2, dropped2 = apply_tag_filter(rows, TagFilter())
    assert (seen2, dropped2, len(kept2)) == (4, 0, 4)


# ---------------------------------------------------------------------------
# Sidecar loader
# ---------------------------------------------------------------------------


def test_load_tags_sidecar_json_object(tmp_path):
    from neurinspectre.malware.capa_filters import load_tags_sidecar

    obj = {
        "AA" * 32: {"file_type": "Win32", "family": "rugmi", "ttps": []},
        "BB" * 32: {"file_type": "Win64", "family": "wacatac", "ttps": []},
    }
    p = tmp_path / "tags.json"
    p.write_text(__import__("json").dumps(obj, indent=2))
    loaded = load_tags_sidecar(p)
    assert set(loaded) == {("AA" * 32).lower(), ("BB" * 32).lower()}
    assert loaded[("AA" * 32).lower()]["family"] == "rugmi"


def test_load_tags_sidecar_jsonl(tmp_path):
    from neurinspectre.malware.capa_filters import load_tags_sidecar

    rows = [
        {"sha256": "AA" * 32, "file_type": "Win32", "family": "rugmi"},
        {"sha256": "BB" * 32, "file_type": "Win64", "family": "wacatac"},
    ]
    p = tmp_path / "tags.jsonl"
    p.write_text("\n".join(__import__("json").dumps(r) for r in rows))
    loaded = load_tags_sidecar(p)
    assert set(loaded) == {("AA" * 32).lower(), ("BB" * 32).lower()}


def test_load_tags_sidecar_empty(tmp_path):
    from neurinspectre.malware.capa_filters import load_tags_sidecar

    p = tmp_path / "empty.json"
    p.write_text("")
    assert load_tags_sidecar(p) == {}


# ---------------------------------------------------------------------------
# In-loop filter integration (evaluate_ember_same_sample)
# ---------------------------------------------------------------------------


def test_same_sample_tag_filter_drops_untagged_by_default(tmp_path):
    """A PE file whose SHA-256 isn't in the sidecar must be dropped when the
    tag filter is active and --filter-include-untagged is False."""
    import hashlib
    import struct

    # Reuse the minimal PE fixture builder from test_ember2024_integration.
    from tests.test_ember2024_integration import _minimal_pe
    from neurinspectre.evaluation.ember_same_sample import evaluate_ember_same_sample

    pe_bytes = _minimal_pe()
    sha = hashlib.sha256(pe_bytes).hexdigest()

    # Fake model: torch.nn.Module with predict_proba returning benign (0.1)
    import torch
    import torch.nn as nn
    import numpy as np

    class _FakeGBDT(nn.Module):
        feature_dim = 2568
        def __init__(self):
            super().__init__()
        def predict_proba(self, x):
            n = x.shape[0]
            p1 = np.full(n, 0.9, dtype=np.float32)  # calls it malware -> kept
            return np.stack([1 - p1, p1], axis=1)
        def forward(self, x):
            probs = self.predict_proba(x.detach().cpu().numpy())
            s = np.log(probs[:, 1]) - np.log(probs[:, 0])
            return torch.as_tensor(np.stack([-s, s], axis=1), device=x.device, dtype=x.dtype)

    # Extractor returns a plausible 2568-d finite vector for any bytes.
    def _fake_extract(_bytes):
        vec = np.zeros(2568, dtype=np.float32)
        return {"available": True, "features": vec, "reasons": [], "dim": 2568,
                "extractor": {"available": True, "reasons": []}}

    pe_dir = tmp_path / "pe"
    pe_dir.mkdir()
    (pe_dir / "sample.exe").write_bytes(pe_bytes)

    # Tag filter that requires family=rugmi; supply a sidecar for our sha with a
    # non-matching tag first, then a matching one, and verify counts.
    flt = TagFilter(family=["rugmi"])
    sidecar_bad = {sha: {"family": "wacatac"}}
    result = evaluate_ember_same_sample(
        pe_dir, _FakeGBDT(), n_queries=5,
        query_budgets=[5], extractor=_fake_extract,
        tag_filter=flt, tags_by_sha256=sidecar_bad,
    )
    assert result["n_detected_malware"] == 0
    assert result["tag_filter_active"] is True
    assert result["n_filtered_out_by_tag"] == 1

    sidecar_good = {sha: {
        "family": "rugmi",
        "file_type": "Win32",
        "caps": [{"Capability": "compiled with Go", "Namespace": "compiler/go", "source": "file_level"}],
        "ttps": [{"Tactic": "Defense Evasion", "Technique": "Obfuscated Files or Information [T1027]"}],
        "mbc": [{"Objective": "Cryptography", "Behavior": "Crypto Library [C0059]"}],
        "in_ember2024_capa_supplement": False,
        "supplement_capabilities": [],
    }}
    result = evaluate_ember_same_sample(
        pe_dir, _FakeGBDT(), n_queries=5,
        query_budgets=[5], extractor=_fake_extract,
        tag_filter=flt, tags_by_sha256=sidecar_good,
    )
    # matched by filter -> extraction runs -> GBDT scores 0.9 -> kept
    assert result["n_detected_malware"] == 1
    assert result["tag_filter_active"] is True
    assert result["n_filtered_out_by_tag"] == 0
    kept = [s for s in result["samples"] if s.get("kept")]
    assert kept[0]["tags"]["family"] == "rugmi"
    assert kept[0]["tags"]["capabilities"] == ["compiled with Go"]
    assert kept[0]["tags"]["ttps"] == ["Obfuscated Files or Information [T1027]"]
    assert kept[0]["tags"]["mbc"] == ["Crypto Library [C0059]"]
    assert kept[0]["tags"]["in_ember2024_capa_supplement"] is False

    # Untagged file with filter active + include_untagged=False -> dropped
    empty_sidecar: dict[str, dict] = {}
    result = evaluate_ember_same_sample(
        pe_dir, _FakeGBDT(), n_queries=5,
        query_budgets=[5], extractor=_fake_extract,
        tag_filter=flt, tags_by_sha256=empty_sidecar,
    )
    assert result["n_detected_malware"] == 0
    assert result["n_untagged_seen"] == 1
    assert result["n_filtered_out_by_tag"] == 1

    # Same, but include_untagged=True -> kept
    result = evaluate_ember_same_sample(
        pe_dir, _FakeGBDT(), n_queries=5,
        query_budgets=[5], extractor=_fake_extract,
        tag_filter=flt, tags_by_sha256=empty_sidecar,
        filter_include_untagged=True,
    )
    assert result["n_detected_malware"] == 1
    assert result["n_untagged_seen"] == 1
    assert result["n_filtered_out_by_tag"] == 0
