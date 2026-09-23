"""Tests for neurinspectre.malware.file_families.

Coverage:

- Magic sniffers for PE, APK, ELF, PDF
- Suffix-based file discovery + rglob depth
- ``load_samples_by_family`` on directory / single file / bytes / mixed
- ``resolve_family`` name normalisation and error handling
- Non-matching bytes literal returns error row (not raise)
"""

from __future__ import annotations

import pytest

from neurinspectre.malware.file_families import (
    APK,
    ELF,
    FAMILIES_BY_NAME,
    PDF,
    PE,
    load_samples_by_family,
    resolve_family,
)


def test_families_registry_names():
    assert set(FAMILIES_BY_NAME) == {"PE", "APK", "ELF", "PDF", "ANY"}


def test_any_family_matches_any_bytes():
    from neurinspectre.malware.file_families import ANY
    assert ANY.matches(b"anything") is True
    assert ANY.matches(b"\x7fELF") is True
    assert ANY.matches(b"MZ\x00\x00") is True
    assert ANY.matches(b"") is False


def test_load_samples_any_family_recurses(tmp_path):
    """ANY family = "the ``all`` detector accepts anything" — recursive
    discovery so nested folders of mixed files are covered."""
    from neurinspectre.malware.file_families import ANY, load_samples_by_family
    (tmp_path / "a.bin").write_bytes(b"anything")
    (tmp_path / "b.dat").write_bytes(b"more")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.hidden").write_bytes(b"deep")
    out = load_samples_by_family(tmp_path, ANY)
    names = {(s["path"] or "").split("/")[-1] for s in out if s["bytes"]}
    assert names == {"a.bin", "b.dat", "c.hidden"}


def test_load_samples_any_family_skips_dot_dirs(tmp_path):
    from neurinspectre.malware.file_families import ANY, load_samples_by_family
    (tmp_path / "keep.bin").write_bytes(b"1")
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "objects").write_bytes(b"2")
    out = load_samples_by_family(tmp_path, ANY)
    names = {(s["path"] or "").split("/")[-1] for s in out if s["bytes"]}
    assert names == {"keep.bin"}


@pytest.mark.parametrize(
    "family, head, expected",
    [
        (PE, b"MZ" + b"\x00" * 30, True),
        (PE, b"PK\x03\x04", False),
        (APK, b"PK\x03\x04" + b"\x00" * 12, True),
        (APK, b"PK\x05\x06", True),
        (APK, b"\x7fELF", False),
        (ELF, b"\x7fELFrest", True),
        (ELF, b"MZ", False),
        (PDF, b"%PDF-1.7\n", True),
        (PDF, b"PDF", False),  # no leading %
    ],
)
def test_magic_matching(family, head, expected):
    assert family.matches(head) is expected


def test_resolve_family_normalises_case_and_rejects_unknown():
    assert resolve_family("pe") is PE
    assert resolve_family("APK") is APK
    with pytest.raises(ValueError):
        resolve_family("dylib")


def test_load_samples_by_family_directory_apk(tmp_path):
    good = tmp_path / "a.apk"
    good.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
    bad = tmp_path / "b.txt"
    bad.write_bytes(b"not an apk file at all")
    out = load_samples_by_family(tmp_path, APK)
    # Both live in the root so both are considered; only the good one loads.
    kinds = {(s.get("path") or "").split("/")[-1]: s for s in out}
    assert kinds["a.apk"]["bytes"] is not None
    assert kinds["b.txt"]["bytes"] is None
    assert kinds["b.txt"]["error"] == "not_apk"


def test_load_samples_by_family_single_file_pe(tmp_path):
    p = tmp_path / "sample.exe"
    p.write_bytes(b"MZ" + b"\x00" * 30)
    out = load_samples_by_family(p, PE)
    assert len(out) == 1
    assert out[0]["bytes"] is not None


def test_load_samples_by_family_bytes_literal_apk_ok_and_bad():
    ok = load_samples_by_family(b"PK\x03\x04" + b"\x00" * 20, APK)
    assert ok[0]["bytes"] is not None
    bad = load_samples_by_family(b"garbage" * 4, APK)
    assert bad[0]["bytes"] is None
    assert bad[0]["error"] == "not_apk"


def test_load_samples_by_family_missing_file(tmp_path):
    out = load_samples_by_family(tmp_path / "nope.pdf", PDF)
    assert out[0]["bytes"] is None
    assert out[0]["error"] == "file_missing"


def test_load_samples_by_family_none_returns_empty():
    assert load_samples_by_family(None, PE) == []
