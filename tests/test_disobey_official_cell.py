"""The Disobey draft must keep the two EMBER 2018 cells distinct."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFP = ROOT / "results/offensive_overnight_20260920/DISOBEY2027_CFP.md"
SLIDES = ROOT / "results/offensive_overnight_20260920/build_disobey_pptx.py"


def test_draft_names_the_official_cell_and_the_august_cell():
    text = CFP.read_text()
    assert "will not claim production-AV evasion, official EMBER2018 reproduction" not in text
    assert "n=58, 500 queries" in text
    assert "LIEF 0.13.2" in text
    assert "0/56" in text
    assert "FeatureSquare was not run" in text
    assert "0.531" in text
    assert "PE 0/4, Win32 0/4, Win64 0/4" in text
    assert "PE 0/6" not in text
    assert text.count("Win32 2/6") == text.count("old Win32 2/6")


def test_slide_source_does_not_deny_the_official_cell():
    text = SLIDES.read_text()
    assert "not official EMBER2018" not in text
    assert "official EMBER2018 cell is LIEF 0.9.0" in text
    assert "The August LIEF 0.13.2 run is not official: 0/56 at 5000 queries." in text
    assert "problem-valid ASR 0.0" in text
    assert "FeatureSquare was not run." in text
    assert "PE 0/4, Win32 0/4, Win64 0/4" in text
    assert "The old Win32 2/6 was not a crossing." in text
    assert "PE 0/6" not in text
    assert text.count("Win32 2/6") == text.count("old Win32 2/6")
