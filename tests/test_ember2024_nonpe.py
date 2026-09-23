"""Tests for the APK/ELF/PDF non-PE evaluation lane.

Coverage:

- Target predicates and normalisation for non-PE targets
- ``evaluate_ember2024_nonpe_same_sample`` on synthetic fixtures
- Report shape: problem_space column is deliberately null; feature_space
  runs; extract_ember2024_features called via a fake extractor
- Skip reason surfacing: non-matching magic → ``not_<family>`` skip
- Click choices accept new targets; audit_cmd knows which are non-PE
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from neurinspectre.cli.audit_cmd import (
    EMBER2024_NONPE_TARGETS,
    _EMBER2024_TARGET_TO_FAMILY,
    _is_ember2024_nonpe_target,
)
from neurinspectre.evaluation.ember_same_sample_nonpe import (
    evaluate_ember2024_nonpe_same_sample,
)
from neurinspectre.malware.file_families import APK, ELF, PDF, resolve_family


class _FakeGBDT(nn.Module):
    feature_dim = 2568

    def __init__(self, p_mal: float = 0.9):
        super().__init__()
        self.p_mal = float(p_mal)

    def predict_proba(self, x):
        n = x.shape[0]
        p1 = np.full(n, self.p_mal, dtype=np.float32)
        return np.stack([1 - p1, p1], axis=1)

    def forward(self, x):
        probs = self.predict_proba(x.detach().cpu().numpy())
        s = np.log(probs[:, 1]) - np.log(probs[:, 0])
        return torch.as_tensor(np.stack([-s, s], axis=1), device=x.device, dtype=x.dtype)


def _fake_extract(_bytes):
    vec = np.zeros(2568, dtype=np.float32)
    return {"available": True, "features": vec, "reasons": [], "dim": 2568,
            "extractor": {"available": True, "reasons": []}}


def test_nonpe_targets_registered():
    assert EMBER2024_NONPE_TARGETS == {
        "ember2024-apk-gbdt", "ember2024-elf-gbdt", "ember2024-pdf-gbdt",
        "ember2024-all-gbdt",
    }
    for t in EMBER2024_NONPE_TARGETS:
        assert _is_ember2024_nonpe_target(t)
    assert not _is_ember2024_nonpe_target("ember2024-gbdt")
    assert not _is_ember2024_nonpe_target("ember2024-dotnet-gbdt")  # PE lane
    assert not _is_ember2024_nonpe_target("ember-gbdt")


def test_nonpe_target_to_family_map():
    assert _EMBER2024_TARGET_TO_FAMILY["ember2024-apk-gbdt"] == "APK"
    assert _EMBER2024_TARGET_TO_FAMILY["ember2024-elf-gbdt"] == "ELF"
    assert _EMBER2024_TARGET_TO_FAMILY["ember2024-pdf-gbdt"] == "PDF"
    assert _EMBER2024_TARGET_TO_FAMILY["ember2024-all-gbdt"] == "ANY"
    for name in ("APK", "ELF", "PDF", "ANY"):
        assert resolve_family(name).name == name


@pytest.mark.parametrize(
    "family, good, bad",
    [
        (APK, b"PK\x03\x04" + b"\x00" * 64, b"not a zip archive"),
        (ELF, b"\x7fELF" + b"\x00" * 64, b"MZ" + b"\x00" * 62),  # PE-magic wrong for ELF
        (PDF, b"%PDF-1.7\n" + b"\x00" * 64, b"garbage"),
    ],
)
def test_nonpe_same_sample_pipeline_on_synthetic_bytes(family, good, bad, tmp_path):
    root = tmp_path / family.name.lower()
    root.mkdir()
    (root / f"good.{family.name.lower()}").write_bytes(good)
    (root / f"bad.txt").write_bytes(bad)

    model = _FakeGBDT(p_mal=0.9)
    result = evaluate_ember2024_nonpe_same_sample(
        root, model, family,
        n_queries=5, query_budgets=[5],
        extractor=_fake_extract,
    )
    assert result["kind"] == "ember2024_nonpe_same_sample"
    assert result["file_family"] == family.name
    # good file kept + detected; bad file dropped as not_<family>
    assert result["n_detected_malware"] == 1
    reasons = result["skip_reasons"]
    assert reasons.get(f"not_{family.name.lower()}") == 1
    # feature_space ran (numeric ASR); problem_space is intentionally null.
    fs = result["feature_space"]
    assert isinstance(fs["attack_success_rate"], float)
    assert 0.0 <= fs["attack_success_rate"] <= 1.0
    assert fs["realizable"] is False
    assert result["problem_space"]["attack_success_rate"] is None
    assert result["problem_space"]["valid_success_rate"] is None
    fvp = result["feature_vs_problem_space"]
    assert fvp["problem_space_asr"] is None
    assert fvp["problem_space_valid_asr"] is None


def test_nonpe_same_sample_tag_filter_applies(tmp_path):
    import hashlib

    data = b"PK\x03\x04" + b"\x00" * 32
    p = tmp_path / "sample.apk"
    p.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()

    from neurinspectre.malware.capa_filters import TagFilter

    flt = TagFilter(family=["rugmi"])
    good_sidecar = {sha: {"family": "rugmi"}}
    bad_sidecar = {sha: {"family": "wacatac"}}

    model = _FakeGBDT()

    # matches -> kept
    r = evaluate_ember2024_nonpe_same_sample(
        p.parent, model, APK,
        n_queries=5, query_budgets=[5],
        extractor=_fake_extract,
        tag_filter=flt, tags_by_sha256=good_sidecar,
    )
    assert r["n_detected_malware"] == 1

    # doesn't match -> filtered out
    r = evaluate_ember2024_nonpe_same_sample(
        p.parent, model, APK,
        n_queries=5, query_budgets=[5],
        extractor=_fake_extract,
        tag_filter=flt, tags_by_sha256=bad_sidecar,
    )
    assert r["n_detected_malware"] == 0
    assert r["n_filtered_out_by_tag"] == 1


def test_click_choices_include_new_targets():
    from click.testing import CliRunner
    from neurinspectre.cli.main import cli

    runner = CliRunner()
    help_text = runner.invoke(cli, ["audit", "--help"]).output
    for t in ("ember2024-apk-gbdt", "ember2024-elf-gbdt", "ember2024-pdf-gbdt",
              "ember2024-dotnet-gbdt", "ember2024-all-gbdt"):
        assert t in help_text


def test_dotnet_target_uses_pe_lane():
    """Dot_Net assemblies ARE PE files (MZ + CLR), so ember2024-dotnet-gbdt
    routes through the PE evaluator (Full DOS / overlay available)."""
    from neurinspectre.cli.audit_cmd import (
        _is_ember2024_target, _is_ember2024_nonpe_target, EMBER2024_PE_TARGETS,
    )
    assert _is_ember2024_target("ember2024-dotnet-gbdt")
    assert not _is_ember2024_nonpe_target("ember2024-dotnet-gbdt")
    assert "ember2024-dotnet-gbdt" in EMBER2024_PE_TARGETS


def test_all_target_uses_any_family_lane():
    """The universal ``all`` detector accepts every file type, so it goes
    through the non-PE (feature-space only) lane with the ANY family."""
    from neurinspectre.cli.audit_cmd import (
        _is_ember2024_nonpe_target, _EMBER2024_TARGET_TO_FAMILY,
    )
    assert _is_ember2024_nonpe_target("ember2024-all-gbdt")
    assert _EMBER2024_TARGET_TO_FAMILY["ember2024-all-gbdt"] == "ANY"


@pytest.mark.parametrize(
    "target, expected_classifier, expected_filename",
    [
        ("ember2024-dotnet-gbdt", "ember2024_dotnet_gbdt", "EMBER2024_Dot_Net.model"),
        ("ember2024-all-gbdt", "ember2024_all_gbdt", "EMBER2024_all.model"),
    ],
)
def test_pipeline_variants_dotnet_and_all(target, expected_classifier, expected_filename):
    from neurinspectre.cli.audit_cmd import characterize_audit_pipeline
    report = characterize_audit_pipeline(target)
    stages = report.get("stages") or report.get("pipeline") or []
    names = [s.get("name") for s in stages if isinstance(s, dict)]
    assert expected_classifier in names
    for s in stages:
        if isinstance(s, dict) and s.get("name") == expected_classifier:
            assert (s.get("params") or {}).get("model_filename") == expected_filename
