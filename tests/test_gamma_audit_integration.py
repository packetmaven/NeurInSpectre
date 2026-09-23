"""Audit wiring for --enable-gamma-sections (secml optional)."""

import pytest
from click.testing import CliRunner

from neurinspectre.cli.audit_cmd import build_audit_config, build_audit_report
from neurinspectre.malware.gamma_section import gamma_secml_status


def test_build_audit_config_records_gamma_flags():
    cfg = build_audit_config(
        target="ember2024-gbdt",
        n_examples=2,
        smoke=True,
        pe_sample="/tmp/pe",
        enable_gamma_sections=True,
        gamma_donor_dir="/tmp/donors",
        gamma_sections_per_population=3,
    )
    audit = cfg["audit"]
    assert audit.get("gamma_sections_enabled") is True
    assert audit.get("gamma_donor_dir") == "/tmp/donors"
    assert audit.get("gamma_sections_per_population") == 3
    scope = audit.get("measurement_scope") or {}
    assert "gamma_section_injection" not in scope.get("not_measured_ids", [])


def test_audit_report_reflects_gamma_scope_from_config():
    config = {
        "audit": {
            "target": "ember2024-gbdt",
            "gamma_sections_enabled": True,
            "same_sample_result": {
                "feature_vs_problem_space": {"n": 0},
                "extractor": {"official_reproduction": True},
            },
        }
    }
    report = build_audit_report({"results": [{"attacks": {}}]}, config=config)
    ms = report.get("measurement_scope") or {}
    assert ms.get("gamma_sections_enabled") is True
    assert "gamma_section_injection" not in ms.get("not_measured_ids", [])


@pytest.mark.skipif(
    not gamma_secml_status().get("available"),
    reason="secml-malware not installed",
)
def test_audit_help_lists_gamma_flags():
    from neurinspectre.cli.main import cli

    r = CliRunner().invoke(cli, ["audit", "--help"])
    assert r.exit_code == 0
    assert "--enable-gamma-sections" in r.output
    assert "--gamma-donor-dir" in r.output


@pytest.mark.skipif(
    not gamma_secml_status().get("available"),
    reason="secml-malware not installed",
)
def test_gamma_readiness_without_smoke_still_requires_donor():
    from neurinspectre.malware.gamma_env import gamma_readiness

    rep = gamma_readiness(donor_dir="/nonexistent/path", run_smoke_inject=False)
    assert rep.get("ready") is False
