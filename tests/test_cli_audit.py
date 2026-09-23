import shutil
import subprocess
import sys

from click.testing import CliRunner

from neurinspectre.cli.main import cli


def test_audit_is_a_top_level_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "audit" in result.output


def test_audit_help_exposes_ember_same_sample_flags():
    runner = CliRunner()
    result = runner.invoke(cli, ["audit", "--help"])
    assert result.exit_code == 0
    assert "ember-gbdt" in result.output
    assert "--pe-sample" in result.output
    assert "--benign-corpus" in result.output
    assert "--query-budgets" in result.output
    assert "--mode" in result.output
    assert "--require-official-rep" in result.output
    assert "--require-detected" in result.output
    assert "--crossing-matrix" in result.output
    assert "--capa-diff-best" in result.output
    assert "--write-diagnosis" in result.output
    assert "--enable-gamma-sections" in result.output
    assert "--enable-iat-edits" in result.output
    assert "--vt-sidecar" in result.output


def test_measurement_frame_commands_registered():
    from neurinspectre.cli.main import _CLICK_COMMANDS

    for name in (
        "scope-pe-corpus",
        "diagnose-ember-audit",
        "capa-diff-audit",
        "ember-pipeline-info",
        "engagement-gaps",
        "gamma-inject",
        "vt-sidecar",
        "redteam-bundle",
        "iat-probe",
    ):
        assert name in _CLICK_COMMANDS


def test_config_audit_embeds_measurement_scope_for_ember():
    from neurinspectre.cli.audit_cmd import build_audit_config

    cfg = build_audit_config(
        target="ember2024-gbdt",
        n_examples=4,
        smoke=True,
        pe_sample="/tmp/pe_corpus",
    )
    audit = cfg.get("audit") or {}
    assert audit.get("measurement_scope")
    assert audit.get("pipeline")
    assert "not_measured" in (audit.get("measurement_scope") or {})


def test_require_official_reproduction_fails_on_mac_or_unverified_lief():
    from neurinspectre.malware.ember_extract import official_reproduction

    assert official_reproduction(platform_name="Darwin", lief_version="0.9.0-") is False
    assert official_reproduction(platform_name="Linux", lief_version="0.9.0-") is True
    assert official_reproduction(platform_name="Linux", lief_version="0.10.1") is True
    assert official_reproduction(platform_name="Linux", lief_version="0.13.2") is False
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["audit", "--target", "ember-gbdt", "--smoke", "--require-official-reproduction"],
    )
    assert result.exit_code != 0
    assert "not Elastic-verified" in (result.output + (result.exception and str(result.exception) or ""))


def test_require_detected_needs_pe_sample():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["audit", "--target", "ember-gbdt", "--smoke", "--require-detected"],
    )
    assert result.exit_code != 0
    assert "--pe-sample" in (result.output + (result.exception and str(result.exception) or ""))


def test_audit_ember_pe_sample_builds_same_sample_config():
    from neurinspectre.cli.audit_cmd import build_audit_config

    cfg = build_audit_config(
        target="ember-gbdt",
        n_examples=4,
        smoke=True,
        pe_sample="/tmp/pe_corpus",
        benign_corpus="/tmp/benign_corpus",
    )
    assert cfg["attacks"] == []
    assert cfg["audit"]["same_sample"] is True
    assert cfg["audit"]["pe_sample"] == "/tmp/pe_corpus"
    assert cfg["audit"]["benign_corpus"] == "/tmp/benign_corpus"
    assert cfg["audit"]["threat_model"] == "malware_evasion"


def test_config_audit_is_a_cli_command():
    runner = CliRunner()
    help_result = runner.invoke(cli, ["config", "--help"])
    assert help_result.exit_code == 0
    assert "audit" in help_result.output
    result = runner.invoke(
        cli,
        ["config", "audit", "--target", "ember-gbdt", "--smoke", "--pe-sample", "/tmp/pe"],
    )
    assert result.exit_code == 0, result.output
    assert "ember-gbdt" in result.output
    assert "same_sample: true" in result.output
    assert "/tmp/pe" in result.output
    assert "name: feature_square" not in result.output


def test_python_module_and_console_script_expose_audit():
    for argv in (
        [sys.executable, "-m", "neurinspectre", "audit", "--help"],
        [sys.executable, "-m", "neurinspectre.cli", "audit", "--help"],
    ):
        proc = subprocess.run(argv, check=False, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert "ember-gbdt" in proc.stdout
        assert "--pe-sample" in proc.stdout
    console = shutil.which("neurinspectre")
    if console:
        proc = subprocess.run([console, "audit", "--help"], check=False, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert "ember-gbdt" in proc.stdout
