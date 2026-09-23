"""IAT transforms, VT read-only sidecar, operator bundle zip, SOW adapters."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from neurinspectre.cli.main import cli
from neurinspectre.evaluation.redteam_bundle import collect_bundle_files, zip_audit_bundle
from neurinspectre.malware.iat_transforms import (
    apply_iat_case_toggle,
    apply_iat_edit,
    import_feature_l1,
    imports_feature_changed,
    probe_iat_primitives,
    text_section_unchanged,
)
from neurinspectre.malware.measurement_scope import build_measurement_scope
from neurinspectre.malware.pe_transforms import evaluate_transform_validity
from neurinspectre.malware.sow_adapters import (
    ADAPTER_COMMERCIAL_AV_EDR,
    ADAPTER_SANDBOX_HANDOFF,
    export_sandbox_handoff,
    run_sow_adapters,
    validate_adapters,
)
from neurinspectre.malware.vt_sidecar import (
    build_vt_sidecar_from_pe_dir,
    compact_vt_for_sample,
    load_vt_sidecar,
    parse_detection_ratio,
)
from neurinspectre.malware.pe_fixtures import build_minimal_pe

_MWB_SAMPLE = Path(
    "/Users/seren3/mwb/pe_only/f97ec11d8c19b36ad546267302c840838e025092b44ea6ef93fa8e2fb34b9d3c.exe"
)


def _pe_with_imports() -> bytes | None:
    if _MWB_SAMPLE.is_file():
        return _MWB_SAMPLE.read_bytes()
    return None


def test_parse_detection_ratio_formats():
    assert parse_detection_ratio("12/70") == {"detected": 12, "total": 70}
    assert parse_detection_ratio({"detected": 3, "total": 10}) == {"detected": 3, "total": 10}
    assert parse_detection_ratio("nope") is None


def test_iat_dll_case_is_thrember_noop_on_real_pe():
    raw = _pe_with_imports()
    if raw is None:
        pytest.skip("no local PE with imports")
    mutated, _ = apply_iat_case_toggle(raw, seed=1)
    assert import_feature_l1(raw, mutated) == 0.0
    assert not imports_feature_changed(raw, mutated)


def test_iat_api_case_moves_import_features():
    raw = _pe_with_imports()
    if raw is None:
        pytest.skip("no local PE with imports")
    mutated, meta = apply_iat_edit(raw, seed=42)
    assert meta.get("iat_mode") == "api_case"
    assert import_feature_l1(raw, mutated) > 0.0
    assert text_section_unchanged(raw, mutated)
    gate = evaluate_transform_validity(raw, mutated, kind="iat_edit")
    assert gate.get("passed") is True


def test_iat_case_toggle_mocked_offset():
    pe = bytearray(256)
    pe[80:92] = b"KERNEL32.dll"
    with patch(
        "neurinspectre.malware.iat_transforms._import_dll_name_offsets",
        return_value=[(80, b"KERNEL32.dll")],
    ):
        out, meta = apply_iat_case_toggle(bytes(pe), seed=0, import_index=0)
    assert meta["dll_before"] == "KERNEL32.dll"
    assert meta["dll_after"] != meta["dll_before"]
    assert len(out) == len(pe)
    assert out[80:92] != pe[80:92]


def test_iat_probe_cli_on_real_pe(tmp_path):
    raw = _pe_with_imports()
    if raw is None:
        pytest.skip("no local PE with imports")
    runner = CliRunner()
    tmp = tmp_path / "s.exe"
    tmp.write_bytes(raw)
    result = runner.invoke(cli, ["iat-probe", str(tmp)])
    assert result.exit_code == 0
    data = json.loads(result.output)
    api_row = next(r for r in data["primitives"] if r["primitive"] == "api_case")
    assert api_row["ok"] is True
    assert api_row["import_feature_l1"] > 0


def test_sow_adapter_does_not_erase_sandbox_engagement_gap():
    scope = build_measurement_scope(
        "ember2024-gbdt",
        sow_adapters_enabled=[ADAPTER_SANDBOX_HANDOFF],
    )
    ids = scope["not_measured_ids"]
    assert "sandbox_execution" in ids
    assert "sandbox_handoff_export" in scope["measured"]


def test_vt_sidecar_build_and_load(tmp_path):
    pe = tmp_path / "sample.exe"
    pe.write_bytes(build_minimal_pe())
    payload = build_vt_sidecar_from_pe_dir(tmp_path)
    assert payload["read_only"] is True
    assert payload["no_live_submit"] is True
    assert payload["n_files"] == 1
    sidecar_path = tmp_path / "vt.json"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_vt_sidecar(sidecar_path)
    sha = list(payload["records"].keys())[0]
    assert sha in loaded
    compact = compact_vt_for_sample(loaded[sha])
    assert compact is not None
    assert compact["read_only"] is True


def test_sow_sandbox_handoff_without_ack(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    best = audit / "best_bytes"
    best.mkdir()
    (best / "abc.mutated.bin").write_bytes(b"MZtest")
    report = {
        "same_sample_detail": {
            "best_bytes_manifest": [
                {
                    "path": str(best / "abc.mutated.bin"),
                    "sha256_original": "a" * 64,
                    "clean_p_malware": 0.9,
                    "best_p_malware": 0.4,
                }
            ]
        }
    }
    (audit / "audit_report.json").write_text(json.dumps(report), encoding="utf-8")
    summary = export_sandbox_handoff(audit)
    assert summary["n_copied"] == 1
    assert (audit / "sandbox_handoff" / "README_HANDOFF.txt").is_file()
    run_sow_adapters(audit, [ADAPTER_SANDBOX_HANDOFF])
    assert (audit / "sow_adapter_results.json").is_file()


def test_sow_commercial_av_requires_ack(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "audit_report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError):
        validate_adapters([ADAPTER_COMMERCIAL_AV_EDR], ack=False)
    run_sow_adapters(
        audit,
        [ADAPTER_COMMERCIAL_AV_EDR],
        ack=True,
        av_system_name="ExampleEDR",
    )
    data = json.loads((audit / "sow_adapter_results.json").read_text(encoding="utf-8"))
    assert data["adapters"][0]["system_name"] == "ExampleEDR"
    assert data["adapters"][0]["executed_in_cli"] is False


def test_zip_audit_bundle(tmp_path):
    audit = tmp_path / "out"
    audit.mkdir()
    (audit / "audit_report.json").write_text('{"ok": true}', encoding="utf-8")
    (audit / "pe_scope.json").write_text('{"n_mz_files": 1}', encoding="utf-8")
    zpath = zip_audit_bundle(audit)
    assert zpath.is_file()
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
    assert "audit_report.json" in names
    assert "bundle_manifest.json" in names
    assert len(collect_bundle_files(audit)) >= 2


def test_cli_help_lists_new_commands_and_audit_flags():
    runner = CliRunner()
    for cmd in ("vt-sidecar", "redteam-bundle", "iat-probe"):
        r = runner.invoke(cli, [cmd, "--help"])
        assert r.exit_code == 0
    r = runner.invoke(cli, ["audit", "--help"])
    assert r.exit_code == 0
    assert "--enable-iat-edits" in r.output
    assert "--vt-sidecar" in r.output
    assert "--sow-adapter" in r.output


def test_vt_sidecar_cli(tmp_path):
    pe = tmp_path / "x.exe"
    pe.write_bytes(build_minimal_pe())
    out = tmp_path / "vt_sidecar.json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["vt-sidecar", str(pe.parent), "-o", str(out)],
    )
    assert result.exit_code == 0
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["kind"] == "vt_sidecar"


def test_probe_iat_primitives_structure():
    raw = _pe_with_imports()
    if raw is None:
        pytest.skip("no local PE")
    report = probe_iat_primitives(raw)
    assert report["n_api_sites"] > 0
    assert report["n_dll_sites"] > 0
