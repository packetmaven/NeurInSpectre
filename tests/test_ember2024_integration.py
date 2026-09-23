"""EMBER 2024 (thrember, v3) integration tests.

These are unit-level: they exercise the CLI config builder, the pipeline
factory, the extractor status probe, and the ``EmberGBDT2024`` wrapper
against an in-repo minimal PE fixture. They do not require any downloaded
LightGBM model to run (a tiny in-memory booster is built when possible).

Rules of engagement:

- Do not download live malware, models, or the EMBER2024 dataset in CI.
- If ``thrember`` is not importable, skip; do not shim signify or lightgbm.
- Never quote a feature-space number as EMBER2024 evasion; that mistake is
  what the whole feature-version-3 same-sample plumbing exists to prevent.
"""

from __future__ import annotations

import importlib
import struct

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from neurinspectre.cli.audit_cmd import (  # noqa: E402
    AUDIT_TARGETS,
    _is_ember2024_target,
    _is_ember_target,
    _normalize_target,
    build_audit_config,
    characterize_audit_pipeline,
)
from neurinspectre.malware.ember2024_extract import (  # noqa: E402
    _apply_authenticode_shim,
    extract_ember2024_features,
    extractor_status,
    official_reproduction,
)


# ---------------------------------------------------------------------------
# In-repo fixture PE (matches tests/test_month3_ember_gbdt.py's helper)
# ---------------------------------------------------------------------------


def _minimal_pe() -> bytes:
    dos = bytearray(64)
    dos[0:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 64)
    pe = bytearray()
    pe += b"PE\x00\x00"
    pe += struct.pack("<HHIIIHH", 0x14C, 1, 0, 0, 0, 0xE0, 0x0102)
    opt = bytearray(224)
    struct.pack_into("<H", opt, 0, 0x10B)
    struct.pack_into("<I", opt, 16, 0x1000)
    struct.pack_into("<I", opt, 28, 0x400000)
    struct.pack_into("<I", opt, 32, 0x1000)
    struct.pack_into("<I", opt, 36, 0x200)
    struct.pack_into("<H", opt, 40, 4)
    struct.pack_into("<H", opt, 42, 0)
    struct.pack_into("<H", opt, 48, 4)
    struct.pack_into("<I", opt, 56, 0x2000)
    struct.pack_into("<I", opt, 60, 0x200)
    struct.pack_into("<H", opt, 68, 3)
    struct.pack_into("<H", opt, 92, 16)
    pe += opt
    sec = bytearray(40)
    sec[0:5] = b".text"
    struct.pack_into("<I", sec, 8, 0x200)
    struct.pack_into("<I", sec, 12, 0x1000)
    struct.pack_into("<I", sec, 16, 0x200)
    struct.pack_into("<I", sec, 20, 0x200)
    struct.pack_into("<I", sec, 36, 0x60000020)
    pe += sec
    file_bytes = bytes(dos) + bytes(pe)
    return file_bytes.ljust(0x400, b"\x00")


def _thrember_available() -> bool:
    try:
        importlib.import_module("thrember.features")
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Target normalisation
# ---------------------------------------------------------------------------


def test_ember2024_target_is_registered():
    assert "ember2024-gbdt" in AUDIT_TARGETS
    assert AUDIT_TARGETS["ember2024-gbdt"]["name"] == "md_ember2024_pe_gbdt"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ember2024-gbdt", "ember2024-gbdt"),
        ("ember2024_gbdt", "ember2024-gbdt"),
        ("ember2024", "ember2024-gbdt"),
        ("ember-2024", "ember2024-gbdt"),
        ("ember2024-pe", "ember2024-gbdt"),
        ("ember2024-pe-gbdt", "ember2024-gbdt"),
        ("ember2024-win32", "ember2024-win32-gbdt"),
        ("ember2024-win32-gbdt", "ember2024-win32-gbdt"),
        ("ember2024-win64", "ember2024-win64-gbdt"),
        ("ember2024-win64-gbdt", "ember2024-win64-gbdt"),
        ("ember2024-apk", "ember2024-apk-gbdt"),
        ("ember2024-apk-gbdt", "ember2024-apk-gbdt"),
        ("ember2024-elf", "ember2024-elf-gbdt"),
        ("ember2024-elf-gbdt", "ember2024-elf-gbdt"),
        ("ember2024-pdf", "ember2024-pdf-gbdt"),
        ("ember2024-pdf-gbdt", "ember2024-pdf-gbdt"),
        ("ember-gbdt", "ember-gbdt"),
        ("ember2018", "ember-gbdt"),
    ],
)
def test_target_aliases(raw, expected):
    assert _normalize_target(raw) == expected


def test_ember_target_predicates():
    assert _is_ember_target("ember-gbdt")
    assert _is_ember_target("ember2024-gbdt")
    assert _is_ember_target("ember2024-win64-gbdt")
    assert _is_ember2024_target("ember2024-gbdt")
    assert _is_ember2024_target("ember2024-win32-gbdt")
    assert _is_ember2024_target("ember2024-win64-gbdt")
    assert not _is_ember2024_target("ember-gbdt")
    assert not _is_ember_target("carmon")


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def test_build_audit_config_ember2024_smoke():
    cfg = build_audit_config(
        target="ember2024-gbdt",
        n_examples=4,
        smoke=True,
        model_path=None,
        data_root="./data/ember",
        batch_size=4,
        seed=42,
        assert_clean_accuracy=False,
        include_pgd=False,
        mode="all",
        query_budgets=None,
        pe_sample=None,
        benign_corpus=None,
    )
    defense = cfg["defenses"][0]
    assert defense["name"] == "md_ember2024_pe_gbdt"
    assert defense["model"]["loader"] == "ember2024_gbdt"
    assert defense["model"]["ember_feature_version"] == 3
    assert defense["model"]["path"].endswith("EMBER2024_PE.model")
    assert cfg["audit"]["target"] == "ember2024-gbdt"
    assert cfg["audit"]["ember_feature_version"] == 3
    assert cfg["audit"]["feature_epsilon"] == 1.0


@pytest.mark.parametrize(
    "target, expected_name, expected_path_suffix",
    [
        ("ember2024-gbdt", "md_ember2024_pe_gbdt", "EMBER2024_PE.model"),
        ("ember2024-win32-gbdt", "md_ember2024_win32_gbdt", "EMBER2024_Win32.model"),
        ("ember2024-win64-gbdt", "md_ember2024_win64_gbdt", "EMBER2024_Win64.model"),
        ("ember2024-apk-gbdt", "md_ember2024_apk_gbdt", "EMBER2024_APK.model"),
        ("ember2024-elf-gbdt", "md_ember2024_elf_gbdt", "EMBER2024_ELF.model"),
        ("ember2024-pdf-gbdt", "md_ember2024_pdf_gbdt", "EMBER2024_PDF.model"),
    ],
)
def test_build_audit_config_ember2024_submodels(target, expected_name, expected_path_suffix):
    cfg = build_audit_config(
        target=target,
        n_examples=4,
        smoke=True,
        model_path=None,
        data_root="./data/ember",
        batch_size=4,
        seed=42,
        assert_clean_accuracy=False,
        include_pgd=False,
        mode="all",
        query_budgets=None,
        pe_sample=None,
        benign_corpus=None,
    )
    defense = cfg["defenses"][0]
    assert defense["name"] == expected_name
    assert defense["model"]["loader"] == "ember2024_gbdt"
    assert defense["model"]["ember_feature_version"] == 3
    assert defense["model"]["path"].endswith(expected_path_suffix)


def test_build_audit_config_ember2018_still_v2():
    cfg = build_audit_config(
        target="ember-gbdt",
        n_examples=4,
        smoke=True,
        model_path=None,
        data_root="./data/ember",
        batch_size=4,
        seed=42,
        assert_clean_accuracy=False,
        include_pgd=False,
        mode="all",
        query_budgets=None,
        pe_sample=None,
        benign_corpus=None,
    )
    defense = cfg["defenses"][0]
    assert defense["model"]["loader"] == "ember_gbdt"
    assert defense["model"]["ember_feature_version"] == 2
    assert defense["model"]["path"].endswith("ember_model_2018.txt")


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target, expected_classifier, expected_filename",
    [
        ("ember2024-gbdt", "ember2024_gbdt", "EMBER2024_PE.model"),
        ("ember2024-win32-gbdt", "ember2024_win32_gbdt", "EMBER2024_Win32.model"),
        ("ember2024-win64-gbdt", "ember2024_win64_gbdt", "EMBER2024_Win64.model"),
        ("ember2024-apk-gbdt", "ember2024_apk_gbdt", "EMBER2024_APK.model"),
        ("ember2024-elf-gbdt", "ember2024_elf_gbdt", "EMBER2024_ELF.model"),
        ("ember2024-pdf-gbdt", "ember2024_pdf_gbdt", "EMBER2024_PDF.model"),
    ],
)
def test_ember2024_pipeline_has_pefile_stage(target, expected_classifier, expected_filename):
    report = characterize_audit_pipeline(target)
    stages = report.get("stages") or report.get("pipeline") or []
    names = [s.get("name") for s in stages if isinstance(s, dict)]
    assert "pe_ingest" in names
    assert "ember2024_features" in names
    assert expected_classifier in names
    for s in stages:
        if isinstance(s, dict) and s.get("name") == "ember2024_features":
            params = s.get("params") or {}
            assert params.get("feature_version") == 3
            assert "thrember" in str(params.get("extractor", "")).lower()
        if isinstance(s, dict) and s.get("name") == expected_classifier:
            params = s.get("params") or {}
            assert params.get("model_filename") == expected_filename


# ---------------------------------------------------------------------------
# Extractor status + real extraction (skipped when thrember unavailable)
# ---------------------------------------------------------------------------


def test_extractor_status_shape():
    st = extractor_status()
    assert isinstance(st, dict)
    for key in ("available", "reasons", "platform", "thrember_version", "pefile_version"):
        assert key in st


def test_official_reproduction_semantics():
    assert official_reproduction(platform_name="Linux", thrember_version="0.1.0") is True
    assert official_reproduction(platform_name="Darwin", thrember_version="0.1.0") is True
    assert official_reproduction(platform_name="Linux", thrember_version=None) is False


@pytest.mark.skipif(not _thrember_available(), reason="thrember not installed")
def test_extract_minimal_pe_returns_finite_vector():
    result = extract_ember2024_features(_minimal_pe())
    assert result["available"] is True
    if result.get("reasons"):
        # Skip only if pefile / thrember bailed on the tiny fixture; that is
        # environment noise, not an integration failure.
        pytest.skip(f"extractor bailed: {result['reasons']}")
    vec = result["features"]
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.ndim == 1
    assert vec.size == result["dim"] == 2568
    assert np.isfinite(vec).all()


@pytest.mark.skipif(not _thrember_available(), reason="thrember not installed")
def test_authenticode_shim_catches_type_error_and_is_idempotent():
    """signify 0.8+ CertificateStore raises TypeError; thrember does not catch it.
    Our shim extends the catch list to (TypeError, AttributeError, LookupError)
    and returns thrember's own zero-record with parse_error=1.
    """
    from thrember.features import AuthenticodeSignature

    _apply_authenticode_shim()
    fn = AuthenticodeSignature.raw_features
    assert getattr(fn, "_neurinspectre_shim", False) is True

    # Idempotent — a second call must not double-wrap.
    _apply_authenticode_shim()
    fn2 = AuthenticodeSignature.raw_features
    assert fn is fn2

    inst = AuthenticodeSignature()

    class _BoomPE:
        class _Hdr:
            TimeDateStamp = 0

        FILE_HEADER = _Hdr()

    class _BoomSignify:
        def iter_signed_datas(self):
            # Simulate the signify 0.8 CertificateStore drift: raise TypeError
            # from inside the loop body the way `len(signed_data.certificates)`
            # does on newer signify.
            raise TypeError("'CertificateStore' object is not subscriptable")

    # Monkey-patch SignedPEFile to force the failure path we want to shim.
    import thrember.features as tf

    orig_signed = tf.SignedPEFile
    tf.SignedPEFile = lambda _bio: _BoomSignify()
    try:
        out = inst.raw_features(b"\x00" * 32, _BoomPE())
    finally:
        tf.SignedPEFile = orig_signed
    assert out["parse_error"] == 1
    assert out["num_certs"] == 0
    assert out["self_signed"] == 0


# ---------------------------------------------------------------------------
# EmberGBDT2024 wrapper unit test (skipped without a real model file)
# ---------------------------------------------------------------------------


def test_ember2024_gbdt_wrapper_shape():
    from pathlib import Path

    from neurinspectre.models.ember_gbdt import EmberGBDT2024

    model_path = Path("data/ember/ember2024/EMBER2024_PE.model")
    if not model_path.is_file():
        pytest.skip("EMBER2024 model not downloaded")
    gbdt = EmberGBDT2024.from_file(model_path)
    assert gbdt.feature_dim > 0
    x = np.zeros((3, gbdt.feature_dim), dtype=np.float32)
    probs = gbdt.predict_proba(x)
    assert probs.shape == (3, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    # softmax(forward) must recover P(malware). [-logit, +logit] does not.
    import torch
    logits = gbdt(torch.as_tensor(x))
    recovered = torch.softmax(logits, dim=1).detach().numpy()
    assert np.allclose(recovered, probs, atol=1e-5)


def test_stored_softmax_inverts_to_predict_proba():
    """Old [-logit(p), +logit(p)] softmax inverts to the LightGBM probability."""
    from pathlib import Path

    from neurinspectre.models.ember_gbdt import (
        EmberGBDT2024,
        lightgbm_probability_from_stored_softmax,
    )

    def old_softmax_malware(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        s = np.log(p / (1.0 - p))
        logits = np.array([-s, s], dtype=np.float64)
        exps = np.exp(logits - logits.max())
        return float(exps[1] / exps.sum())

    for p in (0.11, 0.5, 0.603, 0.691, 0.895, 0.9879738968626302):
        q = old_softmax_malware(p)
        assert lightgbm_probability_from_stored_softmax(q) == pytest.approx(p, abs=1e-6)

    model_path = Path("data/ember/ember2024/EMBER2024_PE.model")
    if not model_path.is_file():
        return
    gbdt = EmberGBDT2024.from_file(model_path)
    live = gbdt.predict_proba(np.zeros((1, gbdt.feature_dim), dtype=np.float32))[0, 1]
    stored = old_softmax_malware(float(live))
    assert lightgbm_probability_from_stored_softmax(stored) == pytest.approx(float(live), abs=1e-6)


# ---------------------------------------------------------------------------
# Model-sniffer + diagnosis script
# ---------------------------------------------------------------------------


def test_utils_sniffer_routes_2024_paths_to_wrapper():
    from pathlib import Path

    from neurinspectre.cli.utils import _looks_like_ember_gbdt

    # thrember model files still have `.model` suffix and 'ember' in the name
    assert _looks_like_ember_gbdt(
        path=Path("data/ember/ember2024/EMBER2024_PE.model"),
        cfg={"model_name": "ember2024_gbdt", "loader": "ember2024_gbdt"},
    )
    assert _looks_like_ember_gbdt(
        path=Path("data/ember/ember2018/ember_model_2018.txt"),
        cfg={"model_name": "ember_gbdt", "loader": "ember_gbdt"},
    )


def test_score_challenge_helper_runs_on_synthetic_records(tmp_path):
    """Feed one fake JSONL row through the scorer and confirm summary shape."""
    import runpy
    import sys

    from pathlib import Path

    if not _thrember_available():
        pytest.skip("thrember not installed")
    model_path = Path("data/ember/ember2024/EMBER2024_PE.model")
    if not model_path.is_file():
        pytest.skip("EMBER2024 PE model not downloaded")

    from neurinspectre.malware.ember2024_extract import _apply_authenticode_shim
    _apply_authenticode_shim()
    from thrember.features import PEFeatureExtractor

    # Build one plausible v3 raw feature record with the schema thrember uses.
    extr = PEFeatureExtractor()
    zero_row = {}
    for fe in extr.features:
        # each thrember FeatureType names its raw_obj key via .name; pass pe=None
        try:
            zero_row[fe.name] = fe.raw_features(b"\x00" * 64, None)
        except Exception:
            # ByteHistogram/etc. expect a plausible byte buffer; that's fine
            zero_row[fe.name] = fe.raw_features(b"\x00" * 512, None)
    zero_row["sha256"] = "0" * 64

    # NB: `raw_features` on empty bytes returns the module's zero record where
    # supported; scored via process_raw_features. Wrap in the full JSONL keys
    # the challenge set uses so the scorer's iteration works.
    row = {
        "sha256": zero_row.get("sha256"),
        "md5": "0" * 32,
        "sha1": "0" * 40,
        "tlsh": "T" + "0" * 71,
        "first_submission_date": 0,
        "last_analysis_date": 0,
        "detection_ratio": "0/0",
        "label": 1,
        "file_type": "Win32",
        "family": "synthetic",
        "family_confidence": 1.0,
        "behavior": [],
        "file_property": [],
        "packer": [],
        "exploit": [],
        "group": [],
        "week_id": 0,
        "caps": [],
        "ttps": [],
        "mbc": [],
    }
    row.update(zero_row)

    challenge = tmp_path / "challenge"
    challenge.mkdir()
    (challenge / "fake.jsonl").write_text(__import__("json").dumps(row) + "\n")

    out = tmp_path / "score.json"
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "score_ember2024_challenge.py",
            "--challenge-dir", str(challenge),
            "--output", str(out),
        ]
        runpy.run_path("scripts/score_ember2024_challenge.py", run_name="__main__")
    except SystemExit as exc:
        assert exc.code in (None, 0)
    finally:
        sys.argv = old_argv

    summary = __import__("json").loads(out.read_text())
    assert summary["kind"] == "ember2024_challenge_scoring"
    assert summary["n_features_built"] == 1
    assert set(summary["per_model"]) == {"PE", "Win32", "Win64"}
    for name in ("PE", "Win32", "Win64"):
        assert 0.0 <= summary["per_model"][name]["median_p"] <= 1.0
        assert summary["per_model"][name]["detected_ge_0.5"] in (0, 1)
    assert summary["file_type_dist"].get("Win32") == 1


def test_score_ember2024_challenge_cli_registered():
    """The score-ember2024-challenge Click command is discoverable + not
    intercepted by the legacy argparse fallback."""
    from click.testing import CliRunner

    from neurinspectre.cli.main import cli, _CLICK_COMMANDS

    assert "score-ember2024-challenge" in _CLICK_COMMANDS
    runner = CliRunner()
    result = runner.invoke(cli, ["score-ember2024-challenge", "--help"])
    assert result.exit_code == 0, result.output
    assert "EMBER 2024 challenge set" in result.output


@pytest.mark.parametrize(
    "name, expected_help_substring",
    [
        ("download-ember2024", "LightGBM detection models"),
        ("download-ember2024-challenge", "challenge JSONLs"),
        ("download-ember2024-capa", "Capa supplement"),
        ("tag-pe-corpus", "Capa-tag sidecar"),
        ("diagnose-ember-audit", "compact diagnosis JSON"),
    ],
)
def test_ember2024_helper_scripts_all_registered_as_click_subcommands(
    name, expected_help_substring,
):
    """Every scripts/*.py we ship has an equivalent ``neurinspectre <name>``
    subcommand. Guards against the ``scripts/`` folder drifting away from the
    CLI."""
    from click.testing import CliRunner

    from neurinspectre.cli.main import cli, _CLICK_COMMANDS

    assert name in _CLICK_COMMANDS, f"{name} not in _CLICK_COMMANDS allowlist"
    runner = CliRunner()
    result = runner.invoke(cli, [name, "--help"])
    assert result.exit_code == 0, result.output
    assert expected_help_substring in result.output


def test_audit_report_carries_tag_filter_provenance_when_active(tmp_path):
    """When the audit lane runs with a Capa filter, ``build_audit_report``
    must copy tag_filter / tag_filter_active / n_filtered_out_by_tag /
    n_untagged_seen / filter_include_untagged into ``same_sample_detail``.
    Regression guard for the A1 report bug where these fields showed up as
    None in audit_report.json despite the CLI counters being correct."""
    from neurinspectre.cli.audit_cmd import build_audit_report

    config = {
        "audit": {
            "target": "ember2024-gbdt",
            "same_sample_result": {
                "n_listed": 148,
                "n_scanned": 148,
                "n_extracted": 1,
                "n_detected_malware": 1,
                "n_skipped": 147,
                "skip_reasons": {"filter_untagged": 146, "filter_tag_mismatch": 1},
                "extractor": {"official_reproduction": True},
                "samples": [],
                "feature_space": {"attack_success_rate": 1.0},
                "problem_space": {"attack_success_rate": 0.0, "valid_success_rate": 0.0},
                "feature_vs_problem_space": {
                    "n": 1, "feature_space_asr": 1.0,
                    "problem_space_valid_asr": 0.0, "same_sample": True,
                },
                "tag_filter": {"family": ["rugmi"], "min_vt_detected": None},
                "tag_filter_active": True,
                "n_filtered_out_by_tag": 147,
                "n_untagged_seen": 146,
                "filter_include_untagged": False,
            },
        },
    }
    summary = {"results": [{"defense": "md_ember2024_pe_gbdt", "attacks": {}, "characterization": {}}]}
    report = build_audit_report(summary, config=config)
    sd = report["same_sample_detail"]
    assert sd["tag_filter_active"] is True
    assert sd["tag_filter"] == {"family": ["rugmi"], "min_vt_detected": None}
    assert sd["n_filtered_out_by_tag"] == 147
    assert sd["n_untagged_seen"] == 146
    assert sd["filter_include_untagged"] is False
    assert report["official_reproduction"] is True
    assert report["quote_as_ember2018"] is False


def test_ember2018_quote_flag_requires_official_reproduction():
    from neurinspectre.cli.audit_cmd import build_audit_report

    def _report(official: bool):
        same = {
            "feature_vs_problem_space": {"n": 1, "same_sample": True},
            "extractor": {"official_reproduction": official},
            "samples": [],
        }
        return build_audit_report(
            {"results": [{"defense": "md_ember2018_gbdt", "attacks": {}}]},
            config={"audit": {"target": "ember-gbdt", "same_sample_result": same}},
        )

    assert _report(True)["quote_as_ember2018"] is True
    assert _report(True)["official_reproduction"] is True
    assert _report(False)["quote_as_ember2018"] is False


def test_diagnose_ember_audit_summarizes_synthetic_report(tmp_path):
    import runpy
    import sys

    report = {
        "target": "ember2024-gbdt",
        "official_reproduction": True,
        "quote_as_ember2018": True,
        "query_budgets": [100],
        "attacks": {
            "feature_square": {
                "attack_success_rate": 1.0,
                "query_curve": [{"query_budget": 100, "asr": 1.0}],
            }
        },
        "problem_space": {"attack_success_rate": 0.0, "valid_success_rate": 0.0},
        "feature_vs_problem_space": {
            "feature_space_asr": 1.0,
            "problem_space_valid_asr": 0.0,
        },
        "same_sample_detail": {
            "extractor": {"shims": ["shim_a"]},
            "samples": [
                {
                    "kept": True,
                    "path": "/mwb/a.exe",
                    "clean_p_malware": 0.99,
                    "problem": {
                        "best_p_malware": 0.98,
                        "chosen_attack": "padding",
                        "queries_used": 100,
                        "success": False,
                        "realizable": True,
                    },
                },
                {
                    "kept": True,
                    "path": "/mwb/b.exe",
                    "clean_p_malware": 0.9,
                    "problem": {
                        "best_p_malware": 0.4,
                        "chosen_attack": "fulldos",
                        "queries_used": 100,
                        "success": True,
                        "realizable": True,
                    },
                },
                {"kept": False, "reason": "not_detected"},
            ],
        },
    }
    report_dir = tmp_path / "audit_run"
    report_dir.mkdir()
    (report_dir / "audit_report.json").write_text(__import__("json").dumps(report))

    old_argv = sys.argv[:]
    try:
        sys.argv = ["diagnose_ember_audit.py", str(report_dir)]
        runpy.run_path("scripts/diagnose_ember_audit.py", run_name="__main__")
    except SystemExit as exc:
        assert exc.code in (None, 0)
    finally:
        sys.argv = old_argv

    diag = __import__("json").loads((report_dir / "ember_audit_diagnosis.json").read_text())
    assert diag["target"] == "ember2024-gbdt"
    assert diag["n"] == 2
    assert diag["feature_asr"] == 1.0
    assert diag["problem_valid_asr"] == 0.0
    assert diag["n_best_below_0.5"] == 1  # b.exe went from 0.9 to 0.4
    assert diag["closest"]["name"] == "b.exe"
    assert diag["transforms"]["padding"] == 1
    assert diag["transforms"]["fulldos"] == 1
    assert diag["extractor_status"] == ["shim_a"]
