"""GBDT audit measurement scope and pipeline characterization."""

from neurinspectre.cli.audit_cmd import build_audit_report, characterize_audit_pipeline
from neurinspectre.malware.measurement_scope import (
    NOT_MEASURED,
    build_measurement_scope,
    enrich_gbdt_pipeline_characterization,
)


def test_measurement_scope_lists_five_engagement_gaps():
    scope = build_measurement_scope("ember2024-gbdt")
    ids = {row["id"] for row in NOT_MEASURED}
    assert "sandbox_execution" in ids
    assert "commercial_av_or_edr" in ids
    assert "gamma_section_injection" in ids
    assert scope["frame"] == "named_model_byte_range_parse_gate_query_budget"


def test_ember_pipeline_characterization_is_problem_space_only():
    pipe = characterize_audit_pipeline("ember-gbdt")
    assert pipe["gradient_available"] is False
    assert pipe["recommended_recipe"] == "problem_space"
    assert "jpeg_bpda" in pipe["do_not_route_to"]
    assert pipe["measurement_scope"]["target"] == "ember-gbdt"


def test_build_audit_report_includes_measurement_scope_for_ember():
    config = {
        "audit": {
            "target": "ember-gbdt",
            "n_examples": 1,
            "smoke": True,
            "modes": ["problem"],
            "query_budgets": [10],
            "pipeline": characterize_audit_pipeline("ember-gbdt"),
            "same_sample_result": {
                "feature_vs_problem_space": {"n": 0},
                "extractor": {"official_reproduction": False},
            },
        }
    }
    report = build_audit_report({"results": [{"attacks": {}}]}, config=config)
    assert report["measurement_scope"] is not None
    assert report["measurement_scope"]["not_measured"]


def test_enrich_does_not_drop_stages():
    base = {"name": "ember_gbdt", "stages": [{"name": "x"}]}
    out = enrich_gbdt_pipeline_characterization(base, "ember-gbdt")
    assert out["stages"] == base["stages"]


def test_crossing_matrix_requires_save_best_bytes_fast():
    from click.testing import CliRunner
    from neurinspectre.cli.main import cli

    r = CliRunner().invoke(
        cli,
        [
            "audit",
            "--target",
            "ember2024-gbdt",
            "--pe-sample",
            "/nonexistent",
            "--crossing-matrix",
            "-o",
            "/tmp/audit_fail_fast",
        ],
    )
    assert r.exit_code != 0
    assert "save-best-bytes" in (r.output or "").lower()
