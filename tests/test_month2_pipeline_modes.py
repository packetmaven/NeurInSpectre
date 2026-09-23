from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neurinspectre.attacks.factory import AttackFactory
from neurinspectre.attacks.square import SquareAttack
from neurinspectre.cli.audit_cmd import build_audit_config, characterize_audit_pipeline
from neurinspectre.cli.utils import _jsonable
from neurinspectre.defenses.wrappers import JPEGCompressionDefense
from neurinspectre.evaluation.problem_space import compute_asr_query_curve, evaluate_pe_parse
from neurinspectre.pipelines import SecurityPipeline


class _Tiny(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, n_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def test_pipeline_from_jpeg_wrapper_routes_bpda():
    model = _Tiny()
    defense = JPEGCompressionDefense(model, quality=75, device="cpu")
    pipeline = SecurityPipeline.from_defense(defense, device="cpu")
    char = pipeline.characterize()
    assert char["requires_bpda"] is True
    assert char["recommended_recipe"] == "bpda"
    assert any(s["kind"] == "ingest" for s in char["stages"])
    assert any(s["kind"] == "classifier" for s in char["stages"])


def test_identity_pipeline_recommends_apgd():
    pipeline = SecurityPipeline.identity(_Tiny(), device="cpu")
    char = pipeline.characterize()
    assert char["recommended_recipe"] == "apgd"
    assert char["requires_bpda"] is False
    assert char["requires_problem_space"] is False


def test_square_init_query_is_counted():
    class _Flip(nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.size(0), 10, device=x.device)
            logits[:, 1] = 5.0
            return logits

    attack = SquareAttack(_Flip(), eps=8 / 255, n_queries=1000, device="cpu")
    x = torch.rand(2, 3, 8, 8)
    y = torch.zeros(2, dtype=torch.long)
    _adv, stats = attack(x, y, verbose=False)
    assert (stats["queries_used"] == 1).all()
    assert stats["success"].all()


def test_label_square_short_budget_only_when_allowed():
    model = _Tiny()
    with pytest.raises(ValueError, match=r">=1000"):
        SquareAttack(model, eps=8 / 255, n_queries=50, loss_type="label", device="cpu")
    attack = SquareAttack(
        model,
        eps=8 / 255,
        n_queries=50,
        loss_type="label",
        device="cpu",
        allow_short_budget=True,
    )
    x = torch.rand(2, 3, 8, 8)
    y = torch.zeros(2, dtype=torch.long)
    x_adv, stats = attack(x, y, verbose=False)
    assert x_adv.shape == x.shape
    assert stats["loss_type"] == "label"
    assert stats["query_floor_relaxed"] is True


def test_default_square_factory_keeps_query_floor():
    model = _Tiny()
    with pytest.raises(ValueError, match=r">=1000"):
        AttackFactory.create_attack("square", model, config={"n_queries": 50}, device="cpu")


def test_factory_scores_and_labels_short_budget():
    model = _Tiny()
    scores = AttackFactory.create_attack(
        "scores",
        model,
        config={"n_queries": 50, "epsilon": 8 / 255},
        device="cpu",
    )
    labels = AttackFactory.create_attack(
        "labels",
        model,
        config={"n_queries": 50, "epsilon": 8 / 255},
        device="cpu",
    )
    assert scores.chosen_attack == "scores"
    assert labels.chosen_attack == "labels"
    x = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        y = model(x).argmax(1)
    s_result = scores.run(x, y)
    l_result = labels.run(x, y)
    assert s_result.metadata["access"] == "scores"
    assert l_result.metadata["access"] == "labels"
    assert s_result.metadata["query_floor_relaxed"] is True
    assert l_result.metadata["query_floor_relaxed"] is True


def test_asr_query_curve_math():
    curve = compute_asr_query_curve(
        queries=[20, 80, 200],
        success=[True, True, False],
        budgets=[50, 100, 500],
    )
    by_q = {row["query_budget"]: row for row in curve}
    assert by_q[50]["asr"] == pytest.approx(1 / 3)
    assert by_q[100]["asr"] == pytest.approx(2 / 3)
    assert by_q[500]["asr"] == pytest.approx(2 / 3)
    assert by_q[500]["successes"] == 2


def test_pe_parse_missing_sample_and_invalid_bytes():
    missing = evaluate_pe_parse(None)
    assert missing["kind"] == "pe_parse"
    assert "no_pe_sample" in missing["reasons"]
    assert missing["passed"] is None

    missing_path = evaluate_pe_parse("/no/such/neurinspectre_pe.exe")
    assert missing_path["passed"] in {False, None}
    assert missing_path["reasons"]

    invalid = evaluate_pe_parse(b"not a pe file")
    if invalid.get("available"):
        assert invalid["passed"] is False
        assert any(r in {"pe_parse_failed", "pe_parse_incomplete"} for r in invalid["reasons"])
    else:
        assert "pefile_not_installed" in invalid["reasons"]


def test_jsonable_converts_numpy_arrays():
    import numpy as np

    payload = _jsonable({"queries_used": np.array([1, 2], dtype=np.int64), "ok": True})
    assert payload["queries_used"] == [1, 2]


def test_audit_modes_and_pipeline_in_config():
    smoke_all = build_audit_config(target="jpeg-carmon", n_examples=8, smoke=True)
    names = [a["name"] for a in smoke_all["attacks"]]
    assert names[:3] == ["aa_official", "aa_bpda", "neurinspectre"]
    assert "scores" in names
    assert "labels" in names
    assert smoke_all["query_budgets"] == [10, 25, 50]
    scores = next(a for a in smoke_all["attacks"] if a["name"] == "scores")
    assert scores["n_queries"] == 50
    assert scores["allow_short_budget"] is True

    whitebox = build_audit_config(
        target="carmon",
        n_examples=8,
        smoke=True,
        mode="whitebox",
    )
    wb_names = [a["name"] for a in whitebox["attacks"]]
    assert "scores" not in wb_names
    assert "labels" not in wb_names

    jpeg_pipe = characterize_audit_pipeline("jpeg-carmon")
    assert jpeg_pipe["recommended_recipe"] == "bpda"
    ident = characterize_audit_pipeline("carmon")
    assert ident["recommended_recipe"] == "apgd"
