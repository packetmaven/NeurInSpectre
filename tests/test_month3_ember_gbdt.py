from __future__ import annotations

import struct

import pytest
import torch
import torch.nn as nn

from neurinspectre.attacks.factory import AttackFactory
from neurinspectre.attacks.feature_square import FeatureSquareAttack
from neurinspectre.attacks.problem_space_pe import OverlayAppendAttack, evaluate_overlay_corpus
from neurinspectre.attacks.square import SquareAttack
from neurinspectre.cli.audit_cmd import build_audit_config, characterize_audit_pipeline
from neurinspectre.evaluation.problem_space import evaluate_pe_functionality, overlay_append
from neurinspectre.pipelines import SecurityPipeline


class _TinyFeat(nn.Module):
    def __init__(self, n_features: int = 16, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.fc(x)


class _AlwaysWrong(nn.Module):
    def forward(self, x):
        # 10-class: predict class 1 for every sample.
        return torch.zeros(x.size(0), 10, device=x.device).index_fill_(1, torch.tensor(1, device=x.device), 10.0)


def _minimal_pe() -> bytes:
    """Tiny PE32 that pefile can parse. Test fixture, not a program."""
    dos = bytearray(64)
    dos[0:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 64)
    pe = bytearray()
    pe += b"PE\x00\x00"
    pe += struct.pack("<HHIIIHH", 0x14C, 1, 0, 0, 0, 0xE0, 0x0102)
    # Optional header PE32
    opt = bytearray(224)
    struct.pack_into("<H", opt, 0, 0x10B)
    struct.pack_into("<I", opt, 16, 0x1000)  # AddressOfEntryPoint
    struct.pack_into("<I", opt, 28, 0x400000)  # ImageBase
    struct.pack_into("<I", opt, 32, 0x1000)  # SectionAlignment
    struct.pack_into("<I", opt, 36, 0x200)  # FileAlignment
    struct.pack_into("<H", opt, 40, 4)
    struct.pack_into("<H", opt, 42, 0)
    struct.pack_into("<H", opt, 48, 4)
    struct.pack_into("<I", opt, 56, 0x2000)  # SizeOfImage
    struct.pack_into("<I", opt, 60, 0x200)  # SizeOfHeaders
    struct.pack_into("<H", opt, 68, 3)  # Subsystem
    struct.pack_into("<H", opt, 92, 16)  # NumberOfRvaAndSizes
    pe += opt
    # One section
    sec = bytearray(40)
    sec[0:5] = b".text"
    struct.pack_into("<I", sec, 8, 0x200)
    struct.pack_into("<I", sec, 12, 0x1000)
    struct.pack_into("<I", sec, 16, 0x200)
    struct.pack_into("<I", sec, 20, 0x200)
    struct.pack_into("<I", sec, 36, 0x60000020)
    pe += sec
    file_bytes = bytes(dos) + bytes(pe)
    file_bytes = file_bytes.ljust(0x400, b"\x00")
    return file_bytes


def test_square_counts_init_query_and_init_success():
    model = _AlwaysWrong()
    attack = SquareAttack(model, eps=8 / 255, n_queries=1000, device="cpu")
    x = torch.rand(2, 3, 8, 8)
    y = torch.zeros(2, dtype=torch.long)
    _x_adv, stats = attack(x, y, verbose=False)
    assert stats["success"].all()
    assert (stats["queries_used"] == 1).all()


def test_feature_square_2d_and_unrealizable():
    model = _TinyFeat()
    attack = FeatureSquareAttack(
        model, eps=1.0, n_queries=50, loss_type="margin", device="cpu", allow_short_budget=True
    )
    x = torch.randn(3, 16)
    y = torch.zeros(3, dtype=torch.long)
    x_adv, stats = attack(x, y)
    assert x_adv.shape == x.shape
    assert stats["realizable"] is False
    assert stats["space"] == "feature"
    assert (x_adv - x).abs().max() <= 1.0 + 1e-5


def test_feature_square_rejects_images_and_short_budget():
    model = _TinyFeat()
    with pytest.raises(ValueError, match=r">=1000"):
        FeatureSquareAttack(model, n_queries=50, device="cpu")
    attack = FeatureSquareAttack(model, n_queries=50, device="cpu", allow_short_budget=True)
    with pytest.raises(ValueError, match="2D"):
        attack(torch.rand(1, 3, 8, 8), torch.zeros(1, dtype=torch.long))


def test_pgd_marks_missing_gradients():
    class _NoGrad(nn.Module):
        def forward(self, x):
            v = x.detach().cpu().numpy().mean(axis=1)
            s = torch.as_tensor(v, device=x.device, dtype=x.dtype)
            return torch.stack([-s, s], dim=1)

    runner = AttackFactory.create_attack(
        "pgd",
        _NoGrad(),
        config={"n_iterations": 2, "epsilon": 1.0},
        device="cpu",
    )
    x = torch.randn(2, 8)
    y = torch.zeros(2, dtype=torch.long)
    result = runner.run(x, y)
    assert result.metadata.get("gradient_unavailable") is True


def test_factory_feature_square():
    model = _TinyFeat()
    runner = AttackFactory.create_attack(
        "feature_square",
        model,
        config={"n_queries": 50, "epsilon": 1.0},
        device="cpu",
    )
    x = torch.randn(2, 16)
    y = torch.zeros(2, dtype=torch.long)
    result = runner.run(x, y)
    assert result.metadata["space"] == "feature"
    assert result.metadata["realizable"] is False
    assert result.metadata["chosen_attack"] == "feature_square"


def test_overlay_append_keeps_original_prefix():
    pe = _minimal_pe()
    mutated = overlay_append(pe, b"AAAA")
    assert mutated.startswith(pe)
    assert mutated.endswith(b"AAAA")
    report = evaluate_pe_functionality(pe, mutated, expect_overlay_only=True)
    if report.get("before", {}).get("available") is False:
        pytest.skip("pefile not installed")
    assert report["passed"] is True
    broken = evaluate_pe_functionality(pe, b"notpe", expect_overlay_only=True)
    assert broken["passed"] is False


def test_overlay_attack_records_extractor_gap():
    pe = _minimal_pe()

    def _no_extract(_data):
        return {"available": False, "features": None, "reasons": ["ember_extractor_unavailable"]}

    row = OverlayAppendAttack(model=None, extractor=_no_extract).run_bytes(pe, y=1)
    assert row["gamma"] is False
    assert row["scored"] is False
    assert row["space"] == "problem"
    corpus = evaluate_overlay_corpus([{"path": "mem", "bytes": pe}], extractor=_no_extract)
    assert corpus["n_scored"] == 0
    assert corpus["kind"] == "problem_space_overlay"


def test_ember_pipeline_recommends_problem_space():
    pipe = SecurityPipeline.from_ember_gbdt(nn.Identity(), device="cpu")
    char = pipe.characterize()
    assert char["requires_problem_space"] is True
    assert char["recommended_recipe"] == "problem_space"
    kinds = [s["kind"] for s in char["stages"]]
    assert kinds == ["ingest", "features", "classifier", "threshold"]


def test_ember_audit_config_is_malware_evasion():
    cfg = build_audit_config(target="ember-gbdt", n_examples=8, smoke=True)
    assert cfg["datasets"]["ember"]["filter_label"] == 1
    assert cfg["perturbation"]["epsilon"] == 1.0
    names = [a["name"] for a in cfg["attacks"]]
    assert "feature_square" in names
    assert "aa_official" not in names
    assert cfg["defenses"][0]["model"]["loader"] == "ember_gbdt"
    pipe = characterize_audit_pipeline("ember-gbdt")
    assert pipe["recommended_recipe"] == "problem_space"


def test_load_pe_samples_recurses_exe_suffix(tmp_path):
    from pathlib import Path

    from neurinspectre.attacks.problem_space_pe import load_pe_samples
    from neurinspectre.malware.pe_fixtures import build_minimal_pe

    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "sample.exe").write_bytes(build_minimal_pe())
    (tmp_path / "readme.txt").write_text("not a pe")
    rows = load_pe_samples(tmp_path)
    names = [Path(r["path"]).name for r in rows if r.get("path")]
    assert "sample.exe" in names
    readme = next(r for r in rows if r.get("path") and Path(r["path"]).name == "readme.txt")
    assert readme.get("bytes") is None
    assert readme.get("error") == "not_a_valid_pe"
    nested_only = load_pe_samples(tmp_path / "nested")
    assert len(nested_only) == 1
    assert nested_only[0]["bytes"][:2] == b"MZ"


def test_ember_audit_pe_sample_skips_unpaired_memmap():
    cfg = build_audit_config(
        target="ember-gbdt",
        n_examples=4,
        smoke=True,
        pe_sample="/tmp/pe_corpus",
    )
    assert cfg["attacks"] == []
    assert cfg["audit"]["same_sample"] is True


def test_fulldos_preserves_pe_image_and_e_lfanew():
    from neurinspectre.malware.pe_fixtures import build_minimal_pe
    from neurinspectre.malware.pe_transforms import (
        apply_fulldos,
        evaluate_transform_validity,
        fulldos_capacity,
        read_e_lfanew,
    )

    pe = build_minimal_pe(e_lfanew=0x80, dos_fill=0x41)
    assert fulldos_capacity(pe) > 0
    e0 = read_e_lfanew(pe)
    mutated = apply_fulldos(pe, b"\x00" * 200)
    assert mutated[:2] == b"MZ"
    assert read_e_lfanew(mutated) == e0
    assert mutated[e0:] == pe[e0:]
    assert mutated[2:0x3C] != pe[2:0x3C]
    report = evaluate_transform_validity(pe, mutated, kind="fulldos")
    if report.get("before", {}).get("available") is False:
        pytest.skip("pefile not installed")
    assert report["passed"] is True
    broken = bytearray(mutated)
    broken[e0 : e0 + 2] = b"XX"
    bad = evaluate_transform_validity(pe, bytes(broken), kind="fulldos")
    assert bad["passed"] is False
    assert "pe_image_rewritten" in bad["reasons"]


def test_same_sample_pairs_feature_and_problem_space():
    import numpy as np

    from neurinspectre.evaluation.ember_same_sample import evaluate_ember_same_sample
    from neurinspectre.malware.pe_fixtures import build_minimal_pe

    class _DosScore(nn.Module):
        def forward(self, x):
            # Malware only if the DOS-derived feature stays very high.
            s = x[:, 0] - 0.8
            return torch.stack([-s, s], dim=1)

    def _extract(data: bytes):
        # High DOS-fill → positive mean → predicted malware.
        chunk = data[2:0x80]
        feat = np.array([sum(chunk) / (len(chunk) * 255.0 + 1e-6)] * 8, dtype=np.float32)
        return {"available": True, "features": feat, "reasons": []}

    malware_like = build_minimal_pe(e_lfanew=0x80, dos_fill=0xFF)
    benign_like = build_minimal_pe(e_lfanew=0x80, dos_fill=0x00)
    report = evaluate_ember_same_sample(
        [malware_like, malware_like, benign_like],
        _DosScore(),
        n_queries=40,
        feature_eps=1.0,
        query_budgets=[10, 25, 40],
        seed=0,
        payload_size=64,
        extractor=_extract,
    )
    assert report["same_sample"] is True
    assert report["n_detected_malware"] == 2
    assert report["n_skipped"] == 1
    cmp_ = report["feature_vs_problem_space"]
    assert cmp_["n"] == 2
    assert cmp_["feature_space_realizable"] is False
    assert cmp_["feature_space_asr"] is not None
    assert cmp_["problem_space_valid_asr"] is not None
    # Full DOS zero-fill lowers the DOS mean and should flip the fixture model.
    assert cmp_["problem_space_valid_asr"] == pytest.approx(1.0)
    assert cmp_["same_sample"] is True


def test_same_sample_max_samples_is_detected_count():
    import numpy as np

    from neurinspectre.evaluation.ember_same_sample import evaluate_ember_same_sample
    from neurinspectre.malware.pe_fixtures import build_minimal_pe

    class _DosScore(nn.Module):
        def forward(self, x):
            s = x[:, 0] - 0.8
            return torch.stack([-s, s], dim=1)

    def _extract(data: bytes):
        chunk = data[2:0x80]
        feat = np.array([sum(chunk) / (len(chunk) * 255.0 + 1e-6)] * 8, dtype=np.float32)
        return {"available": True, "features": feat, "reasons": []}

    benign = build_minimal_pe(e_lfanew=0x80, dos_fill=0x00)
    malware = build_minimal_pe(e_lfanew=0x80, dos_fill=0xFF)
    report = evaluate_ember_same_sample(
        [benign, benign, malware, malware, malware],
        _DosScore(),
        n_queries=8,
        seed=0,
        extractor=_extract,
        max_samples=1,
    )
    assert report["n_detected_malware"] == 1
    assert report["n_scanned"] == 3
    assert report["stopped_early"] is True
    assert report["feature_vs_problem_space"]["n"] == 1


def test_gamma_padding_uses_benign_bytes_and_keeps_prefix():
    from neurinspectre.attacks.problem_space_pe import ProblemSpacePESearch
    from neurinspectre.malware.pe_fixtures import build_minimal_pe
    from neurinspectre.malware.pe_transforms import evaluate_transform_validity

    malware = build_minimal_pe(e_lfanew=0x80, dos_fill=0xFF)
    benign = build_minimal_pe(e_lfanew=0x80, dos_fill=0x11)
    marker = b"BENIGNMARK"
    benign = benign + marker

    class _AlwaysMalware(nn.Module):
        def forward(self, x):
            return torch.tensor([[-1.0, 1.0]]).expand(x.size(0), 2)

    def _extract(data: bytes):
        import numpy as np

        return {"available": True, "features": np.ones(8, dtype=np.float32), "reasons": []}

    search = ProblemSpacePESearch(
        _AlwaysMalware(),
        n_queries=8,
        payload_size=16,
        seed=0,
        benign_payloads=[benign],
        extractor=_extract,
    )
    result = search.run_bytes(malware, y=1)
    assert result["gamma"] is True or result.get("chosen_attack") in {"fulldos", "gamma_padding", "clean"}
    # Direct transform: padding from benign must keep the original prefix.
    from neurinspectre.malware.pe_transforms import apply_padding

    mutated = apply_padding(malware, marker)
    assert mutated.startswith(malware)
    assert mutated.endswith(marker)
    gate = evaluate_transform_validity(malware, mutated, kind="gamma_padding")
    if gate.get("before", {}).get("available") is False:
        pytest.skip("pefile not installed")
    assert gate["passed"] is True


def test_official_extractor_gap_does_not_invent_features():
    import numpy as np

    from neurinspectre.malware.ember_extract import extract_ember_features, extractor_status
    from neurinspectre.malware.pe_fixtures import build_minimal_pe

    pe = build_minimal_pe()
    status = extractor_status()
    out = extract_ember_features(pe)
    if not status.get("available"):
        assert out.get("features") is None
        assert "ember_extractor_unavailable" in (out.get("reasons") or [])
        return
    if out.get("features") is None:
        assert out.get("reasons")
        return
    assert int(np.asarray(out["features"]).size) == 2381
    assert status.get("official_reproduction") is False or str(status.get("lief_version") or "").startswith("0.9.0")


def test_same_sample_skips_non_pe_even_if_scored_malware():
    import numpy as np

    from neurinspectre.evaluation.ember_same_sample import evaluate_ember_same_sample

    class _AlwaysMalware(nn.Module):
        def forward(self, x):
            return torch.tensor([[-1.0, 4.0]]).expand(x.size(0), 2)

    def _extract(_data: bytes):
        return {"available": True, "features": np.ones(8, dtype=np.float32), "reasons": []}

    report = evaluate_ember_same_sample(
        [b"not a pe at all"],
        _AlwaysMalware(),
        n_queries=4,
        seed=0,
        extractor=_extract,
    )
    assert report["n_extracted"] == 0
    assert report["n_detected_malware"] == 0
    assert report["skip_reasons"].get("not_a_valid_pe") == 1


def test_same_sample_skips_nonfinite_features():
    import numpy as np

    from neurinspectre.evaluation.ember_same_sample import evaluate_ember_same_sample
    from neurinspectre.malware.pe_fixtures import build_minimal_pe

    class _Model(nn.Module):
        def forward(self, x):
            return torch.zeros(x.size(0), 2)

    def _extract(_data: bytes):
        return {"available": True, "features": np.array([np.nan] * 8, dtype=np.float32), "reasons": []}

    report = evaluate_ember_same_sample(
        [build_minimal_pe(e_lfanew=0x80, dos_fill=0xFF)],
        _Model(),
        n_queries=4,
        seed=0,
        extractor=_extract,
    )
    assert report["n_detected_malware"] == 0
    assert report["skip_reasons"].get("ember_features_nonfinite") == 1


def test_official_extractor_fixture_is_not_detected_malware():
    from neurinspectre.evaluation.ember_same_sample import evaluate_ember_same_sample
    from neurinspectre.malware.ember_extract import extractor_status
    from neurinspectre.malware.pe_fixtures import build_minimal_pe
    from neurinspectre.models.ember_gbdt import EmberGBDT
    from pathlib import Path

    if not extractor_status().get("available"):
        pytest.skip("official ember extractor unavailable")
    gbdt_path = Path("data/ember/ember2018/ember_model_2018.txt")
    if not gbdt_path.is_file():
        pytest.skip("official EMBER GBDT missing")
    report = evaluate_ember_same_sample(
        [build_minimal_pe(e_lfanew=0x80, dos_fill=0xFF)],
        EmberGBDT.from_file(gbdt_path),
        n_queries=4,
        seed=0,
        max_samples=1,
    )
    assert report["n_extracted"] == 1
    assert report["n_detected_malware"] == 0
    assert report["skip_reasons"].get("not_detected_as_malware") == 1
    assert report["feature_vs_problem_space"]["feature_space_asr"] is None
    assert report["extractor"].get("official_reproduction") is False


def test_audit_report_keeps_extractor_and_skip_reasons():
    from neurinspectre.cli.audit_cmd import build_audit_report

    same = {
        "feature_vs_problem_space": {
            "n": 0,
            "same_sample": True,
            "feature_space_asr": None,
            "problem_space_valid_asr": None,
            "feature_space_realizable": False,
        },
        "feature_space": {"attack_success_rate": None, "realizable": False},
        "problem_space": {"valid_success_rate": None},
        "n_listed": 2,
        "n_scanned": 2,
        "n_extracted": 0,
        "n_detected_malware": 0,
        "n_skipped": 2,
        "stopped_early": False,
        "skip_reasons": {"ember_extractor_unavailable": 2},
        "extractor": {
            "available": False,
            "reasons": ["ember_extractor_unavailable"],
            "official_reproduction": False,
        },
        "samples": [{"kept": False, "reason": "ember_extractor_unavailable"}],
    }
    report = build_audit_report(
        {"results": [{"defense": "md_ember2018_gbdt", "type": "none", "attacks": {}}]},
        config={"audit": {"target": "ember-gbdt", "same_sample": True, "same_sample_result": same}},
    )
    assert report["same_sample"] is True
    assert report["same_sample_detail"]["skip_reasons"]["ember_extractor_unavailable"] == 2
    assert report["same_sample_detail"]["extractor"]["available"] is False
    assert report["feature_vs_problem_space"]["n"] == 0
    assert report["official_reproduction"] is False
    assert report["quote_as_ember2018"] is False
