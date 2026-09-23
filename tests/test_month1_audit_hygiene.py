from __future__ import annotations

import torch
import torch.nn as nn

from neurinspectre.attacks.autoattack import AutoAttack, _square_for_norm
from neurinspectre.attacks.base_interface import AttackConfig
from neurinspectre.attacks.factory import AttackFactory, _resolve_bpda_approximation, _to_attack_config
from neurinspectre.attacks.official_aa import BPDAWrappedModel, IdentityDefenseAdapter, official_aa_norm
from neurinspectre.attacks.square import SquareAttack, SquareAttackL2
from neurinspectre.characterization.defense_analyzer import DefenseCharacterization, ObfuscationType
from neurinspectre.cli.audit_cmd import build_audit_config, build_audit_report
from neurinspectre.cli.table2_cmd import _normalize_table2_spec
from neurinspectre.cli.utils import evaluate_attack_runner


class _Tiny(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, n_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def test_inrepo_aa_standard_includes_apgd_t_and_square_l2():
    model = _Tiny()
    linf = AutoAttack(model, norm="linf", eps=8 / 255, version="standard", device="cpu")
    assert "apgd-t" in linf.attacks
    assert isinstance(linf.attacks["square"], SquareAttack)

    l2 = AutoAttack(model, norm="l2", eps=0.5, version="standard", device="cpu")
    assert "square" in l2.attacks
    assert isinstance(l2.attacks["square"], SquareAttackL2)
    assert isinstance(_square_for_norm(model, eps=0.5, norm="l2", n_queries=10, device="cpu"), SquareAttackL2)


def test_inrepo_aa_records_gradient_skip_fields():
    class _Broken(nn.Module):
        def forward(self, x):
            return torch.zeros(x.size(0), 10)

    model = _Broken()
    aa = AutoAttack(model, norm="linf", eps=8 / 255, version="standard", device="cpu")
    x = torch.rand(2, 3, 8, 8)
    y = torch.zeros(2, dtype=torch.long)
    _x_adv, metrics = aa.run(x, y, verbose=False)
    assert "aa_subattacks_skipped" in metrics
    assert "gradient_unavailable" in metrics
    assert "aa_all_gradient_skipped" in metrics


def test_bpda_default_is_defense_not_identity():
    cfg = AttackConfig()
    assert cfg.bpda_approximation == "defense"
    parsed = _to_attack_config({})
    assert parsed.bpda_approximation == "defense"

    class _Def:
        def get_bpda_approximation(self):
            return lambda x: x * 0.5

    approx = _resolve_bpda_approximation(_Def(), parsed)
    x = torch.ones(1)
    assert torch.allclose(approx(x), x * 0.5)
    identity_cfg = _to_attack_config({"bpda_approximation": "identity"})
    assert torch.allclose(_resolve_bpda_approximation(_Def(), identity_cfg)(x), x)


def test_table2_passes_attack_type_through():
    raw = {
        "defaults": {"evaluation": {"batch_size": 8}, "apgd": {"steps": 100}},
        "datasets": {"content_moderation": {"backing_dataset": "cifar10", "split": "test"}},
        "attacks": {
            "aa_official": {"enabled": True, "type": "aa_official", "config_key": "apgd"},
            "aa_bpda": {"enabled": True, "type": "aa_bpda", "config_key": "apgd"},
            "pgd": {"enabled": True},
        },
        "defenses": [],
    }
    resolved = _normalize_table2_spec(raw, strict_dataset_budgets=True)
    by_name = {a["name"]: a for a in resolved["attacks"]}
    assert by_name["aa_official"]["type"] == "aa_official"
    assert by_name["aa_bpda"]["type"] == "aa_bpda"
    assert by_name["pgd"]["type"] == "pgd"


def test_table2_carmon_model_passthrough():
    raw = {
        "defaults": {"evaluation": {"batch_size": 8}},
        "datasets": {"content_moderation": {"backing_dataset": "cifar10", "split": "test"}},
        "attacks": {"pgd": {"enabled": True}},
        "defenses": [
            {
                "id": "cm_carmon2019",
                "dataset": "content_moderation",
                "model": {
                    "model_name": "Carmon2019Unlabeled",
                    "training_type": "robustbench",
                    "path": "models/cifar10/Linf/Carmon2019Unlabeled.pt",
                },
                "defense": {"type": "none"},
            }
        ],
    }
    resolved = _normalize_table2_spec(raw, strict_dataset_budgets=True)
    d0 = resolved["defenses"][0]
    assert d0["type"] == "none"
    assert d0["model"]["loader"] == "carmon2019"
    assert d0["model"]["model_name"] == "Carmon2019Unlabeled"


def test_characterization_to_dict_exports_chosen_attack():
    char = DefenseCharacterization(
        obfuscation_types=[ObfuscationType.SHATTERED],
        etd_score=0.1,
        alpha_volterra=0.5,
        gradient_variance=0.0,
        jacobian_rank=1.0,
        autocorr_timescale=0.0,
        requires_bpda=True,
        requires_eot=False,
        requires_mapgd=False,
        recommended_eot_samples=1,
        recommended_memory_length=1,
        confidence=0.9,
        metadata={},
    )
    char.chosen_attack = "bpda"
    char.selected_attack_impl = "BPDA"
    payload = char.to_dict()
    assert payload["chosen_attack"] == "bpda"
    assert payload["selected_attack_impl"] == "BPDA"


def test_neurinspectre_records_chosen_attack_on_identity():
    model = _Tiny()
    runner = AttackFactory.create_attack(
        "neurinspectre",
        model,
        config={"n_iterations": 2, "epsilon": 8 / 255, "norm": "linf"},
        device="cpu",
    )
    assert runner.chosen_attack == "apgd"
    assert runner.selected_attack_impl
    x = torch.rand(2, 3, 8, 8)
    y = torch.zeros(2, dtype=torch.long)
    result = runner.run(x, y)
    assert result.metadata["chosen_attack"] == "apgd"


def test_aa_bpda_inrepo_backend_and_cost_logs():
    model = _Tiny()
    defense = IdentityDefenseAdapter(model)
    runner = AttackFactory.create_attack(
        "aa_bpda",
        model,
        config={"backend": "inrepo", "version": "rand", "n_iterations": 1, "epsilon": 8 / 255},
        defense=defense,
        device="cpu",
    )
    x = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        y = model(x).argmax(1)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=2)
    summary = evaluate_attack_runner(runner, defense, loader, num_samples=2, device="cpu")
    assert "cost" in summary
    assert "wall_time_s" in summary["cost"]
    assert "forward_passes" in summary["cost"]
    assert summary.get("backend") == "inrepo"
    assert summary.get("chosen_attack") == "aa_bpda"


def test_official_aa_norm_and_bpda_wrap_is_differentiable():
    assert official_aa_norm("linf") == "Linf"
    assert official_aa_norm("l2") == "L2"
    model = _Tiny()
    defense = IdentityDefenseAdapter(model)
    wrapped = BPDAWrappedModel(defense)
    x = torch.rand(2, 3, 8, 8, requires_grad=True)
    logits = wrapped(x)
    logits.sum().backward()
    assert x.grad is not None


def test_factory_official_aa_uses_fra31(monkeypatch):
    class _FakeAA:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

        def run_standard_evaluation(self, x, y, bs=None):
            return x

    import neurinspectre.attacks.official_aa as official_aa

    monkeypatch.setattr(official_aa, "import_official_autoattack", lambda: _FakeAA)
    model = _Tiny()
    runner = AttackFactory.create_attack(
        "aa_official",
        model,
        config={"version": "custom", "attacks_to_run": ["apgd-ce"], "epsilon": 8 / 255},
        device="cpu",
    )
    x = torch.rand(2, 3, 8, 8)
    y = torch.zeros(2, dtype=torch.long)
    result = runner.run(x, y)
    assert result.metadata["backend"] == "fra31"
    assert result.metadata["chosen_attack"] == "aa_official"


def test_audit_config_and_report_shape():
    cfg = build_audit_config(target="jpeg-carmon", n_examples=8, smoke=True)
    assert cfg["defenses"][0]["type"] == "jpeg_compression"
    assert cfg["defenses"][0]["params"]["quality"] == 75
    names = [a["name"] for a in cfg["attacks"]]
    assert names[:3] == ["aa_official", "aa_bpda", "neurinspectre"]
    assert cfg["attacks"][0]["version"] == "custom"

    summary = {
        "results": [
            {
                "defense": "cm_carmon2019_jpeg",
                "type": "jpeg_compression",
                "dataset": "cifar10",
                "characterization": {"requires_bpda": True, "chosen_attack": "bpda"},
                "attacks": {
                    "aa_official": {
                        "clean_accuracy": 0.9,
                        "attack_success_rate": 0.1,
                        "robust_accuracy": 0.8,
                        "cost": {"wall_time_s": 1.0, "forward_passes": 10},
                        "validity": {"enabled": True, "passed": True},
                    },
                    "neurinspectre": {"chosen_attack": "bpda", "attack_success_rate": 0.4},
                },
            }
        ],
        "timing": {"total_seconds": 1.0},
    }
    report = build_audit_report(summary, config=cfg)
    assert report["kind"] == "neurinspectre_audit"
    assert report["chosen_attack"] == "bpda"
    assert report["target"] == "jpeg-carmon"
    assert "aa_official" in report["attacks"]
