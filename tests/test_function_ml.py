"""Tests for E11 function-level ML pipeline over the Capa supplement.

Covers:

- Deterministic FNV-1a hash bucketing.
- ``_extract_mnemonics`` lowercase + tokenization.
- ``function_features`` unigram + bigram counting and scalar suffix.
- ``StratifiedSampler.consider`` / ``done`` quotas and end-to-end
  ``stratified_sample`` on a synthetic shard.
- ``split_by_sha256`` keeps functions from the same SHA-256 in one split.
- ``train_multilabel_functionml`` on a tiny synthetic dataset produces
  reasonable per-label metrics (recovery of a hand-designed signal).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from neurinspectre.malware.function_ml import (
    DEFAULT_FEATURE_DIM,
    StratifiedSampler,
    _bucket,
    _extract_mnemonics,
    _fnv1a_32,
    function_features,
    iter_function_records,
    split_by_sha256,
    stratified_sample,
    train_multilabel_functionml,
)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_fnv1a_deterministic():
    assert _fnv1a_32(b"") == 0x811C9DC5
    # Reference: FNV-1a of "abc" is 0x1a47e90b (well-known test vector)
    assert _fnv1a_32(b"abc") == 0x1A47E90B


def test_bucket_in_range():
    for tok in ("mov", "call", "push edi", "very-long-token-name"):
        assert 0 <= _bucket(tok, 128) < 128
        assert 0 <= _bucket(tok, 4096) < 4096


# ---------------------------------------------------------------------------
# Mnemonic extraction
# ---------------------------------------------------------------------------


def test_extract_mnemonics_takes_first_token_and_lowercases():
    disasm = ["MOV eax, 1", "  push   edi  ", "call sym.foo", "", None, "RET"]
    m = _extract_mnemonics(disasm)
    assert m == ["mov", "push", "call", "ret"]


def test_extract_mnemonics_empty_returns_empty():
    assert _extract_mnemonics([]) == []
    assert _extract_mnemonics(None) == []


# ---------------------------------------------------------------------------
# function_features
# ---------------------------------------------------------------------------


def test_function_features_shape_and_scalar_suffix():
    rec = {"disasm": ["mov eax, 1", "ret"], "bytes": "cc" * 5}
    vec = function_features(rec, feature_dim=128)
    assert vec.shape == (128 + 3,)
    assert vec.dtype == np.float32
    # 2 instructions -> unigrams "mov","ret" and bigram "mov->ret" -> 3 hits
    assert vec[:128].sum() == 3.0
    # Scalar suffix: n_instructions, byte_len, ratio
    assert vec[128] == 2.0
    assert vec[129] == 5.0
    assert vec[130] == pytest.approx(2.5)


def test_function_features_bigram_ordering_matters():
    a = function_features({"disasm": ["mov eax, 1", "push edi"]}, feature_dim=128)
    b = function_features({"disasm": ["push edi", "mov eax, 1"]}, feature_dim=128)
    # unigrams sum equal, but a specific bigram bin differs
    assert (a != b).any()


# ---------------------------------------------------------------------------
# Streaming sampler
# ---------------------------------------------------------------------------


def _make_shard(tmp_path: Path, name: str, records: list) -> Path:
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        with zf.open("records.json", "w") as fh:
            for r in records:
                fh.write((json.dumps(r) + "\n").encode())
    return p


def _rec(sha, addr, caps, disasm=None):
    return {
        "sha256": sha, "func_addr": addr, "capa": caps,
        "bytes": "cc" * 8,
        "disasm": disasm or ["mov eax, 1", "ret"],
    }


def test_iter_function_records_streams_across_shards(tmp_path):
    _make_shard(tmp_path, "s1.zip", [_rec("aa" * 32, "0x1", ["X"])])
    _make_shard(tmp_path, "s2.zip", [_rec("bb" * 32, "0x2", ["Y"])])
    records = list(iter_function_records(tmp_path))
    assert len(records) == 2
    shas = {r["sha256"] for r in records}
    assert shas == {"aa" * 32, "bb" * 32}


def test_iter_function_records_respects_max_records(tmp_path):
    _make_shard(tmp_path, "s.zip", [
        _rec("aa" * 32, f"0x{i}", ["X"]) for i in range(10)
    ])
    records = list(iter_function_records(tmp_path, max_records=3))
    assert len(records) == 3


def test_stratified_sampler_quotas():
    s = StratifiedSampler(target_labels=["X"], target_per_capability=3,
                          negative_pool_size=2)
    # 5 X's and 3 negatives
    for i in range(5):
        s.consider(_rec(f"{i:064x}", "0x1", ["X"]))
    for i in range(3):
        s.consider(_rec(f"{i:064x}n", "0x2", ["Other"]))
    counts = s.counts()
    # Sampler stops adding positives after quota is hit, so counts reflect
    # positives kept, not positives seen (see kept_x below).
    assert counts["per_label"]["X"] == 3
    assert counts["negatives"] == 2
    # Only 3 X-positive rows were kept though
    kept_x = [r for r in s.samples if "X" in r["capa"]]
    assert len(kept_x) == 3
    kept_neg = [r for r in s.samples if "X" not in r["capa"]]
    assert len(kept_neg) == 2


def test_stratified_sampler_done_signal():
    s = StratifiedSampler(target_labels=["X"], target_per_capability=2,
                          negative_pool_size=1)
    assert not s.done()
    s.consider(_rec("a" * 64, "0x1", ["X"]))
    s.consider(_rec("b" * 64, "0x2", ["X"]))
    s.consider(_rec("c" * 64, "0x3", ["Other"]))
    assert s.done()


def test_stratified_sample_end_to_end(tmp_path):
    records = (
        [_rec(f"a{i:063x}", "0x1", ["A"]) for i in range(4)] +
        [_rec(f"b{i:063x}", "0x2", ["B"]) for i in range(4)] +
        [_rec(f"n{i:063x}", "0x3", ["Other"]) for i in range(6)]
    )
    _make_shard(tmp_path, "s.zip", records)
    samples, counts = stratified_sample(
        tmp_path, ["A", "B"],
        target_per_capability=3, negative_pool_size=3,
    )
    assert counts["per_label"]["A"] >= 3
    assert counts["per_label"]["B"] >= 3
    assert counts["negatives"] == 3
    assert 6 <= len(samples) <= 9  # 3+3 pos + up to 3 neg


# ---------------------------------------------------------------------------
# Split by sha256
# ---------------------------------------------------------------------------


def test_split_by_sha256_keeps_functions_together():
    samples = (
        [_rec("a" * 64, f"0x{i}", ["X"]) for i in range(5)] +
        [_rec("b" * 64, f"0x{i}", ["X"]) for i in range(4)] +
        [_rec("c" * 64, f"0x{i}", ["X"]) for i in range(3)] +
        [_rec("d" * 64, f"0x{i}", ["X"]) for i in range(3)] +
        [_rec("e" * 64, f"0x{i}", ["X"]) for i in range(3)] +
        [_rec("f" * 64, f"0x{i}", ["X"]) for i in range(3)]
    )
    train, val, test = split_by_sha256(samples, val_frac=0.2, test_frac=0.2, seed=7)
    # every sha appears in exactly one split
    def _shas(idxs):
        return {samples[i]["sha256"] for i in idxs}
    train_s, val_s, test_s = _shas(train), _shas(val), _shas(test)
    assert not (train_s & val_s)
    assert not (train_s & test_s)
    assert not (val_s & test_s)
    # each split non-empty
    assert train and val and test


# ---------------------------------------------------------------------------
# Trainer (small synthetic recovery)
# ---------------------------------------------------------------------------


def test_train_recovers_hand_designed_signal():
    """Positive samples contain 'xor'; negatives contain 'mov'/'add'.
    The trainer should learn to distinguish, with test AUC > 0.9.
    """
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    positives = [
        _rec(f"p{i:063x}", "0x1", ["XORencoding"],
             disasm=["xor eax, ebx", "xor ecx, edx", "ret"])
        for i in range(80)
    ]
    negatives = [
        _rec(f"n{i:063x}", "0x2", ["Other"],
             disasm=["mov eax, 1", "add ebx, ecx", "ret"])
        for i in range(80)
    ]
    samples = positives + negatives
    rng.shuffle(samples)
    result = train_multilabel_functionml(
        samples, ["XORencoding"],
        feature_dim=256, n_estimators=50, seed=0,
    )
    m = result["per_label_metrics"]["XORencoding"]
    assert m["auc"] is None or m["auc"] > 0.9, m
    # test set F1 should be strong given a perfectly separable signal
    assert m["f1"] > 0.8, m


def test_train_function_ml_cli_registered():
    from click.testing import CliRunner
    from neurinspectre.cli.main import cli, _CLICK_COMMANDS
    assert "train-function-ml" in _CLICK_COMMANDS
    runner = CliRunner()
    res = runner.invoke(cli, ["train-function-ml", "--help"])
    assert res.exit_code == 0
    assert "--target-per-capability" in res.output
    assert "--feature-dim" in res.output


def test_train_skips_no_variance_labels():
    pytest.importorskip("lightgbm")
    samples = [
        _rec(f"n{i:063x}", "0x1", ["Other"],
             disasm=["mov eax, 1", "ret"])
        for i in range(30)
    ]
    result = train_multilabel_functionml(
        samples, ["NeverPresent"],
        feature_dim=128, n_estimators=10, seed=0,
    )
    assert result["per_label_metrics"]["NeverPresent"] == {"skipped_reason": "no_variance_in_train"}
