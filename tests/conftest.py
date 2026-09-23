"""Test-suite conftest.

Sets thread caps for joblib / OpenMP / BLAS **before** any test module import.
Without this, ``sklearn.feature_extraction.FeatureHasher`` (pulled in through
``thrember.features``) spins up a loky worker pool that can deadlock with
subsequent Square / PGD attack tests on Darwin. We measured a hang here on
2026-09-20: forcing single-threaded backends brings the full EMBER 2018 +
EMBER 2024 + attack-hygiene regression from ``> 15 min timeout`` down to
``~3 s``.

Do not remove without re-running the full regression sweep unpiped.
"""

from __future__ import annotations

import os

_DEFAULTS = {
    "LOKY_MAX_CPU_COUNT": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for key, value in _DEFAULTS.items():
    os.environ.setdefault(key, value)
