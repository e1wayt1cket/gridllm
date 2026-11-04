# test_daily_return_discount.py
"""The discount facts and the C2 telescope, recomputed independently."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_daily_return_discount as audit

GAMMA = 0.99
T = 24


def test_the_telescope_holds_for_random_paths():
    rng = np.random.default_rng(7)
    weights = GAMMA ** np.arange(T)
    for _ in range(50):
        phi = rng.normal(0.0, 100.0, size=T + 1)
        shaped = GAMMA * phi[1:] - phi[:-1]
        left = float(np.sum(weights * shaped))
        right = float(GAMMA ** T * phi[T] - phi[0])
        assert left == pytest.approx(right, abs=1e-9)


def test_the_total_shaping_depends_only_on_the_boundaries():
    rng = np.random.default_rng(8)
    weights = GAMMA ** np.arange(T)
    phi_a = rng.normal(0.0, 100.0, size=T + 1)
    phi_b = phi_a.copy()
    phi_b[1:-1] = rng.normal(0.0, 100.0, size=T - 1)
    total_a = float(np.sum(weights * (GAMMA * phi_a[1:] - phi_a[:-1])))
    total_b = float(np.sum(weights * (GAMMA * phi_b[1:] - phi_b[:-1])))
    assert total_a == pytest.approx(total_b, abs=1e-9)


def test_the_terminal_reward_is_a_fixed_multiple_of_the_day_profit():
    assert GAMMA ** (T - 1) == pytest.approx(0.7936, abs=1e-4)
    assert float(np.sum(GAMMA ** np.arange(T))) == pytest.approx(21.4322,
                                                                 abs=1e-4)


def test_the_recorded_identity_check_passes():
    import json
    path = Path(__file__).resolve().parents[1] / (
        "data/processed/v2_cy_jn_2025/pilot/c2_identity_check.json")
    if not path.exists():
        pytest.skip("identity check not built")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["telescope_max_abs_error"] < 1e-9
    assert payload["telescope_boundary_only_max_abs_error"] < 1e-9
    assert payload["terminal_zero_max_abs_error"] < 1e-9
    assert payload["base_return_over_day_reward"] == pytest.approx(
        payload["terminal_block_discount"])
