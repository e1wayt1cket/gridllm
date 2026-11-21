# test_c2_reward.py
"""C2: terminal zero, telescope, boundary-only dependence, scaling, rejection."""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import c2_reward as c2

T = c2.T


def _soc(seed=0, soc0=0.5):
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, 0.01, T + 1))
    walk = np.clip(walk - walk[0] + soc0, c2.SOC_LO, c2.SOC_HI)
    walk[0] = soc0
    return walk


def _base(seed=0):
    rng = np.random.default_rng(seed)
    return np.zeros(T) + np.where(np.arange(T) == T - 1,
                                  rng.normal(0, 500), 0.0)


def test_terminal_potential_is_exactly_zero():
    for soc in (c2.SOC_LO, 0.37, c2.SOC0, 0.81, c2.SOC_HI):
        assert float(c2.potential(soc, T)) == 0.0


def test_the_telescope_and_the_boundary_term():
    soc = _soc()
    base = _base()
    shaped = c2.c2_reward({"ESS9": base}, {"ESS9": soc})
    weights = c2.GAMMA ** np.arange(T)
    total = float(np.sum(weights * shaped.shaping_terms["ESS9"]))
    assert total == pytest.approx(shaped.boundary["ESS9"], abs=1e-10)
    assert shaped.boundary["ESS9"] == pytest.approx(
        -float(c2.potential(soc[0], 0)), abs=1e-10)


def test_the_boundary_depends_only_on_the_endpoints():
    soc_a = _soc(1)
    soc_b = soc_a.copy()
    rng = np.random.default_rng(2)
    soc_b[1:T] = np.clip(soc_a[1:T] + rng.normal(0, 0.02, T - 1),
                         c2.SOC_LO, c2.SOC_HI)
    soc_b[T] = soc_a[T]
    base = _base()
    a = c2.c2_reward({"ESS9": base}, {"ESS9": soc_a})
    b = c2.c2_reward({"ESS9": base}, {"ESS9": soc_b})
    assert a.episode_return["ESS9"] == pytest.approx(
        b.episode_return["ESS9"], abs=1e-10)


def test_the_potential_is_bounded():
    cfg = c2.C2Config()
    for t in range(T + 1):
        for soc in (cfg.soc_lo, cfg.soc0, cfg.soc_hi):
            assert abs(float(c2.potential(soc, t, cfg))) <= cfg.bound + 1e-12


def test_mid_episode_termination_is_refused():
    soc = _soc()
    base = _base()
    done = np.zeros(T, dtype=bool)
    done[5] = True
    with pytest.raises(ValueError):
        c2.c2_reward({"ESS9": base}, {"ESS9": soc}, done=done)


def test_an_initial_soc_mismatch_is_refused():
    soc = _soc(soc0=0.6)
    with pytest.raises(ValueError):
        c2.c2_reward({"ESS9": _base()}, {"ESS9": soc})


def test_fixed_scaling_preserves_order():
    soc = _soc()
    base = _base()
    a = c2.c2_reward({"ESS9": base}, {"ESS9": soc})
    cfg2 = c2.C2Config(kappa=3.0 * c2.KAPPA)
    b = c2.c2_reward({"ESS9": 3.0 * base}, {"ESS9": soc}, config=cfg2)
    assert b.episode_return["ESS9"] == pytest.approx(
        3.0 * a.episode_return["ESS9"], abs=1e-9)


def test_the_module_does_not_mix_in_the_local_reward():
    source = inspect.getsource(c2)
    assert "local_baseline" not in source
    assert "participant_payoff" not in source


def test_a_numerical_soc_overshoot_is_clipped_not_rejected():
    """The clearing enforces the band; a solver-level overshoot is not a policy
    violation, but a real violation must still be refused."""
    config = c2.C2Config()
    assert np.isfinite(c2.f_soc(config.soc_hi + 1e-9, config))
    assert np.isfinite(c2.f_soc(config.soc_lo - 1e-9, config))
    with pytest.raises(ValueError):
        c2.f_soc(config.soc_hi + 1e-3, config)
