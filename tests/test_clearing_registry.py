"""Tests for the OPF clearing-mechanism registry (dispatch.py)."""

import pytest

from dispatch import (
    CLEARING_MECHANISMS,
    get_clearing_mechanism,
    register_clearing_mechanism,
    solve_opf_gurobi,
    solve_lindist_opf_batch,
    solve_socp_opf_batch,
)
from models import MarketConfig


def test_known_mechanisms_registered():
    assert set(CLEARING_MECHANISMS) >= {"dc", "lindistflow", "socp"}


def test_lindistflow_has_batch_and_single_period():
    mech = get_clearing_mechanism("lindistflow")
    assert mech["batch_fn"] is solve_lindist_opf_batch
    assert mech["single_period_fn"] is not None


def test_socp_batch_only():
    mech = get_clearing_mechanism("socp")
    assert mech["batch_fn"] is solve_socp_opf_batch
    assert mech["single_period_fn"] is None


def test_dc_has_highs_fallback_flag():
    assert get_clearing_mechanism("dc")["requires_gurobi"] is False


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError, match="bogus"):
        get_clearing_mechanism("bogus")


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_clearing_mechanism(
            "dc", single_period_fn=lambda *a, **k: None,
            requires_gurobi=False)


def test_solve_opf_socp_single_period_raises():
    """SOCP has no single-period solver; the dispatcher must refuse up front,
    before touching any solver."""
    config = MarketConfig(opf_mode="socp", verbose=False)
    with pytest.raises(RuntimeError, match="requires the batch solver"):
        solve_opf_gurobi(None, None, 0, "DA", None, 0.0, None, config)


def test_solve_opf_unknown_mode_raises():
    config = MarketConfig(opf_mode="bogus_mode", verbose=False)
    with pytest.raises(ValueError, match="bogus_mode"):
        solve_opf_gurobi(None, None, 0, "DA", None, 0.0, None, config)
