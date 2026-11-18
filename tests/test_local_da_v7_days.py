import numpy as np
import pytest

from local_da_contracts import validate_day
from local_da_days import load_pair, train_pairs, window_day
from run_local_da import build_day

PAIR = "2021-06-04"


def test_a_pair_carries_the_library_curves_and_the_paired_price():
    pair = load_pair(PAIR)
    assert pair["active"].shape == (96, 33)
    assert pair["reactive"].shape == (96, 33)
    assert pair["external_price"].shape == (96,)
    assert pair["price_date"] == "2025-06-02"
    assert np.allclose(pair["reactive"], 0.6190 * pair["active"], atol=1e-3)


def test_a_window_slices_the_day_and_carries_the_soc():
    pair = load_pair(PAIR)
    soc = {f"ESS{b}": 0.55 for b in (9, 14, 17, 27)}
    quotes = {"ESS9": np.array([10., 0.3, 120., 0.2])}
    day = window_day(pair, 44, 64, soc=soc, quotes=quotes)
    validate_day(day)
    assert day.periods == 20
    assert day.terminal_soc_equal is False
    assert [u.soc_initial for u in day.storage] == [0.55] * 4
    assert day.bids["ESS9"].discharge_price_cny_mwh.tolist() == [120.] * 20
    assert day.bids["ESS9"].discharge_quantity_mw.tolist() == [0.2] * 20
    assert np.allclose(day.active_load_mw, pair["active"][44:64])
    day_of = build_day(PAIR)[0]
    assert np.allclose(day.network.r_mw_base, day_of.network.r_mw_base)
    assert day.network.branches == day_of.network.branches


def test_the_window_clears_against_the_committed_case33bw_topology():
    # Literals read off the committed radial_network(), not re-derived here, so a
    # change to that builder fails this test instead of being copied into it.
    soc = {f"ESS{b}": 0.55 for b in (9, 14, 17, 27)}
    network = window_day(load_pair(PAIR), 44, 64, soc=soc, quotes={}).network
    assert len(network.branches) == 32
    assert network.branches[0] == (0, 1)
    assert network.r_mw_base[0] == pytest.approx(0.000575259116172393, rel=1e-12)
    assert network.x_mw_base[0] == pytest.approx(0.0002932448856844086, rel=1e-12)


def test_the_train_pairs_drop_only_exact_infeasible_days_and_name_them():
    # Both v3 screens share the library and the same per-day failure strings,
    # so either one answers the exact-infeasible criterion.
    pairs, selection = train_pairs(
        "reports/candidate_days_v3_clearing_2026-11-17/evidence")
    assert len(pairs) == 87
    assert set(selection["dropped"]) == {"2021-12-24", "2022-10-11", "2022-10-14"}
    assert all(p["block"] == "train" for p in pairs)
