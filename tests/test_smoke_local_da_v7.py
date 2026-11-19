import numpy as np

import rl_da_env
from local_da_days import load_pair
from smoke_local_da_v7 import _walk, summarize


def record(**over):
    base = {"rewards_finite": True, "clears": 24, "none_clears": 0,
            "soc_min": 0.32, "soc_max": 0.71, "declared_point_deviation": 0.0,
            "inventory_terms": 1}
    base.update(over)
    return base


def test_a_clean_day_passes_every_acceptance_flag():
    out = summarize(record())
    assert out == {"rewards_finite": True, "no_none_clears": True,
                   "soc_in_band": True, "one_inventory_term": True,
                   "declared_point_deviation": 0.0}


def test_a_missing_clear_and_a_second_inventory_term_fail_their_flags():
    out = summarize(record(none_clears=1, inventory_terms=2, soc_max=0.95))
    assert out["no_none_clears"] is False
    assert out["one_inventory_term"] is False
    assert out["soc_in_band"] is False


class BelowBandClearingModel:
    """A solve that leaves every unit just below its declared SOC band."""

    def __init__(self, day, *, directions=None, exact=True, seconds=120.0,
                 mip_gap=1e-6):
        self.day = day

    def solve(self, prices=False):
        names = [unit.name for unit in self.day.storage]
        periods = self.day.periods
        energy = 0.1 * rl_da_env.E_MAX_MWH - 1e-9
        result = {"optimal": True, "clearing_objective": 0.0,
                  "charge": {nm: np.zeros(periods) for nm in names},
                  "discharge": {nm: np.zeros(periods) for nm in names},
                  "energy": {nm: np.full(periods + 1, energy) for nm in names}}
        if prices:
            result["lmp_cny_mwh"] = np.zeros((periods, self.day.network.buses))
        return result

    def close(self):
        pass


def test_a_raise_in_the_timing_path_still_leaves_a_report(monkeypatch):
    import smoke_local_da_v7 as smoke_mod

    walked = {"date": "2021-06-04", "price_date": "x",
              "rewards_finite": True, "clears": 24, "none_clears": 0,
              "soc_min": 0.5, "soc_max": 0.5,
              "declared_point_deviation": 0.0, "inventory_terms": 1,
              "block_seconds": [1.0], "blocks": [], "abort": False,
              "episode_seconds": 1.0, "probe": None, "probe_seconds": 1.0}
    monkeypatch.setattr(smoke_mod, "_walk", lambda *a, **k: dict(walked))

    def raise_on_block_zero(*args, **kwargs):
        raise ValueError("block 0 failed")

    monkeypatch.setattr(smoke_mod, "_timed_block", raise_on_block_zero)
    report = smoke_mod.smoke(["2021-06-04"], side_arms=False)
    assert report["episodes"][0]["latency_failure"] == "ValueError: block 0 failed"
    assert report["latency"]["exact"] is None


def test_a_raw_soc_outside_the_band_fails_the_acceptance_flag(monkeypatch):
    # The record must read the solve's raw state, not the clamped handoff the
    # next window inherits, or soc_in_band is true by construction.
    monkeypatch.setattr(rl_da_env, "ClearingModel", BelowBandClearingModel)
    monkeypatch.setattr(rl_da_env, "clear_day",
                        lambda day, **kw: {"qualified_for_settlement": True})
    walked = _walk(load_pair("2021-06-04"), seconds=1.0,
                   quote=(0., 0.4, 0., 0.4))
    assert walked["soc_min"] < 0.1
    assert summarize(walked)["soc_in_band"] is False
