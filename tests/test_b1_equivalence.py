# test_b1_equivalence.py
"""Empty action parameters mean truthful quotes, which is what B1 is."""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ess
import models
from dispatch_core import quote_arrays

T = 96
DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "v2_cy_jn_2025"


@pytest.fixture
def config():
    return models.MarketConfig.from_yaml()


def _prosumer(bid=400.0, offer=300.0):
    zeros = np.zeros(T)
    return models.Agent(name="L1", bus=5, is_prosumer=True,
                        load_forecast=zeros.copy(), pv_forecast=zeros.copy(),
                        load_real=zeros.copy(), pv_real=zeros.copy(),
                        bid_value=bid, offer_cost=offer)


def test_an_absent_storage_action_is_the_truthful_point(config):
    unit = ess.build_ess_fleet(T, config, buses=[9])[0]
    charge, discharge = quote_arrays(unit, {}, T, config)
    assert np.all(charge == 0.0)
    assert np.all(discharge == 0.0)


def test_an_absent_prosumer_action_is_its_own_declared_price(config):
    agent = _prosumer(bid=400.0, offer=300.0)
    bid, offer = quote_arrays(agent, {}, T, config)
    assert np.all(bid == 400.0)
    assert np.all(offer == 300.0)


def test_a_partial_storage_declaration_is_refused_not_completed(config):
    unit = ess.build_ess_fleet(T, config, buses=[9])[0]
    with pytest.raises(KeyError):
        quote_arrays(unit, {unit.name: {"charge_price": 5.0}}, T, config)


def test_a_legacy_multiplier_on_storage_is_refused(config):
    unit = ess.build_ess_fleet(T, config, buses=[9])[0]
    with pytest.raises(KeyError):
        quote_arrays(unit, {unit.name: {"bid_mult": 1.0}}, T, config)


AUDIT = DATA / "benchmark" / "b1_equivalence.csv"


@pytest.mark.skipif(not AUDIT.exists(), reason="B1 audit is not built")
def test_the_b1_audit_passes_every_day():
    with AUDIT.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 96
    assert all(r["passed"] == "True" for r in rows)
    for key in ("storage_quotes_truthful", "prosumer_quotes_truthful",
                "load_digest_ok", "price_digest_ok", "result_flags_ok"):
        assert all(r[key] == "True" for r in rows), key
