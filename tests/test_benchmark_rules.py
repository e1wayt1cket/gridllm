# test_benchmark_rules.py
"""Rule quotes and the train/validation selection protocol."""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_benchmarks as rb


def _price():
    # 96 periods spanning the observed 30-1500 CNY/MWh band.
    return np.linspace(30.0, 900.0, 96)


def test_threshold_quotes_act_only_inside_the_bands():
    price = _price()
    charge, discharge = rb.threshold_quotes(price, 0.20, 0.80)
    lo = np.quantile(price, 0.20)
    hi = np.quantile(price, 0.80)
    assert set(np.unique(charge)) <= {0.0, rb.CHARGE_SUPPRESS}
    assert set(np.unique(discharge)) <= {0.0, rb.DISCHARGE_SUPPRESS}
    assert np.all(charge[price <= lo] == 0.0)
    assert np.all(charge[price > lo] == rb.CHARGE_SUPPRESS)
    assert np.all(discharge[price >= hi] == 0.0)
    assert np.all(discharge[price < hi] == rb.DISCHARGE_SUPPRESS)


def test_the_quotes_stay_inside_the_action_box():
    price = _price()
    charge, discharge = rb.threshold_quotes(price, 0.20, 0.80)
    assert charge.min() >= 0.0 and charge.max() <= 150.0
    assert discharge.min() >= 0.0 and discharge.max() <= 300.0


def test_the_soc_band_caps_the_active_periods():
    price = _price()
    charge, discharge = rb.soc_band_quotes(
        price, 0.20, 0.80, 0.10, 0.90, e_max=1.6, p_max=0.4,
        eta_ch=0.92, eta_dis=0.92, soc0=0.5)
    # (0.9-0.5)*1.6/(0.92*0.4*0.25) = 6 ; (0.5-0.1)*1.6*0.92/(0.4*0.25) = 5
    assert int((charge == 0.0).sum()) == 6
    assert int((discharge == 0.0).sum()) == 5
    # The cheapest charge periods and dearest discharge periods are the ones
    # that may act.
    assert np.all(charge[np.argsort(price)[:6]] == 0.0)
    assert np.all(discharge[np.argsort(price)[-5:]] == 0.0)


def test_a_narrower_band_allows_fewer_periods():
    price = _price()
    wide = rb.soc_band_quotes(price, 0.20, 0.80, 0.10, 0.90, 1.6, 0.4,
                              0.92, 0.92, 0.5)
    narrow = rb.soc_band_quotes(price, 0.20, 0.80, 0.20, 0.80, 1.6, 0.4,
                                0.92, 0.92, 0.5)
    assert int((narrow[0] == 0.0).sum()) < int((wide[0] == 0.0).sum())
    assert int((narrow[1] == 0.0).sum()) < int((wide[1] == 0.0).sum())


def test_selection_ranks_on_train_and_chooses_on_validation(tmp_path):
    path = tmp_path / "raw.csv"
    rows = []
    train_means = {"c1": 10.0, "c2": 20.0, "c3": 30.0, "c4": 5.0}
    for cand, value in train_means.items():
        rows.append({"arm": "B2A", "candidate": cand, "split": "train",
                     "load_date": "2022-01-01", "price_date": "2025-01-01",
                     "ess_total_payoff_cny": value, "error": ""})
    # Both top-two train candidates get a validation score; c2 wins there.
    for cand, value in {"c3": 50.0, "c2": 60.0}.items():
        rows.append({"arm": "B2A", "candidate": cand,
                     "split": "validation", "load_date": "2022-01-02",
                     "price_date": "2025-01-02",
                     "ess_total_payoff_cny": value, "error": ""})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = rb._select(path, tmp_path / "log.txt")
    stage = payload["selection"]["B2A"]
    assert stage["top2_by_train"] == ["c3", "c2"]
    assert stage["selected"] == "c2"
    assert payload["selected"]["B2A"] == "c2"


DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "v2_cy_jn_2025"
RAW = DATA / "benchmark" / "benchmark_raw.csv"


def test_split_map_assigns_the_long_pairs_to_full96_only():
    import make_benchmark_report as report
    common = {
        "common": {"train": ["2022-01-01|2025-01-01"],
                   "validation": ["2022-01-02|2025-01-02"],
                   "test": ["2022-01-03|2025-01-03"],
                   "holiday_block": ["2022-01-04|2025-01-04"]},
        "full96_only": ["2022-02-01|2025-02-01"],
    }
    mapping = report._split_map(common)
    assert mapping["2022-01-01"] == "train"
    assert mapping["2022-01-03"] == "test"
    assert mapping["2022-02-01"] == "full96_only"


@pytest.mark.skipif(not RAW.exists(), reason="benchmark clears are not built")
def test_the_benchmark_covers_every_arm_and_split():
    import csv
    with RAW.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if not r.get("error")]
    by_arm = {}
    for row in rows:
        by_arm.setdefault((row["arm"], row["split"]), set()).add(
            row["candidate"] if row["arm"] != "B0" else "B0")
    assert len([r for r in rows if r["arm"] == "B0"]) == 96
    assert len(by_arm[("B2A", "train")]) == 4
    assert len(by_arm[("B2B", "train")]) == 8
    assert len(by_arm[("B2A", "validation")]) == 2
    assert len(by_arm[("B2B", "validation")]) == 2
    assert len(by_arm[("B2A", "test")]) == 1
    assert len(by_arm[("B2B", "test")]) == 1
    for family in ("B2A", "B2B"):
        days = {r["load_date"] for r in rows
                if r["arm"] == family and r["split"] == "test"}
        assert len(days) == 11
        holiday = {r["load_date"] for r in rows
                   if r["arm"] == family and r["split"] == "holiday_block"}
        assert len(holiday) == 5
        assert not any(r["split"] == "full96_only" and r["arm"] == family
                       for r in rows)
    assert all(r["optimal"] == "True" for r in rows)
