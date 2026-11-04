# test_v2_cy_jn_price.py
"""The 2025 Jiangnan price audit and the dated provider that serves it."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_jn_price_source as audit
import wholesale_price as wp


def _write_day(path: Path, date: str, values, mv=None):
    stamps = pd.date_range(date, periods=len(values), freq="15min")
    frame = pd.DataFrame({
        "time": stamps.strftime("%Y-%m-%d %H:%M"),
        "price_value_day_ahead_north": values,
        "price_value_day_ahead_south": values,
        "mv_value": mv if mv is not None else np.full(len(values), 1000.0),
    })
    frame.to_csv(path, mode="a", index=False,
                 header=not path.exists())


def test_a_day_is_complete_only_with_all_96_quarter_hours(tmp_path):
    path = tmp_path / "price.csv"
    values = np.arange(96, dtype=float)
    _write_day(path, "2025-07-01", values)
    _write_day(path, "2025-07-02", values[:-1])      # one period short
    frame = audit.read_price_csv(path)
    rows = {r["date"]: r for r in audit.daily_coverage(frame)}
    assert rows["2025-07-01"]["complete"] is True
    assert rows["2025-07-01"]["periods"] == 96
    assert rows["2025-07-02"]["complete"] is False
    assert rows["2025-07-02"]["periods"] == 95


def test_the_floor_and_cap_counts_are_reported_not_invented(tmp_path):
    path = tmp_path / "price.csv"
    values = np.full(96, 300.0)
    values[0] = 30.0
    values[1:4] = 1500.0
    _write_day(path, "2025-07-01", values)
    rows = audit.daily_coverage(audit.read_price_csv(path))
    assert rows[0]["south_floor_hits"] == 1
    assert rows[0]["south_cap_hits"] == 3
    assert rows[0]["south_min"] == 30.0
    assert rows[0]["south_max"] == 1500.0


def test_a_curve_digest_moves_with_the_values_not_only_the_date():
    a = audit.curve_digest("2025-07-01", "south", [1.0, 2.0])
    b = audit.curve_digest("2025-07-01", "south", [1.0, 3.0])
    c = audit.curve_digest("2025-07-02", "south", [1.0, 2.0])
    assert len({a, b, c}) == 3


def test_window_supply_counts_only_complete_days(tmp_path):
    path = tmp_path / "price.csv"
    _write_day(path, "2025-07-01", np.full(96, 200.0))
    _write_day(path, "2025-07-02", np.full(95, 200.0))
    rows = audit.daily_coverage(audit.read_price_csv(path))
    windows = audit.window_coverage(rows)
    assert windows["summer"]["complete"] == 1
    assert windows["winter"]["complete"] == 0


def _write_curves(path: Path, curves):
    np.savez_compressed(path, **{f"south/{d}": np.asarray(v, float)
                                 for d, v in curves.items()})


def test_the_provider_slices_the_days_own_periods(tmp_path):
    path = tmp_path / "curves.npz"
    day = np.arange(96, dtype=float)
    _write_curves(path, {"2025-07-15": day})
    provider = wp.JnPriceProvider({"jn2025_file": str(path)})
    np.testing.assert_array_equal(
        provider.get_day_ahead_price("2025-07-15", T=96, t_start=0), day)
    np.testing.assert_array_equal(
        provider.get_day_ahead_price("2025-07-15", T=4, t_start=92), day[92:96])


def test_the_provider_refuses_a_date_it_does_not_have(tmp_path):
    path = tmp_path / "curves.npz"
    _write_curves(path, {"2025-07-15": np.zeros(96)})
    provider = wp.JnPriceProvider({"jn2025_file": str(path)})
    with pytest.raises(ValueError, match="not a complete day"):
        provider.get_day_ahead_price("2025-07-16")
    with pytest.raises(ValueError, match="exceed"):
        provider.get_day_ahead_price("2025-07-15", T=96, t_start=4)


def test_a_jn_episode_cannot_be_served_by_an_undated_index(tmp_path):
    """The day is a date. An index would serve another day's curve."""
    path = tmp_path / "curves.npz"
    _write_curves(path, {"2025-07-15": np.zeros(96)})
    provider = wp.JnPriceProvider({"jn2025_file": str(path)})
    with pytest.raises(ValueError):
        provider.get_day_ahead_price(0)


def test_the_jn_source_is_formal_and_declares_its_unit_is_inferred(tmp_path):
    path = tmp_path / "curves.npz"
    _write_curves(path, {"2025-07-15": np.full(96, 300.0)})
    provider = wp.JnPriceProvider({"jn2025_file": str(path)})
    assert wp.formal_data_readiness("jn2025")[0] is True
    described = provider.describe()
    assert described["price_unit"] == "CNY/MWh"
    assert described["price_unit_confirmed"] is False
    assert described["day_id_selects"] == "iso_date"
