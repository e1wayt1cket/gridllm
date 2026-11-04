# test_near_clearing_reuse.py
"""NEAR may reuse the FULL96 JN-Base clears only where every link holds."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "v2_cy_jn_2025"
REPORTS = Path(__file__).resolve().parents[1] / "reports" / "v2_cy_jn_2025"
INDEX = DATA / "near" / "clearing_reuse.csv"
pytestmark = pytest.mark.skipif(
    not INDEX.exists(), reason="the NEAR reuse audit is not built")


def _rows(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pid(row):
    return f"{row['load_date']}|{row['price_date']}"


@pytest.fixture(scope="module")
def index():
    return _rows(INDEX)


def test_every_near_pair_is_registered_once(index):
    near = _rows(DATA / "near" / "load_price_pairing.csv")
    assert len(index) == len(near) == 80
    assert {r["pair_id"] for r in index} == {_pid(r) for r in near}
    assert len({r["pair_id"] for r in index}) == len(index)


def test_every_registered_pair_is_reusable(index):
    assert all(r["reusable"] == "True" for r in index)
    for key in ("unique_full_record", "dates_match", "load_hash_match",
                "price_hash_match", "result_unique", "arm_is_jn_base",
                "config_match"):
        assert all(r[key] == "True" for r in index), key


def test_the_links_hold_in_the_source_files(index):
    full = {_pid(r): r for r in _rows(DATA / "load_price_pairing.csv")}
    near = {_pid(r): r for r in _rows(DATA / "near" / "load_price_pairing.csv")}
    base = {_pid(r): r for r in _rows(REPORTS / "results.csv")
            if r["arm"] == "JN-Base"}
    for pid, row in near.items():
        assert pid in full
        assert full[pid]["load_curve_sha256"] == row["load_curve_sha256"]
        assert full[pid]["price_curve_sha256"] == row["price_curve_sha256"]
        result = base[pid]
        assert result["optimal"] == "True"
        assert result["solver_status"] == "OPTIMAL"
        assert float(result["voltage_violation_cells"]) == 0
        assert float(result["line_overload_cells"]) == 0
        assert float(result["lmp_fallbacks"]) == 0
        assert abs(float(result["terminal_soc_max_err"])) <= 1e-9


def test_the_index_is_registered_with_its_hash(index):
    datasets = json.loads((DATA / "datasets.json").read_text(encoding="utf-8"))
    entry = datasets["near_jn_base_reuse"]
    assert entry["reusable_pairs"] == 80
    assert entry["total_near_pairs"] == 80
    assert entry["fully_verified"] is True
    assert entry["scope"] == "JN-Base only"
    assert entry["index_sha256"] == hashlib.sha256(INDEX.read_bytes()).hexdigest()


def _clean_row():
    return {
        "optimal": "True", "solver_status": "OPTIMAL",
        "voltage_violation_cells": "0", "min_bus_voltage_pu": "0.95",
        "max_bus_voltage_pu": "1.02", "line_overload_cells": "0",
        "max_line_utilization": "0.8", "p_balance_max_residual_mw": "1e-8",
        "q_balance_checked": "True", "q_balance_max_residual_mvar": "1e-8",
        "terminal_soc_max_err": "0", "exclusive_max_mw2": "0",
        "lmp_fallbacks": "0", "lmp_slack_max_dev_cny_mwh": "0.001",
        "reverse_within_limit": "True", "error": "",
    }


def test_quality_checks_accept_a_clean_clear_and_reject_a_violation():
    import audit_near_clearing_reuse as reuse
    config = {"network_support": {"v_min_pu": 0.93, "v_max_pu": 1.05}}
    assert all(reuse._quality_checks(_clean_row(), config).values())
    bad = dict(_clean_row(), voltage_violation_cells="1")
    assert reuse._quality_checks(bad, config)["voltage_ok"] is False
    bad = dict(_clean_row(), lmp_fallbacks="2")
    assert reuse._quality_checks(bad, config)["lmp_ok"] is False


def test_config_checks_require_the_support_assumption():
    import audit_near_clearing_reuse as reuse
    config = {
        "network_support": {"assumption": "line50_q2.0",
                            "line_impedance_scale": 0.5,
                            "reactive_fraction_of_load_q": 2.0,
                            "v_min_pu": 0.93, "v_max_pu": 1.05},
        "storage": {"terminal_soc_equal": True, "capacity_mwh": 1.6,
                    "power_mw": 0.4},
    }
    assert all(reuse._config_checks(config).values())
    config["network_support"]["line_impedance_scale"] = 1.0
    assert reuse._config_checks(config)["impedance_scale_is_0_5"] is False
