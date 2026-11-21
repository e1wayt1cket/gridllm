# test_pilot_artifacts.py
"""Frozen pilot budget, day pools, and the evaluation's physical integrity."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "v2_cy_jn_2025"
PILOT = DATA / "pilot"
pytestmark = pytest.mark.skipif(
    not (PILOT / "eval.csv").exists(), reason="the pilot has not finished")


def _read(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_hashes_in_the_pilot_config_match_the_files():
    cfg = json.loads((PILOT / "pilot_config.json").read_text(encoding="utf-8"))
    expected = {
        "pairing": DATA / "load_price_pairing.csv",
        "bus_loads": DATA / "bus_loads_96x33.npz",
        "price_curves": DATA / "price_curves.npz",
        "common_split": DATA / "common_split.json",
    }
    for key, path in expected.items():
        assert cfg["hashes"][key] == _sha(path), key


def test_the_two_pools_differ_only_by_the_full96_only_days():
    cfg = json.loads((PILOT / "pilot_config.json").read_text(encoding="utf-8"))
    pools = cfg["pools"]
    assert len(pools["NEAR"]) == 53
    assert len(pools["FULL96"]) == 69
    assert set(pools["NEAR"]) < set(pools["FULL96"])
    split = json.loads((DATA / "common_split.json").read_text(encoding="utf-8"))
    extra = {i.split("|")[0] for i in split["full96_only"]}
    assert set(pools["FULL96"]) - set(pools["NEAR"]) == extra


def test_every_run_has_the_same_budget_and_seed_list():
    cfg = json.loads((PILOT / "pilot_config.json").read_text(encoding="utf-8"))
    seeds = cfg["pilot"]["seeds"]
    assert len(seeds) >= 2
    runs = sorted((PILOT / "train").glob("*.csv"))
    assert len(runs) == 2 * len(seeds)
    budgets = set()
    for path in runs:
        rows = _read(path)
        assert len(rows) == cfg["pilot"]["episodes"]
        budgets.add((rows[-1]["total_steps"],
                     sum(int(float(r["updates"])) for r in rows)))
    assert len(budgets) == 1, budgets


def test_evaluation_covers_the_test_and_holiday_days_only():
    cfg = json.loads((PILOT / "pilot_config.json").read_text(encoding="utf-8"))
    rows = _read(PILOT / "eval.csv")
    expected = set(cfg["evaluation"]["test"]) | set(
        cfg["evaluation"]["holiday_block"])
    assert {r["load_date"] for r in rows} == expected
    train_days = {d for pool in cfg["pools"].values() for d in pool}
    assert not ({r["load_date"] for r in rows} & train_days)
    assert len(cfg["evaluation"]["test"]) == 11
    assert len(cfg["evaluation"]["holiday_block"]) == 5


def test_every_evaluation_clear_is_optimal_and_physically_clean():
    rows = _read(PILOT / "eval.csv")
    assert all(r["optimal"] == "True" for r in rows)
    assert all(int(float(r["lmp_fallbacks"])) == 0 for r in rows)
    assert all(int(float(r["voltage_violation_cells"])) == 0 for r in rows)
    assert all(int(float(r["line_overload_cells"])) == 0 for r in rows)
    assert all(abs(float(r["terminal_soc_max_err"])) <= 1e-9 for r in rows)
