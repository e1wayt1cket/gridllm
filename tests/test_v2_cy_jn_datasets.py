# test_v2_cy_jn_datasets.py
"""The two V2-CY-JN-2025 datasets: filter, partitions, and hashes."""

import csv
import hashlib
import json
import sys
from datetime import date as Date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "v2_cy_jn_2025"
pytestmark = pytest.mark.skipif(
    not (DATA / "datasets.json").exists(),
    reason="V2-CY-JN-2025 datasets are not built")

SPLITS = ("train", "validation", "test", "holiday_block")


def _rows(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _gap(row):
    target = Date.fromisoformat(row["load_date"]).replace(year=2025)
    return abs((Date.fromisoformat(row["price_date"]) - target).days)


def _split(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def full():
    return _rows(DATA / "load_price_pairing.csv")


@pytest.fixture(scope="module")
def near():
    return _rows(DATA / "near" / "load_price_pairing.csv")


def test_full_is_96_unique_pairs(full):
    assert len(full) == 96
    assert len({r["load_date"] for r in full}) == 96
    assert len({r["price_date"] for r in full}) == 96


def test_near_keeps_exactly_the_pairs_within_fourteen_days(full, near):
    kept = {(r["load_date"], r["price_date"]) for r in near}
    expected = {(r["load_date"], r["price_date"]) for r in full
                if _gap(r) <= 14}
    assert kept == expected
    assert len(near) == 80          # computed, not the working name's 83
    assert all(_gap(r) <= 14 for r in near)
    assert all(_gap(r) > 14 for r in full if r not in near)


def test_near_never_repairs_a_pair(full, near):
    """The filter drops rows; it must not alter any that remain."""
    by_load = {r["load_date"]: r for r in full}
    for row in near:
        assert by_load[row["load_date"]] == row


def test_the_recorded_type_mismatch_is_one_and_survives(full, near):
    for rows in (full, near):
        mismatched = [r for r in rows if "mismatch" in r["pairing_type"]]
        assert len(mismatched) == 1
        assert mismatched[0]["load_date"] == "2022-08-20"
        assert _gap(mismatched[0]) == 0


def test_each_split_partitions_its_own_dataset(full, near):
    for rows, name in ((full, "split.json"), (near, "near/split.json")):
        split = _split(name)
        seen = []
        for key in SPLITS:
            seen.extend(split[key])
        assert len(seen) == len(rows)
        assert len(set(seen)) == len(rows)
        expected = {r["load_date"] for r in rows}
        assert set(seen) == expected
        assert set(split["holiday_block"]) == {
            r["load_date"] for r in rows
            if r["group"] == "national_day_2022"}


def test_the_two_splits_are_different_assignments(full, near):
    a = _split("split.json")
    b = _split("near/split.json")
    assert a["seed"] != b["seed"]
    shared = set(b["train"]) & set(b["test"])
    assert not shared
    # NEAR is a subset of FULL96, so it is not an independent sample; the
    # committed seeds put NEAR test days in FULL96 training.
    assert not (set(b["test"]) & set(a["test"]))


def test_dataset_index_hashes_match_the_files():
    index = json.loads((DATA / "datasets.json").read_text(encoding="utf-8"))
    for dataset in index["datasets"].values():
        table = Path(__file__).resolve().parents[1] / dataset["table"]
        digest = hashlib.sha256(table.read_bytes()).hexdigest()
        assert dataset["sha256"]["table"] == digest
    assert index["relationship"]["near_is_subset_of_full"] is True
