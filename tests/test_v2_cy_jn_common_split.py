# test_v2_cy_jn_common_split.py
"""The cross-dataset comparison split: one label per common pair, no leakage."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "v2_cy_jn_2025"
pytestmark = pytest.mark.skipif(
    not (DATA / "common_split.json").exists(),
    reason="the common split is not built")

SPLITS = ("train", "validation", "test", "holiday_block")


def _rows(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pid(row):
    return f"{row['load_date']}|{row['price_date']}"


@pytest.fixture(scope="module")
def split():
    return json.loads((DATA / "common_split.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def full():
    return _rows(DATA / "load_price_pairing.csv")


@pytest.fixture(scope="module")
def near():
    return _rows(DATA / "near" / "load_price_pairing.csv")


def test_common_ids_are_exactly_the_intersection(split, full, near):
    full_ids = {_pid(r) for r in full}
    near_ids = {_pid(r) for r in near}
    common = {i for k in SPLITS for i in split["common"][k]}
    assert common == full_ids & near_ids
    full_only = {i for k in SPLITS for i in split["full96"][k]} - common
    assert full_only == full_ids - near_ids
    assert set(split["full96_only"]) == full_only


def test_every_common_record_has_the_same_label_in_both(split):
    full_label = {i: k for k in SPLITS for i in split["full96"][k]}
    near_label = {i: k for k in SPLITS for i in split["near"][k]}
    for pid in (i for k in SPLITS for i in split["common"][k]):
        assert full_label[pid] == near_label[pid]


def test_the_common_test_cannot_be_trained_on(split):
    full = {k: set(split["full96"][k]) for k in SPLITS}
    near = {k: set(split["near"][k]) for k in SPLITS}
    common_test = set(split["common"]["test"])
    assert common_test == full["test"] == near["test"]
    assert not (common_test & (full["train"] | full["validation"]))
    assert not (common_test & (near["train"] | near["validation"]))
    assert not (full["validation"] & full["train"])
    assert not (near["validation"] & near["train"])


def test_each_view_partitions_its_dataset(split, full, near):
    for rows, key in ((full, "full96"), (near, "near")):
        seen = [i for k in SPLITS for i in split[key][k]]
        assert len(seen) == len(set(seen)) == len(rows)
        assert set(seen) == {_pid(r) for r in rows}


def test_no_load_or_price_date_crosses_a_split(split, full, near):
    for rows, key in ((full, "full96"), (near, "near")):
        label = {i: k for k in SPLITS for i in split[key][k]}
        by_label = {k: set() for k in SPLITS}
        for row in rows:
            by_label[label[_pid(row)]].update(
                {row["load_date"], row["price_date"]})
        keys = [k for k in SPLITS if k != "holiday_block"]
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                assert not (by_label[a] & by_label[b])


def test_full_only_records_are_training_only(split):
    full_only = set(split["full96_only"])
    assert full_only <= set(split["full96"]["train"])
    assert not full_only & set(split["full96"]["test"])
    assert not full_only & set(split["full96"]["validation"])


def test_counts_and_hashes_match_the_index(split):
    index = json.loads((DATA / "datasets.json").read_text(encoding="utf-8"))
    entry = index["common_split"]
    digest = hashlib.sha256((DATA / "common_split.json").read_bytes()).hexdigest()
    assert entry["sha256"] == digest
    for view in ("common", "full96", "near"):
        assert entry["counts"][view] == split["counts"][view]
    test_index = _rows(DATA / "common_test_index.csv")
    assert {r["pair_id"] for r in test_index} == set(split["common"]["test"])


def test_invariants_accept_a_clean_split_and_reject_leakage():
    import build_common_split as bcs
    full = {"a": "train", "b": "train", "x": "train", "c": "validation",
            "d": "test", "h": "holiday_block"}
    near = {"a": "train", "b": "train", "c": "validation", "d": "test",
            "h": "holiday_block"}
    assert all(bcs.invariants(full, near, ["d"]).values())
    # The same common test record labelled train in FULL96: the check must fail.
    leaky = dict(full, d="train")
    assert bcs.invariants(leaky, near, ["d"])[
        "common_test_not_in_full_train_or_val"] is False


def test_date_uniqueness_catches_a_shared_price_date():
    import build_common_split as bcs
    rows = [{"load_date": "2022-01-01", "price_date": "2025-01-01"},
            {"load_date": "2022-01-02", "price_date": "2025-01-02"}]
    labels = {"2022-01-01|2025-01-01": "train",
              "2022-01-02|2025-01-02": "test"}
    assert bcs._date_uniqueness(rows, labels) is True
    # The same price date under two splits is what the check must catch.
    rows[1]["price_date"] = "2025-01-01"
    labels["2022-01-02|2025-01-01"] = labels.pop("2022-01-02|2025-01-02")
    assert bcs._date_uniqueness(rows, labels) is False
