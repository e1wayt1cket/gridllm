"""Read the v3 candidate-day library and cut clearing windows out of it."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from local_da_contracts import DayAheadInput, StorageBid, StorageUnit
from run_local_da import (PRICE_CURVES, V3, production_renewable_curves,
                          radial_network, sha)

LIBRARY = V3
DT_HOURS = 0.25
STORAGE_BUSES = (9, 14, 17, 27)
# The truthful point of the storage-quote box; rl_spec_v7 declares the same vector.
TRUTHFUL_QUOTE = np.array([0., .4, 0., .4])
EXACT_INFEASIBLE = "Exact complementary dispatch was not optimal"


def load_pair(load_date):
    """One library day with its paired price date, curves and split block."""
    split = json.loads((LIBRARY / "split.json").read_text(encoding="utf-8"))
    matches = [(block, entry) for block, rows in split["pairs"].items() for entry in rows
               if entry.split("|")[0] == load_date]
    if len(matches) != 1:
        raise ValueError("The requested date is not a unique member of the v3 library")
    block, entry = matches[0]
    price_date = entry.split("|")[1]
    metadata = json.loads((LIBRARY / "manifest.json").read_text(encoding="utf-8"))
    for name in ["bus_active_load_mw.npz", "bus_reactive_load_mvar.npz"]:
        if sha(LIBRARY / name) != metadata["outputs"][name]["sha256"]:
            raise ValueError(f"Changed frozen v3 load output: {name}")
    with np.load(LIBRARY / "bus_active_load_mw.npz", allow_pickle=False) as archive:
        active = archive[load_date].copy()
    with np.load(LIBRARY / "bus_reactive_load_mvar.npz", allow_pickle=False) as archive:
        reactive = archive[load_date].copy()
    with np.load(PRICE_CURVES, allow_pickle=False) as archive:
        price = archive[f"south/{price_date}"].copy()
    return {"load_date": load_date, "price_date": price_date, "block": block,
            "active": active, "reactive": reactive, "external_price": price}


def train_pairs(screen_dir):
    """Feasible train pairs, plus the days an exact screen dropped and its reason."""
    split = json.loads((LIBRARY / "split.json").read_text(encoding="utf-8"))
    pairs, dropped = [], {}
    for entry in split["pairs"]["train"]:
        load_date, price_date = entry.split("|")
        record = json.loads((Path(screen_dir) / f"{load_date}.json").read_text(encoding="utf-8"))
        failure = record.get("failure")
        if failure and failure.startswith(EXACT_INFEASIBLE):
            dropped[load_date] = failure
            continue
        pairs.append({"load_date": load_date, "price_date": price_date, "block": "train"})
    return pairs, {"dropped": dropped}


def window_day(pair, t_start, t_end, *, soc, quotes):
    """Slice one day into a clearing window with the carried SOC and per-unit quotes."""
    periods = t_end - t_start
    storage = tuple(StorageUnit(f"ESS{bus}", bus, soc_initial=soc[f"ESS{bus}"])
                    for bus in STORAGE_BUSES)
    bids = {}
    for unit in storage:
        charge_price, charge_quantity, discharge_price, discharge_quantity = quotes.get(
            unit.name, TRUTHFUL_QUOTE)
        bids[unit.name] = StorageBid(
            charge_quantity_mw=np.full(periods, charge_quantity),
            discharge_quantity_mw=np.full(periods, discharge_quantity),
            charge_price_cny_mwh=np.full(periods, charge_price),
            discharge_price_cny_mwh=np.full(periods, discharge_price))
    generators = tuple(replace(g, available_mw=g.available_mw[t_start:t_end])
                       for g in production_renewable_curves())
    return DayAheadInput(radial_network(), pair["active"][t_start:t_end],
                         pair["reactive"][t_start:t_end],
                         pair["external_price"][t_start:t_end], storage, generators, bids,
                         0., 600., terminal_soc_equal=False)
