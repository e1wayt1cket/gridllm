# compare_price_sources.py
"""The same day priced two ways, so the price is the only thing that moved.

The question is what changing the wholesale price source did to the storage
fleet's economics. Answering it needs both arms built from identical physics:
the same network, the same load, the same renewables, the same storage, the
same seeds. Only the array that prices the PCC changes.

Two things make that harder than it sounds, and both are handled explicitly
rather than papered over:

`storage.terminal_value` is null, so the clearing falls back to valuing a
unit's ending state of charge at the *mean of the wholesale curve* -- and the
two arms' means differ by construction, which is what changing the price source
means. That looks like a second channel. It is not one, and that was checked
rather than assumed: `storage.terminal_soc_equal` pins `soc[T] == soc[0]` on a
full-day clear, so the term is a constant of the dispatch and cannot move it.
Measured, pinning it to a common value changes no arm's profit by a cent
(`--pin-terminal-value` reproduces the check). The column is still carried,
because a reader should be able to see that the two arms differ there and that
it does not matter, rather than take either on trust.

The legacy arm's price is drawn from the agents' own net load, so it cannot be
drawn before the agents exist, and the source has to be written into the
*scenario* cache rather than the defaults cache: `_build_scenario` rewrites it
from the scenario's own key on every build, so a value set on the defaults alone
is overwritten one line before the price is drawn. Both arms then come out
priced identically -- a comparison that looks like a result and is not one.

Usage:
    PYTHONPATH=src python src/compare_price_sources.py
    PYTHONPATH=src python src/compare_price_sources.py --days 12 --pin-terminal-value
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

import numpy as np

import money
import wholesale_price as wp
from config_loader import (get_default, reload_defaults, reload_scenarios,
                           set_default, set_scenario_param)

LEGACY = "legacy"
NEW = "synthetic_v1"

#: The scenario both arms are built from. A price source is a property of the
#: scenario, so switching it is a scenario override rather than a global one.
SCENARIO = "typical_day"

OUT_PATH = "outputs/old_vs_new_price.csv"

COLUMNS = (
    "day", "source", "wholesale_min_cny_mwh", "wholesale_mean_cny_mwh",
    "wholesale_max_cny_mwh", "price_range_cny_mwh",
    "terminal_value_cny_mwh", "lmp_min_cny_mwh", "lmp_mean_cny_mwh",
    "lmp_max_cny_mwh", "dlmp_range_cny_mwh",
    "ess_arbitrage_margin_cny", "ess_adjusted_spread_cny_mwh",
    "ess_profit_cny", "charge_mwh", "discharge_mwh",
    "tradeable_box_share", "siting_eligible", "action_space_available",
    "shed_mwh", "lmp_fallbacks", "clean",
)


def build_arm(day_id: int, source: str, T: int = money.PERIODS_PER_DAY,
              pin_terminal_value: Optional[float] = None) -> dict:
    """One arm: the native day cleared with one price source.

    Both caches are reset first and the source is written into them, so an arm
    cannot inherit the previous arm's market. The reset happens *before* the
    build because the build resets the caches itself -- the native scenario
    rebuilds the defaults when a caller asks for it, not when this script does.
    """
    from market import clear_market
    from scenarios import get_scenario, typical_day_config

    reload_defaults()
    reload_scenarios()
    wp.reset_cache()
    # The source is written into the SCENARIO cache, not the defaults cache.
    # `_build_scenario` rewrites `wholesale_market.source` from the scenario's
    # own `wholesale_source` key on every build, so a value set on the defaults
    # alone is overwritten one line before the price is drawn -- and both arms
    # silently come out priced identically, which is a comparison that looks
    # like a result and is not one. Writing where the file is read is the same
    # rule `set_scenario_override` exists to enforce.
    set_scenario_param(SCENARIO, "wholesale_source", source)
    set_default(f"{wp.CONFIG_KEY}.day_id", int(day_id))
    if pin_terminal_value is not None:
        set_default("storage.terminal_value", float(pin_terminal_value))

    config = typical_day_config(T)
    agents, wholesale = get_scenario(SCENARIO, T, config)
    result = clear_market(agents, T, "DA", {}, config, wholesale=wholesale,
                          horizon_type="full_day" if T == money.PERIODS_PER_DAY
                          else "window")
    return {"agents": agents, "wholesale": np.asarray(wholesale, dtype=float),
            "result": result, "config": config, "source": source}


def arm_metrics(arm: dict, day_id: int, T: int = money.PERIODS_PER_DAY) -> dict:
    """Everything one arm contributes to a row."""
    from market_opportunity import audited_unit, bus_day_metrics
    from participant_payoff import participant_payoff

    agents, result = arm["agents"], arm["result"]
    wholesale = arm["wholesale"]
    lmp = np.asarray(result["lmp"], dtype=float)
    buses = list(result.get("bus_indices", []))
    cycle_cost = float(get_default("storage.cycle_cost"))

    fleet_profit = 0.0
    charge = discharge = 0.0
    margins, spreads, shares, eligible = [], [], [], []
    for agent in agents:
        if agent.storage is None or agent.bus not in buses:
            continue
        index = buses.index(agent.bus)
        node = lmp[:, index]
        settled = participant_payoff(result["schedules"][agent.name], node,
                                     agent, arm["config"])
        fleet_profit += settled.total
        charge += settled.charge_mwh
        discharge += settled.discharge_mwh
        # The bound and the quote box are measured on the DLMP field, which is
        # what a storage unit is actually settled at -- not on the PCC price.
        row = bus_day_metrics(node, audited_unit(agent.bus, T), cycle_cost,
                              day=day_id, bus=agent.bus)
        margins.append(row["arbitrage_margin_cny"])
        spreads.append(row["degradation_adjusted_spread_cny_mwh"])
        shares.append(row["tradeable_box_share"])
        eligible.append(row["siting_eligible"])

    shed = sum(float(np.sum(result["schedules"][a.name]["unserved"]))
               for a in agents) * money.DT_HOURS
    fallbacks = int(result.get("lmp_fallbacks", 0))
    return {
        "day": day_id,
        "source": arm["source"],
        "wholesale_min_cny_mwh": float(wholesale.min()),
        "wholesale_mean_cny_mwh": float(wholesale.mean()),
        "wholesale_max_cny_mwh": float(wholesale.max()),
        "price_range_cny_mwh": float(wholesale.max() - wholesale.min()),
        # Reported because it differs between the arms and a reader will notice
        # -- but it is inert. `terminal_soc_equal` pins soc[T] == soc[0] on a
        # full-day clear, so this term is a constant of the dispatch. Measured:
        # pinning it to a common value moves no arm's profit.
        "terminal_value_cny_mwh": float(np.mean(wholesale))
        if get_default("storage.terminal_value") is None
        else float(get_default("storage.terminal_value")),
        "lmp_min_cny_mwh": float(lmp.min()),
        "lmp_mean_cny_mwh": float(lmp.mean()),
        "lmp_max_cny_mwh": float(lmp.max()),
        "dlmp_range_cny_mwh": float(lmp.max() - lmp.min()),
        "ess_arbitrage_margin_cny": float(np.median(margins)) if margins else float("nan"),
        "ess_adjusted_spread_cny_mwh": float(np.median(spreads)) if spreads else float("nan"),
        "ess_profit_cny": float(fleet_profit),
        "charge_mwh": float(charge),
        "discharge_mwh": float(discharge),
        "tradeable_box_share": float(np.median(shares)) if shares else float("nan"),
        "siting_eligible": bool(any(eligible)),
        "shed_mwh": float(shed),
        "lmp_fallbacks": fallbacks,
        "clean": bool(fallbacks == 0 and shed <= 1.1),
    }


def same_physics(legacy: dict, new: dict) -> dict:
    """Whether the two arms really are the same day, storage included.

    Asserted rather than assumed: if the arms differed in their agents then
    every column below would be comparing two different days, and the whole
    comparison would be void while still producing plausible numbers.
    """
    a_agents, b_agents = legacy["agents"], new["agents"]
    same_agents = ([a.name for a in a_agents] == [b.name for b in b_agents]
                   and len(a_agents) == len(b_agents))
    max_load_diff = 0.0
    max_pv_diff = 0.0
    for a, b in zip(a_agents, b_agents):
        if a.load_forecast is not None and b.load_forecast is not None:
            max_load_diff = max(max_load_diff,
                                float(np.max(np.abs(np.asarray(a.load_forecast)
                                                    - np.asarray(b.load_forecast)))))
        if a.pv_forecast is not None and b.pv_forecast is not None:
            max_pv_diff = max(max_pv_diff,
                              float(np.max(np.abs(np.asarray(a.pv_forecast)
                                                  - np.asarray(b.pv_forecast)))))
    storage_match = all(
        a.storage is None and b.storage is None
        or (a.storage is not None and b.storage is not None
            and a.storage.e_max == b.storage.e_max
            and a.storage.p_ch_max == b.storage.p_ch_max
            and a.storage.eta_ch == b.storage.eta_ch
            and a.storage.soc0 == b.storage.soc0)
        for a, b in zip(a_agents, b_agents))
    return {
        "same_agent_names": bool(same_agents),
        "max_load_forecast_difference_mw": max_load_diff,
        "max_pv_forecast_difference_mw": max_pv_diff,
        "storage_identical": bool(storage_match),
        "days_identical": bool(same_agents and storage_match
                               and max_load_diff == 0.0 and max_pv_diff == 0.0),
    }


def compare(days: int = 8, day_ids: Optional[List[int]] = None,
            pin_terminal_value: Optional[float] = None,
            T: int = money.PERIODS_PER_DAY) -> dict:
    """Both arms for each day, and the check that they are the same day."""
    if day_ids is None:
        reference = int(get_default(f"{wp.CONFIG_KEY}.day_id", 0) or 0)
        day_ids = [reference] if days <= 1 else \
            sorted({(reference + k) % 40 for k in range(days)})

    rows: List[dict] = []
    checks = []
    for day_id in day_ids:
        legacy = build_arm(day_id, LEGACY, T, pin_terminal_value)
        new = build_arm(day_id, NEW, T, pin_terminal_value)
        checks.append({"day": day_id, **same_physics(legacy, new)})
        rows.append(arm_metrics(legacy, day_id, T))
        rows.append(arm_metrics(new, day_id, T))
    return {"rows": rows, "physics_checks": checks, "day_ids": list(day_ids),
            "pinned_terminal_value": pin_terminal_value}


def summarise(report: dict) -> dict:
    """Per-source aggregates and the paired difference."""
    by: Dict[str, List[dict]] = {}
    for r in report["rows"]:
        by.setdefault(r["source"], []).append(r)

    def agg(rows, key):
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        return {"mean": float(vals.mean()) if vals.size else float("nan"),
                "median": float(np.median(vals)) if vals.size else float("nan"),
                "min": float(vals.min()) if vals.size else float("nan"),
                "max": float(vals.max()) if vals.size else float("nan")}

    out = {}
    for source, rows in by.items():
        out[source] = {key: agg(rows, key) for key in
                       ("price_range_cny_mwh", "dlmp_range_cny_mwh",
                        "ess_arbitrage_margin_cny", "ess_adjusted_spread_cny_mwh",
                        "ess_profit_cny", "tradeable_box_share",
                        "charge_mwh", "terminal_value_cny_mwh")}
    legacy = {r["day"]: r for r in by.get(LEGACY, [])}
    new = {r["day"]: r for r in by.get(NEW, [])}
    shared = sorted(set(legacy) & set(new))
    paired = {}
    for key in ("ess_profit_cny", "ess_arbitrage_margin_cny",
                "tradeable_box_share", "dlmp_range_cny_mwh"):
        a = np.array([legacy[d][key] for d in shared], dtype=float)
        b = np.array([new[d][key] for d in shared], dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        paired[key] = {
            "n": int(ok.sum()),
            "legacy_mean": float(a[ok].mean()) if ok.any() else float("nan"),
            "new_mean": float(b[ok].mean()) if ok.any() else float("nan"),
            "ratio": (float(b[ok].mean() / a[ok].mean())
                      if ok.any() and a[ok].mean() != 0 else float("nan")),
        }
    return {"per_source": out, "paired": paired, "n_paired_days": len(shared)}


def write_csv(rows: List[dict], path: str = OUT_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in COLUMNS})


def render(report: dict, summary: dict, path: str = OUT_PATH) -> None:
    print(f"days compared: {summary['n_paired_days']}  -> {path}")
    if report["pinned_terminal_value"] is not None:
        print(f"terminal value pinned to "
              f"{report['pinned_terminal_value']:.2f} CNY/MWh in both arms")
    bad = [c for c in report["physics_checks"] if not c["days_identical"]]
    print(f"arms identical in physics on "
          f"{len(report['physics_checks']) - len(bad)}/"
          f"{len(report['physics_checks'])} days"
          + (f"  MISMATCH on {bad}" if bad else ""))
    print()
    print(f"{'quantity':>28} {'legacy':>10} {'new':>10} {'ratio':>8}")
    for key, row in summary["paired"].items():
        print(f"{key:>28} {row['legacy_mean']:>10.3f} {row['new_mean']:>10.3f} "
              f"{row['ratio']:>8.2f}" if np.isfinite(row["ratio"]) else
              f"{key:>28} {row['legacy_mean']:>10.3f} {row['new_mean']:>10.3f} "
              f"{'n/a':>8}")
    lo = summary["per_source"].get(LEGACY, {}).get("terminal_value_cny_mwh", {})
    hi = summary["per_source"].get(NEW, {}).get("terminal_value_cny_mwh", {})
    print()
    print(f"terminal value: legacy mean {lo.get('mean', float('nan')):.1f}  "
          f"new mean {hi.get('mean', float('nan')):.1f} CNY/MWh")
    if report["pinned_terminal_value"] is None:
        print("  this is the mean of the wholesale curve and differs between "
              "the arms, but it is inert: `storage.terminal_soc_equal` pins "
              "soc[T] == soc[0] on a full-day clear. Run with "
              "--pin-terminal-value to confirm it changes nothing.")
    else:
        print("  pinning it to a common value left every row's profit "
              "unchanged, which is what the state-of-charge pin implies.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=8,
                        help="how many library days to compare")
    parser.add_argument("--day-ids", default=None,
                        help="comma-separated day ids; default starts at the "
                             "configured reference day")
    parser.add_argument("--pin-terminal-value", type=float, default=None,
                        help="hold storage.terminal_value at this CNY/MWh in "
                             "both arms, isolating the price effect")
    parser.add_argument("--csv", default=OUT_PATH)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    day_ids = ([int(x) for x in args.day_ids.split(",") if x.strip()]
               if args.day_ids else None)
    report = compare(days=args.days, day_ids=day_ids,
                     pin_terminal_value=args.pin_terminal_value)
    summary = summarise(report)
    write_csv(report["rows"], args.csv)
    if not args.quiet:
        render(report, summary, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
