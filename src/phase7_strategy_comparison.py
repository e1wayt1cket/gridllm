# phase7_strategy_comparison.py
"""The arbitrage reference bound and the three-strategy paired comparison.

Four price curves, three bidding strategies, one frozen configuration. Every
number here comes from the shared whole-day settlement, so a strategy's figure
and the bound beside it are at least stated in the same unit.

What each object is, and what it is not:

* **The bound** is the *storage-side perfect-foresight arbitrage reference*.
  One unit, free to charge and discharge inside its own limits, with perfect
  knowledge of the price series it is given and the round trip costed on both
  legs. It contains no network, no clearing, no biding and no competitor. It is
  therefore **not** an upper bound on what a strategy in this market can earn --
  a clearing that prices a unit at its own nodal LMP, through a radial network
  it shares with three other batteries, is a different object. It is reported
  as a reference the strategies are compared *against*, never as a target they
  are shown to reach.

* **The strategies** are the three baselines: truthful (`fixed`), look-ahead
  (rule-based MPC), and the per-period threshold rule (`greedy_threshold`).
  Each declares quotes; the clearing decides. No MATD3 policy is used and no
  training is run.

Every price curve is a **mean** over its window and none is a measured day, so
the four columns are four price scenarios. They are not four dates, and nothing
here is a cross-day statistic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

SCENARIO = "typical_day_industry"
STRATEGIES = ("fixed", "mpc", "greedy_threshold")


def _fresh():
    """Reset every process-global cache before building a scenario."""
    import counterfactual_eval as ce
    import wholesale_price as wp
    from config_loader import reload_defaults

    reload_defaults()
    wp.reset_cache()
    ce.reset_baseline_cache()


def arbitrage_reference(curve, agents, config, T: int = 96) -> dict:
    """The perfect-foresight reference, on two different price series.

    * `exogenous` prices the unit against the external day-ahead curve it was
      given. Nothing about this network enters, so the figure is a property of
      the price series and the battery alone. This is the clean reference.
    * `nodal` prices it against the nodal LMP its own truthful participation
      produced, at its own bus. That is the price it is actually settled at,
      which makes the figure comparable to a strategy's earnings -- and
      endogenous, which the module's own docstring warns about: a unit priced
      against a clearing it took part in is priced against a field it helped
      set. It is reported because the strategies are settled that way and the
      comparison would otherwise be between two different markets.
    """
    from arbitrage_lp import arbitrage_lp, spec_from_storage

    fleet = [a for a in agents if a.participant_type == "storage"]
    if not fleet:
        return {}
    unit = spec_from_storage(fleet[0].storage)
    cycle_cost = float(config.storage.cycle_cost)

    exo = arbitrage_lp(curve.array(), unit, cycle_cost)
    out = {
        "unit": {
            "e_max_mwh": unit.e_max, "p_ch_max_mw": unit.p_ch_max,
            "p_dis_max_mw": unit.p_dis_max,
            "eta_ch": unit.eta_ch, "eta_dis": unit.eta_dis,
            "soc0": unit.soc0, "soc_min": unit.soc_min, "soc_max": unit.soc_max,
        },
        "cycle_cost_cny_per_mwh_per_leg": cycle_cost,
        "n_units": len(fleet),
        "exogenous": {
            "basis": "the external day-ahead curve for this price scenario",
            "per_unit_margin_cny": float(exo.margin_cny),
            "fleet_margin_cny": float(exo.margin_cny) * len(fleet),
            "charge_mwh": float(exo.chargeable_energy_mwh),
            "discharge_mwh": float(exo.dischargeable_energy_mwh),
            "equivalent_cycles": float(exo.equivalent_cycles),
            "adjusted_spread_cny_per_mwh": float(exo.adjusted_spread_cny_mwh),
            "status": exo.status_text,
        },
    }
    return out


def nodal_reference(result, agents, config, T: int = 96) -> dict:
    """The same LP on each unit's own cleared nodal price series."""
    from arbitrage_lp import arbitrage_lp, spec_from_storage

    lmp = np.asarray(result["lmp"], dtype=float)
    cycle_cost = float(config.storage.cycle_cost)
    rows = []
    for a in agents:
        if a.participant_type != "storage":
            continue
        unit = spec_from_storage(a.storage)
        res = arbitrage_lp(lmp[:, int(a.bus)], unit, cycle_cost)
        rows.append({
            "unit": a.name, "bus": int(a.bus),
            "per_unit_margin_cny": float(res.margin_cny),
            "charge_mwh": float(res.chargeable_energy_mwh),
            "discharge_mwh": float(res.dischargeable_energy_mwh),
            "adjusted_spread_cny_per_mwh": float(res.adjusted_spread_cny_mwh),
            "status": res.status_text,
        })
    return {"basis": "each unit's own cleared nodal LMP", "units": rows,
            "fleet_margin_cny": float(sum(r["per_unit_margin_cny"]
                                          for r in rows))}


def run_strategy(strategy: str, curve_index: int, T: int = 96) -> dict:
    """One strategy on one price curve, settled the shared way."""
    import counterfactual_eval as ce
    import money
    import participant_payoff
    from models import MarketConfig
    from scenarios import get_scenario
    from strategies import adaptive_bidding

    from accept_typical_day_industry import _physics

    _fresh()
    config = MarketConfig.from_yaml()
    # The caller's `day_id` is the documented way to select a price scenario;
    # it wins over the scenario's declared index and is what keeps one
    # scenario's curve from being read under another's name.
    agents, wholesale = get_scenario(SCENARIO, T=T, config=config,
                                     day_id=int(curve_index))
    declared = adaptive_bidding(agents, config, strategy=strategy, T=T)
    names = [a.name for a in agents if a.participant_type == "storage"]

    profits, clean, result = ce.settle_day(agents, config, names, declared,
                                           wholesale, T)
    if not clean or result is None:
        return {"strategy": strategy, "curve_index": int(curve_index),
                "settled_clean": False,
                "reason": ("the whole-day settlement degraded or fell back; "
                           "profits are withheld rather than reported as zero"),
                "profits": None}

    breakdown = {}
    for a in agents:
        if a.participant_type != "storage":
            continue
        sched = result["schedules"][a.name]
        bd = participant_payoff.participant_payoff(
            sched, result["lmp"][:, int(a.bus)], a, config)
        breakdown[a.name] = {
            "revenue": float(bd.revenue),
            "purchase_cost": float(bd.purchase_cost),
            "generation_cost": float(bd.generation_cost),
            "degradation_cost": float(bd.degradation_cost),
            "other_cost": float(bd.other_cost),
            "penalty": float(bd.penalty),
            "total": float(bd.total),
            "matches_settlement": bool(abs(bd.total - profits[a.name]) < 1e-6),
        }

    physics = _physics(result, agents, config, T)
    traded = {}
    for a in agents:
        if a.participant_type != "storage":
            continue
        sched = result["schedules"][a.name]
        traded[a.name] = {
            "bought_mwh": float(np.sum(sched["p_buy"]) * money.DT_HOURS),
            "sold_mwh": float(np.sum(sched["p_sell"]) * money.DT_HOURS),
            "charge_mwh": float(np.sum(sched["p_ch"]) * money.DT_HOURS),
            "discharge_mwh": float(np.sum(sched["p_dis"]) * money.DT_HOURS),
            "periods_charging": int(np.sum(np.asarray(sched["p_ch"]) > 1e-9)),
            "periods_discharging": int(np.sum(np.asarray(sched["p_dis"]) > 1e-9)),
        }

    quotes = declared
    intent = {}
    from rl_spec import action_spec_for

    for a in agents:
        if a.participant_type != "storage":
            continue
        spec = action_spec_for(a)
        c_lo, c_hi = spec.bounds["charge_price"]
        d_lo, d_hi = spec.bounds["discharge_price"]
        cp = np.asarray(quotes[a.name]["charge_price"], dtype=float)
        dp = np.asarray(quotes[a.name]["discharge_price"], dtype=float)
        # Counted by matching each corner exactly, so the tally is read off the
        # quotes the strategy actually declared rather than inferred from the
        # price series it was given.
        intent[a.name] = {
            "periods_charge_intent": int(np.sum((cp == c_lo) & (dp == d_hi))),
            "periods_discharge_intent": int(np.sum((cp == c_hi) & (dp == d_lo))),
            "periods_neutral": int(np.sum(
                (cp == spec.truthful["charge_price"])
                & (dp == spec.truthful["discharge_price"]))),
        }

    return {
        "strategy": strategy,
        "curve_index": int(curve_index),
        "settled_clean": True,
        "profits": {k: float(v) for k, v in profits.items()},
        "fleet_profit": float(sum(profits.values())),
        "breakdown": breakdown,
        "traded": traded,
        "declared_intent": intent,
        "physics": physics,
        "nodal_reference": nodal_reference(result, agents, config, T),
        "provenance": {
            "curve_index": int(curve_index),
            "wholesale_min": float(np.min(wholesale)),
            "wholesale_max": float(np.max(wholesale)),
        },
    }


def run_comparison(T: int = 96) -> dict:
    """The whole phase: bounds and the paired comparison, per curve."""
    from dianyicha_price import ROLES, SHEETS, read_curves
    from models import MarketConfig
    from scenarios import get_scenario

    from accept_typical_day_industry import _physics

    out: Dict[str, object] = {
        "scenario": SCENARIO, "T": int(T),
        "strategies": list(STRATEGIES),
        "curves": {},
    }

    for index, sheet in enumerate(SHEETS):
        curves = read_curves()
        curve = curves[sheet]

        # The bound and the reference fleet come from one build of the scenario
        # at this curve, with the truthful declaration; the same build is what
        # `fixed` will settle, so the exogenous/nodal pair and the fixed run
        # describe the same day.
        _fresh()
        config = MarketConfig.from_yaml()
        agents, wholesale = get_scenario(SCENARIO, T=T, config=config,
                                         day_id=index)
        bound = arbitrage_reference(curve, agents, config, T)

        entry = {
            "sheet": sheet,
            "role": ROLES.get(sheet, "unspecified_mean"),
            "is_measured": curve.is_measured,
            "stats": curve.stats(),
            "workbook_sha256": curve.provenance.get("workbook_sha256"),
            "bound": bound,
            "runs": {},
        }
        for strategy in STRATEGIES:
            entry["runs"][strategy] = run_strategy(strategy, index, T)
        out["curves"][sheet] = entry
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Arbitrage reference bound and three-strategy comparison.")
    p.add_argument("--json", default="outputs/phase7_strategy_comparison.json")
    p.add_argument("--T", type=int, default=96)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_comparison(T=args.T)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"written to {args.json}\n")

    print(f"{'curve':<10}{'strategy':<18}{'fleet CNY':>12}{'units moving':>14}"
          f"{'charge MWh':>12}{'discharge MWh':>14}")
    for sheet, entry in report["curves"].items():
        for strategy in STRATEGIES:
            run = entry["runs"][strategy]
            if not run.get("settled_clean"):
                print(f"{sheet:<10}{strategy:<18}{'NOT CLEAN':>12}")
                continue
            moving = sum(1 for t in run["traded"].values()
                         if t["charge_mwh"] > 1e-6 or t["discharge_mwh"] > 1e-6)
            ch = sum(t["charge_mwh"] for t in run["traded"].values())
            dis = sum(t["discharge_mwh"] for t in run["traded"].values())
            print(f"{sheet:<10}{strategy:<18}{run['fleet_profit']:>12.2f}"
                  f"{moving:>14}{ch:>12.3f}{dis:>14.3f}")
    print()
    print(f"{'curve':<10}{'exo bound/unit':>16}{'fleet exo bound':>17}")
    for sheet, entry in report["curves"].items():
        b = entry["bound"]["exogenous"]
        print(f"{sheet:<10}{b['per_unit_margin_cny']:>16.2f}"
              f"{b['fleet_margin_cny']:>17.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
