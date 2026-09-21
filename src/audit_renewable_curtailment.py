# audit_renewable_curtailment.py
"""Why renewable output is curtailed, period by period, traced to the objective.

The renewable consumption rate fell from 98% to 19.7% when the native
scenario's price source changed from the retired net-load map to the external
wholesale market. This module answers *why*, by naming the coefficients that
actually decide it rather than the ones that look like they should.

For a non-storage agent the only objective terms involving PV and wind are

    obj += bid   * served                      (line 557)
    obj -= offer * (pv + wind + dis)           (line 573)
    obj -= wholesale[t] * p_grid[t]            (line 577)

and the carbon, RE-incentive and curtailment-penalty terms are gated behind
`market_design.enable_multi_objective`, which is **false**. So `lambda_re: 50`
and `lambda_curtail: 200` are declared and inert, and there is no curtailment
penalty in the objective at all.

Dispatching one more MWh of local renewable displaces one MWh of import. The
`bid * served` term is identical either way -- consumption is worth the same
whoever supplies it -- so it cannot enter the decision. What is left is

    dispatch iff  wholesale[t]  >  offer_cost

`bid_value` does not appear. A first pass at this explained the drop by
comparing the wholesale price against the prosumer bid values of 350-720; that
is wrong, and this module measures the comparison that is actually made.

Curtailment can also be forced by the network -- a voltage band or a line limit
-- rather than chosen by the objective, so each period's constraint status is
reported alongside. Those two causes look identical in a consumption rate and
call for opposite responses.

Usage:
    PYTHONPATH=src python src/audit_renewable_curtailment.py
    PYTHONPATH=src python src/audit_renewable_curtailment.py --day 11 --json out.json
"""

import argparse
import csv
import json
import os
import sys
from typing import List

import numpy as np

import money
import wholesale_price as wp
from config_loader import get_default, reload_defaults, reload_scenarios, set_default

OUT_CSV = "outputs/renewable_curtailment.csv"
OUT_MD = "outputs/renewable_curtailment_audit.md"


def renewable_agents(agents) -> List:
    """Agents carrying a declared renewable capacity."""
    return [a for a in agents
            if getattr(a, "pv_capacity", 0.0) > 0.0
            or getattr(a, "wind_capacity", 0.0) > 0.0]


def clear(day_id: int, T: int = money.PERIODS_PER_DAY) -> dict:
    """Clear the native day on the external market, unchanged."""
    from market import clear_market
    from scenarios import get_scenario, typical_day_config

    reload_defaults()
    reload_scenarios()
    wp.reset_cache()
    set_default(f"{wp.CONFIG_KEY}.day_id", int(day_id))
    config = typical_day_config(T)
    agents, wholesale = get_scenario("typical_day", T, config)
    result = clear_market(agents, T, "DA", {}, config, wholesale=wholesale,
                          horizon_type="full_day")
    return {"agents": agents, "wholesale": np.asarray(wholesale, dtype=float),
            "result": result, "config": config}


def period_rows(day: dict, T: int = money.PERIODS_PER_DAY) -> List[dict]:
    """One row per renewable agent per period, with the decision made explicit."""
    agents = renewable_agents(day["agents"])
    result = day["result"]
    wholesale = day["wholesale"]
    buses = list(result.get("bus_indices", []))
    voltage = result.get("bus_voltage")
    line_util = result.get("line_utilization")
    v_min = float(day["config"].network.v_min_pu)
    v_max = float(day["config"].network.v_max_pu)

    rows = []
    for a in agents:
        sched = result["schedules"][a.name]
        pv_avail = np.asarray(a.pv_forecast, dtype=float)
        wind_avail = (np.asarray(a.wind_forecast, dtype=float)
                      if a.has_wind and a.wind_forecast is not None
                      else np.zeros(T))
        pv_used = np.asarray(sched["pv_used"], dtype=float)
        wind_used = np.asarray(sched["wind_used"], dtype=float)
        bus_i = buses.index(a.bus) if a.bus in buses else None

        for t in range(T):
            available = float(max(pv_avail[t], 0.0) + max(wind_avail[t], 0.0))
            dispatched = float(pv_used[t] + wind_used[t])
            curtailed = max(0.0, available - dispatched)
            v = (float(np.sqrt(max(voltage[t, bus_i], 0.0)))
                 if voltage is not None and bus_i is not None else float("nan"))
            worst_line = (float(np.max(line_util[t]))
                          if line_util is not None else float("nan"))
            # The objective's own dispatch condition for this period. Strictly,
            # congestion can break the substitution, which is why the network
            # columns are reported beside it.
            wants = float(wholesale[t]) > float(a.offer_cost)
            rows.append({
                "day": int(day.get("day_id", day_id_of(day))),
                "time_index": t,
                "hour": t * 0.25,
                "agent": a.name,
                "bus": int(a.bus),
                "load_type": a.load_type,
                "wholesale_cny_mwh": float(wholesale[t]),
                "offer_cost_cny_mwh": float(a.offer_cost),
                "bid_value_cny_mwh": float(a.bid_value),
                "offer_margin_cny_mwh": float(wholesale[t] - a.offer_cost),
                "objective_wants_dispatch": bool(wants),
                "pv_available_mw": float(max(pv_avail[t], 0.0)),
                "wind_available_mw": float(max(wind_avail[t], 0.0)),
                "renewable_available_mw": available,
                "pv_dispatched_mw": float(pv_used[t]),
                "wind_dispatched_mw": float(wind_used[t]),
                "renewable_dispatched_mw": dispatched,
                "curtailed_mw": curtailed,
                "curtailed_mwh": curtailed * money.DT_HOURS,
                "serve_load_mw": float(sched["served"][t]),
                "voltage_pu": v,
                "voltage_violation": bool(np.isfinite(v)
                                          and (v < v_min - 1e-6 or v > v_max + 1e-6)),
                "worst_line_utilization": worst_line,
                "line_violation": bool(np.isfinite(worst_line)
                                       and worst_line > 1.0 + 1e-6),
            })
    return rows


def day_id_of(day: dict) -> int:
    return int(get_default(f"{wp.CONFIG_KEY}.day_id", 0) or 0)


def summarise(rows: List[dict]) -> dict:
    """Separate the curtailment the objective chose from the network's."""
    avail = np.array([r["renewable_available_mw"] for r in rows])
    disp = np.array([r["renewable_dispatched_mw"] for r in rows])
    curt = np.array([r["curtailed_mw"] for r in rows])
    wants = np.array([r["objective_wants_dispatch"] for r in rows])
    # Curtailment in periods the network was binding is attributed to the
    # network; curtailment in periods the objective did not want it is
    # attributed to the comparison; anything left is a residual the two named
    # causes do not explain and is reported rather than absorbed.
    net_binding = np.array([r["voltage_violation"] or r["line_violation"]
                            for r in rows])
    econ_curt = np.where(~wants & ~net_binding, curt, 0.0)
    net_curt = np.where(net_binding, curt, 0.0)
    unexplained = np.where(wants & ~net_binding, curt, 0.0)

    total_curt_mwh = float(curt.sum()) * money.DT_HOURS
    return {
        "n_periods": len(rows),
        "available_mwh": float(avail.sum()) * money.DT_HOURS,
        "dispatched_mwh": float(disp.sum()) * money.DT_HOURS,
        "curtailed_mwh": total_curt_mwh,
        "consumption_rate": (float(disp.sum() / avail.sum() * 100.0)
                             if avail.sum() > 0 else float("nan")),
        "curtailed_by_objective_mwh": float(econ_curt.sum()) * money.DT_HOURS,
        "curtailed_by_network_mwh": float(net_curt.sum()) * money.DT_HOURS,
        "curtailed_unexplained_mwh": float(unexplained.sum()) * money.DT_HOURS,
        "periods_where_objective_wants_dispatch": int(wants.sum()),
        "periods_network_binding": int(net_binding.sum()),
        "wholesale_mean_when_curtailing": (
            float(np.mean([r["wholesale_cny_mwh"] for r in rows if r["curtailed_mw"] > 1e-9]))
            if any(r["curtailed_mw"] > 1e-9 for r in rows) else float("nan")),
        "offer_cost_mean": float(np.mean([r["offer_cost_cny_mwh"] for r in rows])),
        "wholesale_mean": float(np.mean([r["wholesale_cny_mwh"] for r in rows])),
    }


def by_agent(rows: List[dict]) -> List[dict]:
    out = []
    for name in sorted({r["agent"] for r in rows}):
        sub = [r for r in rows if r["agent"] == name]
        s = summarise(sub)
        s["agent"] = name
        s["bus"] = sub[0]["bus"]
        s["load_type"] = sub[0]["load_type"]
        s["offer_cost_cny_mwh"] = sub[0]["offer_cost_cny_mwh"]
        s["bid_value_cny_mwh"] = sub[0]["bid_value_cny_mwh"]
        # The counterfactual the module exists to test: how much of the day the
        # market's own price stood above this agent's offer cost.
        s["share_of_day_above_offer"] = float(np.mean(
            [r["objective_wants_dispatch"] for r in sub]))
        out.append(s)
    return out


def threshold_sweep(rows: List[dict]) -> List[dict]:
    """What the consumption rate would be at each offer cost.

    The return-on-one-number view of the same finding: the rate is a step
    function of where each agent's `offer_cost` sits relative to the wholesale
    price, so a single changed coefficient moves it a long way.
    """
    avail = {}
    for r in rows:
        avail.setdefault(r["agent"], []).append(r)
    out = []
    for offer in (100.0, 150.0, 180.0, 200.0, 250.0, 300.0, 400.0, 600.0):
        disp = 0.0
        total = 0.0
        for agent, rs in avail.items():
            for r in rs:
                a = r["renewable_available_mw"]
                total += a
                disp += a if r["wholesale_cny_mwh"] > offer else 0.0
        out.append({"offer_cost_cny_mwh": offer,
                    "consumption_rate": (disp / total * 100.0) if total else 0.0})
    return out


def render(report: dict) -> str:
    s, per, sweep = report["summary"], report["by_agent"], report["threshold_sweep"]
    L = []
    L.append("# Renewable curtailment: the coefficients that decide it")
    L.append("")
    L.append(f"Native day {report['day']} at the reference price source "
             f"`{report['source']}`, unchanged. {s['n_periods']} agent-periods.")
    L.append("")
    L.append("## The objective terms that can move PV and wind")
    L.append("")
    L.append("For an agent without storage, `dispatch_socp` writes exactly three "
             "terms touching a renewable:")
    L.append("")
    L.append("| term | coefficient | does it enter the dispatch decision? |")
    L.append("|---|---|---|")
    L.append("| `+ bid * served` | 350 / 720 | **no** -- consumption is worth "
             "the same whether local PV or an import supplies it, so the term "
             "cancels out of the comparison |")
    L.append("| `- offer * (pv + wind + dis)` | 180 / 300 | **yes** |")
    L.append("| `- wholesale[t] * p_grid[t]` | market price | **yes** |")
    L.append("")
    L.append("The carbon, RE-incentive and curtailment-penalty terms are gated "
             "behind `market_design.enable_multi_objective`, which is "
             f"**{report['enable_multi_objective']}**. So `lambda_re` "
             f"({report['lambda_re']}) and `lambda_curtail` "
             f"({report['lambda_curtail']}) are declared and **inert**: there is "
             "no curtailment penalty in the objective at all.")
    L.append("")
    L.append("Dispatching one more MWh locally displaces one MWh of import, so "
             "the whole decision is")
    L.append("")
    L.append("```")
    L.append("dispatch  iff  wholesale[t]  >  offer_cost")
    L.append("```")
    L.append("")
    L.append("**`bid_value` does not appear.** An earlier explanation compared "
             "the wholesale price against the prosumer bid values of 350-720; "
             "that names a coefficient which cannot enter the decision, and it "
             "is withdrawn.")
    L.append("")
    L.append("## What happened on this day")
    L.append("")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append(f"| renewable available | {s['available_mwh']:.4f} MWh |")
    L.append(f"| renewable dispatched | {s['dispatched_mwh']:.4f} MWh |")
    L.append(f"| curtailed | {s['curtailed_mwh']:.4f} MWh |")
    L.append(f"| **consumption rate** | **{s['consumption_rate']:.2f}%** |")
    L.append(f"| mean wholesale price | {s['wholesale_mean']:.2f} CNY/MWh |")
    L.append(f"| mean offer cost | {s['offer_cost_mean']:.2f} CNY/MWh |")
    L.append("")
    L.append("Attribution of the curtailment:")
    L.append("")
    L.append("| cause | MWh |")
    L.append("|---|---|")
    L.append(f"| the objective did not want it (`wholesale <= offer`) | "
             f"{s['curtailed_by_objective_mwh']:.4f} |")
    L.append(f"| the network was binding (voltage or line) | "
             f"{s['curtailed_by_network_mwh']:.4f} |")
    L.append(f"| unexplained (wanted, unconstrained, still curtailed) | "
             f"{s['curtailed_unexplained_mwh']:.4f} |")
    L.append("")
    L.append(f"The network bound in {s['periods_network_binding']} of "
             f"{s['n_periods']} agent-periods.")
    L.append("")
    L.append("## Per renewable agent")
    L.append("")
    L.append("| agent | bus | type | offer cost | bid value | share of day with "
             "wholesale > offer | consumption rate |")
    L.append("|---|---|---|---|---|---|---|")
    for a in per:
        L.append(f"| {a['agent']} | {a['bus']} | {a['load_type']} | "
                 f"{a['offer_cost_cny_mwh']:.0f} | {a['bid_value_cny_mwh']:.0f} | "
                 f"{a['share_of_day_above_offer']:.2f} | "
                 f"{a['consumption_rate']:.1f}% |")
    L.append("")
    L.append("## The rate is a step function of `offer_cost`")
    L.append("")
    L.append("Holding the prices fixed and moving only the offer cost, the "
             "consumption rate is:")
    L.append("")
    L.append("| offer cost CNY/MWh | consumption rate |")
    L.append("|---|---|")
    for row in sweep:
        L.append(f"| {row['offer_cost_cny_mwh']:.0f} | "
                 f"{row['consumption_rate']:.1f}% |")
    L.append("")
    L.append("## What this means")
    L.append("")
    L.append("The drop is **not** a statement about renewables being "
             "uncompetitive in general. It is the arithmetic of one coefficient "
             "per agent meeting a wholesale market whose mean sits below it: "
             "`lambda_re` and `lambda_curtail` are declared at 50 and 200 and "
             "never applied, so nothing in the objective values renewable "
             "output beyond the import it displaces.")
    L.append("")
    L.append("The consumption rate is a **reported outcome**, not a rule. No "
             "floor is asserted anywhere in the suite, and none should be added "
             "unless a documented market or policy rule -- a renewable "
             "obligation, a curtailment penalty, a certificate price -- "
             "justifies an explicit minimum. Adding one to restore a rate would "
             "be fitting the market to the measurement.")
    L.append("")
    L.append(f"Full period-level table: `{OUT_CSV}`.")
    return "\n".join(L)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--day", type=int, default=None,
                        help="library day; default the configured reference day")
    parser.add_argument("--json", default=None)
    parser.add_argument("--csv", default=OUT_CSV)
    parser.add_argument("--report", default=OUT_MD)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    day_id = (args.day if args.day is not None
              else int(get_default(f"{wp.CONFIG_KEY}.day_id", 0) or 0))
    day = clear(day_id)
    day["day_id"] = day_id
    rows = period_rows(day)
    report = {
        "day": day_id,
        "source": wp.current_source(),
        "enable_multi_objective": bool(
            day["config"].market_design.enable_multi_objective),
        "lambda_re": float(day["config"].market_design.lambda_re),
        "lambda_curtail": float(day["config"].market_design.lambda_curtail),
        "summary": summarise(rows),
        "by_agent": by_agent(rows),
        "threshold_sweep": threshold_sweep(rows),
    }
    # The consumption rate is reported, not asserted. It has no threshold here.
    report["consumption_rate_is_a_diagnostic"] = True

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".",
                    exist_ok=True)
        cols = list(rows[0])
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".",
                    exist_ok=True)
        with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render(report))
            fh.write("\n")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")

    if not args.quiet:
        s = report["summary"]
        print(f"day {day_id}  source {report['source']}  "
              f"enable_multi_objective={report['enable_multi_objective']}")
        print(f"available {s['available_mwh']:.4f} MWh  "
              f"dispatched {s['dispatched_mwh']:.4f}  "
              f"rate {s['consumption_rate']:.2f}%")
        print(f"curtailed by objective {s['curtailed_by_objective_mwh']:.4f} MWh | "
              f"by network {s['curtailed_by_network_mwh']:.4f} | "
              f"unexplained {s['curtailed_unexplained_mwh']:.4f}")
        for a in report["by_agent"]:
            print(f"  {a['agent']:<8} offer {a['offer_cost_cny_mwh']:>6.0f}  "
                  f"bid {a['bid_value_cny_mwh']:>6.0f}  "
                  f"above-offer {a['share_of_day_above_offer']:.2f}  "
                  f"rate {a['consumption_rate']:5.1f}%")
        if args.report:
            print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
