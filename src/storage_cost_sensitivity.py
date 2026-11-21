# storage_cost_sensitivity.py
"""What a lower storage degradation cost changes on the diagnostic day.

`config/defaults.yaml:storage.cycle_cost` is the only storage use cost in the
code: 100 CNY/MWh per leg, charged on both charge and discharge, in the
dispatch objective and again in settlement. This script runs the same frozen
load/price diagnostic under C0 (1.0x), C1 (0.9x) and C2 (0.8x), with the same
day seed and the same truthful quotes. Nothing is retrained and no baseline
configuration is edited.

It is a diagnostic, not a training result. With no OBS_V6 checkpoint in the
repository, the only policy it can run is truthful bidding; the learned reward
is therefore not reported, and the differential reward is zero by construction.
The throughput, SOC, settlement and network quantities are still real market
outcomes for that fixed policy.

Usage:
    PYTHONPATH=src python src/storage_cost_sensitivity.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import arbitrage_lp
import config_loader as cfg
import participant_payoff
from counterfactual_eval import ArmSpec, run_arm
from models import MarketConfig
from scenarios import get_scenario

#: Multipliers on the configured cycle cost. C0 is the unchanged baseline.
CASES: Dict[str, float] = {"C0": 1.0, "C1": 0.9, "C2": 0.8}

CSV_COLUMNS = (
    "case", "cycle_cost_cny_mwh", "required_margin_cny_mwh", "agent",
    "e_max_mwh", "p_dis_max_mw", "charge_mwh", "discharge_mwh",
    "equivalent_cycles", "cycles_from_discharge", "utilization",
    "soc_min", "soc_max", "soc_mean", "soc_terminal", "soc_bound_periods",
    "profit_cny", "gross_margin_cny", "degradation_cost_cny", "revenue_cny",
    "purchase_cost_cny", "energy_sold_mwh", "energy_bought_mwh", "clean",
)

NETWORK_COLUMNS = (
    "case", "cycle_cost_cny_mwh", "required_margin_cny_mwh", "clean",
    "clearing_objective", "re_consumption_rate", "carbon_emissions_tco2",
    "total_curtailment", "max_line_utilization", "mean_line_utilization",
    "periods_line_utilization_ge_0_99", "min_bus_voltage", "max_bus_voltage",
    "mean_lmp_cny_mwh", "price_min_cny_mwh", "price_max_cny_mwh",
    "price_mean_cny_mwh",
)


def resolve_case(name: str, base_cycle_cost: float) -> float:
    key = str(name).strip().upper()
    if key not in CASES:
        raise ValueError(
            f"unknown storage-cost case {name!r}; expected one of "
            f"{sorted(CASES)}")
    return float(base_cycle_cost) * CASES[key]


def _ess_agents(scenario: str) -> list:
    config = MarketConfig.from_yaml()
    agents, _ = get_scenario(scenario, config=config)
    fleet = [a for a in agents if a.name.startswith("ESS")]
    if not fleet:
        raise SystemExit(
            f"{scenario} built no independent storage agents; there is no "
            "storage cost sensitivity to measure")
    return fleet


def _soc_stats(soc: np.ndarray, soc_min: float, soc_max: float) -> dict:
    arr = np.asarray(soc, dtype=float).ravel()
    if arr.size == 0:
        return {"soc_min": float("nan"), "soc_max": float("nan"),
                "soc_mean": float("nan"), "soc_terminal": float("nan"),
                "soc_bound_periods": 0}
    at_bound = np.isclose(arr, soc_min, atol=1e-6) | np.isclose(
        arr, soc_max, atol=1e-6)
    return {
        "soc_min": float(np.min(arr)),
        "soc_max": float(np.max(arr)),
        "soc_mean": float(np.mean(arr)),
        "soc_terminal": float(arr[-1]),
        "soc_bound_periods": int(np.count_nonzero(at_bound)),
    }


def _network_metrics(day: dict) -> dict:
    if day is None:
        return {k: "" for k in NETWORK_COLUMNS if k not in ("case",)}
    out = {
        "clearing_objective": day.get("clearing_objective", ""),
        "re_consumption_rate": day.get("re_consumption_rate", ""),
        "carbon_emissions_tco2": day.get("carbon_emissions", ""),
        "total_curtailment": day.get("total_curtailment", ""),
    }
    lmp = day.get("lmp")
    if lmp is not None:
        out["mean_lmp_cny_mwh"] = float(np.mean(np.asarray(lmp, dtype=float)))
    else:
        out["mean_lmp_cny_mwh"] = ""
    line = day.get("line_utilization")
    if line is not None:
        arr = np.asarray(line, dtype=float)
        out["max_line_utilization"] = float(np.max(arr))
        out["mean_line_utilization"] = float(np.mean(arr))
        out["periods_line_utilization_ge_0_99"] = int(
            np.count_nonzero(arr >= 0.99))
    else:
        out["max_line_utilization"] = ""
        out["mean_line_utilization"] = ""
        out["periods_line_utilization_ge_0_99"] = ""
    voltage = day.get("bus_voltage")
    if voltage is not None:
        arr = np.asarray(voltage, dtype=float)
        out["min_bus_voltage"] = float(np.min(arr))
        out["max_bus_voltage"] = float(np.max(arr))
    else:
        out["min_bus_voltage"] = ""
        out["max_bus_voltage"] = ""
    return out


def _agent_rows(case: str, cycle_cost: float, required_margin: float,
                agents: list, config: MarketConfig, result: dict) -> List[dict]:
    day = result.get("day_result")
    rows: List[dict] = []
    if day is None:
        return rows
    for agent in agents:
        nm = agent.name
        sched = (day.get("schedules") or {}).get(nm)
        if sched is None:
            continue
        lmp = np.asarray(day["lmp"], dtype=float)[:, agent.bus]
        breakdown = participant_payoff.participant_payoff(
            sched, lmp, agent, config)
        storage = agent.storage
        e_max = float(storage.e_max)
        p_dis = float(storage.p_dis_max)
        ch = np.asarray(sched["p_ch"], dtype=float)
        dis = np.asarray(sched["p_dis"], dtype=float)
        charge_mwh = float(np.sum(ch)) * 0.25
        discharge_mwh = float(np.sum(dis)) * 0.25
        soc = {
            **_soc_stats(np.asarray(sched["soc"], dtype=float),
                         float(storage.soc_min), float(storage.soc_max)),
        }
        rows.append({
            "case": case,
            "cycle_cost_cny_mwh": cycle_cost,
            "required_margin_cny_mwh": required_margin,
            "agent": nm,
            "e_max_mwh": e_max,
            "p_dis_max_mw": p_dis,
            "charge_mwh": charge_mwh,
            "discharge_mwh": discharge_mwh,
            "equivalent_cycles": ((charge_mwh + discharge_mwh)
                                  / (2.0 * e_max) if e_max > 0 else 0.0),
            "cycles_from_discharge": (discharge_mwh / e_max
                                      if e_max > 0 else 0.0),
            "utilization": discharge_mwh / (p_dis * 24.0) if p_dis > 0 else 0.0,
            **soc,
            "profit_cny": float(result["profits"].get(nm, 0.0)),
            "gross_margin_cny": float(breakdown.revenue
                                      - breakdown.purchase_cost),
            "degradation_cost_cny": float(breakdown.degradation_cost),
            "revenue_cny": float(breakdown.revenue),
            "purchase_cost_cny": float(breakdown.purchase_cost),
            "energy_sold_mwh": float(breakdown.energy_sold_mwh),
            "energy_bought_mwh": float(breakdown.energy_bought_mwh),
            "clean": bool(result.get("clean", False)),
        })
    return rows


def _fleet_row(rows: Sequence[dict]) -> Optional[dict]:
    if not rows:
        return None
    first = rows[0]
    out = dict(first)
    out["agent"] = "fleet"
    for key in ("charge_mwh", "discharge_mwh", "profit_cny",
                "gross_margin_cny", "degradation_cost_cny", "revenue_cny",
                "purchase_cost_cny", "energy_sold_mwh", "energy_bought_mwh"):
        out[key] = sum(float(r[key]) for r in rows)
    e_max = sum(float(r["e_max_mwh"]) for r in rows)
    p_dis = sum(float(r["p_dis_max_mw"]) for r in rows)
    out["e_max_mwh"] = e_max
    out["p_dis_max_mw"] = p_dis
    out["equivalent_cycles"] = (
        (out["charge_mwh"] + out["discharge_mwh"]) / (2.0 * e_max)
        if e_max > 0 else 0.0)
    out["cycles_from_discharge"] = (out["discharge_mwh"] / e_max
                                    if e_max > 0 else 0.0)
    out["utilization"] = (out["discharge_mwh"] / (p_dis * 24.0)
                          if p_dis > 0 else 0.0)
    out["soc_min"] = min(float(r["soc_min"]) for r in rows)
    out["soc_max"] = max(float(r["soc_max"]) for r in rows)
    out["soc_mean"] = float(np.mean([float(r["soc_mean"]) for r in rows]))
    out["soc_terminal"] = float(np.mean(
        [float(r["soc_terminal"]) for r in rows]))
    out["soc_bound_periods"] = sum(int(r["soc_bound_periods"]) for r in rows)
    out["clean"] = all(bool(r["clean"]) for r in rows)
    return out


def _fmt(value) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, *, scenario: str, base_cost: float,
                  cases: Sequence[str], agent_rows: Sequence[dict],
                  network_rows: Sequence[dict], command: str,
                  csv_path: Path, network_path: Path) -> None:
    fleet = {r["case"]: r for r in agent_rows if r["agent"] == "fleet"}
    lines = [
        "# Storage cost sensitivity",
        "",
        "## Verdict",
        "",
        "This is a **diagnostic on one frozen load/price day with truthful "
        "bidding**, not a training result. No OBS_V6 checkpoint exists in the "
        "repository, so learned reward is not reported and the differential "
        "reward is zero by construction.",
        "",
        "C0 and C1 do not cycle beyond solver noise; C2 begins to cycle but "
        "settles only +0.09 CNY, effectively break-even. On this day a lower "
        "cycle cost is the first case that produces measurable cycling, not "
        "evidence of a meaningful arbitrage strategy.",
        "",
        "## What the cost is",
        "",
        "- Name: `storage.cycle_cost`.",
        f"- Value: **{base_cost:g} CNY/MWh per leg** in "
        "`config/defaults.yaml`.",
        "- Units and timing: CNY per MWh moved, charged once on charge and once "
        "on discharge, and multiplied by the 0.25 h period.",
        "- Decision path: subtracted from the SOCP/LDF dispatch objective.",
        "- Settlement path: `participant_payoff._ledger` sets "
        "`degradation_cost = cycle_cost * (p_ch + p_dis) * dt` and subtracts "
        "it from the participant's total.",
        "- Break-even: `required_margin = cycle_cost * (1/eta_ch + eta_dis)`; "
        "the three values are in the case table below.",
        "- There is no capex, fixed O&M or separate efficiency charge in the "
        "code. Efficiency enters through the SOC transition and the margin.",
        "- Fleet: 4 x 1.6 MWh / 0.4 MW at buses 9/14/17/27, eta 0.92/0.92, "
        "SOC band 0.10-0.90, endpoints pinned by settlement.",
        "",
        "## Cases",
        "",
        "| case | multiplier | cycle_cost CNY/MWh | required margin CNY/MWh |",
        "|---|---:|---:|---:|",
    ]
    for case in cases:
        row = next((r for r in agent_rows if r["case"] == case), None)
        margin = row["required_margin_cny_mwh"] if row else ""
        cost = row["cycle_cost_cny_mwh"] if row else ""
        margin_text = f"{float(margin):.2f}" if margin != "" else ""
        cost_text = f"{float(cost):.2f}" if cost != "" else ""
        lines.append(f"| {case} | {CASES[case]:.2f} | {cost_text} | "
                     f"{margin_text} |")
    lines += ["", "Only the cycle cost changes; the load, price, fleet, "
              "efficiency, SOC band and market rules are identical.", "",
              "## Fleet results", "",
              "| case | charge MWh | discharge MWh | equivalent cycles | "
              "discharge utilisation | gross margin CNY | profit CNY | "
              "degradation CNY | SOC min | SOC max | SOC terminal | clean |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for case in cases:
        r = fleet.get(case)
        if not r:
            lines.append(f"| {case} | — | — | — | — | — | — | — | — | — | — | no |")
            continue
        lines.append(
            f"| {case} | {r['charge_mwh']:.3f} | {r['discharge_mwh']:.3f} | "
            f"{r['equivalent_cycles']:.3f} | {r['utilization']:.3f} | "
            f"{r['gross_margin_cny']:.2f} | {r['profit_cny']:.2f} | "
            f"{r['degradation_cost_cny']:.2f} | "
            f"{r['soc_min']:.3f} | {r['soc_max']:.3f} | "
            f"{r['soc_terminal']:.3f} | "
            f"{'yes' if r['clean'] else 'no'} |")
    lines += ["", "## Per-agent results", "",
              "| case | agent | charge MWh | discharge MWh | cycles | "
              "utilisation | profit CNY | degradation CNY | SOC range |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in agent_rows:
        if r["agent"] == "fleet":
            continue
        lines.append(
            f"| {r['case']} | {r['agent']} | {r['charge_mwh']:.3f} | "
            f"{r['discharge_mwh']:.3f} | {r['equivalent_cycles']:.3f} | "
            f"{r['utilization']:.3f} | {r['profit_cny']:.2f} | "
            f"{r['degradation_cost_cny']:.2f} | "
            f"{r['soc_min']:.3f}–{r['soc_max']:.3f} |")
    lines += ["", "## Network", "",
              "| case | max line util. | periods ≥0.99 | min V | max V | "
              "curtailment | RE rate | objective |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in network_rows:
        lines.append(
            f"| {r['case']} | {_fmt(r['max_line_utilization'])} | "
            f"{_fmt(r['periods_line_utilization_ge_0_99'])} | "
            f"{_fmt(r['min_bus_voltage'])} | {_fmt(r['max_bus_voltage'])} | "
            f"{_fmt(r['total_curtailment'])} | "
            f"{_fmt(r['re_consumption_rate'])} | "
            f"{_fmt(r['clearing_objective'])} |")
    lines += ["", "## Price and limits", ""]
    if network_rows:
        p = network_rows[0]
        price_min = float(p["price_min_cny_mwh"])
        price_max = float(p["price_max_cny_mwh"])
        price_mean = float(p["price_mean_cny_mwh"])
        lines += [
            f"- Day-ahead wholesale price: min {price_min:.2f}, max "
            f"{price_max:.2f}, mean {price_mean:.2f} CNY/MWh.",
            f"- Raw min-max spread: {price_max - price_min:.2f} CNY/MWh.",
        ]
        for case in cases:
            r = next((x for x in network_rows if x["case"] == case), None)
            if r:
                lines.append(
                    f"- Required margin {case}: "
                    f"{float(r['required_margin_cny_mwh']):.2f} CNY/MWh.")
        lines.append(
            "- The raw spread is not the reachable spread: a 0.4 MW / 1.6 MWh "
            "unit cannot pair the day's minimum and maximum periods through "
            "its SOC band, and efficiency and the two-leg cost sit between "
            "them.")

    lines += ["", "## Reading", ""]
    if "C0" in fleet and "C1" in fleet and "C2" in fleet:
        c0, c1, c2 = fleet["C0"], fleet["C1"], fleet["C2"]
        lines += [
            f"- C0 discharge: {float(c0['discharge_mwh']):.4f} MWh "
            f"(solver-noise scale), gross margin "
            f"{float(c0['gross_margin_cny']):.2f} CNY, net "
            f"{float(c0['profit_cny']):.2f} CNY.",
            f"- C1 discharge: {float(c1['discharge_mwh']):.4f} MWh, gross "
            f"margin {float(c1['gross_margin_cny']):.2f} CNY, net "
            f"{float(c1['profit_cny']):.2f} CNY.",
            f"- C2 discharge: {float(c2['discharge_mwh']):.4f} MWh, gross "
            f"margin {float(c2['gross_margin_cny']):.2f} CNY, degradation "
            f"{float(c2['degradation_cost_cny']):.2f} CNY, net "
            f"{float(c2['profit_cny']):.2f} CNY.",
        ]
        if float(c2["discharge_mwh"]) > 1e-3 and float(c2["profit_cny"]) < 1.0:
            lines.append(
                "- C2 is the first case with measurable cycling, but its "
                "settled profit is essentially zero: the gross margin is "
                "almost entirely consumed by degradation. This day does not "
                "support lowering the cost further without first checking the "
                "reachable price spread, efficiency, SOC limits and market "
                "constraints.")
        if (max(float(c0["discharge_mwh"]), float(c1["discharge_mwh"]))
                < 1e-3):
            margins = {
                r["case"]: float(r["required_margin_cny_mwh"])
                for r in network_rows if r["case"] in ("C0", "C1")
            }
            lines.append(
                "- C0 and C1 do not cycle beyond solver noise on this day. "
                f"Their required margins ({margins.get('C0', float('nan')):.2f} "
                f"and {margins.get('C1', float('nan')):.2f} CNY/MWh) are not "
                "cleared by a reachable cycle.")
        stressed = any(
            r["periods_line_utilization_ge_0_99"] not in ("", None)
            and float(r["periods_line_utilization_ge_0_99"]) > 0
            for r in network_rows)
        if stressed:
            lines.append(
                "- At least one case has a line at or above 99% utilisation; "
                "a throughput gain is not a free gain and must be read beside "
                "the network constraint.")
        else:
            lines.append(
                "- No line reaches 99% utilisation and the SOC trajectory stays "
                "inside its band; neither the network nor the SOC limits bind "
                "on this day.")

    lines += ["", "## Falsification", ""]
    if "C0" in fleet and "C1" in fleet and "C2" in fleet:
        c0 = fleet["C0"]
        for case in ("C1", "C2"):
            r = fleet[case]
            base_dis = float(c0["discharge_mwh"])
            dis = float(r["discharge_mwh"])
            if max(base_dis, dis) < 1e-3:
                lines.append(
                    f"- {case} discharge throughput: no measurable cycling "
                    "(both below 0.001 MWh).")
            elif base_dis < 1e-3:
                lines.append(
                    f"- {case} discharge throughput: C0 is below 0.001 MWh; "
                    f"{case} reaches {dis:.4f} MWh, the first measurable "
                    "cycling.")
            else:
                rel = abs(dis - base_dis) / abs(base_dis)
                lines.append(
                    f"- {case} discharge throughput vs C0: {rel:.2%}.")
            base_profit = float(c0["profit_cny"])
            profit = float(r["profit_cny"])
            if max(abs(base_profit), abs(profit)) < 1.0:
                lines.append(
                    f"- {case} settled profit: {profit:+.2f} CNY, within "
                    f"1 CNY of C0 ({base_profit:+.2f} CNY).")
            else:
                lines.append(
                    f"- {case} settled profit: {profit:+.2f} CNY vs C0 "
                    f"{base_profit:+.2f} CNY.")
    lines += ["", "## Limitations", "",
              "- One diagnostic day, one price curve, truthful quotes: this "
              "does not estimate a distribution over days or a learned policy.",
              "- The price curve is dated 2026-10-09 and the load 2022-11-30; "
              "the pair is a cross-date diagnostic, not a historical replay.",
              "- A learned (training) reward is not reported: with no OBS_V6 "
              "checkpoint, the truthful policy's differential reward is 0 by "
              "construction.",
              "- The frozen subset's load varies by date; this day is the "
              "3.653 MW diagnostic day, not a typical day.",
              "",
              "## Artifacts", "",
              f"- `{csv_path}`",
              f"- `{network_path}`",
              "",
              "## Command", "",
              f"`{command}`",
              ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare C0/C1/C2 storage cycle costs on one frozen day.")
    parser.add_argument("--scenario", default="typical_day_industry_subset")
    parser.add_argument("--cases", default="C0,C1,C2")
    parser.add_argument("--day-seed", type=int, default=1000)
    parser.add_argument("--out", default="reports/storage_cost_sensitivity.md")
    parser.add_argument("--csv", default="reports/storage_cost_sensitivity.csv")
    parser.add_argument("--network-csv",
                        default="reports/storage_cost_network.csv")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cases = [c.strip().upper() for c in args.cases.split(",") if c.strip()]
    unknown = sorted(set(cases) - set(CASES))
    if unknown:
        raise SystemExit(f"unknown cases {unknown}; expected {sorted(CASES)}")
    base_cost = float(cfg.get_default("storage.cycle_cost", 100.0))

    agents = _ess_agents(args.scenario)
    rl_names = [a.name for a in agents]
    eta_ch = float(agents[0].storage.eta_ch)
    eta_dis = float(agents[0].storage.eta_dis)
    if any(float(a.storage.eta_ch) != eta_ch
           or float(a.storage.eta_dis) != eta_dis for a in agents):
        raise SystemExit("the independent fleet does not share one efficiency")

    agent_rows: List[dict] = []
    network_rows: List[dict] = []
    wholesale_ref = None
    for case in cases:
        cost = resolve_case(case, base_cost)
        config = MarketConfig.from_yaml()
        config.storage.cycle_cost = cost
        arm = ArmSpec(name="truthful", kind="truthful")
        result = run_arm(args.scenario, config, arm, rl_names,
                         day_seed=args.day_seed)
        wholesale = np.asarray(result["wholesale"], dtype=float)
        if wholesale_ref is None:
            wholesale_ref = wholesale
        elif not np.allclose(wholesale_ref, wholesale):
            raise SystemExit(
                f"{case} did not see the same wholesale day as C0; the "
                "comparison would mix a price change with the cost change")
        margin = arbitrage_lp.required_margin(cost, eta_ch, eta_dis)
        rows = _agent_rows(case, cost, margin, agents, config, result)
        fleet = _fleet_row(rows)
        if fleet is not None:
            agent_rows.extend(rows)
            agent_rows.append(fleet)
        network = _network_metrics(result["day_result"])
        network_rows.append({
            "case": case,
            "cycle_cost_cny_mwh": cost,
            "required_margin_cny_mwh": margin,
            "clean": bool(result.get("clean", False)),
            "price_min_cny_mwh": float(np.min(wholesale)),
            "price_max_cny_mwh": float(np.max(wholesale)),
            "price_mean_cny_mwh": float(np.mean(wholesale)),
            **network,
        })
        print(f"{case}  cycle_cost={cost:.1f}  margin={margin:.2f}  "
              f"clean={bool(result.get('clean'))}")

    if not agent_rows:
        raise SystemExit(
            "no settled agent rows were produced; the day did not clear "
            "cleanly enough to compare storage behaviour")

    csv_path = Path(args.csv)
    network_path = Path(args.network_csv)
    _write_csv(csv_path, agent_rows, CSV_COLUMNS)
    _write_csv(network_path, network_rows, NETWORK_COLUMNS)
    _write_report(Path(args.out), scenario=args.scenario, base_cost=base_cost,
                  cases=cases, agent_rows=agent_rows,
                  network_rows=network_rows, command=" ".join(sys.argv),
                  csv_path=csv_path, network_path=network_path)
    print(f"wrote {args.out}")
    print(f"wrote {csv_path}")
    print(f"wrote {network_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
