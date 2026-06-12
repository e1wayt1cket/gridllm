# outputs.py
"""Persist simulation results to CSV under outputs/<timestamp>/."""
import os
import pandas as pd
import numpy as np
from datetime import datetime


def save_run_results(da_results, rt_results, config, scenario, strategy,
                     output_dir=None, nash_improvements=None, is_nash=None):
    """Write DA/RT results and optional Nash test data to CSV files.

    Parameters
    ----------
    da_results, rt_results : dict
        Output from clear_market().
    config : MarketConfig
    scenario, strategy : str
    output_dir : str or None
        Target directory. Defaults to outputs/<YYYY-MM-DD_HH-MM-SS>/.
    nash_improvements : dict or None
        Per-agent Nash test results from test_nash_equilibrium().
    is_nash : bool or None
    """
    if output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join("outputs", ts)
    os.makedirs(output_dir, exist_ok=True)

    # -- summary.csv --
    summary = {
        "scenario": scenario,
        "strategy": strategy,
        "opf_mode": config.opf_mode,
        "multi_obj_method": "constraint" if config.use_constraint_multi_obj else "weighted-sum",
        "da_welfare": da_results.get("welfare", np.nan),
        "da_re_rate_pct": da_results.get("re_consumption_rate", np.nan),
        "da_carbon_tco2": da_results.get("carbon_emissions", np.nan),
        "da_carbon_intensity": da_results.get("carbon_intensity", np.nan),
        "da_curtailment_mwh": da_results.get("total_curtailment", np.nan),
        "rt_welfare": rt_results.get("welfare", np.nan) if rt_results else np.nan,
        "rt_re_rate_pct": rt_results.get("re_consumption_rate", np.nan) if rt_results else np.nan,
        "is_nash": is_nash if is_nash is not None else "",
        "nash_profitable_count": (
            sum(1 for v in nash_improvements.values() if v.get("profitable"))
            if nash_improvements else ""
        ),
    }
    if config.use_constraint_multi_obj:
        summary["carbon_cap_tco2"] = config.carbon_cap_tco2
        summary["re_min_rate_pct"] = config.re_min_rate
        sp = da_results.get("shadow_prices", {})
        if sp:
            for k, v in sp.items():
                summary[f"shadow_price_{k}"] = float(v)

    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "summary.csv"), index=False, float_format="%.4f")

    # -- schedules.csv --
    _save_schedules(da_results, os.path.join(output_dir, "schedules_da.csv"))

    # -- lmp.csv --
    if "lmp" in da_results:
        lmp = da_results["lmp"]
        if isinstance(lmp, np.ndarray):
            cols = [f"bus{b}" for b in range(lmp.shape[1])]
            pd.DataFrame(lmp, columns=cols).to_csv(
                os.path.join(output_dir, "lmp.csv"), index_label="period",
                float_format="%.4f")

    # -- nash_test.csv --
    if nash_improvements:
        nash_rows = []
        for name, imp in nash_improvements.items():
            nash_rows.append({
                "agent": name,
                "is_prosumer": imp.get("is_prosumer", False),
                "base_payoff": imp.get("base_payoff", np.nan),
                "best_payoff": imp.get("best_payoff", np.nan),
                "gain": imp.get("gain", np.nan),
                "rel_gain": imp.get("rel_gain", np.nan),
                "profitable": imp.get("profitable", False),
            })
        pd.DataFrame(nash_rows).to_csv(
            os.path.join(output_dir, "nash_test.csv"), index=False, float_format="%.4f")

    print(f"Results saved to: {output_dir}")


def _save_schedules(results, path):
    """Write per-agent schedules (served, pv_used, wind_used, p_ch, p_dis, soc)."""
    if results is None:
        return
    schedules = results.get("schedules", {})
    if not schedules:
        return
    rows = []
    T = len(next(iter(schedules.values())).get("served", []))
    for period in range(T):
        for name, sched in schedules.items():
            if name == "GRID":
                continue
            rows.append({
                "period": period,
                "agent": name,
                "served": float(sched.get("served", np.zeros(1))[period]),
                "unserved": float(sched.get("unserved", np.zeros(1))[period]),
                "pv_used": float(sched.get("pv_used", np.zeros(1))[period]),
                "wind_used": float(sched.get("wind_used", np.zeros(1))[period]),
                "p_buy": float(sched.get("p_buy", np.zeros(1))[period]),
                "p_sell": float(sched.get("p_sell", np.zeros(1))[period]),
                "p_ch": float(sched.get("p_ch", np.zeros(1))[period]),
                "p_dis": float(sched.get("p_dis", np.zeros(1))[period]),
                "soc": float(sched.get("soc", np.zeros(1))[period]),
            })
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f")
