# outputs.py
"""Persist simulation results to CSV under outputs/<timestamp>/."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
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


def export_curve_csvs(da_results, rt_results, agents, config, scenario,
                      strategy, output_dir, nash_improvements=None, is_nash=None):
    """Export comprehensive curve-level CSVs for dashboard-like analysis.

    Produces per-scenario CSV files suitable for external plotting:
      kpi_summary.csv   — KPI card data
      lmp_matrix.csv    — full 96×33 LMP
      lmp_curves.csv    — representative-bus and aggregate LMP curves
      trade_curves.csv  — aggregate buy/sell per period
      soc_curves.csv    — per-storage-agent SOC + ch/dis
      re_gen_curves.csv — total PV/wind output per period
      load_curves.csv   — total served/unserved/forecast per period
      da_price.csv      — system average price
      nash_test.csv     — per-agent Nash deviation gains
      schedules.csv     — per-agent, per-period full schedules
    """
    os.makedirs(output_dir, exist_ok=True)
    T = 96
    dt = 0.25
    hours = np.arange(T) * dt
    time_labels = [f"{int(h):02d}:{int((h % 1) * 60):02d}" for h in hours]

    # ---- kpi_summary.csv ----
    load_total = sum(np.sum(a.load_forecast) for a in agents) * dt
    served_total = sum(
        np.sum(da_results["schedules"][a.name]["served"]) * dt for a in agents
    )
    satisfaction = (served_total / load_total * 100) if load_total > 0 else 100.0
    kpi = {
        "scenario": scenario,
        "strategy": strategy,
        "opf_mode": config.opf_mode,
        "da_welfare_cny": da_results.get("welfare", np.nan),
        "rt_welfare_cny": rt_results.get("welfare", np.nan) if rt_results else np.nan,
        "re_rate_pct": da_results.get("re_consumption_rate", np.nan),
        "load_satisfaction_pct": satisfaction,
        "carbon_emissions_tco2": da_results.get("carbon_emissions", np.nan),
        "carbon_intensity_tco2_per_mwh": da_results.get("carbon_intensity", np.nan),
        "curtailment_mwh": da_results.get("total_curtailment", np.nan),
        "is_nash": is_nash if is_nash is not None else "",
        "nash_profitable_count": (
            sum(1 for v in nash_improvements.values() if v.get("profitable"))
            if nash_improvements else 0
        ),
    }
    pd.DataFrame([kpi]).to_csv(
        os.path.join(output_dir, "kpi_summary.csv"), index=False, float_format="%.4f")

    # ---- da_price.csv (system average price) ----
    price = da_results.get("price", np.zeros(T))
    pd.DataFrame({
        "period": range(T), "hour": hours, "time": time_labels,
        "system_price_cny_per_mwh": price,
    }).to_csv(os.path.join(output_dir, "da_price.csv"), index=False, float_format="%.4f")

    # ---- lmp_curves.csv (representative bus + IQR) ----
    lmp = da_results.get("lmp", None)
    if lmp is not None and isinstance(lmp, np.ndarray):
        rows = []
        for t in range(T):
            row = {"period": t, "hour": hours[t], "time": time_labels[t]}
            p25 = np.percentile(lmp[t], 25)
            p75 = np.percentile(lmp[t], 75)
            row["mean_lmp"] = float(lmp[t].mean())
            row["median_lmp"] = float(np.median(lmp[t]))
            row["min_lmp"] = float(lmp[t].min())
            row["max_lmp"] = float(lmp[t].max())
            row["p25_lmp"] = float(p25)
            row["p75_lmp"] = float(p75)
            for bus_idx, label, _ in _REPRESENTATIVE_BUSES:
                if bus_idx < lmp.shape[1]:
                    row[f"bus{bus_idx}_lmp"] = float(lmp[t, bus_idx])
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "lmp_curves.csv"), index=False, float_format="%.4f")

    # ---- lmp_matrix.csv (full 96×33) ----
    if lmp is not None and isinstance(lmp, np.ndarray):
        cols = [f"bus{b}" for b in range(lmp.shape[1])]
        df_lmp = pd.DataFrame(lmp, columns=cols)
        df_lmp.insert(0, "time", time_labels)
        df_lmp.insert(0, "hour", hours)
        df_lmp.insert(0, "period", range(T))
        df_lmp.to_csv(os.path.join(output_dir, "lmp_matrix.csv"), index=False, float_format="%.4f")

    # ---- trade_curves.csv ----
    buy = np.zeros(T); sell = np.zeros(T)
    for a in agents:
        s = da_results["schedules"][a.name]
        buy += s["p_buy"]; sell += s["p_sell"]
    pd.DataFrame({
        "period": range(T), "hour": hours, "time": time_labels,
        "total_buy_mw": buy, "total_sell_mw": sell,
    }).to_csv(os.path.join(output_dir, "trade_curves.csv"), index=False, float_format="%.4f")

    # ---- soc_curves.csv ----
    storage_agents = [a for a in agents if a.storage is not None]
    if storage_agents:
        soc_rows = []
        for a in storage_agents:
            s = da_results["schedules"][a.name]
            for t in range(T):
                soc_rows.append({
                    "agent": a.name, "bus": a.bus, "load_type": a.load_type,
                    "period": t, "hour": hours[t], "time": time_labels[t],
                    "soc_pct": float(s["soc"][t] * 100),
                    "p_ch_mw": float(s["p_ch"][t]),
                    "p_dis_mw": float(s["p_dis"][t]),
                })
        pd.DataFrame(soc_rows).to_csv(
            os.path.join(output_dir, "soc_curves.csv"), index=False, float_format="%.4f")

    # ---- re_gen_curves.csv ----
    pv_total = np.zeros(T); wind_total = np.zeros(T)
    for a in agents:
        s = da_results["schedules"][a.name]
        pv_total += s.get("pv_used", np.zeros(T))
        wind_total += s.get("wind_used", np.zeros(T))
    pd.DataFrame({
        "period": range(T), "hour": hours, "time": time_labels,
        "pv_mw": pv_total, "wind_mw": wind_total,
        "re_total_mw": pv_total + wind_total,
    }).to_csv(os.path.join(output_dir, "re_gen_curves.csv"), index=False, float_format="%.4f")

    # ---- load_curves.csv ----
    served = np.zeros(T); unserved = np.zeros(T); forecast = np.zeros(T)
    for a in agents:
        s = da_results["schedules"][a.name]
        served += s.get("served", np.zeros(T))
        unserved += s.get("unserved", np.zeros(T))
        forecast += a.load_forecast
    pd.DataFrame({
        "period": range(T), "hour": hours, "time": time_labels,
        "served_mw": served, "unserved_mw": unserved,
        "total_load_mw": served + unserved, "forecast_mw": forecast,
    }).to_csv(os.path.join(output_dir, "load_curves.csv"), index=False, float_format="%.4f")

    # ---- nash_test.csv ----
    if nash_improvements:
        nash_rows = []
        for name, imp in nash_improvements.items():
            nash_rows.append({
                "agent": name,
                "is_prosumer": imp.get("is_prosumer", False),
                "base_payoff": imp.get("base_payoff", np.nan),
                "best_payoff": imp.get("best_payoff", np.nan),
                "gain_cny": imp.get("gain", np.nan),
                "rel_gain": imp.get("rel_gain", np.nan),
                "profitable_deviation": imp.get("profitable", False),
            })
        pd.DataFrame(nash_rows).to_csv(
            os.path.join(output_dir, "nash_test.csv"), index=False, float_format="%.4f")

    # ---- schedules.csv (per-agent, per-period) ----
    _save_schedules(da_results, os.path.join(output_dir, "schedules.csv"))

    # ---- PNG chart exports ----
    # Build agents_info for topology chart
    agents_info = [{
        "name": a.name, "bus": a.bus, "is_prosumer": a.is_prosumer,
    } for a in agents]

    _export_lmp_chart(lmp, output_dir, title=f"LMP — {scenario}")
    _export_trade_chart(buy, sell, output_dir, title=f"DA Trade — {scenario}")
    _export_soc_chart(da_results, agents, output_dir, title=f"Storage — {scenario}")
    _export_re_gen_chart(pv_total, wind_total, output_dir, title=f"RE Generation — {scenario}")
    _export_load_chart(served, unserved, forecast, output_dir, title=f"Load — {scenario}")
    _export_topology_chart(lmp, agents_info, output_dir, title=f"IEEE 33-Bus — {scenario}")

    # ---- Comparison summary across all scenarios (appended to root if multi) ----
    return kpi


# bus-index → (label, color) mapping matching dashboard's REPRESENTATIVE_BUSES
_REPRESENTATIVE_BUSES = [
    (0,  "Bus1_Grid",       "#e74c3c"),
    (5,  "Bus6_ResProsumer", "#3b82f6"),
    (12, "Bus13_Commercial", "#10b981"),
    (17, "Bus18_Commercial", "#f59e0b"),
    (21, "Bus22_ResProsumer", "#8b5cf6"),
    (24, "Bus25_IndProsumer", "#ec4899"),
    (32, "Bus33_Industrial", "#6366f1"),
]


# ── matplotlib style defaults for chart exports ──
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 14, 'axes.labelsize': 11,
    'figure.facecolor': 'white', 'axes.facecolor': '#f8fafc',
    'axes.edgecolor': '#e2e8f0', 'axes.grid': True,
    'grid.alpha': 0.4, 'grid.color': '#cbd5e1',
    'legend.framealpha': 0.9, 'legend.edgecolor': '#e2e8f0',
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

_TIME_TICKS = list(range(0, 96, 16))
_TIME_LABELS = [f"{h:02d}:00" for h in range(0, 24, 4)]


def _time_axis(ax, T=96):
    """Configure x-axis with time-of-day labels every 4 hours."""
    hours = np.arange(T) * 0.25
    ax.set_xticks([hours[i] for i in _TIME_TICKS])
    ax.set_xticklabels(_TIME_LABELS)
    ax.set_xlim(0, hours[-1])
    ax.set_xlabel("Time")


def _export_lmp_chart(lmp, output_dir, title="LMP Curves"):
    """LMP chart: IQR shaded envelope + representative bus curves + mean."""
    if lmp is None or (isinstance(lmp, np.ndarray) and np.all(lmp == 0)):
        return
    T, n_buses = lmp.shape
    hours = np.arange(T) * 0.25

    p25 = np.percentile(lmp, 25, axis=1)
    p75 = np.percentile(lmp, 75, axis=1)
    p_min = lmp.min(axis=1)
    p_max = lmp.max(axis=1)
    p_mean = lmp.mean(axis=1)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Full range thin envelope
    ax.fill_between(hours, p_min, p_max, color='#cbd5e1', alpha=0.25, linewidth=0)
    ax.plot(hours, p_min, color='#cbd5e1', linewidth=0.4)
    ax.plot(hours, p_max, color='#cbd5e1', linewidth=0.4)

    # IQR envelope
    ax.fill_between(hours, p25, p75, color='#94a3b8', alpha=0.30, linewidth=0,
                    label='IQR (25%-75%)')

    # Mean line
    ax.plot(hours, p_mean, color='#1e293b', linewidth=2.0, linestyle='--',
            label='System mean')

    # Representative bus curves
    for bus_idx, label, color in _REPRESENTATIVE_BUSES:
        if bus_idx < n_buses:
            ax.plot(hours, lmp[:, bus_idx], color=color, linewidth=1.5, label=label)

    _time_axis(ax, T)
    ax.set_ylabel("LMP (CNY/MWh)")
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "lmp_chart.png"))
    plt.close(fig)


def _export_trade_chart(buy, sell, output_dir, title="DA Market Aggregated Trade"):
    """Aggregate buy/sell power curves."""
    T = len(buy)
    hours = np.arange(T) * 0.25

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hours, buy, color='#ef4444', linewidth=2.0, label='Total Buy (MW)')
    ax.plot(hours, sell, color='#10b981', linewidth=2.0, label='Total Sell (MW)')
    ax.fill_between(hours, 0, buy, color='#ef4444', alpha=0.08)
    ax.fill_between(hours, 0, sell, color='#10b981', alpha=0.08)

    _time_axis(ax, T)
    ax.set_ylabel("Power (MW)")
    ax.set_title(title)
    ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "trade_chart.png"))
    plt.close(fig)


def _export_soc_chart(da_results, agents, output_dir, title="Storage SOC & Charge/Discharge"):
    """Dual-panel chart: SOC% above, charge/discharge bars below."""
    storage_agents = [a for a in agents if a.storage is not None]
    if not storage_agents:
        return

    T = len(da_results["schedules"][storage_agents[0].name]["soc"])
    hours = np.arange(T) * 0.25
    soc_colors = [
        '#e74c3c', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6',
        '#ec4899', '#6366f1', '#14b8a6', '#f97316', '#06b6d4',
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                    gridspec_kw={'height_ratios': [0.52, 0.48]})

    for i, a in enumerate(storage_agents):
        color = soc_colors[i % len(soc_colors)]
        s = da_results["schedules"][a.name]
        ax1.plot(hours, s["soc"] * 100, color=color, linewidth=2.0, label=a.name)
        ax1.set_ylabel("SOC (%)")
        ax1.set_title("Storage SOC")
        ax1.legend(loc='upper left', fontsize=8, ncol=2)

        ax2.bar(hours, s["p_ch"], color=color, alpha=0.65, width=0.25,
                label=f"{a.name} ch")
        ax2.bar(hours, -s["p_dis"], color=color, alpha=0.30, width=0.25,
                hatch='//', label=f"{a.name} dis")

    ax2.set_ylabel("Power (MW)")
    ax2.set_title("Charge / Discharge Power")
    _time_axis(ax2, T)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "soc_chart.png"))
    plt.close(fig)


def _export_re_gen_chart(pv_total, wind_total, output_dir, title="Renewable Generation"):
    """PV + wind + total RE output curves."""
    T = len(pv_total)
    hours = np.arange(T) * 0.25
    has_pv = pv_total.max() > 0.001
    has_wind = wind_total.max() > 0.001

    fig, ax = plt.subplots(figsize=(14, 5))

    if has_pv:
        ax.fill_between(hours, 0, pv_total, color='#f59e0b', alpha=0.25)
        ax.plot(hours, pv_total, color='#f59e0b', linewidth=2.0, label='PV')
    if has_wind:
        ax.fill_between(hours, 0, wind_total, color='#3b82f6', alpha=0.25)
        ax.plot(hours, wind_total, color='#3b82f6', linewidth=2.0, label='Wind')
    ax.plot(hours, pv_total + wind_total, color='#10b981', linewidth=1.8,
            linestyle='--', label='Total RE')

    _time_axis(ax, T)
    ax.set_ylabel("Power (MW)")
    ax.set_title(title)
    ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "re_gen_chart.png"))
    plt.close(fig)


def _export_load_chart(served, unserved, forecast, output_dir, title="Load Profile"):
    """Load chart: forecast (dashed), served (filled), unserved (red overlay)."""
    T = len(served)
    hours = np.arange(T) * 0.25
    total_load = served + unserved
    has_unserved = unserved.max() > 0.01

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(hours, forecast, color='#94a3b8', linewidth=1.5, linestyle='--',
            label='Forecast')
    ax.fill_between(hours, 0, served, color='#3b82f6', alpha=0.15)
    ax.plot(hours, served, color='#3b82f6', linewidth=2.0, label='Served')
    if has_unserved:
        ax.fill_between(hours, served, total_load, color='#ef4444', alpha=0.25)
        ax.plot(hours, total_load, color='#ef4444', linewidth=2.0, label='Total Load')

    _time_axis(ax, T)
    ax.set_ylabel("Power (MW)")
    ax.set_title(title)
    ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "load_chart.png"))
    plt.close(fig)


def _export_topology_chart(lmp_arr, agents_info, output_dir, title="IEEE 33-Bus Topology"):
    """IEEE 33-bus topology with LMP-based node coloring and prosumer stars."""
    # Bus coordinates (orthogonal layout)
    node_coords = {
        0: (0, 0), 1: (2, 0), 2: (4, 0), 3: (6, 0), 4: (8, 0),
        5: (10, 0), 6: (12, 0), 7: (14, 0), 8: (16, 0), 9: (18, 0),
        10: (20, 0), 11: (22, 0), 12: (24, 0), 13: (26, 0), 14: (28, 0),
        15: (30, 0), 16: (32, 0), 17: (34, 0),
        18: (2, 3), 19: (4, 3), 20: (6, 3), 21: (8, 3),
        22: (4, -3), 23: (6, -3), 24: (8, -3),
        25: (10, -3), 26: (12, -3), 27: (14, -3), 28: (16, -3),
        29: (18, -3), 30: (20, -3), 31: (22, -3), 32: (24, -3),
    }
    lines = [
        (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10),
        (10,11), (11,12), (12,13), (13,14), (14,15), (15,16), (16,17),
        (1,18), (18,19), (19,20), (20,21),
        (2,22), (22,23), (23,24),
        (5,25), (25,26), (26,27), (27,28), (28,29), (29,30), (30,31), (31,32),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Draw lines
    for i, j in lines:
        x0, y0 = node_coords[i]
        x1, y1 = node_coords[j]
        ax.plot([x0, x1], [y0, y1], color='#cbd5e1', linewidth=2.0, zorder=1)

    # Compute node colors from LMP
    if lmp_arr is not None and not np.all(lmp_arr == 0):
        lmp_avg = lmp_arr.mean(axis=0)
        vmin = float(np.percentile(lmp_avg, 5))
        vmax = float(np.percentile(lmp_avg, 95))
        if vmax - vmin < 1.0:
            vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.RdYlBu_r
        node_colors = [cmap(norm(lmp_avg[b])) for b in range(33)]
    else:
        node_colors = ['#4f46e5'] * 33

    # Draw nodes
    for b in range(33):
        x, y = node_coords[b]
        ax.scatter(x, y, s=200, c=[node_colors[b]], edgecolors='#334155',
                   linewidths=1.5, zorder=3)
        ax.text(x, y + 0.8, f"B{b+1}", ha='center', fontsize=7,
                fontweight='bold', color='#1e293b')

    # Star markers for prosumer buses
    if agents_info:
        prosumer_buses = list({info.get("bus", -1) for info in agents_info
                               if info.get("is_prosumer", False)})
        if prosumer_buses:
            px = [node_coords[b][0] for b in prosumer_buses if b in node_coords]
            py = [node_coords[b][1] for b in prosumer_buses if b in node_coords]
            ax.scatter(px, py, s=120, marker='*', color='#f59e0b',
                       edgecolors='#d0d7e3', linewidths=1.0, zorder=4,
                       label='Prosumer')

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')
    if agents_info and prosumer_buses:
        ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "topology_chart.png"))
    plt.close(fig)


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
