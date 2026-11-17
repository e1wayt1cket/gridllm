# make_demo_figures.py
"""Static figures for a demo snapshot, made from the snapshot files alone.

The clearing environment cannot import matplotlib, so this step runs under a
different interpreter. It reads only the CSVs and JSON that
`demo_snapshot.py` wrote and never touches the model, the inputs or the
clearing. Every figure carries the load day and the price day, because the
case is a cross-date diagnostic and no figure may be read as a same-day
replay.

The six figures are: system load, market price, bus voltage range, line
utilization, served vs shed load, and storage state of charge and power.

Usage:
    python src/make_demo_figures.py --dir outputs/demo_snapshot/<run_id>
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette, light mode. One series per entity, fixed order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9,
    "axes.labelcolor": SECONDARY,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.unicode_minus": False,
})


def panel(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=SECONDARY, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def heading(ax, main: str, sub: str) -> None:
    ax.set_title(main, loc="left", pad=20, color=INK)
    ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=8.5,
            color=SECONDARY, va="bottom")


def footer(fig, case: dict) -> None:
    rule = case["case"]["date_rule"]
    fig.text(
        0.01, 0.012,
        f"Load day {rule['load_date']}  |  Price day {rule['price_date']}  |  "
        "cross-date diagnostic, not a same-day replay",
        fontsize=7.5, color=MUTED, ha="left", va="bottom")


def time_axis(ax, periods: int, dt: float):
    hours = (np.arange(periods) + 0.5) * dt
    ax.set_xlim(0.0, periods * dt)
    ax.set_xticks(np.arange(0.0, periods * dt + 1e-9, 4.0))
    ax.set_xlabel("Hour of day")
    return hours


def fig_load(out: Path, ts: pd.DataFrame, metrics: dict, case: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    panel(ax)
    hours = time_axis(ax, len(ts), metrics["dt_hours"])
    ax.plot(hours, ts["load_asked_mw"], color=SERIES[0], lw=2)
    base = float(metrics["system"]["base_peak_mw"])
    ax.axhline(base, color=AXIS, lw=1.5, linestyle="--")
    ax.text(0.25, base + 0.05, f"{base:.3f} MW subset target",
            fontsize=8, color=SECONDARY)
    peak_i = int(ts["load_asked_mw"].idxmax())
    peak = float(ts["load_asked_mw"].iloc[peak_i])
    ax.scatter([hours[peak_i]], [peak], s=42, color=SERIES[0],
               edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.annotate(f"Coincident peak {peak:.3f} MW at {ts['interval_end'].iloc[peak_i]}",
                xy=(hours[peak_i], peak),
                xytext=(hours[peak_i] - 6.2, peak + 0.22),
                fontsize=8.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 1})
    ax.set_ylim(0, 4.1)
    ax.set_ylabel("Active load (MW)")
    heading(ax, "System active load",
            "Fixed truthful bids; asked load over 96 fifteen-minute periods")
    footer(fig, case)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_price(out: Path, ts: pd.DataFrame, metrics: dict, case: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    panel(ax)
    hours = time_axis(ax, len(ts), metrics["dt_hours"])
    price_meta = case["case"]["price_source"]
    ax.plot(hours, ts["wholesale_cny_mwh"], color=SERIES[0], lw=2,
            label=f"Wholesale input (sheet {price_meta['sheet']}, "
                  f"{price_meta['date']})")
    ax.plot(hours, ts["cleared_system_price_cny_mwh"], color=SERIES[1], lw=2,
            label="Cleared system price (this run)")
    ax.set_ylim(0, float(max(ts["wholesale_cny_mwh"].max(),
                             ts["cleared_system_price_cny_mwh"].max())) * 1.12)
    ax.set_ylabel("Price (CNY/MWh)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    heading(ax, "Market price",
            "The input is a measured spatial mean; the cleared price is this run's output")
    footer(fig, case)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_voltage(out: Path, voltage: pd.DataFrame, metrics: dict,
                case: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    panel(ax)
    bus_columns = [c for c in voltage.columns if c.startswith("bus")]
    buses = np.array([int(c[3:]) for c in bus_columns])
    lows = voltage[bus_columns].min().values
    highs = voltage[bus_columns].max().values
    lo, hi = (float(v) for v in metrics["voltage"]["band_pu"])

    ax.axhspan(lo, hi, color=GRID, alpha=0.65, zorder=0)
    ax.vlines(buses, lows, highs, color=SERIES[0], lw=2, zorder=3)
    ax.scatter(buses, lows, s=16, color=SERIES[0], edgecolor=SURFACE,
               linewidth=1.5, zorder=4)
    ax.scatter(buses, highs, s=16, color=SERIES[0], edgecolor=SURFACE,
               linewidth=1.5, zorder=4)
    ax.axhline(lo, color=AXIS, lw=1.0)
    ax.axhline(hi, color=AXIS, lw=1.0)
    ax.text(0.4, lo + 0.002, f"lower band {lo:.3f} pu", fontsize=8,
            color=SECONDARY)
    ax.text(0.4, hi - 0.008, f"upper band {hi:.3f} pu", fontsize=8,
            color=SECONDARY)
    worst = int(np.argmin(lows))
    ax.annotate(f"min {lows[worst]:.4f} pu at bus {buses[worst]}",
                xy=(buses[worst], lows[worst]),
                xytext=(max(buses[worst] - 13, 0.5), lows[worst] - 0.008),
                fontsize=8.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 1})
    ax.set_xlim(-1, 33)
    ax.set_ylim(0.92, 1.06)
    ax.set_xticks(range(0, 33, 4))
    ax.set_xlabel("Bus")
    ax.set_ylabel("Voltage (pu)")
    heading(ax, "Bus voltage range over the day",
            "Range over 96 periods per bus; shading is the declared band")
    footer(fig, case)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_lines(out: Path, utilization: pd.DataFrame, metrics: dict,
              case: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    panel(ax)
    line_columns = [c for c in utilization.columns if c.startswith("line")]
    maxima = utilization[line_columns].max().sort_values(ascending=False)
    line_ids = [int(c[4:]) for c in maxima.index]
    x = np.arange(len(maxima))
    ax.bar(x, maxima.values, width=0.72, color=SERIES[0])
    ax.axhline(1.0, color=CRITICAL, lw=1.5)
    ax.text(len(maxima) - 0.6, 1.02, "thermal limit 1.0", fontsize=8,
            color=SECONDARY, ha="right")
    ax.annotate(f"max {maxima.values[0]:.4f} on line {line_ids[0]}",
                xy=(0, maxima.values[0]),
                xytext=(3.2, maxima.values[0] + 0.10),
                fontsize=8.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 1})
    ax.set_ylim(0, 1.10)
    # The bars are ranked by utilization, not by line id, so id ticks on the
    # rank axis would read as sequence. The maximum is named in the annotation.
    ax.set_xticks([])
    ax.set_xlabel("Lines, ranked by maximum utilization (highest to lowest)")
    ax.set_ylabel("Max utilization (current / limit)")
    heading(ax, "Line utilization",
            "Maximum over 96 periods per line; every line stays below its limit")
    footer(fig, case)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_supply(out: Path, ts: pd.DataFrame, metrics: dict, case: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    panel(ax)
    hours = time_axis(ax, len(ts), metrics["dt_hours"])
    ax.fill_between(hours, 0.0, ts["served_mw"], color=SERIES[0], alpha=0.10)
    ax.plot(hours, ts["served_mw"], color=SERIES[0], lw=2, label="Served")
    ax.plot(hours, ts["shed_mw"], color=SERIES[1], lw=2, label="Shed")
    ax.set_ylim(0, max(float(ts["served_mw"].max()) * 1.15, 1e-9))
    ax.set_ylabel("Power (MW)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.99, 0.06,
            f"Shed total {metrics['system']['shed_mwh']:.3e} MWh "
            "(solver residual)",
            transform=ax.transAxes, ha="right", fontsize=8, color=SECONDARY)
    heading(ax, "Served load and shed load",
            "The served curve is the load; the shed curve is numerically zero")
    footer(fig, case)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_storage(out: Path, storage: pd.DataFrame, metrics: dict,
                case: dict) -> None:
    units = sorted(storage["unit"].unique(),
                   key=lambda name: int(
                       storage.loc[storage["unit"] == name, "bus"].iloc[0]))
    fig, (ax_soc, ax_power) = plt.subplots(
        2, 1, figsize=(10, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]})
    for ax in (ax_soc, ax_power):
        panel(ax)

    hours = (np.arange(int(metrics["periods"])) + 0.5) * float(metrics["dt_hours"])
    for i, unit in enumerate(units):
        rows = storage[storage["unit"] == unit].sort_values("period")
        bus = int(rows["bus"].iloc[0])
        ax_soc.plot(hours, rows["soc_fraction"].values * 100.0,
                    color=SERIES[i % len(SERIES)], lw=2,
                    label=f"{unit} (bus {bus})")
    ax_soc.set_ylim(0, 100)
    ax_soc.set_ylabel("State of charge (%)")
    ax_soc.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)

    aggregate = storage.groupby("period")[["p_ch_mw", "p_dis_mw"]].sum()
    aggregate = aggregate.sort_index()
    ax_power.plot(hours, aggregate["p_ch_mw"].values, color=SERIES[0], lw=2,
                  label="Charge")
    ax_power.plot(hours, -aggregate["p_dis_mw"].values, color=SERIES[1], lw=2,
                  label="Discharge (negative)")
    ax_power.axhline(0.0, color=AXIS, lw=1.0)
    ax_power.set_ylabel("Fleet power (MW)")
    ax_power.set_xlabel("Hour of day")
    ax_power.set_xlim(0.0, int(metrics["periods"]) * float(metrics["dt_hours"]))
    ax_power.set_xticks(np.arange(0.0, int(metrics["periods"])
                                  * float(metrics["dt_hours"]) + 1e-9, 4.0))
    ax_power.legend(frameon=False, fontsize=8, loc="upper right")
    scale = float(np.max(np.abs(aggregate.values))) if aggregate.size else 0.0
    limit = max(0.02, scale * 1.25)
    ax_power.set_ylim(-limit, limit)

    if metrics["storage"] and all(row["idle"] for row in metrics["storage"]):
        ax_soc.text(0.01, 0.06,
                    "All four units idle: daily charge and discharge stay "
                    "below 1e-4 MWh.", transform=ax_soc.transAxes,
                    fontsize=8, color=SECONDARY)
        ax_power.text(0.01, 0.06,
                      "Fleet power is solver-scale zero; the state of charge "
                      "stays at its initial value.", transform=ax_power.transAxes,
                      fontsize=8, color=SECONDARY)

    fig.suptitle("Storage state of charge and power", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    footer(fig, case)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def verify(metrics: dict, ts: pd.DataFrame, voltage: pd.DataFrame,
           utilization: pd.DataFrame, storage: pd.DataFrame) -> dict:
    """The figure sources must hold the same numbers the report quotes."""
    bus_columns = [c for c in voltage.columns if c.startswith("bus")]
    line_columns = [c for c in utilization.columns if c.startswith("line")]
    dt = float(metrics["dt_hours"])
    checks = []

    def check(name, actual, expected, atol):
        actual = float(actual)
        expected = float(expected)
        ok = abs(actual - expected) <= atol
        checks.append({"check": name, "actual": actual,
                       "expected": expected, "atol": atol, "pass": ok})
        if not ok:
            raise SystemExit(
                f"figure source disagrees with metrics.json on {name}: "
                f"{actual} vs {expected}")

    check("timeseries rows", len(ts), metrics["periods"], 0)
    check("load peak", ts["load_asked_mw"].max(),
          metrics["system"]["coincident_peak_mw"], 1e-9)
    check("served MWh", ts["served_mw"].sum() * dt,
          metrics["system"]["served_mwh"], 1e-6)
    check("shed MWh", ts["shed_mw"].sum() * dt,
          metrics["system"]["shed_mwh"], 1e-9)
    check("voltage min", voltage[bus_columns].min().min(),
          metrics["voltage"]["min_pu"], 1e-9)
    check("voltage max", voltage[bus_columns].max().max(),
          metrics["voltage"]["max_pu"], 1e-9)
    check("line max utilization", utilization[line_columns].max().max(),
          metrics["lines"]["max_utilization"], 1e-9)
    check("storage charged MWh",
          storage.groupby("unit")["p_ch_mw"].sum().sum() * dt,
          metrics["energy"]["storage_charged_mwh"], 1e-9)
    check("storage discharged MWh",
          storage.groupby("unit")["p_dis_mw"].sum().sum() * dt,
          metrics["energy"]["storage_discharged_mwh"], 1e-9)
    return {"all_pass": all(c["pass"] for c in checks), "checks": checks}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", required=True,
                        help="snapshot directory written by demo_snapshot.py")
    args = parser.parse_args(argv)
    root = Path(args.dir)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    case = json.loads((root / "case.json").read_text(encoding="utf-8"))
    ts = pd.read_csv(root / "timeseries.csv")
    voltage = pd.read_csv(root / "voltage_pu.csv")
    utilization = pd.read_csv(root / "line_utilization.csv")
    storage = pd.read_csv(root / "storage.csv")

    verification = verify(metrics, ts, voltage, utilization, storage)
    if not verification["all_pass"]:
        raise SystemExit("figure sources disagree with metrics.json")
    (root / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8")

    figures = []

    fig_load(fig_dir / "fig1_system_load.png", ts, metrics, case)
    figures.append({"file": "figures/fig1_system_load.png",
                    "title": "System active load, coincident peak and the "
                             "3.715 MW baseline",
                    "sources": ["timeseries.csv", "metrics.json"]})
    fig_price(fig_dir / "fig2_market_price.png", ts, metrics, case)
    figures.append({"file": "figures/fig2_market_price.png",
                    "title": "Wholesale price input and this run's cleared "
                             "system price",
                    "sources": ["timeseries.csv", "case.json"]})
    fig_voltage(fig_dir / "fig3_bus_voltage.png", voltage, metrics, case)
    figures.append({"file": "figures/fig3_bus_voltage.png",
                    "title": "Bus voltage range over the day and the band",
                    "sources": ["voltage_pu.csv", "metrics.json"]})
    fig_lines(fig_dir / "fig4_line_utilization.png", utilization, metrics, case)
    figures.append({"file": "figures/fig4_line_utilization.png",
                    "title": "Maximum line utilization and the thermal limit",
                    "sources": ["line_utilization.csv", "metrics.json"]})
    fig_supply(fig_dir / "fig5_served_vs_shed.png", ts, metrics, case)
    figures.append({"file": "figures/fig5_served_vs_shed.png",
                    "title": "Served load and shed load over the day",
                    "sources": ["timeseries.csv", "metrics.json"]})
    fig_storage(fig_dir / "fig6_storage_soc_power.png", storage, metrics, case)
    figures.append({"file": "figures/fig6_storage_soc_power.png",
                    "title": "Storage state of charge and charge/discharge "
                             "power",
                    "sources": ["storage.csv", "metrics.json"]})

    manifest = {
        "run_id": root.name,
        "figures": figures,
        "figure_environment": {
            "executable": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "verified_against": "metrics.json",
        "verification": "verification.json",
        "note": ("All figures are made from snapshot files only. The load day "
                 "and the price day are printed on every figure; the case is a "
                 "cross-date diagnostic."),
    }
    (fig_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[figures] wrote {len(figures)} figures under {fig_dir}")
    print(f"[figures] all figure-source checks passed: "
          f"{verification['all_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
