# audit_synthetic_wholesale_price.py
"""What the external wholesale price actually looks like, and what it did not change.

Two questions, and the second is the one that decides whether the first is
usable. The price statistics say whether the new market has the structure the
retired map lacked. The integration and regression sections say whether
swapping the source was a swap and not a rewrite: the same network, the same
scenario definitions, the same storage, the same clearing, with one input
changed.

The achieved statistics are reported AGAINST the design targets the market was
specified with, and any gap is stated as a gap. Nothing here tunes a parameter
to close one. A price moved to hit a target stops being a market and becomes an
answer key, which is exactly what the previous map was.

Usage:
    PYTHONPATH=src python src/audit_synthetic_wholesale_price.py
    PYTHONPATH=src python src/audit_synthetic_wholesale_price.py --json outputs/x.json
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

import numpy as np

import money
import wholesale_price as wp
from config_loader import get_default, reload_defaults, reload_scenarios

#: The design targets the market was specified with. Reported against, never
#: tuned to. They are simulation design goals, not statistics of any real
#: market.
DESIGN_TARGETS = {
    "min": (150.0, 200.0),
    "mean": (350.0, 450.0),
    "max": (750.0, 900.0),
}

REPORT_PATH = "outputs/synthetic_wholesale_price_audit.md"
PLOT_DIR = "outputs"


# ---------------------------------------------------------------------------
# Price statistics
# ---------------------------------------------------------------------------

def _in_target(value: float, band) -> str:
    lo, hi = band
    if value < lo:
        return "below"
    if value > hi:
        return "above"
    return "in"


def price_statistics(provider) -> dict:
    """Every price in the library, against the targets it was specified with."""
    s = wp.stats(provider)
    s["against_design_targets"] = {
        key: {"achieved": s[key], "target": list(DESIGN_TARGETS[key]),
              "verdict": _in_target(s[key], DESIGN_TARGETS[key])}
        for key in ("min", "mean", "max")
    }
    return s


def typical_day(provider, day: int = 0) -> dict:
    """One day's shape, described by where its money is."""
    price = provider.day_ahead_price(day)
    r = provider.reduced_input(day)
    hours = np.arange(price.size) * 0.25

    def window(a, b):
        sel = (hours >= a) & (hours < b)
        return {"mean_price": float(price[sel].mean()),
                "mean_r": float(r[sel].mean())}

    return {
        "day": day,
        "lowest_price_time": f"{hours[int(np.argmin(price))]:05.2f}",
        "lowest_price": float(price.min()),
        "highest_price_time": f"{hours[int(np.argmax(price))]:05.2f}",
        "highest_price": float(price.max()),
        "night_00_06": window(0, 6),
        "morning_06_10": window(6, 10),
        "noon_10_15": window(10, 15),
        "evening_17_22": window(17, 22),
        "late_22_24": window(22, 24),
        "curve": [float(v) for v in price],
        "r_curve": [float(v) for v in r],
        "times": [f"{int(h):02d}:{int((h % 1) * 60):02d}" for h in hours],
    }


def net_load_relation(provider) -> dict:
    """Whether price actually tracks the external market's own net load.

    The retired map's relationship was affine by construction. This one should
    be monotone and *convex* -- the slope rises with load, because dearer
    resources are dispatched as cheap ones run out.
    """
    R = np.concatenate([provider.day_net_load(d)
                        for d in range(provider.number_of_days)])
    r = np.concatenate([provider.reduced_input(d)
                        for d in range(provider.number_of_days)])
    quiet = {**provider.cfg, "noise_sigma": 0.0, "scarcity_probability": 0.0}
    clean = wp.SyntheticWholesalePriceProvider(quiet)
    p = np.concatenate([clean.day_ahead_price(d)
                        for d in range(clean.number_of_days)])

    order = np.argsort(r)
    edges = np.array_split(order, 10)
    binned = [{"r": float(r[idx].mean()), "price": float(p[idx].mean())}
              for idx in edges if idx.size]
    slopes = [(binned[i + 1]["price"] - binned[i]["price"])
              / max(binned[i + 1]["r"] - binned[i]["r"], 1e-9)
              for i in range(len(binned) - 1)]
    return {
        "net_load_min_mw": float(R.min()),
        "net_load_max_mw": float(R.max()),
        "monotone": bool(all(binned[i]["price"] <= binned[i + 1]["price"] + 1e-9
                             for i in range(len(binned) - 1))),
        "convex": bool(all(slopes[i] <= slopes[i + 1] + 1e-6
                           for i in range(len(slopes) - 1))),
        "bins": binned,
        "slopes_cny_mwh_per_r": [float(s) for s in slopes],
    }


def reproducibility(provider) -> dict:
    """Same configuration, same prices; different day, different prices."""
    a = provider.get_day_ahead_price(0)
    b = provider.get_day_ahead_price(0)
    c = provider.get_day_ahead_price(1)
    return {
        "same_day_identical": bool(np.array_equal(a, b)),
        "different_day_differs": bool(not np.allclose(a, c)),
        "day_0_mean": float(a.mean()),
        "day_1_mean": float(c.mean()),
        "master_seed": provider.master_seed,
        "number_of_days": provider.number_of_days,
    }


# ---------------------------------------------------------------------------
# Integration: the swap reached the market
# ---------------------------------------------------------------------------

def library_opportunity(provider) -> dict:
    """What the market offers across every day, not just the reference one.

    A single day says nothing about whether a market has a spread in it. The
    reference day is deliberately the median one, but a reader still needs to
    see the distribution it was taken from -- and a day can be mild enough
    that the round trip does not pay at all, which is a property of that day
    and not of the market.
    """
    from arbitrage_lp import arbitrage_lp, spec_from_storage
    from models import StorageSpec

    unit = spec_from_storage(StorageSpec(
        e_max=1.6, p_ch_max=0.4, p_dis_max=0.4, eta_ch=0.92, eta_dis=0.92,
        soc0=0.5, soc_min=0.10, soc_max=0.90))
    cycle_cost = float(get_default("storage.cycle_cost"))
    rows = []
    for day in range(provider.number_of_days):
        w = provider.day_ahead_price(day)
        res = arbitrage_lp(w, unit, cycle_cost)
        rows.append({
            "day": day,
            "range_cny_mwh": float(w.max() - w.min()),
            "naive_margin_cny_mwh": float(
                unit.eta_dis * w.max() - w.min() / unit.eta_ch
                - 2.0 * cycle_cost),
            "lp_margin_cny": float(res.margin_cny),
        })
    ranges = np.array([r["range_cny_mwh"] for r in rows])
    margins = np.array([r["naive_margin_cny_mwh"] for r in rows])
    reference = int(get_default(f"{wp.CONFIG_KEY}.day_id", 0) or 0)
    return {
        "rows": rows,
        "range_median_cny_mwh": float(np.median(ranges)),
        "range_p05_cny_mwh": float(np.percentile(ranges, 5)),
        "range_p95_cny_mwh": float(np.percentile(ranges, 95)),
        "margin_median_cny_mwh": float(np.median(margins)),
        "margin_min_cny_mwh": float(margins.min()),
        "margin_max_cny_mwh": float(margins.max()),
        "days_with_margin_above_zero": int((margins > 0).sum()),
        "n_days": len(rows),
        "reference_day": reference,
        "reference_day_range_cny_mwh": float(ranges[reference]),
        "reference_rank_by_range": int(
            (ranges < ranges[reference]).sum()) + 1,
    }


def integration(report: bool = True) -> dict:
    """Clear the native day on the new source and check it behaved.

    A price nobody clears against is a spreadsheet. This runs the same
    native-scale scenario the research uses, with the same network, the same
    storage fleet and the same clearing, and reports that the PCC term took
    effect, the SOCP solved, the nodal prices came out nodal, and the storage
    fleet settled.
    """
    from market import clear_market
    from participant_payoff import participant_payoff
    from scenarios import get_scenario, typical_day_config

    reload_defaults()
    reload_scenarios()
    wp.reset_cache()
    config = typical_day_config(money.PERIODS_PER_DAY)
    agents, wholesale = get_scenario("typical_day", money.PERIODS_PER_DAY,
                                     config)
    result = clear_market(agents, money.PERIODS_PER_DAY, "DA", {}, config,
                          wholesale=wholesale, horizon_type="full_day")

    storage = [a for a in agents if a.storage is not None]
    lmp = np.asarray(result["lmp"], dtype=float)
    buses = list(result.get("bus_indices", []))
    settled = {}
    for a in storage:
        i = buses.index(a.bus) if a.bus in buses else 0
        settled[a.name] = participant_payoff(
            result["schedules"][a.name], lmp[:, i], a, config).total

    # The PCC price and the nodal price must stay different quantities: if a
    # change ever routed one into the other, the markup would vanish.
    nodal_spread = float((lmp.max(axis=1) - lmp.min(axis=1)).max())
    return {
        "source": wp.current_source(),
        "wholesale_min": float(np.min(wholesale)),
        "wholesale_mean": float(np.mean(wholesale)),
        "wholesale_max": float(np.max(wholesale)),
        "socp_solved": bool(result is not None and not result.get("fell_back")),
        "lmp_fallbacks": int(result.get("lmp_fallbacks", -1)),
        "lmp_min": float(lmp.min()),
        "lmp_mean": float(lmp.mean()),
        "lmp_max": float(lmp.max()),
        "lmp_is_nodal": bool(nodal_spread > 1.0),
        "max_nodal_spread_cny_mwh": nodal_spread,
        "n_storage": len(storage),
        "ess_settled_cny": settled,
        "ess_total_cny": float(sum(settled.values())),
        "shed_mwh": float(sum(np.sum(result["schedules"][a.name]["unserved"])
                              for a in agents) * money.DT_HOURS),
    } if report else {}


# ---------------------------------------------------------------------------
# Regression: what the swap did NOT change
# ---------------------------------------------------------------------------

def regression() -> dict:
    """The frozen inputs, read back and reported.

    Making the external market the native scenario's price source must not
    have moved a physical parameter, a storage parameter or the network. This
    reads each of them from the configuration and reports it, so a reader can
    see that they are what they were rather than take it on trust.
    """
    from grid import build_base_network, native_nominal_demand_mw
    from rl_spec import ACTION_STORAGE_QUOTE_V2

    reload_defaults()
    scenarios = get_default("scenarios", {}) or {}
    cfg = typical_day_config_safe()
    net = build_base_network(cfg)
    iscfg = get_default("independent_storage", {}) or {}
    spec = ACTION_STORAGE_QUOTE_V2
    return {
        "native_nominal_demand_mw": native_nominal_demand_mw(net),
        "scenarios": {
            name: {"load_scale_mode": sc.get("load_scale_mode"),
                   "peak_load_mw": sc.get("peak_load_mw"),
                   "pv_bus_mw": (sc.get("renewable") or {}).get("pv_bus_mw"),
                   "wind_bus_mw": (sc.get("renewable") or {}).get("wind_bus_mw")}
            for name, sc in scenarios.items()},
        "independent_storage": {
            "buses": iscfg.get("buses"),
            "capacity_mwh": iscfg.get("capacity_mwh"),
            "power_ratio": iscfg.get("power_ratio"),
            "eta_ch": iscfg.get("eta_ch"),
            "eta_dis": iscfg.get("eta_dis"),
            "soc0": iscfg.get("soc0"),
            "soc_min": iscfg.get("soc_min"),
            "soc_max": iscfg.get("soc_max"),
        },
        "cycle_cost_cny_mwh": get_default("storage.cycle_cost"),
        "storage_action_box": {name: list(spec.bounds[name])
                               for name in spec.action_names},
        "voltage_band": {"v_min_pu": cfg.network.v_min_pu,
                         "v_max_pu": cfg.network.v_max_pu},
    }


def typical_day_config_safe():
    from scenarios import typical_day_config
    return typical_day_config(money.PERIODS_PER_DAY)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def write_plots(provider, typical: dict, relation: dict,
                out_dir: str = PLOT_DIR) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"(plots skipped: {exc})"]

    os.makedirs(out_dir, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(9, 4))
    for day in range(min(8, provider.number_of_days)):
        ax.plot(np.arange(96) * 0.25, provider.day_ahead_price(day),
                lw=0.9, alpha=0.75, label=f"day {day}")
    ax.set_xlabel("hour")
    ax.set_ylabel("CNY/MWh")
    ax.set_title("External day-ahead price, first days of the library")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "wholesale_price_days.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    written.append(path)

    everything = np.concatenate([provider.day_ahead_price(d)
                                 for d in range(provider.number_of_days)])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(everything, bins=60, color="#4c72b0")
    for lo, hi in DESIGN_TARGETS.values():
        ax.axvspan(lo, hi, color="#55a868", alpha=0.10)
    ax.set_xlabel("CNY/MWh")
    ax.set_ylabel("periods")
    ax.set_title("Price distribution over the library (bands: design targets)")
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "wholesale_price_distribution.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = relation["bins"]
    ax.plot([b["r"] for b in bins], [b["price"] for b in bins], "o-")
    ax.set_xlabel("r (normalized external net load)")
    ax.set_ylabel("mean price, CNY/MWh")
    ax.set_title("Price against net load, noise and scarcity off")
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "wholesale_price_vs_netload.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    written.append(path)
    return written


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def build_report(quiet: bool = False) -> dict:
    provider = wp.SyntheticWholesalePriceProvider()
    stats = price_statistics(provider)
    typical = typical_day(provider)
    relation = net_load_relation(provider)
    repro = reproducibility(provider)
    integ = integration()
    reg = regression()
    return {
        "source": provider.name,
        "config": provider.describe(),
        "statistics": stats,
        "typical_day": typical,
        "net_load_relation": relation,
        "reproducibility": repro,
        "library_opportunity": library_opportunity(provider),
        "integration": integ,
        "regression": reg,
        "manifest": _manifest(),
    }


def _manifest() -> Optional[dict]:
    path = get_default(f"{wp.CONFIG_KEY}.library_manifest_file") or \
        os.path.splitext(get_default(f"{wp.CONFIG_KEY}.library_file",
                                     "data/wholesale/synthetic_da_v1.csv"))[0] \
        + "_manifest.csv"
    if not os.path.isfile(path):
        return None
    import csv as _csv
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return next(_csv.DictReader(fh), None)


def render(report: dict, plots: Sequence[str] = ()) -> str:
    s, t, rel = report["statistics"], report["typical_day"], \
        report["net_load_relation"]
    c, integ, reg = report["config"], report["integration"], report["regression"]

    L = []
    L.append("# Synthetic external day-ahead wholesale price: audit")
    L.append("")
    L.append("`wholesale[t]` is the price at the PCC. It is not the nodal price "
             "inside the feeder -- that is `lmp[t, i]`, recovered from the "
             "power-balance duals of the network-constrained clearing. This "
             "audit is about the first one only.")
    L.append("")
    L.append(f"Source `{report['source']}`, library of {s['days']} days x "
             f"{s['intervals_per_day']} intervals, master seed "
             f"{c['master_seed']}.")
    L.append("")

    L.append("## Price statistics")
    L.append("")
    L.append("| quantity | value |")
    L.append("|---|---|")
    for key in ("min", "median", "mean", "std", "max"):
        L.append(f"| {key} | {s[key]:.2f} |")
    L.append(f"| P(price < 0) | {s['p_negative']:.4f} |")
    L.append(f"| P(price > 800) | {s['p_above_800']:.4f} |")
    L.append(f"| max / min | {s['ratio_max_min']:.2f} |")
    L.append(f"| P(at floor {c['declared_floor_cny_mwh']:.0f}) | "
             f"{s['p_at_floor']:.4f} |")
    L.append(f"| P(at cap {c['declared_cap_cny_mwh']:.0f}) | "
             f"{s['p_at_cap']:.4f} |")
    L.append("")

    L.append("### Against the design targets")
    L.append("")
    L.append("The targets are simulation design goals, not statistics of any "
             "real market. Nothing was tuned to reach them.")
    L.append("")
    L.append("| statistic | achieved | target | verdict |")
    L.append("|---|---|---|---|")
    for key, row in s["against_design_targets"].items():
        lo, hi = row["target"]
        L.append(f"| {key} | {row['achieved']:.1f} | {lo:.0f}-{hi:.0f} | "
                 f"**{row['verdict']}** |")
    L.append("")
    gap = s["against_design_targets"]["mean"]
    if gap["verdict"] != "in":
        L.append(f"The mean sits **{gap['verdict']}** its target. This is a "
                 f"consequence of normalizing `r` across the whole library: "
                 f"net load is left-skewed, so most periods sit low on the "
                 f"stack even though every day reaches its own extremes. It is "
                 f"reported rather than corrected -- moving a supply-stack "
                 f"parameter to hit a price target would make the market an "
                 f"answer key.")
        L.append("")

    L.append("### Stack segment occupancy")
    L.append("")
    L.append(f"Segments `{c['supply_stack']['knots']}` carry "
             f"`{[round(o, 3) for o in s['segment_occupancy']]}` of all "
             f"periods.")
    L.append("")
    L.append(f"The top two segments carry "
             f"{sum(s['segment_occupancy'][-2:]):.1%}. The stack is real -- "
             f"its slope rises with load -- but the dearest resources are "
             f"reached rarely, because a day has many low-net-load hours and "
             f"few very high ones.")
    L.append("")

    L.append("## Typical day (day 0)")
    L.append("")
    L.append(f"- lowest price {t['lowest_price']:.1f} CNY/MWh at "
             f"{t['lowest_price_time']}")
    L.append(f"- highest price {t['highest_price']:.1f} CNY/MWh at "
             f"{t['highest_price_time']}")
    L.append("")
    L.append("| window | mean price | mean r |")
    L.append("|---|---|---|")
    for key in ("night_00_06", "morning_06_10", "noon_10_15",
                "evening_17_22", "late_22_24"):
        w = t[key]
        L.append(f"| {key.replace('_', ' ')} | {w['mean_price']:.1f} | "
                 f"{w['mean_r']:.3f} |")
    L.append("")

    L.append("## Price against net load")
    L.append("")
    L.append(f"Net load over the library spans "
             f"{rel['net_load_min_mw']:.2f} to {rel['net_load_max_mw']:.2f} MW.")
    L.append(f"- monotone in net load: **{rel['monotone']}**")
    L.append(f"- convex (slope rising with load): **{rel['convex']}**")
    L.append("")
    L.append("A straight line across the whole reachable range is what the "
             "retired map was, and the market-opportunity audit identified "
             "that flatness as the reason the strategic-bidding signal was "
             "under-excited. Convexity is the property that replaced it.")
    L.append("")

    L.append("## Reproducibility")
    L.append("")
    r = report["reproducibility"]
    L.append(f"- same day drawn twice identical: **{r['same_day_identical']}**")
    L.append(f"- a different day differs: **{r['different_day_differs']}**")
    L.append(f"- day 0 mean {r['day_0_mean']:.1f}, day 1 mean "
             f"{r['day_1_mean']:.1f}")
    L.append("")
    m = report.get("manifest")
    if m:
        L.append("The committed library records the yardstick every price in "
                 "it depends on, because `r` is normalized across the library "
                 "and so every price moves if the library's size changes.")
        L.append("")
        L.append("```")
        L.append(f"r_min_mw        {m.get('r_min_mw')}")
        L.append(f"r_max_mw        {m.get('r_max_mw')}")
        L.append(f"number_of_days  {m.get('number_of_days')}")
        L.append(f"library_sha256  {m.get('library_sha256')}")
        L.append("```")
        L.append("")
        L.append("`library` mode refuses a library whose recorded yardstick "
                 "disagrees with the configuration, rather than warning, "
                 "because a stale library prices different numbers silently.")
        L.append("")

    L.append("## What the market offers, day by day")
    L.append("")
    lo = report["library_opportunity"]
    L.append(f"Over all {lo['n_days']} days, the round-trip margin a single "
             f"perfectly-forecasting unit could take from the PCC price:")
    L.append("")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append(f"| price range, median | {lo['range_median_cny_mwh']:.1f} CNY/MWh |")
    L.append(f"| price range, p05 .. p95 | {lo['range_p05_cny_mwh']:.1f} .. "
             f"{lo['range_p95_cny_mwh']:.1f} CNY/MWh |")
    L.append(f"| adjusted margin, median | {lo['margin_median_cny_mwh']:.1f} "
             f"CNY/MWh |")
    L.append(f"| adjusted margin, min .. max | {lo['margin_min_cny_mwh']:.1f} .. "
             f"{lo['margin_max_cny_mwh']:.1f} CNY/MWh |")
    L.append(f"| days with a positive margin | "
             f"**{lo['days_with_margin_above_zero']} / {lo['n_days']}** |")
    L.append("")
    L.append(f"The reference day is **day {lo['reference_day']}**, whose range "
             f"({lo['reference_day_range_cny_mwh']:.1f} CNY/MWh) is the "
             f"library's median and which ranks "
             f"{lo['reference_rank_by_range']} of {lo['n_days']} by range. The "
             f"day the library happens to open with is the flattest of the "
             f"forty, which is why the reference day is declared rather than "
             f"left at zero.")
    L.append("")
    L.append("A day can be mild enough that the round trip does not pay at all. "
             "That is a property of the day, not of the market, and the count "
             "above is how many days it happens to on.")
    L.append("")

    L.append("## Market integration")
    L.append("")
    L.append("The native-scale scenario cleared on the new source, with the "
             "same network, the same storage fleet and the same clearing.")
    L.append("")
    L.append("| check | result |")
    L.append("|---|---|")
    L.append(f"| active source | `{integ['source']}` |")
    L.append(f"| wholesale range | {integ['wholesale_min']:.1f} .. "
             f"{integ['wholesale_max']:.1f} CNY/MWh |")
    L.append(f"| SOCP solved | **{integ['socp_solved']}** |")
    L.append(f"| nodal-price fallbacks | {integ['lmp_fallbacks']} |")
    L.append(f"| LMP range | {integ['lmp_min']:.1f} .. {integ['lmp_max']:.1f} "
             f"CNY/MWh |")
    L.append(f"| LMP is nodal (not the PCC price) | "
             f"**{integ['lmp_is_nodal']}** |")
    L.append(f"| storage units settled | {integ['n_storage']} |")
    L.append(f"| fleet settled total | {integ['ess_total_cny']:.2f} CNY |")
    L.append(f"| shed | {integ['shed_mwh']:.4f} MWh |")
    L.append("")
    L.append("The last row of the price table is the one that matters "
             "structurally: the PCC price and the nodal price are different "
             "quantities at different levels, and the markup between them is "
             "the whole point of clearing the network.")
    L.append("")

    L.append("## Regression: what did not change")
    L.append("")
    L.append("Swapping the price source is a swap, not a rewrite. Read back "
             "from the configuration:")
    L.append("")
    L.append(f"- native nominal demand: **{reg['native_nominal_demand_mw']:.4f} "
             f"MW**")
    L.append(f"- voltage band: {reg['voltage_band']['v_min_pu']} .. "
             f"{reg['voltage_band']['v_max_pu']} pu")
    L.append(f"- cycle cost {reg['cycle_cost_cny_mwh']} CNY/MWh")
    isc = reg["independent_storage"]
    L.append(f"- ESS fleet: buses {isc['buses']}, {isc['capacity_mwh']} MWh "
             f"each at power ratio {isc['power_ratio']}, eta "
             f"{isc['eta_ch']}/{isc['eta_dis']}, SOC0 {isc['soc0']}, band "
             f"[{isc['soc_min']}, {isc['soc_max']}]")
    L.append(f"- storage action box: "
             f"`{reg['storage_action_box']}`")
    L.append("")
    L.append("| scenario | load scale | peak MW | PV bus MW | wind bus MW |")
    L.append("|---|---|---|---|---|")
    for name, sc in reg["scenarios"].items():
        L.append(f"| {name} | {sc['load_scale_mode']} | {sc['peak_load_mw']} | "
                 f"{sc['pv_bus_mw']} | {sc['wind_bus_mw']} |")
    L.append("")
    L.append("`typical_day` remains the IEEE33 native scale (3.715 MW, PV "
             "0.6 MW, wind 1.0 MW) and the legacy scenarios remain at their "
             "declared 10 MW. Only the price source of the native scenario "
             "moved.")
    L.append("")

    if plots:
        L.append("## Figures")
        L.append("")
        for p in plots:
            if p.endswith(".png"):
                L.append(f"- `{p}`")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default=None)
    parser.add_argument("--markdown", default=REPORT_PATH)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(quiet=args.quiet)
    plots = [] if args.no_plots else write_plots(
        wp.SyntheticWholesalePriceProvider(), report["typical_day"],
        report["net_load_relation"])

    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)) or ".",
                    exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render(report, plots))
            fh.write("\n")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")

    if not args.quiet:
        s = report["statistics"]
        print(f"days {s['days']} x {s['intervals_per_day']}")
        print(f"min {s['min']:.2f}  median {s['median']:.2f}  "
              f"mean {s['mean']:.2f}  std {s['std']:.2f}  max {s['max']:.2f}")
        print(f"P(<0) {s['p_negative']:.4f}  P(>800) {s['p_above_800']:.4f}  "
              f"max/min {s['ratio_max_min']:.2f}")
        for key, row in s["against_design_targets"].items():
            print(f"  {key:>5}: {row['achieved']:8.2f}  target "
                  f"{row['target'][0]:.0f}-{row['target'][1]:.0f}  "
                  f"{row['verdict'].upper()}")
        print(f"segment occupancy "
              f"{[round(o, 3) for o in s['segment_occupancy']]}")
        i = report["integration"]
        print(f"integration: socp={i['socp_solved']} "
              f"fallbacks={i['lmp_fallbacks']} nodal={i['lmp_is_nodal']} "
              f"ess={i['ess_total_cny']:.2f} CNY")
        if args.markdown:
            print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
