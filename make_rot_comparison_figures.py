# make_rot_comparison_figures.py
"""Comparison figures for the rot (multi-scenario rotation) training run.

Reads the 4-scenario combined-fleet evaluation CSVs (v4 seeds + rot variants)
and the rot training metrics, then writes four figures:

  1. Profit delta vs truthful baseline, grouped by scenario x policy variant.
  2. Genuine welfare delta (valuation artifact removed), same grouping.
  3. Profit-delta vs genuine-welfare-delta tradeoff scatter.
  4. Rot training reward curve with the in-training best checkpoint marker.

Usage:
  python make_rot_comparison_figures.py [--out-dir results]
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCENARIOS = ["baseline", "high_re", "peak_load", "congestion"]
SCEN_LABEL = {
    "baseline": "baseline",
    "high_re": "high RE",
    "peak_load": "peak load",
    "congestion": "congestion",
}

# Ordered variants for the grouped bars, with display labels and hatch.
VARIANTS = [
    ("multi_matd3_v4_eval.csv", "v4 seed42", None),
    ("multi_matd3_v4_seed7_p2_eval.csv", "v4 seed7", None),
    ("multi_matd3_v4_seed123_p2_eval.csv", "v4 seed123", None),
    ("multi_matd3_rot_seed7_partial_eval.csv", "rot partial (interrupted)", "//"),
    ("multi_matd3_rot_seed7_eval.csv", "rot final (ep200)", "xx"),
    ("multi_matd3_rot_seed7_best_eval.csv", "rot best (ep100)", ".."),
]

COLORS = {
    "baseline": "#4C72B0",
    "high_re": "#55A868",
    "peak_load": "#C44E52",
    "congestion": "#8172B3",
}

# Distinct scatter marker per variant (valid matplotlib marker codes).
VARIANT_MARKERS = {
    "v4 seed42": "o",
    "v4 seed7": "s",
    "v4 seed123": "^",
    "rot partial (interrupted)": "D",
    "rot final (ep200)": "P",
    "rot best (ep100)": "X",
}


def load_eval_rows(results_dir: str) -> dict:
    """Load the ALL-aggregated row per variant/scenario.

    Returns {variant_label: {scenario: {profit_delta, genuine_welfare_delta}}}.
    """
    data = {}
    for fname, label, _ in VARIANTS:
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] missing {fname}")
            continue
        df = pd.read_csv(path)
        rows = {}
        for _, r in df.iterrows():
            sc = str(r["scenario"])
            if sc not in SCENARIOS:
                continue
            rows[sc] = {
                "profit_delta": float(r["profit_delta"]),
                "genuine_welfare_delta": float(r["genuine_welfare_delta"]),
            }
        data[label] = rows
    return data


def fig_profit_delta(data: dict, out_dir: str):
    """Grouped bar: profit delta by scenario x variant (units: 1e3 CNY)."""
    x = np.arange(len(SCENARIOS))
    width = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (label, rows) in enumerate(data.items()):
        vals = [rows[sc]["profit_delta"] / 1e3 for sc in SCENARIOS]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=label, color="#4C72B0", alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCENARIOS])
    ax.set_ylabel("profit delta vs baseline (k CNY)")
    ax.set_title("Fleet profit gain by scenario and policy variant")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    out = os.path.join(out_dir, "rot_profit_delta_by_scenario.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_welfare_delta(data: dict, out_dir: str):
    """Grouped bar: genuine welfare delta (valuation artifact removed)."""
    x = np.arange(len(SCENARIOS))
    width = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (label, rows) in enumerate(data.items()):
        vals = [rows[sc]["genuine_welfare_delta"] / 1e3 for sc in SCENARIOS]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=label, color="#55A868", alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCENARIOS])
    ax.set_ylabel("genuine welfare delta (k CNY)")
    ax.set_title("Genuine welfare gain by scenario and policy variant")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    out = os.path.join(out_dir, "rot_welfare_delta_by_scenario.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_tradeoff(data: dict, out_dir: str):
    """Scatter: profit delta vs genuine welfare delta, colored by scenario."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, rows in data.items():
        m = VARIANT_MARKERS.get(label, "o")
        for sc in SCENARIOS:
            px = rows[sc]["profit_delta"] / 1e3
            py = rows[sc]["genuine_welfare_delta"] / 1e3
            ax.scatter(px, py, marker=m, s=90, color=COLORS[sc],
                       edgecolor="k", linewidths=0.5, zorder=3)
            ax.annotate(sc, (px, py), textcoords="offset points",
                        xytext=(5, 4), fontsize=8, color=COLORS[sc])
    handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=c,
                   label=SCEN_LABEL[s], markeredgecolor="k")
        for s, c in COLORS.items()
    ]
    var_handles = [
        plt.Line2D([], [], marker=m, color="w", markerfacecolor="gray",
                   label=label, markeredgecolor="k", markersize=8)
        for label, m in VARIANT_MARKERS.items()
    ]
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("profit delta vs baseline (k CNY)")
    ax.set_ylabel("genuine welfare delta (k CNY)")
    ax.set_title("Profit vs genuine welfare: per scenario and variant")
    leg1 = ax.legend(handles=handles, loc="upper left", fontsize=8,
                     title="scenario")
    ax.add_artist(leg1)
    ax.legend(handles=var_handles, loc="lower right", fontsize=8,
              title="variant")
    fig.tight_layout()
    out = os.path.join(out_dir, "rot_profit_welfare_tradeoff.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_training_curve(metrics_csv: str, eval_csv: str, out_dir: str):
    """Line: rot training mean reward + welfare, with best-checkpoint marker."""
    m = pd.read_csv(metrics_csv)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(m["episode"], m["mean_reward"] / 1e3, color="#4C72B0",
             label="mean reward (k CNY)")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("mean reward (k CNY)", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.axhline(0, color="k", lw=0.6)

    ax2 = ax1.twinx()
    ax2.plot(m["episode"], m["welfare"] / 1e3, color="#C44E52", alpha=0.7,
             label="episode welfare (k CNY)")
    ax2.set_ylabel("episode welfare (k CNY)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    if os.path.exists(eval_csv):
        e = pd.read_csv(eval_csv)
        best = e[e["is_best"] == True]
        if not best.empty:
            b = best.iloc[-1]
            ax1.axvline(b["episode"], color="k", ls="--", lw=1,
                        alpha=0.7)
            ax1.annotate(f"best ep{b['episode']:.0f}\n"
                         f"mean_reward={b['mean_reward']/1e3:.1f}k",
                         xy=(b["episode"], b["mean_reward"] / 1e3),
                         xytext=(b["episode"] + 8, b["mean_reward"] / 1e3),
                         fontsize=8, arrowprops=dict(arrowstyle="->"))

    ax1.set_title("rot training: reward and welfare over episodes (seed7)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
               fontsize=8)
    fig.tight_layout()
    out = os.path.join(out_dir, "rot_training_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--metrics-csv",
                        default="outputs/rl/run-20260826-215128-9a1f29ce/"
                                "metrics.csv")
    parser.add_argument("--eval-csv",
                        default="outputs/rl/run-20260826-215128-9a1f29ce/"
                                "eval.csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_eval_rows(args.results_dir)
    if not data:
        print("No evaluation CSVs loaded; nothing to plot.")
        return

    fig_profit_delta(data, args.out_dir)
    fig_welfare_delta(data, args.out_dir)
    fig_tradeoff(data, args.out_dir)
    fig_training_curve(args.metrics_csv, args.eval_csv, args.out_dir)


if __name__ == "__main__":
    main()
