# make_rot_comparison_figures.py
"""Comparison figures for the unified-scenario rotation MATD3 runs.

Three seeds (7, 42, 123) are trained by rotating over the four unified
physical-facility scenarios (baseline / high_re / peak_load / congestion),
then evaluated in combined-fleet mode on the same four scenarios. This script
plots the three-seed distribution of evaluation outcomes and the seed-7
training curve faceted by scenario.

Writes four figures into --out-dir (default results/):

  1. unified_profit_delta_by_scenario.png
  2. unified_welfare_delta_by_scenario.png
  3. unified_profit_welfare_tradeoff.png
  4. unified_training_by_scenario.png

Usage:
  python make_rot_comparison_figures.py [--results-dir results] \
      [--metrics-csv outputs/rl/run-.../metrics.csv] [--out-dir results]
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
SCEN_SHAPE = {
    "baseline": "o",
    "high_re": "s",
    "peak_load": "^",
    "congestion": "D",
}

# The three unified-rotation seeds; colors follow make_seed_curves.py.
SEEDS = [(7, "runs/train-multi-20260902-151407"),
         (42, "runs/train-multi-20260902-162229"),
         (123, "runs/train-multi-20260902-162230")]
SEED_FILES = ["multi_matd3_unified_seed7_eval.csv",
              "multi_matd3_unified_seed42_eval.csv",
              "multi_matd3_unified_seed123_eval.csv"]
SEED_COLORS = {7: "#1baf7a", 42: "#2a78d6", 123: "#eb6834"}

# Ink palette shared with the seed-curves figure.
SURFACE = "#fcfcfb"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"


def load_family(results_dir: str) -> dict:
    """Read the three seed eval CSVs into {scenario: {profit_deltas, genuine_welfare_deltas}}."""
    fam = {sc: {"profit_deltas": [], "genuine_welfare_deltas": []}
           for sc in SCENARIOS}
    for fname in SEED_FILES:
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] missing {fname}")
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            sc = str(r["scenario"])
            if sc in SCENARIOS:
                fam[sc]["profit_deltas"].append(float(r["profit_delta"]))
                fam[sc]["genuine_welfare_deltas"].append(
                    float(r["genuine_welfare_delta"]))
    return fam


def _k(v):
    """Convert a CNY value to k CNY for plotting."""
    return v / 1e3


def _seed_legend_handles():
    return [plt.Line2D([], [], marker="o", color="w", markerfacecolor=c,
                       markeredgecolor="k", label=f"seed {s}")
            for s, c in SEED_COLORS.items()]


def fig_profit_delta(fam: dict, out_dir: str):
    """Profit delta per scenario: three-seed mean with min-max range and per-seed points."""
    x = np.arange(len(SCENARIOS))
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, sc in enumerate(SCENARIOS):
        vals = np.asarray(fam[sc]["profit_deltas"]) / 1e3
        if vals.size == 0:
            continue
        mean, lo, hi = vals.mean(), vals.min(), vals.max()
        ax.errorbar(x[i], mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=SECONDARY, ms=7, capsize=3, elinewidth=1.4,
                    zorder=3)
        # Per-seed points next to the mean.
        for dx, (s, c) in zip((-0.09, 0.0, 0.09), SEED_COLORS.items()):
            seed_val = fam[sc]["profit_deltas"][list(SEED_COLORS).index(s)]
            # profit_deltas order follows SEED_FILES order = seed 7/42/123.
            seed_val = vals[list(SEED_COLORS).index(s)]
            ax.scatter([x[i] + dx], [seed_val], marker="o", s=45, color=c,
                       edgecolor="k", linewidths=0.4, zorder=4)
        ax.text(x[i], hi + 0.5, f"{mean:+,.0f}", ha="center", va="bottom",
                fontsize=8, color=SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCENARIOS])
    ax.set_ylabel("profit delta vs truthful baseline (k CNY)")
    ax.set_title("Fleet profit gain by scenario (unified, 3 seeds)")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(handles=_seed_legend_handles(), fontsize=8, frameon=False,
              loc="upper left")
    fig.tight_layout()
    out = os.path.join(out_dir, "unified_profit_delta_by_scenario.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_welfare_delta(fam: dict, out_dir: str):
    """Genuine welfare delta per scenario, same grouping as profit delta."""
    x = np.arange(len(SCENARIOS))
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, sc in enumerate(SCENARIOS):
        vals = np.asarray(fam[sc]["genuine_welfare_deltas"]) / 1e3
        if vals.size == 0:
            continue
        mean, lo, hi = vals.mean(), vals.min(), vals.max()
        ax.errorbar(x[i], mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=SECONDARY, ms=7, capsize=3, elinewidth=1.4,
                    zorder=3)
        for dx, (s, c) in zip((-0.09, 0.0, 0.09), SEED_COLORS.items()):
            idx = list(SEED_COLORS).index(s)
            ax.scatter([x[i] + dx], [vals[idx]], marker="o", s=45, color=c,
                       edgecolor="k", linewidths=0.4, zorder=4)
        ax.text(x[i], hi + 0.5, f"{mean:+,.0f}", ha="center", va="bottom",
                fontsize=8, color=SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCENARIOS])
    ax.set_ylabel("genuine welfare delta (k CNY)")
    ax.set_title("Genuine welfare gain by scenario (unified, 3 seeds)")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(handles=_seed_legend_handles(), fontsize=8, frameon=False,
              loc="upper left")
    fig.tight_layout()
    out = os.path.join(out_dir, "unified_welfare_delta_by_scenario.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_tradeoff(fam: dict, out_dir: str):
    """Profit vs welfare tradeoff: color encodes seed, shape encodes scenario."""
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for sc in SCENARIOS:
        shape = SCEN_SHAPE[sc]
        for i, (s, c) in enumerate(SEED_COLORS.items()):
            px = _k(fam[sc]["profit_deltas"][i])
            py = _k(fam[sc]["genuine_welfare_deltas"][i])
            ax.scatter([px], [py], marker=shape, s=110, color=c,
                       edgecolor="k", linewidths=0.7, zorder=4)
        # Scenario label at the seed-7 point.
        ax.annotate(SCEN_LABEL[sc],
                    (_k(fam[sc]["profit_deltas"][0]),
                     _k(fam[sc]["genuine_welfare_deltas"][0])),
                    textcoords="offset points", xytext=(6, 4), fontsize=8,
                    color=SECONDARY)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("profit delta vs truthful baseline (k CNY)")
    ax.set_ylabel("genuine welfare delta (k CNY)")
    ax.set_title("Profit vs welfare tradeoff by scenario (unified, 3 seeds)")
    scen_handles = [plt.Line2D([], [], marker=SCEN_SHAPE[sc], color="w",
                               markerfacecolor="#b0b0b0",
                               markeredgecolor="k", label=SCEN_LABEL[sc])
                    for sc in SCENARIOS]
    seed_handles = _seed_legend_handles()
    leg1 = ax.legend(handles=seed_handles, loc="upper left", fontsize=8,
                     title="seed", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=scen_handles, loc="lower right", fontsize=8,
              title="scenario", frameon=False)
    fig.tight_layout()
    out = os.path.join(out_dir, "unified_profit_welfare_tradeoff.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_training_by_scenario(metrics_csv: str, out_dir: str):
    """Seed-7 training reward per scenario: one single-axis panel per scenario."""
    m = pd.read_csv(metrics_csv)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, sc in zip(axes.ravel(), SCENARIOS):
        ax.set_facecolor(SURFACE)
        sub = m[m["scenario"] == sc].sort_values("episode")
        if sub.empty:
            ax.set_title(f"{SCEN_LABEL[sc]} (no episodes)")
            continue
        ax.scatter(sub["episode"], _k(sub["mean_reward"]), s=8,
                   color=SEED_COLORS[7], alpha=0.5)
        ax.plot(sub["episode"],
                _k(sub["mean_reward"].rolling(7, min_periods=1).mean()),
                color=SEED_COLORS[7], lw=2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(SCEN_LABEL[sc], fontsize=9)
    fig.suptitle("Training reward by scenario (unified rotation, seed 7)",
                 fontsize=11)
    for ax in axes.ravel():
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.6)
    fig.supxlabel("episode", fontsize=9)
    fig.supylabel("mean reward (k CNY)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(out_dir, "unified_training_by_scenario.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--metrics-csv",
                        default="outputs/rl/run-20260902-151405-9a1f29ce/"
                                "metrics.csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fam = load_family(args.results_dir)
    if not any(fam[sc]["profit_deltas"] for sc in SCENARIOS):
        print("No unified seed evaluation CSVs loaded; nothing to plot.")
        return

    fig_profit_delta(fam, args.out_dir)
    fig_welfare_delta(fam, args.out_dir)
    fig_tradeoff(fam, args.out_dir)
    if os.path.exists(args.metrics_csv):
        fig_training_by_scenario(args.metrics_csv, args.out_dir)
    else:
        print(f"[skip] missing metrics csv {args.metrics_csv}")


if __name__ == "__main__":
    main()
