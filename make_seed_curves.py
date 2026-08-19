# make_seed_curves.py
"""Overlay the MATD3 training curves (Reward/mean) for the three seeds.

Reads the per-seed TensorBoard event logs under runs/ and renders one line per
seed into results/multi_seed_training_curves.png. Chart follows the shared
palette: categorical blue/orange/aqua for seeds 42/123/7, thin 2px lines,
recessive grid, legend plus direct end labels.

Usage:
  python make_seed_curves.py
"""

import os
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "rotation · seed 42": "runs/train-multi-20260817-122650",
    "rotation · seed 123": "runs/train-multi-20260818-151447",
    "rotation · seed 7": "runs/train-multi-20260818-210651",
    "fixed baseline · seed 42": "runs/train-multi-20260819-111944",
}
COLORS = {"rotation · seed 42": "#2a78d6",
          "rotation · seed 123": "#eb6834",
          "rotation · seed 7": "#1baf7a",
          "fixed baseline · seed 42": "#eda100"}
SURFACE = "#fcfcfb"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def load_curve(path: str):
    ea = EventAccumulator(path)
    ea.Reload()
    if "Reward/mean" not in ea.Tags().get("scalars", []):
        raise RuntimeError(f"no Reward/mean in {path}")
    steps = [s.step for s in ea.Scalars("Reward/mean")]
    vals = [s.value for s in ea.Scalars("Reward/mean")]
    return steps, vals


def main():
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    finals = {}
    for label, path in RUNS.items():
        steps, vals = load_curve(path)
        finals[label] = vals[-1]
        ax.plot(steps, vals, color=COLORS[label], lw=2, label=label)
        # Direct end label: final converged value, in secondary ink (not the
        # series color), offset past the last point.
        ax.annotate(f"{vals[-1]:+,.0f}",
                    xy=(steps[-1], vals[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=SECONDARY, fontsize=9, va="center")

    ax.axhline(0, color=BASELINE, lw=1.2)
    ax.set_xlabel("episode", color=MUTED, fontsize=10)
    ax.set_ylabel("mean reward (differential)", color=MUTED, fontsize=10)
    ax.set_title("MATD3 training: mean reward (rotation vs fixed baseline)",
                 color=PRIMARY, fontsize=13, pad=12)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 210)

    legend = ax.legend(frameon=False, fontsize=10, loc="lower right",
                       handlelength=2.0)
    for txt in legend.get_texts():
        txt.set_color(PRIMARY)

    # Convergence table caption (secondary encoding / relief for aqua).
    cap = "  ".join(f"{k}: {v:+,.0f}" for k, v in finals.items())
    fig.text(0.06, 0.02, f"final mean reward — {cap}",
             color=SECONDARY, fontsize=9)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join("results", "multi_seed_training_curves.png")
    os.makedirs("results", exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved: {out}")
    for k, v in finals.items():
        print(f"  {k}: final={v:+,.0f}")


if __name__ == "__main__":
    main()
