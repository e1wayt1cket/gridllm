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
import numpy as np
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "rotation · seed 7": "runs/train-multi-20260902-151407",
    "rotation · seed 42": "runs/train-multi-20260902-162229",
    "rotation · seed 123": "runs/train-multi-20260902-162230",
}
COLORS = {"rotation · seed 7": "#1baf7a",
          "rotation · seed 42": "#2a78d6",
          "rotation · seed 123": "#eb6834"}
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
        vals = np.asarray(vals, dtype=float)
        # Converged value: mean of the last 50 episodes, far less sensitive to
        # per-episode environment noise than the last single value.
        conv = float(vals[-50:].mean())
        finals[label] = conv
        ax.plot(steps, vals, color=COLORS[label], lw=2, label=label)
        # Rolling-mean (window 10) overlay in the same hue, thinner and
        # lighter, so the convergence trend is visible through the noise.
        if len(vals) >= 10:
            roll = np.convolve(vals, np.ones(10) / 10, mode="valid")
            ax.plot(steps[9:], roll, color=COLORS[label], lw=1.2, alpha=0.6)
        # Direct end label: converged value (last-50 mean), in secondary ink
        # (not the series color), offset past the last point.
        ax.annotate(f"{conv:+,.0f}",
                    xy=(steps[-1], conv),
                    xytext=(6, 0), textcoords="offset points",
                    color=SECONDARY, fontsize=9, va="center")

    ax.axhline(0, color=BASELINE, lw=1.2)
    ax.set_xlabel("episode", color=MUTED, fontsize=10)
    ax.set_ylabel("mean reward (differential)", color=MUTED, fontsize=10)
    ax.set_title("MATD3 training: mean reward (rotation over 4 unified scenarios)",
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
    fig.text(0.06, 0.02, f"converged mean reward (last-50 eps) — {cap}",
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
