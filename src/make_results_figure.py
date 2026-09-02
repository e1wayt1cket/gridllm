# make_results_figure.py
"""One comprehensive figure summarizing the RL bidding work.

Panel forms:
  1. Stat tiles (headline numbers)
  2. Line chart : training curves (MATD3 vs TD3, differential reward)
  3. Grouped bar: profit delta vs baseline by approach x scenario
  4. Grouped bar: welfare decomposition (bid-shading artifact vs genuine)
  5. Grouped bar: arbitrage vs market-power per storage agent

Usage:
  python make_results_figure.py [--out results/rl_work_summary.png]
"""

import argparse
import os

# NOTE: import the diagnose_profit / ortools / gurobi chain BEFORE pandas and
# matplotlib. On this Windows environment, importing pandas (pyarrow) first
# alters the DLL search path and breaks ortools' native DLL load.
import numpy as np
import torch
from diagnose_profit import (build_config, run_day, valuation_artifact,
                             agent_profit, DEFAULT_ACTION)
from scenarios import get_scenario
from rl_env import BiddingEnv
from rl_td3 import load_policy

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCEN = ["baseline", "high_re", "peak_load", "congestion", "no_congestion"]
LABELS = {"logitreg": "旧 TD3\n(logit 惩罚)",
          "td3": "TD3\n(差分奖励)",
          "matd3": "MATD3\n(差分奖励)"}
C_SERIES = {"logitreg": "#2a78d6", "td3": "#eb6834", "matd3": "#1baf7a"}
C_ARTI, C_GENU = "#e34948", "#1baf7a"
SURF, INK, SEC, MUT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def load_policies():
    pols = {}
    for tag in ["td3", "matd3"]:
        d = f"policies/multi_{tag}"
        config = build_config()
        agents, _ = get_scenario("baseline", T=96, config=config)
        env0 = BiddingEnv(agents, config,
                          rl_agent_names=[a.name for a in agents if a.storage])
        pols[tag] = {
            a.name: load_policy(os.path.join(d, f"{a.name}.pt"),
                                env0.get_state_dim(), env0.get_action_bounds())
            for a in env0.rl_agents
        }
    return pols


def analyze(tag, scenario, pols):
    """Run baseline + RL day, return artifact, fleet profit delta, and raw data."""
    config = build_config()
    agents, _ = get_scenario(scenario, T=96, config=config)
    rl_names = [a.name for a in agents if a.storage is not None]
    env = BiddingEnv(agents, config, rl_agent_names=rl_names)

    def rl_provider(nm, obs):
        with torch.no_grad():
            return pols[tag][nm](
                torch.as_tensor(obs[None], dtype=torch.float32)).numpy()[0]

    s_base, lmp_base, _ = run_day(env, rl_names,
                                  lambda nm, obs: DEFAULT_ACTION)
    s_rl, lmp_rl, decl_rl = run_day(env, rl_names, rl_provider)
    artifact = valuation_artifact(s_rl, decl_rl, agents, config)
    dprofit = sum(
        agent_profit(s_rl[a.name], lmp_rl[:, a.bus], a, config)
        - agent_profit(s_base[a.name], lmp_base[:, a.bus], a, config)
        for a in agents if a.storage is not None)
    return dict(artifact=artifact, dprofit=dprofit, s_base=s_base,
                lmp_base=lmp_base, s_rl=s_rl, lmp_rl=lmp_rl, agents=agents)


def eval_csv_delta(tag, scenario):
    df = pd.read_csv(f"results/multi_{tag}_eval.csv")
    df = df[df["agent"] == "ALL"]
    row = df[df["scenario"] == scenario]
    if len(row):
        return float(row["welfare_delta"].iloc[0])
    return None


def training_curves():
    out = {}
    for tag, run in [("matd3", "runs/train-multi-20260817-122650"),
                     ("td3", "runs/train-multi-20260817-144717")]:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(
            run, size_guidance={"scalars": 0})
        ea.Reload()
        s = ea.Scalars("Reward/mean")
        out[tag] = (np.array([x.step for x in s]),
                    np.array([x.value for x in s]))
    return out


def panel_style(ax):
    ax.set_facecolor(SURF)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=MUT, labelsize=8, length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rl_work_summary.png")
    args = ap.parse_args()

    pols = load_policies()
    curves = training_curves()

    # ---- profit delta by approach x scenario (from eval CSVs) ----
    prof = {tag: [] for tag in ["logitreg", "td3", "matd3"]}
    for tag in prof:
        df = pd.read_csv(f"results/multi_{tag}_eval.csv")
        df = df[df["agent"] == "ALL"].set_index("scenario")
        prof[tag] = [float(df.loc[s, "profit_delta"]) for s in SCEN]

    # ---- welfare decomposition: artifact + genuine for 4 combos ----
    combos = [("matd3", "baseline"), ("td3", "baseline"),
              ("matd3", "peak_load"), ("td3", "peak_load")]
    decomp = []
    arb_power = None  # per-agent for matd3 baseline
    for tag, sc in combos:
        r = analyze(tag, sc, pols)
        eval_delta = eval_csv_delta(tag, sc)
        genuine = eval_delta - r["artifact"] if eval_delta is not None else None
        decomp.append(dict(tag=tag, sc=sc, artifact=r["artifact"],
                           genuine=genuine, eval_delta=eval_delta))
        if tag == "matd3" and sc == "baseline":
            rows = []
            for a in r["agents"]:
                if a.storage is None:
                    continue
                nm = a.name
                qb = r["s_base"][nm]["p_sell"] - r["s_base"][nm]["p_buy"]
                qr = r["s_rl"][nm]["p_sell"] - r["s_rl"][nm]["p_buy"]
                lmp_b = r["lmp_base"][:, a.bus]
                lmp_r = r["lmp_rl"][:, a.bus]
                arb = float(np.sum((qr - qb) * (lmp_b + lmp_r) / 2.0))
                pwr = float(np.sum((qb + qr) / 2.0 * (lmp_r - lmp_b)))
                rows.append((nm, arb, pwr))
            arb_power = rows

    # ---- figure ----
    fig = plt.figure(figsize=(11.5, 13.5), dpi=150, facecolor=SURF)
    gs = GridSpec(4, 2, height_ratios=[1.0, 2.4, 3.1, 3.1],
                  hspace=0.85, wspace=0.28,
                  left=0.07, right=0.97, top=0.945, bottom=0.05)
    fig.suptitle("RL 储能出价工作总览 — CTDE + 差分奖励", fontsize=15,
                 color=INK, x=0.07, ha="left")

    # ---- stat tiles ----
    tax = fig.add_subplot(gs[0, :])
    tax.axis("off")
    pct = [prof[t][i] / (pd.read_csv(f"results/multi_{t}_eval.csv")
                         .query("agent=='ALL' and scenario=='baseline'")
                         ["baseline_profit"].iloc[0]) * 100
           for t, i in [("td3", 0), ("matd3", 0)]]
    tiles = [
        ("利润增量 (TD3/MATD3)", f"+{pct[0]:.0f}% ~ +{pct[1]:.0f}%",
         "相对真实出价 baseline，5 场景"),
        ("真实调度效应 baseline", "+9k ~ +11k",
         "剔除出价假象后，RL 出清为正"),
        ("福利下降中假象占比", "≈100%",
         "baseline 场景福利下降全为度量假象"),
        ("唯一真实损失", "peak_load −25k",
         "紧张场景储能激进套利所致"),
    ]
    tw, th = 0.23, 0.82
    for i, (lab, val, sub) in enumerate(tiles):
        x0 = 0.012 + i * (tw + 0.012)
        tax.add_patch(plt.Rectangle((x0, 0.08), tw, th, transform=tax.transAxes,
                                    facecolor="#f2f2ee", edgecolor=GRID, lw=0.8))
        tax.text(x0 + 0.012, 0.78, lab, transform=tax.transAxes,
                 fontsize=8.5, color=SEC)
        tax.text(x0 + 0.012, 0.52, val, transform=tax.transAxes,
                 fontsize=14, color=INK, fontweight="bold")
        tax.text(x0 + 0.012, 0.22, sub, transform=tax.transAxes,
                 fontsize=7, color=MUT)

    # ---- panel 1: training curves (line) ----
    ax = fig.add_subplot(gs[1, :])
    panel_style(ax)
    for tag in ["matd3", "td3"]:
        x, y = curves[tag]
        ax.plot(x, y, color=C_SERIES[tag], lw=2)
        ax.annotate(f"{LABELS[tag].split(chr(10))[0]} 终值 {y[-1]:+.0f}",
                    (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=INK, va="center")
    ax.axhline(0, color=AXIS, lw=1)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 210)
    ax.set_ylim(-3000, 9000)
    ax.set_title("训练曲线 — 差分奖励（mean_reward 相对真实出价）", fontsize=11,
                 color=INK, loc="left", pad=8)
    ax.set_xlabel("episode", fontsize=8, color=SEC)
    ax.set_ylabel("mean reward", fontsize=8, color=SEC)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.legend([plt.Line2D([0], [0], color=C_SERIES[t], lw=2)
               for t in ["matd3", "td3"]],
              [LABELS[t].replace("\n", " ") for t in ["matd3", "td3"]],
              frameon=False, fontsize=8, loc="lower right")

    # ---- panel 2: profit delta (grouped bar) ----
    ax = fig.add_subplot(gs[2, 0])
    panel_style(ax)
    x = np.arange(len(SCEN))
    w = 0.26
    offs = (np.arange(3) - 1) * w
    for i, tag in enumerate(["logitreg", "td3", "matd3"]):
        ax.bar(x + offs[i], prof[tag], w, color=C_SERIES[tag])
        for xi, v in zip(x + offs[i], prof[tag]):
            ax.annotate(f"{v/1000:+.1f}", (xi, v), xytext=(0, 2),
                        textcoords="offset points", ha="center",
                        fontsize=6, color=INK, va="bottom")
    ax.axhline(0, color=AXIS, lw=1)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("利润增量 vs 基线（k CNY/日）", fontsize=11, color=INK,
                 loc="left", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n").upper() for s in SCEN],
                       fontsize=7, color=SEC)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=C_SERIES[t])
               for t in ["logitreg", "td3", "matd3"]],
              [LABELS[t] for t in ["logitreg", "td3", "matd3"]],
              frameon=False, fontsize=7.5, loc="upper left")

    # ---- panel 3: welfare decomposition (artifact vs genuine) ----
    ax = fig.add_subplot(gs[2, 1])
    panel_style(ax)
    cl = [f"{d['tag']}\n{d['sc'].replace('_', ' ')}" for d in decomp]
    xc = np.arange(len(decomp))
    wc = 0.34
    art = [d["artifact"] for d in decomp]
    gen = [d["genuine"] if d["genuine"] is not None else 0 for d in decomp]
    ax.bar(xc - wc / 2, art, wc, color=C_ARTI, label="出价假象")
    ax.bar(xc + wc / 2, gen, wc, color=C_GENU, label="真实调度效应")
    for i, d in enumerate(decomp):
        ax.annotate(f"{d['artifact']/1000:+.0f}k", (xc[i] - wc/2, d["artifact"]),
                    xytext=(0, 2 if d["artifact"] >= 0 else -2),
                    textcoords="offset points", ha="center", fontsize=6,
                    color=INK, va="bottom" if d["artifact"] >= 0 else "top")
        ax.annotate(f"{d['genuine']/1000:+.0f}k", (xc[i] + wc/2, d["genuine"]),
                    xytext=(0, 2 if d["genuine"] >= 0 else -2),
                    textcoords="offset points", ha="center", fontsize=6,
                    color=INK, va="bottom" if d["genuine"] >= 0 else "top")
    ax.axhline(0, color=AXIS, lw=1)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("福利下降的构成（k CNY）", fontsize=11, color=INK,
                 loc="left", pad=8)
    ax.set_xticks(xc)
    ax.set_xticklabels(cl, fontsize=7, color=SEC)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    # ---- panel 4: arbitrage vs market power per agent ----
    ax = fig.add_subplot(gs[3, :])
    panel_style(ax)
    names = [r[0] for r in arb_power]
    arb = [r[1] for r in arb_power]
    pwr = [r[2] for r in arb_power]
    xa = np.arange(len(names))
    wa = 0.38
    ax.bar(xa - wa / 2, arb, wa, color="#1baf7a", label="套利（价值创造）")
    ax.bar(xa + wa / 2, pwr, wa, color="#e34948", label="市场力（价格影响）")
    ax.axhline(0, color=AXIS, lw=1)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    tot_a = sum(arb)
    tot_p = sum(pwr)
    ax.set_title(f"利润来源分解（matd3 baseline，逐代理）— 套利 {tot_a:+.0f} "
                 f"({100*tot_a/(tot_a+abs(tot_p)) if tot_a+abs(tot_p) else 0:.0f}%)"
                 f" vs 市场力 {tot_p:+.0f}", fontsize=11, color=INK, loc="left", pad=8)
    ax.set_xticks(xa)
    ax.set_xticklabels(names, fontsize=7.5, color=SEC, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.savefig(args.out, facecolor=SURF, bbox_inches="tight")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
