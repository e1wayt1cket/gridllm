# diagnose_profit.py
"""Decompose RL profit gains into arbitrage vs market-power extraction.

Runs a scenario twice - baseline (truthful bids) and the trained RL fleet -
records each storage agent's full-day dispatch and nodal LMP from the rolling
market, then splits the per-agent market-payment delta into two factors:

  * arbitrage term   : dispatch re-timing valued at the average nodal price
                       (``sum(Delta q * LMP_mid)``)
  * market-power term: nodal price movement times average net position
                       (``sum(q_mid * Delta LMP)``)

A positive market-power term means the agent's bidding moved its nodal LMP in
its own favour (net seller pushed prices up, or net buyer pulled them down),
i.e. surplus transfer rather than value creation.

Usage:
  python diagnose_profit.py --policies policies/multi_matd3 --scenario baseline
  python diagnose_profit.py --policies policies/multi_matd3 --scenario baseline \
      --save-plot results/extraction_diagnosis.png
"""

import argparse
import os
import numpy as np
import torch

from models import MarketConfig
from scenarios import get_scenario
from rl_env import BiddingEnv, N_BLOCKS, BLOCK_SIZE
from rl_td3 import load_policy

N_BUS = 33
DEFAULT_ACTION = np.array([1.0, 0.0], dtype=np.float32)


def build_config() -> MarketConfig:
    config = MarketConfig(opf_mode="socp", verbose=False)
    config.market_design.enable_multi_objective = False
    config.storage.self_schedule = False
    config.storage.use_nodal_price = False
    return config


def run_day(env: BiddingEnv, rl_names, action_provider) -> tuple:
    """Run one full day, stitching committed-period schedules and nodal LMP.

    Returns (sched, lmp, declared): per-agent full-day schedule dicts (ALL
    agents), the (T, 33) nodal LMP matrix, and the DECLARED bid_mult /
    offer_adder actually applied per period (storage agents only). Only the
    committed periods of each rolling window are recorded, matching the
    reward the environment hands to training.
    """
    T = env.T
    sched = {
        a.name: {k: np.zeros(T) for k in
                 ["p_buy", "p_sell", "p_ch", "p_dis", "served", "unserved",
                  "pv_used", "wind_used"]}
        for a in env.all_agents
    }
    declared = {nm: {"bid_mult": np.zeros(T), "offer_adder": np.zeros(T)}
                for nm in rl_names}
    lmp = np.zeros((T, N_BUS))
    obs = env.reset()
    for block in range(N_BLOCKS):
        t_start = block * BLOCK_SIZE
        acts = {nm: action_provider(nm, obs[nm]) for nm in rl_names}
        next_obs, _, done, _ = env.step(acts)
        res = env._last_result
        n_commit = min(BLOCK_SIZE,
                       min(t_start + env.roll_horizon, T) - t_start)
        if res is not None:
            for nm in sched:
                ws = res["schedules"].get(nm)
                if ws is None:
                    continue
                for key in sched[nm]:
                    sched[nm][key][t_start:t_start + n_commit] = ws[key][:n_commit]
            lmp[t_start:t_start + n_commit, :] = res["lmp"][:n_commit, :]
        # Record the declared (clipped) bid/offer the optimizer actually used
        for nm in rl_names:
            ca = env.current_actions.get(nm)
            if ca is not None:
                declared[nm]["bid_mult"][t_start:t_start + n_commit] = \
                    ca["bid_mult"][t_start:t_start + n_commit]
                declared[nm]["offer_adder"][t_start:t_start + n_commit] = \
                    ca["offer_adder"][t_start:t_start + n_commit]
        obs = next_obs
        if done:
            break
    return sched, lmp, declared


def agent_profit(s: dict, lmp_node: np.ndarray, agent, config) -> float:
    """Raw profit (CNY) over a full day, matching the env reward formula."""
    mkt = np.sum(s["p_sell"] * lmp_node - s["p_buy"] * lmp_node)
    cons = float(np.sum(agent.bid_value * s["served"]))
    gen = float(np.sum(agent.offer_cost * (s["pv_used"] + s["wind_used"])))
    pen = float(config.market_design.penalty_unserved * np.sum(s["unserved"]))
    cyc = float(config.storage.cycle_cost * np.sum(s["p_ch"] + s["p_dis"]))
    return mkt + cons - gen - pen - cyc


def valuation_artifact(sched: dict, declared: dict, agents, config) -> float:
    """Welfare (ObjVal) drop caused purely by RL bid shading.

    The OPF objective values load ``served`` and storage charge/discharge at
    the agent's DECLARED bid/offer. A shaded bid_mult<1 therefore understates
    both the storage agent's load value and its charging value in the metric,
    even when the real dispatch is unchanged. Revaluing the same RL dispatch
    at truthful bid_value/offer_cost isolates that distortion.

    Mirrors dispatch_socp.py: ``bid*served`` is undiscounted; the storage
    ``bid*ch - offer*dis`` term carries the discount factor.
    """
    gamma = config.storage.discount_factor
    A = 0.0
    for a in agents:
        if a.storage is None:
            continue
        s = sched[a.name]
        bm = declared[a.name]["bid_mult"]
        oa = declared[a.name]["offer_adder"]
        # load-value (served) term: undiscounted
        A += float(np.sum(a.bid_value * (bm - 1.0) * s["served"]))
        # storage charge/discharge valuation: discounted
        disc = gamma ** np.arange(len(s["p_ch"]))
        A += float(np.sum(
            disc * (a.bid_value * (bm - 1.0) * s["p_ch"] - oa * s["p_dis"])))
    return A


def decompose(rows: list, s_base, lmp_base, s_rl, lmp_rl,
              agents, config) -> list:
    """Append (arbitrage, market_power) to each row for the fleet."""
    for row in rows:
        nm, dp = row[0], row[1]
        a = next(x for x in agents if x.name == nm)
        b, r = s_base[nm], s_rl[nm]
        lmp_b = lmp_base[:, a.bus]
        lmp_r = lmp_rl[:, a.bus]
        q_b = b["p_sell"] - b["p_buy"]
        q_r = r["p_sell"] - r["p_buy"]
        arb = float(np.sum((q_r - q_b) * (lmp_b + lmp_r) / 2.0))
        pow_t = float(np.sum((q_b + q_r) / 2.0 * (lmp_r - lmp_b)))
        other = dp - (arb + pow_t)
        # net position over the RL day: >0 net seller, <0 net buyer
        net = float(np.sum(q_r))
        row.extend([arb, pow_t, other, net])
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Decompose RL profit gains into arbitrage vs market power")
    parser.add_argument("--policies", type=str, required=True,
                        help="Directory of trained policy .pt files")
    parser.add_argument("--scenario", type=str, default="baseline",
                        help="Scenario name")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Optional PNG path for the arbitrage vs market "
                             "power bar chart")
    args = parser.parse_args()

    config = build_config()
    agents, _ = get_scenario(args.scenario, T=96, config=config)
    rl_names = [a.name for a in agents if a.storage is not None]
    env = BiddingEnv(agents, config, rl_agent_names=rl_names)

    action_bounds = env.get_action_bounds()
    obs_dim = env.get_state_dim()
    policies = {}
    for nm in rl_names:
        path = os.path.join(args.policies, f"{nm}.pt")
        policies[nm] = load_policy(path, obs_dim, action_bounds)

    def rl_provider(nm, obs):
        with torch.no_grad():
            return policies[nm](
                torch.as_tensor(obs[None], dtype=torch.float32)).numpy()[0]

    def base_provider(nm, obs):
        return DEFAULT_ACTION

    s_base, lmp_base, decl_base = run_day(env, rl_names, base_provider)
    s_rl, lmp_rl, decl_rl = run_day(env, rl_names, rl_provider)

    rows = []
    for a in agents:
        if a.storage is None or a.name not in s_base:
            continue
        dp = (agent_profit(s_rl[a.name], lmp_rl[:, a.bus], a, config)
              - agent_profit(s_base[a.name], lmp_base[:, a.bus], a, config))
        rows.append([a.name, dp])
    rows = decompose(rows, s_base, lmp_base, s_rl, lmp_rl, agents, config)

    print(f"\n=== {args.policies} | scenario={args.scenario} ===\n")
    print(f"{'agent':<9s} {'dProfit':>9s} {'arbitrage':>10s} "
          f"{'marketPwr':>10s} {'other':>8s} {'net(MWh)':>9s}  role")
    tot_dp = tot_arb = tot_pow = 0.0
    for nm, dp, arb, pow_t, other, net in rows:
        role = "seller" if net >= 0 else "buyer"
        print(f"{nm:<9s} {dp:>9.0f} {arb:>10.0f} {pow_t:>10.0f} "
              f"{other:>8.0f} {net:>9.1f}  {role}")
        tot_dp += dp
        tot_arb += arb
        tot_pow += pow_t
    if tot_dp != 0:
        print(f"\nfleet: dProfit={tot_dp:+.0f} | arbitrage={tot_arb:+.0f} "
              f"({100 * tot_arb / tot_dp:+.0f}%) | "
              f"market_power={tot_pow:+.0f} ({100 * tot_pow / tot_dp:+.0f}%)")

    # ---- Welfare side: which agents lose the surplus ----
    storage_names = set(rl_names)
    d_all = {}
    for a in agents:
        if a.name not in s_base:
            continue
        d_all[a.name] = (
            agent_profit(s_rl[a.name], lmp_rl[:, a.bus], a, config)
            - agent_profit(s_base[a.name], lmp_base[:, a.bus], a, config))
    d_stor = sum(v for n, v in d_all.items() if n in storage_names)
    d_other = sum(v for n, v in d_all.items() if n not in storage_names)
    d_total = sum(d_all.values())

    def components(s, lm, a):
        return (float(np.sum(s["p_sell"] * lm - s["p_buy"] * lm)),
                float(np.sum(a.bid_value * s["served"])),
                float(np.sum(a.offer_cost * (s["pv_used"] + s["wind_used"]))),
                float(config.market_design.penalty_unserved
                      * np.sum(s["unserved"])),
                float(config.storage.cycle_cost
                      * np.sum(s["p_ch"] + s["p_dis"])))

    print(f"\nwelfare-side (sum over ALL agents = surrogate welfare delta):")
    print(f"  storage   : {d_stor:+.0f}")
    print(f"  non-storage: {d_other:+.0f}")
    print(f"  total     : {d_total:+.0f}   (compare vs eval CSV welfare_delta)")
    print("  top non-storage losers (component deltas: mkt/cons/gen/pen/cyc):")
    losers = sorted(((n, v) for n, v in d_all.items()
                     if n not in storage_names), key=lambda x: x[1])[:5]
    for n, v in losers:
        a = next(x for x in agents if x.name == n)
        cb = components(s_base[n], lmp_base[:, a.bus], a)
        cr = components(s_rl[n], lmp_rl[:, a.bus], a)
        ds = " ".join(f"{d:+.0f}" for d in
                      (cr[0]-cb[0], cr[1]-cb[1], cr[2]-cb[2],
                       cr[3]-cb[3], cr[4]-cb[4]))
        print(f"    {n:<9s} {v:>9.0f}  ({ds})")
    print("  top storage gainers (arbitrage / market_power / other):")
    gainers = sorted(rows, key=lambda r: r[1], reverse=True)[:4]
    for r in gainers:
        print(f"    {r[0]:<9s} {r[1]:>9.0f}  (arb={r[2]:+.0f} "
              f"pwr={r[3]:+.0f} other={r[4]:+.0f})")

    # ---- Valuation-artifact check ----
    # eval welfare_delta = genuine dispatch effect + bid-shading artifact.
    # The artifact is computed exactly (only the storage bid/offer valuation
    # terms in the objective differ between declared and truthful values).
    tag = os.path.basename(args.policies.rstrip("/\\"))
    csv_path = os.path.join("results", f"{tag}_eval.csv")
    eval_delta = None
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        df = df[df["agent"] == "ALL"]
        row = df[df["scenario"] == args.scenario]
        if len(row):
            eval_delta = float(row["welfare_delta"].iloc[0])
    artifact = valuation_artifact(s_rl, decl_rl, agents, config)
    genuine = (eval_delta - artifact) if eval_delta is not None else None
    print(f"\nwelfare decomposition (eval welfare_delta = genuine + artifact):")
    print(f"  eval welfare_delta      : "
          f"{eval_delta:+,.0f}" if eval_delta is not None
          else "  eval welfare_delta      : (no eval CSV found)")
    print(f"  bid-shading artifact     : {artifact:+,.0f}   "
          f"(storage valued at declared bid/offer vs truthful)")
    if genuine is not None:
        print(f"  genuine dispatch effect : {genuine:+,.0f}")

    # Optional bar chart: arbitrage vs market power per agent
    if args.save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r[0] for r in rows]
        arb = [r[2] for r in rows]
        pow_t = [r[3] for r in rows]
        x = np.arange(len(names))
        w = 0.38
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
        ax.axhline(0, color="#c3c2b7", lw=1.2)
        ax.bar(x - w / 2, arb, w, color="#1baf7a", label="arbitrage")
        ax.bar(x + w / 2, pow_t, w, color="#e34948",
               label="market power")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("profit component (CNY)")
        ax.legend(frameon=False)
        ax.set_title(f"{args.scenario}: arbitrage vs market-power "
                     f"decomposition")
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(args.save_plot, facecolor="#fcfcfb", bbox_inches="tight")
        print(f"\nsaved plot: {args.save_plot}")


if __name__ == "__main__":
    main()
