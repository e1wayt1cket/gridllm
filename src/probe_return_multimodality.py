# probe_return_multimodality.py
"""Falsification probe for the diffusion value-distribution (MAD3PG) premise.

The premise being tested: for a fixed decision context (state, action) the
conditional return distribution is multimodal, so a generative (diffusion)
critic adds value over a point estimate. In this market load/PV/wind profiles
are deterministic per scenario; only the day-ahead wholesale curve is re-drawn
per episode from the global RNG (merit-order price + Gaussian noise, no
spikes). Running a deterministic truthful policy (bid_mult=1.0, offer=0)
over many price days therefore samples the conditional return distribution
for near-identical contexts. If per-block returns stay unimodal, modeling the
full distribution is unlikely to pay off here.

Method per block t: N episode samples of committed-block profit -> (1) KDE/
histogram peak count and (2) 1-vs-2 component Gaussian EM BIC difference with
a separation check. Outputs a CSV and a console verdict.
"""

import argparse
import csv
import os
import sys

import numpy as np

from models import MarketConfig
from scenarios import get_scenario
from rl_env import BiddingEnv, N_BLOCKS

MAIN_AGENT = "Bus23I"


def build_config() -> MarketConfig:
    """Eval-style config: market-dispatched storage, no multi-objective."""
    cfg = MarketConfig(opf_mode="socp", verbose=False)
    cfg.market_design.enable_multi_objective = False
    cfg.storage.self_schedule = False
    cfg.storage.use_nodal_price = False
    return cfg


def _hist_n_modes(x: np.ndarray, nbins: int = 48, smooth: int = 2) -> int:
    """Smoothed-histogram peak count (>=1). Bins span the interior quantiles so
    a single wide peak with ragged tails does not read as multimodal."""
    x = np.asarray(x, dtype=float)
    lo, hi = np.percentile(x, [1.0, 99.0])
    if hi - lo < 1e-9:
        return 1
    counts, _ = np.histogram(x, bins=nbins, range=(lo, hi))
    for _ in range(int(smooth)):
        counts = np.convolve(counts, [0.25, 0.5, 0.25], mode="same")
    counts = counts.astype(float)
    floor = counts.mean() * 0.4
    peaks = 0
    n = len(counts)
    for i in range(1, n - 1):
        if counts[i] > counts[i - 1] and counts[i] >= counts[i + 1] \
                and counts[i] > floor:
            peaks += 1
    return max(1, peaks)


def _log_gauss(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    return (-0.5 * np.log(2.0 * np.pi * var) - 0.5 * (x - mu) ** 2 / var)


def _bic_two_vs_one(x: np.ndarray) -> dict:
    """BIC(2-component) - BIC(1-component); negative favors two components.
    Also returns mean separation in pooled-std units."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    if n < 8 or np.var(x) < 1e-12:
        return {"bic_delta": 0.0, "separation": 0.0, "weight_min": 0.0}

    # 1-component
    mu1 = float(np.mean(x))
    var1 = float(np.var(x)) + 1e-9
    ll1 = float(_log_gauss(x, mu1, var1).sum())

    # 2-component EM with floor weights to avoid degenerate collapse
    q = np.quantile(x, [0.25, 0.5, 0.75])
    mu = np.array([q[0], q[2]], dtype=float)
    var = np.array([var1, var1], dtype=float)
    w = np.array([0.5, 0.5], dtype=float)
    for _ in range(300):
        l1 = _log_gauss(x, mu[0], var[0]) + np.log(w[0])
        l2 = _log_gauss(x, mu[1], var[1]) + np.log(w[1])
        m = np.maximum(l1, l2)
        a1 = np.exp(l1 - m)
        a2 = np.exp(l2 - m)
        r1 = a1 / (a1 + a2)                     # P(component 0 | x)
        r1 = np.clip(r1, 1e-3, 1 - 1e-3)
        w[0] = r1.mean()
        w[1] = 1.0 - w[0]
        s0 = r1.sum()
        s1 = n - s0
        mu[0] = (r1 * x).sum() / s0
        mu[1] = ((1 - r1) * x).sum() / s1
        var[0] = (r1 * (x - mu[0]) ** 2).sum() / s0 + 1e-9
        var[1] = ((1 - r1) * (x - mu[1]) ** 2).sum() / s1 + 1e-9

    logmix = np.logaddexp(_log_gauss(x, mu[0], var[0]) + np.log(w[0]),
                          _log_gauss(x, mu[1], var[1]) + np.log(w[1]))
    ll2 = float(logmix.sum())
    # params: 1-comp=2, 2-comp=5
    bic1 = 2 * np.log(n) - 2 * ll1
    bic2 = 5 * np.log(n) - 2 * ll2
    pooled = np.sqrt(0.5 * (var[0] + var[1]))
    separation = float(abs(mu[0] - mu[1]) / pooled) if pooled > 0 else 0.0
    return {"bic_delta": float(bic2 - bic1), "separation": separation,
            "weight_min": float(min(w[0], w[1]))}


def run_scenario(scenario: str, episodes: int, seed0: int,
                 out_path: str) -> None:
    cfg = build_config()
    agents, _ = get_scenario(scenario, T=96, config=cfg)
    storage_names = [a.name for a in agents if a.storage is not None]
    if MAIN_AGENT not in {a.name for a in agents}:
        raise SystemExit(f"Scenario {scenario} has no {MAIN_AGENT} agent")
    env = BiddingEnv(agents, cfg)
    truthful = {a.name: np.array([1.0, 0.0], dtype=np.float32)
                for a in env.all_agents}

    fleet = {t: [] for t in range(N_BLOCKS)}
    main = {t: [] for t in range(N_BLOCKS)}
    fallbacks = 0
    for e in range(episodes):
        np.random.seed(seed0 + e)
        env.reset()
        for t in range(N_BLOCKS):
            _, rewards, done, info = env.step(truthful)
            if info.get("fell_back"):
                fallbacks += 1
            fleet[t].append(sum(rewards.get(nm, 0.0) for nm in storage_names))
            main[t].append(rewards.get(MAIN_AGENT, 0.0))
            if done:
                break
        if (e + 1) % max(1, episodes // 5) == 0 or e == 0:
            print(f"  ep {e + 1}/{episodes}", flush=True)

    print(f"\nScenario {scenario}: N={episodes} price days, "
          f"{len(storage_names)} storage agents, fell_back={fallbacks}")
    print(f"{'blk':>3} | {'n':>3} | {'fleet_mean':>11} {'fleet_std':>10} "
          f"{'peaks':>5} {'BIC_delta':>9} {'sep':>5} | main_peaks")
    header = ["scenario", "block", "n", "fleet_mean", "fleet_std",
              "fleet_peaks", "fleet_bic_delta", "fleet_sep",
              "fleet_wmin", "fleet_multimodal", "main_peaks",
              "main_bic_delta", "main_sep", "main_wmin",
              "main_multimodal"]
    rows = []
    n_multimodal = 0
    for t in range(N_BLOCKS):
        fs = np.asarray(fleet[t])
        ms = np.asarray(main[t])
        fb = _bic_two_vs_one(fs)
        mb = _bic_two_vs_one(ms)
        fp = _hist_n_modes(fs)
        mp = _hist_n_modes(ms)
        # Robust rule: strong BIC preference for two components, well-separated
        # means, and both components carrying non-trivial weight (guards against
        # a tiny-noise component or a skewed-unimodal false positive).
        fm = (fb["bic_delta"] < -10 and fb["separation"] >= 2.0
              and fb["weight_min"] >= 0.15)
        mm = (mb["bic_delta"] < -10 and mb["separation"] >= 2.0
              and mb["weight_min"] >= 0.15)
        n_multimodal += int(fm or mm)
        print(f"{t:>3} | {len(fs):>3} | {fs.mean():>11.1f} "
              f"{fs.std():>10.1f} {fp:>5} {fb['bic_delta']:>9.1f} "
              f"{fb['separation']:>5.2f} {fb['weight_min']:>5.2f} | {mp}")
        rows.append({"scenario": scenario, "block": t, "n": len(fs),
                     "fleet_mean": float(fs.mean()), "fleet_std": float(fs.std()),
                     "fleet_peaks": fp, "fleet_bic_delta": fb["bic_delta"],
                     "fleet_sep": fb["separation"],
                     "fleet_wmin": fb["weight_min"],
                     "fleet_multimodal": fm,
                     "main_peaks": mp, "main_bic_delta": mb["bic_delta"],
                     "main_sep": mb["separation"],
                     "main_wmin": mb["weight_min"],
                     "main_multimodal": mm})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    verdict = ("MULTIMODAL present" if n_multimodal > 0
               else "UNIMODAL (no multimodal block)")
    print(f"\nVerdict: {verdict}  ({n_multimodal}/{N_BLOCKS} blocks flagged) "
          f"-> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=str, default="baseline",
                        help="Comma-separated scenario names")
    parser.add_argument("--episodes", type=int, default=60,
                        help="Number of price-day episodes per scenario")
    parser.add_argument("--seed", type=int, default=1000,
                        help="Seed offset for price-day draws")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for sc in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        out = os.path.join(args.out_dir,
                           f"probe_multimodality_{sc}.csv")
        print(f"=== {sc} ===")
        run_scenario(sc, args.episodes, args.seed, out)


if __name__ == "__main__":
    main()
