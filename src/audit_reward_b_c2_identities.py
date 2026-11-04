# audit_reward_b_c2_identities.py
"""Numerical identities for B and C2. Pure functions; no env, no training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import c2_reward as c2
import daily_reward_b as b

OUT = "data/processed/v2_cy_jn_2025/pilot/reward_b_c2_identities.json"
UNITS = b.UNITS


def main(argv=None) -> int:
    rng = np.random.default_rng(20261102)
    T = b.T
    policy = {u: rng.normal(0.0, 200.0, size=T) for u in UNITS}
    baseline = {u: rng.normal(0.0, 200.0, size=T) for u in UNITS}
    # B: only the last block pays.
    cfg = b.RewardBConfig(fixed_scale=1000.0)
    day = b.day_reward_b(policy, baseline, pair_id="p", config=cfg)
    b_terminal_only = all(
        np.all(arr[:T - 1] == 0.0) and arr[T - 1] != 0.0
        for arr in day.reward.values())
    b_discounted = {u: float(np.sum(cfg.gamma ** np.arange(T)
                                    * day.reward[u])) for u in UNITS}
    b_discounted_ok = all(
        abs(b_discounted[u] - cfg.gamma ** (T - 1) * day.scaled_delta[u])
        < 1e-12 for u in UNITS)

    # C2: random bounded SOC paths starting at soc0.
    ccfg = c2.C2Config()
    soc = {}
    for u in UNITS:
        walk = np.cumsum(rng.normal(0.0, 0.02, size=T + 1))
        walk = walk - walk[0] + ccfg.soc0
        soc[u] = np.clip(walk, ccfg.soc_lo, ccfg.soc_hi)
        soc[u][0] = ccfg.soc0
    shaped = c2.c2_reward(day.reward, soc, config=ccfg)
    telescope_err = 0.0
    boundary_err = 0.0
    for u in UNITS:
        weights = ccfg.gamma ** np.arange(T)
        total = float(np.sum(weights * shaped.shaping_terms[u]))
        telescope_err = max(telescope_err, abs(total - shaped.boundary[u]))
        boundary_err = max(
            boundary_err,
            abs((shaped.episode_return[u]
                 - float(np.sum(weights * day.reward[u])))
                - shaped.boundary[u]))
    terminal_zero = max(
        abs(float(c2.potential(soc[u][T], T, ccfg))) for u in UNITS)
    bounded = max(abs(float(c2.potential(s, t, ccfg)))
                  for u in UNITS for t, s in enumerate(soc[u]))
    # Boundary-only: same endpoints, different intermediate SOC.
    soc2 = {u: soc[u].copy() for u in UNITS}
    for u in UNITS:
        mid = np.clip(soc[u][1:T] + rng.normal(0, 0.01, size=T - 1),
                      ccfg.soc_lo, ccfg.soc_hi)
        soc2[u][1:T] = mid
        soc2[u][T] = soc[u][T]
    shaped2 = c2.c2_reward(day.reward, soc2, config=ccfg)
    boundary_only_err = max(
        abs(shaped.episode_return[u] - shaped2.episode_return[u])
        for u in UNITS)
    # Fixed-scaling invariance: double the potential and the base.
    ccfg2 = c2.C2Config(kappa=2.0 * ccfg.kappa)
    day2 = b.day_reward_b({u: 2.0 * policy[u] for u in UNITS},
                          {u: 2.0 * baseline[u] for u in UNITS},
                          pair_id="p", config=cfg)
    shaped_scaled = c2.c2_reward(day2.reward, soc, config=ccfg2)
    scaling_err = max(
        abs(shaped_scaled.episode_return[u] - 2.0 * shaped.episode_return[u])
        for u in UNITS)
    mid_term_rejected = False
    try:
        done = np.zeros(T, dtype=bool)
        done[10] = True
        c2.c2_reward(day.reward, soc, done=done, config=ccfg)
    except ValueError:
        mid_term_rejected = True
    payload = {
        "b_terminal_only": bool(b_terminal_only),
        "b_discounted_is_gamma23": bool(b_discounted_ok),
        "b_fixed_scale": cfg.fixed_scale,
        "c2_telescope_max_abs_error": float(telescope_err),
        "c2_boundary_max_abs_error": float(boundary_err),
        "c2_terminal_potential_max_abs": float(terminal_zero),
        "c2_potential_max_abs": float(bounded),
        "c2_potential_bound": float(ccfg.bound),
        "c2_boundary_only_max_abs_error": float(boundary_only_err),
        "c2_scaling_max_abs_error": float(scaling_err),
        "c2_mid_episode_termination_rejected": bool(mid_term_rejected),
        "gamma": ccfg.gamma,
    }
    payload["all_pass"] = bool(
        payload["b_terminal_only"] and payload["b_discounted_is_gamma23"]
        and telescope_err < 1e-9 and boundary_err < 1e-9
        and terminal_zero == 0.0 and bounded <= ccfg.bound + 1e-9
        and boundary_only_err < 1e-9 and scaling_err < 1e-9
        and mid_term_rejected)
    path = Path.cwd() / OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
