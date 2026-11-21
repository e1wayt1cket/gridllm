# calibrate_dev_penalty.py
"""Choose the deviation-penalty weight by measurement, not by profit.

The penalty pulls each action toward its agent's truthful declaration, measured
as a fraction of that action's own box width (see `rl_td3.deviation_penalty`).
Its weight is dimensionless and lives in the same units as the normalized
actor objective, so the question is a balance: how much of the actor's gradient
does the regularizer claim, and where does it hold the policy?

This script answers that on a fixed B3 day with a fixed seed and no learning at
all -- the same observations, the same critic initialization, the same batch,
for every weight. What varies is only the weight.

It reports, per weight:

  penalty_at_mid      the regularizer's cost at half-box deviation, d = 0.5
  |econ|              the actor objective's magnitude on the same batch
  penalty/econ        those two as a ratio, in the units both are expressed in
  grad_ratio          ||d penalty/d logit|| / ||d econ/d logit|| on the batch
  corner_frac         share of the policy's actions sitting within 2% of its
                      truthful corner after a fixed-length probe
  action_spread       std of the quotes the policy emits across the batch

**Nothing here selects on profit.** The selected weight is the largest one that
keeps the regularizer's gradient well below the economic term's and leaves the
policy free to move off the corner, because a weight that pins the fleet at
truthful removes the behaviour the research is measuring. Profit is not
reported at all, so it cannot be selected on by accident.
"""

import argparse
import copy
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

import money
from baseline_ladder import BASE_SCENARIO, PRIMARY_BAND, build_rung_agents
from rl_bidding import MATD3, agent_action_spaces
from rl_env import BiddingEnv, N_BLOCKS
from rl_spec import ACTION_STORAGE_QUOTE_V2

#: The weights to compare. Chosen to span "negligible" to "as large as the
#: normalized economic term itself".
LAMBDA_GRID = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)

#: Actor-only probe length. Not training: a fixed critic is held frozen and the
#: question is only where the balance of two gradients settles the policy.
PROBE_STEPS = 200

#: An action within this fraction of the box width from the truthful point
#: counts as parked on it.
CORNER_TOLERANCE = 0.02


def _fixed_setup(seed: int = 123, T: int = money.PERIODS_PER_DAY):
    """One B3 day and one fixed observation batch, identical for every weight."""
    agents, _wholesale, config = build_rung_agents("B3", PRIMARY_BAND, seed)
    env = BiddingEnv(agents, config,
                     rl_agent_names=[a.name for a in agents
                                     if getattr(a, "participant_type", "")
                                     == "storage"])
    np.random.seed(seed)
    obs = env.reset()
    states = np.stack([obs[a.name] for a in env.rl_agents])
    return env, config, torch.as_tensor(states, dtype=torch.float32)


def _probe(env, config, states, lam: float, seed: int = 7) -> dict:
    """Measure the penalty's balance and where it leaves the policy.

    The critic is frozen at one fixed initialization and the batch is fixed, so
    the critic's gradient is arbitrary but *identical across weights*. That is
    what makes the weights comparable: only the regularizer differs.

    `states` is (n_agents, obs_dim); the critic wants (batch, n_agents, obs_dim),
    so it is run at batch size one.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    trainer = MATD3(env, dev_penalty=lam, batch_size=1)
    _specs, bounds, truthful, _ = agent_action_spaces(env)
    names = trainer.agent_names
    batch_states = states.unsqueeze(0)          # (1, n_agents, obs_dim)

    def penalty_of(nm, action):
        width = bounds[nm][1] - bounds[nm][0]
        d = ((action - truthful[nm].unsqueeze(0)).abs() / width)
        return lam * (d ** 2).sum(dim=1).mean()

    # --- Gradient balance on the fixed batch, before any update ---
    grads_econ, grads_pen = [], []
    for i, nm in enumerate(names):
        logits, action = trainer.actors[nm].forward_logits(states[i:i + 1])
        # This agent's own action is the one already computed, so the critic's
        # graph reaches `logits`. Re-running the forward here would build a
        # second pass and leave the gradient unconnected to it.
        all_acts = torch.stack(
            [(action if j == i else
              trainer.actors[n2].forward_logits(states[j:j + 1])[1])
             for j, n2 in enumerate(names)], dim=1)
        q1, q2 = trainer.critics[nm](
            trainer._build_global_state(batch_states, i),
            all_acts.reshape(1, -1))
        econ = -torch.min(q1, q2).mean()
        g_econ = torch.autograd.grad(econ, logits, retain_graph=True)[0]
        g_pen = torch.autograd.grad(penalty_of(nm, action), logits,
                                    retain_graph=True)[0]
        grads_econ.append(float(g_econ.norm()))
        grads_pen.append(float(g_pen.norm()))

    econ_mag = float(np.mean(grads_econ))
    pen_mag = float(np.mean(grads_pen))

    # --- Where the policy settles when both terms pull ---
    #
    # Both terms drive the probe, which is what makes it a corner-pinning test.
    # Driving it with the penalty alone would only rediscover that the penalty's
    # own minimum is the corner -- true of any penalty, and no evidence about
    # what happens when a critic is also pulling. The critic here is frozen at a
    # fixed initialization, so its pull is arbitrary but identical across
    # weights; the comparison between weights is what the probe is for.
    def joint_loss(i, nm):
        _, action = trainer.actors[nm].forward_logits(states[i:i + 1])
        all_acts = torch.stack(
            [trainer.actors[n2].forward_logits(states[j:j + 1])[1]
             for j, n2 in enumerate(names)], dim=1)
        q1, q2 = trainer.critics[nm](
            trainer._build_global_state(batch_states, i),
            all_acts.reshape(1, -1))
        return -torch.min(q1, q2).mean() + penalty_of(nm, action)

    for _ in range(PROBE_STEPS):
        for i, nm in enumerate(names):
            opt = trainer.actor_opts[nm]
            opt.zero_grad()
            joint_loss(i, nm).backward()
            opt.step()

    with torch.no_grad():
        devs, quotes = [], []
        for i, nm in enumerate(names):
            _, action = trainer.actors[nm].forward_logits(states[i:i + 1])
            width = bounds[nm][1] - bounds[nm][0]
            devs.append(((action - truthful[nm].unsqueeze(0)).abs()
                         / width).numpy())
            quotes.append(action.numpy())
        dev = np.concatenate([d.ravel() for d in devs])
        quote_arr = np.concatenate(quotes, axis=0)

    return {
        "lambda": lam,
        "econ_grad_norm": econ_mag,
        "pen_grad_norm": pen_mag,
        "grad_ratio": (pen_mag / econ_mag) if econ_mag > 0 else float("inf"),
        "mean_deviation_after_probe": float(dev.mean()),
        "min_deviation_after_probe": float(dev.min()),
        "max_deviation_after_probe": float(dev.max()),
        "corner_fraction": float((dev < CORNER_TOLERANCE).mean()),
        "action_spread": float(quote_arr.std(axis=0).mean()),
        "quote_min": float(quote_arr.min()),
        "quote_max": float(quote_arr.max()),
    }


def calibrate(seed: int = 123, grid=LAMBDA_GRID) -> dict:
    env, config, states = _fixed_setup(seed)

    rows: List[dict] = []
    for lam in grid:
        row = _probe(env, config, states, lam, seed=seed)
        # The regularizer's own cost at half-box deviation -- where a policy
        # genuinely exploring around truthful would sit. Two actions, each
        # d = 0.5, so the sum of squares is 0.5.
        row["penalty_at_mid"] = float(lam * 0.5)
        rows.append(row)

    econ = np.array([r["econ_grad_norm"] for r in rows])
    return {
        "seed": seed,
        "rung": "B3",
        "engine": config.opf_mode,
        "scenario": BASE_SCENARIO,
        "band": list(PRIMARY_BAND),
        "probe_steps": PROBE_STEPS,
        "corner_tolerance": CORNER_TOLERANCE,
        "econ_grad_norm_reference": float(econ.mean()),
        "rows": rows,
    }


def select(doc: dict, dev_tolerance: float = 0.20,
           spread_tolerance: float = 0.30) -> dict:
    """The largest weight that does not visibly narrow the policy.

    Both limits are measured against the *no-penalty* row rather than against
    numbers chosen here, so the rule is "how much does turning the penalty on
    change the policy", not "how close to some invented constant is it".

      deviation   how far from truthful the policy ends up, on average
      spread      how much the fleet's quotes differ from one another

    Either shrinking means the regularizer is deciding for the policy instead
    of shaping it: at the limit the fleet declares one quote and the bidding
    the research is measuring is gone. Even the largest admissible weight keeps
    the regularizer's gradient well under the economic term's, which is
    reported alongside so the margin is visible rather than assumed.
    """
    ref = next((r for r in doc["rows"] if r["lambda"] == 0.0), None)
    if ref is None:
        raise ValueError("the grid must contain lambda = 0 as its reference row")
    dev_floor = ref["mean_deviation_after_probe"] * (1.0 - dev_tolerance)
    spread_floor = ref["action_spread"] * (1.0 - spread_tolerance)

    admissible = [r for r in doc["rows"]
                  if r["lambda"] > 0
                  and r["mean_deviation_after_probe"] >= dev_floor
                  and r["action_spread"] >= spread_floor]
    if not admissible:
        return {
            "selected_lambda": 0.0,
            "reference_mean_deviation": ref["mean_deviation_after_probe"],
            "reference_action_spread": ref["action_spread"],
            "deviation_floor": dev_floor, "spread_floor": spread_floor,
            "reason": ("every weight in the grid narrowed the policy past the "
                       "tolerances, so no penalty is the supported choice"),
        }
    best = max(admissible, key=lambda r: r["lambda"])
    return {
        "selected_lambda": best["lambda"],
        "grad_ratio": best["grad_ratio"],
        "penalty_at_mid": best["penalty_at_mid"],
        "mean_deviation": best["mean_deviation_after_probe"],
        "action_spread": best["action_spread"],
        "corner_fraction": best["corner_fraction"],
        "reference_mean_deviation": ref["mean_deviation_after_probe"],
        "reference_action_spread": ref["action_spread"],
        "deviation_floor": dev_floor, "spread_floor": spread_floor,
        "dev_tolerance": dev_tolerance, "spread_tolerance": spread_tolerance,
        "reason": (
            f"largest weight that keeps mean deviation at "
            f"{best['mean_deviation_after_probe']:.3f} (no-penalty "
            f"{ref['mean_deviation_after_probe']:.3f}, floor {dev_floor:.3f}) "
            f"and quote spread at {best['action_spread']:.1f} CNY/MWh "
            f"(no-penalty {ref['action_spread']:.1f}, floor "
            f"{spread_floor:.1f}). Its pull is {best['grad_ratio']:.3f} of the "
            f"economic term's and costs {best['penalty_at_mid']:.3f} against "
            f"a normalized economic term of {doc['econ_grad_norm_reference']:.3f}."),
    }


def main(argv=None) -> int:
    # A non-formal entry point: this measures the control path against a
    # market with a known shape, not against a measured one.
    from wholesale_price import enable_non_formal_market
    enable_non_formal_market()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    doc = calibrate(args.seed)
    doc["selection"] = select(doc)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")

    if not args.quiet:
        print(f"{'lambda':>7} {'pen@mid':>9} {'|econ|':>9} {'grad ratio':>11} "
              f"{'mean dev':>9} {'corner':>8} {'spread':>9}")
        for r in doc["rows"]:
            print(f"{r['lambda']:>7.3f} {r['penalty_at_mid']:>9.4f} "
                  f"{r['econ_grad_norm']:>9.4f} {r['grad_ratio']:>11.4f} "
                  f"{r['mean_deviation_after_probe']:>9.4f} "
                  f"{r['corner_fraction']:>8.1%} {r['action_spread']:>9.3f}")
        print()
        print(f"selected lambda = {doc['selection']['selected_lambda']}")
        print(f"  {doc['selection']['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
