"""Re-derive the deviation weight for the v7 quote box by measurement.

The v5 quote is four entries wide per unit, so the weight selected for the
two-entry box describes a different regularizer. This measures the v7
trainer's own anchor -- the sum of squared box-normalised deviations from
`TRUTHFUL` -- against the critic term on one fixed day with no learning, and
leaves the selection to the same rule `calibrate_dev_penalty.select` applies.

The grid, the probe length and the selection rule are imported rather than
restated, so the v7 weight is chosen by the frozen procedure and not by a new
one. Nothing here clears a market and nothing computes a profit.
"""

from __future__ import annotations

import numpy as np
import torch

from calibrate_dev_penalty import (CORNER_TOLERANCE, LAMBDA_GRID, PROBE_STEPS,
                                   select)
from da_quote_map_v5 import ACTION_BOX_HIGH, TRUTHFUL, forward_policy_v5
from local_da_days import load_pair, train_pairs
from rl_da_env import LocalDABiddingEnv
from rl_da_gate_v7 import DAGateTrainerV7, DAV7ExperimentConfig

from local_da_audit import SCREEN  #: the one screen definition


def _fixed_setup(seed=123):
    """One v7 training day and its initial observations, identical per weight."""
    pair = load_pair(train_pairs(SCREEN)[0][0]["load_date"])
    env = LocalDABiddingEnv()
    np.random.seed(seed)
    obs = env.reset(pair)
    states = np.stack([obs[name] for name in env.agent_names])
    return env, torch.as_tensor(states, dtype=torch.float32)


def _policy_quotes(trainer, states, index, *, train):
    """The v5 quotes one actor emits for its own observation row."""
    out = trainer.actors[trainer.agent_names[index]](states[index:index + 1])
    return forward_policy_v5(out, train=train)["quotes"]


def _joint_econ_and_anchor(trainer, states, index, lam):
    """The critic term and the weighted anchor for one agent's actor loss."""
    out = trainer.actors[trainer.agent_names[index]](states[index:index + 1])
    quotes = forward_policy_v5(out, train=True)["quotes"]
    joint = torch.stack(
        [quotes if j == index else _policy_quotes(trainer, states, j, train=True)
         for j in range(len(trainer.agent_names))], dim=1)
    # The gate trainer's centralized critic reads every agent's observation
    # flattened, the same input its own update builds.
    q1, q2 = trainer.critics[trainer.agent_names[index]](
        states.reshape(1, -1), joint.reshape(1, -1))
    econ = -torch.min(q1, q2).mean()
    # The trainer's own anchor: its bounds and its truthful point, not a
    # re-derived width, so the share describes this trainer's regularizer.
    anchor = lam * trainer._anchor_tensor(quotes)
    return out, econ, anchor


def _probe(env, states, lam, seed):
    """Measure the anchor's gradient share and where the policy settles."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    trainer = DAGateTrainerV7(env, DAV7ExperimentConfig(anchor_weight=lam))
    n_agents = len(trainer.agent_names)

    # --- Gradient balance on the initial policy, before any update ---
    econ_norms, anchor_norms = [], []
    for index in range(n_agents):
        out, econ, anchor = _joint_econ_and_anchor(trainer, states, index, lam)
        econ_norms.append(float(torch.autograd.grad(
            econ, out, retain_graph=True)[0].norm()))
        anchor_norms.append(float(torch.autograd.grad(
            anchor, out, retain_graph=True)[0].norm()))
    econ_magnitude = float(np.mean(econ_norms))
    anchor_magnitude = float(np.mean(anchor_norms))

    # --- Where the policy settles when both terms pull ---
    for _ in range(PROBE_STEPS):
        for index, name in enumerate(trainer.agent_names):
            _, econ, anchor = _joint_econ_and_anchor(trainer, states, index, lam)
            opt = trainer.actor_opts[name]
            opt.zero_grad()
            (econ + anchor).backward()
            opt.step()

    with torch.no_grad():
        anchor_values = np.array(
            [float(trainer._anchor_tensor(
                _policy_quotes(trainer, states, index, train=False)))
             for index in range(n_agents)])
        quote_rows = np.concatenate(
            [_policy_quotes(trainer, states, index, train=False).numpy()
             for index in range(n_agents)], axis=0)
        # Per-dimension unsigned distance, the same statistic the rule's
        # corner fraction was defined on.
        deviation = np.abs(
            (quote_rows - np.asarray(TRUTHFUL)) / np.asarray(ACTION_BOX_HIGH))
        corner = float((deviation < CORNER_TOLERANCE).mean())

    return {
        "lambda": lam,
        "econ_grad_norm": econ_magnitude,
        "pen_grad_norm": anchor_magnitude,
        "grad_ratio": (anchor_magnitude / econ_magnitude
                       if econ_magnitude > 0 else float("inf")),
        "penalty_at_mid": float(lam * len(ACTION_BOX_HIGH) * 0.5 ** 2),
        "mean_deviation_after_probe": float(anchor_values.mean()),
        "min_deviation_after_probe": float(anchor_values.min()),
        "max_deviation_after_probe": float(anchor_values.max()),
        "corner_fraction": corner,
        "action_spread": float(quote_rows.std(axis=0).mean()),
        "quote_min": float(quote_rows.min()),
        "quote_max": float(quote_rows.max()),
    }


def calibrate(seed=123, grid=LAMBDA_GRID):
    """Probe every weight on the fixed v7 day and select one by the frozen rule."""
    env, states = _fixed_setup(seed)
    rows = [_probe(env, states, lam, seed) for lam in grid]
    doc = {
        "seed": seed,
        "trainer": "DAGateTrainerV7",
        "quote_map": "storage_quote_v5",
        "probe_steps": PROBE_STEPS,
        "corner_tolerance": CORNER_TOLERANCE,
        "econ_grad_norm_reference": float(
            np.mean([row["econ_grad_norm"] for row in rows])),
        "rows": rows,
    }
    selection = select(doc)
    weight = selection["selected_lambda"]
    share = next(row["grad_ratio"] for row in rows if row["lambda"] == weight)
    return dict(doc, selected_weight=weight, gradient_share=share,
                selection=selection)
