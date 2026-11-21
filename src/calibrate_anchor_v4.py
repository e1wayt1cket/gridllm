# calibrate_anchor_v4.py
"""Choose the v4 anchor weight by measurement, not by profit.

v4 actor loss:  L = -min(Q1, Q2) + lam * mean((quote / high)^2)

The anchor pulls every quote toward the truthful point (0, 0) of the
storage_quote_v4 space. Its weight lam is dimensionless against the scaled Q
units, so the question is a balance: how much of the actor gradient does the
anchor claim, and where does it leave the policy?

Method (mirrors `calibrate_dev_penalty.py`): one fixed Train day, one fixed
observation, one frozen critic at a fixed initialisation, no learning. The
critic's gradient is arbitrary but *identical across weights*; only the
anchor differs. Per weight we report:

  anchor_at_quotes   anchor value at the initial policy's quotes
  grad_ratio         ||lam * d anchor / d policy_out|| / ||d econ / d policy_out||
  corner_frac        after a 200-step actor-only probe, share of legs within
                     2% of the truthful corner (quote / high < 0.02)
  quote_mean         mean quote / high after the probe
  quote_spread       std of quote / high across legs after the probe

**Nothing here selects on profit.** Selection rule (pre-registered): the
largest lam with grad_ratio <= 0.20 and corner_frac(lam) - corner_frac(0)
== 0 — the anchor must not park a single leg on the truthful corner beyond
what the economic gradient alone already parks (the frozen critic itself
pushes most quotes down on this day, so the lam = 0 probe is the honest
reference). The quadratic anchor's gradient grows with the quote, so a
weight that looks subordinate on the batch average can still override the
econ term on the one leg the critic pushes *up*; the zero-excess rule is
the guard against that. On a flat Q surface even a small anchor is the
only gradient present, and Adam's per-parameter normalisation makes it
decisive there — small weights do not mean no anchoring.

Fixed day: `2022-12-09|2025-12-09` (Train split, winter; the same day the
ranking probes used). Read-only: no clearing, no buffer writes, no saves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import da_quote_map as qmap
import rl_da_gate as mod
import v2_cy_jn_env as envmod

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "algorithm_diagnostics" / "v4_anchor_calibration.json"

DAY = {"load_date": "2022-12-09", "price_date": "2025-12-09",
       "pair_id": "2022-12-09|2025-12-09"}
LAMBDA_GRID = (0.0, 0.01, 0.03, 0.1, 0.3)
PROBE_STEPS = 200
CORNER_TOLERANCE = 0.02
GRAD_RATIO_CAP = 0.20
CORNER_EXCESS_CAP = 0.0
SEED = 123
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")


def _setup():
    env = envmod.build_env_for_pair(DAY, differential_reward=False,
                                    terminal_soc_target=0.5)
    np.random.seed(SEED)
    random_state = np.random.get_state()
    obs = env.reset()
    np.random.set_state(random_state)
    torch.manual_seed(SEED)
    trainer = mod.DAGateTrainer(env, mod.DAExperimentConfig("calibration"),
                                agent_names=list(NAMES))
    states = torch.as_tensor(
        np.stack([obs[nm] for nm in NAMES]), dtype=torch.float32)
    return trainer, states


def _econ_and_anchor_grads(trainer, states):
    """Gradient of -minQ and of the anchor wrt policy_out, per unit, at the
    frozen initialisation. Identical critic for every lambda."""
    econ_grads, anchor_grads, anchor_vals = {}, {}, {}
    highs = torch.as_tensor(qmap.LEG_HIGHS, dtype=torch.float32)
    for i, nm in enumerate(NAMES):
        policy_out = trainer.actors[nm](states[i:i + 1])
        quotes = trainer._actor_loss_quotes(policy_out)
        q1, q2 = trainer.critics[nm](
            states.reshape(1, -1),
            torch.cat([quotes if j == i else
                       torch.as_tensor(np.zeros((1, 2)), dtype=torch.float32)
                       for j in range(len(NAMES))], dim=-1).reshape(1, -1))
        econ = -torch.min(q1, q2).mean()
        anchor = ((quotes / highs) ** 2).mean()
        g_econ, = torch.autograd.grad(econ, policy_out, retain_graph=True)
        g_anchor, = torch.autograd.grad(anchor, policy_out)
        econ_grads[nm] = g_econ.detach()
        anchor_grads[nm] = g_anchor.detach()
        anchor_vals[nm] = float(anchor.detach())
    return econ_grads, anchor_grads, anchor_vals


def _probe(trainer, states, lam: float) -> dict:
    """200 actor-only steps with the frozen critic; where does the policy go?"""
    actors = {nm: torch.optim.Adam(trainer.actors[nm].parameters(),
                                   lr=trainer.experiment.lr)
              for nm in NAMES}
    highs = torch.as_tensor(qmap.LEG_HIGHS, dtype=torch.float32)
    for step in range(PROBE_STEPS):
        for i, nm in enumerate(NAMES):
            actor = trainer.actors[nm]
            policy_out = actor(states[i:i + 1])
            quotes = trainer._actor_loss_quotes(policy_out)
            replaced = torch.cat(
                [quotes if j == i else torch.zeros((1, 2))
                 for j in range(len(NAMES))], dim=-1)
            q1, q2 = trainer.critics[nm](states.reshape(1, -1),
                                         replaced.reshape(1, -1))
            loss = -torch.min(q1, q2).mean() \
                + lam * ((quotes / highs) ** 2).mean()
            actors[nm].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actors[nm].step()
    fracs = []
    with torch.no_grad():
        for i, nm in enumerate(NAMES):
            policy_out = trainer.actors[nm](states[i:i + 1])
            quotes = qmap.target_quotes_v3(policy_out)
            fracs.extend((quotes[0] / highs).tolist())
    fracs = np.asarray(fracs, dtype=float)
    return {"corner_frac": float(np.mean(fracs < CORNER_TOLERANCE)),
            "quote_mean": float(np.mean(fracs)),
            "quote_spread": float(np.std(fracs))}


def calibrate() -> dict:
    trainer, states = _setup()
    econ_grads, anchor_grads, anchor_vals = _econ_and_anchor_grads(
        trainer, states)
    econ_norm = float(np.mean([float(g.norm()) for g in econ_grads.values()]))
    anchor_norm = float(np.mean([float(g.norm())
                                 for g in anchor_grads.values()]))
    init_snapshot = {nm: {k: v.detach().clone()
                          for k, v in trainer.actors[nm].state_dict().items()}
                     for nm in NAMES}
    rows = {}
    for lam in LAMBDA_GRID:
        for nm in NAMES:
            trainer.actors[nm].load_state_dict(init_snapshot[nm])
        probe = _probe(trainer, states, lam)
        rows[str(lam)] = {
            "grad_ratio": (lam * anchor_norm / econ_norm
                           if econ_norm > 0 else float("inf")),
            "anchor_at_init": float(np.mean(list(anchor_vals.values()))),
            **probe,
        }
        print(f"lam {lam}: grad_ratio {rows[str(lam)]['grad_ratio']:.3f} "
              f"corner {probe['corner_frac']:.2f} "
              f"quote_mean {probe['quote_mean']:.3f}", flush=True)
    corner0 = rows["0.0"]["corner_frac"]
    for lam in LAMBDA_GRID:
        rows[str(lam)]["corner_excess_vs_lam0"] = \
            rows[str(lam)]["corner_frac"] - corner0
    eligible = [lam for lam in LAMBDA_GRID if lam > 0.0
                and rows[str(lam)]["grad_ratio"] <= GRAD_RATIO_CAP
                and rows[str(lam)]["corner_excess_vs_lam0"]
                <= CORNER_EXCESS_CAP + 1e-12]
    selected = max(eligible) if eligible else None
    return {"day": DAY["pair_id"], "seed": SEED,
            "econ_grad_norm_at_init": econ_norm,
            "anchor_grad_norm_at_init": anchor_norm,
            "grid": rows,
            "rule": {"grad_ratio_cap": GRAD_RATIO_CAP,
                     "corner_excess_cap": CORNER_EXCESS_CAP,
                     "reference": "corner_frac at lam = 0 (frozen-critic "
                                  "econ-only baseline)",
                     "select": "largest lam > 0 passing both caps"},
            "selected": selected}


def main() -> int:
    doc = calibrate()
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps({"selected": doc["selected"], "out": str(OUT)},
                     indent=2))
    return 0 if doc["selected"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
