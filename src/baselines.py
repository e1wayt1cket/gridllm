# baselines.py
"""The baseline bidding policies an AI-assisted policy is measured against.

Each baseline is a callable that turns one agent's observation into the same
action a trained actor would emit, ``[bid_mult, offer_adder]``, so the
evaluator can run every arm through one code path and cannot compare against
something other than what it thinks it is comparing against. The rules read
named observation features rather than raw indices, so a change to the
observation layout breaks them loudly instead of silently re-pointing them.

The horizon-wide strategies in `strategies/` remain the interface for
`run.py`'s offline simulation; these are the per-block, causal policies the
rolling-horizon environment and the paired evaluator drive.

`truthful` is the headline baseline: the counterfactual the paper's benefit is
measured against.
"""

from typing import Callable, Dict, Optional

import numpy as np

from models import MarketConfig
from rl_spec import OBS_V3

# The action a policy returns, in the declared action order.
Action = np.ndarray
Policy = Callable[[np.ndarray], Action]

BASELINE_NAMES = ("truthful", "rule", "myopic")


def _feature_index(spec=OBS_V3) -> Dict[str, int]:
    """Name -> position for every feature the observation carries."""
    return {name: i for i, name in enumerate(spec.feature_order)}


_FEATURE_INDEX = _feature_index()

# Action bounds, kept in step with rl_spec.ACTION_BID_OFFER_V1.
_BID_LOW, _BID_HIGH = 0.3, 1.8
_OFFER_HIGH = 50.0


def _read(obs: np.ndarray, name: str) -> float:
    return float(np.asarray(obs, dtype=float)[_FEATURE_INDEX[name]])


def truthful_policy() -> Policy:
    """Declare the agent's stated prices unchanged.

    bid_mult = 1.0 and offer_adder = 0.0, i.e. each unit bids exactly what it is
    worth to it. This is the baseline the reported benefit is measured against,
    so it must be the do-nothing-special case and nothing else.
    """
    def policy(obs: np.ndarray) -> Action:
        return np.array([1.0, 0.0], dtype=np.float32)
    return policy


def rule_policy(soc_low: float = 0.25, soc_high: float = 0.75,
                cheap: float = 0.9, dear: float = 1.1) -> Policy:
    """A deterministic state-of-charge and price rule, stated in full.

    The observation's ``soc`` is the fraction of capacity, ``last_lmp_norm`` and
    ``avg_lmp_norm`` are prices as ratios of the day average. In order:

      1. soc <= soc_low    charge hard:  bid_mult 1.40, offer_adder 45
      2. soc >= soc_high   discharge:    bid_mult 0.60, offer_adder 0
      3. price <= cheap * average   charge:  bid_mult 1.20, offer_adder 30
      4. price >= dear  * average   discharge: bid_mult 0.80, offer_adder 5
      5. otherwise         leave the declared prices alone

    Every branch is a constant, so the policy is reproducible without a seed.
    """
    def policy(obs: np.ndarray) -> Action:
        soc = _read(obs, "soc")
        if soc <= soc_low:
            return np.array([1.40, 45.0], dtype=np.float32)
        if soc >= soc_high:
            return np.array([0.60, 0.0], dtype=np.float32)
        average = _read(obs, "avg_lmp_norm")
        price = _read(obs, "last_lmp_norm")
        if average > 0 and price <= cheap * average:
            return np.array([1.20, 30.0], dtype=np.float32)
        if average > 0 and price >= dear * average:
            return np.array([0.80, 5.0], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)
    return policy


def myopic_policy(band: float = 0.05) -> Policy:
    """Act on this period's price alone, with no view forward or of state.

    Charging is declared cheap and discharging dear, judged against the running
    average price, and the state of charge is ignored entirely. That is what
    makes it myopic: it will happily discharge a flat battery or charge a full
    one, because it never looks. It is the control that separates "reacts to
    the price" from "plans around the battery".
    """
    def policy(obs: np.ndarray) -> Action:
        average = _read(obs, "avg_lmp_norm")
        price = _read(obs, "last_lmp_norm")
        if average <= 0:
            return np.array([1.0, 0.0], dtype=np.float32)
        if price < average * (1.0 - band):
            return np.array([_BID_HIGH * 0.78, _OFFER_HIGH * 0.8],
                            dtype=np.float32)
        if price > average * (1.0 + band):
            return np.array([_BID_LOW + 0.25, 0.0], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)
    return policy


def make_baseline_policy(name: str,
                         config: Optional[MarketConfig] = None) -> Policy:
    """The policy for one baseline arm, by name.

    Returns a callable taking a single observation, so a baseline and a trained
    actor are interchangeable to the caller. ``config`` is accepted because a
    baseline may need the market's own rules; none currently does, and taking it
    keeps the evaluator from having to special-case them later.
    """
    if name == "truthful":
        return truthful_policy()
    if name == "rule":
        return rule_policy()
    if name == "myopic":
        return myopic_policy()
    raise ValueError(f"unknown baseline {name!r}; expected one of "
                     f"{BASELINE_NAMES}")
