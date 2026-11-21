# baselines.py
"""The baseline bidding policies an AI-assisted policy is measured against.

Each baseline is a callable that turns one agent's observation into the same
action a trained actor would emit, in the arm's own action space, so the
evaluator can run every arm through one code path and cannot compare against
something other than what it thinks it is comparing against. The rules read
named observation features rather than raw indices, so a change to the
observation layout breaks them loudly instead of silently re-pointing them.

The arms are measured on the storage fleet, so their action space is the
storage quote space: ``[charge_price, discharge_price]`` in CNY/MWh, both zero
at truthful. The direction convention is worth stating because it is easy to
invert and was, in fact, inverted here for two revisions: a unit charges when
it asks **little** to charge and refuses to sell (a high discharge price), and
discharges when it asks little to discharge and refuses to buy. A high charge
price is a unit declining to charge, not a unit bidding to charge.

The horizon-wide strategies in `strategies/` remain the interface for
`run.py`'s offline simulation; these are the per-block, causal policies the
rolling-horizon environment and the paired evaluator drive.

`truthful` is the headline baseline: the counterfactual the paper's benefit is
measured against.
"""

from typing import Callable, Dict, Optional

import numpy as np

from models import MarketConfig
from rl_spec import ACTION_STORAGE_QUOTE_V2, OBS_V6, ActionSpec

# The action a policy returns, in the declared action order.
Action = np.ndarray
Policy = Callable[[np.ndarray], Action]

BASELINE_NAMES = ("truthful", "rule", "myopic")


def _feature_index(spec=OBS_V6) -> Dict[str, int]:
    """Name -> position for every feature the observation carries."""
    return {name: i for i, name in enumerate(spec.feature_order)}


_FEATURE_INDEX = _feature_index()

# Bounds read off the action space the arms bid in, so a change to the space
# cannot leave a baseline declaring a price outside it.
_CHARGE_LOW, _CHARGE_HIGH = ACTION_STORAGE_QUOTE_V2.bounds["charge_price"]
_DISCHARGE_LOW, _DISCHARGE_HIGH = ACTION_STORAGE_QUOTE_V2.bounds["discharge_price"]
_TRUTHFUL = ACTION_STORAGE_QUOTE_V2.truthful_array()


def _read(obs: np.ndarray, name: str) -> float:
    return float(np.asarray(obs, dtype=float)[_FEATURE_INDEX[name]])


def truthful_policy(spec: ActionSpec = ACTION_STORAGE_QUOTE_V2) -> Policy:
    """Declare no strategic premium on either leg.

    A battery's round trip is costed separately, through cycle_cost, so zero
    here is the declaration that asks for nothing beyond what a cycle already
    costs -- and it is the counterfactual the reported benefit is measured
    against, so it must be the do-nothing-special case and nothing else.
    """
    truthful = spec.truthful_array()

    def policy(obs: np.ndarray) -> Action:
        return truthful.copy()
    return policy


def rule_policy(soc_low: float = 0.25, soc_high: float = 0.75,
                cheap: float = 0.9, dear: float = 1.1) -> Policy:
    """A deterministic state-of-charge and price rule, stated in full.

    The observation's ``soc`` is the fraction of capacity, ``last_lmp_norm`` and
    ``avg_lmp_hist_norm`` are prices as ratios of a frozen reference price. In
    order:

      1. soc <= soc_low    charge hard:  charge_price low, discharge_price high
      2. soc >= soc_high   discharge:    charge_price high, discharge_price low
      3. price <= cheap * average   charge preference, as above
      4. price >= dear  * average   discharge preference, as above
      5. otherwise         declare nothing

    Every branch is a constant, so the policy is reproducible without a seed.

    The direction is the part to be careful with. A battery charges when it
    asks *little* to charge -- a price below what the market charges it is a
    price it will be taken up on -- and it discharges when it asks little to
    discharge. A *high* charge price is a unit declining to buy. This rule was
    written the other way round while the objective still paid a unit for
    charging, and the tests only asserted an ordering, so the inversion went
    unnoticed; the steps below are named so the intent is checkable.
    """
    def policy(obs: np.ndarray) -> Action:
        soc = _read(obs, "soc")
        if soc <= soc_low:
            return np.array([_CHARGE_LOW, _DISCHARGE_HIGH], dtype=np.float32)
        if soc >= soc_high:
            return np.array([_CHARGE_HIGH, _DISCHARGE_LOW], dtype=np.float32)
        average = _read(obs, "avg_lmp_hist_norm")
        price = _read(obs, "last_lmp_norm")
        if average > 0 and price <= cheap * average:
            return np.array([_CHARGE_LOW + 0.2 * (_CHARGE_HIGH - _CHARGE_LOW),
                             0.8 * _DISCHARGE_HIGH], dtype=np.float32)
        if average > 0 and price >= dear * average:
            return np.array([0.8 * _CHARGE_HIGH,
                             _DISCHARGE_LOW + 0.2 * (_DISCHARGE_HIGH
                                                     - _DISCHARGE_LOW)],
                            dtype=np.float32)
        return _TRUTHFUL.copy()
    return policy


def myopic_policy(band: float = 0.05) -> Policy:
    """Act on this period's price alone, with no view forward or of state.

    Charging is declared cheap and discharging dear, judged against the running
    average price, and the state of charge is ignored entirely. That is what
    makes it myopic: it will happily discharge a flat battery or charge a full
    one, because it never looks. It is the control that separates "reacts to
    the price" from "plans around the battery". The direction convention is the
    one stated above the rule policy.
    """
    def policy(obs: np.ndarray) -> Action:
        average = _read(obs, "avg_lmp_hist_norm")
        price = _read(obs, "last_lmp_norm")
        if average <= 0:
            return _TRUTHFUL.copy()
        if price < average * (1.0 - band):
            return np.array([_CHARGE_LOW, _DISCHARGE_HIGH], dtype=np.float32)
        if price > average * (1.0 + band):
            return np.array([_CHARGE_HIGH, _DISCHARGE_LOW], dtype=np.float32)
        return _TRUTHFUL.copy()
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
