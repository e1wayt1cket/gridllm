# participant_payoff.py
"""What each market participant actually earns.

The clearing objective is not a payoff. It values a storage unit at that unit's
own declared bid and offer, while the unit is settled at the nodal price when the
day is accounted; the two are different quantities, and only the second one is
money the participant keeps. This module holds the settlement view.

Each participant type gets its own model, and each model returns a breakdown
rather than a single number, so a change in earnings can be attributed to a
cause instead of merely observed.

Every figure is an energy multiplied by a price (see money.py): power MW, price
CNY/MWh, time h, money CNY.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

import money
from models import Agent, MarketConfig

# Participant kinds a payoff model is registered for. A `storage` participant is
# a battery alone: no load to serve and no on-site generation.
PARTICIPANT_TYPES = ("prosumer", "storage", "retailer")


def _periods(schedule: dict, n_periods: Optional[int]) -> int:
    """Number of leading periods to settle; None means the whole schedule."""
    if n_periods is not None:
        return int(n_periods)
    return len(np.asarray(schedule["served"] if "served" in schedule
                          else schedule["p_sell"]))


def _head(values, n: int) -> np.ndarray:
    return np.asarray(values, dtype=float)[:n]


@dataclass(frozen=True)
class PayoffBreakdown:
    """One participant's settled period, decomposed.

    ``total`` is derived from the components rather than stored alongside them,
    so a component can never drift out of the total.

    ``terminal_adjustment`` stays outside the components on purpose. It corrects
    for ending the day at a different state of charge; it is not a cash flow, so
    folding it in would corrupt every decomposition that sums the components.
    """

    revenue: float = 0.0
    purchase_cost: float = 0.0
    generation_cost: float = 0.0
    degradation_cost: float = 0.0
    other_cost: float = 0.0
    penalty: float = 0.0
    terminal_adjustment: float = 0.0
    energy_sold_mwh: float = 0.0
    energy_bought_mwh: float = 0.0
    charge_mwh: float = 0.0
    discharge_mwh: float = 0.0
    equivalent_cycles: float = 0.0

    @property
    def total(self) -> float:
        """Settled profit (CNY), excluding the terminal adjustment."""
        return (self.revenue - self.purchase_cost - self.generation_cost
                - self.degradation_cost - self.other_cost - self.penalty)

    @property
    def soc_neutral_total(self) -> float:
        """Profit adjusted to a common terminal state of charge."""
        return self.total + self.terminal_adjustment

    def as_dict(self) -> dict:
        out = {name: getattr(self, name)
               for name in self.__dataclass_fields__}
        out["total"] = self.total
        out["soc_neutral_total"] = self.soc_neutral_total
        return out


def soc_neutral_adjustment(soc_T: float, soc_0: float, e_max: float,
                           terminal_price: float) -> float:
    """Value of ending the day at a different state of charge (CNY).

    A day that finishes with a fuller battery has banked energy it did not
    sell, so comparing two arms on cash alone rewards the one that ran the
    battery down. Valuing the difference at one common price makes the
    comparison like for like. Apply the same price to both arms; the adjustment
    removes the level difference, not the risk difference, which remains a
    documented limitation.
    """
    return float(terminal_price) * (float(soc_T) - float(soc_0)) * float(e_max)


def resolve_terminal_price(config: MarketConfig,
                           wholesale: Optional[np.ndarray] = None) -> float:
    """The one terminal-SOC price to use for a pair of arms.

    ``config.storage.terminal_value`` when set, otherwise the day's mean
    wholesale price — the same fallback the clearing optimizers use, so the
    optimizer's incentive and the reported adjustment are the same number.
    """
    configured = config.storage.terminal_value
    if configured is not None:
        return float(configured)
    if wholesale is None or len(wholesale) == 0:
        raise ValueError(
            "no terminal price: config.storage.terminal_value is unset and no "
            "wholesale curve was given to fall back to")
    return float(np.mean(np.asarray(wholesale, dtype=float)))


class PayoffModel:
    """Settlement view for one participant type."""

    name: str = ""

    def payoff(self, schedule: dict, lmp_node: np.ndarray, agent: Agent,
               config: MarketConfig,
               n_periods: Optional[int] = None) -> PayoffBreakdown:
        """Settle one participant over ``n_periods`` (default: the whole day)."""
        raise NotImplementedError

    def profit(self, schedule: dict, lmp_node: np.ndarray, agent: Agent,
               config: MarketConfig,
               n_periods: Optional[int] = None) -> float:
        """Convenience: the settled total only."""
        return self.payoff(schedule, lmp_node, agent, config,
                           n_periods).total


class _LedgerPayoff(PayoffModel):
    """The shared five-term ledger.

    revenue          energy sold at the nodal price, plus the value the agent
                     places on the load actually served
    purchase_cost    energy bought at the nodal price
    generation_cost  the offer cost of the on-site generation that was used
    degradation_cost battery cycling
    penalty          unserved load

    ``with_load`` and ``with_generation`` let a specialization drop terms that
    are zero for it by definition, rather than relying on the caller's data
    happening to be zero.
    """

    def _ledger(self, schedule: dict, lmp_node: np.ndarray, agent: Agent,
                config: MarketConfig, n_periods: Optional[int],
                with_load: bool = True,
                with_generation: bool = True) -> PayoffBreakdown:
        n = _periods(schedule, n_periods)
        lmp = _head(lmp_node, n)
        p_sell = _head(schedule["p_sell"], n)
        p_buy = _head(schedule["p_buy"], n)
        ch = _head(schedule["p_ch"], n)
        dis = _head(schedule["p_dis"], n)
        cycle_cost = float(config.storage.cycle_cost)

        revenue = money.total_money(lmp, p_sell)
        generation_cost = 0.0
        penalty = 0.0
        if with_load:
            revenue += money.total_money(agent.bid_value,
                                         _head(schedule["served"], n))
            penalty = money.total_money(config.market_design.penalty_unserved,
                                        _head(schedule["unserved"], n))
        if with_generation:
            generation_cost = money.total_money(
                agent.offer_cost,
                _head(schedule["pv_used"], n) + _head(schedule["wind_used"], n))

        charge_mwh = float(np.sum(ch)) * money.DT_HOURS
        discharge_mwh = float(np.sum(dis)) * money.DT_HOURS
        storage = getattr(agent, "storage", None)
        e_max = float(getattr(storage, "e_max", 0.0) or 0.0)
        # One full cycle moves 2 * e_max of energy through the battery.
        cycles = ((charge_mwh + discharge_mwh) / (2.0 * e_max)
                  if e_max > 0 else 0.0)

        return PayoffBreakdown(
            revenue=revenue,
            purchase_cost=money.total_money(lmp, p_buy),
            generation_cost=generation_cost,
            degradation_cost=money.total_money(cycle_cost, ch + dis),
            penalty=penalty,
            energy_sold_mwh=float(np.sum(p_sell)) * money.DT_HOURS,
            energy_bought_mwh=float(np.sum(p_buy)) * money.DT_HOURS,
            charge_mwh=charge_mwh,
            discharge_mwh=discharge_mwh,
            equivalent_cycles=cycles,
        )


class ProsumerPayoff(_LedgerPayoff):
    """A site with load, and possibly on-site generation and a battery.

    Consumption is valued at the agent's truthful ``bid_value``, which the
    bidding action never touches: RL shades ``bid_mult`` at clearing time only.
    Charging the battery is a purchase, not a source of value — a battery does
    not earn the clearing objective's credit for declaring a high charge bid.
    """

    name = "prosumer"

    def payoff(self, schedule, lmp_node, agent, config, n_periods=None):
        return self._ledger(schedule, lmp_node, agent, config, n_periods,
                            with_load=True, with_generation=True)


class IndependentStoragePayoff(_LedgerPayoff):
    """A battery alone: no load to serve, no on-site generation.

    Profit = energy sales revenue - energy purchase cost - degradation, all
    settled at the nodal price over the period length. The load and generation
    terms are zero for this participant *by definition*, so they are not read
    from the schedule at all: a storage unit cannot earn a consumption value,
    and must not inherit one from whatever load profile happens to sit at its
    bus.
    """

    name = "independent_storage"

    def payoff(self, schedule, lmp_node, agent, config, n_periods=None):
        return self._ledger(schedule, lmp_node, agent, config, n_periods,
                            with_load=False, with_generation=False)


class RetailerPayoff(PayoffModel):
    """Placeholder for a retail supplier; not implemented.

    A retailer's payoff needs a retail tariff and its own served-load
    accounting (retail revenue minus wholesale purchase cost), neither of which
    the simulator models yet. Declared so the dispatch below is total and a
    retailer can never silently be settled as a prosumer.
    """

    name = "retailer"

    def payoff(self, schedule, lmp_node, agent, config, n_periods=None):
        raise NotImplementedError(
            "RetailerPayoff needs a retail tariff and the retailer's own "
            "served-load book; the simulator does not model either yet")


# participant_type -> the model that settles it. The key is the agent's declared
# type, not the model's descriptive name, which is only used in reports.
PAYOFF_REGISTRY: Dict[str, PayoffModel] = {
    "prosumer": ProsumerPayoff(),
    "storage": IndependentStoragePayoff(),
    "retailer": RetailerPayoff(),
}
assert set(PAYOFF_REGISTRY) == set(PARTICIPANT_TYPES), \
    "every declared participant type needs a payoff model"


def payoff_model_for(agent) -> PayoffModel:
    """The payoff model for one participant, by its declared type.

    Defaults to the prosumer ledger for an agent that predates the
    ``participant_type`` field, which is what every existing agent was.
    """
    kind = getattr(agent, "participant_type", "prosumer") or "prosumer"
    try:
        return PAYOFF_REGISTRY[kind]
    except KeyError:
        raise ValueError(
            f"unknown participant_type {kind!r}; expected one of "
            f"{PARTICIPANT_TYPES}") from None


def participant_payoff(schedule: dict, lmp_node: np.ndarray, agent: Agent,
                       config: MarketConfig,
                       n_periods: Optional[int] = None) -> PayoffBreakdown:
    """Settle one participant with its own payoff model."""
    return payoff_model_for(agent).payoff(schedule, lmp_node, agent, config,
                                          n_periods)
