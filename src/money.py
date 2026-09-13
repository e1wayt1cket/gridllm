# money.py
"""The project's single money convention.

Every monetary quantity in the simulator is an energy multiplied by a price.
A period is ``DT_HOURS`` long, so a period's power has to be scaled by it before
it can be priced: ``price * power`` alone is not money, it is money per period
expressed as if a period were an hour.

Units: power MW, time h, energy MWh, price CNY/MWh, money CNY.

This module is the one place that convention lives. It is a leaf — it imports
nothing from the project — because it is used both by the solvers and by the
numpy-only metrics layer, and `dispatch_core` pre-loads ortools to work around a
pandapower DLL ordering issue that the metrics layer must not inherit.
"""

import numpy as np

DT_HOURS = 0.25
PERIODS_PER_DAY = 96


def energy(power):
    """Energy (MWh) in one period from power (MW)."""
    return np.asarray(power, dtype=float) * DT_HOURS


def total_money(price, power) -> float:
    """Money (CNY) for a price series (CNY/MWh) times a power series (MW)."""
    return float(np.sum(np.asarray(price, dtype=float)
                        * np.asarray(power, dtype=float))) * DT_HOURS


def price_from_balance_dual(dual: float, dt: float = DT_HOURS) -> float:
    """Price (CNY/MWh) from the dual of a period power-balance constraint.

    The balance constrains power in MW over one period, so its dual is CNY per
    MW *for that period*, not per MWh. Dividing by the period length is what
    makes it a price. An objective rescaled into money without this step leaves
    every nodal price four times too low while still looking self-consistent.

    ``dt`` must be the same factor the objective's energy-valued terms were
    scaled by, so that an unscaled objective (``dt=1``) still yields the prices
    its own convention implies.
    """
    return -float(dual) / dt
