# strategies/greedy_bidding.py
"""A per-period price-threshold rule, with no look-ahead.

The rule is one comparison per period and nothing else:

    price[t] < low    ->  the unit wants to charge
    price[t] > high   ->  the unit wants to discharge
    otherwise         ->  the unit stays neutral

Each period is decided on its own price and on nothing else. No forecast
horizon, no dynamic program, no linear program, no knowledge of what later
periods will cost. That is the whole point of having it beside `mpc`: the MPC
solves a look-ahead LP over the next `H` periods, and this answers the opposite
question -- how far does a rule that cannot see forward get?

**It declares quotes; it does not dispatch.** The output is a pair of
reservation prices per period in the unit's own action space, and the clearing
decides what is taken. A unit cannot force itself to charge, and the quantities
it finally trades are the market's answer, not this rule's wish. Anything that
wrote `p_ch`/`p_dis` directly would be bypassing the market this study is about.

**Which corner of the quote box.** A unit expresses a direction by asking
little for the leg it wants and much for the leg it does not, so the declaration
is a corner of its own box (`mpc_storage_bidding` uses the same convention):

    wants to charge     charge_price = low,  discharge_price = high
    wants to discharge  charge_price = high, discharge_price = low
    neutral             both at the truthful point, which asks for nothing

**What price it reads, and what that makes it.** The signal is the external
day-ahead curve -- `market_history["price"]` when a caller supplies one, else
`day_ahead_price_china`. That curve is published before the day, so reading a
period's own value is information a real day-ahead bidder has, and the rule is
therefore a genuine *ex ante* rule rather than an after-the-fact one. But the
unit is settled at its **nodal** LMP, not at this curve, and the two differ by
the network's congestion and loss terms. So the signal is a forecast of the
settlement price and not the settlement price itself, and the strategy is
described accordingly wherever it is reported.

A run built from a *mean* price curve is weaker still: the curve is an average
over a window, so its shape is smoother than any single day's. The rule then
reads a statistic, not a day, and no result on it may be read as evidence about
day-to-day behaviour.

**Thresholds are declared, never fitted.** They come from configuration
(`greedy.*`), and the values shipped are a pre-registered heuristic chosen from
the real price band -- not a search over outcomes. Nothing here reads the curve
being evaluated to set them.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from models import Agent, MarketConfig
from rl_spec import action_spec_for
from strategies import BiddingStrategy, register_strategy

#: Config block holding the thresholds.
CONFIG_KEY = "greedy"

#: Sources a threshold pair may come from.
#:
#: `fixed` reads the declared constants. `historical_mean` derives them from
#: the historical mean curves by quantile, which is legitimate only because
#: those curves are information available before the day -- and is NOT
#: available for the curve under evaluation, which would be look-ahead.
THRESHOLD_SOURCES = ("fixed", "historical_mean")


def resolve_thresholds(config: Optional[MarketConfig] = None) -> Tuple[float, float]:
    """The (low, high) thresholds in CNY/MWh, and where they came from.

    Read from configuration. A `historical_mean` source derives them from the
    trailing-window curves by quantile; see `historical_thresholds`.
    """
    from config_loader import get_default

    source = str(get_default(f"{CONFIG_KEY}.threshold_source", "fixed"))
    if source not in THRESHOLD_SOURCES:
        raise ValueError(
            f"unknown {CONFIG_KEY}.threshold_source {source!r}; expected one "
            f"of {THRESHOLD_SOURCES}")
    if source == "historical_mean":
        return historical_thresholds()

    low = float(get_default(f"{CONFIG_KEY}.low_threshold", 300.0))
    high = float(get_default(f"{CONFIG_KEY}.high_threshold", 400.0))
    if not low < high:
        raise ValueError(
            f"the greedy thresholds must satisfy low < high; got "
            f"low={low!r}, high={high!r}. With low >= high no period could be "
            f"both below the low threshold and above the high one, and the "
            f"rule would have no neutral band.")
    return low, high


def historical_thresholds(
        sheets: Tuple[str, ...] = ("7days", "30days", "halfyear"),
        quantiles: Tuple[float, float] = (0.25, 0.75)) -> Tuple[float, float]:
    """Thresholds from the trailing-window mean curves, per curve.

    Each curve is reduced on its OWN quantiles and the results are pooled; the
    three are never concatenated into one sample, because they cover different
    windows and pooling them would weight the longest window by its length.

    This is a *descriptive sensitivity* source. The curves' window end dates
    have not been verified as preceding the evaluation day (see
    `reports/jiangsu_market_mechanism.md`), so a pair derived this way may not
    be called strictly non-anticipative.
    """
    from dianyicha_price import read_curves

    curves = read_curves()
    lows, highs = [], []
    for sheet in sheets:
        if sheet not in curves:
            continue
        arr = curves[sheet].array()
        lows.append(float(np.quantile(arr, quantiles[0])))
        highs.append(float(np.quantile(arr, quantiles[1])))
    if not lows:
        raise ValueError(
            f"no historical curve among {sheets} was readable; cannot derive "
            f"a historical_mean threshold pair")
    return float(np.mean(lows)), float(np.mean(highs))


def price_signal(agents: List[Agent], config: MarketConfig, T: int,
                 market_history: Optional[dict] = None) -> np.ndarray:
    """The per-period price the rule compares against.

    `market_history["price"]` when the caller supplies one, else the scenario's
    own external day-ahead curve. Either way it is a published-forward series,
    and each period is read on its own.
    """
    if isinstance(market_history, dict) and market_history.get("price") is not None:
        arr = np.asarray(market_history["price"], dtype=float)
        if arr.size >= T:
            return arr[:T]
    from grid import day_ahead_price_china
    return np.asarray(day_ahead_price_china(T, agents=agents, config=config),
                      dtype=float)


def greedy_quotes(price, low: float, high: float,
                  spec) -> Dict[str, np.ndarray]:
    """The quote pair each period, from that period's price alone.

    `price[t]` decides period `t` and nothing else; the arrays are built by a
    single elementwise comparison, so a future price cannot reach an earlier
    decision even by accident.
    """
    price = np.asarray(price, dtype=float)
    if not np.all(np.isfinite(price)):
        raise ValueError("the price signal carries a non-finite value")

    charge_price = np.full(price.size, float(spec.truthful["charge_price"]))
    discharge_price = np.full(price.size, float(spec.truthful["discharge_price"]))
    c_lo, c_hi = spec.bounds["charge_price"]
    d_lo, d_hi = spec.bounds["discharge_price"]

    wants_charge = price < low
    wants_discharge = price > high
    if np.any(wants_charge & wants_discharge):          # pragma: no cover
        raise ValueError(
            "a period is both below the low threshold and above the high one; "
            "the thresholds are not ordered")

    # Charge: ask little to charge, decline to sell.
    charge_price[wants_charge] = c_lo
    discharge_price[wants_charge] = d_hi
    # Discharge: ask little to discharge, decline to buy.
    charge_price[wants_discharge] = c_hi
    discharge_price[wants_discharge] = d_lo
    return {"charge_price": charge_price, "discharge_price": discharge_price}


@register_strategy("greedy_threshold")
class GreedyThresholdStrategy(BiddingStrategy):
    """Each storage unit bids a threshold rule on the current period's price.

    Non-storage participants are left at their own truthful declaration: this
    strategy is a statement about storage bidding, and shading a load's quote
    would change the comparison it exists to make.
    """

    def formulate(
        self,
        agents: List[Agent],
        config: MarketConfig,
        market_history: Optional[dict] = None,
        T: int = 96,
    ) -> dict:
        low, high = resolve_thresholds(config)
        price = price_signal(agents, config, T, market_history)

        actions: Dict[str, Dict[str, np.ndarray]] = {}
        for a in agents:
            spec = action_spec_for(a)
            if a.storage is None or "charge_price" not in spec.action_names:
                # Not a battery in the storage quote space: leave it truthful.
                actions[a.name] = {n: np.full(T, spec.truthful[n], dtype=float)
                                   for n in spec.action_names}
                continue
            actions[a.name] = greedy_quotes(price[:T], low, high, spec)
        return actions
