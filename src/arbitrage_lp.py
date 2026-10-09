# arbitrage_lp.py
"""The best a single storage unit can do against a price series it cannot move.

This is the opportunity *bound*: one unit, perfect foresight of the day's nodal
prices, free to charge and discharge within its own limits, with the round trip
costed at `cycle_cost` on both legs and the endpoints pinned. Nothing about
bidding enters, so the answer does not depend on any policy being trained or on
any quote being declared. If a perfect forecaster cannot clear the degradation
floor, no bidding strategy can, which is what makes this the instrument that
decides whether a market has arbitrage in it at all.

The price vector is exogenous and that is the point. A unit priced against a
clearing it took part in is priced against a field it helped set, and its own
optimum then moves the prices it is optimizing over. The bound only means
something if the prices are given.

`soc` is a *fraction of energy capacity*, not MWh, and the transition carries
the same `DT_HOURS / e_max` factor the clearing writes, so this LP and the market
describe the same battery rather than two batteries that happen to share a name.

This module is a leaf by construction: numpy, scipy and `money`, nothing else.
The repo's clearing machinery pre-loads ortools to work around a pandapower DLL
ordering issue, and a bound that could only be evaluated with a Gurobi licence
would be a bound that cannot be tested.

Units, throughout: power MW, time h, energy MWh, price CNY/MWh, money CNY.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import money

#: A period's flow below this (MW) counts as the unit not using that period.
#: The tolerance is on power because the schedule is a power trajectory; a
#: period at 1e-9 MW is the solver's noise, not a decision to charge.
DEFAULT_TOL_MW = 1e-6


def required_margin(cycle_cost: float, eta_ch: float, eta_dis: float) -> float:
    """The round-trip margin a cycle must clear to be worth running, CNY/MWh.

    Derived from this module's own objective, per MWh **stored**. Put one MWh
    into the battery: it costs `p_c / eta_ch` to buy at the charge price, and
    releases `eta_dis` MWh at the discharge price, so it earns `eta_dis * p_d`.
    `cycle_cost` is charged on **both** legs, and the two legs are not equal in
    energy -- the charge leg draws `1 / eta_ch` MWh from the grid to store one,
    and the discharge leg returns `eta_dis`. So the degradation on one stored
    MWh is `cycle_cost * (1 / eta_ch + eta_dis)`, and the cycle pays only if

        eta_dis * p_d - p_c / eta_ch  >  cycle_cost * (1 / eta_ch + eta_dis)

    The right-hand side is this function. It is **not** `2 * cycle_cost`: that
    is the lossless-round-trip special case, and it is short by
    `cycle_cost * (1/eta + eta - 2)`, which at the configured 100 CNY/MWh per
    leg and 0.92 efficiency is 0.6957 CNY/MWh -- small, but it is the whole
    margin on a day that sits near the threshold, so it is not dropped.

    The break-even was confirmed against the LP itself: with the two prices
    chosen so the left-hand side equals this value exactly, the LP returns an
    optimum of zero and is indifferent about charging, and half a CNY/MWh above
    it the optimum turns positive.
    """
    if not eta_ch > 0 or not eta_dis > 0:
        raise ValueError("efficiencies must be positive")
    return float(cycle_cost) * (1.0 / float(eta_ch) + float(eta_dis))


@dataclass(frozen=True)
class UnitSpec:
    """One battery's own parameters, as scalars.

    Spelled out here rather than taking a `StorageSpec` so this module stays a
    leaf. `StorageSpec` is a dataclass with no behaviour, so the adapter in the
    caller is a field-for-field copy and the two cannot drift without the
    caller's construction failing.
    """

    e_max: float
    p_ch_max: float
    p_dis_max: float
    eta_ch: float
    eta_dis: float
    soc0: float
    soc_min: float
    soc_max: float


@dataclass(frozen=True)
class ArbitrageResult:
    """The LP's optimum, decomposed into the quantities an audit reports.

    The spread terms are computed from the LP's *own* charge and discharge
    prices rather than from the day's minimum and maximum. Those differ, and the
    difference is not small: on the native day the cheapest period is not
    jointly reachable with a peak through a 0.4 MW / 1.6 MWh window, so the
    day's extremes overstate the opportunity by roughly a factor of two. Pricing
    the spread off the schedule the unit can actually run is what keeps this a
    bound it can reach.
    """

    margin_cny: float
    """The LP optimum: settled profit over the day, CNY."""

    status: int
    status_text: str

    ch_mw: np.ndarray
    dis_mw: np.ndarray
    soc: np.ndarray
    """State of charge as a fraction of `e_max`, length T+1."""

    chargeable_energy_mwh: float
    dischargeable_energy_mwh: float
    feasible_charge_periods: int
    feasible_discharge_periods: int
    equivalent_cycles: float

    best_charge_price: float
    """Mean price over the periods the LP chose to charge at, CNY/MWh.

    A mean rather than the minimum: the optimum spreads its charging over
    several periods, and quoting the single cheapest of them would describe a
    schedule the unit did not run. NaN when the unit never charges.
    """

    best_discharge_price: float
    """Mean price over the periods the LP chose to discharge at. NaN if none."""

    gross_spread_cny_mwh: float
    """`best_discharge_price - best_charge_price`, before any cost."""

    efficiency_loss_cny_mwh: float
    """What the round trip's losses cost, per MWh stored."""

    degradation_cost_cny_mwh: float
    """`cycle_cost * (1/eta_ch + eta_dis)` per MWh stored -- see
    `required_margin`. Charged on both legs, and the two legs differ in energy,
    so this is not `2 * cycle_cost`."""

    adjusted_spread_cny_mwh: float
    """`gross_spread - efficiency_loss - degradation_cost`, per MWh stored."""

    @property
    def margin_per_mwh_charged(self) -> float:
        """The optimum, per MWh the unit actually put in. NaN if it never charges."""
        if self.chargeable_energy_mwh <= 0:
            return float("nan")
        return self.margin_cny / self.chargeable_energy_mwh


def _build_equalities(T: int, unit: UnitSpec, dt: float) -> tuple:
    """The (T+2, 3T+1) equality block: endpoints pinned, dynamics between them.

    Row 0 pins `soc[0]`; rows 1..T are the transitions; row `T+1` pins
    `soc[T] = soc[0]`. The terminal pin is a *separate row* rather than an
    appendage to the last transition. Writing it onto that row would silently
    replace the `t = T-1` dynamics and leave the final period's energy
    unaccounted for, which shows up as a unit discharging energy it never
    charged -- an error that looks like a very good day rather than a bug.
    """
    n_vars = 3 * T + 1
    g = dt / unit.e_max
    rows, cols, vals, b = [], [], [], []

    # Row 0: soc[0] = soc0.
    rows.append(0); cols.append(2 * T); vals.append(1.0)
    b.append(unit.soc0)

    # Rows 1..T: soc[t+1] - soc[t] - (eta_ch * ch[t] - dis[t] / eta_dis) * g = 0.
    for t in range(T):
        r = 1 + t
        rows.append(r); cols.append(2 * T + t + 1); vals.append(1.0)     # soc[t+1]
        rows.append(r); cols.append(2 * T + t); vals.append(-1.0)        # -soc[t]
        rows.append(r); cols.append(t); vals.append(-unit.eta_ch * g)    # -eta_ch*ch*g
        rows.append(r); cols.append(T + t); vals.append(g / unit.eta_dis)  # +dis*g/eta_dis
        b.append(0.0)

    # Row T+1: soc[T] - soc[0] = 0.
    rows.append(T + 1); cols.append(2 * T + T); vals.append(1.0)
    rows.append(T + 1); cols.append(2 * T); vals.append(-1.0)
    b.append(0.0)

    A_eq = coo_matrix((vals, (rows, cols)), shape=(T + 2, n_vars)).tocsr()
    return A_eq, np.array(b, dtype=float)


def arbitrage_lp(price_cny_per_mwh, unit: UnitSpec, cycle_cost: float,
                 dt: float = money.DT_HOURS,
                 tol_mw: float = DEFAULT_TOL_MW) -> ArbitrageResult:
    """The most one unit can earn from `price_cny_per_mwh`, and how.

    Maximizes, over charge `ch` and discharge `dis` in MW,

        sum_t  price[t] * (dis[t] - ch[t]) * dt  -  cycle_cost * (ch[t] + dis[t]) * dt

    subject to the state-of-charge transition, the power limits, the state-of-
    charge band, and `soc[T] == soc[0]`. The terminal pin is what makes the
    figure a whole-day profit rather than a profit plus whatever energy the unit
    chose to leave banked.

    No charge/discharge exclusivity is imposed, and none is needed: with both
    legs positive at a period the two price terms cancel exactly and the
    contribution is `-2 * cycle_cost * dt`, so a positive `cycle_cost` makes
    simultaneous flow strictly suboptimal on its own. That argument fails at
    `cycle_cost <= 0`, where the LP would be free to churn for a self-cancelling
    zero, so a non-positive cost is refused rather than answered.
    """
    price = np.asarray(price_cny_per_mwh, dtype=float)
    if price.ndim != 1:
        raise ValueError(f"price must be one series; got shape {price.shape}")
    T = int(price.size)
    if T < 1:
        raise ValueError("price series is empty")
    if not np.all(np.isfinite(price)):
        raise ValueError("price series carries a non-finite value")
    if cycle_cost <= 0:
        raise ValueError(
            f"cycle_cost must be positive to keep charging and discharging "
            f"exclusive; got {cycle_cost!r}")
    for name in ("e_max", "eta_ch", "eta_dis"):
        if float(getattr(unit, name)) <= 0:
            raise ValueError(f"{name} must be positive; got {getattr(unit, name)!r}")
    if not unit.soc_min <= unit.soc0 <= unit.soc_max:
        raise ValueError(
            f"soc0 {unit.soc0!r} is outside the band "
            f"[{unit.soc_min!r}, {unit.soc_max!r}]")

    # Objective: maximize price*(dis - ch)*dt - cycle_cost*(ch + dis)*dt, which
    # linprog minimizes negated. The two legs separate cleanly:
    #   ch  coefficient  (price + cycle_cost) * dt
    #   dis coefficient  (cycle_cost - price) * dt
    c = np.empty(3 * T + 1, dtype=float)
    c[0:T] = (price + cycle_cost) * dt
    c[T:2 * T] = (cycle_cost - price) * dt
    c[2 * T:] = 0.0

    bounds = ([(0.0, unit.p_ch_max)] * T
              + [(0.0, unit.p_dis_max)] * T
              + [(unit.soc_min, unit.soc_max)] * (T + 1))

    A_eq, b_eq = _build_equalities(T, unit, dt)
    # Presolve is off deliberately. On one bus-day of a real clearing -- day 20
    # of the native library at bus 17, whose nodal prices carry a near-tie of
    # 5e-6 CNY/MWh between two periods -- HiGHS presolve returned status 15
    # ("Unknown", primal feasible) and the bound simply failed to exist for
    # that bus. The same instance solves to the same optimum with presolve
    # disabled, and so does every other instance tried. It is also faster here
    # (1.8 ms against 2.2 ms over a sample of 96-period vectors), because this
    # LP is small enough that the presolve pass costs more than it returns.
    # A bound that is unavailable on a small fraction of inputs is not a bound.
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs",
                  options={"presolve": False})

    if not res.success:
        raise RuntimeError(
            f"the arbitrage LP did not solve: status {res.status} "
            f"({res.message})")

    x = np.asarray(res.x, dtype=float)
    ch = x[0:T]
    dis = x[T:2 * T]
    soc = x[2 * T:3 * T + 1]
    margin = -float(res.fun)

    charge_mwh = float(np.sum(ch)) * dt
    discharge_mwh = float(np.sum(dis)) * dt
    ch_mask = ch > tol_mw
    dis_mask = dis > tol_mw

    # The spread is quoted on the periods the unit actually ran, weighted by how
    # much it moved in each, so it describes the schedule rather than an
    # arbitrary pair of hours the unit never chose.
    best_ch = (float(np.sum(price[ch_mask] * ch[ch_mask]) / np.sum(ch[ch_mask]))
               if ch_mask.any() else float("nan"))
    best_dis = (float(np.sum(price[dis_mask] * dis[dis_mask]) / np.sum(dis[dis_mask]))
                if dis_mask.any() else float("nan"))

    if np.isfinite(best_ch) and np.isfinite(best_dis):
        gross = best_dis - best_ch
        # Buying one MWh at `best_ch` to put it in costs best_ch/eta_ch, and one
        # MWh in yields eta_dis at best_dis. The gap between those and the naive
        # spread is what the round trip loses to conversion, separately from
        # what it loses to degradation.
        eff_loss = best_ch * (1.0 / unit.eta_ch - 1.0) + best_dis * (1.0 - unit.eta_dis)
        # The degradation on one MWh *stored*, which is the unit `gross` and
        # `eff_loss` are quoted in: the charge leg draws 1/eta_ch MWh from the
        # grid, the discharge leg returns eta_dis. `2 * cycle_cost` would be
        # the lossless special case; see `required_margin`.
        deg = required_margin(cycle_cost, unit.eta_ch, unit.eta_dis)
        adjusted = gross - eff_loss - deg
    else:
        gross = eff_loss = adjusted = float("nan")
        deg = required_margin(cycle_cost, unit.eta_ch, unit.eta_dis)

    return ArbitrageResult(
        margin_cny=margin,
        status=int(res.status),
        status_text=str(res.message),
        ch_mw=ch,
        dis_mw=dis,
        soc=soc,
        chargeable_energy_mwh=charge_mwh,
        dischargeable_energy_mwh=discharge_mwh,
        feasible_charge_periods=int(np.count_nonzero(ch_mask)),
        feasible_discharge_periods=int(np.count_nonzero(dis_mask)),
        equivalent_cycles=((charge_mwh + discharge_mwh) / (2.0 * unit.e_max)
                           if unit.e_max > 0 else 0.0),
        best_charge_price=best_ch,
        best_discharge_price=best_dis,
        gross_spread_cny_mwh=gross,
        efficiency_loss_cny_mwh=eff_loss,
        degradation_cost_cny_mwh=deg,
        adjusted_spread_cny_mwh=adjusted,
    )


def lp_schedule(result: ArbitrageResult) -> dict:
    """The LP's dispatch as a settlement-shaped schedule.

    Exists so the LP's own objective can be checked against the repo's one
    settlement implementation. `IndependentStoragePayoff` reads `p_sell` and
    `p_buy`, and for a unit with no load and no on-site generation those are the
    two sides of one net flow, exactly as `dispatch_core.split_power` splits it.
    Reproduced here in two lines rather than imported because `dispatch_core`
    pre-loads ortools, which is the dependency this module exists to avoid.

    The load and generation keys are present and zero so the same dict can be
    handed to any payoff model without a KeyError; `IndependentStoragePayoff`
    does not read them, by definition.
    """
    T = int(result.ch_mw.size)
    net = np.asarray(result.dis_mw, dtype=float) - np.asarray(result.ch_mw, dtype=float)
    p_sell = np.clip(net, 0.0, None)
    p_buy = np.clip(-net, 0.0, None)
    zeros = np.zeros(T)
    return {
        "served": zeros.copy(),
        "unserved": zeros.copy(),
        "pv_used": zeros.copy(),
        "wind_used": zeros.copy(),
        "p_ch": np.asarray(result.ch_mw, dtype=float).copy(),
        "p_dis": np.asarray(result.dis_mw, dtype=float).copy(),
        "p_buy": p_buy,
        "p_sell": p_sell,
        # The clearing writes the final state of charge into the last slot, so
        # the schedule's `soc` runs T long while the trajectory is T+1.
        "soc": np.asarray(result.soc, dtype=float)[1:].copy(),
    }


def spec_from_storage(storage, **overrides) -> UnitSpec:
    """Adapt a `models.StorageSpec` to a `UnitSpec`, with optional overrides.

    Field-for-field, so a property the caller needs that the spec does not carry
    fails at construction rather than defaulting to zero behind the answer.
    """
    fields = dict(
        e_max=float(storage.e_max),
        p_ch_max=float(storage.p_ch_max),
        p_dis_max=float(storage.p_dis_max),
        eta_ch=float(storage.eta_ch),
        eta_dis=float(storage.eta_dis),
        soc0=float(storage.soc0),
        soc_min=float(storage.soc_min),
        soc_max=float(storage.soc_max),
    )
    fields.update(overrides)
    return UnitSpec(**fields)
