# surplus_metrics.py
"""Three-layer accounting metrics for RL-vs-truthful evaluation.

Splits one cleared day into user outcomes (consumer payment / consumer
surplus), mechanism outcomes (LMP markup, market-power decomposition) and the
existing system-level welfare, so profit gains can be traced to who pays.

Pure numpy + models; no torch, no rl_env, no eval_agents at import time so the
module stays lightweight and reusable from eval, diagnostics and reports.

Accounting convention (load-bearing): the whole profit/price ledger values a
period's power array as the energy of that period, WITHOUT the 0.25 h DT
factor (matches `eval_agents.compute_agent_profit`, the env reward and the
SOCP objective). New metrics follow the same convention (T_SCALE = 1.0), or
they would not close against the profit layer.
"""

import numpy as np

from models import Agent, MarketConfig

# Period-power accounting convention; matches profit/welfare. Keep at 1.0.
T_SCALE = 1.0
# Wholesale floor (CNY/MWh) for the LMP-markup denominator: periods at or
# below this wholesale carry no meaningful bill weight and are dropped.
WHOLESALE_MIN = 1.0

_SCHED_KEYS = ["p_buy", "p_sell", "p_ch", "p_dis", "served", "unserved",
               "pv_used", "wind_used"]


def _has_load(agent: Agent) -> bool:
    """True when the agent carries a (possibly zero elsewhere) load profile."""
    prof = agent.load_forecast if agent.load_forecast is not None \
        else agent.load_real
    return prof is not None and bool(np.any(np.asarray(prof) > 0))


def load_agents(agents) -> list:
    """Load/consuming agents = those with a load profile.

    Includes storage-owning prosumers (their served load is real consumption);
    excludes pure wind / pure-storage units.
    """
    return [a for a in agents if _has_load(a)]


def _bus_lmp(lmp: np.ndarray, bus: int) -> np.ndarray:
    """Nodal LMP column for a bus (column index == bus id convention)."""
    if lmp.ndim != 2 or lmp.shape[1] <= bus:
        raise ValueError(
            f"lmp shape {lmp.shape} incompatible with bus {bus}; "
            "expected (T, n_buses) with column index == bus id")
    return lmp[:, bus]


def _load_import(sched_a: dict) -> np.ndarray:
    """qbuy[t] = max(served - (pv_used + wind_used), 0).

    Energy a load stream must import from the bus/market after on-site
    generation. Storage discharge (p_dis) does NOT offset the consumer bill
    (its value is settled separately in the storage layer). Storage charging
    (p_ch) never enters qbuy either.
    """
    g = (sched_a["pv_used"] + sched_a["wind_used"])
    return np.maximum(sched_a["served"] - g, 0.0)


def consumer_metrics(sched: dict, lmp: np.ndarray, agents) -> dict:
    """Consumer payment and surplus over load agents for one cleared day.

    Per agent a, per period t:
        qbuy_a[t]   = served - (pv_used + wind_used), floored at 0
        CP_a[t]     = lmp[t, a.bus] * qbuy_a[t]        (bill for load import)
        CS_a[t]     = a.bid_value * served_a[t] - CP_a[t]

    Value uses the TRUTHFUL scalar bid_value (RL never touches bid_value, only
    bid_mult at clearing) so consumer surplus is not contaminated by RL bid
    self-depreciation.

    Returns
    -------
    dict with cp (total), cs (total), cp_by_agent, cs_by_agent, qbuy.
    """
    cp_total = 0.0
    cs_total = 0.0
    cp_by = {}
    cs_by = {}
    qbuy_by = {}
    for a in load_agents(agents):
        sa = sched[a.name]
        qbuy = _load_import(sa)
        lmp_node = _bus_lmp(lmp, a.bus)
        cp_a = float(np.sum(lmp_node * qbuy)) * T_SCALE
        cs_a = float(np.sum(a.bid_value * sa["served"])) * T_SCALE - cp_a
        cp_total += cp_a
        cs_total += cs_a
        cp_by[a.name] = cp_a
        cs_by[a.name] = cs_a
        qbuy_by[a.name] = qbuy
    return {"cp": cp_total, "cs": cs_total,
            "cp_by_agent": cp_by, "cs_by_agent": cs_by, "qbuy": qbuy_by}


def load_payment_weighted_markup(sched: dict, lmp: np.ndarray,
                                 wholesale: np.ndarray, agents,
                                 w_min: float = WHOLESALE_MIN) -> float:
    """One LMP-markup scalar per day, weighted by consumer load-import energy.

    markup = sum_{a,t} (lmp[t,a.bus] - wholesale[t]) * qbuy_a[t]
           / sum_{a,t}  wholesale[t]            * qbuy_a[t]

    computed over periods with wholesale[t] > w_min. This is the load-payment-
    weighted mean of (lmp - wholesale) / wholesale, not a bus-count-weighted
    mean (which tiny buses would dominate). NaN when no qualifying weight.
    """
    num = 0.0
    den = 0.0
    w = np.asarray(wholesale, dtype=float)
    for a in load_agents(agents):
        qbuy = _load_import(sched[a.name])
        lmp_node = _bus_lmp(lmp, a.bus)
        keep = w > w_min
        num += float(np.sum((lmp_node[keep] - w[keep]) * qbuy[keep]))
        den += float(np.sum(w[keep] * qbuy[keep]))
    return num / den if den > 0 else float("nan")


def agent_profit(sched_a: dict, lmp_node: np.ndarray, agent: Agent,
                 config: MarketConfig) -> float:
    """Raw profit over a full day; matches `diagnose_profit.agent_profit` and
    the env reward formula (mkt + cons - gen - pen - cyc)."""
    mkt = float(np.sum(sched_a["p_sell"] * lmp_node
                       - sched_a["p_buy"] * lmp_node))
    cons = float(np.sum(agent.bid_value * sched_a["served"]))
    gen = float(np.sum(agent.offer_cost * (sched_a["pv_used"]
                                           + sched_a["wind_used"])))
    pen = float(config.market_design.penalty_unserved
                * np.sum(sched_a["unserved"]))
    cyc = float(config.storage.cycle_cost
                * np.sum(sched_a["p_ch"] + sched_a["p_dis"]))
    return mkt + cons - gen - pen - cyc


def market_power_split(s_base: dict, s_rl: dict, lmp_base: np.ndarray,
                       lmp_rl: np.ndarray, agents, config) -> list:
    """diagnose_profit-style arb / market-power decomposition per storage agent.

    For each storage agent a (q = p_sell - p_buy; subscript b = truthful
    baseline, r = RL fleet):
        arb   = sum((q_r - q_b) * (lmp_b + lmp_r) / 2)   dispatch re-timing
        power = sum((q_b + q_r) / 2 * (lmp_r - lmp_b))   price movement x position
        dp    = agent_profit(rl) - agent_profit(base)
        other = dp - (arb + power)                        residual
        net   = sum(q_r)                                  net position RL day

    Parity with `diagnose_profit.decompose` is pinned by a unit test.
    """
    rows = []
    for a in agents:
        if a.storage is None:
            continue
        b, r = s_base[a.name], s_rl[a.name]
        lmp_b = _bus_lmp(lmp_base, a.bus)
        lmp_r = _bus_lmp(lmp_rl, a.bus)
        q_b = b["p_sell"] - b["p_buy"]
        q_r = r["p_sell"] - r["p_buy"]
        arb = float(np.sum((q_r - q_b) * (lmp_b + lmp_r) / 2.0))
        power = float(np.sum((q_b + q_r) / 2.0 * (lmp_r - lmp_b)))
        dp = (agent_profit(r, lmp_r, a, config)
              - agent_profit(b, lmp_b, a, config))
        other = dp - (arb + power)
        net = float(np.sum(q_r))
        rows.append({"name": a.name, "profit_delta": dp, "arb": arb,
                     "market_power": power, "other": other, "net": net})
    return rows


def reconciliation(sched: dict, lmp: np.ndarray, wholesale: np.ndarray,
                   agents, slack_bus: int = 0) -> dict:
    """Funds-flow identity check for one cleared day.

    Per period: net meter draw d_a[t] = p_buy - p_sell; import_t = sum_a d_a[t]
    (all external flow enters at the slack). Partition each meter bill lmp*d
    into the consumer bill CP (load-import energy) and the residual business
    cash BC = lmp * (d - qbuy) (net gen/export/other streams):

        sum_a CP_a[t] + sum_a BC_a[t]  ==  wholesale[t] * import_t + rent_t
        rent_t = sum_a lmp[t,a.bus]*d_a[t] - wholesale[t]*import_t

    rent is the merchandising surplus (congestion rent + marginal-loss
    surcharge), >= 0 in an LMP market; it is exactly 0 on a lossless,
    congestion-free day. The per-period identity holds by construction of the
    partition; the informative checks are rent >= ~0 and bill_total positive.

    Returns per-period and daily aggregates plus the identity residual.
    """
    T = lmp.shape[0]
    w = np.asarray(wholesale, dtype=float)
    if w.shape[0] != T:
        raise ValueError(f"wholesale length {w.shape[0]} != lmp T {T}")
    cp_t = np.zeros(T)
    bc_t = np.zeros(T)
    d_sum = np.zeros(T)
    for a in agents:
        sa = sched[a.name]
        d_a = sa["p_buy"] - sa["p_sell"]
        d_sum += d_a
        lmp_node = _bus_lmp(lmp, a.bus)
        if _has_load(a):
            qbuy = _load_import(sa)
            cp_t += lmp_node * qbuy
            bc_t += lmp_node * (d_a - qbuy)
        else:
            bc_t += lmp_node * d_a
    import_t = d_sum
    nodal_bill_t = cp_t + bc_t          # == sum_a lmp*d by partition
    wholesale_bill_t = w * import_t
    rent_t = nodal_bill_t - wholesale_bill_t
    residual = nodal_bill_t - (wholesale_bill_t + rent_t)
    return {"rent": rent_t,
            "rent_total": float(np.sum(rent_t)),
            "bill_total": float(np.sum(wholesale_bill_t)),
            "cp_total": float(np.sum(cp_t)),
            "bc_total": float(np.sum(bc_t)),
            "import_total": float(np.sum(import_t)),
            "max_identity_residual": float(np.max(np.abs(residual))),
            "cp": cp_t, "bc": bc_t, "nodal_bill": nodal_bill_t,
            "wholesale_bill": wholesale_bill_t}
