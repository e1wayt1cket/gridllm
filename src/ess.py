# ess.py
"""Independent storage participants: a battery alone.

The first case study. An independent storage unit has no load to serve and no
on-site generation, so nothing it earns can be a consumption value: its payoff
is energy sold minus energy bought at the nodal price, less degradation (see
`participant_payoff.IndependentStoragePayoff`).

It takes part in the market the same way every other participant does. It
declares what it will pay to charge and what it asks to discharge; the clearing
decides whether to take either, and settlement happens at the nodal price. The
declared prices are anchored to an economically meaningful level rather than
inherited from the local load type, whose willingness to pay for consumption has
no meaning for a battery. That level is a margin on the unit's own degradation
cost, not the market price: the clearing costs charging at the declared bid, so
a unit cycles only when the day's spread covers both quotes and the round trip,
and an anchor set to the mean wholesale price demands a spread no day offers.
See `StorageConfig.quote_anchor`.
"""

from typing import List, Optional

import numpy as np

from config_loader import get_default
from models import Agent, MarketConfig, StorageSpec


def independent_storage_config() -> dict:
    """The `independent_storage` block from config/defaults.yaml."""
    return get_default("independent_storage", {}) or {}


def build_ess_fleet(T: int, config: MarketConfig,
                    wholesale: Optional[np.ndarray] = None,
                    buses: Optional[List[int]] = None,
                    scale: float = 1.0) -> List[Agent]:
    """Construct the independent storage fleet for one scenario.

    Parameters
    ----------
    T : int
        Horizon in periods; the units carry zero load and zero generation
        profiles of this length.
    config : MarketConfig
        Supplies the price anchor (``storage.bid_anchor``) and the enabled flag
        is read from YAML.
    wholesale : np.ndarray, optional
        The scenario's price curve. Used for the anchor when
        ``storage.bid_anchor`` is unset, matching what the clearing optimizers
        fall back to for terminal SOC value.
    buses : list of int, optional
        Override the configured placement, for the node-location experiment.
    scale : float
        Fleet-wide size multiplier; the round-one ablation knob.

    Returns
    -------
    list of Agent, empty when the fleet is disabled.
    """
    cfg = independent_storage_config()
    if not cfg.get("enabled", False):
        return []

    placements = list(buses if buses is not None
                      else cfg.get("buses", []))
    if not placements:
        return []

    capacity = float(cfg.get("capacity_mwh", 0.0)) * float(scale)
    power_ratio = float(cfg.get("power_ratio", 0.4))
    power = capacity * power_ratio

    anchor = config.storage.quote_anchor()

    zeros = np.zeros(T)
    fleet = []
    for bus in placements:
        storage = StorageSpec(
            e_max=capacity,
            p_ch_max=power,
            p_dis_max=power,
            eta_ch=float(cfg.get("eta_ch", 0.92)),
            eta_dis=float(cfg.get("eta_dis", 0.92)),
            soc0=float(cfg.get("soc0", 0.5)),
            soc_min=float(cfg.get("soc_min", 0.10)),
            soc_max=float(cfg.get("soc_max", 0.90)),
        )
        fleet.append(Agent(
            name=f"ESS{bus}",
            bus=int(bus),
            is_prosumer=False,
            load_forecast=zeros.copy(),
            load_real=zeros.copy(),
            pv_forecast=zeros.copy(),
            pv_real=zeros.copy(),
            wind_forecast=None,
            wind_real=None,
            bid_value=float(anchor),
            offer_cost=float(anchor),
            storage=storage,
            load_type="storage",
            pv_capacity=0.0,
            wind_capacity=0.0,
            participant_type="storage",
        ))
    return fleet
