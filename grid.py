"""grid.py
Network topology builder, agent factory, and day-ahead price curve generator.
Reads default parameters from config/defaults.yaml; accepts optional overrides.
"""
# Pre-load ortools to work around DLL load order issue in pandapower
from ortools.linear_solver import pywraplp  # noqa: F401

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from typing import Any, Dict, List, Optional

from models import MarketConfig, StorageSpec, Agent
from config_loader import (
    get_default, get_prosumer_cfg, get_load_type_cfg, load_defaults,
)


_net_cache: Optional[pp.pandapowerNet] = None
_net_cache_mult: Optional[float] = None


def build_base_network(config: MarketConfig) -> pp.pandapowerNet:
    global _net_cache, _net_cache_mult
    mult = config.line_capacity_multiplier
    if _net_cache is not None and _net_cache_mult == mult:
        return _net_cache
    net_name = get_default("network.name", "case33bw")
    net = getattr(pn, net_name)()
    net.line["max_i_ka"] = net.line["max_i_ka"].fillna(1.0) * mult  # type: ignore[union-attr]
    _net_cache = net
    _net_cache_mult = mult
    return net  # type: ignore


def create_agents_from_network(
    net: pp.pandapowerNet,
    T: int,
    with_wind: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Agent]:
    """Create Agent list from network, reading defaults from config/defaults.yaml.

    Args:
        net: pandapower network
        T: number of time periods
        with_wind: enable wind generation for eligible prosumers
        overrides: optional dict to override config values (not yet used;
                   reserved for runtime customization)

    Returns:
        List of Agent objects
    """
    cfg = load_defaults()
    hours = np.arange(T)

    # ---- Profile generators (local wrappers reading config defaults) ----
    pv_cfg = cfg["profiles"]["pv"]
    _pv_base = pv_profile(hours,
                          amplitude=pv_cfg.get("amplitude", 1.0))

    wind_cfg = cfg["profiles"]["wind"]
    _wind_base = wind_profile(hours,
                              seed=wind_cfg.get("seed", 42)) if with_wind else None

    load_cfg = cfg["profiles"]["load"]
    load_fcast_sigma = load_cfg.get("forecast_noise_sigma", 0.06)
    load_real_sigma = load_cfg.get("real_noise_sigma", 0.12)

    def noisy(x, sigma=0.1):
        return np.clip(x * (1 + np.random.normal(0, sigma, size=x.shape)), 0, None)

    # ---- Build bus → load_type map from config ----
    bus_to_type: Dict[int, str] = {}
    for lt_name, lt_cfg in cfg.get("load_types", {}).items():
        for bus in lt_cfg.get("buses", []):
            bus_to_type[bus] = lt_name

    # ---- Build prosumer bus sets ----
    prosumer_buses: Dict[str, set] = {}
    for lt_name, ps_cfg in cfg.get("prosumers", {}).items():
        prosumer_buses[lt_name] = set(ps_cfg.get("prosumer_buses", []))

    agents: List[Agent] = []

    for idx, load_row in net.load.iterrows():
        bus = int(load_row.bus)
        p_mw = float(load_row.p_mw)

        load_type = bus_to_type.get(bus, "industrial")
        lt_cfg = get_load_type_cfg(load_type)
        base_load = p_mw * lt_cfg.get("load_multiplier", 1.0)

        load_forecast = noisy(np.full(T, base_load), load_fcast_sigma)
        load_real = noisy(np.full(T, base_load), load_real_sigma)

        is_prosumer = bus in prosumer_buses.get(load_type, set())
        if is_prosumer and load_type == "industrial" and not with_wind:
            pass  # industrial prosumer needs with_wind flag

        pv_forecast = np.zeros(T)
        pv_real = np.zeros(T)
        wind_forecast: Optional[np.ndarray] = None
        wind_real: Optional[np.ndarray] = None
        storage: Optional[StorageSpec] = None

        if is_prosumer:
            ps_cfg = get_prosumer_cfg(load_type)

            if load_type == "residential":
                pv_cap = base_load * ps_cfg.get("pv_capacity_factor", 3.8)
                pv_fcast_noise = ps_cfg.get("pv_forecast_noise", 0.15)
                pv_real_noise = ps_cfg.get("pv_real_noise", 0.20)
                pv_forecast = noisy(_pv_base * pv_cap, pv_fcast_noise)
                pv_real = noisy(_pv_base * pv_cap, pv_real_noise)

            elif load_type == "industrial" and with_wind:
                wind_cap = base_load * ps_cfg.get("wind_capacity_factor", 4.5)
                wind_fcast_noise = ps_cfg.get("wind_forecast_noise", 0.12)
                wind_real_noise = ps_cfg.get("wind_real_noise", 0.18)
                if _wind_base is not None:
                    wind_forecast = noisy(_wind_base * wind_cap, wind_fcast_noise)
                    wind_real = noisy(_wind_base * wind_cap, wind_real_noise)
                else:
                    wind_forecast = np.zeros(T)
                    wind_real = np.zeros(T)

            st_cfg = ps_cfg.get("storage", {})
            storage_capacity = base_load * st_cfg.get("capacity_factor", 2.0)
            storage_power = storage_capacity * st_cfg.get("power_ratio", 0.2)
            storage = StorageSpec(
                e_max=storage_capacity,
                p_ch_max=storage_power,
                p_dis_max=storage_power,
                eta_ch=st_cfg.get("eta_ch", 0.93),
                eta_dis=st_cfg.get("eta_dis", 0.93),
                soc0=st_cfg.get("soc0", 0.5),
                soc_min=st_cfg.get("soc_min", 0.0),
                soc_max=st_cfg.get("soc_max", 1.0),
            )

            bid_val = ps_cfg.get("bid_value", 350.0)
            offer_cost = ps_cfg.get("offer_cost", 180.0)
        else:
            bid_val = lt_cfg.get("bid_value", 580.0)
            offer_cost = 999.0

        agent = Agent(
            name=f"bus{bus}{load_type.capitalize()[:3]}{idx}",
            bus=bus,
            is_prosumer=is_prosumer,
            load_forecast=load_forecast,
            pv_forecast=pv_forecast,
            load_real=load_real,
            pv_real=pv_real,
            bid_value=bid_val,
            offer_cost=offer_cost,
            wind_forecast=wind_forecast,
            wind_real=wind_real,
            storage=storage,
            load_type=load_type,
        )
        agents.append(agent)

    return agents


def day_ahead_price_china(T: int = 96) -> np.ndarray:
    """Generate synthetic day-ahead price curve, reading params from config."""
    cfg = get_default("price_curve", {})
    hours = np.arange(T) * 0.25
    base = cfg.get("base", 420.0)
    amp1 = cfg.get("amplitude_1", 300.0)
    phase1 = cfg.get("phase_1_hours", -9)
    amp2 = cfg.get("amplitude_2", 150.0)
    phase2 = cfg.get("phase_2_hours", -20)
    price = base + amp1 * np.sin((hours + phase1) / 24 * 2 * np.pi) \
            + amp2 * np.sin((hours + phase2) / 24 * 2 * np.pi)

    dip = cfg.get("noon_dip_amplitude", -80.0)
    dip_center = cfg.get("noon_dip_center", 13)
    dip_width = cfg.get("noon_dip_width", 8)
    if dip != 0:
        price += dip * np.exp(-((hours - dip_center) ** 2) / dip_width)

    min_p = cfg.get("min_price", 150.0)
    max_p = cfg.get("max_price", 800.0)
    noise_sigma = cfg.get("noise_sigma", 30.0)
    hard_min = cfg.get("hard_min", -50.0)
    hard_max = cfg.get("hard_max", 1200.0)

    price = np.clip(price, min_p, max_p)
    price = price + np.random.normal(0, noise_sigma, size=T)
    price = np.clip(price, hard_min, hard_max)
    return price


# ---------------------------------------------------------------------------
# Shared profile generators (used by llm.py and create_agents_from_network)
# ---------------------------------------------------------------------------

def pv_profile(hours: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    """Generate synthetic PV generation profile (per-unit, sine-based)."""
    return np.clip(amplitude * np.sin((hours - 6) / 24 * 2 * np.pi), 0, None)


def wind_profile(hours: np.ndarray, seed: int = 42) -> np.ndarray:
    """Generate synthetic wind generation profile (per-unit, composite sine + noise)."""
    rng = np.random.RandomState(seed)
    w = np.clip(
        0.7 * (0.5 + 0.5 * np.sin((hours - 3) / 12 * np.pi))
        + 0.2 * np.sin((hours - 14) / 8 * np.pi)
        + rng.normal(0, 0.12, size=len(hours)),
        0, 1.3,
    )
    return np.maximum(w, 0.05)
