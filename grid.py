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
_net_cache_cfg: Optional[tuple] = None


def build_base_network(config: MarketConfig) -> pp.pandapowerNet:
    global _net_cache, _net_cache_cfg
    mult = config.network.line_capacity_multiplier
    base_ka = get_default("network.base_ampacity_ka", 0.50)
    min_ka = get_default("network.min_line_ka", 0.15)
    max_ka = get_default("network.max_line_ka", 0.80)
    line_overrides = config.network.line_capacity_overrides or get_default("network.line_capacity_overrides", {})
    cache_key = (mult, base_ka, min_ka, max_ka, tuple(sorted(line_overrides.items())))
    if _net_cache is not None and _net_cache_cfg == cache_key:
        return _net_cache
    net_name = get_default("network.name", "case33bw")
    net = getattr(pn, net_name)()
    # Line ampacity from thermal limit: I_max ∝ sqrt(1/R), i.e. thinner
    # conductor → higher R → lower ampacity. The 1/sqrt(R) form follows
    # from conductor cross-section ∝ 1/R and thermal limit ∝ sqrt(A).
    # Clipped to [min_ka, max_ka] to avoid extremes from near-zero R or
    # very thin lines that might otherwise get unrealistically low ratings.
    for idx in net.line.index:
        r_ohm = net.line.at[idx, 'r_ohm_per_km'] * net.line.at[idx, 'length_km']
        net.line.at[idx, 'max_i_ka'] = float(np.clip(
            base_ka / np.sqrt(max(r_ohm, 0.005)), min_ka, max_ka))
    net.line["max_i_ka"] = net.line["max_i_ka"] * mult
    # Per-line capacity overrides — applied after global multiplier
    for line_idx, ov_mult in line_overrides.items():
        if line_idx in net.line.index:
            net.line.at[line_idx, 'max_i_ka'] *= ov_mult
    _net_cache = net
    _net_cache_cfg = cache_key
    return net  # type: ignore


def _noisy_load(x: np.ndarray, sigma: float = 0.1, seed: int = None) -> np.ndarray:
    """Apply multiplicative Gaussian noise, clipped to non-negative."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    return np.clip(x * (1 + rng.normal(0, sigma, size=x.shape)), 0, None)


def _make_load_profile(
    hours: np.ndarray,
    load_type: str,
    load_cfg: dict,
) -> np.ndarray:
    """Generate type-specific diurnal load profile (per-unit, mean=1.0)."""
    ov = load_cfg.get("type_overrides", {}).get(load_type, {})
    return load_profile(
        hours,
        morning_peak_hour=load_cfg.get("morning_peak_hour", 8.0),
        morning_peak_amplitude=load_cfg.get("morning_peak_amplitude", 0.30),
        morning_peak_width=load_cfg.get("morning_peak_width", 3.0),
        evening_peak_hour=load_cfg.get("evening_peak_hour", 19.0),
        evening_peak_amplitude=load_cfg.get("evening_peak_amplitude", 0.50),
        evening_peak_width=load_cfg.get("evening_peak_width", 4.0),
        night_base=load_cfg.get("night_base", 0.60),
        phase_shift=ov.get("phase_shift_hours", 0.0),
        amplitude_scale=ov.get("amplitude_scale", 1.0),
    )


def create_agents_from_network(
    net: pp.pandapowerNet,
    T: int,
    with_wind: bool = False,
) -> List[Agent]:
    """Create Agent list from network, reading defaults from config/defaults.yaml."""
    cfg = load_defaults()
    hours = np.arange(T)

    # ---- Profile generators (local wrappers reading config defaults) ----
    pv_cfg = cfg["profiles"]["pv"]
    _pv_base = pv_profile(hours,
                          amplitude=pv_cfg.get("amplitude", 1.0),
                          cloud_prob=pv_cfg.get("cloud_prob", 0.25),
                          cloud_attenuation_min=pv_cfg.get("cloud_attenuation_min", 0.2),
                          cloud_attenuation_max=pv_cfg.get("cloud_attenuation_max", 0.6),
                          cloud_persistence=pv_cfg.get("cloud_persistence", 0.70),
                          cloud_seed=pv_cfg.get("cloud_seed", 99),
                          dt_h=1.0)

    wind_cfg = cfg["profiles"]["wind"]
    wind_base_seed = wind_cfg.get("seed", 42)
    ar_coef = wind_cfg.get("ar_coef", 0.90)
    noise_std = wind_cfg.get("noise_std", 0.15)
    wind_peak_hour = wind_cfg.get("peak_hour", 3.0)

    load_cfg = cfg["profiles"]["load"]
    load_fcast_sigma = load_cfg.get("forecast_noise_sigma", 0.06)
    load_real_sigma = load_cfg.get("real_noise_sigma", 0.12)

    # ---- Build bus → load_type map from config ----
    bus_to_type: Dict[int, str] = {}
    for lt_name, lt_cfg in cfg.get("load_types", {}).items():
        for bus in lt_cfg.get("buses", []):
            bus_to_type[bus] = lt_name

    # ---- Build prosumer bus sets ----
    prosumer_buses: Dict[str, set] = {}
    for lt_name, ps_cfg in cfg.get("prosumers", {}).items():
        prosumer_buses[lt_name] = set(ps_cfg.get("prosumer_buses", []))

    # Non-prosumer standalone storage config
    nps_cfg = cfg.get("non_prosumer_storage", {})
    nps_buses = set(nps_cfg.get("buses", []))

    agents: List[Agent] = []

    for idx, load_row in net.load.iterrows():
        bus = int(load_row.bus)
        p_mw = float(load_row.p_mw)

        load_type = bus_to_type.get(bus, "industrial")
        lt_cfg = get_load_type_cfg(load_type)
        base_load = p_mw * lt_cfg.get("load_multiplier", 1.0)
        diurnal = _make_load_profile(hours, load_type, load_cfg)

        load_forecast = _noisy_load(diurnal * base_load, load_fcast_sigma, seed=bus * 10 + 0)
        load_real = _noisy_load(diurnal * base_load, load_real_sigma, seed=bus * 10 + 1)

        is_prosumer = bus in prosumer_buses.get(load_type, set())
        pv_cap_installed = 0.0
        wind_cap_installed = 0.0
        if is_prosumer and load_type == "industrial" and not with_wind:
            continue  # industrial prosumer needs with_wind flag

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
                # Per-bus capacity scale from YAML override
                pv_scale = ps_cfg.get("bus_overrides", {}).get(bus, {}).get("pv_capacity_scale", 1.0)
                pv_cap *= pv_scale
                pv_cap_installed = pv_cap
                pv_forecast = _noisy_load(_pv_base * pv_cap, pv_fcast_noise, seed=bus * 10 + 2)
                pv_real = _noisy_load(_pv_base * pv_cap, pv_real_noise, seed=bus * 10 + 3)

            elif load_type == "industrial" and with_wind:
                wind_cap = base_load * ps_cfg.get("wind_capacity_factor", 4.5)
                wind_fcast_noise = ps_cfg.get("wind_forecast_noise", 0.12)
                wind_real_noise = ps_cfg.get("wind_real_noise", 0.18)
                # Per-bus capacity scale from YAML override
                wind_scale = ps_cfg.get("bus_overrides", {}).get(bus, {}).get("wind_capacity_scale", 1.0)
                wind_cap *= wind_scale
                wind_cap_installed = wind_cap
                # Per-bus wind profile for spatial diversity
                bus_wind = wind_profile(hours, seed=wind_base_seed + bus,
                                        ar_coef=ar_coef, noise_std=noise_std,
                                        peak_hour=wind_peak_hour, dt_h=1.0)
                wind_forecast = _noisy_load(bus_wind * wind_cap, wind_fcast_noise, seed=bus * 10 + 4)
                wind_real = _noisy_load(bus_wind * wind_cap, wind_real_noise, seed=bus * 10 + 5)

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
                soc_min=st_cfg.get("soc_min", 0.10),
                soc_max=st_cfg.get("soc_max", 1.0),
            )

            # Apply per-bus overrides from YAML config
            bus_ov = ps_cfg.get("bus_overrides", {}).get(bus, {})
            st_ov = bus_ov.get("storage", {})
            for attr in ("soc0", "soc_min", "soc_max", "eta_ch", "eta_dis"):
                if attr in st_ov:
                    setattr(storage, attr, st_ov[attr])
            pr_mult = st_ov.get("power_ratio_mult", 1.0)
            if pr_mult != 1.0:
                storage.p_ch_max = storage_power * pr_mult
                storage.p_dis_max = storage_power * pr_mult

            # Assign PV to industrial prosumer buses that have pv:true in override
            if load_type == "industrial" and bus_ov.get("pv", False):
                pv_cap = base_load * ps_cfg.get("pv_capacity_factor", 2.5)
                pv_fcast_noise = ps_cfg.get("pv_forecast_noise", 0.15)
                pv_real_noise = ps_cfg.get("pv_real_noise", 0.20)
                pv_scale = bus_ov.get("pv_capacity_scale", 1.0)
                pv_cap *= pv_scale
                pv_cap_installed = pv_cap
                pv_forecast = _noisy_load(_pv_base * pv_cap, pv_fcast_noise, seed=bus * 10 + 2)
                pv_real = _noisy_load(_pv_base * pv_cap, pv_real_noise, seed=bus * 10 + 3)

            bid_val = ps_cfg.get("bid_value", 350.0)
            offer_cost = ps_cfg.get("offer_cost", 180.0)
        else:
            bid_val = lt_cfg.get("bid_value", 580.0)
            offer_cost = 999.0
            # Attach standalone storage to selected non-prosumer nodes
            if bus in nps_buses and nps_cfg:
                nps_storage = StorageSpec(
                    e_max=base_load * nps_cfg.get("capacity_factor", 2.5),
                    p_ch_max=base_load * nps_cfg.get("capacity_factor", 2.5)
                              * nps_cfg.get("power_ratio", 0.30),
                    p_dis_max=base_load * nps_cfg.get("capacity_factor", 2.5)
                               * nps_cfg.get("power_ratio", 0.30),
                    eta_ch=nps_cfg.get("eta_ch", 0.92),
                    eta_dis=nps_cfg.get("eta_dis", 0.92),
                    soc0=nps_cfg.get("soc0", 0.5),
                    soc_min=nps_cfg.get("soc_min", 0.10),
                    soc_max=nps_cfg.get("soc_max", 0.95),
                )
                storage = nps_storage

        agent = Agent(
            name=f"Bus{bus}{load_type[0].upper()}",
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
            pv_capacity=pv_cap_installed,
            wind_capacity=wind_cap_installed,
        )
        agents.append(agent)

    return agents


def day_ahead_price_china(T: int = 96, agents=None, config=None) -> np.ndarray:
    """Generate day-ahead price curve.

    Routes to the configured forecast method (defaults.yaml price_curve.forecast_method):
      - "merit_order": builds supply/demand stacks from agent fundamentals, so
        the price reflects actual net-load conditions (tight supply → high price,
        excess RE → low / negative price). Requires *agents* and *config*.
      - "synthetic": legacy sinusoidal model — no agents needed.
    """
    cfg = get_default("price_curve", {})
    method = cfg.get("forecast_method", "synthetic")

    if method == "merit_order" and agents is not None and len(agents) > 0:
        return _forecast_price_merit_order(agents, T, config, cfg)
    else:
        return _forecast_price_synthetic(T, cfg)


def _forecast_price_synthetic(T: int, cfg: dict) -> np.ndarray:
    """Legacy sinusoidal day-ahead price model (no agents needed)."""
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

    spike_prob = cfg.get("spike_probability", 0.05)
    spike_mag = cfg.get("spike_magnitude", 300.0)
    if spike_prob > 0:
        spike_mask = np.random.random(T) < spike_prob
        spike_signs = np.random.choice([-1, 1], size=T)
        price += spike_mask * spike_signs * spike_mag

    price = np.clip(price, hard_min, hard_max)
    return price


def _forecast_price_merit_order(agents, T: int, config, cfg: dict) -> np.ndarray:
    """Build supply/demand stacks from agent fundamentals for each period.

    Approach (inspired by ASSUME's calculate_naive_price):
      1. Compute total demand per period (sum of load forecasts).
      2. Compute total renewable generation per period (PV + wind forecasts).
      3. Net load = demand - RE generation.
      4. Price = base_price * (1 + elasticity * net_load / max_demand).

    When net load is high (tight supply, heavy grid imports), prices rise above
    base.  When net load is negative (RE surplus, exports), prices fall and can
    go negative.  This directly ties the price forecast to physical fundamentals
    instead of a static sinusoidal template.
    """
    base = cfg.get("base", 420.0)
    elasticity = cfg.get("merit_order_price_elasticity", 0.35)
    floor = cfg.get("merit_order_min_price", 50.0)
    cap = cfg.get("merit_order_max_price", 900.0)
    noise_sigma = cfg.get("noise_sigma", 30.0)
    hard_min = cfg.get("hard_min", -200.0)
    hard_max = cfg.get("hard_max", 1200.0)

    demand = np.zeros(T)
    re_gen = np.zeros(T)

    for a in agents:
        # Use forecasts (DA perspective) or real values (RT perspective)
        load = a.load_forecast if config is None or not hasattr(config, 'use_real') else a.load_real
        demand += load
        re_gen += a.pv_forecast
        if a.has_wind and a.wind_forecast is not None:
            re_gen += a.wind_forecast

    # Avoid division by zero
    max_demand = float(np.max(demand)) if np.max(demand) > 0 else 1.0
    net_load = demand - re_gen

    # Price = base * (1 + elasticity * net_load / max_demand)
    #  net_load = +1 * max_demand → price = base * (1 + elasticity)  [high]
    #  net_load = 0              → price = base                      [balanced]
    #  net_load = -max_demand    → price = base * (1 - elasticity)  [low]
    price = base * (1.0 + elasticity * net_load / max_demand)
    price = np.clip(price, floor, cap)

    if noise_sigma > 0:
        price = price + np.random.normal(0, noise_sigma, size=T)

    spike_prob = cfg.get("spike_probability", 0.0)
    spike_mag = cfg.get("spike_magnitude", 300.0)
    if spike_prob > 0:
        spike_mask = np.random.random(T) < spike_prob
        spike_signs = np.random.choice([-1, 1], size=T)
        price += spike_mask * spike_signs * spike_mag

    price = np.clip(price, hard_min, hard_max)
    return price


# ---------------------------------------------------------------------------
# Shared profile generators (used by llm.py and create_agents_from_network)
# ---------------------------------------------------------------------------

def load_profile(hours: np.ndarray,
                 morning_peak_hour: float = 8.0, morning_peak_amplitude: float = 0.30,
                 morning_peak_width: float = 2.0,
                 evening_peak_hour: float = 19.0, evening_peak_amplitude: float = 0.50,
                 evening_peak_width: float = 2.5, night_base: float = 0.60,
                 phase_shift: float = 0.0, amplitude_scale: float = 1.0) -> np.ndarray:
    """Dual-Gaussian diurnal load profile, normalized to mean=1.0.

    Accepts either decimal hours (0..24) or integer periods (0..95 at 15-min
    resolution). Auto-detects and converts internally.

    amplitude_scale blends between flat (0.0) and full diurnal shape (1.0).
    """
    max_h = float(np.max(hours))
    if max_h > 30:
        h = hours * 0.25
    else:
        h = hours
    # phase_shift > 0 → peak occurs LATER (matching natural intuition)
    morning = morning_peak_amplitude * np.exp(
        -((h - (morning_peak_hour + phase_shift)) ** 2) / morning_peak_width)
    evening = evening_peak_amplitude * np.exp(
        -((h - (evening_peak_hour + phase_shift)) ** 2) / evening_peak_width)
    raw = night_base + morning + evening
    norm = raw / np.mean(raw)
    return norm * amplitude_scale + (1.0 - amplitude_scale)


def pv_profile(hours: np.ndarray, amplitude: float = 1.0,
               cloud_prob: float = 0.15, cloud_attenuation_min: float = 0.55,
               cloud_attenuation_max: float = 0.85, cloud_persistence: float = 0.85,
               cloud_seed: int = 42, dt_h: float = 0.25) -> np.ndarray:
    """Synthetic PV profile with clear-sky envelope and smooth cloud attenuation.

    Accepts either decimal hours (0..24) or integer periods (0..95 at 15-min
    resolution). Auto-detects: if max(hours) > 30, treats as integer periods
    and converts to decimal hours internally.

    dt_h controls the native modelling time step in hours. When dt_h > 0.25,
    the AR(1) cloud process runs at the coarser resolution, then the result is
    linearly interpolated back to the original length.

    Cloud cover is modelled as a continuous AR(1) latent process mapped through
    a sigmoid to [cloud_attenuation_min, 1.0] so irradiance never drops to zero
    during daytime.
    """
    T = len(hours)
    max_h = float(np.max(hours))
    if max_h > 30:
        hours_dec = hours * 0.25
    else:
        hours_dec = hours

    # If modelling at coarser resolution, generate coarse then interpolate
    if dt_h > 0.25:
        T_coarse = int(24 / dt_h)
        hours_coarse = np.arange(T_coarse) * dt_h
        coarse = pv_profile(hours_coarse, amplitude=amplitude,
                            cloud_prob=cloud_prob,
                            cloud_attenuation_min=cloud_attenuation_min,
                            cloud_attenuation_max=cloud_attenuation_max,
                            cloud_persistence=cloud_persistence,
                            cloud_seed=cloud_seed, dt_h=0.25)
        x_coarse = hours_coarse
        x_fine = hours_dec
        return np.interp(x_fine, x_coarse, coarse)

    # Clear-sky envelope: clipped sine peaking at 12:00 (solar noon)
    clear_sky = np.clip(amplitude * np.sin((hours_dec - 6) / 24 * 2 * np.pi), 0, None)
    daytime = clear_sky > 1e-6

    # AR(1) latent process with temporal correlation
    rng = np.random.RandomState(cloud_seed)
    innovations = rng.normal(0, 1, size=T)
    latent = np.zeros(T)
    latent[0] = innovations[0]
    for t in range(1, T):
        latent[t] = (cloud_persistence * latent[t - 1]
                     + np.sqrt(1 - cloud_persistence ** 2) * innovations[t])

    # Map latent through sigmoid scaled to [cloud_attenuation_min, 1.0]
    # Shift so that ~cloud_prob fraction of daytime periods get attenuation
    if daytime.sum() > 0:
        threshold = np.percentile(latent[daytime], 100 * cloud_prob)
    else:
        threshold = 0.0
    atten = np.ones(T)
    for t in range(T):
        if daytime[t]:
            raw = 1.0 / (1.0 + np.exp(-3.0 * (latent[t] - threshold)))
            atten[t] = cloud_attenuation_min + raw * (1.0 - cloud_attenuation_min)

    return clear_sky * atten


def wind_profile(hours: np.ndarray, seed: int = 42,
                 ar_coef: float = 0.88, noise_std: float = 0.18,
                 peak_hour: float = 3.0, floor: float = 0.0,
                 dt_h: float = 0.25) -> np.ndarray:
    """Synthetic wind generation with AR(1) temporal correlation.

    Accepts either decimal hours (0..24) or integer periods (0..95 at 15-min
    resolution). Diurnal baseline peaks at peak_hour (default 3:00 AM for
    complementarity with solar PV).

    dt_h controls the native modelling time step in hours. When dt_h > 0.25,
    the AR(1) process runs at the coarser resolution, then the result is
    linearly interpolated back to the original length.

    Each bus can receive a different seed for spatial diversity.
    """
    T = len(hours)
    max_h = float(np.max(hours))
    if max_h > 30:
        hours_dec = hours * 0.25
    else:
        hours_dec = hours

    # If modelling at coarser resolution, generate coarse then interpolate
    if dt_h > 0.25:
        T_coarse = int(24 / dt_h)
        hours_coarse = np.arange(T_coarse) * dt_h
        coarse = wind_profile(hours_coarse, seed=seed,
                              ar_coef=ar_coef, noise_std=noise_std,
                              peak_hour=peak_hour, floor=floor, dt_h=0.25)
        x_coarse = hours_coarse
        x_fine = hours_dec
        return np.interp(x_fine, x_coarse, coarse)

    # Diurnal: peak at peak_hour, trough 12h later (complementary to PV)
    diurnal = 0.55 + 0.35 * np.sin((hours_dec - peak_hour + 6) / 24 * 2 * np.pi)

    # AR(1) process — preserves temporal structure across periods
    rng = np.random.RandomState(seed)
    innovations = rng.normal(0, noise_std, size=T)
    ar_component = np.zeros(T)
    ar_component[0] = innovations[0]
    for t in range(1, T):
        ar_component[t] = (ar_coef * ar_component[t - 1]
                           + np.sqrt(1 - ar_coef ** 2) * innovations[t])

    w = diurnal + ar_component
    w = np.clip(w, floor, 1.2)
    return w
