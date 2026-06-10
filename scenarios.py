# scenarios.py
import numpy as np
from typing import List, Tuple
from models import MarketConfig, Agent
from grid import build_base_network, create_agents_from_network, day_ahead_price_china
from config_loader import get_scenario_cfg


def _build_base(config: MarketConfig, T: int, with_wind: bool) -> Tuple[List[Agent], np.ndarray]:
    net = build_base_network(config)
    agents = create_agents_from_network(net, T, with_wind=with_wind)
    wholesale = day_ahead_price_china(T)
    return agents, wholesale


# ======================== Core scenario functions ========================

def scenario_baseline(T: int = 96) -> Tuple[List[Agent], np.ndarray]:
    config = MarketConfig(use_ac_opf=False)
    return _build_base(config, T, with_wind=True)


def scenario_high_re(T: int = 96) -> Tuple[List[Agent], np.ndarray]:
    cfg = get_scenario_cfg("high_re")
    multipliers = cfg.get("multipliers", {})
    pv_mult = multipliers.get("pv", 2.0)
    wind_mult = multipliers.get("wind", 2.0)
    config = MarketConfig(use_ac_opf=False)
    agents, wholesale = _build_base(config, T, with_wind=True)
    for a in agents:
        if a.is_prosumer:
            a.pv_forecast = a.pv_forecast * pv_mult
            a.pv_real = a.pv_real * pv_mult
        if a.has_wind:
            a.wind_forecast = a.wind_forecast * wind_mult  # type: ignore[operator]
            a.wind_real = a.wind_real * wind_mult  # type: ignore[operator]
    return agents, wholesale


def scenario_peak_load(T: int = 96) -> Tuple[List[Agent], np.ndarray]:
    cfg = get_scenario_cfg("peak_load")
    load_mult = cfg.get("multipliers", {}).get("load", 1.8)
    config = MarketConfig(use_ac_opf=False)
    agents, wholesale = _build_base(config, T, with_wind=True)
    for a in agents:
        a.load_forecast = a.load_forecast * load_mult
        a.load_real = a.load_real * load_mult
    return agents, wholesale


def scenario_congestion(T: int = 96) -> Tuple[List[Agent], np.ndarray]:
    cfg = get_scenario_cfg("congestion")
    line_mult = cfg.get("line_capacity_multiplier", 0.5)
    config = MarketConfig(use_ac_opf=False, line_capacity_multiplier=line_mult)
    return _build_base(config, T, with_wind=True)


def scenario_re_ramp_drop(T: int = 96) -> Tuple[List[Agent], np.ndarray]:
    cfg = get_scenario_cfg("re_ramp_drop")
    return _scenario_re_ramp_event(T, cfg.get("ramp_type", "sudden_drop"),
                                   cfg.get("ramp_multiplier", 0.1))


def scenario_re_ramp_surge(T: int = 96) -> Tuple[List[Agent], np.ndarray]:
    cfg = get_scenario_cfg("re_ramp_surge")
    return _scenario_re_ramp_event(T, cfg.get("ramp_type", "sudden_surge"))


def _scenario_re_ramp_event(T: int, ramp_type: str,
                            ramp_mult: float = 0.1) -> Tuple[List[Agent], np.ndarray]:
    config = MarketConfig(use_ac_opf=False)
    agents, wholesale = _build_base(config, T, with_wind=True)
    for a in agents:
        if not (a.is_prosumer or a.has_wind):
            continue
        pv_orig = a.pv_forecast.copy()
        wind_orig = a.wind_forecast.copy() if a.has_wind and a.wind_forecast is not None else np.zeros(T)
        pv_new = np.zeros(T)
        wind_new = np.zeros(T)
        mid = T // 2
        if ramp_type == "sudden_drop":
            pv_new[:mid] = pv_orig[:mid]
            pv_new[mid:] = pv_orig[mid:] * ramp_mult
            if a.has_wind:
                wind_new[:mid] = wind_orig[:mid]
                wind_new[mid:] = wind_orig[mid:] * ramp_mult
        else:  # sudden_surge
            pv_new[:mid] = pv_orig[:mid] * ramp_mult
            pv_new[mid:] = pv_orig[mid:]
            if a.has_wind:
                wind_new[:mid] = wind_orig[:mid] * ramp_mult
                wind_new[mid:] = wind_orig[mid:]
        a.pv_forecast = pv_new
        a.pv_real = pv_new * 0.95
        if a.has_wind:
            a.wind_forecast = wind_new
            a.wind_real = wind_new * 0.95
    return agents, wholesale


# ======================== Scenario registry ========================
SCENARIO_REGISTRY = {
    "baseline":      scenario_baseline,
    "high_re":       scenario_high_re,
    "peak_load":     scenario_peak_load,
    "congestion":    scenario_congestion,
    "re_ramp_drop":  scenario_re_ramp_drop,
    "re_ramp_surge": scenario_re_ramp_surge,
}


def get_scenario(name: str, T: int = 96) -> Tuple[List[Agent], np.ndarray]:
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(SCENARIO_REGISTRY.keys())}")
    return SCENARIO_REGISTRY[name](T=T)


def list_scenarios() -> List[str]:
    return list(SCENARIO_REGISTRY.keys())


def print_scenario_info():
    """Print all available scenarios with descriptions from config."""
    print("=" * 60)
    print("Available scenarios")
    print("=" * 60)
    for key in SCENARIO_REGISTRY:
        cfg = get_scenario_cfg(key)
        desc = cfg.get("description", "") if cfg else ""
        print(f"  {key:<20s}: {desc}")
    print("=" * 60)
