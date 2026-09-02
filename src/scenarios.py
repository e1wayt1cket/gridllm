# scenarios.py
import numpy as np
from typing import List, Tuple
from models import MarketConfig, Agent
from grid import build_base_network, create_agents_from_network, day_ahead_price_china
from config_loader import get_scenario_cfg


def _build_scenario(name: str, T: int, config: MarketConfig = None) -> Tuple[List[Agent], np.ndarray]:
    """Generic scenario builder driven by config/scenarios.yaml.

    Supported YAML keys per scenario:
      - with_wind: bool (default True)
      - multipliers: {pv, wind, load, storage} -> float factor applied to agent forecasts
      - override_config: dict of MarketConfig field overrides; may also carry
        custom_loads: [{buses: [int], factor: float}] for per-bus load scaling
      - re_ramp: {type: sudden_drop|sudden_surge, multiplier: float}
        Transforms PV/wind forecasts at midpoint by the given multiplier.

    If *config* is provided, scenario override_config is applied to it in-place
    so callers (clear_market, Nash, etc.) use the same parameters as the network.
    """
    cfg = get_scenario_cfg(name)
    if not cfg:
        raise ValueError(f"Unknown scenario '{name}'")

    with_wind = cfg.get("with_wind", True)
    multipliers = cfg.get("multipliers", {})
    override = dict(cfg.get("override_config", {}))
    # custom_loads is a per-bus load scaling directive applied after agent
    # creation; take it out so it never reaches MarketConfig construction
    # (an unknown key would raise TypeError on the config=None path). Copy
    # the dict first because get_scenario_cfg returns cached YAML state.
    custom_loads = override.pop("custom_loads", [])
    re_ramp = cfg.get("re_ramp", None)

    if config is not None:
        for k, v in override.items():
            # Support dotted keys like "network.line_capacity_multiplier"
            if "." in k:
                parts = k.split(".")
                obj = config
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                setattr(obj, parts[-1], v)
            else:
                setattr(config, k, v)
    else:
        # Dotted keys in override are resolved into nested dataclass instances
        net_kwargs = {}
        stor_kwargs = {}
        md_kwargs = {}
        rt_kwargs = {}
        top_kwargs = {}
        for k, v in override.items():
            if k.startswith("network."):
                net_kwargs[k.split(".", 1)[1]] = v
            elif k.startswith("storage."):
                stor_kwargs[k.split(".", 1)[1]] = v
            elif k.startswith("market_design."):
                md_kwargs[k.split(".", 1)[1]] = v
            elif k.startswith("rt."):
                rt_kwargs[k.split(".", 1)[1]] = v
            else:
                top_kwargs[k] = v
        from models import NetworkConfig, StorageConfig, MarketDesignConfig, RTConfig
        config = MarketConfig(
            network=NetworkConfig(**net_kwargs) if net_kwargs else NetworkConfig(),
            storage=StorageConfig(**stor_kwargs) if stor_kwargs else StorageConfig(),
            market_design=MarketDesignConfig(**md_kwargs) if md_kwargs else MarketDesignConfig(),
            rt=RTConfig(**rt_kwargs) if rt_kwargs else RTConfig(),
            **top_kwargs,
        )

    net = build_base_network(config)
    agents = create_agents_from_network(net, T, with_wind=with_wind)
    wholesale = day_ahead_price_china(T, agents=agents, config=config)

    # --- apply multipliers ---
    pv_mult = multipliers.get("pv", 1.0)
    wind_mult = multipliers.get("wind", 1.0)
    load_mult = multipliers.get("load", 1.0)
    storage_mult = multipliers.get("storage", 1.0)

    for a in agents:
        a.load_forecast = a.load_forecast * load_mult
        a.load_real = a.load_real * load_mult
        if a.is_prosumer:
            a.pv_forecast = a.pv_forecast * pv_mult
            a.pv_real = a.pv_real * pv_mult
        if a.has_wind:
            a.wind_forecast = a.wind_forecast * wind_mult  # type: ignore[operator]
            a.wind_real = a.wind_real * wind_mult  # type: ignore[operator]
        if a.storage is not None and storage_mult != 1.0:
            a.storage.e_max *= storage_mult
            a.storage.p_ch_max *= storage_mult
            a.storage.p_dis_max *= storage_mult

    # --- apply per-bus custom load scaling (spatial redistribution) ---
    # Runs after the global load multiplier so the factor acts on the
    # already-scaled load profile. Same shape as llm.apply_llm_config_to_agents.
    for cl in custom_loads:
        buses = set(cl.get("buses", []))
        factor = float(cl.get("factor", 1.0))
        if factor == 1.0:
            continue
        for a in agents:
            if a.bus in buses:
                a.load_forecast = a.load_forecast * factor
                a.load_real = a.load_real * factor

    # --- apply RE ramp event ---
    if re_ramp is not None:
        ramp_type = re_ramp.get("type", "sudden_drop")
        ramp_mult = re_ramp.get("multiplier", 0.1)
        mid = T // 2
        for a in agents:
            if not (a.is_prosumer or a.has_wind):
                continue
            pv_orig = a.pv_forecast.copy()
            wind_orig = a.wind_forecast.copy() if a.has_wind and a.wind_forecast is not None else np.zeros(T)
            pv_new = np.zeros(T)
            wind_new = np.zeros(T)
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


# ---- registry ----
SCENARIO_REGISTRY: dict = {}  # populated lazily from scenarios.yaml


def _load_registry():
    """Lazy-load scenario names from YAML config."""
    from config_loader import load_scenarios
    if SCENARIO_REGISTRY:
        return
    scenarios_dict = load_scenarios().get("scenarios", {})
    for name in scenarios_dict:
        SCENARIO_REGISTRY[name] = None  # value unused; name is the key


def get_scenario(name: str, T: int = 96, config: MarketConfig = None) -> Tuple[List[Agent], np.ndarray]:
    _load_registry()
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(SCENARIO_REGISTRY.keys())}")
    return _build_scenario(name, T, config)


def list_scenarios() -> List[str]:
    _load_registry()
    return list(SCENARIO_REGISTRY.keys())


def print_scenario_info():
    """Print all available scenarios with descriptions from config."""
    print("=" * 60)
    print("Available scenarios")
    print("=" * 60)
    for key in list_scenarios():
        cfg = get_scenario_cfg(key)
        desc = cfg.get("description", "") if cfg else ""
        print(f"  {key:<20s}: {desc}")
    print("=" * 60)
