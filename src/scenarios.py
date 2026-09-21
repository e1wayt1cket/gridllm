# scenarios.py
import numpy as np
from typing import List, Optional, Tuple
import money
from models import MarketConfig, Agent
from ess import build_ess_fleet
from grid import build_base_network, create_agents_from_network, day_ahead_price_china
from config_loader import get_scenario_cfg, set_default


def scenario_scale(cfg: dict) -> dict:
    """A scenario's declaration of its own operating scale.

    Four quantities that used to be one number in `defaults.yaml`:

      load_scale_mode       "native_peak" (the network's own nominal demand) or
                            "explicit_peak" (a declared `peak_load_mw`)
      peak_load_mw          the target for the explicit mode
      load_shape            "native" or "diversified"
      renewable.{pv,wind}_bus_mw
                            installed capacity per bus, in MW

    Read from the scenario rather than from the defaults because none of them
    is a property of the benchmark: they are choices this scenario makes, and a
    scenario that inherits another's choice cannot be described.
    """
    missing = [k for k in ("load_scale_mode", "load_shape", "renewable")
               if k not in cfg]
    if missing:
        raise ValueError(
            f"scenario {cfg.get('name', '?')!r} declares no {missing}. Every "
            f"scenario must state its own operating scale and renewable "
            f"capacity: a scenario that inherits another's cannot be described, "
            f"and a reader cannot tell the IEEE33 native scale from a study "
            f"scale. See config/scenarios.yaml.")
    renewable = cfg.get("renewable", {}) or {}
    return {
        "mode": cfg["load_scale_mode"],
        "peak_load_mw": cfg.get("peak_load_mw"),
        "shape": cfg["load_shape"],
        "pv_bus_mw": renewable.get("pv_bus_mw", {}) or {},
        "wind_bus_mw": renewable.get("wind_bus_mw", {}) or {},
    }


def apply_scenario_overrides(config: MarketConfig, override: dict) -> None:
    """Apply a scenario's `override_config` onto a configuration, in place.

    Dotted keys address a nested field ("network.v_min_pu"); a bare key
    addresses a top-level one ("opf_mode"). Idempotent, so a caller that
    already applied them -- `typical_day_config` does, so that a bare call
    returns the model the diagnostic actually clears -- may pass the same
    config through `get_scenario` again without harm.

    Every key must name something that already exists. `setattr` on a dataclass
    instance happily creates a stray attribute for a misspelled or retired
    name, and the run then proceeds with the key silently ignored -- which is
    how a scenario ends up configured by a comment. A key that names nothing is
    an error here.
    """
    for key, value in override.items():
        if "." not in key:
            if key.startswith("_") or not hasattr(config, key):
                raise ValueError(
                    f"scenario override {key!r} names no field of "
                    f"{type(config).__name__}")
            setattr(config, key, value)
            continue

        parts = key.split(".")
        obj = config
        for i, part in enumerate(parts[:-1]):
            if not hasattr(obj, part):
                raise ValueError(
                    f"scenario override {key!r} names no field "
                    f"{'.'.join(parts[:i + 1])!r} of {type(obj).__name__}")
            obj = getattr(obj, part)
        leaf = parts[-1]
        if leaf.startswith("_") or not hasattr(obj, leaf):
            raise ValueError(
                f"scenario override {key!r} names no field {leaf!r} of "
                f"{type(obj).__name__}")
        setattr(obj, leaf, value)


def relocate_renewables(agents: List[Agent], moves) -> None:
    """Move a renewable resource between buses, in place.

    A move rather than a re-rating: the array keeps its size and its own
    profile and only its bus changes, so the installed MW are conserved by
    construction. That is the invariant a layout experiment has to hold --
    re-rating a site would confound "where the array is" with "how big it is",
    and the question is the first.

    Parameters
    ----------
    moves : sequence of (from_bus, to_bus, kind, fraction)
        `kind` is "pv" or "wind". `fraction` defaults to 1.0 and is the share
        of what the source holds **at that point in the sequence**, so a site
        split across two buses is `[(b, x, k, 0.5), (b, y, k, 1.0)]` -- half
        then the remainder -- and a site emptied onto one bus is a single
        default move. Expressed that way the shares are readable in order and
        every move conserves what it touches.

    Raises when a bus is asked to give up a resource it does not carry, or to
    receive one it already has. A layout expressed against buses that have
    moved is a layout whose author was thinking about a different network, and
    silently relocating the wrong array is worse than refusing.
    """
    by_bus = {a.bus: a for a in agents}
    for move in moves:
        from_bus, to_bus, kind = int(move[0]), int(move[1]), move[2]
        fraction = float(move[3]) if len(move) > 3 else 1.0
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"move {move}: fraction must be in (0, 1]")
        src, dst = by_bus.get(from_bus), by_bus.get(to_bus)
        if src is None or dst is None:
            missing = from_bus if src is None else to_bus
            raise ValueError(f"move {move}: no agent at bus {missing}")

        if kind == "pv":
            if src.pv_capacity <= 0:
                raise ValueError(
                    f"move {move}: bus {from_bus} carries no PV to move")
            if fraction == 1.0 and dst.pv_capacity > 0:
                raise ValueError(
                    f"move {move}: bus {to_bus} already carries "
                    f"{dst.pv_capacity:.4g} MW of PV")
            dst.pv_capacity += src.pv_capacity * fraction
            dst.pv_forecast = (dst.pv_forecast
                               + src.pv_forecast * fraction)
            dst.pv_real = dst.pv_real + src.pv_real * fraction
            src.pv_capacity *= (1.0 - fraction)
            src.pv_forecast = src.pv_forecast * (1.0 - fraction)
            src.pv_real = src.pv_real * (1.0 - fraction)
        elif kind == "wind":
            if not src.has_wind or src.wind_capacity <= 0:
                raise ValueError(
                    f"move {move}: bus {from_bus} carries no wind to move")
            if fraction == 1.0 and dst.wind_capacity > 0:
                raise ValueError(
                    f"move {move}: bus {to_bus} already carries "
                    f"{dst.wind_capacity:.4g} MW of wind")
            dst.wind_capacity += src.wind_capacity * fraction
            dst_forecast = dst.wind_forecast if dst.has_wind \
                else np.zeros_like(src.wind_forecast)
            dst_real = dst.wind_real if dst.has_wind \
                else np.zeros_like(src.wind_real)
            dst.wind_forecast = dst_forecast + src.wind_forecast * fraction
            dst.wind_real = dst_real + src.wind_real * fraction
            src.wind_capacity *= (1.0 - fraction)
            if src.wind_capacity <= 0:
                # `has_wind` is a None check, not a capacity check, so an
                # emptied source has to be set back to None or the bus goes on
                # being modelled as a wind site with a zero-rating turbine.
                src.wind_forecast = src.wind_real = None
            else:
                src.wind_forecast = src.wind_forecast * (1.0 - fraction)
                src.wind_real = src.wind_real * (1.0 - fraction)
        else:
            raise ValueError(f"move {move}: unknown kind {kind!r}")


def _build_scenario(name: str, T: int, config: MarketConfig = None,
                    ess_buses: Optional[List[int]] = None,
                    der_moves=None) -> Tuple[List[Agent], np.ndarray]:
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
    # The config's power factor is this scenario's, not a global default: the
    # native scale needs the one that makes the reactive load the benchmark's
    # own, and a legacy scenario must keep the one it was defined with rather
    # than inherit a change made for another scenario.
    if "load_power_factor" in cfg:
        config_load_pf = float(cfg["load_power_factor"])
    else:
        config_load_pf = None
    multipliers = cfg.get("multipliers", {})
    override = dict(cfg.get("override_config", {}))
    # custom_loads is a per-bus load scaling directive applied after agent
    # creation; take it out so it never reaches MarketConfig construction
    # (an unknown key would raise TypeError on the config=None path). Copy
    # the dict first because get_scenario_cfg returns cached YAML state.
    custom_loads = override.pop("custom_loads", [])
    re_ramp = cfg.get("re_ramp", None)

    # Both branches start from config/defaults.yaml, so the file is the source
    # of truth whichever way a scenario is reached. The only difference is
    # whether a caller supplied an existing config to override.
    if config is None:
        config = MarketConfig.from_yaml()
    apply_scenario_overrides(config, override)
    if config_load_pf is not None:
        config.network.load_power_factor = config_load_pf

    net = build_base_network(config)
    agents = create_agents_from_network(net, T, with_wind=with_wind,
                                        scale=scenario_scale(cfg))
    # Which market prices this scenario, written where every regeneration site
    # reads it. Written on every build, including when the scenario declares
    # nothing, so that building the native scenario and then a legacy one
    # cannot leave the legacy one priced by the external market: the price has
    # ten creation sites and only this one knows which scenario is being built.
    #
    # The fallback is the scenario level's default, which is `legacy` -- the
    # retired net-load map. Every scenario that is not the native research day
    # exists only for comparison against results recorded before the external
    # market existed, and is priced the way those results were.
    set_default("wholesale_market.source",
                str(cfg.get("wholesale_source", "legacy")))
    wholesale = day_ahead_price_china(T, agents=agents, config=config)

    # --- apply multipliers ---
    pv_mult = multipliers.get("pv", 1.0)
    wind_mult = multipliers.get("wind", 1.0)
    load_mult = multipliers.get("load", 1.0)
    storage_mult = multipliers.get("storage", 1.0)

    # The network's own batteries shrink or grow with the scenario multiplier
    # and with `storage.prosumer_storage_scale`, so an experiment can make the
    # independent storage fleet a material arbitrageur instead of a marginal
    # one. The fleet itself is built after this and is not affected.
    fleet_storage_mult = storage_mult * float(config.storage.prosumer_storage_scale)

    for a in agents:
        a.load_forecast = a.load_forecast * load_mult
        a.load_real = a.load_real * load_mult
        if a.is_prosumer:
            a.pv_forecast = a.pv_forecast * pv_mult
            a.pv_real = a.pv_real * pv_mult
        if a.has_wind:
            a.wind_forecast = a.wind_forecast * wind_mult  # type: ignore[operator]
            a.wind_real = a.wind_real * wind_mult  # type: ignore[operator]
        if a.storage is not None and fleet_storage_mult != 1.0:
            a.storage.e_max *= fleet_storage_mult
            a.storage.p_ch_max *= fleet_storage_mult
            a.storage.p_dis_max *= fleet_storage_mult

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

    # --- independent storage fleet ---
    # Appended last so the load, RE-ramp and storage-scaling steps above apply
    # only to the network's own participants: the fleet carries no load and no
    # generation for those steps to act on, and its capacity is its own knob.
    # `ess_buses` re-places the fleet for the siting experiment. An empty list
    # is a real request -- no fleet at all, which is how a single unit is
    # measured against a market that has no other battery in it -- so it must
    # stay distinguishable from the default, which is "whatever the file says".
    agents.extend(build_ess_fleet(T, config, wholesale=wholesale,
                                  buses=ess_buses))

    # After the fleet, because it is a placement of the same facilities the
    # steps above have finished scaling: a move is a relocation of what is
    # already installed, not a sizing.
    if der_moves:
        relocate_renewables(agents, der_moves)

    # Two agents sharing a name silently corrupt the clearing model: the batch
    # solvers key charge, discharge and state of charge by agent name, so a
    # second unit with the same name overwrites the first unit's variables and
    # that unit then reports zero dispatch for the whole horizon without any
    # error being raised. Refusing up front is cheaper than finding it later.
    names = [a.name for a in agents]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate agent names in scenario {name!r}: {duplicates}")

    return agents, wholesale


def typical_day_config(T: int = money.PERIODS_PER_DAY) -> MarketConfig:
    """The configuration the typical-day physical baseline is defined against.

    Makes the canonical entry point executable rather than a description. The
    horizon is checked instead of coerced because the terminal state of charge
    is pinned only on a whole day: at any other T this would quietly be a
    different model, and the baselines are all quoted for the whole day.
    """
    if T != money.PERIODS_PER_DAY:
        raise ValueError(
            f"the typical day is {money.PERIODS_PER_DAY} periods; got T={T}")
    # Built from config/defaults.yaml, then given this scenario's own choices
    # from config/scenarios.yaml. Nothing is assigned here: a value that lives
    # in code rather than in the file is a value the next reader cannot find,
    # and the point of the binding is that the file says what the model is.
    config = MarketConfig.from_yaml(opf_mode="socp")
    cfg = get_scenario_cfg("typical_day")
    apply_scenario_overrides(config, cfg.get("override_config", {}))
    # The scenario's own power factor, applied here as well as in
    # `_build_scenario`: this entry point returns a model to clear with, so a
    # field the file declares and this function omits would make the
    # diagnostic's market differ from the one the scenario actually builds.
    if "load_power_factor" in cfg:
        config.network.load_power_factor = float(cfg["load_power_factor"])
    return config


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


def get_scenario(name: str, T: int = 96, config: MarketConfig = None,
                 ess_buses: Optional[List[int]] = None, der_moves=None,
                 day_id: Optional[int] = None) \
        -> Tuple[List[Agent], np.ndarray]:
    """Build one scenario.

    `ess_buses` overrides the independent fleet's placement -- None keeps the
    configured one, [] requests no fleet. `der_moves` relocates renewable
    sites, as a sequence of (from_bus, to_bus, kind[, fraction]).

    `day_id` selects which day of the wholesale price library the day is
    priced on. It is written into the configuration caches rather than passed
    down, the way `diagnose_typical_day.run_day` moves `load_scale`: the price
    has ten creation sites and threading a new argument through all of them
    would be a refactor in service of one number.

    None leaves the cache alone, which -- like every other `set_default` knob
    -- means it leaves the last value a caller set rather than reverting to the
    file. A caller that wants the file's value after setting one calls
    `reload_defaults()` first. The source itself is different: `_build_scenario`
    rewrites it on every build, so a scenario can never inherit another's
    market.
    """
    _load_registry()
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(SCENARIO_REGISTRY.keys())}")
    if day_id is not None:
        set_default("wholesale_market.day_id", int(day_id))
    return _build_scenario(name, T, config, ess_buses=ess_buses,
                           der_moves=der_moves)


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
