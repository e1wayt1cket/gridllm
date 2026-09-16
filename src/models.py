# models.py
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Optional, Tuple


@dataclass
class NetworkConfig:
    """Physical network parameters for OPF and topology."""
    base_kv: float = 12.66
    base_mva: float = 1.0
    v_min_pu: float = 0.93          # min voltage (GB/T 12325 ±7%)
    v_max_pu: float = 1.05          # max voltage
    line_capacity_multiplier: float = 1.5
    line_capacity_overrides: Optional[Dict[int, float]] = None
    reverse_power_limit_mw: float = 2.5  # max reverse flow to main grid (MW), 0=no export
    ramp_limit_mw_per_period: Optional[float] = None  # max MW change per 15-min, None=disabled
    reactive_support: bool = True    # PV/storage inverters provide reactive power
    load_power_factor: float = 0.9   # lagging, for reactive load allocation
    # Inverter apparent-power rating as a multiple of the real-power rating it
    # is sized from. At 1.0 the inverter's two outputs share one circle exactly,
    # so every MVar it supplies for voltage support is a MWh of real output it
    # cannot deliver -- which is where this network's renewable curtailment
    # comes from, not from the price. Real inverters are commonly rated a little
    # above their panels for exactly this reason. See the headroom sweep in
    # diagnose_typical_day.
    inverter_smax_multiplier: float = 1.0
    n_loss_iters: int = 3            # I²R loss linearization iterations (0=no losses)


@dataclass
class StorageConfig:
    """Storage bidding thresholds, MPC self-scheduling, and degradation params."""
    charge_discount: float = 0.75          # bid * rt_eff * this → charge trigger
    discharge_premium: float = 1.30        # offer / rt_eff * this → discharge trigger
    soc_buffer: float = 0.05               # SOC buffer from min/max before forced action
    terminal_value: Optional[float] = None  # None = use day-ahead mean price
    cycle_cost: float = 100.0              # cycling degradation cost (CNY/MWh per ch+dis)
    discount_factor: float = 0.997         # per-period discount on future storage revenue
    self_schedule: bool = False            # MPC pre-computes schedule, OPF treats as fixed.
                                           # Default False so storage is dispatched by the market
                                           # on declared bids (the RL path); MPC flows opt in.
    mpc_horizon: int = 64                  # MPC look-ahead periods (16 hours)
    mpc_price_noise_pct: float = 5.0       # per-agent MPC price forecast noise (%)
    mpc_bus_markup_pct: float = 30.0       # max bus-distance markup for MPC prices (%)
    use_nodal_price: bool = True           # two-pass: re-solve with storage at nodal LMP
    mpc_fast_heuristic: bool = False      # use O(H) threshold heuristic instead of LP for MPC
    mpc_congestion_pass: bool = False     # enable second congestion-aware MPC pass
    # Retired. This capped a storage unit's declared charge bid at its declared
    # discharge offer, which was what kept the objective concave and stopped a
    # unit bidding its way into a simultaneous charge and discharge. Charge and
    # discharge are now the parts of one net power flow, so the overlap is not
    # representable and no quote needs capping; the flag is kept because
    # callers and tests still pass it, and it no longer changes any dispatch.
    # The cap was also what made storage charge without ever discharging, by
    # forbidding the price spread a battery uses to express arbitrage.
    churn_free_quotes: bool = True
    # "off": rely on churn_free_quotes, which keeps the model a QCP.
    # "binary": additionally impose an explicit per-period exclusive-or on
    # charge/discharge. Exact by construction, but it makes the clearing model a
    # MIQCP, and Gurobi reports no duals for a MIP — so the power-balance duals
    # every LMP is read from are lost until the directions are frozen and the
    # model re-solved as a continuous QCP. Kept for cross-checking the rule.
    # Measured on the baseline scenario at the RL window (16 periods, shaded
    # bid): "off" clears in 0.18 s with zero overlap, while "strategy_only" —
    # binaries on the three independent-storage units only — did not finish in
    # eight minutes. Allowing a unit to bid to charge above what it asks to
    # discharge is what a battery needs in order to arbitrage, and it is also
    # what makes the exclusive-or expensive to resolve, so the exact mode is
    # not usable at the horizon a policy trains through. Default is the mode
    # that is both fast and overlap-free.
    exclusive_mode: str = "off"
    # Require each storage unit to end the horizon at the state of charge it
    # started with. A day that finishes with a fuller battery has banked energy
    # it never sold, so comparing two bidding policies on cash alone rewards
    # whichever one ran the battery down; pinning the endpoints makes the
    # comparison like for like without needing a terminal price at all. It is
    # expressible only on a whole-horizon clear, so it belongs to evaluation
    # rather than to the rolling window the policy acts through. This is the
    # outer switch; whether a given clear is the settled horizon is the
    # caller's `horizon_type` argument, because a window that happens to be a
    # whole day long is still a window. See dispatch_core.pins_terminal_soc.
    terminal_soc_equal: bool = True
    # A storage unit's quotes are its declared charge and discharge prices, and
    # `ess.build_ess_fleet` sets both to zero: zero strategic premium is the
    # truthful declaration, because the round trip's own cost is charged
    # separately through `cycle_cost`. `bid_value` and `offer_cost` are the
    # settlement-side prices of a load (what it will pay) and a generator (what
    # it must receive); a battery alone has neither, and
    # `participant_payoff.IndependentStoragePayoff` reads neither of them. Any
    # separate anchor would be a second source of truth for the same quote.
    # Scale applied to every battery already in the network, prosumer and
    # non-prosumer alike, when a scenario is built. The independent-storage
    # experiments reduce it so that fleet is a material arbitrageur rather than
    # a marginal one; it is part of the config snapshot, so a run records the
    # market it was cleared in. 1.0 leaves the existing fleet unchanged.
    prosumer_storage_scale: float = 1.0


@dataclass
class MarketDesignConfig:
    """Market rules, bidding ranges, multi-objective weights and constraints."""
    penalty_unserved: float = 2500.0
    bid_mult_range: Tuple[float, float] = (0.3, 1.8)
    offer_adder_range: Tuple[float, float] = (0.0, 50.0)
    default_bid_mult: float = 1.0
    default_offer_adder: float = 0.0
    emission_factor_grid: float = 0.58     # tCO2/MWh
    # Weighted-sum multi-objective
    enable_multi_objective: bool = False
    lambda_re: float = 50.0                # RE incentive (CNY/MWh)
    lambda_curtail: float = 200.0          # curtailment penalty (CNY/MWh)
    lambda_carbon: float = 80.0            # carbon cost (CNY/tCO2)
    # Constraint-based multi-objective (hard constraints with soft slack)
    use_constraint_multi_obj: bool = False
    carbon_cap_tco2: Optional[float] = None    # total CO2 emissions cap (tonnes)
    re_min_rate: Optional[float] = None        # minimum RE consumption rate (fraction 0-1)
    penalty_carbon_slack: float = 5000.0       # slack penalty for carbon cap (CNY/tCO2)
    penalty_re_slack: float = 3000.0           # slack penalty for RE rate (CNY/MWh)


@dataclass
class RTConfig:
    """Real-time market and rolling-horizon DA parameters."""
    rt_forecast_mode: str = "noisy_da"    # perfect | da_as_forecast | noisy_da
    rt_forecast_noise_pct: float = 10.0   # noise std as % of DA price (noisy_da)
    rt_horizon: int = 8                    # periods per RT optimization window
    rt_step: int = 1                       # RT step size (periods)
    da_rolling_enabled: bool = False       # toggle rolling-horizon DA
    da_window_length: int = 48             # periods per rolling window (12 hours)
    da_window_step: int = 8                # periods to advance per iteration (2 hours)
    da_forecast_noise_pct: float = 0.0    # noise std as % of DA price within window


# ---------------------------------------------------------------------------
# Binding configuration to config/defaults.yaml
# ---------------------------------------------------------------------------
# `config/defaults.yaml` is the configuration's single source of truth. The
# maps below are the only declaration of which YAML key feeds which dataclass
# field, and `MarketConfig.from_yaml` is the only reader.
#
# This exists because the file used to be misleading rather than authoritative:
# the `storage:`, `market_design:` and `rt:` blocks had no reader at all, so
# their values were documentation while the dataclass defaults in this module
# were the model. One of them had drifted -- `enable_multi_objective` read
# `true` in the file against an effective `false` in code -- and nobody could
# have noticed, because editing the file changed nothing.
#
# Two invariants are enforced by `tests/test_yaml_config_agreement.py` and are
# the reason this is an explicit map rather than `**kwargs`:
#   1. A YAML key with no field behind it must fail loudly, not be skipped.
#   2. Every field of the four config dataclasses must be either mapped here or
#      named in KNOWN_INERT with a reason, so an unwired field is a decision
#      rather than an omission.
YAML_FIELD_MAP = {
    "network": {
        "base_kv": "network.base_kv",
        "v_min_pu": "network.v_min_pu",
        "v_max_pu": "network.v_max_pu",
        "line_capacity_multiplier": "network.line_capacity_multiplier",
        "line_capacity_overrides": "network.line_capacity_overrides",
        "reverse_power_limit_mw": "network.reverse_power_limit_mw",
        "ramp_limit_mw_per_period": "network.ramp_limit_mw_per_period",
        "reactive_support": "network.reactive_support",
        "load_power_factor": "network.load_power_factor",
        "inverter_smax_multiplier": "network.inverter_smax_multiplier",
        "n_loss_iters": "network.n_loss_iters",
    },
    "storage": {
        "charge_discount": "storage.charge_discount",
        "discharge_premium": "storage.discharge_premium",
        "soc_buffer": "storage.soc_buffer",
        "terminal_value": "storage.terminal_value",
        "cycle_cost": "storage.cycle_cost",
        "discount_factor": "storage.discount_factor",
        "self_schedule": "storage.self_schedule",
        "mpc_horizon": "storage.mpc_horizon",
        "mpc_price_noise_pct": "storage.mpc_price_noise_pct",
        "mpc_bus_markup_pct": "storage.mpc_bus_markup_pct",
        "use_nodal_price": "storage.use_nodal_price",
        "mpc_fast_heuristic": "storage.mpc_fast_heuristic",
        "mpc_congestion_pass": "storage.mpc_congestion_pass",
        "churn_free_quotes": "storage.churn_free_quotes",
        "exclusive_mode": "storage.exclusive_mode",
        "terminal_soc_equal": "storage.terminal_soc_equal",
        "prosumer_storage_scale": "storage.prosumer_storage_scale",
    },
    "market_design": {
        "penalty_unserved": "market_design.penalty_unserved",
        "bid_mult_range": "market_design.bid_mult_range",
        "offer_adder_range": "market_design.offer_adder_range",
        "default_bid_mult": "market_design.default_bid_mult",
        "default_offer_adder": "market_design.default_offer_adder",
        "emission_factor_grid": "market_design.emission_factor_grid",
        "enable_multi_objective": "market_design.enable_multi_objective",
        "lambda_re": "market_design.lambda_re",
        "lambda_curtail": "market_design.lambda_curtail",
        "lambda_carbon": "market_design.lambda_carbon",
        "use_constraint_multi_obj": "market_design.use_constraint_multi_obj",
        "carbon_cap_tco2": "market_design.carbon_cap_tco2",
        "re_min_rate": "market_design.re_min_rate",
        "penalty_carbon_slack": "market_design.penalty_carbon_slack",
        "penalty_re_slack": "market_design.penalty_re_slack",
    },
    "rt": {
        "rt_forecast_mode": "rt.rt_forecast_mode",
        "rt_forecast_noise_pct": "rt.rt_forecast_noise_pct",
        "rt_horizon": "rt.rt_horizon",
        "rt_step": "rt.rt_step",
        "da_rolling_enabled": "rt.da_rolling_enabled",
        "da_window_length": "rt.da_window_length",
        "da_window_step": "rt.da_window_step",
        "da_forecast_noise_pct": "rt.da_forecast_noise_pct",
    },
}

# Fields deliberately left unbound, with the reason. Kept short on purpose: an
# entry here is a decision, and adding one is how a field stops being wired.
KNOWN_INERT = {
    "NetworkConfig.base_mva":
        "the per-unit conversion in all three engines is written as "
        "z_base = base_kv**2, which is the textbook Z_base = V**2 / S_base "
        "with S_base = 1 MVA. Nothing reads this field, and wiring it means "
        "reworking the base in dispatch_socp, dispatch_ldf and dispatch_dc "
        "together -- a change of its own, not a config edit.",
}

# Keys inside the bound blocks that belong to some other reader. They are
# configuration, just not of MarketConfig: the network topology and its line
# ratings are read by grid.build_base_network, which asks for them directly.
YAML_KEYS_READ_ELSEWHERE = {
    "network.name",
    "network.base_ampacity_ka",
    "network.min_line_ka",
    "network.max_line_ka",
}


@dataclass
class MarketConfig:
    """Top-level configuration composing network, storage, market-design, and RT settings."""
    opf_mode: str = "socp"            # "dc", "lindistflow", or "socp"
    # Scale the clearing objective's energy-valued terms, and de-scale the
    # power-balance duals they price, by the period length. False reproduces the
    # objective used before the money-unit unification, which valued a period's
    # power as if a period were an hour and therefore overstated every base
    # economic term against the already-correct carbon, RE and terminal-SOC
    # terms. Verification and historical reproduction only.
    dt_scaled_money: bool = True
    opf_tolerance: float = 1e-6
    opf_max_iter: int = 100
    verbose: bool = False
    network: NetworkConfig = field(default_factory=NetworkConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    market_design: MarketDesignConfig = field(default_factory=MarketDesignConfig)
    rt: RTConfig = field(default_factory=RTConfig)

    @classmethod
    def from_yaml(cls, **overrides) -> "MarketConfig":
        """Build the configuration from `config/defaults.yaml`.

        The file is the source of truth; this is the only reader. Fields the
        YAML does not carry keep their dataclass defaults, which is why the
        top-level fields (`opf_mode`, the tolerance, `verbose`) are passed as
        keyword overrides rather than living in the file -- they describe how a
        run is executed, not what the market is.

        A key named in YAML_FIELD_MAP but missing from the file raises, rather
        than quietly falling back to a default: a file that has lost a key is a
        file that has stopped being the source of truth, and that is the
        failure mode this replaced.
        """
        # Imported here rather than at module scope: config_loader reads this
        # module's dataclass defaults for nothing, but keeping the dependency
        # inside the one method that needs it leaves the import graph flat.
        from config_loader import get_default

        sections = {
            "network": NetworkConfig(),
            "storage": StorageConfig(),
            "market_design": MarketDesignConfig(),
            "rt": RTConfig(),
        }
        for section, mapping in YAML_FIELD_MAP.items():
            target = sections[section]
            for field_name, yaml_key in mapping.items():
                sentinel = object()
                value = get_default(yaml_key, sentinel)
                if value is sentinel:
                    raise KeyError(
                        f"{yaml_key} is bound to {type(target).__name__}."
                        f"{field_name} but is not in config/defaults.yaml; add "
                        f"it, or unbind the field")
                # YAML has no tuple, so a range written as [lo, hi] arrives as a
                # list and would quietly change the field's type. Restore what
                # the field declares; a value of the wrong shape is an error.
                current = getattr(target, field_name)
                if isinstance(current, tuple) and isinstance(value, list):
                    value = tuple(value)
                setattr(target, field_name, value)

        config = cls(network=sections["network"], storage=sections["storage"],
                     market_design=sections["market_design"],
                     rt=sections["rt"])
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise KeyError(f"MarketConfig has no field {key!r}")
            setattr(config, key, value)
        return config


@dataclass
class StorageSpec:
    e_max: float
    p_ch_max: float
    p_dis_max: float
    eta_ch: float
    eta_dis: float
    soc0: float
    soc_min: float
    soc_max: float
    p_ch_min: float = 0.0
    p_dis_min: float = 0.0
    ramp_up_ch: Optional[float] = None
    ramp_down_ch: Optional[float] = None
    ramp_up_dis: Optional[float] = None
    ramp_down_dis: Optional[float] = None

    def max_discharge_feasible(self, soc, dt=0.25):
        if soc <= self.soc_min + 1e-6:
            return 0.0
        energy_available = (soc - self.soc_min) * self.e_max * self.eta_dis
        return min(energy_available / dt, self.p_dis_max)

    def max_charge_feasible(self, soc, dt=0.25):
        if soc >= self.soc_max - 1e-6:
            return 0.0
        energy_headroom = (self.soc_max - soc) * self.e_max / self.eta_ch
        return min(energy_headroom / dt, self.p_ch_max)

@dataclass
class Agent:
    name: str
    bus: int
    is_prosumer: bool
    load_forecast: np.ndarray
    pv_forecast: np.ndarray
    load_real: np.ndarray
    pv_real: np.ndarray
    bid_value: float
    offer_cost: float
    wind_forecast: Optional[np.ndarray] = None
    wind_real: Optional[np.ndarray] = None
    storage: Optional[StorageSpec] = None
    load_type: str = "unknown"
    pv_capacity: float = 0.0
    wind_capacity: float = 0.0
    # Which payoff model settles this participant (see participant_payoff.py).
    # "prosumer" is the general site ledger: load, on-site generation and an
    # optional battery. "storage" is a battery alone, with no load and no
    # generation, whose payoff must not inherit either term.
    participant_type: str = "prosumer"

    @property
    def has_wind(self) -> bool:
        return self.wind_forecast is not None and self.wind_real is not None

    def get_wind_forecast(self):
        return self.wind_forecast if self.has_wind else np.array([])

    def get_wind_real(self):
        return self.wind_real if self.has_wind else np.array([])

