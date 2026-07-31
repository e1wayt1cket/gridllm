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
    self_schedule: bool = True             # MPC pre-computes schedule, OPF treats as fixed
    mpc_horizon: int = 64                  # MPC look-ahead periods (16 hours)
    mpc_price_noise_pct: float = 5.0       # per-agent MPC price forecast noise (%)
    mpc_bus_markup_pct: float = 30.0       # max bus-distance markup for MPC prices (%)
    use_nodal_price: bool = True           # two-pass: re-solve with storage at nodal LMP
    mpc_fast_heuristic: bool = False      # use O(H) threshold heuristic instead of LP for MPC
    mpc_congestion_pass: bool = False     # enable second congestion-aware MPC pass


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
    enable_multi_objective: bool = True
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


@dataclass
class MarketConfig:
    """Top-level configuration composing network, storage, market-design, and RT settings."""
    opf_mode: str = "socp"            # "dc", "lindistflow", or "socp"
    opf_tolerance: float = 1e-6
    opf_max_iter: int = 100
    verbose: bool = False
    network: NetworkConfig = field(default_factory=NetworkConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    market_design: MarketDesignConfig = field(default_factory=MarketDesignConfig)
    rt: RTConfig = field(default_factory=RTConfig)

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

    @property
    def has_wind(self) -> bool:
        return self.wind_forecast is not None and self.wind_real is not None

    def get_wind_forecast(self):
        return self.wind_forecast if self.has_wind else np.array([])

    def get_wind_real(self):
        return self.wind_real if self.has_wind else np.array([])

