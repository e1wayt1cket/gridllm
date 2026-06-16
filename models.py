# models.py
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple

@dataclass
class MarketConfig:
    opf_mode: str = "lindistflow"      # "dc" or "lindistflow"
    use_ac_opf: bool = False           # retained for compatibility
    opf_tolerance: float = 1e-6
    opf_max_iter: int = 100
    verbose: bool = False
    bid_mult_range: Tuple[float, float] = (0.8, 1.2)
    offer_adder_range: Tuple[float, float] = (0.0, 50.0)
    default_bid_mult: float = 1.0
    default_offer_adder: float = 0.0
    line_capacity_multiplier: float = 1.0
    penalty_unserved: float = 5000.0
    base_mva: float = 1.0
    base_kv: float = 12.66
    v_min_pu: float = 0.78      # min voltage in LinDistFlow (accounts for linearization error vs AC PF)
    v_max_pu: float = 1.05      # max voltage
    # Multi-objective optimization weights
    lambda_re: float = 50.0           # RE incentive (CNY/MWh)
    lambda_curtail: float = 200.0     # Curtailment penalty (CNY/MWh), comparable to avg LMP
    lambda_carbon: float = 50.0       # Carbon cost (CNY/tCO2)
    emission_factor_grid: float = 0.58  # Grid emission factor (tCO2/MWh)
    enable_multi_objective: bool = False
    # Constraint-based multi-objective (hard constraints) — disabled
    use_constraint_multi_obj: bool = False
    carbon_cap_tco2: Optional[float] = None
    re_min_rate: Optional[float] = None
    # Storage mode thresholds (per-unit, relative to bid/offer)
    storage_charge_discount: float = 0.85   # bid * roundtrip_eff * this → charge trigger
    storage_discharge_premium: float = 1.15  # offer / roundtrip_eff * this → discharge trigger
    storage_soc_buffer: float = 0.05  # SOC buffer from min/max before forced charge/discharge
    storage_terminal_value: Optional[float] = None  # None = use day-ahead mean price
    lambda_cycle: float = 50.0     # Cycling degradation cost (CNY/MWh per ch+dis)
    load_power_factor: float = 0.9  # Load power factor (lagging) for reactive power
    n_loss_iters: int = 2          # Iterations for I²R loss linearization (0=no losses)
    use_nodal_storage_price: bool = True  # Two-pass: re-solve with storage at nodal LMP
    rt_forecast_mode: str = "perfect"  # perfect | da_as_forecast | noisy_da
    rt_forecast_noise_pct: float = 10.0  # noise std as % of DA price (noisy_da)
    rt_horizon: int = 8        # periods per RT optimization window
    rt_step: int = 1           # RT step size (periods)
    reactive_support: bool = True  # PV/storage inverters provide reactive power

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

