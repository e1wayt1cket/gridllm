# models.py
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple

@dataclass
class MarketConfig:
    opf_mode: str = "lindistflow"      # "dc" 或 "lindistflow"
    use_ac_opf: bool = False           # 保留兼容
    opf_tolerance: float = 1e-6
    opf_max_iter: int = 100
    verbose: bool = False
    bid_mult_range: Tuple[float, float] = (0.8, 1.2)
    offer_adder_range: Tuple[float, float] = (0.0, 50.0)
    default_bid_mult: float = 1.0
    default_offer_adder: float = 0.0
    line_capacity_multiplier: float = 3.0
    penalty_unserved: float = 800.0
    base_mva: float = 1.0
    base_kv: float = 12.66
    # Multi-objective optimization weights
    lambda_re: float = 50.0           # RE incentive (CNY/MWh)
    lambda_curtail: float = 15.0      # Curtailment penalty (CNY/MWh)
    lambda_carbon: float = 50.0       # Carbon cost (CNY/tCO2)
    emission_factor_grid: float = 0.58  # Grid emission factor (tCO2/MWh)
    enable_multi_objective: bool = True
    # Constraint-based multi-objective (hard constraints instead of weighted-sum)
    use_constraint_multi_obj: bool = True
    carbon_cap_tco2: Optional[float] = 200.0   # hard cap on total carbon
    re_min_rate: Optional[float] = 95.0       # minimum RE consumption rate (0-100)
    # Storage mode thresholds (per-unit, relative to bid/offer)
    storage_charge_discount: float = 0.85   # bid * roundtrip_eff * this → charge trigger
    storage_discharge_premium: float = 1.15  # offer / roundtrip_eff * this → discharge trigger
    storage_soc_buffer: float = 0.02  # SOC buffer from min/max before forced charge/discharge
    rt_horizon: int = 4        # RT 每次优化的时段数
    rt_step: int = 1           # RT 步长（时段）

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
    self_discharge_rate: float = 0.001
    p_ch_min: float = 0.0
    p_dis_min: float = 0.0
    ramp_up_ch: Optional[float] = None
    ramp_down_ch: Optional[float] = None
    ramp_up_dis: Optional[float] = None
    ramp_down_dis: Optional[float] = None
    cost_ch: float = 0.0
    cost_dis: float = 0.0
    hot_start_cost: float = 0.0
    warm_start_cost: float = 0.0
    cold_start_cost: float = 0.0
    downtime_hot: float = 8.0
    downtime_warm: float = 48.0
    min_operating_time: float = 0.0
    min_down_time: float = 0.0

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

    @property
    def has_wind(self) -> bool:
        return self.wind_forecast is not None and self.wind_real is not None

    def get_wind_forecast(self):
        return self.wind_forecast if self.has_wind else np.array([])

    def get_wind_real(self):
        return self.wind_real if self.has_wind else np.array([])