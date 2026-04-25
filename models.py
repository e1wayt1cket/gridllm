# models.py
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple


@dataclass
class MarketConfig:
    use_ac_opf: bool = False
    opf_tolerance: float = 1e-6
    opf_max_iter: int = 100
    verbose: bool = False
    bid_mult_range: Tuple[float, float] = (0.8, 1.2)
    offer_adder_range: Tuple[float, float] = (0.0, 50.0)
    default_bid_mult: float = 1.0
    default_offer_adder: float = 0.0
    line_capacity_multiplier: float = 3.0
    w_re_consume: float = 0.0
    penalty_unserved: float = 800.0
    base_mva: float = 1.0
    base_kv: float = 12.66
    w_re_consume: float = 0.0          # 保留原有激励系数（向后兼容）
    lambda_re: float = 0.0             # 可再生消纳奖励权重（多目标加权系数）

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

    # ── ASSUME-inspired 精细化约束 ──
    p_ch_min: float = 0.0       # 最小充电功率 (MW)，避免低效区
    p_dis_min: float = 0.0      # 最小放电功率 (MW)
    ramp_up_ch: Optional[float] = None   # 充电功率增加速率 (MW/15min)
    ramp_down_ch: Optional[float] = None # 充电功率减少速率 (MW/15min)
    ramp_up_dis: Optional[float] = None  # 放电功率增加速率 (MW/15min)
    ramp_down_dis: Optional[float] = None# 放电功率减少速率 (MW/15min)

    # ── 成本与启停 ──
    cost_ch: float = 0.0        # 充电边际成本 ¥/MWh
    cost_dis: float = 0.0       # 放电边际成本 ¥/MWh
    hot_start_cost: float = 0.0
    warm_start_cost: float = 0.0
    cold_start_cost: float = 0.0
    downtime_hot: float = 8.0   # 小时
    downtime_warm: float = 48.0 # 小时
    min_operating_time: float = 0.0
    min_down_time: float = 0.0

    def __post_init__(self):
        if self.e_max <= 0:
            raise ValueError(f"e_max={self.e_max} must be > 0")
        if not (0 <= self.soc_min <= 1):
            raise ValueError(f"soc_min={self.soc_min} must be in [0,1]")
        if not (0 <= self.soc_max <= 1):
            raise ValueError(f"soc_max={self.soc_max} must be in [0,1]")
        if self.soc_max < self.soc_min:
            raise ValueError(f"soc_max={self.soc_max} must be >= soc_min={self.soc_min}")
        if not (self.soc_min <= self.soc0 <= self.soc_max):
            raise ValueError(f"soc0={self.soc0} must be in [{self.soc_min},{self.soc_max}]")
        if self.p_ch_max < 0:
            raise ValueError(f"p_ch_max={self.p_ch_max} must be >= 0")
        if self.p_dis_max < 0:
            raise ValueError(f"p_dis_max={self.p_dis_max} must be >= 0")
        if self.p_ch_min < 0:
            raise ValueError(f"p_ch_min={self.p_ch_min} must be >= 0")
        if self.p_dis_min < 0:
            raise ValueError(f"p_dis_min={self.p_dis_min} must be >= 0")
        if self.p_ch_min > self.p_ch_max:
            raise ValueError(f"p_ch_min={self.p_ch_min} must be <= p_ch_max={self.p_ch_max}")
        if self.p_dis_min > self.p_dis_max:
            raise ValueError(f"p_dis_min={self.p_dis_min} must be <= p_dis_max={self.p_dis_max}")
        if not (0 < self.eta_ch <= 1):
            raise ValueError(f"eta_ch={self.eta_ch} must be in (0,1]")
        if not (0 < self.eta_dis <= 1):
            raise ValueError(f"eta_dis={self.eta_dis} must be in (0,1]")
        if self.self_discharge_rate < 0:
            raise ValueError("self_discharge_rate must be >= 0")

    def max_discharge_feasible(self, soc: float, dt: float = 0.25) -> float:
        """根据 SOC 下限计算当前最大可放电功率"""
        if soc <= self.soc_min + 1e-6:
            return 0.0
        energy_available = (soc - self.soc_min) * self.e_max * self.eta_dis
        return min(energy_available / dt, self.p_dis_max)

    def max_charge_feasible(self, soc: float, dt: float = 0.25) -> float:
        """根据 SOC 上限计算当前最大可充电功率"""
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
    wind_used: Optional[np.ndarray] = None
    wind_real: Optional[np.ndarray] = None
    storage: Optional[StorageSpec] = None
    load_type: str = "unknown"

    @property
    def has_wind(self) -> bool:
        return self.wind_forecast is not None and self.wind_real is not None

    def get_wind_forecast(self) -> np.ndarray:
        if not self.has_wind or self.wind_forecast is None:
            return np.array([])
        return self.wind_forecast

    def get_wind_real(self) -> np.ndarray:
        if not self.has_wind or self.wind_real is None:
            return np.array([])
        return self.wind_real