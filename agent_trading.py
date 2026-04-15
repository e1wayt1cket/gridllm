import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

np.random.seed(1)

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
    storage: Optional[StorageSpec] = None

@dataclass
class Network:
    cap01: float
    cap12: float

def build_demo_case(T=24) -> Tuple[List[Agent], Network, np.ndarray]:
    hours = np.arange(T)
    wholesale = np.array([
        1.14758475, 1.13159475, 1.07758475, 1.03258475, 0.88410775,
        0.82860775, 1.50748175, 1.47349775, 1.43748175, 1.39248175,
        1.20401675, 1.14851675
    ])
    return [], Network(6.5, 3.5), wholesale

def run_one_day_demo():
    agents, network, wholesale = build_demo_case()
    print(wholesale)

run_one_day_demo()