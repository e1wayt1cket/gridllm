"""grid.py
电网拓扑构建、智能体工厂、日前电价曲线生成
"""
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from typing import List

from models import MarketConfig, StorageSpec, Agent


def build_base_network(config: MarketConfig) -> pp.pandapowerNet:
    net = pn.case33bw()
    net.line['max_i_ka'] = net.line['max_i_ka'].fillna(1.0) * config.line_capacity_multiplier #type: ignore[union-attr]
    return net #type: ignore


def create_agents_from_network(net: pp.pandapowerNet, T: int,
                               with_wind: bool = False) -> List[Agent]:
    hours = np.arange(T)

    def pv_profile():
        return np.clip(1.0 * np.sin((hours - 6) / 24 * 2 * np.pi), 0, None)

    def wind_profile():
        rng = np.random.RandomState(42)
        w = np.clip(
            0.7 * (0.5 + 0.5 * np.sin((hours - 3) / 12 * np.pi))
            + 0.2 * np.sin((hours - 14) / 8 * np.pi)
            + rng.normal(0, 0.12, size=T),
            0, 1.3
        )
        return np.maximum(w, 0.05)

    def noisy(x, sigma=0.1):
        return np.clip(x * (1 + np.random.normal(0, sigma, size=x.shape)), 0, None)

    agents = []
    pv_base = pv_profile()
    wind_base = wind_profile() if with_wind else None

    for idx, load in net.load.iterrows():
        bus = load.bus
        p_mw = load.p_mw

        if bus in [0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21]:
            load_type = "residential"
        elif bus in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
            load_type = "commercial"
        else:
            load_type = "industrial"
        if load_type == "residential":
            base_load = p_mw * 0.9
        elif load_type == "commercial":
            base_load = p_mw * 1.0
        else:
            base_load = p_mw * 1.1
        
        load_forecast = noisy(np.full(T, base_load), 0.06)
        load_real = noisy(np.full(T, base_load), 0.12)

        is_prosumer = (load_type == "residential" and bus in [5, 12]) or \
                      (load_type == "industrial" and with_wind and bus in [22, 25])

        pv_forecast = np.zeros(T)
        pv_real = np.zeros(T)
        wind_forecast = None
        wind_real = None
        storage = None

        if is_prosumer:
            if load_type == "residential":
                pv_cap = base_load * 1.8
                pv_forecast = noisy(pv_base * pv_cap, 0.15)
                pv_real = noisy(pv_base * pv_cap, 0.20)
                
                storage_capacity = base_load * 3.0
                storage_power = storage_capacity * 0.2
                storage = StorageSpec(
                    e_max=storage_capacity,
                    p_ch_max=storage_power,
                    p_dis_max=storage_power,
                    eta_ch=0.93,
                    eta_dis=0.93,
                    soc0=0.5,
                    soc_min=0.0,
                    soc_max=1.0,
                    self_discharge_rate=0.001
                )
                
                bid_val = 550.0
                offer_cost = 280.0
                
            else:
                wind_cap = base_load * 2.5
                if wind_base is not None:
                    wind_forecast = noisy(wind_base * wind_cap, 0.12)
                    wind_real = noisy(wind_base * wind_cap, 0.18)
                else:
                    wind_forecast = np.zeros(T)
                    wind_real = np.zeros(T)
                
                storage_capacity = base_load * 4.0
                storage_power = storage_capacity * 0.2
                storage = StorageSpec(
                    e_max=storage_capacity,
                    p_ch_max=storage_power,
                    p_dis_max=storage_power,
                    eta_ch=0.94,
                    eta_dis=0.94,
                    soc0=0.5,
                    soc_min=0.0,
                    soc_max=0.95,
                    self_discharge_rate=0.0008
                )
                
                bid_val = 720.0
                offer_cost = 300.0
        else:
            if load_type == "residential":
                bid_val = 580.0
            elif load_type == "commercial":
                bid_val = 750.0
            else:
                bid_val = 680.0
            offer_cost = 999.0

        agent = Agent(
            name=f"L{bus}_{load_type[:3]}{idx}",
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
            load_type=load_type
        )
        agents.append(agent)

    return agents


def day_ahead_price_china(T: int = 96) -> np.ndarray:
    hours = np.arange(T) * 0.25
    base = 420.0
    price = base + 300.0 * np.sin((hours - 9) / 24 * 2 * np.pi) \
            + 150.0 * np.sin((hours - 20) / 24 * 2 * np.pi)
    noon_dip = -80.0 * np.exp(-((hours - 13) ** 2) / 8)
    price += noon_dip
    price = np.clip(price, 150.0, 800.0)
    noise = np.random.normal(0, 30.0, size=T)
    price = price + noise
    price = np.clip(price, -50.0, 1200.0)
    return price