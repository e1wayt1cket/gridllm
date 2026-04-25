# scenarios.py
import numpy as np
from typing import List, Tuple

from models import MarketConfig, Agent
from grid import build_base_network, create_agents_from_network, day_ahead_price_china


def _build_base(config: MarketConfig, T: int, with_wind: bool) -> Tuple[List[Agent], np.ndarray]:
    """通用构建：网络 + 智能体 + 电价"""
    net = build_base_network(config)
    agents = create_agents_from_network(net, T, with_wind=with_wind)
    wholesale = day_ahead_price_china(T)
    return agents, wholesale


# ======================== 场景函数 ========================
def scenario_baseline(T: int = 96) -> Tuple[List[Agent], np.ndarray]:
    """baseline：基准风光储"""
    config = MarketConfig(use_ac_opf=False)
    return _build_base(config, T, with_wind=True)


def scenario_no_pv(T: int = 96) -> Tuple[List[Agent], np.ndarray]:
    """no_pv：仅有风电，光伏置零"""
    config = MarketConfig(use_ac_opf=False)
    agents, wholesale = _build_base(config, T, with_wind=True)
    for a in agents:
        if a.is_prosumer:
            a.pv_forecast = np.zeros(T)
            a.pv_real = np.zeros(T)
    return agents, wholesale


def scenario_high_re(T: int = 96, multiplier: float = 2.0) -> Tuple[List[Agent], np.ndarray]:
    """high_re：高可再生渗透（光伏/风电翻倍）"""
    config = MarketConfig(use_ac_opf=False)
    agents, wholesale = _build_base(config, T, with_wind=True)
    for a in agents:
        if a.is_prosumer:
            a.pv_forecast *= multiplier
            a.pv_real *= multiplier
        if a.has_wind:
            a.wind_forecast *= multiplier   #type: ignore
            a.wind_real *= multiplier       #type: ignore
    return agents, wholesale


def scenario_peak_load(T: int = 96, multiplier: float = 1.8) -> Tuple[List[Agent], np.ndarray]:
    """peak_load：所有负荷增大"""
    config = MarketConfig(use_ac_opf=False)
    agents, wholesale = _build_base(config, T, with_wind=True)
    for a in agents:
        a.load_forecast *= multiplier
        a.load_real *= multiplier
    return agents, wholesale


def scenario_unbalanced(T: int = 96) -> Tuple[List[Agent], np.ndarray]:
    """unbalanced：移除全部商业负荷"""
    config = MarketConfig(use_ac_opf=False)
    agents, wholesale = _build_base(config, T, with_wind=True)
    agents = [a for a in agents if "commercial" not in a.load_type]
    return agents, wholesale


def scenario_low_load_high_re(T: int = 96, load_factor: float = 0.3, re_factor: float = 3.0) -> Tuple[List[Agent], np.ndarray]:
    """low_load_high_re：极低负荷+极大新能源"""
    config = MarketConfig(use_ac_opf=False)
    agents, wholesale = _build_base(config, T, with_wind=True)
    for a in agents:
        a.load_forecast *= load_factor
        a.load_real *= load_factor
        if a.is_prosumer:
            a.pv_forecast *= re_factor
            a.pv_real *= re_factor
        if a.has_wind:
            a.wind_forecast *= re_factor #type: ignore
            a.wind_real *= re_factor    #type: ignore
    return agents, wholesale


def scenario_high_load_low_re(T: int = 96, load_factor: float = 2.0, re_factor: float = 0.2) -> Tuple[List[Agent], np.ndarray]:
    """high_load_low_re：极大负荷+极低新能源"""
    config = MarketConfig(use_ac_opf=False)
    agents, wholesale = _build_base(config, T, with_wind=True)
    for a in agents:
        a.load_forecast *= load_factor
        a.load_real *= load_factor
        if a.is_prosumer:
            a.pv_forecast *= re_factor
            a.pv_real *= re_factor
        if a.has_wind:
            a.wind_forecast *= re_factor #type: ignore
            a.wind_real *= re_factor #type: ignore
    return agents, wholesale


def scenario_congestion(T: int = 96, capacity_mult: float = 0.5) -> Tuple[List[Agent], np.ndarray]:
    """congestion：线路容量下降至50%，无负荷变化"""
    config = MarketConfig(use_ac_opf=False, line_capacity_multiplier=capacity_mult)
    return _build_base(config, T, with_wind=True)


def scenario_re_ramp_drop(T: int = 96) -> Tuple[List[Agent], np.ndarray]:
    """re_ramp_drop：新能源骤降（中期突降至10%）"""
    return _scenario_re_ramp_event(T, "sudden_drop")


def scenario_re_ramp_surge(T: int = 96) -> Tuple[List[Agent], np.ndarray]:
    """re_ramp_surge：新能源骤升（中期突增至正常）"""
    return _scenario_re_ramp_event(T, "sudden_surge")


def _scenario_re_ramp_event(T: int, ramp_type: str) -> Tuple[List[Agent], np.ndarray]:
    """内部实现：爬坡事件"""
    config = MarketConfig(use_ac_opf=False)
    agents, wholesale = _build_base(config, T, with_wind=True)

    for a in agents:
        if not (a.is_prosumer or a.has_wind):
            continue

        # 保存原始曲线（取当前值）
        pv_orig = a.pv_forecast.copy()
        wind_orig = a.wind_forecast.copy() if a.has_wind and a.wind_forecast is not None else np.zeros(T)

        # 生成变化的曲线
        pv_new = np.zeros(T)
        wind_new = np.zeros(T)
        mid = T // 2

        if ramp_type == "sudden_drop":
            pv_new[:mid] = pv_orig[:mid]
            pv_new[mid:] = pv_orig[mid:] * 0.1
            if a.has_wind:
                wind_new[:mid] = wind_orig[:mid]
                wind_new[mid:] = wind_orig[mid:] * 0.1
        else:  # sudden_surge
            pv_new[:mid] = pv_orig[:mid] * 0.1
            pv_new[mid:] = pv_orig[mid:]
            if a.has_wind:
                wind_new[:mid] = wind_orig[:mid] * 0.1
                wind_new[mid:] = wind_orig[mid:]

        a.pv_forecast = pv_new
        a.pv_real = pv_new * 0.95
        if a.has_wind:
            a.wind_forecast = wind_new
            a.wind_real = wind_new * 0.95

    return agents, wholesale


def scenario_peak_congestion(T: int = 96, load_factor: float = 1.8, capacity_mult: float = 0.5) -> Tuple[List[Agent], np.ndarray]:
    """peak_congestion：高峰负荷 + 线路阻塞"""
    config = MarketConfig(use_ac_opf=False, line_capacity_multiplier=capacity_mult)
    agents, wholesale = _build_base(config, T, with_wind=True)
    for a in agents:
        a.load_forecast *= load_factor
        a.load_real *= load_factor
    return agents, wholesale


# ======================== 场景注册表 ========================
SCENARIO_REGISTRY = {
    "baseline": scenario_baseline,
    "no_pv": scenario_no_pv,
    "high_re": scenario_high_re,
    "peak_load": scenario_peak_load,
    "unbalanced": scenario_unbalanced,
    "low_load_high_re": scenario_low_load_high_re,
    "high_load_low_re": scenario_high_load_low_re,
    "congestion": scenario_congestion,
    "re_ramp_drop": scenario_re_ramp_drop,
    "re_ramp_surge": scenario_re_ramp_surge,
    "peak_congestion": scenario_peak_congestion,
}


def get_scenario(name: str, T: int = 96) -> Tuple[List[Agent], np.ndarray]:
    """按名称获取场景，返回 (agents, wholesale)"""
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"未知场景 '{name}'，可选：{list(SCENARIO_REGISTRY.keys())}")
    return SCENARIO_REGISTRY[name](T=T)


def list_scenarios() -> List[str]:
    """列出所有可用场景名称"""
    return list(SCENARIO_REGISTRY.keys())


def print_scenario_info():
    """打印所有场景的描述信息"""
    desc = {
        "baseline": "基准-风光储：居民/商业/工业，光伏、风电、储能",
        "no_pv": "无光伏-仅风电：光伏置零",
        "high_re": "高可再生：光伏、风电容量翻倍",
        "peak_load": "高峰负荷：负荷×1.8",
        "unbalanced": "资源不平衡：移除商业负荷",
        "low_load_high_re": "极低负荷+极大新能源：负荷30%，新能源3倍",
        "high_load_low_re": "极大负荷+极低新能源：负荷2倍，新能源20%",
        "congestion": "网络阻塞：线路容量50%",
        "re_ramp_drop": "新能源骤降：中期突降至10%",
        "re_ramp_surge": "新能源骤升：中期突增至正常",
        "peak_congestion": "高峰负荷+网络阻塞：负荷×1.8+线路容量50%",
    }
    print("=" * 60)
    print("可用场景")
    print("=" * 60)
    for key in SCENARIO_REGISTRY:
        print(f"  {key:<20s}: {desc.get(key, '')}")
    print("=" * 60)