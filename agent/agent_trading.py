import numpy as np
import copy
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any, Callable
import pandapower as pp
import pandapower.networks as pn
import warnings


warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*MessageStream size changed.*")

np.random.seed(1)


# ========================= 配置类 =========================
@dataclass
class MarketConfig:
    """
    Market configuration
    """
    # 默认使用 DC OPF（AC OPF 对 storage 兼容性差且速度慢）
    use_ac_opf: bool = False
    opf_tolerance: float = 1e-6
    opf_max_iter: int = 100
    verbose: bool = False
    # offer_adder 不允许负值，防止售电成本为负
    bid_mult_range: Tuple[float, float] = (0.8, 1.2)
    offer_adder_range: Tuple[float, float] = (0.0, 50.0)
    default_bid_mult: float = 1.0
    default_offer_adder: float = 0.0
    line_capacity_multiplier: float = 3.0
    w_re_consume: float = 0.0
    penalty_unserved: float = 800.0
    base_mva: float = 1.0
    base_kv: float = 12.66


# ========================= 储能参数 =========================
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
    self_discharge_rate: float = 0.001  # 自放电率（每小时约 0.1%，每天约 2.4%）


# ========================= 智能体 =========================
@dataclass
class Agent:
    name: str
    bus: int
    is_prosumer: bool
    load_forecast: np.ndarray   # (T,)
    pv_forecast: np.ndarray
    load_real: np.ndarray
    pv_real: np.ndarray
    bid_value: float            # 购电意愿价格 (元/MWh)
    offer_cost: float           # 售电边际成本 (元/MWh)
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


# ========================= 网络构建 =========================
# [修改3] 恢复签名：接受 config 而不是 T，使用 config.line_capacity_multiplier
def build_base_network(config: MarketConfig) -> pp.pandapowerNet:
    net = pn.case33bw()
    net.line['max_i_ka'] = net.line['max_i_ka'].fillna(1.0) * config.line_capacity_multiplier
    return net


# ========================= 智能体与标准网络负荷映射 =========================
def create_agents_from_network(net: pp.pandapowerNet, T: int,
                               with_wind: bool = False) -> List[Agent]:
    """
    将 IEEE 33 节点中的原始负荷转换为智能体。
    根据负荷母线的位置，分配居民/商业/工业类型，并生成光伏/风电预测数据。
    - IEEE 33节点总负荷约 3.7 MW，典型配电网规模
    - 居民负荷：0.05-0.15 MW/户，日均用电量 10-30 kWh
    - 商业负荷：0.1-0.3 MW/户，日均用电量 50-100 kWh  
    - 工业负荷：0.2-0.5 MW/户，日均用电量 200-500 kWh
    - 光伏配置：产消者光伏容量通常为负荷峰值的 1.5-2 倍
    - 风电配置：工业风电容量通常为负荷峰值的 2-3 倍
    - 储能配置：容量为负荷峰值的 1-2 倍，充放电功率为容量的 0.25-0.5C
    """
    hours = np.arange(T)

    def pv_profile():
        """
        光伏出力曲线（标幺值）
        - 峰值在中午12点（hours=12）
        - 日出约6点，日落约18点
        - 峰值系数1.0（考虑天气波动可达1.2）
        """
        return np.clip(1.0 * np.sin((hours - 6) / 24 * 2 * np.pi), 0, None)

    def wind_profile():
        """
        风电出力曲线（标幺值）
        - 考虑昼夜风速变化和风湍流
        - 平均利用小时数约 2000-2500 小时/年
        - 容量系数约 0.25-0.35
        """
        rng = np.random.RandomState(42)
        w = np.clip(
            0.7 * (0.5 + 0.5 * np.sin((hours - 3) / 12 * np.pi))  # 基础风型
            + 0.2 * np.sin((hours - 14) / 8 * np.pi)               # 日间波动
            + rng.normal(0, 0.12, size=T),                         # 随机扰动
            0, 1.3
        )
        return np.maximum(w, 0.05)  # 最小出力5%

    def noisy(x, sigma=0.1):
        return np.clip(x * (1 + np.random.normal(0, sigma, size=x.shape)), 0, None)

    agents = []
    pv_base = pv_profile()
    wind_base = wind_profile() if with_wind else None

    for idx, load in net.load.iterrows():
        bus = load.bus
        p_mw = load.p_mw  # IEEE 33节点原始负荷（MW）

        # 负荷类别分配（基于母线位置）
        if bus in [0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21]:
            load_type = "residential"
        elif bus in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
            load_type = "commercial"
        else:
            load_type = "industrial"
        if load_type == "residential":
            # 居民负荷：保持原始或轻微放大（0.04-0.10 MW）
            base_load = p_mw * 0.9
        elif load_type == "commercial":
            # 商业负荷：保持原始或轻微放大（0.10-0.25 MW）
            base_load = p_mw * 1.0
        else:  # industrial
            # 工业负荷：轻微放大（0.15-0.35 MW）
            base_load = p_mw * 1.1
        
        # 负荷预测和实际值（添加预测误差）
        # 日前预测误差约 5-8%，实时偏差约 10-15%
        load_forecast = noisy(np.full(T, base_load), 0.06)
        load_real = noisy(np.full(T, base_load), 0.12)

        # [修改4] 恢复产消者数量为 4 个（避免过度扩展导致 OPF 不稳定）
        is_prosumer = (load_type == "residential" and bus in [5, 12]) or \
                      (load_type == "industrial" and with_wind and bus in [22, 25])

        pv_forecast = np.zeros(T)
        pv_real = np.zeros(T)
        wind_forecast = None
        wind_real = None
        storage = None

        if is_prosumer:
            if load_type == "residential":
                # 居民产消者：光伏+储能
                pv_cap = base_load * 1.8
                pv_forecast = noisy(pv_base * pv_cap, 0.15)
                pv_real = noisy(pv_base * pv_cap, 0.20)
                
                # 储能配置
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
                
            else:  # industrial + wind
                # 工业产消者：风电+储能
                wind_cap = base_load * 2.5
                if wind_base is not None:
                    wind_forecast = noisy(wind_base * wind_cap, 0.12)
                    wind_real = noisy(wind_base * wind_cap, 0.18)
                else:
                    wind_forecast = np.zeros(T)
                    wind_real = np.zeros(T)
                
                # 储能配置
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
            # 纯负荷：根据中国实际电价设置购电意愿
            if load_type == "residential":
                bid_val = 580.0  # 居民电价 ~0.58 元/kWh
            elif load_type == "commercial":
                bid_val = 750.0  # 商业电价 ~0.75 元/kWh（商业电价最高）
            else:  # industrial
                bid_val = 680.0  # 工业电价 ~0.68 元/kWh（大工业有优惠）
            offer_cost = 999.0  # 纯负荷不卖电（设置为极高值）

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


# ========================= [修改6] 重构：时序数据管理器 =========================
class TimestepDataManager:
    """
    修复版：根据 determine_storage_mode() 结果，只创建单一方向元件。
    charge -> 可控负荷(load), discharge -> 可控发电机(sgen), idle -> 无元件
    """
    def __init__(self, net: pp.pandapowerNet, agents: List[Agent], t: int, stage: str,
                 prev_soc: Optional[Dict[str, float]] = None,
                 wholesale_t: float = 400.0,
                 action_params: Optional[Dict[str, Dict]] = None,
                 config: Optional[MarketConfig] = None):
        self.net = net
        self.agents = agents
        self.t = t
        self.stage = stage
        self.prev_soc = prev_soc if prev_soc is not None else {}
        self.wholesale_t = wholesale_t
        self.action_params = action_params or {}
        self.config = config or MarketConfig()
        self.storage_mode_map: Dict[str, str] = {}

        net.load.drop(net.load.index, inplace=True)
        net.sgen.drop(net.sgen.index, inplace=True)
        if hasattr(net, 'storage'):
            net.storage.drop(net.storage.index, inplace=True)
        if hasattr(net, 'poly_cost'):
            net.poly_cost.drop(net.poly_cost.index, inplace=True)

        self.load_idx_map: Dict[str, int] = {}
        self.pv_idx_map: Dict[str, int] = {}
        self.wind_idx_map: Dict[str, int] = {}
        self.storage_idx_map: Dict[str, int] = {}

        self._attach_all()

    def _attach_all(self):
        for a in self.agents:
            bus = a.bus
            load_val = a.load_forecast[self.t] if self.stage == "DA" else a.load_real[self.t]
            idx = pp.create_load(self.net, bus=bus, p_mw=load_val, q_mvar=0,
                                 name=f"{a.name}_load", controllable=False)
            self.load_idx_map[a.name] = int(idx)  # type: ignore[arg-type]

            if a.is_prosumer:
                pv_val = a.pv_forecast[self.t] if self.stage == "DA" else a.pv_real[self.t]
                if pv_val > 0:
                    idx = pp.create_sgen(self.net, bus=bus, p_mw=pv_val, q_mvar=0,
                                         min_p_mw=0, max_p_mw=pv_val,
                                         name=f"{a.name}_pv", controllable=True)
                    self.net.sgen.loc[idx, ['min_q_mvar', 'max_q_mvar']] = 0.0, 0.0
                    self.pv_idx_map[a.name] = int(idx)  # type: ignore[arg-type]

                if a.has_wind:
                    wind_val = a.get_wind_forecast()[self.t] if self.stage == "DA" else a.get_wind_real()[self.t]
                    if wind_val > 0:
                        idx = pp.create_sgen(self.net, bus=bus, p_mw=wind_val, q_mvar=0,
                                             min_p_mw=0, max_p_mw=wind_val,
                                             name=f"{a.name}_wind", controllable=True)
                        self.net.sgen.loc[idx, ['min_q_mvar', 'max_q_mvar']] = 0.0, 0.0
                        self.wind_idx_map[a.name] = int(idx)  # type: ignore[arg-type]

                # [修改6续] 储能：单一元件建模
                if a.storage:
                    current_soc = self.prev_soc.get(a.name, a.storage.soc0)
                    mode = determine_storage_mode(
                        a, self.wholesale_t, current_soc,
                        self.config, self.action_params.get(a.name)
                    )
                    self.storage_mode_map[a.name] = mode

                    if mode == "charge":
                        idx = pp.create_load(
                            self.net, bus=bus, p_mw=0.0, q_mvar=0,
                            min_p_mw=0.0, max_p_mw=a.storage.p_ch_max,
                            name=f"{a.name}_storage", controllable=True
                        )
                        self.net.load.loc[idx, ['min_q_mvar', 'max_q_mvar']] = 0.0, 0.0
                        self.storage_idx_map[a.name] = int(idx)  # type: ignore[arg-type]
                    elif mode == "discharge":
                        idx = pp.create_sgen(
                            self.net, bus=bus, p_mw=0.0, q_mvar=0,
                            min_p_mw=0.0, max_p_mw=a.storage.p_dis_max,
                            name=f"{a.name}_storage", controllable=True
                        )
                        self.net.sgen.loc[idx, ['min_q_mvar', 'max_q_mvar']] = 0.0, 0.0
                        self.storage_idx_map[a.name] = int(idx)  # type: ignore[arg-type]
                    # else idle: do nothing

    def get_storage_mode(self, agent_name: str) -> str:
        return self.storage_mode_map.get(agent_name, "idle")


# ========================= 中国大陆日前电价曲线生成 =========================
def day_ahead_price_china(T: int = 96) -> np.ndarray:
    
    hours = np.arange(T) * 0.25
    
    # 基准价格曲线
    base = 420.0   # 基准均价
    
    # 双峰曲线：上午高峰（9-11点）和晚高峰（18-21点）
    price = base + 150.0 * np.sin((hours - 9) / 24 * 2 * np.pi) \
            + 130.0 * np.sin((hours - 20) / 24 * 2 * np.pi)
    
    # 午间低谷（12-14点，光伏大发时段）
    noon_dip = -80.0 * np.exp(-((hours - 13) ** 2) / 8)
    price += noon_dip
    
    # 价格范围初步控制
    price = np.clip(price, 150.0, 800.0)
    
    # 添加噪声
    noise = np.random.normal(0, 30.0, size=T)
    price = price + noise
    
    # 最终限制：允许负电价和极端高价
    price = np.clip(price, -50.0, 1200.0)
    
    return price


# ========================= [修改5] 新增：储能模式决策 =========================
def determine_storage_mode(
    agent: Agent,
    wholesale_t: float,
    prev_soc: float,
    config: MarketConfig,
    action_params: Optional[Dict] = None
) -> str:
    """
    根据电价、SOC状态决定储能行为模式。
    在OPF求解前确定单一行为方向，避免线性OPF的充放电互斥问题。
    返回: "charge" | "discharge" | "idle"
    """
    storage = agent.storage
    if storage is None:
        return "idle"

    ap = action_params or {}
    bid_mult = ap.get("bid_mult", 1.0)
    offer_adder = ap.get("offer_adder", 0.0)

    # 物理边界检查
    if prev_soc <= storage.soc_min + 0.02:
        if wholesale_t < agent.bid_value * bid_mult * 0.9:
            return "charge"
        return "idle"

    if prev_soc >= storage.soc_max - 0.02:
        if wholesale_t > agent.offer_cost + offer_adder:
            return "discharge"
        return "idle"

    # 经济信号
    round_trip_eff = storage.eta_ch * storage.eta_dis
    charge_threshold = agent.bid_value * bid_mult * round_trip_eff * 0.85
    discharge_threshold = (agent.offer_cost + offer_adder) / round_trip_eff * 1.15

    # SOC调节因子
    soc_range = storage.soc_max - storage.soc_min
    soc_ratio = (prev_soc - storage.soc_min) / soc_range if soc_range > 0 else 0.5
    charge_threshold *= (0.7 + 0.6 * soc_ratio)
    discharge_threshold *= (1.3 - 0.6 * soc_ratio)

    if wholesale_t < charge_threshold and prev_soc < storage.soc_max * 0.98:
        return "charge"
    elif wholesale_t > discharge_threshold and prev_soc > storage.soc_min * 1.02:
        return "discharge"
    else:
        return "idle"


# ========================= [修改7] 重构：成本函数 =========================
def setup_cost_functions_weighted(
    net: pp.pandapowerNet,
    agents: List[Agent],
    action_params: Dict[str, Dict],
    config: MarketConfig,
    external_price: float
):
    """
    修复版：储能成本全部使用正值，消除套利空间。
    充电负荷成本 = 愿意支付的价格（正值）
    放电sgen成本 = 发电边际成本（正值）
    """
    pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=external_price)
    re_incentive = config.w_re_consume * 10.0

    # 光伏/风电
    for sgen in net.sgen.itertuples():
        name = sgen.name
        if '_pv' in name or '_wind' in name:
            agent_name = name.replace('_pv', '').replace('_wind', '')
            ap = action_params.get(agent_name, {})
            a = next((ag for ag in agents if ag.name == agent_name), None)
            if a is None:
                continue
            offer_adder = ap.get("offer_adder", 0.0)
            base_cost = a.offer_cost + offer_adder
            net_cost = base_cost - re_incentive
            pp.create_poly_cost(net, sgen.Index, 'sgen', cp1_eur_per_mw=net_cost)

    # 储能放电（sgen类型）
    for sgen in net.sgen.itertuples():
        name = sgen.name
        if '_storage' in name:
            agent_name = name.replace('_storage', '')
            ap = action_params.get(agent_name, {})
            a = next((ag for ag in agents if ag.name == agent_name), None)
            if a is None or not a.storage:
                continue
            offer_adder = ap.get("offer_adder", 0.0)
            # 放电成本：低于外部电价（能卖出去），但高于 offer_cost（有利润）
            discharge_cost = max(a.offer_cost + offer_adder, external_price * 0.88)
            discharge_cost = max(discharge_cost, a.offer_cost * 1.05)
            pp.create_poly_cost(net, sgen.Index, 'sgen',
                                cp1_eur_per_mw=discharge_cost, cp0_eur=0)

    # 储能充电（load类型）
    for load in net.load.itertuples():
        name = load.name
        if '_storage' in name:
            agent_name = name.replace('_storage', '')
            ap = action_params.get(agent_name, {})
            a = next((ag for ag in agents if ag.name == agent_name), None)
            if a is None or not a.storage:
                continue
            bid_mult = ap.get("bid_mult", 1.0)
            # 充电成本：低于外部电价（比买电划算），但为正值
            charge_cost = min(a.bid_value * bid_mult * 0.85, external_price * 0.92)
            charge_cost = max(charge_cost, 50.0)
            pp.create_poly_cost(net, load.Index, 'load',
                                cp1_eur_per_mw=charge_cost, cp0_eur=0)



# ========================= 输出辅助 =========================
def calc_load_satisfaction(result: Dict, agents: List[Agent], stage: str) -> float:
    total_load = sum(np.sum(a.load_forecast if stage == "DA" else a.load_real) for a in agents)
    total_served = sum(np.sum(result["schedules"][a.name]["served"]) for a in agents)
    return (total_served / total_load * 100) if total_load > 0 else 100.0


def print_summary(da: Dict, rt: Dict, payment: Dict, agents: List[Agent]):
    print("\n" + "=" * 70)
    print("市场出清结果")
    print("=" * 70)
    print(f"{'指标':<20} {'日前(DA)':>15} {'实时(RT)':>15}")
    print("-" * 50)
    print(f"{'社会福利(¥)':<20} {da['welfare']:>15.2f} {rt['welfare']:>15.2f}")
    print(f"{'可再生消纳率(%)':<20} {da['re_consumption_rate']:>15.1f} {rt['re_consumption_rate']:>15.1f}")
    da_load = calc_load_satisfaction(da, agents, 'DA')
    rt_load = calc_load_satisfaction(rt, agents, 'RT')
    print(f"{'负荷满足率(%)':<20} {da_load:>15.1f} {rt_load:>15.1f}")

    print("\n结算结果 (正数=成本, 负数=收益):")
    print("-" * 50)
    total = sum(payment.values())
    for name, val in payment.items():
        ptype = "成本" if val > 0 else "收益" if val < 0 else "平衡"
        print(f"{name:14s}  {val:10.2f} ¥ ({ptype})")
    print(f"{'总计':14s}  {total:10.2f} ¥")


def print_snapshots(agents: List[Agent], da: Dict, rt: Dict):
    print("\n各智能体运行快照 (显示部分):")
    print("-" * 70)
    for a in agents[:8]:  # 只显示前8个，避免刷屏
        s_da, s_rt = da["schedules"][a.name], rt["schedules"][a.name]
        print(f"\n🔹 {a.name} @ Bus {a.bus} ({a.load_type})")
        if a.is_prosumer:
            print(f"  PV: DA {np.sum(s_da['pv_used']):.2f} MWh")
        print(f"  净购电: DA {np.sum(s_da['p_buy'])-np.sum(s_da['p_sell']):.2f} MWh")


# ========================= 主入口 =========================
if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    config = MarketConfig(
        use_ac_opf="--ac" in args,  # 默认使用DC OPF，只有明确指定--ac才使用AC OPF
        verbose="--verbose" in args,
        w_re_consume=2.0 if "--green" in args else 0.0
    )
    T = 24 * 4  # 96 个 15 分钟时段

    net = build_base_network(config)
    agents = create_agents_from_network(net, T, with_wind="--wind" in args)

    print("=" * 70)
    print(f"IEEE 33节点电力市场模拟 (智能体 {len(agents)} 个, AC OPF: {config.use_ac_opf})")
    print(f"可再生能源激励系数: {config.w_re_consume}")
    print("=" * 70)

    da_actions = random_actions(agents, config)
    rt_actions = random_actions(agents, config)

    da = clear_market(agents, T, "DA", da_actions, config)
    rt = clear_market(agents, T, "RT", rt_actions, config)
    payment = two_settlement(agents, da, rt)

    print_summary(da, rt, payment, agents)
    print_snapshots(agents, da, rt)
    print("\n模拟完成")