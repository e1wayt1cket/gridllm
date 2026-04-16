"""
第四轮修改:
  1. build_demo_case: 新增风电Agent F(WIND+ESS) + 使用新Network格式
  2. 新增 solve_pareto_front_epsilon() ε-约束法（任务2）
  3. 新增 build_scenarios() 多场景（任务3）
  4. 更新 random_actions / 演示函数适配新字段名
"""

with open("/mnt/c/Python/MakerB/agent/agent_trading.py", "r", encoding="utf-8") as f:
    content = f.read()

# ============================================================
# 1) 替换 build_demo_case 函数
# ============================================================
old_build = '''def build_demo_case(T=24) -> Tuple[List[Agent], Network, np.ndarray]:
    hours = np.arange(T)

    # 北京工业电价替代曲线（元/千瓦时）
    base_price = np.array([
        1.14758475, 1.13159475, 1.07758475, 1.03258475, 0.88410775,
        0.82860775, 1.50748175, 1.47349775, 1.43748175, 1.39248175,
        1.20401675, 1.14851675
    ])
    wholesale = np.tile(base_price, 2)

    # 预测与实际（简单加噪）
    def noisy(x, sigma=0.1):
        return np.clip(
            x * (1 + np.random.normal(0, sigma, size=x.shape)), 0, None)

    # 两个 prosumer (A,B) 在 bus1 和 bus2
    base_pv_A = np.clip(6 * np.sin((hours - 6) / 24 * 2 * np.pi), 0, None)
    base_pv_B = np.clip(5.5 * np.sin((hours - 7) / 24 * 2 * np.pi), 0, None)

    load_A = 2.0 + 0.5 * np.exp(-0.5 * ((hours - 10) / 3.0)**2)
    load_B = 1.8 + 0.4 * np.exp(-0.5 * ((hours - 15) / 3.5)**2)

    # 两个纯负荷 (C,D) 在 bus1/bus2
    load_C = 4.5 + 1.2 * np.exp(-0.5 * ((hours - 9) / 2.5)**2) + \\
        1.5 * np.exp(-0.5 * ((hours - 18) / 2.5)**2)
    load_D = 3.8 + 1.0 * np.exp(-0.5 * ((hours - 8) / 2.7)**2) + \\
        1.2 * np.exp(-0.5 * ((hours - 20) / 2.7)**2)

    A = Agent(
        name="A(RES+ESS)",
        bus=1,
        is_prosumer=True,
        load_forecast=noisy(
            load_A,
            0.05),
        pv_forecast=noisy(
            base_pv_A,
            0.12),
        load_real=noisy(
            load_A,
            0.08),
        pv_real=noisy(
            base_pv_A,
            0.18),
        bid_value=90.0,
        offer_cost=5.0,
        storage=StorageSpec(
            e_max=8.0,
            p_ch_max=2.5,
            p_dis_max=2.5,
            eta_ch=0.95,
            eta_dis=0.95,
            soc0=3.0,
            soc_min=0.8,
            soc_max=7.5))
    B = Agent(
        name="B(RES+ESS)",
        bus=2,
        is_prosumer=True,
        load_forecast=noisy(
            load_B,
            0.05),
        pv_forecast=noisy(
            base_pv_B,
            0.12),
        load_real=noisy(
            load_B,
            0.08),
        pv_real=noisy(
            base_pv_B,
            0.18),
        bid_value=88.0,
        offer_cost=6.0,
        storage=StorageSpec(
            e_max=6.0,
            p_ch_max=2.0,
            p_dis_max=2.0,
            eta_ch=0.94,
            eta_dis=0.94,
            soc0=2.5,
            soc_min=0.6,
            soc_max=5.6))
    C = Agent(
        name="C(LOAD)", bus=1, is_prosumer=False,
        load_forecast=noisy(load_C, 0.06), pv_forecast=np.zeros(T),
        load_real=noisy(load_C, 0.10), pv_real=np.zeros(T),
        bid_value=110.0, offer_cost=999.0,
        storage=None
    )
    D = Agent(
        name="D(LOAD)", bus=2, is_prosumer=False,
        load_forecast=noisy(load_D, 0.06), pv_forecast=np.zeros(T),
        load_real=noisy(load_D, 0.10), pv_real=np.zeros(T),
        bid_value=105.0, offer_cost=999.0,
        storage=None
    )

    E = Agent(
        name="E(LOAD)", bus=2, is_prosumer=False,
        load_forecast=noisy(load_D, 0.06), pv_forecast=np.zeros(T),
        load_real=noisy(load_D, 0.10), pv_real=np.zeros(T),
        bid_value=102.0, offer_cost=999.0,
        storage=None
    )

    # 简化配网容量（MW）：如果容量紧，会产生拥塞导致储能/本地PV价值凸显
    network = Network(cap01=6.5, cap12=3.5)

    return [A, B, C, D, E], network, wholesale'''

new_build = '''def build_demo_case(T=24, with_wind: bool = False) -> Tuple[List[Agent], Network, np.ndarray]:
    """
    构建演示算例（任务1扩展：支持风电Agent）

    参数：
      T: 时段数（默认24小时）
      with_wind: 是否包含风电Agent F（默认False保持向后兼容）

    返回：(agents, network, wholesale_price)
    """
    hours = np.arange(T)

    # 北京工业电价替代曲线（元/千瓦时）
    base_price = np.array([
        1.14758475, 1.13159475, 1.07758475, 1.03258475, 0.88410775,
        0.82860775, 1.50748175, 1.47349775, 1.43748175, 1.39248175,
        1.20401675, 1.14851675
    ])
    wholesale = np.tile(base_price, 2)

    # 预测与实际（简单加噪）
    def noisy(x, sigma=0.1):
        return np.clip(
            x * (1 + np.random.normal(0, sigma, size=x.shape)), 0, None)

    # ---- PV出力曲线（正弦模拟日间发电）----
    base_pv_A = np.clip(6 * np.sin((hours - 6) / 24 * 2 * np.pi), 0, None)
    base_pv_B = np.clip(5.5 * np.sin((hours - 7) / 24 * 2 * np.pi), 0, None)

    # ---- 风电出力曲线（任务1：Weibull风格，昼夜波动大）----
    # 风电特点：无明显日间峰，夜间也可能较大出力，随机波动强
    if with_wind:
        # 基础风功率：用多正弦叠加模拟昼夜变化 + 随机湍流
        np_rng = np.random.RandomState(42)
        base_wind_F = np.clip(
            8.0 * (0.5 + 0.5 * np.sin((hours - 3) / 12 * np.pi))   # 昼夜分量
            + 3.0 * np.sin((hours - 14) / 8 * np.pi)                  # 中午低谷
            + np_rng.normal(0, 1.5, size=T),                          # 随机湍流
            0, 15.0                                                    # 额定15MW
        )
        base_wind_F = np.maximum(base_wind_F, 0.3)  # 最小出力（避免完全无风）

    # ---- 负荷曲线 ----
    load_A = 2.0 + 0.5 * np.exp(-0.5 * ((hours - 10) / 3.0)**2)
    load_B = 1.8 + 0.4 * np.exp(-0.5 * ((hours - 15) / 3.5)**2)
    load_C = 4.5 + 1.2 * np.exp(-0.5 * ((hours - 9) / 2.5)**2) + \\\n        1.5 * np.exp(-0.5 * ((hours - 18) / 2.5)**2)
    load_D = 3.8 + 1.0 * np.exp(-0.5 * ((hours - 8) / 2.7)**2) + \\\n        1.2 * np.exp(-0.5 * ((hours - 20) / 2.7)**2)
    # 新增负荷E/F节点（用于多节点场景）
    load_E = load_D.copy()
    load_F_load = 3.0 + 0.8 * np.exp(-0.5 * ((hours - 12) / 3.0)**2) + \\\n        1.0 * np.exp(-0.5 * ((hours - 20) / 3.0)**2)

    # ---- Agent A: PV+储能 @ bus1 ----
    A = Agent(
        name="A(PV+ESS)", bus=1, is_prosumer=True,
        load_forecast=noisy(load_A, 0.05), pv_forecast=noisy(base_pv_A, 0.12),
        load_real=noisy(load_A, 0.08), pv_real=noisy(base_pv_A, 0.18),
        wind_forecast=None, wind_real=None,
        bid_value=90.0, offer_cost=5.0,
        storage=StorageSpec(e_max=8.0, p_ch_max=2.5, p_dis_max=2.5,
                            eta_ch=0.95, eta_dis=0.95, soc0=3.0, soc_min=0.8, soc_max=7.5))

    # ---- Agent B: PV+储能 @ bus2 ----
    B = Agent(
        name="B(PV+ESS)", bus=2, is_prosumer=True,
        load_forecast=noisy(load_B, 0.05), pv_forecast=noisy(base_pv_B, 0.12),
        load_real=noisy(load_B, 0.08), pv_real=noisy(base_pv_B, 0.18),
        wind_forecast=None, wind_real=None,
        bid_value=88.0, offer_cost=6.0,
        storage=StorageSpec(e_max=6.0, p_ch_max=2.0, p_dis_max=2.0,
                            eta_ch=0.94, eta_dis=0.94, soc0=2.5, soc_min=0.6, soc_max=5.6))

    # ---- Agent C: 纯负荷 @ bus1 ----
    C = Agent(name="C(LOAD)", bus=1, is_prosumer=False,
              load_forecast=noisy(load_C, 0.06), pv_forecast=np.zeros(T),
              load_real=noisy(load_C, 0.10), pv_real=np.zeros(T),
              wind_forecast=None, wind_real=None,
              bid_value=110.0, offer_cost=999.0, storage=None)

    # ---- Agent D: 纯负荷 @ bus2 ----
    D = Agent(name="D(LOAD)", bus=2, is_prosumer=False,
              load_forecast=noisy(load_D, 0.06), pv_forecast=np.zeros(T),
              load_real=noisy(load_D, 0.10), pv_real=np.zeros(T),
              wind_forecast=None, wind_real=None,
              bid_value=105.0, offer_cost=999.0, storage=None)

    # ---- Agent E: 纯负荷 @ bus2 ----
    E = Agent(name="E(LOAD)", bus=2, is_prosumer=False,
              load_forecast=noisy(load_E, 0.06), pv_forecast=np.zeros(T),
              load_real=noisy(load_E, 0.10), pv_real=np.zeros(T),
              wind_forecast=None, wind_real=None,
              bid_value=102.0, offer_cost=999.0, storage=None)

    agents_base = [A, B, C, D, E]

    # ---- 任务1：新增风电Agent F(WIND+ESS) ----
    if with_wind:
        F = Agent(
            name="F(WIND+ESS)", bus=3, is_prosumer=True,
            load_forecast=noisy(load_F_load, 0.05), pv_forecast=np.zeros(T),
            load_real=noisy(load_F_load, 0.08), pv_real=np.zeros(T),
            wind_forecast=noisy(base_wind_F, 0.10),     # 风电预测
            wind_real=noisy(base_wind_F, 0.20),         # 风电实际（更大误差）
            bid_value=85.0,                              # 风电边际成本低
            offer_cost=3.0,                              # 风电机会成本更低
            storage=StorageSpec(e_max=10.0, p_ch_max=4.0, p_dis_max=4.0,
                                eta_ch=0.96, eta_dis=0.96, soc0=5.0,
                                soc_min=1.0, soc_max=9.5))
        agents_base.append(F)

        # 如果有风电，增加一个纯负荷G来平衡bus3（可选）
        G = Agent(name="G(LOAD)", bus=3, is_prosumer=False,
                  load_forecast=noisy(load_F_load * 0.8, 0.06), pv_forecast=np.zeros(T),
                  load_real=noisy(load_F_load * 0.8, 0.10), pv_real=np.zeros(T),
                  wind_forecast=None, wind_real=None,
                  bid_value=100.0, offer_cost=999.0, storage=None)
        agents_base.append(G)

    # ---- 网络（任务3：使用新的通用结构）----
    if with_wind:
        # 5节点网络: 0(电网)-1-2-3-4
        network = Network.from_edges([
            (0, 1, 8.0),   # cap01 = 8 MW
            (1, 2, 5.0),   # cap12 = 5 MW
            (2, 3, 6.0),   # cap23 = 6 MW
            (3, 4, 4.0),   # cap34 = 4 MW
        ])
    else:
        # 兼容旧版3节点
        network = Network.simple_2bus(cap01=6.5, cap12=3.5)

    return agents_base, network, wholesale'''

content = content.replace(old_build, new_build)

with open("/mnt/c/Python/MakerB/agent/agent_trading.py", "w", encoding="utf-8") as f:
    f.write(content)

print("build_demo_case: 加风电Agent F + 新Network ✓")
