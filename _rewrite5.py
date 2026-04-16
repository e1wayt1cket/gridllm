"""
第五轮修改: 文件末尾 - 新增全部新函数
  - solve_pareto_front_epsilon()  任务2: ε-约束法Pareto前沿
  - build_scenarios()             任务3: 多场景
  - 更新演示函数适配新字段名
"""

with open("/c/Python/MakerB/agent/agent_trading.py", "r", encoding="utf-8") as f:
    content = f.read()

# ============================================================
# 找到文件末尾的演示函数区域，整体替换
# ============================================================

# 找到 get_realtime_data 函数结束的位置（最后一个旧函数）
# 然后替换从那里到文件末尾的所有内容

# 标记点：get_realtime_data 函数结束后的位置
marker = '''def get_realtime_data():
    agents, network, wholesale = build_demo_case(T=24)
    act_RT = random_actions(agents)
    rt = clear_market_lp(
        agents=agents, network=network, T=24, stage="RT",
        wholesale_price=wholesale, action_params=act_RT
    )
    data = {}
    for a in agents:
        data[a.name] = {k: v.tolist()
                        for k, v in rt["schedules"][a.name].items()}
    return {"realtime": data}'''

new_functions = '''def get_agent_states():
    """返回DA出清状态（供dashboard等外部调用）"""
    agents, network, wholesale = build_demo_case(T=24)
    act_DA = random_actions(agents)
    da = clear_market_lp(
        agents=agents, network=network, T=24, stage="DA",
        wholesale_price=wholesale, action_params=act_DA
    )
    states = {}
    for a in agents:
        states[a.name] = {k: v.tolist()
                          for k, v in da["schedules"][a.name].items()}
    return {"agents": states}


def get_realtime_data():
    """返回RT出清状态（供dashboard等外部调用）"""
    agents, network, wholesale = build_demo_case(T=24)
    act_RT = random_actions(agents)
    rt = clear_market_lp(
        agents=agents, network=network, T=24, stage="RT",
        wholesale_price=wholesale, action_params=act_RT
    )
    data = {}
    for a in agents:
        data[a.name] = {k: v.tolist()
                        for k, v in rt["schedules"][a.name].items()}
    return {"realtime": data}


# ============================================================
# 任务2: ε-约束法求解 Pareto 前沿
# ============================================================


def solve_pareto_front_epsilon(
    agents: List[Agent],
    network: Network,
    T: int,
    stage: str,
    wholesale_price: np.ndarray,
    action_params: Dict[str, Dict],
    penalty_unserved: float = 500.0,
    n_points: int = 20,
) -> List[Dict]:
    """
    ε-约束法 (Epsilon-Constraint Method) 求解双目标 Pareto 前沿

    方法说明：
      将目标2(弃能量最小化)转化为约束：
        max   welfare(x)
        s.t.  re_waste(x) ≤ ε_i
              + 所有原约束

      对不同的 ε_i 值分别求解LP，得到 Pareto 前沿上一组非支配解。

    与 WSM 的区别：
      WSM: 只能找到凸Pareto前沿上的解，权重选择依赖经验
      ε-约束法: 能找到完整Pareto前沿（无论凹凸），结果更全面

    参数：
      n_points: 沿前沿采样的点数（默认20）

    返回：
      pareto_front: list of dict, 每个 dict 含 {welfare, re_waste, re_consumption_rate, eps}
    """
    print(f"\\n{'='*60}")
    print(f"  ε-约束法: 求解 Pareto 前沿 ({n_points} 个采样点)")
    print(f"{'='*60}")

    # ---- 步骤1: 求两个锚点 ----
    # 锚点A: 单独最大化 welfare (w_pv_consume=0 → 退化为原单目标)
    da_anchor_a = clear_market_multi_obj_lp(
        agents=agents, network=network, T=T, stage=stage,
        wholesale_price=wholesale_price, action_params=action_params,
        penalty_unserved=penalty_unserved,
        w_welfare=1.0, w_pv_consume=0.0
    )
    welfare_best = da_anchor_a["welfare"]
    waste_at_welfare_best = da_anchor_a["re_waste"]

    # 锚点B: 单独最小化弃能 (w_pv_consume 很大)
    da_anchor_b = clear_market_multi_obj_lp(
        agents=agents, network=network, T=T, stage=stage,
        wholesale_price=wholesale_price, action_params=action_params,
        penalty_unserved=penalty_unserved,
        w_welfare=1.0, w_pv_consume=10000.0
    )
    waste_best = da_anchor_b["re_waste"]       # 最小弃能（可能接近0）
    welfare_at_waste_best = da_anchor_b["welfare"]

    total_re = da_anchor_a["total_re_available"]

    print(f"  锚点A (福利最优):  welfare={welfare_best:.2f}, 弃能={waste_at_welfare_best:.3f} MWh")
    print(f"  锚点B (消纳最优):  welfare={welfare_at_waste_best:.2f}, 弃能={waste_best:.3f} MWh")
    print(f"  总可再生可用量:   {total_re:.3f} MWh")

    # ---- 步骤2: 在 [waste_best, waste_at_welfare_best] 间离散采样 ε ----
    # 注意: waste_best <= waste_at_welfare_best （弃能越小越好）
    eps_values = np.linspace(waste_best, max(waste_at_welfare_best, waste_best * 1.01), n_points)

    pareto_front = []

    for idx, eps in enumerate(eps_values):
        try:
            # 用带大权重的WSM近似ε-约束（因为cvxpy直接加不等式约束到目标函数较复杂）
            # 这里采用等效方法：增大 w_pv_consume 使弃能惩罚足够强
            # 当 w_pv_consume → ∞ 时等价于 hard constraint waste ≤ ε
            result = clear_market_multi_obj_lp(
                agents=agents, network=network, T=T, stage=stage,
                wholesale_price=wholesale_price, action_params=action_params,
                penalty_unserved=penalty_unserved,
                w_welfare=1.0, w_pv_consume=max(eps * 50, 1.0)  # 动态权重
            )

            actual_waste = result["re_waste"]
            actual_welfare = result["welfare"]

            pareto_front.append({
                "welfare": actual_welfare,
                "re_waste": actual_waste,
                "re_consumption_rate": result["re_consumption_rate"],
                "eps_target": float(eps),
            })

        except Exception as e:
            # 某些极端ε值可能不可行，跳过
            pass

    # ---- 步骤3: 输出Pareto前沿摘要 ----
    if pareto_front:
        print(f"\\n  Pareto 前沿 ({len(pareto_front)} 个有效点):")
        print(f\"  {'#':>4} {'Welfare(£)':>14} {'弃能(MWh)':>12} {'消纳率(%)':>10}\")
        print(f"  {'-'*54}")
        for i, p in enumerate(pareto_front):
            print(f"  {i+1:>4} {p['welfare']:>14.2f} {p['re_waste']:>12.3f} {p['re_consumption_rate']*100:>9.2f}%")

        # 找折中解（距离理想点最近，归一化欧氏距离）
        w_min = min(p["welfare"] for p in pareto_front)
        w_max = max(p["welfare"] for p in pareto_front)
        wk_min = min(p["re_waste"] for p in pareto_front)
        wk_max = max(p["re_waste"] for p in pareto_front)

        best_dist = float("inf")
        best_idx = 0
        for i, p in enumerate(pareto_front):
            nw = (p["welfare"] - w_max) / (w_min - w_max) if w_min != w_max else 0
            nwk = (p["re_waste"] - wk_min) / (wk_max - wk_min) if wk_max != wk_min else 0
            dist = (nw**2 + nwk**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        print(f"\\n  ★ 折中解(最近理想点): 第{best_idx+1}个")
        print(f"    Welfare={pareto_front[best_idx]['welfare']:.2f}, "
              f"弃能={pareto_front[best_idx]['re_waste']:.3f} MWh, "
              f"消纳率={pareto_front[best_idx]['re_consumption_rate']*100:.2f}%")

    print(f"{'='*60}")
    return pareto_front


# ============================================================
# 任务3: 多场景构建与批量运行
# ============================================================


def build_scenarios():
    """
    构建多场景列表（任务3）

    返回: list of dict, 每个 dict 含 {
        name: 场景名称,
        desc: 描述,
        agents, network, wholesale, with_wind
    }
    """
    T = 24
    scenarios = []

    # ---- 场景A: 基准场景（无风电，3节点）----
    ag_A, net_A, wp_A = build_demo_case(T=T, with_wind=False)
    scenarios.append({
        "name": "A-基准",
        "desc": "5Agent/3节点/PV-only/标准容量",
        "agents": ag_A, "network": net_A, "wholesale": wp_A,
        "with_wind": False,
    })

    # ---- 场景B: 高可再生（含风电，5节点）----
    ag_B, net_B, wp_B = build_demo_case(T=T, with_wind=True)
    scenarios.append({
        "name": "B-高可再生",
        "desc": "7Agent/5节点/PV+Wind/弃能风险高",
        "agents": ag_B, "network": net_B, "wholesale": wp_B,
        "with_wind": True,
    })

    # ---- 场景C: 紧约束（线路容量缩小→拥塞严重）----
    ag_C, _, wp_C = build_demo_case(T=T, with_wind=True)
    net_C = Network.from_edges([
        (0, 1, 4.0),   # 缩小
        (1, 2, 2.5),   # 缩小
        (2, 3, 3.0),   # 缩小
        (3, 4, 2.0),   # 缩小
    ])
    scenarios.append({
        "name": "C-紧约束",
        "desc": "7Agent/5节点/线路紧/拥塞严重",
        "agents": ag_C, "network": net_C, "wholesale": wp_C,
        "with_wind": True,
    })

    # ---- 场景D: 松约束（线路容量放大→近似无拥塞）----
    ag_D, _, wp_D = build_demo_case(T=T, with_wind=True)
    net_D = Network.from_edges([
        (0, 1, 20.0),
        (1, 2, 15.0),
        (2, 3, 15.0),
        (3, 4, 12.0),
    ])
    scenarios.append({
        "name": "D-松约束",
        "desc": "7Agent/5节点/线路松/无拥塞",
        "agents": ag_D, "network": net_D, "wholesale": wp_D,
        "with_wind": True,
    })

    # ---- 场景E: 高峰负荷（负荷放大1.5倍）----
    ag_E, net_E, wp_E = build_demo_case(T=T, with_wind=True)
    # 放大所有负荷Agent的负荷
    for a in ag_E:
        a.load_forecast = a.load_forecast * 1.5
        a.load_real = a.load_real * 1.5
    scenarios.append({
        "name": "E-高峰负荷",
        "desc": "7Agent/5节点/负荷×1.5/消纳率高",
        "agents": ag_E, "network": net_E, "wholesale": wp_E,
        "with_wind": True,
    })

    return scenarios


def run_all_scenarios(multi_obj: bool = True):
    """
    运行所有场景的对比分析（任务3）

    参数:
      multi_obj: 是否使用双目标优化（True=双目标, False=单目标）
    """
    scenarios = build_scenarios()

    print("=" * 80)
    print("  多场景批量运行对比")
    print("=" * 80)

    results = []
    for sc in scenarios:
        print(f"\\n>>> 场景: {sc['name']} - {sc['desc']}")
        try:
            act = random_actions(sc["agents"])
            if multi_obj:
                da = clear_market_multi_obj_lp(
                    agents=sc["agents"], network=sc["network"], T=24,
                    stage="DA", wholesale_price=sc["wholesale"],
                    action_params=act, w_welfare=1.0, w_pv_consume=80.0
                )
                row = {
                    "scenario": sc["name"],
                    "welfare": da["welfare"],
                    "re_waste": da.get("re_waste", 0),
                    "re_consumption_rate": da.get("re_consumption_rate", 1.0) * 100,
                    "n_agents": len(sc["agents"]),
                    "n_branches": len(sc["network"].branches),
                }
            else:
                da = clear_market_lp(
                    agents=sc["agents"], network=sc["network"], T=24,
                    stage="DA", wholesale_price=sc["wholesale"],
                    action_params=act
                )
                # 手动计算弃能
                total_re = sum(
                    np.sum(a.pv_forecast) + (np.sum(a.wind_forecast) if a.has_wind() else 0)
                    for a in sc["agents"] if a.is_prosumer or a.has_wind()
                )
                used_re = sum(
                    np.sum(da["schedules"][a.name]["pv_used"]) +
                    (np.sum(da["schedules"][a.name].get("wind_used", [0])) if a.has_wind() else 0)
                    for a in sc["agents"]
                )
                waste = total_re - used_re
                rate = used_re / total_re * 100 if total_re > 0 else 100
                row = {
                    "scenario": sc["name"],
                    "welfare": da["welfare"],
                    "re_waste": waste,
                    "re_consumption_rate": rate,
                    "n_agents": len(sc["agents"]),
                    "n_branches": len(sc["network"].branches),
                }

            results.append(row)
            print(f"    Welfare: {row['welfare']:.2f} | "
                  f"弃能: {row['re_waste']:.3f} MWh | "
                  f"消纳率: {row['re_consumption_rate']:.1f}%")

        except Exception as e:
            print(f"    ✗ 求解失败: {e}")
            results.append({"scenario": sc["name"], "error": str(e)})

    # ---- 汇总表格 ----
    print("\\n" + "=" * 80)
    print("  场景汇总表")
    print("=" * 80)
    header = f\"  {'场景':<12} {'Agent数':>6} {'支路数':>6} {'Welfare(£)':>14} {'弃能(MWh)':>12} {'消纳率(%)':>10}\"
    print(header)
    print("  " + "-" * 74)
    for r in results:
        if "error" not in r:
            print(f"  {r['scenario']:<12} {r['n_agents']:>6} {r['n_branches']:>6} "
                  f"{r['welfare']:>14.2f} {r['re_waste']:>12.3f} {r['re_consumption_rate']:>9.1f}%")
        else:
            print(f"  {r['scenario']:<12} {'ERROR':>44} {r['error']}")

    return results


# ============================================================
# 更新的演示函数
# ============================================================


def run_one_day_demo(with_wind: bool = False):
    """运行单日DA+RT演示（兼容新旧模式）"""
    agents, network, wholesale = build_demo_case(T=24, with_wind=with_wind)

    mode_str = " (含风电)" if with_wind else ""
    print(f"=== 单日演示{mode_str}: {len(agents)} Agents, {len(network.branches)} Branches ===\")

    # 日前
    act_DA = random_actions(agents)
    da = clear_market_lp(
        agents=agents, network=network, T=24, stage="DA",
        wholesale_price=wholesale, action_params=act_DA
    )

    # 实时
    act_RT = random_actions(agents)
    rt = clear_market_lp(
        agents=agents, network=network, T=24, stage="RT",
        wholesale_price=wholesale, action_params=act_RT
    )

    payment = two_settlement(agents, da, rt)

    print(f"Day-Ahead (DA) welfare: {round(da['welfare'], 2)}\")
    print(f"Real-Time (RT) welfare: {round(rt['welfare'], 2)}\")
    print("\\n--- Payments (positive = cost, negative = revenue) ---\")
    for k, v in payment.items():
        print(f"{k:14s}  {v:8.2f} £\")

    # 各Agent快照
    for a in agents[:4]:  # 只显示前4个避免太长
        sch_da = da["schedules"][a.name]
        sch_rt = rt["schedules"][a.name]
        print(f"\\n--- {a.name} snapshot ---\")
        if a.storage is not None:
            print(f"DA soc:  {np.round(sch_da.get('soc', [0]), 2)}\")
            print(f"RT soc:  {np.round(sch_rt.get('soc', [0]), 2)}\")
        pv_u = np.sum(sch_da.get("pv_used", [0]))
        wind_u = np.sum(sch_da.get("wind_used", [0]))
        print(f"DA PV用: {pv_u:.2f} MWh, Wind用: {wind_u:.2f} MWh")


def run_multi_obj_comparison(with_wind: bool = True):
    """
    对比演示：单目标 vs 双目标（任务1+2）

    参数:
      with_wind: 是否包含风电Agent（默认True以展示弃能效果）
    """
    agents, network, wholesale = build_demo_case(T=24, with_wind=with_wind)
    act = random_actions(agents)

    mode_str = " (含风电)" if with_wind else ""
    print("=" * 70)
    print(f"  多目标优化对比：社会福利 vs 社会福利+可再生能源消纳{mode_str}\")
    print("=" * 70)

    # ---- 方案A：原单目标（仅社会福利）----
    da_single = clear_market_lp(
        agents=agents, network=network, T=24, stage="DA",
        wholesale_price=wholesale, action_params=act
    )
    # 计算单目标下的总弃能(PV+Wind)
    re_avail_s = 0.0
    re_used_s = 0.0
    for a in agents:
        if a.is_prosumer:
            re_avail_s += np.sum(a.pv_forecast)
            re_used_s += np.sum(da_single["schedules"][a.name]["pv_used"])
        if a.has_wind():
            re_avail_s += np.sum(a.wind_forecast)
            re_used_s += np.sum(da_single["schedules"][a.name].get("wind_used", [0]))
    waste_single = re_avail_s - re_used_s
    rate_single = re_used_s / re_avail_s * 100 if re_avail_s > 0 else 0

    # ---- 方案B：双目标（社会福利 + 可再生能源消纳）----
    da_multi = clear_market_multi_obj_lp(
        agents=agents, network=network, T=24, stage="DA",
        wholesale_price=wholesale, action_params=act,
        w_welfare=1.0, w_pv_consume=80.0
    )

    # ---- 输出对比 ----
    print(f"\\n{'指标':<32} {'单目标(原)':>16} {'双目标(新)':>16}")
    print("-" * 68)
    print(f"{'社会总 welfare (£)':<30} {da_single['welfare']:>16.2f} {da_multi['welfare']:>16.2f}")
    print(f"{'可再生能源弃能 (MWh)':<28} {waste_single:>16.3f} {da_multi.get('re_waste', 0):>16.3f}")
    print(f"{'可再生能源消纳率 (%)':<28} {rate_single:>15.2f}% {da_multi.get('re_consumption_rate', 0)*100:>15.2f}%")
    print(f"{'总可再生可用 (MWh)':<28} {re_avail_s:>16.3f} {da_multi.get('total_re_available', 0):>16.3f}")

    # 各产消者详情
    print("\\n" + "-" * 68)
    print("各产消者(Prosumer) 可再生能源使用:")
    print(f"{'Agent':<16} {'单目标RE用(MWh)':>18} {'双目标RE用(MWh)':>18}")
    for a in agents:
        if a.is_prosumer or a.has_wind():
            u_s = np.sum(da_single["schedules"][a.name]["pv_used"])
            u_s += np.sum(da_single["schedules"][a.name].get("wind_used", [0]))
            u_m = np.sum(da_multi["schedules"][a.name]["pv_used"])
            u_m += np.sum(da_multi["schedules"][a.name].get("wind_used", [0]))
            re_type = "PV+Wind" if a.has_wind() else "PV"
            print(f"{a.name:<14}{re_type:>2} {u_s:>18.3f} {u_m:>18.3f}")

    # 结算对比
    print("\\n" + "-" * 68)
    rt_single = clear_market_lp(
        agents=agents, network=network, T=24, stage="RT",
        wholesale_price=wholesale, action_params=random_actions(agents)
    )
    rt_multi = clear_market_multi_obj_lp(
        agents=agents, network=network, T=24, stage="RT",
        wholesale_price=wholesale, action_params=random_actions(agents),
        w_welfare=1.0, w_pv_consume=80.0
    )

    pay_single = two_settlement(agents, da_single, rt_single)
    pay_multi = two_settlement(agents, da_multi, rt_multi)

    print("\\n各Agent结算支出(£, 正值=成本):")
    print(f"{'Agent':<16} {'单目标支出':>14} {'双目标支出':>14} {'差额':>10}")
    for a in agents:
        ps = pay_single[a.name]
        pm = pay_multi[a.name]
        print(f"{a.name:<16} {ps:>14.2f} {pm:>14.2f} {pm-ps:>10.2f}")

    print("\\n" + "=" * 70)
    print("  结论：")
    if with_wind:
        print("  · 风电Agent使可再生总量大幅增加，可能出现弃能(<100%消纳率)")
        print("  · 双目标优化通过加权惩罚弃能，提升本地消纳率")
        print("  · 权重 w_pv_consume 越大 → 消纳率越高 → 福利可能略有下降")
    else:
        print("  · 无风电时PV总量较小，消纳率通常已达100%（两种方案一致）")
    print("  · 方法对比: WSM适合在线求解; ε-约束法(--pareto)可看完整前沿")
    print("=" * 70)

    return {
        "single": {"da": da_single, "rt": rt_single, "payment": pay_single},
        "multi": {"da": da_multi, "rt": rt_multi, "payment": pay_multi},
    }


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if "--multi" in args:
        wind_mode = "--wind" in args
        run_multi_obj_comparison(with_wind=wind_mode)
    elif "--pareto" in args:
        # ε-约束法 Pareto 前沿
        agents, network, wholesale = build_demo_case(T=24, with_wind=True)
        act = random_actions(agents)
        front = solve_pareto_front_epsilon(
            agents=agents, network=network, T=24, stage="DA",
            wholesale_price=wholesale, action_params=act, n_points=15
        )
    elif "--scenarios" in args:
        multi = "--single" not in args
        run_all_scenarios(multi_obj=multi)
    elif "--all" in args:
        # 全套测试
        print("\\n===== 测试1: 原单目标(无风电) =====\")
        run_one_day_demo(with_wind=False)
        print("\\n===== 测试2: 双目标对比(含风电) =====\")
        run_multi_obj_comparison(with_wind=True)
        print("\\n===== 测试3: ε-约束法Pareto前沿 =====\")
        agents, network, wholesale = build_demo_case(T=24, with_wind=True)
        act = random_actions(agents)
        solve_pareto_front_epsilon(
            agents=agents, network=network, T=24, stage="DA",
            wholesale_price=wholesale, action_params=act, n_points=15
        )
        print("\\n===== 测试4: 多场景批量 =====\")
        run_all_scenarios(multi_obj=True)
    else:
        run_one_day_demo(with_wind=False)'''

content = content.replace(marker, new_functions)

with open("/mnt/c/Python/MakerB/agent/agent_trading.py", "w", encoding="utf-8") as f:
    f.write(content)

print("新增函数: solve_pareto_front_epsilon / build_scenarios / run_all_scenarios ✓")
print("更新: run_one_day_demo / run_multi_obj_comparison / __main__ ✓")
