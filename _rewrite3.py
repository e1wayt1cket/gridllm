"""
第三轮修改:
  1. clear_market_multi_obj_lp 同步加wind（部分已由上面的全局替换完成，需补目标函数中的弃风）
  2. build_demo_case 新增风电Agent F + 兼容新Network
  3. 新增 solve_pareto_front_epsilon() ε-约束法
  4. 新增 build_scenarios() 多场景
  5. 更新 random_actions 和演示函数
"""

with open("/mnt/c/Python/MakerB/agent/agent_trading.py", "r", encoding="utf-8") as f:
    content = f.read()

# ============================================================
# 1) 多目标函数中：弃光→弃可再生(PV+Wind)
# ============================================================

# 替换 total_pv_available 计算为总可再生
old_total_pv = '''    total_pv_available = float(sum(np.sum(pv[a.name]) for a in agents if a.is_prosumer))
    pv_waste_expr = 0
    for a in agents:
        if a.is_prosumer:
            # 弃光量 = 可用 - 已用
            pv_waste_expr += cp.sum(pv[a.name] - pv_used[a.name])'''

new_total_re = '''    # ---- 总可再生能源可用量（PV+Wind, 任务1扩展）----
    total_re_available = 0.0
    for a in agents:
        if a.is_prosumer:
            total_re_available += float(np.sum(pv[a.name]))
        if a.has_wind():
            w_arr = a.wind_forecast if stage == "DA" else a.wind_real
            total_re_available += float(np.sum(w_arr))

    # ---- 弃能量表达式（PV waste + Wind waste）----
    re_waste_expr = 0
    for a in agents:
        if a.is_prosumer:
            # PV弃光量
            re_waste_expr += cp.sum(pv[a.name] - pv_used[a.name])
        if a.has_wind():
            # Wind弃风量
            w_avail = a.wind_forecast if stage == "DA" else a.wind_real
            re_waste_expr += cp.sum(w_avail - wind_used[a.name])'''

content = content.replace(old_total_pv, new_total_re)

# 替换目标函数中引用
old_obj_if = '''    if total_pv_available > 0:
        objective = w_welfare * welfare - w_pv_consume * (pv_waste_expr / total_pv_available)
    else:
        objective = w_welfare * welfare'''

new_obj_if = '''    if total_re_available > 0:
        objective = w_welfare * welfare - w_pv_consume * (re_waste_expr / total_re_available)
    else:
        objective = w_welfare * welfare'''

content = content.replace(old_obj_if, new_obj_if)

# 替换返回值计算
old_ret_calc = '''    # 计算各目标的实际值（用于输出分析）
    pv_waste_actual = sum(
        float(np.sum(pv[a.name]) - np.sum(pv_used[a.name].value))
        for a in agents if a.is_prosumer
    )
    pv_consumption_rate = 1.0 - (pv_waste_actual / total_pv_available) if total_pv_available > 0 else 0.0

    return {
        "price": clearing_price,
        "schedules": schedules,
        "welfare": float(welfare.value) if welfare.value is not None else 0.0,
        # ---- 多目标新增返回项 ----
        "pv_waste": pv_waste_actual,
        "pv_consumption_rate": pv_consumption_rate,
        "total_pv_available": total_pv_available,
        "obj_detail": {
            "welfare": float(welfare.value) if welfare.value is not None else 0.0,
            "pv_waste": pv_waste_actual,
            "pv_consumption_rate": pv_consumption_rate,
        },
    }'''

new_ret_calc = '''    # 计算各目标的实际值（任务1: PV+Wind 弃能）
    re_waste_actual = 0.0
    for a in agents:
        if a.is_prosumer:
            re_waste_actual += float(np.sum(pv[a.name]) - np.sum(pv_used[a.name].value))
        if a.has_wind():
            w_avail_arr = a.wind_forecast if stage == "DA" else a.wind_real
            re_waste_actual += float(np.sum(w_avail_arr) - np.sum(wind_used[a.name].value))

    re_consumption_rate = 1.0 - (re_waste_actual / total_re_available) if total_re_available > 0 else 0.0

    return {
        "price": clearing_price,
        "schedules": schedules,
        "welfare": float(welfare.value) if welfare.value is not None else 0.0,
        # ---- 多目标新增返回项（任务1: 扩展为含风电）----
        "re_waste": re_waste_actual,           # 总弃能量(MWh), 含PV+Wind
        "re_consumption_rate": re_consumption_rate,  # 总消纳率
        "total_re_available": total_re_available,
        "obj_detail": {
            "welfare": float(welfare.value) if welfare.value is not None else 0.0,
            "re_waste": re_waste_actual,
            "re_consumption_rate": re_consumption_rate,
        },
    }'''

content = content.replace(old_ret_calc, new_ret_calc)

with open("/mnt/c/Python/MakerB/agent/agent_trading.py", "w", encoding="utf-8") as f:
    f.write(content)

print("多目标函数: 弃能计算扩展为PV+Wind ✓")
