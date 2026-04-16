"""
第二轮修改:
  1. clear_market_lp 加 wind_used 变量/约束
  2. clear_market_multi_obj_lp 同步更新
  3. 网络约束通用化(去硬编码)
  4. build_demo_case 新增风电Agent F + 兼容新Network
"""

with open("/mnt/c/Python/MakerB/agent/agent_trading.py", "r", encoding="utf-8") as f:
    content = f.read()

# ============================================================
# 1) clear_market_lp 中添加 wind_used 支持
# ============================================================

# 1a: 在变量声明区，pv_used 后面添加 wind_used
old_var_decl = '''        pv_used[a.name] = cp.Variable(T, nonneg=True)

        if a.storage is not None:'''

new_var_decl = '''        pv_used[a.name] = cp.Variable(T, nonneg=True)

        # ---- 风电使用量（任务1）----
        if a.has_wind():
            wind_used[a.name] = cp.Variable(T, nonneg=True)
        else:
            wind_used[a.name] = None

        if a.storage is not None:'''

content = content.replace(old_var_decl, new_var_decl)

# 1b: 在 g_grid 声明后添加 wind_used 字典初始化
old_ggrid = '''    # 外部电网供给（系统电源）：g_grid >= 0
    g_grid = cp.Variable(T, nonneg=True)'''

new_ggrid = '''    # 外部电网供给（系统电源）：g_grid >= 0
    g_grid = cp.Variable(T, nonneg=True)

    # 风电使用量字典（任务1）
    wind_used = {}'''

content = content.replace(old_ggrid, new_ggrid)

# 1c: PV使用约束后面添加 Wind 使用约束
old_pv_constr = '''    # PV使用上限
    for a in agents:
        if a.is_prosumer:
            constraints += [pv_used[a.name] <= pv[a.name]]
        else:
            constraints += [pv_used[a.name] == 0]'''

new_pv_constr = '''    # PV使用上限
    for a in agents:
        if a.is_prosumer:
            constraints += [pv_used[a.name] <= pv[a.name]]
        else:
            constraints += [pv_used[a.name] == 0]

    # Wind使用上限（任务1）
    for a in agents:
        if a.has_wind():
            if stage == "DA":
                w_avail = a.wind_forecast
            else:
                w_avail = a.wind_real
            constraints += [wind_used[a.name] <= w_avail]
        else:
            if wind_used.get(a.name) is not None:
                constraints += [wind_used[a.name] == 0]'''

content = content.replace(old_pv_constr, new_pv_constr)

# 1d: 功率平衡中加入 wind_used
old_supply = '''        total_supply = g_grid[t] + cp.sum([pv_used[a.name][t]
                                          for a in agents]) + cp.sum([p_sell[a.name][t] for a in agents])'''

new_supply = '''        total_supply = g_grid[t] + cp.sum([pv_used[a.name][t]
                                          for a in agents])
        # 加入风电供给（任务1）
        total_supply += cp.sum([wind_used[a.name][t]
                                for a in agents if wind_used.get(a.name) is not None])
        total_supply += cp.sum([p_sell[a.name][t] for a in agents])'''

content = content.replace(old_supply, new_supply)

# 1e: Agent能量守恒中加入 wind_used
old_energy_bal = '''            lhs_supply = pv_used[a.name][t] + p_buy[a.name][t]'''
new_energy_bal = '''            lhs_supply = pv_used[a.name][t] + p_buy[a.name][t]
            if wind_used.get(a.name) is not None:
                lhs_supply += wind_used[a.name][t]'''

content = content.replace(old_energy_bal, new_energy_bal)

# 1f: 网络约束中注入计算加入 wind_used
old_inj = '''            inj = pv_used[a.name][t] + p_sell[a.name][t] - \\
                served[a.name][t] - p_buy[a.name][t]'''
new_inj = '''            inj = pv_used[a.name][t] + p_sell[a.name][t] - \\
                served[a.name][t] - p_buy[a.name][t]
            if wind_used.get(a.name) is not None:
                inj += wind_used[a.name][t]'''

content = content.replace(old_inj, new_inj)

# 1g: 网络约束通用化（任务3：去硬编码）
old_net_constr = '''        # 这里假设 bus 分别为 1,2；bus0 为上级电网
        # flow12 = net_inj(bus2)
        # flow01 = net_inj(bus1) + net_inj(bus2)
        if 1 in net_inj_bus and 2 in net_inj_bus:
            flow12 = net_inj_bus[2]
            flow01 = net_inj_bus[1] + net_inj_bus[2]
            constraints += [flow12 <= network.cap12, flow12 >= -network.cap12]
            constraints += [flow01 <= network.cap01, flow01 >= -network.cap01]'''

new_net_constr = '''        # ---- 通用网络潮流约束（任务3）----
        # 对每条支路 (from_bus -> to_bus)，计算流过该支路的功率
        # 径向网假设：flow_ij = Σ net_inj(bus_k), k为j侧所有下游节点
        for branch in network.branches:
            bus_from = branch["from"]
            bus_to = branch["to"]
            cap = branch["cap"]

            # 找出 bus_to 及其所有下游节点的集合
            # 用BFS从bus_to出发沿支路方向搜索下游
            downstream = set()
            queue = [bus_to]
            visited_bfs = set()
            while queue:
                node = queue.pop(0)
                if node in visited_bfs:
                    continue
                visited_bfs.add(node)
                downstream.add(node)
                for br in network.branches:
                    if br["from"] == node and br["to"] not in visited_bfs:
                        queue.append(br["to"])

            # 支路潮流 = 所有下游节点净注入之和
            flow_ij = sum(net_inj_bus.get(b, 0) for b in downstream)
            constraints += [flow_ij <= cap, flow_ij >= -cap]'''

content = content.replace(old_net_constr, new_net_constr)

# 1h: schedules 返回中添加 wind_used
old_sched_ret = '''        schedules[a.name] = {
            "p_buy": np.array(p_buy[a.name].value).reshape(-1),
            "p_sell": np.array(p_sell[a.name].value).reshape(-1),
            "served": np.array(served[a.name].value).reshape(-1),
            "unserved": np.array(unserved[a.name].value).reshape(-1),
            "pv_used": np.array(pv_used[a.name].value).reshape(-1),
        }'''
new_sched_ret = '''        sched_entry = {
            "p_buy": np.array(p_buy[a.name].value).reshape(-1),
            "p_sell": np.array(p_sell[a.name].value).reshape(-1),
            "served": np.array(served[a.name].value).reshape(-1),
            "unserved": np.array(unserved[a.name].value).reshape(-1),
            "pv_used": np.array(pv_used[a.name].value).reshape(-1),
        }
        # 风电使用量（任务1）
        if wind_used.get(a.name) is not None:
            sched_entry["wind_used"] = np.array(wind_used[a.name].value).reshape(-1)
        schedules[a.name] = sched_entry'''

# 需要精确替换两处（clear_market_lp 和 clear_market_multi_obj_lp 各一处）
count = content.count(old_sched_ret)
content = content.replace(old_sched_ret, new_sched_ret, 1)
if count > 1:
    content = content.replace(old_sched_ret, new_sched_ret, 1)

with open("/mnt/c/Python/MakerB/agent/agent_trading.py", "w", encoding="utf-8") as f:
    f.write(content)

print("clear_market_lp: wind_used 变量/约束/能量守恒/网络通用化 ✓")
print(f"当前文件大小: {len(content)} 字符")
