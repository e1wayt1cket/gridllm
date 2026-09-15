# 典型日物理基线 —— 变更说明与审查请求

> **状态：本文档描述的是本基线的第一版，其内容已被后续一轮机制重构取代。**
> 当前的基线记录以 `TYPICAL_DAY_PHYSICAL_BASELINE.md` 为准；本文保留，是因为
> 第 6 节的九个待攻击点在那一轮里被逐条处理，处理结果记录在此，可作对照。
>
> **已修**（细节见 `TYPICAL_DAY_PHYSICAL_BASELINE.md`）：
> - §3.3 / §5.4 价格不可复现 → `price_curve.seed`，同配置逐字节一致。
> - §5.3 碳排按毛 import 计高估 → 上下网合并为单一自由净交换变量，
>   碳排改按 `max(net, 0)`，实测 115.79 → 80.99 tCO2（−34.80，与预测一致）。
> - §2.2 的 `bid ≤ offer` 钳位 → 已退役。充放电改为单一净功率流的正负部，
>   churn 结构性为零，报价可自由交叉。
> - §2.2 目标函数里 `+bid·ch` 的充电收益 → 翻号为成本项。代价是 `bid_mult`
>   含义反转（越高越不愿充电），已记入基线文档。
> - §2.2 整日语义用 `T == 96` 魔数 → 改为调用方显式声明 `horizon_type`。
> - §5.9 窗口 agent 丢字段 → `_make_window_agents` 已保留
>   `pv_capacity` / `wind_capacity` / `participant_type`。
>
> **已核实推翻**（见当时的核实结论，实施按核实结果）：
> - **§6 Q2 的 per-unit 前提不成立。** 标幺系统三引擎自洽，`U²/S_base` 在
>   `S_base = 1 MVA` 下即标准式；LMP 均值与批发曲线均值完全相等（782.8 对
>   782.8），故 2743 CNY/MWh 是边际线损的**空间离散**而非缩放错误。
>   未做 per-unit 改动，改为在基线文档 §7.5 记录为待定向诊断项。
>
> **仍未处理**：LDF/DC 对产消者电池仍按批发价计价（引擎分歧）；深馈线边际
> 线损因子的独立复算；`pseudo_realtime.py` 的调用签名错误。

本文是一份自包含的审查材料。读者没有任何本项目上下文，仅凭本文与仓库代码即可判断这次改动是否成立。

文中的「主张」都是可被证伪的陈述，第 6 节列出作者认为最容易被攻破的九个点，希望审查者优先从那里入手，而不是复述第 2 节的改动清单。

---

## 1. 背景

GridLLM 是一个配电网 agent 电力市场仿真：IEEE 33-bus 辐射网，96 个时段（24 小时 × 15 分钟），日前 + 实时两结算，出清为 Gurobi 求解的多时段联合 OPF（两种引擎：`socp` 与 `lindistflow`）。研究目标是产消者与储能的 RL 报价策略能否超过 truthful bidding 基线。

这次改动重做**物理基线**——所有报价实验的度量基准。原因：旧基线上网络峰值 5.75 MW、网内 15 台电池，峰谷价比 1.39，低于 0.92/0.92 电池在 100 CNY/MWh 退化成本下的往返盈亏平衡比 1.68。**储能套利在旧基线上物理上不存在**，15 台电池全部只充不放。「价格信号能否驱动储能套利」这个研究问题在旧基线上无法被提出。

本次改动不含任何 RL 结果。它只做一件事：让这一天在物理与账目上闭环——负荷、光伏、风电进入网络，出清按节点定价，电池被价格驱动，荷电状态回到起点，现金被结算。

---

## 2. 变更清单

工作树状态：10 个文件被修改、3 个文件新增，**全部未提交**（`git status` 无 commit）。最后一次文件写入时间为 09-14 18:38。

### 2.1 物理参数

| 文件 | 改动 | 意图 |
|---|---|---|
| `config/defaults.yaml` | 新增 `profiles.load.load_scale: 1.914` | 把全网峰值从 5.75 MW 抬到 11.0 MW。**只乘负荷曲线**，不乘 `base_load`——否则抬负荷会连带把每一条母线上的 PV/风电/电池按比例放大 |
| 同上 | `network.base_ampacity_ka` 0.22 → 0.25，`line_capacity_multiplier` 1.5 | 线路载流量（配合抬高的负荷） |
| 同上 | `price_curve.merit_order_price_elasticity` 0.35 → 1.4 | 让日前价格曲线的峰谷比越过往返盈亏平衡，否则电池仍然只充不放 |
| 同上 | `independent_storage`：3 台 × 6 MWh → **4 台 × (1 MW / 4 MWh)**，母线 `[2,12,24]` → `[6,20,24,31]` | 电池成为唯一储能，四台同型、一台一区，供后续节点位置实验扫描 |
| 同上 | 产消者与 `non_prosumer_storage` 加 `enabled` 开关，全部置 false | 撤掉网内 15 台电池 |
| 同上 | 各母线 `pv_capacity_scale` / `wind_capacity_scale` 显式归零或改写 | 让装机容量精确且空间分离：PV 3.5 MW(bus 21/23)、风电 1.5 MW(bus 30)，且 PV/风电/电池不共母线 |

### 2.2 代码

| 文件 | 改动 | 意图 |
|---|---|---|
| `src/grid.py` | `load_scale` 只作用于成品负荷曲线；`storage.enabled` 为假时**跳过**而不是把容量缩放到 0 | `e_max = 0` 会让 SOC 递推除零；且跳过的位置必须在「报价锚点」应用之前，否则失去电池的产消者会继承电池的支付意愿、其可再生出力不再值得被调度 |
| `src/models.py` | `StorageConfig.terminal_soc_equal` 默认 `False` → `True` | 整日比较不应奖励「期末囤电」。旧基线所有电池都结束在 SOC 上限，正是缺少这条约束 |
| `src/dispatch_socp.py` | 终值 SOC 约束改为仅在 `T == money.PERIODS_PER_DAY` 时生效；**首次**对独立储能（`storage_units`）也加上该约束 | 滚动窗口若钉住端点，就等于禁止窗口本来要选的那条轨迹 |
| `src/dispatch_ldf.py` | 补上同一条约束（含同样的整日门控） | 此前 LDF 引擎静默忽略该 flag，而 SOCP 执行它——同一请求两个引擎答案不同 |
| `src/dispatch_socp.py` | 结果字典新增 `bus_voltage` / `bus_indices` / `line_utilization` / `line_indices` / `line_flow_mw` / `grid_import_mw` / `grid_export_mw` | 电压、电流、线路潮流本就是模型变量，只是从未被导出，导致下游无法报告线路负载率与电压跌落。纯增量，旧 key 语义不变。LDF 下不提供（该引擎无电流变量） |
| `src/scenarios.py` | 新增 `typical_day` 场景（`config/scenarios.yaml`）与 `typical_day_config(T)` | 让基线的规范配置**可执行**而非仅可描述；`T != 96` 直接抛错 |
| `src/scenarios.py` | 构建场景时检查 agent 重名并抛错 | 批量求解器按 agent 名索引充/放/SOC 变量，重名会让后者覆盖前者且该单元静默报告全天零调度 |

### 2.3 测试

| 文件 | 改动 |
|---|---|
| `tests/test_baseline.py` | `test_baseline_da_welfare` 由「目标落在一个拟合出的数值区间」改为「价格节点化 + 资金流恒等式闭合」的不变量断言；`test_peak_load_higher_welfare` 改为 `test_peak_load_stresses_the_feeder`（断言甩负荷显著增加且目标下降，而非目标上升） |
| `tests/test_money_units.py` | 逐期调度容差 `atol=1e-4` → `1e-3`，附退化 LP 的说明 |
| `tests/test_surplus_metrics.py` | 端到端测试不再硬编码 `policies/matd3_cc_rot_seed42/best`，改为自动发现「所有策略文件都以 `ESS` 开头」的目录，找不到则 `skip` |
| `tests/test_typical_day_physical_baseline.py`（新增） | 典型日的物理契约：装机、空间分布、**恰好 4 台电池且网内无其他电池**、单元参数、母线互异、agent 名唯一、峰值、机队相对规模、可再生被实际消纳、端点 SOC 只在整日生效 |
| `src/diagnose_typical_day.py`（新增） | 独立诊断入口，输出系统/电池/网络/价格/选址/账目六节，12 条硬门 + 「已接受偏差」节，门失败则退出码非零 |

---

## 3. 验证证据

### 3.1 本次实际执行（2026-09-15）

```
$ git status --short
 M config/defaults.yaml
 M config/scenarios.yaml
 M src/dispatch_ldf.py
 M src/dispatch_socp.py
 M src/grid.py
 M src/models.py
 M src/scenarios.py
 M tests/test_baseline.py
 M tests/test_money_units.py
 M tests/test_surplus_metrics.py
?? TYPICAL_DAY_PHYSICAL_BASELINE.md
?? src/diagnose_typical_day.py
?? tests/test_typical_day_physical_baseline.py

$ python -m pytest tests/ -q
214 passed, 31 deselected, 17 warnings in 58.62s

$ PYTHONPATH=src python src/diagnose_typical_day.py --json outputs/typical_day_check.json
...
  12/12 gates passed
wrote outputs/typical_day_check.json
```

**注意 `31 deselected`**：`pytest.ini` 设了 `addopts = -m "not slow"`，被排除的 31 项标记为 `slow`（注释称「hours」）。也就是说上面这次通过**不覆盖**耗时路径，其中包含 `tests/test_surplus_metrics.py` 的端到端 SOCP 用例（它现在会因找不到适配机队的策略集而 `skip`）。

### 3.2 本次复跑与既有报告的数字对照

左列为本次复跑（`outputs/typical_day_check.json`），右列为既有报告 `TYPICAL_DAY_PHYSICAL_BASELINE.md` 记录值。

| 量 | 本次复跑 | 报告记录 |
|---|---|---|
| 峰值负荷（各 agent 最大值之和） | 11.0005 MW | 11.000 MW |
| 峰值负荷（同时） | 9.7723 MW | 9.772 MW |
| 日负荷电量 | 174.92 MWh | 174.9 MWh |
| PV 装机 / 发电 | 3.500 MW / 23.029 MWh | 3.500 / 23.03 |
| 风电装机 / 发电 | 1.500 MW / 18.562 MWh | 1.500 / 18.56 |
| 网络净交换 | 133.574 MWh | 133.6 |
| 毛 import / 毛 export | 199.619 / 60.000 MWh | 199.7 / 60 |
| 弃电 | 1.6399 MWh | 1.64 |
| 甩负荷 | 1.0194 MWh（0.583%） | 1.017（0.58%） |
| RE 消纳率 | 96.207% | 96.21% |
| LMP fallback 数 | 0 | 0 |
| LMP min / mean / max | 485.9 / 874.0 / 2764.8 | 485 / 867 / 2751 |
| LMP 系统峰谷比 | 2.393 | 2.39 |
| 最大线路利用率 | 0.5661 | 0.560 |
| 母线电压 min / max | 0.930 / 1.005 pu | 同 |
| merchandising surplus (rent) | 16957.9 CNY | 16889 |
| 消费者账单 | 100694.7 CNY | 99761 |
| 资金流恒等式残差 | 0 | 0 |
| 碳排（按毛 import） | 115.78 tCO2 | 115.8 |
| **清算目标** | **5647.93** | **6531** |
| **机队净收益** | **+539.6 CNY/日** | **+435.17** |

12 条门：节点价格真实、充放互斥（churn 1.055e-07 MWh）、四台全部放电、日内回到起始 SOC（偏差 0）、网络越限 0、甩负荷 ≤1%、可再生被使用、峰值达标、机队规模达标、装机达标、循环次数合理（0.42/0.65/0.42/0.40）、电池母线电气互异（最小间隔 2.292 Ω）。

### 3.3 关于两处不一致

清算目标与机队净收益两次运行不同，其余量一致。原因是既知缺陷：`_forecast_price_merit_order` 从全局未播种的 `np.random` 抽样，同一配置重复 5 次的实测区间为机队净收益 +486 ~ +665 CNY、LMP 系统比 2.24 ~ 2.52。既有报告据此要求「本日任何单次金额数字按 ±500 CNY 看待」。本次 +539.6 落在该区间内，**不构成回归，但也说明这个基线目前不可复现**。

---

## 4. 作者主张（可被证伪）

- **C1** 典型日的物理量（峰值 11 MW、174.9 MWh 电量、3.5 MW PV、1.5 MW 风电、四台 1 MW/4 MWh）由诊断脚本测量而非断言，12 条硬门全部通过。
- **C2** `load_scale` 只影响负荷曲线，不影响任何设备的装机。理由是它作用在 `_noisy_load(...)` 的返回值上，而 `base_load` 在缩放之前已被用于给 PV/风电/电池定容。**可证伪方式**：把 `load_scale` 从 1.914 改到其它值，检查 `system.pv_installed_mw` 与各 agent 的 `storage.e_max` 是否移动。
- **C3** 「恰好 4 台电池、网内无其他电池」由测试按后果而非按配置检查（`len(storage) == 4` 而非 `assert not enabled`），因此配置开关即使写错也会被测出。
- **C4** 跳过无电池产消者这件事发生在报价锚点应用**之前**，因此这类 agent 保留自身负荷类型的 `bid_value`/`offer_cost`；若顺序反了，其可再生出力会被定价过高而弃掉。测试 `test_renewables_are_dispatched_not_merely_installed` 用 RE 消纳率 ≥90% 钉住这个后果。
- **C5** 终值 SOC 约束只在整日生效，且两个引擎行为一致；测试对 `T=16` 的窗口断言「至少一台电池没有回到起点」，以此证明门控在起作用。
- **C6** 新增的网络读数（电压/线路利用率/潮流）不改变任何旧 key 的语义，是纯增量。
- **C7** 4 台电池的选址是按电气结构选的（距 slack 2.17/2.34/2.83/5.12 Ω，最小两两间隔 2.292 Ω，一台一区），不是按单点套利收益选的；报告明确说明「电池会消掉它自己吃的价差」，因此单点最优选址不可信。
- **C8** 所有金额来自 `participant_payoff`，与训练器和评估器走同一实现，没有在这份报告里重新推导。

---

## 5. 已知偏差与未修问题

以下为既有报告第 6 节的内容摘要，作者认为它们是**记录在案而非已修**。审查者应把它们当作尚未关闭的风险。

1. **甩负荷 1.019 MWh（0.58%）**。馈线受电压约束：在 `v_min_pu = 0.93`（±7%）下，峰值约 8.3 MW 起开始甩负荷。关掉逆变器无功支撑可把弃电降到 0，但甩负荷升到 11.6 MWh。报告结论是「在同一网络上同时满足 11 MW 与零甩负荷不可达」，选择保留电压带、接受甩负荷。
2. **弃电 1.6399 MWh（可用的 3.8%）不是经济性弃电**，而是逆变器视在功率圆被无功挤占。该量在负荷水平、价格水平、选址变化下不变。
3. **碳排按毛 import 计算，高估约 34.8 tCO2**。`p_grid_import` 与 `p_grid_export` 同价 ⇒ 目标函数在两者同增方向完全平坦，求解器把 export 停在 2.5 MW 上限且 96 个时段全部如此，import 被同额抬高（毛 199.62 MWh vs 物理净 133.57 MWh）。出清与 LMP 不受影响。修法是把两个变量重构为单个自由净交换变量再取正负部，属于市场机制改动，本次未做。
4. **价格曲线不可复现**（见 3.3）。`price_curve.seed` 是推荐的下一步，但会改动所有现存场景的价格，故未纳入本次。
5. **套利窗口压在盈亏平衡点上**，不显著高于它。逐母线 margin：ESS6 +0.228、ESS20 +1.117、ESS24 +0.085、ESS31 −0.021。只有 ESS20 明显为正，机队净收益依赖单台，且随价格实现变号（历史上曾测到 −330 CNY）。
6. **深馈线节点价格达 ~2750 CNY/MWh**（约谷段 5.7 倍）。原因是 `_build_line_params` 用 `z_base = v_base²`，隐含 `S_base = 1 MVA`，而 `NetworkConfig.base_mva` 在 SOCP 路径中从未被读取；11 MW 下标幺电流约 11 pu，边际线损因子被放大近十倍。物理自洽但标度极差。修它要同时改三个引擎的标幺换算，超出本次范围。
7. **引擎分歧**：同一天用 `lindistflow` 出清得到 RE 消纳率 100.0%（SOCP 为 96.21%）、甩负荷 0.315 MWh（SOCP 为 1.019）、目标差两个数量级。LDF 无电流变量且线性化线损与电压，看不到造成弃电的无功/有功耦合。典型日被 `typical_day_config()` 钉在 `socp`。
8. **`config/defaults.yaml` 里有不生效的配置块**：`storage:` / `market_design:` / `rt:` 三个块没有任何模块读取，生效值是 `models.py` 的数据类默认值；`network.v_min_pu` 同理。改这些 key 无效。未删除是因为删除是另一件事。
9. **`market._make_window_agents` 在切滚动窗口时丢掉了 `pv_capacity` / `wind_capacity` / `participant_type`**，导致每个窗口出清看到的逆变器无功上限为 0、储能机队看起来像产消者。这在 RL 训练路径上，不在本次基线的路径上（基线整日一次出清）。
10. **两个乘负荷的研究场景被新基线抬高成严重压力场景**：`peak_load`（1.5×）现为 16.5 MW、甩负荷 13.7 MWh（5.2%）；`congestion` 为 13.2 MW、甩负荷 18.5 MWh（9.0%）。

另有一条不属于模型、但决定后续工作起点的事实：**`policies/` 下 39 个目录与全部 `results/*_eval.csv` 都是旧 12 台网内电池机队的策略集，与新场景没有任何共同 agent；磁盘上目前不存在任何 `ESS*` 策略。** 现有 RL 结果与这一天不可比，必须重训。

---

## 6. 请重点审查的问题

作者认为以下九点最可能站不住，请优先攻击。

**Q1（最高优先）测试被放宽了，这是不是在掩盖真实回归？**
三处改动方向一致地变松：`test_baseline_da_welfare` 删掉了 12500–50000 的数值区间，换成一组不变量；`test_peak_load_higher_welfare` 原本断言 peak_load 目标**高于**基线，现在断言**低于**基线（理由是新基线下 1.5× 峰值超出馈线可输送能力，多出来的需求表现为甩负荷，被按失负荷价值定价，因此压低目标）；`test_money_units` 的逐期调度容差从 `1e-4` 放宽到 `1e-3`。请判断：**这些是新物理带来的正确重述，还是把测试改到能过？** 特别是第二个——「压力场景的目标应当更低」是否是一条真的不变量，还是在为一个不该接受的模型行为背书？

**Q2 用 `T == 96` 作为「整日」的判据是否可靠？**
终值 SOC 约束的门控写成了 `config.storage.terminal_soc_equal and T == money.PERIODS_PER_DAY`。这把「是否整日」编码成了一个魔数比较。若将来有 48 期（30 分钟粒度）的整日出清，或 96 期但非整日的窗口，门控会给出错误答案。是否应该让「整日」成为显式信号（由调用方传入，或由场景声明），而不是由 T 推断？

**Q3 基线的套利窗口压在盈亏平衡点上，这个基线还成立吗？**
只有 ESS20 显著为正，ESS31 为负，机队净收益随价格实现变号。既然这条基线存在的唯一理由是「让储能套利在物理上存在」，那么「勉强为正、且靠一台」是否已经不足以支撑后续所有报价实验？可选修法各有代价：降 `cycle_cost`（被决定保持 100）、再抬弹性（会同时抬高 LMP 与消费者账单）、换电池参数。请给出判断。

**Q4 甩负荷 0.58% 与弃电 3.8% 应该被接受吗？**
报告的论证是「±7% 电压带下不可兼得」。但这个取舍把两个都由网络物理决定的量变成了基线的既成事实：任何后续 RL 策略的比较都要在一个已经甩掉 1.019 MWh 负荷、弃掉 1.64 MWh 可再生的日子上进行。另一个选择是放松 `v_min_pu` 或抬载流量把两个都消掉。请判断哪个更适合作为研究基线。

**Q5 把「毛 import 高估碳排 34.8 tCO2」留在基线里是否可接受？**
出清与 LMP 不受影响，但任何以碳排为指标或目标的实验（本项目有 `carbon_cap_tco2` 的约束型多目标路径）都会读到这个偏高的数。它是已知的、有明确修法的、且本次明确不修。请判断这算「记录在案的已知缺陷」还是「必须先修才能发布基线」。

**Q6 引擎分歧会不会让后续结论不可比？**
同一天 LDF 与 SOCP 的 RE 消纳率差 3.8 个百分点、甩负荷差 3 倍、目标差两个数量级。基线钉在 SOCP，但训练/评估路径（`market._make_window_agents`）与部分测试走的是 LDF。如果两个引擎在同一场景上给出定性不同的答案，那么「策略 A 优于策略 B」的结论是否跨引擎成立？

**Q7 测试里的策略集自动发现是否过于宽松？**
`_fleet_policy_dir` 的判据是「目录下所有 `.pt` 文件名都以 `ESS` 开头」，取排序后第一个匹配。这既可能选中一个与当前场景 agent 集合并不完全对应的目录，也让「没有可用策略」这一真实阻塞在测试里表现为 `skip`（即绿色）。是否应该改成显式失败或显式列出所需 agent 名？

**Q8 `load_scale` 这个旋钮的设计对不对？**
它只缩放负荷曲线而不缩放设备，好处是抬负荷不会连带放大装机；代价是机队相对规模指标（P_ESS/P_peak = 36.4%、E_ESS/load = 9.1%）是被这个选择决定的，而不是被一个物理规律决定的。换言之，负荷水平与设备规模现在是两个独立旋钮，基线报告里的「机队规模合理」是靠把两个旋钮调到一起得到的。请判断这在方法论上是否站得住。

**Q9 那些「门」的阈值本身合理吗？**
12 条硬门里，有几条的阈值是作者自定的：甩负荷 ≤ 负荷电量的 1%、RE 消纳率 ≥ 90%、peak/P_ESS 在 0.30–0.42、E_ESS/load 在 0.08–0.10、每台循环 0.2–0.9 次、母线最小间隔 > 0.5 Ω。请检查这些区间是否有外部依据，还是仅仅把当前这次运行的结果框起来的。

---

## 7. 复现方式

需要 Gurobi 授权与 `requirements.txt` 依赖。本机解释器为 `C:/Users/26036/.conda/envs/energy_env/python.exe`，`src/` 在 `PYTHONPATH` 上（`pytest.ini` 已设 `pythonpath = src`）。

```bash
git status --short
git diff                                  # 全部改动，约 282 行插入 / 64 行删除
git diff --stat

python -m pytest tests/ -q                # 期望 214 passed, 31 deselected
python -m pytest tests/ -q -m slow        # 被排除的 31 项，耗时长
PYTHONPATH=src python src/diagnose_typical_day.py --json outputs/typical_day_check.json
```

诊断脚本若任一门失败则退出码非零；`--quiet` 只出 JSON；`--sweep-load-scale` / `--sweep-elasticity` 可复现参数扫描。

相关文件：

| 内容 | 位置 |
|---|---|
| 既有基线报告（英文，含第 3 节选址推导与第 5 节验收表） | `TYPICAL_DAY_PHYSICAL_BASELINE.md` |
| 物理契约测试 | `tests/test_typical_day_physical_baseline.py` |
| 诊断脚本 | `src/diagnose_typical_day.py` |
| 独立储能机队构建 | `src/ess.py`（`build_ess_fleet`） |
| 负荷缩放与设备开关 | `src/grid.py`（`create_agents_from_network`） |
| 端值 SOC | `src/models.py`（`StorageConfig.terminal_soc_equal`），在 `src/dispatch_socp.py` / `src/dispatch_ldf.py` 中执行 |
| 网络读数 | `src/dispatch_socp.py` 结果字典 |
