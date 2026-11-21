# 信息集与市场时序审计

目的：把「策略在什么时刻、基于什么信息、对哪些时段报价」逐条对齐到代码，
并区分「日前投标」与「日内动态报价」两种叙事。

---

## 1. 价格标签的时间语义

`reports/v2_cy_jn_2025/data_audit.md` 记载：

> 一个完整日包含 96 个互不相同的标签 `00:00` 到 `23:45`，因此标签是**时段起点**，
> 每个数值属于以该标签开始的 15 分钟。

代码侧一致：`money.PERIODS_PER_DAY = 96`、`DT_HOURS = 0.25`，
`JnPriceProvider.get_day_ahead_price(day_id, T, t_start)` 返回 `curve[t_start : t_start+T]`
（`src/wholesale_price.py`），即**按起点索引的连续切片**。

因此时段 `t` 覆盖 `[t·15min, (t+1)·15min)`，`t=0` 是 `00:00–00:15`。
本审计不发现任何「起点/终点混用」的证据（PASS，证据为上述两处一致性）。
但价格曲线自身的单位仍未确认：`JnPriceProvider.price_unit_confirmed: False`
（`src/wholesale_price.py`），这是**数据侧未决项**，不是时序问题。

---

## 2. 一个决策块的时间关系

```
block k (k = 0..23)
  t_start = 4k,  t_end = 4k+4,  window_end = min(4k+16, 96)
  ┌──── 已观测：LMP[0 : 4k) 的真实值 + 该日负荷/风光预测 ────┐
  │  策略输出 4 个常数报价对（charge_price, discharge_price）│
  └───────────────────────────────────────────────────────┘
        │ 写入 current_actions[t_start : t_end]（仅 4 段）
        ▼
  clear_market(window = [4k, window_end), T_win = 16)
        │ 外部价 = curve[4k : 4k+16]（整段已知，含 12 段前瞻）
        ▼
  执行前 4 段 → 块收益 → 观测更新 → block k+1
```

代码：`rl_env.step`（`src/rl_env.py:864-1157`），
关键行：写入 `current_actions` `:903-909`、`window_end/window_T/n_commit` `:921-923`、
外部价 `:953`、horizon 选择 `:961-963`、`clear_market` `:965`、块收益 `:1054-1059`、
`prev_soc` 更新 `:1080-1085`。

**报价覆盖的时段**：只有 16 段窗口的**前 4 段**。后 12 段用的是
`current_actions` 中尚未被覆盖的**真申报占位值**（见 §4）。

---

## 3. 策略能看见什么（信息集）

`BiddingEnv._obs`（`src/rl_env.py:536-645`），OBS_V6 的 12 维：

| 特征 | 来源 | 是否含未来信息 |
| --- | --- | --- |
| `load_feat` | `agent.load_forecast[4k:4k+4]` 均值 / 该主体负荷峰值 | 否（日前预测） |
| `re_feat` | `pv_forecast`、`wind_forecast` 同期均值 / RE 峰值 | 否（日前预测） |
| `soc` | `prev_soc`（上一块第 4 段）或 `soc0` | 否 |
| `last_lmp_norm` | 已清算时段的最新跨节点均值 LMP ÷ `PRICE_REF_CNY_MWH` | 否（历史） |
| `block_pos` | `block_idx / 23` | 否 |
| `avg_lmp_hist_norm` | 已观测 LMP 的均值 ÷ `PRICE_REF` | 否（历史，空历史时回落到 `PRICE_REF`） |
| `slr_norm` | 系统负荷/新能源比（来自**预测**） | 否 |
| `avg_other_dev` / `all_dev_std` | 其他/全体机组上一块的**有符号方向归一化偏离** | 否（上一块已决策） |
| `ema_dev` / `price_trend` | 本主体 forecaster 的 EMA 偏离与 lately 斜率 | 否（历史） |
| `pred_lmp_norm` | forecaster 的下一步预测（EMA 持续）÷ `PRICE_REF` | 否 |

要点：

- **策略从不见到价格曲线本身**。它只见已实现 LMP（历史）与自己的 EMA 外推。
  代码注释明确写下这一点（`rl_env.py:546-552`），并有测试
  `test_the_observation_ignores_prices_after_the_decision` 钉住。
- 所有价格类特征都除以**冻结常量** `PRICE_REF_CNY_MWH = 288.19757259048833`（`rl_env.py:66`），
  不用当日均值。这是 OBS_V4 起就固定的口径。
- 决策块的负荷/风光特征在 DA/RT 两个阶段**都取预测值**（`rl_env.py:558-567`），
  实现曲线只通过清算后的历史（价格、SOC）进入观测。这一条已由注释与测试钉住。

---

## 4. 未来窗口的占位报价

`reset` 时 `self.current_actions = self._default_actions()`（`rl_env.py:837`），
即**全部 96 段填满每个主体自己的真申报点**（储能 `(0,0)`，负荷/prosumer `(1,0)`）。
每个 block 只覆盖自己那 4 段（`:909`）。

所以在窗口 `[4k, 4k+16)` 的清算中：

- 前 4 段：策略本块的报价；
- 后 12 段：**真申报价**，不是「策略未来会报的价」，也不是任何形式的预测。

**这不是实现缺陷，而是一个必须写明的事实的含义**：窗口前瞻所优化的是
「接下来 12 段我会诚实申报」的条件，而不是「接下来 12 段我会最优报价」的条件。
前瞻因此高估了未来时段的"报价约束松紧度"，对当前 4 段的边际条件有实际影响
（通过 SOC 路径与预期价格），但该影响的方向与幅度需要重新清算才能量化 → NOT_TESTED。

---

## 5. 「日前投标」还是「日内动态报价」？

代码事实：

1. 外部价格序列名为 day-ahead（`day_ahead_price_china`、`JnPriceProvider` 的
   `get_day_ahead_price`），但**在 reset 时把整日曲线一次取入**（`rl_env.reset` → `:806-807`），
   每个窗口再切片传给清算（`:953`）。
2. 清算发生在**模拟日内**的 24 个时点，每块只执行 4 段（1 小时）。
   所以这是一个**顺序日内再清算**过程，不是一次性的 D-1 拍卖。
3. **清算器对窗口内 16 段的外部价是完全知的**（整段切片直接进目标 O7），
   而**策略对价格曲线完全不知**（只能用历史 LMP 外推）。
   这是「市场有完美预见、参与者没有」的不对称信息结构。
4. 在 `jn2025` 提供者下，外部价**不依赖本地网络**（`wholesale_price` 文档字符串：
   独立外部市场的价格；`agents` 被显式忽略）。所以外部价对本网络是外生的、
   并被当作已知的（PASS）。
5. 负荷侧：`ledger` 模式下 `load_forecast = ledger_load.copy()` 与
   `load_real = ledger_load.copy()`（`src/grid.py:321-324`）——**预测等于实现**。
   只有 PV/风电有独立的 forecast/real 噪声（`grid.py:361-370`），
   而 `stage == "DA"` 的清算只读预测（`dispatch_socp.py:407-410`）。

合并 3–5 的结论（INFERRED）：在正式路径上，**给定日期后整个日内的清算过程是确定性的**，
唯一的随机性是 PV/风电的 forecast/real 差异，而正式路径只用 forecast。
这既不是「D-1 报价后等结果」，也不是「RT 实时不确定性下的动态报价」，
而是**第三种：价格已知、负荷已知、逐小时重开的确定性滚动市场**。

这一点与 `formal_training_config.yaml` 中 `episode: commit_periods 4 / lookahead_periods 16`
的描述一致，也与 `information_timing_audit` 想要区分的两类叙事都不同——
把它叫作"日前"或"日内"都会误导。

---

## 6. 时间/信息相关风险清单

| # | 风险 | 证据 | 等级 |
| --- | --- | --- | --- |
| T1 | 价格曲线单位未确认（`price_unit_confirmed: False`） | `src/wholesale_price.py` | 数据侧未决；影响所有 CNY 结论的量纲 |
| T2 | 前瞻 12 段用真申报占位价，而非策略未来价 | `rl_env.py:837, 909` | PASS（代码事实）；影响幅度 NOT_TESTED |
| T3 | 清算器对窗口内外部价完美预见，策略没有 | `rl_env.py:953` + `:536-645` | PASS（代码事实）；影响幅度 NOT_TESTED |
| T4 | `stage == "DA"` 时不使用实现曲线，因此 forecast≠real 的 PV 噪声在正式路径上不产生任何不确定性 | `dispatch_socp.py:407-410` | PASS；这意味着「实时市场」在本路径上不承担其名义功能 |
| T5 | `reactive_fraction_of_load_q = 2.0` 使 `q_ratio < 0`（负荷表现为无功注入） | `dispatch_socp.py:314-319` | PASS（代码事实）；物理含义见 `constraint_register.csv` C03 |
