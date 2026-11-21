# 循环成本数据流

参数：`cycle_cost = 100.0 CNY/MWh`，配置位置 `config/defaults.yaml` → `storage.cycle_cost`
（`MarketConfig.storage`），正式 manifest 亦记为 `cycle_cost_cny_per_mwh: 100.0`。

计费口径（**全部实现都是这一个口径**）：

```
Degradation = cycle_cost × (ch_t + dis_t) × DT_HOURS        per unit, per period
```

即**按毛吞吐量**（充电量 + 放电量各算一次），不是按净吞吐量、不是按循环次数。

---

## 1. 六个出现点

| 位置 | 表达式 | 代码 | 是否影响钱 | 状态 |
| --- | --- | --- | --- | --- |
| **A. 出清目标（prosumer 储能）** | `− cycle_cost·(ch+dis)·dt` | `src/dispatch_socp.py:587-589` | 影响充放量（间接影响钱） | 生效 |
| **A'. 出清目标（独立储能）** | `− cycle_cost·(ch+dis)·dt` | `src/dispatch_socp.py:74-98`（`_add_independent_storage`，:92-95） | 同上 | 生效 |
| **B. 参与者结算** | `Degradation = Σ cycle_cost·(ch+dis)·dt` | `src/participant_payoff.py:154` | **是**，扣一次 | 生效 |
| **C. 单日收益** | 由 B 累加（`payoff.total`） | `participant_payoff.py:71-74`；`surplus_metrics.agent_profit` 委托同处 | 是 | 生效 |
| **D. RL 奖励** | 经 C 进入 `G_i`，再进 `R_i = G_i^policy − G_i^B1` | `src/rl_env.py:1054`→`participant_payoff`；`src/train_da_gate.py:287`；`src/daily_reward_b.py:137-160` | 是（两臂各扣一次，差额相消） | 生效 |
| **E. B1 对照臂** | 同一账本、同一 `100.0` | `src/b1_daily_baseline.py:75-78`，`build_env_for_pair` 默认 `cycle_cost=100.0` | 是 | 生效 |
| **F. C2 势函数整形** | **不含循环成本**，只有 SOC 势 | `src/c2_reward.py:80-92` | 否 | 生效但不涉及 |
| **G. 其他整形** | 无。`rl_env` 的 market-impact 惩罚（`rl_env.py:1062-1071`）作用于价格×量，不含循环成本 | `src/rl_env.py` | 否 | 默认关闭 |
| **H. KPI 整日结算** | 经 B（`_settle_full_day` 调 `participant_payoff`） | `src/counterfactual_eval.py:264` | 是 | 生效 |
| **I. `two_settlement`（遗留）** | **完全不含循环成本** | `src/market.py:21-50` | — | 只在 CLI/仪表盘 |

**I 是唯一的口径缺口**：`two_settlement` 只算 `price × (p_buy − p_sell)`，
既没有循环成本也没有负荷价值。它不参与任何正式结论，但 `run.py`/`sim_page.py` 的
展示数字与正式账本不可比。

---

## 2. 数据流图

```
policy quotes (charge_price, discharge_price)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ dispatch_socp.solve_socp_opf_batch                           │
│   obj: −b·ch·γ^t − o·dis·γ^t − c·(ch+dis)        [A, A']     │
│   s.t. net 包络 + SOC 转移 + 电压/潮流/容量                   │
│   ↓                                                          │
│   ch*[t], dis*[t], soc*[t], lmp[t,b], clearing_objective      │
└──────────────────────────────────────────────────────────────┘
        │                     │
        │                     └────────────► settlement? 否
        ▼                                     （目标不是结算）
  schedules (ch, dis, p_buy, p_sell, ...)
        │
        ▼
participant_payoff._ledger                                  [B]
  Revenue       = Σ lmp·p_sell·dt  (+ Σ bid_value·served·dt)
  Purchase      = Σ lmp·p_buy·dt
  Generation    = Σ offer_cost·(pv+wind)·dt
  Degradation   = Σ cycle_cost·(ch+dis)·dt          ◄────── c 在这里扣一次
  Penalty       = Σ penalty_unserved·unserved·dt
  Net = R − P − G − D − O − Pen
        │
        ├──► rl_env.step → raw_profit（块收益，已执行 4 段）      [D]
        ├──► b1_daily_baseline.run_truthful_day → Σ raw_profit     [E]
        ├──► counterfactual_eval._settle_full_day → pi_ai / pi_b1  [H]
        └──► surplus_metrics.agent_profit / consumer_metrics
        │
        ▼
daily_reward_b.day_reward_b
  R_i = (G_i^policy − G_i^B1)/scale，全部放在 block 23
        │
        ▼
c2_reward.c2_reward                                          [F]
  r_{i,t} = R_{i,t} + γΦ(soc_{t+1},t+1) − Φ(soc_t,t)         （无循环成本）
```

---

## 3. 是否重复计费：结论与边界

**结论：在账本层面不重复。** `Degradation` 在 (S1) 里只出现一次；
出清目标里出现的是另一个对象（选择器的边际成本），不是第二笔扣款。

三条需要写清的区别：

1. **目标里的 `c` 与账本里的 `c` 不是同一笔钱的两份，而是「决策成本」与「事后扣款」。**
   目标用 `c` 决定 `ch/dis`；账本用同一个 `c` 对已经决定的 `ch/dis` 扣一次。
   两者数值上必然相等（同一 `c`、同一最优解），所以看数字容易误以为重复。
2. **`c` 会进入边际机组的 LMP**（INFERRED，见 `price_formation_audit.csv` P12）。
   若某储能单元在自己母线上是边际机组，它面对的买价/卖价里已经包含自己的 `c`，
   于是「按 LMP 结算」会把它自己的折旧成本也一起结算回来。
   **这不是重复计费，而是价格接受者自设价格的自反性**；它对
   「储能能否靠价差覆盖折旧」这一问题有实质影响，本阶段不量化。
3. **效率损失本身不额外计价。** `η_ch = η_dis = 0.92` 使 1.6 MWh 充电只放出 1.354 MWh。
   这 0.246 MWh 的损失**没有单独的成本项**，它通过 (S2) 自动体现为「多买少卖」。
   所以循环的经济门槛是

   ```
   每交付 1 MWh 的循环成本等价 = c/η_rt + c = 100/0.8464 + 100 = 218.1 CNY/MWh
   一次完整循环（1.6 MWh 充 → 1.354 MWh 放）的绝对额 = 100 × (1.6 + 1.354) = 295.4 CNY
   ```

   这是代码算术（PASS），不是实测收益。

---

## 4. 与折扣的交互

| 出现点 | 是否乘 `γ^t` |
| --- | --- |
| prosumer 储能的申报报价项 | **是**（`:584-586`） |
| prosumer 储能的循环成本 | **否**（`:587-589`） |
| 独立储能的报价项与循环成本 | **否**（`_add_independent_storage`） |
| 账本 `Degradation` | **否**（账本无折扣概念） |

同一台电池的「报价」随 `t` 被折扣而「折旧」不被折扣，两者在目标里的相对权重随 `t` 变化。
当 `T=96` 时 `γ^96 = 0.7494`，报价项权重比末期折旧项低 25%。这是代码事实（PASS），
其经济含义（是否应统一折扣）属判断题，本阶段不下结论。

---

## 5. 未覆盖

- `cycle_cost` 敏感性实验脚本 `src/storage_cycle_cost_sensitivity.py` 存在，但**本阶段未运行**。
- 目标里 `c` 对 `ch/dis` 的实际影响幅度（报价 vs 折旧谁主导）需要重新清算 → NOT_TESTED。
- `reports/phase7a_*` 曾报告「Z=10 增益来自循环成本规避」；本审计**不引用旧报告的结论**，
  只记录该结论与本阶段认定的机制（`c` 在目标与账本各出现一次、口径一致）不矛盾。
