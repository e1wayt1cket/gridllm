# 结算对账（当前实现）

本文件把「谁付给谁多少」按代码逐层还原，并核对三条不同的账：
出清目标（model objective）、参与者账（participant ledger）、系统对账（merchanting surplus）。
**只有第二条是钱。** 第一条是模型内部的选择器，第三条是恒等式检查。

---

## 1. 存在两套结算实现，正式路径只用其中一套

| 实现 | 位置 | 调用方 | 是否正式 |
| --- | --- | --- | --- |
| `two_settlement(agents, da, rt)` | `src/market.py:21-50` | `src/run.py:221`、`src/ui/sim_page.py:700,1137` | **否**。CLI 与仪表盘的演示路径 |
| `participant_payoff(...)` | `src/participant_payoff.py:117-160` | `src/rl_env.py`（训练奖励）、`src/b1_daily_baseline.py`、`src/counterfactual_eval.py:_settle_full_day`、`src/surplus_metrics.py` | **是**。训练、基线、KPI 全走这条 |

**正式配置里写的 `settlement: two_settlement_ledger` 是一个标签，不是一个调用**
（`src/make_formal_training_manifests.py:150` 只是把字符串写进 manifest 的
`formal_training_config.yaml`）。正式结算由 `participant_payoff` 完成，
其结构是「节点价 × 电量 + 申报负荷价值 − 成本项」，**没有日前/实时两段结算**。
`clear_rt_rolling_mpc` 存在于 `src/market.py:460`，但 `rl_env`/`b1_daily_baseline`/
`counterfactual_eval` 都不调用它。

> 这是本审计中**最大的命名—实现落差**：配置声称两段结算，实际是单时段的节点价结算。
> 它不改变任何数字（因为正式路径从没走过两段结算），但会让读配置的人误判机制。

---

## 2. 正式账本（`participant_payoff`）

对主体 `a`、时段集合 `P`（一次调用通常是已执行的 4 段或整日 96 段）：

```
Net_a = Revenue_a − Purchase_a − Generation_a − Degradation_a − Other_a − Penalty_a      (S1)
```

| 项 | 定义 | 位置 |
| --- | --- | --- |
| `Revenue` | `Σ_t LMP_{t,b(a)}·p_sell_{a,t}·dt` `+ (with_load)` `Σ_t bid_value_a·served_{a,t}·dt` | `:129-134` |
| `Purchase` | `Σ_t LMP_{t,b(a)}·p_buy_{a,t}·dt` | `:152` |
| `Generation` | `Σ_t offer_cost_a·(pv_used+wind_used)_{a,t}·dt` | `:137-139` |
| `Degradation` | `Σ_t cycle_cost·(ch+dis)_{a,t}·dt`，`cycle_cost = 100` | `:154` |
| `Other` | 恒为 0（保留字段） | `:62` |
| `Penalty` | `Σ_t penalty_unserved·unserved_{a,t}·dt`，2500 | `:135-136` |

`with_load` / `with_generation` 由 `payoff_model_for` 选择：
负载与 prosumer 用 `(True, True)`；**独立储能**用 `(False, False)`（`:176-196`），
即储能既没有负荷价值项，也没有发电机会成本项。

**`PayoffBreakdown` 没有期末库存项**（`participant_payoff.py` 的类注释明示），
所以「期末电量」不进入任何参与者的钱。

### 2.1 每主体能量恒等式

`p_buy` 与 `p_sell` 由 `dispatch_core.split_power(net_gen, net_con)` 给出，
`net_gen = pv+wind+dis`、`net_con = served+ch`（`:604-609`），故逐时逐主体：

```
p_buy_{a,t} − p_sell_{a,t} = served_{a,t} + ch_{a,t} − pv_{a,t} − wind_{a,t} − dis_{a,t}   (S2)
```

即「买−卖 = 负荷 + 充电 − 本地发电 − 放电」。这条恒等式是结算可对账的基础：
把 (S2) 代入 (S1) 的节点价项，得到

```
LMP-terms_a = Σ_t LMP_{t,b}·dt·( served + ch − pv − wind − dis )
```
再叠加申报负荷价值 `Σ bid_value·served`、发电成本 `Σ offer_cost·(pv+wind)`、循环成本
`Σ cycle_cost·(ch+dis)`，就是净额。**注意 `served` 同时被计价两次：一次按 `LMP`（作为购买），
一次按 `bid_value`（作为价值）。** 这不是重复收费，而是「支付 = 价×量、收益 = 估值」的分账；
但如果有人把 `Net_a` 当作「该主体的货币现金流」，`bid_value·served` 这一项就是**非现金的**估值项。
本报告的用词：`Net_a` 是**核算收益**，不等于现金流。现金流是 `LMP` 项减去实际支出。

### 2.2 系统级对账

对全部主体求和，用 (S2) 与节点平衡 C2：

```
Σ_a (p_buy − p_sell)_{a,t} = Σ_a (served+ch−pv−wind−dis) = p_grid_t − Σ_l r_l·I_sq_{t,l}
```
即「全体净购电 = 对外净交换 − 网损」。因此

```
Σ_a LMP·d_a = wholesale·p_grid + Rent                                             (S3)
Rent = Σ_{t,b} LMP_{t,b}·(净注入_{t,b}) − wholesale·p_grid
```
`Rent` 是节点价与外部价之间的价差现金流（含阻塞租金与网损租金），
`src/surplus_metrics.py` 的 `reconciliation` 实现了这条恒等式（`CP + BC == wholesale*import + rent`）。
**本阶段未运行 `surplus_metrics.reconciliation` 验证数值** → NOT_TESTED。

`(S3)` 的两个推论：
1. 用户总支出（`CP`）不等于系统对外购电成本，差额是 `Rent`；
2. 由于 `p_grid` 可正可负（出口上限 2.5 MW），`wholesale·p_grid` 在净出口时段是**收入**，
   这会改变 `Rent` 的符号语义（不是恒正的 merchandising surplus）。

---

## 3. 出清目标 ≠ 结算

| 对象 | 出清目标里的系数 | 账本里的系数 |
| --- | --- | --- |
| 负荷 | `bid = bid_mult × bid_value`（**申报**） | `bid_value`（**原申报值**），且另按 `LMP` 计购买 |
| 发电 | `offer = offer_adder + offer_cost`（申报） | `offer_cost`（原值） |
| 储能充电 | `γ^t·charge_quote + cycle_cost` | `LMP·(+ch)` + `cycle_cost·ch`（**无** `γ^t`，**无** charge_quote） |
| 储能放电 | `γ^t·discharge_quote + cycle_cost` | `LMP·(−dis)` + `cycle_cost·dis` |
| 对外购电 | `w_t·p_grid` | 不出现在任何单一主体账上（由 `Rent` 承担） |
| 期末库存 | `γ^T·TV·soc_T·e_max` | **不存在** |

所以：
- **出清目标不是任何一笔钱的汇总**；把它读成「社会福利」会同时错两处（用申报价、含期末估值）。
- 储能的申报报价**只影响出清，不影响自己的账单**：账单按 `LMP` 结算。
  报价影响 `LMP`（它改变边际条件），但那是间接、均衡层面的影响，不是「按报价结算」。
- 若某储能单元在自己的母线上是边际机组，则 `LMP` 已经包含它自己的报价与循环成本
  （见 `price_formation_audit.csv` P12，INFERRED）。此时「按 LMP 结算」等价于「按自己的保留价结算」，
  这是价格接受者设定自身价格的经典自反性，不是重复计费。

---

## 4. 循环成本的账（与 `cycle_cost_dataflow.md` 配合）

出现位置：出清目标 O4/O9a（成本）、账本 `Degradation`（一次）、B 奖励（继承账本）、
B1 对照臂（同一账本，同样的电量）。**账本只扣一次。** 目标里出现不是第二次收费，
而是让清算在决定充放量时把折旧计入边际成本。

---

## 5. 与 B1 的差额

```
delta_pi_i = pi_ai_i − pi_baseline_i            (counterfactual_eval.py:61-64)
delta_pi_percent = delta_pi / |pi_baseline| × 100   (:66)
p_delta_pi_positive = P(delta_pi_i > 0) over the day set
```

两臂的结算方式**必须相同**才是配对：`_settle_full_day` 用
`dataclasses.replace(config.storage, terminal_soc_equal=True)`、`horizon_type="full_day"`、
两端 SOC 钉住（`counterfactual_eval.py:203-264`）。因此 KPI 里的 `pi_ai` 与 `pi_baseline`
都来自**整日重清算 + 端点钉住**的账本，而训练奖励的 24 个块收益来自**逐窗清算**的账本
（`rl_env.step`，中段窗口不钉端点）。两者数值不同是设计使然：
**奖励不是收益，收益不是奖励**（CLAUDE.md 与 `model_constraints.md` 均如此规定）。
本阶段不比较两者数值。

`_settle_full_day` 在 `result is None / fell_back / has_wholesale_fallback` 时返回零收益
（`:248-255`、`is_degraded` `:111-126`）。所以「KPI = 0」可能意味着「退化」而不是「无收益」，
读报告时必须看 `settled` 与 `wholesale` 标志位。

---

## 6. 社会剩余：可定义，但当前代码没有把它当作结算对象

代码里可对账的三层（`src/surplus_metrics.py`）：

1. `consumer_metrics`：消费者支付 `CP`、消费者剩余 `CS`（按 `bid_value` 计的机会价值）、
   `qbuy = max(served − pv_used − wind_used, 0)`；
2. `agent_profit`：主体利润（= `participant_payoff` 口径）；
3. `market_power_split`：利润的「套利分量 / 市场力分量」分解。

`reconciliation` 把 `CP`、`BC`、`wholesale·import`、`rent` 串成一条恒等式。
**这就是本项目里「社会剩余」的可定义形式**：它是上述分量之和，是一个**事后核算对象**，
与出清目标 O1–O9 不是同一个函数（申报价 vs 真值、有无期末估值、有无罚项）。
本阶段声明这一点，但**不计算它** → NOT_TESTED（需重新清算）。
