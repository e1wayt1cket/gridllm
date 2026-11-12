# SOC / MPC / 终端条件审计

只登记机制与边界，不重新设计。所有数值为代码常量与算术，非清算结果。

---

## 1. 时间结构（代码事实）

| 概念 | 值 | 位置 |
| --- | --- | --- |
| 一天 | 96 个 15 分钟时段，`DT_HOURS = 0.25` | `src/money.py:21-22` |
| 决策块 | `BLOCK_SIZE = 4` 时段 = 1 h | `src/rl_env.py:76` |
| 块数 | `N_BLOCKS = 24` | `src/rl_env.py:77` |
| 每窗期数 | `roll_horizon = 16` | `BiddingEnv.__init__` 默认 `roll_horizon=16` |
| 每窗执行 | `n_commit = min(4, window_T) = 4` | `rl_env.py:923` |
| 每窗前瞻 | 12 段（16 − 4），只清算、不执行 | `rl_env.py:921` |

所以「MPC」在本系统里的全部含义是：**每次滚动优化 16 段、只执行前 4 段**。
不存在独立的 MPC 求解器在运行（见 §2）。

---

## 2. MPC 模块状态：代码存在，路径为死代码

- `config.storage.self_schedule = false`（`config/defaults.yaml`）。
- `_compute_mpc_schedules` 第一行判断：`if not config.storage.self_schedule or not all_storage: return schedules`
  （`src/dispatch_core.py:703-705`）→ 返回空 dict。
- `solve_socp_opf_batch` 只在 `mpc_schedules` 非空时把 `ch/dis/soc` 的 `LB=UB`
  （`src/dispatch_socp.py:302-312`）。空 dict ⇒ 该分支不执行。
- `mpc_horizon = 64`、`mpc_price_noise_pct` 等参数因此**全部无效**。
- 副作用：因为 MPC 不固定，`_needs_binaries` 走的路径也不触发互斥二进制。

**结论**：MPC 自调度是登记在册但未启用的机制（PASS）。仓库里另有一份
`src/mpc_storage.py`，是否被 `_run_mpc_pass` 使用不影响本节结论，因为整条链在入口处就返回了。

---

## 3. SOC 的滚动传递

```
窗口 k: soc[0] = prev_soc      （init_soc 约束，dispatch_socp.py:260/268）
        soc[T] 由转移方程决定   （:463-467）
执行后: prev_soc = soc[n_commit]（rl_env.py:1080-1085）
```

即每个新窗口从**上一个窗口第 4 段的 SOC**重新开始，丢弃 12 段前瞻的尾部。
这与「MPC 用滚动实现」一致，但意味着前瞻期末的 SOC（`soc[16]`）**不被执行、也不被记账**。

---

## 4. 终端条件：三种 horizon_type 与是否钉住

`pins_terminal_soc(config, horizon_type)` 只在
`horizon_type in ("full_day", "day_end")` 且 `config.storage.terminal_soc_equal` 时返回 True
（`src/dispatch_core.py:31-57`）。`terminal_soc_equal` 默认 True。

| horizon_type | 使用场景 | 是否钉住 | RHS | 目标值来源 |
| --- | --- | --- | --- | --- |
| `window` | 训练/基线的前 23 个窗口 | **否** | — | — |
| `day_end` | 训练/基线的第 24 个窗口（`window_end == T`） | **是** | `terminal_soc_target` | `rl_env`/`b1_daily_baseline`/`train_da_gate` 传 `0.5` |
| `full_day` | `counterfactual_eval._settle_full_day`、`run.py`、`compare_methods` | **是** | `soc[0]`（= `soc0 = 0.5`） | end-to-start 钉住 |

选择逻辑在 `rl_env.py:961-963`：
`horizon = "day_end" if terminal_soc_target is not None and window_end == self.T else "window"`。

**后果**：一天的末态 SOC 被硬等式钉在 0.5，23 个中间窗口自由。
所以「跨期囤电」既不能带来日终结算收益（账本无期末项），也不能通过末窗口逃逸。

---

## 5. 期末库存价值：存在哪、不存在哪

| 对象 | 是否存在 | 位置 |
| --- | --- | --- |
| 出清目标中的期末估值 `+γ^T·TV·soc_T·e_max` | **存在**，每个窗口都有 | `dispatch_socp.py:614-618`；独立储能 `:98` |
| `TV` 取值 | `config.storage.terminal_value`，为 `None` 时取该窗口 `mean(wholesale)` | `:554-556` |
| 参与者账本中的期末项 | **不存在**（`PayoffBreakdown` 明示无 terminal 项） | `src/participant_payoff.py` 类注释 |
| RL 奖励中的期末项 | **不存在**，只有 SOC 势函数（C2） | `src/c2_reward.py:80-92` |
| 结算期（块）残值 | **不存在**：块收益只看已执行 4 段 | `rl_env.py:1054` `n_commit` |

**注意**：在 `day_end`/`full_day` 窗口里 `soc_T` 被钉成常数，因此 `γ^T·TV·soc_T·e_max`
在该窗口是一个**常数项**，不改变 argmax。也就是说：期末估值只在**中段窗口**起作用，
作为「16 段之后那一点库存的规划影子价」；到了真的日终，它自动失效。这是自洽的（PASS）。

---

## 6. 四个必须分开的「库存」概念

1. **MPC 窗口末 SOC**（`soc[16]`）：不执行、不记账，只在目标里被估值。**不存在独立的 MPC 末态约束。**
2. **24 h 日末 SOC**（`soc[96]`）：被 `day_end` 硬等式钉在 0.5。**存在且唯一。**
3. **结算块残留库存**：块收益只覆盖已执行的 4 段，第 4 段末的库存**不计价、不入账**。
   库存的跨块价值只通过下一窗口的初始化与 16 段前瞻间接体现。**这是设计，不是遗漏**
   （否则每块都要做存货估值，会与 24 h 结算重复计价）。
4. **会计库存价值**：日终为 0（因为 `soc_96 = soc_0 = 0.5`，且账本无该项）；
   日内没有账面价值。所以**任何「库存价值」数字都只能来自出清目标，不能来自账本**。

---

## 7. 末窗口可行域风险（结构性，未在训练日志中核实）

末窗口 `T = 4`，起点 `soc_0 = prev_soc`（第 23 块执行后的 SOC），终点被钉在 0.5。
在 4 个时段内可达的末态区间是：

```
charge:    Δsoc ≤ η_ch · p_ch_max · 4·dt / e_max   = 0.92·0.4·1.0/1.6 = +0.2300
discharge: Δsoc ≥ − (p_dis_max/η_dis) · 4·dt/e_max = −(0.4/0.92)·1.0/1.6 = −0.2717
⇒ 可行起点 soc_0 ∈ [0.5 − 0.23, 0.5 + 0.2717] = [0.2700, 0.7717]
```

而 SOC 的物理箱是 `[0.10, 0.90]`。**代码中没有任何约束阻止前 23 个窗口把 SOC 停在该区间之外。**
若末窗口的 `soc_0 ∉ [0.27, 0.77]`，该窗口的 SOCP 模型**不可行**。

发生时的代码行为（INFERRED，逐行可追）：
`dispatch_socp` 返回 `None` → `clear_market` 打印 `Batch solve failed, falling back...`
（`market.py:88`）→ 走**逐时段 DC/LinDistFlow 回退引擎**（`market.py:116-138`）→
返回带 `fell_back` 的结果 → `rl_env` 仍从该结果计算块收益（`rl_env.py:1043-1059`）。
于是第 23 块（**B 奖励唯一付费的块**）会在另一套物理模型下定价，
且 LMP 可能整体回落到批发电价。

- 代码层：**PASS**（无任何防护，路径可追）。
- 后果层：**INFERRED**（不可行 → 回退 → 换模型定价）。
- 是否在既有正式 run 中发生过：**NOT_TESTED**。核实需要读取 `train.log` / `episodes.csv` /
  审计产物，而本阶段授权只覆盖「当前正式源码 + 当前正式配置 + 历史机制文档」，
  故不越界。风险等级由人工裁决。

同一算术也适用于 `full_day` 结算：整日 96 段可达区间是 `[0.5 − 5.52, 0.5 + 6.52] ⊃ [0.1,0.9]`，
所以 `_settle_full_day` 的端点钉住**永远可行**，KPI 路径不受此风险影响。

---

## 8. 小结（不做设计建议）

- 时间结构：24 块 × 4 段执行 / 16 段前瞻，属实。
- MPC：未启用，参数无效。
- 终端条件：只有日末窗口与整日结算钉住；中段窗口自由。
- 期末库存价值：只存在于出清目标，不进账本、不进奖励；在钉住的窗口退化为常数。
- 风险点：末窗口钉住是一个可能不可行的硬等式，且失败时静默切换到另一套引擎。
