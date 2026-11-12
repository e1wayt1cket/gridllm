# 当前出清数学模型（从源码还原）

审计对象：MakerB 工作树 `63b008d` + 未提交改动，`src/` 与 `config/` 的实际内容。
方式：只读源码静态还原，不重新清算。所有表达式后标注 `文件:行号`。

证据分级：**PASS** = 源码/配置直接给出；**INFERRED** = 由源码做代数推导但未重新求解验证；
**NOT_TESTED** = 需要重新清算或有限差分才能确认，本阶段禁做。

---

## 0. 一句话结论

当前出清是一个**单次求解、多时段、辐射网 SOCP 最优潮流**，目标函数在 MAXIMIZE 方向上是
「负荷申报价值 − 发电/储能机会成本 − 弃负荷罚 − 对外购电成本 − 循环成本 + 期末库存估值」的混合式，
**不是**标准意义上的社会剩余最大化，也**不是**结算金额表。网络（电阻、网损、电压、线路容量）是真约束，
外部电网是唯一的边际能量来源，其价格外生给定。

---

## 1. 代码身份：被审计的是 V4 候选，不是 v3 冻结协议

| 项 | 结果 | 证据 |
| --- | --- | --- |
| 工作树 vs v3 代码清单（19 个 pin） | **8 个不一致 / 11 个一致** | 见 `verification_report.json` `protocol_identity` |
| 工作树 vs v4 代码清单（22 个 pin） | **22 个全部一致** | 同上 |
| v3 配置归一化哈希 | `1b47e0ed…` 复算一致 | `formal_training_v3/formal_training_manifest.json` |
| v4 配置归一化哈希 | `7834d0aa…` 复算一致 | `formal_training_v4/formal_training_manifest.json` |
| 归一化方式 | `sha256(json.dumps(yaml.safe_load(cfg), sort_keys=True))` | `src/make_formal_training_manifests.py:274` |

不一致的 8 个文件：`rl_da_gate.py`、`rl_b_c2_gate.py`、`train_da_gate.py`、`da_quote_map.py`、
`rl_spec.py`、`run_formal_training.py`、`training_contract.py`、`make_formal_training_manifests.py`。

**含义**：`dispatch_socp.py`、`rl_env.py`、`c2_reward.py`、`daily_reward_b.py`、`b1_daily_baseline.py`、
`v2_cy_jn_env.py` 等出清与奖励路径文件在 v3/v4 中哈希一致，因此**本次出清模型审计对两个协议都成立**；
但动作/映射/奖励缩放/门控层的描述对应的是 **v4（`V4_candidate_001`，`fixed_scale=100`，
`action storage_quote_v4`，gate logit 上界 4.0）**，不是 CLAUDE.md 中所写的 v3（`fixed_scale=1000`）。
本报告凡涉及这些文件处，一律以工作树为准并注明。

CLAUDE.md 里的 `AGENTS.md`/规则文件仍把 `formal_training_v3` 称为 live manifest，与工作树实际状态不符：
**这是文档与代码的状态差，不是出清机制的差**。按规则「不得为让检查通过而修改冻结文件」，
本阶段只记录、不修正。

---

## 2. 索引与集合

- `t ∈ {0,…,T-1}`：时段。RL 训练路径每次滚动窗 `T = 16`；全天结算 `T = 96`。
- `b ∈ B`：IEEE 33 节点母线（`dispatch_socp.py:186`）。
- `l ∈ L`：从 slack 出发的辐射状线路集合（`_select_radial_lines`，`dispatch_socp.py:187/265`）。
- `a ∈ A`：参与主体（居民/商业/工业用户、prosumer）。
- `su ∈ SU`：独立储能机组，本场景 4 台，`ESS9/ESS14/ESS17/ESS27`（`src/ess.py` `build_ess_fleet`）。
- `slack_bus = net.ext_grid.at[0,'bus']`（`dispatch_socp.py:188`），IEEE33 即母线 0。

单位：功率 MW，时间 h，能量 MWh，价格 CNY/MWh，金额 CNY（`src/money.py` 文档字符串）。
`DT_HOURS = 0.25`（`src/money.py:21`）。

---

## 3. 决策变量

| 变量 | 含义 | 界 | 出处 |
| --- | --- | --- | --- |
| `v[t,b]` | 电压平方 (pu²) | `[v_min², v_max²] = [0.93², 1.05²] = [0.8649, 1.1025]` | `dispatch_socp.py:224` |
| `P[t,l]`,`Q[t,l]` | 线路有功/无功潮流 (pu) | 自由 | `:226-227` |
| `I_sq[t,l]` | 电流平方 (pu²) | `≥ 0` | `:228` |
| `p_grid[t]` | 与外部电网净交换 (pu, 由 add_grid_exchange) | `[-reverse_power_limit_mw, +∞) = [-2.5, +∞)` | `dispatch_core.py:178` |
| `imp_aux[t]` | 购电辅助变量 | `≥ 0`，且 `≥ p_grid[t]` | `dispatch_core.py:180-182` |
| `q_grid[t]` | slack 无功交换 | 自由 | `:230` |
| `served[a][t]`,`unserved[a][t]` | 已供/未供负荷 | `≥ 0` | `:242-243` |
| `pv[a][t]`,`wind[a][t]` | 实际使用的风光 | `≥ 0` | `:244-245` |
| `ch[a][t]`,`dis[a][t]` | 充/放功率 | `≥ 0` | `:246-247`, `:264-265` |
| `net[a][t]` | 净功率（正=放） | `[-p_ch_max, +p_dis_max]` | `dispatch_core.py:94` |
| `soc[a][t]`, `t=0..T` | 荷电状态 | `[soc_min, soc_max] = [0.10, 0.90]` | `:258`, `:266` |
| `q_re[a][t]` | DER 无功注入 | `[-s_max, +s_max]` | `:253`, `:445-446` |
| `is_ch/is_dis` | 充放互斥二进制 | 0/1 | `:296-298`，本场景**不生成** |

`net` 变量是充放的母变量：`ch ≥ -net`，`dis ≥ net`（`dispatch_core.py:97-98`）。
因为目标里 `ch`、`dis` 的系数在所有生效分支都为负（申报充电价 `≥0` + `cycle_cost>0`，
申报放电价 `≥0` + `cycle_cost>0`），最优点上 `ch = max(-net,0)`、`dis = max(net,0)`，
同时充放在**表达式层面不可表示**（`dispatch_core.py:63-99`）。这也是动作箱必须非负的原因（`rl_spec.py:340-345`）。

---

## 4. 目标函数（完整）

`m.setObjective(obj, GRB.MAXIMIZE)`（`dispatch_socp.py:628`）。令：

- `dt = dt_scale = DT_HOURS = 0.25`，仅当 `config.dt_scaled_money=False` 时取 1.0（`:566`）；
- `γ = config.storage.discount_factor = 0.997`（`models.py` `StorageConfig`）；
- `c = config.storage.cycle_cost = 100.0` CNY/MWh；
- `π^u = config.market_design.penalty_unserved = 2500.0` CNY/MWh；
- `w_t` = 外生批发电价（CNY/MWh），见 §6；
- `TV` = 期末库存估值（CNY/MWh），`= config.storage.terminal_value`，为 `None` 时取
  `TV = mean(w_0..w_{T-1})`（`:554-556`）；
- `N_a` = `a.storage` 非空的 prosumer 集合；`SU` = 独立储能。

```
max  Σ_{t=0}^{T-1} dt · [  Σ_{a∈A}  b_{a,t} · served_{a,t}                                    (O1)
                         +  Σ_{a∈N_a} ( −γ^t · [ b_{a,t}·ch_{a,t} + o_{a,t}·dis_{a,t} ] )      (O2,O3)
                         +  Σ_{a∈N_a} ( −c · (ch_{a,t} + dis_{a,t}) )                         (O4)
                         −  Σ_{a∉N_a}  o_{a,t} · (pv_{a,t} + wind_{a,t} + dis_{a,t})          (O5)
                         −  Σ_{a∈A}  π^u · unserved_{a,t}                                  (O6)
                         −  w_t · p_grid_t ]                                              (O7)
    +  γ^T · TV · Σ_{a∈N_a} e_a · soc_{a,T}                                                   (O8)
    +  Σ_{su∈SU} [ −dt·Σ_t ( b_{su,t}·ch_{su,t} + o_{su,t}·dis_{su,t}
                              + c·(ch_{su,t}+dis_{su,t}) )
                   + γ^T · TV · e_{su} · soc_{su,T} ]                                        (O9)
```

逐项：

| # | 表达式 | 位置 | 说明 |
| --- | --- | --- | --- |
| O1 | `+b·served·dt` | `:574` | 负荷按**申报**购电价计价（`b = bid_mult × bid_value`，非储能） |
| O2 | `−γ^t·b·ch·dt` | `:584-586` | 充电按申报充电价计**成本**（越贵越少充） |
| O3 | `−γ^t·o·dis·dt` | `:584-586` | 放电按申报放电价计**成本**（越贵越少放） |
| O4 | `−c·(ch+dis)·dt` | `:587-589` | 循环成本，**不带 γ^t** |
| O5 | `−o·(pv+wind+dis)·dt` | `:590-591` | 非储能主体的风光发电机会成本 |
| O6 | `−π^u·unserved·dt` | `:592-593` | 弃负荷罚，2500 CNY/MWh ≈ 8.67 × `PRICE_REF_CNY_MWH` |
| O7 | `−w_t·p_grid_t·dt` | `:594` | 对外净购电成本（净额为负时即卖电收入），**不带折扣** |
| O8 | `+γ^T·TV·e·soc_T` | `:614-618` | prosumer 储能期末库存估值 |
| O9 | `_add_independent_storage` | `:74-98` | 独立储能：三项成本**均不带 γ^t**，期末估值同 O8 |

### 4.1 目标函数的性质（必须明确的三点）

1. **不是社会福利**：负荷侧用申报 `b`（不是真实支付意愿），发电侧用申报 `o`（不是真实成本），
   储能两侧都是「申报价 × 功率」的成本形式。这是**申报参数驱动的模型目标**，
   与 `surplus_metrics.reconciliation` 里定义的社会剩余不是同一对象。
   项目自己的措辞也一致：`clearing_objective` 是「出清模型自身的目标」，不是参与者收益
   （CLAUDE.md「Result dict schema」条）。
2. **折扣不一致**：prosumer 储能的报价项乘 `γ^t`，其循环成本、独立储能的三项成本、
   购电成本 `w_t·p_grid_t`、负荷价值都不乘 `γ^t`。当 `T=16` 时 `γ^t ∈ [1.0, 0.9531]`，
   当 `T=96` 时 `γ^t ∈ [1.0, 0.7494]`。也就是同一个「未来一块钱」在目标里最多差 25%。
   这是代码事实（PASS）；它是否构成经济偏差是判断题，本阶段不做。
3. **期末估值是线性项、只作用于时段末**：`γ^T·TV·soc_T·e_max`，`TV = mean(w)`。
   它是一个合法的「末日能量影子价值」，但**只在窗口末端出现一次**，窗口内部的跨期价值
   只能靠 SOC 转移方程与各时段报价体现。

### 4.2 生效 / 不生效清单（当前配置）

| 机制 | 状态 | 证据 |
| --- | --- | --- |
| 加权多目标碳项 | **不生效** | `enable_multi_objective: false`（`config/defaults.yaml market_design`） |
| 加权 RE 奖励 / 弃风弃光罚 | **不生效** | 同上 |
| 碳上限约束 / RE 最低比例约束 + 影子价 | **不生效** | `use_constraint_multi_obj: false` |
| 碳/RE 松弛罚 | **不生效** | 同上 |
| 充放互斥二进制 | **不生效** | `exclusive_mode: off`（`models.py:83`），`_needs_binaries` 返回 False |
| MPC 自调度固定 | **不生效** | `self_schedule: false` → `_compute_mpc_schedules` 直接返回 `{}`（`dispatch_core.py:703-704`） |
| 双向潮流闭锁 | **不生效**（无二进制，`_freeze_exclusive_directions` 不调用） | `:642` |
| 电网爬坡约束 | **不生效** | `ramp_limit_mw_per_period: None` |
| 储能爬坡约束 | **不生效** | `StorageSpec` 各 ramp 为 None |
| 无功支撑 | **生效** | `reactive_support: true` |
| `use_nodal_price` 重解环 | 代码会跑，但**代数上是恒等重解**（见 §7） | `:651-714` |

---

## 5. 约束（完整）

| # | 约束 | 数学形式 | 索引 | 位置 |
| --- | --- | --- | --- | --- |
| C1 | 参考电压 | `v_{t,0} = 1` | `t` | `:325` |
| C2 | **节点有功平衡** | `1_{b=0}·p_grid_t + Σ_{a@b}(pv+wind+dis−served−ch) + Σ_{su@b}(dis−ch) − Σ_{l:from=b}P + Σ_{l:to=b}(P − r_l·I²) = 0` | `(t,b)` | `:329-353` |
| C3 | 节点无功平衡 | `1_{b=0}·q_grid + Σ_{a@b}(−served·q_ratio + q_re) − Σ_{l:from=b}Q + Σ_{l:to=b}(Q − x_l·I²) = 0` | `(t,b)` | `:356-377` |
| C4 | DistFlow 压降 | `v_{t,to} = v_{t,from} − 2(r_l P + x_l Q) + (r_l²+x_l²)·I²` | `(t,l)` | `:386-391` |
| C5 | 二阶锥 | `P² + Q² ≤ v_{t,from}·I_sq` | `(t,l)` | `:395-399` |
| C6 | 线路容量 | `I_sq ≤ I_max²` | `(t,l)` | `:402` |
| C7/C8 | 风光上限 | `pv ≤ max(pv_max,0)`，`wind ≤ max(wind_max,0)` | `(a,t)` | `:416-417` |
| C9 | 负荷平衡 | `served + unserved = max(load,0)` | `(a,t)` | `:418` |
| C10 | 充放功率上限 | `ch ≤ p_ch_max`，`dis ≤ p_dis_max` | `(a,t)` | `:425-426` |
| C11 | 净功率包络与分解 | `net ∈ [−p_ch_max, p_dis_max]`，`ch ≥ −net`，`dis ≥ net` | `(a,t)` | `dispatch_core.py:94-98` |
| C12 | 逆变器容量（菱形） | `q_re + p_total ≤ s_max`，`−q_re + p_total ≤ s_max`，`p_total = pv+wind+dis` | `(a,t)` | `:445-451` |
| C13 | SOC 转移 | `soc_{t+1} = soc_t + (η_ch·ch − dis/η_dis)·dt/e_max` | `(a,t)` | `:463-467` |
| C14 | 初始 SOC | `soc_0 = storage.soc0 = 0.5` | `a` | `:260`, `:268` |
| C15 | 期末 SOC 钉住 | `soc_T = soc_0`（`full_day`）或 `= terminal_soc_target`（`day_end`） | `a` | `:486-497` |
| C16 | SOC 箱 | `soc ∈ [0.10, 0.90]` | `(a,t)` | `:258`, `:266` |
| C17 | 碳上限 | `carbon_expr − slack_carbon ≤ cap` | — | `:524`（不生效） |
| C18 | RE 最低比例 | `re_expr + slack_re ≥ target` | — | `:536`（不生效） |

要点：

- **C2 的对偶就是 LMP 的来源**（`:353` 保存 `p_bal_constr[(t,b)]`，`:717` 提取）。
  C3 的对偶**不提取**，故没有「节点无功价格」。
- C2 中的网损项 `r_l·I²` 记在**受端母线**的平衡式里（`:348-350`），这是辐射网的标准做法，
  但意味着「母线 b 的注入 = 母线 b 的负荷 + 该母线出线的全部网损」。
- `q_ratio = tan(arccos(pf))·(1 − reactive_fraction_of_load_q)`（`:318`）。本场景
  `load_power_factor = 0.8503`、`reactive_fraction_of_load_q = 2.0`，故
  `q_ratio = −0.6190`，无功平衡里 `−served·q_ratio = +0.619·served`：
  **负荷在无功上表现为注入（超前）**。这是 `scenarios.yaml` 的模型化选择，不是符号错误，
  但它决定了无功电压特性，从而影响电压约束是否 bind。
- C12 的 `s_max = (pv_cap + wind_cap)·1.10 (+ p_dis_max·1.10)`，用的是**装机容量**而非当前出力上限；
  `p_total` 不含 `ch`，故充电时逆变器容量约束不收紧。这是代码事实（PASS），
  其物理合理性不在本阶段裁决范围。
- 本场景 4 台 ESS：`capacity 1.6 MWh`、`power = capacity × 0.25 = 0.4 MW`、
  `η_ch = η_dis = 0.92`、`soc0 = 0.5`、`soc ∈ [0.10,0.90]`、`bid_value = offer_cost = 0`（`src/ess.py`）。
  单位满功率一小时使 SOC 变化 `0.92 × 0.4 × 1.0 / 1.6 = 0.23`。

---

## 6. 外部电网 / PCC / slack bus

- 只有一个净交换变量 `p_grid[t]`：`lb = −reverse_power_limit_mw = −2.5`，`ub = +∞`
  （`dispatch_core.py:178`）。**购电无上限，反送上限 2.5 MW**。
- 净买入的量：`imp_aux[t] ≥ max(p_grid,0)`（`dispatch_core.py:180-182`）；
  但在目标里 `imp_aux` 只出现在 `lambda_carbon > 0` 的碳项（`:596-599`），
  当前配置下**没有被定价**，因此碳排放在目标里不产生任何边际成本。
  实际碳排放在结果里另算：`grid_carbon_tco2(p_grid_mw, 0.58)`（`dispatch_core.py:140-148`）。
- `p_grid` 直接进 slack 母线的 C2（`:331`），所以 slack 母线是**平衡机 + 唯一外部接口**；
  母线 0 的电压固定 `v=1`（C1）。
- 外部价 `w_t` **外生**：`wholesale_price.JnPriceProvider.get_day_ahead_price` 返回
  `curve[t_start : t_start+T]`（`src/wholesale_price.py`），来源
  `data/processed/v2_cy_jn_2025/price_curves.npz`，`price_unit_confirmed: False`。
  清算过程**不能改变 `w_t`**：模型里没有任何反向作用通道。
- 因此：**外部价是能量锚，网络把锚价空间化**。slack 母线的 LMP 等于 `w_t`
  （INFERRED，见 §7.2）；其余母线的 LMP = `w_t` + 网损/阻塞分量。
  后者占多少比例，需要重新清算才能量化 → **NOT_TESTED**。

---

## 7. 求解与价格提取

### 7.1 求解设置（`dispatch_socp.py:206-221`）

`Method=2`（barrier）、`QCPDual=1`、`BarHomogeneous=1`、`OutputFlag=0`。
注释明确说明 `BarHomogeneous` 存在的原因：某些价格实现下普通 barrier 返回 OPTIMAL 但**不出对偶**，
届时每个母线都会静默回落到批发电价，看起来像「节点价被抹平」而不是像失败。

### 7.2 LMP

```
λ_{t,b} = − Pi_{p_bal(t,b)} / dt_scale
```
（`dispatch_socp.py:123` → `money.price_from_balance_dual`，`src/money.py:35`）

- 取的是 C2 的对偶 `Pi`，符号翻转后除以**同一个** `dt_scale`（`store` 的是 `dt_scale`，
  不是硬编码 `DT_HOURS`）。取 `dt_scale` 而不是 `DT_HOURS` 是刻意的一致化：
  目标按 `dt_scale` 缩放，价格也必须按同一个因子还原（`money.py` 文档字符串）。
  当前两条路径 `dt_scaled_money = True`，故 `dt_scale = 0.25`。
- 失败回落：`except (AttributeError, gp.GurobiError)` → `lmp[t,i] = wholesale[t]`（`:126-130`），
  并记录 `lmp_fallbacks / lmp_fallback_periods / lmp_price_source`
  （`dispatch_core.py:555-570`）。回落是**逐母线逐时段**的，不是整体的。
- `result["price"] = lmp.mean(axis=1)`：96 长度的**跨母线算术平均**。这是给旧 UI /
  `two_settlement` 用的聚合价，**不是**节点价，也不是正式结算价。
- 混合整数：本路径无二进制（`exclusive_mode: off`、`self_schedule: false`），
  故不存在「QCP 对偶不可用」结构性问题；若将来开启，代码路径是
  `_freeze_exclusive_directions` 后**再解一次**（`:642-649`），对偶取自第二次。
- **slack 母线 LMP = w_t**：INFERRED。推导：C2 在 slack 母线上的对偶满足
  `Pi = −dt·(∂obj/∂inj)`；目标中注入侧的边际成本是 `−w_t·dt`（每多注 1 MW·dt 少购 1 MW·dt），
  故 `Pi = −w_t·dt`，`λ = −Pi/dt = w_t`。**未重新求解验证**。
- 储能边际时的 LMP（INFERRED）：充/放两腿的目标系数为
  `−(γ^t·b + c)`（充）与 `−(γ^t·o + c)`（放），故边际储能所在母线
  `λ^charge_b ≈ γ^t·b + c`、`λ^discharge_b ≈ γ^t·o + c`。
  即**循环成本 c 会被打进该母线的边际价格**；在真申报 `(0,0)`、`γ^t≈1`、`c=100` 时，
  一个完整循环所需的价差下限约为「每交付 1 MWh 需 100/η_rt + 100 ≈ 218 CNY/MWh」。
  这是代数推导，**未做有限差分验证**（按任务书要求，有限差分属后续独立任务）。

### 7.3 求解流程

单次 `m.optimize()` → （无二进制，跳过方向闭锁）→ **`use_nodal_price` 重解环**：

- 触发条件：`all_storage 非空 且 config.storage.use_nodal_price`（`:656`）。当前
  `use_nodal_price: true`，所以**RL 路径也会进入该环**——`dispatch_socp.py:653-654` 的注释
  写「It is skipped entirely when use_nodal_price is False (RL path)」，
  与实际配置不符（注释错，代码行为为准）。
- 环内 `obj2` 与 `obj` **代数恒等**（同样 5 个分支、同样 `dt_scale`、同样期末项；
  多目标分支因 `enable_multi_objective=False` 不进入）。因此 `max_change` 恒为 0 < 1.0，
  循环在第 1 次即 `break`。**净效果：每次出清多解一次同样的模型，解与对偶不应改变。**
  这是纯开销（PASS：代数恒等；「不应改变对偶」严格说只在非退化情形成立——两次求解若落到
  不同最优解，对偶可能不同，本阶段未测 → 记为风险）。
- 之后 `clearing_objective = m.ObjVal`（`:754`）。

---

## 8. 时间尺度与单位换算

| 量 | 变换 | 位置 |
| --- | --- | --- |
| 功率→能量 | `× DT_HOURS = 0.25` | `money.total_money`，`money.energy` |
| 目标中的金额 | 全部 `× dt_scale`（多目标三项除外，用 `DT_HOURS`） | `:566`, `:596-608` |
| 平衡式对偶→价格 | `÷ dt_scale` | `:123` |
| SOC | `Δsoc = (η_ch·ch − dis/η_dis)·0.25 / e_max` | `:463-467` |
| 折扣 | `γ^t`, `γ = 0.997`；`γ^16 = 0.95307`，`γ^96 = 0.74944` | `:614` |
| 净交换 | `grid_import_mw/grid_export_mw` 由 `split` 得到（`dispatch_core.grid_import_export`） | `:126-137` |

---

## 9. 与「标准市场模型」的结构差异（只列结构，不评价优劣）

1. **没有 bid-ask 撮合**：负荷报价不是价格曲线，而是一个常数 `b`（`bid_value × bid_mult`），
   加上硬需求 `served + unserved = load`，弃负荷以固定罚价 2500 结算。
   「价格」由对偶给出，不参与工作量出清。
2. **储能报价不是报价单，而是保留价**：两侧都是 `≥ 0` 的成本加成，真申报点 `(0,0)`
   在箱的**下边界**上。动作空间**只能加价、不能让价**（下界即真申报）——
   这是 `rl_spec.py:340-345` 明示的机制约束，也是当前动作空间的结构性不对称。
3. **同一条母线同一时段只有一个能量价**（对偶），没有容量市场、没有辅助服务市场、
   没有分区电价以外的其它价格。
4. **滚动窗清算**：每次清 `T=16` 个 15 分钟时段，只执行前 `n_commit=4` 段（`rl_env.py:921-923`），
   其余 12 段用作前瞻、不执行。这就是「MPC 式的窗口」的全部含义——**没有独立的 MPC 模块在跑**
   （`self_schedule: false`）。
5. **外部价对清算完全外生且无弹性**：`p_grid` 无上限，故外部电网在本模型中
   永远能补足缺口，削峰填谷只可能来自储能的价差。
