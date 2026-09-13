# 学术研究口径审计

本文档记录 2026-09-13 对项目研究口径的重构：改了什么、为什么改、哪些指标被保留、
哪些被降级、当前的定义是什么、还有什么没做、以及已知限制。

**本文档不含任何实验结果。** 凡未实测的项一律标注「未验证」。下面的所有数字都是
机制验证（证明代码行为符合预期），不是研究结论。

---

## 一、当前的研究口径（已固定）

| 项 | 定义 |
|---|---|
| 研究主题 | 基于 MATD3 的电力市场报价辅助决策 |
| 核心问题 | 报价辅助能否帮助市场主体在不同市场环境下稳定获得更高经济收益，以及提升多少 |
| 第一主案例 | 独立储能 |
| 主算法 | MATD3（CTDE），保留；TD3 等兼容但不干扰主流程 |
| 市场 | 日前市场；网络约束出清与 LMP 保留 |
| 时间尺度 | 96 × 15min（未改） |
| RL 输出 | 仅报价 `[bid_mult, offer_adder]`，不输出充放电功率 |
| 储能运行 | 由市场出清优化器决定 |

---

## 二、收益定义（当前）

统一口径：`Money = Price(CNY/MWh) × Power(MW) × DT_HOURS(0.25)`。
唯一来源是 `src/money.py`。

### 独立储能（`participant_payoff.IndependentStoragePayoff`）

```
Profit_ESS = Energy_Sales_Revenue − Energy_Purchase_Cost − Degradation_Cost
             − Other_Operating_Cost − Penalty
```

- `Energy_Sales_Revenue = Σ_t LMP_t · p_sell_t · DT`
- `Energy_Purchase_Cost = Σ_t LMP_t · p_buy_t · DT`
- `Degradation_Cost = cycle_cost · Σ_t (p_ch_t + p_dis_t) · DT`

**独立储能不获得任何消费价值。** 目标函数里 `+bid·ch` 给充电的「信用」不是它收到的钱，
它在 LMP 结算。这条是硬约束，有测试钉住（改 `bid_value`/`offer_cost` 不改变其收益）。

### Prosumer（`ProsumerPayoff`）

在通用五项账本上结算：`revenue − purchase_cost − generation_cost − degradation_cost
− other_cost − penalty`，其中 consumption 用**真实** `bid_value`（RL 只动 `bid_mult`）。

### 终端 SOC 中性化

`terminal_adjustment = terminal_price · (SOC_T − SOC_0) · e_max`，**独立于五个分量**，
不并入 `total`，以免污染任何求和恒等式。两臂必须用同一个 `terminal_price`。

---

## 三、Baseline 定义（当前）

统一在 `src/baselines.py`，全部是 `(obs) -> [bid_mult, offer_adder]`，与训练好的 actor 同形：

| 臂 | 定义 |
|---|---|
| `truthful` | `[1.0, 0.0]`——**论文 benefit 的反事实基准** |
| `rule` | SOC 优先（≤0.25 强充 / ≥0.75 强放），否则按价格带（0.9×/1.1× 均价）充放；全常量分支 |
| `myopic` | 只看当期价格、完全忽略 SOC |

规则按特征名读观测，布局变化会直接报错而不是静默错位。

`strategies/` 注册表（fixed/random/mpc/rl）保留给 `run.py` 的离线仿真，未改动。

---

## 四、评测协议（当前）

`src/counterfactual_eval.py`。

- 四臂：all-truthful / all-MATD3 / all-rule / all-myopic。
- **配对**：同 seed 下每臂前 `np.random.seed(day_seed)`，负荷、新能源、初始 SOC、网络、
  价格实现完全相同，唯一变量是报价策略。`assert_paired` **逐元素比对两臂价格曲线**，
  不一致即拒绝比较。
- **整日结算**：先因果滚动得到动作序列（策略看不到它被评判的那一天），再用同一序列
  整日出清并硬约束 `soc[T] == soc[0]`（`storage.terminal_soc_equal`）。这样不需要
  选 terminal price，两臂在「日内」上可比。
- 输出：`profit_ai / profit_baseline / profit_delta / profit_delta_rate /
  positive_benefit_rate` + `mean/std/median/p05/p25/p50/p75/p95`，per-agent 与 fleet 两套。
- 降级或回退的出清块**丢弃并计数**，绝不平摊进均值。

### 训练 reward ≠ 评测 benefit（重要）

| | 训练信号 | 论文 benefit |
|---|---|---|
| 字段 | `reward` = `raw_profit − local_baseline_profit` | `benefit` = `profit_ai − profit_baseline` |
| 反事实 | 窗口内、其他 agent 保持各自 RL 动作 | 整日、所有主体 truthful |
| 范围 | 单 block（4 期） | 整天 |

两者不可互相替代。`reward` 只是学习信号；`run_artifacts` 现在同时记录
`raw_profit_mean` / `local_baseline_profit_mean` / `profit_delta_local_mean`，
避免「actor loss 下降但利润不涨」被误读为成功。

---

## 五、改了什么、为什么

### 1. 储能同时充放电（`dispatch_socp.py` / `models.py`）

**问题**：`ch`/`dis` 是各自独立的连续变量且**无任何互斥约束**。同时充放电在功率平衡中
完全抵消，却在目标函数里净得 `discount_t·(bid − offer) − 2·cycle_cost`，即凭空制造
目标值。实测（baseline，truthful）：4 个工业 prosumer 共 churn 71.2 MW，目标虚增 15,664；
shaded bid（1.8×）下 12 个主体全部 churn，177.6 MW，虚增 85,453。

**根因不是 churn 系数，而是充电侧的虚构补贴**：储能主体的 `bid_value` 直接沿用了
其**负荷**类型的支付意愿（工业 720），远高于真实能量成本（wholesale≈500），
于是求解器满功率充电到 SOC 触顶、再放电腾空间。

**当前处理**：市场规则 `churn_free_quotes`——储能申报的充电报价不得超过放电报价
（`bid_eff = min(bid, offer)`）。这使 churn 系数 ≤ `−2·cycle_cost`，纯 QCP 下严格劣。

**代价（已知限制）**：该规则同时压掉了「充电报价 > 放电报价」这一电池表达套利的方式，
因此**当前设置下储能不会放电**。见第七节。

### 2. 货币量纲（全项目）

**问题**：整条利润/支付账本把「一期的功率」当作「一期的能量」，未乘 `DT_HOURS`；
而碳排、新能源、弃电、SOC 动态、终端 SOC 都已经乘了。同一目标函数混着两种量纲。

**处理**：`src/money.py` 作为唯一口径来源（`DT_HOURS`、`total_money`、
`price_from_balance_dual`）。所有货币项接入该口径。

**同时必须做的对偶反缩放**：目标函数乘 `DT` 会让功率平衡约束的对偶 `Pi` 也乘 `DT`，
`lmp = −Pi` 会变成真实价格的 ¼。所有对偶提取点都已改为 `−Pi / DT`。

**可验证性**：`config.dt_scaled_money = False` 复现旧目标函数，因此可以**证明**
（而非声称）该修改在纯经济项下是正标量缩放——实测 {socp, lindistflow} × {nodal reprice
开, 关} 四种组合下调度与去缩放后的节点价格均不变，SOCP 目标值精确 ×0.25。

**行为变化**：终端 SOC 项与多目标项本就量纲正确、不随之缩放，其**相对权重变为原来的 4 倍**。
这是经济上正确的权重（旧口径下基础经济项被高估了 4 倍），但确实改变出清行为。

### 3. 主体收益层（`participant_payoff.py`）

原先同一份 profit 公式在 4 处重复（`eval_agents` / `rl_env` / `surplus_metrics` /
`diagnose_profit`），各自手改量纲、可自由漂移。现全部委托给按 `participant_type`
分派的 payoff model。`RetailerPayoff` 为已声明占位（`NotImplementedError`），
防止售电公司被静默按 prosumer 结算。

### 4. 独立储能主体（`ess.py`）

零负荷、零新能源、`participant_type="storage"` 的 Agent，共址于已有节点（默认 2/12/24，
上中下游各一），报价锚定 `mean(wholesale)`（可用 `storage.bid_anchor` 覆盖）。
`storage.prosumer_storage_scale` 用于缩放网络自有电池，使 ESS 成为实质套利者；
该系数进入 config snapshot。

### 5. 训练 reward 与评测 benefit 分离

见第四节表格。

### 6. 其他修正

- `rl_bidding` / `rl_td3`：actor 正则项原为 `(penalty / scale) · logits²`，
  而 `−min(q1,q2)` 已在 reward-std 归一化单位，再除一次 `scale` 等于反归一化，
  使有效正则强度**反比于 reward 尺度**。已改为无量纲。**含义：有效强度与修复前不同，
  重训时需重新标定 `bid_dev_penalty` / `offer_dev_penalty`**；`update()` 现返回
  `reward_scale` 以便观测。
- `dispatch_ldf._opf_cache_key`：原先不包含 `exclusive_mode` / `dt_scaled_money` / 等
  影响模型结构的配置，缓存模型会拿一份配置的结构回答另一份配置的问题。已补全。
- `eval_agents`：原先不传 `--consumer-metrics` 时必然崩溃（`run_combined_episode`
  在 `capture=False` 时 pop 掉 `sched`，而 `main()` 无条件读它）。已改为 combined run
  始终 capture。
- `data_aggregator`：新增 `MONEY_CONVENTIONS` / `MONEY_UNIT_FIX_TS` 与
  `money_convention` 列，**默认不过滤**（磁盘上所有结果都属旧口径，默认过滤会清空所有视图）。

---

## 六、指标去留

### 保留为第一主指标

`profit_ai`、`profit_baseline`、`profit_delta`、`profit_delta_rate`、
`positive_benefit_rate`，以及其 mean/std/median/p05…p95。

### 保留但降级为辅助（secondary）

- **Consumer surplus / consumer payment / LMP markup**：`surplus_metrics` 保留并已统一量纲，
  但不再作为主目标或主 KPI。
- **Market power**：`surplus_metrics.market_power_split` 与 `diagnose_profit.decompose`
  保留（两者相等有测试钉住），作为解释性次要分析。**不再作为正面收益来源的默认解释。**
- **Nash / regret**：`nash.py` 保留，`--nash-regret` 仍可选，定位为「策略稳定性辅助分析」。
- **Social welfare**：不再作为主体收益的代名词。

### 明确不再作为性能证据

`actor_loss` / `critic_loss` / `q1_mean` / `q2_mean` / `q_gap` 仍记录用于诊断，
但**不得用来论证经济收益提升**。

### 结果字典的改名

`welfare` 实为 `m.ObjVal`（市场出清目标，且把储能按**其申报价格**计价，不是任何人的利润）。
现同时提供 `objective` 并保留 `welfare` 作为兼容别名；新增 `lmp_fallbacks` 以便
对偶不可用时的静默退化可见。

---

## 七、已知限制（必须写进论文或至少知悉）

1. **当前设置下独立储能不会放电。** `churn_free_quotes` 规则（`bid ≤ offer`）是
   零 churn 与快速求解的代价所在，但它同时禁止了电池表达套利所需的
   「充电报价 > 放电报价」。实测：truthful 下 ESS 充 2.61 MWh、放 0 MWh、SOC 收于 0.9。
   - 「差价 ≤ 2·cycle」是数学上正确的无 churn 条件，但**不充分**：shaded bid 下仍有
     39.5 MW churn（充电信用的贪婪充电机制）。加严格边际也无效——边际量级远低于
     QCP 障碍法的数值分辨率。
   - 精确二元互斥有效，但 **RL 工况下不可行**：16 期窗口单次出清 0.22s → 38.1s
     （全部储能）或 >8 分钟未完成（仅 ESS 三台）。`exclusive_mode="strategy_only"` 已实现，
     默认关闭并注明不可用于训练时段。
   - **唯一实测有效并使储能真正日内套利的是 `terminal_soc_equal`**（整日出清钉住端点）：
     钉住后 ESS 充 1.13 / 放 0.96 MWh、SOC 精确回到 0.500、等效循环 0.174。
2. **标量 `terminal_value` 对线性目标产生角点解。** 在 anchor × terminal_value 网格上
   每一格都是角点（要么充满 soc_max 要么放空 soc_min），无内点均衡。这正是改用
   整日端点约束的原因。
3. **旧结果与旧 policy 全部不可比。** 所有 CNY 绝对值变为原来的 ¼；
   `lmp_markup_*` 与 `benefit_rate` 这类比值不变。**主实验必须重训重评。**
4. **训练超参需重新标定**（见第五节第 6 条）。
5. **SOC 中性化只消除水平差、不消除风险差。** 两臂 SOC_T 分布差异仍影响日内收益方差。
6. **LinDistFlow 路径不满足精确的缩放不变性。** 它迭代线损外近似（及可选的节点价格重解）
   且最优解退化，因此其逐期调度、对偶与目标值在两次求解间不完全可复现；
   日级能量总量不变。SOCP（训练与评测实际使用的路径）是单次求解，检验更严。
7. **未修的既有问题**（不属于本次范围，已记录）：
   - `dispatch_socp.py` 的 `storage_units` 分支只有 ch/dis 变量与目标项，
     **没有 SOC 转移、没有功率上限、没有爬坡**，`soc[T]` 自由且被目标估价。
     该路径全项目无人构造，**不要使用**。
   - `dispatch_ldf.py` 的负荷约束写死 `a.load_forecast if True else a.load_real`，
     RT 阶段也用 DA 预测。只影响 LinDistFlow（SOCP 正确）。
8. **ESS 规模与 prosumer 缩放系数未标定。** 容量过大则 LMP 被自身影响（退化为市场力故事），
   过小则 benefit 落在噪声量级。需要消融。
9. **报价锚点 `mean(wholesale)` 的影响未充分验证。** 实测在标量 terminal_value 下
   锚点分位数选择对结论影响很小（因为终端项主导），但端点约束生效后的锚点行为**未验证**。

---

## 八、本次未做（计划内、待续）

- Stage 8 收益来源分解（`Δprofit = Δrevenue − Δpurchase − Δdegradation − Δother`，
  再做 timing / price / network 三分）
- Stage 9 训练/测试隔离（`--train-scenarios` / `--test-scenarios`，拒绝重叠，写入 manifest）
- Stage 10 节点位置实验
- Stage 11 Dashboard 报价辅助页（并把 welfare / market power / Nash 从 KPI 中心挪开）
- Stage 12 `run_experiments.py` 统一入口
- Stage 13 §25 的 10 项自动检查（单位一致性、无 NaN、失败块不入指标、部分日不得当作全天等）
- Stage 14 本文档的后续补全

---

## 九、验证状态

- **非慢速测试：203 项全部通过。**
- **慢速测试：未全绿，两项需说明。** 全套耗时约 54 分钟，最后一次完整结果为
  `29 passed, 2 failed`：
  - `test_surplus_metrics.py::test_consumer_metrics_end_to_end_capture`
    因 `KeyError: 'ESS2'` 失败——该测试的 `rl_names` 取「所有带电池的主体」，
    而主体集合现已包含独立储能机组，那套旧 checkpoint 没有对应的 policy。
    已修正为只驱动实际加载到 policy 的主体，单独复跑通过（13s）。
  - `test_baseline.py::test_stackelberg_improves_leader_payoff`
    在一次完整运行中失败、单独复跑通过，属**容差边界**上的不稳定测试：
    它用绝对值 `−5.0` 做容差，而全部货币量已变为原来的 ¼，该容差相对而言紧了 4 倍。
    **尚未修正，属已知遗留项**，重训前应改为相对容差。
  - **因此慢速套件的全绿状态仍是未验证项**，需要在上述两项稳定后重跑一次完整套件。
- 已实测的机制验证（**均非研究结论**）：
  - 储能互斥：默认配置与 RL 路径下 churn = 0；负向对照（关规则）可复现 churn。
  - 量纲不变性：四种 (solver × nodal reprice) 组合下调度与节点价格不变。
  - 收益与报价无关：改 `bid_value`/`offer_cost` 不改变 ESS 收益。
  - 端到端配对评测可跑通，产出 headline 三数，配对校验通过，0 天被丢弃。
  - 整日端点约束：滚动结算 → 钉住端点的收益由 −1519/−1665/−1669 变为 −246/−41/−31。
- **未验证**：慢速套件全绿；任何真实训练产出的 policy；任何场景下的真实 benefit 数字。

---

## 十、下一步（按优先级）

1. **完整重跑慢速测试套件**并确认全绿；先把 `test_stackelberg_improves_leader_payoff`
   的绝对容差改为相对容差。
2. **用新口径训练一套 ESS 的 MATD3 policy**（`train_rl.py` 默认已选中 ESS 机组）。
   这是拿到任何真实 benefit 数字的前置条件，也是第一次真实压力测试——
   重点观察 `reward_scale` 与正则项有效强度。
3. 依据实测重新标定 `bid_dev_penalty` / `offer_dev_penalty`。
4. 完成 Stage 8–13。

### 必须继续研究的设计问题

**储能目标项 `+bid·ch` 的写法仍未被证明正确。** 它是当前唯一让储能愿意充电的机制，
也是 churn 的根源，而抑制它的手段（`bid ≤ offer`）同时抑制了套利。候选方向：
把充电**计成本**而非计收益、或改成单一报价阈值（`disc·π·(dis − ch)`，churn 系数结构上恒为负）。
后者会改变动作语义（二维降为一维），偏离 §11 的规格，需要明确决策后才能动。
在这一点解决之前，独立储能作为第一主案例的行为是退化的。
