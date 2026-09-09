# 三层指标层落地 + full-obs MATD3 3-seed baseline 重测（2026-09-08）

> **口径修正（2026-09-09）**：`qbuy` 定义改为 `served−(pv_used+wind_used)`，**不再以 p_dis 抵消**（见本文 §1 表）。此修正使下方 cs/cp/markup 数值过时、符号可能翻转（RL 在四场景 CS 由负转正）；`profit/genuine/power` 不受影响。跨 run 比较消费者侧请用修正后语义（最新结果见 `docs/2026-09-09-quickval-masac-phys.md`）。
>
> 状态：代码+单测完成，三 seed 训练与 K=3 paired eval 完成。方法学第一步：回答"储能利润从哪来、谁在承担代价"。上游决策见 `docs/2026-09-08-mad3pg-diffusion-handoff.md`（MAD3PG 封存、P0=full-obs 视为已完成）。

## 1. 动机与口径

原主指标 `genuine_welfare_delta` 是系统净值，无法区分储能赚的钱来源、用户是否受损。新增**三层账**把单个出清日切开，全部基于**真实出清价/真实 bid_value**、与 profit 账同约定（**不乘 0.25 DT 因子**）。

| 层 | 量 | 定义（负荷代理 a，时段 t） |
|---|---|---|
| 用户 | 支付 CP | `CP_a[t]=lmp[t,a.bus]·qbuy_a[t]`，`qbuy=max(served−(pv_used+wind_used),0)`（就地出力先满足；**储能放电 p_dis 不抵扣**——其价值在储能层单独结算；充电 p_ch **不进** CP） |
| 用户 | 剩余 CS | `CS_a=Σ bid_value·served_a − CP_a`（价值恒用真实标量 bid_value，绝不用 bid_mult） |
| 机制 | LMP markup | 负荷支付加权 `Σ(lmp−w)qbuy / Σ w·qbuy`（w=wholesale，w>1 时段） |
| 机制 | 市场力 | per-storage `power=Σ(q_b+q_r)/2·(lmp_r−lmp_b)`，`arb=Σ(q_r−q_b)(lmp_b+lmp_r)/2`（同 `diagnose_profit.decompose`，parity 测试锁定） |
| 系统 | genuine_welfare | 沿用（辅助） |
| 校验 | reconciliation | `ΣCP+ΣBC ≡ wholesale·import + rent`，rent=Σ lmp·d − w·import（拥塞/损耗盈余，无损无拥塞例为 0；实测见 §4） |

实现：`src/surplus_metrics.py`（纯 numpy）。eval 接入 `--consumer-metrics`（capture 整日 sched/lmp/wholesale，CSV 追加 `cs_*/cp_*/lmp_markup_*/market_power_*`）。测试 `tests/test_surplus_metrics.py`（9 快速 + 1 slow 端到端；全量回归 71 passed）。

## 2. baseline 重测（当前代码 full-obs MATD3 rotation，200 集，四场景轮换）

- seed 42（run 1，09-08 复用）；**新增 seed 7、123**（`policies/matd3_cc_rot_seed{7,123}`，各 ~56 min，detached 串行；critic loss 均平滑有界，健康）。
- K=3 paired eval（`--eval-seed 2026 --combined-only --consumer-metrics`）→ `results/matd3_cc_rot_seed{42,7,123}_three_layer_eval.csv`；种子均值 `results/three_layer_rot_seed_means.csv`。

**三种子均值表（RL − truthful，k CNY，markup 无量纲）**

| 场景 | profit_d | genuine_d | cs_d | cp_d | markup base→rl (Δ) | arb | power |
|---|---|---|---|---|---|---|---|
| baseline | +15.09 | +7.65 | −5.18 | +5.18 | 0.0478→0.0448 (−0.0029) | +9.33 | +0.33 |
| high_re | +10.31 | +30.04 | −0.48 | +0.48 | 0.0455→0.0428 (−0.0027) | +7.57 | +0.26 |
| peak_load | +14.64 | **−16.08** | −8.66 | +8.45 | 0.2270→0.2156 (−0.0115) | +8.62 | +1.28 |
| congestion | +18.03 | +1.68 | +0.18 | −0.22 | 0.3617→0.3154 (−0.0463) | +8.88 | **+3.75** |

CS/CP 基线与 RL（k CNY）：baseline 99.3/94.1（cs）、153.0/158.2（cp）；high_re 114.5/114.0、137.8/138.3；peak_load 88.6/79.9、288.9/297.3；congestion 50.4/50.6、238.8/238.6。

## 3. 结果解读（初读）

1. **跨 seed 高度一致**（best 策略收敛）：genuine 仅在 peak_load 为负（−15~−16k，与旧 best-K3 −12.6~−15.5k 同量级），重新锚定在 full-obs 基线上成立。
2. **利润主要来自套利时序而非价格影响**：`arb` 7.6–9.3k 全面占主导；`power`（价格影响项）在 baseline/high_re 仅 ~0.3k，但在 **peak_load (+1.3k) 与 congestion (+3.8k)** 显著 —— 市场力出现在网络拥塞/紧张时。
3. **markup（支付加权溢价）在四场景都下降**（congestion −4.6pp 最明显）—— RL 没有抬高"对 wholesale 的溢价"；但**绝对消费者支付 CP 在 3/4 场景上升**（baseline +5.2k、peak_load +8.5k），只因导入量被重新时序化到更贵时段。congestion 例外（cp_d ≈ −0.2k）。
4. **CS 镜像 −CP（baseline/high_re 精确）**：负荷总价值不变 → 是纯支付转移；用户损失最大在 peak_load（−8.7k），恰是 genuine 最负之处；congestion 用户侧近中性却带最大 power 项 —— 转移方向（power 提取是否被更低 premium 抵消）需按 agent/时段再拆，暂不断言。
5. **reconciliation 校验通过**：rent 均非负、占账单比例 baseline/high_re ~5%、peak_load ~22%、congestion ~35%；RL 同时略降账单与 rent（拥塞缓解的方向与"压低溢价"一致）。

## 4. reconciliation 实测（seed42，单日）

| 场景 | rent base→rl (k CNY) | rent/bill base→rl |
|---|---|---|
| baseline | 7.67 → 6.61 | 0.050 → 0.046 |
| high_re | 6.21 → 5.38 | 0.048 → 0.044 |
| peak_load | 54.88 → 52.24 | 0.221 → 0.216 |
| congestion | 64.26 → 56.64 | 0.352 → 0.326 |

## 5. 口径与注意

- CP/CS/markup 基于 DA 节点 LMP 单结算 + rolling 出清口径；**两结算（DA+RT）与真实零售电价不在本口径**，需单独扩展。
- 储能代理若是储能业主，其自有负荷计入用户层；仅充电流 p_ch 被排除（计入储能层 profit）。
- `power/arb` 用 base vs RL 双价格差（两日配对同 seed），沿用 diagnose_profit 定义；解读为"相对真实出清的获利来源"，非反事实均衡。
- 复现：见 §6 命令；`results/_smoke_three_layer.csv` 为开发期冒烟产物（scratch）。

## 6. 复现命令

```bash
# 训练（本轮新增 seed 7/123；seed42 为 run1）
PYTHONPATH=src ... python src/train_rl.py --algo matd3 --seed 7 --episodes 200 \
  --scenarios baseline,high_re,peak_load,congestion --eval-scenarios baseline \
  --save-dir policies/matd3_cc_rot_seed7

# K=3 paired eval（三层指标）
PYTHONPATH=src ... python src/eval_agents.py \
  --policies policies/matd3_cc_rot_seed42/best \
  --scenarios baseline,high_re,peak_load,congestion \
  --eval-episodes 3 --eval-seed 2026 --combined-only --consumer-metrics \
  --output results/matd3_cc_rot_seed42_three_layer_eval.csv

# 测试
python -m pytest tests/test_surplus_metrics.py -v
```

## 7. 产物与下一步

- 结果：`results/matd3_cc_rot_seed{42,7,123}_three_layer_eval.csv`、`results/three_layer_rot_seed_means.csv`
- 代码：`src/surplus_metrics.py`（新）、`src/eval_agents.py`（capture + `--consumer-metrics`）、`src/rl_env.py`（`_last_wholesale`）、`tests/test_surplus_metrics.py`（新）—— 未提交
- 下一步（里程碑，按 `docs/2026-09-08-mad3pg-vs-matd3-matrix.md` 暂停项 + 升级路线）：Quantile distributional critic（`train_rl --algo` 注入 trainer 模式）；物理引导交互（电气距离，注意 `get_scenario` 对四场景重钉 capacity 1.5）；capacity 消融（E8）；reward 消融（E9）。三层指标 + 本表作为各实验统一口径。
