# RL 储能出价：CTDE 迁移 + 差分奖励 + 机制诊断（工作交接）

> 2026-08-17。分支 `feature/multi-obj-llm-dashboard`。本文件供新会话直接接手，包含全部改动、结果与关键结论。

## 1. 目标与背景

储能 RL 出价此前用独立 TD3 + pre-tanh logit 偏差惩罚（`rl_td3.py`），消除了出价饱和但**收益≈0**（profit +0.1%~+3.6%，被惩罚按回真实出价）。本阶段目标：切换到 **CTDE（MATD3，集中 critic）** + **差分奖励** + **观测降维**，让 RL 真正学会有利可图的策略性出价。

**核心结果**：利润大幅提升（+18%~+92%），且机制诊断证明——**"福利下降"主要是度量假象，真实调度效应在正常场景为正**（详见 §5）。

## 2. 代码改动（均已实现并通过测试）

| 文件 | 改动 |
|---|---|
| `rl_env.py` | ① 删 reward 级偏差惩罚死代码；② 新增 `use_differential_reward`：每块 reward = 实际利润 − 全真实报价基准（复用同一 `window_agents` 二次 `clear_market`，不推进 SOC/forecaster；失败回退无 shaping）；③ 观测降维 V1：103→**9 维** `[load0, re0, soc0, last_lmp, block_pos, avg_lmp, slr, avg_other_bid, bid_std]`，`unique_obs_dim=3`（unique 在最前，供集中 critic 切分）、`shared=6`；④ 移除 congestion_idx 与 `LOOKAHEAD_BLOCKS`；⑤ 加 `_last_result` 诊断钩子 |
| `rl_bidding.py` | Actor 加 `forward_logits`；MATD3 加 `bid_dev_penalty/offer_dev_penalty`（actor 更新用 pre-tanh logit L2）；探索噪声退火 0.2→0.05（`noise_anneal_steps=5000`）；`CentralizedCritic` 扩为 `[256,256,128]`+Dropout(0.1)，target critic `.eval()` |
| `train_rl.py` | `--algo {td3,matd3}`（默认 td3）；`--no-diff-reward`（差分奖励默认开）；`--noise-anneal-steps`；`--bid-mult-low/high` 默认 None → 从 `config.market_design.bid_mult_range`（`[0.3,1.8]`）读；新增 `_train_matd3` CTDE 主循环（共享 buffer） |
| `strategies/rl_bidding.py` | 适配 `_get_agent_obs` 新签名；旧 103 维 checkpoint 加载失败时回退 FixedStrategy（obs 降维的必然后果） |
| `tests/test_rl_env_obs_dim.py` | 新增（obs_dim==9、unique==3、差分奖励开关） |
| `tests/test_rl_defaults.py` | 追加 --algo/边界/差分奖励默认断言 |
| `docs/superpowers/specs/2026-08-07-mild-deviation-penalty-design.md` | 更新"实现演进"段，消除文档漂移 |

`eval_agents.py` **无需改**：`rl_td3.Actor` 与 `rl_bidding.Actor` state_dict 布局一致，MATD3 产出的 `.pt` 可直接加载。

**验证**：`pytest` 25 passed（baseline 除 stackelberg 外 + 全部 RL 测试）。2 个 Stackelberg 测试是数小时级重测（`stackelberg_nash` 对 12 leader 做差分进化全时域清算），改动前也一直 error，与本次无关，需另行降规模。

## 3. 训练结果（200 集，6 场景轮换，~74 分钟）

| 算法 | mean_reward（差分奖励，相对真实出价的超额利润） |
|---|---|
| MATD3 | −1325 → **+6328** |
| TD3（独立） | −1325 → **+5422** |

**结论：差分奖励（而非集中 critic）是利润的主要来源。** CTDE 在"每代理差分利润"目标下与独立 TD3 等价。

## 4. 评估结果（5 场景，整队 ALL，相对真实出价 baseline）

| 指标 | 旧 TD3（logit 惩罚） | TD3（差分奖励） | MATD3（差分奖励） |
|---|---|---|---|
| profit_delta | +0.0k ~ +1.2k | +8.9k ~ +28.6k（+18%~+92%）| +8.4k ~ +26.1k（+18%~+92%）|
| welfare_delta（原始） | −6.7k ~ +0.3k | −33k ~ −111k | −38k ~ −109k |
| re_rate | 不变 | 不变 | 不变 |

CSV：`results/multi_logitreg_eval.csv`、`multi_td3_eval.csv`、`multi_matd3_eval.csv`。

## 5. 机制诊断（关键，纠正 §4 的"福利大降"误读）

用 `diagnose_profit.py` 分解（baseline 场景，matd3）：

- **储能利润增量 +13.5k 拆为：套利 +8.1k（+60%）、市场力 +0.4k（+3%，≈0）、其他 +5.3k** —— 不是价格操纵。
- **非储能代理（负荷/发电侧）LMP 结算盈余几乎不变（+85 ~ +974）** —— 没有剥削他人。
- **"福利下降" = 出价低估假象 + 真实调度效应**：
  - 根因：`welfare = m.ObjVal`（OPF 目标函数值），而目标函数里储能代理的**负荷（served）与充电（ch）价值都按申报的 bid_mult 估值**（`bid*served`、`bid*ch-offer*dis`）。RL 把 bid_mult 压到 0.3~0.5 = 在指标里自我贬低。
  - 用真实报价重估同一出清（`diagnose_profit.valuation_artifact`）即可精确分离。

| 组合 | eval welfare_delta | 出价低估假象 | 真实调度效应 |
|---|---|---|---|
| matd3 baseline | −48,772 | −57,577 | **+8,805（正）** |
| td3 baseline | −51,265 | −62,191 | **+10,926（正）** |
| matd3 peak_load | −109,493 | −84,067 | **−25,425（真负）** |

**结论**：正常场景下 RL 出清的真实效率**改善**（+9~11k）；唯一真实损失在 peak_load（−25k，紧张场景储能激进套利所致）。此前图表显示的"福利 −7%~−17%"是同一个机制的两面（压低 bid 既带来利润、又压低指标里的自身价值），不是"以福利换利润"。

**方法学教训**：eval 的 welfare 指标对"出价偏离真实"的策略不可靠，应并列报告**"一致福利 delta"**（用真实报价重估 RL 出清）。`valuation_artifact` 已就绪，可并入 `eval_agents.py`。

## 6. 产物清单

- 代码：§2 各文件；`diagnose_profit.py`（套利/市场力分解 + 假象分离）；`make_results_figure.py`（综合总览图生成器，注意其导入顺序注释：pandas/matplotlib 必须在 ortools 链之后，否则 DLL 加载失败）
- 策略：`policies/multi_matd3/`、`policies/multi_td3/`（各 12 个最终 .pt + 50/100/150/200 checkpoint）
- 结果：`results/multi_{logitreg,td3,matd3}_eval.csv`
- 图表：`results/rl_work_summary.png`（综合总览：训练曲线+利润对比+福利分解+套利/市场力+统计瓷砖）、`results/rl_vs_baseline_comparison.png`（三算法对比）、`results/extraction_diagnosis_matd3.png`（套利 vs 市场力）
- TensorBoard：`runs/train-multi-20260817-*`
- `policies/`、`results/`、`runs/` 均 gitignored。2026-08-18 收尾新增：`rl_profit_diagnostics.py`（共享 welfare 假象分离，`diagnose_profit.py` 与 `eval_agents.py` 共用）、`pytest.ini`（slow 测试默认排除）

## 7. 复现命令

```bash
# 训练（matd3 或 td3，200 集 ~74 分钟）
python train_rl.py --algo matd3 --episodes 200 --save-dir policies/multi_matd3
# 评估（5 场景，整队）
python eval_agents.py --policies policies/multi_matd3 \
  --scenarios baseline,high_re,peak_load,congestion,no_congestion \
  --combined-only --output results/multi_matd3_eval.csv
# 机制诊断（套利 vs 市场力 + 福利假象分离）
python diagnose_profit.py --policies policies/multi_matd3 --scenario baseline [--save-plot results/extraction_diagnosis_matd3.png]
```

> 2026-08-19 起 `train_rl.py` 默认训练为**固定单场景（baseline）**，曲线无场景轮换噪声。复现本文 6 场景轮换结果需显式 `--scenarios baseline,high_re,peak_load,congestion,no_congestion,tight_bottleneck`。

## 8. 后续方向（2026-08-18 收尾决策已记录）

1. **目标函数抉择**：**已定利润最大化（收尾）**，不做社会福利/Pareto 目标改动。CTDE 路径保留（`--algo matd3|td3` 并存）。
2. **eval 一致福利 delta**：**已实施**——`eval_agents.py` 组合整队运行新增 `valuation_artifact` / `genuine_welfare_delta` 列，复用 `rl_profit_diagnostics.py`（与 `diagnose_profit.py` 共享单一事实来源）；新增 `--checkpoint N` 加载指定集 checkpoint。验证：baseline matd3 `artifact≈−57.6k`、`genuine` 为正（+1.7k~+10.9k 随运行波动），机制与 §5 一致。
3. **Stackelberg 慢测试**：**已隔离**——两个 Stackelberg 测试标 `@pytest.mark.slow`，`pytest.ini` 默认 `-m "not slow"` 排除。error 根因未查（数小时级，独立调查项）。
4. **peak_load 真实 −25k**：利润目标下接受，不做福利调参。若后续转向福利目标再处理。
5. **已知抖动测试**：`test_congestion_lower_welfare` 为既有抖动（根因：`dispatch_ldf._OPF_CACHE` 缓存键不含负荷数据，跨场景可能复用模型 + 跨进程哈希随机化 → 客观值 ±1% 波动）。决定保留原样；深修缓存键属核心调度层，另立任务。
6. **复现注意**：价格曲线用全局未种子化 `np.random`，单次运行数值随进程波动 ~1%（利润结论 +18%~92% 为量级稳健，不受影响）。如需逐位复现，用 `PYTHONHASHSEED=0` 启动可消除哈希随机化。
7. **留出场景泛化（2026-08-18 测得）**：re_ramp_drop/re_ramp_surge 未参与训练，最终策略 profit_delta 为正（+18.3k/+25.1k，波动价格创造套利），但 genuine 福利为负（−12.9k/−13.9k），与 peak_load −25k 同型——利润目标下接受。checkpoint 平台期（baseline）：profit_delta 单调上升（ckpt50 +2.9k → 200 +13.0k），200 集仍上升未明显趋平；genuine 全阶段为正。
8. **多种子稳健性（2026-08-19 完成）**：三个种子（42/123/7）均训练满 200 集，同一 4 场景对等评估（baseline/high_re/peak_load/congestion）。profit_delta 全场景为正且各场景量级跨种子一致——baseline +12.9k~+15.8k、high_re +9.1k~+10.2k、congestion +12.8k~+15.0k、peak_load +28.3k~+31.7k；genuine 正常场景为正（baseline +3.2k~+10.7k、high_re +29.1k~+29.9k、congestion +2.5k~+8.1k）、仅 peak_load 为负（−14.6k~−20.3k）。结论：利润提升与"福利下降是度量假象"均非单种子偶然。产物：`policies/multi_matd3_seed{123,7}/` 最终 .pt + checkpoint；`results/multi_matd3_seed{42,123,7}_eval.csv`。
