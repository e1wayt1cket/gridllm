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
- `policies/`、`results/`、`runs/` 均 gitignored；`diagnose_profit.py`、`tests/test_rl_env_obs_dim.py` 未提交

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

## 8. 未决事项 / 后续方向

1. **把"一致福利 delta"并入 `eval_agents.py`**（低风险，直接复用 `valuation_artifact`）。
2. **peak_load 真实 −25k**：若在意社会福利，需调查紧张场景的激进套利，或引入福利项调参。
3. **目标函数抉择**（核心问题）：(a) 利润最大化 → 现状即完成，收尾；(b) 社会福利 → 需改目标（如 `reward = 差分利润 + λ×系统福利`，CTDE 在此目标下才有意义）；(c) 权衡 → λ 调参找 Pareto 前沿。
4. **2 个 Stackelberg 测试降规模**（限 leader 数）或标 slow。
5. 是否保留 CTDE 路径：当前目标下与 TD3 等价；若目标改协作式则 CTDE 值得保留。
