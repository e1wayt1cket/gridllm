# qmatd3（Quantile Distributional Critic）vs full-obs MATD3（2026-09-08）

> 状态：实现 + 单测 + 3-seed rotation 训练 + K3 三层 eval 完成。结论：**qmatd3 ≈ MATD3（等价，无实质差异）** —— 见 §4 解读。

## 1. 方法

`qmatd3`（`src/rl_quantile.py`）是把单点估计集中 critic 换成 **固定网格 Quantile 分位 critic** 的 drop-in CTDE trainer（镜像 MATD3 接口，`_train_matd3` 注入）。Actor/ReplayBuffer/obs/action/save/eval 全复用。

- 每代理 twin（q1/q2）critic，各自输出 **N=32 分位头**（固定对称网格 τ_i=(i+0.5)/N），拓扑镜像 `CentralizedCritic`。
- **TD 目标 = 期望-min 标量近似**（owner 锁定）：`y = r_norm + γ(1−done)·min(mean Z1_t, mean Z2_t)`；在线 q1、q2 的 N 个头都用**分位 Huber** 回归该标量。
- Reward 尺度复用 MATD3 动态 `buffer.reward_std()`（actor logit 惩罚同除以 scale）。
- actor：`value=min(mean(Z1), mean(Z2))`，loss = −value + 惩罚。
- **近似口径（如实报告）**：对标量目标，分位头学习到的"分布"退化到标量附近 → 本方法的机制价值 = 鲁棒/非对称回归，**非校准分位数**。

CLI：`--algo qmatd3`、`--n-quantiles 32`。测试 `tests/test_rl_quantile.py`（6）+ `test_rl_defaults`（2）；fast 全量 79 passed；3-ep 端到端冒烟通过。

## 2. 训练健康（rotation 200 集，四场景轮换，~60 min/seed）

| seed | final_mean_reward | best_ep | critic_loss（quantile-Huber） |
|---|---|---|---|
| 42 | 6877 | 150 | 0.264 → 0.169 |
| 7 | 7117 | 100 | 0.262 → 0.163 |
| 123 | 7490 | 175 | 0.261 → 0.182 |

critic loss O(0.2) 单调有界，无发散；收敛跨 seed 一致。策略目录 `policies/qmatd3_rot_seed{42,7,123}`。

## 3. 三层指标结果（K=3 paired eval，`--consumer-metrics`，seed 均值；RL − truthful，k CNY；markup 无量纲）

**qmatd3**

| 场景 | profit_d | genuine_d | cs_d | cp_d | markup base→rl | arb | power |
|---|---|---|---|---|---|---|---|
| baseline | +15.09 | +7.73 | −5.15 | +5.15 | 0.0478→0.0448 | +9.33 | +0.33 |
| high_re | +10.31 | +29.98 | −0.48 | +0.48 | 0.0455→0.0428 | +7.57 | +0.26 |
| peak_load | +14.87 | −15.82 | −8.32 | +8.12 | 0.2270→0.2141 | +8.69 | +1.37 |
| congestion | +18.27 | +1.78 | +0.98 | −1.01 | 0.3617→0.3118 | +8.83 | +4.02 |

**qmatd3 − matd3（vs `matd3_cc_rot` 同口径，seed 均值；差异单位 k CNY，markup 差异无量纲）**

| 场景 | Δprofit | Δgenuine | Δcs_d | Δcp_d | Δarb | Δpower |
|---|---|---|---|---|---|---|
| baseline | +0.002 | +0.085 | +0.031 | −0.031 | +0.003 | −0.000 |
| high_re | −0.001 | −0.061 | −0.006 | +0.006 | −0.001 | −0.000 |
| peak_load | +0.22 | +0.26 | +0.34 | −0.33 | +0.08 | +0.10 |
| congestion | +0.24 | +0.10 | **+0.81** | −0.78 | −0.05 | +0.27 |

（Δ 均为 k CNY 量级；全部差异 ≤ ~0.8k，对照基准量级 10–18k。）

## 4. 解读

1. **qmatd3 ≈ MATD3（等价）**：三层指标上逐场景差异 ≤ ~0.8k CNY（多数 <0.3k），跨 3 seed 一致 —— 标量期望-min 近似下，分位 critic 收敛到的策略/行为与单点估计 MATD3 **无实质差异**。这不是失败而是**预期内的等价结论**：N 个头回归同一标量，均值即点估计，故行为不变。
2. 微弱且**方向一致的位移**在紧张场景：peak_load/congestion 的 profit_d、genuine_d 与 congestion 的 cs_d 略优（+0.2~+0.8k，congestion CS 由 +0.18k→+0.98k），power 亦略升 —— 但都在噪声/差异界内，不足以支撑"分位 critic 更稳健"的强主张。
3. **机制含义**：若想让分布 critic 带来真收益，需**真分布目标**（随机 τ 的 QR 式目标 / IQN），而非本标量近似；或本 MDP 下分布信息本就无增量价值。是否投入真 QR 目标属研究决策（见 §6）。
4. 健康度：qmatd3 与 matd3 的 critic loss 都有界（quantile-Huber O(0.2) vs MSE O(1)），均无 Q 膨胀；未观察到分位损失带来的稳健性差异（按本评估口径）。

## 5. 产物与复现

- 结果：`results/qmatd3_rot_seed{42,7,123}_three_layer_eval.csv`、`results/qmatd3_vs_matd3_delta.csv`（qmatd3−matd3 逐场景差异）、基线 `results/matd3_cc_rot_seed*_three_layer_eval.csv`。
- 训练：`PYTHONPATH=src python src/train_rl.py --algo qmatd3 --seed S --episodes 200 --scenarios baseline,high_re,peak_load,congestion --eval-scenarios baseline --save-dir policies/qmatd3_rot_seedS`。
- eval：`eval_agents.py --policies policies/qmatd3_rot_seedS/best --scenarios baseline,high_re,peak_load,congestion --eval-episodes 3 --eval-seed 2026 --combined-only --consumer-metrics --output results/qmatd3_rot_seedS_three_layer_eval.csv`。
- 代码：`src/rl_quantile.py`（新）、`src/train_rl.py`（`qmatd3`/`--n-quantiles`）、`tests/test_rl_quantile.py`（新）、`tests/test_rl_defaults.py` —— 未提交。

## 6. 下一步（待定）

- 接受等价结论：主表保留 matd3（点估计更简）；qmatd3 作为方法学对照组。→ 进入 **物理引导交互 / capacity 消融 / reward 消融** 里程碑。
- 或验证"分布是否本无增量"：给 qmatd3 上**真 QR 目标**（随机 τ 采样 + 双网 min-over-mean 标量）再看 —— 但预计收益低、成本高。
- CVaR 仅在真分布目标下才有意义；当前标量近似下 CVaR 无内容。
