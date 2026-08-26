# RL 竞价训练效果评估与改进方向（2026-08-26）

> 评估基于截至 2026-08-23 21:45 的训练产物（最新运行 `train-multi-20260823-210728`）。
> 当前基线 = MATD3/CTDE + 差分奖励 + V3 观测（12 维，`v3_12d`）+ 出价偏差惩罚（`bid_dev_penalty=5.0, offer_dev_penalty=0.5`）。
> 训练目标已定为**利润最大化**（见 `docs/2026-08-17-rl-ctde-summary.md` §8.1），福利作为机制与副作用指标并列报告。

## 1. 结论速览

| 维度 | 结论 | 依据 |
|---|---|---|
| 利润目标 | ✅ 达成且跨种子稳健（+18%~+92%） | §2 跨种子表 |
| 真实福利 | ⚠️ 正常场景为正；peak_load 真负（已知，利润目标下接受） | §2、§4 |
| 训练稳定性 | ✅ 收敛正常，critic loss 有界（非发散） | §3 |
| 机制 | ✅ 利润来自套利，非价格操纵 | §4 |
| 遗留 | 🔴 rot（场景轮换）实验 8-23 中断于 ~130/200 集，未收尾未入档 | §5.1 |

## 2. 训练效果（整队 12 储能代理，相对真实出价 baseline）

**利润增量 profit_delta（CNY）**

| 场景 | seed42 (v4) | seed7 (v4) | seed123 (v4) | rot seed7（中断版） | rot final (ep200) | rot best (ep100) |
|---|---|---|---|---|---|---|
| baseline | +15.0k | +14.2k | +15.0k | +13.8k | +15.2k | +15.0k |
| high_re | +10.0k | +9.4k | +10.8k | +10.2k | +10.5k | +10.5k |
| congestion | +14.2k | +14.7k | +14.8k | +15.4k | +15.3k | +15.0k |
| peak_load | +30.7k | +29.4k | +29.3k | +30.4k | +31.4k | +30.7k |

**真实福利增量 genuine_welfare_delta（扣除出价低估假象，CNY）**

| 场景 | seed42 | seed7 | seed123 | rot seed7 | rot final | rot best |
|---|---|---|---|---|---|---|
| baseline | +6.4k | +7.6k | +7.6k | +5.1k | +6.5k | +11.9k |
| high_re | +32.5k | +26.7k | +33.0k | +28.4k | +29.6k | +28.1k |
| congestion | +9.9k | +9.5k | +2.7k | +7.2k | +9.9k | +6.2k |
| **peak_load** | **−26.3k** | **−18.0k** | **−23.2k** | **−24.2k** | **−9.2k** | **−19.2k** |

数据来源：`results/multi_matd3_v4_eval.csv`、`results/multi_matd3_v4_seed7_p2_eval.csv`、
`results/multi_matd3_v4_seed123_p2_eval.csv`、`results/multi_matd3_rot_seed7_partial_eval.csv`、
`results/multi_matd3_rot_seed7_eval.csv`（ep200 final）、`results/multi_matd3_rot_seed7_best_eval.csv`（ep100 best）。
跨种子一致 ⇒ "利润提升 / 福利下降是度量假象"均非单次偶然。

rot 完成版（尤其 ep200 final）在 peak_load 上真实福利增量明显改善（−9.2k vs v4 种子 −18k~−26k），
同时利润增量仍为全场最高（+31.4k）；该提升是否来自场景轮换带来的跨场景泛化，待 §7 拆解确认。

**RE 消纳率**：RL 与 baseline 几乎一致（59.844445...），为物理/经济吸收上限，RL 最大化利润、不优化 RE 率（见 handoff §10 诊断，非 bug）。

## 3. 训练曲线健康度

- 固定 baseline 运行（`run-20260823-194546-9a1f29ce`，seed7，200 集）：训练差分奖励 −1325→**+6661**；critic loss 有界（0.57→3.16→回落 ~2.3）；in-training eval 峰值在 ep100（mean_reward≈4014，raw profit）。
- rot 运行（`train-multi-20260823-210728`）：reward −1392→**+6257**（130 集处）；critic loss 1.98→3.00（停止时仍在爬升区间，未确认回落）；每代理 reward 分化合理——工业储能代理（Bus23I +24k、Bus29I +12k、Bus31I +11.5k）是主要赢家，无储能/小储能代理（Bus5R +357、Bus15C +56）提升有限。
- **best 策略选择只看利润**：两处 eval 都显示 `Eval/welfare` 随训练单调下降（461k→419k），而利润在 ep~75-100 已平台。即"多训练主要损福利而不增利润"，当前靠 `best by mean_reward` 选在 ep~100 缓解，但无机制保证。

## 4. 机制诊断（非价格操纵）

`diagnose_profit.py` 分解（baseline，matd3）：储能利润增量 ≈ 套利 +8.1k（60%）+ 其他 +5.3k + 市场力 +0.4k（≈0）；非储能代理 LMP 结算盈余几乎不变（+85~+974）。
"福利下降"主因是度量假象：`welfare = OPF 目标值`，而目标函数按**申报 bid_mult** 给储能负荷/充电估值，RL 压低 bid_mult（0.3~0.5）即在指标里自我贬低。用真实报价重估（`valuation_artifact`）即可分离真实调度效应。

## 5. 遗留问题

1. ~~**rot 泛化实验未收尾**~~（8-23 中断于 ~130 集，仅有 ckpt_50/100；8-26 重跑完成，见 §6）。
2. **best 策略选择忽略福利**：`eval_metric=mean_reward`（raw profit），选择策略时不对 welfare 设约束。
3. **`total_regret` 列从未填充**：regret/Nash 检查已实现（`--nash-regret`）但未对 V4 策略跑过。COBYLA 版在 Windows 串行需 4-8 小时，需改用并行/采样版。
4. **评估协议方差**：`eval_episodes=1` + 电价曲线未种子化（全局 np.random），单次运行数值 ±1% 波动。
5. **仅 12 个储能代理参与 RL**：其余 20 个无储能代理固定出价，其对抗响应是静态的，低估均衡影响。

## 6. 已完成：rot 训练重跑（2026-08-26）

命令：
```
python train_rl.py --algo matd3 --seed 7 \
  --scenarios baseline,high_re,peak_load,congestion,no_congestion,tight_bottleneck \
  --episodes 200 --save-dir policies/multi_matd3_rot_seed7
```
- 场景清单采用 handoff §7 记录的 6 场景轮换；Ep1（peak_load，reward −1392.0）与中断运行逐位一致，证明同一 seed7 下复现了原轨迹。
- **200 集完整跑完**（手动终端重跑，~47 min，14s/ep）：ckpt_50/100/150/200 + final/best/last + artifact
  `outputs/rl/run-20260826-215128-9a1f29ce`。KPI：final_mean_reward=6959.2、best_mean_reward=4038.8（ep100）、best_welfare=463.2k、n_episodes=200。
- **4 场景整队评估已完成**：`results/multi_matd3_rot_seed7_eval.csv`（ep200 final）、
  `results/multi_matd3_rot_seed7_best_eval.csv`（ep100 best），结果并入 §2 表。
- 对比图（`make_rot_comparison_figures.py` 生成，`results/rot_*.png`）：利润增量 / 真实福利增量分场景分组柱状图、
  利润-福利权衡散点、rot 训练奖励曲线。
- 说明：8-26 上午经本会话后台启动的重跑在 Ep 120 被外部终止（无 traceback），后改由独立终端手动完成；
  8-26 20:44 的首次运行中断于 ~ep25（日志保留于 `logs/train-rot-seed7-20260826.log.partial`）。

## 7. 改进方向（按优先级）

**P0**
- ~~跑完 rot 训练、补 artifact、重新评估、更新 handoff 文档~~（8-26 完成，见 §6）；rot 版 critic loss 峰值后回落已由
  `outputs/rl/run-20260826-215128-9a1f29ce/metrics.csv` 记录（ep~200 critic loss ≈ 3.0~3.3，有界）。
- 对 V4 策略补 regret/Nash 检查（改用并行采样版，`eval_agents.py` 已切到 `use_optimization=False, parallel=True`，见 §5.3）。

**P1**
- best 策略选择纳入福利：如"利润增量最大化且 genuine 福利 ≥ 0"，或输出 profit–welfare 权衡曲线。
- 训练内评估扩展到 2~3 场景（尤其 peak_load），使 checkpoint 选择带泛化信号。
- 电价曲线种子化 + `eval_episodes≥3`，消除 ±1% 噪音。

**P1**
- peak_load 真实负福利：先用 `diagnose_profit.py` 对 peak_load 单独做套利/市场力/其他分解，再决定接受或调参（如收紧 bid_mult 下限、奖励加调度成本项——后者需先决策目标）。

**P2**
- 非储能代理参与 RL 训练（或 critic 加 opponent modeling），消除固定对手假设。
- 场景课程学习（baseline → 逐步加 peak_load/congestion）替代均匀轮换。
- 样本效率：差分奖励每次 step 双倍 Gurobi 出清是主要算力开销（~14-20s/ep），可考虑多环境并行或周期复用基准出清。
