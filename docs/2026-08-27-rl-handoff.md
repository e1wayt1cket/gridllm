# RL 储能竞价：当前状态总交接（2026-08-27）

> 本文是 RL 工作流的**总交接入口**，整合截至今日的全部逻辑、结果、产物与待办。历史演进细节见 §9 文档索引。新会话先读本文件，再按需查历史文档。
>
> 项目：GridLLM（IEEE 33 节点配电网，96 时段两结算市场，12 储能代理参与 RL 出价）。
> 当前基准：**CTDE MATD3 + 差分奖励 + V3-12 维观测（`v3_12d`）+ 出价偏差惩罚（`bid_dev_penalty=5.0, offer_dev_penalty=0.5`）**，训练目标为**利润最大化**（福利作为机制/副作用指标并列报告）。

## 1. 工作全景

| 工作流 | 状态 | 入口 |
|---|---|---|
| RL 训练（MATD3/TD3，差分奖励） | 完成，V4 为基准 | `train_rl.py` |
| 多种子稳健性（V4 seed 42/7/123） | 完成 | `policies/multi_matd3_v4*` |
| rot 场景轮换实验（seed7, 200 集） | 完成（8-26 重跑收尾） | `policies/multi_matd3_rot_seed7` |
| 整队评估（4 场景，`genuine_welfare_delta`） | 完成 | `eval_agents.py` |
| 机制诊断（套利 vs 市场力 vs 出价假象） | 完成 | `rl_profit_diagnostics.py` / `diagnose_profit.py` |
| 训练编排 + 结构化产物 | 完成 | `rl_training.py` / `run_artifacts.py` |
| 对比图表生成 | 完成（今日重设计） | `make_rot_comparison_figures.py` |
| regret/Nash 检查（V4 策略） | 未跑（见 §8） | `eval_agents.py --nash-regret` |

## 2. 核心逻辑

### 2.1 RL 竞价管线
`train_rl.py`（matd3 默认）→ `rl_env.py`（环境，11 维→12 维观测，差分奖励）→ `rl_bidding.py`（MATD3，集中 critic `[256,256,128]`+Dropout，logit 惩罚，噪声退火 0.2→0.05）→ `strategies/rl_bidding.py`（出价接入）。观测/动作空间经 `rl_spec.py` 可插拔（`ObservationSpec`/`ActionSpec` 注册表，策略 `.pt` 带 `_makerb_policy_meta`，obs-spec 不匹配拒绝加载）。

### 2.2 差分奖励（利润的主要来源）
每块 reward = 实际利润 − 全真实报价基准（复用同一 `window_agents` 二次 `clear_market`，不推进 SOC/forecaster；失败回退无 shaping）。8-23 起基准线**共享单批发价曲线**（`clear_market` 新增可选 `wholesale`），消除双清价格曲线噪声。结论（8-17）：利润提升主要来自差分奖励而非集中 critic。

### 2.3 观测演进
V1 103 维 → V2 9 维 → V3 11 维 → **V3-12 维（`v3_12d`，当前基准）**：unique 3 = `[load_feat, re_feat, soc]`（load/re 整块均值按各自日峰值归一，RE 不按负荷峰值归一——PV 峰值是负荷 1.4~5.8 倍会饱和）；shared 9 = `[last_lmp_norm, block_pos, avg_lmp_norm, slr_norm, avg_other_bid, bid_std, ema_dev, price_trend, pred_lmp_norm]`。每次观测升级使旧策略失效需重训。

### 2.4 福利假象分离（关键方法学）
`welfare = OPF 目标值`，目标函数按申报 `bid_mult` 给储能负荷/充电估值；RL 压低 bid_mult（0.3~0.5）即"在指标里自我贬低"。"福利下降"主因是**度量假象**而非真实效率损失。用真实报价重估同一出清得到 `valuation_artifact`，从而 `genuine_welfare_delta = welfare_delta − valuation_artifact`。实现集中在 `rl_profit_diagnostics.py`（`eval_agents.py` 与 `diagnose_profit.py` 共享单一事实来源）。评估 CSV 的 `genuine_welfare_delta` 列即真值。

### 2.5 best / checkpoint 选择
`PolicyTracker`（`rl_training.py`）：确定性评估、best/last 策略、早停默认关。**`eval_metric=mean_reward`（raw profit）**，best 策略只按利润选，不对 welfare 设约束。rot 运行中 in-training eval 每 25 集一次，best 落在 ep100（`mean_reward≈4039`）。

### 2.6 训练编排与结构化产物
`run_artifacts.py`：每次训练写 `outputs/rl/<run_id>/{manifest,config}.json, metrics.csv, eval.csv, kpi.json`。`eval.csv` 的 `is_best` 列按当前最优标记（注意：reward 单调上升时多个点会同时为 True，取最后一个即全局最优）。逐 agent critic/actor loss 写 TensorBoard（`Loss/critic/{agent}`）。

## 3. 当前训练结果（4 场景整队评估，相对真实出价 baseline）

### 3.1 利润增量 profit_delta（k CNY）

| 场景 | V4 seed42 | V4 seed7 | V4 seed123 | rot best(ep100) | rot final(ep200) |
|---|---|---|---|---|---|
| baseline | +15.0 | +14.2 | +15.0 | +15.0 | +15.2 |
| high_re | +10.0 | +9.4 | +10.8 | +10.5 | +10.5 |
| peak_load | +30.7 | +29.4 | +29.3 | +30.7 | +31.4 |
| congestion | +14.2 | +14.7 | +14.8 | +15.0 | +15.3 |

### 3.2 真实福利增量 genuine_welfare_delta（k CNY，扣除出价假象）

| 场景 | V4 seed42 | V4 seed7 | V4 seed123 | rot best(ep100) | rot final(ep200) |
|---|---|---|---|---|---|
| baseline | +6.4 | +7.6 | +7.6 | +11.9 | +6.5 |
| high_re | +32.5 | +26.7 | +33.0 | +28.1 | +29.6 |
| peak_load | **−26.3** | **−18.0** | **−23.2** | **−19.2** | **−9.2** |
| congestion | +9.9 | +9.5 | +2.7 | +6.2 | +9.9 |

数据来源：`results/multi_matd3_v4_eval.csv`、`multi_matd3_v4_seed7_p2_eval.csv`、`multi_matd3_v4_seed123_p2_eval.csv`、`multi_matd3_rot_seed7_best_eval.csv`、`multi_matd3_rot_seed7_eval.csv`。

### 3.3 结论
- **利润提升跨种子稳健**（+9.4k~+31.4k 全场景为正，量级一致）——非单种子偶然。
- **真实福利正常场景为正**（baseline/high_re/congestion），**仅 peak_load 为负**（利润目标下接受，紧张场景储能激进套利）。
- **rot final(ep200) 明显改善 peak_load 真实福利**（−9.2k vs V4 种子 −18k~−26k），且利润仍为全场最高（+31.4k）。此提升是否来自场景轮换的跨场景泛化，尚未拆解确认（见 §8）。

### 3.4 训练健康度
- rot 运行（`run-20260826-215128-9a1f29ce`，seed7，200 集）：`final_mean_reward=6959.2`、`best_mean_reward=4038.8`(ep100)、`best_welfare=463.2k`、`n_episodes=200`，critic loss 有界（~2.3~3.3，回落）。
- 每集轮换 6 场景（metrics.csv `scenario` 列：peak_load 43 / no_congestion 35 / tight_bottleneck 34 / baseline 31 / high_re 31 / congestion 26）——**这是训练曲线必须按场景分面画的原因**。
- RE 消纳率恒定 59.84% 是物理/经济吸收上限，非 bug（RL 最大化利润、不优化 RE 率）。

## 4. 图表生成（`make_rot_comparison_figures.py`，今日重设计）

### 4.1 重设计逻辑
原图把 **6 个异构变体**（3 个 V4 种子 + rot partial/final/best 三个 checkpoint）并列成 24 根同色柱，维度混淆、不可读；训练曲线用双轴混 6 场景，误导。今日按"**rot vs V4 方法对比**"重设计：

- **两个族**：V4 = 3 种子分布 → **均值 ± min-max 范围**（灰，上下文）；rot = **best(ep100) + final(ep200) 两个 checkpoint**（各独立颜色，主体）。
- **每项独立颜色**（经 dataviz 校验器通过，blue↔orange CVD ΔE 24.7）：V4 灰 `#8a8a8a`、rot best 蓝 `#2a78d6`、rot final 橙 `#eb6834`。
- **rot partial（8-23 中断运行）完全移除**——已被完整重跑取代。
- **训练曲线去双轴、按 4 个评估场景分面**（单轴 reward），解决"一条线混 6 场景"。

### 4.2 输出文件（`results/`）
| 文件 | 内容 |
|---|---|
| `rot_v4_profit_delta_by_scenario.png` | 利润增量：V4 均值±范围 vs rot best/final，rot 值直接标注 |
| `rot_v4_welfare_delta_by_scenario.png` | 真实福利增量，同结构，peak_load 负值醒目 |
| `rot_v4_profit_welfare_tradeoff.png` | 权衡散点：颜色=项、形状=场景，rot 标注场景名 |
| `rot_training_by_scenario.png` | 训练 reward 按 4 场景分面（2×2），单轴 |

运行：`python make_rot_comparison_figures.py [--results-dir results] [--out-dir results] [--metrics-csv outputs/rl/run-20260826-215128-9a1f29ce/metrics.csv]`

**注意**：旧的 `rot_profit_delta_by_scenario.png` / `rot_welfare_delta_by_scenario.png` / `rot_profit_welfare_tradeoff.png` / `rot_training_curve.png` 仍留在 `results/`（未删除），本会话未提交，属旧版，可清理。

## 5. 产物清单

**代码**（均已提交，除 `make_rot_comparison_figures.py` 今日未提交）
- 训练/环境：`train_rl.py`、`rl_env.py`、`rl_bidding.py`、`rl_td3.py`、`rl_spec.py`、`rl_training.py`、`run_artifacts.py`
- 评估/诊断：`eval_agents.py`、`rl_profit_diagnostics.py`、`diagnose_profit.py`、`price_forecaster.py`
- 图表：`make_rot_comparison_figures.py`（今日重设计）、`make_results_figure.py`（注意其导入顺序注释：pandas/matplotlib 必须在 ortools 链之后）

**策略**（均 gitignored）
- `policies/multi_matd3_v4`（seed42）、`policies/multi_matd3_v4_seed7`、`policies/multi_matd3_v4_seed123`
- `policies/multi_matd3_rot_seed7`（final `.pt` + `ckpt_{50,100,150,200}` + `best/`、`last/`）

**评估结果**（`results/`，gitignored）：`multi_matd3_v4_eval.csv`、`multi_matd3_v4_seed7_p2_eval.csv`、`multi_matd3_v4_seed123_p2_eval.csv`、`multi_matd3_rot_seed7_eval.csv`、`multi_matd3_rot_seed7_best_eval.csv`、`multi_matd3_rot_seed7_partial_eval.csv`（历史，可弃）+ 各类历史 eval。

**运行产物**（`outputs/rl/`，gitignored）
- `run-20260826-215128-9a1f29ce/` = **rot 完整重跑**（config/metrics/eval/kpi/manifest，git_sha `bfa185a`）
- `run-20260823-194546-9a1f29ce/` = **V4 seed7 固定 baseline 训练**
- TensorBoard：`runs/train-multi-20260826-215130`（rot 完整版）；`runs/train-multi-20260826-205018`（中断 ~ep25）、`runs/train-multi-20260826-203355`（后台重跑中断 ~ep120）、`runs/train-multi-20260823-210728`（原始 rot 中断 ~130 集）——后三者历史记录。

## 6. 复现命令

```bash
# rot 场景轮换训练（200 集，~47 min，14s/ep）
python train_rl.py --algo matd3 --seed 7 \
  --scenarios baseline,high_re,peak_load,congestion,no_congestion,tight_bottleneck \
  --episodes 200 --save-dir policies/multi_matd3_rot_seed7

# V4 固定 baseline 训练（默认即固定单场景）
python train_rl.py --algo matd3 --seed 7 --episodes 200 --save-dir policies/multi_matd3_v4_seed7

# 整队评估（4 场景；best 策略用 --checkpoint 100 或 policies/.../best）
python eval_agents.py --policies policies/multi_matd3_rot_seed7 \
  --scenarios baseline,high_re,peak_load,congestion \
  --combined-only --output results/multi_matd3_rot_seed7_eval.csv

# 对比图表（今日重设计版）
python make_rot_comparison_figures.py

# 测试（慢测试默认排除）
python -m pytest tests/ -v
```

## 7. 运行环境注意

- **venv/ 是 WSL 创建的**（`pyvenv.cfg` home=/usr/bin），原生 Windows Python 不可用。实际用系统 **Anaconda Python 3.12.7**（`/c/ProgramData/anaconda3/python`，pandas 2.3.3 / matplotlib 3.8.4）。
- **Gurobi 必需**（`_HAS_GUROBI` 守卫，无开源求解器回退）。
- 价格曲线全局未种子化 `np.random`，单次运行数值 ±1% 波动；逐位复现需 `PYTHONHASHSEED=0`。
- `policies/`、`results/`、`runs/`、`outputs/` 均 gitignored——产物不在 git 里，交接依赖本文档定位。

## 8. 未决事项与改进方向（按优先级）

**P0**
- V4 策略 regret/Nash 检查：`eval_agents --nash-regret` 已就绪（并行采样版），未对 V4 跑过（COBYLA 串行需 4-8 小时，需并行/采样版）。

**P1**
- **best 策略选择纳入福利**：如"利润增量最大且 genuine 福利 ≥ 0"，或输出 profit–welfare 权衡曲线。
- **训练内评估扩展到 2~3 场景**（尤其 peak_load），使 checkpoint 选择带泛化信号。
- **rot 提升 peak_load 福利的成因拆解**：确认是否来自场景轮换跨场景泛化（`diagnose_profit.py` 对 peak_load 单独分解）。
- 电价曲线种子化 + `eval_episodes≥3`，消除 ±1% 噪音。

**P2**
- 非储能代理参与 RL 训练（或 critic 加 opponent modeling）。
- 场景课程学习（baseline → 逐步加 peak_load/congestion）替代均匀轮换。
- 样本效率：差分奖励双 Gurobi 出清是主要算力开销，可多环境并行或周期复用基准出清。

## 9. 文档索引（历史演进）

| 文档 | 覆盖 |
|---|---|
| `docs/2026-08-17-rl-ctde-summary.md` | CTDE 迁移、差分奖励、机制诊断（§5 福利假象）、观测演进 V1→V3、critic 发散修复、多种子重训、编排移植（截至 8-23） |
| `docs/2026-08-26-rl-training-eval.md` | rot 实验中断与 8-26 完整重跑、跨种子评估表、改进方向 |
| `docs/2026-08-27-rl-handoff.md`（本文） | 当前状态总交接 + 今日图表重设计 |
| `docs/superpowers/specs/2026-08-07-mild-deviation-penalty-design.md` | 出价偏差惩罚设计（已实现演进记录） |
| `docs/superpowers/plans/2026-08-07-mild-deviation-penalty.md` | 对应实施计划 |

## 10. Git 状态（2026-08-27）

- HEAD：`610f4f9 Complete rot scenario-rotation training and evaluation`。
- 未提交：`make_rot_comparison_figures.py`（今日重设计）、`logs/`（训练日志，未跟踪）。
- 产物目录均 gitignored，不进入版本库。
