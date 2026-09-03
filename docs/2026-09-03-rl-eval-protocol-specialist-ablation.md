# RL 评估协议修正与 specialist 消融设计（2026-09-03）

> 状态：**Ⅰ 协议重评已完成（2026-09-03）**；**Ⅱ specialist 消融暂停待启动**。A 主表 checkpoint 口径决策见 §7。

## 1. 决策背景

论文比较集 `baseline / high_re / peak_load / congestion` 已统一物理设施（同网同储能，仅负荷/RE 波形不同，见 memory `unified-facility-scenarios`）。
RL 上报方式经讨论锁定为**主张 A：generalist（单策略通吃）进正文主表**——即现有 `unified` 系列（三种子 42/7/123，每集轮换四场景训练 200 集）。
"四场景单独训练、单独评估"的 specialist 方案**降级为可选消融**，用于量化"通吃 vs 专精"的泛化代价。

两个前置缺陷必须先修，否则 A 的数字不可直接进论文：

1. **评估噪声与配对缺失**：见 §3。
2. **checkpoint 选择不对等**：unified 训练时未传 `--eval-scenarios`，`PolicyTracker` 的 in-training eval 全部落在 **baseline** 上，best 按 baseline 表现选出（三 seed 的 best_episode 为 100/150/175）。
   ⇒ 跨方案比较统一以 **best（in-training 选型 checkpoint）** 为准——§7 实证 final 可能是 overtraining 退化点；final 仅作训练稳定性证据。

## 2. 范围

**Ⅰ（本次执行）**
- 给 `eval_agents.py` 加最小改动：`--eval-episodes N`（默认 1，保持向后兼容）+ 每 episode 重播种、base/comb 同种配对的聚合逻辑。
- 用新协议重评 `policies/multi_matd3_unified_seed{42,7,123}` 的 **final(ep200)** 策略 × 四场景，`--combined-only`，输出聚合 CSV。
- 结果写入本文档与交接文档。

**Ⅱ（不做，等 Ⅰ）**
- 暂不新训练任何 specialist；是否补跑 peak_load-specialist 探针，待 Ⅰ 噪声量化后再定。
- 若做：4×3 种子全矩阵另立 spec。

**明确不做**：不改训练逻辑、不动场景定义、不新增奖励项。

## 3. 噪声机制（查证结论）

- 场景物理曲线（负荷/PV/风电 real+forecast）逐总线确定性（`grid.py` 固定 `RandomState`）。
- `BiddingEnv.step` 每块现画批发价（`rl_env.py:552`，24 块/日，`roll_horizon=16` 滑窗），用**全局 `np.random`**。
- `eval_agents.py` 的 base 与 comb 是**两个独立 env**，各画各的价 → `profit_delta = profit(comb 日) − profit(base 日)`，双份独立日噪声。
- 现各结果是单 episode（K=1）、价格未种子化 ⇒ ±1% 波动。

**修正方法**：每 episode k，先 `np.random.seed(S_k)` 再跑 base，再 `np.random.seed(S_k)` 再跑 comb。
base/comb 的 RNG 调用序列同构（均 24 块 × 每块一次 `day_ahead_price_china` + reset 一次全日曲线；OPF/Gurobi 与代理计算不消费 `np.random`），
故两者逐窗口拿到**同一批发价日** → 差分配对。跨 episode 递增 seed → 对批发价日做 Monte-Carlo 平均。

## 4. 验收标准

- 重播种可复现：同一 (policy, scenario, episode seed) 两次运行数值一致。
- 保留默认单 episode 行为（不传新参数结果不变/等价）。
- 输出聚合：`profit_delta`、`welfare_delta`、`valuation_artifact`、`genuine_welfare_delta` 为 **逐 episode 配对后求均值**（artifact 与 delta 分开平均，`genuine_mean = welfare_delta_mean − artifact_mean`）。
- 置信：K≥3 时列出各 episode 的 min–max 或 std 作误差带。
- 新值相对历史单次值在预期噪声内（genuine_welfare 几 k~几十 k 量级），方向与量级不大改。

## 5. 风险与注意

- **配对前提是 base/comb 的 RNG 消耗序列完全一致**。若某路径多/少消费一次随机数（如 price 尖峰分支、OPF 内部随机），配对会失准但仍在"各自独立但同种子起点"下成立，仅削弱降噪能力——不破坏正确性。实现后用小 episode 数冒烟核对 base/comb 各窗口价格一致再全量跑。
- **Gurobi 学术许可**：训练/评估均串行；四种子、多策略的并行需确认许可并发数，本阶段不并发。
- checkpoint：主表用 best；`final` 用于对照退化风险；`--checkpoint` 语义不动。
- 产物目录 gitignored，交接靠本文档 + 生成 CSV 路径。

## 6. 执行步骤（Ⅰ）

1. 改 `eval_agents.py`：新增 `--eval-episodes`、`--eval-seed`；把 main() 的场景循环包一层 episode 循环，aggregation 按 §4。
2. 冒烟：seed7 × baseline × K=2，打印每块 base/comb wholesale 是否逐窗口一致。
3. 全量重评三种子 × 四场景：`eval_agents.py --policies policies/multi_matd3_unified_seed{42,7,123} --scenarios baseline,high_re,peak_load,congestion --combined-only --eval-episodes 3 --output results/..._pN_eval.csv`。
4. 结果写回本文档 §7 与 `docs/2026-08-27-rl-handoff.md`，标注新协议与旧值差异。

## 7. Ⅰ 执行结果（2026-09-03）

**代码改动**：`eval_agents.py` 新增 `--eval-episodes N`（默认 1，向后兼容）与 `--eval-seed`；每 episode 在 base 与 comb 前重播种同一 seed → 差分配对，跨 episode 递增 seed 平均。
配对前提已实证：同 seed 下两次 base 全日 run 的 wholesale 与 welfare 逐位一致（welfare 差异 <1e-6）。

**重评协议**：`final(ep200)` 与 `best`（in-training best，三种子 best_episode=100/150/175）各跑一遍，K=3、`--eval-seed 2026`（三种子共享同一批价格日），`--combined-only`，12 代理整队。
产物：`results/multi_matd3_unified_seed{42,7,123}_{p3,best_p3}_eval.csv`。

**关键发现：seed123 final 的 Bus31I 在 ep200 后退化**。final 表中该代理四场景一致负 delta（baseline −5.6k…），把 seed123 的 profit/genuine 拖垮；但同 seed **best(ep175)** 该代理完全正常（+1220.6），且四场景 best 表三种子高度一致。即该假象来自"选 final 记录"而非真实能力差异——见 §1 已知的"ep100 后多训练损福利不增利润"，此为极端形态。

**final(ep200) 结果（k CNY，profit_delta / genuine_welfare_delta）**

| 场景 | seed42 | seed7 | seed123 |
|---|---|---|---|
| baseline | +15.1 / +7.6 | +15.1 / +7.9 | **+8.3 / −4.6** |
| high_re | +10.3 / +29.9 | +10.3 / +30.1 | **+3.6 / +15.7** |
| peak_load | +14.8 / −15.2 | +14.6 / −15.1 | +11.6 / −14.7 |
| congestion | +17.4 / +1.9 | +17.6 / +2.1 | **+12.2 / −9.0** |

**best 结果（k CNY，同口径）**

| 场景 | seed42 | seed7 | seed123 |
|---|---|---|---|
| baseline | +15.1 / +7.6 | +15.1 / +7.9 | +15.1 / +7.9 |
| high_re | +10.3 / +29.9 | +10.3 / +30.1 | +10.3 / +30.0 |
| peak_load | +15.0 / −15.5 | +15.0 / −15.3 | +13.5 / −12.6 |
| congestion | +18.0 / +1.7 | +18.2 / +2.0 | +18.3 / +2.0 |

**解读**：best 口径下跨种子几乎重合（同价格日 + 三种子收敛到相近策略），profit 全场景为正、genuine 仅 peak_load 为负（−12.6~−15.5k）。
final 口径的跨种子离散来自单点 overtraining 退化。旧单 episode（未种子化、final）值在量级内一致（如 seed7 peak_load genuine −21.3k→新 final −15.1k、best −15.3k，旧值偏负含 ±1% 噪声）。

**决策（2026-09-03）**：A 主表采用 **best** 口径；final 表仅作 overtraining 稳定性证据。
后续 **Ⅱ specialist 消融**与结果图制作**暂停**，待另行启动。

## 8. 外部评审采纳项修正记录（2026-09-03）

对一份外部评审逐条核验后（多数主张无效/已修复/设计使然），用户确认采纳三项，均已落地：

| 项 | 改动 | 验证 |
|---|---|---|
| **I.1 critic 全量观测** | `rl_bidding.py`：`CentralizedCritic` 输入改为 `obs_dim*n_agents + act_dim*n_agents`；`_build_global_state` 返回所有代理**全量 obs**（此前仅他人 unique 3 维）。Actor/obs spec 不变。 | py_compile + 8 集 smoke 训练无维度错误 + 60 pytest 通过 |
| **I.4 Nash 混合采样** | `nash.py`：`_generate_variations` 由纯局部高斯改为 边界顶点 + local Gaussian + 全局 uniform 混合；默认变体数 150→300（run.py 20 保持可调）。 | smoke：300/代理、含边界 0.3；`test_regret` 通过 |
| **II.3 self_schedule 默认 False** | `models.py` + `defaults.yaml` 默认 `False`；按用户决策**全局采纳新默认**（非 RL 脚本同步改用市场调度语义），run.py 帮助文案同步。RL 路径原强制 False 保留。 | 60 pytest 通过 |

**注意**：I.1 改变 critic 架构 ⇒ 需重训才能生效。**full-obs 重训已由用户中止（2026-09-03，Ep1 后停止，无残留产物）**——I.1 代码改动保留但**未经重训验证**；现行 A 主表仍为旧 critic 的 best-K3 数字。若需让 I.1 生效须重启 3 seed × 200 集重训。

**阻塞**：run.py 冒烟因 energy_env 内 matplotlib 损坏（命名空间包，`__file__=None`、无 pyplot）无法启动——预存环境问题，非本次改动；RL 训练/评估路径不依赖 matplotlib，不受影响。

