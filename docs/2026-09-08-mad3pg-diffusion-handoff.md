# MAD3PG（diffusion-critic）实验与 10-run 矩阵交接（2026-09-08）

> 本文记录 2026-09-08 会话：MAD3PG pilot 验证、10-run 全矩阵获批与部分执行、后台任务被环境强停问题、以及**暂停状态与恢复步骤**。计划与验收细节见 `docs/2026-09-08-mad3pg-vs-matd3-matrix.md`。总交接入口：`docs/2026-08-27-rl-handoff.md`。

## 1. 会话起点（前置状态）

- RL 主基线 = CTDE MATD3 + 差分奖励 + V3-12 维观测 + logit 出价偏差惩罚；论文 A 主表 = `multi_matd3_unified_seed{42,7,123}` **best** checkpoint × 四场景 paired K=3（旧 critic 口径）。
- `f600b31`（09-03）已落地但**未大规模重训**：
  - **I.1 full-obs 集中 critic**（`rl_bidding.py`）——代码在，用户中止过重训，旧主表数字未受该改动影响。
  - **MAD3PG**（`rl_diffusion.py` / `train_rl.py --algo mad3pg`）——diffusion value-distribution critic，T=50/K=4 默认，试图根治 TD3/MATD3 的 critic Q 高估。仅单测过，未跑过训练。
  - multimodality 前提探针（`probe_return_multimodality.py`）：baseline 2/24、peak_load 5/24 块判 multimodal —— 证据偏弱。

## 2. 本会话工作

### 2.1 MAD3PG pilot（baseline fixed, seed 7, 30 集）——健康

`logs/mad3pg_pilot_seed7.log`，artifacts `outputs/rl/run-20260908-105400-9a1f29ce/`。

| 指标 | 实测 | 判定 |
|---|---|---|
| 墙钟 | 平均 22s/集；学习稳态 ~52s/集（MATD3 ~12s/集） | 可接受 |
| critic loss（标准化回报上 diffusion MSE） | ep21 起 0.85 → 0.18 单调下降 | 无 Q 膨胀发散 |
| reward | 随机 −1.6k → ep28-30 +6.2k~+6.7k | 学习 ≈ MATD3 |
| 产物 | final/best 12 代理 .pt + KPI 正常，rc=0 | OK |

### 2.2 全矩阵获批（用户决策）

范围：**双算法全矩阵 10 条** = {current-code full-obs MATD3, MAD3PG} × {4 specialist + 1 rotation}；**1 seed=42 × 200 集**；串行。见 `docs/2026-09-08-mad3pg-vs-matd3-matrix.md`（含完整 CLI、矩阵表、评估协议、验收、风险）。

对照组必须同代码重训 MATD3（旧 A 主表混入 full-obs 改动），故本矩阵顺带补上 I.1 重训与 Ⅱ specialist 消融（暂停项）。

### 2.3 Run 1 完成并验收通过

`matd3_cc_rot_seed42`（当前代码 full-obs MATD3, rotation, 200 集）。`logs/matd3_cc_rot_seed42.log`，artifacts `outputs/rl/run-20260908-114118-9a1f29ce/`。

- 200 集 3256s（16.3s/集）；critic loss ep21~200 从 ~1.96 平滑降到 ~1.13（无发散）；final_mean_reward=6856；best_episode=175。
- **验证门过**：full-obs 集中 critic 端到端可训练（此前仅 8 集 smoke），放行后 9 条的前提成立。

### 2.4 环境问题：后台长任务被强停 → detached 对策

- run 2 `mad3pg_rot_seed42` 用**工具后台任务**跑时被外部 killed 两次（第一次 ~ep60/2234s，第二次 ~ep50+；日志无报错 = 进程被停非代码崩溃）。单命令的 run 1（54 min）与 pilot 均正常完成 → 被停对象是任务管理器跟踪的长任务本身。
- 对策：剩余 9 条打包 `logs/run_matrix_9.sh`，`nohup` detached 启动（脱离任务管理器跟踪），脚本自行串行、各 run 独立日志 + `logs/matrix_chain.log` 打点；配 30-min 看门狗 cron。
- run 2 第三次 detached 启动后被用户暂停，只到 ep1（无进展损失）。**恢复时务必用 detached 方式，勿用工具后台任务跑长训练**。

### 2.5 暂停（用户指令）

整批已停：python 全清（无 `python.exe` 残留）、runner 已杀、看门狗 cron 已取消。产物/日志/脚本全保留。

## 3. 当前矩阵状态

| # | run（seed42, 200 集） | 状态 |
|---|---|---|
| 1 | `matd3_cc_rot_seed42` | ✅ 完成（2.3，验收通过） |
| 2 | `mad3pg_rot_seed42` | 未完成（partial 已清，需重跑） |
| 3-6 | `matd3_cc_spec_{baseline,high_re,peak_load,congestion}_seed42` | 未启动 |
| 7-10 | `mad3pg_spec_{baseline,high_re,peak_load,congestion}_seed42` | 未启动 |

产物均为 gitignored：`policies/<dir>`、`outputs/rl/<run_id>/`、`logs/`、`runs/`。

## 4. 恢复步骤（下次接续）

1. `docs/2026-09-08-mad3pg-vs-matd3-matrix.md` 是规范；`logs/run_matrix_9.sh` 覆盖 run 2-10（run 2 开头 `rm -rf` 保证干净）。若已有 run 部分完成，写**续跑版 runner**（只跑未完成项），勿重复已完成项。
2. **detached 启动**（勿用工具后台任务）：
   ```bash
   nohup bash logs/run_matrix_9.sh >/dev/null 2>&1 < /dev/null &
   ```
3. 重新布置 30-min 看门狗（查 `logs/matrix_chain.log` + 当前 run 日志；某 run 非零退出则诊断并续跑剩余项；见 `ALL_RUNS_DONE` 取消）。
4. 会话持久性：detached 进程属 OS 会话，关窗口即中断；长批需保持会话或另作真正脱离启动。
5. 训练后按 matrix 文档评估协议跑 paired K=3 eval（best checkpoint × 四场景），产出 generalist 头对头 / specialist-vs-generalist / vs 旧 A 主表三张对比表，结论落回本文档与 `docs/2026-08-27-rl-handoff.md`。

## 5. 运行时成本为何高（决策参考）

- 10 条串行 ~17.5h：MATD3 5×~0.9h + MAD3PG 5×~2.6h。Gurobi 学术许可单并发。
- MAD3PG 单集 ~52s vs MATD3 ~16s，瓶颈在 diffusion critic：TD 目标 = K=4 × T=50 去噪反演（每 critic 更新 200 次 target 前向），actor 每 2 步反向传播穿过 50 步链；CPU 无 GPU。
- 想缩短的杠杆（从大到小）：砍 specialist（8 条 ≈14.5h，核心问题只需 2 条 generalist 头对头 ~2.6h）；降 T/K（偏离 paper 默认）；上 GPU；减集数（~25%，偏离主表口径）。

## 6. Git 状态（2026-09-08）

- HEAD 仍为 `f600b31`（09-03）；本会话**未改任何源码、未提交**。
- 新增未提交：本文档、`docs/2026-09-08-mad3pg-vs-matd3-matrix.md`、`docs/2026-08-27-rl-handoff.md` 索引行更新；`logs/`（脚本 + 日志，未跟踪）。

## 7. 文档索引

| 文档 | 覆盖 |
|---|---|
| `docs/2026-09-08-mad3pg-vs-matd3-matrix.md` | 10-run 矩阵规范：矩阵表/CLI/评估协议/验收/风险 |
| `docs/2026-09-08-mad3pg-diffusion-handoff.md`（本文） | 会话交接：pilot 结果、矩阵执行状态、环境强停与 detached 对策、暂停与恢复 |
| `docs/2026-09-03-rl-eval-protocol-specialist-ablation.md` | 评估协议 + A 主表口径 + I.1/Nash/self_schedule 修正记录 |
| `docs/2026-08-27-rl-handoff.md` | RL 总交接入口（本文档的上级索引） |
