# MAD3PG vs full-obs MATD3 全矩阵训练计划（2026-09-08）

> 状态：pilot 已完成并健康（见下）；10-run 全矩阵已获批，**1 seed × 200 集，串行执行中**。

## 背景与动机

`rl_diffusion.py`（diffusion value-distribution critic, MAD3PG）已提交、单测通过。30-ep baseline pilot 验证：
- 墙钟：学习稳态 ~52 s/ep（对照旧 MATD3 ~12 s/ep）；200 集/条约 2.6 h。
- critic loss（标准化回报上的 diffusion MSE）ep21→30 从 0.85 单调降到 0.18，无 Q 膨胀发散；reward 由随机 −1.6k 升至 +6.2k~6.7k，学习速度 ≈ MATD3。
- 产物 normal（best/final 12 agent .pt，artifacts 正常）。

**混杂项（本矩阵的对照组意义）**：现 A 主表 `multi_matd3_unified_seed{42,7,123}` 用**旧 critic**（full-obs 集中 critic 改动已提交但重训被中止）。因此"diffusion 是否优于 MATD3"必须在**同一当前代码**下重训 MATD3 对照组，否则比较混入 full-obs 改动。本矩阵同时补齐两者。

## 矩阵（10 runs，1 seed=42，各 200 集）

四条比较场景沿用论文统一设施集合：`baseline / high_re / peak_load / congestion`。

| # | 算法 | 模式 | scenarios 参数 | save-dir | 预计 |
|---|---|---|---|---|---|
| 1 | matd3 | rotation | baseline,high_re,peak_load,congestion | `matd3_cc_rot_seed42` | ~0.8 h |
| 2 | mad3pg | rotation | 同上 | `mad3pg_rot_seed42` | ~2.6 h |
| 3 | matd3 | spec | baseline | `matd3_cc_spec_baseline_seed42` | ~0.8 h |
| 4 | matd3 | spec | high_re | `matd3_cc_spec_high_re_seed42` | ~0.8 h |
| 5 | matd3 | spec | peak_load | `matd3_cc_spec_peak_load_seed42` | ~0.8 h |
| 6 | matd3 | spec | congestion | `matd3_cc_spec_congestion_seed42` | ~0.8 h |
| 7 | mad3pg | spec | baseline | `mad3pg_spec_baseline_seed42` | ~2.6 h |
| 8 | mad3pg | spec | high_re | `mad3pg_spec_high_re_seed42` | ~2.6 h |
| 9 | mad3pg | spec | peak_load | `mad3pg_spec_peak_load_seed42` | ~2.6 h |
| 10 | mad3pg | spec | congestion | `mad3pg_spec_congestion_seed42` | ~2.6 h |

串行合计约 17 h（Gurobi 学术许可单并发，不并行）。

## 每 run CLI

```bash
PYTHONPATH=src C:/Users/26036/.conda/envs/energy_env/python.exe src/train_rl.py \
  --algo <matd3|mad3pg> --episodes 200 --seed 42 \
  --scenarios <LIST_OR_SINGLE> --eval-scenarios <SAME_OR_baseline> \
  --save-dir policies/<dir>
```

- specialist：`--scenarios <sc> --eval-scenarios <sc>`（in-training best 按其自身场景选）。
- rotation：`--scenarios baseline,high_re,peak_load,congestion --eval-scenarios baseline`（best 选型口径沿用 A 主表：baseline in-training eval）。
- mad3pg 额外用默认 `--diff-steps 50 --diff-k 4 --batch-size 128`；matd3 忽略这些参数。LR/noise/penalty 均默认。
- stdout → `logs/<save-dir>.log`；artifacts → `outputs/rl/<run_id>/`。

## 训练后评估协议（本轮目标产物）

对 10 个 save-dir 的 **best** checkpoint × 四场景跑配对 K=3 seeded eval（`--eval-episodes 3 --eval-seed 2026 --combined-only`），与 `multi_matd3_unified_seed42_best_p3_eval.csv` 同口径。产出对比表：

1. **generalist head-to-head（主问题）**：`mad3pg_rot` vs `matd3_cc_rot`（同代码，只换 critic），每场景 profit/genuine delta。
2. **specialist-vs-generalist 泛化代价**：对两算法各自比 rotation best 与 4 个 spec best 在其场景上（本矩阵把被暂停的 Ⅱ specialist 消融一并补上，当前代码口径）。
3. **是否优于旧 A 主表**：`matd3_cc_rot` best vs `multi_matd3_unified_seed42` best-K3（量化 full-obs 改动效应）。

## 验收

- run 1（full-obs MATD3 200 ep）成功训练且 critic loss 正常 → 证明 I.1 改动端到端可用，再放行后 9 条。
- 每条 run exit 0，artifacts KPI/逐集 metrics 存在，`best/` 与 `final/` 12 agent .pt 齐全。
- critic loss（mad3pg ~O(1) 下降；matd3 归一化后 ~O(1)）无发散。
- 评估按配对 K=3，结论落回本文档与 `docs/2026-08-27-rl-handoff.md`。

## 风险

- **full-obs MATD3 未大规模验证**（此前仅 8 集 smoke）→ 先跑 run 1 验证，失败即停。
- 长跑崩溃：串行链某 run 非零退出即停并上报，先修再续。
- 会话持久性：后台任务随本会话存活；会话关闭可能中断，需重续。
- 价格日配对评估与旧 CSV 同口径，保证可比。
