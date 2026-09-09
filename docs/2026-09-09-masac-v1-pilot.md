# MASAC V1 gated pilot（2026-09-09）

> 状态：代码 + 单测 + 4-run gated pilot（MASAC vs MATD3 × {baseline, peak_load}，seed 42，100 集）+ K3 三层 eval 完成。**Gate 判定：MASAC 训练稳定、成本≈MATD3，但等集数下收敛明显更慢、评估增益小而混杂 ⇒ 暂不作为主线；MASAC 先不进入 V2/V3。** 见 §4。

## 1. 实现（M1/M2/S1 代码，全部 fast 全量 93 passed）

- **M1 多场景 best**（`src/rl_training.py`/`train_rl.py`）：`PolicyTracker` 取 `eval_scenarios` 列表，`evaluate` 跨场景平均 + per-scenario；in-training best 默认按**训练场景集**选（rotation→四场景）。影响 matd3/mad3pg/qmatd3/td3 全部。
- **M2 物理引导 critic**（`src/rl_physics.py`）：无向树路径电阻 + exp 行归一 W + `PhysicsCentralizedCritic`（twin/多分位头）；`MATD3`/`QuantileMATD3` `critic_factory` 注入；`--physics`（train-only，saved actors 不变）。短冒烟通过。
- **S1 MASAC**（`src/rl_masac.py`）：`StochasticActor`（pre-tanh 高斯 + tanh + 仿射，`forward()`=均值保 eval）；`MASAC` drop-in（SAC 目标减自身熵、actor `α·logπ−min(Q)`、per-agent auto-α，reward 归一化同 MATD3）；`rl_td3` save/load 加 `actor_type` 工厂（legacy 不变）；`--algo masac` + `--alpha-init 0.1`/`--target-entropy −2`。

## 2. 训练健康（100 集，seed 42，baseline/peak_load fixed）

| run | final_r | best_ep | critic loss | actor loss | 单集 |
|---|---|---|---|---|---|
| matd3 baseline | 6401 | 75 | 1.97→0.86 | +0.48→−4.73 | ~13 s |
| masac baseline | 5147 | 100 | 1.97→1.01 | −0.36→−3.61 | ~17 s |
| matd3 peak_load | 7433 | 50 | 1.86→0.69 | +0.50→−4.57 | ~13 s |
| masac peak_load | 4957 | 100 | 1.85→0.85 | −0.34→−3.32 | ~15 s |

critic/actor 均有界、无发散；auto-α 工作（actor_loss 平滑）；成本 ≈ MATD3（+10–15%）。**注意**：masac 的 final training reward 明显低于 matd3（baseline −20%、peak_load −33%），且 best_ep=100 说明到终点仍在上升（熵探索 → 收敛更慢）。α 值未记录到 TensorBoard（缺口，见 §5）。

## 3. 评估（K3 三层，各策略在 4 场景上，k CNY）— 训练场景对

**baseline-trained（bl）在 baseline 上**

| | profit_d | genuine_d | cs_d | power |
|---|---|---|---|---|
| matd3_bl | +14.80 | +7.66 | −5.25 | 0.32 |
| masac_bl | +15.10 | **+8.08** | −5.13（更小负） | 0.33 |

**peak_load-trained（pl）在 peak_load 上**

| | profit_d | genuine_d | cs_d | power |
|---|---|---|---|---|
| matd3_pl | +13.71 | −15.99 | −10.40 | 0.92 |
| masac_pl | +13.85 | **−14.68**（+1.3k） | −11.22（更负） | 0.71 |

**泛化（非训练场景）观测**：baseline-trained 的 MASAC 在 congestion 上 cs_d 明显更差（−2.22k vs matd3_bl +0.03k），genuine 略低；peak_load-trained 的 MASAC 在其余场景 genuine 大体同或略高。

## 4. Gate 判定与建议

判据检查：
1. 训练稳定、无发散：✅（双 loss 有界，α 自动调）。
2. 单集成本 ≈ MATD3：✅（+10–15%）。
3. **至少无回退（尤其 peak_load 稳健性）：部分** —— 评估上 peak_load genuine 改善 +1.3k、baseline genuine +0.4k、profit 基本持平；但 **masac 等集数下训练收敛明显更慢、final reward 显著低于 matd3**，且消费者侧泛化混杂（congestion cs_d 更差）。

**结论**：MASAC V1 在此 MDP/口径下**不构成对 MATD3 主线更优的替代** —— 增益（genuine 小幅改善）被收敛慢 + 泛化混杂抵消。**暂不进入 V2（真 QR）/V3（physics+risk）**；MASAC 与 qmatd3 同为"已实现、未采纳"的方法学对照组。若日后要复核，先跑 200 集 head-to-head（弥补收敛差）再判。

**下一步建议（与既有评估一致）**：回归机制方向（M2 physics 已就绪、capacity 消融 E8、market-impact reward 消融 E9），它们不受 actor 范式影响且直接回答 RQ；M1 多场景 best 已自动让未来一切训练受益。

## 5. 缺口 / 记录

- α（log_alpha）未写入 TensorBoard/artifacts —— 后续算法 run 应记录，便于判"α 是否进入合理区间"。
- MASAC eval 未在 200 集口径验证（pilot 100 集）；主线若回 MASAC 需补。
- 代码/产物均未提交；结果 CSV：`results/masac_pilot_{matd3_bl,masac_bl,matd3_pl,masac_pl}_eval.csv`；策略 `policies/{matd3,masac}_pilot_{bl,pl}_seed42`。
