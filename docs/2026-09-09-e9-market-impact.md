# E9 market-impact reward 惩罚探针（2026-09-09）

> 状态：`--market-impact-penalty`（λ）实现 + λ=0.3 探针（matd3 rotation 200ep seed42）+ K3 三层 eval 完成。**结论：λ=0.3 未降低实测价格影响项（power 反略升），但 congestion/peak_load 的 CS 与 profit 略增 —— 该惩罚定义未命中"抑制市场力"通道，先不扫 λ，需重新审视定义。**

## 1. 方法

训练 reward（每块，仅训练期）：
```
r_i = 差分利润_i − λ · power_i,
power_i = Σ_t (q_rl+q_base)/2 · (lmp_rl − lmp_base)   （同窗 RL vs 真实出清的 per-agent 价格影响项）
```
复用差分奖励的第二次真实出清（base_result）即可算出，eval 不设惩罚 → 评估仍真实市场。λ=0 = 原差分奖励（对照 `matd3_cc_rot_seed42`，K3 重评在新 CS 语义 = `quickval_matd3_cc_ref`）。

## 2. 结果（K3 三层，k CNY；对照 λ=0 vs λ=0.3，均同 matd3）

| 场景 | 指标 | λ=0 | λ=0.3 | Δ(0.3−0) |
|---|---|---|---|---|
| baseline | profit_d | 15.08 | 15.08 | −0.00 |
| | genuine_d | 7.51 | 7.55 | +0.03 |
| | cs_d | 0.43 | 0.43 | +0.00 |
| | power | 0.33 | 0.33 | +0.00 |
| high_re | profit_d | 10.31 | 10.31 | +0.00 |
| | genuine_d | 29.90 | 29.96 | +0.06 |
| | power | 0.26 | 0.26 | −0.00 |
| peak_load | profit_d | 14.51 | 14.70 | +0.20 |
| | genuine_d | −16.41 | −16.44 | −0.03 |
| | cs_d | 3.64 | 3.67 | +0.03 |
| | power | 1.35 | 1.37 | +0.02 |
| congestion | profit_d | 18.01 | 18.24 | **+0.24** |
| | genuine_d | 1.55 | 1.62 | +0.07 |
| | cs_d | 7.67 | 8.05 | **+0.38** |
| | power | 3.72 | 3.94 | **+0.22** |

训练健康：final_r 6112、best_eval_r 3280（ep175），正常。

## 3. 判读

1. **惩罚未压低实测价格影响**：λ=0.3 时 congestion 的 power 反而 **+0.22k**（peak_load +0.02k），baseline/high_re 不变。即每块"RL vs 真实出清"的 power 项被罚，但净效果却是在 congestion 抬高了整日 price-impact —— 说明该 per-block power 主要是**套利重排的副作用**，优化器换一种时序仍保有价格影响，λ 惩罚没有锁住这个通道。
2. **消费者侧小幅更友好**：congestion cs_d +0.38k、profit +0.24k、genuine +0.07k；peak_load cs_d +0.03k —— 不是"更好压市场力"，更像策略微移后的略优解。
3. 结论：**这个 λ·power 的惩罚定义不成立（未达"允许赚钱但抑制价格操纵"目标），先不值得扫 λ**。若要继续 E9，应先改定义，候选：(a) 罚 `|power|` 或罚 consumer-payment 增量（直接对着新口径 CS 目标）；(b) 罚 agent 对 nodal LMP 抬升（`lmp_rl−wholesale` 水平 vs 真实）；(c) 在 actor loss 加（而非 env reward）以避免与差分基线纠缠。选 (a)/(b) 需另立小实验；若本线收益存疑，可与 TD3-vs-matd3（何时需要集中式）并行评估后决定。

## 4. 产物与下一步

- eval：`results/e9_matd3_p03_eval.csv`；策略 `policies/matd3_e9p03_rot_seed42`；代码 `--market-impact-penalty`（env reward，默认 0，向后兼容）未提交。
- 下一步建议：**不改 λ 扫**，先重定义 market-impact/consumer 惩罚（a 或 b）做一次小对照，或转 TD3-vs-matd3 @ capacity（把"何时需要集中 critic"做实）。代码/文档均未提交。
