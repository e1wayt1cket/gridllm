# 轻度偏离惩罚防止储能出价策略饱和

## Context

RL 出价管线（`train_rl.py` + `dispatch_socp.py` + `rl_env.py`）已打通：12 个储能 agent 各自训练独立 TD3，出价（bid_mult/offer_adder）经 OPF 目标函数指挥储能调度。150 ep 训练的评估显示机制工作正常（储能响应报价、critic 学习、福利 4/5 场景上升），但存在两类问题：

1. **策略饱和**：12 个策略全部塌缩到 action 边界的常数（bid_mult 恒为 0.6 或 1.4，offer_adder 恒为 0 或 50），不随状态变化。
2. **输家自损**：Bus29I/30I/31I/21R/16I 恒报 `bid_mult=1.4`，使优化器以 1.4×bid_value 给储能充电定价 → 疯狂充电 → 高价买电 → 单日亏损 1 万~1.7 万。赢家恒报 `0.6` → 充电价值低 → 只在极低价充电 → 盈利。

诊断（baseline，12 策略逐块输出）确认：**输家不是被赢家挤出，是自身策略饱和在高位、逼储能过度充电自损**。根因是去掉偏离惩罚后，critic 的 Q 函数在 action 方向近似平坦，TD3 的 actor 被驱动到边界。

## 目标

防止策略饱和到自损的边界，让输家（29/30/31 等）不再过度充电亏损，同时保留赢家（23I/25I）的真实套利收益。

## 方案：轻度偏离惩罚

在训练 reward 中引入**小量级**偏离惩罚，给 actor 一个精确的、远离边界的梯度（不依赖 critic 学对）：

- `bid_dev_penalty = 5`（每时段每单位 |bid_mult−1|）
- `offer_dev_penalty = 0.5`（每时段每单位 offer_adder）

**校准逻辑**：惩罚在 bid=0.6/1.4 处为 `5×0.4×4 = 8/块`。赢家 Bus23I 在 0.6 的边际收益 ~270/块 ≫ 8 → 0.6 策略保留；输家在 1.4 的边际收益为负，惩罚把 Q 在 1.4 处压低 → 被拉回 1.0 附近。惩罚只拦"边际收益 < 8/块"的偏离。

## 改动范围（初版）

- `train_rl.py`：`--bid-dev-penalty` 默认 `0 → 5`，`--offer-dev-penalty` 默认 `0 → 0.5`。
- `eval_agents.py`：不加惩罚。评估要真实市场利润；baseline（真实报价）也无惩罚，对比公平。训练带整形项、评估不带是标准做法。

## 实现演进（现状，已偏离初版设计）

- **惩罚载体三次演进**：reward 级 → actor loss 上对 squash 后 action 惩罚 → **actor loss 上对 pre-tanh logit 做 L2**（`rl_td3.py`、`rl_bidding.py` 的 actor 更新）。改到 logit 的原因：饱和处 d(tanh)/dx→0，对 squash 后 action 的惩罚梯度恰好消失；logit 的 L2 梯度 `2·coef·logit` 在饱和时仍存活。
- **env reward 级惩罚已删除**：`rl_env.py` 不再含 `bid_dev_penalty`/`offer_dev_penalty` 参数与扣除逻辑，避免与 actor loss 惩罚重复。
- **差分奖励**：`rl_env.py` 新增 `use_differential_reward`，每块 reward = 实际利润 − 同一窗口下"所有 RL 代理真实报价（bid=1.0/offer=0.0）"的基准利润（复用 `_make_window_agents` 二次 `clear_market`，不推进 SOC/forecaster）。把真实报价锚定为 0 优势，给平坦的 critic 一个可学梯度。
- **观测降维 V1**：观测由 103 维（4 组 24 期序列）降为 9 维；`unique_obs_dim = 3`（load[0]/re_gen[0]/soc[0]，置于观测最前，供 MATD3 集中 critic 切分）。代价：丢失 24h 电价曲线形状与阻塞指数。
- **CTDE（MATD3）路径**：`train_rl.py --algo matd3` 使用 `rl_bidding.py` 的 MATD3（集中 critic 看所有代理 unique 观测 + 所有动作），共享回放缓冲；`--noise-anneal-steps` 探索噪声退火；critic 扩为 [256,256,128]+Dropout(0.1)（target critic `.eval()`）。直接针对本 spec 诊断的"独立学习者 critic Q 平坦"根因。
- **动作边界单一来源**：`train_rl.py --bid-mult-low/high` 默认 `None`，从 `config.market_design.bid_mult_range`（YAML `[0.3, 1.8]`）读取。

## 验证

1. 150 ep 重训（带惩罚），确认策略**不再饱和**（bid_mult 逐块有变化，不钉在边界）。
2. 全量评估（`eval_agents.py --combined-only`，5 场景）对比：
   - 输家（29/30/31/21R）亏损收敛、尽量接近盈亏平衡
   - 赢家（23I/25I）收益保留
   - 总利润不再大幅为负
   - RE 消纳率变化（记录，不设硬指标）
3. 若惩罚量级不当（赢家受损或输家仍亏），迭代调参（bid 2~10、offer 0.2~1）。
4. CTDE 全量训练后（`--algo matd3`，200 ep）用 `eval_agents.py` 评估 5 场景，对照 `results/multi_logitreg_eval.csv`：profit_delta 显著 > 0、welfare 不降、`std_bid > 0.001`（不饱和）。

## 风险

- 惩罚可能削掉部分真实偏离（量级迭代解决）。
- RE 下降是独立问题，本方案不承诺改善。
- 独立学习者的非平稳性在独立 TD3 下仍在；CTDE（集中 critic）路径直接针对该根因，但差分奖励使训练成本约翻倍（每块二次 `clear_market`），可观测降维/`--no-diff-reward` 折衷。
