# 轻度偏离惩罚防止储能出价饱和 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 12 个储能 agent 的 TD3 出价策略不再饱和到自损边界（输家恒报 bid_mult=1.4 过度充电），通过给训练 reward 引入小量级偏离惩罚。

**Architecture:** 复用 `rl_env.py` 已有的偏离惩罚代码（`bid_dev_penalty`/`offer_dev_penalty`，由 `BiddingEnv` 构造传入）。只改 `train_rl.py` 两个 CLI 默认值：`bid_dev_penalty` 0→5、`offer_dev_penalty` 0→0.5。惩罚给 actor 一个远离边界的精确梯度，只拦"边际收益 < 8/块"的偏离，不牺牲赢家（23I/25I）的套利收益。

**Tech Stack:** Python, PyTorch (TD3), Gurobi (SOCP-OPF), TensorBoard。

## Global Constraints

- 代码只使用英文（标识符、注释、字符串）。
- `rl_env.py` 和 `eval_agents.py` 不改（惩罚代码已存在；评估保持真实利润、无惩罚）。
- 评估在 `self_schedule=False`、`use_nodal_price=False` 机制下进行（与训练一致）。
- 运行环境：`C:/Users/26036/.conda/envs/energy_env/python.exe`。
- 训练生成产物（policies/runs/logs）已被 .gitignore 覆盖，不提交。

---

### Task 1: 提取 parser 并改默认惩罚值 + 单元测试

**Files:**
- Modify: `train_rl.py`（把 `main()` 内的 `argparse.ArgumentParser(...)` 提取为 `build_parser()`，改两个默认值，`main()` 改调 `build_parser().parse_args()`）
- Test: `tests/test_rl_defaults.py`（新建）

**Interfaces:**
- Produces: `train_rl.build_parser() -> argparse.ArgumentParser`。`build_parser().parse_args([])` 返回的 namespace 中 `bid_dev_penalty == 5.0`、`offer_dev_penalty == 0.5`。

- [ ] **Step 1: 写失败测试**

`tests/test_rl_defaults.py`:
```python
"""Unit tests for train_rl CLI defaults."""

import pytest


def test_deviation_penalty_defaults():
    from train_rl import build_parser
    args = build_parser().parse_args([])
    assert args.bid_dev_penalty == 5.0
    assert args.offer_dev_penalty == 0.5
```

- [ ] **Step 2: 运行确认失败**

Run: `C:/Users/26036/.conda/envs/energy_env/python.exe -m pytest tests/test_rl_defaults.py -v`
Expected: FAIL（`ImportError: cannot import name 'build_parser'`）

- [ ] **Step 3: 实现——提取 parser + 改默认值**

在 `train_rl.py` 顶部新增函数（放在 `list_agents_command` 之后、`main` 之前）：

```python
def build_parser():
    """Construct the training CLI parser (testable in isolation)."""
    parser = argparse.ArgumentParser(
        description="Train independent TD3 bidding policies for all storage agents")
    parser.add_argument("--agent-names", type=str, default=None,
                        help="Comma-separated agent names to train. "
                             "Default: all agents with storage.")
    parser.add_argument("--list-agents", action="store_true",
                        help="Print all available agent names and exit")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="Comma-separated scenario names for training. "
                             "Default: all scenarios except re_ramp variants")
    parser.add_argument("--eval-scenarios", type=str, default=None,
                        help="Comma-separated scenario names held out for "
                             "post-training evaluation")
    parser.add_argument("--episodes", type=int, default=N_EPISODES,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for actor and critic")
    parser.add_argument("--noise-std", type=float, default=0.2,
                        help="Exploration noise standard deviation")
    parser.add_argument("--bid-dev-penalty", type=float, default=5.0,
                        help="Penalty per unit |bid_mult - 1.0| per period "
                             "(0 = no penalty)")
    parser.add_argument("--offer-dev-penalty", type=float, default=0.5,
                        help="Penalty per unit offer_adder per period "
                             "(0 = no penalty)")
    parser.add_argument("--bid-mult-low", type=float, default=0.6,
                        help="Lower bound for bid_mult action space")
    parser.add_argument("--bid-mult-high", type=float, default=1.4,
                        help="Upper bound for bid_mult action space")
    parser.add_argument("--start-steps", type=int, default=500,
                        help="Random exploration steps before TD3 learning")
    parser.add_argument("--save-dir", type=str, default="policies/multi_agent",
                        help="Directory for saved policy files")
    return parser
```

在 `main()` 里把 `parser = argparse.ArgumentParser(...)` 起的整段 `parser.add_argument(...)` 替换为：

```python
    parser = build_parser()
    args = parser.parse_args()
```

（`main()` 其余逻辑不变；`parser.error` 若在 `main` 中用到，保留在 `main` 内。）

- [ ] **Step 4: 运行确认通过**

Run: `C:/Users/26036/.conda/envs/energy_env/python.exe -m pytest tests/test_rl_defaults.py -v`
Expected: PASS（2 个断言通过）

- [ ] **Step 5: 提交**

```bash
git add train_rl.py tests/test_rl_defaults.py
git commit -m "Add mild deviation penalty defaults to prevent bid saturation"
```

---

### Task 2: 重训验证策略不再饱和

**Files:** 无（运行命令；产物被 .gitignore 覆盖）

**Interfaces:**
- Consumes: `train_rl.py`（Task 1 的 `build_parser` 与新默认惩罚值）
- Produces: `policies/multi_penalty/*.pt`（12 个重训策略）

- [ ] **Step 1: 后台启动重训（150 ep）**

Run（后台，直接输出便于监控）:
```
C:/Users/26036/.conda/envs/energy_env/python.exe train_rl.py --episodes 150 --start-steps 200 --bid-mult-low 0.6 --bid-mult-high 1.4 --save-dir policies/multi_penalty
```
Expected: 完成 150 ep，12 个 `policies/multi_penalty/{name}.pt` 写出；stdout 无 `[rl_env] WARNING` 回退刷屏。

- [ ] **Step 2: 验证策略不饱和（跑一个 episode，检查 bid_mult 有方差）**

`diag_saturation.py`（临时，跑完删除）:
```python
import numpy as np, torch
from scenarios import get_scenario
from models import MarketConfig
from rl_env import BiddingEnv, N_BLOCKS
from rl_td3 import load_policy
from eval_agents import actor_predict

config = MarketConfig(opf_mode="socp", verbose=False)
config.market_design.enable_multi_objective = False
config.storage.self_schedule = False
config.storage.use_nodal_price = False
agents, _ = get_scenario("baseline", T=96, config=config)
rl_names = [a.name for a in agents if a.storage is not None]
bounds = torch.tensor([[0.6, 0.0], [1.4, 50.0]], dtype=torch.float32)
obs_dim = BiddingEnv(agents, config).get_state_dim()
policies = {nm: load_policy(f"policies/multi_penalty/{nm}.pt", obs_dim, bounds)
            for nm in rl_names}
env = BiddingEnv(agents, config, rl_agent_names=rl_names,
                 bid_dev_penalty=5.0, offer_dev_penalty=0.5,
                 bid_mult_low=0.6, bid_mult_high=1.4)
obs = env.reset()
bids = {nm: [] for nm in rl_names}
for _ in range(N_BLOCKS):
    acts = {nm: actor_predict(policies[nm], obs[nm]) for nm in rl_names}
    for nm in rl_names:
        bids[nm].append(float(acts[nm][0]))
    obs, _, done, _ = env.step(acts)
    if done:
        break
print(f"{'agent':<7s} {'mean_bid':>8s} {'std_bid':>8s} {'saturated':>10s}")
for nm in rl_names:
    b = np.mean(bids[nm]); s = np.std(bids[nm])
    sat = s < 1e-3
    print(f"{nm:<7s} {b:>8.3f} {s:>8.4f} {str(sat):>10s}")
```
Run: `C:/Users/26036/.conda/envs/energy_env/python.exe diag_saturation.py`
Expected: 全部 agent `std_bid > 0.001`，`saturated = False`。若仍有 agent 饱和，记录其名字和均值，回到 Task 1 调惩罚量级（bid 2~10、offer 0.2~1）重训。

- [ ] **Step 3: 提交验证结论**

无代码改动则不提交；若 Step 2 发现饱和需调参，改动落在 `train_rl.py` 默认值并提交。

---

### Task 3: 全量评估对比

**Files:** 无（运行命令；结果写 `results/multi_penalty_eval.csv`，*.csv 被忽略不提交）

**Interfaces:**
- Consumes: `policies/multi_penalty/*.pt`（Task 2）
- Produces: 对比结论（相对 `results/multi_eval50.csv` 的 ep50 基线）

- [ ] **Step 1: 复制策略到干净目录并全量评估**

```bash
mkdir -p policies/multi_penalty_eval
for f in policies/multi_penalty/*.pt; do cp "$f" "policies/multi_penalty_eval/$(basename "$f")"; done
C:/Users/26036/.conda/envs/energy_env/python.exe eval_agents.py --policies policies/multi_penalty_eval \
  --scenarios baseline,high_re,peak_load,congestion,no_congestion --combined-only \
  --bid-mult-low 0.6 --bid-mult-high 1.4 --output results/multi_penalty_eval.csv
```
Expected: 5 场景评估完成，CSV 写出。

- [ ] **Step 2: 对比输出**

用 `results/multi_penalty_eval.csv` 与 `results/multi_eval50.csv` 对比，报告：
1. 输家（29/30/31/21R/16I）profit_delta 是否收敛（向 0 靠近）
2. 赢家（23I/25I/5R/6R/13C/15C）profit_delta 是否保留
3. 12 agent 总利润（`agent=ALL` 行）是否不再大幅为负
4. RE 消纳率变化（记录，不设硬指标）

Expected: 输家亏损显著缩小、赢家收益基本保留、总利润明显改善。

- [ ] **Step 3: 汇报并删除临时脚本**

汇报 Step 2 结果；删除 `diag_saturation.py`。

---

## 风险与回退

- 惩罚量级不当（赢家受损或输家仍亏）：回 Task 1 调 `--bid-dev-penalty`（2~10）与 `--offer-dev-penalty`（0.2~1）。
- 训练再次被外部终止：分段重跑（每段 `--episodes 50` + 复用已有 checkpoint 语义），或换 `--start-steps 200` 保持一致性。
- 非平稳性/RE 下降不属本方案范围，如复现则单独立项。
