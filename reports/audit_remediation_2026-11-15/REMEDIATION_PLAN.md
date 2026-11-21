# 审计整改方案（剩余项）

依据：`reports/full_project_audit_2026-11-15/REPORT.md`。
已落地部分与验收见同目录 `REPORT.md`。
本方案只覆盖**尚未处理**的条目，不含已完成项。

> 执行状态见同目录 `REPORT.md`。方案里写的 `formal_training_v5` 已在执行中被
> `formal_training_v6` 取代：B1 基线是在错误径向树上算出来的，重算后必须换
> 目录，于是这次重冻结升为 v6。v5 目录保留为历史证据，不再是指针目标。

范围裁决：负荷峰值与负荷映射本轮不动（`typical_day_industry_subset` 的尺度、
`bus 28` 的集中度，见下文“已知测量”）。`da_market*` / `local_da*` 新引擎不动。

---

## 0. 已完成，不再进入本方案

`F1` 径向树、`F2` 新能源超铭牌、`F3` 部分日 headline、`F5` 出清内折现、
`F7` 缓存键、`F8` 报价切片、`3.7` 两种多目标互斥、`4.4` `delta_pi_percent`
重复定义、`9.5.6` `positive_rate` 的 NaN、`8.1` `clarabel` 依赖、
`3.5` I2R 不可行时丢时段、`served_covers_forecast` 命名。

与审计无关但同批完成：`tests/test_rl_spec.py` 的场景换成 `typical_day`，
去掉 `xfail`，`test_a_blocks_realized_lmp_reaches_only_the_next_observation`
恢复可测。

当前基线：`PYTHONPATH=src python -m pytest tests/ -q` → 1485 passed。

---

## 1. 已知测量（本方案依赖的事实）

| 事实 | 值 | 来源 |
|---|---|---|
| 53 个训练日全天出清，无回退、无弃电 | 53/53 | 全天 24 block 汇总 |
| 电压越界 | 0 / 167,904 格 | 同上 |
| 电压贴边 | 26 天触及 0.9300，5.0% 格低于 0.95 | 同上 |
| 日均 LMP / 批发曲线均值 | 1.002 / 1.052 / 1.090 (min/median/max) | 同上 |
| `PRICE_REF_CNY_MWH` 饱和 | 198 / 160,908 格，2/53 天 | 同上 |
| 循环成本在出清目标中 | 是，费率 100→300 使吞吐 21.207→0.000 MWh | `tmp/diag_cycle_in_clearing.py` |
| 出清费率偏离真实费率的代价 | 真实 100、出清 50 → 利润 656.77→494.41 CNY | `tmp/diag_cycle_mismatch.py` |
| 两套结算在同一日的结果 | 支付 −715.80 vs 利润 +164.40（ESS9） | `tmp/diag_settle_paths.py` |

---

## 2. 硬阻塞：re-freeze 通道

`F4` `F6` `F9` `F10` `5.3` `5.10` 以及 `4.2`/`5.9`/`10.2` 需要写冻结文件。
`.claude/hooks/protect_frozen.py` **按路径**保护（`src/<name>` for every entry in
the live manifest's `source_sha256`），因此新建 v5 manifest 再切指针**不能**
放行——v5 会保护同样的路径。唯一可行路径是在钩子里加显式放行。

三种形态，边界不同，需裁决：

| 形态 | 放行范围 | 代码量 | 风险 |
|---|---|---|---|
| 环境变量开关 | 该会话内所有写操作 | 最小 | 最宽 |
| 令牌文件 | 存在期间所有写操作 | 小 | 仍不限定路径 |
| 路径白名单 | 仅补丁声明的那 6 个路径 + 令牌 | 最大 | 最窄（建议） |

放行只是手段，真正的控制是 v5 manifest 的新哈希。v4 目录在本方案中保持只读、
哈希不变。

---

## 3. 阶段

每个阶段一个提交，独立可回滚。任何一段验收不过，停在该段，不进入下一段。

### S1 — 机械项（不改变任何数值结果）

| 条目 | 改动面 | 要点 |
|---|---|---|
| `3.8` 求解器参数单一来源 | `dispatch_socp` / `dispatch_ldf` / `dispatch_dc` / `dispatch_core` 等 6 处的字面量 | 收敛到一处；`opf_tolerance` / `opf_max_iter` 当前无读取者，按 `8.2` 一并删除或在模型中真正生效，二选一，不能留悬空参数 |
| `3.8` 许可缺失明确报错 | `dispatch.py` 已有的提示文本 | 在 `optimize()` 抛裸 `GurobiError` 的路径上转成带说明的错误 |
| `9.2` 金额断言改相对容差 | `tests/` 中 21 处 `assertAlmostEqual(places=)` | 只改金额类；计数、维度类保持原样 |
| `10.3` 忽略规则收窄 | `.gitignore` 的全局 `*.csv` | 改为按目录忽略，把 DA 输入映射与证据 CSV 纳入版本控制 |

**验收**
- `PYTHONPATH=src python -m pytest tests/ -q` → 0 failed
- 固定日期出清前后**逐位一致**：对 `typical_day` 的 SOCP 结果取 `lmp`、
  `clearing_objective`、各机组的 `p_ch` / `p_dis` 做二进制比对，必须完全相同
- `git status` 中新增的 CSV 是预期的那些，不是意外产物

**回滚**：单提交 revert。无数据依赖。

---

### S2 — re-freeze 与 v5 manifest

1. 按裁决实现 `protect_frozen.py` 的显式放行。
2. 建 `reports/v2_cy_jn_2025/formal_training_v5/`，**复制 v4 内容**后修改，
   不原地改 v4。
3. 应用 `tmp/refreeze_v5/frozen_patches.diff`（6 项）。
4. 把 `4.2` 结算声明、`5.9` 线程数、`10.2` 源码清单一并写进 v5。
5. 更新 `MANIFEST_DIRS[0]`、`CLAUDE.md` 与 `.claude/rules/*` 中的 live 指向。

**验收**
- `PYTHONPATH=src python src/run_formal_training.py --check --config-dir reports/v2_cy_jn_2025/formal_training_v5`
  → `all_pass: true`、`budget_contract_ok: true`、`status: GO_FORMAL_TRAINING`
- v4 的 22 个 `source_sha256` 与 v4 目录内容哈希不变（v4 仍是只读证据）
- 放行关闭后，对 6 个冻结路径的写入再次被拒（手工验证一次）

**回滚**：撤销 v5 目录与钩子改动；src 用本段提交前的哈希恢复。

**风险**：`F4` 改的是节点重解不带 slack 惩罚，`F6` 改的是端点钉住时仍加终端
价值——两项都直接影响价格形成。S2 之后、S5 之前的所有价格数字都不能作为结论。

---

### S3 — 冻结补丁的验证

`F4` / `F6` / `F9` / `F10` / `5.3` / `5.10` 在 S2 之前从未执行过任何测试。

**验收**
- 全测 0 failed
- 固定日前后对照表：`baseline` 与 `typical_day_industry_subset` 各一天，
  记录 LMP 均值/最大值、`clearing_objective`、每台机组 payoff、是否 fallback
- `F10` 的定向验证：一个部分日（96 段中有未运行段）不再按整日结算
- `5.10` 的定向验证：故意在写检查点中途中断，文件要么完整要么不存在

**回滚**：回到 v4 哈希 + 关闭放行。

---

### S4 — 行为项（每条一个提交）

按“是否改变数值”排序，会改数值的排在 S5 之前。

| 条目 | 改动面 | 数值影响 |
|---|---|---|
| `3.5` 线损迭代改残差判据 | `dispatch_ldf` 的迭代循环 | **会改** |
| `6.3` 出清/奖励/`update()` 的 NaN/Inf 检测 | `dispatch_socp` 返回处、奖励组装、`update()` | 不改数值，可能把静默 NaN 变成抛错 |
| `5.2` 惩罚量级可观测 | `rl_da_gate.update()` 已算的 `anchor_by_agent` / `actor_loss_by_agent` | 不改数值，只落盘 |
| `5.8` 失败块不入池 | legacy `train_rl.py` 的无条件 `remember()` | 只影响 legacy 训练路径 |
| `4.4` 奖励与收益分行 | `run_artifacts` 的 `metrics.csv` | 不改数值，改表结构 |
| `3.5` 汇总加引擎列 | `make_training_summary` | 不改数值 |
| `6.2` SOC 换算收敛 | 当前散在 9 个文件的 `soc*e_max` / `e/e_max` | 只允许逐位不变的写法 |
| `8.2` 删除无人调用的 `capacity_basis` | `config/independent_da.json` | 无 |

`82` 处 `except Exception` 的收窄**单列**，取决于裁决（见 §4）。

**验收**：每条自带命令 + 全测 0 failed。`3.5` 与 `6.2` 需附固定日的逐位比对
或前后数值表。

---

### S5 — 重算基线

前置：S2、S3、S4 中会改数值的项全部落地。

- `PYTHONPATH=src python src/baseline_ladder.py --rung all --bands both --json <新路径>`
- `PYTHONPATH=src python src/accept_typical_day_industry.py --json <新路径>`
- benchmark 表与 training-readiness headroom 重出

**验收**
- 新产物写入新路径，旧产物不动
- 新旧差异表：每个量的变化、以及变化能否归因到具体某一条改动
- TD-12 的状态更新为“已重算”，并列出被替代的旧产物

---

### S6 — 收口

- `REPORT.md` 的验证块更新为最终命令与计数
- `TECHNICAL_DEBT.md` 逐条更新状态（含 TD-12 的关闭）
- `CLAUDE.md` 与 `.claude/rules/*` 的 live 指向、冻结集描述、预算数字

---

## 4. 待裁决

| 编号 | 问题 | 建议 |
|---|---|---|
| D-A | re-freeze 放行形态（三选一） | 路径白名单 |
| D-B | `F11` 三组结构重构（两套结算、三种锥、两套定价）本轮做不做 | 不做；单独立项 |
| D-C | `82` 处宽 `except` 本轮做不做 | 不做；每一处都要判断“静默失败是否应该变成抛错” |
| D-D | 循环成本是否移出出清目标 | **保留**。测量显示出清费率低于真实费率会让机组多循环且少赚（656.77→494.41）；移出必须同时改报价符号守卫、动作规格与观测锚点，属新 candidate id |
| D-E | `5.4` 冗余 `ema_dev` | 不做。它是 `OBS_V6` 的 12 维之一，删除即改冻结观测接口 |
| D-F | `4.2` 的声明改成什么 | 建议改成实际口径（`participant_payoff` 五项账本 + 端点钉住的整天重解），而不是把代码改成 `two_settlement` |

---

## 5. 风险登记

1. `S2` 之后价格形成改变，任何在 S2 与 S5 之间产生的数字都不能引用。
2. `F4`/`F6` 是已发布结论（`JN-Base` benchmark、三日结算）的依赖项，重算前
   不得引用旧值。
3. 放行形态若选环境变量，该会话内任何写操作都能越过守卫；必须配合“关闭后
   复验一次拒绝”的步骤。
4. `6.2` 与 `6.3` 的改动面宽，若出现逐位差异，应回退该条而不是调容差。
5. 负荷尺度未动，所以 S5 重算出的电压贴边、线路满载现象会原样保留；报告中
   必须继续标明这是映射结果而非网络能力。

---

## 6. 总验收

```
PYTHONPATH=src python -m pytest tests/ -q                                   # 0 failed
PYTHONPATH=src python src/run_formal_training.py --check \
    --config-dir reports/v2_cy_jn_2025/formal_training_v5                   # all_pass / GO
git apply --reverse --check -p1 tmp/refreeze_v5/frozen_patches.diff         # 成功 = 已应用
```

`git apply --check` 在应用前成功、应用后失败，所以最终态用 `--reverse --check`
判断。
