# 09 当前状态（Current State）

> 本文件回答一个问题：**今天这个仓库里，什么已经发生、什么正在发生、什么被阻塞。**
> 历史实验结论集中记录在 `reports/` 与 `10_DECISION_LOG.md`，此处只留索引与当前设计状态。
> 最后更新：2026-11-12

---

## 1. 工作区身份

| 项 | 值 | 证据 |
| --- | --- | --- |
| 分支 | `review` | `git branch --show-current` |
| HEAD | `63b008d` | `git log --oneline -1` |
| 工作树 | 有未提交改动（13 个已跟踪文件 + 大量未跟踪目录） | `git status --porcelain` |
| 未提交改动的处理 | **不得覆盖** | 见本文 §7 |

## 2. 协议版本

| 协议 | 配置哈希 | 状态 | 证据 |
| --- | --- | --- | --- |
| `formal_training_v4` | `7834d0aa…` | **live**，`finalized`，`GO_FORMAL_TRAINING` | `reports/v2_cy_jn_2025/formal_training_v4/formal_training_manifest.json` |
| `formal_training_v3` | `1b47e0ed…` | 历史，只读 | M0 审计 §0.1 |
| `formal_training_v2` | `2148a577…` | 历史，只读 | 同上 |
| `formal_training_ready` | `b21a97c1…` | `SUPERSEDED_READ_ONLY` | 同上 |

**工作树与 v4 代码清单 22/22 pin 一致，与 v3 的 19 个 pin 有 8 个不一致。**
因此任何"live manifest 是 v3"的表述都是过期的；`.claude/rules/model_constraints.md`
里的 `storage_quote_v3` / `fixed_scale 1000` 也对应 v3，不是工作树。
本文件按 **v4** 记录现状；该表述冲突已登记到 `11_OPEN_ISSUES.md`。

## 3. 已有正式训练运行

| run | 协议 | 状态 | 备注 |
| --- | --- | --- | --- |
| `runs/FORMAL_1b47e0ed_seed11` | v3 | `completed_early_stop` | 历史证据，只读 |
| `runs/FORMAL_1b47e0ed_seed23` | v3 | `completed_early_stop` | 历史证据，只读 |
| `runs/FORMAL_1b47e0ed_seed37` | v3 | `exploratory_development` | 中途停止，仍是证据 |
| `runs/FORMAL_7834d0aa_seed11` | v4 | `completed_early_stop` | 当前协议 |
| `runs/FORMAL_7834d0aa_seed23` | v4 | `completed_early_stop` | 当前协议 |

上述 run 目录、`snapshots/` 对应树、三套 manifest 目录均属冻结集，`.claude/hooks/protect_frozen.py` 拒绝对其 Edit/Write。

**本文件不复述这些 run 的任何数值结论。** 需要引用时，以各自的 `run_manifest.json` /
`eval_history.json` / `episodes.csv` 路径为准。

## 4. 市场机制审计进度

| 阶段 | 内容 | 状态 | 产物 |
| --- | --- | --- | --- |
| M0 | 当前出清机制审计（还原现有模型） | **已完成** | `reports/market_mechanism_audit/M0_current_clearing_audit_2026-11-12/` |
| M1 | 江苏日前市场规则审计 + 新模型方案 | **本次进行中** | `reports/market_mechanism_audit/M1_jiangsu_da_rule_audit_2026-11-12/` |
| M2 | 出清器实现 | **未开始**，需用户批准 | — |
| M3 | 训练 | **未开始**，需用户批准 | — |

### M0 已确认的结构事实（摘要，细节见 M0 原文）

现有模型是 **确定性多时段最优潮流**，不是"报价撮合 + 出清价结算"的拍卖模型。以下为代码事实（PASS）：

- 报价以"成本/估值参数"形式直接进入目标函数，不是量价曲线；没有 bid/ask 撮合。
- 动作箱 `charge (0,150)` / `discharge (0,300)`，真值点在 **下边界 (0,0)**，策略只能加价。
- 每块 16 时段窗口只覆盖 4 段申报，其余 12 段用真申报占位。
- 窗口内外单价完全已知，策略只见到已发生的 LMP。
- `terminal_value` 项只作用于窗口末端；在 `day_end` / `full_day` 窗口被硬等式钉住后退化为常数。
- 周期成本 100 CNY/MWh 同时进入出清目标与结算账本（同一参数两处各用一次，非重复计费）。
- 结算账本 `participant_payoff` 按 **节点 LMP** 计价，因此"报价 → 出清 → LMP → 收益"通路存在，但幅度 **NOT_TESTED**。

## 5. 当前设计与新研究定义之间的已知差距

下表的对照对象是 `00_PROJECT_CHARTER.md`。差距 **尚未修复，也未获批准修复**。

| # | 宪章要求 | 当前状态 | 类别 |
| --- | --- | --- | --- |
| G1 | 只研究日前市场 | 环境按 24 个 4 时段滚动窗顺序决策，逐步看到已实现的 LMP | **结构性冲突**：这是日内滚动，不是日前 |
| G2 | 日终 SOC 不强制 0.5 | 冻结协议 `terminal_soc.final = 0.5`，`day_end_window_only: true` | 契约冲突 |
| G3 | 终端库存价值需经济依据 | 出清目标里有 `γ^T·TV·soc_T`，`TV = mean(wholesale)`，且日终被钉住后退化为常数 | 设计待重做 |
| G4 | 报价是量价曲线 | 动作是单个价格点，无分段、无功率轴 | 接口待重做 |
| G5 | 申报价格范围遵循规则 | 动作箱 0–150 / 0–300，与规则 0–1500 无对应关系 | 接口待重做 |
| G6 | 内部成本参数不得当作市场申报 | 循环成本进入出清目标，与申报价同量纲耦合 | 设计待重做 |
| G7 | 当日真实日前价格不得泄漏 | 策略观察不含当日价格（已满足）；但策略在日内看到已实现 LMP | **与 G1 同源** |

这些差距 **不是缺陷清单**，是重构的设计输入。每一条的处理方式必须在 M2 前由用户裁决。

## 6. 当前阶段的权限

| 允许 | 禁止 |
| --- | --- |
| 读源码、读文档、读官方规则 | 改正式 Python 源码 / 出清器 / MATD3 / 动作与观察接口 |
| 写项目上下文文档 | 改现有报价映射、写 Replay、改 checkpoint |
| 写 M1 研究规格与验证计划 | 启动训练、正式清算实验、修改历史实验结果 |
| | commit / push |
| | 读 Test / Holiday / H2 封存数据 |

## 7. 工作树未提交改动的处理纪律

工作树有用户未提交的改动（`src/rl_spec.py`、`src/da_quote_map.py`、`src/run_formal_training.py` 等）。
本次任务 **只写新文档目录**，不触碰这些文件。

判定方式：任务前后各取一次 `git status --porcelain` 并比对；本次新增项应只包含
`docs/project_context/`、`tasks/`、`reports/market_mechanism_audit/M1_.../` 与 `CLAUDE.md`。

## 8. 数据边界

- 训练 53 日 / 验证 11 日 / 测试 11 日 / 节假日 5 日（`.claude/rules/data_governance.md`）。
- Test 与 Holiday 带 `LEGACY_SEEN_TEST` 标签；H2 价格日期 holdout 仍封存（`h2_opened: false`）。
- 本次任务 **未读取** 上述任何封存数据。
