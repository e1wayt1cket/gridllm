# C0 candidate 消融判定(common split 重跑,双报)

label: NON_FORMAL_DIAGNOSTIC_PILOT
merged(预注册对称带): **MIXED:DESIGN_FLAW/NOT_TESTED_RUN_FAILURE**
merged(探索性单向带,事后修订): **MIXED:NEGATIVE/NOT_TESTED_RUN_FAILURE**

| seed | 对照 r4 | 实验 r4 | A 差 [CI] | C r1 差 | verdict(对称) | verdict(单向) |
|---|---|---|---|---|---|---|
| 11 | RUN_FAILURE(r1/4) | — | NOT_TESTED | +31.9 | NOT_TESTED_RUN_FAILURE | NOT_TESTED_RUN_FAILURE |
| 23 | -101.3 | -69.7 | +31.7 [-59.1, +139.2] | +26.1 | DESIGN_FLAW | NEGATIVE |

判据 B 分 ESS 明细(r4;贴边=评估模式开价边缘比例,梯度留存=训练模式映射导数中位/SLOPE):

- seed11: NOT_TESTED(run 未完成 r4)
- seed23:
  - ESS9.charge: 贴边 0.000→NA (NA), 梯度留存 0.001, 命中=n
  - ESS9.discharge: 贴边 0.000→NA (NA), 梯度留存 1.000, 命中=Y
  - ESS14.charge: 贴边 NA→0.000 (NA), 梯度留存 1.000, 命中=Y
  - ESS14.discharge: 贴边 1.000→0.973 (+2.7pp), 梯度留存 1.000, 命中=Y
  - ESS17.charge: 贴边 1.000→NA (NA), 梯度留存 0.001, 命中=n
  - ESS17.discharge: 贴边 1.000→0.000 (+100.0pp), 梯度留存 1.000, 命中=Y
  - ESS27.charge: 贴边 NA→NA (NA), 梯度留存 1.000, 命中=Y
  - ESS27.discharge: 贴边 1.000→NA (NA), 梯度留存 1.000, 命中=Y

逐轮曲线(mean_fleet_delta):

- seed11 对照: r1:-65.7 → r2:-91.4 → r3:-84.7 → r4:-98.8
- seed11 实验: r1:-33.8
- seed23 对照: r1:-61.1 → r2:-96.0 → r3:-87.3 → r4:-101.3
- seed23 实验: r1:-35.0 → r2:-69.8 → r3:-62.1 → r4:-69.7