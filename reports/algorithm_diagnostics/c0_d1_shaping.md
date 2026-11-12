# C0 D1:C2 shaping 贡献(回放分解)

label: NON_FORMAL_DIAGNOSTIC_PILOT; verdict: **SHAPING_SIGNIFICANT**

判据:ESS-日 级 S/|B|(S=逐 block |shaping| 之和,B=全日差分)的日聚类中位 ≥20% → C0 candidate = C2 消融。无信息 ESS-日(|G_policy|<1 且 |G_b1|<1 CNY)剔除并单列。

## seed11

| checkpoint | 自检 | N(剔除) | 比例中位 | 95% CI | 净 shaping |max| | 判定 |
|---|---|---|---|---|---|---|
| best_r1 | Y | 44(16) | 4.679 | [0.669, 8.773] | 11.1118 | SHAPING_SIGNIFICANT |
| final_r10 | Y | 44(16) | 1.883 | [0.000, 3.491] | 11.1122 | SHAPING_SIGNIFICANT |

## seed23

| checkpoint | 自检 | N(剔除) | 比例中位 | 95% CI | 净 shaping |max| | 判定 |
|---|---|---|---|---|---|---|
| best_r1 | Y | 44(16) | 3.934 | [0.800, 8.620] | 11.1118 | SHAPING_SIGNIFICANT |
| final_r10 | Y | 44(16) | 5.206 | [0.655, 12.147] | 11.1313 | SHAPING_SIGNIFICANT |
