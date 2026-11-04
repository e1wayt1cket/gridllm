# Physical screen (V2-CY-JN-2025)

Every paired day is cleared once with the `line50_q2.0` support assumption and the Jiangnan day-ahead price. The support assumption is **idealised**: it is the same screening assumption the frozen V2 subset screen used, not a selected device or a costed engineering design.

## Summary by arm

| arm | days | optimal | 0 LMP fallbacks | voltage OK | lines OK | terminal SOC OK | reverse flow OK | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JN-Base | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 0 |
| JN-C80 | 21 | 21 | 21 | 21 | 21 | 21 | 21 | 0 |
| JN-C90 | 21 | 21 | 21 | 21 | 21 | 21 | 21 | 0 |
| JN-C100 | 21 | 21 | 21 | 21 | 21 | 21 | 21 | 0 |
| JN-NoESS | 21 | 21 | 21 | 21 | 21 | 21 | 21 | 0 |

The terminal-SOC column counts rows whose `terminal_soc_max_err` is exactly 0; the no-storage arm has no SOC to pin and reads as OK.

## Checks performed

| # | check | how it is read |
|---|---|---|
| 1 | Gurobi status | `solver.final.optimal`, and no exception |
| 2 | nodal active-power balance | injections + net line flow with the model's own I^2R term at the receiving end; the residual must be ~0 |
| 3 | voltage band | every bus-period inside 0.93–1.05 p.u. |
| 4 | line thermal limit | every line-period at or below 100% utilisation |
| 5 | storage power and SOC | p_ch/p_dis within the 0.4 MW rating, SOC inside [0.10, 0.90] |
| 6 | terminal SOC | SOC(T) equals SOC(0) to 1e-6 |
| 7 | charge/discharge exclusion | p_ch * p_dis is zero; a net-flow formulation makes this structural, not a binary |
| 8 | reactive power balance | rebuilt from the published line Q flows, the agents' `q_re`, the slack Q and the model's I^2X term; the residual must be ~0 |
| 9 | LMP extraction | zero fallback cells, finite values, and the slack LMP equals the Jiangnan price |
| 10 | PCC exchange | import/export extremes and the 2.5 MW reverse limit |

**Reactive balance is rebuilt, not assumed.** The SOCP engine publishes `line_flow_mvar` and `grid_q_mvar`; with the agents' own `q_re` and the configured compensation, the reactive balance closes to **4.56e-07 MVar** at worst over 180 checked day(s). 

## JN-Base by group

| group | days | min V (p.u.) | max V (p.u.) | max line util. | ESS payoff (CNY/day, median) |
|---|---:|---:|---:|---:|---:|
| national_day_2022 | 5 | 0.9681 | 1.0195 | 0.5181 | -0.0 |
| summer | 21 | 0.9488 | 1.0231 | 0.6160 | -0.0 |
| autumn | 47 | 0.9330 | 1.0231 | 0.8150 | 111.3 |
| winter | 23 | 0.9348 | 1.0231 | 0.7142 | 456.7 |

Every paired day cleared. No infeasibility file was produced.
