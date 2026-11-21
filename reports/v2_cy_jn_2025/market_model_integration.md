# How the Jiangnan price enters the market (V2-CY-JN-2025)

## The external price

For pair `(load_date, price_date)` the clearing receives the 96-point `price_value_day_ahead_south` curve of `price_date` as `wholesale`, the PCC price `lambda_ws(t)`. The objective term is unchanged:

    - sum_t  p_grid(t) * lambda_ws(t) * dt        dt = 0.25 h

`p_grid > 0` is import, so the term is the cost of buying from the upper grid. `p_grid < 0` is export and earns the same price; no buy/sell spread was added. The PCC reverse-flow limit (2.5 MW) and every other constraint are the existing ones.

The unit is CNY/MWh by inference (`data_audit.md`), consistent with `MW x h` in the same expression. It is not externally confirmed.

## The nodal price

LMPs are the power-balance duals of the same clearing:

    LMP(i,t) = -Pi[balance(i,t)] / dt

They are **not** the Jiangnan price copied to 33 buses. The slack bus's LMP is the marginal cost of one more MW at the PCC and equals `lambda_ws(t)`; every other bus differs by its network constraints, losses and congestion. `results.csv` records the slack-bus deviation from `lambda_ws` as an extraction check.

## Storage settlement

ESS9, ESS14, ESS17 and ESS27 settle at their own node's LMP through `participant_payoff`, the same function the rest of the project uses. Efficiency, SOC bounds and the cycle cost are unchanged.

## What the storage fleet can and cannot move

The fleet receives the Jiangnan price and its quotes change the distribution market's clearing and therefore its LMPs. It has no effect on the Jiangsu or Jiangnan day-ahead price, which is an input.

## Price formation: the two layers, kept apart

| layer | assumption | who sets it |
|---|---|---|
| Jiangsu / Jiangnan day-ahead | exogenous, price taking | the recorded 2025 curve; no participant here can move it |
| distribution node LMP | endogenous, price influencing | the clearing; storage quotes change the nodal prices they are settled at |

Every arm in this version is a price taker at the regional layer and a price influencer at the node layer. No arm is claimed to move the Jiangnan price, and none is a stackelberg model of the province.

## Arm settings

| arm | price | mv_value | cycle cost (CNY/MWh) | storage |
|---|---|---|---:|---|
| JN-Base | south | no | 100 | 4 units |
| JN-C80 | south | no | 80 | 4 units |
| JN-C90 | south | no | 90 | 4 units |
| JN-C100 | south | no | 100 | 4 units |
| JN-NoESS | south | no | 100 | none |
| JN-RE | south | yes | 100 | 4 units | disabled, not run |

C80/C90/C100 change the cycle cost alone. JN-RE is declared and disabled: `mv_value` has no confirmable unit or availability time and would require a new observation version.
