# Winter 15-minute quote acceptance audit (P0)

Read-only. For each checkpoint, winter day and loss unit, the worst block is
re-cleared at 15-minute resolution while only that unit's **discharge** quote
changes. The other three ESS keep the checkpoint's deterministic quotes and the
other quote leg keeps its actor value. Every price and every accepted quantity
comes from the real rolling-window SOCP clearing, not a block average or a
merit-order shortcut. Data: `winter_15min_quote_acceptance.csv`,
`winter_15min_quote_acceptance.json`.

The B1 arm is a full truthful day under the same terminal-neutral mechanism.
Its per-period block sums match `b1_terminal_neutral_block_payoffs.json` to
≤ 3e-14 CNY for every unit and day, so the per-15-minute B1 series is the
registered mechanism, not a re-derivation.

## Worst blocks

| ckpt | day | block | unit | actor quote | discharge gate | truth (q=0) payoff | B1 block | q=300 payoff | truth − B1 | q=300 − B1 | truth MWh | q=300 MWh |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 12-13 | 17 | ESS14 | 0.0 | closed | 36.07 | 71.46 | 0.00 | −35.39 | −71.46 | 0.175 | 0.0 |
| 2 | 12-13 | 12 | ESS27 | 50.7 | open | 88.18 | 88.21 | 0.00 | −0.03 | −88.21 | 0.340 | 0.0 |
| 2 | 12-30 | 23 | ESS14 | 0.0 | closed | −52.70 | −21.87 | −52.74 | −30.83 | −30.87 | 0.0 | 0.0 |
| 2 | 12-30 | 23 | ESS27 | 47.6 | open | −48.58 | −24.27 | −48.55 | −24.31 | −24.28 | 0.0 | 0.0 |
| 2 | 12-31 | 23 | ESS14 | 0.0 | closed | −52.71 | −19.54 | −52.69 | −33.17 | −33.15 | 0.0 | 0.0 |
| 2 | 12-31 | 0 | ESS27 | 62.9 | open | 53.44 | 53.41 | 0.00 | +0.02 | −53.41 | 0.300 | 0.0 |
| 10 | 12-13 | 3 | ESS14 | 299.7 | open | 72.95 | 71.55 | 0.00 | +1.40 | −71.55 | 0.400 | 0.0 |
| 10 | 12-13 | 12 | ESS27 | 274.2 | open | 88.21 | 88.21 | 0.00 | 0.00 | −88.21 | 0.340 | 0.0 |
| 10 | 12-30 | 10 | ESS14 | 299.4 | open | 111.14 | 111.11 | 0.00 | +0.03 | −111.11 | 0.400 | 0.0 |
| 10 | 12-30 | 10 | ESS27 | 282.1 | open | 128.18 | 128.14 | 0.00 | +0.04 | −128.14 | 0.400 | 0.0 |
| 10 | 12-31 | 17 | ESS14 | 272.5 | open | 98.16 | 98.11 | 0.00 | +0.04 | −98.11 | 0.400 | 0.0 |
| 10 | 12-31 | 17 | ESS27 | 258.4 | open | 108.18 | 108.13 | 0.00 | +0.05 | −108.13 | 0.400 | 0.0 |

`truth (q=0)` is the same-state counterfactual, not the B1 path. At the R10
loss states it recovers the B1 block payoff almost exactly, so the entire gap
there is the quote. At R2 12-13 b17 ESS14 the truthful counterfactual is only
36.07 against B1's 71.46: that loss is the policy's lower SOC and the closed
discharge gate, not the quote. At R2 12-30/12-31 b23 the block clears at
≈32 CNY/MWh and the unit is charging; the quote is economically irrelevant
there.

## Per-period detail: R10, 2022-12-30, block 10, ESS14

| quote | period | LMP | effective offer (quote+100) | price-taking condition | accepted MWh | SOC end | payoff | B1 payoff | payoff − B1 |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 0 | 40 | 377.7 | 100 | yes | 0.100 | 0.900 | 27.769 | 27.760 | +0.008 |
| 0 | 41 | 377.8 | 100 | yes | 0.100 | 0.832 | 27.783 | 27.775 | +0.008 |
| 0 | 42 | 377.8 | 100 | yes | 0.100 | 0.764 | 27.776 | 27.768 | +0.008 |
| 0 | 43 | 378.1 | 100 | yes | 0.100 | 0.696 | 27.814 | 27.807 | +0.007 |
| 258 | 40 | 384.1 | 358 | yes | 0.000 | 0.900 | 0.000 | 27.760 | −27.760 |
| 258 | 41 | 384.3 | 358 | yes | 0.000 | 0.900 | 0.000 | 27.775 | −27.775 |
| 258 | 42 | 384.2 | 358 | yes | 0.000 | 0.900 | 0.000 | 27.768 | −27.768 |
| 258 | 43 | 384.6 | 358 | yes | 0.000 | 0.900 | 0.000 | 27.807 | −27.807 |
| 300 | 40 | 384.1 | 400 | no | 0.000 | 0.900 | 0.000 | 27.760 | −27.760 |
| 300 | 41 | 384.3 | 400 | no | 0.000 | 0.900 | 0.000 | 27.775 | −27.775 |
| 300 | 42 | 384.2 | 400 | no | 0.000 | 0.900 | 0.000 | 27.768 | −27.768 |
| 300 | 43 | 384.6 | 400 | no | 0.000 | 0.900 | 0.000 | 27.807 | −27.807 |

The 258 quote passes the price-taking test in every period (358 < 384) and is
still not dispatched: the other three units at their truthful offer (effective
100) satisfy the block first. The actor quote 299.4 fails the test and also
idles. The battery stays at SOC 0.900 instead of ending the block at 0.696.

## Per-period detail: R10, 2022-12-30, block 10, ESS27

Selected periods 40 and 43 are shown to keep the graded response readable; the
full four-period rows are in the CSV.

| quote | period | LMP | effective offer | price-taking | accepted MWh | SOC end | payoff | B1 payoff | payoff − B1 |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 0 | 40 | 418.1 | 100 | yes | 0.100 | 0.900 | 31.808 | 31.799 | +0.010 |
| 0 | 43 | 422.3 | 100 | yes | 0.100 | 0.696 | 32.228 | 32.219 | +0.010 |
| 150 | 40 | 420.3 | 250 | yes | 0.100 | 0.900 | 32.530 | 31.799 | +0.731 |
| 150 | 43 | 422.3 | 250 | yes | 0.042 | 0.814 | 17.725 | 32.219 | −14.494 |
| 258 | 40 | 423.0 | 358 | yes | 0.000 | 0.900 | 0.000 | 31.799 | −31.799 |
| 258 | 43 | 427.3 | 358 | yes | 0.000 | 0.900 | 0.000 | 32.219 | −32.219 |
| 300 | 40 | 423.0 | 400 | yes | 0.000 | 0.900 | 0.000 | 31.799 | −31.799 |
| 300 | 43 | 427.3 | 400 | yes | 0.000 | 0.900 | 0.000 | 32.219 | −32.219 |

The 150 quote is partially accepted and can beat B1 in early periods, but the
accepted energy falls and the block total is still 23.1 CNY below the truthful
counterfactual. The 282 actor quote and 300 are both zero-acceptance. This is a
graded response, not a single cliff.

## Per-period detail: R2, 2022-12-13, block 17, ESS14

| quote | period | LMP | effective offer | price-taking | accepted MWh | SOC end | payoff | B1 payoff | payoff − B1 |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 0 | 68 | 305.2 | 100 | yes | 0.000 | 0.219 | 0.000 | 11.640 | −11.640 |
| 0 | 69 | 306.5 | 100 | yes | 0.100 | 0.219 | 20.650 | 20.193 | +0.457 |
| 0 | 70 | 306.2 | 100 | yes | 0.041 | 0.151 | 8.555 | 19.822 | −11.267 |
| 0 | 71 | 306.2 | 100 | yes | 0.033 | 0.123 | 6.864 | 19.808 | −12.944 |
| 258 | 68 | 306.2 | 358 | no | 0.000 | 0.219 | 0.000 | 11.640 | −11.640 |
| 258 | 71 | 308.1 | 358 | no | 0.000 | 0.219 | 0.000 | 19.808 | −19.808 |

The actor statement at this state is **truthful** (0.0) but the discharge gate
is closed, so the executed quote is zero and the unit cannot capture the block
even though the truthful counterfactual would. The remaining gap to the B1
block (36.07 vs 71.46) is the different SOC path: the policy enters the block
at SOC 0.219, while the B1 arm has more energy available.

## Findings

1. **R10 high quotes are confirmed as the direct cause.** At 12-30 b10 and
   12-31 b17, the actor's 258–299 quote produces zero accepted discharge and
   zero payoff in every 15-minute period, while q=0 or q=100 reproduces the B1
   block payoff. At 12-13 b3 ESS14 and 12-13 b12 ESS27 only q=0 recovers the
   block (q=100 is already displaced); the high quote is still zero. The lost
   block is 71–128 CNY per unit. The idle battery stays at SOC 0.9 (12-30) or
   0.44/0.88 (12-31).
2. **The price-taking condition is necessary but not sufficient.** With four
   homogeneous batteries, a 258 quote can be below the block LMP and still be
   displaced by the other units at effective 100. The audit therefore cannot
   use `quote + cycle_cost < LMP` as an acceptance criterion; the joint
   clearing is the only valid test.
3. **The economically harmful premium can be much smaller than 258.** At R2
   12-13 b12 ESS27 a 50.7 quote loses the whole 88.21 CNY block; at R2
   12-31 b0 ESS27 a 62.9 quote loses 53.41 CNY. The 258–299 region is where the
   R10 policy sits, not the boundary of the problem.
4. **R2's worst blocks are not all quote losses.** 12-13 b17 ESS14 is a
   closed discharge gate plus a low-SOC path; 12-30/12-31 b23 are low-price
   charging blocks where the quote is immaterial. The phase-15 block-level
   attribution needs this qualification.
5. **The per-15-minute response is graded.** At 12-30 ESS27, 150 is partially
   accepted and can beat B1 early, while 258 and 300 are zero. A continuous
   quote branch that can reach the middle is not merely cosmetic; it is able to
   recover part of the block without returning all the way to truthful.

## Answers

- **Was the high discharge quote confirmed as the loss cause per 15 minutes?**
  Yes for R10 at 12-13 b3 ESS14, 12-13 b12 ESS27, 12-30 b10 ESS14/27 and
  12-31 b17 ESS14/27. No for R2 12-30/12-31 b23 and no for R2 12-13 b17 ESS14,
  where the gate or the state path is the cause. The distinction is necessary
  because the two checkpoints fail for different reasons.

## Boundary

One target leg per clearing; the other ESS actions are the checkpoint's
deterministic quotes, held fixed and recorded in the JSON. No future SOC is
forced. No H2, no legacy test, no training, no checkpoint write.
