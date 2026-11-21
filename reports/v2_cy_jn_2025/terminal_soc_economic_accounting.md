# Terminal SOC economic accounting (V2-CY-JN-2025)

## The adopted rule

Every RL storage unit starts and must end the day at

\[
SOC_{i,24}=SOC_{i,0}=0.5,
\]

enforced **per unit**, not on the fleet average. The constraint is absolute; it
is not the window-relative `soc[T] == soc[0]` pin used by a one-shot full-day
clear.

The rolling interface is unchanged: 24 blocks, each clearing a 16-period
look-ahead window and committing 4 periods. A window whose own end is the
day's last period is cleared as `horizon_type="day_end"` and pins the absolute
target; every earlier window stays `window` and is unpinned. Blocks 20–23 all
look ahead to period 96, so their plans must be able to land on 0.5; only the
committed actions of block 23 determine the executed day-end state.

The target lives on the per-agent `StorageSpec` (`terminal_soc_target`), so the
`MarketConfig` hash is unchanged. The mechanism is recorded in the B1 manifest
and in the trainer checkpoint's reward block.

The B1 cache key is the manifest's `mechanism_hash`, not the bare market config
hash. The old rolling B1 and the new one share the same market config hash and
initial SOC; without the rule in the key, a trainer could silently reuse the
no-pin benchmark. `load_b1_cache` refuses a manifest that lacks the mechanism
hash.

## Why this rule

- The policy arm and B1 share the same terminal inventory, so neither can
  manufacture a cash advantage by running the battery down.
- No terminal inventory valuation price is introduced; the comparison is on
  actual settlement plus a physical endpoint.
- The day-level metric remains the actual settled cash difference.

The earlier rolling B1 did **not** pin terminal SOC. It is kept unchanged and
tagged `CASHFLOW_NO_TERMINAL`; it is a historical cash-flow reference, not the
benchmark for the new experiment.

## Feasibility audit

Source: `b1_terminal_neutral_manifest.json`,
`b1_terminal_neutral_results.csv`, `b1_terminal_neutral_audit.md`.

| item | result |
|---|---|
| days | 96 |
| feasible (24 clears, 0 fallbacks, end SOC within 1e-5) | **96** |
| infeasible | 0 |
| all clears OPTIMAL | yes |
| clearing fallbacks | 0 |
| LMP fallback cells | 0 |
| max per-unit terminal error | **0.0** |
| terminal SOC range, every ESS | exactly 0.5 |
| NEAR records found by `pair_id` | 80 / 80 |
| NEAR pair ids absent from FULL96 | none |

No IIS was needed. Had a day failed, the runner records the first failing
block and the SOC the units carried into it and does not relax the constraint
or drop the day.

The truthful fleet's terminal-neutral day payoff ranges −164.10 to 3,369.99
CNY/day, mean 165.39. This is the terminal-neutral B1, not the old rolling
B1 and not JN-Base.

## The registered economics

For each storage unit \(i\) and day \(d\):

\[
\Delta G_i(d)=G_i^{\pi}(d)-G_i^{B1}(d),
\qquad
\Delta G_{\mathrm{fleet}}(d)=\sum_i \Delta G_i(d).
\]

Both arms use the same initial SOC, load, price, network, storage parameters,
cycle cost, block structure and terminal rule. Only the quotes differ. The B1
arm is all four RL units at `(0,0)`; the 32 load/prosumer participants keep
their truthful declarations in both arms.

For each day the paper reports, per ESS:

- policy actual settlement \(G_i^{\pi}\);
- B1 actual settlement \(G_i^{B1}\);
- \(\Delta G_i\);
- charge and discharge energy;
- equivalent cycles;
- day-end SOC;
- network losses and voltage/line constraint outcomes.

The fleet quantities are the sum of the per-unit ones. No training reward and
no clearing objective is substituted for these quantities.

## Implementation facts

- `G_i` is `participant_payoff` summed over the 24 committed blocks. Cycle
  cost is inside that ledger once; it is not subtracted again in the reward.
- The B reward is `delta / fixed_scale` paid at block 23 only. The C2 mode adds
  the SOC-time potential shaping every block, divided by the same fixed scale.
- Terminal SOC is neutralized by the clearing constraint, not by a money
  adjustment. `PayoffBreakdown` carries no terminal term.
- The B1 day record carries per-unit `charge_mwh`, `discharge_mwh`,
  `cycles`, `losses_mwh`, `voltage_violation_cells` and
  `line_overload_cells`. Equivalent cycles are
  `(charge_mwh + discharge_mwh) / (2 * e_max)`, the definition already used by
  `arbitrage_lp`. The policy trainer records the same quantities in its
  per-day checks.

## Risk carried forward

The terminal constraint can make a **policy** day infeasible even though the
truthful B1 day is feasible: the policy can leave the battery outside the reach
of the last four periods. The trainer treats a failed clearing as fatal for
that seed and keeps the diagnostic checkpoint; it does not relax the target,
drop the day, or substitute an approximate B1.
