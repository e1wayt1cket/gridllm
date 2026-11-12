# Model constraints

Applies to any change to the environment, action, reward, gate or trainer.

## Episode

- 24 blocks per day; 4 committed periods and 16 lookahead periods per block.
- Storage reference SOC starts and ends at 0.5, enforced only in the day-end
  window.
- The network is `line50_q2.0`; clearing engine is `socp`.
- Cycle cost is 100 CNY/MWh.

## Accounting

- Node LMP and the PCC exogenous price are different objects. Settlement uses
  the two-settlement ledger; a clearing objective is not a participant payoff.
- B1 is terminal-neutral and pays at block 23 only. Payoff differences must
  match the settlement ledger.
- The research KPIs are `pi_ai`, `pi_baseline`, `delta_pi`,
  `delta_pi_percent` and `p_delta_pi_positive`. A training reward is not a
  payoff.

## Frozen interfaces

- Observation `v6_12d`, 12 dimensions.
- Action `storage_quote_v3`, 2 dimensions per unit, four ESS in order
  ESS9, ESS14, ESS17, ESS27.
- Gate: 8 total, straight-through Bernoulli estimator.
- Quote map `linear_core_v1`.
- Actor 256-128-4; centralized twin critic 256-256.
- Reward mode `B_C2`, gamma 0.99, fixed scale 1000.

Any change to these requires a new candidate id and a new frozen manifest.
Changing them in place is not a model improvement; it is a protocol change.

## Enforcement

- The 15 pinned sources are hashed by the live code manifest.
- `formal_training_code_manifest.json` includes the audit and test files that
  pin the interfaces.
- Frozen-file edits are denied by `.claude/hooks/protect_frozen.py`.
