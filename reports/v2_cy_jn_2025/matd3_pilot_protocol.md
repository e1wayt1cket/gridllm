# MATD3 pilot protocol (V2-CY-JN-2025)

This is a **controlled small pilot**, not formal training and not a
performance claim. It asks whether MATD3 runs stably in this environment, with
the audited reward, and whether the extra 16 training days change anything on
the common test set. The design below was frozen before the first episode; the
seeds and budget are in `data/processed/v2_cy_jn_2025/pilot/pilot_config.json`.

## Pre-training gates (completed before any episode)

| gate | result | report |
|---|---|---|
| Same-day price availability at bid time | **not proven**; no policy observation reads it | `decision_information_timing_audit.md` |
| OBS_V6 price features causal | pass: no feature reads the wholesale curve or a counterfactual result | same |
| Truthful baseline is state-matched | pass, 9/9 argument checks on one instrumented block | `difference_reward_state_matching_audit.md` |
| Cycle cost charged once in settlement | pass; clearing objective and payoff are different ledgers | `reward_accounting_audit.md` |
| Deviation penalty only in the actor loss | pass | same |

The timing gate means the same-day actual price is **not** treated as a
decision-time input for the policy. It is the clearing's exogenous PCC price,
which is the market mechanism. B2A/B2B did quote from the same-day curve and
carry a look-ahead caveat; they are retained as historical baselines only.

## Datasets

| run | training pool | days |
|---|---|---:|
| `MATD3-NEAR` | common training split | 53 |
| `MATD3-FULL96` | common training split + FULL96-only long pairs | 69 |

The two runs differ in **nothing else**. The extra 16 days are inserted by the
same seeded permutation; they are not extra epochs, extra updates or extra
episodes.

## Budget and seeds

| item | value |
|---|---|
| seeds | 11, 22 |
| episodes per run | 20 |
| decision blocks per episode | 24 |
| environment steps per run | 480 |
| gradient updates per run | one per block after `start_steps` |
| `start_steps` | 48 |
| batch size | 64 |
| buffer capacity | 20000 |
| learning rate / gamma / tau | 3e-4 / 0.99 / 0.005 |
| policy delay | 2 |
| exploration noise | 0.2 → 0.05 over 2000 steps, clip 0.5 |
| deviation penalty | calibrated 0.10 |
| observation / action | OBS_V6 / `storage_quote_v2` |

The clock is **steps and updates**, not wall time, so the two pools cannot be
given different computational budgets without it being visible in the log.

## What is logged

Per episode: day, environment steps, updates, episode reward, raw profit,
local truthful baseline, critic and actor loss, Q means and gap, deviation
penalty value, action means, SOC range, clears, fallbacks, LMP fallbacks,
voltage and line violations, end-of-day SOC gap, wall time.

The reward, raw profit and local baseline are written separately. A reward is
not a payoff and is not reported as one; the evaluation table reports settled
payoffs.

## Evaluation

- Days: the 11 common test days and the 5 national-day days, never used in
  training or tuning.
- Rollout: deterministic policy (no noise) over the 24 blocks; the per-block
  quotes are written into a 96-period schedule.
- Settlement: **one full-day clear** with that schedule, `horizon_type="full_day"`,
  terminal SOC pinned, the same config as B1. This makes the number comparable
  with B1/B2A, whose days were also cleared whole.
- Reported per day and per unit: settled payoff, B1 payoff, B2A payoff,
  differences, charge/discharge energy, equivalent cycles, terminal SOC error,
  physical violations, LMP fallbacks, and the rollout's reward components.
- The block-level rolling reward is **not** the settlement; both are reported
  and labelled.

## Comparisons

- `MATD3-FULL96` vs `MATD3-NEAR`: per-day paired on the common test, same seed,
  same budget. The question is whether the extra days change the paired
  difference, not whether either mean is larger in isolation.
- `MATD3` vs B1: B1 is the day-level truthful benchmark.
- `MATD3` vs B2A: B2A is exploratory and carries the same-day-price caveat.

Eleven days cannot support a significance claim. The tables are descriptive.

## Pass criteria

The pilot succeeds if the logs show numeric stability, no clearing failures or
physical violations, no information leak, a reward that decomposes as audited,
and policies whose actions are explicable. **Beating B1 is not required**, and
a higher payoff in a few days is not evidence of economic advantage unless it
survives the paired day table and is not produced by a corner quote or a
single outlier day.
