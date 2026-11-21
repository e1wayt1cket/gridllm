# FULL96 vs NEAR exposure experiment design (V2-CY-JN-2025)

Design only; no training is run. The sixth phase found that 20 episodes mean
20 distinct days once, and that the FULL96 runs spent 3–5 of them on the
extra 16 days, so the pilot's comparison changed **pool composition and
common-day exposure together**. Two protocols separate those questions.

## Protocol 1 — fixed computation budget

| item | value |
|---|---|
| question | at the **same** environment steps and gradient updates, does a larger day pool help? |
| pools | NEAR 53; FULL96 69 |
| episodes per seed | 20 |
| blocks / steps per episode | 24 / 24 |
| environment steps per run | 480 |
| gradient updates | one per block after `start_steps=48` → 417 |
| day sampling | seeded permutation of the pool; 20 distinct days, each once |
| season stratification | shuffle within season, then interleave seasons so every 3-episode window contains summer/autumn/winter |
| seeds | 3 (11, 22, 33) |
| common-day exposure | NEAR 20 of 53; FULL96 20 − (extra days drawn) of 53 |
| reporting | per-seed and seed-averaged paired day differences; exposure table mandatory |

This is the protocol the pilot approximated. It answers "same compute, bigger
pool" and deliberately allows the common exposure to differ; that difference is
part of the treatment and must be reported.

## Protocol 2 — fixed common-sample exposure

| item | value |
|---|---|
| question | **holding the 53 common days' exposure equal**, does adding the 16 extra days help? |
| exposure per day | k = 2 |
| NEAR episodes | 53 × 2 = 106 |
| FULL96 episodes | 53 × 2 + 16 × 2 = 138 |
| environment steps | 2544 / 3312 |
| gradient updates | 2496 / 3264 (48 warmup steps) |
| day sampling | for each round r = 1,2, a seeded permutation of the 53 common days, identical across datasets for the same seed; FULL96 appends a seeded permutation of the 16 extra days in each round |
| common-day exposure | exactly 2 per common day in both |
| budget ratio | 138/106 = 1.30 |
| reporting | separate from Protocol 1; never pooled |

Protocol 2 costs 30 % more updates. Its result is about the **data**, not about
a fixed compute envelope, and must be labelled as such.

## Shared rules

- **Seeds and pairing:** the same seed gives the same common-day order in both
  arms; the extra days are appended, not interleaved into the common order, so
  the common exposure is identical by construction.
- **Seasonal stratification:** shuffling happens within each season and the
  rounds interleave seasons; report per-season exposure.
- **No repeats inside a round:** a day appears at most once per round; across
  rounds it appears exactly k times.
- **Episode count is the budget:** steps = episodes × 24; updates are counted,
  not assumed. Both runs in a protocol must match exactly.
- **Evaluation:** the current 11-day test has been viewed and is no longer a
  blind holdout. Tuning uses train/validation only; any new performance claim
  needs a new holdout or a statement that the benchmark is no longer blind.
- **Seeds:** at least 3. The pilot's within-dataset seed spread was 51–64
  CNY/day, larger than the observed dataset difference (−20.9); two seeds
  cannot resolve the question.

## What each protocol can and cannot show

- P1: whether extra days help **when buying them with the same compute** —
  the practical training question. It cannot separate "extra days" from
  "fewer common days seen".
- P2: whether extra days help **when common exposure is held equal** — the
  data question. It cannot be compared with P1 as if the budgets matched.
- Neither can settle the question with 2 seeds; both need a fresh holdout for
  a final claim.
