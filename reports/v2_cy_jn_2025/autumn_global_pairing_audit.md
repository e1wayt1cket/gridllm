# Autumn global pairing audit (V2-CY-JN-2025)

**Verdict: keep the existing pairing.** The matcher is already a global one-to-one assignment. A min-max variant caps the worst distance at 36 days, but it does so by moving 23 pairs and raising the median from 3 to 12 days, with the count over 30 days going from 11 to 12. The long distances are a coverage limit, not a search failure.

## Is the matcher greedy?

No. `pair_v2_cy_jn_2025.pair_group` builds a cost matrix -- `gap` for a same-type candidate, `1e6 + gap` for the other type -- and solves it with `scipy.optimize.linear_sum_assignment`, one-to-one and globally. Re-solving it here reproduces the committed assignment on **47/47** autumn load days.

## Pool

- Autumn load days: 47
- Complete autumn price days: 57 (2025-09-01..11-30, complete only)
- Reserved and excluded: 2025-10-03..07 (national-day group)
- Same-month-day pairs held fixed in both schemes: 14

## Distance

| scheme | pairs | mean | median | max | >14 days | >30 days | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| current (min-sum) | 47 | 15.17 | 3 | 67 | 13 | 11 | 713 |
| min-max tie-break | 47 | 15.17 | 12 | 36 | 23 | 12 | 713 |

The bottleneck for a same-type one-to-one matching, with the same-month-day pairs fixed, is **36 days**.

The two schemes have the *same total distance*, so the min-max choice does not buy its lower maximum by adding distance overall -- it redistributes it. It is still not adopted: 23 pairs move, most from 1-5 days to 12-36 days, and the number of pairs over 30 days does not fall. Trading a concentrated tail for a wider mid-range is the degradation the instruction rules out.

## Pairs that would change

| load_date | current | gap | min-max | gap |
|---|---|---:|---|---:|
| 2022-10-11 | 2025-10-10 | 1 | 2025-09-29 | 12 |
| 2022-10-12 | 2025-10-09 | 3 | 2025-09-26 | 16 |
| 2022-10-22 | 2025-10-19 | 3 | 2025-09-21 | 31 |
| 2022-10-26 | 2025-10-24 | 2 | 2025-09-30 | 26 |
| 2022-10-29 | 2025-10-26 | 3 | 2025-09-28 | 31 |
| 2022-10-30 | 2025-10-25 | 5 | 2025-09-27 | 33 |
| 2022-11-01 | 2025-10-30 | 2 | 2025-10-02 | 30 |
| 2022-11-02 | 2025-10-29 | 4 | 2025-10-01 | 32 |
| 2022-11-08 | 2025-11-06 | 2 | 2025-10-08 | 31 |
| 2022-11-09 | 2025-11-05 | 4 | 2025-10-09 | 31 |
| 2022-11-15 | 2025-11-13 | 2 | 2025-10-10 | 36 |
| 2022-11-16 | 2025-11-12 | 4 | 2025-10-15 | 32 |
| 2022-11-19 | 2025-11-15 | 4 | 2025-10-19 | 31 |
| 2022-11-20 | 2025-09-28 | 53 | 2025-10-26 | 25 |
| 2022-11-22 | 2025-10-15 | 38 | 2025-11-13 | 9 |
| 2022-11-23 | 2025-10-08 | 46 | 2025-11-06 | 17 |
| 2022-11-24 | 2025-10-02 | 53 | 2025-10-30 | 25 |
| 2022-11-25 | 2025-10-01 | 55 | 2025-10-29 | 27 |
| 2022-11-26 | 2025-09-27 | 60 | 2025-10-25 | 32 |
| 2022-11-27 | 2025-09-21 | 67 | 2025-11-15 | 12 |
| 2022-11-28 | 2025-09-30 | 59 | 2025-10-24 | 35 |
| 2022-11-29 | 2025-09-29 | 61 | 2025-11-05 | 24 |
| 2022-11-30 | 2025-09-26 | 65 | 2025-11-12 | 18 |

## `2022-11-27`

Its nearest same-type price dates and who holds them now:

| price_date | gap (days) | current load day | that load's gap |
|---|---:|---|---:|
| 2025-11-15 | 12 | 2022-11-19 | 4 |
| 2025-11-09 | 18 | 2022-11-12 | 3 |
| 2025-11-08 | 19 | 2022-11-13 | 5 |
| 2025-11-02 | 25 | 2022-11-05 | 3 |
| 2025-11-01 | 26 | 2022-11-06 | 5 |
| 2025-10-26 | 32 | 2022-10-29 | 3 |

A closer date exists — 2025-11-15 is 12 days away — but it is held by a load day 4 days from it. Any one-to-one matching that gives it to 2022-11-27 must move that load day, and the global min-sum optimum already prices that trade-off. The min-max scheme does give 2022-11-27 the 12-day date and moves the other load to 2025-10-19 (31 days). The current 67-day pair is the cost of leaving every other pair near its own month-day.

## Why the distances are long

The source has no complete day from 2025-11-16 to 11-30 (15 dates) and none from 2025-08-31 to 09-14 (15 dates). The autumn pool is 57 dates for 47 loads, but the ones near late November are missing entirely, so the late-November loads compete for a short supply of October and early-November dates.

**Diagnostic, not a proposal:** if the five national-day dates (2025-10-03..07) were also available to autumn, the bottleneck would fall to 36 days. They stay reserved for the national-day group; the number only shows that scarcity, not proximity, drives the long pairs.

## Optimality evidence

Two solves agree on the optimum of the defined cost. The cost matrix is `gap` for a same-type candidate and `1e6 + gap` for the other type; a same-month-day pair is kept if it is of the same type, and 2025-10-03..07 are outside the pool.

| solve | method | objective | status |
|---|---|---:|---|
| production matcher | `scipy.optimize.linear_sum_assignment` (exact LAP) | 713.0000 | exact |
| certificate | Gurobi binary assignment, 2679 variables | 713.0000 | OPTIMAL |

The objectives differ by 0.00e+00; the MILP bound is 713.0000 and the MIP gap is 0.0e+00. Within the defined cost, no one-to-one assignment of the 47 autumn load days to the 57 candidate price dates is cheaper.

What the certificate does not say: the cost function is a modelling choice. It encodes same-type preference, distance, the fixed same-month-day pairs and the reserved national-day window. Optimality is relative to those choices; a different rule can have a different optimum. The committed pairing is optimal for this cost, not the uniquely correct pairing of the two calendars.

## Answers

1. **Greedy or global?** Global. The committed matcher is a one-to-one linear assignment, and re-solving it reproduces the committed pairing on 47/47 autumn days.
2. **The alternative's objective and constraints.** Minimise the *largest* distance, subject to same season and same weekday/weekend type, one distinct price date per load day, the national-day dates reserved, and every same-month-day pair held fixed; among matchings at that bottleneck (36 days) it then minimises total distance.
3. **Mean / median / max.** Committed 15.17 / 3 / 67 days; alternative 15.17 / 12 / 36 days.
4. **Over 14 and over 30 days.** Committed 13 and 11; alternative 23 and 12.
5. **Weekday/weekend mismatches.** Zero in both schemes. The autumn pool has enough same-type dates, so neither scheme is forced into a mismatch.
6. **`2022-11-27`.** Committed price date 2025-09-21 (67 days). A closer same-type date exists: 2025-11-15 is 12 days away, held by 2022-11-19 at 4 days. The alternative moves it to 2025-11-15 and moves 2022-11-19 to 2025-10-19 (31 days).
7. **Same result as the committed one?** Yes, and it is the *same optimum*, not a constraint that blocks change: two exact solves return the same objective, and the alternative is another point of the optimum set with the same total distance. So the committed pairing is optimal **for the defined cost** — it is not uniquely optimal, and this is not a claim that the cost is the only reasonable one.

- Per-day comparison: `data/processed/v2_cy_jn_2025/autumn_global_pairing.csv`

Command: `src/audit_autumn_pairing.py`
