# B1 daily equivalence audit (V2-CY-JN-2025)

**Answer: JN-Base is not equivalent to the 24-block day-level B1.** The two
execute different mechanisms, produce different SOC paths and different
returns. The rolling B1 is registered as the day benchmark; JN-Base is kept as
a separate full-day benchmark and the two are never mixed.

## The two mechanisms

| property | JN-Base | 24-block B1 |
|---|---|---|
| clearing | one 96-period joint clear | 24 windows of 16 periods, each committing 4 |
| terminal SOC | pinned to the start | not pinned |
| settlement | one full-day clearing; LMPs from its duals | sum of per-block `participant_payoff` on the committed periods |
| SOC update | one trajectory from one solve | 24 sequential updates |
| reuse status | unchanged historical result | **new** registered benchmark |

Equal dates, inputs and network configuration are necessary but not sufficient:
the execution interface differs, and the task requires the B1 arm to share the
policy arm's interface.

## Representative evidence

From `b1_jnbase_comparison.csv` (six days; rolling arm vs JN-Base):

| pair_id | rolling | JN-Base | difference | rolling end SOC (4 units) |
|---|---:|---:|---:|---|
| 2022-10-04\|2025-10-04 | 554.79 | 349.88 | +204.91 | 0.432 |
| 2022-07-22\|2025-07-15 | 106.91 | 0.00 | +106.91 | 0.432 |
| 2022-07-23\|2025-07-13 | 31.01 | 0.00 | +31.01 | 0.432 |
| 2022-10-03\|2025-10-03 | 0.00 | 0.00 | 0.00 | 0.500 |
| 2022-10-11\|2025-10-10 | 0.00 | 0.00 | 0.00 | 0.500 |
| 2022-10-12\|2025-10-09 | 0.00 | 0.00 | 0.00 | 0.500 |

The differences are not rounding: the rolling arm may end the day with stored
energy that the pinned full-day arm must return, and it can re-plan every four
periods against a 16-period look-ahead.

## Full registration (96 days)

`b1_daily_results.csv` — one row per pair, all four units:

- 96 rows, 96 unique pair_ids; every day 24 blocks;
- 96/96 OPTIMAL, 0 clearing fallbacks, 0 LMP fallback cells;
- fleet \(G\) mean 182.6 CNY/day, median 0 (many days carry no spread), max
  3844.1;
- **terminal SOC is not 0.5 on 71 days**; the observed end-SOC range is
  0.10–0.90, i.e. the rolling arm can end at either bound.

The manifest (`b1_daily_manifest.json`) records the mechanism, `not_jn_base:
true`, `soc0 = 0.5`, the config hash, input hashes (pairing, bus loads, price
curves) and the output hashes.

## NEAR mapping

All 80 NEAR pair_ids are present in the 96-row table (80/80 checked by test).
The B1 value depends only on the day, the initial SOC, the config and the data
hashes, not on the training pool, so the NEAR rows are the same records, keyed
by `pair_id`; they are not a separate solve and are not averaged.

## Consequences

1. **The B reward uses the rolling B1**, never JN-Base.
2. The rolling arm does not pin terminal SOC, so the day differential includes
   any difference in ending SOC between the two arms. That is a property of the
   shared execution interface, not a hidden adjustment. A future variant that
   pins terminal SOC must change both arms together.
3. JN-Base remains a valid **one-shot** benchmark for the market result it is,
   and is not the counterfactual of the rolling policy arm.

## Limits

- Six representative days were compared in detail; the full 96 days were then
  computed with the same interface. The mechanism difference is categorical and
  does not depend on the sample.
- The rolling B1 uses no differential baseline (there is none for the truthful
  arm), so the run is a single clear per block.
