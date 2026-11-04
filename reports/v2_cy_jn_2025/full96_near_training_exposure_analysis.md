# FULL96 vs NEAR: exposure and test behaviour (V2-CY-JN-2025)

## What actually entered training

| run | pool | distinct days | FULL96-only visited | common visited |
|---|---:|---:|---:|---:|
| NEAR/11 | 53 | 20 | 0 | 20 |
| NEAR/22 | 53 | 20 | 0 | 20 |
| FULL96/11 | 69 | 20 | 3 | 17 |
| FULL96/22 | 69 | 20 | 5 | 15 |

Each run sees 20 distinct days once; 20 episodes do **not** cover the 53- or 69-day pool. The FULL96 runs spend 3 or 5 of their 20 episodes on the extra days, so they see fewer common days than the NEAR runs at the same budget. The comparison therefore changes two things at once: the pool's composition and the common-day exposure.

## Test results and seed spread

| dataset | seed 11 | seed 22 | mean | seed spread |
|---|---:|---:|---:|---:|
| NEAR | 106.74 | 42.86 | 74.80 | 63.88 |
| FULL96 | 79.42 | 28.36 | 53.89 | 51.06 |

The seed spread within a dataset (64 CNY/day for NEAR, 51 for FULL96) is larger than the FULL96 − NEAR difference (−20.9 CNY/day). Two seeds cannot separate a pool effect from seed noise.

## The extra days' character

The 16 FULL96-only days have a higher median price level (329.7 vs 237.8 CNY/MWh) but flatter curves: truthful fleet payoff median 4.4 vs 119.9 CNY/day, and only 6 of 16 days above 50 CNY. They add price- and load-level coverage, not arbitrage signal.

## What can and cannot be concluded

- **Can:** at this fixed budget, FULL96 did not outperform NEAR on the common test set; every run was below B1.
- **Cannot:** that the 16 days are useless or harmful. They were visited only 3-5 times, the seed spread exceeds the dataset difference, and the pools also differ in common-day exposure.
- **Cannot:** that 69 days is a larger effective sample than 53 at this budget; both runs used 20 days once.

Separating distribution from exposure from seed noise needs more episodes per seed and more seeds, with the common-day exposure held equal; none of that is done here.
