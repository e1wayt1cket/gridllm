# The 16 FULL96-only long pairs (V2-CY-JN-2025)

These are the pairs NEAR excludes: `month_day_gap_days > 14`. They are **training-only** in FULL96 and must not enter the common test set or any NEAR result.

## Coverage

- 16 pairs: autumn 13, summer 3.
- Months covered by their price dates: 2025-07, 2025-09, 2025-10.

| load_date | price_date | group | gap (days) | price mean | price range | network peak MW | B1 fleet payoff | B0 served MWh |
|---|---|---|---:|---:|---|---:|---:|---:|
| 2022-11-27 | 2025-09-21 | autumn | 67 | 235.7 | 30-390 | 6.91 | 635.3 | 100.9 |
| 2022-11-30 | 2025-09-26 | autumn | 65 | 382.2 | 314-391 | 7.14 | -0.0 | 129.6 |
| 2022-11-29 | 2025-09-29 | autumn | 61 | 378.4 | 312-612 | 7.19 | 8.9 | 117.8 |
| 2022-11-26 | 2025-09-27 | autumn | 60 | 394.9 | 312-470 | 7.07 | -0.0 | 112.6 |
| 2022-11-28 | 2025-09-30 | autumn | 59 | 306.1 | 100-391 | 6.77 | 18.7 | 108.2 |
| 2022-11-25 | 2025-10-01 | autumn | 55 | 410.2 | 30-1500 | 7.16 | 3433.3 | 110.7 |
| 2022-11-20 | 2025-09-28 | autumn | 53 | 366.8 | 312-418 | 6.45 | -0.0 | 108.3 |
| 2022-11-24 | 2025-10-02 | autumn | 53 | 332.6 | 252-419 | 7.15 | -0.0 | 116.7 |
| 2022-11-23 | 2025-10-08 | autumn | 46 | 326.9 | 272-391 | 7.02 | -0.0 | 120.5 |
| 2022-08-13 | 2025-07-05 | summer | 39 | 180.2 | 30-330 | 6.30 | 63.2 | 112.6 |
| 2022-11-22 | 2025-10-15 | autumn | 38 | 382.9 | 310-420 | 7.03 | -0.0 | 120.8 |
| 2022-11-21 | 2025-10-16 | autumn | 36 | 387.8 | 300-448 | 6.94 | -0.0 | 113.8 |
| 2022-08-12 | 2025-07-11 | summer | 32 | 198.5 | 80-380 | 6.52 | 67.1 | 115.5 |
| 2022-11-18 | 2025-10-22 | autumn | 27 | 220.8 | 30-391 | 7.24 | 427.9 | 125.8 |
| 2022-11-17 | 2025-10-23 | autumn | 25 | 254.0 | 30-391 | 6.89 | 360.7 | 124.6 |
| 2022-07-30 | 2025-07-06 | summer | 24 | 184.8 | 80-250 | 6.43 | -0.0 | 105.5 |

## Distribution shift against the common 80

| statistic | common 80 | FULL96-only 16 |
|---|---:|---:|
| price daily mean, median | 237.8 | 329.7 |
| price daily max, median | 391.0 | 391.0 |
| network peak MW, median | 6.48 | 6.98 |
| B1 fleet payoff, median | 119.9 | 4.4 |
| B1 days with payoff > 50 CNY | 45 / 80 | 6 / 16 |

## Are they worth keeping as training samples?

They are not interchangeable with the common days. Their loads are late-November and August days, and their prices come from July, September and October, so they add load-level and price-level coverage the common 80 lack. Their price level is also higher: the median daily mean is 329.7 against 237.8. What they do not add is arbitrage signal. Their curves are flatter, so the truthful fleet earns a median 4.4 CNY/day against 119.9 on the common days, and only 6 of 16 days clear 50 CNY against 45 of 80. One day (2022-11-25 to 2025-10-01) carries a 30-1500 range and 3433 CNY and pulls the mean up; it is one day, not a pattern.

**Recommendation:** keep them in FULL96 training only, where they may teach a policy what a higher-price, lower-spread day looks like, and never let them define a test. Whether they help is a question for the FULL96-vs-NEAR comparison on the common test set; their own payoffs are not evidence for or against the filtering rule, and because they are far more often idle than the common days they cannot be used to argue that more training days are better.
