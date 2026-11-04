# Pilot training-date exposure (V2-CY-JN-2025)

## Answer

Twenty episodes means **twenty distinct training days per run**; no day is drawn twice (maximum count 1). The training pool is therefore larger than what the run actually used, and must not be described as the model's sample size.

| run | pool | distinct visited | max count | unvisited | FULL96-only visited | common visited | seasons visited |
|---|---:|---:|---:|---:|---:|---:|---|
| NEAR seed 11 | 53 | 20 | 1 | 33 | 0 | 20 | autumn 8, summer 7, winter 5 |
| NEAR seed 22 | 53 | 20 | 1 | 33 | 0 | 20 | autumn 8, summer 4, winter 8 |
| FULL96 seed 11 | 69 | 20 | 1 | 49 | 3 | 17 | autumn 9, summer 7, winter 4 |
| FULL96 seed 22 | 69 | 20 | 1 | 49 | 5 | 15 | autumn 10, summer 5, winter 5 |

The 16 FULL96-only days were visited 3 times at seed 11, 5 times at seed 22. A FULL96 run therefore spent that many of its 20 episodes on the extra days and saw fewer common days than the NEAR run. With the same 20-episode budget, the two pools differ in composition **and** in how much of the common pool each sees.

## What this does and does not mean

- It is not evidence that the extra days are useless; they were visited, just rarely, and never repeated.
- It is evidence that "69 training days" overstates what the run learned from: each run saw 20 days once.
- Any comparison of pools at this budget carries both the composition difference and the reduced common-day exposure.
