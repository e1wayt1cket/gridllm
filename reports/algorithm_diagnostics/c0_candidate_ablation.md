# C0 candidate 消融判定(B_C2 vs B,4 轮 pilot)

label: NON_FORMAL_DIAGNOSTIC_PILOT; merged verdict: **DESIGN_FLAW**

| seed | 对照 r4 | 实验 r4 | A 差 [CI] | C r1 差 | B 腿命中 | verdict |
|---|---|---|---|---|---|---|
| 11 | -166.0 | -40.4 | +125.6 [+0.5, +277.2] | +62.4 | {'charge': 'Y', 'discharge': 'Y'} | DESIGN_FLAW |
| 23 | -134.4 | -96.4 | +38.0 [-64.6, +159.0] | -37.9 | {'charge': 'Y', 'discharge': 'Y'} | DESIGN_FLAW |

逐轮曲线(mean_fleet_delta):

- seed11 对照: r1:-98.7 → r2:-148.9 → r3:-165.2 → r4:-166.0
- seed11 实验: r1:-36.3 → r2:-94.5 → r3:-35.7 → r4:-40.4
- seed23 对照: r1:-78.6 → r2:-105.4 → r3:-129.3 → r4:-134.4
- seed23 实验: r1:-116.6 → r2:-105.2 → r3:-102.2 → r4:-96.4