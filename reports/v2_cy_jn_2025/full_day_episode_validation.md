# Full 24-block episode validation

One fixed training day through the B+C2 + truthful-gate branch. This is an engineering check, not a policy result.

- pair: `2022-07-22|2025-07-15`
- experiment: `SMOKE_B_C2_GATE` (B_C2)
- blocks cleared: 24/24
- fallbacks: 0, LMP fallback cells: 0
- all clears OPTIMAL: True
- voltage violation cells: 0
- line overload cells: 0
- terminal SOC: {'ESS9': 0.5, 'ESS14': 0.5, 'ESS17': 0.5, 'ESS27': 0.5}
- quotes legal: True
- every closed gate exactly zero: True
- charge / discharge / equivalent cycles: {'ESS9': 0.11815599864780206, 'ESS14': 0.11814793208387601, 'ESS17': 0.11815687046342147, 'ESS27': 4.581703324173823e-07} / {'ESS9': 0.10000723865212768, 'ESS14': 0.10000041120980395, 'ESS17': 0.10000797655651476, 'ESS27': 3.858844177782221e-07} / {'ESS9': 0.06817601165622804, 'ESS14': 0.06817135727927498, 'ESS17': 0.06817651469373007, 'ESS27': 2.6376710943612634e-07}
- network losses: 2.5319 MWh

## Reward accounting

- `delta = G_policy - G_B1`, max error 0.000e+00
- B paid only at block 23: True
- B at block 23 equals delta: True
- C2 shaping terms present: 96 of 96
- C2 telescope error: 3.390e-15
- buffer reward equals (B+C2)/scale, max error 5.531e-09
- fixed reward scale: 1000.0 (no running standardisation)

The B1 arm is the registered terminal-neutral rolling B1: same pair, same initial SOC, same 24-block interface, same exogenous day. The manifest config hash and the three input hashes match.

## Gate behaviour

- gate open frequency: {'ESS9.charge': 0.7083333333333334, 'ESS9.discharge': 0.5, 'ESS14.charge': 0.4166666666666667, 'ESS14.discharge': 0.375, 'ESS17.charge': 0.375, 'ESS17.discharge': 0.4583333333333333, 'ESS27.charge': 0.625, 'ESS27.discharge': 0.625}
- gate mean probability: {'ESS9.charge': 0.5344352175792059, 'ESS9.discharge': 0.4748254232108593, 'ESS14.charge': 0.4931062137087186, 'ESS14.discharge': 0.47027332459886867, 'ESS17.charge': 0.4981752708554268, 'ESS17.discharge': 0.4916407950222492, 'ESS27.charge': 0.5090438177188238, 'ESS27.discharge': 0.4687254863480727}
- observed fleet modes: ['102', '107', '109', '136', '149', '171', '182', '193', '194', '20', '201', '202', '209', '217', '231', '235', '241', '243', '247', '4', '5', '51', '75']
- saturated upper / lower (open legs only): 0 / 0

A closed gate is an exact zero by construction and is reported through the open frequency, not as a saturated quote.
