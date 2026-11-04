# B1 Truthful equivalence audit (V2-CY-JN-2025)

**Result: 96/96 days pass. B1 is JN-Base, not an independent repeated experiment.**

B1 reuses the JN-Base clears; no second solve was run. Equivalence is established by the code path, the construction and the stored results, not by re-solving a deterministic engine and comparing it with itself.

## The mapping

`clear_market(action_params={})` sends every participant through `dispatch_core.quote_arrays`. A storage agent with no declaration falls back to `ACTION_STORAGE_QUOTE_V2.truthful`, `(charge_price, discharge_price) = (0, 0)`; a partial declaration raises instead of being completed. A load or prosumer with no declaration uses `(bid_mult, offer_adder) = (1, 0)`, so its bid array is its own `bid_value` and its offer array its own `offer_cost`. This was checked for every agent on every one of the 96 days: the largest storage quote deviation is 0.00e+00 CNY/MWh and the largest prosumer deviation is 0.00e+00.

## Checks

| check | pass |
|---|---:|
| agents ok | 96/96 |
| storage specs ok | 96/96 |
| storage quotes truthful | 96/96 |
| prosumer quotes truthful | 96/96 |
| load digest ok | 96/96 |
| price digest ok | 96/96 |
| result unique | 96/96 |
| result flags ok | 96/96 |
| inputs ok | 96/96 |
| voltage ok | 96/96 |
| line ok | 96/96 |
| p balance ok | 96/96 |
| q balance ok | 96/96 |
| terminal soc ok | 96/96 |
| exclusive ok | 96/96 |
| soc band ok | 96/96 |
| lmp ok | 96/96 |
| pcc ok | 96/96 |
| no error | 96/96 |

### Configuration and inputs

| check | result |
|---|---|
| opf is socp | PASS |
| cycle cost is 100 | PASS |
| terminal soc pinned | PASS |
| impedance scale is 0 5 | PASS |
| reactive fraction is 2 | PASS |
| band is 0 93 1 05 | PASS |
| reverse limit is 2 5 | PASS |
| pairing hash matches manifest | PASS |
| bus loads hash matches manifest | PASS |
| price curves hash matches manifest | PASS |

### Numeric margins (maximum over 96 days)

| quantity | max | tolerance |
|---|---:|---:|
| active-balance residual (MW) | 8.36e-07 | 1e-05 |
| reactive-balance residual (MVAr) | 4.56e-07 | 1e-05 |
| terminal SOC error | 0.00e+00 | 1e-09 |
| charge/discharge product (MW^2) | 1.37e-08 | 1e-06 |
| slack-LMP deviation (CNY/MWh) | 5.83e-03 | 0.01 |

## Hashes

- `results.csv`: `769d1467b64b7ec76f642860aa3d90fd581daf68aa51abb3362d8f3b212a5eae`
- `config.json`: `5237cdc753d59f870a3cf715c05295ee84794f644bc16ab57f90927065702737`
- run manifest inputs match the files on disk: yes
- per-day audit: `data/processed/v2_cy_jn_2025/benchmark/b1_equivalence.csv`

## What this does not show

It does not show that a fresh B1 run would reproduce these numbers; the engine is deterministic, so that would be a weaker check than the one above. It also does not make B1 an independent sample: B1 and JN-Base are one clearing described twice, and a benchmark table must not count them as two arms of evidence.
