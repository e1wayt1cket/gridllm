# Superseded: pre-S1 load artefacts

These files were produced before the S1 unit correction. They record the old
reading of a slot value as "kWh per 15-minute slot", which made the capacity
check compute `max|slot| * 综合倍率 * 60/15` and so overstate every implied peak
four fold. Every figure below is superseded and must not be cited as a current
result.

| | |
|---|---|
| screening rule version | `f647fa60d88c3b29` (superseded; current is in the live manifest) |
| dataset version | `industry-blocks-B/310fbfb3ccde` |
| gated meters | 87 |
| meters contradicting their rating | 113 |
| qualified blocks | 39,365 |
| contributing meters | 84 |
| users | 36 |
| mapping | `R3_meter_spread`, 72 meters over 29 buses |
| fully covered days | 312 |

The raw workbook was never modified, and these files are the record of what the
old rule produced from it. See `reports/s0_code_and_data_baseline.md` for the
baseline these belong to and `reports/industry_load_rebuild.md` for the
recomputed figures.
