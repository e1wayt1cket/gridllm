# V2 price-source audit

## Frozen input

- Manifest: `data\processed\v2\v2_scenario_library_manifest.json`
- Version: **v2-frozen-dfc9d60cc222**
- Status: **frozen**
- Manifest sha256: `ebed1801c43c32a9a4aeb2758171778b8481491f948ea73a1d179bddd251730f`
- Load workbook sha256: `63b826d9206f239f31f977bde281253ee5d61e94b02f3d8c23897478b52aebbb`
- Price workbook sha256: `5fc8f9a8461b680fd38a6d2fefb3d2ff99f239c8bd13a6c77df27de3bb9612f3`

The frozen manifest was read only. The six user sets, the 31 representative days, the load curves and the `line50_q2.0` support assumption were not changed.

## Existing price artefacts

| source | sheet | date / range | periods | interval | labels | units | kind | source kind | traceable | external confirmation | min / max / mean CNY/MWh |
|---|---|---|---:|---:|---|---|---|---|---|---|---|
| dianyicha:261009 | 261009 | 2026-10-09 (raw 26/10/09) | 96 | 15 | interval_end | CNY/MWh | single_day | real_market_data | True | False | 225.9 / 445.0 / 386.8302083333333 |
| dianyicha:7days | 7days | undated window mean | 96 | 15 | interval_end | CNY/MWh | window_mean | real_market_data | True | True | 257.5 / 477.871 / 395.54803125000007 |
| dianyicha:30days | 30days | undated window mean | 96 | 15 | interval_end | CNY/MWh | window_mean | real_market_data | True | True | 333.077 / 450.857 / 407.82115625000006 |
| dianyicha:halfyear | halfyear | undated window mean | 96 | 15 | interval_end | CNY/MWh | window_mean | real_market_data | True | True | 284.927 / 393.855 / 346.3266354166667 |
| single_day_diagnostic:price_96 |  | 2026-10-09 | 96 | 15 | interval_end | CNY/MWh | single_day | derived_from_real | True | False | 225.9 / 445.0 / 386.8302083333333 |
| synthetic_v1 |  | 40 day_ids; no calendar dates | 96 | 15 | interval_start | CNY/MWh | synthetic | synthetic | True | False | 131.9641038121654 / 896.9291514004817 / 238.13164929693218 |
| price_ref_calibration |  | none | 0 | 0 | unknown | CNY/MWh | derived_reference | derived | True | False | — |
| exports:da_price |  | none (scenario output) | 96 | 15 | interval_start | CNY/MWh | scenario_output | simulation_output | True | False | 219.4749 / 613.7432 / 421.33438541666663 |

## Source-by-source notes

### `261009` (single day, 2026-10-09)

- The sheet's own header reads `日前均值（CNY/MWh）`; the workbook does not state a timezone, tax status or spatial weighting.
- The single-day reading is a **project ruling**, not an external confirmation. `externally_confirmed` is false.
- The curve is a **spatial mean**; no bus, zone or settlement point is represented.
- It is the only real single-day curve in the project. Its actual date is 2026-10-09, which does not overlap any frozen load date.

### `7days`, `30days`, `halfyear`

- These are trailing window means, not daily observations. They have no actual date and cannot match a load date. They are recorded here as rejected, not used.

### `single_day_diagnostic/price_96.csv`

- An extracted copy of the `261009` sheet. It is traceable to the same workbook and has the same 2026-10-09 date; it adds no independent coverage.

### `synthetic_v1` and `price_ref_calibration.json`

- `synthetic_v1` is generated, not measured market data. It has 40 day ids and no calendar dates.
- `price_ref_calibration.json` is one derived scalar, not a daily curve.

### `exports/**/da_price.csv`

- Scenario outputs from earlier runs. They are not raw market observations and carry no calendar date.

## Data exists versus legally matchable

- **Data exists** for one real single day (2026-10-09) and three real window means.
- **Legally matchable to a frozen load day**: **none**. The single day has the wrong actual date; the window means are not days; the copy is the same single day; the synthetic library is not real; the exports are not observations.
- A same-month-day cross-year pair is not a same-date price and is not counted as a match.

## External source assessment

The following is a **source survey**, not a data download. No subscription was purchased and no account was created.

| candidate source | authority / access | what it could provide | confirmed for 2022 15-min daily data? | gap |
|---|---|---|---|---|
| 江苏电力交易中心 / Jiangsu spot-market disclosure platform | official market operator; registration/authorization likely | day-ahead clearing prices, 96 x 15-min periods | **not confirmed**; no public bulk download found in the survey | request an authorized export with market/product/unit/tax/interval metadata |
| 电易电查 (the existing workbook's source) | commercial data vendor; subscription/authorization | historical daily curves | **not confirmed** beyond the single 2026-10-09 sheet | request a dated 2022 export and its field definitions |
| Mysteel / 北极星 / 我的煤炭 and similar news aggregators | secondary; public snippets or paid reports | single-day snapshots or monthly/medium-term averages | **no**; not a 15-min daily series in the public results | do not use as a daily price source |
| academic / repository copies | provenance varies | possibly a shared Jiangsu spot dataset | **not verified** | require source, license, date and interval documentation before use |

Public search on 2026-10-21 found secondary descriptions of Jiangsu's spot market design (96 x 15-min day-ahead periods, 00:15–24:00, zonal pricing) and single-day or monthly price news, but **not** a confirmed public 2022 96-point daily series. The official market operator remains the authoritative route. Secondary sources are listed only to show what was checked, not as admissible data.

### Survey references (secondary, not data sources)

- 2022 Jiangsu electricity market summary: http://news.mcar.com.cn/kx/20230103/103658.html
- Jiangsu day-ahead market snapshot (Mysteel): https://www.mysteel.com/report/2609/B021AAF51145C26D.html
- Jiangsu electricity market news index (我的煤炭): https://www.mycoal.cn/news/search-htm-kw-%E6%B1%9F%E8%8B%8F%E7%94%B5%E5%8A%9B%E4%BA%A4%E6%98%93%E4%B8%AD%E5%BF%83.html
- Jiangsu electricity market news index (北极星): https://www.bjx.com.cn/search?kw=%E6%B1%9F%E8%8B%8F%E7%94%B5%E5%8A%9B%E5%B8%82%E5%9C%BA&type=5

These are secondary/contextual pages. None is recorded as the authoritative price series for this study.

## Specific data request

At minimum, provide real 15-minute day-ahead price curves for the frozen representative dates (31 days) or, preferably, all 123 candidate dates. Each curve must state:

- actual trading date and market/region (Jiangsu day-ahead spot);
- product and settlement/pricing scope (zonal vs nodal, if relevant);
- 96 periods per day, 15-minute granularity, interval convention;
- timezone and daylight-saving handling;
- CNY/MWh and tax status (included/excluded/unknown);
- source, export path, file sha256 and field definitions;
- no window means, no synthetic values, no copied or adjacent dates.

## Command

`src/audit_v2_price_matching.py`
