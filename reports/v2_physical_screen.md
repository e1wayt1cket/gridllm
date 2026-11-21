# V2 seasonal physical screen

## Verdict

This is a **forced-serve power-flow screen**, not a market clearing. There is no bid, no storage, no shedding decision and no objective. It asks whether the selected users' measured load, on the proposed replace buses with native load elsewhere, stays inside the network's voltage band and line ratings.

**No group passes the voltage band on any tested representative day.** The binding constraint is voltage for every group; two autumn days also overload a line. The screen therefore does not establish a physically operable full-load V2 scenario under the current IEEE33 network.

`peak load MW` below is the **whole network** load: the selected users' measured load on replace buses plus the benchmark's native load on preserve buses. It is not the subset-only aggregate.

| group | dates tested | converged | voltage violations | line overloads | worst min V | worst max V | max line loading | binding |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| spring | 7 | 7 | 74 | 0 | 0.9112 | 1.0091 | 0.344 | ['voltage'] |
| summer | 6 | 6 | 512 | 0 | 0.7403 | 1.0021 | 0.884 | ['voltage'] |
| autumn | 6 | 6 | 552 | 3 | 0.6658 | 1.0021 | 1.109 | ['voltage'] |
| winter | 6 | 6 | 513 | 0 | 0.6945 | 1.0011 | 0.841 | ['voltage'] |
| spring_festival_2022 | 3 | 3 | 193 | 0 | 0.8820 | 1.0076 | 0.331 | ['voltage'] |
| national_day_2022 | 3 | 3 | 226 | 0 | 0.7620 | 1.0032 | 0.816 | ['voltage'] |

The closest group is **spring** (worst min V 0.9112); the others are further below the band. A uniform small voltage support is therefore unlikely to make every group feasible.

## Per-date results

| group | date | peak load MW | converged | min V | max V | voltage periods | line overload periods | max line | binding |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| spring | 2022-04-29 | 3.522 | yes | 0.9136127522783984 | 1.0074173452676771 | 16 | 0 | 0.2777118476181895 | voltage |
| spring | 2022-04-30 | 3.516 | yes | 0.9208943507007606 | 1.007587240943864 | 9 | 0 | 0.27396291657939237 | voltage |
| spring | 2022-05-21 | 3.791 | yes | 0.9173838160720279 | 1.006081563644997 | 9 | 0 | 0.30940731876649336 | voltage |
| spring | 2022-05-23 | 3.989 | yes | 0.911239417009928 | 1.0090862505980693 | 11 | 0 | 0.3443670852071742 | voltage |
| spring | 2022-05-28 | 3.927 | yes | 0.9146307070243678 | 1.0033086221216891 | 8 | 0 | 0.3353179157073493 | voltage |
| spring | 2022-05-29 | 3.880 | yes | 0.9158998570562209 | 1.0054144492693116 | 7 | 0 | 0.32152224218976144 | voltage |
| spring | 2022-05-30 | 3.621 | yes | 0.9177307014021122 | 1.0066133068478622 | 14 | 0 | 0.29010448489887175 | voltage |
| summer | 2022-07-22 | 6.458 | yes | 0.7536953501405136 | 1.0018328171034745 | 84 | 0 | 0.8352323078725619 | voltage |
| summer | 2022-07-23 | 6.259 | yes | 0.7609998089704367 | 1.0019411130510514 | 86 | 0 | 0.8158563857028304 | voltage |
| summer | 2022-08-14 | 5.816 | yes | 0.7965442829488224 | 1.0020388044826485 | 83 | 0 | 0.650319766330673 | voltage |
| summer | 2022-08-24 | 6.527 | yes | 0.7550938520640388 | 1.0020767307959202 | 85 | 0 | 0.8353351793799492 | voltage |
| summer | 2022-08-25 | 5.576 | yes | 0.7935037600570558 | 1.0020223474378558 | 85 | 0 | 0.6655084344269321 | voltage |
| summer | 2022-08-27 | 6.506 | yes | 0.7403484993524335 | 1.002080219457892 | 89 | 0 | 0.884202449059106 | voltage |
| autumn | 2022-10-15 | 6.822 | yes | 0.6958024723396958 | 1.0020651199876471 | 95 | 0 | 0.9885735131005942 | voltage |
| autumn | 2022-11-10 | 6.648 | yes | 0.6696533715075157 | 1.0015513680492514 | 93 | 2 | 1.0412534594434526 | voltage |
| autumn | 2022-11-11 | 6.457 | yes | 0.74150092329316 | 1.0019577377705529 | 94 | 0 | 0.8535394151552835 | voltage |
| autumn | 2022-11-13 | 6.484 | yes | 0.7312789357675613 | 1.0021378732704593 | 90 | 0 | 0.8884479744777368 | voltage |
| autumn | 2022-11-20 | 6.451 | yes | 0.7316967536032619 | 1.0021084491570773 | 95 | 0 | 0.8906664015075958 | voltage |
| autumn | 2022-11-25 | 7.164 | yes | 0.6658206264507899 | 1.0016456294170368 | 85 | 1 | 1.1085273800529134 | voltage |
| winter | 2022-12-10 | 6.159 | yes | 0.7495215196255577 | 1.0011410673615784 | 89 | 0 | 0.690839965309384 | voltage |
| winter | 2022-12-11 | 5.794 | yes | 0.7655778318342752 | 1.000978027928578 | 94 | 0 | 0.6389901920196316 | voltage |
| winter | 2022-12-24 | 5.693 | yes | 0.7564882528699651 | 1.0010841417976035 | 79 | 0 | 0.6710170696795591 | voltage |
| winter | 2022-12-27 | 6.354 | yes | 0.698945168800675 | 1.0003817990695345 | 81 | 0 | 0.8294222437021079 | voltage |
| winter | 2022-12-28 | 6.479 | yes | 0.6945240090799232 | 1.000002813940054 | 87 | 0 | 0.8406549558755736 | voltage |
| winter | 2022-12-29 | 6.297 | yes | 0.7086752442043589 | 1.0004193145805271 | 83 | 0 | 0.80449215603408 | voltage |
| spring_festival_2022 | 2022-01-31 | 3.097 | yes | 0.892603540742067 | 1.006283478657115 | 62 | 0 | 0.2542756373852928 | voltage |
| spring_festival_2022 | 2022-02-01 | 3.989 | yes | 0.885585692933228 | 1.007648064553549 | 64 | 0 | 0.33109971747495315 | voltage |
| spring_festival_2022 | 2022-02-02 | 3.941 | yes | 0.8820253605940132 | 1.0063681054552 | 67 | 0 | 0.3206946938756671 | voltage |
| national_day_2022 | 2022-10-03 | 5.782 | yes | 0.7877231595227288 | 1.002705470833172 | 73 | 0 | 0.7249557551679464 | voltage |
| national_day_2022 | 2022-10-04 | 5.389 | yes | 0.8147055180979441 | 1.0031811195510174 | 73 | 0 | 0.6433885966728304 | voltage |
| national_day_2022 | 2022-10-07 | 6.154 | yes | 0.7619520872232697 | 1.0025766268465426 | 80 | 0 | 0.8161133987273097 | voltage |

## Limits

- The screen uses the V1 replace/preserve composition and the scenario's renewable placement; it is not a new market scenario.
- It does not model storage, bids, congestion pricing, shedding or the two-settlement market. A passing screen is a necessary condition, not a market result.
- It uses the scenario's line-capacity multiplier and power factor; the exact V2 scenario declaration is not frozen yet.
- The price blocker remains: no real dated prices exist for these days, so no training or priced evaluation is possible.

## Artifacts

| path | sha256 |
|---|---|
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_seasonal_subsets.csv` | `3ea3b009289194aa3b5d8e7b0afdbb148d0da5086423194cfbd1ad6aaec014ca` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2_seasonal_candidates.csv` | `05b63da48f2585044635af745a0c18e76565001a78fead03b122cd285891a369` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_user_summary.csv` | `8464c14a92e1d0ff777a9b0e7b5ef5b6c2bc962cdbe1b9c40c9f6ae52d0b92de` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_user_day_curves.npz` | `8ff8cbc00618189d001565068bf8a95c9b9975960baa0ce6343ad508075be472` |
| `C:\Projects\00_WorkSpace\01_Projects\MakerB\data\processed\v2\v2_physical_screen.csv` | `0717996a44cea56e3976fca700d4cf375e1879d017bc108f8c9dba9b29d4fbce` |

## Command

`src/screen_v2_physical.py`
