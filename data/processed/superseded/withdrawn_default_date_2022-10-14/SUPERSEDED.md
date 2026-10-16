# Superseded: the 2022-10-14 audit ledger

This is the audit ledger `build_industry_node_loads.py` wrote for **2022-10-14**,
kept here because the ledger it came from is written to a single path and would
otherwise have been overwritten.

2022-10-14 was never chosen by a rule. It was picked by "the date with the most
complete qualified meters, earliest wins ties", which is a **coverage rank and a
monotone proxy for load**: a date with more reporting meters carries more load,
so the rule selected from the high end of the distribution and the day was then
described as representative. On bus 16 the date it picked ranked 694th of 730.
`reports/s3_date_selection_audit.md` withdrew it as a default; the builder then
required `--date` and designated nothing.

On 2026-10-15 a replacement rule was adopted — the **earliest** date, in
calendar order, on which every mapped bus is covered for all 96 periods — which
selects **2022-11-30**. The live ledger is that one. This file is the record of
what the withdrawn rule produced, kept so the audit of its bias stays
reproducible.

Nothing here was edited; both files are byte copies of what the builder wrote.
See `reports/single_day_diagnostic_case.md`.
