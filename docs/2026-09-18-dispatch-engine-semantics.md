# Dispatch engines: what each one means

Three engines clear a day. They do not implement the same market, and the
differences are load-bearing for every economic quantity this project reports.
This is the authoritative comparison; the module docstrings of
`dispatch_socp.py`, `dispatch_ldf.py` and `dispatch_dc.py` point here.

**`socp` is the primary engine.** Every reported result -- the B0-B3 baseline
ladder, the paired counterfactual evaluation, the RL environment, the
diagnostics -- is cleared with `solve_socp_opf_batch`. `lindistflow` and `dc`
are auxiliary engines kept for sensitivity work. A number produced by either of
them is not the number the study reports.

| | **socp** (primary) | **lindistflow** (auxiliary) | **dc** (auxiliary) |
|---|---|---|---|
| formulation | DistFlow rotated cone, exact relaxation | LinDistFlow | lossless DC, single period only |
| voltage variables | yes | yes | **none** |
| losses | `r * I_sq` inside the cone | iterative linearization | **ignored** |
| multi-period | yes | yes | **no** (`batch_fn` is None) |
| prosumer-attached storage valued at | its **own declared bid/offer** | **wholesale**, then **nodal LMP** in the re-solve | **wholesale** |
| independent storage (`storage_units`) | declared `bid_value`/`offer_cost` | declared `bid_value`/`offer_cost` | **not modelled at all** |
| LMP source | power-balance dual | dual + marginal-loss-factor overwrite | angle-only dual |
| charge/discharge exclusivity | single net flow | single net flow | **two free variables, no exclusivity** (Gurobi path); a binary on the HiGHS path |
| emits `clearing_objective` | yes | yes | yes, but a **different function** |

## The three that matter most

**Independent storage units are not in the DC model.** `clear_market` never
passes `storage_units` to it, and the fleet is not in the `agents` list either,
so a day cleared with `--opf-mode dc` has no independent battery in it. Any
comparison of fleet economics across engines is meaningless for this reason
alone.

**Prosumer-attached batteries are priced differently.** The SOCP costs charging
at the unit's own declared charge price; LinDistFlow and DC value the battery's
net flow at the wholesale price (and, in the LinDistFlow nodal re-solve, at the
nodal LMP). Since the declared price is what a participant actually bids, only
the SOCP's dispatch is the one that participant asked for.

**The three `clearing_objective` values are not the same quantity.** The SOCP
and LinDistFlow objectives are the same construction; the DC one is a simpler
expression (`served * bid - (pv + wind + dis) * offer`). Comparing an objective
across engines compares two different functions.

## Charge and discharge

A battery is modelled as one net flow whose parts are `ch` and `dis`. The
decomposition is exact only while the combined objective coefficient on each
part stays negative, which is what `dispatch_core.check_storage_quote_signs`
enforces and why both quote bounds are non-negative: charging is costed at the
declared charge price plus the cycle cost on both legs. A quote below
`-cycle_cost` would flip the coefficient and make simultaneous charge and
discharge profitable again.

`use_nodal_price` (default `true`) adds a second pass that re-values storage at
the nodal LMP and re-solves. It rebuilds the objective with `setObjective`,
**overwriting the first one** — so a term that differs in sign between the two
passes changes the dispatch silently while the reported prices come from the
second pass. Both passes now build the independent-storage terms through
`dispatch_socp._add_independent_storage`, so there is no second sign to get
wrong. (`storage_units` currently has no caller: the ESS fleet reaches the
solver as ordinary agents. The defect was real but dormant, which is why it
never surfaced.)

## Money and prices

`money.py` is the single convention. A period is `DT_HOURS = 0.25` long, so
power must be scaled by it before it can be priced. A power-balance dual is CNY
per MW *for a period*, so `money.price_from_balance_dual` divides by `dt` —
without it every nodal price comes out four times too low while still looking
self-consistent. `dispatch_socp.py` used to redefine `DT_HOURS = 0.25` locally,
shadowing the imported one; that second source of truth is gone.
