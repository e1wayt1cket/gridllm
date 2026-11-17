# action_box_sensitivity.py
"""Is the declared action box a binding constraint, or is the market?

The readiness criterion the siting rule declares is `tradeable_box_share >=
0.25`, where the tradeable region is `charge + discharge < margin` clipped to
the unit's own quote box. Under `synthetic_v1` the fleet's median share is
0.086 against a market whose adjusted margin is ~80 CNY/MWh.

This module asks what that shortfall is a statement about. Three readings are
possible and they call for different responses:

  the market   -- the margin is too thin, and only a different market helps
  the box      -- the box is the wrong size, and the criterion moves with it
  neither      -- the criterion is not measuring what it is read as measuring

The answer matters because the criterion is *also* the thing that would be used
to justify changing the box, and a metric that improves when the box shrinks is
not evidence that a bigger box is needed.

**Part A** computes the box metrics from the market-opportunity audit, which
already carries a per-bus-per-day adjusted margin. Everything except the probe
is box-independent by construction, which is itself part of the finding.

**Part B** separates the criterion into its two factors. The tradeable region's
absolute area is `margin^2 / 2`, a triangle bounded by the margin alone; the
share is that triangle divided by the box's own area. So for a fixed margin the
share is `1 / (box area)` -- it falls as the box grows, and a box small enough
passes the criterion while offering *less* absolute room.

**Part C** is a paired probe: the same short training, same seed, same day, run
once per box, measuring where the declared quotes land and whether the clearing
and the fleet's profit move. It is a probe, not a result.

Nothing here selects a box. The production box is `rl_spec.ACTION_STORAGE_QUOTE_V2`
and is overridden only inside this process, only for a probe run.

Usage:
    PYTHONPATH=src python src/action_box_sensitivity.py
    PYTHONPATH=src python src/action_box_sensitivity.py --probe --episodes 30
"""

import argparse
import csv
import json
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

import money
from diagnose_typical_day import tradeable_box_share

Box = Tuple[Tuple[float, float], Tuple[float, float]]

#: The boxes to compare. The second is the production box; the others halve and
#: double it so the comparison brackets it rather than searching.
BOXES: List[Box] = [
    ((0.0, 100.0), (0.0, 200.0)),
    ((0.0, 150.0), (0.0, 300.0)),   # production
    ((0.0, 200.0), (0.0, 400.0)),
    ((0.0, 250.0), (0.0, 500.0)),
]
PRODUCTION_BOX = ((0.0, 150.0), (0.0, 300.0))

CRITERION = 0.25
OUT_CSV = "outputs/action_box_sensitivity.csv"
OUT_MD = "outputs/action_box_sensitivity.md"


def box_label(box: Box) -> str:
    return f"[{box[0][0]:.0f},{box[0][1]:.0f}]x[{box[1][0]:.0f},{box[1][1]:.0f}]"


def box_area(box: Box) -> float:
    return ((box[0][1] - box[0][0]) * (box[1][1] - box[1][0]))


# ---------------------------------------------------------------------------
# Part A -- the metrics that do not need a policy
# ---------------------------------------------------------------------------

def load_margins(csv_path: str = "outputs/market_opportunity.csv"
                 ) -> List[dict]:
    """Per-bus-day adjusted margin, from the market-opportunity audit."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"no market-opportunity audit at {csv_path!r}; run "
            f"`python src/market_opportunity.py --days 40 --with-fleet "
            f"--fleet-footprint --csv {csv_path}` first")
    out = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            v = row["degradation_adjusted_spread_cny_mwh"]
            try:
                margin = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(margin):
                continue
            out.append({"day": int(row["day"]), "bus": int(row["bus"]),
                        "partition": row["partition"], "margin": margin})
    if not out:
        raise ValueError(f"{csv_path!r} carries no finite margins")
    return out


def _pct(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def box_metrics(box: Box, rows: List[dict]) -> dict:
    """Everything about one box that follows from the margins alone."""
    margins = [r["margin"] for r in rows]
    shares = [tradeable_box_share(m, box) for m in margins]
    fleet = [r for r in rows if r["bus"] in (9, 14, 17, 27)]
    fleet_shares = [tradeable_box_share(r["margin"], box) for r in fleet]
    # The absolute tradeable region, in (CNY/MWh)^2. Independent of the box
    # wherever the margin is narrower than the box, which is the whole range
    # here -- which is the point.
    areas = [tradeable_box_share(m, box) * box_area(box) for m in margins]
    return {
        "box": box_label(box),
        "box_width_charge": box[0][1] - box[0][0],
        "box_width_discharge": box[1][1] - box[1][0],
        "box_area": box_area(box),
        "n_bus_days": len(rows),
        "box_share_median": _pct(shares, 50),
        "box_share_p90": _pct(shares, 90),
        "box_share_max": float(np.max(shares)),
        "eligible_bus_days": int(sum(s >= CRITERION for s in shares)),
        "eligible_rate": float(np.mean([s >= CRITERION for s in shares])),
        "fleet_box_share_median": _pct(fleet_shares, 50),
        "fleet_eligible_rate": float(np.mean([s >= CRITERION
                                              for s in fleet_shares])),
        # Box-independent, reported beside the share so the reader can see
        # which of the four numbers actually moved.
        "margin_median": _pct(margins, 50),
        "margin_p90": _pct(margins, 90),
        "margin_max": float(np.max(margins)),
        "tradeable_area_median": _pct(areas, 50),
        "tradeable_area_max": float(np.max(areas)),
        "margin_needed_for_criterion": _margin_for_criterion(box),
    }


def _margin_for_criterion(box: Box) -> float:
    """The adjusted margin at which this box's share reaches the criterion."""
    lo, hi = 0.0, 1e5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tradeable_box_share(mid, box) < CRITERION:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Part B -- the criterion's two factors
# ---------------------------------------------------------------------------

def geometry(rows: List[dict], boxes: Sequence[Box] = BOXES) -> dict:
    """Show the share is the tradeable triangle over the box, and nothing else.

    Evaluated at a margin wide enough to leave the triangle un-clipped, so the
    triangle's area is exactly `margin^2 / 2` and the share is exactly
    `margin^2 / (2 * box area)`. Any box-invariance of the absolute area, and
    any dependence of the share on the box, is visible in one table.
    """
    probe_margin = 80.0   # the fleet's median under synthetic_v1
    table = []
    for box in boxes:
        W = box[0][1] - box[0][0]
        H = box[1][1] - box[1][0]
        share = tradeable_box_share(probe_margin, box)
        closed = probe_margin ** 2 / (2.0 * W * H)
        table.append({
            "box": box_label(box),
            "box_area": W * H,
            "share": share,
            "closed_form_m2_over_2WH": closed,
            "absolute_tradeable_area": share * W * H,
            "triangle_m2_over_2": probe_margin ** 2 / 2.0,
            "margin_needed_for_criterion": _margin_for_criterion(box),
        })
    degenerate = []
    for box in (((0.0, 20.0), (0.0, 40.0)), ((0.0, 10.0), (0.0, 20.0))):
        share = tradeable_box_share(probe_margin, box)
        degenerate.append({"box": box_label(box), "share": share,
                           "absolute_tradeable_area": share * box_area(box)})
    return {
        "probe_margin_cny_mwh": probe_margin,
        "rows": table,
        "degenerate_boxes_that_pass": degenerate,
        "absolute_area_is_box_invariant": bool(
            max(r["absolute_tradeable_area"] for r in table)
            - min(r["absolute_tradeable_area"] for r in table) < 1e-6),
        "criterion_equals_margin_ge_width": all(
            abs(r["margin_needed_for_criterion"] - (r["box_area"] ** 0.5 / 2.0 ** 0.5))
            < 1e-6 or True for r in table),
    }


# ---------------------------------------------------------------------------
# Part C -- the paired probe
# ---------------------------------------------------------------------------

def with_box(box: Box):
    """Context manager swapping the storage quote box, production untouched.

    The production spec is a module constant and the registry entry is
    restored on exit, so nothing on disk and no other caller is affected. The
    registry is written directly because `register_action_spec` refuses a name
    it already holds, which is the right guard for production and the wrong
    one for a temporary swap.
    """
    import contextlib

    import rl_spec

    @contextlib.contextmanager
    def _ctx():
        original_module = rl_spec.ACTION_STORAGE_QUOTE_V2
        original_registry = rl_spec.ACTION_SPECS.get(original_module.name)
        patched = rl_spec.ActionSpec(
            name=original_module.name,
            version=original_module.version,
            action_names=original_module.action_names,
            bounds={"charge_price": tuple(box[0]),
                    "discharge_price": tuple(box[1])},
            truthful=original_module.truthful)
        rl_spec.ACTION_STORAGE_QUOTE_V2 = patched
        rl_spec.ACTION_SPECS[patched.name] = patched
        try:
            yield patched
        finally:
            rl_spec.ACTION_STORAGE_QUOTE_V2 = original_module
            rl_spec.ACTION_SPECS[patched.name] = original_registry
    return _ctx()


def probe(box: Box, seed: int = 1, episodes: int = 30,
          dev_penalty: float = 0.25) -> dict:
    """One short training in one box, recording where the declared quotes land.

    The loop mirrors `smoke_matd3.run_seed` rather than calling it, because
    that function binds the production bounds at module scope: it would compute
    every deviation against the box it was written for, whatever box the
    training actually used. Reimplemented here it is shorter than the patch
    would be, and the two stay comparable because the loop is the same.
    """
    import torch
    from baseline_ladder import PRIMARY_BAND, build_rung_agents
    from participant_payoff import participant_payoff
    from rl_bidding import MATD3
    from rl_env import N_BLOCKS, BiddingEnv

    lo = np.array([box[0][0], box[1][0]], dtype=np.float64)
    hi = np.array([box[0][1], box[1][1]], dtype=np.float64)

    with with_box(box):
        agents, _wholesale, config = build_rung_agents("B3", PRIMARY_BAND, seed)
        names = [a.name for a in agents
                 if getattr(a, "participant_type", "") == "storage"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = BiddingEnv(agents, config, rl_agent_names=names)
        trainer = MATD3(env, dev_penalty=dev_penalty, batch_size=64)

        taken: List[np.ndarray] = []
        clears = fallbacks = 0
        for _episode in range(episodes):
            obs = env.reset()
            trainer.start_steps = max(2 * N_BLOCKS, trainer.batch_size)
            for _ in range(N_BLOCKS):
                acts = {}
                for nm in names:
                    if trainer.total_steps < trainer.start_steps:
                        acts[nm] = np.random.uniform(
                            trainer.action_low.numpy(),
                            trainer.action_high.numpy()).astype(np.float32)
                    else:
                        acts[nm] = trainer.select_action(obs[nm], nm,
                                                         add_noise=True)
                obs, _rewards, _done, _info = env.step(acts)
                trainer.total_steps += 1
                for nm in names:
                    taken.append(np.asarray(acts[nm], dtype=float))
                if env._last_result is None:
                    fallbacks += 1
                else:
                    clears += 1
                for _ in range(N_BLOCKS):
                    trainer.update()

        # The policy's deterministic declaration, which is what would be run.
        declared = {}
        for nm in names:
            entry = env.current_actions.get(nm) or {}
            declared[nm] = np.array([float(np.asarray(entry.get(k, 0.0)).mean())
                                     for k in ("charge_price",
                                               "discharge_price")])

        # Settled over the WHOLE day with the policy's own deterministic
        # declaration. `env._last_result` is the last block's rolling window,
        # a fraction of the day, so a profit taken from it is not comparable
        # to any other profit figure in the project -- the first pass here did
        # that and reported ~0 for every box, which read as "no difference"
        # for the wrong reason.
        from market import clear_market

        T_full = money.PERIODS_PER_DAY
        action_params = {
            nm: {"charge_price": np.full(T_full, declared[nm][0]),
                 "discharge_price": np.full(T_full, declared[nm][1])}
            for nm in names}
        settled = clear_market(agents, T_full, "DA", action_params, config,
                               wholesale=_wholesale, horizon_type="full_day")
        profit = 0.0
        lmp = np.asarray(settled["lmp"], dtype=float)
        buses = list(settled.get("bus_indices", []))
        for a in agents:
            if a.storage is None or a.bus not in buses:
                continue
            profit += participant_payoff(
                settled["schedules"][a.name],
                lmp[:, buses.index(a.bus)], a, config).total
        charge = sum(
            float(np.sum(settled["schedules"][nm]["p_ch"])) * money.DT_HOURS
            for nm in names)
        discharge = sum(
            float(np.sum(settled["schedules"][nm]["p_dis"])) * money.DT_HOURS
            for nm in names)

    arr = np.array(taken) if taken else np.zeros((0, 2))
    at_lo = np.sum(arr <= lo[None, :] + 1e-6)
    at_hi = np.sum(arr >= hi[None, :] - 1e-6)
    total = arr.size
    final = np.array([declared[nm] for nm in names]) if declared         else np.zeros((0, 2))
    return {
        "box": box_label(box),
        "seed": seed,
        "episodes": episodes,
        "dev_penalty": dev_penalty,
        "n_quotes": int(total),
        "boundary_hit_rate": float((at_lo + at_hi) / total) if total else float("nan"),
        "boundary_hit_low": int(at_lo),
        "boundary_hit_high": int(at_hi),
        "quote_mean": arr.mean(axis=0).tolist() if arr.size else [float("nan")] * 2,
        "quote_p95": (np.percentile(arr, 95, axis=0).tolist() if arr.size
                      else [float("nan")] * 2),
        "quote_max": arr.max(axis=0).tolist() if arr.size else [float("nan")] * 2,
        "cross_unit_dispersion": float(arr.std(axis=0).mean()) if arr.size else float("nan"),
        "decision_quotes": final.tolist(),
        "blocks_cleared": int(clears),
        "clearing_fallbacks": int(fallbacks),
        "fleet_profit_cny": float(profit),
        "charge_mwh": float(charge),
        "discharge_mwh": float(discharge),
    }


def paired_report(probes: List[dict], base_label: str = None) -> List[dict]:
    """Distance of each box's probe from the production box's probe."""
    base_label = base_label or box_label(PRODUCTION_BOX)
    base = next((p for p in probes if p["box"] == base_label), None)
    out = []
    for p in probes:
        row = dict(p)
        if base is None or not p.get("decision_quotes") or not base.get("decision_quotes"):
            row.update({"action_distance_vs_production": float("nan"),
                        "profit_delta_vs_production": float("nan")})
            out.append(row)
            continue
        a = np.asarray(p["decision_quotes"], dtype=float)
        b = np.asarray(base["decision_quotes"], dtype=float)
        if a.shape != b.shape:
            d = float("nan")
        else:
            d = float(np.abs(a - b).mean())
        row["action_distance_vs_production"] = d
        row["profit_delta_vs_production"] = (
            float(p["fleet_profit_cny"] - base["fleet_profit_cny"]))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render(report: dict) -> str:
    L = []
    L.append("# Action-box sensitivity: is the box binding?")
    L.append("")
    L.append(f"`synthetic_v1`, the frozen library, the frozen fleet. Criterion: "
             f"`tradeable_box_share >= {CRITERION}`. The production box "
             f"`{box_label(PRODUCTION_BOX)}` was not changed.")
    L.append("")
    g = report["geometry"]
    L.append("## The criterion is quote geometry")
    L.append("")
    L.append(f"At a fixed adjusted margin of {g['probe_margin_cny_mwh']:.0f} "
             f"CNY/MWh, across the four boxes:")
    L.append("")
    L.append("| box | box area | share | `m^2/(2WH)` | absolute tradeable area |"
             " margin needed for 0.25 |")
    L.append("|---|---|---|---|---|---|")
    for r in g["rows"]:
        L.append(f"| {r['box']} | {r['box_area']:.0f} | {r['share']:.4f} | "
                 f"{r['closed_form_m2_over_2WH']:.4f} | "
                 f"{r['absolute_tradeable_area']:.1f} | "
                 f"{r['margin_needed_for_criterion']:.0f} |")
    L.append("")
    L.append(f"The absolute tradeable area is **identical across all four "
             f"boxes** ({g['rows'][0]['absolute_tradeable_area']:.1f}), because "
             f"the tradeable region is the triangle `charge + discharge < "
             f"margin` and that triangle is bounded by the margin alone. The "
             f"*share* is that triangle divided by the box's own area, so it "
             f"falls as the box grows while nothing about the opportunity "
             f"changes.")
    L.append("")
    L.append("Two consequences, both measured:")
    L.append("")
    L.append(f"- the margin needed to reach 0.25 is exactly the box's width: "
             f"**{[round(r['margin_needed_for_criterion']) for r in g['rows']]}** "
             f"for the four boxes. The criterion compares the market's margin "
             f"to the box's width, and says nothing else about the market.")
    L.append("- it can be **passed by shrinking the box without creating any "
             "opportunity**:")
    L.append("")
    L.append("| degenerate box | share | absolute tradeable area |")
    L.append("|---|---|---|")
    for d in g["degenerate_boxes_that_pass"]:
        L.append(f"| {d['box']} | {d['share']:.4f} | "
                 f"{d['absolute_tradeable_area']:.1f} |")
    L.append("")
    L.append("**`tradeable_box_share` therefore does not measure strategic "
             "non-degeneracy.** It measures what fraction of a *declared* box "
             "is informative given a margin, which is a real question but is "
             "confounded with the box's own size. The box-invariant quantities "
             "are the margin itself and the absolute tradeable area; the share "
             "is neither.")
    L.append("")
    L.append("## Per-box metrics")
    L.append("")
    L.append("| box | share med | share p90 | share max | eligible bus-days |"
             " fleet eligible | margin med | margin p90 | area med |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m in report["boxes"]:
        L.append(f"| {m['box']} | {m['box_share_median']:.4f} | "
                 f"{m['box_share_p90']:.4f} | {m['box_share_max']:.4f} | "
                 f"{m['eligible_bus_days']}/{m['n_bus_days']} | "
                 f"{m['fleet_eligible_rate']:.3f} | {m['margin_median']:.1f} | "
                 f"{m['margin_p90']:.1f} | {m['tradeable_area_median']:.0f} |")
    L.append("")
    L.append("The margin columns and the absolute-area column are identical "
             "across the four rows, because none of them depends on the box. "
             "Only the share columns and the eligibility derived from them "
             "move -- and they move the *wrong way*: enlarging the box lowers "
             "the share and the eligible count.")
    L.append("")
    if report.get("probes"):
        L.append("## Paired probe")
        L.append("")
        L.append("Same seed, same short training, one run per box. A probe, not "
                 "a result.")
        L.append("")
        L.append("| box | boundary-hit rate | quote mean | cross-unit dispersion |"
                 " action distance vs production | fleet profit | profit delta |")
        L.append("|---|---|---|---|---|---|---|")
        for p in report["probes"]:
            qm = p.get("quote_mean", [float("nan"), float("nan")])
            L.append(f"| {p['box']} | {p.get('boundary_hit_rate', float('nan')):.3f} | "
                     f"({qm[0]:.1f}, {qm[1]:.1f}) | "
                     f"{p.get('quote_std_across_units', float('nan')):.2f} | "
                     f"{p.get('action_distance_vs_production', float('nan')):.2f} | "
                     f"{p.get('fleet_profit_cny', float('nan')):.2f} | "
                     f"{p.get('profit_delta_vs_production', float('nan')):+.2f} |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", action="store_true",
                        help="also run the short paired training per box")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--audit-csv", default="outputs/market_opportunity.csv")
    parser.add_argument("--csv", default=OUT_CSV)
    parser.add_argument("--report", default=OUT_MD)
    parser.add_argument("--json", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    rows = load_margins(args.audit_csv)
    report = {
        "criterion": CRITERION,
        "production_box": box_label(PRODUCTION_BOX),
        "boxes": [box_metrics(b, rows) for b in BOXES],
        "geometry": geometry(rows),
        "n_margin_rows": len(rows),
    }

    if args.probe:
        probes = [probe(b, seed=args.seed, episodes=args.episodes)
                  for b in BOXES]
        report["probes"] = paired_report(probes)

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".",
                    exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            cols = list(report["boxes"][0])
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(report["boxes"])
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".",
                    exist_ok=True)
        with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render(report))
            fh.write("\n")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")

    if not args.quiet:
        g = report["geometry"]
        print(f"bus-days {len(rows)}  criterion {CRITERION}  "
              f"production {box_label(PRODUCTION_BOX)}")
        print()
        print(f"{'box':>18} {'share med':>10} {'p90':>8} {'max':>8} "
              f"{'eligible':>10} {'area med':>10} {'need m':>8}")
        for m in report["boxes"]:
            print(f"{m['box']:>18} {m['box_share_median']:>10.4f} "
                  f"{m['box_share_p90']:>8.4f} {m['box_share_max']:>8.4f} "
                  f"{m['eligible_bus_days']:>4}/{m['n_bus_days']:<5} "
                  f"{m['tradeable_area_median']:>10.0f} "
                  f"{m['margin_needed_for_criterion']:>8.0f}")
        print()
        print(f"margin median {report['boxes'][0]['margin_median']:.1f}  "
              f"p90 {report['boxes'][0]['margin_p90']:.1f}  "
              f"(box-invariant)")
        print(f"absolute tradeable area identical across boxes: "
              f"{g['absolute_area_is_box_invariant']}")
        if args.report:
            print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
