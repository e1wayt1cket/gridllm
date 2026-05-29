"""Compare weighted-sum vs constraint-based multi-objective optimization.

Runs both methods on the baseline scenario and compares welfare, RE rate,
carbon emissions, solve time, feasibility, and shadow prices.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from models import MarketConfig
from market import clear_market, adaptive_bidding
from scenarios import get_scenario

T = 96

def run_one(config, label):
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="random")
    t0 = time.perf_counter()
    try:
        results = clear_market(agents, T, "DA", actions, config)
        elapsed = time.perf_counter() - t0
        if results is None:
            return {"label": label, "feasible": False, "time": elapsed}
        return {
            "label": label,
            "feasible": True,
            "welfare": results["welfare"],
            "re_rate": results["re_consumption_rate"],
            "carbon": results["carbon_emissions"],
            "carbon_intensity": results["carbon_intensity"],
            "curtailment": results["total_curtailment"],
            "time": elapsed,
            "shadow_prices": results.get("shadow_prices", {}),
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"label": label, "feasible": False, "time": elapsed, "error": str(e)}


def main():
    rows = []

    # ---- 1. Weighted-sum baseline ----
    cfg_ws = MarketConfig(opf_mode="lindistflow", verbose=False)
    rows.append(run_one(cfg_ws, "WeightedSum (re=50,curtail=15,carbon=50)"))

    # ---- 2. Constraint-based: vary carbon cap ----
    carbon_caps = [250, 200, 180, 160, 140, 120, 100]
    for cap in carbon_caps:
        cfg = MarketConfig(opf_mode="lindistflow", verbose=False,
                           use_constraint_multi_obj=True,
                           carbon_cap_tco2=cap)
        rows.append(run_one(cfg, f"Constraint: carbon<={cap}"))

    # ---- 3. Constraint-based: vary RE min rate ----
    re_rates = [90, 93, 95, 97, 98, 99]
    for rate in re_rates:
        cfg = MarketConfig(opf_mode="lindistflow", verbose=False,
                           use_constraint_multi_obj=True,
                           re_min_rate=rate)
        rows.append(run_one(cfg, f"Constraint: RE>={rate}%"))

    # ---- 4. Combined constraints ----
    combos = [
        (200, 95), (180, 95), (160, 97), (140, 97), (120, 98),
    ]
    for cap, rate in combos:
        cfg = MarketConfig(opf_mode="lindistflow", verbose=False,
                           use_constraint_multi_obj=True,
                           carbon_cap_tco2=cap, re_min_rate=rate)
        rows.append(run_one(cfg, f"Constraint: C<={cap} + RE>={rate}%"))

    # ---- Print table ----
    print(f"{'Method':<45} {'Welfare':>10} {'RE%':>7} {'Carbon':>8} {'Curtail':>9} {'Time':>6} {'Shadow C':>10} {'Shadow RE':>10}")
    print("-" * 120)
    for r in rows:
        if not r["feasible"]:
            print(f"{r['label']:<45} {'INFEASIBLE':>10}")
            continue
        w = r['welfare']
        re_r = r['re_rate']
        c = r['carbon']
        curt = r['curtailment']
        t = r['time']
        sp = r.get('shadow_prices', {})
        sc = sp.get('carbon_cap', None)
        sr = sp.get('re_min_rate', None)
        sc_s = f"{sc:10.2f}" if sc is not None else "       N/A"
        sr_s = f"{sr:10.2f}" if sr is not None else "       N/A"
        print(f"{r['label']:<45} {w:10.0f} {re_r:6.1f}% {c:8.1f} {curt:9.1f} {t:5.2f}s {sc_s} {sr_s}")

    # ---- Analysis ----
    print("\n=== Analysis ===")
    ws = rows[0]
    print(f"WeightedSum welfare: {ws['welfare']:.0f}, RE: {ws['re_rate']:.1f}%, Carbon: {ws['carbon']:.1f}")

    # Find constraint runs with similar RE rate to WS
    for r in rows[1:]:
        if r['feasible'] and 'RE>=' in r['label']:
            diff = abs(r['re_rate'] - ws['re_rate'])
            if diff < 2:
                w_diff = r['welfare'] - ws['welfare']
                print(f"  {r['label']}: welfare delta={w_diff:+.0f} vs WS")

    # Find constraint runs with similar carbon to WS
    for r in rows[1:]:
        if r['feasible'] and 'carbon<=' in r['label'] and 'RE' not in r['label']:
            diff = abs(r['carbon'] - ws['carbon'])
            if diff < 20:
                w_diff = r['welfare'] - ws['welfare']
                print(f"  {r['label']}: welfare delta={w_diff:+.0f} vs WS (carbon diff={r['carbon']-ws['carbon']:+.1f})")

    # Interpret shadow prices
    print("\nShadow price interpretation:")
    for r in rows[1:]:
        if r['feasible'] and r.get('shadow_prices'):
            sp = r['shadow_prices']
            if 'carbon_cap' in sp and sp['carbon_cap'] is not None and abs(sp['carbon_cap']) > 1e-6:
                print(f"  {r['label']}: carbon shadow = {sp['carbon_cap']:.2f} CNY/tCO2 "
                      f"(marginal welfare gain per unit relaxation)")
            if 're_min_rate' in sp and sp['re_min_rate'] is not None and abs(sp['re_min_rate']) > 1e-6:
                print(f"  {r['label']}: RE shadow = {sp['re_min_rate']:.2f} CNY/% "
                      f"(marginal welfare cost per 1% tighter RE constraint)")

    print("\nDone.")


if __name__ == "__main__":
    main()
