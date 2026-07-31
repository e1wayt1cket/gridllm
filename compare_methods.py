"""Compare weighted-sum vs constraint-based multi-objective optimization.

Runs both methods on the baseline scenario and compares welfare, RE rate,
carbon emissions, solve time, feasibility, and shadow prices.

Usage:
    python compare_methods.py           # default comparison table
    python compare_methods.py --pareto  # Pareto frontier scan
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from models import MarketConfig
from market import clear_market
from strategies import adaptive_bidding
from scenarios import get_scenario

T = 96

def run_one(config, label):
    agents, _ = get_scenario("baseline", T=T)
    actions = adaptive_bidding(agents, config, strategy="rl")
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


def pareto_scan():
    """Grid-scan carbon caps and RE rates to map the Pareto frontier.

    Sweeps carbon_cap_tco2 over [20, 300] and re_min_rate over [0.3, 1.0],
    recording (welfare, carbon, re_rate) for every feasible point.
    Computes the non-dominated set and prints a summary.
    """
    carbon_range = np.linspace(20, 300, 6)
    re_range = np.linspace(0.3, 1.0, 5)
    results = []
    total = len(carbon_range) * len(re_range)
    count = 0

    agents, _ = get_scenario("baseline", T=T)

    for c_cap in carbon_range:
        for r_rate in re_range:
            count += 1
            cfg = MarketConfig(opf_mode="lindistflow", verbose=False)
            cfg.market_design.use_constraint_multi_obj = True
            cfg.market_design.carbon_cap_tco2 = float(c_cap)
            cfg.market_design.re_min_rate = float(r_rate)

            actions = adaptive_bidding(agents, cfg, strategy="random")
            t0 = time.perf_counter()
            try:
                res = clear_market(agents, T, "DA", actions, cfg)
                elapsed = time.perf_counter() - t0
                if res is not None:
                    sp = res.get("shadow_prices", {})
                    results.append({
                        "carbon_cap": float(c_cap),
                        "re_rate_target": float(r_rate),
                        "feasible": True,
                        "welfare": res["welfare"],
                        "carbon": res["carbon_emissions"],
                        "re_rate_actual": res["re_consumption_rate"],
                        "curtailment": res["total_curtailment"],
                        "time": elapsed,
                        "carbon_slack": sp.get("carbon_slack_tco2", 0.0),
                        "re_slack": sp.get("re_slack_mwh", 0.0),
                        "shadow_carbon": sp.get("carbon_cap", None),
                        "shadow_re": sp.get("re_min_rate", None),
                    })
                else:
                    results.append({
                        "carbon_cap": float(c_cap),
                        "re_rate_target": float(r_rate),
                        "feasible": False,
                    })
            except Exception as e:
                results.append({
                    "carbon_cap": float(c_cap),
                    "re_rate_target": float(r_rate),
                    "feasible": False,
                    "error": str(e),
                })
            print(f"\r  Scanning... {count}/{total} ({(count/total)*100:.0f}%)", end="", flush=True)

    print()

    feasible = [r for r in results if r["feasible"]]
    infeasible = [r for r in results if not r["feasible"]]
    print(f"\nFeasible: {len(feasible)}, Infeasible: {len(infeasible)}")
    if infeasible:
        # With slack variables, infeasible should be rare
        print("  Infeasible points (should be empty with slack):")
        for r in infeasible[:5]:
            print(f"    C<={r['carbon_cap']}, RE>={r['re_rate_target']}")

    # Compute Pareto frontier: non-dominated in (carbon, welfare) space
    # A point dominates another if it has BOTH lower carbon AND higher welfare.
    pareto = []
    for i, a in enumerate(feasible):
        dominated = False
        for j, b in enumerate(feasible):
            if i == j:
                continue
            # b dominates a: b has lower carbon AND higher welfare
            if b["carbon"] <= a["carbon"] and b["welfare"] >= a["welfare"]:
                if b["carbon"] < a["carbon"] or b["welfare"] > a["welfare"]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(a)

    # Sort by carbon ascending
    pareto.sort(key=lambda r: r["carbon"])

    print(f"\nPareto frontier ({len(pareto)} points):")
    print(f"{'Carbon Cap':>11} {'RE Target':>10} {'Welfare':>10} {'Carbon':>9} {'RE Act%':>8} {'Curtail':>9} {'C-Slack':>9} {'RE-Slack':>9}")
    print("-" * 85)
    for r in pareto:
        print(f"{r['carbon_cap']:11.0f} {r['re_rate_target']:10.2f} {r['welfare']:10.0f} "
              f"{r['carbon']:9.1f} {r['re_rate_actual']:7.1f}% {r['curtailment']:9.1f} "
              f"{r['carbon_slack']:9.2f} {r['re_slack']:9.2f}")

    # Carbon-welfare trade-off summary
    if len(pareto) >= 2:
        best_w = pareto[-1]  # highest welfare (relaxed constraints)
        best_c = pareto[0]   # lowest carbon (tightest constraints)
        print(f"\nTrade-off summary:")
        print(f"  Min carbon: {best_c['carbon']:.1f} tCO2 → welfare {best_c['welfare']:.0f} CNY "
              f"(cap={best_c['carbon_cap']:.0f}, slack={best_c['carbon_slack']:.2f})")
        print(f"  Max welfare: {best_w['welfare']:.0f} CNY → carbon {best_w['carbon']:.1f} tCO2 "
              f"(cap={best_w['carbon_cap']:.0f})")
        delta_w = best_w['welfare'] - best_c['welfare']
        delta_c = best_c['carbon'] - best_w['carbon']
        if delta_c > 0:
            print(f"  Abatement cost: {delta_w/delta_c:.0f} CNY/tCO2 "
                  f"(welfare reduction per tonne CO2 avoided)")

    print("\nPareto scan done.")


if __name__ == "__main__":
    if "--pareto" in sys.argv:
        pareto_scan()
    else:
        main()
