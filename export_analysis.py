"""Export simulation results to CSV and analyze data inconsistencies."""
from ortools.linear_solver import pywraplp  # noqa: F401 — preload for DLL order
import numpy as np
import pandas as pd
import os
from models import MarketConfig
from market import clear_market, adaptive_bidding
from scenarios import get_scenario

os.makedirs("output_csv", exist_ok=True)

config = MarketConfig(opf_mode="lindistflow", verbose=False)
agents, _ = get_scenario("baseline", T=96)
actions = adaptive_bidding(agents, config, strategy="random")
result = clear_market(agents, 96, "DA", actions, config)

T = 96
dt = 0.25

# =====================================================================
# 1. Per-agent time series
# =====================================================================
rows = []
for a in agents:
    nm = a.name
    s = result["schedules"][nm]
    soc_arr = s.get("soc", np.zeros(T + 1))
    for t in range(T):
        rows.append({
            "agent": nm, "bus": a.bus, "load_type": a.load_type,
            "is_prosumer": int(a.is_prosumer),
            "has_storage": int(a.storage is not None),
            "t": t, "hour": round(t * dt, 2),
            "load_scheduled_MW": round(s["served"][t], 6),
            "unserved_MW": round(s["unserved"][t], 6),
            "pv_used_MW": round(s["pv_used"][t], 6),
            "wind_used_MW": round(s["wind_used"][t], 6),
            "p_ch_MW": round(s["p_ch"][t], 6),
            "p_dis_MW": round(s["p_dis"][t], 6),
            "p_buy_MW": round(s["p_buy"][t], 6),
            "p_sell_MW": round(s["p_sell"][t], 6),
            "soc": round(float(soc_arr[t]), 6),
            "lmp_bus_CNY": round(float(result["lmp"][t, a.bus]), 2),
            "storage_mode": s.get("storage_mode", ["idle"] * T)[t],
        })
df_agent = pd.DataFrame(rows)
df_agent.to_csv("output_csv/agent_timeseries.csv", index=False)
print(f"[1/7] agent_timeseries.csv — {len(df_agent)} rows")

# =====================================================================
# 2. LMP heatmap (96 x 33)
# =====================================================================
lmp_df = pd.DataFrame(result["lmp"], columns=[f"bus{b}" for b in range(33)])
lmp_df.to_csv("output_csv/lmp_heatmap.csv")
print(f"[2/7] lmp_heatmap.csv — {lmp_df.shape[0]} periods x {lmp_df.shape[1]} buses")

# =====================================================================
# 3. Global summary
# =====================================================================
summary = {
    "welfare_CNY": result["welfare"],
    "re_consumption_rate_pct": result["re_consumption_rate"],
    "total_re_available_MWh": result["total_re_available"],
    "carbon_emissions_tCO2": result["carbon_emissions"],
    "carbon_intensity_tCO2_per_MWh": result["carbon_intensity"],
    "total_curtailment_MWh": result["total_curtailment"],
    "shadow_carbon_cap_CNY": result["shadow_prices"].get("carbon_cap", np.nan),
    "shadow_re_min_rate_CNY": result["shadow_prices"].get("re_min_rate", np.nan),
    "carbon_cap_tCO2": config.carbon_cap_tco2,
    "re_min_rate_pct": config.re_min_rate,
}
pd.DataFrame([summary]).to_csv("output_csv/summary.csv", index=False)
print(f"[3/7] summary.csv")

# =====================================================================
# 4. Storage agent summary
# =====================================================================
st_rows = []
for a in agents:
    if a.storage is not None:
        s = result["schedules"][a.name]
        soc_arr = np.array(s["soc"])
        ch_arr = s["p_ch"]
        dis_arr = s["p_dis"]
        st_rows.append({
            "agent": a.name, "bus": a.bus,
            "e_max_MWh": a.storage.e_max,
            "p_ch_max_MW": a.storage.p_ch_max,
            "p_dis_max_MW": a.storage.p_dis_max,
            "eta_ch": a.storage.eta_ch, "eta_dis": a.storage.eta_dis,
            "soc0": a.storage.soc0,
            "soc_min_cfg": a.storage.soc_min,
            "soc_max_cfg": a.storage.soc_max,
            "total_ch_MWh": round(np.sum(ch_arr) * dt, 6),
            "total_dis_MWh": round(np.sum(dis_arr) * dt, 6),
            "expected_soc_change": round(
                (np.sum(ch_arr) * a.storage.eta_ch -
                 np.sum(dis_arr) / a.storage.eta_dis) * dt / a.storage.e_max, 6),
            "actual_soc_change": round(
                float(s.get("soc_final", soc_arr[-1]) - soc_arr[0]), 6),
            "soc_final": round(float(s.get("soc_final", soc_arr[-1])), 6),
            "soc_mean": round(float(np.mean(soc_arr[:T])), 6),
            "soc_min_actual": round(float(np.min(soc_arr)), 6),
            "soc_max_actual": round(float(np.max(soc_arr)), 6),
            "n_periods_ch": int(np.sum(ch_arr > 0.001)),
            "n_periods_dis": int(np.sum(dis_arr > 0.001)),
            "n_periods_simul": int(np.sum((ch_arr > 0.001) & (dis_arr > 0.001))),
            "n_periods_idle": int(np.sum((ch_arr <= 0.001) & (dis_arr <= 0.001))),
        })
df_st = pd.DataFrame(st_rows)
df_st.to_csv("output_csv/storage_summary.csv", index=False)
print(f"[4/7] storage_summary.csv — {len(df_st)} agents")

# =====================================================================
# 5. Per-agent energy balance
# =====================================================================
bal_rows = []
for a in agents:
    nm = a.name
    s = result["schedules"][nm]
    total_load = float(np.sum(s["served"] + s["unserved"]) * dt)
    total_buy = float(np.sum(s["p_buy"]) * dt)
    total_sell = float(np.sum(s["p_sell"]) * dt)
    total_pv = float(np.sum(s["pv_used"]) * dt)
    total_wind = float(np.sum(s["wind_used"]) * dt)
    total_ch = float(np.sum(s["p_ch"]) * dt)
    total_dis = float(np.sum(s["p_dis"]) * dt)
    total_served = float(np.sum(s["served"]) * dt)
    total_unserved = float(np.sum(s["unserved"]) * dt)
    supply = total_buy + total_pv + total_wind + total_dis
    demand = total_served + total_sell + total_ch
    bal_rows.append({
        "agent": nm, "bus": a.bus, "load_type": a.load_type,
        "is_prosumer": a.is_prosumer,
        "total_load_MWh": round(total_load, 6),
        "served_MWh": round(total_served, 6),
        "unserved_MWh": round(total_unserved, 6),
        "buy_MWh": round(total_buy, 6),
        "sell_MWh": round(total_sell, 6),
        "pv_MWh": round(total_pv, 6),
        "wind_MWh": round(total_wind, 6),
        "ch_MWh": round(total_ch, 6),
        "dis_MWh": round(total_dis, 6),
        "supply_MWh": round(supply, 6),
        "demand_MWh": round(demand, 6),
        "imbalance_MWh": round(supply - demand, 8),
        "imbalance_pct": round(abs(supply - demand) / max(total_load, 0.001) * 100, 6),
    })
df_bal = pd.DataFrame(bal_rows)
df_bal.to_csv("output_csv/agent_balance.csv", index=False)
n_bad = int(np.sum(df_bal["imbalance_pct"] > 0.1))
print(f"[5/7] agent_balance.csv — {n_bad} agents with >0.1% imbalance")

# =====================================================================
# 6. System-level per-period balance
# =====================================================================
sys_rows = []
for t in range(T):
    total_buy_t = sum(result["schedules"][a.name]["p_buy"][t] for a in agents)
    total_sell_t = sum(result["schedules"][a.name]["p_sell"][t] for a in agents)
    total_pv_t = sum(result["schedules"][a.name]["pv_used"][t] for a in agents)
    total_wind_t = sum(result["schedules"][a.name]["wind_used"][t] for a in agents)
    total_ch_t = sum(result["schedules"][a.name]["p_ch"][t] for a in agents)
    total_dis_t = sum(result["schedules"][a.name]["p_dis"][t] for a in agents)
    total_served_t = sum(result["schedules"][a.name]["served"][t] for a in agents)
    total_unserved_t = sum(result["schedules"][a.name]["unserved"][t] for a in agents)
    price_arr = result["price"]  # shape (96,)
    wholesale_t = float(price_arr[t])
    lmp_mean_t = float(np.mean(result["lmp"][t]))
    sys_rows.append({
        "t": t, "hour": round(t * dt, 2),
        "wholesale_CNY": round(float(wholesale_t), 2),
        "lmp_mean_CNY": round(lmp_mean_t, 2),
        "total_load_MW": round(total_served_t + total_unserved_t, 6),
        "total_served_MW": round(total_served_t, 6),
        "total_unserved_MW": round(total_unserved_t, 6),
        "total_buy_MW": round(total_buy_t, 6),
        "total_sell_MW": round(total_sell_t, 6),
        "total_pv_MW": round(total_pv_t, 6),
        "total_wind_MW": round(total_wind_t, 6),
        "total_ch_MW": round(total_ch_t, 6),
        "total_dis_MW": round(total_dis_t, 6),
        "grid_import_MW": round(total_buy_t - total_sell_t, 6),
    })
df_sys = pd.DataFrame(sys_rows)
df_sys.to_csv("output_csv/system_timeseries.csv", index=False)
print(f"[6/7] system_timeseries.csv — {len(df_sys)} periods")

# =====================================================================
# 7. ANOMALY ANALYSIS
# =====================================================================
print("\n" + "=" * 70)
print("ANOMALY ANALYSIS")
print("=" * 70)

anomalies = []

# --- (A) Agent energy imbalance ---
bad_bal = df_bal[df_bal["imbalance_pct"] > 0.1]
if len(bad_bal) > 0:
    print(f"\n[A] ENERGY IMBALANCE (>0.1%): {len(bad_bal)} agents")
    for _, row in bad_bal.iterrows():
        msg = (f"  {row['agent']} (bus{row['bus']}): "
               f"supply={row['supply_MWh']:.4f} MWh, demand={row['demand_MWh']:.4f} MWh, "
               f"gap={row['imbalance_MWh']:.6f} MWh ({row['imbalance_pct']:.2f}%)")
        print(msg)
        anomalies.append({"type": "energy_imbalance", "agent": row["agent"],
                          "detail": f"gap={row['imbalance_MWh']:.6f} MWh ({row['imbalance_pct']:.2f}%)"})
else:
    print(f"\n[A] ENERGY IMBALANCE: All agents within 0.1% tolerance")

# --- (B) Simultaneous ch+dis ---
for _, row in df_st.iterrows():
    if row["n_periods_simul"] > 0:
        msg = (f"[B] SIMULTANEOUS CH+DIS: {row['agent']}: "
               f"{int(row['n_periods_simul'])} periods")
        print(msg)
        anomalies.append({"type": "simul_ch_dis", "agent": row["agent"],
                          "detail": f"{int(row['n_periods_simul'])} periods"})

# --- (C) SOC boundary violations ---
for _, row in df_st.iterrows():
    violations = []
    if row["soc_min_actual"] < row["soc_min_cfg"] - 0.001:
        violations.append(f"soc_min={row['soc_min_actual']:.4f} < cfg={row['soc_min_cfg']}")
    if row["soc_max_actual"] > row["soc_max_cfg"] + 0.001:
        violations.append(f"soc_max={row['soc_max_actual']:.4f} > cfg={row['soc_max_cfg']}")
    if violations:
        print(f"[C] SOC BOUNDS: {row['agent']}: " + ", ".join(violations))
        anomalies.append({"type": "soc_bounds", "agent": row["agent"],
                          "detail": "; ".join(violations)})

# --- (D) SOC change mismatch ---
for _, row in df_st.iterrows():
    diff = abs(row["expected_soc_change"] - row["actual_soc_change"])
    if diff > 0.001:
        msg = (f"[D] SOC MISMATCH: {row['agent']}: "
               f"expected_dSOC={row['expected_soc_change']:.6f}, "
               f"actual_dSOC={row['actual_soc_change']:.6f}, diff={diff:.6f}")
        print(msg)
        anomalies.append({"type": "soc_mismatch", "agent": row["agent"],
                          "detail": f"diff={diff:.6f}"})

# --- (E) Unserved load ---
bad_unserved = df_bal[df_bal["unserved_MWh"] > 0.001]
if len(bad_unserved) > 0:
    print(f"\n[E] UNSERVED LOAD: {len(bad_unserved)} agents")
    for _, row in bad_unserved.iterrows():
        pct = row["unserved_MWh"] / max(row["total_load_MWh"], 0.001) * 100
        msg = (f"  {row['agent']}: unserved={row['unserved_MWh']:.4f} MWh "
               f"({pct:.2f}% of load)")
        print(msg)
        anomalies.append({"type": "unserved", "agent": row["agent"],
                          "detail": f"{row['unserved_MWh']:.4f} MWh ({pct:.2f}%)"})

# --- (F) Zero PV/wind but non-zero sell ---
for _, row in df_bal.iterrows():
    if row["sell_MWh"] > 0.001 and row["pv_MWh"] < 0.001 and row["wind_MWh"] < 0.001 and row["dis_MWh"] < 0.001:
        msg = (f"[F] SELL WITHOUT GENERATION: {row['agent']}: "
               f"sell={row['sell_MWh']:.4f} MWh, pv={row['pv_MWh']:.4f}, "
               f"wind={row['wind_MWh']:.4f}, dis={row['dis_MWh']:.4f}")
        print(msg)
        anomalies.append({"type": "sell_no_gen", "agent": row["agent"],
                          "detail": f"sell={row['sell_MWh']:.4f} without gen"})

# --- (G) Consumer selling (non-prosumer with sell > 0) ---
for _, row in df_bal.iterrows():
    if not row["is_prosumer"] and row["sell_MWh"] > 0.001:
        msg = (f"[G] CONSUMER SELLING: {row['agent']}: "
               f"sell={row['sell_MWh']:.4f} MWh (consumer should not sell)")
        print(msg)
        anomalies.append({"type": "consumer_selling", "agent": row["agent"],
                          "detail": f"sell={row['sell_MWh']:.4f} MWh"})

# --- (H) LMP anomalies ---
lmp_min = np.min(result["lmp"])
lmp_max = np.max(result["lmp"])
lmp_neg = np.sum(result["lmp"] < -1)
print(f"\n[H] LMP RANGE: min={lmp_min:.1f}, max={lmp_max:.1f}")
if lmp_neg > 0:
    print(f"  WARNING: {lmp_neg} negative LMP values detected!")
    anomalies.append({"type": "negative_lmp", "agent": "system",
                      "detail": f"{lmp_neg} negative LMPs"})

# --- (I) Curtailment check ---
total_pv_avail = sum(np.sum(a.pv_forecast) * dt for a in agents)
total_wind_avail = sum(np.sum(a.get_wind_forecast()) * dt for a in agents if a.has_wind)
total_pv_used = sum(float(np.sum(result["schedules"][a.name]["pv_used"])) * dt for a in agents)
total_wind_used = sum(float(np.sum(result["schedules"][a.name]["wind_used"])) * dt for a in agents)
pv_curt = total_pv_avail - total_pv_used
wind_curt = total_wind_avail - total_wind_used
print(f"\n[I] CURTAILMENT: PV={pv_curt:.4f}/{total_pv_avail:.2f} MWh, "
      f"Wind={wind_curt:.4f}/{total_wind_avail:.2f} MWh")

# =====================================================================
# Summary
# =====================================================================
print(f"\n{'=' * 70}")
print(f"TOTAL ANOMALIES FOUND: {len(anomalies)}")
print(f"{'=' * 70}")
if anomalies:
    df_anom = pd.DataFrame(anomalies)
    df_anom.to_csv("output_csv/anomalies.csv", index=False)
    print("[7/7] anomalies.csv")
    for cat in df_anom["type"].unique():
        cat_anoms = df_anom[df_anom["type"] == cat]
        print(f"  {cat}: {len(cat_anoms)} occurrences")
else:
    print("[7/7] No anomalies found.")

print(f"\nAll files in output_csv/:")
for f in sorted(os.listdir("output_csv")):
    sz = os.path.getsize(f"output_csv/{f}")
    print(f"  {f} ({sz:,} bytes)")
