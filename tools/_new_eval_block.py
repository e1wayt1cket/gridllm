def eval_day(trainer: mod.MATD3BC2Gate, pair: dict, b1_cache, *,
             mechanism_hash: str, pair_row: dict) -> dict:
    """Deterministic full-day evaluation; no buffer write, no exposure.

    Records settlement, per-ESS energy/cycles, raw SOC extrema and the
    deterministic gate behaviour of every quote leg. It does not call
    `train_day`, so validation cannot change the model or the exposure log.
    """
    env = envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    obs = env.reset()
    names = trainer.agent_names
    block_payoffs = {nm: [] for nm in names}
    soc_path = {nm: [float(next(a for a in env.all_agents
                                if a.name == nm).storage.soc0)]
                for nm in names}
    charge_mwh = {nm: 0.0 for nm in names}
    discharge_mwh = {nm: 0.0 for nm in names}
    gate_open = {nm: [0, 0] for nm in names}
    gate_prob_sum = {nm: [0.0, 0.0] for nm in names}
    gate_closed = {nm: [0, 0] for nm in names}
    quote_open = {nm: [[], []] for nm in names}
    mode_counts: Dict[int, int] = {}
    sat_upper = sat_lower = 0
    soc_min = {nm: math.inf for nm in names}
    soc_max = {nm: -math.inf for nm in names}
    decisions = {nm: 0 for nm in names}
    for block in range(24):
        acts, unit_modes = {}, []
        for nm in names:
            d = trainer.select_action(obs[nm], nm, train=False)
            acts[nm] = np.asarray(d["quotes"], dtype=float)
            unit_modes.append(int(d["mode"]))
            gates = np.asarray(d["gates"], dtype=float)
            probs = np.asarray(d["probs"], dtype=float)
            price_quotes = np.asarray(d["price_quotes"], dtype=float)
            decisions[nm] += 1
            for leg in (0, 1):
                gate_prob_sum[nm][leg] += float(probs[leg])
                if gates[leg] > 0:
                    gate_open[nm][leg] += 1
                    quote_open[nm][leg].append(float(price_quotes[leg]))
                    if price_quotes[leg] >= 0.99 * mod.LEG_HIGHS[leg]:
                        sat_upper += 1
                    if price_quotes[leg] <= 0.01 * mod.LEG_HIGHS[leg]:
                        sat_lower += 1
                else:
                    gate_closed[nm][leg] += 1
        mode = mod.fleet_mode(unit_modes)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        next_obs, _, done, info = env.step(acts)
        if env._last_result is None:
            raise RuntimeError(f"{pair}: clearing failed at block {block}")
        result = env._last_result
        n_commit = min(4, env.T - block * 4)
        for nm in names:
            block_payoffs[nm].append(
                float(info.get("raw_profit", {}).get(nm, 0.0)))
            soc_path[nm].append(float(env.prev_soc.get(nm, soc_path[nm][-1])))
            ws = (result.get("schedules") or {}).get(nm)
            if ws is None:
                continue
            charge_mwh[nm] += float(
                np.sum(np.asarray(ws["p_ch"])[:n_commit]) * 0.25)
            discharge_mwh[nm] += float(
                np.sum(np.asarray(ws["p_dis"])[:n_commit]) * 0.25)
            soc_arr = ws.get("soc")
            if soc_arr is not None:
                vals = np.asarray(soc_arr, dtype=float)[:n_commit + 1]
                if vals.size:
                    soc_min[nm] = min(soc_min[nm], float(np.min(vals)))
                    soc_max[nm] = max(soc_max[nm], float(np.max(vals)))
        obs = next_obs
        if done:
            break
    pair_id = f"{pair['load_date']}|{pair['price_date']}"
    b1_day = b1_cache.get(pair_id, 0.5, mechanism_hash,
                          mod.pair_data_hash(pair_row))
    G_pi = {nm: float(sum(block_payoffs[nm])) for nm in names}
    G_b1 = {nm: float(sum(b1_day.block_payoffs[nm])) for nm in names}
    delta = {nm: G_pi[nm] - G_b1[nm] for nm in names}
    e_max = {nm: float(next(a for a in env.all_agents
                            if a.name == nm).storage.e_max) for nm in names}
    cycles = {
        nm: ((charge_mwh[nm] + discharge_mwh[nm]) / (2.0 * e_max[nm])
             if e_max[nm] > 0 else 0.0)
        for nm in names}
    for nm in names:
        if not math.isfinite(soc_min[nm]):
            soc_min[nm] = min(soc_path[nm])
            soc_max[nm] = max(soc_path[nm])
    result = env._last_result
    voltage = result.get("bus_voltage")
    v = np.asarray(voltage, dtype=float) if voltage is not None else np.zeros(0)
    util_src = result.get("line_utilization")
    util = np.asarray(util_src, dtype=float) \
        if util_src is not None else np.zeros(0)
    return {
        "pair_id": pair_id,
        "season": pair.get("season", ""),
        "G_policy": G_pi,
        "G_b1": G_b1,
        "delta": delta,
        "delta_fleet": float(sum(delta.values())),
        "terminal_soc": {nm: soc_path[nm][-1] for nm in names},
        "charge_mwh": charge_mwh,
        "discharge_mwh": discharge_mwh,
        "cycles": cycles,
        "soc_min": soc_min,
        "soc_max": soc_max,
        "clears": 24,
        "lmp_fallbacks": int(result.get("lmp_fallbacks", 0)),
        "voltage_violation_cells": int(
            ((v < env.config.network.v_min_pu - 1e-7)
             | (v > env.config.network.v_max_pu + 1e-7)).sum())
        if v.size else 0,
        "line_overload_cells": int((util > 1.0 + 1e-7).sum())
        if util.size else 0,
        "gate_open_frequency": {
            nm: [gate_open[nm][leg] / max(decisions[nm], 1)
                 for leg in (0, 1)] for nm in names},
        "gate_mean_probability": {
            nm: [gate_prob_sum[nm][leg] / max(decisions[nm], 1)
                 for leg in (0, 1)] for nm in names},
        "gate_closed_fraction": {
            nm: [gate_closed[nm][leg] / max(decisions[nm], 1)
                 for leg in (0, 1)] for nm in names},
        "quote_open_values": quote_open,
        "gate_saturation_upper": sat_upper,
        "gate_saturation_lower": sat_lower,
        "mode_counts": {str(k): int(v) for k, v in sorted(mode_counts.items())},
        "distinct_modes": len(mode_counts),
    }


def _quote_stats(values: Sequence[float]):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {
        "count": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "p05": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def evaluate(trainer: mod.MATD3BC2Gate, pairs: Sequence[dict], b1_cache,
             *, mechanism_hash: str, pair_rows: Dict[str, dict]) -> dict:
    days = []
    for pair in pairs:
        pair_id = f"{pair['load_date']}|{pair['price_date']}"
        row = pair_rows[pair_id]
        days.append(eval_day(trainer, pair, b1_cache,
                             mechanism_hash=mechanism_hash, pair_row=row))
    units = trainer.agent_names
    fleet = [d["delta_fleet"] for d in days]
    per_unit = {nm: float(np.mean([d["delta"][nm] for d in days]))
                for nm in units}
    terminal_err = max(
        abs(v - 0.5) for d in days for v in d["terminal_soc"].values())
    voltage_cells = int(sum(d["voltage_violation_cells"] for d in days))
    line_cells = int(sum(d["line_overload_cells"] for d in days))
    c2 = mod.C2Config()
    raw_soc_min = min(d["soc_min"][nm] for d in days for nm in units)
    raw_soc_max = max(d["soc_max"][nm] for d in days for nm in units)
    raw_band_violation = max(0.0, c2.soc_lo - raw_soc_min,
                             raw_soc_max - c2.soc_hi)
    if terminal_err > 1e-5 or voltage_cells or line_cells \
            or raw_band_violation > 1e-6:
        raise RuntimeError(
            "validation/evaluation physical check failed: "
            f"terminal_err={terminal_err:.2e} voltage={voltage_cells} "
            f"line={line_cells} raw_soc_band={raw_band_violation:.2e}")
    gate_open = {
        f"{nm}.{leg}": float(np.mean([d["gate_open_frequency"][nm][i]
                                      for d in days]))
        for nm in units for i, leg in enumerate(("charge", "discharge"))}
    gate_prob = {
        f"{nm}.{leg}": float(np.mean([d["gate_mean_probability"][nm][i]
                                      for d in days]))
        for nm in units for i, leg in enumerate(("charge", "discharge"))}
    gate_closed = {
        f"{nm}.{leg}": float(np.mean([d["gate_closed_fraction"][nm][i]
                                      for d in days]))
        for nm in units for i, leg in enumerate(("charge", "discharge"))}
    quote_dist = {
        f"{nm}.{leg}": _quote_stats(
            [value for d in days for value in d["quote_open_values"][nm][i]])
        for nm in units for i, leg in enumerate(("charge", "discharge"))}
    modes: Dict[str, int] = {}
    for d in days:
        for key, count in d["mode_counts"].items():
            modes[key] = modes.get(key, 0) + int(count)
    return {
        "days": days,
        "mean_fleet_delta": float(np.mean(fleet)),
        "median_fleet_delta": float(np.median(fleet)),
        "mean_delta_by_unit": per_unit,
        "positive_days": int(sum(1 for x in fleet if x > 0)),
        "mean_charge_mwh": {nm: float(np.mean(
            [d["charge_mwh"][nm] for d in days])) for nm in units},
        "mean_discharge_mwh": {nm: float(np.mean(
            [d["discharge_mwh"][nm] for d in days])) for nm in units},
        "mean_cycles": {nm: float(np.mean(
            [d["cycles"][nm] for d in days])) for nm in units},
        "mean_terminal_soc": {nm: float(np.mean(
            [d["terminal_soc"][nm] for d in days])) for nm in units},
        "soc_min": raw_soc_min,
        "soc_max": raw_soc_max,
        "raw_soc_band_violation": raw_band_violation,
        "voltage_violation_cells": voltage_cells,
        "line_overload_cells": line_cells,
        "lmp_fallbacks": int(sum(d["lmp_fallbacks"] for d in days)),
        "terminal_soc_max_err": terminal_err,
        "gate": {
            "open_frequency": gate_open,
            "mean_probability": gate_prob,
            "closed_fraction": gate_closed,
            "quote_distribution": quote_dist,
            "mode_counts": {k: int(v) for k, v in sorted(
                modes.items(), key=lambda kv: int(kv[0]))},
            "distinct_modes": int(len(modes)),
            "permanent_open": sorted(
                k for k, v in gate_open.items() if v >= 1.0 - 1e-12),
            "permanent_closed": sorted(
                k for k, v in gate_open.items() if v <= 1e-12),
            "saturation_upper": int(sum(
                d["gate_saturation_upper"] for d in days)),
            "saturation_lower": int(sum(
                d["gate_saturation_lower"] for d in days)),
        },
    }


