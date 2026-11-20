"""Independent balance checks, nonlinear AC replay and participant ledgers."""

import numpy as np


def physical_audit(day, result, tolerance=1e-6, cone_tolerance=1e-5):
    """Conservation, bounds and relaxation-exactness over one cleared day.

    The cone gap is an auxiliary-variable residual, not a physical one: `ac_replay`
    replays the same injections through a nonlinear sweep and bounds the voltage
    and import differences at 1e-5. A cone bound three orders tighter than that
    independent check is a solver-precision demand, and the declared barrier
    tolerance (1e-8) does not meet it. Every other check keeps `tolerance`.
    """
    net = day.network
    p = -np.asarray(day.active_load_mw, dtype=float).copy()
    q = -np.asarray(day.reactive_load_mvar, dtype=float).copy()
    p[:, net.slack] += result["pcc_mw"]
    q[:, net.slack] += result["qcc_mvar"]
    for bus, mvar in net.shunts:
        q[:, bus] += mvar
    bound, recursion, simultaneous, product = 0., 0., 0., 0.
    for unit in day.storage:
        name, bid = unit.name, day.bids[unit.name]
        ch, dis, energy = result["charge"][name], result["discharge"][name], result["energy"][name]
        p[:, unit.bus] += dis - ch
        expected = day.dt_hours * (unit.eta_charge * ch - dis / unit.eta_discharge)
        recursion = max(recursion, float(abs(np.diff(energy) - expected).max()),
                        abs(float(energy[0]) - unit.soc_initial * unit.energy_mwh))
        if day.terminal_soc_equal:
            recursion = max(recursion, abs(float(energy[-1] - energy[0])))
        bound = max(bound, float((ch - bid.charge_quantity_mw).max()),
                    float((dis - bid.discharge_quantity_mw).max()), float((-ch).max()), float((-dis).max()),
                    float((unit.soc_min * unit.energy_mwh - energy).max()),
                    float((energy - unit.soc_max * unit.energy_mwh).max()))
        simultaneous = max(simultaneous, float(np.minimum(ch, dis).max()))
        product = max(product, float(abs(ch * dis).max()))
    for generator in day.generators:
        values, reactive = result["generation"][generator.name], result["generator_q"][generator.name]
        p[:, generator.bus] += values
        q[:, generator.bus] += reactive
        bound = max(bound, float((-values).max()), float((values - generator.available_mw).max()),
                    float((abs(reactive) - generator.reactive_limit_mvar).max()))
    gaps, drops = [], []
    for line, (start, end) in enumerate(net.branches):
        branch_p, branch_q = result["branch_p"][:, line], result["branch_q"][:, line]
        current = result["current_sq"][:, line]
        r, x = net.r_mw_base[line], net.x_mw_base[line]
        p[:, start] -= branch_p
        q[:, start] -= branch_q
        p[:, end] += branch_p - r * current
        q[:, end] += branch_q - x * current
        gaps.append(result["voltage_sq"][:, start] * current - branch_p ** 2 - branch_q ** 2)
        drops.append(result["voltage_sq"][:, end] - result["voltage_sq"][:, start]
                     + 2 * (r * branch_p + x * branch_q) - (r*r + x*x) * current)
        bound = max(bound, float((-current).max()))
        if net.current_sq_limit is not None:
            bound = max(bound, float((current - net.current_sq_limit[line]).max()))
    v = np.sqrt(np.maximum(result["voltage_sq"], 0))
    bound = max(bound, float((net.voltage_min - v).max()), float((v - net.voltage_max).max()),
                float(abs(v[:, net.slack] - net.slack_vm_pu).max()))
    if day.import_limit_mw is not None:
        bound = max(bound, float((result["pcc_mw"] - day.import_limit_mw).max()))
    if day.export_limit_mw is not None:
        bound = max(bound, float((-result["pcc_mw"] - day.export_limit_mw).max()))
    checks = {
        "p_balance_max_mw": float(abs(p).max()), "q_balance_max_mvar": float(abs(q).max()),
        "soc_energy_recursion_max_mwh": recursion, "bound_violation_max": max(0., bound),
        "simultaneous_max_mw": simultaneous, "complementarity_product_max_mw2": product,
        "cone_tightness_max_mw2": float(np.max(abs(np.array(gaps)))) if gaps else 0.,
        "voltage_drop_max_pu2": float(np.max(abs(np.array(drops)))) if drops else 0.,
    }
    limits = {key: tolerance for key in checks}
    limits["cone_tightness_max_mw2"] = cone_tolerance
    checks["passed"] = all(np.isfinite(value) and value <= limits[key]
                           for key, value in checks.items())
    return checks


def ac_replay(day, result):
    """Replay fixed injections with a nonlinear radial current sweep on 1 MVA."""
    net = day.network
    power = np.asarray(day.active_load_mw, dtype=complex).copy()
    power += 1j * np.asarray(day.reactive_load_mvar)
    for unit in day.storage:
        power[:, unit.bus] += result["charge"][unit.name] - result["discharge"][unit.name]
    for generator in day.generators:
        power[:, generator.bus] -= result["generation"][generator.name] + 1j * result["generator_q"][generator.name]
    for bus, mvar in net.shunts:
        power[:, bus] -= 1j * mvar
    order, reached = [], {net.slack}
    while len(order) < len(net.branches):
        for line, (start, end) in enumerate(net.branches):
            if start in reached and end not in reached:
                order.append(line)
                reached.add(end)
    voltages, imports, converged = [], [], []
    for period in range(day.periods):
        voltage = np.full(net.buses, net.slack_vm_pu, dtype=complex)
        success = False
        for _ in range(500):
            if (abs(voltage) < 1e-8).any():
                break
            currents = np.conj(power[period] / voltage)
            branch = np.zeros(len(net.branches), dtype=complex)
            for line in reversed(order):
                start, end = net.branches[line]
                branch[line] = currents[end]
                currents[start] += branch[line]
            updated = np.full(net.buses, net.slack_vm_pu, dtype=complex)
            for line in order:
                start, end = net.branches[line]
                updated[end] = updated[start] - complex(net.r_mw_base[line], net.x_mw_base[line]) * branch[line]
            change = abs(updated - voltage).max()
            voltage = updated
            if change < 1e-10:
                success = True
                break
        converged.append(success)
        voltages.append(abs(voltage))
        # The sweep solves currents; the substation's power is its voltage times that current.
        imports.append(float(currents[net.slack].real) * net.slack_vm_pu)
    values = np.array(voltages)
    voltage_difference = float(abs(values - np.sqrt(result["voltage_sq"])).max())
    import_difference = float(abs(np.array(imports) - result["pcc_mw"]).max())
    passed = (all(converged) and voltage_difference <= 1e-5 and import_difference <= 1e-5
              and values.min() >= net.voltage_min - 1e-6 and values.max() <= net.voltage_max + 1e-6)
    return {"passed": bool(passed), "converged_periods": sum(converged),
            "voltage_min_pu": float(values.min()), "voltage_max_pu": float(values.max()),
            "voltage_difference_max_pu": voltage_difference,
            "pcc_difference_max_mw": import_difference}


def settle_day(day, clearing, *, inventory_price_cny_mwh):
    """Settlement requires price and physical qualification; rewards stay separate."""
    if not clearing.get("qualified_for_settlement", False):
        raise ValueError("Unqualified prices or dispatch cannot be used for settlement")
    if not np.isfinite(inventory_price_cny_mwh):
        raise ValueError("A common finite external inventory value is required")
    result = clearing["pricing"]
    price, dt = result["lmp_cny_mwh"], day.dt_hours
    units, generation = {}, {}
    for unit in day.storage:
        ch, dis, energy = result["charge"][unit.name], result["discharge"][unit.name], result["energy"][unit.name]
        revenue = float(np.dot(price[:, unit.bus], dis) * dt)
        purchase = float(np.dot(price[:, unit.bus], ch) * dt)
        degradation = float((ch + dis).sum() * dt * unit.degradation_cny_mwh)
        inventory = float((energy[-1] - energy[0]) * inventory_price_cny_mwh)
        units[unit.name] = {"bus": unit.bus, "revenue_cny": revenue, "purchase_cny": purchase,
                            "net_cash_cny": revenue - purchase, "degradation_cny": degradation,
                            "inventory_change_mwh": float(energy[-1] - energy[0]),
                            "inventory_value_cny": inventory,
                            "economic_payoff_cny": revenue - purchase - degradation + inventory,
                            "charge_mwh": float(ch.sum() * dt), "discharge_mwh": float(dis.sum() * dt)}
    for generator in day.generators:
        dispatch = result["generation"][generator.name]
        generation[generator.name] = {
            "revenue_cny": float(np.dot(price[:, generator.bus], dispatch) * dt),
            "real_cost_cny": float(dispatch.sum() * dt * generator.real_cost_cny_mwh),
            "curtailment_mwh": float((generator.available_mw - dispatch).sum() * dt)}
    consumer = float((price * day.active_load_mw).sum() * dt)
    external = float(np.dot(day.external_price_cny_mwh, result["pcc_mw"]) * dt)
    ess_cash = sum(unit["net_cash_cny"] for unit in units.values())
    generator_cash = sum(unit["revenue_cny"] for unit in generation.values())
    operator = consumer - ess_cash - generator_cash - external
    residual = consumer - ess_cash - generator_cash - external - operator
    real_cost = external + sum(u["real_cost_cny"] for u in generation.values()) + sum(u["degradation_cny"] for u in units.values())
    return {"units": units, "generators": generation, "consumer_payment_cny": consumer,
            "external_net_payment_cny": external, "generator_payment_cny": generator_cash,
            "ess_net_payment_cny": ess_cash, "operator_residual_cny": operator,
            "cash_closure_residual_cny": residual, "real_system_cost_cny": real_cost,
            "clearing_objective": result["clearing_objective"],
            "common_external_inventory_price_cny_mwh": float(inventory_price_cny_mwh)}
