"""Why the cleared voltage sits on the lower bound.

Three questions, each answered from the network rather than from the clearing:
what the benchmark itself does at nominal load, which component drives the sag
at the library's peak period, and what minimum the market would choose if the
band did not stop it.
"""

import dataclasses

import grid  # noqa: F401  ortools must load before pandas on this platform
import numpy as np
import pandapower as pp
import pandapower.networks as networks

from local_da_clearing import clear_day
from run_local_da import build_day

DAYS = ["2022-10-14", "2021-09-20", "2021-06-04", "2021-10-14"]


def power_flow(benchmark, p, q, gen=None, extra=None):
    """Voltage of one dispatch; p and q are indexed by the 33 bus numbers."""
    net = networks.case33bw()
    loads = net.load.bus.values.astype(int)
    net.load.p_mw = np.asarray(p, float)[loads]
    net.load.q_mvar = np.asarray(q, float)[loads]
    for bus, mw in (gen or {}).items():
        pp.create_sgen(net, bus, p_mw=mw, q_mvar=0.)
    for bus, mw in (extra or {}).items():
        pp.create_load(net, bus, p_mw=mw, q_mvar=0.)
    pp.runpp(net, numba=False)
    return float(net.res_bus.vm_pu.min())


def main():
    net = networks.case33bw()
    loads = net.load.bus.values.astype(int)
    p0, q0 = np.zeros(33), np.zeros(33)
    p0[loads], q0[loads] = net.load.p_mw.values, net.load.q_mvar.values
    native_ratio = np.divide(q0, p0, out=np.zeros(33), where=p0 > 0)
    print(f"no market, benchmark at its own {p0.sum():.3f} MW and {q0.sum():.3f} MVAr "
          f"(aggregate Q/P {q0.sum() / p0.sum():.3f}): Vmin {power_flow(None, p0, q0):.4f}")
    print(f"reactive equipment in the network: {len(net.shunt)} shunts, {len(net.gen)} "
          f"generators, {len(net.trafo)} transformers, slack {net.ext_grid.vm_pu.values[0]:.2f} pu")

    for date in DAYS:
        day, _ = build_day(date)
        t = int(np.argmax(day.active_load_mw.sum(axis=1)))
        p, q = day.active_load_mw[t], day.reactive_load_mvar[t]
        gen = {g.bus: float(g.available_mw[t]) for g in day.generators}
        charge = {u.bus: .4 for u in day.storage}
        discharge = {u.bus: -.4 for u in day.storage}
        print(f"\n{date} peak period {t}: {p.sum():.3f} MW, {q.sum():.3f} MVAr, "
              f"renewables {sum(gen.values()):.3f} MW")
        for label, args in [("load alone, declared Q = 0.619 P", (p, q)),
                            ("load alone, native per-bus Q/P", (p, p * native_ratio)),
                            ("+ renewables at available output", (p, q, gen)),
                            ("+ storage charging 4 x 0.4 MW", (p, q, gen, charge)),
                            ("+ storage discharging 4 x 0.4 MW", (p, q, gen, discharge))]:
            print(f"  {label:34s} Vmin {power_flow(None, *args):.4f}")
        loose = dataclasses.replace(day, network=dataclasses.replace(day.network,
                                                                     voltage_min=.80))
        free = clear_day(loose, seconds=120, price_probes=False)["dispatch"]
        volts = np.sqrt(free["voltage_sq"])
        print(f"  market with the band opened to 0.80       Vmin {volts.min():.4f} "
              f"(median {np.median(volts):.4f})")


if __name__ == "__main__":
    main()
