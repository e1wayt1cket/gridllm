"""Size the two conventional remedies against the worst cleared hour.

Neither remedy exists in the market model: the slack is pinned at 1.00 pu and
the network carries no shunt. This probe measures both on the same load, with
the fleet at available output and every battery charging, which is the hour the
clearing actually has to survive.
"""

import grid  # noqa: F401  ortools must load before pandas on this platform
import numpy as np
import pandapower as pp
import pandapower.networks as networks

from run_local_da import build_day

DAYS = ["2022-10-14", "2021-09-20"]


def voltage(benchmark, p, q, gen, charge, slack=1., shunts=None):
    net = networks.case33bw()
    loads = net.load.bus.values.astype(int)
    net.ext_grid.vm_pu = slack
    net.load.p_mw = np.asarray(p, float)[loads]
    net.load.q_mvar = np.asarray(q, float)[loads]
    for bus, mw in gen.items():
        pp.create_sgen(net, bus, p_mw=mw, q_mvar=0.)
    for bus, mw in charge.items():
        pp.create_load(net, bus, p_mw=mw, q_mvar=0.)
    for bus, mvar in (shunts or {}).items():
        pp.create_shunt(net, bus, q_mvar=-mvar, p_mw=0.)
    pp.runpp(net, numba=False)
    return float(net.res_bus.vm_pu.min()), float(net.res_bus.vm_pu.max())


def main():
    for date in DAYS:
        day, _ = build_day(date)
        t = int(np.argmax(day.active_load_mw.sum(axis=1)))
        p, q = day.active_load_mw[t], day.reactive_load_mvar[t]
        gen = {g.bus: float(g.available_mw[t]) for g in day.generators}
        charge = {unit.bus: .4 for unit in day.storage}
        print(f"\n{date} peak period {t}: {p.sum():.3f} MW, {q.sum():.3f} MVAr")
        for slack in [1., 1.02, 1.03, 1.04]:
            lo, hi = voltage(None, p, q, gen, charge, slack=slack)
            print(f"  slack {slack:.2f} pu                        Vmin {lo:.4f}  Vmax {hi:.4f}")
        for total in [.5, 1., 1.5, 2.]:
            lo, hi = voltage(None, p, q, gen, charge, shunts={17: total / 2, 30: total / 2})
            print(f"  {total:.1f} MVAr capacitors at 17 and 30   Vmin {lo:.4f}  Vmax {hi:.4f}")
        lo, hi = voltage(None, p, q, gen, charge, slack=1.02, shunts={17: .75, 30: .75})
        print(f"  slack 1.02 + 1.5 MVAr capacitors       Vmin {lo:.4f}  Vmax {hi:.4f}")
        lo, hi = voltage(None, p, q, gen, charge, slack=1.04, shunts={17: .75, 30: .75})
        print(f"  slack 1.04 + 1.5 MVAr capacitors       Vmin {lo:.4f}  Vmax {hi:.4f}")


if __name__ == "__main__":
    main()
