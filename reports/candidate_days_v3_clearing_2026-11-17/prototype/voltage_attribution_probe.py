"""Split the voltage drop between the load level and its reactive draw.

Each sweep moves one input at the hardest hour of the day: the whole load,
the reactive load alone, the active load alone. The generators stay at their
available output, so the comparison isolates the load model.
"""

import grid  # noqa: F401  ortools must load before pandas on this platform
import numpy as np
import pandapower as pp
import pandapower.networks as networks

from run_local_da import build_day

DAYS = ["2022-10-14", "2021-09-20"]


def voltage(benchmark, p, q, gen):
    net = networks.case33bw()
    loads = net.load.bus.values.astype(int)
    net.load.p_mw = np.asarray(p, float)[loads]
    net.load.q_mvar = np.asarray(q, float)[loads]
    for bus, mw in gen.items():
        pp.create_sgen(net, bus, p_mw=mw, q_mvar=0.)
    pp.runpp(net, numba=False)
    return float(net.res_bus.vm_pu.min())


def main():
    for date in DAYS:
        day, _ = build_day(date)
        t = int(np.argmax(day.active_load_mw.sum(axis=1)))
        p, q = day.active_load_mw[t], day.reactive_load_mvar[t]
        gen = {g.bus: float(g.available_mw[t]) for g in day.generators}
        print(f"\n{date} peak period {t}: {p.sum():.3f} MW, {q.sum():.3f} MVAr")
        print("  P and Q together: " + " ".join(
            f"{s:.2f}x:{voltage(None, p * s, q * s, gen):.4f}" for s in [1.05, 1., .95, .9, .85]))
        print("  reactive only:    " + " ".join(
            f"{f:.2f}x:{voltage(None, p, q * f, gen):.4f}" for f in [1.2, 1., .75, .5, .25, 0.]))
        print("  active only:      " + " ".join(
            f"{s:.2f}x:{voltage(None, p * s, q, gen):.4f}" for s in [1., .95, .9, .85]))


if __name__ == "__main__":
    main()
