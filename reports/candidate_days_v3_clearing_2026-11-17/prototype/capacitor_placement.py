"""Where a fixed capacitor buys the most voltage, and how much is needed."""

import numpy as np
import pandapower as pp
import pandapower.networks as networks


def solve(shunts=()):
    net = networks.case33bw()
    for bus, mvar in shunts:
        pp.create_shunt(net, bus=bus, q_mvar=-mvar, p_mw=0.)
    pp.runpp(net, numba=False)
    volts = net.res_bus.vm_pu
    return float(volts.min()), int(volts.idxmin()), float(net.res_line.pl_mw.sum())


print("single 1.0 MVAr capacitor, one bus at a time")
for bus in [6, 12, 16, 17, 24, 25, 28, 30, 31, 32]:
    minimum, worst, loss = solve([(bus, 1.)])
    print(f"  bus {bus:>2}: Vmin {minimum:.4f} at bus {worst:>2}, losses {loss:.4f} MW")

print("\nsplit across buses 17 and 32")
for total in [1., 1.5, 2., 2.5, 3.]:
    minimum, worst, loss = solve([(17, total / 2), (32, total / 2)])
    print(f"  {total:.1f} MVAr: Vmin {minimum:.4f} at bus {worst:>2}, losses {loss:.4f} MW")

print("\nbest single-bus placement at several sizes")
for bus in [17, 32]:
    for total in [1., 2., 3.]:
        minimum, worst, loss = solve([(bus, total)])
        print(f"  bus {bus} {total:.1f} MVAr: Vmin {minimum:.4f} at {worst:>2}, losses {loss:.4f}")
