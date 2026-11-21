"""Which input moves the case33bw benchmark's 0.9131 pu minimum voltage."""

import numpy as np
import pandapower as pp
import pandapower.networks as networks


def solve(slack=1., load_scale=1., q_ratio=None, shunt_mvar=0., shunt_buses=(17,),
          impedance=None, sgen_mw=0.):
    net = networks.case33bw()
    net.ext_grid.vm_pu = slack
    net.load.p_mw *= load_scale
    net.load.q_mvar *= load_scale
    if q_ratio is not None:
        net.load.q_mvar = net.load.p_mw * q_ratio
    if impedance is not None:
        net.line.r_ohm_per_km *= impedance
        net.line.x_ohm_per_km *= impedance
    for bus in shunt_buses:
        if shunt_mvar:
            pp.create_shunt(net, bus=bus, q_mvar=-shunt_mvar / len(shunt_buses), p_mw=0.)
    if sgen_mw:
        pp.create_sgen(net, bus=17, p_mw=sgen_mw, q_mvar=0.)
    pp.runpp(net, numba=False)
    volts = net.res_bus.vm_pu
    return float(volts.min()), int(volts.idxmin()), float(net.res_line.pl_mw.sum())


def row(name, **kw):
    minimum, bus, loss = solve(**kw)
    print(f"{name:<34} {minimum:.4f}  bus {bus:>2}  losses {loss:.4f} MW")


print("nominal case33bw: 3.715 MW / 2.300 MVAr, slack 1.00, no compensation")
row("nominal")
print()
for s in [0.98, 1.00, 1.02, 1.04]:
    row(f"slack {s:.2f} pu", slack=s)
print()
for s in [0.85, 0.90, 1.00, 1.10]:
    row(f"load scale {s:.2f}", load_scale=s)
print()
for r in [0.619, 0.50, 0.40, 0.20, 0.]:
    row(f"Q/P {r:.3f}", q_ratio=r)
print()
for q in [0.5, 1.0, 1.5]:
    row(f"capacitor {q:.1f} MVAr at bus 17", shunt_mvar=q)
row("capacitor 1.0 MVAr split 17+30", shunt_mvar=1.0, shunt_buses=(17, 30))
print()
for z in [0.8, 1.0, 1.2]:
    row(f"line r,x scale {z:.1f}", impedance=z)
print()
for g in [0.3, 0.6]:
    row(f"0.3 MW sgen at bus 17 ({g:.1f})", sgen_mw=g)
print("\ntarget: 0.9300 pu")
