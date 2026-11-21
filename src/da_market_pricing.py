"""Qualify direction-conditional prices without changing the dispatched award."""

import numpy as np

from da_market_model import MarketModel, audit
from da_market_conic import ConicMarketModel
from da_market_cutting import CuttingMarketModel
from local_da_clearing import choose_directions


def _log(folder,name):
    return None if folder is None else folder/name


def _valid(result):
    return result.get("optimal",False) and np.isfinite(result.get("lmp_cny_mwh",np.array([np.nan]))).all()


def _backend(day,ratings,rules,name,**kw):
    return {"cutting":lambda:CuttingMarketModel(day,ratings,rules,**kw),
            "conic":lambda:ConicMarketModel(day,ratings,rules,**kw),
            "relaxed":lambda:MarketModel(day,ratings,rules,exact=False,**kw)}[name]()


def qualify_prices(day,ratings,rules,dispatch,*,folder=None,backend="cutting"):
    if backend not in {"cutting","conic","relaxed"}:
        raise ValueError("An explicit pricing backend is required")
    output={"qualified_for_settlement":False,"price_qualification":"NOT_TESTED", "dispatch":dispatch,
            "pricing_backend":backend,
            "price_basis":"Direction-conditional active-demand marginal price with Q fixed"}
    if not dispatch.get("optimal"):
        output["failure"]="Dispatch not optimal"
        return output
    output["dispatch_audit"]=audit(day,ratings,rules,dispatch)
    if not output["dispatch_audit"]["passed"]:
        output["failure"]="Dispatch physical qualification failed"
        return output
    directions,idle=choose_directions(dispatch)
    tolerance=max(.01,abs(dispatch["clearing_objective"])*1e-6)
    model=_backend(day,ratings,rules,backend,directions=directions,log_path=_log(folder,"pricing.log"))
    try:
        pricing=model.solve(prices=True)
        output["pricing"]=pricing
        if not _valid(pricing):
            output["failure"]="Conditional prices unavailable or nonoptimal"
            return output
        output["pricing_audit"]=audit(day,ratings,rules,pricing)
        gap=abs(pricing["clearing_objective"]-dispatch["clearing_objective"])
        output["conditional_objective_difference_cny"]=gap
        if not output["pricing_audit"]["passed"] or gap>tolerance:
            output["failure"]="Conditional physical or objective consistency failed"
            return output
        voltage=np.sqrt(pricing["voltage_sq"])
        low,high=day.network.voltage_min,day.network.voltage_max
        binding=(voltage<=low+1e-6)|(voltage>=high-1e-6)
        periods={int(pricing["pcc_mw"].argmax())}
        times=np.flatnonzero(binding.any(axis=1))
        if len(times):
            periods.add(int(times[np.argmin(voltage[times].min(axis=1))]))
        buses={day.network.slack,*(u.bus for u in day.storage)}
        if any(g.name=="EXTERNAL33" for g in day.generators):
            buses.add(rules.external_q_bus)
        for t in periods:
            buses.update(int(b) for b in np.flatnonzero(binding[t]))
        records=[]
        for t in sorted(periods):
            for b in sorted(buses):
                constraint=model.balance[t,b]
                original=constraint.RHS
                dual=float(pricing["lmp_cny_mwh"][t,b])
                ptol=max(.1,abs(dual)*.01)
                for epsilon in [1e-4,1e-5]:
                    slopes={}
                    statuses={}
                    try:
                        for sign,name in [(-1,"left"),(1,"right")]:
                            constraint.RHS=original+sign*epsilon
                            result=model.solve()
                            statuses[name]=result["solver_status_name"]
                            slopes[name]=(result["clearing_objective"]-pricing["clearing_objective"])/(sign*epsilon*day.dt_hours) if result["optimal"] else None
                    finally:
                        constraint.RHS=original
                    both=all(value is not None and np.isfinite(value) for value in slopes.values())
                    smooth=both and all(abs(value-dual)<=ptol for value in slopes.values())
                    bracket=both and slopes["left"]-ptol<=dual<=slopes["right"]+ptol
                    records.append({"period":t,"bus":b,"epsilon_mw":epsilon,"dual_cny_mwh":dual,
                                    "left_cny_mwh":slopes["left"],"right_cny_mwh":slopes["right"],
                                    "statuses":statuses,"smooth_pass":bool(smooth),"subgradient_pass":bool(bracket)})
        output["price_probes"]=records
        output["probe_scope"]={"periods":sorted(periods),"buses_zero_based":sorted(buses),"all_96_derivatives_tested":False}
        if all(r["smooth_pass"] for r in records):
            output["price_qualification"]="SMOOTH_PASS"
        elif all(r["subgradient_pass"] for r in records):
            output["price_qualification"]="SUBGRADIENT_PASS"
        else:
            output["price_qualification"]="FAILED"
    finally:
        model.close()
    other_directions={n:np.where(idle[n],-1,values) for n,values in directions.items()}
    stable=True
    if any(mask.any() for mask in idle.values()):
        model=_backend(day,ratings,rules,backend,directions=other_directions,
                       log_path=_log(folder,"idle_alternative.log"))
        try:
            other=model.solve(prices=True)
        finally:
            model.close()
        output["idle_alternative"]=other
        stable=_valid(other)
        if stable:
            checked=audit(day,ratings,rules,other)
            output["idle_alternative_audit"]=checked
            delta=abs(other["lmp_cny_mwh"]-pricing["lmp_cny_mwh"])
            output["idle_price_difference_max_cny_mwh"]=float(delta.max())
            stable=bool(checked["passed"] and abs(other["clearing_objective"]-pricing["clearing_objective"])<=tolerance
                        and (delta<=np.maximum(.1,abs(pricing["lmp_cny_mwh"])*.01)).all())
    output["idle_direction_stable"]=bool(stable)
    output["conditional_schedule_difference_max_mw"]=max([float(abs(pricing[leg][n]-dispatch[leg][n]).max())
                 for leg in ["charge","discharge"] for n in day.bids] or [0.])
    output["qualified_for_settlement"]=bool(stable and output["price_qualification"] in {"SMOOTH_PASS","SUBGRADIENT_PASS"})
    if not output["qualified_for_settlement"]:
        output["failure"]="Marginal-price or idle-direction qualification failed"
    return output
