"""Equivalent conic expressions and normalized currency for stable DA prices."""

import numpy as np
from gurobipy import GRB

from local_da_operating import OperatingModel, OperatingRules, audit_operating


class MarketModel(OperatingModel):
    def __init__(self,day,ratings,rules,*,exact=True,directions=None,log_path=None):
        rules.validate(day)
        self.execution=rules
        super().__init__(day,OperatingRules(),ratings,exact=exact,directions=directions,
                         seconds=rules.seconds,log_path=log_path)
        m=self.model
        if any(g.name=="EXTERNAL33" for g in day.generators):
            for t in range(day.periods):
                self.generator_q["EXTERNAL33"][t].LB=0.
                self.generator_q["EXTERNAL33"][t].UB=rules.external_q_capacity_mvar
        if not exact and rules.cone_form=="standard":
            m.remove([c for c in m.getQConstrs() if c.QCName.startswith("cone_")])
            for t in range(day.periods):
                for l,(a,_) in enumerate(day.network.branches):
                    radius=m.addVar(lb=day.network.voltage_min**2,name=f"cone_radius_{t}_{l}")
                    axis=m.addVar(lb=-GRB.INFINITY,name=f"cone_axis_{t}_{l}")
                    m.addConstr(radius==self.v[t,a]+self.i[t,l])
                    m.addConstr(axis==self.v[t,a]-self.i[t,l])
                    m.addQConstr(4*self.p[t,l]*self.p[t,l]+4*self.q[t,l]*self.q[t,l]+axis*axis<=radius*radius,
                                 name=f"standard_cone_{t}_{l}")
        m.setObjective(self.market_objective/rules.objective_scale,GRB.MINIMIZE)
        if exact:
            m.Params.MIPGapAbs=.001/rules.objective_scale
        m.update()

    def solve(self,*,prices=False):
        result=super().solve(prices=prices)
        scale=self.execution.objective_scale
        for key in ["clearing_objective","objective_bound"]:
            if key in result:
                result[key]*=scale
        if "lmp_cny_mwh" in result:
            result["lmp_cny_mwh"]*=scale
        return result


def audit(day,ratings,rules,result):
    report=audit_operating(day,OperatingRules(),ratings,result)
    values=result.get("generator_q",{}).get("EXTERNAL33")
    extra=0. if values is None else max(0.,float(-values.min()),float(values.max()-rules.external_q_capacity_mvar))
    report["capacitive_q_violation_mvar"]=extra
    report["passed"]=bool(report["passed"] and extra<=1e-6 and result["shed_mwh"]<=1e-8)
    return report
