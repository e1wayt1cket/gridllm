"""LP supporting-plane solution of the unchanged conditional second-order cones."""

import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from da_market_model import MarketModel


class CuttingMarketModel(MarketModel):
    def __init__(self,day,ratings,rules,*,directions,log_path=None):
        super().__init__(day,ratings,rules,exact=False,directions=directions,log_path=log_path)
        m=self.model
        m.remove(m.getQConstrs())
        self.cone_vectors=[]
        for t in range(day.periods):
            for l,(a,_) in enumerate(day.network.branches):
                self.cone_vectors.append((self.v[t,a]+self.i[t,l],
                    [2*self.p[t,l],2*self.q[t,l],self.v[t,a]-self.i[t,l]]))
            for g in day.generators:
                if g.name not in ratings:
                    continue
                p=self.generation[g.name][t]
                if g.name.startswith('Q_'):
                    n=g.name[2:]
                    p=self.discharge[n][t]-self.charge[n][t]
                self.cone_vectors.append((gp.LinExpr(ratings[g.name]),[p,self.generator_q[g.name][t]]))
        for radius,vector in self.cone_vectors:
            for term in vector:
                m.addConstr(term<=radius)
                m.addConstr(-term<=radius)
        m.Params.Method=1
        m.Params.FeasibilityTol=1e-9
        m.Params.OptimalityTol=1e-9
        m.update()

    def solve(self,*,prices=False):
        start=time.monotonic()
        maximum=float('inf')
        for iteration in range(150):
            remaining=self.execution.seconds-(time.monotonic()-start)
            if remaining<=0:
                break
            self.model.Params.TimeLimit=remaining
            result=super().solve(prices=prices)
            if not result['optimal']:
                return result
            maximum=0.
            cuts=0
            for radius,vector in self.cone_vectors:
                values=np.array([term.X if isinstance(term,gp.Var) else term.getValue() for term in vector])
                length=float(np.linalg.norm(values))
                rv=radius.X if isinstance(radius,gp.Var) else radius.getValue()
                gap=length-rv
                maximum=max(maximum,gap)
                if gap>1e-10:
                    self.model.addConstr(1000*gp.quicksum(float(a/length)*term for a,term in zip(values,vector))<=1000*radius)
                    cuts+=1
            if not cuts:
                result.update(cutting_iterations=iteration+1,cone_distance_max=maximum,
                              solver_backend='gurobi-lp-cone-cuts',runtime_seconds=time.monotonic()-start)
                return result
            self.model.update()
        return {'solver_status':12,'solver_status_name':'CONE_CUT_LIMIT','optimal':False,'solution_count':0,
                'runtime_seconds':time.monotonic()-start,'cone_distance_max':maximum}
