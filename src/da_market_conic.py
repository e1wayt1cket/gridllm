"""Native sparse conic backend for the same direction-conditional market."""

import time

import clarabel
import numpy as np
from scipy import sparse

from da_market_model import MarketModel


class ConicMarketModel:
    def __init__(self,day,ratings,rules,*,directions,log_path=None):
        self.day,self.rules,self.ratings=day,rules,ratings
        self.log_path=log_path
        self.base=MarketModel(day,ratings,rules,exact=False,directions=directions)
        m=self.base.model
        variables=m.getVars()
        self.positions={v.VarName:i for i,v in enumerate(variables)}
        self.count=len(variables)
        self.objective=np.array([v.Obj for v in variables])
        self.constant=m.ObjCon
        rows,bounds,cones=[],[],[]
        self.row_by_name={}

        def row(terms,rhs,cone):
            index=len(rows)
            rows.append(terms)
            bounds.append(float(rhs))
            cones.append(cone)
            return index

        for c in m.getConstrs():
            expression=m.getRow(c)
            terms={self.positions[expression.getVar(i).VarName]:expression.getCoeff(i) for i in range(expression.size())}
            sign=-1 if c.Sense=='>' else 1
            terms={k:sign*v for k,v in terms.items()}
            self.row_by_name[c.ConstrName]=row(terms,sign*c.RHS,
                clarabel.ZeroConeT(1) if c.Sense=='=' else clarabel.NonnegativeConeT(1))
        for i,v in enumerate(variables):
            if v.LB==v.UB:
                row({i:1.},v.LB,clarabel.ZeroConeT(1))
            else:
                if v.LB>-1e20:
                    row({i:-1.},-v.LB,clarabel.NonnegativeConeT(1))
                if v.UB<1e20:
                    row({i:1.},v.UB,clarabel.NonnegativeConeT(1))

        def pos(v):
            return self.positions[v.VarName]

        for t in range(day.periods):
            for l,(a,_) in enumerate(day.network.branches):
                v,i,p,q=[pos(x) for x in [self.base.v[t,a],self.base.i[t,l],self.base.p[t,l],self.base.q[t,l]]]
                rows.extend([{v:-1.,i:-1.},{p:-2.},{q:-2.},{v:-1.,i:1.}])
                bounds.extend([0.]*4)
                cones.append(clarabel.SecondOrderConeT(4))
            for g in day.generators:
                if g.name not in ratings:
                    continue
                reactive=pos(self.base.generator_q[g.name][t])
                if g.name.startswith('Q_'):
                    n=g.name[2:]
                    active={pos(self.base.discharge[n][t]):-1.,pos(self.base.charge[n][t]):1.}
                else:
                    active={pos(self.base.generation[g.name][t]):-1.}
                rows.extend([{},active,{reactive:-1.}])
                bounds.extend([ratings[g.name],0.,0.])
                cones.append(clarabel.SecondOrderConeT(3))
        rr,cc,vv=[],[],[]
        for i,terms in enumerate(rows):
            for j,value in terms.items():
                rr.append(i)
                cc.append(j)
                vv.append(value)
        self.matrix=sparse.csc_matrix((vv,(rr,cc)),shape=(len(rows),self.count))
        self.rhs=np.array(bounds)
        self.cones=cones
        fixed={i:float(v.LB) for i,v in enumerate(variables) if v.LB==v.UB}
        equalities=[]
        for c in m.getConstrs():
            if c.Sense!='=' or c.ConstrName.startswith(('active_balance_','reactive_balance_')):
                continue
            expression=m.getRow(c)
            equalities.append(({self.positions[expression.getVar(i).VarName]:expression.getCoeff(i)
                                for i in range(expression.size())},float(c.RHS)))
        changed=True
        while changed:
            changed=False
            for terms,rhs in equalities:
                unknown={j:v for j,v in terms.items() if j not in fixed and v!=0}
                if len(unknown)==1:
                    j,coefficient=next(iter(unknown.items()))
                    fixed[j]=(rhs-sum(v*fixed[k] for k,v in terms.items() if k in fixed))/coefficient
                    changed=True
        self.fixed=fixed
        self.free=np.array([j for j in range(self.count) if j not in fixed],dtype=int)
        self.fixed_vector=np.array([fixed.get(j,0.) for j in range(self.count)])
        self.reduced_matrix=self.matrix[:,self.free].tocsc()
        self.reduced_rhs=self.rhs-self.matrix@self.fixed_vector
        self.reduced_objective=self.objective[self.free]
        self.reduced_constant=self.constant+float(self.objective@self.fixed_vector)
        keep=[]
        reduced_cones=[]
        offset=0
        mutable_rows={self.row_by_name[c.ConstrName] for c in self.base.balance.values()}
        for cone in self.cones:
            dimension=cone.dim
            drop=False
            if dimension==1 and self.reduced_matrix[offset].nnz==0 and offset not in mutable_rows:
                if isinstance(cone,clarabel.ZeroConeT):
                    drop=abs(self.reduced_rhs[offset])<1e-10
                elif isinstance(cone,clarabel.NonnegativeConeT):
                    drop=self.reduced_rhs[offset]>=-1e-10
            if not drop:
                keep.extend(range(offset,offset+dimension))
                reduced_cones.append(cone)
            offset+=dimension
        self.keep=np.array(keep,dtype=int)
        self.row_reduced={original:i for i,original in enumerate(keep)}
        self.reduced_matrix=self.reduced_matrix[self.keep,:].tocsc()
        self.reduced_cones=reduced_cones
        self.balance=self.base.balance
        settings=clarabel.DefaultSettings()
        settings.verbose=False
        settings.max_iter=300
        settings.time_limit=rules.seconds
        settings.tol_gap_abs=1e-10
        settings.tol_gap_rel=1e-10
        settings.tol_feas=1e-10
        settings.static_regularization_constant=1e-12
        settings.iterative_refinement_abstol=1e-14
        settings.iterative_refinement_reltol=1e-14
        settings.iterative_refinement_max_iter=20
        self.settings=settings

    def solve(self,*,prices=False):
        self.base.model.update()
        for (t,b),constraint in self.balance.items():
            self.rhs[self.row_by_name[constraint.ConstrName]]=constraint.RHS
        started=time.monotonic()
        rhs=self.rhs-self.matrix@self.fixed_vector
        solver=clarabel.DefaultSolver(sparse.csc_matrix((len(self.free),len(self.free))),self.reduced_objective,
                                    self.reduced_matrix,rhs[self.keep],self.reduced_cones,self.settings)
        solution=solver.solve()
        status=str(solution.status)
        optimal=status=='Solved'
        result={"solver_status":2 if optimal else 12,"solver_status_name":"OPTIMAL" if optimal else status,
                "optimal":optimal,"solution_count":int(optimal),"runtime_seconds":time.monotonic()-started,
                "declared_integer_variables":0,"declared_binary_variables":0,"solver_backend":"clarabel-0.11.1",
                "conic_primal_residual":float(solution.r_prim),"conic_dual_residual":float(solution.r_dual)}
        if self.log_path:
            with self.log_path.open('a',encoding='utf-8') as stream:
                stream.write(f"status={status} iterations={solution.iterations} primal={solution.obj_val} dual={solution.obj_val_dual}\n")
        if not optimal:
            return result
        x=self.fixed_vector.copy()
        x[self.free]=solution.x

        def value(v):
            return x[self.positions[v.VarName]]

        def curve(variables):
            return np.array([value(variables[t]) for t in range(len(variables))])

        base,day=self.base,self.day
        result.update(clearing_objective=float((self.objective@x+self.constant)*self.rules.objective_scale),
                      objective_bound=float((solution.obj_val_dual+self.reduced_constant)*self.rules.objective_scale),
                      pcc_mw=curve(base.pcc),qcc_mvar=curve(base.qcc),
                      voltage_sq=np.array([[value(base.v[t,b]) for b in range(day.network.buses)] for t in range(day.periods)]),
                      branch_p=np.array([[value(base.p[t,l]) for l in range(len(day.network.branches))] for t in range(day.periods)]),
                      branch_q=np.array([[value(base.q[t,l]) for l in range(len(day.network.branches))] for t in range(day.periods)]),
                      current_sq=np.array([[value(base.i[t,l]) for l in range(len(day.network.branches))] for t in range(day.periods)]),
                      charge={n:curve(v) for n,v in base.charge.items()},discharge={n:curve(v) for n,v in base.discharge.items()},
                      energy={n:curve(v) for n,v in base.energy.items()},generation={n:curve(v) for n,v in base.generation.items()},
                      generator_q={n:curve(v) for n,v in base.generator_q.items()},shed_mw=np.zeros_like(day.active_load_mw),
                      shed_q_mvar=np.zeros_like(day.reactive_load_mvar),served_mw=day.active_load_mw.copy(),shed_mwh=0.)
        if prices:
            result['lmp_cny_mwh']=np.array([[-solution.z[self.row_reduced[self.row_by_name[base.balance[t,b].ConstrName]]]
                           *self.rules.objective_scale/day.dt_hours for b in range(day.network.buses)] for t in range(day.periods)])
        return result

    def close(self):
        self.base.close()
