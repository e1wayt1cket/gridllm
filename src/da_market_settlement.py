"""Settle actual awards using qualified conditional prices and neutral inventory."""

import numpy as np


def settle(day,result):
    if not result.get('qualified_for_settlement') or not day.terminal_soc_equal:
        raise ValueError('Qualified prices and equal terminal SOC are required')
    dispatch=result['dispatch']
    price=result['pricing']['lmp_cny_mwh']
    if price.shape!=day.active_load_mw.shape or not np.isfinite(price).all():
        raise ValueError('Invalid settlement price matrix')
    dt=day.dt_hours
    units={}
    for unit in day.storage:
        n=unit.name
        c,p,e=dispatch['charge'][n],dispatch['discharge'][n],dispatch['energy'][n]
        if abs(e[-1]-e[0])>1e-6:
            raise ValueError('Dispatched terminal SOC is not neutral')
        revenue=float(price[:,unit.bus]@p*dt)
        purchase=float(price[:,unit.bus]@c*dt)
        degradation=float((c+p).sum()*dt*unit.degradation_cny_mwh)
        units[n]={'bus_zero_based':unit.bus,'revenue_cny':revenue,'purchase_cny':purchase,
                  'net_cash_cny':revenue-purchase,'degradation_cny':degradation,
                  'economic_payoff_cny':revenue-purchase-degradation,
                  'charge_mwh':float(c.sum()*dt),'discharge_mwh':float(p.sum()*dt),
                  'initial_energy_mwh':float(e[0]),'terminal_energy_mwh':float(e[-1])}
    generators={g.name:{'revenue_cny':float(price[:,g.bus]@dispatch['generation'][g.name]*dt),
                        'real_cost_cny':float(dispatch['generation'][g.name].sum()*dt*g.real_cost_cny_mwh)}
                for g in day.generators}
    consumer=float((price*day.active_load_mw).sum()*dt)
    external=float(day.external_price_cny_mwh@dispatch['pcc_mw']*dt)
    ess=sum(v['net_cash_cny'] for v in units.values())
    generation=sum(v['revenue_cny'] for v in generators.values())
    residual=consumer-ess-generation-external
    system_cost=external+sum(v['degradation_cny'] for v in units.values())+sum(v['real_cost_cny'] for v in generators.values())
    return {'units':units,'generators':generators,'consumer_payment_cny':consumer,
            'external_payment_cny':external,'storage_net_payment_cny':ess,'generator_payment_cny':generation,
            'operator_residual_cny':residual,'real_system_cost_cny':system_cost,
            'dispatch_basis':'Exact complementary dispatched awards',
            'price_basis':result['price_basis'],'inventory_term_cny':0.,
            'reactive_and_capital_costs_included':False,'currency':'CNY'}
