"""Shared versioned public entry for independent day-ahead market clearing."""

from dataclasses import replace

import numpy as np

from da_market_data import MECHANISM_VERSION,MarketRules,zero_bids
from da_market_model import MarketModel,audit
from da_market_pricing import qualify_prices
from da_market_settlement import settle
from da_market_config import default_rules


def classify(dispatch):
    if dispatch.get('optimal'):
        return 'DISPATCH_OPTIMAL'
    return {3:'PHYSICAL_INFEASIBLE',9:'SOLVE_TIME_LIMIT',12:'NUMERICAL_FAILURE',13:'SOLVE_SUBOPTIMAL'}.get(
                  dispatch.get('solver_status'),'SOLVE_FAILED')


def supply_witness(day,ratings,rules,*,folder=None):
    candidate=zero_bids(day)
    model=MarketModel(candidate,ratings,rules,log_path=None if folder is None else folder/'supply.log')
    try:
        dispatch=model.solve()
    finally:
        model.close()
    checked=audit(candidate,ratings,rules,dispatch) if dispatch['optimal'] else {'passed':False}
    return {'dispatch':dispatch,'audit':checked,'qualified':bool(checked['passed'])}


def clear_market_day(day,ratings,*,rules=None,folder=None,price_checks=True):
    rules=rules or default_rules()
    result={'mechanism_version':MECHANISM_VERSION,'qualified_for_settlement':False,
            'price_qualification':'NOT_TESTED','status':'INPUT_INVALID','training_started':False}
    try:
        rules.validate(day)
    except (ValueError,TypeError,AttributeError) as error:
        result['failure']=str(error)
        return result
    base=supply_witness(day,ratings,rules,folder=folder)
    result['supply_witness']=base
    if not base['qualified']:
        result['status']='SUPPLY_WITNESS_FAILED'
        result['failure']=classify(base['dispatch'])
        return result
    model=MarketModel(day,ratings,rules,log_path=None if folder is None else folder/'dispatch.log')
    try:
        dispatch=model.solve()
    finally:
        model.close()
    result['dispatch']=dispatch
    result['status']=classify(dispatch)
    if not dispatch['optimal']:
        return result
    result['dispatch_audit']=audit(day,ratings,rules,dispatch)
    if not result['dispatch_audit']['passed']:
        result['status']='PHYSICAL_AUDIT_FAILED'
        return result
    if not price_checks:
        result['status']='PHYSICAL_ONLY'
        return result
    priced=qualify_prices(day,ratings,rules,dispatch,folder=folder)
    result.update(priced)
    if not result['qualified_for_settlement']:
        result['status']='PRICE_QUALIFICATION_FAILED'
        return result
    result['settlement']=settle(day,result)
    result['status']='SETTLED'
    return result


def stress_cases(day):
    cases=[('reference',day)]
    for n in day.bids:
        cases.append((f'zero_{n}',replace(day,bids={k:(zero_bids(day).bids[k] if k==n else v)
                                                   for k,v in day.bids.items()})))
    cases.append(('all_zero_quantity',zero_bids(day)))
    for label,charge,discharge in [('high_discharge',0.,600.),('high_charge',600.,0.)]:
        cases.append((label,replace(day,bids={n:replace(b,charge_price_cny_mwh=np.full(day.periods,charge),
                    discharge_price_cny_mwh=np.full(day.periods,discharge)) for n,b in day.bids.items()})))
    for label,ch,dis in [('charge_only',.4,0.),('discharge_only',0.,.4),('small_quantity',.01,.01)]:
        cases.append((label,replace(day,bids={n:replace(b,charge_quantity_mw=np.full(day.periods,ch),
                    discharge_quantity_mw=np.full(day.periods,dis)) for n,b in day.bids.items()})))
    rng=np.random.default_rng(20261114)
    bids={n:replace(b,charge_quantity_mw=rng.uniform(0,.4,day.periods),
                   discharge_quantity_mw=rng.uniform(0,.4,day.periods),
                   charge_price_cny_mwh=rng.uniform(0,600,day.periods),
                   discharge_price_cny_mwh=rng.uniform(0,600,day.periods)) for n,b in day.bids.items()}
    cases.append(('random_legal',replace(day,bids=bids)))
    return cases
