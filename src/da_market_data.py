"""Versioned research contract and frozen historical training-day inputs."""

from dataclasses import dataclass, replace
import json

import numpy as np

from local_da_contracts import Generator, validate_day
from local_da_operating import OperatingRules, build_historical_day, support_day
from run_local_da import ROOT, sha

MECHANISM_VERSION = "independent_da_v2"
# Charged per MWh of throughput in settlement only; it never enters the clearing
# objective, so it cannot move dispatch, LMP or traded volume.
DEGRADATION_CNY_MWH = 80.


@dataclass(frozen=True)
class MarketRules:
    # 2.0 MVAr is the capacity selected in reports/da_market_system_2026-11-14/
    # supply_capacity/selected.json; 1.20 cannot serve 2022-12-25.
    external_q_capacity_mvar: float = 2.0
    external_q_bus: int = 32
    objective_scale: float = 100.
    cone_form: str = "standard"
    seconds: float = 120.
    # Frozen contract: the external price is an exogenous marginal cost of PCC
    # import, so a local LMP is a balance dual that may differ from it, never a
    # supply curve recovered from local load.
    external_price_exogenous: bool = True
    import_cost_form: str = "linear_w_buy_times_p_imp"
    import_export_same_price: bool = True
    local_lmp_formation: str = "balance_dual_direction_conditional"

    def validate(self, day):
        if (not np.isfinite([self.external_q_capacity_mvar,self.objective_scale,self.seconds]).all()
                or min(self.external_q_capacity_mvar,self.objective_scale,self.seconds)<=0
                or self.cone_form not in {"standard","rotated"}
                or not 0<=self.external_q_bus<day.network.buses):
            raise ValueError("Invalid market execution rules")
        if not (self.external_price_exogenous and self.import_export_same_price
                and self.import_cost_form=="linear_w_buy_times_p_imp"
                and self.local_lmp_formation=="balance_dual_direction_conditional"):
            raise ValueError("The external-price contract is frozen and cannot be redefined in place")
        validate_day(day)
        if not day.terminal_soc_equal:
            raise ValueError("The unified day-ahead market requires equal terminal SOC")


def load_day(date="2022-12-22", rules=None):
    rules = rules or MarketRules()
    day, context, mapping = build_historical_day(date)
    pf=.8503
    op=OperatingRules()
    day=replace(day, reactive_load_mvar=day.active_load_mw*np.tan(np.arccos(pf)),terminal_soc_equal=True,
                storage=tuple(replace(u,degradation_cny_mwh=DEGRADATION_CNY_MWH) for u in day.storage))
    day,ratings=support_day(day,op)
    day=replace(day,generators=day.generators+(Generator("EXTERNAL33",rules.external_q_bus,
                np.zeros(day.periods),reactive_limit_mvar=rules.external_q_capacity_mvar),))
    rules.validate(day)
    context.update(mechanism_version=MECHANISM_VERSION,load_power_factor=pf,
                   degradation_cny_mwh=DEGRADATION_CNY_MWH,
                   q_basis="Uniform PF research assumption",terminal_soc="equal",
                   price_series_kind="platform_mean_curve",measured_daily_series=False,
                   load_price_pairing="cross_year_controlled_simulation",
                   external_price_exogenous=rules.external_price_exogenous,
                   import_cost_form=rules.import_cost_form,
                   import_export_same_price=rules.import_export_same_price,
                   local_lmp_formation=rules.local_lmp_formation,
                   conclusion_label="local_market_under_given_external_price_scenario",
                   external_q_capacity_mvar=rules.external_q_capacity_mvar,
                   external_q_bus_one_based=rules.external_q_bus+1,
                   generator_q_limits_mvar="Declared bounded inverter capabilities",
                   inverter_rating_basis="1.10 nameplate multiplier research assumption",
                   reactive_obligation="Retained independently of active-energy bids",
                   reactive_and_capital_costs_included=False,thermal_limits_qualified=False,
                   formal_jiangsu_compliance=False,training_protocol_compatible=False)
    return day,ratings,context,mapping


def zero_bids(day):
    return replace(day,bids={n:replace(b,charge_quantity_mw=np.zeros(day.periods),
                   discharge_quantity_mw=np.zeros(day.periods)) for n,b in day.bids.items()})


def read_bids(day,path):
    payload=json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mechanism_version")!=MECHANISM_VERSION or set(payload.get("bids",{}))!=set(day.bids):
        raise ValueError("Bid file version or independent participants mismatch")
    result={}
    for name,original in day.bids.items():
        values=payload["bids"][name]
        if set(values)!=set(original.__dataclass_fields__):
            raise ValueError("A bid must declare all four quantity and price curves")
        result[name]=replace(original,**{k:np.asarray(v,dtype=float) for k,v in values.items()})
    candidate=replace(day,bids=result)
    validate_day(candidate)
    return candidate


def source_fingerprint():
    names=["reports/multi_user_load_2026-11-13/evidence/manifest.json",
           "data/processed/industry_meter_bus_map_v2.csv",
           "data/processed/v2_cy_jn_2025/common_split.json",
           "data/processed/v2_cy_jn_2025/price_curves.npz"]
    return {name:sha(ROOT/name) for name in names}
