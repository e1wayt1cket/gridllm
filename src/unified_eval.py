# unified_eval.py
"""One evaluation entry point for every strategy.

Before this module the comparison paths did not meet. `strategies.adaptive_bidding`
drives `fixed` / `mpc` / `greedy_threshold` / `rl` but returns **declared quotes
only** -- no money. `counterfactual_eval` returns **settled per-agent profits**
but accepts only `truthful` / `rule` / `myopic` / `rl`. Neither could answer
"which strategy earned more on the same day", because each was missing half of
what the question needs.

This module is the adapter, not a second settlement. Every arm is settled through
`counterfactual_eval.settle_day` -- one definition of what a participant earns.
An arm is driven either as a whole-day declaration (the strategy registry) or as
a per-block policy (the counterfactual baselines and the trained actors), and
both are normalised into one record shape.

Three metrics are kept apart, and none substitutes for another:

| metric | what it is |
|---|---|
| **settled net profit** | what the participant was actually paid, through the clearing, from `participant_payoff`. The primary metric. |
| **training reward** | the learning signal. Recorded in its own field, never as profit. |
| **LP reference** | what the reference unit could capture on the same price curve, in isolation. A bound, not an outcome. |

Evaluation mode is asserted, not assumed: `run_arm` performs inference only, no
actor noise and no gradient step, and a test checks that a policy's parameters
are bit-identical before and after an evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import counterfactual_eval as ce
import synthetic_scenarios as ss
from arbitrage_lp import arbitrage_lp
from money import DT_HOURS

T_DEFAULT = 96

#: Arms driven by a whole-day declaration through the strategy registry.
WHOLE_DAY_STRATEGIES: Tuple[str, ...] = ("fixed", "mpc", "greedy_threshold")
#: Arms that act per block inside the rolling environment.
PER_BLOCK_KINDS: Tuple[str, ...] = ("truthful", "rule", "myopic", "rl")

#: How a result is marked when it may not enter the aggregate.
EXCLUDED_SOLVER_FAILURE = "solver_failure_or_fallback"
#: The clear succeeded but its price field is not nodal for every period, so
#: the profit was settled on a price the day did not make. Kept apart from a
#: solver failure: nothing went wrong with the solve.
EXCLUDED_MIXED_PRICE_FIELD = "mixed_wholesale_price_fallback"
EXCLUDED_INCOMPLETE = "incomplete_settlement"
EXCLUDED_SOC_INFEASIBLE = "soc_constraint_violated"


@dataclass(frozen=True)
class ArmRef:
    """One strategy under evaluation, with the identity to report it by."""

    label: str
    family: str                       # "whole_day" | "per_block"
    name: str                         # registry name, or baseline kind
    #: Loaded actors for `family == "per_block"` and `name == "rl"`.
    policies: Optional[Dict[str, object]] = None
    #: Free-form version/checkpoint identity for the report.
    version: Optional[str] = None

    def __post_init__(self) -> None:
        if self.family not in ("whole_day", "per_block"):
            raise ValueError(f"unknown arm family {self.family!r}")
        if self.family == "whole_day" and \
                self.name not in WHOLE_DAY_STRATEGIES:
            raise ValueError(
                f"{self.name!r} is not a whole-day strategy; expected one of "
                f"{WHOLE_DAY_STRATEGIES}")
        if self.family == "per_block" and self.name not in PER_BLOCK_KINDS:
            raise ValueError(
                f"{self.name!r} is not a per-block kind; expected one of "
                f"{PER_BLOCK_KINDS}")


@dataclass(frozen=True)
class ScenarioRef:
    """One scenario day, with the provenance a result must carry."""

    scenario: str
    day_id: int
    split: str                        # "train" | "validation" | "test"
    family: Optional[str] = None
    data_digest: Optional[str] = None

    @property
    def scenario_id(self) -> str:
        return f"{self.scenario}::day{int(self.day_id)}"


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=float).encode("utf-8")
    ).hexdigest()


def scenario_input_digest(agents, config, wholesale: np.ndarray,
                          T: int = T_DEFAULT) -> Dict[str, object]:
    """A fingerprint of **every exogenous input** at the evaluation boundary.

    Price alone is not enough to say two runs faced the same scenario: two days
    can share a price curve, an initial SOC and an economics block while
    differing in load or in renewable output, and a profit difference between
    them would then be attributed to the strategy when it belongs to the day.

    Components are digested separately so a mismatch says *which* input moved,
    and combined into one value that `assert_comparable` tests. What is
    **excluded** is as deliberate as what is included: policy actions, policy
    parameters and any other strategy-internal state are not exogenous, and
    hashing them would turn a difference between strategies into an apparent
    difference between scenarios -- exactly the confusion this guards against.

    Network topology is covered by the scenario/config fingerprint rather than by
    hashing pandapower's element tables: the network is a pure function of the
    scenario and the config, both of which are in the digest. That is a
    statement about this model, and it is stated rather than assumed.
    """
    price = np.asarray(wholesale, dtype=float)[:T]
    load, re_out = {}, {}
    for a in agents:
        load[a.name] = [float(x) for x in np.asarray(
            getattr(a, "load_forecast", []), dtype=float)[:T]]
        pv = [float(x) for x in np.asarray(
            getattr(a, "pv_forecast", []), dtype=float)[:T]]
        wind = ([float(x) for x in np.asarray(a.get_wind_forecast(),
                                              dtype=float)[:T]]
                if getattr(a, "has_wind", False) else [])
        re_out[a.name] = {"pv": pv, "wind": wind}

    agent_spec = {}
    for a in agents:
        st = getattr(a, "storage", None)
        agent_spec[a.name] = {
            "bus": int(getattr(a, "bus", -1)),
            "participant_type": getattr(a, "participant_type", None),
            "is_prosumer": bool(getattr(a, "is_prosumer", False)),
            "has_wind": bool(getattr(a, "has_wind", False)),
            "storage": None if st is None else {
                "e_max": float(st.e_max), "p_ch_max": float(st.p_ch_max),
                "p_dis_max": float(st.p_dis_max),
                "eta_ch": float(st.eta_ch), "eta_dis": float(st.eta_dis),
                "soc0": float(st.soc0), "soc_min": float(st.soc_min),
                "soc_max": float(st.soc_max),
            },
        }
    economics = {
        "cycle_cost": float(config.storage.cycle_cost),
        "terminal_soc_equal": bool(config.storage.terminal_soc_equal),
        "opf_mode": str(getattr(config, "opf_mode", "")),
        "line_capacity_multiplier": float(
            getattr(config, "line_capacity_multiplier", float("nan"))),
    }
    # The scenario + config fingerprint stands in for the network topology.
    try:
        network = ce._scenario_fingerprint(getattr(config, "_scenario_name", ""))
    except Exception:
        network = ""
    try:
        cfg_digest = ce._config_fingerprint(config)
    except Exception:
        cfg_digest = ""
    parts = {
        "T": int(T),
        "price": _digest([float(x) for x in price]),
        "load": _digest(load),
        "renewables": _digest(re_out),
        "agents": _digest(agent_spec),
        "economics": _digest(economics),
        "network": _digest([network, cfg_digest]),
    }
    parts["combined"] = _digest(parts)
    parts["note"] = (
        "exogenous inputs only; policy actions, parameters and internal state "
        "are excluded by design")
    return parts


def _scenario_freeze(agents, config, wholesale: np.ndarray) -> Dict[str, object]:
    """Everything about the scenario a comparison must hold identical.

    Recorded per result so "the two arms saw the same day" is checkable from the
    artefacts rather than asserted in prose.
    """
    soc0 = {a.name: (float(a.storage.soc0)
                     if getattr(a, "storage", None) is not None else None)
            for a in agents if getattr(a, "storage", None) is not None}
    price = np.asarray(wholesale, dtype=float)
    return {
        "n_agents": len(agents),
        "soc0": soc0,
        "wholesale_sha256": hashlib.sha256(
            np.ascontiguousarray(price).tobytes()).hexdigest(),
        "wholesale_min": float(price.min()),
        "wholesale_max": float(price.max()),
        "wholesale_mean": float(price.mean()),
        "cycle_cost_per_leg": float(config.storage.cycle_cost),
        "terminal_soc_equal": bool(config.storage.terminal_soc_equal),
    }


def _soc_report(agents, result, rl_names: Sequence[str]) -> Dict[str, object]:
    """SOC feasibility, read off the settled schedule.

    A bound violation is a failure of the run, not a number to average in, so
    the verdict is reported alongside the values that produced it.
    """
    per_unit: Dict[str, object] = {}
    ok = True
    for a in agents:
        if getattr(a, "storage", None) is None:
            continue
        sched = (result or {}).get("schedules", {}).get(a.name)
        if sched is None or "soc" not in sched:
            per_unit[a.name] = {"present": False, "within_bounds": None,
                                "terminal_equals_initial": None}
            continue
        soc = np.asarray(sched["soc"], dtype=float)
        lo, hi = float(a.storage.soc_min), float(a.storage.soc_max)
        within = bool(soc.min() >= lo - 1e-6 and soc.max() <= hi + 1e-6)
        balanced = bool(abs(soc[-1] - soc[0]) <= 1e-6)
        per_unit[a.name] = {
            "present": True, "soc_min_observed": float(soc.min()),
            "soc_max_observed": float(soc.max()),
            "soc_bounds": [lo, hi],
            "soc_terminal": float(soc[-1]), "soc_initial": float(soc[0]),
            "within_bounds": within, "terminal_equals_initial": balanced,
        }
        ok = ok and within and balanced
    return {"per_unit": per_unit, "all_feasible": bool(ok)}


def _regime_of(wholesale: np.ndarray) -> Dict[str, object]:
    """The scenario's economic regime, measured on the reference unit.

    Measured, not inherited from how the day was generated -- a generator label
    and a measured outcome are different claims and only the second belongs in a
    result.
    """
    try:
        res = arbitrage_lp(np.asarray(wholesale, dtype=float),
                           ss.REFERENCE_UNIT, ss.CYCLE_COST_PER_LEG)
        diag = ss.classify(res, ss.CYCLE_COST_PER_LEG, ss.REFERENCE_UNIT)
    except Exception as exc:                     # pragma: no cover - defensive
        return {"regime": None, "schedule_class": None,
                "diagnostic": f"regime not measurable: {exc}"}
    return {
        "regime": diag["regime"],
        "schedule_class": diag.get("schedule_class"),
        "m_lp_cny_per_mwh": diag["m_lp_cny_per_mwh"],
        "diagnostic": diag["reason"],
    }


def evaluate_arm(arm: ArmRef, scen: ScenarioRef, config,
                 seed: int, T: int = T_DEFAULT,
                 rl_names: Optional[Sequence[str]] = None,
                 n_blocks: int = ce.N_BLOCKS) -> Dict[str, object]:
    """One (arm, scenario, seed) result. Never raises for a solver failure.

    A failed clear is recorded as a failed record with its reason, not as a zero
    profit and not by dropping the day: either would let a comparison improve by
    failing, which is the one thing an evaluation interface must not permit.
    """
    from scenarios import get_scenario

    day_seed = ce.day_seed_for(scen.day_id) if seed is None else int(seed)
    record: Dict[str, object] = {
        "arm_label": arm.label, "arm_family": arm.family, "arm_name": arm.name,
        "arm_version": arm.version,
        "scenario": scen.scenario, "scenario_id": scen.scenario_id,
        "scenario_family": scen.family, "split": scen.split,
        "scenario_data_digest": scen.data_digest,
        "seed": day_seed, "T": int(T),
        "mode": "evaluation",
        "price_ref_cny_per_mwh": __import__("rl_env").PRICE_REF_CNY_MWH,
        "price_ref_artifact": ss.PRICE_REF_CALIBRATION_FILE,
    }

    np.random.seed(day_seed)
    try:
        agents, wholesale = get_scenario(scen.scenario, T=T, config=config,
                                         day_id=scen.day_id)
    except Exception as exc:
        record.update({"settled_clean": False, "included": False,
                       "exclusion_reason": EXCLUDED_INCOMPLETE,
                       "failure": f"scenario build failed: {exc}",
                       "settled_net_profit_cny": None,
                       "training_reward": None})
        return record

    record["scenario_freeze"] = _scenario_freeze(agents, config, wholesale)
    record["scenario_input_digest"] = scenario_input_digest(
        agents, config, wholesale, T=T)
    record["economic_regime"] = _regime_of(wholesale)

    if arm.family == "whole_day":
        from strategies import adaptive_bidding
        try:
            declared = adaptive_bidding(agents, config, strategy=arm.name,
                                        market_history=None, T=T)
            profits, clean, result = ce.settle_day(
                agents, config, [], declared, wholesale, T=T)
        except Exception as exc:
            profits, clean, result = None, False, None
            record["failure"] = f"whole-day arm failed: {exc}"
    else:
        spec = ce.ArmSpec(name=arm.label, kind=arm.name,
                          policies=arm.policies)
        names = list(rl_names) if rl_names is not None else \
            [a.name for a in agents if getattr(a, "storage", None) is not None]
        try:
            out = ce.run_arm(scen.scenario, config, spec, names, day_seed,
                             T=T, n_blocks=n_blocks, settle_full_day=True,
                             day_id=scen.day_id)
            profits, clean = out["profits"], out["clean"]
            result = out.get("day_result")
            record["training_reward"] = {
                k: float(v) for k, v in (out.get("rolling_profits") or {}).items()
            }
        except Exception as exc:
            profits, clean, result = None, False, None
            record["failure"] = f"per-block arm failed: {exc}"

    record.setdefault("training_reward", None)
    record["settled_clean"] = bool(clean)
    record["solver_status"] = (result or {}).get("solver") if isinstance(
        result, dict) else None
    # How much of the price field the profit was settled on is not a nodal
    # price. Carried on every record, clean or not, so a reader of the file can
    # always tell what the price column was.
    record["lmp_price_source"] = (result or {}).get("lmp_price_source",
                                                   "nodal") if isinstance(
        result, dict) else None
    record["lmp_fallback_periods"] = (
        list((result or {}).get("lmp_fallback_periods", []))
        if isinstance(result, dict) else None)
    record["lmp_fallback_reason"] = (result or {}).get("lmp_fallback_reason", "")         if isinstance(result, dict) else None
    record["has_wholesale_fallback"] = bool(
        (result or {}).get("has_wholesale_fallback")) if isinstance(
        result, dict) else None
    record["soc"] = _soc_report(agents, result, [])

    if not clean or profits is None:
        mixed = bool((result or {}).get("has_wholesale_fallback"))             if isinstance(result, dict) else False
        record.update({
            "settled_net_profit_cny": None, "per_agent_profit_cny": None,
            "traded": None, "included": False,
            "exclusion_reason": (EXCLUDED_MIXED_PRICE_FIELD if mixed
                                 else EXCLUDED_SOLVER_FAILURE),
            "failure": record.get(
                "failure",
                "price field is not nodal for every period"
                if mixed else "settlement not clean"),
        })
        return record

    fleet = float(sum(float(v) for v in profits.values()))
    sched = (result or {}).get("schedules", {})
    traded = {}
    for name in profits:
        s = sched.get(name, {})
        traded[name] = {
            "charge_mwh": float(np.sum(s.get("p_ch", [])) * DT_HOURS),
            "discharge_mwh": float(np.sum(s.get("p_dis", [])) * DT_HOURS),
        }
    record.update({
        "settled_net_profit_cny": fleet,
        "per_agent_profit_cny": {k: float(v) for k, v in profits.items()},
        "traded": traded,
        "fleet_charge_mwh": float(sum(v["charge_mwh"] for v in traded.values())),
        "fleet_discharge_mwh": float(sum(v["discharge_mwh"] for v in traded.values())),
        "n_units": len(profits),
    })
    if not record["soc"]["all_feasible"]:
        record.update({"included": False,
                       "exclusion_reason": EXCLUDED_SOC_INFEASIBLE})
    else:
        record.update({"included": True, "exclusion_reason": None})
    return record


def evaluate(arms: Sequence[ArmRef], scenarios: Sequence[ScenarioRef], config,
             seed: Optional[int] = None, T: int = T_DEFAULT,
             rl_names: Optional[Sequence[str]] = None,
             n_blocks: int = ce.N_BLOCKS) -> Dict[str, object]:
    """Every arm on every scenario, under one seed and one settlement path.

    The scenario list is iterated identically for every arm and is not filtered
    per arm: a scenario an arm fails is reported as failed, never skipped in its
    favour. `assert_comparable` then verifies from the artefacts that the arms
    really did see the same days.
    """
    if not arms:
        raise ValueError("no arms supplied")
    if not scenarios:
        raise ValueError("no scenarios supplied")
    records: List[Dict[str, object]] = []
    for arm in arms:
        for scen in scenarios:
            records.append(evaluate_arm(arm, scen, config, seed=seed, T=T,
                                        rl_names=rl_names, n_blocks=n_blocks))
    assert_comparable(records)
    out: Dict[str, object] = {
        "records": records,
        "summary": summarize_records(records),
        "arms": [a.label for a in arms],
        "scenario_ids": [s.scenario_id for s in scenarios],
        "n_scenarios": len(scenarios),
    }
    # A two-arm run also gets the paired view, on the common valid set. The
    # per-arm summaries above each use their own denominator and must not be
    # read as a ranking.
    if len(arms) == 2:
        out["paired"] = paired_summary(records, arms[0].label, arms[1].label)
    return out


def assert_comparable(records: Sequence[Dict[str, object]]) -> None:
    """Arms compared on a scenario must have seen identical exogenous inputs.

    Verified against the **combined** input digest, which covers price, load,
    renewable output, agent/storage specification, economics and the
    scenario/config fingerprint -- not against price alone. Two runs can share a
    price curve, an initial SOC and an economics block and still differ in load
    or in renewable output; a price-only check would admit that pair and the
    profit difference would be read as a strategy effect.

    The component digests are compared too, so the refusal names which input
    moved rather than only that something did.
    """
    by_scenario: Dict[str, List[Dict[str, object]]] = {}
    for r in records:
        by_scenario.setdefault(str(r["scenario_id"]), []).append(r)
    for sid, group in by_scenario.items():
        first = group[0].get("scenario_input_digest")
        if first is None:
            continue
        for r in group[1:]:
            other = r.get("scenario_input_digest")
            if other is None:
                continue
            for key in ("price", "load", "renewables", "agents", "economics",
                        "network", "combined"):
                if other[key] != first[key]:
                    raise AssertionError(
                        f"{sid}: arms saw different {key} inputs "
                        f"({str(first[key])[:12]} vs {str(other[key])[:12]}); "
                        "the comparison is between two different scenarios")
        # The older price/SOC/economics cross-check is kept as well: it is
        # cheap, it is independent of the digest construction, and a
        # disagreement between the two would itself be a defect.
        f0 = group[0].get("scenario_freeze")
        if f0 is None:
            continue
        for r in group[1:]:
            o = r.get("scenario_freeze")
            if o is None:
                continue
            if o["wholesale_sha256"] != f0["wholesale_sha256"]:
                raise AssertionError(f"{sid}: arms saw different price series")
            if o["soc0"] != f0["soc0"]:
                raise AssertionError(f"{sid}: arms started from different SOC")
            if o["cycle_cost_per_leg"] != f0["cycle_cost_per_leg"]:
                raise AssertionError(f"{sid}: different economics")


#: Fewer common valid scenarios than this and no ranking is offered.
MIN_COMMON_SCENARIOS_FOR_RANKING = 3


def paired_summary(records: Sequence[Dict[str, object]],
                   arm_a: str, arm_b: str) -> Dict[str, object]:
    """Paired per-scenario comparison of two arms on their COMMON valid set.

    The single-arm summaries answer "how did this arm do", each over its own
    denominator. They cannot answer "which is better", because the two arms may
    have failed on different scenarios: if A fails the low-price days and B
    fails the high-price ones, both means carry a selection effect and the
    difference between them is partly an artefact of which days each kept.

    So the comparison runs on `S_common = S_included(A) ∩ S_included(B)` and
    reports per-scenario differences `dPi_d = Pi_A,d - Pi_B,d`. Failure counts
    are reported beside it, because runnability is a separate question from
    return and both matter.

    When the common set is empty or too small, no ranking is returned.
    Falling back to the unequal-denominator means would silently present a
    selection-biased number as a comparison.
    """
    inc: Dict[str, Dict[str, Dict[str, object]]] = {arm_a: {}, arm_b: {}}
    for r in records:
        lab = str(r["arm_label"])
        if lab in inc and r.get("included") and \
                r.get("settled_net_profit_cny") is not None:
            inc[lab][str(r["scenario_id"])] = r
    common = sorted(set(inc[arm_a]) & set(inc[arm_b]))
    fails = {lab: sum(1 for r in records
                      if str(r["arm_label"]) == lab and not r.get("included"))
             for lab in (arm_a, arm_b)}
    out: Dict[str, object] = {
        "arm_a": arm_a, "arm_b": arm_b,
        "n_included_a": len(inc[arm_a]), "n_included_b": len(inc[arm_b]),
        "n_failed_a": fails[arm_a], "n_failed_b": fails[arm_b],
        "n_common": len(common), "common_scenario_ids": common,
        "min_common_for_ranking": MIN_COMMON_SCENARIOS_FOR_RANKING,
    }
    if len(common) < MIN_COMMON_SCENARIOS_FOR_RANKING:
        out["comparable"] = False
        out["reason"] = (
            f"only {len(common)} scenario(s) valid for both arms; a paired "
            f"comparison needs at least {MIN_COMMON_SCENARIOS_FOR_RANKING}. No "
            "ranking is offered rather than falling back to means computed over "
            "different denominators")
        out["delta_settled_net_profit_cny"] = None
        return out

    deltas = [float(inc[arm_a][s]["settled_net_profit_cny"])
              - float(inc[arm_b][s]["settled_net_profit_cny"]) for s in common]
    arr = np.asarray(deltas, dtype=float)
    out.update({
        "comparable": True,
        "per_scenario_delta_cny": dict(zip(common, deltas)),
        "delta_settled_net_profit_cny": {
            "mean": float(arr.mean()), "median": float(np.median(arr)),
            "std": float(arr.std()), "min": float(arr.min()),
            "max": float(arr.max()),
            "n_positive": int((arr > 0).sum()),
            "n_negative": int((arr < 0).sum()),
            "n_zero": int((arr == 0).sum()),
        },
        "denominator": "n_common (scenarios valid for BOTH arms)",
    })
    return out


def summarize_records(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate over **included** records only, with failures reported apart.

    The denominator is stated explicitly. A failed record is neither dropped
    silently nor counted as zero profit: either would let an arm's average
    improve by failing, so failures are listed with their reasons and the
    excluded count is published beside the mean.
    """
    by_arm: Dict[str, Dict[str, object]] = {}
    for r in records:
        label = str(r["arm_label"])
        bucket = by_arm.setdefault(label, {"included": [], "excluded": []})
        (bucket["included"] if r.get("included") else bucket["excluded"]).append(r)

    out: Dict[str, object] = {}
    for label, bucket in sorted(by_arm.items()):
        vals = [float(r["settled_net_profit_cny"]) for r in bucket["included"]
                if r.get("settled_net_profit_cny") is not None]
        reasons: Dict[str, int] = {}
        for r in bucket["excluded"]:
            key = str(r.get("exclusion_reason") or "unspecified")
            reasons[key] = reasons.get(key, 0) + 1
        entry: Dict[str, object] = {
            "n_records": len(bucket["included"]) + len(bucket["excluded"]),
            "n_included": len(bucket["included"]),
            "n_excluded": len(bucket["excluded"]),
            "exclusion_reasons": reasons,
            "excluded_scenario_ids": [r["scenario_id"] for r in bucket["excluded"]],
            "denominator": "n_included (settled, SOC-feasible records only)",
        }
        if vals:
            arr = np.asarray(vals, dtype=float)
            entry.update({
                "settled_net_profit_cny": {
                    "mean": float(arr.mean()), "std": float(arr.std()),
                    "median": float(np.median(arr)),
                    "min": float(arr.min()), "max": float(arr.max()),
                    "p25": float(np.percentile(arr, 25)),
                    "p75": float(np.percentile(arr, 75)),
                },
            })
        else:
            entry["settled_net_profit_cny"] = None
        out[label] = entry
    return out


def to_json(payload: Dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
