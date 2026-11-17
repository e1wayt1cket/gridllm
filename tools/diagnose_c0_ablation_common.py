"""C0 candidate ablation, common-split rerun: dual-report judgment.

NON_FORMAL_DIAGNOSTIC_PILOT. Reads the four C0PILOT_COMMON runs only; no
training, no checkpoint writes. Mirrors diagnose_c0_ablation_v4.py with
three review-mandated changes (2026-11-10 ruling):

  1. Runs: C0PILOT_COMMON_{BC2,B}_seed{11,23} (--split common, matching the
     formal runs' day partition). Outputs go to *_common.{json,md}; the
     first batch's c0_candidate_ablation.{json,md} is left untouched.
  2. Dual reporting: the pre-registered symmetric C band (|r1 diff| <= 10)
     is kept verbatim; an exploratory one-sided read (r1 diff >= -10) is
     reported alongside and labelled post-hoc protocol revision.
  3. Criterion B is additionally broken down per ESS (per-leg edge fraction
     drop and gradient retention), per the rerun approval conditions.

Criteria (unchanged from b/C0_candidate_预注册_2026-11-10.md):
  A: r4 val mean_fleet_delta(B) - mean_fleet_delta(B_C2) > +10 CNY,
     day-clustered bootstrap (1000) 95% CI lower > 0.
  B: r4 open-quote edge fraction drops >= 20pp vs control (per leg, 5%
     edge band), OR training-mode gradient retention median >= 0.1.
  C (pre-registered, symmetric): |r1 diff| <= 10 CNY.
  C (exploratory, one-sided, post-hoc): r1 diff >= -10 CNY.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import da_quote_map as qmap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "algorithm_diagnostics"
JSON_PATH = OUT / "c0_candidate_ablation_common.json"
MD_PATH = OUT / "c0_candidate_ablation_common.md"
SEEDS = (11, 23)
ARMS = ("BC2", "B")
ROUNDS = 4
A_THRESHOLD = 10.0
C_THRESHOLD = 10.0
B_EDGE_PP = 20.0
B_GRAD = 0.1
EDGE = 0.05
BOOTSTRAP = 1000
RNG_SEED = 20261110
LABEL = "NON_FORMAL_DIAGNOSTIC_PILOT"
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")
LEGS = ("charge", "discharge")
RUN_PREFIX = "C0PILOT_COMMON"


def run_dir(arm: str, seed: int) -> Path:
    return ROOT / "runs" / f"{RUN_PREFIX}_{arm}_seed{seed}"


def load_eval(run: Path) -> dict:
    history = json.loads((run / "eval_history.json").read_text(
        encoding="utf-8"))
    return {int(e["round"]): e for e in history}


def day_fleet(entry: dict) -> dict:
    return {d["pair_id"]: float(d["delta_fleet"]) for d in entry["days"]}


def boot_ci(diffs: np.ndarray, rng) -> tuple:
    med_mean = float(np.mean(diffs))
    boots = []
    n = diffs.size
    for _ in range(BOOTSTRAP):
        boots.append(float(np.mean(diffs[rng.integers(0, n, n)])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return med_mean, float(lo), float(hi)


def edge_fracs_eval(entry: dict) -> dict:
    """Per-ESS per-leg open-quote edge fraction (eval mode)."""
    out = {}
    for nm in NAMES:
        for leg in range(2):
            lo = EDGE * qmap.LEG_HIGHS[leg]
            hi = (1 - EDGE) * qmap.LEG_HIGHS[leg]
            vals = []
            for day in entry["days"]:
                vals.extend(day["quote_open_values"][nm][leg])
            vals = np.asarray(vals)
            out.setdefault(nm, {})[LEGS[leg]] = float(
                np.mean((vals <= lo) | (vals >= hi))) if vals.size else None
    return out


def fleet_edge(per_ess: dict) -> dict:
    out = {}
    for leg in LEGS:
        vals = [per_ess[nm][leg] for nm in NAMES
                if per_ess[nm][leg] is not None]
        out[leg] = float(np.mean(vals)) if vals else None
    return out


def grad_retention_train(run: Path, round_no: int) -> dict:
    """Per-ESS per-leg gradient retention median (training mode)."""
    vals = {nm: {leg: [] for leg in range(2)} for nm in NAMES}
    with (run / "block_decisions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if int(d["round"]) != round_no:
                continue
            nm = d["agent"]
            for leg in range(2):
                vals[nm][leg].append(float(d["price_logits"][leg]))
    out = {}
    for nm in NAMES:
        out[nm] = {}
        for leg in range(2):
            arr = np.asarray(vals[nm][leg])
            out[nm][LEGS[leg]] = float(np.median(
                qmap.map_derivative_np(arr) / qmap.SLOPE)) \
                if arr.size else None
    return out


def verdict_from(a_hit: bool, b_hit: bool, c_hit: bool, a_diff: float,
                 a_lo: float, a_hi: float) -> str:
    inconclusive = abs(a_diff) <= A_THRESHOLD and a_lo <= 0 <= a_hi
    if not c_hit:
        return "DESIGN_FLAW"
    if inconclusive:
        return "INCONCLUSIVE"
    if a_hit and b_hit:
        return "SAME_CHAIN"
    if a_hit:
        return "TWO_CHAINS"
    return "NEGATIVE"


def judge_seed(seed: int, ctrl: dict, exp: dict, exp_run: Path, rng) -> dict:
    c_diff = None
    if 1 in ctrl and 1 in exp:
        c_diff = float(exp[1]["mean_fleet_delta"]) \
            - float(ctrl[1]["mean_fleet_delta"])
    if ROUNDS not in ctrl or ROUNDS not in exp:
        # A run died before round 4 (e.g. eval-fatal clearing failure):
        # report what is computable (C from r1) and mark the rest
        # NOT_TESTED rather than crashing or guessing.
        return {"run_failure": True,
                "rounds_ctrl": sorted(ctrl), "rounds_exp": sorted(exp),
                "a_diff_r4": None, "a_ci_lo": None, "a_ci_hi": None,
                "a_hit": False, "c_diff_r1": c_diff,
                "c_hit_symmetric": (c_diff is not None
                                    and abs(c_diff) <= C_THRESHOLD),
                "c_hit_one_sided": (c_diff is not None
                                    and c_diff >= -C_THRESHOLD),
                "b_per_ess": None, "b_hit": False,
                "ctrl_r4_mean": None, "exp_r4_mean": None,
                "ctrl_curve": {str(r): float(ctrl[r]["mean_fleet_delta"])
                               for r in sorted(ctrl)},
                "exp_curve": {str(r): float(exp[r]["mean_fleet_delta"])
                              for r in sorted(exp)},
                "verdict_symmetric": "NOT_TESTED_RUN_FAILURE",
                "verdict_one_sided": "NOT_TESTED_RUN_FAILURE"}
    c4, e4 = day_fleet(ctrl[ROUNDS]), day_fleet(exp[ROUNDS])
    days = sorted(c4)
    diffs = np.asarray([e4[d] - c4[d] for d in days])
    a_diff, a_lo, a_hi = boot_ci(diffs, rng)
    a_hit = a_diff > A_THRESHOLD and a_lo > 0
    c1, e1 = float(ctrl[1]["mean_fleet_delta"]), float(exp[1]["mean_fleet_delta"])
    c_diff = e1 - c1
    c_hit_sym = abs(c_diff) <= C_THRESHOLD
    c_hit_one = c_diff >= -C_THRESHOLD

    ctrl_edge = edge_fracs_eval(ctrl[ROUNDS])
    exp_edge = edge_fracs_eval(exp[ROUNDS])
    exp_grad = grad_retention_train(exp_run, ROUNDS)
    per_ess = {}
    for nm in NAMES:
        legs = {}
        for lg in LEGS:
            ce, ee = ctrl_edge[nm][lg], exp_edge[nm][lg]
            pp = 100 * (ce - ee) if (ce is not None and ee is not None) \
                else None
            gr = exp_grad[nm][lg]
            legs[lg] = {"ctrl_edge": ce, "exp_edge": ee,
                        "edge_pp_drop": pp, "grad_retention_median": gr,
                        "hit": (pp is not None and pp >= B_EDGE_PP)
                        or (gr is not None and gr >= B_GRAD)}
        per_ess[nm] = legs
    b_hit = any(v["hit"] for nm in per_ess for v in per_ess[nm].values())

    base = {"a_diff_r4": a_diff, "a_ci_lo": a_lo, "a_ci_hi": a_hi,
            "a_hit": a_hit, "c_diff_r1": c_diff,
            "c_hit_symmetric": c_hit_sym, "c_hit_one_sided": c_hit_one,
            "b_per_ess": per_ess, "b_hit": b_hit,
            "ctrl_r4_mean": float(ctrl[ROUNDS]["mean_fleet_delta"]),
            "exp_r4_mean": float(exp[ROUNDS]["mean_fleet_delta"]),
            "ctrl_curve": {str(r): float(ctrl[r]["mean_fleet_delta"])
                           for r in sorted(ctrl)},
            "exp_curve": {str(r): float(exp[r]["mean_fleet_delta"])
                          for r in sorted(exp)}}
    base["verdict_symmetric"] = verdict_from(
        a_hit, b_hit, c_hit_sym, a_diff, a_lo, a_hi)
    base["verdict_one_sided"] = verdict_from(
        a_hit, b_hit, c_hit_one, a_diff, a_lo, a_hi)
    return base


def merged_of(per_seed: dict, key: str) -> str:
    verdicts = {per_seed[s][key] for s in per_seed}
    if len(verdicts) == 1:
        return verdicts.pop()
    a_hits = [per_seed[s]["a_hit"] for s in per_seed]
    if sum(a_hits) == 1:
        return "PARTIAL_SINGLE_SEED"
    return "MIXED:" + "/".join(sorted(verdicts))


def main() -> int:
    t0 = time.time()
    missing = [f"{RUN_PREFIX}_{a}_seed{s}" for s in SEEDS for a in ARMS
               if not (run_dir(a, s) / "eval_history.json").exists()]
    if missing:
        print(json.dumps({"error": "missing runs", "runs": missing}))
        return 2
    rng = np.random.default_rng(RNG_SEED)
    per_seed = {}
    for seed in SEEDS:
        ctrl = load_eval(run_dir("BC2", seed))
        exp = load_eval(run_dir("B", seed))
        per_seed[str(seed)] = judge_seed(
            seed, ctrl, exp, run_dir("B", seed), rng)
        r = per_seed[str(seed)]
        if r.get("run_failure"):
            cd = "NA" if r["c_diff_r1"] is None else f"{r['c_diff_r1']:+.1f}"
            print(f"seed{seed}: RUN_FAILURE "
                  f"(exp rounds {r['rounds_exp']}, C r1 diff {cd})",
                  flush=True)
            continue
        print(f"seed{seed}: symmetric={r['verdict_symmetric']} "
              f"one_sided={r['verdict_one_sided']} "
              f"(A diff {r['a_diff_r4']:.1f} CI[{r['a_ci_lo']:.1f},"
              f"{r['a_ci_hi']:.1f}], C r1 diff {r['c_diff_r1']:+.1f})",
              flush=True)
    m_sym = merged_of(per_seed, "verdict_symmetric")
    m_one = merged_of(per_seed, "verdict_one_sided")
    report = {"label": LABEL, "generated_at":
              time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "runs_prefix": RUN_PREFIX, "split": "common",
              "criterion_C_note":
                  "symmetric band is the pre-registered rule; one-sided "
                  "read is a post-hoc protocol revision (2026-11-10 ruling)",
              "criteria": {"A_diff_cny": A_THRESHOLD,
                           "C_band_cny_symmetric": C_THRESHOLD,
                           "C_floor_cny_one_sided": -C_THRESHOLD,
                           "B_edge_pp": B_EDGE_PP, "B_grad": B_GRAD,
                           "bootstrap": BOOTSTRAP, "rng_seed": RNG_SEED,
                           "rounds": ROUNDS},
              "per_seed": per_seed,
              "merged_verdict_symmetric": m_sym,
              "merged_verdict_one_sided_exploratory": m_one,
              "runtime_s": round(time.time() - t0, 1)}
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# C0 candidate 消融判定(common split 重跑,双报)", "",
             f"label: {LABEL}",
             f"merged(预注册对称带): **{m_sym}**",
             f"merged(探索性单向带,事后修订): **{m_one}**", "",
             "| seed | 对照 r4 | 实验 r4 | A 差 [CI] | C r1 差 | "
             "verdict(对称) | verdict(单向) |",
             "|---|---|---|---|---|---|---|"]
    for seed, r in report["per_seed"].items():
        if r.get("run_failure"):
            cd = "NA" if r["c_diff_r1"] is None else f"{r['c_diff_r1']:+.1f}"
            lines.append(
                f"| {seed} | RUN_FAILURE(r{max(r['rounds_exp'])}/"
                f"{ROUNDS}) | — | NOT_TESTED | {cd} "
                f"| {r['verdict_symmetric']} | {r['verdict_one_sided']} |")
            continue
        lines.append(
            f"| {seed} | {r['ctrl_r4_mean']:.1f} | {r['exp_r4_mean']:.1f} "
            f"| {r['a_diff_r4']:+.1f} [{r['a_ci_lo']:+.1f}, "
            f"{r['a_ci_hi']:+.1f}] | {r['c_diff_r1']:+.1f} "
            f"| {r['verdict_symmetric']} | {r['verdict_one_sided']} |")
    lines += ["", "判据 B 分 ESS 明细(r4;贴边=评估模式开价边缘比例,"
              "梯度留存=训练模式映射导数中位/SLOPE):", ""]
    for seed, r in report["per_seed"].items():
        if r.get("run_failure"):
            lines.append(f"- seed{seed}: NOT_TESTED(run 未完成 r4)")
            continue
        lines.append(f"- seed{seed}:")
        for nm, legs in r["b_per_ess"].items():
            for lg, v in legs.items():
                ce = "NA" if v["ctrl_edge"] is None else f"{v['ctrl_edge']:.3f}"
                ee = "NA" if v["exp_edge"] is None else f"{v['exp_edge']:.3f}"
                pp = "NA" if v["edge_pp_drop"] is None else f"{v['edge_pp_drop']:+.1f}pp"
                gr = "NA" if v["grad_retention_median"] is None else f"{v['grad_retention_median']:.3f}"
                lines.append(f"  - {nm}.{lg}: 贴边 {ce}→{ee} ({pp}), "
                             f"梯度留存 {gr}, 命中={'Y' if v['hit'] else 'n'}")
    lines += ["", "逐轮曲线(mean_fleet_delta):", ""]
    for seed, r in report["per_seed"].items():
        lines.append(f"- seed{seed} 对照: "
                     + " → ".join(f"r{k}:{v:.1f}"
                                 for k, v in r["ctrl_curve"].items()))
        lines.append(f"- seed{seed} 实验: "
                     + " → ".join(f"r{k}:{v:.1f}"
                                 for k, v in r["exp_curve"].items()))
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"merged_symmetric": m_sym,
                      "merged_one_sided_exploratory": m_one}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
