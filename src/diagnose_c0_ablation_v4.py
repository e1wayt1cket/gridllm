"""C0 candidate ablation analysis: B_C2 vs B, pre-registered criteria A/B/C.

NON_FORMAL_DIAGNOSTIC_PILOT. Reads the four C0PILOT runs only; no training,
no checkpoint writes. Criteria per b/C0_candidate_预注册_2026-11-10.md:

  A (economic, per seed): r4 val mean_fleet_delta(B) - mean_fleet_delta(B_C2)
      > +10 CNY, day-clustered bootstrap (1000, day unit) 95% CI lower > 0.
      Directional threshold, not a formal significance level.
  B (chain test, per seed per leg): r4 open-quote edge fraction (eval mode,
      per-leg 5% edge band) drops >= 20pp vs control, OR training-mode
      gradient retention median >= 0.1. Descriptive, not a gate.
  C (sanity, fleet-level, 11-day aggregate): |r1 diff| <= 10 CNY.

Verdicts (per seed first, then merged; ties required for a joint verdict):
  SAME_CHAIN (A&B), TWO_CHAINS (A&~B -> C1 next), NEGATIVE (~A&C),
  DESIGN_FLAW (~C), INCONCLUSIVE (|diff| <= 10 and CI covers 0),
  PARTIAL_SINGLE_SEED (A on exactly one seed).
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
JSON_PATH = OUT / "c0_candidate_ablation.json"
MD_PATH = OUT / "c0_candidate_ablation.md"
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


def run_dir(arm: str, seed: int) -> Path:
    return ROOT / "runs" / f"C0PILOT_{arm}_seed{seed}"


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
    out = {}
    for leg in range(2):
        lo = EDGE * qmap.LEG_HIGHS[leg]
        hi = (1 - EDGE) * qmap.LEG_HIGHS[leg]
        vals = []
        for day in entry["days"]:
            for nm in NAMES:
                vals.extend(day["quote_open_values"][nm][leg])
        vals = np.asarray(vals)
        out[LEGS[leg]] = float(np.mean((vals <= lo) | (vals >= hi))) \
            if vals.size else None
    return out


def grad_retention_train(run: Path, round_no: int) -> dict:
    vals = {lg: [] for lg in range(2)}
    with (run / "block_decisions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if int(d["round"]) != round_no:
                continue
            for leg in range(2):
                vals[leg].append(float(d["price_logits"][leg]))
    return {LEGS[leg]: float(np.median(
                qmap.map_derivative_np(np.asarray(vals[leg])) / qmap.SLOPE))
            if vals[leg] else None for leg in range(2)}


def judge_seed(seed: int, ctrl: dict, exp: dict, rng) -> dict:
    c4, e4 = day_fleet(ctrl[ROUNDS]), day_fleet(exp[ROUNDS])
    days = sorted(c4)
    diffs = np.asarray([e4[d] - c4[d] for d in days])
    a_diff, a_lo, a_hi = boot_ci(diffs, rng)
    a_hit = a_diff > A_THRESHOLD and a_lo > 0
    c1, e1 = float(ctrl[1]["mean_fleet_delta"]), float(exp[1]["mean_fleet_delta"])
    c_diff = e1 - c1
    c_hit = abs(c_diff) <= C_THRESHOLD
    ctrl_edge = edge_fracs_eval(ctrl[ROUNDS])
    exp_edge = edge_fracs_eval(exp[ROUNDS])
    exp_grad = grad_retention_train(run_dir("B", seed), ROUNDS)
    b_legs = {}
    for lg in LEGS:
        pp = None
        if ctrl_edge[lg] is not None and exp_edge[lg] is not None:
            pp = 100 * (ctrl_edge[lg] - exp_edge[lg])
        b_legs[lg] = {"edge_pp_drop": pp,
                      "grad_retention_median": exp_grad[lg],
                      "hit": (pp is not None and pp >= B_EDGE_PP)
                      or (exp_grad[lg] is not None
                          and exp_grad[lg] >= B_GRAD)}
    b_hit = any(v["hit"] for v in b_legs.values())
    inconclusive = abs(a_diff) <= A_THRESHOLD and a_lo <= 0 <= a_hi
    if not c_hit:
        verdict = "DESIGN_FLAW"
    elif inconclusive:
        verdict = "INCONCLUSIVE"
    elif a_hit and b_hit:
        verdict = "SAME_CHAIN"
    elif a_hit:
        verdict = "TWO_CHAINS"
    else:
        verdict = "NEGATIVE"
    return {"a_diff_r4": a_diff, "a_ci_lo": a_lo, "a_ci_hi": a_hi,
            "a_hit": a_hit, "c_diff_r1": c_diff, "c_hit": c_hit,
            "b_legs": b_legs, "b_hit": b_hit, "verdict": verdict,
            "ctrl_r4_mean": float(ctrl[ROUNDS]["mean_fleet_delta"]),
            "exp_r4_mean": float(exp[ROUNDS]["mean_fleet_delta"]),
            "ctrl_curve": {str(r): float(ctrl[r]["mean_fleet_delta"])
                           for r in sorted(ctrl)},
            "exp_curve": {str(r): float(exp[r]["mean_fleet_delta"])
                          for r in sorted(exp)}}


def main() -> int:
    t0 = time.time()
    missing = [f"C0PILOT_{a}_seed{s}" for s in SEEDS for a in ARMS
               if not (run_dir(a, s) / "eval_history.json").exists()]
    if missing:
        print(json.dumps({"error": "missing runs", "runs": missing}))
        return 2
    rng = np.random.default_rng(RNG_SEED)
    per_seed = {}
    for seed in SEEDS:
        ctrl = load_eval(run_dir("BC2", seed))
        exp = load_eval(run_dir("B", seed))
        per_seed[str(seed)] = judge_seed(seed, ctrl, exp, rng)
        print(f"seed{seed}: {per_seed[str(seed)]['verdict']} "
              f"(A diff {per_seed[str(seed)]['a_diff_r4']:.1f} "
              f"CI[{per_seed[str(seed)]['a_ci_lo']:.1f},"
              f"{per_seed[str(seed)]['a_ci_hi']:.1f}])", flush=True)
    a_hits = [per_seed[str(s)]["a_hit"] for s in SEEDS]
    verdicts = {per_seed[str(s)]["verdict"] for s in SEEDS}
    if len(verdicts) == 1:
        merged = verdicts.pop()
    elif sum(a_hits) == 1:
        merged = "PARTIAL_SINGLE_SEED"
    else:
        merged = "MIXED:" + "/".join(sorted(verdicts))
    report = {"label": LABEL, "generated_at":
              time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "criteria": {"A_diff_cny": A_THRESHOLD, "C_band_cny":
                           C_THRESHOLD, "B_edge_pp": B_EDGE_PP,
                           "B_grad": B_GRAD, "bootstrap": BOOTSTRAP,
                           "rng_seed": RNG_SEED, "rounds": ROUNDS},
              "per_seed": per_seed, "merged_verdict": merged,
              "runtime_s": round(time.time() - t0, 1)}
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# C0 candidate 消融判定(B_C2 vs B,4 轮 pilot)", "",
             f"label: {LABEL}; merged verdict: **{merged}**", "",
             "| seed | 对照 r4 | 实验 r4 | A 差 [CI] | C r1 差 | "
             "B 腿命中 | verdict |",
             "|---|---|---|---|---|---|---|"]
    for seed, r in report["per_seed"].items():
        blegs = {k: ("Y" if v["hit"] else "n") for k, v in
                 r["b_legs"].items()}
        lines.append(
            f"| {seed} | {r['ctrl_r4_mean']:.1f} | {r['exp_r4_mean']:.1f} "
            f"| {r['a_diff_r4']:+.1f} [{r['a_ci_lo']:+.1f}, "
            f"{r['a_ci_hi']:+.1f}] | {r['c_diff_r1']:+.1f} "
            f"| {blegs} | {r['verdict']} |")
    lines += ["", "逐轮曲线(mean_fleet_delta):", ""]
    for seed, r in report["per_seed"].items():
        lines.append(f"- seed{seed} 对照: "
                     + " → ".join(f"r{k}:{v:.1f}"
                                 for k, v in r["ctrl_curve"].items()))
        lines.append(f"- seed{seed} 实验: "
                     + " → ".join(f"r{k}:{v:.1f}"
                                 for k, v in r["exp_curve"].items()))
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"merged_verdict": merged}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
