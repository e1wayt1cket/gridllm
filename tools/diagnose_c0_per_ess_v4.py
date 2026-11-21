"""C0 D4: per-ESS share of the validation degradation.

NON_FORMAL_DIAGNOSTIC_PILOT. Pure eval_history.json analysis; no checkpoint
load, no clearing, no training.

Pre-registered criterion (b/C0任务书_草稿_2026-11-09.md v2):
  negative days := validation days with delta_fleet < 0 (per round);
  share_i := |delta_i| / sum(|negative delta|) over the ESS with delta_i < 0
  on that day;
  an ESS with median share > 50% over negative days AND day-clustered
  bootstrap 95% CI lower bound > 40% -> "degradation concentrated".
Reported per seed at r1 and r10, plus the all-round trajectory. A per-ESS
group ablation is a candidate suggestion only and needs separate approval.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "algorithm_diagnostics"
JSON_PATH = OUT / "c0_d4_per_ess.json"
MD_PATH = OUT / "c0_d4_per_ess.md"
SEEDS = (11, 23)
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")
BOOTSTRAP = 1000
RNG_SEED = 20261109
LABEL = "NON_FORMAL_DIAGNOSTIC_PILOT"


def day_shares(day: dict) -> dict:
    neg = {nm: -float(day["delta"][nm]) for nm in NAMES
           if float(day["delta"][nm]) < 0}
    total = sum(neg.values())
    if total <= 0:
        return {}
    return {nm: v / total for nm, v in neg.items()}


def boot_ci(shares_by_day: list, nm: str, rng) -> tuple:
    vals = np.asarray([s.get(nm, 0.0) for s in shares_by_day])
    if vals.size == 0:
        return (None, None, None)
    med = float(np.median(vals))
    if vals.size < 2:
        return (med, None, None)
    boots = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, vals.size, vals.size)
        boots.append(float(np.median(vals[idx])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (med, float(lo), float(hi))


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    report = {"label": LABEL, "generated_at":
              time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "criteria": {"median_share": 0.50, "ci_lower": 0.40,
                           "bootstrap": BOOTSTRAP, "rng_seed": RNG_SEED,
                           "cluster_unit": "validation day"},
              "seeds": {}, "runtime_s": None}
    any_concentrated = False
    for seed in SEEDS:
        run_dir = ROOT / "runs" / f"FORMAL_7834d0aa_seed{seed}"
        history = json.loads((run_dir / "eval_history.json")
                             .read_text(encoding="utf-8"))
        rounds = {}
        for entry in history:
            r = int(entry["round"])
            neg_days = [d for d in entry["days"] if d["delta_fleet"] < 0]
            shares = [day_shares(d) for d in neg_days]
            shares = [s for s in shares if s]
            per_ess = {}
            for nm in NAMES:
                med, lo, hi = boot_ci(shares, nm, rng)
                per_ess[nm] = {"median_share": med, "ci_lo": lo,
                               "ci_hi": hi}
            rounds[r] = {"n_negative_days": len(neg_days),
                         "n_days": len(entry["days"]), "per_ess": per_ess}
        r10 = rounds[max(rounds)]
        conc = []
        for nm, st in r10["per_ess"].items():
            if (st["median_share"] is not None and st["ci_lo"] is not None
                    and st["median_share"] > 0.50 and st["ci_lo"] > 0.40):
                conc.append(nm)
        any_concentrated = any_concentrated or bool(conc)
        report["seeds"][str(seed)] = {
            "rounds": {str(r): rounds[r] for r in sorted(rounds)},
            "r10_concentrated": conc}
        print(f"seed{seed}: r10 negative days "
              f"{r10['n_negative_days']}/{r10['n_days']}, concentrated: "
              f"{conc or '无'}", flush=True)
    report["runtime_s"] = round(time.time() - t0, 1)
    report["verdict"] = ("DEGRADATION_CONCENTRATED" if any_concentrated
                         else "DEGRADATION_NOT_CONCENTRATED")
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# C0 D4:分 ESS 退化份额", "",
             f"label: {LABEL}; verdict: **{report['verdict']}**", "",
             "判据:负 ΔG 验证日上某 ESS 的负收益份额中位 >50% 且日聚类 "
             "bootstrap 95% CI 下界 >40% → 退化集中。份额=该日负 delta 的 "
             "|delta_i|/Σ|负 delta|。", ""]
    for seed, rep in report["seeds"].items():
        lines += [f"## seed{seed}", ""]
        for r in sorted(rep["rounds"], key=int):
            rr = rep["rounds"][r]
            lines.append(f"### round {r}(负日 {rr['n_negative_days']}/"
                         f"{rr['n_days']})")
            lines += ["", "| ESS | 份额中位 | CI 下界 | CI 上界 |",
                      "|---|---|---|---|"]
            for nm in NAMES:
                st = rr["per_ess"][nm]
                fmt = lambda v: "—" if v is None else f"{v:.3f}"
                lines.append(f"| {nm} | {fmt(st['median_share'])} "
                             f"| {fmt(st['ci_lo'])} | {fmt(st['ci_hi'])} |")
            lines.append("")
        lines.append(f"r10 退化集中:{rep['r10_concentrated'] or '无'}")
        lines.append("")
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"],
                      "runtime_s": report["runtime_s"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
