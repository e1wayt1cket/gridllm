"""C0 D2: gate straight-through / sampling analysis, dual mode.

NON_FORMAL_DIAGNOSTIC_PILOT. Pure file analysis of the frozen formal runs;
no checkpoint load, no clearing, no training.

Pre-registered criteria (b/C0任务书_草稿_2026-11-09.md v2):
  training mode (block_decisions.jsonl, Bernoulli-sampled gates):
    T1: r10 fraction of ESS-leg-block records with prob in [0.2, 0.8] > 30%
    T2: any ESS-leg E|g - p| at r10 > 1.5x its r1 value
  eval mode (eval_history.json, deterministic hard gates):
    E1: r10 fraction of ESS-leg-days with gate_mean_probability in
        [0.2, 0.8] > 30%
  Any of T1/T2/E1 -> "ST/gate significant" -> recommend C3 earlier.
Training-mode and eval-mode numbers are never mixed in one criterion.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "algorithm_diagnostics"
JSON_PATH = OUT / "c0_d2_gate_st.json"
MD_PATH = OUT / "c0_d2_gate_st.md"
SEEDS = (11, 23)
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")
LEGS = ("charge", "discharge")
UNC_LO, UNC_HI = 0.2, 0.8
LABEL = "NON_FORMAL_DIAGNOSTIC_PILOT"


def training_mode(run_dir: Path) -> dict:
    """Per round: uncertain-prob fraction and per-leg E|g-p|."""
    per_round: dict[int, dict] = {}
    leg_err: dict[int, dict[str, list]] = {}
    with (run_dir / "block_decisions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            r = int(d["round"])
            rec = per_round.setdefault(r, {"n": 0, "unc": 0})
            err = leg_err.setdefault(r, {f"{nm}.{lg}": []
                                         for nm in NAMES for lg in LEGS})
            for i, lg in enumerate(LEGS):
                p = float(d["probs"][i])
                g = float(d["gates"][i])
                rec["n"] += 1
                if UNC_LO <= p <= UNC_HI:
                    rec["unc"] += 1
                err[f"{d['agent']}.{lg}"].append(abs(g - p))
    out = {}
    for r in sorted(per_round):
        rec = per_round[r]
        out[r] = {
            "n_records": rec["n"],
            "uncertain_frac": rec["unc"] / max(rec["n"], 1),
            "e_abs_g_minus_p": {k: float(np.mean(v)) if v else None
                                for k, v in leg_err[r].items()},
        }
    return out


def eval_mode(run_dir: Path) -> dict:
    """Per round: fraction of ESS-leg-days with mean prob in [0.2, 0.8]."""
    history = json.loads((run_dir / "eval_history.json")
                         .read_text(encoding="utf-8"))
    out = {}
    for entry in history:
        r = int(entry["round"])
        n = unc = 0
        for day in entry["days"]:
            for nm in NAMES:
                for i, _lg in enumerate(LEGS):
                    p = float(day["gate_mean_probability"][nm][i])
                    n += 1
                    if UNC_LO <= p <= UNC_HI:
                        unc += 1
        out[r] = {"n_ess_leg_days": n, "uncertain_frac": unc / max(n, 1)}
    return out


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"label": LABEL, "generated_at":
              time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "criteria": {"T1": "r10 train uncertain prob frac > 0.30",
                           "T2": "any leg E|g-p| r10 > 1.5x r1",
                           "E1": "r10 eval uncertain prob frac > 0.30",
                           "uncertain_band": [UNC_LO, UNC_HI]},
              "seeds": {}, "runtime_s": None}
    any_hit = False
    for seed in SEEDS:
        run_dir = ROOT / "runs" / f"FORMAL_7834d0aa_seed{seed}"
        tr = training_mode(run_dir)
        ev = eval_mode(run_dir)
        r1, r10 = min(tr), max(tr)
        t1 = tr[r10]["uncertain_frac"] > 0.30
        rises = {}
        for leg, v in tr[r10]["e_abs_g_minus_p"].items():
            base = tr[r1]["e_abs_g_minus_p"][leg]
            if v is not None and base:
                rises[leg] = v / base
        t2_legs = {k: round(v, 3) for k, v in rises.items() if v > 1.5}
        e1 = ev[max(ev)]["uncertain_frac"] > 0.30
        hit = bool(t1 or t2_legs or e1)
        any_hit = any_hit or hit
        report["seeds"][str(seed)] = {
            "training_mode": {str(r): tr[r] for r in sorted(tr)},
            "eval_mode": {str(r): ev[r] for r in sorted(ev)},
            "T1_hit": t1, "T2_hit_legs_ratio_vs_r1": t2_legs, "E1_hit": e1,
            "significant": hit}
        print(f"seed{seed}: T1={t1} (r10 unc "
              f"{tr[r10]['uncertain_frac']:.3f}), T2 legs={t2_legs}, "
              f"E1={e1} (r10 unc {ev[max(ev)]['uncertain_frac']:.3f})",
              flush=True)
    report["runtime_s"] = round(time.time() - t0, 1)
    report["verdict"] = ("ST_GATE_SIGNIFICANT" if any_hit
                         else "ST_GATE_NOT_SIGNIFICANT")
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# C0 D2:Gate ST / 采样双模式分析", "",
             f"label: {LABEL}; verdict: **{report['verdict']}**", "",
             "判据:T1 训练模式 r10 高不确定门控占比>30%;T2 任一腿 "
             "E|g−p| r10 较 r1 升 >50%;E1 评估模式 r10 高不确定占比>30%。"
             "两模式分列,不混用。", ""]
    for seed, rep in report["seeds"].items():
        lines += [f"## seed{seed}", "",
                  "| round | train unc frac | eval unc frac |",
                  "|---|---|---|"]
        for r in sorted(rep["training_mode"], key=int):
            tm = rep["training_mode"][r]
            em = rep["eval_mode"].get(r, {})
            lines.append(f"| {r} | {tm['uncertain_frac']:.3f} "
                         f"| {em.get('uncertain_frac', float('nan')):.3f} |")
        lines += ["",
                  f"- T1: {'HIT' if rep['T1_hit'] else 'no'}; "
                  f"E1: {'HIT' if rep['E1_hit'] else 'no'}; "
                  f"T2 legs(r10/r1 比值>1.5): "
                  f"{rep['T2_hit_legs_ratio_vs_r1'] or '无'}", ""]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"],
                      "runtime_s": report["runtime_s"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
