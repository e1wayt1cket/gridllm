"""C0 D3: quote-mapping saturation and gradient retention.

NON_FORMAL_DIAGNOSTIC_PILOT. Pure file analysis of the frozen formal runs
plus a static read of the frozen mapping code; no checkpoint load, no
clearing, no training.

Mapping facts (da_quote_map.py, frozen): per-leg boxes [0,150] (charge) and
[0,300] (discharge); quote = high * g(z), g = linear core 0.5+0.196z for
|z|<=2.5, quadratic tails to (0,1); dg/dz in (0, 0.196], polynomially decayed
in the tails, never zero. The plan's "5% edge of [0,300]" is corrected here
to a per-leg 5% edge band: charge <=7.5 or >=142.5; discharge <=15 or >=285.

Pre-registered criterion (b/C0任务书_草稿_2026-11-09.md v2):
  edge-band fraction of open-gate quotes, r1 -> r10 increase > 20 pp
  -> "mapping saturation significant".
Gradient retention (median dg/dz / SLOPE per round per leg) is descriptive.
Five-path static consistency cites the P1-A mapping regression evidence.
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
JSON_PATH = OUT / "c0_d3_mapping.json"
MD_PATH = OUT / "c0_d3_mapping.md"
P1_EVIDENCE = OUT / "P1_v4_evidence.json"
SEEDS = (11, 23)
LEGS = ("charge", "discharge")
HIGHS = qmap.LEG_HIGHS
EDGE = 0.05
LABEL = "NON_FORMAL_DIAGNOSTIC_PILOT"


def edge_band(leg: int) -> tuple:
    return (EDGE * HIGHS[leg], (1 - EDGE) * HIGHS[leg])


def analyse_quotes(values: np.ndarray, leg: int) -> dict:
    lo, hi = edge_band(leg)
    if values.size == 0:
        return {"n": 0, "edge_frac": None, "lo_frac": None, "hi_frac": None}
    return {"n": int(values.size),
            "edge_frac": float(np.mean((values <= lo) | (values >= hi))),
            "lo_frac": float(np.mean(values <= lo)),
            "hi_frac": float(np.mean(values >= hi))}


def training_mode(run_dir: Path) -> dict:
    per_round: dict[int, dict] = {}
    with (run_dir / "block_decisions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            r = int(d["round"])
            rec = per_round.setdefault(
                r, {lg: {"open_quotes": [], "logits": []}
                    for lg in range(2)})
            for leg in range(2):
                rec[leg]["logits"].append(float(d["price_logits"][leg]))
                if float(d["gates"][leg]) > 0.5:
                    rec[leg]["open_quotes"].append(
                        float(d["quotes"][leg]))
    out = {}
    for r in sorted(per_round):
        row = {}
        for leg in range(2):
            q = np.asarray(per_round[r][leg]["open_quotes"])
            z = np.asarray(per_round[r][leg]["logits"])
            deriv = qmap.map_derivative_np(z) / qmap.SLOPE
            row[LEGS[leg]] = {
                "open_quotes": analyse_quotes(q, leg),
                "n_logits": int(z.size),
                "grad_retention_median": float(np.median(deriv)),
                "grad_retention_p10": float(np.percentile(deriv, 10)),
                "frac_beyond_core": float(np.mean(np.abs(z) > qmap.Z_CORE)),
            }
        out[r] = row
    return out


def eval_mode(run_dir: Path) -> dict:
    history = json.loads((run_dir / "eval_history.json")
                         .read_text(encoding="utf-8"))
    out = {}
    for entry in history:
        r = int(entry["round"])
        row = {}
        for leg in range(2):
            vals = []
            for day in entry["days"]:
                for nm in day["quote_open_values"]:
                    vals.extend(day["quote_open_values"][nm][leg])
            row[LEGS[leg]] = analyse_quotes(np.asarray(vals), leg)
        out[r] = row
    return out


def mapping_consistency() -> dict:
    if not P1_EVIDENCE.exists():
        return {"status": "NOT_TESTED",
                "note": "P1_v4_evidence.json missing"}
    ev = json.loads(P1_EVIDENCE.read_text(encoding="utf-8"))
    mapping = ev.get("mapping", {})
    return {"status": "PASS" if mapping.get("all_pass") else "FAIL",
            "source": "P1_v4_evidence.json (G1c P1-A regression, V4D)",
            "checks": list(mapping.get("checks", {}).keys())}


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"label": LABEL, "generated_at":
              time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "criteria": {"edge_band_pp_increase": 20,
                           "edge_bands": {
                               LEGS[i]: {"lo": edge_band(i)[0],
                                         "hi": edge_band(i)[1]}
                               for i in range(2)}},
              "mapping_static": mapping_consistency(),
              "seeds": {}, "runtime_s": None}
    any_hit = False
    for seed in SEEDS:
        run_dir = ROOT / "runs" / f"FORMAL_7834d0aa_seed{seed}"
        tr, ev = training_mode(run_dir), eval_mode(run_dir)
        r1, r10 = min(tr), max(tr)
        pp = {}
        for leg in LEGS:
            a = tr[r1][leg]["open_quotes"]["edge_frac"]
            b = tr[r10][leg]["open_quotes"]["edge_frac"]
            if a is not None and b is not None:
                pp[leg] = round(100 * (b - a), 1)
        hit = any(v > 20 for v in pp.values())
        any_hit = any_hit or hit
        report["seeds"][str(seed)] = {
            "training_mode": {str(r): tr[r] for r in sorted(tr)},
            "eval_mode": {str(r): ev[r] for r in sorted(ev)},
            "edge_pp_r1_to_r10": pp, "significant": hit}
        print(f"seed{seed}: edge-band pp r1->r10 {pp} -> "
              f"{'SIGNIFICANT' if hit else 'no'}", flush=True)
    report["runtime_s"] = round(time.time() - t0, 1)
    report["verdict"] = ("MAPPING_SATURATION_SIGNIFICANT" if any_hit
                         else "MAPPING_SATURATION_NOT_SIGNIFICANT")
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# C0 D3:报价映射饱和与梯度留存", "",
             f"label: {LABEL}; verdict: **{report['verdict']}**", "",
             "判据:开腿报价贴边比例(各腿箱 5% 边缘带:充电 ≤7.5/≥142.5,"
             "放电 ≤15/≥285)r1→r10 增幅 >20pp → 饱和显著。", "",
             "注:计划稿的'箱 [0,300] 5% 边缘带'已按冻结代码更正为分腿"
             "(充电箱 [0,150]);映射为线性核+二次尾,梯度多项式衰减、"
             "永不为零。", "",
             f"五路径静态一致性:{report['mapping_static']['status']}"
             f"({report['mapping_static'].get('source', '')})", ""]
    for seed, rep in report["seeds"].items():
        lines += [f"## seed{seed}",
                  "",
                  "| round | 充电贴边 | 放电贴边 | 充电梯度留存中位 | "
                  "放电梯度留存中位 |",
                  "|---|---|---|---|---|"]
        for r in sorted(rep["training_mode"], key=int):
            row = rep["training_mode"][r]
            c, dch = row["charge"], row["discharge"]
            lines.append(
                f"| {r} | {c['open_quotes']['edge_frac']:.3f} "
                f"| {dch['open_quotes']['edge_frac']:.3f} "
                f"| {c['grad_retention_median']:.3f} "
                f"| {dch['grad_retention_median']:.3f} |")
        lines += ["", f"- 贴边增幅 r1→r10(pp): {rep['edge_pp_r1_to_r10']}",
                  ""]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"],
                      "runtime_s": report["runtime_s"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
