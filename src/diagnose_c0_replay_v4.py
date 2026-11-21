"""C0 stage 0b: replay consistency of the frozen v4 formal checkpoints.

NON_FORMAL_DIAGNOSTIC_PILOT. Read-only: loads the frozen checkpoints into an
in-memory trainer and re-evaluates the 11 validation days with the exact
function that produced eval_history (`train_da_gate.eval_day`, deterministic
hard gates, no exploration, no parameter update, no checkpoint/buffer write).
The B1 baseline is a registered cache lookup; all 11 validation pairs are
already cached, so nothing is written.

Gate (pre-registered in b/C0任务书_草稿_2026-11-09.md v2):
  PASS  := for both seeds and both checkpoints, >=10 of 11 days satisfy
           |replay - logged| <= max(0.5 CNY, 1% x |logged|) on delta_fleet
           and per-ESS delta, and |mean_fleet_delta diff| <= 1 CNY.
  Otherwise D1 is BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import rl_da_gate as mod
import train_da_gate as T
import v2_cy_jn_env as envmod

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "algorithm_diagnostics"
JSON_PATH = OUT / "c0_d0_replay_consistency.json"
MD_PATH = OUT / "c0_d0_replay_consistency.md"
SEEDS = (11, 23)
TOL_ABS = 0.5
TOL_REL = 0.01
MEAN_TOL = 1.0
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")
LABEL = "NON_FORMAL_DIAGNOSTIC_PILOT"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_provenance(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    step = int(manifest["step"])
    ckpt = root / "checkpoints" / f"step_{step:08d}"
    actor_hashes = {
        p.name: sha256_file(p)[:16]
        for p in sorted((ckpt / "actors").glob("*.pt"))}
    return {"root": str(root.relative_to(ROOT)), "step": step,
            "run_id": manifest.get("run_id"),
            "manifest_sha256_16": sha256_file(root / "manifest.json")[:16],
            "actor_sha256_16": actor_hashes}


def load_val_pairs(run_dir: Path) -> list:
    manifest = json.loads((run_dir / "run_manifest.json")
                          .read_text(encoding="utf-8"))
    val_ids = manifest["validation_pairs"]
    full = {f"{r['load_date']}|{r['price_date']}": r
            for r in T.read_csv(T.FULL_PAIRING)}
    near = {f"{r['load_date']}|{r['price_date']}": r
            for r in T.read_csv(T.NEAR_PAIRING)}
    triples = []
    for pid in val_ids:
        row = full[pid]
        triples.append((pid,
                        {"load_date": row["load_date"],
                         "price_date": row["price_date"],
                         "season": near.get(pid, {}).get("season", ""),
                         "pair_id": pid},
                        row))
    return triples, manifest


def tol(logged: float) -> float:
    return max(TOL_ABS, TOL_REL * abs(logged))


def compare(day: dict, logged: dict) -> dict:
    fleet_diff = abs(day["delta_fleet"] - logged["delta_fleet"])
    unit_diffs = {nm: abs(day["delta"][nm] - logged["delta"][nm])
                  for nm in NAMES}
    unit_pass = all(unit_diffs[nm] <= tol(logged["delta"][nm])
                    for nm in NAMES)
    return {"pair_id": logged["pair_id"],
            "logged_fleet": logged["delta_fleet"],
            "replay_fleet": day["delta_fleet"],
            "fleet_diff": fleet_diff,
            "fleet_pass": fleet_diff <= tol(logged["delta_fleet"]),
            "unit_diffs": unit_diffs,
            "unit_pass": unit_pass,
            "pass": fleet_diff <= tol(logged["delta_fleet"]) and unit_pass}


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"label": LABEL, "generated_at":
              time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "tolerance": {"abs_cny": TOL_ABS, "rel": TOL_REL,
                            "mean_fleet_abs_cny": MEAN_TOL},
              "frozen_prefix": "7834d0aa", "seeds": {}, "runtime_s": None}
    overall = True
    for seed in SEEDS:
        run_dir = ROOT / "runs" / f"FORMAL_7834d0aa_seed{seed}"
        best_dir = ROOT / "runs" / f"FORMAL_7834d0aa_seed{seed}_best"
        triples, manifest = load_val_pairs(run_dir)
        history = json.loads((run_dir / "eval_history.json")
                             .read_text(encoding="utf-8"))
        logged_by_round = {int(r["round"]): {d["pair_id"]: d
                                             for d in r["days"]}
                           for r in history}
        mech = manifest["b1_mechanism_hash"]
        b1_cache = mod.load_b1_cache()
        template_env = envmod.build_env_for_pair(
            triples[0][1], differential_reward=False, terminal_soc_target=0.5)
        seed_rep = {"validation_pairs": [t[0] for t in triples],
                    "b1_mechanism_hash": mech, "checkpoints": {}}
        for label, ckroot, round_no in (
                ("best_r1", best_dir, 1), ("final_r10", run_dir, 10)):
            prov = checkpoint_provenance(ckroot)
            trainer = mod.DAGateTrainer(
                template_env, mod.DAExperimentConfig("diagnostic"))
            trainer.load(ckroot)
            rows = []
            for pid, pair, row in triples:
                day = T.eval_day(trainer, pair, b1_cache,
                                 mechanism_hash=mech, pair_row=row)
                cmp_row = compare(day, logged_by_round[round_no][pid])
                rows.append(cmp_row)
                print(f"seed{seed} {label} {pid}: replay "
                      f"{cmp_row['replay_fleet']:.3f} vs logged "
                      f"{cmp_row['logged_fleet']:.3f} "
                      f"(diff {cmp_row['fleet_diff']:.4f})", flush=True)
            n_pass = sum(1 for r in rows if r["pass"])
            replay_mean = float(np.mean([r["replay_fleet"] for r in rows]))
            logged_mean = float(np.mean([r["logged_fleet"] for r in rows]))
            mean_diff = abs(replay_mean - logged_mean)
            ok = n_pass >= 10 and mean_diff <= MEAN_TOL
            overall = overall and ok
            seed_rep["checkpoints"][label] = {
                "provenance": prov, "round": round_no,
                "days_pass": n_pass, "days_total": len(rows),
                "replay_mean_fleet_delta": replay_mean,
                "logged_mean_fleet_delta": logged_mean,
                "mean_diff": mean_diff, "pass": ok, "days": rows}
            print(f"seed{seed} {label}: {n_pass}/11 days within tol, "
                  f"mean diff {mean_diff:.4f} -> "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
        report["seeds"][str(seed)] = seed_rep
    report["runtime_s"] = round(time.time() - t0, 1)
    report["verdict"] = "PASS" if overall else "FAIL"
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# C0 阶段 0b:回放一致性", "",
             f"label: {LABEL}; frozen prefix 7834d0aa; "
             f"verdict: **{report['verdict']}**", "",
             "判据:11 日中 ≥10 日 |replay−logged| ≤ max(0.5 CNY, 1%×|logged|)"
             "(delta_fleet 与 per-ESS delta),且 mean_fleet_delta 差 ≤ 1 CNY。",
             ""]
    for seed, seed_rep in report["seeds"].items():
        for label, cp in seed_rep["checkpoints"].items():
            lines += [f"## seed{seed} {label} (round {cp['round']})",
                      f"- checkpoint `{cp['provenance']['root']}` step "
                      f"{cp['provenance']['step']}, manifest sha "
                      f"{cp['provenance']['manifest_sha256_16']}",
                      f"- days pass {cp['days_pass']}/{cp['days_total']}, "
                      f"replay mean {cp['replay_mean_fleet_delta']:.3f} vs "
                      f"logged {cp['logged_mean_fleet_delta']:.3f} "
                      f"(diff {cp['mean_diff']:.4f}) -> "
                      f"{'PASS' if cp['pass'] else 'FAIL'}", "",
                      "| day | logged | replay | diff | pass |",
                      "|---|---|---|---|---|"]
            for r in cp["days"]:
                lines.append(f"| {r['pair_id']} | {r['logged_fleet']:.3f} "
                             f"| {r['replay_fleet']:.3f} "
                             f"| {r['fleet_diff']:.4f} "
                             f"| {'Y' if r['pass'] else 'N'} |")
            lines.append("")
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"],
                      "runtime_s": report["runtime_s"]}))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
