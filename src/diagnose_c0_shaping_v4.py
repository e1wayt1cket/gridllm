"""C0 D1: C2 potential-shaping contribution, decomposed by replay.

NON_FORMAL_DIAGNOSTIC_PILOT. Loads the frozen v4 formal checkpoints into an
in-memory trainer and replays the 11 validation days with the deterministic
evaluation path (hard gates, no exploration, no parameter update, no
checkpoint/buffer write). The B1 baseline is a registered cache lookup.

Gated on stage 0b (c0_d0_replay_consistency.json verdict PASS). The rollout
replicates `train_da_gate.eval_day` and additionally keeps the block-level
SOC path; a self-check asserts the rollout's delta_fleet equals `eval_day`
to 1e-6 on the first validation day of each checkpoint (the pattern proven
by `diagnose_actor_update_v4.rollout_check`).

Shaping terms are recomputed with the frozen `c2_reward.c2_reward` on a zero
base reward, i.e. exactly the training-time formula (gamma=0.99 from the
formal run's da_config).

Pre-registered criterion (b/C0任务书_草稿_2026-11-09.md v2):
  atomic unit ESS-day; S_i,d = sum over blocks |shaping|; ratio S/|B|;
  ESS-days with |G_policy| < 1 CNY AND |G_b1| < 1 CNY are uninformative and
  excluded (counted separately); day-clustered median >= 20% (bootstrap 95%
  CI over days, 1000 resamples) -> C0 candidate = C2 ablation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import rl_da_gate as mod
import train_da_gate as T
import v2_cy_jn_env as envmod
from c2_reward import C2Config, c2_reward

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "algorithm_diagnostics"
JSON_PATH = OUT / "c0_d1_shaping.json"
MD_PATH = OUT / "c0_d1_shaping.md"
GATE_PATH = OUT / "c0_d0_replay_consistency.json"
SEEDS = (11, 23)
NAMES = ("ESS9", "ESS14", "ESS17", "ESS27")
NOINFO_TOL = 1.0
BOOTSTRAP = 1000
RNG_SEED = 20261109
RATIO_THRESHOLD = 0.20
LABEL = "NON_FORMAL_DIAGNOSTIC_PILOT"


def rollout_soc(trainer, pair, b1_cache, *, mechanism_hash, pair_row):
    """eval_day replica that also returns the block-level SOC path."""
    env = envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    obs = env.reset()
    names = trainer.agent_names
    block_payoffs = {nm: [] for nm in names}
    soc_path = {nm: [float(next(a for a in env.all_agents
                                if a.name == nm).storage.soc0)]
                for nm in names}
    for block in range(24):
        acts = {}
        for nm in names:
            d = trainer.select_action(obs[nm], nm, train=False)
            acts[nm] = np.asarray(d["quotes"], dtype=float)
        next_obs, _, done, info = env.step(acts)
        if env._last_result is None:
            raise RuntimeError(f"{pair['pair_id']}: clearing failed "
                               f"at block {block}")
        for nm in names:
            block_payoffs[nm].append(
                float(info.get("raw_profit", {}).get(nm, 0.0)))
            soc_path[nm].append(
                float(env.prev_soc.get(nm, soc_path[nm][-1])))
        obs = next_obs
        if done:
            break
    pair_id = pair["pair_id"]
    b1_day = b1_cache.get(pair_id, 0.5, mechanism_hash,
                          mod.pair_data_hash(pair_row))
    g_pi = {nm: float(sum(block_payoffs[nm])) for nm in names}
    g_b1 = {nm: float(sum(b1_day.block_payoffs[nm])) for nm in names}
    return {"pair_id": pair_id, "G_policy": g_pi, "G_b1": g_b1,
            "delta": {nm: g_pi[nm] - g_b1[nm] for nm in names},
            "delta_fleet": float(sum(g_pi[nm] - g_b1[nm] for nm in names)),
            "soc_path": soc_path}


def shaping_terms(soc_path, gamma):
    zero = {nm: np.zeros(24) for nm in soc_path}
    shaped = c2_reward(zero, soc_path, done=None,
                       config=C2Config(gamma=gamma))
    return shaped


def cluster_bootstrap_median(ratios_by_day: dict, rng) -> tuple:
    days = list(ratios_by_day)
    all_vals = np.asarray([v for d in days for v in ratios_by_day[d]])
    if all_vals.size == 0:
        return (None, None, None)
    med = float(np.median(all_vals))
    boots = []
    for _ in range(BOOTSTRAP):
        pick = rng.choice(days, size=len(days), replace=True)
        vals = np.asarray([v for d in pick for v in ratios_by_day[d]])
        boots.append(float(np.median(vals)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (med, float(lo), float(hi))


def main() -> int:
    if not GATE_PATH.exists():
        print("stage 0b evidence missing; D1 BLOCKED", flush=True)
        return 2
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    if gate.get("verdict") != "PASS":
        print(f"stage 0b verdict {gate.get('verdict')}; D1 BLOCKED",
              flush=True)
        return 2

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    report = {"label": LABEL, "generated_at":
              time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "criteria": {"ratio_threshold": RATIO_THRESHOLD,
                           "noinfo_tol_cny": NOINFO_TOL,
                           "bootstrap": BOOTSTRAP, "rng_seed": RNG_SEED,
                           "cluster_unit": "validation day",
                           "atomic_unit": "ESS-day"},
              "seeds": {}, "runtime_s": None}
    import diagnose_c0_replay_v4 as stage0
    for seed in SEEDS:
        run_dir = ROOT / "runs" / f"FORMAL_7834d0aa_seed{seed}"
        best_dir = ROOT / "runs" / f"FORMAL_7834d0aa_seed{seed}_best"
        triples, manifest = stage0.load_val_pairs(run_dir)
        mech = manifest["b1_mechanism_hash"]
        gamma = float(manifest["da_config"]["gamma"])
        b1_cache = mod.load_b1_cache()
        template_env = envmod.build_env_for_pair(
            triples[0][1], differential_reward=False,
            terminal_soc_target=0.5)
        seed_rep = {"gamma": gamma, "checkpoints": {}}
        for label, ckroot in (("best_r1", best_dir), ("final_r10", run_dir)):
            trainer = mod.DAGateTrainer(
                template_env, mod.DAExperimentConfig("diagnostic"))
            trainer.load(ckroot)
            # self-check: rollout must equal eval_day on the first day
            pid0, pair0, row0 = triples[0]
            mine = rollout_soc(trainer, pair0, b1_cache,
                               mechanism_hash=mech, pair_row=row0)
            ref = T.eval_day(trainer, pair0, b1_cache,
                             mechanism_hash=mech, pair_row=row0)
            agree = abs(mine["delta_fleet"] - ref["delta_fleet"]) <= 1e-6
            days = {}
            ratios_by_day: dict[str, list] = {}
            excluded = []
            for pid, pair, row in triples:
                r = mine if pid == pid0 else rollout_soc(
                    trainer, pair, b1_cache, mechanism_hash=mech,
                    pair_row=row)
                shaped = shaping_terms(r["soc_path"], gamma)
                for nm in NAMES:
                    terms = np.asarray(shaped.shaping_terms[nm])
                    s_abs = float(np.sum(np.abs(terms)))
                    entry = {"G_policy": r["G_policy"][nm],
                             "G_b1": r["G_b1"][nm],
                             "B": r["delta"][nm],
                             "shaping_abs_sum": s_abs,
                             "shaping_net": float(np.sum(terms)),
                             "soc_min": float(np.min(r["soc_path"][nm])),
                             "soc_max": float(np.max(r["soc_path"][nm]))}
                    days[f"{pid}|{nm}"] = entry
                    noinfo = (abs(r["G_policy"][nm]) < NOINFO_TOL
                              and abs(r["G_b1"][nm]) < NOINFO_TOL)
                    if noinfo:
                        excluded.append(f"{pid}|{nm}")
                    elif abs(r["delta"][nm]) > 1e-9:
                        ratios_by_day.setdefault(pid, []).append(
                            s_abs / abs(r["delta"][nm]))
            med, lo, hi = cluster_bootstrap_median(ratios_by_day, rng)
            nets = [v["shaping_net"] for v in days.values()]
            hit = med is not None and med >= RATIO_THRESHOLD
            seed_rep["checkpoints"][label] = {
                "self_check_agrees_1e_6": agree,
                "n_ess_days": len(days), "n_excluded_noinfo": len(excluded),
                "excluded": excluded,
                "ratio_median": med, "ratio_ci_lo": lo, "ratio_ci_hi": hi,
                "shaping_net_max_abs": float(np.max(np.abs(nets))),
                "verdict": ("SHAPING_SIGNIFICANT" if hit
                            else "SHAPING_NOT_SIGNIFICANT"),
                "days": days}
            print(f"seed{seed} {label}: self-check {agree}, "
                  f"ratio median {med if med is None else round(med, 3)} "
                  f"CI [{None if lo is None else round(lo, 3)}, "
                  f"{None if hi is None else round(hi, 3)}], "
                  f"excluded {len(excluded)}, "
                  f"net |max| {max(abs(n) for n in nets):.4f}", flush=True)
        report["seeds"][str(seed)] = seed_rep
    report["runtime_s"] = round(time.time() - t0, 1)
    finals = [rep["checkpoints"]["final_r10"]["verdict"]
              for rep in report["seeds"].values()]
    report["verdict"] = ("SHAPING_SIGNIFICANT"
                         if any(v == "SHAPING_SIGNIFICANT" for v in finals)
                         else "SHAPING_NOT_SIGNIFICANT")
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# C0 D1:C2 shaping 贡献(回放分解)", "",
             f"label: {LABEL}; verdict: **{report['verdict']}**", "",
             "判据:ESS-日 级 S/|B|(S=逐 block |shaping| 之和,B=全日差分)"
             "的日聚类中位 ≥20% → C0 candidate = C2 消融。无信息 ESS-日"
             "(|G_policy|<1 且 |G_b1|<1 CNY)剔除并单列。", ""]
    for seed, rep in report["seeds"].items():
        lines += [f"## seed{seed}", "",
                  "| checkpoint | 自检 | N(剔除) | 比例中位 | 95% CI | "
                  "净 shaping |max| | 判定 |",
                  "|---|---|---|---|---|---|---|"]
        for label, cp in rep["checkpoints"].items():
            fmt = lambda v: "—" if v is None else f"{v:.3f}"
            lines.append(
                f"| {label} | {'Y' if cp['self_check_agrees_1e_6'] else 'N'} "
                f"| {cp['n_ess_days']}({cp['n_excluded_noinfo']}) "
                f"| {fmt(cp['ratio_median'])} "
                f"| [{fmt(cp['ratio_ci_lo'])}, {fmt(cp['ratio_ci_hi'])}] "
                f"| {cp['shaping_net_max_abs']:.4f} | {cp['verdict']} |")
        lines.append("")
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"],
                      "runtime_s": report["runtime_s"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
