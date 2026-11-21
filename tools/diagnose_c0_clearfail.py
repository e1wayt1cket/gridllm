"""Read-only diagnosis: why did C0PILOT_COMMON_B_seed11 r2 eval fail clearing?

NON_FORMAL_DIAGNOSTIC_PILOT. Replays the failed validation day
(2022-12-23|2025-12-24) with the run's latest checkpoint (r2 policy) in eval
mode. rl_env swallows the clearing exception (`except Exception: result =
None`), so this script wraps rl_env.clear_market to record the traceback
before re-raising, and dumps SOC trajectory, quotes and solver metadata at
the failing block. No training, no checkpoint writes, no file modifications
outside reports/algorithm_diagnostics/.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

import market as marketmod
import rl_da_gate as mod
import v2_cy_jn_env as envmod
import train_da_gate as T
from dispatch_core import last_solve_metadata

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "C0PILOT_COMMON_B_seed11"
CRASH_PAIR = "2022-12-23|2025-12-24"
OUT = ROOT / "reports" / "algorithm_diagnostics" / "c0_clearfail_diag.json"

captured = {"traceback": None, "block": None}
_orig_clear = marketmod.clear_market


def capturing_clear(*args, **kwargs):
    try:
        return _orig_clear(*args, **kwargs)
    except Exception:
        captured["traceback"] = traceback.format_exc()
        raise


def main() -> int:
    # rl_env imports clear_market locally from market at each step, so
    # patching the market module attribute intercepts every call.
    marketmod.clear_market = capturing_clear

    doc = json.loads((ROOT / "data/processed/v2_cy_jn_2025"
                      / "common_split.json").read_text(encoding="utf-8"))
    view = doc["near"]
    # Replicates train_da_gate.main()'s nested pairs_from_ids (not importable).
    near_rows = T.read_csv(T.NEAR_PAIRING)
    full_rows = T.read_csv(T.FULL_PAIRING)
    near_by_pair = {f"{r['load_date']}|{r['price_date']}": r
                    for r in near_rows}
    full_by_pair = {f"{r['load_date']}|{r['price_date']}": r
                    for r in full_rows}

    def pairs_from_ids(pair_ids):
        return [{"load_date": near_by_pair[pid]["load_date"],
                 "price_date": full_by_pair[pid]["price_date"],
                 "season": near_by_pair[pid].get("season", ""),
                 "pair_id": pid} for pid in pair_ids]

    train = pairs_from_ids(view["train"])
    validation = pairs_from_ids(view["validation"])
    pair = next(p for p in validation
                if f"{p['load_date']}|{p['price_date']}" == CRASH_PAIR)

    saved = json.loads((RUN / "run_manifest.json").read_text(
        encoding="utf-8"))
    da_cfg = dict(saved["da_config"])
    da_cfg.pop("experiment_id", None)
    da_cfg.pop("reward_mode", None)
    template_env = envmod.build_env_for_pair(
        train[0], differential_reward=False, terminal_soc_target=0.5)
    experiment = mod.DAExperimentConfig("DIAG_CLEARFAIL", "B", **da_cfg)
    trainer = mod.DAGateTrainer(template_env, experiment)
    bundle = trainer.load(RUN)
    print(f"loaded checkpoint step={bundle['manifest']['step']} "
          f"episodes={trainer.episodes}")

    env = envmod.build_env_for_pair(
        pair, differential_reward=False, terminal_soc_target=0.5)
    obs = env.reset()
    names = trainer.agent_names
    log = []
    fail_block = None
    for block in range(24):
        acts = {}
        quotes_rec = {}
        for nm in names:
            d = trainer.select_action(obs[nm], nm, train=False)
            acts[nm] = np.asarray(d["quotes"], dtype=float)
            quotes_rec[nm] = {
                "quotes": [float(x) for x in d["quotes"]],
                "price_quotes": [float(x) for x in d["price_quotes"]],
                "probs": [float(x) for x in d["probs"]],
                "gates": [float(x) for x in d["gates"]],
            }
        soc_before = {nm: float(env.prev_soc.get(nm, 0.5)) for nm in names}
        next_obs, _, done, info = env.step(acts)
        log.append({"block": block, "soc_before": soc_before,
                    "quotes": quotes_rec,
                    "cleared": env._last_result is not None})
        if env._last_result is None:
            fail_block = block
            break
        obs = next_obs
        if done:
            break

    meta = last_solve_metadata()
    print(f"fail_block={fail_block}")
    print(f"solve metadata: {json.dumps(meta, default=str)[:400]}")
    if captured["traceback"]:
        print("=== captured traceback (swallowed by rl_env) ===")
        print(captured["traceback"][-2000:])

    report = {"label": "NON_FORMAL_DIAGNOSTIC_PILOT",
              "run": RUN.name, "pair": CRASH_PAIR,
              "checkpoint_step": int(bundle["manifest"]["step"]),
              "fail_block": fail_block,
              "solve_metadata": {k: str(v) for k, v in meta.items()},
              "traceback": captured["traceback"],
              "blocks": log}
    OUT.write_text(json.dumps(report, indent=2, default=str),
                   encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
