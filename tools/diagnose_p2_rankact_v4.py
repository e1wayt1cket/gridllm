"""Chunked runner for the P2 `rankact` stage (activity-selected ranking sweep).

The monolithic `diagnose_p2.py rankact` was killed by the environment's memory
reaper after 4 of 11 states: one process held two traced days plus ~500
clearings. This runner keeps every process small:

  pickle     -- trace the two winter Train days, select the activity states
                with the exact `diagnose_p2.activity_states` logic and pickle
                them (one short process, no sweeps);
  sweep <i>  -- one fresh process per state: five quote points, writes
                `rankact_chunks_v4/state_<i>.csv` and `state_<i>.json`;
  merge      -- rebuild `state_records` from the chunk JSONs, call the exact
                `diagnose_p2.summarize_rank`, write the merged
                `P2_winter_critic_ranking_activity.csv` and update
                `P2_evidence.json -> rank_activity`.

Read-only with respect to every frozen artifact, same as `diagnose_p2`: no
training, no buffer add, no checkpoint writes; the only writes are under
`reports/algorithm_diagnostics`.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import diagnose_p2_v4 as P2

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = P2.OUT / "rankact_chunks_v4"
STATES_PKL = CHUNK_DIR / "states.pkl"
META_PKL = CHUNK_DIR / "meta.pkl"


def _build_states() -> None:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    triples_, mechanism_hash = P2.triples()
    by_id = {pid: (pair, row) for pid, pair, row in triples_}
    b1_cache = P2.mod.load_b1_cache()
    trainer, _ = P2.load_trainer(P2.CHECKPOINTS[23], triples_[0][1])
    states, selected = [], {}
    for pid in P2.WINTER_PAIRS:
        pair, row = by_id[pid]
        t0 = time.time()
        trace = P2.trace_day(trainer, pair, b1_cache, mechanism_hash, row)
        day_states = P2.activity_states(trace, pid, pair, row, per_leg=2)
        selected[pid] = [[s["block"], s["unit"],
                          "charge" if s["leg"] == 0 else "discharge"]
                         for s in day_states]
        states.extend(day_states)
        print(f"traced {pid}: {len(trace)} blocks, "
              f"{len(day_states)} states ({time.time() - t0:.0f}s)",
              flush=True)
        del trace
    with STATES_PKL.open("wb") as fh:
        pickle.dump(states, fh)
    with META_PKL.open("wb") as fh:
        pickle.dump({"selected": selected, "n_states": len(states)}, fh)
    print(f"pickled {len(states)} states -> {STATES_PKL}", flush=True)


def _sweep_one(index: int) -> None:
    with STATES_PKL.open("rb") as fh:
        states = pickle.load(fh)
    st = states[index]
    triples_, mechanism_hash = P2.triples()
    by_id = {pid: (pair, row) for pid, pair, row in triples_}
    pair, row = by_id[st["pair_id"]]
    b1_cache = P2.mod.load_b1_cache()
    trainer, _ = P2.load_trainer(P2.CHECKPOINTS[23], triples_[0][1])

    grid = P2.CHARGE_GRID if st["leg"] == 0 else P2.DISCHARGE_GRID
    points, rows = [], []
    t0 = time.time()
    for quote in grid:
        acts = {k: v.copy() for k, v in st["acts"].items()}
        acts[st["unit"]][st["leg"]] = float(quote)
        q = P2.q_at_state(trainer, st["obs"], acts, st["unit"], st["leg"])
        roll = P2.rollout_override(trainer, pair, b1_cache, mechanism_hash,
                                   row, st["block"], st["unit"], st["leg"],
                                   quote)
        points.append({
            "quote": float(quote), "q1": q["q1"], "q2": q["q2"],
            "q_min": q["q_min"], "twin_gap": q["twin_gap"],
            "dq_dquote": q["dq_dquote"],
            "payoff_unit": roll["payoff"][st["unit"]],
            "payoff_fleet": roll["fleet"],
            "accept_mwh": roll["accept_mwh"],
            "clearing_failed": roll["clearing_failed"],
            "soc_before": roll["soc_before"],
        })
        rows.append({
            "state_id": index, "pair_id": st["pair_id"],
            "block": st["block"], "unit": st["unit"],
            "leg": ("charge" if st["leg"] == 0 else "discharge"),
            "gate_open": st["gate_open"],
            "leg_switched": st["leg_switched"],
            "quote": float(quote), "q1": q["q1"], "q2": q["q2"],
            "q_min": q["q_min"], "twin_gap": q["twin_gap"],
            "dq_dquote": q["dq_dquote"],
            "payoff_remaining_unit": roll["payoff"][st["unit"]],
            "payoff_remaining_fleet": roll["fleet"],
            "accept_mwh": roll["accept_mwh"],
            "soc_before": roll["soc_before"],
            "clearing_failed": roll["clearing_failed"],
        })
        print(f"state {index} quote {quote}: payoff_unit "
              f"{roll['payoff'][st['unit']]:+.6f} "
              f"accept {roll['accept_mwh']:.4g} MWh "
              f"({time.time() - t0:.0f}s)", flush=True)
    record = {**{k: st[k] for k in ("pair_id", "block", "unit",
                                    "leg_switched", "gate_open", "soc")},
              "leg": ("charge" if st["leg"] == 0 else "discharge"),
              "points": points}
    csv_path = CHUNK_DIR / f"state_{index:02d}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=P2.RANK_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path = CHUNK_DIR / f"state_{index:02d}.json"
    json_path.write_text(json.dumps(P2.jsonable(record), indent=2),
                         encoding="utf-8")
    print(f"state {index} done -> {csv_path} ({time.time() - t0:.0f}s)",
          flush=True)


def _merge() -> int:
    with META_PKL.open("rb") as fh:
        meta = pickle.load(fh)
    n = meta["n_states"]
    records, rows = [], []
    for i in range(n):
        json_path = CHUNK_DIR / f"state_{i:02d}.json"
        csv_path = CHUNK_DIR / f"state_{i:02d}.csv"
        if not json_path.exists() or not csv_path.exists():
            print(f"missing chunk {i}; run `sweep {i}` first", flush=True)
            return 2
        records.append(json.loads(json_path.read_text(encoding="utf-8")))
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows.extend(list(csv.DictReader(fh)))
    summary = P2.summarize_rank(records)
    summary.update({"checkpoint_seed": 23,
                    "pairs": list(P2.WINTER_PAIRS),
                    "blocks": list(P2.RANK_BLOCKS),
                    "units": list(P2.RANK_UNITS),
                    "selected_states": meta["selected"],
                    "runner": "diagnose_p2_rankact_chunk.py (chunked, "
                              "one process per state; memory-reaper "
                              "workaround for the monolithic rankact)"})
    with P2.RANK_ACTIVITY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=P2.RANK_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    evidence = P2.load_evidence()
    evidence["rank_activity"] = summary
    P2.save_evidence(evidence, "PARTIAL_IN_PROGRESS")
    print(json.dumps({k: summary[k] for k in
                      ("states_total", "states_open", "no_signal",
                       "median_spearman", "top1_agreement",
                       "pairwise_accuracy", "gradient_direction_agreement",
                       "ranking_verdict")},
                     indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
          flush=True)
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: diagnose_p2_rankact_chunk.py pickle|sweep <i>|merge",
              flush=True)
        return 2
    mode = sys.argv[1]
    if mode == "pickle":
        _build_states()
        return 0
    if mode == "sweep":
        _sweep_one(int(sys.argv[2]))
        return 0
    if mode == "merge":
        return _merge()
    print(f"unknown mode {mode}", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
