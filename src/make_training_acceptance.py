"""Build the acceptance artifacts for the development training run."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path.cwd()
RUN_ID = "EXPLORATORY_NEAR_B_C2_GATE_seed11_v2"
RUN = ROOT / "runs" / RUN_ID
SNAP = ROOT / "snapshots" / RUN_ID
DATA = ROOT / "data/processed/v2_cy_jn_2025"
REPORTS = ROOT / "reports/v2_cy_jn_2025"
UNITS = ("ESS9", "ESS14", "ESS17", "ESS27")
GATES = tuple(f"{u}.{leg}" for u in UNITS for leg in ("charge", "discharge"))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    audit = load(RUN / "validation_audit_by_round.json")
    c2 = load(DATA / "c2_soc_training_audit.json")
    index = load(RUN / "checkpoint_index.json")
    manifest = load(RUN / "run_manifest.json")
    episodes = read_csv(RUN / "episodes.csv")
    gate_rows = [json.loads(line) for line in
                 (RUN / "gate_stats.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    # ------------------------------------------------------------------
    # Canonical round table
    # ------------------------------------------------------------------
    columns = ["round", "episode", "step", "mean_fleet_delta",
               "median_fleet_delta", "positive_days"]
    columns += [f"delta_{u}" for u in UNITS]
    columns += [f"charge_mwh_{u}" for u in UNITS]
    columns += [f"discharge_mwh_{u}" for u in UNITS]
    columns += [f"cycles_{u}" for u in UNITS]
    columns += [f"terminal_soc_{u}" for u in UNITS]
    columns += [f"gate_open_{g}" for g in GATES]
    columns += [f"gate_prob_{g}" for g in GATES]
    columns += ["distinct_modes", "permanent_open", "permanent_closed",
                "saturation_upper", "saturation_lower",
                "voltage_violation_cells", "line_overload_cells",
                "terminal_soc_max_err", "soc_min", "soc_max",
                "raw_soc_band_violation", "reproduced"]
    rows = []
    for n in range(1, 11):
        r = audit[str(n)]
        v = r["validation"]
        g = v["gate"]
        row = {
            "round": n, "episode": r["episode"], "step": r["step"],
            "mean_fleet_delta": v["mean_fleet_delta"],
            "median_fleet_delta": v["median_fleet_delta"],
            "positive_days": v["positive_days"],
            "distinct_modes": g["distinct_modes"],
            "permanent_open": ";".join(g["permanent_open"]),
            "permanent_closed": ";".join(g["permanent_closed"]),
            "saturation_upper": g["saturation_upper"],
            "saturation_lower": g["saturation_lower"],
            "voltage_violation_cells": v["voltage_violation_cells"],
            "line_overload_cells": v["line_overload_cells"],
            "terminal_soc_max_err": v["terminal_soc_max_err"],
            "soc_min": v["soc_min"], "soc_max": v["soc_max"],
            "raw_soc_band_violation": v["raw_soc_band_violation"],
            "reproduced": r["reproduced"],
        }
        for u in UNITS:
            row[f"delta_{u}"] = v["mean_delta_by_unit"][u]
            row[f"charge_mwh_{u}"] = v["mean_charge_mwh"][u]
            row[f"discharge_mwh_{u}"] = v["mean_discharge_mwh"][u]
            row[f"cycles_{u}"] = v["mean_cycles"][u]
            row[f"terminal_soc_{u}"] = v["mean_terminal_soc"][u]
        for gate in GATES:
            row[f"gate_open_{gate}"] = g["open_frequency"][gate]
            row[f"gate_prob_{gate}"] = g["mean_probability"][gate]
        rows.append(row)
    with (DATA / "training_validation_rounds.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    gate_payload = {
        str(n): audit[str(n)]["validation"]["gate"] for n in range(1, 11)}
    (DATA / "gate_behavior_by_round.json").write_text(
        json.dumps(gate_payload, indent=2), encoding="utf-8")

    batch = int(manifest["experiment"]["batch_size"])
    total_steps = int(manifest["total_steps"])
    update_calls = total_steps - batch + 1
    actor_calls = sum(1 for s in range(batch, total_steps + 1)
                      if s % int(manifest["experiment"]["policy_delay"]) == 0)
    episode_audit = {
        "run_id": RUN_ID,
        "status": manifest["status"],
        "budget_episodes": manifest["budget_episodes"],
        "episodes_run": len(episodes),
        "unique_episode_ids": len({int(e["episode"]) for e in episodes}),
        "episode_id_min": min(int(e["episode"]) for e in episodes),
        "episode_id_max": max(int(e["episode"]) for e in episodes),
        "episode_duplicates": len(episodes)
        - len({int(e["episode"]) for e in episodes}),
        "rounds": {str(k): sum(1 for e in episodes
                               if int(e["round"]) == k)
                   for k in range(1, 11)},
        "block_steps": total_steps,
        "blocks_per_episode": 24,
        "update_calls": update_calls,
        "critic_updates": update_calls * len(UNITS),
        "actor_update_calls": actor_calls,
        "actor_updates": actor_calls * len(UNITS),
        "buffer_len": int(manifest["buffer_len"]),
        "gate_stats_rows": len(gate_rows),
        "gate_stats_unique_episodes": len({g["episode"] for g in gate_rows}),
        "round_distinct_dates": {
            str(r): len({e["pair_id"] for e in episodes
                         if int(e["round"]) == r})
            for r in range(1, 11)},
        "all_rounds_same_date_set": len({
            frozenset(e["pair_id"] for e in episodes
                      if int(e["round"]) == r) for r in range(1, 11)}) == 1,
        "min_exposures_per_pair": min(
            sum(1 for e in episodes if e["pair_id"] == pid)
            for pid in {e["pair_id"] for e in episodes}),
        "max_exposures_per_pair": max(
            sum(1 for e in episodes if e["pair_id"] == pid)
            for pid in {e["pair_id"] for e in episodes}),
        "snapshots": sorted(p.name for p in SNAP.glob("round_*.json")),
        "checkpoints": sorted(p.name for p in (RUN / "checkpoints").iterdir()),
        "resumes": [
            {"checkpoint_step": 5088, "episodes_in_checkpoint": 212,
             "resume_episode": 212, "partial_round_discarded": False,
             "note": "first external stop at episode 252"},
            {"checkpoint_step": 6360, "episodes_in_checkpoint": 265,
             "resume_episode": 265, "partial_round_discarded": False,
             "note": "second external stop at episode 301"},
        ],
        "interruption_note": (
            "Two external stops occurred. The checkpointed round boundary was "
            "resumed; uncheckpointed partial-round episodes were re-run and "
            "the episode CSV was truncated to the checkpoint, so the final "
            "episode ids are 1..530 unique. train.log retains the interrupted "
            "attempts; the CSV is the authoritative episode log."),
        "earlier_failed_attempt": {
            "run_id": "EXPLORATORY_NEAR_B_C2_GATE_seed11",
            "stopped_at_episode": 45,
            "reason": ("C2 rejected a 6e-7 SOC numerical overshoot with its "
                       "1e-12 check"),
            "fix": "1e-6 tolerance plus clipping and a regression test",
            "diagnostic_checkpoint": "checkpoints/step_00001056",
            "retained": True,
        },
        "best_round": manifest["best"]["round"],
        "best_checkpoint": manifest["best"]["checkpoint"],
        "final_checkpoint": manifest["final_checkpoint"],
        "h2_holdout_read": manifest["h2_holdout_read"],
        "legacy_test_run": manifest.get("test_evaluation") is not None,
    }
    (DATA / "training_episode_audit.json").write_text(
        json.dumps(episode_audit, indent=2), encoding="utf-8")
    shutil.copy2(RUN / "checkpoint_index.json",
                 DATA / "training_checkpoint_index.json")
    shutil.copy2(RUN / "validation_audit_by_round.json",
                 DATA / "training_validation_audit.json")
    shutil.copy2(RUN / "run_manifest.json",
                 DATA / "training_run_manifest.json")

    best = manifest["best"]
    best_step = int(Path(best["checkpoint"]).name.split("_")[-1])
    final_step = int(Path(manifest["final_checkpoint"]).name.split("_")[-1])

    # ------------------------------------------------------------------
    # Checkpoint recovery report
    # ------------------------------------------------------------------
    lines = [
        "# Checkpoint recovery audit",
        "",
        f"Run: `{RUN_ID}`. All ten round checkpoints were reconstructed into "
        "loadable roots, loaded, and used to re-evaluate the validation split. "
        "The re-evaluated mean fleet ΔG matches the training-time snapshot "
        "for every round.",
        "",
        "| round | episode | step | checkpoint tree sha256 (first 16) | "
        "reproduced |",
        "|---:|---:|---:|---|---|",
    ]
    for n in range(1, 11):
        idx = index[str(n)]
        lines.append(f"| {n} | {idx['episode']} | {idx['step']} | "
                     f"`{idx['checkpoint_tree_sha256'][:16]}` | "
                     f"{idx['reproduced']} |")
    lines += [
        "",
        f"- Best validation checkpoint: `{best['checkpoint']}` "
        f"(round {best['round']}, step {best_step}).",
        f"- Final checkpoint: `{manifest['final_checkpoint']}` "
        f"(step {final_step}).",
        f"- Best root: `runs/{RUN_ID}_best/` (manifest step {best_step}).",
        "- Full hash index: `data/processed/v2_cy_jn_2025/"
        "training_checkpoint_index.json`.",
        "- The reconstructed roots use a manifest whose `step` and `counters` "
        "are set from the step's own `counters.json`; the network, optimizer, "
        "RNG and buffer files are hard-linked from the original step.",
        "- No test or H2 evaluation was used in this recovery audit.",
        "",
    ]
    (REPORTS / "checkpoint_recovery_audit.md").write_text(
        "\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # Main acceptance report
    # ------------------------------------------------------------------
    fleet = [(n, audit[str(n)]["validation"]) for n in range(1, 11)]
    best_v = audit[str(best["round"])]["validation"]
    final_v = audit["10"]["validation"]
    lines = [
        "# Development training acceptance",
        "",
        f"Run: `{RUN_ID}`. **Exploratory/development only**: H1 joint holdout "
        "is 0; the legacy test was not run; the H2 price-date holdout was not "
        "read.",
        "",
        "## 1. Training completeness",
        "",
        "| item | value |",
        "|---|---:|",
        f"| rounds completed | 10 / 10 |",
        f"| episodes | {episode_audit['episodes_run']} |",
        f"| unique episode ids | {episode_audit['unique_episode_ids']} "
        f"({episode_audit['episode_id_min']}..{episode_audit['episode_id_max']}) |",
        f"| block interactions | {total_steps} |",
        f"| update calls | {update_calls} |",
        f"| critic updates | {episode_audit['critic_updates']} |",
        f"| actor update calls | {actor_calls} |",
        f"| actor updates | {episode_audit['actor_updates']} |",
        f"| replay buffer length | {episode_audit['buffer_len']} |",
        f"| snapshots | {len(episode_audit['snapshots'])} |",
        f"| checkpoints | {len(episode_audit['checkpoints'])} |",
        "",
        "Two external stops occurred. The run resumed from the last complete "
        "round checkpoint (steps 5088 and 6360); uncheckpointed partial-round "
        "episodes were re-run, and the episode CSV was truncated to the "
        "checkpoint. The final episode ids are 1..530, unique and contiguous. "
        "`train.log` retains the interrupted attempts; the CSV is "
        "authoritative. All ten round checkpoints were reloaded and "
        "re-evaluated; every ΔG reproduced.",
        "",
        "An earlier attempt, `EXPLORATORY_NEAR_B_C2_GATE_seed11`, stopped at "
        "episode 45 because the C2 check rejected a 6e-7 SOC numerical "
        "overshoot with a 1e-12 tolerance. The tolerance was changed to 1e-6 "
        "with clipping and a regression test, and the run restarted as "
        "`_v2`. The failed run is retained at `runs/"
        "EXPLORATORY_NEAR_B_C2_GATE_seed11/` with its diagnostic checkpoint "
        "`step_00001056`; it is not part of the 530 completed episodes.",
        "",
        "Each round contains 53 distinct dates; the date set is identical in "
        "all ten rounds, and every training date has exactly 10 exposures. "
        "No date repeats within a round. The 530-row episode CSV, the 530 "
        "gate-stat rows and the 10 round snapshots are consistent with the "
        "10 checkpoints (each step equals 24 × its episode count).",
        "",
        "## 2. Validation performance",
        "",
        "| round | episode | mean ΔG | median ΔG | positive | ESS9 | ESS14 | "
        "ESS17 | ESS27 | modes | v/l |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, v in fleet:
        d = v["mean_delta_by_unit"]
        lines.append(
            f"| {n} | {audit[str(n)]['episode']} | "
            f"{v['mean_fleet_delta']:.1f} | {v['median_fleet_delta']:.1f} | "
            f"{v['positive_days']}/11 | {d['ESS9']:.1f} | {d['ESS14']:.1f} | "
            f"{d['ESS17']:.1f} | {d['ESS27']:.1f} | "
            f"{v['gate']['distinct_modes']} | "
            f"{v['voltage_violation_cells']}/"
            f"{v['line_overload_cells']} |")
    lines += [
        "",
        f"Best validation round: **{best['round']}** (mean fleet ΔG "
        f"{best_v['mean_fleet_delta']:.1f}, {best_v['positive_days']}/11 "
        f"positive); checkpoint `{best['checkpoint']}`. The legacy test was "
        "not used for selection.",
        "",
        "### Per-ESS energy and cycles (validation means)",
        "",
        "| round | " + " | ".join(
            f"{u} ch/dis/cyc" for u in UNITS) + " |",
        "|---:|" + "---:|" * len(UNITS),
    ]
    for n, v in fleet:
        cells = []
        for u in UNITS:
            cells.append(f"{v['mean_charge_mwh'][u]:.3f}/"
                         f"{v['mean_discharge_mwh'][u]:.3f}/"
                         f"{v['mean_cycles'][u]:.3f}")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Terminal SOC is 0.5 for every ESS in every round "
        f"(max error {max(v['terminal_soc_max_err'] for _, v in fleet):.1e}); "
        "voltage and line violation cells are 0 in every round. The complete "
        "10-round table is `data/processed/v2_cy_jn_2025/"
        "training_validation_rounds.csv`.",
        "",
        "## 3. Gate learning",
        "",
        "Open frequency / mean probability per gate, deterministic validation:",
        "",
        "| round | " + " | ".join(GATES) + " | distinct modes | sat U/L | "
        "permanent open / closed |",
        "|---:|" + "---:|" * len(GATES) + "---:|---:|---|",
    ]
    for n, v in fleet:
        g = v["gate"]
        cells = [f"{g['open_frequency'][k]:.2f}/{g['mean_probability'][k]:.2f}"
                 for k in GATES]
        lines.append(
            f"| {n} | " + " | ".join(cells) + " | "
            f"{g['distinct_modes']} | {g['saturation_upper']}/"
            f"{g['saturation_lower']} | "
            f"{len(g['permanent_open'])}/{len(g['permanent_closed'])} |")
    lines += [
        "",
        "### Continuous quote distribution, best and final rounds",
        "",
        "| round | gate | n | mean | std | min | p50 | max |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n in (best["round"], 10):
        g = audit[str(n)]["validation"]["gate"]
        for key in GATES:
            q = g["quote_distribution"][key]
            if q is None:
                lines.append(f"| {n} | {key} | 0 | closed | - | - | - | - |")
            else:
                lines.append(
                    f"| {n} | {key} | {q['count']} | {q['mean']:.2f} | "
                    f"{q['std']:.2f} | {q['min']:.2f} | {q['p50']:.2f} | "
                    f"{q['max']:.2f} |")
    lines += [
        "",
        "**Assessment.** The deterministic policy does **not** form a "
        "meaningful differentiated gate strategy. The joint fleet mode is "
        "almost static: 1 distinct mode in rounds 1–6, only 2–3 of 256 in the "
        "last rounds. Several gates are permanently open or permanently "
        "closed over all 264 validation decisions, and the open price branch "
        "reaches the lower boundary (264 saturated legs at the final round). "
        "The gate has learned a fixed mask plus near-boundary quotes, not a "
        "state-dependent declaration of truthful versus premium quotes. The "
        "training-time Bernoulli exploration does not rescue the "
        "deterministic policy. Full per-round gate data: `gate_behavior_by_"
        "round.json`.",
        "",
        "## 4. C2 and SOC audit",
        "",
        "| check | result |",
        "|---|---|",
        f"| C2 identity day | `{c2['day']}` |",
        f"| max raw vs clipped SOC deviation | "
        f"{c2['checks']['max_raw_vs_clipped_deviation']:.3e} |",
        f"| max raw SOC band overshoot (audited day) | "
        f"{c2['checks']['max_raw_soc_band_violation']:.3e} |",
        f"| max raw SOC band overshoot (all validation rounds) | "
        f"{max(v['raw_soc_band_violation'] for _, v in fleet):.3e} |",
        f"| terminal SOC max error | "
        f"{c2['checks']['terminal_soc_max_err']:.3e} |",
        f"| voltage / line violation cells | "
        f"{c2['checks']['voltage_violation_cells']} / "
        f"{c2['checks']['line_overload_cells']} |",
        f"| B vs C2 discounted identity, max error | "
        f"{c2['checks']['b_vs_c2_max_error']:.3e} |",
        f"| shaping sum vs boundary, max error | "
        f"{c2['checks']['shaping_vs_boundary_max_error']:.3e} |",
        f"| B paid only at block 23 | {c2['checks']['b_paid_only_at_23']} |",
        f"| B at block 23 equals delta | "
        f"{c2['checks']['b_at_23_equals_delta']} |",
        f"| C2 shaping present | {c2['checks']['all_shaping_nonzero']} |",
        "",
        "The C2 tolerance is 1e-6, matching solver feasibility. The largest "
        "raw SOC overshoot observed across all validation rounds is "
        f"{max(v['raw_soc_band_violation'] for _, v in fleet):.2e}, which is "
        "numerical and was clipped; a larger excursion would raise. Raw and "
        "clipped SOC paths are recorded in `c2_soc_training_audit.json`. The "
        "identity audit day had no raw overshoot. Terminal SOC is exactly 0.5 "
        "and no voltage or line violation was observed.",
        "",
        "## 5. Checkpoints and recovery",
        "",
        f"- Best validation: `{best['checkpoint']}` (round {best['round']}, "
        f"step {best_step}).",
        f"- Final: `{manifest['final_checkpoint']}` (step {final_step}).",
        f"- Best root: `runs/{RUN_ID}_best/`.",
        f"- All 10 round snapshots: `snapshots/{RUN_ID}/round_001..010.json`.",
        "- Full checkpoint hashes: `data/processed/v2_cy_jn_2025/"
        "training_checkpoint_index.json`.",
        "- All 10 checkpoints were reconstructed and re-evaluated; every "
        "mean fleet ΔG reproduced the snapshot. See "
        "`checkpoint_recovery_audit.md`.",
        "",
        "## 6. Holdout boundary",
        "",
        "- H2 was **not read**: `h2_holdout_read=false` in the run manifest.",
        "- The legacy test was **not run**: `test_evaluation=null`.",
        "- The 40 frozen H2 price dates remain untouched for a later "
        "one-shot evaluation after the model and five-seed protocol are "
        "frozen.",
        "",
        "## 7. Final conclusion",
        "",
        "- **Budget:** completed 10/10 rounds, 530 episodes, 12,720 block "
        f"steps, {update_calls} update calls, {episode_audit['critic_updates']} "
        f"critic and {episode_audit['actor_updates']} actor updates.",
        "- **Stable improvement:** no. Validation mean fleet ΔG goes from "
        "−35.5 (R1) to a best of −30.7 (R2), then worsens to −77.7 (R10). "
        "Only 4–6 of 11 validation days are positive.",
        "- **Better than B1:** no. Mean fleet ΔG is negative in every round; "
        "B1 is the zero reference by construction.",
        "- **Gate degeneration:** yes. The deterministic gate collapses to "
        "1–3 joint modes, with permanent open/closed gates and lower-bound "
        "quote saturation; it is not a meaningful differentiated quote "
        "strategy.",
        "- **Other seeds:** not recommended until the gate collapse and the "
        "worsening validation curve are diagnosed. A five-seed matrix would "
        "likely reproduce the degeneration and is not a fix.",
        "- **H2 unseen:** yes. The 40 price dates were not read.",
        "- **Old code/data:** old MATD3, the clearing engines and the frozen "
        "data were not modified in this training run. The only numerical "
        "change is the C2 1e-6 SOC tolerance described above.",
        "",
        "Do not report this single-seed development run as a formal "
        "generalisation result.",
        "",
    ]
    (REPORTS / "training_run_acceptance.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "episodes": episode_audit["episodes_run"],
        "unique": episode_audit["unique_episode_ids"],
        "steps": total_steps,
        "updates": update_calls,
        "critic": episode_audit["critic_updates"],
        "actor": episode_audit["actor_updates"],
        "best_round": best["round"],
        "best_mean": best_v["mean_fleet_delta"],
        "final_mean": final_v["mean_fleet_delta"],
        "all_reproduced": all(index[str(n)]["reproduced"] for n in range(1, 11)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
