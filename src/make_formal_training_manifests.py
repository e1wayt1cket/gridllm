"""Build and check the formal-training data, config and code manifests.

The formal data definition is the v3 candidate-day library: 136 paired days of
historical Jiangsu bus load and 2025 day-ahead price, split into 90 train, 18
validation, 18 test and 10 holiday blocks. Its price dates come from the same
2025 series the H2 holdout definition counts, so the data manifest declares how
many of those dates entered training and what the result still supports.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

import training_contract_v7 as contract_v7

ROOT = Path(".")
LIBRARY = ROOT / "reports/candidate_days_v3_2026-11-17/evidence"
PRICE_DIR = ROOT / "data/processed/v2_cy_jn_2025"
OUT = ROOT / "reports/v2_cy_jn_2025/formal_training_v7"
CANDIDATE = "V7_candidate_002"
INPUTS = (
    ("manifest.json", LIBRARY),
    ("split.json", LIBRARY),
    ("bus_active_load_mw.npz", LIBRARY),
    ("bus_reactive_load_mvar.npz", LIBRARY),
    ("price_curves.npz", PRICE_DIR),
    # Reads the H2 definition the library's price dates are checked against.
    ("holdout_manifest.json", PRICE_DIR),
)
SOURCE_FILES = (
    # The environment, the quote map, the actor/critic stack and the trainer.
    "rl_da_env.py", "rl_da_gate.py", "rl_da_gate_v7.py",
    "da_quote_map.py", "da_quote_map_v5.py", "rl_spec.py", "rl_spec_v7.py",
    "rl_env.py", "rl_td3.py", "rl_bidding.py", "rl_b_c2_gate.py",
    "truthful_gate_actor_adapter.py", "da_replay.py",
    "c2_reward.py", "daily_reward_b.py", "price_forecaster.py",
    "participant_payoff.py", "money.py",
    # The market: library, contracts, clearing, operating rules and pricing.
    "local_da_days.py", "local_da_contracts.py", "local_da_clearing.py",
    "local_da_operating.py", "local_da_accounting.py", "local_da_audit.py",
    "run_local_da.py",
    "checkpoint_writer_v2.py", "checkpoint_loader_v2.py",
    "validate_checkpoint_schema.py",
    # The launch, the contract and this generator supply and check the budget,
    # the stop rule and the interfaces, so they are part of the protocol.
    "train_local_da_v7.py", "training_contract_v7.py", "training_contract.py",
    "run_formal_training.py", "make_formal_training_manifests.py",
)
TESTS = (
    "tests/test_local_da_env_v7.py",
    "tests/test_local_da_env_v7_probe.py",
    "tests/test_local_da_env_v7_reward.py",
    "tests/test_local_da_v7_days.py",
    "tests/test_local_da_audit_v7.py",
    "tests/test_rl_da_gate_v7.py",
    "tests/test_rl_spec_v7.py",
    "tests/test_local_da_market.py",
    "tests/test_smoke_local_da_v7.py",
    "tests/test_training_contract_v7.py",
    "tests/test_frozen_guard.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip()
        return {"revision": rev, "dirty": bool(dirty),
                "dirty_paths": [line[3:] for line in dirty.splitlines()]}
    except Exception:
        return {"revision": "working-tree", "dirty": True,
                "dirty_paths": []}


def code_payload() -> dict:
    """The pinned sources and tests; the launcher verifies against this."""
    return {
        "candidate": CANDIDATE,
        "source_sha256": {
            name: sha(ROOT / "src" / name) for name in SOURCE_FILES
        },
        "git": git_state(),
        "tests": {Path(name).name: sha(ROOT / name) for name in TESTS},
    }


def data_payload() -> dict:
    split_doc = json.loads((LIBRARY / "split.json").read_text(
        encoding="utf-8"))
    sets = {name: set(rows) for name, rows in split_doc["pairs"].items()}
    names = list(sets)
    overlaps = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlaps[f"{left}/{right}"] = sorted(sets[left] & sets[right])
    legacy = sets["test"] | sets["holiday"]
    library = json.loads((LIBRARY / "manifest.json").read_text(
        encoding="utf-8"))
    price_dates = {entry.split("|")[1] for rows in split_doc["pairs"].values()
                   for entry in rows}
    holdout = json.loads((PRICE_DIR / "holdout_manifest.json").read_text(
        encoding="utf-8"))
    h2_dates = {pair.split("|")[1] for pair in holdout["h2_pair_ids"]}
    return {
        "split_file": str(LIBRARY / "split.json"),
        "split_view": "v3_candidate_library",
        "split_seed": split_doc["seed"],
        "pair_key": "load_date|price_date",
        "library_size": split_doc["library_size"],
        "library_dataset_version": library.get("dataset_version"),
        "counts": {name: len(sets[name]) for name in names},
        "train_pairs": sorted(sets["train"]),
        "validation_pairs": sorted(sets["validation"]),
        "test_pairs": sorted(sets["test"]),
        "holiday_pairs": sorted(sets["holiday"]),
        "legacy_seen_test_pairs": sorted(legacy),
        "legacy_label": "LEGACY_SEEN_TEST",
        "overlaps": overlaps,
        "all_disjoint": all(not v for v in overlaps.values()),
        # The holdout is not sealed. These are the price dates the H2
        # definition enumerates that the library also used, and the surfaces
        # the declaration still rules out.
        "price_dates_used": len(price_dates),
        "h2_price_dates_used": len(h2_dates & price_dates),
        "h2_opened": True,
        "h2_use": contract_v7.H2_USE,
        "independent_generalisation_claim": False,
        "input_sha256": {
            name: sha(root / name) for name, root in INPUTS
        },
        "test_label": "LEGACY_SEEN_TEST",
    }


def config_payload(data: dict, code: dict) -> dict:
    return {
        "formal_training": {
            "experiment_id": "FORMAL_LOCAL_DA_V7",
            "protocol_version": 7,
            "accepted_candidate": CANDIDATE,
            "dataset": "v3 candidate-day library: Jiangsu 2021-2022 bus load "
                       "paired with 2025 day-ahead price dates",
            "split": data["split_file"],
            "split_view": data["split_view"],
            "split_seed": data["split_seed"],
            "trainer": "src/train_local_da_v7.py",
            "trainer_split": "local_da_v7",
            "reward_mode": "B_C2",
            "gamma": 0.99,
            "fixed_scale": 100.0,
            # The day ends on whatever SOC it reached; the endpoint is valued
            # by the settlement's inventory term, never pinned.
            "terminal_soc": {"mode": "free", "initial": 0.5,
                             "valuation": "inventory_value at the day end"},
            "cycle_cost_cny_per_mwh": 100.0,
            "clearing_engine": "exact MIP dispatch with a binary charge/"
                               "discharge disjunction, fixed-direction "
                               "conditional pricing",
            # A training window may read its payoff off a feasible incumbent
            # when the solver stopped at its time limit and its own reported
            # relative gap is inside this bound. Settlement and audit paths
            # pass 0.0 and keep demanding the certified optimum.
            "training_gap_tolerance": 0.01,
            # The limit is a search budget, not a criterion. The disjunction
            # gives the exact model a feasible incumbent in seconds; blocks that
            # cannot be certified inside the limit are still cleared and counted
            # in gap_solved_blocks.
            "training_window_seconds": 300.0,
            "voltage_band_pu": [0.93, 1.07],
            "reactive_support": "declared zero-active shunt asset at buses "
                                "17, 31, 32 carried by the library day and "
                                "by every training window",
            "settlement": "local day-ahead settlement ledger",
            "observation": {"name": "v7_12d", "version": 7,
                            "dimensions": 12},
            "action": {"name": "storage_quote_v5", "version": 5,
                       "quote_map": "linear_core_v1",
                       "dimensions_per_unit": 4,
                       "quote_order": ["charge_price", "charge_quantity",
                                       "discharge_price",
                                       "discharge_quantity"],
                       "bounds": {"charge_price": [0.0, 150.0],
                                  "charge_quantity": [0.0, 0.4],
                                  "discharge_price": [0.0, 300.0],
                                  "discharge_quantity": [0.0, 0.4]},
                       "truthful": {"charge_price": 0.0,
                                    "charge_quantity": 0.4,
                                    "discharge_price": 0.0,
                                    "discharge_quantity": 0.4},
                       "ess_order": ["ESS9", "ESS14", "ESS17", "ESS27"]},
            "gate": {"per_unit": 2, "total": 8, "estimator":
                     "straight_through_bernoulli",
                     "temperature_start": 4.0, "temperature_end": 1.0,
                     "warmup_episodes": 106,
                     "evaluation_temperature": 1.0,
                     "logit_bound": 4.0},
            "networks": {"actor": "256-128-4",
                         "critic": "centralized twin 256-256 no dropout"},
            "optimizer": {"name": "Adam", "lr": 3e-4, "tau": 0.005,
                          "policy_delay": 2, "batch_size": 128,
                          "buffer_capacity": 6000, "start_steps": 480,
                          "actor_weight_decay": 1e-5,
                          "calibration_interval": 8},
            "anchor": {"weight": 0.05,
                       "form": "sum over a unit's quotes of squared "
                               "box-normalised deviations from truthful, "
                               "averaged over the batch",
                       "calibrated_by":
                           "reports/local_da_v7_2026-11-18/audit/"
                           "summary.json (acceptance_6)",
                       "selection_rule": "largest weight whose gradient share "
                                         "stays at or below one half",
                       "note": "not a profit-selected hyperparameter"},
            "hinge": {"advantage_hinge_delta": 0.10,
                      "twin_gap_weight": 1.0},
            "sampler": {"uniform_fraction": 0.5, "beta": 0.5,
                        "max_upweight": 5.0},
            "exploration": {"epsilon_start": 0.20, "epsilon_end": 0.05,
                            "decay_episodes": 159,
                            "charge_scale": 0.5,
                            "high_fraction": 0.5},
            "target": {"policy_noise": 0.2, "noise_clip": 0.5},
            "episode": {"blocks": 24, "commit_periods": 4,
                        "lookahead_periods": 16, "roll_horizon": 20,
                        "probe_block": 11},
            "budget": {"seeds": [11, 23],
                       "minimum_rounds": 10,
                       "exposures_per_day_min": 10,
                       "exposures_per_day_max": 15,
                       "episodes_per_seed_min": 900,
                       "episodes_per_seed_max": 1350,
                       "block_steps_per_seed_min": 21600,
                       "block_steps_per_seed_max": 32400,
                       "critic_updates_per_block": 1,
                       "actor_updates_every_blocks": 2},
            "wall_clock": {
                "hours_per_seed_min": 12.0,
                "hours_per_seed_max": 18.0,
                "status": "recorded for the reader; not a gate",
                "source": "reports/local_da_v7_2026-11-18/RESULTS.md "
                          "latency table at roll horizon 20",
                "hard_block_seconds": 300,
                "hard_block_note": "a block that reaches the solver limit "
                                   "spends it; the projection is the typical "
                                   "block rate and does not bound a day whose "
                                   "blocks all bind",
            },
            "validation": {
                "passes": "after each full pass over the 90 training days",
                "metric": "mean fleet delta G over the 18 validation days "
                          "(settlement, not reward)",
                "best_checkpoint": "highest validation mean fleet delta G",
                "resume_checkpoint": "every round",
                "final_checkpoint": "end of budget or early stop",
                "early_stop_patience": 3,
            },
            "compatibility": {
                "supersedes": "reports/v2_cy_jn_2025/formal_training_v6",
                "reward_scale": "fixed_scale 100, unchanged from v4 to v7",
                "checkpoints": "v7 runs refuse to load a checkpoint from any "
                               "earlier protocol: the observation name, the "
                               "quote box and the clearing market all differ, "
                               "so an earlier checkpoint does not describe a "
                               "v7 policy",
            },
            "boundaries": {
                "train_only_in_replay": True,
                "validation_not_in_replay": True,
                "test_evaluated": False,
                "holiday_evaluated": False,
                "h2_opened": True,
                "h2_use": contract_v7.H2_USE,
                "test_label": "LEGACY_SEEN_TEST",
                "independent_generalisation_claim": False,
            },
            "claims": {
                "allowed": "limited price-date adaptation on 40 unpaired "
                           "2025 price dates with historical load scenarios, "
                           "under this frozen protocol",
                "not_allowed": "a joint source-load generalisation claim or "
                               "any independent generalisation claim",
                "gap_accepted_payoffs": "a training payoff may be read off an "
                                        "incumbent inside "
                                        "training_gap_tolerance and is then "
                                        "known only to that gap; no settlement "
                                        "payment and no audit price was read "
                                        "off an uncertified incumbent",
            },
            "code": {
                "candidate": CANDIDATE,
                "source_sha256": code["source_sha256"],
                "git": code["git"],
            },
            "data": {
                "counts": data["counts"],
                "legacy_seen_test_pairs": data["legacy_seen_test_pairs"],
                "h2_price_dates_used": data["h2_price_dates_used"],
                "input_sha256": data["input_sha256"],
            },
        }
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify existing manifests instead of writing")
    parser.add_argument("--finalize", action="store_true",
                        help="write the GO/NO-GO status from "
                             "pretraining_validation_results.json")
    parser.add_argument("--out", default=str(OUT),
                        help="manifest directory (defaults to the original)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    data = data_payload()
    code = code_payload()
    config = config_payload(data, code)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    manifest = {
        "status": "PENDING_PRETRAINING_VALIDATION",
        "training_readiness_status": None,
        "scope": ("formal training on the v3 local day-ahead library; Test "
                  "and holiday are LEGACY_SEEN_TEST; the 40 H2 price dates "
                  "were used for price-date adaptation only"),
        "accepted_candidate": CANDIDATE,
        "data_manifest": "formal_training_data_manifest.json",
        "code_manifest": "formal_training_code_manifest.json",
        "config_file": "formal_training_config.yaml",
        "config_sha256": config_hash,
        "data_manifest_sha256": hashlib.sha256(
            json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest(),
        "code_manifest_sha256": hashlib.sha256(
            json.dumps(code, sort_keys=True).encode("utf-8")).hexdigest(),
        "checks": {
            "library_counts": data["counts"] == contract_v7.EXPECTED_COUNTS,
            "all_disjoint": data["all_disjoint"],
            "legacy_label_declared":
                data["legacy_label"] == "LEGACY_SEEN_TEST",
            "h2_declared": (data["h2_price_dates_used"]
                            == contract_v7.H2_PRICE_DATES_USED
                            and data["independent_generalisation_claim"]
                            is False),
        },
    }
    manifest["checks"]["all_pass"] = all(manifest["checks"].values())
    files = {
        "formal_training_data_manifest.json": data,
        "formal_training_code_manifest.json": code,
        "formal_training_config.yaml": config,
        "formal_training_manifest.json": manifest,
    }
    if args.check:
        problems = []
        paths = {name: out / name for name in files}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            print(json.dumps({"ok": False, "problems":
                              [f"missing {p}" for p in missing]}, indent=2))
            return 1
        saved_data = json.loads(
            paths["formal_training_data_manifest.json"].read_text(
                encoding="utf-8"))
        saved_code = json.loads(
            paths["formal_training_code_manifest.json"].read_text(
                encoding="utf-8"))
        saved_config = yaml.safe_load(
            paths["formal_training_config.yaml"].read_text(encoding="utf-8"))
        saved_manifest = json.loads(
            paths["formal_training_manifest.json"].read_text(
                encoding="utf-8"))
        if saved_data != data:
            problems.append("formal_training_data_manifest.json: "
                            "content mismatch")
        # Git state is provenance, not integrity: an untracked file must not
        # turn a frozen manifest red. Source and test hashes are the contract.
        for key in ("candidate", "source_sha256", "tests"):
            if saved_code.get(key) != code.get(key):
                problems.append(
                    f"formal_training_code_manifest.json: {key} mismatch")
        for key, payload in (
                ("config_sha256", saved_config),
                ("data_manifest_sha256", saved_data),
                ("code_manifest_sha256", saved_code)):
            expected = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if saved_manifest.get(key) != expected:
                problems.append(f"formal_training_manifest.json: "
                                f"{key} mismatch")
        print(json.dumps({
            "ok": not problems, "problems": problems,
            "git_drift": saved_code.get("git") != code.get("git"),
            "checks": manifest["checks"]}, indent=2))
        return 0 if not problems else 1
    if args.finalize:
        results_path = out / "pretraining_validation_results.json"
        if not results_path.exists():
            raise FileNotFoundError(results_path)
        results = json.loads(results_path.read_text(encoding="utf-8"))
        manifest_path = out / "formal_training_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "finalized":
            print(json.dumps({
                "refused": "manifest is already finalized",
                "manifest": str(manifest_path),
            }, indent=2))
            return 4
        status = ("GO_FORMAL_TRAINING" if results.get("all_pass")
                  else "NO_GO")
        existing.update({
            "status": "finalized",
            "training_readiness_status": status,
            "pretraining_validation_results": str(results_path),
            "training_readiness": str(
                out / "training_readiness.md"),
            "finalized_on": dt.date.today().isoformat(),
        })
        manifest_path.write_text(
            json.dumps(existing, indent=2), encoding="utf-8")
        print(json.dumps({
            "training_readiness_status": status,
            "all_pass": results.get("all_pass"),
            "manifest": str(manifest_path),
        }, indent=2))
        return 0 if results.get("all_pass") else 1
    conflicts = [str(out / name) for name in files if (out / name).exists()]
    if conflicts:
        print(json.dumps({
            "refused": "target already contains formal-training manifests; "
                       "write a new version directory instead",
            "paths": conflicts,
        }, indent=2))
        return 4
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        text = (yaml.safe_dump(payload, sort_keys=False)
                if name.endswith(".yaml") else json.dumps(payload, indent=2))
        (out / name).write_text(text, encoding="utf-8")
    (out / "manifest_generation.json").write_text(json.dumps({
        "generated_on": dt.datetime.now().isoformat(timespec="seconds"),
        "candidate": CANDIDATE,
        "config_sha256": config_hash,
        "source_sha256": code["source_sha256"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(manifest["checks"], indent=2))
    print(f"wrote {out}")
    return 0 if manifest["checks"]["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
