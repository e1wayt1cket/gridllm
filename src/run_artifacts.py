# run_artifacts.py
"""Structured persistence for RL training runs.

Port of ASSUME's WriteOutput idea (structured per-run artifacts with a
manifest) scoped to training runs: every training invocation writes a
self-describing directory under outputs/rl/<run_id>/ containing the CLI args,
a JSON-safe config snapshot, the observation/action specs, per-episode
metrics, evaluation history, and the paths of every policy artifact produced.

outputs/ is gitignored, so these artifacts never enter version control.
"""

import os
import json
import hashlib
import dataclasses
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
# pandas is imported lazily inside save()/load() to keep it out of the
# module-load path: this project loads ortools before pandas on Windows,
# where the reverse order can fail DLL resolution (see make_results_figure.py).


def _json_safe(obj):
    """Recursively convert a value to a JSON-serializable Python object.

    Handles dataclasses (via asdict), dicts, lists/tuples, numpy arrays and
    scalars, and falls back to str() for anything unrecognized.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _json_safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def _config_hash(config) -> str:
    """Short md5 of the JSON-safe config snapshot, used in the run id."""
    payload = json.dumps(_json_safe(config), sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()


# Base metrics columns, kept first and in this order so existing readers and
# human diffing of metrics.csv across runs keep working.
_BASE_METRIC_COLUMNS = ["episode", "mean_reward", "welfare", "re_rate",
                        "critic_loss", "actor_loss", "scenario"]
# Prefix for the per-agent reward columns ({prefix}{agent_name}).
_AGENT_REWARD_PREFIX = "reward__"
# Fleet profit columns, reported next to the reward they are the basis of.
# `mean_reward` is the learning signal; these are the settled money it is
# measured against and the in-window truthful counterfactual it subtracts.
# Neither is the paper's benefit, which uses a whole-day all-truthful baseline.
_PROFIT_COLUMNS = ["raw_profit_mean", "local_baseline_profit_mean",
                   "profit_delta_local_mean"]
# Critique diagnostics, ordered before the day-level economics.
_Q_COLUMNS = ["q1_mean", "q2_mean", "q_gap"]
# Trailing integrity marker for the day-level economics.
_CAPTURE_COLUMNS = ["capture_fellbacks"]


def _ordered_columns(rows: List[dict]) -> List[str]:
    """Stable metrics.csv column order over a possibly ragged row set.

    The base columns come first, then per-agent rewards sorted by agent name,
    then critic diagnostics, then the day-level economic columns (whose names
    are owned by surplus_metrics), then the capture marker. Ragged rows - a
    run that recorded economics on some episodes only - keep one header with
    missing cells left blank.

    Anything unrecognized lands at the end alphabetically, so a caller can add
    a field without this function silently dropping it.
    """
    seen = set()
    for r in rows:
        seen.update(r.keys())
    cols = [c for c in _BASE_METRIC_COLUMNS if c in seen]
    cols += sorted(c for c in seen if c.startswith(_AGENT_REWARD_PREFIX))
    cols += [c for c in _Q_COLUMNS if c in seen]
    known = set(cols) | set(_BASE_METRIC_COLUMNS)
    # Day-level economics and the capture marker: eval names first, then any
    # other extra, alphabetically, so unknown additions are never dropped.
    cols += sorted(c for c in seen
                   if c not in known and c not in _CAPTURE_COLUMNS)
    cols += [c for c in _CAPTURE_COLUMNS if c in seen]
    return cols


def _git_sha() -> Optional[str]:
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


@dataclasses.dataclass
class RunArtifacts:
    """One training run's structured artifact set."""

    run_id: str
    root_dir: str
    cli_args: dict
    config: dict
    obs_spec: dict
    action_spec: dict
    metrics: List[dict]
    evals: List[dict]
    policy_paths: Dict[str, List[str]]
    kpi: dict
    git_sha: Optional[str] = None
    timestamp: str = ""
    # TensorBoard log directory for this run, set by the trainer once the
    # writer exists. Recorded so a consumer can find the per-agent scalar
    # history by reading the manifest instead of guessing from timestamps.
    log_dir: Optional[str] = None

    @classmethod
    def create(cls, config, cli_args, obs_spec, action_spec,
               base_dir: str = "outputs/rl") -> "RunArtifacts":
        """Create a fresh artifacts container for a training run."""
        ts = datetime.now()
        run_id = f"run-{ts:%Y%m%d-%H%M%S}-{_config_hash(config)[:8]}"
        root_dir = os.path.join(base_dir, run_id)
        return cls(
            run_id=run_id, root_dir=root_dir,
            cli_args=_json_safe(cli_args),
            config=_json_safe(config),
            obs_spec=obs_spec, action_spec=action_spec,
            metrics=[], evals=[], policy_paths={}, kpi={},
            git_sha=_git_sha(), timestamp=ts.isoformat())

    def record_episode(self, ep: int, mean_reward: float, welfare: float,
                       re_rate: float, critic_loss=None, actor_loss=None,
                       scenario=None, per_agent_reward: dict = None,
                       q_stats: dict = None, econ: dict = None,
                       capture_fellbacks: int = None,
                       raw_profit: float = None,
                       local_baseline_profit: float = None,
                       profit_delta_local: float = None):
        """Append one training-episode metrics row.

        Parameters
        ----------
        per_agent_reward : dict or None
            {agent_name: episode reward}; written as reward__<agent> columns
            so per-agent learning curves need no TensorBoard round trip.
        raw_profit, local_baseline_profit, profit_delta_local : float or None
            The fleet's settled profit this episode, the truthful-bidding
            counterfactual taken inside each training window, and the
            difference between them. Recorded so a run reports the money
            alongside the signal it is learning from: a falling actor loss with
            a flat raw profit is a failed run, and that is only visible if the
            profit is a column.
        q_stats : dict or None
            Critic diagnostics (q1_mean / q2_mean / q_gap). q_gap is twin
            disagreement, in the critic's normalized reward units.
        econ : dict or None
            Day-level consumer/mechanism metrics; the names are owned by
            surplus_metrics.DAY_ECON_COLUMNS and shared verbatim with the
            evaluation CSV.
        capture_fellbacks : int or None
            Blocks whose clearing degraded, making the econ columns unusable
            for this episode. 0 means the day-level metrics are complete.
        """
        row = {
            "episode": ep, "mean_reward": mean_reward, "welfare": welfare,
            "re_rate": re_rate, "critic_loss": critic_loss,
            "actor_loss": actor_loss, "scenario": scenario,
        }
        if per_agent_reward:
            row.update({f"{_AGENT_REWARD_PREFIX}{nm}": v
                        for nm, v in per_agent_reward.items()})
        if raw_profit is not None:
            row["raw_profit_mean"] = raw_profit
        if local_baseline_profit is not None:
            row["local_baseline_profit_mean"] = local_baseline_profit
        if profit_delta_local is not None:
            row["profit_delta_local_mean"] = profit_delta_local
        if q_stats:
            row.update(q_stats)
        if econ:
            row.update(econ)
        if capture_fellbacks is not None:
            row["capture_fellbacks"] = capture_fellbacks
        self.metrics.append(row)

    def record_eval(self, ep: int, metrics: dict, is_best: bool = False,
                    stopped: bool = False):
        """Append one in-training evaluation row."""
        self.evals.append({
            "episode": ep,
            "mean_reward": metrics.get("mean_reward"),
            "welfare": metrics.get("welfare"),
            "re_rate": metrics.get("re_rate"),
            "is_best": is_best,
            "early_stopped": stopped,
        })

    def add_policy_paths(self, role: str, paths: list):
        """Record the policy files produced under a role (final/best/last/ckpt)."""
        self.policy_paths[role] = sorted(str(p) for p in paths)

    def save(self) -> str:
        """Write manifest.json, config.json, metrics.csv, eval.csv, kpi.json."""
        import pandas as pd
        os.makedirs(self.root_dir, exist_ok=True)
        manifest = {
            "run_id": self.run_id, "timestamp": self.timestamp,
            "git_sha": self.git_sha, "cli_args": self.cli_args,
            "obs_spec": self.obs_spec, "action_spec": self.action_spec,
            "policy_paths": self.policy_paths, "kpi": self.kpi,
            "log_dir": self.log_dir,
        }
        with open(os.path.join(self.root_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        with open(os.path.join(self.root_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        pd.DataFrame(self.metrics,
                     columns=_ordered_columns(self.metrics) or None).to_csv(
            os.path.join(self.root_dir, "metrics.csv"), index=False)
        pd.DataFrame(self.evals).to_csv(
            os.path.join(self.root_dir, "eval.csv"), index=False)
        with open(os.path.join(self.root_dir, "kpi.json"), "w") as f:
            json.dump(self.kpi, f, indent=2, default=str)
        return self.root_dir

    @classmethod
    def load(cls, root_dir: str) -> "RunArtifacts":
        """Reconstruct a container from a saved artifact directory."""
        import pandas as pd
        with open(os.path.join(root_dir, "manifest.json")) as f:
            manifest = json.load(f)
        with open(os.path.join(root_dir, "config.json")) as f:
            config = json.load(f)
        metrics_csv = os.path.join(root_dir, "metrics.csv")
        evals_csv = os.path.join(root_dir, "eval.csv")
        metrics = pd.read_csv(metrics_csv).to_dict("records") \
            if os.path.exists(metrics_csv) else []
        evals = pd.read_csv(evals_csv).to_dict("records") \
            if os.path.exists(evals_csv) else []
        with open(os.path.join(root_dir, "kpi.json")) as f:
            kpi = json.load(f)
        return cls(
            run_id=manifest["run_id"], root_dir=root_dir,
            cli_args=manifest["cli_args"], config=config,
            obs_spec=manifest["obs_spec"],
            action_spec=manifest["action_spec"],
            metrics=metrics, evals=evals,
            policy_paths=manifest["policy_paths"], kpi=kpi,
            git_sha=manifest.get("git_sha"),
            timestamp=manifest.get("timestamp", ""),
            log_dir=manifest.get("log_dir"))
