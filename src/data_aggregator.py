# data_aggregator.py
"""Unified read layer over training runs, evaluations and policy directories.

The research UI needs one place that knows where results live and how they are
shaped, so pages never splice CSVs and patch missing fields themselves:

  outputs/rl/<run_id>/   per-run artifacts (manifest, config, metrics, kpi)
  runs/train-*/          TensorBoard scalar history, including per-agent
                         reward curves for runs predating the metrics columns
  results/*_eval.csv     evaluation output, optionally with consumer metrics
  policies/<dir>/        final / best / last / ckpt_N policy files per agent

Only the standard library is imported at module scope. pandas is imported
lazily inside the functions that need it: this project loads ortools before
pandas on Windows, and importing pandas earlier can fail DLL resolution (see
the note in run_artifacts.py). Dashboard pages import this module alongside the
market modules, so that ordering constraint applies here.
"""

import os
import datetime
from datetime import date
import re
import json
from typing import List, Optional

# Columns every metrics frame is guaranteed to expose, whatever generation of
# metrics.csv it came from. Readers can address them without a presence check;
# ones a given file lacks come back empty rather than missing.
CANONICAL_METRIC_COLUMNS = [
    "episode", "mean_reward", "welfare", "re_rate", "critic_loss",
    "actor_loss", "scenario",
    "q1_mean", "q2_mean", "q_gap",
    "cs_baseline", "cs_rl", "cs_delta",
    "cp_baseline", "cp_rl", "cp_delta",
    "lmp_markup_baseline", "lmp_markup_rl", "lmp_markup_delta",
    "market_power_arb", "market_power_power",
    "capture_fellbacks",
]

# Columns that carry the day-level consumer/mechanism accounting. Their
# presence marks a run as recording economics during training.
_ECON_METRIC_COLUMNS = [
    "cs_baseline", "cs_rl", "cs_delta", "cp_baseline", "cp_rl", "cp_delta",
    "lmp_markup_baseline", "lmp_markup_rl", "lmp_markup_delta",
    "market_power_arb", "market_power_power",
]

_AGENT_REWARD_PREFIX = "reward__"
_CKPT_RE = re.compile(r"^(?P<agent>.+)_ckpt_(?P<step>\d+)\.pt$")
# A bare YYYYmmdd-HHMMSS stamp, matched anywhere in a directory name (log
# directories also carry a 'train-...' prefix, which this deliberately ignores)
# and also used for the run's own timestamp once normalized to that form.
_TS_RE = re.compile(r"(\d{8}-\d{6})")


def _stamp_seconds(stamp: str) -> Optional[int]:
    """Seconds-of-epoch-day for a YYYYmmdd-HHMMSS stamp, for window compare."""
    m = _TS_RE.search(stamp or "")
    if not m:
        return None
    try:
        day, clock = m.group(1).split("-")
        return (int(day) * 86400 + int(clock[0:2]) * 3600
                + int(clock[2:4]) * 60 + int(clock[4:6]))
    except (ValueError, IndexError):
        return None

# Run directories whose manifest could not be read, surfaced instead of
# raising so one bad directory cannot take down a whole page.
_WARNINGS: List[str] = []

# TensorBoard event files are re-read only when they change on disk; a full
# scan of the run history is otherwise repeated on every page render.
_TB_CACHE = {}


def _pd():
    """Import pandas lazily; see the module docstring for why."""
    import pandas as pd
    return pd


def warnings() -> List[str]:
    """Non-fatal problems found during discovery since the last clear()."""
    return list(_WARNINGS)


def clear_warnings() -> None:
    _WARNINGS.clear()


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
def repo_root() -> str:
    """Repository root, derived from this file's location (src/ lives in it)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def artifacts_root() -> str:
    return os.path.join(repo_root(), "outputs", "rl")


def results_root() -> str:
    return os.path.join(repo_root(), "results")


def policies_root() -> str:
    return os.path.join(repo_root(), "policies")


def runs_root() -> str:
    return os.path.join(repo_root(), "runs")


def _run_dir(run_id: str, root: Optional[str]) -> str:
    return os.path.join(root or artifacts_root(), run_id)


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _run_ids(root: Optional[str] = None) -> List[str]:
    base = root or artifacts_root()
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base)
                  if d.startswith("run-")
                  and os.path.isdir(os.path.join(base, d)))


def load_manifest(run_id: str, root: Optional[str] = None) -> dict:
    return _read_json(os.path.join(_run_dir(run_id, root),
                                   "manifest.json")) or {}


def load_config(run_id: str, root: Optional[str] = None) -> dict:
    return _read_json(os.path.join(_run_dir(run_id, root),
                                   "config.json")) or {}


def load_kpi(run_id: str, root: Optional[str] = None) -> dict:
    return _read_json(os.path.join(_run_dir(run_id, root), "kpi.json")) or {}


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
def _metrics_columns(run_id: str, root: Optional[str]) -> List[str]:
    """Header of metrics.csv, read without parsing the rows."""
    pd = _pd()
    path = os.path.join(_run_dir(run_id, root), "metrics.csv")
    if not os.path.exists(path):
        return []
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def list_runs(root: Optional[str] = None,
              with_metrics: bool = False) -> "object":
    """One row per training run under outputs/rl/.

    Parameters
    ----------
    with_metrics : bool
        Also load each run's metrics frame into the ``metrics`` column. Off by
        default because discovery is meant to be cheap.
    """
    pd = _pd()
    rows = []
    for run_id in _run_ids(root):
        manifest = load_manifest(run_id, root)
        if not manifest:
            _WARNINGS.append(f"{run_id}: unreadable manifest.json, skipped")
            continue
        args = manifest.get("cli_args") or {}
        kpi = manifest.get("kpi") or {}
        cols = _metrics_columns(run_id, root)
        log_path, log_mode = link_log_dir(run_id, root)
        run_dir = _run_dir(run_id, root)
        save_dir = args.get("save_dir")
        rows.append({
            "run_id": run_id,
            "timestamp": manifest.get("timestamp", ""),
            "git_sha": (manifest.get("git_sha") or "")[:8],
            "algo": args.get("algo"),
            "seed": args.get("seed"),
            "scenarios": args.get("scenarios"),
            "episodes": args.get("episodes"),
            "diff_reward": args.get("diff_reward"),
            "save_dir": save_dir,
            "save_dir_exists": bool(save_dir)
            and os.path.isdir(os.path.join(repo_root(), save_dir)),
            "best_episode": kpi.get("best_episode"),
            "best_mean_reward": kpi.get("best_mean_reward"),
            "best_welfare": kpi.get("best_welfare"),
            "final_mean_reward": kpi.get("final_mean_reward"),
            "n_episodes": kpi.get("n_episodes"),
            "early_stopped": kpi.get("early_stopped"),
            "has_eval": os.path.exists(os.path.join(run_dir, "eval.csv")),
            "has_econ": any(c in cols for c in _ECON_METRIC_COLUMNS),
            "metrics_columns": cols,
            "log_dir": log_path,
            "tb_link": log_mode,
            "path": run_dir,
        })
    df = pd.DataFrame(rows)
    if len(df) and with_metrics:
        df["metrics"] = [load_metrics(rid, root)
                         for rid in df["run_id"]]
    return df


# ---------------------------------------------------------------------------
# Per-run frames
# ---------------------------------------------------------------------------
def load_metrics(run_id: str, root: Optional[str] = None,
                 columns: Optional[List[str]] = None):
    """Per-episode training metrics, with every canonical column addressable.

    A metrics.csv from an older run is returned as-is with the newer columns
    present but empty, so a consumer never has to branch on schema generation.
    """
    pd = _pd()
    path = os.path.join(_run_dir(run_id, root), "metrics.csv")
    frame = pd.DataFrame()
    if os.path.exists(path):
        try:
            frame = pd.read_csv(path)
        except Exception:
            _WARNINGS.append(f"{run_id}: unreadable metrics.csv")
            frame = pd.DataFrame()
    for col in CANONICAL_METRIC_COLUMNS:
        if col not in frame.columns:
            frame[col] = float("nan")
    frame.attrs["run_id"] = run_id
    if columns is not None:
        keep = [c for c in columns if c in frame.columns]
        missing = [c for c in columns if c not in frame.columns]
        frame = frame[keep]
    return frame


def load_evals(run_id: str, root: Optional[str] = None):
    """In-training evaluation history; empty when the run recorded none."""
    pd = _pd()
    path = os.path.join(_run_dir(run_id, root), "eval.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        _WARNINGS.append(f"{run_id}: unreadable eval.csv")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------
def _parse_run_timestamp(timestamp: str) -> Optional[str]:
    """Run timestamps ('2026-09-10T12:00:00') to the log dir's form."""
    if not timestamp or "T" not in timestamp:
        return None
    return timestamp.split("T", 1)[0].replace("-", "") + "-" \
        + timestamp.split("T", 1)[1][:8].replace(":", "")


def link_log_dir(run_id: str, root: Optional[str] = None,
                 runs_dir: Optional[str] = None,
                 window_s: float = 10.0):
    """Find the TensorBoard log directory for a run.

    Returns (path, mode) where mode is one of:

      exact      the manifest recorded it; trust it
      nearest    exactly one log directory starts within ``window_s`` of the
                 run's timestamp
      ambiguous  two or more candidates; path is None rather than a guess,
                 because plotting another run's curve is worse than a gap
      none       nothing close enough

    Older runs predate the recorded link, and their log directories start a few
    seconds after the run directory is created.
    """
    manifest = load_manifest(run_id, root)
    recorded = manifest.get("log_dir")
    if recorded:
        full = recorded if os.path.isabs(recorded) \
            else os.path.join(repo_root(), recorded)
        if os.path.isdir(full):
            return full, "exact"

    runs_dir = runs_dir or runs_root()
    if not os.path.isdir(runs_dir):
        return None, "none"
    target = _stamp_seconds(_parse_run_timestamp(
        manifest.get("timestamp", "")))
    if target is None:
        return None, "none"
    hits = []
    for entry in sorted(os.listdir(runs_dir)):
        full = os.path.join(runs_dir, entry)
        if not os.path.isdir(full):
            continue
        secs = _stamp_seconds(entry)
        if secs is None:
            continue
        delta = secs - target
        # The log directory is created just after the run directory, so only
        # a small non-negative offset counts as this run's log.
        if 0 <= delta <= window_s:
            hits.append((delta, full))
    if not hits:
        return None, "none"
    if len(hits) > 1:
        _WARNINGS.append(
            f"{run_id}: {len(hits)} TensorBoard logs start within "
            f"{window_s:.0f}s; refusing to guess which is this run's")
        return None, "ambiguous"
    return hits[0][1], "nearest"


def load_agent_rewards_tb(log_dir: str, max_points: Optional[int] = None):
    """Per-agent reward curves from a TensorBoard log directory.

    The event files are cached against their size and mtime, so a page render
    does not re-scan a 200-episode run.
    """
    pd = _pd()
    key = (os.path.abspath(log_dir), max_points)
    stamp = _tb_stamp(log_dir)
    cached = _TB_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1].copy()

    from tensorboard.backend.event_processing.event_accumulator \
        import EventAccumulator
    acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    rows = []
    for tag in tags:
        if not tag.startswith("Reward/"):
            continue
        agent = tag.split("/", 1)[1]
        if agent == "mean":
            continue        # the fleet mean is not an agent
        events = acc.Scalars(tag)
        if max_points is not None and len(events) > max_points:
            step = max(1, len(events) // max_points)
            events = events[::step]
        for e in events:
            rows.append({"episode": e.step, "agent": agent, "reward": e.value,
                         "source": "tensorboard"})
    frame = pd.DataFrame(rows, columns=["episode", "agent", "reward", "source"])
    _TB_CACHE[key] = (stamp, frame)
    return frame.copy()


def _tb_stamp(log_dir: str):
    try:
        parts = []
        for name in sorted(os.listdir(log_dir)):
            if name.startswith("events.out.tfevents"):
                st = os.stat(os.path.join(log_dir, name))
                parts.append((name, st.st_size, st.st_mtime))
        return tuple(parts)
    except OSError:
        return ()


def agent_reward_frame(run_id: str, root: Optional[str] = None):
    """Long frame of (episode, agent, reward, source) for one run.

    Prefers the per-agent columns of metrics.csv: they sit inside the run's own
    directory, so no log-directory linking is involved, and they survive the
    log directory being pruned. Both sources are written from the same
    per-episode reward accumulation, so falling back cannot introduce a
    discontinuity in the curve. A partially populated wide block (an agent
    missing) is rejected in favour of the complete TensorBoard history rather
    than silently dropping that agent.
    """
    pd = _pd()
    metrics = load_metrics(run_id, root)
    agent_cols = [c for c in metrics.columns
                  if c.startswith(_AGENT_REWARD_PREFIX)]
    if agent_cols and len(agent_cols) > 0 and len(metrics) > 0:
        # Complete = every agent column present on every row.
        if bool(metrics[agent_cols].notna().all().all()):
            rows = []
            for _, r in metrics.iterrows():
                for col in agent_cols:
                    rows.append({"episode": r["episode"],
                                 "agent": col[len(_AGENT_REWARD_PREFIX):],
                                 "reward": r[col],
                                 "source": "metrics"})
            return pd.DataFrame(rows, columns=["episode", "agent", "reward",
                                               "source"])

    log_dir, mode = link_log_dir(run_id, root)
    if log_dir is None:
        if mode == "ambiguous":
            _WARNINGS.append(
                f"{run_id}: per-agent reward curves unavailable "
                "(ambiguous TensorBoard link)")
        return pd.DataFrame(columns=["episode", "agent", "reward", "source"])
    return load_agent_rewards_tb(log_dir)


def run_detail(run_id: str, root: Optional[str] = None) -> dict:
    """Everything the training-monitor page needs for one run, in one call."""
    return {
        "run_id": run_id,
        "manifest": load_manifest(run_id, root),
        "kpi": load_kpi(run_id, root),
        "config": load_config(run_id, root),
        "metrics": load_metrics(run_id, root),
        "evals": load_evals(run_id, root),
        "per_agent_rewards": agent_reward_frame(run_id, root),
        "log_dir": link_log_dir(run_id, root)[0],
        "tb_link": link_log_dir(run_id, root)[1],
    }


# ---------------------------------------------------------------------------
# Policy directories
# ---------------------------------------------------------------------------
def list_policy_dirs(root: Optional[str] = None):
    """One row per trained-policy directory with its available roles.

    ``checkpoints`` lists only steps every agent has, so a partially written
    checkpoint is not offered as though it were complete.
    """
    pd = _pd()
    base = root or policies_root()
    rows = []
    if not os.path.isdir(base):
        return pd.DataFrame()
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        finals, by_step = [], {}
        for fn in os.listdir(d):
            if not fn.endswith(".pt"):
                continue
            m = _CKPT_RE.match(fn)
            if m:
                by_step.setdefault(int(m.group("step")), set()).add(
                    m.group("agent"))
            else:
                finals.append(fn)
        if not finals and not by_step:
            continue
        n_agents = len(finals)
        # Only steps present for every agent count as available.
        checkpoints = sorted(step for step, agents in by_step.items()
                             if n_agents and len(agents) == n_agents)
        roles = []
        if finals:
            roles.append("final")
        for role in ("best", "last"):
            if os.path.isdir(os.path.join(d, role)) and \
                    any(f.endswith(".pt")
                        for f in os.listdir(os.path.join(d, role))):
                roles.append(role)
        rows.append({
            "name": name, "path": d, "n_agents": n_agents,
            "roles": roles, "checkpoints": checkpoints,
            "has_best": "best" in roles, "has_last": "last" in roles,
            "n_final": len(finals),
            "n_ckpt_agents": max((len(v) for v in by_step.values()),
                                 default=0),
            "mtime": os.path.getmtime(d),
        })
    return pd.DataFrame(rows)


def policy_dir_run_id(policy_dir_name: str,
                      root: Optional[str] = None) -> Optional[str]:
    """Run id whose manifest wrote into policies/<policy_dir_name>.

    The manifest records save_dir with whatever path separators the platform
    used, so both sides are normalized before comparing.
    """
    want = os.path.normpath(os.path.join(repo_root(), "policies",
                                         policy_dir_name))
    for run_id in _run_ids(root):
        save_dir = (load_manifest(run_id, root).get("cli_args") or {}) \
            .get("save_dir")
        if not save_dir:
            continue
        if os.path.normpath(os.path.join(repo_root(), save_dir)) == want:
            return run_id
    return None


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------
def list_eval_results(root: Optional[str] = None):
    """Concatenated results/*_eval.csv with a source tag per row.

    Evaluation files come in two generations (with and without the consumer
    columns), so the block is unioned and the missing cells are left empty
    rather than dropped. Each file's rows keep their ``agent`` column, which
    carries the per-agent rows plus the fleet's ``ALL`` row.
    """
    pd = _pd()
    base = root or results_root()
    if not os.path.isdir(base):
        return pd.DataFrame()
    frames = []
    for fn in sorted(os.listdir(base)):
        if not fn.endswith(".csv") or "eval" not in fn:
            continue
        path = os.path.join(base, fn)
        try:
            frame = pd.read_csv(path)
        except Exception:
            _WARNINGS.append(f"{fn}: unreadable evaluation CSV")
            continue
        frame["source_file"] = fn
        frame["has_consumer_metrics"] = "cs_delta" in frame.columns
        m = re.search(r"seed(\d+)", fn)
        frame["seed_guess"] = int(m.group(1)) if m else float("nan")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in CANONICAL_METRIC_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan")
    return out


# Columns the fleet-level consumer accounting is expressed in.
FLEET_METRIC_COLUMNS = [
    "profit_delta", "genuine_welfare_delta", "welfare_delta",
    "cs_baseline", "cs_rl", "cs_delta",
    "cp_baseline", "cp_rl", "cp_delta",
    "lmp_markup_baseline", "lmp_markup_rl", "lmp_markup_delta",
    "market_power_arb", "market_power_power",
    "re_rate_baseline", "re_rate_rl",
]


# Consumer-accounting conventions that appear in results/ on disk.
#
# The 2026-09-09 fix to consumer surplus stopped letting storage discharge
# offset the consumer bill (qbuy = served - (pv_used + wind_used), with p_dis
# ignored). That flipped the sign of cs_delta and cp_delta on three of the four
# scenarios, turning "storage bidding harms consumers" into "consumers pay
# less". Anything computed before the fix is still on disk, so a chart that
# pools the two generations asserts the opposite of the truth.
CONSUMER_CONVENTIONS = ("p_dis_excluded", "p_dis_offset_legacy", "unknown")

# When the corrected convention first produced results on disk. The earliest
# file carrying it is 2026-09-09 12:23; the day's earlier files (00:52-00:58)
# are still on the old semantics, so the cutoff is a time, not a date.
P_DIS_FIX_TS = datetime.datetime(2026, 9, 9, 12, 0, 0).timestamp()


def _convention_of(frame) -> str:
    """Which consumer-surplus convention produced these rows.

    Emission time separates the generations. A file with no consumer columns
    is 'unknown' rather than guessed, and one carrying them is dated from its
    mtime against P_DIS_FIX_TS.
    """
    if "cs_delta" not in frame.columns or frame["cs_delta"].isna().all():
        return "unknown"
    path = frame.attrs.get("source_path")
    if not path or not os.path.exists(path):
        return "unknown"
    return ("p_dis_excluded" if os.path.getmtime(path) >= P_DIS_FIX_TS
            else "p_dis_offset_legacy")


# Money-unit conventions that appear in results/ on disk.
#
# The 2026-09-13 unification made every monetary quantity an energy multiplied
# by a price, so a period's power now carries the 0.25 h period length (see
# money.py). Before it, the profit and payment ledger valued a period's power as
# if a period were an hour, leaving every absolute money figure on disk at four
# times the same quantity today. Pooling the generations compares a 4x larger
# delta against a 4x smaller one, silently, because the sign and the ordering
# survive. Unit-rate metrics (lmp_markup_*) and any ratio of two money figures
# are invariant and stay comparable across generations; absolute money is not.
MONEY_CONVENTIONS = ("mwh_dt", "period_power_legacy", "unknown")

# When the unified convention first produced results on disk. The newest result
# written before the change is dated 2026-09-10, so a date is a safe cutoff.
MONEY_UNIT_FIX_TS = datetime.datetime(2026, 9, 13, 0, 0, 0).timestamp()


def _money_convention_of(frame) -> str:
    """Which money-unit convention produced these rows.

    Same dated-generation mechanism as `_convention_of`: emission time
    separates the generations, and a file that cannot be dated, or that carries
    no money columns, is 'unknown' rather than guessed.
    """
    if "profit_delta" not in frame.columns \
            or frame["profit_delta"].isna().all():
        return "unknown"
    path = frame.attrs.get("source_path")
    if not path or not os.path.exists(path):
        return "unknown"
    return ("mwh_dt" if os.path.getmtime(path) >= MONEY_UNIT_FIX_TS
            else "period_power_legacy")


def fleet_rows(root: Optional[str] = None,
               convention: Optional[str] = None,
               money_convention: Optional[str] = None):
    """One row per (evaluation file, scenario) for the fleet as a whole.

    Evaluation CSVs hold a per-agent row per scenario plus one ``agent="ALL"``
    row carrying the day-level consumer accounting; only the ALL row has the
    fleet numbers, so this keeps just those. ``policy_label`` is the filename
    without its ``_eval.csv`` suffix, which is how a result is identified in
    conversation ("e8_matd3_10", "quickval_matd3_cc_ref").

    ``cs_convention`` records which consumer-surplus convention produced the
    row (see CONSUMER_CONVENTIONS); pass ``convention`` to keep only that one,
    which is what a consumer-facing comparison should do.

    ``money_convention`` does the same for the money unit (see
    MONEY_CONVENTIONS). It defaults to None rather than to the current
    convention because every result on disk predates the unification, so
    filtering by default would empty every view; a comparison of absolute money
    figures across policies should pass a convention explicitly.
    """
    pd = _pd()
    rows = list_eval_results(root)
    cols = ["source_file", "policy_label", "scenario", "agent",
            "cs_convention", "money_convention"] + FLEET_METRIC_COLUMNS
    if not len(rows) or "agent" not in rows.columns:
        return pd.DataFrame(columns=cols)
    base = root or results_root()
    fleet = rows[rows["agent"] == "ALL"].copy()
    fleet["policy_label"] = fleet["source_file"].str.replace(
        r"_eval\.csv$", "", regex=True)
    # One read per file: both conventions are decided from the same frame.
    peeks = [_peek(os.path.join(base, fn)) for fn in fleet["source_file"]]
    fleet["cs_convention"] = [_convention_of(f) for f in peeks]
    fleet["money_convention"] = [_money_convention_of(f) for f in peeks]
    for col in FLEET_METRIC_COLUMNS:
        if col not in fleet.columns:
            fleet[col] = float("nan")
    if convention is not None:
        fleet = fleet[fleet["cs_convention"] == convention]
    if money_convention is not None:
        fleet = fleet[fleet["money_convention"] == money_convention]
    return fleet[cols]


def pareto_pairs(root: Optional[str] = None,
                 convention: Optional[str] = "p_dis_excluded",
                 money_convention: Optional[str] = None):
    """Profit against consumer surplus, one point per policy and scenario.

    Each row pairs a policy's profit gain with the change in consumer surplus
    on the same cleared day, so the two can be plotted against each other and
    the non-conflicting quadrant (both rise) told apart from the trade-off
    quadrant (storage gains, consumers pay).

    Rows without consumer accounting are dropped: without a surplus figure
    there is no point to place. ``convention`` defaults to the corrected
    consumer-surplus semantics, since the pre-fix generation reports the
    opposite sign on three of the four scenarios; pass None to keep every
    generation, with ``cs_convention`` marking each point.
    """
    pd = _pd()
    fleet = fleet_rows(root, convention=convention,
                       money_convention=money_convention)
    cols = ["policy_label", "scenario", "cs_convention", "money_convention",
            "profit_delta", "genuine_welfare_delta", "cs_delta", "cp_delta",
            "lmp_markup_delta", "market_power_power",
            "profit_rises", "cs_rises", "conflict"]
    if not len(fleet):
        return pd.DataFrame(columns=cols)
    paired = fleet[fleet["cs_delta"].notna()
                   & fleet["profit_delta"].notna()].copy()
    paired["profit_rises"] = paired["profit_delta"] > 0
    paired["cs_rises"] = paired["cs_delta"] > 0
    # The conflict the mechanism design is meant to resolve: the fleet gains
    # while consumers are made worse off.
    paired["conflict"] = paired["profit_rises"] & ~paired["cs_rises"]
    return paired[cols].reset_index(drop=True)


# Experiment families, in the order the paginated view lists them. The prefix
# is matched against an evaluation file's policy_label; the dict's order is
# the display order, so the main baseline comes first.
_EXPERIMENT_FAMILIES = (
    ("matd3_cc_rot", "MATD3 主基线"),
    ("multi_matd3", "MATD3 其他族"),
    ("qmatd3", "qmatd3 对照"),
    ("quickval", "快速验证"),
    ("masac", "MASAC pilot"),
    ("e8", "E8 网络耦合"),
    ("e9", "E9 市场力惩罚"),
)

_VARIANT_PATTERNS = (
    # E8 varies network line capacity; the filename spells the value without
    # its decimal point (10 -> 1.0, 08 -> 0.8).
    (re.compile(r"^e8_(?:matd3|phys)_(\d)(\d)$"),
     lambda m: f"capacity={m.group(1)}.{m.group(2)}"),
    # E9's market-impact penalty weight.
    (re.compile(r"^e9_matd3_p(\d)(\d)$"),
     lambda m: f"λ={m.group(1)}.{m.group(2)}"),
    # Seed is the varying factor for the multi-seed families.
    (re.compile(r"seed(\d+)"), lambda m: f"seed={m.group(1)}"),
)


def experiment_family(policy_label: str) -> str:
    """Which experiment a result belongs to, from its file name."""
    for prefix, name in _EXPERIMENT_FAMILIES:
        if policy_label.startswith(prefix):
            return name
    return "其他"


def experiment_variant(policy_label: str) -> str:
    """The setting a result varied, as text, or "" when the name encodes none.

    Without this a reader sees only file names; the manipulated axis is what
    makes a row interpretable.
    """
    for pattern, render in _VARIANT_PATTERNS:
        m = pattern.search(policy_label)
        if m:
            return render(m)
    return ""


def experiment_table(root: Optional[str] = None,
                     convention: Optional[str] = None,
                     families: Optional[List[str]] = None):
    """Fleet results annotated with their experiment family and variant.

    One row per (evaluation file, scenario), grouping the results on disk into
    the experiments they came from. ``convention`` defaults to None here: the
    experiment view spans both accounting generations, and each row carries
    ``cs_convention`` so a consumer-surplus comparison can filter or label.
    """
    pd = _pd()
    fleet = fleet_rows(root, convention=convention)
    cols = ["family", "variant", "policy_label", "scenario", "cs_convention"] \
        + FLEET_METRIC_COLUMNS
    if not len(fleet):
        return pd.DataFrame(columns=cols)
    fleet = fleet.copy()
    fleet["family"] = fleet["policy_label"].map(experiment_family)
    fleet["variant"] = fleet["policy_label"].map(experiment_variant)
    if families is not None:
        fleet = fleet[fleet["family"].isin(families)]
    order = {name: i for i, (_, name) in enumerate(_EXPERIMENT_FAMILIES)}
    order["其他"] = len(order)
    fleet["_fam"] = fleet["family"].map(lambda f: order.get(f, len(order)))
    fleet = fleet.sort_values(["_fam", "policy_label", "scenario"])
    return fleet[cols].reset_index(drop=True)


def run_summary(root: Optional[str] = None):
    """One row per training run: what it trained and how the curve ended.

    The critic-loss first-quarter / last-quarter means and their trend give a
    cheap divergence check without loading every curve. A run whose loss was
    never recorded (all episodes still in random exploration) reports no
    figures rather than zeros.
    """
    pd = _pd()
    rows = []
    for meta in list_runs(root).to_dict("records"):
        metrics = load_metrics(meta["run_id"], root)
        loss = metrics["critic_loss"].dropna() if "critic_loss" in metrics \
            else pd.Series(dtype=float)
        quarter = max(1, len(loss) // 4)
        first = float(loss.head(quarter).mean()) if len(loss) else None
        last = float(loss.tail(quarter).mean()) if len(loss) else None
        if first is None or last is None:
            trend = "unknown"
        else:
            trend = "falling" if last < first else "rising"
        rows.append({
            "run_id": meta["run_id"],
            "date": (meta["timestamp"] or "")[:16].replace("T", " "),
            "algo": meta["algo"],
            "seed": meta["seed"],
            "scenarios": meta["scenarios"],
            "episodes": meta["n_episodes"],
            "best_episode": meta["best_episode"],
            "final_mean_reward": meta["final_mean_reward"],
            "best_mean_reward": meta["best_mean_reward"],
            "critic_first": first,
            "critic_last": last,
            "critic_max": float(loss.max()) if len(loss) else None,
            "critic_trend": trend,
            "has_econ": meta["has_econ"],
            "tb_link": meta["tb_link"],
            "save_dir": meta["save_dir"],
        })
    cols = ["run_id", "date", "algo", "seed", "scenarios", "episodes",
            "best_episode", "final_mean_reward", "best_mean_reward",
            "critic_first", "critic_last", "critic_max", "critic_trend",
            "has_econ", "tb_link", "save_dir"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("date").reset_index(drop=True)


def eval_summary(root: Optional[str] = None):
    """Mean fleet outcome per (accounting convention, scenario).

    Averaging across conventions would mix two definitions of the same column
    whose signs disagree, so they are kept apart and the caller decides which
    generation to publish. ``n_points`` is the number of fleet days averaged,
    so a thin cell is visible as thin.
    """
    pd = _pd()
    cols = ["cs_convention", "scenario", "n_points"] + FLEET_METRIC_COLUMNS
    fleet = fleet_rows(root, convention=None)
    if not len(fleet):
        return pd.DataFrame(columns=cols)
    keep = [c for c in FLEET_METRIC_COLUMNS if c in fleet.columns]
    grouped = fleet.groupby(["cs_convention", "scenario"])[keep] \
        .mean().reset_index()
    counts = fleet.groupby(["cs_convention", "scenario"]).size() \
        .reset_index(name="n_points")
    out = counts.merge(grouped, on=["cs_convention", "scenario"])
    return out[cols].sort_values(["cs_convention", "scenario"]) \
        .reset_index(drop=True)


def _peek(path: str):
    """Read just the columns that identify an evaluation file's conventions,
    tagged with its path so the caller can date it.

    ``cs_delta``/``cp_delta`` mark the consumer-surplus generation and
    ``profit_delta`` marks the money-unit generation; a file that predates a
    column simply will not carry it, which is how the 'unknown' case arises.
    """
    pd = _pd()
    frame = pd.DataFrame()
    if os.path.exists(path):
        try:
            frame = pd.read_csv(path, usecols=lambda c: c in
                                ("cs_delta", "cp_delta", "profit_delta"))
        except Exception:
            frame = pd.DataFrame()
    frame.attrs["source_path"] = path
    return frame
