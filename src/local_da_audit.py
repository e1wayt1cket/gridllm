"""Pure comparisons over a cleared day: prices, window gaps, evaluation counts.

The training stack prices a block on a 20-period window; the audit clears the
whole day at once. These functions describe how far the two disagree without
deciding which one is right, so the decision stays with the reader of the
report rather than with the tool that produced the table.
"""

import json

import numpy as np

#: The per-day feasibility screen, rewritten from the full-library measurement
#: that followed the market's declared reactive asset. One definition, because a
#: screen and the day set citing it must not drift.
SCREEN = "reports/local_da_v7_2026-11-18/qualification/screen"


def compare_stacks(training_lmp, audit_lmp, *, floor=1.0):
    """Sign agreement and relative gap where the audit price clears a floor.

    A cell is compared only where the audit price exceeds `floor` in magnitude:
    a relative gap against a near-zero price is a ratio of two noises, and a
    sign on such a cell carries no information.
    """
    training = np.asarray(training_lmp, dtype=float)
    audit = np.asarray(audit_lmp, dtype=float)
    if training.shape != audit.shape:
        raise ValueError("the two price stacks must cover the same cells")
    mask = np.abs(audit) > floor
    cells = int(mask.sum())
    if cells == 0:
        return {"cells": 0, "sign_agreement": None, "median_rel_gap": None}
    agreement = float((np.sign(training[mask]) == np.sign(audit[mask])).mean())
    gap = float(np.median(np.abs(training[mask] - audit[mask]) / np.abs(audit[mask])))
    return {"cells": cells, "sign_agreement": agreement, "median_rel_gap": gap}


def rolling_gap(day_lmp, block_lmps, committed):
    """Per-period relative gap between the block prices and the day prices.

    `committed` is the number of periods each block actually commits, so only
    those periods are compared against the day's own price for them.
    """
    day = np.asarray(day_lmp, dtype=float).ravel()
    blocks = np.asarray(block_lmps, dtype=float).ravel()
    periods = min(len(day), len(blocks) * max(int(committed), 1))
    if periods == 0:
        return {"median": None, "p95": None, "periods": 0}
    day = day[:periods]
    blocks = np.repeat(blocks, max(int(committed), 1))[:periods]
    gap = np.abs(blocks - day) / np.abs(day)
    return {"median": float(np.median(gap)), "p95": float(np.percentile(gap, 95)),
            "periods": int(periods)}


def feasibility_selected(screen_dir=SCREEN):
    """The library's block counts and the train days the exact screen dropped.

    `train_selected` is the count `train_pairs` returns, so the two numbers a
    manifest records -- the block size and the days actually trained on -- are
    read from the same pass over the same screen.
    """
    from local_da_days import LIBRARY, train_pairs

    split = json.loads((LIBRARY / "split.json").read_text(encoding="utf-8"))
    counts = {block: len(rows) for block, rows in split["pairs"].items()}
    pairs, selection = train_pairs(screen_dir)
    counts["train_selected"] = len(pairs)
    return {"counts": counts, "dropped": selection["dropped"]}


def eval_counts(days, results):
    """How many of the days evaluated, and which ones could not be cleared."""
    attempted = list(days)
    infeasible = [day for day in attempted if not results.get(day, {}).get("settled")]
    return {"attempted": len(attempted),
            "evaluated": len(attempted) - len(infeasible),
            "infeasible": infeasible}
