# audit_action_mapping_gradients.py
"""Offline comparison of the tanh and truthful-gate quote mappings.

No training, no env, no clearing. Values and analytic derivatives on a fixed
logit grid: old `a = mid + tanh(z) * half`, new `a = 0` (gate closed) or
`a = h * sigmoid(z)` (gate open). A central difference checks the analytic
derivatives. The discrete gate's estimator is not a derivative of this map and
is discussed in the report, not tabulated as if it were.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from action_gate import gated_quote

GRID = (-50.0, -10.0, -5.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 5.0, 10.0,
        50.0)
LEGS = (("charge", 150.0), ("discharge", 300.0))
CSV_OUT = "data/processed/v2_cy_jn_2025/pilot/action_mapping_grid.csv"
JSON_OUT = "data/processed/v2_cy_jn_2025/pilot/action_mapping_gradient_check.json"


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + np.exp(-x))
    e = np.exp(x)
    return e / (1.0 + e)


def old_map(z: float, high: float) -> float:
    return high / 2.0 + np.tanh(z) * high / 2.0


def old_derivative(z: float, high: float) -> float:
    t = np.tanh(z)
    return high / 2.0 * (1.0 - t * t)


def open_map(z: float, high: float) -> float:
    return high * _sigmoid(z)


def open_derivative(z: float, high: float) -> float:
    s = _sigmoid(z)
    return high * s * (1.0 - s)


def _numeric(fn, z, high, eps=1e-5):
    return (fn(z + eps, high) - fn(z - eps, high)) / (2.0 * eps)


def main(argv=None) -> int:
    root = Path.cwd()
    rows = []
    max_err_old = 0.0
    max_err_open = 0.0
    for leg, high in LEGS:
        for z in GRID:
            old = old_map(z, high)
            old_d = old_derivative(z, high)
            open_v = open_map(z, high)
            open_d = open_derivative(z, high)
            closed = float(gated_quote([False], [z], high=[high])[0])
            max_err_old = max(max_err_old,
                              abs(old_d - _numeric(old_map, z, high)))
            max_err_open = max(max_err_open,
                               abs(open_d - _numeric(open_map, z, high)))
            rows.append({
                "leg": leg, "high": high, "logit": z,
                "old_quote": old, "old_d_dz": old_d,
                "old_exact_truth": abs(old) == 0.0,
                "old_tanh_abs": abs(float(np.tanh(z))),
                "gate_closed_quote": closed,
                "gate_closed_d_dz": 0.0,
                "gate_closed_exact_truth": closed == 0.0,
                "gate_open_quote": open_v, "gate_open_d_dz": open_d,
                "gate_open_exact_truth": open_v == 0.0,
                "gate_open_sigmoid": _sigmoid(z),
            })
    path = root / CSV_OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def _at(leg, z, key):
        return next(r[key] for r in rows if r["leg"] == leg
                    and r["logit"] == z)

    summary = {
        "grid": list(GRID),
        "max_abs_derivative_error_old": max_err_old,
        "max_abs_derivative_error_open": max_err_open,
        "old_quote_at_minus_5_charge": _at("charge", -5.0, "old_quote"),
        "old_quote_at_minus_10_charge": _at("charge", -10.0, "old_quote"),
        "gate_closed_quote_any_logit": 0.0,
        "gate_open_quote_at_minus_5_charge":
            _at("charge", -5.0, "gate_open_quote"),
        "old_derivative_at_minus_5_charge":
            _at("charge", -5.0, "old_d_dz"),
        "old_derivative_at_minus_50_charge":
            _at("charge", -50.0, "old_d_dz"),
        "gate_open_derivative_at_0_charge":
            _at("charge", 0.0, "gate_open_d_dz"),
        "gate_open_derivative_at_minus_10_charge":
            _at("charge", -10.0, "gate_open_d_dz"),
        "gate_open_derivative_at_10_charge":
            _at("charge", 10.0, "gate_open_d_dz"),
        "old_exact_truth_on_grid": any(r["old_exact_truth"] for r in rows),
        "gate_closed_exact_truth_on_grid": all(
            r["gate_closed_exact_truth"] for r in rows),
        "gate_open_exact_truth_on_grid": any(
            r["gate_open_exact_truth"] for r in rows),
        "note": "the discrete gate has no analytic derivative; straight-through "
                "and REINFORCE are estimators discussed in the report",
    }
    (root / JSON_OUT).write_text(json.dumps(summary, indent=2),
                                 encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}\nwrote {root / JSON_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
