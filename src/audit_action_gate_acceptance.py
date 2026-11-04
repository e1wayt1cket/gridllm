# audit_action_gate_acceptance.py
"""Run the truthful-gate acceptance checks and save the result. Pure functions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from action_gate import gated_quote

OUT = "data/processed/v2_cy_jn_2025/pilot/action_gate_test_results.json"
HIGHS = (150.0, 300.0)


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def run() -> dict:
    checks = {}
    logits = (-1e6, -5.0, 0.0, 5.0, 1e6)
    checks["closed_is_exact_zero"] = all(
        gated_quote([False, False], [z, z])[i] == 0.0
        for z in logits for i in (0, 1))
    grid = np.linspace(-20.0, 20.0, 401)
    open_q = gated_quote(np.ones((grid.size, 2), dtype=bool),
                         np.stack([grid, grid], axis=1))
    checks["open_is_legal"] = bool(
        np.all(open_q >= 0.0)
        and np.all(open_q[:, 0] <= HIGHS[0])
        and np.all(open_q[:, 1] <= HIGHS[1]))
    checks["open_is_monotone"] = bool(
        np.all(np.diff(open_q[:, 0]) >= 0)
        and np.all(np.diff(open_q[:, 1]) >= 0))
    q = gated_quote([False, True], [0.0, 0.0])
    checks["gates_are_independent"] = bool(q[0] == 0.0 and q[1] > 0.0)
    with np.errstate(over="raise", invalid="raise"):
        extreme = gated_quote([True, True], [1e6, -1e6])
    checks["extreme_logits_stable"] = bool(np.isfinite(extreme).all()
                                           and (extreme >= 0).all())
    checks["no_overshoot"] = bool(
        gated_quote([True, True], [60.0, 60.0]).tolist() == [150.0, 300.0])
    checks["deterministic"] = bool(np.array_equal(
        gated_quote([True, False], [1.3, -2.1]),
        gated_quote([True, False], [1.3, -2.1])))
    old_minus5 = 75.0 + np.tanh(-5.0) * 75.0
    checks["old_tanh_is_not_exact_at_minus_5"] = bool(old_minus5 > 0.0)
    checks["new_gate_is_exact_at_minus_5"] = bool(
        gated_quote([False], [-5.0], high=[150.0])[0] == 0.0)
    checks["bad_shape_rejected"] = _raises(
        lambda: gated_quote([True, False], [0.0, 0.0, 0.0]))
    checks["bad_bound_rejected"] = _raises(
        lambda: gated_quote([True], [0.0], high=[-1.0]))
    checks["nonfinite_rejected"] = _raises(
        lambda: gated_quote([True], [float("inf")], high=[150.0]))
    values = {
        "old_tanh_quote_at_minus_5": float(old_minus5),
        "old_tanh_quote_at_minus_10": float(75.0 + np.tanh(-10.0) * 75.0),
        "old_tanh_quote_at_minus_50": float(75.0 + np.tanh(-50.0) * 75.0),
        "new_gate_closed_quote": 0.0,
        "new_gate_open_quote_at_minus_5": float(
            gated_quote([True], [-5.0], high=[150.0])[0]),
    }
    return {"checks": checks, "values": values,
            "all_pass": all(checks.values())}


def main(argv=None) -> int:
    payload = run()
    path = Path.cwd() / OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
