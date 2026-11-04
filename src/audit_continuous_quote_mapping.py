"""Compare continuous quote maps for the open gate branch.

The gate already makes truthful exactly zero when closed. This script only
compares the mapping used when the gate is open:

    q = h * g(z)

Candidates:
  A. logistic, g = sigmoid(z)
  B. temperature-scaled logistic, g = sigmoid(z / tau)
  C. bounded piecewise map with a linear core on [0.01, 0.99] and quadratic
     tails that match value and slope at the joint, g(0) = 0.5.

Every candidate is monotone, bounded in [0, h] without clipping, and leaves the
gate-closed zero exact. The comparison reports reachability, the logit needed
to reach the high quote region, and dq/dz there.

Read-only; no trainer, checkpoint or action spec is changed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path("data/processed/v2_cy_jn_2025")
HIGHS = {"charge": 150.0, "discharge": 300.0}
Q_FRACTIONS = (0.01, 0.05, 0.25, 0.5, 0.75, 0.86, 0.9, 0.95, 0.99,
               0.997, 0.99967, 0.9999)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1.0 - s)


# Candidate C: linear core, quadratic tails.
C_Q_HI = 0.99
C_Z_C = 2.5
C_SLOPE = (C_Q_HI - 0.5) / C_Z_C
C_B = C_SLOPE / (2.0 * (1.0 - C_Q_HI))


def piecewise(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    core = np.abs(z) <= C_Z_C
    out[core] = 0.5 + C_SLOPE * z[core]
    hi = z > C_Z_C
    d = z[hi] - C_Z_C
    out[hi] = 1.0 - (1.0 - C_Q_HI) / (1.0 + C_B * d) ** 2
    lo = z < -C_Z_C
    d = -z[lo] - C_Z_C
    out[lo] = (1.0 - C_Q_HI) / (1.0 + C_B * d) ** 2
    return out


def piecewise_deriv(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    core = np.abs(z) <= C_Z_C
    out[core] = C_SLOPE
    hi = z > C_Z_C
    d = z[hi] - C_Z_C
    out[hi] = C_SLOPE / (1.0 + C_B * d) ** 3
    lo = z < -C_Z_C
    d = -z[lo] - C_Z_C
    out[lo] = C_SLOPE / (1.0 + C_B * d) ** 3
    return out


def candidates():
    yield "A_logistic", 1.0, sigmoid, sigmoid_deriv
    for tau in (0.5, 1.0, 2.0, 4.0):
        yield (f"B_logistic_tau{tau:g}", tau,
               lambda z, t=tau: sigmoid(z / t),
               lambda z, t=tau: sigmoid_deriv(z / t) / t)
    yield "C_piecewise_linear_core", 1.0, piecewise, piecewise_deriv


def z_for_fraction(fn, target, lo=-200.0, hi=200.0):
    """Bisection on a monotone map; 0.5 is exact at z=0."""
    if abs(target - 0.5) < 1e-12:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main(argv=None) -> int:
    z_grid = np.linspace(-20.0, 20.0, 40001)
    rows = []
    summary = {"candidates": {}, "constants": {
        "C_q_hi": C_Q_HI, "C_z_core": C_Z_C, "C_slope": C_SLOPE,
        "C_tail_b": C_B,
        "note": "C is symmetric: g(-z)=1-g(z), slope 0.196 per logit on "
                "0.01-0.99, quadratic tails matched in value and slope.",
    }}
    for name, tau, fn, dfn in candidates():
        vals = fn(z_grid)
        dvals = dfn(z_grid)
        mono = bool(np.all(np.diff(vals) >= -1e-12))
        finite = bool(np.all(np.isfinite(vals)) and np.all(np.isfinite(dvals)))
        in_box = bool(np.all(vals >= 0.0) and np.all(vals <= 1.0))
        z_lo = z_for_fraction(fn, 0.01)
        z_hi = z_for_fraction(fn, 0.99)
        metrics = {
            "tau": tau, "monotone": mono, "finite": finite,
            "in_unit_interval": in_box, "clipping": False,
            "gate_closed_exact_zero": True,
            "logit_for_1pct": float(z_lo),
            "logit_for_99pct": float(z_hi),
            "logit_span_1_to_99": float(z_hi - z_lo),
            "logit_for_258_cny_300": float(z_for_fraction(fn, 258.0 / 300.0)),
            "logit_for_299_cny_300": float(
                z_for_fraction(fn, 299.0 / 300.0)),
            "logit_for_299_9_cny_300": float(
                z_for_fraction(fn, 299.9 / 300.0)),
            "logit_to_299_from_150": float(
                z_for_fraction(fn, 299.0 / 300.0)
                - z_for_fraction(fn, 150.0 / 300.0)),
        }
        ratios = []
        for frac in Q_FRACTIONS:
            z = z_for_fraction(fn, frac)
            for leg, h in HIGHS.items():
                grad = float(dfn(np.array([z]))[0]) * h
                rows.append({
                    "candidate": name, "tau": tau, "leg": leg,
                    "quote_fraction": frac, "quote_cny": frac * h,
                    "logit": z, "dq_dlogit_cny": grad,
                })
                if leg == "discharge" and 0.86 <= frac <= 0.997:
                    ratios.append(grad)
        mid = float(dfn(np.array([0.0]))[0]) * HIGHS["discharge"]
        high_min = float(min(ratios)) if ratios else float("nan")
        metrics["midpoint_gradient_cny"] = mid
        metrics["high_region_min_gradient_cny"] = high_min
        metrics["high_min_over_mid"] = (high_min / mid if mid else float("nan"))
        metrics["gradient_at_299_9_cny"] = float(
            dfn(np.array([z_for_fraction(fn, 299.9 / 300.0)]))[0] * 300.0)
        metrics["gradient_at_299_cny"] = float(
            dfn(np.array([z_for_fraction(fn, 299.0 / 300.0)]))[0] * 300.0)
        summary["candidates"][name] = metrics

    out_csv = DATA / "continuous_quote_mapping_grid.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (DATA / "continuous_quote_mapping_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # Self-check: every candidate is monotone, bounded and finite; the chosen
    # linear-core map reaches the high region with a higher gradient than the
    # logistic over 0.86-0.997.
    for name, tau, fn, dfn in candidates():
        v = fn(np.linspace(-50.0, 50.0, 20001))
        assert np.all(np.diff(v) >= -1e-12), name
        assert np.all(v >= 0.0) and np.all(v <= 1.0), name
        assert np.all(np.isfinite(v)), name
    c = summary["candidates"]["C_piecewise_linear_core"]
    a = summary["candidates"]["A_logistic"]
    assert c["high_region_min_gradient_cny"] > \
        a["high_region_min_gradient_cny"], (c, a)
    assert abs(piecewise(np.array([0.0]))[0] - 0.5) < 1e-12
    print(json.dumps(summary["candidates"], indent=1))
    print(f"wrote {out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
