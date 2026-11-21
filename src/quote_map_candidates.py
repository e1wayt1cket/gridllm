"""Candidate quote maps for the offline saturation audit.

Nothing in the training stack imports this module. `linear_core_v1` stays the
only registered map; these candidates exist so a diagnostic process can score
them on identical logits without touching `da_quote_map.py`.

All candidates keep the v1 shape and differ only in the core half-width:
`slope = 0.49 / Z`, `tail_b = slope / 0.02`, `g(0) = 0.5`, `g(+-Z) = 0.99/0.01`,
quadratic tails outside the core. `Z = 2.5` therefore reproduces v1 exactly and
serves as the in-process control for every comparison.

`patched()` swaps `da_quote_map.map_fraction` for the duration of a forward, so
the gate estimator, the gate-logit bound and the straight-through gradient all
stay the frozen ones and only the price branch changes.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

import da_quote_map as qmap

Q_HI = qmap.Q_HI
Q_LO = 1.0 - Q_HI
SPAN = Q_HI - 0.5
CANDIDATE_Z = (2.5, 6.0, 8.0, 10.0)


def params(z_core: float) -> dict:
    if z_core <= 0.0:
        raise ValueError("z_core must be positive")
    slope = SPAN / float(z_core)
    return {"z_core": float(z_core), "slope": slope,
            "tail_b": slope / (2.0 * (1.0 - Q_HI)),
            "q_hi": Q_HI, "q_lo": Q_LO}


def candidate_id(z_core: float) -> str:
    return "candidate_linear_core_z" + str(z_core).replace(".", "p")


def fraction_torch(z: torch.Tensor, z_core: float) -> torch.Tensor:
    p = params(z_core)
    zc = p["z_core"]
    z = torch.clamp(z, -qmap.Z_GUARD, qmap.Z_GUARD)
    core = 0.5 + p["slope"] * z
    upper = 1.0 - Q_LO / (1.0 + p["tail_b"] * (z - zc)) ** 2
    lower = Q_LO / (1.0 + p["tail_b"] * (-z - zc)) ** 2
    return torch.where(z > zc, upper, torch.where(z < -zc, lower, core))


def derivative_torch(z: torch.Tensor, z_core: float) -> torch.Tensor:
    p = params(z_core)
    zc = p["z_core"]
    z = torch.clamp(z, -qmap.Z_GUARD, qmap.Z_GUARD)
    upper = p["slope"] / (1.0 + p["tail_b"] * (z - zc)) ** 3
    lower = p["slope"] / (1.0 + p["tail_b"] * (-z - zc)) ** 3
    return torch.where(z > zc, upper,
                       torch.where(z < -zc, lower,
                                   torch.full_like(z, p["slope"])))


def fraction_np(z: np.ndarray, z_core: float) -> np.ndarray:
    p = params(z_core)
    zc = p["z_core"]
    z = np.clip(np.asarray(z, dtype=float), -qmap.Z_GUARD, qmap.Z_GUARD)
    core = 0.5 + p["slope"] * z
    upper = 1.0 - Q_LO / (1.0 + p["tail_b"] * (z - zc)) ** 2
    lower = Q_LO / (1.0 + p["tail_b"] * (-z - zc)) ** 2
    return np.where(z > zc, upper, np.where(z < -zc, lower, core))


def derivative_np(z: np.ndarray, z_core: float) -> np.ndarray:
    # `np.asarray(..., dtype=float)` is load-bearing: an integer input would
    # otherwise make `np.full_like` truncate the core slope to 0.
    p = params(z_core)
    zc = p["z_core"]
    z = np.clip(np.asarray(z, dtype=float), -qmap.Z_GUARD, qmap.Z_GUARD)
    upper = p["slope"] / (1.0 + p["tail_b"] * (z - zc)) ** 3
    lower = p["slope"] / (1.0 + p["tail_b"] * (-z - zc)) ** 3
    return np.where(z > zc, upper,
                    np.where(z < -zc, lower, np.full_like(z, p["slope"])))


@contextmanager
def patched(z_core: float):
    """Serve `fraction_torch(z_core)` wherever `da_quote_map.map_fraction` is
    looked up, then restore it. The gauge is a diagnostic; no code outside the
    with-block ever sees the swap."""
    original = qmap.map_fraction
    qmap.map_fraction = lambda z: fraction_torch(z, z_core)
    try:
        yield
    finally:
        qmap.map_fraction = original


def run_tests(z_core: float) -> dict:
    """Math and numerical checks for one candidate. Returns named booleans."""
    checks: dict = {}
    p = params(z_core)
    zc, slope = p["z_core"], p["slope"]

    grid = np.linspace(-40.0, 40.0, 400001)
    g = fraction_np(grid, z_core)
    checks["monotone"] = bool(np.all(np.diff(g) >= -1e-15))
    checks["bounded_0_1"] = bool(np.all(g >= 0.0) and np.all(g <= 1.0))
    checks["finite"] = bool(np.all(np.isfinite(g)))
    checks["symmetric"] = bool(np.allclose(
        fraction_np(-grid, z_core), 1.0 - g, atol=1e-12, rtol=0.0))
    checks["endpoint_centre"] = bool(
        abs(float(fraction_np(np.array([0.0]), z_core)[0]) - 0.5) < 1e-15)
    checks["endpoint_upper"] = bool(
        abs(float(fraction_np(np.array([zc]), z_core)[0]) - Q_HI) < 1e-12)
    checks["endpoint_lower"] = bool(
        abs(float(fraction_np(np.array([-zc]), z_core)[0]) - Q_LO) < 1e-12)

    # One-sided samples one guard step either side of the joint. The offset
    # itself moves the tail derivative by O(slope * eps), so the tolerance is
    # scaled by the slope rather than absolute.
    d_left = float(derivative_np(np.array([zc - 1e-9]), z_core)[0])
    d_right = float(derivative_np(np.array([zc + 1e-9]), z_core)[0])
    checks["derivative_continuous"] = bool(
        abs(d_left - slope) < 1e-12
        and abs(d_right - d_left) <= 1e-6 * slope)

    zt = torch.tensor([-zc * 3.0, -zc, -0.5, 0.0, 0.5, zc, zc * 3.0],
                      dtype=torch.float64)
    checks["torch_numpy_parity"] = bool(np.allclose(
        fraction_torch(zt, z_core).numpy(), fraction_np(zt.numpy(), z_core),
        atol=1e-14, rtol=0.0))
    checks["derivative_numpy_torch_parity"] = bool(np.allclose(
        derivative_torch(zt, z_core).numpy(),
        derivative_np(zt.numpy(), z_core), atol=1e-14, rtol=0.0))

    zz = zt.clone().requires_grad_(True)
    fraction_torch(zz, z_core).sum().backward()
    checks["autograd_matches_analytic"] = bool(torch.allclose(
        zz.grad, derivative_torch(zt, z_core), atol=1e-12))

    fd_ok = True
    for point in (-20.0, -zc * 2.0, -zc, -1.0, 0.0, 1.0, zc, zc * 2.0, 20.0):
        h = 1e-6
        num = float((fraction_torch(torch.tensor([point + h],
                                                 dtype=torch.float64), z_core)
                     - fraction_torch(torch.tensor([point - h],
                                                   dtype=torch.float64),
                                      z_core)) / (2.0 * h))
        ana = float(derivative_torch(torch.tensor([point],
                                                  dtype=torch.float64),
                                     z_core))
        # The second derivative jumps at the joint, so a central difference
        # there is first-order accurate; elsewhere it is second-order.
        tol = 1e-5 if abs(abs(point) - zc) < 1e-12 else 1e-8
        fd_ok = fd_ok and abs(num - ana) < tol
    checks["finite_difference_matches"] = bool(fd_ok)

    extreme = np.array([-1.0e9, -1.0e6, 1.0e6, 1.0e9])
    ge = fraction_np(extreme, z_core)
    de = derivative_np(extreme, z_core)
    checks["extreme_logits_finite"] = bool(
        np.all(np.isfinite(ge)) and np.all(np.isfinite(de))
        and np.all(ge >= 0.0) and np.all(ge <= 1.0))

    integer_input = derivative_np(np.array([0, 1, 2]), z_core)
    checks["integer_input_not_truncated"] = bool(
        np.all(integer_input > 0.0) and float(integer_input[0]) == slope)

    checks["closed_gate_zero"] = _closed_gate_zero(z_core)
    return {"z_core": zc, "params": p, "checks": checks,
            "all_pass": all(checks.values())}


def _closed_gate_zero(z_core: float) -> bool:
    """The price branch must not move a quote whose gate is shut."""
    with patched(z_core):
        out = qmap.forward_policy_v3(
            torch.tensor([[5.0, 5.0, 8.0, 8.0]], dtype=torch.float64),
            train=False)
        zero = qmap.forward_policy_v3(
            torch.tensor([[-5.0, -5.0, 8.0, 8.0]], dtype=torch.float64),
            train=False)
        positive = qmap.forward_policy_v3(
            torch.tensor([[5.0, 5.0, 8.0, 8.0]], dtype=torch.float64),
            train=False)
    return bool(float(zero["quotes"].sum()) == 0.0
                and float(out["quotes"].sum()) == float(positive["quotes"].sum())
                and float(out["quotes"].sum()) > 0.0)


if __name__ == "__main__":
    for z in CANDIDATE_Z:
        res = run_tests(z)
        failed = [k for k, v in res["checks"].items() if not v]
        print(f"Z={z}: {'ok' if res['all_pass'] else 'FAIL ' + str(failed)}")
        if z == 2.5:
            zz = torch.tensor([-5.0, 0.0, 3.7], dtype=torch.float64)
            assert torch.allclose(fraction_torch(zz, 2.5), qmap.map_fraction(zz))
            assert torch.allclose(derivative_torch(zz, 2.5),
                                  qmap.map_derivative(zz))
