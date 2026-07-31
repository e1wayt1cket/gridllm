# dispatch.py
"""Unified OPF dispatch entry point and public API re-exports.

Solver availability:
  - Gurobi: required for LinDistFlow and SOCP modes
  - HiGHS (via ortools): automatic fallback for DC-OPF mode only

To install Gurobi: https://www.gurobi.com/downloads/
Free academic licenses available.
"""

# Re-export public API for backward compatibility
from dispatch_core import StorageConstraints  # noqa: F401
from dispatch_ldf import solve_lindist_opf_batch  # noqa: F401
from dispatch_socp import solve_socp_opf_batch  # noqa: F401

_GUROBI_MISSING_MSG = (
    "Gurobi is required for opf_mode='{mode}'. "
    "Use opf_mode='dc' for the built-in HiGHS fallback, "
    "or install Gurobi from https://www.gurobi.com/downloads/"
)


def solve_opf_gurobi(net, agents, t, stage, prev_soc, wholesale_t,
                     action_params, config):
    """Single-period OPF dispatcher: routes to DC-OPF or LinDistFlow by config."""
    from dispatch_dc import solve_dc_opf_gurobi, _solve_dc_opf_highs
    from dispatch_ldf import solve_lindist_opf_gurobi

    try:
        import gurobipy as gp  # noqa: F401
        _has_gurobi = True
    except ImportError:
        _has_gurobi = False

    if not _has_gurobi:
        if config.opf_mode == "dc":
            return _solve_dc_opf_highs(net, agents, t, stage, prev_soc,
                                       wholesale_t, action_params, config)
        raise RuntimeError(_GUROBI_MISSING_MSG.format(mode=config.opf_mode))
    if config.opf_mode == "dc":
        return solve_dc_opf_gurobi(net, agents, t, stage, prev_soc,
                                   wholesale_t, action_params, config)
    elif config.opf_mode == "lindistflow":
        return solve_lindist_opf_gurobi(net, agents, t, stage, prev_soc,
                                        wholesale_t, action_params, config)
    elif config.opf_mode == "socp":
        raise RuntimeError("SOCP OPF requires batch solver; use opf_mode='lindistflow' or 'dc' for single-period")
    else:
        raise ValueError(f"Unknown OPF mode: {config.opf_mode}")
