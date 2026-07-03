# dispatch.py
"""Unified OPF dispatch entry point and public API re-exports."""

# Re-export public API for backward compatibility
from dispatch_core import StorageConstraints  # noqa: F401
from dispatch_ldf import solve_lindist_opf_batch  # noqa: F401
from dispatch_socp import solve_socp_opf_batch  # noqa: F401


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
        raise RuntimeError("Gurobi unavailable and no HiGHS fallback for LinDistFlow")
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
