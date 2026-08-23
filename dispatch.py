# dispatch.py
"""Unified OPF dispatch entry point and public API re-exports.

Clearing engines are registered per opf_mode in CLEARING_MECHANISMS (mirroring
the strategy registry in strategies/__init__.py and the forecast registry in
price_forecaster.py). Adding a new OPF mode is a one-line registration.

Solver availability:
  - Gurobi: required for LinDistFlow and SOCP modes
  - HiGHS (via ortools): automatic fallback for DC-OPF mode only

To install Gurobi: https://www.gurobi.com/downloads/
Free academic licenses available.
"""

# Re-export public API for backward compatibility
from dispatch_core import StorageConstraints  # noqa: F401
from dispatch_ldf import solve_lindist_opf_batch  # noqa: F401
from dispatch_dc import solve_dc_opf_gurobi
from dispatch_ldf import solve_lindist_opf_gurobi
from dispatch_socp import solve_socp_opf_batch

_GUROBI_MISSING_MSG = (
    "Gurobi is required for opf_mode='{mode}'. "
    "Use opf_mode='dc' for the built-in HiGHS fallback, "
    "or install Gurobi from https://www.gurobi.com/downloads/"
)

# ---------------------------------------------------------------------------
# Clearing mechanism registry
# ---------------------------------------------------------------------------

CLEARING_MECHANISMS: dict = {}


def register_clearing_mechanism(mode, *, batch_fn=None, single_period_fn=None,
                                requires_gurobi=True, description=""):
    """Register an OPF clearing engine under an opf_mode key.

    batch_fn solves a multi-period joint optimization (returning the
    clear_market result schema); single_period_fn solves one period (used by
    the fallback loop and by market.py's per-period path). A mode may supply
    either or both.
    """
    if mode in CLEARING_MECHANISMS:
        raise ValueError(f"Clearing mechanism '{mode}' already registered")
    entry = {"mode": mode, "batch_fn": batch_fn,
             "single_period_fn": single_period_fn,
             "requires_gurobi": requires_gurobi, "description": description}
    CLEARING_MECHANISMS[mode] = entry
    return entry


def get_clearing_mechanism(mode):
    """Look up an OPF clearing engine by opf_mode."""
    if mode not in CLEARING_MECHANISMS:
        raise ValueError(f"Unknown OPF mode '{mode}'. "
                         f"Available: {list(CLEARING_MECHANISMS)}")
    return CLEARING_MECHANISMS[mode]


register_clearing_mechanism(
    "dc",
    single_period_fn=solve_dc_opf_gurobi,
    requires_gurobi=False,
    description="DC-OPF; Gurobi with HiGHS fallback",
)
register_clearing_mechanism(
    "lindistflow",
    batch_fn=solve_lindist_opf_batch,
    single_period_fn=solve_lindist_opf_gurobi,
    requires_gurobi=True,
    description="LinDistFlow; batch multi-period or single-period",
)
register_clearing_mechanism(
    "socp",
    batch_fn=solve_socp_opf_batch,
    requires_gurobi=True,
    description="SOCP relaxation; batch multi-period only",
)


def solve_opf_gurobi(net, agents, t, stage, prev_soc, wholesale_t,
                     action_params, config):
    """Single-period OPF dispatcher: routes via the clearing registry.

    DC and LinDistFlow are supported as single-period solvers; SOCP requires
    the batch solver (see clear_market). The Gurobi/ HiGHS choice for DC-OPF
    is handled inside the registered solver.
    """
    mech = get_clearing_mechanism(config.opf_mode)
    if mech["single_period_fn"] is None:
        raise RuntimeError(
            f"OPF mode '{config.opf_mode}' requires the batch solver; "
            "use clear_market() with a batch-capable mode.")
    return mech["single_period_fn"](net, agents, t, stage, prev_soc,
                                    wholesale_t, action_params, config)
