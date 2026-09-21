# legacy_schema.py
"""Reading artifacts that were written under names the project has since renamed.

A rename cannot rewrite what is already on disk, and the artifacts already on
disk are superseded rather than discarded -- so the read side translates the old
names instead of failing on them or letting two names for one quantity coexist.
New output carries only the canonical name.

`welfare` is the case that prompted this. It was the clearing model's own
objective value and never a social-welfare figure: it prices each storage unit
at that unit's *declared* bid and offer, while the unit is settled at the nodal
price. Only the second is money the participant keeps. Calling the first one
"welfare" invited reading it as the second.

A file that carries both names is reported rather than resolved. There is no
basis for preferring either value, and silently preferring one would hide that
the file disagrees with itself.

This module is a leaf -- it imports nothing from the project -- so both the
numpy-only metrics layer and the dataframe layer can use it.
"""

import warnings

# Legacy name -> the canonical name that replaced it.
LEGACY_ALIASES = {
    "welfare": "clearing_objective",
    "objective": "clearing_objective",
    "best_welfare": "best_clearing_objective",
    "welfare_baseline": "clearing_objective_baseline",
    "welfare_rl": "clearing_objective_rl",
    "welfare_delta": "clearing_objective_delta",
    "genuine_welfare_delta": "genuine_clearing_objective_delta",
    "welfare_da": "clearing_objective_da",
    "da_welfare": "da_clearing_objective",
    "rt_welfare": "rt_clearing_objective",
    "da_welfare_cny": "da_clearing_objective_cny",
    "rt_welfare_cny": "rt_clearing_objective_cny",
    # Retired key-performance-indicator names, from the pair-comparison CSV.
    # Only names that current output no longer writes belong here. The
    # evaluation CSV's own `baseline_profit` / `rl_profit` / `profit_delta`
    # columns are still written under those spellings and are deliberately
    # NOT aliased: mapping a name that is still in use would rename live data
    # on read, which is the failure this module exists to prevent.
    "benefit": "delta_pi",
    "benefit_rate": "delta_pi_percent",
    "mean_benefit": "mean_delta_pi",
    "mean_benefit_rate": "mean_delta_pi_percent",
    "ai_profit": "pi_ai",
}


def canonical_name(name: str) -> str:
    """The canonical name for a possibly-legacy field name."""
    return LEGACY_ALIASES.get(name, name)


def upgrade_columns(names) -> list:
    """A column list with legacy names mapped to their canonical ones."""
    return [canonical_name(n) for n in names]


def upgrade_row(row: dict) -> dict:
    """A mapping with legacy keys renamed to canonical ones.

    The canonical key wins when both are present. They must agree; when they
    do not, the mismatch is reported and both values are kept reachable, with
    the canonical spelling holding the canonical key. Choosing a winner here
    would turn a file that contradicts itself into a file that quietly looks
    consistent, which is the failure this whole translation exists to avoid.
    """
    out = {}
    for key, value in row.items():
        out[canonical_name(key)] = value
    for legacy, canon in LEGACY_ALIASES.items():
        if legacy not in row or canon not in row or legacy == canon:
            continue
        if row[legacy] != row[canon]:
            warnings.warn(
                f"artifact carries both {legacy!r} and {canon!r} with "
                f"different values ({row[legacy]!r} vs {row[canon]!r}); "
                f"keeping {canon!r}. The file disagrees with itself and one "
                f"of the two is wrong.", stacklevel=2)
    return out


def clearing_objective(row: dict, default=None):
    """The clearing objective from a row, under either spelling."""
    for key in ("clearing_objective", "welfare", "objective"):
        if key in row:
            return row[key]
    return default
