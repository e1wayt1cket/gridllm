# agent_seed.py
"""Stable RNG seeds for named participants.

`hash()` on a string is salted per process, so seeding a draw from
`hash(agent.name)` made the same configuration produce a different draw in
every process: a run could not be reproduced from its own record, and two runs
that should have been identical disagreed. The seed below is a digest of the
name, so it is a function of the participant's identity and of nothing else --
not of the process, and not of the order the participants were visited in.
"""

import hashlib

__all__ = ["seed_for_agent"]

_SEED_MODULUS = 2 ** 31


def seed_for_agent(name: str, seed: int = 0) -> int:
    """A seed for `name`, repeatable across processes.

    `seed` folds an experiment-level seed into the draw without renaming
    anyone. It is an argument rather than a configuration knob because the
    callers that draw per participant do not all read a configuration, and a
    knob nothing sets would claim a control that does not exist.
    """
    digest = hashlib.blake2b(f"{seed}|{name}".encode("utf-8"),
                             digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS
