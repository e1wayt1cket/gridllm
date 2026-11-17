"""C0 pilot entry: reward_mode ablation without touching frozen files.

NON_FORMAL_DIAGNOSTIC_PILOT. `train_da_gate.py` hard-codes the
`DAExperimentConfig` reward_mode positional as "B_C2" (train_da_gate.py:641)
and exposes no CLI for it. This wrapper is the review-approved isolation:
frozen files keep their hashes; the control arm calls the frozen entry
verbatim; the B arm installs an explicit, logged constructor patch that
rewrites only the reward_mode positional.

Usage:
    python src/train_c0_pilot.py --arm B_C2 --seed 11 \
        --exposures 15 --min-rounds 4 --max-rounds 4 --max-episodes 212 \
        --early-stop-patience 99 --run-id C0PILOT_BC2_seed11 --trace-blocks
"""

from __future__ import annotations

import sys

import rl_da_gate as mod
import train_da_gate as T

ARM_FLAG = "--arm"


def extract_arm(argv: list) -> tuple:
    if ARM_FLAG not in argv:
        raise SystemExit(f"missing {ARM_FLAG} {{B_C2,B}}")
    i = argv.index(ARM_FLAG)
    arm = argv[i + 1]
    if arm not in ("B_C2", "B"):
        raise SystemExit(f"{ARM_FLAG} must be B_C2 or B, got {arm!r}")
    return arm, argv[:i] + argv[i + 2:]


def main() -> int:
    arm, passthrough = extract_arm(sys.argv[1:])
    # Review condition 1 (2026-11-10): --split must be explicit; the first
    # C0 pilot batch silently fell back to the `near` default and trained on
    # a different day partition than the formal runs. Never allow that again.
    if "--split" not in passthrough:
        raise SystemExit(
            "C0PILOT refuses to run without an explicit --split "
            "(near|common); the default burned the first batch")
    if arm == "B":
        original = mod.DAExperimentConfig

        def patched(*args, **kwargs):
            if len(args) < 2 or args[1] != "B_C2":
                raise RuntimeError(
                    "C0 pilot patch expected reward_mode 'B_C2' at "
                    f"positional 2, got {args[1:]!r}")
            print("C0PILOT patch active: reward_mode 'B_C2' -> 'B'; "
                  "all other constructor arguments pass through verbatim",
                  flush=True)
            return original(args[0], "B", *args[2:], **kwargs)

        mod.DAExperimentConfig = patched
    else:
        print("C0PILOT control arm: frozen entry called verbatim "
              "(reward_mode stays B_C2)", flush=True)
    sys.argv = [sys.argv[0], *passthrough]
    return T.main()


if __name__ == "__main__":
    sys.exit(main())
