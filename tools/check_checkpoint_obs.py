# check_checkpoint_obs.py
"""Classify every saved policy by the observation it was trained against.

A checkpoint is only usable by the environment whose observation it was trained
in, and the recorded spec name is the only thing that says which that was. The
input width does not: V3, V4, the unreleased V5 and V6 all take twelve numbers
and mean different things by them, and several older encodings take other
widths.

So each file gets one of three statuses:

| status | what it is | may the current environment load it? |
|---|---|---|
| `current` | metadata names the current spec | yes |
| `incompatible_<current>` | metadata names a different spec | no -- refused by name mismatch |
| `legacy_obs_unverified` | no metadata at all | no -- refused, because nothing in the file says |

The third row is the one that needed making explicit. Those files used to load
unchanged, and the ones that happen to be twelve wide would have been read as
whatever the environment currently builds.

Writes `policies/checkpoint_obs_compat_manifest.json` and prints a summary.
Read-only with respect to the checkpoints themselves: nothing here rewrites,
moves or deletes a `.pt`.

Usage:
    PYTHONPATH=src python src/check_checkpoint_obs.py
    PYTHONPATH=src python src/check_checkpoint_obs.py --check   # no write
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from rl_spec import OBS_V6  # noqa: E402

#: Where the manifest goes. Beside the checkpoints, because that is where
#: somebody wondering about a `.pt` will be.
MANIFEST_PATH = os.path.join("policies", "checkpoint_obs_compat_manifest.json")

#: What a file with no metadata is called.
UNVERIFIED = "legacy_obs_unverified"

#: What each status means, written once instead of on every entry. Two thousand
#: repetitions of the same sentence is bulk, not information.
STATUS_MEANING = {
    "current": "trained against the spec the environment builds now",
    UNVERIFIED: ("no observation metadata in the file, so the observation it "
                 "was trained against is unknown; the input width is not "
                 "evidence"),
    "unreadable": "not a readable policy file",
}

_PATTERNS = ("policies/**/*.pt", "outputs/*.pt")


def _actor_input_width(state: dict):
    """The observation width the weights were built for, or None.

    Read off the first layer rather than the recorded `obs_dim`, so a file with
    no metadata still reports a width -- as evidence, not as a verdict.
    """
    for key, value in state.items():
        if not hasattr(value, "shape") or len(value.shape) != 2:
            continue
        if "net.0" in key or key.endswith("weight"):
            return int(value.shape[1])
    return None


def classify(path: str) -> dict:
    """One checkpoint's status, and the evidence for it."""
    record = {"path": path.replace(os.sep, "/")}
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # a file that is not a checkpoint at all
        record["status"] = "unreadable"
        return record
    if not isinstance(state, dict):
        record["status"] = "unreadable"
        return record

    meta = state.get("_makerb_policy_meta") or state.get("_meta")
    record["input_width"] = _actor_input_width(
        {k: v for k, v in state.items() if hasattr(v, "shape")})

    if meta is None:
        record["status"] = UNVERIFIED
        return record

    obs = (meta or {}).get("obs_spec") or {}
    record["recorded_obs_spec"] = obs.get("name")
    record["recorded_obs_version"] = obs.get("version")
    if obs.get("name") == OBS_V6.name:
        record["status"] = "current"
    else:
        record["status"] = f"incompatible_{OBS_V6.name}"
    return record


def scan() -> dict:
    paths = []
    for pattern in _PATTERNS:
        paths.extend(glob.glob(pattern, recursive=True))
    records = [classify(p) for p in sorted(set(paths))]
    counts = collections.Counter(r["status"] for r in records)
    return {
        "manifest_version": "checkpoint_obs_compat_v1",
        "current_obs_spec": OBS_V6.name,
        "current_obs_version": OBS_V6.version,
        "note": ("A checkpoint may only be loaded by an environment whose "
                 "observation spec it records. The input width is not "
                 "evidence: several different observations share a width."),
        "status_meaning": {**STATUS_MEANING,
                           f"incompatible_{OBS_V6.name}":
                               ("trained against an observation this "
                                "environment does not build; refused by the "
                                "recorded name, not by the width")},
        "counts": dict(sorted(counts.items())),
        "checkpoints": records,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report only; do not write the manifest")
    parser.add_argument("--out", default=MANIFEST_PATH)
    args = parser.parse_args(argv)

    manifest = scan()
    for status, n in manifest["counts"].items():
        print(f"{n:6d}  {status}")
    if not args.check:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
