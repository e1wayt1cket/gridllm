# test_checkpoint_obs_compat.py
"""A checkpoint is only loadable by the observation it was trained against.

The recorded spec name is the only thing that says which observation that was.
The input width is not evidence: V3, V4, the unreleased V5 and V6 all take
twelve numbers and mean different things by them, so a twelve-wide file with no
header is exactly the case that must not load quietly.

`policies/checkpoint_obs_compat_manifest.json` classifies every checkpoint in
the repository, and `load_policy` refuses the two classes the current
environment may not take.
"""

from __future__ import annotations

import glob
import json
import pathlib

import pytest
import torch

import rl_td3
from rl_spec import (ACTION_STORAGE_QUOTE_V2, OBS_V3, OBS_V6)

_REPO = pathlib.Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "policies" / "checkpoint_obs_compat_manifest.json"


def _bounds():
    return torch.tensor([[-1.0, -1.0], [1.0, 1.0]])


def _untagged_checkpoint(tmp_path):
    """A 12-dim actor saved with no spec -- as the pre-metadata files are."""
    actor = rl_td3.Actor(OBS_V6.total_dim, ACTION_STORAGE_QUOTE_V2.dim, _bounds())
    path = tmp_path / "untagged.pt"
    rl_td3.save_policy(actor, str(path))          # no obs_spec, no action_spec
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert "_makerb_policy_meta" not in state, "the fixture must be untagged"
    return path


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------

def test_the_manifest_classifies_every_checkpoint_in_the_repository():
    """A checkpoint that is not classified is one nobody has looked at."""
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["current_obs_spec"] == OBS_V6.name

    listed = {e["path"] for e in manifest["checkpoints"]}
    on_disk = set()
    for pattern in ("policies/**/*.pt", "outputs/*.pt"):
        for p in glob.glob(str(_REPO / pattern), recursive=True):
            on_disk.add(pathlib.Path(p).relative_to(_REPO).as_posix())
    assert on_disk <= listed, f"unclassified: {sorted(on_disk - listed)}"
    assert manifest["counts"] == {
        status: sum(1 for e in manifest["checkpoints"]
                    if e["status"] == status)
        for status in manifest["counts"]}


def test_only_a_file_that_names_the_current_spec_is_classified_current():
    """The classification follows the tag in the file, not the width or the age.

    This asserted `counts["current"] == 0`, on the grounds that *every*
    checkpoint in the repository predated V6. That was a statement about the
    repository at one moment rather than about the files, and the S8 smoke test
    made it false by producing the first V6 policies. What replaces it is the
    rule the snapshot stood for: a file reads as `current` **only** when its own
    metadata names the current observation, and every other file is one of the
    two refusals.
    """
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    allowed = {"current", "legacy_obs_unverified",
               f"incompatible_{OBS_V6.name}"}
    for entry in manifest["checkpoints"]:
        assert entry["status"] in allowed, entry

    # Checked against the files, not read back from the manifest: a width may
    # not admit a checkpoint, and neither may the directory it sits in.
    for entry in manifest["checkpoints"]:
        if entry["status"] != "current":
            continue
        state = torch.load(_REPO / entry["path"], map_location="cpu",
                           weights_only=False)
        meta = state.get(rl_td3._POLICY_META_KEY) or {}
        assert (meta.get("obs_spec") or {}).get("name") == OBS_V6.name, \
            entry["path"]

    # The refusals are still the overwhelming majority, and the unverifiable
    # ones still exist -- they are the case the whole rule is for.
    assert manifest["counts"].get("legacy_obs_unverified", 0) > 0
    assert manifest["counts"].get(f"incompatible_{OBS_V6.name}", 0) > 0


def test_the_manifest_records_a_width_for_the_unverified_files():
    """The width is recorded as evidence, which is what it is -- never as a
    verdict. These files are unverified whatever width they happen to be."""
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    unverified = [e for e in manifest["checkpoints"]
                  if e["status"] == "legacy_obs_unverified"]
    assert unverified, "the fixture premise is gone; re-derive this test"
    twelve_wide = [e for e in unverified if e["input_width"] == OBS_V6.total_dim]
    assert twelve_wide, (
        "no twelve-wide unverified file remains, so this test no longer "
        "demonstrates why width is not evidence -- re-derive it")


# ---------------------------------------------------------------------------
# The loader refuses them
# ---------------------------------------------------------------------------

def test_a_twelve_wide_untagged_checkpoint_is_refused(tmp_path):
    """The case the rule exists for: same width, unknown meaning."""
    path = _untagged_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="legacy_obs_unverified"):
        rl_td3.load_policy(str(path), OBS_V6.total_dim, _bounds(),
                           obs_spec=OBS_V6, action_spec=ACTION_STORAGE_QUOTE_V2)


def test_an_untagged_checkpoint_is_refused_by_the_plural_loader_too(tmp_path):
    import rl_bidding

    actor = rl_td3.Actor(OBS_V6.total_dim, ACTION_STORAGE_QUOTE_V2.dim, _bounds())
    path = tmp_path / "multi.pt"
    torch.save({"ESS9": actor.state_dict()}, path)
    with pytest.raises(ValueError, match="legacy_obs_unverified"):
        rl_bidding.load_policies(str(path), OBS_V6.total_dim, _bounds(),
                                 obs_spec=OBS_V6, action_spec=ACTION_STORAGE_QUOTE_V2)


def test_an_untagged_checkpoint_still_loads_when_no_spec_is_claimed(tmp_path):
    """Passing no spec is the caller taking responsibility for the weights,
    which is why the refusal lives behind a supplied spec."""
    path = _untagged_checkpoint(tmp_path)
    net = rl_td3.load_policy(str(path), OBS_V6.total_dim, _bounds())
    assert net is not None


def test_a_checkpoint_tagged_with_an_older_observation_is_refused(tmp_path):
    """Tagged but wrong is a name mismatch; untagged is unverifiable. Both are
    refusals, and the messages must not be the same one."""
    actor = rl_td3.Actor(OBS_V3.total_dim, ACTION_STORAGE_QUOTE_V2.dim, _bounds())
    path = tmp_path / "v3.pt"
    rl_td3.save_policy(actor, str(path), obs_spec=OBS_V3,
                       action_spec=ACTION_STORAGE_QUOTE_V2)
    with pytest.raises(ValueError) as excinfo:
        rl_td3.load_policy(str(path), OBS_V6.total_dim, _bounds(),
                           obs_spec=OBS_V6, action_spec=ACTION_STORAGE_QUOTE_V2)
    message = str(excinfo.value)
    assert OBS_V3.name in message and OBS_V6.name in message
    assert "legacy_obs_unverified" not in message


def test_shape_alone_never_admits_a_checkpoint(tmp_path):
    """Two files of identical width, opposite verdicts: one tagged V6 loads,
    the untagged one does not."""
    tagged = tmp_path / "tagged.pt"
    rl_td3.save_policy(rl_td3.Actor(OBS_V6.total_dim, ACTION_STORAGE_QUOTE_V2.dim, _bounds()),
                       str(tagged), obs_spec=OBS_V6,
                       action_spec=ACTION_STORAGE_QUOTE_V2)
    untagged = _untagged_checkpoint(tmp_path)

    sizes = {
        torch.load(p, map_location="cpu", weights_only=False)["net.0.weight"]
        .shape[1] for p in (tagged, untagged)}
    assert sizes == {OBS_V6.total_dim}, "the two files must share a width"

    assert rl_td3.load_policy(str(tagged), OBS_V6.total_dim, _bounds(),
                              obs_spec=OBS_V6,
                              action_spec=ACTION_STORAGE_QUOTE_V2) is not None
    with pytest.raises(ValueError):
        rl_td3.load_policy(str(untagged), OBS_V6.total_dim, _bounds(),
                           obs_spec=OBS_V6, action_spec=ACTION_STORAGE_QUOTE_V2)
