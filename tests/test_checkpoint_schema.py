# test_checkpoint_schema.py
"""The checkpoint schema validator accepts valid bundles and catches gaps."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import validate_checkpoint_schema as v


def test_the_three_levels_validate():
    for level in v.LEVELS:
        assert v.validate(v._sample(level)) == [], level


def test_an_inference_checkpoint_needs_no_buffer():
    assert v.validate(v._sample("inference")) == []


def test_a_resume_checkpoint_without_critics_is_refused():
    bundle = copy.deepcopy(v._sample("resume"))
    bundle["networks"].pop("critics")
    assert v.validate(bundle)


def test_an_unsaved_buffer_must_state_why():
    bundle = copy.deepcopy(v._sample("resume"))
    bundle["buffer"] = {"saved": False}
    assert v.validate(bundle)
    bundle["buffer"]["reason"] = "too large to store"
    assert v.validate(bundle) == []


def test_a_malformed_hash_is_refused():
    bundle = copy.deepcopy(v._sample("bundle"))
    bundle["data"]["pairing"] = "not-a-hash"
    assert v.validate(bundle)


def test_a_bundle_requires_data_and_provenance():
    bundle = copy.deepcopy(v._sample("bundle"))
    bundle.pop("data")
    bundle.pop("eval_provenance")
    errors = v.validate(bundle)
    assert any("data" in e for e in errors)
    assert any("eval_provenance" in e for e in errors)
