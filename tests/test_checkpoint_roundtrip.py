# test_checkpoint_roundtrip.py
"""The writer/loader round trip on tiny modules. No training, no env."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_checkpoint_roundtrip as audit


def test_roundtrip_reproduces_everything(tmp_path):
    payload = audit.run(tmp_path / "run")
    assert payload["all_pass"], payload["checks"]
    assert payload["checks"]["incomplete_rejected"] is True
