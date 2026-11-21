# test_gate_interface_audit.py
"""The adapter contract checks, independent of the RL environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_truthful_gate_interface as audit


def test_the_interface_contract_passes():
    payload = audit.run()
    assert payload["all_pass"], payload["checks"]
