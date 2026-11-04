# test_b1_daily_baseline.py
"""The rolling B1 table: completeness, hashes, and the NEAR mapping."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "processed" / "v2_cy_jn_2025" / "pilot"
RESULTS = PILOT / "b1_daily_results.csv"
MANIFEST = PILOT / "b1_daily_manifest.json"
pytestmark = pytest.mark.skipif(not RESULTS.exists(),
                                reason="rolling B1 not built")


def _read(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_the_rolling_b1_covers_all_days():
    rows = _read(RESULTS)
    assert len(rows) == 96
    assert len({r["pair_id"] for r in rows}) == 96
    assert all(int(float(r["blocks"])) == 24 for r in rows)
    assert all(r["all_optimal"] == "True" for r in rows)
    assert all(int(float(r["lmp_fallbacks"])) == 0 for r in rows)
    for r in rows:
        total = sum(float(r[f"G_{u}"]) for u in
                    ("ESS9", "ESS14", "ESS17", "ESS27"))
        assert total == pytest.approx(float(r["G_total"]), abs=1e-6)


def test_the_manifest_records_it_is_not_jn_base():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["not_jn_base"] is True
    assert manifest["soc0"] == 0.5
    digest = hashlib.sha256(RESULTS.read_bytes()).hexdigest()
    assert manifest["outputs"]["results"]["sha256"] == digest
    assert len(manifest["representative_comparison"]) >= 5


def test_every_near_day_has_a_b1_row_by_pair_id():
    near = _read(ROOT / "data/processed/v2_cy_jn_2025/near/load_price_pairing.csv")
    b1 = {r["pair_id"] for r in _read(RESULTS)}
    near_ids = {f"{r['load_date']}|{r['price_date']}" for r in near}
    assert len(near_ids) == 80
    assert near_ids <= b1
