import numpy as np

from local_da_audit import compare_stacks, eval_counts, rolling_gap


def test_sign_and_gap_bounds_are_computed_on_the_cells_above_the_floor():
    training = np.array([[0.5, -2.0], [100.0, 300.0]])
    audit = np.array([[0.4, -2.5], [110.0, 250.0]])
    out = compare_stacks(training, audit, floor=1.0)
    # The 0.4 cell falls under the floor; the other three agree in sign and
    # their relative gaps are 0.2, 0.0909..., 0.2.
    assert out["cells"] == 3
    assert out["sign_agreement"] == 1.0
    assert abs(out["median_rel_gap"] - 0.2) < 1e-9


def test_a_sign_flip_drops_the_agreement_below_one():
    out = compare_stacks(np.array([[100.0]]), np.array([[-100.0]]))
    assert out["sign_agreement"] == 0.0


def test_the_rolling_gap_reports_median_and_p95():
    day = np.array([100.0, 100.0, 100.0, 100.0])
    blocks = np.array([110.0, 80.0])     # two blocks, each committing two periods
    out = rolling_gap(day, blocks, committed=2)
    # gaps are 0.1, 0.1, 0.2, 0.2
    assert abs(out["median"] - 0.15) < 1e-9
    assert abs(out["p95"] - 0.2) < 1e-9


def test_the_eval_counts_name_the_infeasible_day():
    out = eval_counts(["2022-08-01", "2022-08-02"],
                      {"2022-08-01": {"settled": True},
                       "2022-08-02": {"settled": False}})
    assert out["attempted"] == 2 and out["evaluated"] == 1
    assert out["infeasible"] == ["2022-08-02"]
