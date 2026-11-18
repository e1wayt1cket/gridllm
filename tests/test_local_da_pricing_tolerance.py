from local_da_clearing import ClearingModel, clear_day
from run_local_da import build_day


def test_the_conditional_price_solve_is_optimal_on_a_library_day():
    day, _ = build_day("2021-09-02")
    result = clear_day(day, seconds=120, price_probes=False)
    assert result["dispatch"]["optimal"] is True
    assert result["pricing"]["optimal"] is True
    assert result["pricing"]["solver_status"] == 2
    assert result["conditional_objective_difference_cny"] <= max(
        .01, abs(result["dispatch"]["clearing_objective"]) * 1e-6)


def test_the_exact_model_keeps_the_tight_barrier_tolerance():
    day, _ = build_day("2021-06-04")
    exact = ClearingModel(day, exact=True)
    convex = ClearingModel(day, exact=False)
    try:
        assert exact.model.Params.BarQCPConvTol == 1e-10
        assert convex.model.Params.BarQCPConvTol == 1e-8
    finally:
        exact.close()
        convex.close()
