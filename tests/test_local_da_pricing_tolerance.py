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


def test_the_cone_barrier_tolerance_is_per_model():
    # Exclusivity is a disjunction, not a bilinear product, so neither model
    # carries a quadratic apart from the cones. The second gateway tranche
    # doubled the residual the barrier leaves behind, and the exact model at
    # 1e-8 puts a sampled day past the audit's 1e-5 bound (1.04e-5, 8.6e-7 at
    # 1e-9). The continuous pricing model cannot certify 1e-9 at all, so it
    # keeps the looser barrier.
    day, _ = build_day("2021-06-04")
    exact = ClearingModel(day, exact=True)
    convex = ClearingModel(day, exact=False)
    try:
        assert exact.model.Params.BarQCPConvTol == 1e-9
        assert convex.model.Params.BarQCPConvTol == 1e-8
    finally:
        exact.close()
        convex.close()
