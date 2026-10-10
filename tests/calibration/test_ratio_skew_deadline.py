from calibration.ratio_skew import run_cell


def test_ratio_skew_cell_stops_before_starting_work_after_deadline():
    assert run_cell(50, 1.5, 10_000, seed=1, deadline=0.0) is None
