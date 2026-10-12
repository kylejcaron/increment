import numpy as np

from calibration.ratio_skew import Cell, choose_cutoff, run_cell


def test_ratio_skew_cell_stops_before_starting_work_after_deadline():
    assert run_cell(50, 1.5, 10_000, seed=1, deadline=0.0) is None


def test_cutoff_does_not_join_occupied_bins_across_an_empty_bin():
    cell = Cell(
        n=50,
        sigma=1.0,
        coverage=0.0,
        evaluated=2,
        refused=0,
        stats=np.array([0.01, 0.11]),
        covered=np.array([False, False]),
    )
    assert choose_cutoff([cell]) is None
