"""Planning for the runtime's exact binomial risk-ratio decision.

A binomial-eligible conversion or retention plan reports the probability that
the unchanged runtime decision (``binomial_rr.p_plus``/``p_minus`` below the
compiled tail allocation) rejects, integrated over the binomial count law at
the analyzed integer counts.
"""

from __future__ import annotations

import json
import math
import weakref
from typing import Any

import numpy as np
import pytest
from scipy.stats import binom

from increment.estimation import binomial_rr
from increment.estimation.arm_contract import ArmPlanningProcedure, RelativeDecisionPolicy
from increment.power import (
    Baseline,
    PowerDesign,
    PowerResult,
    achieved_power,
    minimum_detectable_effect,
    power_curve,
    required_sample_size,
)
from increment.power._binomial import BinomialDecision, RejectionGeometry
from tests.power._procedures import make_procedure


def _runtime_power(
    n_c: int, n_t: int, p_c: float, p_t: float, procedure: ArmPlanningProcedure
) -> float:
    """Independent enumeration of the unchanged runtime decision over every
    count pair."""
    decision = procedure.decision
    assert isinstance(decision, RelativeDecisionPolicy)
    r0 = 1.0 + decision.null_lift
    beta = binomial_rr.nuisance_beta(procedure.compiled_alpha)
    tail = procedure.compiled_tail_alpha
    total = 0.0
    w_t = binom.pmf(np.arange(n_t + 1), n_t, p_t)
    for x_c, w_c in enumerate(binom.pmf(np.arange(n_c + 1), n_c, p_c)):
        for x_t in range(n_t + 1):
            plus = binomial_rr.p_plus(r0, x_c, n_c, x_t, n_t, beta, tail=tail)
            minus = binomial_rr.p_minus(r0, x_c, n_c, x_t, n_t, beta, tail=tail)
            if decision.alternative == "greater":
                rejects = plus < tail
            elif decision.alternative == "less":
                rejects = minus < tail
            else:
                rejects = min(plus, minus) < tail
            total += w_c * w_t[x_t] * rejects
    return total


def _conversion(**overrides: Any) -> ArmPlanningProcedure:
    return make_procedure(
        metric_type="conversion",
        identification="randomized",
        population="assigned",
        variance_adjustment="none",
        **overrides,
    )


class TestExactRouteMatchesRuntime:
    """(a) The exact route integrates the runtime's own decisions."""

    @pytest.mark.parametrize(
        ("n_t", "allocation", "p_c", "lift", "overrides"),
        [
            (15, 0.6, 0.3, 0.8, {}),
            (12, 0.5, 0.25, 0.9, {"alternative": "greater", "null_lift": 0.2, "alpha": 0.01}),
            (10, 0.4, 0.5, -0.6, {"alternative": "less", "null_lift": -0.2}),
        ],
    )
    def test_power_equals_enumerated_runtime_rejections(
        self, n_t, allocation, p_c, lift, overrides
    ):
        procedure = _conversion(**overrides)
        design = PowerDesign(allocation=allocation)
        result = achieved_power(n_t, lift, Baseline.from_proportion(p_c), procedure, design)
        n_c = result.n_total - result.n_per_arm
        expected = _runtime_power(n_c, n_t, p_c, p_c * (1.0 + lift), procedure)
        assert result.power_basis == "exact"
        assert result.power == pytest.approx(expected, abs=1e-11)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("n_c", "n_t", "null_ratio", "alpha", "tail", "alternative"),
        [
            (20, 30, 1.0, 0.05, 0.025, "two-sided"),
            (40, 60, 1.2, 0.01, 0.01, "greater"),
            (60, 40, 0.8, 0.05, 0.05, "less"),
        ],
    )
    def test_every_decision_equals_the_runtime(
        self, n_c, n_t, null_ratio, alpha, tail, alternative
    ):
        beta = binomial_rr.nuisance_beta(alpha)
        decision = BinomialDecision(n_c, n_t, null_ratio, beta, tail, alternative)
        plus_cells, minus_cells = RejectionGeometry(decision, "exact").cells(0, n_c, 0, n_t)
        for x_c in range(n_c + 1):
            for x_t in range(n_t + 1):
                plus = binomial_rr.p_plus(null_ratio, x_c, n_c, x_t, n_t, beta, tail=tail) < tail
                minus = binomial_rr.p_minus(null_ratio, x_c, n_c, x_t, n_t, beta, tail=tail) < tail
                assert plus_cells[x_c, x_t] == ("plus" in decision.kinds and plus)
                assert minus_cells[x_c, x_t] == ("minus" in decision.kinds and minus)

    @pytest.mark.parametrize(
        ("n_c", "n_t", "null_ratio", "tail", "alternative", "controls"),
        [
            (300, 300, 1.0, 0.025, "two-sided", (12, 30, 75)),
            (400, 250, 1.3, 0.05, "greater", (40, 80)),
            (350, 350, 0.7, 0.05, "less", (3, 90)),
        ],
    )
    def test_decisions_with_a_windowed_control_support_equal_the_runtime(
        self, n_c, n_t, null_ratio, tail, alternative, controls
    ):
        """Above the enumeration threshold the runtime sums only a Chernoff window of control
        counts and the replay sums that same window: the counts on each side of every
        rejection boundary, and the ends of each row, are decided as the runtime decides them."""
        beta = binomial_rr.nuisance_beta(0.05)
        decision = BinomialDecision(n_c, n_t, null_ratio, beta, tail, alternative)
        first, last = controls[0], controls[-1]
        cells = RejectionGeometry(decision, "exact").cells(first, last, 0, n_t)
        for kind, mask in zip(("plus", "minus"), cells, strict=True):
            runtime = binomial_rr.p_plus if kind == "plus" else binomial_rr.p_minus
            for x_c in controls:
                row = mask[x_c - first]
                steps = np.flatnonzero(np.diff(row.astype(int)))
                probes = {0, n_t, *steps.tolist(), *(steps + 1).tolist()}
                for x_t in sorted(probes):
                    decided = (
                        kind in decision.kinds
                        and runtime(null_ratio, x_c, n_c, x_t, n_t, beta, tail=tail) < tail
                    )
                    assert row[x_t] == decided, (kind, x_c, x_t)


class TestConversionExample:
    """(b) At 701 per arm, 10% baseline and a 50% lift the Wald model
    reported 0.8004; the runtime decision rejects with probability 0.7144."""

    _BASELINE = Baseline.from_proportion(0.1)
    _PROCEDURE = ArmPlanningProcedure.standard("conversion")

    @pytest.mark.slow
    def test_achieved_power_is_the_runtime_rejection_probability(self):
        result = achieved_power(701, 0.5, self._BASELINE, self._PROCEDURE)
        assert result.power_basis == "exact"
        assert result.power == pytest.approx(0.714423480362035, abs=1e-11)
        # The companion effect reaches the target at the runtime decision.
        assert result.mde_relative is not None
        at_mde = achieved_power(701, result.mde_relative, self._BASELINE, self._PROCEDURE)
        assert at_mde.power >= PowerDesign().power

    @pytest.mark.slow
    def test_required_sample_size_reaches_target_at_its_integer_size(self):
        sized = required_sample_size(0.5, self._BASELINE, self._PROCEDURE)
        assert sized.power_basis == "exact"
        assert sized.n_per_arm == 831
        assert sized.power >= 0.8
        assert sized.power == pytest.approx(0.8001957842538467, abs=1e-11)
        smaller = achieved_power(sized.n_per_arm - 1, 0.5, self._BASELINE, self._PROCEDURE)
        assert smaller.power < 0.8


def test_triggered_sizing_agrees_with_achieved_power_at_its_assigned_size():
    """The size search runs over assigned units, so the returned size and
    its assigned predecessor are judged at the analyzed counts
    ``achieved_power`` uses."""
    baseline = Baseline(mean=0.3, var=0.21, trigger_rate=0.35)
    procedure = ArmPlanningProcedure.standard("conversion")
    sized = required_sample_size(0.8, baseline, procedure)
    replay = achieved_power(sized.n_per_arm, 0.8, baseline, procedure)
    assert (replay.power, replay.power_basis) == (sized.power, sized.power_basis)
    assert replay.n_triggered_per_arm == sized.n_triggered_per_arm
    assert sized.power_basis == "exact"
    assert sized.power >= PowerDesign().power
    assert achieved_power(sized.n_per_arm - 1, 0.8, baseline, procedure).power < 0.8


def test_sizing_is_refused_only_by_the_selected_route(monkeypatch):
    """At a (constructed) arm ceiling of 150 treatment units the approximate
    replay stays below target while the exact decision reaches it: the
    approximate proposal must not refuse a design the exact route can size."""
    from increment.power import core

    procedure = _conversion(alternative="greater", null_lift=0.2, alpha=0.01)
    baseline = Baseline.from_proportion(0.1)
    exact = achieved_power(150, 2.5, baseline, procedure, PowerDesign(allocation=0.6667))
    n_c = exact.n_total - exact.n_per_arm
    decision = BinomialDecision(n_c, 150, 1.2, binomial_rr.nuisance_beta(0.01), 0.01, "greater")
    approximate = RejectionGeometry(decision, "approximate").evaluate(0.1, 0.35).power
    target = (approximate + exact.power) / 2.0
    assert exact.power_basis == "exact" and approximate < target < exact.power
    design = PowerDesign(power=target, allocation=0.6667)

    monkeypatch.setattr(core, "_binomial_arm_ceiling", lambda design, floor, baseline: 150)
    sized = required_sample_size(2.5, baseline, procedure, design)
    assert sized.power_basis == "exact"
    assert sized.n_per_arm <= 150
    assert sized.power >= target
    assert achieved_power(sized.n_per_arm, 2.5, baseline, procedure, design).power == sized.power
    assert achieved_power(sized.n_per_arm - 1, 2.5, baseline, procedure, design).power < target


@pytest.mark.parametrize(
    ("reject_margin", "inferred"),
    [(2.0 * 5e-11, False), (math.nextafter(2.0 * 5e-11, 1.0), True)],
)
def test_rejection_is_inferred_only_beyond_twice_the_rounding(monkeypatch, reject_margin, inferred):
    """The replay rejects only at a strictly positive margin, so a neighbour's
    rejection margin of exactly twice the rounding room certifies nothing;
    one representable step above it does."""
    from increment.power import _binomial

    assert _binomial._ROOT_ROUNDING == 5e-11

    def margins(decision, groups, g, j):
        return np.full(j.size, -1.0), np.full(j.size, reject_margin)

    monkeypatch.setattr(_binomial, "_root_margins", margins)
    decision = BinomialDecision(50, 50, 1.0, binomial_rr.nuisance_beta(0.05), 0.05, "greater")
    ints = np.array([0])
    groups = _binomial._Groups(
        kind=ints,
        x_c=np.array([5]),
        a=np.array([0.1]),
        hi=np.array([0.2]),
        wlo=ints,
        width=np.array([51]),
        omitted=np.zeros(1),
        margin=np.zeros(1),
        j0=np.array([0]),
        j1=np.array([20]),
        dmin=ints,
        dmax=ints,
        offsets=np.zeros((1, 1), np.int64),
    )
    j = np.arange(21)
    settled, rejected = _binomial._root_settled(decision, groups, np.zeros(21, np.int64), j)
    assert settled.tolist() == [inferred] * 21
    assert rejected.tolist() == [inferred] * 21


class TestApproximateRoute:
    """(c) The Normal-tail replay stays within the errors the feasibility
    study measured against the runtime on its retained witness rows."""

    # ((n_c, n_t), (p_c, p_t), alpha, null ratio, alternative, runtime power,
    # the replay's power measured in the feasibility study).
    _WITNESSES = (
        ((20, 20), (0.05, 0.25), 0.05, 1.0, "two-sided", 0.123815805538, 0.123815805538),
        ((40, 60), (0.1, 0.25), 0.01, 1.2, "greater", 0.014667364180, 0.012439421369),
        ((60, 40), (0.3, 0.1), 0.05, 0.8, "less", 0.083402470113, 0.083402470113),
        ((25, 25), (0.8, 0.95), 0.05, 1.0, "two-sided", 0.051897357556, 0.051897357556),
        ((100, 100), (0.01, 0.15), 0.05, 1.0, "two-sided", 0.816115665003, 0.816115665003),
        ((75, 150), (0.1, 0.35), 0.01, 1.2, "greater", 0.625496290137, 0.617596569857),
        ((100, 100), (0.5, 0.75), 0.05, 1.0, "two-sided", 0.943014903387, 0.943014903387),
        ((20, 30), (0.4, 0.95), 0.05, 2.0, "greater", 0.085496937736, 0.085496937736),
    )

    @pytest.mark.parametrize("witness", _WITNESSES)
    def test_power_error_within_measured_error(self, witness):
        (n_c, n_t), (p_c, p_t), alpha, ratio, alternative, runtime, measured = witness
        tail = alpha / 2.0 if alternative == "two-sided" else alpha
        decision = BinomialDecision(
            n_c, n_t, ratio, binomial_rr.nuisance_beta(alpha), tail, alternative
        )
        geometry = RejectionGeometry(decision, "approximate")
        approximate = geometry.evaluate(p_c, p_t).power
        assert geometry.route == "approximate"
        # Twelve printed digits plus the windows' omitted mass.
        assert abs(approximate - runtime) <= abs(measured - runtime) + 2e-12

    def test_rare_large_arm_witness_decisions(self):
        """Counts ``(1e6, 4e6, 100, j)``: the runtime rejects from ``j = 510``
        (p+ 0.024564) where the replay rejects only from 512; neither rejects at 509."""
        n_c, n_t = 1_000_000, 4_000_000
        beta = binomial_rr.nuisance_beta(0.05)
        decision = BinomialDecision(n_c, n_t, 1.0, beta, 0.025, "greater")
        plus, _ = RejectionGeometry(decision, "approximate").cells(100, 100, 509, 512)
        runtime = [
            binomial_rr.p_plus(1.0, 100, n_c, j, n_t, beta, tail=0.025) < 0.025
            for j in (509, 510, 511, 512)
        ]
        assert runtime == [False, True, True, True]
        assert plus[0].tolist() == [False, False, False, True]

    def test_plans_beyond_the_cell_budget_are_approximate(self):
        result = achieved_power(2_600, 0.1, Baseline.from_proportion(0.1), _conversion())
        assert result.power_basis == "approximate"
        assert 0.0 < result.power < 1.0

    @pytest.mark.parametrize(
        ("n_c", "n_t", "p", "ratio", "alternative"),
        [
            (300, 300, 0.1, 1.0, "two-sided"),
            (150, 600, 0.3, 0.9, "greater"),
            (400, 400, 0.5, 1.2, "less"),
        ],
    )
    def test_inferred_root_exits_match_replaying_every_count(
        self, monkeypatch, n_c, n_t, p, ratio, alternative
    ):
        """Counts whose root exit is inferred from a neighbour's margin get
        the decision their own replay gives, across whole rows (including
        rows constant at both ends) and the band around each crossing."""
        from increment.power import _binomial

        tail = 0.025 if alternative == "two-sided" else 0.05
        decision = BinomialDecision(
            n_c, n_t, ratio, binomial_rr.nuisance_beta(0.05), tail, alternative
        )
        wc = _binomial._window(n_c, p)
        j_lo = _binomial._window(n_t, ratio * p * 0.6).lo
        j_hi = _binomial._window(n_t, min(1.0, ratio * p * 1.7)).hi
        requests = [
            _binomial._Request(x, kind, j_lo, j_hi)
            for x in range(max(0, wc.lo - 60), min(n_c, wc.hi + 60) + 1)
            for kind in decision.kinds
        ]
        inferred = _binomial.classify(decision, "approximate", requests)

        def replay_everything(decision, groups, group, j):
            return np.zeros(j.size, bool), np.zeros(j.size, bool)

        monkeypatch.setattr(_binomial, "_root_settled", replay_everything)
        replayed = _binomial.classify(decision, "approximate", requests)
        rows = [np.asarray(mask) for mask in replayed]
        assert any(row.all() or not row.any() for row in rows)
        assert any(row.any() and not row.all() for row in rows)
        for got, expected in zip(inferred, rows, strict=True):
            assert np.array_equal(got, expected)


class _LeafMeter:
    """Bytes of `_Leaves` arrays alive at once in the real replay, and the cells it keeps searching.

    Wraps the replay's own allocations (`_Leaves._blank`, `empty`, `take`) and adds nothing to
    what they return: each array is tracked by a weak reference and leaves the total when the
    replay drops it. `take` also counts the one field its fancy indexing gathers beside both
    sets of arrays, which is a temporary the assignment frees at once."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from increment.power import _binomial

        self.live = 0
        self.peak = 0
        self.searching: list[tuple[int, _binomial.Kind, int]] = []
        self._refs: dict[int, Any] = {}
        self._batch: Any = None
        leaves = _binomial._Leaves
        blank, empty, take = leaves._blank, leaves.empty.__func__, leaves.take
        replay = _binomial._replay

        def metered_blank(name, rows, capacity):
            array = blank(name, rows, capacity)
            self._track(array)
            return array

        def metered_empty(cls, rows, capacity):
            created = empty(cls, rows, capacity)
            self._track(created.count)
            return created

        def metered_take(this, rows, *, capacity):
            taken = take(this, rows, capacity=capacity)
            self._track(taken.count)
            gather = max(getattr(this, name).dtype.itemsize for name in leaves._FIELDS)
            self.peak = max(self.peak, self.live + gather * this.u.shape[1] * rows.size)
            batch = self._batch
            for row in rows.tolist():
                group = int(batch.group[row])
                kind = "plus" if batch.groups.kind[group] == 0 else "minus"
                self.searching.append((int(batch.groups.x_c[group]), kind, int(batch.j[row])))
            return taken

        def metered_replay(batch, tails, *, exact):
            self._batch = batch
            return replay(batch, tails, exact=exact)

        monkeypatch.setattr(leaves, "_blank", staticmethod(metered_blank))
        monkeypatch.setattr(leaves, "empty", classmethod(metered_empty))
        monkeypatch.setattr(leaves, "take", metered_take)
        monkeypatch.setattr(_binomial, "_replay", metered_replay)

    def _track(self, array: np.ndarray) -> None:
        key, size = id(array), array.nbytes
        self.live += size
        self.peak = max(self.peak, self.live)

        def release(_ref: Any) -> None:
            self.live -= size
            del self._refs[key]

        self._refs[key] = weakref.ref(array, release)

    def reset(self) -> None:
        self.peak = 0
        self.searching = []


class TestReplayFollowsTheRuntimeStopContract:
    """Planning reads ``binomial_rr.NUISANCE_STOP`` when it replays, so a search the cap ends
    is decided as the runtime decides it, and a longer search never rejects less."""

    _GAP = binomial_rr.NUISANCE_STOP.gap_fraction
    #: Caps that end most searches, end the hard ones, and the shipped contract.
    _RULES = (
        binomial_rr._StopRule(_GAP, 6),
        binomial_rr._StopRule(_GAP, 60),
        binomial_rr.NUISANCE_STOP,
    )
    _DESIGNS = [
        (40, 60, 1.0, 0.025, "two-sided", (3, 9, 14)),
        (50, 50, 1.2, 0.05, "greater", (4, 12)),
        (60, 40, 0.8, 0.05, "less", (6, 18)),
    ]

    @pytest.mark.parametrize(("n_c", "n_t", "ratio", "tail", "alternative", "controls"), _DESIGNS)
    def test_exact_replay_decides_as_the_runtime_under_each_contract(
        self, monkeypatch, n_c, n_t, ratio, tail, alternative, controls
    ):
        beta = binomial_rr.nuisance_beta(0.05)
        decision = BinomialDecision(n_c, n_t, ratio, beta, tail, alternative)
        first, last = controls[0], controls[-1]
        masks = []
        for rule in self._RULES:
            monkeypatch.setattr(binomial_rr, "NUISANCE_STOP", rule)
            cells = RejectionGeometry(decision, "exact").cells(first, last, 0, n_t)
            masks.append(cells)
            for kind, mask in zip(("plus", "minus"), cells, strict=True):
                runtime = binomial_rr.p_plus if kind == "plus" else binomial_rr.p_minus
                for x_c in controls:
                    row = mask[x_c - first]
                    steps = np.flatnonzero(np.diff(row.astype(int)))
                    for x_t in sorted({0, n_t, *steps.tolist(), *(steps + 1).tolist()}):
                        decided = (
                            kind in decision.kinds
                            and runtime(ratio, x_c, n_c, x_t, n_t, beta, tail=tail) < tail
                        )
                        assert row[x_t] == decided, (rule, kind, x_c, x_t)
        self._assert_longer_searches_never_reject_less(masks)

    @pytest.mark.parametrize(("n_c", "n_t", "ratio", "tail", "alternative", "controls"), _DESIGNS)
    def test_approximate_replay_follows_the_contract(
        self, monkeypatch, n_c, n_t, ratio, tail, alternative, controls
    ):
        decision = BinomialDecision(
            n_c, n_t, ratio, binomial_rr.nuisance_beta(0.05), tail, alternative
        )
        masks = []
        for rule in self._RULES:
            monkeypatch.setattr(binomial_rr, "NUISANCE_STOP", rule)
            masks.append(
                RejectionGeometry(decision, "approximate").cells(controls[0], controls[-1], 0, n_t)
            )
        self._assert_longer_searches_never_reject_less(masks)

    @staticmethod
    def _assert_longer_searches_never_reject_less(masks) -> None:
        for shorter, longer in zip(masks, masks[1:], strict=False):
            for short_mask, long_mask in zip(shorter, longer, strict=True):
                assert not (short_mask & ~long_mask).any()
        gained = sum(
            int((long & ~short).sum()) for short, long in zip(masks[0], masks[-1], strict=True)
        )
        assert gained > 0  # the cap binds somewhere on this design

    def test_deep_searches_continued_in_several_chunks_decide_as_one_pass_does(self, monkeypatch):
        """Rows still searching after the common splits continue in chunks sized from the leaf
        budget: with few common splits, many rows continue, and a budget cut to one row a chunk
        spreads them over several chunks without changing a decision."""
        from increment.power import _binomial

        decision = BinomialDecision(
            300, 300, 1.0, binomial_rr.nuisance_beta(0.05), 0.025, "two-sided"
        )
        whole = RejectionGeometry(decision, "approximate").cells(20, 40, 0, 300)
        monkeypatch.setattr(_binomial, "_COMMON_SPLITS", 6)
        monkeypatch.setattr(_binomial, "_LEAF_BUDGET_BYTES", 1e5)
        chunked = RejectionGeometry(decision, "approximate").cells(20, 40, 0, 300)
        for one_pass, in_chunks in zip(whole, chunked, strict=True):
            assert np.array_equal(one_pass, in_chunks)

    @pytest.mark.parametrize(
        ("route", "n_c", "n_t", "x_lo", "x_hi"),
        [("exact", 40, 60, 3, 14), ("approximate", 300, 300, 20, 40)],
    )
    def test_no_replay_exceeds_its_row_budget_even_for_one_request_wider_than_it(
        self, monkeypatch, route, n_c, n_t, x_lo, x_hi
    ):
        """A request spans a whole run of treatment counts. With the leaf budget cut to a
        megabyte a single request is several batches wide: it is replayed in pieces, no replay
        holds more rows than the budget allows, and no decision changes."""
        from increment.power import _binomial

        decision = BinomialDecision(
            n_c, n_t, 1.0, binomial_rr.nuisance_beta(0.05), 0.025, "two-sided"
        )
        whole = RejectionGeometry(decision, route).cells(x_lo, x_hi, 0, n_t)

        monkeypatch.setattr(_binomial, "_LEAF_BUDGET_BYTES", 1e6)
        batch_rows = _binomial._batch_rows(exact=route == "exact")
        assert batch_rows < n_t + 1
        replayed: list[int] = []
        live_batch = _binomial._classify_live

        def spy(decision, route, live, results):
            replayed.append(sum(item[1].j1 - item[1].j0 + 1 for item in live))
            return live_batch(decision, route, live, results)

        monkeypatch.setattr(_binomial, "_classify_live", spy)
        split = RejectionGeometry(decision, route).cells(x_lo, x_hi, 0, n_t)

        assert replayed and max(replayed) <= batch_rows
        for one_batch, in_pieces in zip(whole, split, strict=True):
            assert np.array_equal(one_batch, in_pieces)

    @pytest.mark.parametrize(
        ("route", "n_c", "n_t", "x_lo", "x_hi"),
        [("exact", 40, 60, 3, 14), ("approximate", 300, 300, 20, 40)],
    )
    def test_batches_of_six_split_searches_stay_within_the_leaf_budget(
        self, monkeypatch, route, n_c, n_t, x_lo, x_hi
    ):
        """A replay capped at six splits is one stage: its batch is largest while the rows still
        searching are copied out of the root's arrays, beside which the copy and the field being
        gathered are held. Handed only counts that keep searching, with the budget cut to about a
        hundred rows, every batch fills with such rows; the live leaf arrays must stay within
        the budget and no decision differ from an uncut budget's."""
        from increment.power import _binomial

        beta = binomial_rr.nuisance_beta(0.05)
        decision = BinomialDecision(n_c, n_t, 1.0, beta, 0.025, "two-sided")
        monkeypatch.setattr(binomial_rr, "NUISANCE_STOP", binomial_rr._StopRule(self._GAP, 6))
        meter = _LeafMeter(monkeypatch)
        wide = [
            _binomial._Request(x_c, kind, 0, n_t)
            for x_c in range(x_lo, x_hi + 1)
            for kind in decision.kinds
        ]
        _binomial.classify(decision, route, wide)
        # One request a count, so each batch is filled to its row limit.
        requests = [_binomial._Request(x_c, kind, j, j) for x_c, kind, j in meter.searching[:300]]
        assert len(requests) == 300
        uncut = _binomial.classify(decision, route, requests)

        budget = 6.5e4
        monkeypatch.setattr(_binomial, "_LEAF_BUDGET_BYTES", budget)
        assert _binomial._batch_rows(exact=route == "exact") < len(requests)
        meter.reset()
        cut = _binomial.classify(decision, route, requests)

        assert meter.live == 0
        assert meter.peak <= budget
        assert meter.peak >= 0.95 * budget  # the batches really fill the budget
        decided = np.concatenate(cut)
        assert decided.any()
        assert not decided.all()
        assert np.array_equal(decided, np.concatenate(uncut))
        if route == "exact":
            for req, rejected in zip(requests, decided, strict=True):
                runtime = binomial_rr.p_plus if req.kind == "plus" else binomial_rr.p_minus
                p = runtime(1.0, req.x_c, n_c, req.j0, n_t, beta, tail=0.025)
                assert rejected == (p < 0.025)


class TestRouting:
    """(d) The basis is a function of the inputs, and every plan the runtime
    does not decide with the binomial test keeps the log-ratio model."""

    def test_repeated_calls_agree(self):
        baseline = Baseline.from_proportion(0.2)
        first = achieved_power(120, 0.4, baseline, _conversion())
        second = achieved_power(120, 0.4, baseline, _conversion())
        assert first == second
        assert first.power_basis == "exact"

    @pytest.mark.parametrize(
        "procedure",
        [
            make_procedure(),
            make_procedure(metric_type="conversion"),
            _conversion(decision_method="cuped"),
            ArmPlanningProcedure.standard("conversion", clustered=True),
        ],
        ids=["mean", "absorbed_conversion", "cuped_conversion", "clustered_conversion"],
    )
    def test_non_binomial_plans_are_asymptotic(self, procedure):
        overrides: dict[str, float] = {}
        if procedure.decision_method.variance_reduction == "cuped":
            overrides["cuped_rho"] = 0.3
        if procedure.dependence == "cluster":
            overrides.update(cluster_icc=0.01, avg_cluster_size=5.0)
        baseline = Baseline(mean=0.1, var=0.09, **overrides)
        for result in (
            achieved_power(2_000, 0.2, baseline, procedure),
            minimum_detectable_effect(2_000, baseline, procedure),
            required_sample_size(0.2, baseline, procedure),
        ):
            assert result.power_basis == "asymptotic"

    def test_sequential_conversion_is_asymptotic(self):
        from increment.semantics.models import InferenceSpec

        procedure = ArmPlanningProcedure.standard(
            "conversion", inference=InferenceSpec(kind="asymptotic_mean")
        )
        result = achieved_power(2_000, 0.2, Baseline.from_proportion(0.1), procedure)
        assert result.power_basis == "asymptotic"


class TestPowerBasisSurvivesSerialization:
    """(e) The basis travels with every serialized form."""

    def test_result_round_trips(self):
        result = achieved_power(60, 0.8, Baseline.from_proportion(0.2), _conversion())
        assert result.power_basis == "exact"
        dumped = result.model_dump(mode="json")
        assert dumped["power_basis"] == "exact"
        assert PowerResult.model_validate(json.loads(json.dumps(dumped))) == result

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_curve_rows_and_frames_carry_the_basis(self, backend):
        curve = power_curve(
            n_per_arm=[60],
            relative_lift=[0.5, 0.8],
            baseline=Baseline.from_proportion(0.2),
            procedure=[_conversion(), make_procedure()],
        )
        expected = ["exact", "asymptotic", "exact", "asymptotic"]
        assert [row["power_basis"] for row in curve.to_dicts()] == expected
        frame: Any = curve.to_frame(backend=backend)
        column = frame["power_basis"]
        values = column.to_pylist() if backend == "pyarrow" else list(column)
        assert [str(value) for value in values] == expected

    def test_curve_matches_scalar_solvers(self):
        baseline = Baseline.from_proportion(0.2)
        curve = power_curve(
            n_per_arm=60, relative_lift=[0.5, 0.8], baseline=baseline, procedure=_conversion()
        )
        for point, lift in zip(curve, (0.5, 0.8), strict=True):
            direct = achieved_power(60, lift, baseline, _conversion())
            assert (point.power, point.power_basis, point.mde_relative) == (
                direct.power,
                direct.power_basis,
                direct.mde_relative,
            )
            assert math.isfinite(point.power)
