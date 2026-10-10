"""Registered likelihood runtime and separate Gaussian planning boundaries."""

import pytest

from increment.errors import CapabilityError, InvalidRequestError

# Importing here (not just inside each test) pays scipy.stats' import cost
# during collection, keeping every test under the <100ms budget (--durations).
from increment.estimation.engine import Method, estimate_lift
from increment.estimation.sequential import AlwaysValid
from increment.semantics.models import MeanMetric


def test_estimate_lift_refuses_sequential_cuped_before_summary_conversion():
    from tests.sequential_cases import registration

    inference = AlwaysValid(registration=registration())

    def unread():
        raise AssertionError("summary read before adjustment rejection")
        yield

    with pytest.raises(CapabilityError) as raised:
        estimate_lift(
            [MeanMetric(name="outcome", entity="unit", fact="outcome")],
            unread(),
            control_group="control",
            methods=[Method(name="cuped", variance_reduction="cuped")],
            inference=inference,
        )
    assert raised.value.code == "sequential.transform.unpredictable"


class TestGaussianScorePlanning:
    """Historical Gaussian boundary oracles; no deployed-likelihood claim."""

    @pytest.mark.parametrize("se", [0.5, 0.02, 123.456])
    def test_pinned_gaussian_planning_boundary(self, se):
        """The mixture is tuned at its own se-independent optimum
        (mixture_r_star), so the z-scale boundary no longer varies with
        se_full -- only the drift does."""
        from increment.estimation.sequential import GaussianScoreMixture
        from increment.power.sequential import planning_bounds

        boundary = planning_bounds(GaussianScoreMixture(), (1.0,), se, 0.05)[0]
        assert boundary == pytest.approx(3.035122413028551, rel=1e-12)


def test_unchanged_prefix_advances_reveal_cursor():
    from datetime import date

    from increment import capture_sequential_snapshot
    from tests.sequential_cases import records, registration

    reg = registration()
    rows = records([0], [1])
    first = capture_sequential_snapshot(
        reg,
        rows,
        source_id=reg.source_id,
        definitions_id=reg.definitions_id,
        finalized=True,
        reveal_cursor=date(2025, 1, 1),
    )
    later = capture_sequential_snapshot(
        reg,
        rows,
        source_id=reg.source_id,
        definitions_id=reg.definitions_id,
        finalized=True,
        previous=first,
        reveal_cursor=date(2025, 1, 2),
    )

    assert later.prefix_id == first.prefix_id
    assert later.reveal_cursor == date(2025, 1, 2)
    assert later.records == first.records


def test_reveal_cursor_cannot_move_backward_on_an_unchanged_prefix():
    from datetime import date

    from increment import capture_sequential_snapshot
    from tests.sequential_cases import records, registration

    reg = registration()
    rows = records([0], [1])
    first = capture_sequential_snapshot(
        reg,
        rows,
        source_id=reg.source_id,
        definitions_id=reg.definitions_id,
        finalized=True,
        reveal_cursor=date(2025, 1, 2),
    )

    with pytest.raises(CapabilityError) as raised:
        capture_sequential_snapshot(
            reg,
            rows,
            source_id=reg.source_id,
            definitions_id=reg.definitions_id,
            finalized=True,
            previous=first,
            reveal_cursor=date(2025, 1, 1),
        )
    assert raised.value.code == "sequential.source.invalid"


@pytest.mark.parametrize("operation", ["capture", "link"])
def test_unchanged_prefix_without_cursor_preserves_asof_label(operation):
    from datetime import date

    from increment import capture_sequential_snapshot
    from increment.sequential_source import link_snapshot
    from tests.sequential_cases import records, registration

    reg = registration()
    rows = records([0], [1])
    first = capture_sequential_snapshot(
        reg,
        rows,
        source_id=reg.source_id,
        definitions_id=reg.definitions_id,
        finalized=True,
        reveal_cursor=date(2025, 1, 1),
    )
    if operation == "capture":
        continued = capture_sequential_snapshot(
            reg,
            rows,
            source_id=reg.source_id,
            definitions_id=reg.definitions_id,
            finalized=True,
            previous=first,
        )
    else:
        unlabeled = capture_sequential_snapshot(
            reg,
            rows,
            source_id=reg.source_id,
            definitions_id=reg.definitions_id,
            finalized=True,
        )
        continued = link_snapshot(unlabeled, first)
    assert continued.reveal_cursor == first.reveal_cursor
    assert continued.prefix_id == first.prefix_id


def test_bernoulli_negative_null_ratio_refused():
    from fractions import Fraction

    from increment import SequentialCell, SequentialRegistration
    from tests.sequential_cases import registration

    base = registration()
    with pytest.raises(InvalidRequestError):
        SequentialRegistration.model_validate(
            {
                **base.model_dump(),
                "roster": (
                    SequentialCell(
                        metric="outcome",
                        group_id="treatment",
                        null_lift=Fraction(-2),
                    ),
                ),
            }
        )


def _bernoulli_checkpoint(control=(0, 1, 0, 1, 0, 0, 1, 0), treatment=(1, 1, 0, 1, 1, 0, 1, 1)):
    from increment import AlwaysValid
    from tests.sequential_cases import (
        capture,
        estimate_sequential,
        records,
        registration,
    )

    reg = registration("bernoulli")
    snapshot = capture(reg, records(list(control), list(treatment)))
    bundle = estimate_sequential(snapshot, AlwaysValid(registration=reg))
    return bundle.results[0].require_exact_sequential_result().checkpoint


def _scalar_mean_checkpoint():
    from increment import AsymptoticMean, estimate_sequential
    from tests.asymptotic_cases import mean_capture, mean_records, mean_registration

    reg = mean_registration()
    snapshot = mean_capture(reg, mean_records([1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]))
    bundle = estimate_sequential(snapshot, AsymptoticMean(registration=reg))
    return bundle.results[0].require_asymptotic_sequential_result().checkpoint


def _respecify(checkpoint, **changes):
    """Rebuild through full validation, so no invariant is skipped by the edit."""
    from increment.estimation.sequential_result import SequentialCheckpoint

    payload = checkpoint.model_dump()
    for field, change in changes.items():
        payload[field] = {**payload[field], **change} if isinstance(change, dict) else change
    return SequentialCheckpoint.model_validate(payload)


def _usage():
    from increment.estimation.sequential_result import checkpoint_memo_usage

    return {
        usage.name: (usage.hits, usage.misses, usage.entries) for usage in checkpoint_memo_usage()
    }


# Provenance a checkpoint carries but no computation reads: which registration
# and verified prefix produced the state, how long that prefix was, and whether
# the cell is still current.
_PROVENANCE = (
    {"registration_id": "0" * 64},
    {"prefix_id": "0" * 64},
    {"filtration_id": "a-different-joint-reveal"},
    {"revealed_units": 4096},
    {"status": "frozen"},
)


class TestCheckpointMemoKey:
    """Evidence and inversion are functions of retained state, not of provenance."""

    @pytest.mark.parametrize("change", _PROVENANCE, ids=lambda c: next(iter(c)))
    def test_provenance_change_returns_the_memoised_evidence(self, change):
        from fractions import Fraction

        from increment.estimation.sequential_result import (
            checkpoint_bounds,
            checkpoint_certificate,
            clear_checkpoint_memos,
        )
        from increment.estimation.sequential_runtime import display_estimate, evaluate_checkpoint

        base = _bernoulli_checkpoint()
        other = _respecify(base, **change)
        assert other != base
        clear_checkpoint_memos()
        alpha = Fraction(1, 20)
        assert checkpoint_certificate(other) == checkpoint_certificate(base)
        assert checkpoint_bounds(other, alpha) == checkpoint_bounds(base, alpha)
        assert _usage()["certificate"] == (1, 1, 1)
        assert _usage()["bounds"] == (1, 1, 1)
        base_lift = display_estimate(evaluate_checkpoint(base, alpha=alpha))
        other_lift = display_estimate(evaluate_checkpoint(other, alpha=alpha))
        assert base_lift is not None and other_lift is not None
        assert other_lift.value == base_lift.value

    @pytest.mark.parametrize("change", _PROVENANCE, ids=lambda c: next(iter(c)))
    def test_provenance_change_returns_the_memoised_asymptotic_set(self, change):
        from increment.estimation.sequential_result import (
            checkpoint_mean_bounds,
            clear_checkpoint_memos,
        )

        base = _scalar_mean_checkpoint()
        other = _respecify(base, **change)
        assert other != base
        clear_checkpoint_memos()
        assert checkpoint_mean_bounds(other) == checkpoint_mean_bounds(base)
        assert _usage()["mean_bounds"] == (1, 1, 1)


def _beta(a, b):
    from increment import PredictivePrior

    return PredictivePrior(kind="beta", a=a, b=b).model_dump()


class TestCheckpointMemoSeparation:
    """A too-narrow key is a silent wrong answer, so every key field must separate."""

    @pytest.mark.parametrize(
        ("field", "change"),
        [
            ("cell", {"cell": {"null_lift": "1/5"}}),
            ("model", {"model": {"control_prior": _beta(2, 3)}}),
            ("control", {"control": {"successes": 4}}),
            ("treatment", {"treatment": {"successes": 4}}),
        ],
    )
    def test_key_field_separates_the_certificate(self, field, change):
        from increment.estimation.sequential_result import (
            checkpoint_certificate,
            clear_checkpoint_memos,
        )

        base = _bernoulli_checkpoint()
        other = _respecify(base, **change)
        clear_checkpoint_memos()
        assert checkpoint_certificate(other) != checkpoint_certificate(base)
        assert _usage()["certificate"] == (0, 2, 2)

    @pytest.mark.parametrize(
        ("field", "change"),
        [
            ("cell", {"cell": {"alternative": "greater"}}),
            ("model", {"model": {"control_prior": _beta(2, 3)}}),
            ("control", {"control": {"successes": 4}}),
            ("treatment", {"treatment": {"successes": 4}}),
        ],
    )
    def test_key_field_separates_the_inverted_set(self, field, change):
        from fractions import Fraction

        from increment.estimation.sequential_result import (
            checkpoint_bounds,
            clear_checkpoint_memos,
        )

        base = _bernoulli_checkpoint()
        other = _respecify(base, **change)
        clear_checkpoint_memos()
        alpha = Fraction(1, 20)
        assert checkpoint_bounds(other, alpha) != checkpoint_bounds(base, alpha)
        assert _usage()["bounds"] == (0, 2, 2)

    def test_error_level_separates_the_inverted_set(self):
        from fractions import Fraction

        from increment.estimation.sequential_result import (
            checkpoint_bounds,
            clear_checkpoint_memos,
        )

        base = _bernoulli_checkpoint()
        clear_checkpoint_memos()
        wide = checkpoint_bounds(base, Fraction(1, 10))
        narrow = checkpoint_bounds(base, Fraction(1, 100))
        assert wide != narrow
        assert _usage()["bounds"] == (0, 2, 2)

    @pytest.mark.parametrize(
        ("field", "change"),
        [
            ("cell.alpha", {"cell": {"alpha": "1/10"}}),
            ("cell.null_lift", {"cell": {"null_lift": "1/5"}}),
            ("cell.alternative", {"cell": {"alternative": "greater"}}),
            ("model.rho", {"model": {"rho": "1/4"}}),
            ("model.start_count", {"model": {"start_count": 9}}),
            ("control", {"control": {"mean": ("4",)}}),
            ("treatment", {"treatment": {"mean": ("6",)}}),
        ],
    )
    def test_key_field_separates_the_asymptotic_set(self, field, change):
        from increment.estimation.sequential_result import (
            checkpoint_mean_bounds,
            clear_checkpoint_memos,
        )

        base = _scalar_mean_checkpoint()
        other = _respecify(base, **change)
        clear_checkpoint_memos()
        assert checkpoint_mean_bounds(other) != checkpoint_mean_bounds(base)
        assert _usage()["mean_bounds"] == (0, 2, 2)


class TestCheckpointMemoBound:
    def test_bound_is_tunable_and_evicts_in_use_order(self):
        from increment.estimation.sequential_result import (
            _DEFAULT_MEMO_ENTRIES,
            checkpoint_certificate,
            clear_checkpoint_memos,
            set_checkpoint_memo_size,
        )

        base = _bernoulli_checkpoint()
        others = [_respecify(base, control={"successes": k}) for k in range(5)]
        try:
            set_checkpoint_memo_size(2)
            clear_checkpoint_memos()
            first = [checkpoint_certificate(other) for other in others]
            assert _usage()["certificate"] == (0, 5, 2)
            assert [checkpoint_certificate(other) for other in others[-2:]] == first[-2:]
            assert _usage()["certificate"] == (2, 5, 2)
        finally:
            set_checkpoint_memo_size(_DEFAULT_MEMO_ENTRIES)
            clear_checkpoint_memos()

    @pytest.mark.parametrize("entries", [0, -1])
    def test_nonpositive_bound_refused(self, entries):
        from increment.estimation.sequential_result import set_checkpoint_memo_size

        with pytest.raises(InvalidRequestError) as raised:
            set_checkpoint_memo_size(entries)
        assert raised.value.code == "sequential.memo.invalid"
