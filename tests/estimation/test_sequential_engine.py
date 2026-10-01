"""Sequential inference plumbed through infer_lift / estimate_lift."""

import math
import warnings
from typing import Any

import pytest

from increment.errors import InvalidRequestError
from increment.estimation.armstats import centered_row_from_raw_sums
from increment.estimation.engine import estimate_lift
from increment.estimation.inference import Normal, infer_lift
from increment.estimation.results import Estimate, LiftEstimate
from increment.estimation.sequential import AlwaysValid
from increment.semantics.models import (
    ConversionMetric,
    MeanMetric,
    Measure,
    RatioMetric,
    RetentionMetric,
)

# Two-arm group_summary rows: n=400 each, mean 10 vs 11, modest variance.
_ROWS = [
    centered_row_from_raw_sums(
        {
            "experiment_id": "exp",
            "metric": "revenue",
            "group_id": g,
            "n": 400,
            "sum_y": s,
            "sum_y2": q,
        }
    )
    for g, s, q in [("control", 4000.0, 41600.0), ("treatment", 4400.0, 50000.0)]
]

# Ratio-metric rows: denominator.window_days left None (numerator windowed) -
# exercises the "one or both windows unset" ratio branch of the open-ended warning.
_RATIO_ROWS = [
    centered_row_from_raw_sums(
        {
            "experiment_id": "exp",
            "metric": "conv_rate",
            "group_id": g,
            "n": 400,
            "sum_y": s,
            "sum_y2": q,
            "sum_den": d,
            "sum_den2": d2,
            "sum_yden": yd,
        }
    )
    for g, s, q, d, d2, yd in [
        ("control", 4000.0, 41600.0, 400.0, 400.0, 4000.0),
        ("treatment", 4400.0, 50000.0, 400.0, 400.0, 4400.0),
    ]
]

# Retention-metric rows: same two-arm shape, binary y (y2 == y) - exercises the
# "unbounded band" retention branch, and its bounded-band non-warning counterpart.
_RETENTION_ROWS = [
    centered_row_from_raw_sums(
        {
            "experiment_id": "exp",
            "metric": "d7_retention",
            "group_id": g,
            "n": 400,
            "sum_y": s,
            "sum_y2": s,
        }
    )
    for g, s in [("control", 150.0), ("treatment", 180.0)]
]


def _ratio_metric() -> RatioMetric:
    return RatioMetric(
        name="conv_rate",
        entity="user_id",
        numerator=Measure(fact="spend", window_days=7),
        denominator=Measure(fact="visit", window_days=None),
    )


def _mean_metric(window_days: int | None) -> MeanMetric:
    return MeanMetric(name="revenue", fact="spend", entity="user_id", window_days=window_days)


def _retention_metric(threshold_days: int | tuple[int, int]) -> RetentionMetric:
    return RetentionMetric(
        name="d7_retention", entity="user_id", fact="return_visit", threshold_days=threshold_days
    )


def _registered_case(*, alternative="two-sided", alpha=None):
    from fractions import Fraction as F

    from increment import SequentialCell, SequentialRegistration
    from tests.sequential_cases import capture, registration

    base = registration("gaussian")
    model = base.models[0].model_copy(update={"metric": "revenue"})
    cell = SequentialCell(
        metric="revenue",
        group_id="treatment",
        alternative=alternative,
        alpha=F(1, 20) if alpha is None else alpha,
    )
    reg = SequentialRegistration.model_validate(
        {**base.model_dump(), "models": (model,), "roster": (cell,)}
    )
    policy = AlwaysValid(registration=reg)
    rows = [
        {"unit_id": f"{i:04d}-{arm}", "group_id": arm, "values": {"revenue": value}}
        for i in range(400)
        for arm, value in (("control", 8 + 4 * (i % 2)), ("treatment", 9 + 4 * (i % 2)))
    ]
    return capture(reg, rows), policy


@pytest.mark.parametrize(
    "foreign_reference",
    [
        "additive",
        "binomial",
        "absolute_scale",
        "log_scale",
        "prior_state",
        "abs_bounds",
        "abs_reference",
        "cluster_reference",
        "fixed_reference",
    ],
)
def test_sequential_replay_rejects_fixed_horizon_metadata(foreign_reference):
    from increment import estimate_sequential
    from increment.errors import CodedError
    from increment.estimation.binomial_rr import confidence_interval, nuisance_beta
    from increment.estimation.results import BinomialConfidenceSet
    from tests.sequential_cases import registered_bernoulli

    snapshot, policy = registered_bernoulli(n=4)
    row = estimate_sequential(snapshot, policy).results[0]
    assert LiftEstimate.model_validate_json(row.model_dump_json()) == row
    payload = row.model_dump()
    if foreign_reference == "additive":
        payload.update(null_abs=0.0, abs_diff=10.0, abs_se=1.0)
    elif foreign_reference == "absolute_scale":
        payload["value_scale"] = "absolute"
    elif foreign_reference == "log_scale":
        payload["scale"] = "log"
    elif foreign_reference == "prior_state":
        payload["prior_shrunk"] = True
    elif foreign_reference == "abs_bounds":
        payload.update(abs_lb=-1.0, abs_ub=1.0)
    elif foreign_reference == "fixed_reference":
        payload["reference_kind"] = "normal"
    elif foreign_reference == "abs_reference":
        payload["abs_reference_kind"] = "normal"
    elif foreign_reference == "cluster_reference":
        payload["n_clusters"] = 8
    else:
        interval = confidence_interval(0, 4, 2, 4, alpha=0.05, alternative="two-sided")
        payload["binomial_set"] = BinomialConfidenceSet(
            lower=interval.lower - 1.0,
            upper=None,
            alpha=0.05,
            level=0.95,
            decision_alpha=0.05,
            geometry="central",
            x_c=0,
            n_c=4,
            x_t=2,
            n_t=4,
            nuisance_beta=nuisance_beta(0.05),
        )
    with pytest.raises(CodedError) as caught:
        LiftEstimate.model_validate(payload)
    expected_code = (
        "estimation.results.lift.inference_reference_kind_mismatch"
        if foreign_reference == "fixed_reference"
        else "sequential.source.invalid"
    )
    assert caught.value.code == expected_code


class TestInferLiftSequential:
    def test_fixed_default_is_unchanged(self):
        result = infer_lift(
            metric="m",
            group_id="t",
            method="unadjusted",
            method_role="decision",
            log_rr=math.log(1.1),
            se_t=0.02,
            se_c=0.02,
        )
        assert result.inference == "fixed"
        assert result.require_lift().value == pytest.approx(0.1)

    @pytest.mark.parametrize("n_comparison", [None, 800])
    def test_se_only_entrypoint_cannot_claim_a_certified_process(self, n_comparison):
        from increment.errors import CapabilityError
        from tests.sequential_cases import registration

        policy = AlwaysValid(registration=registration("gaussian"))
        with pytest.raises(CapabilityError) as raised:
            infer_lift(
                metric="m",
                group_id="t",
                method="unadjusted",
                method_role="decision",
                log_rr=0.1,
                se_t=0.02,
                se_c=0.02,
                inference_spec=policy,
                n_comparison=n_comparison,
            )
        assert raised.value.code == "sequential.route.unsupported"


class TestEstimateLiftSequential:
    def test_raw_state_public_path_refuses_unadmitted_gaussian(self):
        from increment.errors import CapabilityError

        snapshot, policy = _registered_case()
        with pytest.raises(CapabilityError) as raised:
            estimate_lift([_mean_metric(7)], snapshot, control_group="control", inference=policy)
        assert raised.value.code == "sequential.route.unsupported"

    def test_registered_sensitivity_role_refuses_before_evidence(self):
        from increment.errors import CapabilityError
        from tests.sequential_cases import registered_bernoulli

        snapshot, policy = registered_bernoulli(n=4)
        with pytest.raises(CapabilityError) as raised:
            estimate_lift(
                [ConversionMetric(name="revenue", entity="unit_id", fact="revenue")],
                snapshot,
                control_group="control",
                inference=policy,
                method_roles={"unadjusted": "sensitivity"},
            )
        assert raised.value.code == "sequential.route.unsupported"

    @pytest.mark.parametrize("summary", [_ROWS, _RATIO_ROWS, _RETENTION_ROWS])
    def test_rounded_moments_cannot_certify_a_raw_likelihood(self, summary):
        from increment.errors import CapabilityError
        from tests.sequential_cases import registration

        with pytest.raises(CapabilityError) as raised:
            estimate_lift(
                [_mean_metric(7)],
                summary,
                control_group="control",
                inference=AlwaysValid(registration=registration("gaussian")),
            )
        assert raised.value.code == "sequential.source.invalid"

    def test_no_inference_no_warning_even_when_open_ended(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = estimate_lift([_mean_metric(None)], _ROWS, control_group="control").results
        assert [row.inference for row in results] == ["fixed"]

    def test_prior_refused_before_summary_conversion(self):
        from increment.errors import CapabilityError
        from tests.sequential_cases import registration

        def unread():
            raise AssertionError("summary read before prior refusal")
            yield

        with pytest.raises(CapabilityError) as raised:
            estimate_lift(
                [_mean_metric(7)],
                unread(),
                control_group="control",
                prior=Normal(mu=0, sigma=1),
                inference=AlwaysValid(registration=registration("gaussian")),
            )
        assert raised.value.code == "sequential.route.unsupported"

    def test_alternative_forwarded_per_arm(self):
        results = estimate_lift(
            [_mean_metric(7)],
            _ROWS,
            control_group="control",
            alternative="greater",
        ).results
        assert [row.alternative for row in results] == ["greater"]
        assert results[0].require_lift().level == pytest.approx(0.90)


class TestLiftEstimateAlternativeField:
    def test_defaults_to_two_sided(self):
        r = LiftEstimate(
            metric="m",
            group_id="t",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
        )
        assert r.alternative == "two-sided"

    def test_one_sided_label_round_trips(self):
        r = LiftEstimate(
            metric="m",
            group_id="t",
            method="unadjusted",
            method_role="decision",
            lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
            alternative="greater",
        )
        assert r.alternative == "greater"


class TestOneSidedInferLift:
    _KW: dict[str, Any] = {
        "metric": "m",
        "group_id": "t",
        "method": "unadjusted",
        "method_role": "decision",
        "log_rr": math.log(1.1) - 0.0,
        "se_t": 0.02,
        "se_c": 0.02,
    }

    def test_greater_is_the_doubled_alpha_two_sided_interval_labeled(self):
        one = infer_lift(**self._KW, alpha=0.05, alternative="greater")
        two = infer_lift(**self._KW, alpha=0.10)  # two-sided at 2*alpha
        # Same interval numbers, honest two-sided level, directional label.
        assert one.require_lift().lb == pytest.approx(two.require_lift().lb, rel=1e-12)
        assert one.require_lift().ub == pytest.approx(two.require_lift().ub, rel=1e-12)
        assert one.require_lift().level == pytest.approx(0.90)
        assert one.alternative == "greater"
        assert two.alternative == "two-sided"

    def test_less_same_interval_different_label(self):
        one = infer_lift(**self._KW, alpha=0.05, alternative="less")
        two = infer_lift(**self._KW, alpha=0.10)
        assert one.require_lift().ub == pytest.approx(two.require_lift().ub, rel=1e-12)
        assert one.alternative == "less"

    @pytest.mark.slow
    def test_one_sided_diagnostic_shares_the_central_lower_endpoint(self):
        from fractions import Fraction as F

        from increment.estimation.sequential_runtime import _evaluate_sequential_diagnostic

        one_snapshot, one_policy = _registered_case(alternative="greater", alpha=F(1, 20))
        two_snapshot, two_policy = _registered_case(alpha=F(1, 20))
        (one,) = _evaluate_sequential_diagnostic(one_snapshot, one_policy)
        (two,) = _evaluate_sequential_diagnostic(two_snapshot, two_policy)
        lower_one, lower_two = one.bounds, two.bounds
        assert lower_one.lower == lower_two.lower
        assert lower_one.upper is None
        assert lower_two.upper is not None

    def test_two_sided_default_bit_for_bit_unchanged(self):
        a = infer_lift(**self._KW, alpha=0.05)
        b = infer_lift(**self._KW, alpha=0.05, alternative="two-sided")
        assert a == b

    def test_unknown_alternative_rejected(self):
        with pytest.raises(InvalidRequestError) as raised:
            infer_lift(**self._KW, alternative="bigger")
        assert raised.value.code == "estimation.binomial.unknown_alternative"


def _wire_reveal():
    from increment import JointReveal

    return JointReveal(
        filtration_id="joint-units-v1",
        independent_unit_vectors=True,
        simultaneous_metrics=True,
        outcome_independent_order=True,
        immutable_finalized_outcomes=True,
        longest_window_days=14,
    )


def _wire_registration(law):
    """One registration per digested state shape: counts, one rational moment, two."""
    from fractions import Fraction as F

    from increment import (
        PredictivePrior,
        SequentialCell,
        SequentialModel,
        SequentialRegistration,
    )
    from increment.semantics.sequential import ScalarMeanModel

    models: tuple[SequentialModel | ScalarMeanModel, ...]
    if law == "bernoulli":
        prior = PredictivePrior(kind="beta", a=1, b=1)
        models = (
            SequentialModel(
                metric="outcome",
                law="bernoulli",
                control_prior=prior,
                treatment_prior=prior,
                positive_population_control=True,
            ),
        )
        # A plain cell and a segmented cell, so the digested state tuple carries
        # both an empty and a populated segment key.
        roster = (
            SequentialCell(metric="outcome", group_id="treatment", alpha=F(1, 40)),
            SequentialCell(
                metric="outcome",
                group_id="treatment",
                segment=(("plan", "pro"),),
                alpha=F(1, 40),
            ),
        )
    elif law == "scalar_mean":
        models = (
            ScalarMeanModel(
                metric="revenue",
                rho=F(1, 4),
                start_count=2,
                assignment="iid_fixed_bernoulli_randomization",
                treatment_probability=F(1, 2),
                consistency_and_no_interference=True,
                segment_membership="pre_assignment",
                unit_model="iid_stationary_potential_outcomes",
                moments="finite_2_plus_delta",
                positive_limiting_variance=True,
                positive_population_control=True,
            ),
        )
        roster = (SequentialCell(metric="revenue", group_id="treatment", alpha=F(1, 20)),)
    else:
        prior = PredictivePrior(kind="niw", kappa=1, nu=4, mean=(0, 1), scale=((2, 0), (0, 2)))
        models = (
            SequentialModel(
                metric="spend_per_visit",
                law="gaussian_ratio",
                control_prior=prior,
                treatment_prior=prior,
                positive_population_control=True,
                positive_population_denominators=True,
            ),
        )
        roster = (SequentialCell(metric="spend_per_visit", group_id="treatment", alpha=F(1, 20)),)
    return SequentialRegistration(
        source_id="experiment",
        definitions_id="immutable-definition-v1",
        control_group="control",
        committed_before_data=True,
        reveal=_wire_reveal(),
        models=models,
        roster=roster,
    )


# Unit identities deliberately include non-ASCII and JSON-escapable characters:
# the canonical record bytes must escape exactly as a whole-payload dump does.
_WIRE_MARKS = ("plain", "caf\u00e9", "sn\u00f8w", 'quo"te', "back\\slash", "\u2603")


def _wire_looks(law):
    """Reveal batches per look; the Bernoulli case opens with an empty prefix."""
    from fractions import Fraction as F

    if law == "bernoulli":
        looks = [[]]
        for look in range(3):
            looks.append(
                [
                    {
                        "unit_id": f"{_WIRE_MARKS[(look * 3 + i) % len(_WIRE_MARKS)]}-{look}-{i}-{arm}",
                        "group_id": arm,
                        "values": {"outcome": (look + i + (arm == "treatment")) % 2},
                        "segments": {"plan": "pro" if i % 2 else "basic"},
                    }
                    for i in range(3)
                    for arm in ("control", "treatment")
                ]
            )
        return looks
    if law == "scalar_mean":
        return [
            [
                {
                    "unit_id": f"u-{look}-{i}-{arm}",
                    "group_id": arm,
                    "values": {"revenue": F(look * 10 + i + (arm == "treatment"), 3)},
                    "segments": {},
                }
                for i in range(2)
                for arm in ("control", "treatment")
            ]
            for look in range(3)
        ]
    return [
        [
            {
                "unit_id": f"r-{look}-{i}-{arm}",
                "group_id": arm,
                "values": {"spend_per_visit": (F(look + i + 1, 2), F(i + 2))},
                "segments": {},
            }
            for i in range(2)
            for arm in ("control", "treatment")
        ]
        for look in range(3)
    ]


def _wire_snapshot(law):
    """Append every look onto the previous verified checkpoint, as a producer does."""
    from increment import capture_sequential_snapshot
    from increment.sequential_state import _capture_sequential_diagnostic_snapshot

    registration = _wire_registration(law)
    capture = (
        _capture_sequential_diagnostic_snapshot
        if law == "gaussian_ratio"
        else capture_sequential_snapshot
    )
    snapshot = None
    for rows in _wire_looks(law):
        snapshot = capture(
            registration,
            rows,
            source_id=registration.source_id,
            definitions_id=registration.definitions_id,
            finalized=True,
            previous=snapshot,
            append=snapshot is not None,
        )
    return snapshot


# Pinned wire identities: persisted checkpoints carry these digests, and
# continuation and family selection re-derive them, so any moved digest
# silently invalidates stored state.
_WIRE_IDENTITY = {
    "bernoulli": {
        "registration_id": "56efc0d1c325943b25dc9250e01d0674974d92f0507430c779552f922d63bfef",
        "prefix_id": "f51d1425e27a524acc0fb76e955221bf7db15aa0fbf760392e841eeb01b0dbc9",
        "parent_id": "3a258dccac69ab9da521ed64aa801b80f6d5c884435d6d4a3862d09cf74f1c4b",
        "ancestors": (
            (0, "6c3657d26f467d658b405bc9b58bdfff24be138cfb46781b98f253f4954099a5"),
            (6, "39c2cc8432fdf503bf4ac715f09f8003fb68296659c55529e83d26833f803f8a"),
            (12, "3a258dccac69ab9da521ed64aa801b80f6d5c884435d6d4a3862d09cf74f1c4b"),
        ),
        "records": 18,
        "first_digest": "e01a7608f68a4d7bada00b5ae64228057f8103b88539a6ebc8a54833d88a1745",
        "last_digest": "890b61478b281669c92356f971413c921782b19cece579097edbfbd94a21e6f5",
    },
    "scalar_mean": {
        "registration_id": "eddb98818eece6caf042a12c09b74f45440609f5e2af0c90e10ef47800ecc266",
        "prefix_id": "f66a1f4815b7dfb05f16ff179469d37927e16a96731165353e7e70c7916aad82",
        "parent_id": "cc18eb2a0ea3d3c3c7ffc3643492e53fe2aee31cd954c5c286bcadadcdfdc849",
        "ancestors": (
            (4, "b5765ce1cf1947137563b3652775a21bf2f52bcb0242f621745ae70ae0841689"),
            (8, "cc18eb2a0ea3d3c3c7ffc3643492e53fe2aee31cd954c5c286bcadadcdfdc849"),
        ),
        "records": 12,
        "first_digest": "39b6ca9a1f77adfc846058fa99c7aebefe84486f6bd2078f7d413b141dc44bbb",
        "last_digest": "02198ea9eaa66835940f97b3a297b1cb6728d263d63fccc528535f52cd3a9c24",
    },
    "gaussian_ratio": {
        "registration_id": "97c2ae5098aeac24e8029944d09ecc8bef16332af43d7de2ebaa915d2574ff10",
        "prefix_id": "274ab333796479ba1f48cb244a986091ef5d3f551bc561b6b816700e60a89df8",
        "parent_id": "8a95b262aeb7c42f1eec870cb364fe0d75b8eab74cfc0fee693faad9ef53591c",
        "ancestors": (
            (4, "c982f11efa81220a505e166099f9e8b89d8a4c8f466a01fe7bfd6f5c8a357b75"),
            (8, "8a95b262aeb7c42f1eec870cb364fe0d75b8eab74cfc0fee693faad9ef53591c"),
        ),
        "records": 12,
        "first_digest": "4cfbfdf8c2d329d7bdb53749e0f0dc007ab596651119afc270652a2ef71d17a9",
        "last_digest": "aa961825a6b3cdf91d8ada49db4174512e0834d54345056de663914d069bbb0f",
    },
}


class TestSnapshotWireIdentity:
    """Every digest a snapshot publishes, and the cost of re-deriving them."""

    @pytest.mark.parametrize("law", list(_WIRE_IDENTITY))
    def test_multi_look_digests_match_the_released_wire_values(self, law):
        expected = _WIRE_IDENTITY[law]
        snapshot = _wire_snapshot(law)
        assert len(snapshot.records) == expected["records"]
        assert snapshot.registration_id == expected["registration_id"]
        assert snapshot.prefix_id == expected["prefix_id"]
        assert snapshot.parent_id == expected["parent_id"]
        assert snapshot.content_id() == expected["prefix_id"]
        assert (
            tuple((a.n_records, a.prefix_id) for a in snapshot.ancestors) == expected["ancestors"]
        )
        assert snapshot.records[0].digest == expected["first_digest"]
        assert snapshot.records[-1].digest == expected["last_digest"]

    @pytest.mark.parametrize("law", list(_WIRE_IDENTITY))
    def test_every_prefix_digest_equals_a_whole_payload_serialization(self, law):
        """The streamed prefix digest must equal the reference spelling exactly.

        `canonical_id` over the assembled payload is the definition of a snapshot
        identity. Validation streams the shared record prefix instead, so the two
        spellings are asserted to agree byte-for-byte on the head and on every
        ancestor prefix, including an ancestor that retains no records at all.
        """
        from increment.sequential_state import canonical_id

        snapshot = _wire_snapshot(law)

        def reference(records, states):
            return canonical_id(
                {
                    "version": snapshot.version,
                    "registration_id": snapshot.registration_id,
                    "records": [r.model_dump(mode="json") for r in records],
                    "states": [s.model_dump(mode="json") for s in states],
                }
            )

        assert snapshot.prefix_id == reference(snapshot.records, snapshot.states)
        assert [a.n_records for a in snapshot.ancestors][:1] == ([0] if law == "bernoulli" else [4])
        for ancestor in snapshot.ancestors:
            assert ancestor.prefix_id == reference(
                snapshot.records[: ancestor.n_records], ancestor.states
            )

    @pytest.mark.parametrize("law", list(_WIRE_IDENTITY))
    def test_stored_payload_revalidates_to_the_same_identity(self, law):
        """A persisted checkpoint re-validates without the capture-side digest."""
        from increment.sequential_state import snapshot_from_json

        snapshot = _wire_snapshot(law)
        restored = snapshot_from_json(snapshot.model_dump_json())
        assert restored.prefix_id == _WIRE_IDENTITY[law]["prefix_id"]
        assert restored == snapshot

    @pytest.mark.parametrize(
        "tamper", ["count_exceeds_prefix", "empty_prefix_claims_units", "binds_a_longer_prefix"]
    )
    def test_each_ancestor_is_checked_against_its_own_record_prefix(self, tamper):
        """A re-sealed ancestor must not borrow the head prefix's units.

        Ancestor digests and retained assignment counts are derived in one
        forward pass over the records. Each forgery below is internally
        self-consistent -- its digest binds the states it carries -- and
        disagrees only with the prefix it claims. Reusing the head's counts for
        every ancestor, or taking a boundary at the wrong position, accepts them.
        """
        from increment.errors import CapabilityError
        from increment.sequential_state import (
            SequentialAncestor,
            SequentialSnapshot,
            canonical_id,
        )

        snapshot = _wire_snapshot("bernoulli")

        def sealed(records, states):
            return canonical_id(
                {
                    "version": 2,
                    "registration_id": snapshot.registration_id,
                    "records": [r.model_dump(mode="json") for r in records],
                    "states": [s.model_dump(mode="json") for s in states],
                }
            )

        index = 0 if tamper == "empty_prefix_claims_units" else 1
        target = snapshot.ancestors[index]
        if tamper == "binds_a_longer_prefix":
            forged = SequentialAncestor(
                prefix_id=sealed(snapshot.records[: target.n_records + 2], target.states),
                n_records=target.n_records,
                states=target.states,
            )
        else:
            claimed = (
                target.states[0].model_copy(update={"n": target.states[0].n + 3, "successes": 1}),
                *target.states[1:],
            )
            forged = SequentialAncestor(
                prefix_id=sealed(snapshot.records[: target.n_records], claimed),
                n_records=target.n_records,
                states=claimed,
            )
        ancestors = list(snapshot.ancestors)
        ancestors[index] = forged
        # The tampered ancestor is never the last one, so parent_id still binds.
        assert index < len(ancestors) - 1
        with pytest.raises(CapabilityError) as raised:
            SequentialSnapshot(
                registration=snapshot.registration,
                registration_id=snapshot.registration_id,
                parent_id=snapshot.parent_id,
                ancestors=tuple(ancestors),
                prefix_id=snapshot.prefix_id,
                records=snapshot.records,
                states=snapshot.states,
                finalized=True,
            )
        assert raised.value.code == "sequential.source.invalid"

    def test_validation_hashes_each_record_once_per_prefix_family(self, monkeypatch):
        """Re-deriving L ancestor digests must not re-serialize L record prefixes.

        The identity check is O(records) per validation, not O(records x looks):
        a deep ancestry must not multiply the number of record serializations.
        """
        from increment import capture_sequential_snapshot
        from increment.sequential_state import SequentialSnapshot, UnitRecordProof

        registration = _wire_registration("scalar_mean")
        snapshot = None
        for look in range(12):
            snapshot = capture_sequential_snapshot(
                registration,
                [
                    {
                        "unit_id": f"u-{look}-{arm}",
                        "group_id": arm,
                        "values": {"revenue": look + (arm == "treatment")},
                        "segments": {},
                    }
                    for arm in ("control", "treatment")
                ],
                source_id=registration.source_id,
                definitions_id=registration.definitions_id,
                finalized=True,
                previous=snapshot,
                append=snapshot is not None,
            )
        assert len(snapshot.ancestors) == 11
        dumped = []
        original = UnitRecordProof.model_dump

        def counting(self, *args, **kwargs):
            dumped.append(self)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(UnitRecordProof, "model_dump", counting)
        revalidated = SequentialSnapshot.model_validate(snapshot, strict=False)
        assert revalidated.prefix_id == snapshot.prefix_id
        # One dump per retained record, not one per record per ancestor prefix.
        assert len(dumped) == len(snapshot.records)
