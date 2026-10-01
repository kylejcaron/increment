"""Pre-data declarations and deterministic finalized scalar outcome fixtures."""

from fractions import Fraction as F

from increment import JointReveal, ScalarMeanModel, SequentialCell, SequentialRegistration


def mean_model(metric="outcome", *, rho=F(1, 10), start_count=2, law="scalar_mean"):
    return ScalarMeanModel(
        metric=metric,
        law=law,
        rho=rho,
        start_count=start_count,
        assignment="iid_fixed_bernoulli_randomization",
        treatment_probability=F(1, 2),
        consistency_and_no_interference=True,
        segment_membership="pre_assignment",
        unit_model="iid_stationary_potential_outcomes",
        moments="finite_2_plus_delta",
        positive_limiting_variance=True,
        positive_population_control=True,
        positive_population_denominators=law.endswith("ratio_mean"),
    )


def mean_registration(*, models=None, cells=None, **changes):
    return SequentialRegistration.model_validate(
        {
            "source_id": "experiment",
            "definitions_id": "stationary-unit-mean-v1",
            "control_group": "control",
            "committed_before_data": True,
            "reveal": JointReveal(
                filtration_id="finalized-iid-units-v1",
                independent_unit_vectors=True,
                simultaneous_metrics=True,
                outcome_independent_order=True,
                immutable_finalized_outcomes=True,
                longest_window_days=14,
            ),
            "models": models or (mean_model(),),
            "roster": cells or (SequentialCell(metric="outcome", group_id="treatment"),),
            **changes,
        }
    )


def mean_records(control, treatment, *, metrics=("outcome",), segment=None, offset=0):
    return [
        {
            "unit_id": f"{i + offset:08d}-{arm}",
            "group_id": arm,
            "values": dict.fromkeys(metrics, value),
            "segments": segment or {},
        }
        for i in range(max(len(control), len(treatment)))
        for arm, values in (("control", control), ("treatment", treatment))
        if i < len(values)
        for value in (values[i],)
    ]


def mean_capture(registration, records, **kwargs):
    from increment import capture_sequential_snapshot

    return capture_sequential_snapshot(
        registration,
        records,
        source_id=registration.source_id,
        definitions_id=registration.definitions_id,
        finalized=True,
        **kwargs,
    )
