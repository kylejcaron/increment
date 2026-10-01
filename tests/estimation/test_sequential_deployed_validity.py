"""Behavioral public cutover witnesses. Main owns execution and the scientific grid."""

import hashlib
import json
from fractions import Fraction as F

import pytest

from increment import (
    AlwaysValid,
    AsymptoticMean,
    SequentialCell,
    SequentialRegistration,
    capture_sequential_snapshot,
    snapshot_from_json,
)
from increment import (
    estimate_sequential as _public_estimate_sequential,
)
from increment.errors import CapabilityError
from increment.estimation.decision_types import (
    AsymptoticSequentialEvidence,
    EValueEvidence,
    sequential_hypothesis_key,
)
from increment.estimation.family import select_sequential_family
from increment.estimation.sequential_runtime import (
    _evaluate_sequential_diagnostic,
    reinvert_selected,
)
from increment.sequential_state import (
    _capture_sequential_diagnostic_snapshot,
    declare_sequential_freeze,
)
from tests.estimation._sequential_acceptance import (
    CASE_COVERAGE,
    CERTIFICATION_DESIGN,
    CERTIFICATION_LEDGER,
    COVERAGE_MANIFEST,
    INTEGRATION_CASES,
    INTEGRATION_CEILING,
    IntegrationCase,
    IntegrationWork,
    integration_work,
    principal_manifest,
    require_integration_budget,
)
from tests.sequential_cases import capture, estimate_sequential, records, registration


def test_capture_rejects_assignments_outside_registered_roster():
    reg = registration("bernoulli")
    rows = records([0], [1])
    rows.append(
        {
            "unit_id": "00000001-unknown",
            "group_id": "not-registered",
            "values": {"outcome": 1},
        }
    )
    with pytest.raises(CapabilityError) as raised:
        capture(reg, rows)
    assert raised.value.code == "sequential.source.invalid"


@pytest.mark.slow
@pytest.mark.parametrize(
    "law,c,t",
    [
        ("bernoulli", [0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24),
    ],
)
def test_actual_public_raw_likelihood_has_power_with_unknown_unequal_variances(law, c, t):
    reg = registration(law)
    bundle = estimate_sequential(capture(reg, records(c, t)), AlwaysValid(registration=reg))
    row = bundle.results[0]
    evidence = next(iter(bundle.evidence.values()))
    assert isinstance(evidence, EValueEvidence)
    assert row.stat_sig()
    assert evidence.log_e > 3
    lower = row.require_sequential_result().bounds.lower
    assert lower is not None and lower > 1
    assert row.require_lift().value > 0
    assert row.require_lift().log_se is None
    assert row.require_sequential_result().checkpoint.control.n == len(c)


def test_public_gaussian_capture_and_estimate_refuse():
    reg = registration("gaussian")
    with pytest.raises(CapabilityError) as raised:
        capture_sequential_snapshot(
            reg,
            records([1], [2]),
            source_id=reg.source_id,
            definitions_id=reg.definitions_id,
            finalized=True,
        )
    assert raised.value.code == "sequential.route.unsupported"
    diagnostic = _capture_sequential_diagnostic_snapshot(
        reg,
        records([1], [2]),
        source_id=reg.source_id,
        definitions_id=reg.definitions_id,
        finalized=True,
    )
    with pytest.raises(CapabilityError) as raised:
        _public_estimate_sequential(diagnostic, AlwaysValid(registration=reg))
    assert raised.value.code == "sequential.route.unsupported"


@pytest.mark.slow
def test_scalar_mean_retains_typed_asymptotic_result_across_checkpoint_resume():
    from tests.asymptotic_cases import mean_capture, mean_records, mean_registration

    require_integration_budget(INTEGRATION_CASES)
    case = next(c for c in INTEGRATION_CASES if c.law == "scalar_mean")
    reg = mean_registration()
    policy = AsymptoticMean(registration=reg)
    early_rows = mean_records(case.control * case.repeats[0], case.treatment * case.repeats[0])
    fresh = case.repeats[1] - case.repeats[0]
    later_rows = mean_records(
        case.control * fresh,
        case.treatment * fresh,
        offset=len(case.control) * case.repeats[0],
    )
    early = mean_capture(reg, early_rows)
    initial = estimate_sequential(early, policy)
    first = initial.results[0]
    first_result = first.require_asymptotic_sequential_result()
    assert all(isinstance(e, AsymptoticSequentialEvidence) for e in initial.evidence.values())
    assert first.require_lift().value == float(case.ratio - 1)
    replayed = snapshot_from_json(early.model_dump_json())
    assert estimate_sequential(replayed, policy).results == initial.results
    later = mean_capture(reg, later_rows, previous=replayed, append=True)
    full = mean_capture(reg, early_rows + later_rows, previous=early)
    resumed = estimate_sequential(later, policy)
    uninterrupted = estimate_sequential(full, policy)
    assert resumed.results == uninterrupted.results
    assert resumed.evidence == uninterrupted.evidence
    assert resumed.results[0].require_asymptotic_sequential_result().bounds != first_result.bounds
    declared = declare_sequential_freeze(replayed, ["outcome"])
    later = mean_capture(reg, later_rows, previous=declared, append=True)
    retained = estimate_sequential(later, policy).results[0]
    assert retained.stat_sig() == first.stat_sig()
    assert retained.require_lift() == first.require_lift()
    assert retained.require_asymptotic_sequential_result().bounds == first_result.bounds
    assert retained.require_asymptotic_sequential_result().checkpoint.status == "frozen"
    assert retained.require_asymptotic_sequential_result().checkpoint.prefix_id == early.prefix_id
    with pytest.raises(CapabilityError):
        retained.p_value()


@pytest.mark.slow
def test_zero_events_remain_in_process_then_become_informative():
    reg = registration()
    spec = AlwaysValid(registration=reg)
    early = records([0] * 8, [0] * 8)
    first = capture(reg, early)
    row = estimate_sequential(first, spec).results[0]
    assert row.lift is None
    assert not row.stat_sig()
    assert row.require_sequential_result().point_reason == "observed control mean is zero"
    assert row.require_sequential_result().bounds.upper is None
    later = early + records([0, 0, 0, 1] * 32, [0, 1, 1, 1] * 32, offset=8)
    last = estimate_sequential(capture(reg, later, first), spec).results[0]
    assert last.stat_sig()
    assert last.require_sequential_result().checkpoint.control.n == 136
    assert last.lift is not None


@pytest.mark.parametrize(
    "law,c,t",
    [
        ("gaussian", [1], [2]),
        ("gaussian_ratio", [(1, 1)], [(2, 1)]),
    ],
)
def test_singular_prefix_abstains_without_discarding_observations(law, c, t):
    reg = registration(law)
    spec = AlwaysValid(registration=reg)
    early = records(c, t)
    first = capture(reg, early)
    row = _evaluate_sequential_diagnostic(first, spec)[0]
    assert row.certificate.status == "zero"
    assert row.bounds.status == "abstained"
    if law == "gaussian":
        later = records([1, 2, 3, 4] * 20, [6, 10, 14, 18] * 20, offset=1)
    else:
        later = records(
            [(1, 1), (2, 1), (1, 2), (2, 2)] * 20, [(8, 1), (9, 1), (8, 2), (9, 2)] * 20, offset=1
        )
    last = _evaluate_sequential_diagnostic(capture(reg, early + later, first), spec)[0]
    assert last.certificate.status == "finite"
    assert last.checkpoint.control.n == 81
    assert last.rejects()


@pytest.mark.slow
@pytest.mark.parametrize(
    "alternative,null,c,t",
    [
        ("greater", F(1, 5), [0, 1, 1, 1] * 24, [0, 0, 0, 1] * 24),
        ("less", -F(1, 5), [0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24),
    ],
)
def test_shifted_composite_null_wrong_direction_never_becomes_evidence(alternative, null, c, t):
    reg = registration(
        cells=(
            SequentialCell(
                metric="outcome", group_id="treatment", alternative=alternative, null_lift=null
            ),
        )
    )
    row = estimate_sequential(capture(reg, records(c, t)), AlwaysValid(registration=reg)).results[0]
    assert row.require_exact_sequential_result().log_e < 0
    assert not row.stat_sig()
    assert row.require_sequential_result().bounds.alternative == alternative


def test_registration_identity_refuses_before_reading_records():
    reg = registration()

    def unread():
        raise AssertionError("producer was read before registration rejection")
        yield

    with pytest.raises(CapabilityError) as caught:
        capture_sequential_snapshot(
            reg, unread(), source_id="wrong", definitions_id="wrong", finalized=True
        )
    assert caught.value.code == "sequential.source.invalid"


def test_snapshot_equal_append_and_value_or_assignment_rewrite_are_distinct():
    reg = registration()
    early = records([0, 1], [1, 0])
    first = capture(reg, early)
    assert capture(reg, early, first) is first
    rows = early + records([1, 0], [1, 1], offset=2)
    appended = capture(reg, rows, first)
    assert appended.parent_id == first.prefix_id
    assert appended.arm("outcome", "control").n == 4
    replayed = snapshot_from_json(appended.model_dump_json())
    replayed.verify_parent(first)
    for patch in ({"values": {"outcome": 1}}, {"group_id": "treatment"}):
        corrupted = [dict(row) for row in rows]
        corrupted[0].update(patch)
        with pytest.raises(CapabilityError) as exc:
            capture(reg, corrupted, first)
        assert exc.value.code == "sequential.continuation.rewrite"


@pytest.mark.slow
def test_missing_cells_stay_in_family_and_selected_interval_is_reinverted():
    reg = registration(
        cells=(
            SequentialCell(metric="outcome", group_id="treatment", family=True),
            SequentialCell(metric="outcome", group_id="never-enrolled", family=True),
        )
    )
    spec = AlwaysValid(registration=reg)
    bundle = estimate_sequential(capture(reg, records([0, 0, 0, 1] * 32, [0, 1, 1, 1] * 32)), spec)
    cells = [
        (sequential_hypothesis_key(r.require_sequential_result().checkpoint.cell), r)
        for r in bundle.results
    ]
    outcome = select_sequential_family(cells, reg.q, spec, F(1, 20), computation=bundle)
    assert outcome.n_family == 2
    assert len(outcome.selected) == 1
    missing = next(r for r in bundle.results if r.group_id == "never-enrolled")
    missing_evidence = bundle.evidence[
        sequential_hypothesis_key(missing.require_sequential_result().checkpoint.cell)
    ]
    assert isinstance(missing_evidence, EValueEvidence)
    assert missing_evidence.log_e == -float("inf")
    selected = next(r for key, r in cells if key in outcome.selected)
    wider = reinvert_selected(selected, F(1, 100), ceiling=F(1, 100))
    assert (
        wider.require_sequential_result().checkpoint
        == selected.require_sequential_result().checkpoint
    )
    assert wider.require_sequential_result().bounds.alpha == F(1, 100)
    outer = wider.require_sequential_result().bounds
    inner = selected.require_sequential_result().bounds
    assert outer.lower is not None and inner.lower is not None
    assert outer.lower <= inner.lower
    assert outer.upper is None or (inner.upper is not None and outer.upper >= inner.upper)


@pytest.mark.slow
def test_shared_controls_overlapping_metrics_and_cell_freeze_keep_exact_stopped_state():
    base = registration()
    twin = base.models[0].model_copy(update={"metric": "correlated"})
    cells = tuple(
        SequentialCell(metric=m, group_id="treatment", family=True)
        for m in ("outcome", "correlated")
    )
    reg = registration(cells=cells, models=(base.models[0], twin))
    spec = AlwaysValid(registration=reg)
    rows = records([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24, extra=("correlated",))
    first = capture(reg, rows)
    initial = estimate_sequential(first, spec)
    frozen_before = initial.results[0].require_sequential_result()
    declared = declare_sequential_freeze(first, [frozen_before.checkpoint.cell.metric])
    later = capture(
        reg,
        rows + records([1, 1, 1, 0] * 8, [1, 0, 0, 0] * 8, offset=96, extra=("correlated",)),
        declared,
    )
    bundle = estimate_sequential(later, spec)
    stopped = next(
        r.require_sequential_result()
        for r in bundle.results
        if r.metric == frozen_before.checkpoint.cell.metric
    )
    assert stopped.checkpoint.status == "frozen"
    assert stopped.checkpoint.control == frozen_before.checkpoint.control
    assert stopped.log_e == frozen_before.log_e
    active = next(
        r.require_sequential_result()
        for r in bundle.results
        if r.metric != frozen_before.checkpoint.cell.metric
    )
    assert active.checkpoint.control.n == 128
    assert active.log_e < stopped.log_e


@pytest.mark.slow
def test_evidence_survives_exp_overflow_and_subnormal_alpha():
    reg = registration(alpha=F(1, 10**320))
    bundle = estimate_sequential(
        capture(reg, records([0] * 600, [1] * 600)), AlwaysValid(registration=reg)
    )
    evidence = next(iter(bundle.evidence.values()))
    assert isinstance(evidence, EValueEvidence)
    assert evidence.log_e > 710
    assert evidence.e_value is None
    assert bundle.results[0].stat_sig()
    assert bundle.results[0].require_sequential_result().bounds.alpha == F(1, 10**320)
    assert bundle.results[0].lift is None


@pytest.mark.slow
def test_overlapping_segment_cells_share_joint_units_without_shrinking_roster():
    cells = (
        SequentialCell(
            metric="outcome", group_id="treatment", segment=(("country", "US"),), family=True
        ),
        SequentialCell(
            metric="outcome", group_id="treatment", segment=(("paid", "yes"),), family=True
        ),
        SequentialCell(
            metric="outcome", group_id="treatment", segment=(("country", "CA"),), family=True
        ),
    )
    reg = registration(cells=cells)
    spec = AlwaysValid(registration=reg)
    snapshot = capture(
        reg,
        records(
            [0, 0, 0, 1] * 24,
            [0, 1, 1, 1] * 24,
            segments={"country": "US", "paid": "yes"},
        ),
    )
    bundle = estimate_sequential(snapshot, spec)
    rows = [
        (sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)
        for row in bundle.results
    ]
    selection = select_sequential_family(rows, reg.q, spec, F(1, 20), computation=bundle)
    assert selection.n_family == 3
    assert len(selection.selected) == 2
    present = [row for key, row in rows if key in selection.selected]
    assert (
        present[0].require_exact_sequential_result().log_e
        == present[1].require_exact_sequential_result().log_e
    )
    assert all(row.require_sequential_result().checkpoint.revealed_units == 192 for row in present)


@pytest.mark.slow
def test_portable_result_cannot_change_certificate_or_confidence_geometry():
    from increment.estimation.results import LiftEstimate

    reg = registration()
    row = estimate_sequential(
        capture(reg, records([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24)),
        AlwaysValid(registration=reg),
    ).results[0]
    assert LiftEstimate.model_validate_json(row.model_dump_json()).stat_sig()
    payload = row.model_dump(mode="json")
    payload["sequential_result"]["point_reason"] = "forged missing point"
    with pytest.raises(CapabilityError) as raised:
        LiftEstimate.model_validate(payload)
    assert raised.value.code == "sequential.source.invalid"
    changed = row.model_dump(mode="json")
    changed["lift"]["value"] = 100.0
    with pytest.raises(CapabilityError) as raised:
        LiftEstimate.model_validate(changed)
    assert raised.value.code == "sequential.source.invalid"


@pytest.mark.parametrize("law", ["bernoulli", "gaussian", "gaussian_ratio"])
def test_raw_append_capture_matches_full_prefix_and_rejects_rewrites(law):
    if law == "bernoulli":
        early, later = records([0, 1], [1, 0]), records([1, 1], [0, 1], offset=2)
    elif law == "gaussian":
        early, later = records([1, 2], [3, 4]), records([5, 8], [9, 12], offset=2)
    else:
        early = records([(1, 2), (2, 1)], [(3, 2), (4, 1)])
        later = records([(5, 1), (8, 3)], [(9, 2), (12, 1)], offset=2)
    reg = registration(law)
    first = capture(reg, early)
    full = capture(reg, early + later, first)
    appended = capture(reg, iter(later), first, append=True)
    assert appended == full
    assert capture(reg, early + later, full) is full
    with pytest.raises(CapabilityError) as raised:
        capture(reg, early, first, append=True)
    assert raised.value.code == "sequential.source.invalid"


def test_canonical_manifest_is_immutable_and_ordered():
    cases = principal_manifest()
    assert tuple(case for case, _ in COVERAGE_MANIFEST) == cases
    rows = []
    for case in cases:
        if case.law == "scalar_mean":
            continue
        row = {name: getattr(case, name) for name in case.__dataclass_fields__}
        for name in (
            "shape",
            "scalar_control_mean",
            "scalar_treatment_shift",
            "scalar_rho",
            "scalar_start_count",
        ):
            row.pop(name)
        row["allocation"] = list(row["allocation"])
        row["null"] = str(row["null"])
        row["wrong_direction"] = str(row["wrong_direction"])
        rows.append(row)
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == "69931ef196c36516c6d8d2ffc724ccad962f4bdb2b3303a9d20b1e5f2e22f7fa"
    assert len(rows) == 807
    assert len(cases) == len(CASE_COVERAGE) == 817
    marginals = [
        c for c in cases if "family-" not in c.name and not c.name.startswith(("wrong-", "power-"))
    ]
    assert sum(c.law != "bernoulli" for c in marginals) == 81

    def encode_design(value):
        if isinstance(value, F):
            return str(value)
        if isinstance(value, tuple):
            return [encode_design(item) for item in value]
        return value

    design_row = {
        name: encode_design(getattr(CERTIFICATION_DESIGN, name))
        for name in CERTIFICATION_DESIGN.__dataclass_fields__
    }
    design_digest = hashlib.sha256(
        json.dumps(design_row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert design_digest == "e75b5c926e25a460069c39567c473e93443715f54fdd890727f8faf699e769fd"
    ledger = CERTIFICATION_LEDGER
    values = [getattr(CERTIFICATION_DESIGN, n) for n in CERTIFICATION_DESIGN.__dataclass_fields__]
    assert not any(v is None for v in values)
    assert not any(isinstance(v, str) and v.startswith("unresolved") for v in values)
    assert ledger.per_decision_error == CERTIFICATION_DESIGN.mc_family_error / (
        ledger.budgeted_gate_count * ledger.interim_looks
    )
    assert ledger.max_stage_ladder[-1] == ledger.max_case_replications
    # The retained ladder cannot reach the derived requirement; the campaign is unsized.
    assert not ledger.historical_stages_sufficient
    families = [c for c in cases if "family-" in c.name]
    assert sum(c.law == "bernoulli" for c in marginals) == 144
    assert len(families) == 577
    assert sum(c.name.startswith("wrong-") for c in cases) == 12
    assert sum(c.name.startswith("power-") for c in cases) == 3
    for case, binding in zip(cases, CASE_COVERAGE, strict=True):
        is_family = "family-" in case.name
        if case.law == "scalar_mean":
            expected_claims = ("asymptotic_sequential_set_exclusion",)
        elif is_family and case.family > 1:
            expected_claims = ("stopped_FDR", "selected_FCR")
        else:
            expected_claims = ("ever_null_rejection", "stopped_FDR", "selected_FCR")
        assert binding.proofs == (
            () if case.law == "scalar_mean" else ("P1", "P2", "P3", "P4", "P5", "P6")
        )
        assert binding.upper_claims == expected_claims
        assert binding.power_reference_required == (
            case.name.startswith("power-")
            or (is_family and case.family > 1 and case.null_fraction == 0.5)
        )
        expected_bridge = (
            "Bernoulli-threshold-coupling"
            if case.law == "bernoulli"
            else (
                "registered-iid-scalar-mean-asymptotic-target"
                if case.law == "scalar_mean"
                else "Gaussian-rounding-and-declared-sampler-target"
            )
        )
        assert binding.law_bridge == expected_bridge
        assert binding.availability_width == (
            "preserve-unavailable-and-unbounded-geometry"
            if case.law == "scalar_mean"
            else "freeze-applicability-and-certify"
        )
    assert sum(c.family == 1 for c in families) == 192
    assert sum(c.family > 1 for c in families) == 385
    assert sum(b.power_nonvacuity_required for b in CASE_COVERAGE) == 3
    assert all(b.status == "prospective-unproved" for b in CASE_COVERAGE)


def test_deterministic_integration_cost_ceiling_precedes_record_expansion():
    from dataclasses import replace

    assert INTEGRATION_CEILING == IntegrationWork(600, 800, 12)
    work = require_integration_budget(INTEGRATION_CASES)
    # The shipped cases must do real work and stay inside every declared ceiling.
    assert work.raw_records > 0 and work.checkpoint_evaluations > 0
    assert work.raw_records <= INTEGRATION_CEILING.raw_records
    assert work.scalar_observations <= INTEGRATION_CEILING.scalar_observations
    assert work.checkpoint_evaluations <= INTEGRATION_CEILING.checkpoint_evaluations
    # Both oversized fixtures are well-formed (integration_work accepts each)
    # and exceed the record ceiling, so the ValueError below can only be the
    # budget refusal rather than a validation one.
    oversized = ((replace(INTEGRATION_CASES[0], repeats=(6, 4096)),), INTEGRATION_CASES * 2)
    for cases in oversized:
        assert integration_work(cases).raw_records > INTEGRATION_CEILING.raw_records
        with pytest.raises(ValueError):
            require_integration_budget(cases)


def test_campaign_detects_changed_shared_registration_source(tmp_path, monkeypatch):
    from calibration import sequential as campaign

    relative = "tests/sequential_cases.py"
    fixture = tmp_path / relative
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes((campaign.ROOT / relative).read_bytes())
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    before = campaign._sources()
    fixture.write_bytes(fixture.read_bytes() + b"\n# changed registration implementation\n")
    after = campaign._sources()
    assert after[relative] != before[relative]


def test_sufficient_state_campaign_worker_journals_completed_replication(tmp_path):
    import subprocess
    import sys

    from calibration import sequential as campaign
    from calibration.journal import verify
    from calibration.profile import load
    from tests.estimation._sequential_acceptance import campaign_identity

    plan = campaign.campaign_plan(load("smoke", campaign="sequential"), repetitions=1)
    case_plan = next(item for item in plan["cases"] if item["case_index"] == 0)
    campaign._write(
        tmp_path / "manifest.json",
        {
            "source_files": campaign._sources(),
            "scientific_identity": campaign_identity(principal_manifest()),
            "plan": plan,
        },
    )
    directory = tmp_path / "case"
    directory.mkdir()
    campaign._write(directory / "plan.json", case_plan)

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "from calibration.sequential import _worker_entry; "
            "raise SystemExit(_worker_entry(Path(sys.argv[1])))",
            str(directory),
        ],
        cwd=campaign.ROOT,
        check=True,
        timeout=30,
    )

    result = json.loads((directory / "result.json").read_text())
    totals = verify(directory, case_id=case_plan["case_id"])
    assert (totals.started, totals.completed) == (1, 1)
    assert totals.counters["completed"] == 1
    assert result["journal_digest"] == totals.digest
    assert (result["attempted"], result["completed"], result["failed"], result["unfinished"]) == (
        1,
        1,
        0,
        0,
    )


def test_scalar_manifest_freezes_the_declared_science_and_cli_selection():
    from dataclasses import replace

    from calibration.profile import load as load_profile
    from calibration.sequential import CAMPAIGN, _indices, _parser, campaign_plan
    from tests.estimation._sequential_acceptance import (
        RuntimeCase,
        campaign_declaration,
        campaign_identity,
        campaign_science,
    )

    cases = principal_manifest()
    scalar = tuple(case for case in cases if case.law == "scalar_mean")
    expected = (
        ("gaussian", (1, 1), 6, F(-1, 5), "two-sided", 1, 127),
        ("gaussian", (1, 1), 14, F(0), "greater", 4, 128),
        ("gaussian", (1, 4), 6, F(1, 5), "less", 1, 129),
        ("gaussian", (1, 4), 14, F(-1, 5), "two-sided", 4, 130),
        ("finite_moment", (1, 1), 6, F(0), "greater", 1, 131),
        ("finite_moment", (1, 1), 14, F(1, 5), "less", 4, 132),
        ("finite_moment", (1, 4), 6, F(-1, 5), "two-sided", 1, 133),
        ("finite_moment", (1, 4), 14, F(0), "greater", 4, 134),
    )
    assert scalar[:8] == tuple(
        RuntimeCase(
            name=f"scalar_mean-{shape}-{allocation}-{looks}",
            law="scalar_mean",
            shape=shape,
            allocation=allocation,
            looks=looks,
            null=null,
            alternative=alternative,
            variance_ratio=variance,
            seed=seed,
            stopping="first_rejection",
        )
        for shape, allocation, looks, null, alternative, variance, seed in expected
    )
    assert scalar[8] == RuntimeCase(
        name="scalar_mean-family-missing-unequal-peek",
        law="scalar_mean",
        shape="finite_moment",
        allocation=(1, 4),
        family=3,
        null_fraction=1 / 3,
        dependence="shared",
        looks=6,
        batch=10,
        stopping="first_discovery",
        missing=True,
        seed=139,
    )
    assert scalar[9] == RuntimeCase(
        name="scalar_mean-near-zero-control",
        law="scalar_mean",
        scalar_control_mean=F(1, 1000),
        scalar_treatment_shift=F(5),
        null_fraction=0,
        looks=6,
        batch=10,
        stopping="horizon",
        seed=149,
    )
    for case in scalar:
        reg, policy, _, _, _ = campaign_declaration(case)
        science = campaign_science(case)
        assert isinstance(policy, AsymptoticMean)
        assert all(
            m.treatment_probability == F(case.allocation[1], sum(case.allocation))
            for m in reg.models
        )
        assert science["alpha"] == F(1, 20 * case.family)
        assert science["validity_regime"] == "asymptotic_sequential"
        assert not science["finite_start_anytime_claim"]
    original = scalar[0]
    for changes in (
        {"allocation": (4, 1)},
        {"scalar_control_mean": F(1, 1000)},
        {"scalar_treatment_shift": F(5)},
        {"scalar_rho": F(1, 5)},
        {"scalar_start_count": 4},
        {"seed": 199},
        {"shape": "finite_moment"},
        {"stopping": "horizon"},
        {"null": F(1, 10)},
        {"alternative": "less"},
    ):
        changed = replace(original, **changes)
        assert campaign_identity((changed,)) != campaign_identity((original,))
        assert (
            campaign_declaration(changed)[0].definitions_id
            != campaign_declaration(original)[0].definitions_id
        )
    # Selection is the run profile's certified cell set, in manifest order.
    args = _parser().parse_args(["--output", "/tmp/unused"])
    plan = campaign_plan(load_profile(args.profile, campaign=CAMPAIGN))
    selected = _indices(args.cases, plan)
    assert {cases[i] for i in selected if cases[i].law == "scalar_mean"} == set(scalar)
    assert all(i in selected for i, case in enumerate(cases) if case.name.startswith("power-"))
    args = _parser().parse_args(["--output", "/tmp/unused", "--case", "216", "--case", "225"])
    assert _indices(args.cases, plan) == (216, 225)
    assert cases[216] == scalar[0] and cases[225] == scalar[-1]
    # A run cannot execute a cell its profile does not certify.
    smoke = campaign_plan(load_profile("smoke", campaign=CAMPAIGN))
    uncertified = sorted(set(range(len(cases))) - {e["case_index"] for e in smoke["cases"]})
    with pytest.raises(ValueError, match="not certified by profile"):
        _indices([uncertified[0]], smoke)


class _DeclaredScalarDraws:
    """Fixed draws exercise routing and stopping, not empirical calibration."""

    def __init__(self, assignments, noise, *, shape):
        from types import SimpleNamespace

        self.assignments = iter(assignments)
        self.noise = iter(noise)
        self.shape = shape
        self.bit_generator = SimpleNamespace(state={"draws": 0})

    def integers(self, high):
        if high == 1_000_000:
            return 1
        value = next(self.assignments)
        assert 0 <= value < high
        self.bit_generator.state = {"draws": self.bit_generator.state["draws"] + 1}
        return value

    def normal(self, *, size):
        assert self.shape == "gaussian"
        return [next(self.noise)] * size

    def standard_t(self, df, *, size):
        assert self.shape == "finite_moment" and df == 5
        return [next(self.noise)] * size


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["gaussian", "finite_moment"])
@pytest.mark.parametrize("stopping", ["first_rejection", "first_discovery"])
def test_scalar_campaign_public_stopping_retains_missing_member_budget(shape, stopping):
    from dataclasses import replace

    from increment.estimation.results import LiftEstimate
    from tests.estimation._sequential_acceptance import campaign_declaration, campaign_replication

    case = replace(
        next(
            c for c in principal_manifest() if c.name == "scalar_mean-family-missing-unequal-peek"
        ),
        shape=shape,
        stopping=stopping,
        scalar_control_mean=F(20),
        scalar_rho=F(1),
        scalar_start_count=3,
        looks=3,
        batch=1,
    )
    declaration = campaign_declaration(case)
    reg, policy, _, _, _ = declaration
    # Unequal realized counts (2,3), then (4,6), are possible Bernoulli prefixes.
    rng = _DeclaredScalarDraws([4, 0, 4, 0, 0] * 2, [-1, -1, 1, 1, 0] * 2, shape=shape)
    events = []
    result = campaign_replication(case, rng, declaration, events.append)
    assert result["completed_looks"] == 2
    assert result["retained_cells"] == 3 and result["selected_cells"] == 1
    assert result["nonnull_discovery"] == 1
    assert result["nominal_ever_null_rejection"] == 0
    assert result["validity_regime"] == "asymptotic_sequential"
    looks = [e for e in events if e["phase"] == "look_complete"]
    assert [e["stop"] for e in looks] == [False, True]
    assert looks[0]["fcr_alpha"] is None
    assert F(str(looks[1]["fcr_alpha"])) == F(1, 60)
    captures = [e for e in events if e["phase"] == "captured"]
    computation = _public_estimate_sequential(
        snapshot_from_json(json.dumps(captures[-1]["snapshot"])), policy
    )
    assert all(isinstance(e, AsymptoticSequentialEvidence) for e in computation.evidence.values())
    rows = {row.metric: row for row in computation.results}
    assert not rows["m0"].stat_sig() and rows["m1"].stat_sig() and not rows["m2"].stat_sig()
    assert rows["m2"].require_asymptotic_sequential_result().bounds.reason == "missing_arm"
    for row in rows.values():
        assert row.require_asymptotic_sequential_result().decision_alpha == F(1, 60)
        assert LiftEstimate.model_validate_json(row.model_dump_json()) == row
    family = select_sequential_family([], reg.q, policy, F(1, 20), computation=computation)
    assert family.n_family == 3 and len(family.selected) == 1
    assert family.fcr_alpha == F(1, 60) and family.realized_threshold == pytest.approx(1 / 60)


@pytest.mark.slow
@pytest.mark.parametrize("truth_in_gap", [False, True])
def test_near_zero_campaign_preserves_disconnected_unbounded_sets(truth_in_gap):
    from dataclasses import replace

    from tests.estimation._sequential_acceptance import campaign_declaration, campaign_replication

    case = replace(
        next(c for c in principal_manifest() if c.name == "scalar_mean-near-zero-control"),
        scalar_rho=F(1),
        looks=2,
        batch=2,
        **({"scalar_treatment_shift": F(0), "null_fraction": 1.0} if truth_in_gap else {}),
    )
    noise = [-1, 4, 1, 6] if truth_in_gap else [-1, -1, 1, 1]
    rng = _DeclaredScalarDraws([1, 0, 1, 0] * 2, noise * 2, shape="gaussian")
    events = []
    result = campaign_replication(case, rng, campaign_declaration(case), events.append)
    assert result["completed_looks"] == 2
    assert result["all_look_interval_statuses"] == {"disconnected": 2}
    assert result["available_points"] == 1
    assert result["selected_cells"] == 1
    assert result["selected_widths"][0]["width"] is None
    components = result["selected_widths"][0]["components"]
    assert len(components) == 2
    assert components[0]["lower"] is None and components[1]["upper"] is None
    assert F(components[0]["upper"]) < 1 < F(components[1]["lower"])
    assert result["nominal_fcp_lower"] == result["nominal_fcp_upper"] == str(int(truth_in_gap))


@pytest.mark.slow
@pytest.mark.parametrize("stopping,expected_looks", [("first_rejection", 3), ("horizon", 4)])
def test_deferring_interim_geometry_moves_no_gated_number(stopping, expected_looks):
    """Turning off the per-look census changes nothing a gate decides.

    The certification campaign runs with ``interim_geometry=False`` because no
    statistic in ``replication_statistics`` reads the per-look event stream or
    the every-look interval census, and serialising them dominates the
    campaign's byte budget. That substitution is only sound if the rest of the
    record is untouched, so this pins the WHOLE record -- not merely the gated
    statistics -- for a replication that stops early and one that runs to the
    declared horizon.
    """
    from dataclasses import replace

    import numpy as np

    from tests.estimation._sequential_acceptance import (
        campaign_declaration,
        campaign_replication,
        replication_statistics,
    )

    case = replace(
        next(c for c in principal_manifest() if c.name == "power-bernoulli"),
        looks=4,
        batch=50,
        stopping=stopping,
        null_fraction=0.0,
        rate=0.5,
    )
    declaration = campaign_declaration(case)
    observed = campaign_replication(case, np.random.default_rng(case.seed), declaration, [].append)
    events = []
    deferred = campaign_replication(
        case,
        np.random.default_rng(case.seed),
        declaration,
        events.append,
        interim_geometry=False,
    )
    assert observed["completed_looks"] == deferred["completed_looks"] == expected_looks
    assert replication_statistics(observed) == replication_statistics(deferred)
    geometry = ("all_look_interval_statuses", "interim_geometry_observed")
    assert {key: value for key, value in observed.items() if key not in geometry} == {
        key: value for key, value in deferred.items() if key not in geometry
    }
    # An absent census says nothing looked; it is never an empty one.
    assert observed["interim_geometry_observed"] and observed["all_look_interval_statuses"]
    assert not deferred["interim_geometry_observed"]
    with pytest.raises(KeyError):
        deferred["all_look_interval_statuses"]
    # The saving is the per-look payloads, so none may survive the deferral.
    assert {event["phase"] for event in events}.isdisjoint(
        {"look_started", "captured", "evaluated", "look_complete"}
    )


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["gaussian", "finite_moment"])
def test_campaign_capture_rng_resume_reproduces_public_inference(shape):
    from dataclasses import replace

    import numpy as np

    from increment.estimation.sequential_result import AsymptoticSequentialResult
    from tests.estimation._sequential_acceptance import (
        campaign_batch,
        campaign_declaration,
        campaign_replication,
    )

    case = replace(
        next(c for c in principal_manifest() if c.law == "scalar_mean"),
        shape=shape,
        looks=2,
        batch=5,
        allocation=(1, 4),
        stopping="horizon",
    )
    declaration = campaign_declaration(case)
    reg, policy, truths, arms, schedule = declaration
    events = []
    campaign_replication(case, np.random.default_rng(case.seed), declaration, events.append)
    captured = [e for e in events if e["phase"] == "captured"]
    first = snapshot_from_json(json.dumps(captured[0]["snapshot"]))
    resumed_rng = np.random.default_rng()
    resumed_rng.bit_generator.state = captured[0]["rng_state"]
    next_rows = campaign_batch(
        case, resumed_rng, reg, truths, arms, schedule[1], len(first.records)
    )
    resumed = capture(reg, next_rows, first, append=True)
    original = snapshot_from_json(json.dumps(captured[1]["snapshot"]))
    expected = _public_estimate_sequential(original, policy)
    actual = _public_estimate_sequential(resumed, policy)
    assert actual.results == expected.results and actual.evidence == expected.evidence
    recorded = [e for e in events if e["phase"] == "evaluated"][-1]
    assert tuple(row.require_asymptotic_sequential_result() for row in actual.results) == tuple(
        AsymptoticSequentialResult.model_validate(payload) for payload in recorded["results"]
    )


@pytest.mark.parametrize("allocation", [(1, 1), (1, 4), (4, 1)])
def test_scalar_sampler_does_not_balance_or_sort_assignments(allocation):
    from tests.estimation._sequential_acceptance import (
        RuntimeCase,
        campaign_batch,
        campaign_declaration,
    )

    case = RuntimeCase(name="assignment-witness", law="scalar_mean", allocation=allocation)
    reg, _, truths, arms, _ = campaign_declaration(case)
    high = sum(allocation)
    assignments = [0] * high + [high - 1] * high
    rng = _DeclaredScalarDraws(assignments, [1] * (2 * high), shape="gaussian")
    first = campaign_batch(case, rng, reg, truths, arms, 1, 0)
    second = campaign_batch(case, rng, reg, truths, arms, 1, len(first))
    assert [r["group_id"] for r in first + second] == ["treatment"] * high + ["control"] * high
    assert len({r["unit_id"] for r in first + second}) == 2 * high
    assert all(m.treatment_probability == F(allocation[1], high) for m in reg.models)


@pytest.mark.parametrize("schedule,expected", [("equal", (4, 4, 4)), ("front_loaded", (5, 4, 3))])
def test_campaign_preserves_integer_schedule_endpoints(schedule, expected):
    from tests.estimation._sequential_acceptance import RuntimeCase, campaign_schedule

    assert (
        campaign_schedule(RuntimeCase(name="schedule", looks=3, batch=4, schedule=schedule))
        == expected
    )


@pytest.mark.parametrize("law", ["bernoulli", "gaussian", "gaussian_ratio"])
def test_legacy_campaign_retains_seed_draw_order_and_transformations(law):
    import numpy as np

    from tests.estimation._sequential_acceptance import (
        RuntimeCase,
        campaign_batch,
        campaign_declaration,
    )

    case = RuntimeCase(
        name="legacy-draws",
        law=law,
        allocation=(1, 4),
        variance_ratio=4,
        null=F(1, 5),
    )
    reg, _, truths, arms, _ = campaign_declaration(case)
    rng = np.random.default_rng(case.seed)
    actual = campaign_batch(case, rng, reg, truths, arms, 2, 7)
    reference = np.random.default_rng(case.seed)
    expected = []
    for arm, count, ratio, sd in (("control", 2, 1.0, 1.0), ("t0", 8, 1.2, 2.0)):
        draws = (
            reference.random((count, 1))
            if law == "bernoulli"
            else reference.normal(size=(count, 1, 2))
        )
        for j in range(count):
            z = draws[j, 0]
            if law == "bernoulli":
                value = int(z < case.rate * ratio)
            else:
                numerator = 5.0 * ratio + sd * z[0]
                value = (
                    float(numerator)
                    if law == "gaussian"
                    else (float(2 * numerator), float(2 + 0.2 * (z[0] + z[1])))
                )
            expected.append(
                {
                    "unit_id": f"{7 + len(expected):09d}",
                    "group_id": arm,
                    "values": {"m0": value},
                    "segments": {"segment": "A" if j % 2 == 0 else "B"},
                }
            )
    assert actual == expected
    assert rng.bit_generator.state == reference.bit_generator.state


@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    tuple(case for case in INTEGRATION_CASES if case.law == "bernoulli"),
    ids=lambda case: case.law,
)
def test_bounded_public_family_and_interval_replay(case: IntegrationCase):
    """Fixed raw witnesses preserve points and outward selected intervals."""
    require_integration_budget(INTEGRATION_CASES)
    reg = registration(
        case.law,
        cells=(SequentialCell(metric="outcome", group_id="treatment", family=True),),
    )
    reg = SequentialRegistration.model_validate({**reg.model_dump(), "q": F(1, 40)})
    spec = AlwaysValid(registration=reg)
    snapshot = None
    previous_repeat = 0
    for repeat in case.repeats:
        fresh = repeat - previous_repeat
        snapshot = capture_sequential_snapshot(
            reg,
            records(
                case.control * fresh,
                case.treatment * fresh,
                offset=len(case.control) * previous_repeat,
            ),
            source_id=reg.source_id,
            definitions_id=reg.definitions_id,
            finalized=True,
            previous=snapshot,
            append=snapshot is not None,
        )
        previous_repeat = repeat
        bundle = estimate_sequential(snapshot, spec)
        row = bundle.results[0]
        result = row.require_sequential_result()
        assert row.require_lift().value == float(case.ratio - 1)
        assert result.point_reason is None and result.certificate.status == "finite"
        bounds = result.bounds
        assert bounds.status == "interval" and not bounds.empty
        if bounds.lower is not None:
            assert bounds.lower < case.ratio
        else:
            assert bounds.lower_certificate.reason
        if bounds.upper is not None:
            assert case.ratio < bounds.upper
        else:
            assert bounds.upper_certificate.reason
    cells = [(sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)]
    outcome = select_sequential_family(cells, reg.q, spec, F(1, 20), computation=bundle)
    assert outcome.n_family == 1
    if case.law == "gaussian_ratio":
        # This fixed NIW witness does not cross the stricter family threshold.
        assert not outcome.selected and outcome.fcr_alpha is None
        return
    assert set(outcome.selected) == {cells[0][0]}
    assert outcome.fcr_alpha == F(1, 40)
    assert outcome.fcr_alpha is not None
    selected = reinvert_selected(row, outcome.fcr_alpha, ceiling=outcome.fcr_alpha)
    result = selected.require_sequential_result()
    assert result.checkpoint == row.require_sequential_result().checkpoint
    assert result.point_reason is None
    assert selected.require_lift().value == row.require_lift().value
    bounds = result.bounds
    assert bounds.alpha == outcome.fcr_alpha and bounds.status == "interval"
    parent = row.require_sequential_result().bounds
    if parent.lower is None:
        assert bounds.lower is None
    else:
        assert bounds.lower is None or bounds.lower <= parent.lower < case.ratio
    if parent.upper is None:
        assert bounds.upper is None
    else:
        assert bounds.upper is None or case.ratio < parent.upper <= bounds.upper
    if bounds.lower is not None and bounds.upper is not None:
        assert parent.lower is not None and parent.upper is not None
        assert parent.upper - parent.lower <= bounds.upper - bounds.lower
