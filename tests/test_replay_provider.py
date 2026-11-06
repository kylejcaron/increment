"""A run's recorded primitive answers replay that run through the provider seam, warehouse-free."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from increment import View
from increment.errors import CapabilityError
from tests.evidence_fixtures.recording import (
    CaptureManifest,
    Recording,
    assert_same_recording,
    frozen_manifests,
    load_recording,
    record_journey,
    record_run,
    replay,
    replay_journey,
    write_recording,
)
from tests.warning_codes import allow_warning_codes

# The scenario declares CUPED covariates its default decision method does not use.
_CUPED_ADVISORY = "analysis.cuped.covariate_unused"
_PILOT = ("audit-retention-breakout-cohorts-single-segment", "from_definitions")


def _pilot_manifest() -> CaptureManifest:
    (manifest,) = (
        item for item in frozen_manifests() if (item.scenario_id, item.ingress) == _PILOT
    )
    return manifest


def _rows(readout: Any) -> list[dict[str, Any]]:
    return [row.model_dump() for row in readout]


@allow_warning_codes(_CUPED_ADVISORY)
def test_replayed_journey_yields_the_pilot_rows():
    """Mature d7 retention in one store: 5 of 20 control and 10 of 20 treatment units return."""
    manifest = _pilot_manifest()
    pooled_view, breakout_view = manifest.scopes()

    result = replay_journey(manifest)
    pooled: list[Any] = list(result.readout(pooled_view))
    segments: list[Any] = list(result.readout(breakout_view))

    assert {(row.metric, row.dimension_value) for row in segments} == {("d7_retention", "s0")}
    # The one segment holds every unit, so each segment row is its pooled row.
    assert len(segments) == len(pooled)
    for segment, total in zip(segments, pooled, strict=True):
        assert segment.lift is not None and total.lift is not None
        assert segment.group_id == total.group_id
        assert segment.lift.value == pytest.approx(1.0)
        assert segment.lift.lb is not None and segment.lift.ub is not None
        assert segment.lift.value == pytest.approx(total.lift.value)
        assert segment.lift.lb == pytest.approx(total.lift.lb)
        assert segment.lift.ub == pytest.approx(total.lift.ub)


def test_replay_refuses_a_primitive_outside_the_recording():
    recording = load_recording(_pilot_manifest())

    with pytest.raises(CapabilityError) as refused:
        replay(recording, (View(view="daily"),))

    assert refused.value.code == "evidence.request_unsupported"
    assert refused.value.context["constructor"] == "ReplayProvider"


def test_recording_round_trips_through_the_codec():
    recording = load_recording(_pilot_manifest())

    assert_same_recording(recording, Recording.from_payload(recording.to_payload()))


@pytest.mark.integration
@allow_warning_codes(_CUPED_ADVISORY)
def test_scoped_replay_survives_closing_the_warehouse():
    manifest = _pilot_manifest()
    # The scenario's DuckDB connection is closed before the recording is returned.
    recording, live = record_run(manifest)

    replayed = replay(recording, manifest.scopes())

    for view in manifest.scopes():
        assert _rows(replayed.readout(view)) == _rows(live.readout(view))


def test_two_manifests_with_different_views_keep_separate_files(tmp_path):
    frozen = load_recording(_pilot_manifest())
    pooled_only = replace(frozen.manifest, views=frozen.manifest.views[:1])

    for manifest in (frozen.manifest, pooled_only):
        recorded = Recording(
            manifest, providers=frozen.providers, scopes=frozen.scopes, records=frozen.records
        )
        write_recording(recorded, tmp_path)

    assert set(frozen_manifests(tmp_path)) == {frozen.manifest, pooled_only}


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.exhaustive
@pytest.mark.parametrize("manifest", frozen_manifests(), ids=CaptureManifest.file_name)
@allow_warning_codes(_CUPED_ADVISORY)
def test_frozen_recording_matches_a_fresh_recording(manifest):
    assert_same_recording(load_recording(manifest), record_journey(manifest))
