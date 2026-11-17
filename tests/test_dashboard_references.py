"""Captured-result selection must not substitute another source or view."""

import json
import re
from dataclasses import replace
from html import unescape

import pytest

from increment.dashboard import render_dashboard
from increment.estimation.advisory import ReadoutAdvisory
from increment.estimation.readout_types import CellKey
from increment.results import Estimate, LiftEstimate, LiftEstimates
from increment.tables import estimates_to_readout
from tests.dashboard_snapshots import (
    COUNTRY,
    captured_snapshot,
    experiment_for,
    frozen,
    snapshot,
)


def _payload(value, *, experiment=None):
    dashboard = render_dashboard(
        experiment if experiment is not None else experiment_for(value), snapshot=value
    )
    (embedded,) = re.findall(
        r'<script type="application/json" id="dashboard-data">(.*?)</script>',
        dashboard.widget.get_state()["document"],
        re.S,
    )
    return json.loads(embedded)


def _view(references, coordinates):
    return next(
        identities
        for key, identities in references["views"].items()
        if json.loads(key) == coordinates
    )


def _source_collection(base, source, *, lower, alpha, metric=None):
    original = base.readout_collection
    row = original[0].model_copy(
        update={
            "source_snapshot_id": source,
            "metric": metric or original[0].metric,
            "lift": Estimate(value=0.1, lb=lower, ub=0.4, level=1 - alpha, alpha=alpha),
        }
    )
    cell = CellKey.from_row(row)
    source_scope = original.metadata.scope.by_source["presentation"].model_copy(
        update={"source_snapshot_id": source, "cells": (cell,), "decision_cells": (cell,)}
    )
    metadata = original.metadata.model_copy(
        update={
            "scope": original.metadata.scope.model_copy(
                update={
                    "snapshot_id": source,
                    "cells": (cell,),
                    "decision_cells": (cell,),
                    "by_source": {source: source_scope},
                }
            ),
            "cells": (
                original.metadata.cells[0].model_copy(
                    update={"cell": cell, "source_snapshot_id": source}
                ),
            ),
        }
    )
    return LiftEstimates((row,), metadata=metadata)


@pytest.mark.integration
def test_reference_selection_keeps_source_and_view_specific_evidence():
    # A metric-only or cell-only index would open the first source/view instead.
    base = snapshot(explore={})
    headline = _source_collection(base, "source-a", lower=-0.1, alpha=0.05)
    other = _source_collection(base, "source-b", lower=-0.2, alpha=0.05)
    corrected = _source_collection(base, "source-b", lower=-0.3, alpha=0.01)
    captures = {
        ("assigned", "cumulative_lift", None, complete, None): rows
        for complete, rows in ((False, other), (True, corrected))
    }
    value = captured_snapshot(base, readout=headline, explore=captures)
    references = _payload(value)["references"]
    identities = [_view(references, ["explore", *key])[0] for key in captures]
    selected = [references["results"][identity] for identity in identities]
    assert [item["sourceId"] for item in selected] == ["source-b", "source-b"]
    records = [json.loads(item["recordJson"]) for item in selected]
    readings = estimates_to_readout(
        [LiftEstimate.model_validate(item["reading"]) for item in records]
    )
    assert [row["lower"] for row in readings] == pytest.approx([-0.2, -0.3])
    assert [row["coverage_alpha"] for row in readings] == pytest.approx([0.05, 0.01])
    assert identities[0] != identities[1]
    for item in records:
        assert item["record"] is not None
        assert item["record"]["source_snapshot_id"] == "source-b"


@pytest.mark.integration
def test_missing_reference_metadata_never_borrows_another_source_check():
    # A fallback to headline metadata would turn unknown checks into recorded checks.
    base = snapshot(explore={})
    rows = LiftEstimates((base.estimates[0].model_copy(update={"source_snapshot_id": None}),))
    key = ("assigned", "cumulative_lift", None, False, None)
    value = captured_snapshot(base, explore={key: rows})
    references = _payload(value)["references"]
    (identity,) = _view(references, ["explore", *key])
    reference = references["results"][identity]
    raw = json.loads(reference["recordJson"])
    assert reference["sourceId"] is None
    assert raw["record"] is None
    assert reference["groupIds"] == []
    assert raw["reading"] is not None
    assert "Checks were not recorded" in reference["readingHtml"]
    assert "missing evidence is not a pass" in reference["readingHtml"]


@pytest.mark.integration
@pytest.mark.parametrize("view", ["daily_values", "cumulative_lift"])
def test_trajectory_references_keep_original_arm_and_segment_identity(view):
    saved = frozen("explore")
    base = saved.snapshot
    key = ("assigned", view, base.primary_metric, False, COUNTRY)
    capture = base.explore[key]
    rows = capture.load()
    payload = _payload(
        replace(base, explore={key: capture}, overview=None), experiment=saved.experiment
    )
    references = payload["references"]
    identities = _view(
        references, ["explore", "assigned", view, base.primary_metric, False, list(COUNTRY)]
    )
    chosen = next(row for row in rows if row.dimension_value == "US")
    selected = [
        identity
        for identity in identities
        if json.loads(references["results"][identity]["recordJson"])["reading"]
        == chosen.model_dump(mode="json")
    ]
    assert len(selected) == 1
    reference = references["results"][selected[0]]
    assert reference["cell"] == CellKey.from_row(chosen).model_dump(mode="json")
    assert chosen.source_snapshot_id is not None
    assert reference["sourceId"] == chosen.source_snapshot_id
    scope = next(
        name
        for name, declaration in payload["scopes"].items()
        if (declaration["source"], declaration["dimension"]) == COUNTRY
    )
    packet = payload["explore"][scope][base.primary_metric][view]
    linked = {
        identity
        for encoded in re.findall(r'data-inc-result-refs="([^"]*)"', packet["html"])
        for identity in json.loads(unescape(encoded))
    }
    assert selected[0] in linked


@pytest.mark.integration
def test_corrected_overview_keeps_its_original_capture_checks():
    template = snapshot(explore={})
    metric = template.metrics[0].model_copy(update={"name": "overview_only"})
    base = snapshot(explore={}, offered_metrics=(metric,), exploratory_metrics=(metric,))
    original = _source_collection(
        base, "overview-source", lower=-0.2, alpha=0.05, metric=metric.name
    )
    corrected = original[0].model_copy(
        update={"lift": Estimate(value=0.1, lb=-0.3, ub=0.4, level=0.99, alpha=0.01)}
    )
    key = ("assigned", "overview_whole", None, False, None)
    value = captured_snapshot(base, explore={key: original}, overview=(corrected,))
    references = _payload(value)["references"]
    reference = next(
        references["results"][identity]
        for identity in _view(references, ["overview", None, None])
        if references["results"][identity]["sourceId"] == "overview-source"
    )
    raw = json.loads(reference["recordJson"])
    assert raw["record"] is not None
    assert raw["record"]["source_snapshot_id"] == "overview-source"
    assert raw["reading"] is not None
    (reading,) = estimates_to_readout([LiftEstimate.model_validate(raw["reading"])])
    assert reading["coverage_alpha"] == pytest.approx(0.01)
    source = next(
        group["source"]
        for identity in reference["groupIds"]
        for group in (json.loads(references["groups"][identity]["recordJson"]),)
        if "source" in group
    )
    assert source["integrity"][0]["code"] == "integrity.allocation_missing_or_nonconstant"


@pytest.mark.integration
def test_diagnostic_counts_union_result_identities_across_codes_and_views():
    base = snapshot(explore={})
    advisories = tuple(
        ReadoutAdvisory(code=code, context={"threshold": 0.3}, message="Recorded caution.")
        for code in (
            "estimation.engine.ratio_denominator_skew",
            "estimation.engine.ratio_denominator_precision",
        )
    )

    def cautioned(source, lower=-0.2, codes=advisories):
        rows = _source_collection(base, source, lower=lower, alpha=0.05)
        metadata = rows.metadata.model_copy(
            update={
                "cells": tuple(
                    record.model_copy(update={"advisories": codes})
                    for record in rows.metadata.cells
                ),
            }
        )
        return LiftEstimates(rows, metadata=metadata)

    original = cautioned("source-a")
    captures = {}

    def retain(rows, metric, complete):
        key = ("assigned", "cumulative_lift", metric, complete, None)
        captures[key] = rows

    def affected(shape_count, precision_count):
        value = captured_snapshot(base, readout=original, explore=captures)
        payload = _payload(value)
        references = payload["references"]
        by_code = {
            advisory.code: {
                identity
                for identity, reference in references["results"].items()
                if advisory.code in reference["diagnosticCodes"]
            }
            for advisory in advisories
        }
        assert {code: len(identities) for code, identities in by_code.items()} == {
            advisories[0].code: shape_count,
            advisories[1].code: precision_count,
        }
        for code, identities in by_code.items():
            for identity in identities:
                raw = json.loads(references["results"][identity]["recordJson"])
                assert raw["record"] is not None
                assert any(advisory["code"] == code for advisory in raw["record"]["advisories"])
        buttons = re.findall(
            r'<button[^>]*class="diagnostic-select"[^>]*data-result-refs="([^"]*)"[^>]*>'
            r"(.*?)</button>",
            payload["health"],
        )
        affected_ids = set().union(*by_code.values())
        assert [int(re.findall(r"\d+", label)[0]) for _, label in buttons] == [len(affected_ids)]
        assert set(json.loads(unescape(buttons[0][0]))) == affected_ids
        return references

    retain(original, None, False)
    retain(original, None, True)
    partial = LiftEstimates(
        original,
        metadata=original.metadata.model_copy(
            update={"partial": True, "partial_reason": ("filter",)}
        ),
    )
    retain(partial, original[0].metric, False)
    first = affected(1, 1)
    assert len(first["results"]) == 1
    reference = next(iter(first["results"].values()))
    source = next(
        group
        for identity in reference["groupIds"]
        for group in (json.loads(first["groups"][identity]["recordJson"]),)
        if "source" in group
    )
    captures_by_view = {
        tuple(json.loads(key)): captures for key, captures in source["captures_by_view"].items()
    }
    assert captures_by_view[("readout",)][0]["partial"] is False
    partial_key = ("explore", "assigned", "cumulative_lift", original[0].metric, False, None)
    assert captures_by_view[partial_key][0]["partial"] is True
    assert captures_by_view[partial_key][0]["partial_reason"] == ["filter"]
    retain(cautioned("source-b", codes=(advisories[1],)), original[0].metric, False)
    assert len(affected(1, 2)["results"]) == 2
    retain(cautioned("source-a", lower=-0.4), original[0].metric, True)
    assert len(affected(2, 3)["results"]) == 3
