"""The Explore interaction fragments keep the selected pair and its stored evidence together."""

from __future__ import annotations

import re
from html import unescape

import pytest

pytest.importorskip("coeftable")
pytest.importorskip("marimo")

from increment.semantics.interaction_scan import (
    InteractionScan,
    PairDetail,
    ScanReason,
    ScanRow,
)
from increment.semantics.overlap import CellId, OverlapCell
from tests.test_interaction_scan_contract import (
    SAVE,
    declared_inventory,
    freeze,
    inventory_entry,
    observed_inventory,
)

pytestmark = pytest.mark.integration

_NAME = "<script>same & name</script>"
_CELLS: tuple[CellId, ...] = ("00", "10", "01", "11")


def _binary_cell(cell: CellId, n: int, k: int | None) -> OverlapCell:
    return OverlapCell(
        metric="save",
        cell=cell,
        n_assigned=n,
        n_observed=n,
        n_missing=0,
        n_pending=0,
        n_success=k,
        anchor=0.0 if n else None,
        mean_delta=k / n if k is not None and n else None,
        m2=k * (n - k) / n if k is not None and n else None,
    )


@pytest.fixture(scope="module")
def _scan() -> InteractionScan:
    first = inventory_entry("cards").model_copy(update={"display_name": _NAME})
    second = inventory_entry("tips")
    second = second.model_copy(
        update={
            "display_name": _NAME,
            "identity": second.identity.model_copy(update={"revision": "r2"}),
        }
    )
    scale = SAVE.model_copy(update={"name": "revenue", "unit": "USD/account", "kind": "mean"})
    plan = freeze(declared_inventory(first, second), metrics=(SAVE, scale))
    cells = (
        (
            *(
                _binary_cell(cell, 10_000, k)
                for cell, k in zip(_CELLS, (3_100, 3_400, 3_000, 2_800), strict=True)
            ),
            OverlapCell(
                metric="revenue",
                cell="00",
                n_assigned=2,
                n_observed=2,
                n_missing=0,
                n_pending=0,
                n_success=None,
                anchor=25.0,
                mean_delta=2.5,
                m2=12.5,
            ),
        ),
        (
            _binary_cell("00", 50, 0),
            _binary_cell("10", 127, 1).model_copy(
                update={
                    "n_assigned": 131,
                    "n_missing": 4,
                    "anchor": None,
                    "mean_delta": None,
                    "m2": None,
                }
            ),
            _binary_cell("01", 60, 0),
            _binary_cell("11", 0, None),
        ),
    )
    details = tuple(
        PairDetail(
            candidate=candidate,
            status="captured",
            reasons=(),
            focal_side=side,
            evidence=None,
            eligibility_counts=None,
            gate_counts=None,
            delivery_counts=(),
            cells=groups,
        )
        for candidate, side, groups in zip(plan.candidates, ("A", "B"), cells, strict=True)
    )
    rates = iter(
        (
            (0.03, 0.01163917, 0.04836083),
            (-0.02, -0.03779279, -0.00220721),
            (-0.05, -0.07556624, -0.02443376),
        )
    )
    amounts = iter(((2.0, -1.0, 3.0), (6.0, -1.6e308, 1.6e308), (4.0, -2.0, 12.0)))
    rows = []
    for key in plan.family_members:
        if key.metric == "save" and key.pair.candidate == first.identity:
            estimate, lower, upper = next(rates)
            availability, reasons, decision = "decided", (), "detected"
        elif key.metric == "revenue" and key.pair.candidate == first.identity:
            estimate, lower, upper = next(amounts)
            availability = "fitted_small" if key.contrast == "focal_given_other_on" else "decided"
            reasons = (
                (ScanReason(code="small_cell", cell="11"),)
                if availability == "fitted_small"
                else ()
            )
            decision = None if reasons else "none_detected"
        elif key.metric == "revenue" and key.contrast == "interaction":
            estimate, lower, upper = -5.0, -7.0, -3.0
            availability, reasons, decision = "decided", (), "detected"
        elif key.metric == "save" and key.contrast == "focal_given_other_off":
            estimate, lower, upper = 0.0, None, None
            availability, reasons, decision = (
                "point_only",
                (ScanReason(code="zero_cell_variance", cell="00"),),
                None,
            )
        else:
            estimate = lower = upper = decision = None
            availability = "unavailable"
            reasons = (
                ScanReason(code="empty_cell", cell="11"),
                ScanReason(code="missing_outcomes", cell="10"),
            )
        rows.append(
            ScanRow(
                key=key,
                estimate=estimate,
                se=None,
                df=None,
                lower=lower,
                upper=upper,
                availability=availability,
                reasons=reasons,
                cells=(),
                evidence_sha256=None,
                decision=decision,
            )
        )
    return InteractionScan(
        plan=plan,
        capture_id="a" * 64,
        observed_inventory=observed_inventory(plan.inventory),
        coverage="complete",
        rows=tuple(rows),
        details=details,
    )


def _text(markup: str) -> str:
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", markup)).split())


def _accessible_numbers(markup: str) -> str:
    exposed = re.sub(r"<svg\b[^>]*>.*?</svg>", "", markup, flags=re.S)
    assert not re.search(r'\baria-hidden="true"|<[^>]+\shidden[\s=>]', exposed)
    return _text(exposed)


def _candidate_row(markup: str, pair_id: str) -> str:
    (row,) = [
        row
        for row in re.findall(r"<tr\b[^>]*>.*?</tr>", markup, re.S)
        if f'data-inc-pair-id="{pair_id}"' in row
    ]
    assert re.search(r'<button\b[^>]*\btype="button"', row)
    return row


def _assignment_rows(markup: str) -> list[tuple[str, ...]]:
    return [
        tuple(map(_text, re.findall(r"<(?:th|td)\b[^>]*>(.*?)</(?:th|td)>", row, re.S)))
        for row in re.findall(r'<tr data-inc-cell="(?:00|10|01|11)">(.*?)</tr>', markup, re.S)
    ]


def test_same_name_buttons_open_their_own_four_cell_evidence(_scan):
    from increment.dashboard._interaction_html import interaction_tables

    fragments = interaction_tables(_scan, metric_key="save")
    first, second = (detail.candidate.key.pair_id for detail in _scan.details)
    assert tuple(fragments.details) == (first, second)
    for pair, revision in ((first, "r1"), (second, "r2")):
        row = _candidate_row(fragments.overview, pair)
        assert _NAME in _text(row)
        assert "<script>" not in row
        assert revision in _text(row)
    assert [row[1:4] for row in _assignment_rows(fragments.details[first])] == [
        ("10,000", "10,000", "3,100"),
        ("10,000", "10,000", "3,400"),
        ("10,000", "10,000", "3,000"),
        ("10,000", "10,000", "2,800"),
    ]
    second_rows = _assignment_rows(fragments.details[second])
    # Source A/B is transposed for the second pair, not the actual counts.
    assert [row[1:4] for row in second_rows] == [
        ("50", "50", "0"),
        ("60", "60", "0"),
        ("131", "127", "1"),
        ("0", "0", "—"),
    ]
    assert "missing" in second_rows[2][-1].lower()
    assert "assigned" in second_rows[-1][-1].lower()
    assert {"Focal experiment", "Other experiment", "Metric", "Outcome window", "Target"} <= set(
        re.findall(r"<dt>(.*?)</dt>", fragments.details[second])
    )


def test_stored_rate_numbers_and_absence_are_visible_in_both_levels(_scan):
    from increment.dashboard._interaction_html import interaction_tables

    fragments = interaction_tables(_scan, metric_key="save")
    first, second = tuple(fragments.details)
    first_row = _accessible_numbers(_candidate_row(fragments.overview, first))
    first_detail = _accessible_numbers(fragments.details[first])
    assert "percentage points" in first_detail and "percentage points" in _text(fragments.overview)
    for number in ("−5", "−7.56", "−2.44"):
        assert number in first_row and number in first_detail
    for number in ("+3", "+1.16", "+4.84", "−2", "−3.78", "−0.22"):
        assert number in first_detail
    unavailable = _candidate_row(fragments.overview, second)
    assert "Couldn’t estimate · nobody was assigned both on" in _text(unavailable)
    assert "<svg" not in unavailable
    detail = fragments.details[second]
    assert "Estimate only · no interval" in _text(detail)
    point_only = next(
        row for row in re.findall(r"<tr\b[^>]*>.*?</tr>", detail, re.S) if "Estimate only" in row
    )
    assert re.search(r">0</span>", point_only)
    assert "<svg " in point_only and "<polygon" not in point_only


def test_relative_pair_shows_its_own_lifts_and_supported_metric_direction(_scan):
    from increment.dashboard._interaction_html import interaction_tables

    (first, second) = tuple(detail.candidate.key.pair_id for detail in _scan.details)
    fragments = interaction_tables(_scan, metric_key="save", relative=True)
    comparison = _accessible_numbers(fragments.details[first])
    assert "Change in relative lift" in comparison
    assert re.search(r"\+9\.7% \[\+[\d.]+%, \+[\d.]+%\]", comparison)
    assert "−6.7%" in comparison
    assert "−16.3 pp [" in comparison and "−16.3%" not in comparison
    detected = _candidate_row(fragments.overview, first)
    overview = _accessible_numbers(detected)
    assert "−16.3 pp [" in overview and "%" not in overview
    assert re.search(r'<rect\b[^>]+\bfill="var\(--inc-dashboard-theme-accent,', detected)
    assert "Favorable" in comparison and "Unfavorable" in comparison
    assert "Lower with other on" in comparison
    assert "baseline is zero or negative" in _accessible_numbers(fragments.details[second])


def test_outlying_stored_bound_shares_an_unclipped_axis_and_small_fit_keeps_interval(_scan):
    from increment.dashboard._interaction_html import interaction_tables

    fragments = interaction_tables(_scan, metric_key="revenue")
    first, second = tuple(fragments.details)
    overview, detail = fragments.overview, fragments.details[first]
    assert "USD/account" in _text(overview) and "USD/account" in _text(detail)
    assert "None detected" in _text(_candidate_row(overview, first))
    assert "insufficient evidence" in _text(detail).lower()
    assert "−1.6e+308" in _text(detail) and "+1.6e+308" in _text(detail)
    assert "−5 [−7, −3]" in _accessible_numbers(_candidate_row(overview, second))
    assert "−5 [−7, −3]" in _accessible_numbers(fragments.details[second])
    assert _assignment_rows(detail)[0][1:4] == ("2", "2", "27.5")
    axes = [re.findall(r"<text\b[^>]*>([^<]+)</text>", fragment) for fragment in (overview, detail)]
    assert axes[0] and axes[0] == axes[1]
    assert "1.6e+308" in " ".join(axes[0])
    for fragment in (overview, detail):
        assert "<line " in fragment and "<rect " in fragment
        assert re.search(r"\b(?:nan|inf)\b|<polygon|<linearGradient", fragment, re.I) is None
        assert re.search(r'<svg\b[^>]*\baria-hidden="true"', fragment)
