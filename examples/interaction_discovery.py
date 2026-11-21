"""Find concurrent experiments that changed the effect of a synthetic search layout.

Run ``uv run --extra demo --extra dashboard python -m examples.interaction_discovery interaction.html``.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, TypedDict

import duckdb
import ibis

from increment import from_definitions, load_definitions
from increment.artifacts import SourceSnapshotEvidence
from increment.dashboard import DashboardConfig, dashboard_html, prepare_dashboard
from increment.overlap import (
    ExperimentIdentity,
    FocalStudy,
    InteractionDuckDBSource,
    InteractionInventory,
    InventoryEntry,
    ObservedInventory,
    OverlapCapture,
    OverlapClaim,
    OverlapEvidence,
    OverlapFactor,
    OverlapFeed,
    OverlapLaw,
    OverlapMetric,
    OverlapPlan,
    SourceFact,
    SourceGenerationManifest,
    StudyFeedFact,
    bind_interaction_scan,
    capture_interaction_scan,
    capture_overlap_warehouse,
    freeze_interaction_scan,
    interaction_duckdb_fact,
    interaction_metric_sha256,
)
from increment.query.overlap import bind_overlap_warehouse

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ibis.backends.duckdb import Backend

    from increment.dashboard import DashboardSnapshot
    from increment.overlap import InteractionScanPlan

_FOCAL = "search_layout"
_SAVE = "save_rate"
_SHARE = "share_rate"
_UNIT = "account_id"
_ENTRY = datetime(2026, 1, 1, tzinfo=UTC)
_REGISTERED = _ENTRY - timedelta(days=1)
_START = _ENTRY + timedelta(days=1)
_END = _START + timedelta(days=2)
_CUTOFF = _END + timedelta(hours=12)
_SOURCE = "synthetic-assignment-archive"
_GENERATION = "synthetic-search-2026-01-04"
_ALLOCATION = {"off": 0.5, "on": 0.5}

# Each candidate overlaps a different eligible cohort of the focal experiment.
# These are synthetic realized counts, not quotas in the declared assignment law.
# A rare tips assignment makes an empty 11 group plausible even with a large off group.
# Each tuple lists (assigned, saved) in cell order 00, 10, 01, 11.
_CANDIDATES = (
    (
        "product_cards",
        "Product cards",
        1,
        ((10_000, 3_100), (10_000, 3_400), (10_000, 3_000), (10_000, 2_800)),
    ),
    (
        "quick_filters",
        "Quick filters",
        1,
        ((10_000, 3_100), (10_000, 3_200), (10_000, 3_000), (10_000, 3_100)),
    ),
    ("search_tips", "Search tips", 1_999, ((10_000, 3_100), (10_000, 3_400), (6, 2), (0, 0))),
)
_FeedRole = Literal["roster", "decisions", "deliveries", "outcomes"]
_FEED_WINDOWS: tuple[tuple[_FeedRole, datetime, datetime], ...] = (
    ("roster", _ENTRY, _START),
    ("decisions", _ENTRY, _START),
    ("deliveries", _ENTRY, _END),
    ("outcomes", _START, _END),
)


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _definitions(path: Path, *, triggered: bool) -> None:
    trigger = "    trigger: opened_search\n" if triggered else ""
    path.write_text(
        f"""dialect: duckdb
fact_sources:
  - name: event_log
    sql: SELECT * FROM analytics.event_log
    timestamp_column: event_at
    entities: [{_UNIT}]
    facts:
      - {{name: assignment, column: null}}
      - {{name: search_opened, column: null}}
      - {{name: saved, column: null}}
      - {{name: shared, column: null}}
exposures:
  - {{name: assigned, fact: assignment}}
  - {{name: opened_search, fact: search_opened}}
metrics:
  - {{name: {_SAVE}, type: conversion, entity: {_UNIT}, fact: saved, window_days: 2}}
  - {{name: {_SHARE}, type: conversion, entity: {_UNIT}, fact: shared, window_days: 2}}
experiments:
  - name: {_FOCAL}
    exposure: assigned
{trigger}    unit: {_UNIT}
    start: {_START.date()}
    end: {_START.date()}
    observation_end: {(_END - timedelta(days=1)).date()}
    control_group: 'off'
    allocation: {{'off': 0.5, 'on': 0.5}}
    plan:
      primary: {_SAVE}
""",
        encoding="utf-8",
    )


def _regime(name: str, label: str, off_weight: int) -> tuple[InventoryEntry, OverlapFactor, str]:
    factor = OverlapFactor(
        target=name,
        experiment=name,
        revision="v1",
        off="off",
        on="on",
        off_weight=off_weight,
        on_weight=1,
        arm_draw_id=f"{name}-arm-draw",
        layer=None,
    )
    archive = factor.model_dump_json()
    entry = InventoryEntry(
        identity=ExperimentIdentity(target=name, experiment_id=name, revision=factor.revision),
        archive_sha256=_digest(archive),
        unit_namespace=_UNIT,
        active_from=_ENTRY,
        active_until=_END,
        arms=(factor.off, factor.on),
        display_name=label,
    )
    return entry, factor, archive


class _RosterFields(TypedDict):
    universe_id: str
    source_id: str
    digest: str
    covered_from: datetime
    complete_through: datetime
    coverage: Literal["complete"]
    entries: tuple[InventoryEntry, ...]


def _roster_fields(entries: tuple[InventoryEntry, ...]) -> _RosterFields:
    schedule = [
        item.model_dump(mode="json")
        for item in sorted(entries, key=lambda entry: entry.identity.sort_key)
    ]
    return {
        "universe_id": "search-changes",
        "source_id": _SOURCE,
        "digest": _digest(json.dumps(schedule, sort_keys=True)),
        "covered_from": _ENTRY,
        "complete_through": _END,
        "coverage": "complete",
        "entries": entries,
    }


_WAREHOUSE = """
    CREATE TEMP TABLE joint_units AS
    SELECT study_id, concat_ws(':', study_id, cell, slot) AS unit_id,
           cell % 2 = 1 AS focal_on, cell >= 2 AS other_on, slot,
           slot < saves AS saved
    FROM cell_counts, LATERAL range(assigned) AS units(slot);

    CREATE TABLE analytics.roster AS
    SELECT study_id, unit_id, entry_at, true AS eligible_a, true AS eligible_b,
           entry_at AS eligibility_at_a, entry_at AS eligibility_at_b,
           true AS gate_a, true AS gate_b,
           entry_at + INTERVAL '1 hour' AS gate_at_a,
           entry_at + INTERVAL '2 hours' AS gate_at_b
    FROM joint_units, fixed_look;

    CREATE TABLE analytics.decisions AS
    SELECT study_id, unit_id, factor->>'target' AS target, window_start AS decided_at,
           factor->>'revision' AS revision, sha256(archive_json) AS archive_sha256,
           factor->>'experiment' AS selected_experiment, 'assigned' AS status,
           CASE WHEN arm_on THEN 'on' ELSE 'off' END AS arm,
           (factor->>CASE WHEN arm_on THEN 'on_weight' ELSE 'off_weight' END)::BIGINT
               AS assigned_weight
    FROM joint_units, fixed_look, analytics.assignment_archive,
         LATERAL (SELECT archive_json::JSON AS factor),
         LATERAL (SELECT CASE WHEN experiment_id = focal THEN focal_on ELSE other_on END AS arm_on)
    WHERE experiment_id IN (focal, study_id);

    CREATE TABLE analytics.deliveries AS
    SELECT study_id, concat_ws(':', unit_id, target) AS event_id, unit_id, target,
           selected_experiment AS experiment, revision, arm AS delivered_arm,
           decided_at AS delivered_at
    FROM analytics.decisions;

    CREATE TABLE analytics.event_log AS
    SELECT unit_id AS account_id, decided_at AS event_at, 'assignment' AS event,
           arm AS group_id, selected_experiment AS experiment_id
    FROM analytics.decisions, fixed_look
    WHERE selected_experiment = focal
    UNION ALL
    SELECT unit_id, window_start + INTERVAL '1 hour', 'search_opened', NULL, focal
    FROM joint_units, fixed_look WHERE slot % 2 = 0
    UNION ALL
    SELECT unit_id, window_end - INTERVAL '1 hour', 'saved', NULL, focal
    FROM joint_units, fixed_look WHERE saved
    UNION ALL
    SELECT unit_id, window_end - INTERVAL '1 hour', 'shared', NULL, focal
    FROM joint_units, fixed_look WHERE slot % 5 = 0;

    CREATE TABLE analytics.outcomes AS
    SELECT study_id, unit_id, metric, definition_sha256, window_start, window_end,
           (events.account_id IS NOT NULL)::DOUBLE AS value, cutoff AS finalized_at
    FROM joint_units CROSS JOIN fixed_look
    LEFT JOIN analytics.event_log AS events ON events.account_id = unit_id AND events.event = 'saved';
"""


def _write_warehouse(
    path: Path,
    regimes: tuple[tuple[InventoryEntry, OverlapFactor, str], ...],
    metric: OverlapMetric,
) -> None:
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE SCHEMA analytics")
        con.execute("""CREATE TABLE analytics.assignment_archive (
            experiment_id VARCHAR, entry_json VARCHAR, archive_json VARCHAR
        )""")
        con.executemany(
            "INSERT INTO analytics.assignment_archive VALUES (?, ?, ?)",
            [
                (factor.experiment, entry.model_dump_json(), archive)
                for entry, factor, archive in regimes
            ],
        )
        con.execute("""CREATE TEMP TABLE cell_counts (
            study_id VARCHAR, cell BIGINT, assigned BIGINT, saves BIGINT
        )""")
        con.executemany(
            "INSERT INTO cell_counts VALUES (?, ?, ?, ?)",
            [
                (study, cell, n, saved)
                for study, _, _, cells in _CANDIDATES
                for cell, (n, saved) in enumerate(cells)
            ],
        )
        con.execute(
            """CREATE TEMP TABLE fixed_look AS SELECT ? AS focal, ? AS metric,
            ? AS definition_sha256, ? AS entry_at, ? AS window_start, ? AS window_end, ? AS cutoff""",
            [_FOCAL, metric.name, metric.definition_sha256, _ENTRY, _START, _END, _CUTOFF],
        )
        con.execute(_WAREHOUSE)
        con.execute(
            "CREATE TABLE main.increment_interaction_generation AS "
            "SELECT 1 AS version, CAST(? AS VARCHAR) AS generation_id",
            [_GENERATION],
        )
        con.execute("CHECKPOINT")


def _capture_pair(con: Backend, plan: OverlapPlan, catalog: str) -> OverlapEvidence:
    capture = OverlapCapture(
        generation_id="synthetic-v1",
        origin=plan.law.origin,
        law_sha256=plan.law_sha256,
        archive_sha256=plan.law.archive_sha256,
        cutoff=_CUTOFF,
        feeds=tuple(
            OverlapFeed(
                role=role,
                source_id=f"synthetic-{role}",
                version="v1",
                recipe_sha256=_digest(f"{role}-from-synthetic-joint-units-v1"),
                covered_from=covered_from,
                complete_through=complete_through,
            )
            for role, covered_from, complete_through in _FEED_WINDOWS
        ),
    )
    roster, decisions, deliveries, outcomes = (
        con.table(role, database=(catalog, "analytics")) for role, _, _ in _FEED_WINDOWS
    )
    bound = bind_overlap_warehouse(
        plan,
        capture=capture,
        roster=roster,
        decisions=decisions,
        deliveries=deliveries,
        outcomes=outcomes,
    )
    return capture_overlap_warehouse(
        plan,
        capture=bound,
        roster=roster,
        decisions=decisions,
        deliveries=deliveries,
        outcomes=outcomes,
    )


def _pair_plan(
    focal: FocalStudy, a: OverlapFactor, b: OverlapFactor, archive: str, metric: OverlapMetric
) -> OverlapPlan:
    assert focal.entry.archive_sha256 is not None
    return OverlapPlan(
        study_id=b.experiment,
        unit_namespace=_UNIT,
        population_id=focal.population_id,
        eligibility_rule_sha256=focal.eligibility_rule_sha256,
        enrollment_rule_sha256=focal.enrollment_rule_sha256,
        registered_at=_REGISTERED,
        entry_open=_ENTRY,
        start=_START,
        end=_END,
        target=focal.target,
        a=a,
        b=b,
        law=OverlapLaw(
            scheme="independent_unit_bernoulli",
            origin="external",
            archive_sha256=(focal.entry.archive_sha256, _digest(archive)),
            gate_basis="preassigned_roster",
            admission="both_guaranteed",
            rule_source_id=_SOURCE,
            rule_version="v1",
            serving_barrier_id="both-decisions-before-showing-search",
        ),
        metrics=(metric,),
        claims=(
            OverlapClaim(metric=metric.name, contrast="interaction", alternative="difference"),
        ),
        alpha=0.05,
    )


@dataclass(frozen=True, slots=True)
class Example:
    """The definitions reader and pair receipts from one synthetic archive."""

    source: InteractionDuckDBSource
    plan: InteractionScanPlan
    observed_inventory: ObservedInventory
    evidence_by_study: dict[str, OverlapEvidence]

    def prepare(
        self, *, show_second_metric: bool = False, replay_pairs: bool = False
    ) -> DashboardSnapshot:
        """Prepare the report with original or restored pair evidence."""
        pairs = self.evidence_by_study
        if replay_pairs:
            pairs = {
                study: OverlapEvidence.from_json(
                    pair.to_json(), expected_sha256=pair.evidence_sha256
                )
                for study, pair in pairs.items()
            }
        capture = capture_interaction_scan(
            self.plan, observed_inventory=self.observed_inventory, evidence_by_study=pairs
        )
        attachment = bind_interaction_scan(capture, source=self.source)
        return prepare_dashboard(
            self.source.experiment,
            self.source.reader,
            config=DashboardConfig(
                title="Search layout",
                source_label="Synthetic account history",
                expected_allocation=_ALLOCATION,
                exploratory_metrics=(_SHARE,) if show_second_metric else (),
            ),
            interactions=attachment,
        )


@contextmanager
def open_example(directory: Path, *, triggered: bool = False) -> Iterator[Example]:
    """Build one synthetic archive in an empty directory and open it read-only."""
    definitions = directory / "search.yaml"
    _definitions(definitions, triggered=triggered)
    metrics = tuple(
        OverlapMetric(
            name=metric.name,
            unit="proportion",
            definition_sha256=interaction_metric_sha256(metric),
            kind="binary_mean",
            role="primary" if metric.name == _SAVE else "secondary",
            better="higher",
        )
        for metric in load_definitions(definitions).metrics
    )
    primary = next(metric for metric in metrics if metric.name == _SAVE)
    focal_regime = _regime(_FOCAL, "Search layout", 1)
    candidates = tuple(_regime(name, label, weight) for name, label, weight, _ in _CANDIDATES)
    regimes = (focal_regime, *candidates)
    focal = FocalStudy(
        entry=focal_regime[0],
        population_id="accounts-eligible-for-both-changes",
        eligibility_rule_sha256=_digest("eligible accounts in each candidate's search cohort"),
        enrollment_rule_sha256=_digest("record both gates before either decision"),
        target="jointly_eligible",
    )
    inventory = InteractionInventory(
        version="schedule-v1",
        frozen_at=_REGISTERED,
        **_roster_fields(tuple(entry for entry, _, _ in regimes)),
    )
    plan = freeze_interaction_scan(
        inventory=inventory,
        focal=focal,
        metrics=metrics,
        entry_open=_ENTRY,
        start=_START,
        end=_END,
        cutoff=_CUTOFF,
        alpha=0.05,
    )
    pair_plans = tuple(
        _pair_plan(focal, focal_regime[1], factor, archive, primary)
        for _, factor, archive in candidates
    )

    database = directory / "search.duckdb"
    _write_warehouse(database, regimes, primary)
    con = ibis.duckdb.connect(database, read_only=True)
    try:
        [(catalog,)] = con.con.execute("SELECT current_database()").fetchall()
        observed_entries = tuple(
            InventoryEntry.model_validate_json(raw)
            for (raw,) in con.con.execute(
                "SELECT entry_json FROM analytics.assignment_archive"
            ).fetchall()
        )
        observed = ObservedInventory(
            version="observed-v1", cutoff=_CUTOFF, **_roster_fields(observed_entries)
        )
        receipts = {pair.study_id: _capture_pair(con, pair, catalog) for pair in pair_plans}
        experiment, reader = from_definitions(
            definitions,
            con,
            source_snapshot_evidence=SourceSnapshotEvidence(
                observation_cutoff_ts=_CUTOFF, complete_through_by_feed={"event_log": _END}
            ),
        )
        try:
            event_log = interaction_duckdb_fact(
                con.table("event_log", database=(catalog, "analytics"))
            )
            pair_feeds = tuple(
                StudyFeedFact(
                    study_id=study,
                    role=role,
                    fact=SourceFact(
                        coordinate=(catalog, "analytics", role),
                        content_sha256=receipt.capture.feed(role).version.removeprefix(
                            "duckdb-rows-v1:"
                        ),
                    ),
                )
                for study, receipt in receipts.items()
                for role, _, _ in _FEED_WINDOWS
            )
            with database.open("rb") as stream:
                dataset_sha256 = file_digest(stream, "sha256").hexdigest()
            manifest = SourceGenerationManifest(
                source_id=_SOURCE,
                generation_id=_GENERATION,
                dataset_sha256=dataset_sha256,
                cutoff=_CUTOFF,
                scan_plan_id=plan.plan_id,
                definition_sha256=experiment.hash(),
                focal_assignments=event_log,
                focal_outcomes=tuple((metric.name, event_log) for metric in metrics),
                pair_feeds=pair_feeds,
            )
            source = InteractionDuckDBSource(experiment, reader, manifest)
            yield Example(source, plan, observed, receipts)
        finally:
            reader.close()
    finally:
        con.disconnect()


def main() -> None:
    """Export the prepared dashboard without keeping its data source open."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="output HTML path")
    parser.add_argument(
        "--show-second-metric", action="store_true", help="show the offered share rate"
    )
    parser.add_argument(
        "--replay-pairs", action="store_true", help="restore the original format-1 pair JSON"
    )
    args = parser.parse_args()
    with TemporaryDirectory() as directory, open_example(Path(directory)) as example:
        experiment = example.source.experiment
        snapshot = example.prepare(
            show_second_metric=args.show_second_metric, replay_pairs=args.replay_pairs
        )
    args.output.write_text(dashboard_html(experiment, snapshot=snapshot), encoding="utf-8")


if __name__ == "__main__":
    main()
