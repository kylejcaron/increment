"""Recorded differences between a frozen capture and the current route.

Each record names one scenario and ingress, the identity field that differs and
why. Every other value of the case is still compared. Both the frozen and the
current value are pinned, so a change on either side fails the comparison, and
a record whose difference no longer occurs fails until it is removed. Fields a
row gained after the capture are projected away only when their value follows
from the row's own captured evidence, and a value whose construction changed
after the capture is projected back only when both its captured and its current
value follow from that evidence under their own constructions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, cast

from tests.parity_harness.comparison import TOLERANCE, nested_close

BinomialCounts = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ExpectedDifference:
    scenario: str
    ingress: str
    field: str
    frozen: str | float | Mapping[str, Any] | None
    current: str | float | Mapping[str, Any] | None
    reason: str
    key: str | None = None

    def applied_to(self, expected: Any) -> Any:
        """The frozen outcome with every occurrence of the frozen value, as a value or a
        mapping key, replaced by the current one; a record naming a ``key`` replaces only
        values held under that mapping key. Refuses a record the outcome does not contain."""
        found = False

        def replace(value: Any, held_under: Any = None) -> Any:
            nonlocal found
            if (self.key is None or held_under == self.key) and value == self.frozen:
                found = True
                return self.current
            if isinstance(value, dict):
                return {replace(key): replace(item, key) for key, item in value.items()}
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, tuple):
                return tuple(replace(item) for item in value)
            return value

        migrated = replace(expected)
        assert found, f"{self.scenario}/{self.ingress} no longer records {self.field} {self.frozen}"
        return migrated


# A compiled field the current route no longer records.
REMOVED = object()
# The fields of a plan entry a capture recorded as a bare metric name.
_PLAN_ENTRY = {
    "decision_method": None,
    "margin": None,
    "margin_abs": None,
    "prior": None,
    "sensitivity_methods": [],
}


@dataclass(frozen=True, slots=True)
class GuardrailMigration:
    """A guardrail the frozen capture recorded without a margin, which every constructor now
    refuses, migrated to the margin the scenario now declares.

    Every mapping naming ``metric`` that holds each ``frozen`` value -- a compiled procedure,
    in a plan or its ``decision_plan`` JSON, a retained row, or the projection fields of that
    row's cell -- takes the ``current`` values (``REMOVED`` deletes a field). A plan
    declaration's guardrail entry for ``metric``, a bare name or an entry without a margin,
    gains ``declared``. Refuses an outcome holding no compiled match, or, when the record
    declares a margin, no such entry.
    """

    scenario: str
    ingress: str
    metric: str
    frozen: Mapping[str, Any]
    current: Mapping[str, Any]
    reason: str
    declared: Mapping[str, Any] = field(default_factory=dict)

    def applied_to(self, expected: Any) -> Any:
        found = {"compiled": False, "declared": False}

        def declaration(plan: dict[str, Any]) -> dict[str, Any]:
            guardrails: list[Any] = list(plan.get("guardrails") or ())
            names: list[Any] = [
                item if isinstance(item, str) else item.get("metric") for item in guardrails
            ]
            if self.metric not in names:
                return plan
            index = names.index(self.metric)
            entry = guardrails[index]
            entry = {**_PLAN_ENTRY, "metric": entry} if isinstance(entry, str) else dict(entry)
            assert entry["margin"] is None and entry["margin_abs"] is None, entry
            found["declared"] = True
            guardrails[index] = {**entry, **self.declared}
            return {**plan, "guardrails": guardrails}

        def migrate(value: Any, held_under: Any = None, *, in_cell: bool = False) -> Any:
            if held_under == "decision_plan" and isinstance(value, str):
                return json.dumps(migrate(json.loads(value)), sort_keys=True, separators=(",", ":"))
            if isinstance(value, dict):
                row = value.get("retained_row")
                cell = isinstance(row, dict) and row.get("metric") == self.metric
                migrated = {key: migrate(item, key, in_cell=cell) for key, item in value.items()}
                names = migrated.get("metric") == self.metric or (
                    in_cell and held_under == "projection_fields"
                )
                if names and all(
                    key in migrated and migrated[key] == item for key, item in self.frozen.items()
                ):
                    found["compiled"] = True
                    migrated.update(self.current)
                    migrated = {key: item for key, item in migrated.items() if item is not REMOVED}
                if "guardrails" in migrated and "primary" in migrated:
                    migrated = declaration(migrated)
                return migrated
            if isinstance(value, list):
                return [migrate(item) for item in value]
            if isinstance(value, tuple):
                return tuple(migrate(item) for item in value)
            return value

        migrated = migrate(expected)
        assert found["compiled"], (
            f"{self.scenario}/{self.ingress} no longer records {self.metric} as {self.frozen}"
        )
        assert found["declared"] or not self.declared, (
            f"{self.scenario}/{self.ingress} no longer declares {self.metric} without a margin"
        )
        return migrated


Difference = ExpectedDifference | GuardrailMigration

# The favorable side each declared direction names; any other declaration names none.
_DECLARED_SIDES = {"increase": "greater", "decrease": "less"}


def declared_directions(experiment: Any) -> dict[str, str | None]:
    """Each metric's declared ``preferred_direction``, None where ``experiment`` declares none;
    an experiment's metric payload holds the field only when it is declared."""
    return {
        name: dict(metric).get("preferred_direction") for name, metric in experiment.metrics.items()
    }


def with_declared_favorable_sides(expected: Any, declared: Mapping[str, str | None]) -> Any:
    """The frozen capture with each compiled plan it records given the favorable sides its
    metrics declare, which plans captured earlier left out.

    A plan is a mapping holding ``procedures`` and ``view_policies``, or that mapping as JSON
    under ``decision_plan``. An arm procedure gains "greater" when ``declared`` names its
    metric's ``preferred_direction`` "increase" and "less" for "decrease"; any other
    declaration adds none, so a side the current plan records for it fails the caller's
    comparison. Refuses a plan that already records a side or names a metric ``declared``
    lacks, so the rule fails once a capture records the sides itself.
    """

    def migrate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
        procedures = {}
        for metric, procedure in plan["procedures"].items():
            side = None
            if procedure.get("kind") in {"relative", "absolute"}:
                assert "favorable_side" not in procedure, f"{metric} already records its side"
                assert metric in declared, f"{metric} has no recorded declaration"
                side = _DECLARED_SIDES.get(declared[metric] or "")
            procedures[metric] = (
                procedure if side is None else {**procedure, "favorable_side": side}
            )
        return {**plan, "procedures": procedures}

    def migrate(value: Any, held_under: Any = None) -> Any:
        if held_under == "decision_plan" and isinstance(value, str):
            return json.dumps(
                migrate_plan(json.loads(value)), sort_keys=True, separators=(",", ":")
            )
        if isinstance(value, dict):
            if "procedures" in value and "view_policies" in value:
                return migrate_plan(value)
            return {key: migrate(item, key) for key, item in value.items()}
        if isinstance(value, list):
            return [migrate(item) for item in value]
        if isinstance(value, tuple):
            return tuple(migrate(item) for item in value)
        return value

    return migrate(expected)


def without_stamped_sequential_direction(
    expected: Mapping[str, Any], actual: Mapping[str, Any], declared: Mapping[str, str | None]
) -> dict[str, Any]:
    """``actual`` without the ``preferred_direction`` a sequential row gained after its capture.

    Only a sequential row whose captured value is null is projected, and only when its value
    is its metric's declared direction in ``declared``; every other row is returned as is.
    """
    projected = dict(actual)
    if (
        actual.get("inference") in {"always_valid", "asymptotic_mean"}
        and expected.get("preferred_direction") is None
        and actual.get("preferred_direction") is not None
        and actual["preferred_direction"] == declared.get(actual["metric"])
    ):
        projected["preferred_direction"] = None
    return projected


VALUE_SCALE_SNAPSHOT_REASON = (
    "A values snapshot digests each emitted row, and a daily or as-of value row now records "
    "value_scale='absolute'; the values are unchanged."
)
_RECORDS: tuple[Difference, ...] = (
    ExpectedDifference(
        scenario="triggered_encouragement_compliance_asof",
        ingress="from_definitions",
        # Cell and row `source_snapshot_id`, `scope.snapshot_id` and the `scope.by_source` key.
        field="source_snapshot_id",
        frozen="sha256:250f3b61e63d68f59821d42f3f56bbdd3969d660903abb8e649ddbf6c3541bce",
        current="sha256:555fa9fd774dee87a087655dc870ca4bf2123da6d3298bb69e83ba7c9ad9c782",
        reason=(
            "A compliance-only view run through `run` requests no outcome metrics, so its "
            "snapshot request differs from the removed "
            "`run_asof_lift(estimands=('compliance',))`, which requested the declared "
            "outcome metrics; the estimates are unchanged."
        ),
    ),
    ExpectedDifference(
        scenario="audit-compliance-only-outcome-missingness",
        ingress="from_unit_panel",
        field="refusal code",
        frozen="frame.missing_policy.panel_drop",
        current="frame.missing_policy.panel_skip",
        reason=(
            "The panel refusal of an observed-only outcome policy is renamed with the policy; "
            "the retired code maps to the current one in `RETIRED_CODES`."
        ),
    ),
    ExpectedDifference(
        scenario="audit-compliance-only-outcome-missingness",
        ingress="from_unit_panel",
        field="refusal context missing",
        frozen="drop",
        current="skip",
        reason="The refusal names the declared policy, now spelled missing='skip'.",
    ),
    ExpectedDifference(
        scenario="sequential-mixed-quantile-catalog-checkpoint",
        ingress="from_unit_summary",
        field="retained_row.preferred_direction",
        key="preferred_direction",
        frozen=None,
        current="increase",
        reason=(
            "A sequential row carries its metric's declared preferred_direction, as a "
            "fixed-horizon row does; the captured row recorded none."
        ),
    ),
    ExpectedDifference(
        scenario="breakout_store_dimension",
        ingress="from_definitions",
        # Cell, row and family `source_snapshot_id`, `scope.snapshot_id` and the
        # `scope.by_source` key.
        field="source_snapshot_id",
        frozen="sha256:b36013595a199643bc99407709b3253008aa20dfdbd310dfed3a1144aea57fc9",
        current="sha256:fd017e16f0999da4f298a1692e00322a54d7335ce7c4d1b0f467675ee82feddb",
        reason=(
            "The breakout snapshot digests each row's additive reference, which the exact "
            "binomial segment rows now name as the score interval; their bounds are checked "
            "by `without_gained_arm_facts`."
        ),
    ),
    ExpectedDifference(
        scenario="breakout_store_dimension",
        ingress="from_definitions",
        field="family_id",
        frozen="sha256:020683b7c4135af28e830fa95ecc26a1be457017b8a35ec14550f603bf28022e",
        current="sha256:fd2bc5d54f902f2d2daa1c0f8c18079b9f139218e47f508d75fbac5bb7849f36",
        reason="The breakout family ID digests the snapshot ID above.",
    ),
    *(
        ExpectedDifference(
            scenario=scenario,
            ingress="from_definitions",
            # Cell and row `source_snapshot_id`, `scope.snapshot_id` and the `scope.by_source` key.
            field="source_snapshot_id",
            frozen=frozen,
            current=current,
            reason=VALUE_SCALE_SNAPSHOT_REASON,
        )
        for scenario, frozen, current in (
            (
                "triggered_daily",
                "sha256:26a5b7884c72bcee1a286281f2c5b0044a02fa96ebbfd7c5ce4b7220ad9c16a3",
                "sha256:6df2055343e38762994c98bdb8c82fadbd87fb11867dd9e554751623c19107d8",
            ),
            (
                "triggered_asof",
                "sha256:b5c8643bfd5b02f42eff25575d68ac638a323dad4902e8b5ec44f995b0040557",
                "sha256:5216ff36ab506d2521c8e926c49109f9a9231f7aa19a5d29f3b4a510f94bdea0",
            ),
        )
    ),
)
# A breakout view on an experiment that declares no breakout, or a triggered view no source
# can read, refuses in the check `runnable()` and `run` share, before any read; the frozen
# refusals were raised while the view ran.
_RECORDS += tuple(
    ExpectedDifference(
        scenario=scenario,
        ingress="from_moments",
        field="refusal context",
        key="context",
        frozen={
            "constructor": "MomentsSource",
            "offered": [],
            "requested": requested,
            "route": "use a source with the requested breakout dimensions",
        },
        current={
            "constructor": "from_moments",
            "offered": [],
            "requested": requested,
            "route": "declare breakouts=[...] on from_frame, or on the experiment in the definitions",
        },
        reason=(
            "The refusal names the public builder and how to declare a breakout, where the "
            "frozen refusal named the source class; both name the breakout the view requests, "
            "a registered segment family's dimension or none."
        ),
    )
    for scenario, requested in (
        ("artifact-breakout-exposure-coverage", []),
        ("audit-binary-breakout-ancillary-uptake", []),
        ("audit-encouragement-multi-metric-breakout", []),
        ("audit-retention-breakout-cohorts", []),
        ("breakout_boolean_store_dimension", []),
        ("breakout_store_dimension", []),
        ("dashboard-breakout-reads", []),
        ("sequential_automatic_breakout_continuous_bonferroni", ["store"]),
        ("sequential_automatic_breakout_discrete_bh", ["store"]),
        ("sequential_boolean_segments_automatic_asymptotic_mean", ["store"]),
        ("sequential_boolean_segments_automatic_exact_bernoulli", ["store"]),
        ("sequential_boolean_segments_registered_asymptotic_mean", ["store"]),
        ("sequential_boolean_segments_registered_exact_bernoulli", ["store"]),
        ("sequential_registered_breakout_continuous_bonferroni", ["store"]),
        ("sequential_registered_breakout_discrete_bh", ["store"]),
    )
) + tuple(
    ExpectedDifference(
        scenario="triggered_encouragement_compliance_asof",
        ingress=ingress,
        field="refusal context",
        key="context",
        frozen={
            "experiment": "<unknown>",
            "method": "run_asof_lift",
            "route": route,
            "supported_sources": supported,
            "trigger": None,
        },
        current={
            "experiment": "frame",
            "method": "run_asof_lift",
            "route": route,
            "supported_sources": ["from_definitions", "from_unit_day_artifact"],
            "trigger": None,
        },
        reason=(
            "The refusal names the experiment and the builders the whole-window triggered "
            "refusal lists; the frozen refusal named no experiment and, on the moments route, "
            "also listed from_moments."
        ),
    )
    for ingress, route, supported in (
        ("from_moments", "moments", ["from_definitions", "from_unit_day_artifact", "from_moments"]),
        (
            "from_unit_panel",
            "moments",
            ["from_definitions", "from_unit_day_artifact", "from_moments"],
        ),
        (
            "from_unit_summary",
            "moments",
            ["from_definitions", "from_unit_day_artifact", "from_moments"],
        ),
        ("from_switchback_panel", "switchback", ["from_definitions", "from_unit_day_artifact"]),
    )
)


def _rpo_margin(ingress: str, *, declared: bool) -> GuardrailMigration:
    return GuardrailMigration(
        scenario="mixed-cuped-multicovariate-fixed-horizon",
        ingress=ingress,
        metric="rpo",
        frozen={"null_lift": 0.0},
        current={"null_lift": -0.02},
        declared={"margin": 0.02} if declared else {},
        reason=(
            "The rpo guardrail declares margin=0.02, since a guardrail needs a margin, so its "
            "null on this increase metric is a 2% decrease; the capture tested it against zero."
        ),
    )


def _errors_margin(ingress: str, *, declared: float | None) -> GuardrailMigration:
    return GuardrailMigration(
        scenario="observational_multiplicity_primary_secondary_guardrail",
        ingress=ingress,
        metric="errors",
        frozen={"null_lift": 0.0, "kind": "relative", "axis": "relative", "scale": "relative"},
        current={
            "null_lift": REMOVED,
            "null_abs": 1.0,
            "kind": "absolute",
            "axis": "absolute",
            "scale": "absolute",
        },
        declared={} if declared is None else {"margin_abs": declared},
        reason=(
            "The errors guardrail declares margin_abs=1.0, since a guardrail needs a margin and "
            "an observational design tests only an absolute one, so its procedure is absolute "
            "with a null one error above control; the capture tested a relative lift of zero."
        ),
    )


def _ratio_margin(ingress: str, *, declared: bool) -> GuardrailMigration:
    return GuardrailMigration(
        scenario="categorical-cuped-moments-format-13",
        ingress=ingress,
        metric="revenue_per_spend",
        frozen={"null_lift": 0.0},
        current={"null_lift": -0.02},
        declared={"margin": 0.02} if declared else {},
        reason=(
            "The ratio guardrail declares margin=0.02, since a guardrail needs a margin, so its "
            "null on this increase metric is a 2% decrease; the capture tested it against zero."
        ),
    )


_RECORDS += (
    _rpo_margin("from_unit_day_artifact", declared=False),
    ExpectedDifference(
        scenario="mixed-cuped-multicovariate-fixed-horizon",
        ingress="from_unit_day_artifact",
        field="rpo p_value",
        key="p_value",
        frozen=0.004322126574,
        current=0.0016885046464289928,
        reason="The rpo guardrail's one-sided test is against its 2% margin, not zero.",
    ),
    _ratio_margin("from_moments", declared=False),
    ExpectedDifference(
        scenario="categorical-cuped-moments-format-13",
        ingress="from_moments",
        field="revenue_per_spend p_value",
        key="p_value",
        frozen=1.48663558e-06,
        current=8.485173020721271e-08,
        reason="The ratio guardrail's one-sided test is against its 2% margin, not zero.",
    ),
)
# Each recorded cell key of a frozen day-axis values readout.
_CELL_KEY_RECORDS = tuple(
    ExpectedDifference(
        scenario=scenario,
        ingress=ingress,
        field="metadata cells[].cell.value_scale",
        key="value_scale",
        frozen=None,
        current="absolute",
        reason=(
            "A daily or as-of value row records value_scale='absolute', so its cell key "
            "carries the scale its result cell already named; the values are unchanged."
        ),
    )
    for scenario, ingress in (
        ("triggered_daily", "from_definitions"),
        ("triggered_daily", "from_unit_day_artifact"),
        ("triggered_asof", "from_definitions"),
        ("triggered_asof", "from_unit_day_artifact"),
        *(
            (f"asof-{base}", ingress)
            for base in (
                "breakout_store_dimension",
                "breakout_boolean_store_dimension",
                "conversion_route_breakout_segments",
                "audit-retention-breakout-cohorts",
            )
            for ingress in ("from_definitions", "from_unit_panel")
        ),
    )
)
# Typed consumer inputs are frozen separately from the outcomes above.
_CONSUMER_INPUT_RECORDS: tuple[Difference, ...] = (
    ExpectedDifference(
        scenario="audit-compliance-only-outcome-missingness",
        ingress="from_unit_summary",
        # Source `specs[].missing` and the consumer context's `metric_missingness` value.
        field="specs[].missing",
        frozen="drop",
        current="skip",
        reason="The source declares the observed-only policy, now spelled missing='skip'.",
    ),
    ExpectedDifference(
        scenario="audit-compliance-only-outcome-missingness",
        ingress="from_moments",
        # Source and consumer `context.metric_missingness`.
        field="context.metric_missingness",
        key="metric_missingness",
        frozen={},
        current={"revenue": "skip"},
        reason=(
            "A moments source records each metric declared missing='skip', which qualifies "
            "its rows' population; the captured source recorded none."
        ),
    ),
)
_INGRESSES = (
    "from_definitions",
    "from_moments",
    "from_unit_day_artifact",
    "from_unit_panel",
    "from_unit_summary",
)
# Ingresses whose consumer inputs record the plan declaration, not only its compiled form.
_DECLARING = frozenset({"from_definitions", "from_unit_day_artifact"})


def _artifact_digest(scenario: str, name: str, frozen: str, current: str) -> ExpectedDifference:
    return ExpectedDifference(
        scenario=scenario,
        ingress="from_unit_day_artifact",
        field=name,
        frozen=frozen,
        current=current,
        reason="The published context records the guardrail's margin, so each digest over it moves.",
    )


_CONSUMER_INPUT_RECORDS += (
    *(_rpo_margin(ingress, declared=ingress in _DECLARING) for ingress in _INGRESSES),
    _artifact_digest(
        "mixed-cuped-multicovariate-fixed-horizon",
        "context_sha256",
        "e8bfcd6bb30b18b90550e27e2c0a3d807849283e413518e6b5c3539918efe11b",
        "239e8fdc6cfa4d84fa981310cd1ea5f73da030a1b27d8b878e8eecdd1f1a820e",
    ),
    _artifact_digest(
        "mixed-cuped-multicovariate-fixed-horizon",
        "canonical_source_recipe_json",
        '{"recipe_sha256":"28000a70d60daf8355f0d350ec056ef5a42a88fa838724774d16defca8743c0e",'
        '"source_recipe_format":2}',
        '{"recipe_sha256":"3a21ea631b88996c190d73333050cfe5527c9e3796f1735e23f89ef1bf3c37cf",'
        '"source_recipe_format":2}',
    ),
    _artifact_digest(
        "mixed-cuped-multicovariate-fixed-horizon",
        "source_provenance_sha256",
        "599637a75aba64f2f5e5fa41893b35a479b8a349049e91eb6f57012ec75fd7d9",
        "d99e6eee29444e1bdc311df7d8a24effbd5aa443aa71357317aeaff653a07b75",
    ),
    _errors_margin("from_definitions", declared=1.0),
    # The artifact context's canonical JSON writes the integral margin as an integer.
    _errors_margin("from_unit_day_artifact", declared=1),
    _errors_margin("from_unit_panel", declared=None),
    _errors_margin("from_unit_summary", declared=None),
    _artifact_digest(
        "observational_multiplicity_primary_secondary_guardrail",
        "context_sha256",
        "d3918218fee95d52a0f61e956744c26f00fefdf2be97219a328855856c2a198b",
        "205f58117b0ebe14d997c6deb2550e4f39fb7fa4a1e4b334220a06ce55185294",
    ),
    _artifact_digest(
        "observational_multiplicity_primary_secondary_guardrail",
        "canonical_source_recipe_json",
        '{"recipe_sha256":"04bfa333ecdfd550c658869304bb5ff1282f4f159b417a850debffe42c5720da",'
        '"source_recipe_format":2}',
        '{"recipe_sha256":"cea071a54c5af791fe3e4063978b11a58e8604c9de189da4a5f9202bdedd7560",'
        '"source_recipe_format":2}',
    ),
    _artifact_digest(
        "observational_multiplicity_primary_secondary_guardrail",
        "source_provenance_sha256",
        "15f5add124d2cdbd1ceb06ba85215bfb5df5ebbf01a642db48ce9924696dcd6f",
        "8aa6920277ebdef00bb5dd8f1a57c07bb9de2686ef90f1aaf04bbaf5e08c6073",
    ),
    *(_ratio_margin(ingress, declared=ingress in _DECLARING) for ingress in _INGRESSES),
    _artifact_digest(
        "categorical-cuped-moments-format-13",
        "context_sha256",
        "9af7380106e977018cb523d70c132212c5ee88d6c78276d93bd89038eccfd46e",
        "0d4bf8c70eceb016860bad056fd756941a66299c22d79467749fe6073659a140",
    ),
    _artifact_digest(
        "categorical-cuped-moments-format-13",
        "canonical_source_recipe_json",
        '{"recipe_sha256":"75692b0744a585230ee65ccb21df51a6c7bd8f823cb08d063f13854a8a92d1c6",'
        '"source_recipe_format":2}',
        '{"recipe_sha256":"0b70761dc104485b8a6a97a360a289f147765895ff49d7b13cf6b200add40338",'
        '"source_recipe_format":2}',
    ),
    _artifact_digest(
        "categorical-cuped-moments-format-13",
        "source_provenance_sha256",
        "42e84e6064ef492986126d002a6e5db58c1c3beb31b838387cff1d635f580ce5",
        "0b7786c8796a24d479746ddf996180d1a906216e46a30fa24c3e2be710f2869a",
    ),
)
# Typed records digest the published context too, apart from the identity records above.
_TYPED_INPUT_RECORDS: tuple[Difference, ...] = (
    _artifact_digest(
        "mixed-cuped-multicovariate-fixed-horizon",
        "context_sha256",
        "e8bfcd6bb30b18b90550e27e2c0a3d807849283e413518e6b5c3539918efe11b",
        "239e8fdc6cfa4d84fa981310cd1ea5f73da030a1b27d8b878e8eecdd1f1a820e",
    ),
    _artifact_digest(
        "categorical-cuped-moments-format-13",
        "context_sha256",
        "9af7380106e977018cb523d70c132212c5ee88d6c78276d93bd89038eccfd46e",
        "0d4bf8c70eceb016860bad056fd756941a66299c22d79467749fe6073659a140",
    ),
)
# A whole frozen disposition the current route replaces with another one.
_PANEL_ASOF_WITHOUT_EXPOSURE_DATE = {
    "kind": "refusal",
    "code": "evidence.grain_unavailable",
    "stage": "run",
    "context": {
        "constructor": "FramePanelSource",
        "offered": ["total"],
        "requested": "asof",
        "route": (
            "pass exposure_date=<column> naming each unit's exposure date; if every unit was "
            "exposed on the same day, add a column holding that date"
        ),
    },
}
# The failures a retired whole-run refusal listed, which the run now returns as unavailable rows.
_ALL_CELLS_FAILED = (
    (
        "edge-zero-variance-arm",
        {
            "code": "estimation.engine.lift_guard",
            "context": {
                "display": (
                    "Degenerate data: an arm has zero variance (e.g. all-identical outcomes or a "
                    "saturated conversion arm), cannot estimate lift uncertainty for that arm"
                ),
                "group_id": "treatment",
                "method": "unadjusted",
                "metric": "revenue",
                "reason": "zero_variance",
            },
            "estimand": "itt",
            "group_id": "treatment",
            "method": "unadjusted",
            "metric": "revenue",
        },
    ),
    (
        "edge-near-zero-ratio-denominator",
        {
            "code": "estimation.variance.ratio_moments_nonpositive_denominator_mean",
            "context": {"d_bar": 0.0, "group_id": "control", "method": "unadjusted", "metric": "r"},
            "estimand": "itt",
            "group_id": "treatment",
            "method": "unadjusted",
            "metric": "r",
        },
    ),
)
_DISPOSITION_RECORDS: tuple[Difference, ...] = tuple(
    ExpectedDifference(
        scenario=f"{view}-{base}",
        ingress="from_unit_panel",
        field="disposition",
        frozen=frozen,
        current=_PANEL_ASOF_WITHOUT_EXPOSURE_DATE,
        reason=(
            "The scenario's panel declares no exposure_date, so it offers only whole-window "
            "grains and refuses the as-of view before reading; the frozen outcome counted days "
            "from a common start the panel never declared."
        ),
    )
    for view, base, frozen in (
        ("asof", "breakout_store_dimension", {"kind": "cells"}),
        ("asof", "breakout_boolean_store_dimension", {"kind": "cells"}),
        ("asof", "conversion_route_breakout_segments", {"kind": "cells"}),
        (
            "asof_lift",
            "breakout_store_dimension",
            {
                "kind": "refusal",
                "code": "estimation.cuped.arm_no_covariate",
                "stage": "run",
                "context": {"arm_group": "control", "arm_metric": "revenue_cuped"},
            },
        ),
        (
            "asof_lift",
            "breakout_boolean_store_dimension",
            {
                "kind": "refusal",
                "code": "estimation.cuped.arm_no_covariate",
                "stage": "run",
                "context": {"arm_group": "control", "arm_metric": "revenue_cuped"},
            },
        ),
        ("asof_lift", "conversion_route_breakout_segments", {"kind": "cells"}),
    )
) + tuple(
    ExpectedDifference(
        scenario=scenario,
        ingress="from_unit_summary",
        field="disposition",
        frozen={
            "kind": "refusal",
            "code": "readout.estimate_lift_every",
            "stage": "run",
            "context": {"failures": [failure]},
        },
        current={"kind": "unavailable", "stage": "run", "cells": [failure]},
        reason=(
            "The run's only cell failed; the retired `readout.estimate_lift_every` refusal gives "
            "way to that cell's unavailable row, carrying the failure it listed."
        ),
    )
    for scenario, failure in _ALL_CELLS_FAILED
)
_MARGIN_LESS_GUARDRAIL_REFUSAL = {
    "kind": "refusal",
    "code": "plan.guardrail.margin_required",
    "stage": "construct",
    "context": {
        "metric": "latency",
        "route": "declare ExperimentMetric(metric=..., margin=...) or margin_abs=...",
    },
}
_DISPOSITION_RECORDS += tuple(
    ExpectedDifference(
        scenario=scenario,
        ingress=ingress,
        field="disposition",
        frozen={"kind": "cells"},
        current=_MARGIN_LESS_GUARDRAIL_REFUSAL,
        reason=(
            "The scenario's latency guardrail declares no margin, which every constructor now "
            "refuses before reading; the frozen outcome tested it against zero."
        ),
    )
    for scenario in (
        "encouragement_guardrail_late_rows",
        "encouragement_guardrail_late_only_request",
    )
    for ingress in (
        "from_definitions",
        "from_unit_day_artifact",
        "from_unit_summary",
        "from_unit_panel",
        "from_moments",
    )
)


def _by_case(
    records: tuple[Difference, ...],
) -> dict[tuple[str, str], tuple[Difference, ...]]:
    return {
        pair: tuple(record for record in records if (record.scenario, record.ingress) == pair)
        for pair in dict.fromkeys((record.scenario, record.ingress) for record in records)
    }


EXPECTED_DIFFERENCES = _by_case(_RECORDS)
CONSUMER_INPUT_DIFFERENCES = _by_case(_CONSUMER_INPUT_RECORDS)
TYPED_INPUT_DIFFERENCES = _by_case(_TYPED_INPUT_RECORDS)
CELL_KEY_DIFFERENCES = _by_case(_CELL_KEY_RECORDS)
DISPOSITION_DIFFERENCES = _by_case(_DISPOSITION_RECORDS)


def _d7_retention_margin(ingress: str) -> GuardrailMigration:
    return GuardrailMigration(
        scenario="parity_baseline",
        ingress=ingress,
        metric="d7_retention",
        frozen={"null_lift": 0.0},
        current={"null_lift": -0.02},
        reason=(
            "The example d7_retention guardrail declares margin=0.02, since a guardrail needs a "
            "margin, so its null on this increase metric is a 2% decrease; the capture tested it "
            "against zero."
        ),
    )


# The root readout fixture and its exports, frozen while these guardrails declared no margin.
EXPORT_DIFFERENCES = {
    "parity_baseline": _d7_retention_margin("from_definitions"),
    "mixed_warehouse": _rpo_margin("from_definitions", declared=False),
    "mixed_categorical_replay": _ratio_margin("from_moments", declared=False),
}


def _replayed(scenario: str, field: str, frozen: str | float, current: str | float) -> Difference:
    return ExpectedDifference(
        scenario=scenario,
        ingress="from_moments",
        field=field,
        key=field,
        frozen=frozen,
        current=current,
        reason=(
            "The replayed export's guardrail carries its margin, so its one-sided test is against "
            "the margin, not zero, and each snapshot and family id that digests its procedure "
            "moves."
        ),
    )


# Each export's frozen replay, its recorded plan given the margin above.
REPLAY_DIFFERENCES: dict[str, tuple[Difference, ...]] = {
    "parity_baseline": (
        _d7_retention_margin("from_moments"),
        _replayed("parity_baseline", "p_value", 3.65387567e-06, 1.962906411e-06),
        _replayed(
            "parity_baseline",
            "source_snapshot_id",
            "sha256:ab2a6e33bc818afba62cb99f14b3cec6e5c032a49c7ba4f3e7950a1b89eb60d0",
            "sha256:7baac129d5fbefb861252df41482fce0980b79044fc6fe95eb6ecd8eb02cf560",
        ),
        _replayed(
            "parity_baseline",
            "family_id",
            "sha256:24abdf725e68129f3eac1c346dbc2153e2776046e7fc1fca25fbe565f3345d39",
            "sha256:774eb11bc9776bf4114215b749058817ccf3ac2ea072b7f1f2a414f41473959a",
        ),
        _replayed(
            "parity_baseline",
            "family_id",
            "sha256:d3185e13450d0baae047ffb1f2291540dfb4c2077026ad69efdbfd6d5e48ad30",
            "sha256:bec6c86a15183f59773e763e6797149a5d7e7287de0409f7a0a2e2675c5aaf3e",
        ),
    ),
    **{
        name: (
            _rpo_margin("from_moments", declared=False),
            _replayed(name, "p_value", 0.004322126573866, 0.001688504646429),
        )
        for name in ("mixed_warehouse", "mixed_fitted_replay")
    },
    "mixed_categorical_replay": (
        _ratio_margin("from_moments", declared=False),
        _replayed("mixed_categorical_replay", "p_value", 1.48663558e-06, 8.485173e-08),
        _replayed(
            "mixed_categorical_replay",
            "source_snapshot_id",
            "sha256:42a91378ea3f0f723cbdf30f4b3c22c6e55ce0dd02087e2c6a4990fdb61aa76b",
            "sha256:5de0cac0113fa38cc26ba2e8b84cb4f3d5a8dd8cf70aa8fad23535955f702920",
        ),
        _replayed(
            "mixed_categorical_replay",
            "family_id",
            "sha256:0dd5ccc05702e246d291afe845d04e62f5b54e4dc9020f77dca5b18967693567",
            "sha256:8267e5954fe947a5b6eaa03f42deaba9b90f48d39515c1df38ea917fe376a072",
        ),
        _replayed(
            "mixed_categorical_replay",
            "family_id",
            "sha256:8b164da095832eb816f95add98f81b0c044489184cb5d0f1c780d6fd4dd7acbe",
            "sha256:7eb542b21701f5045cd93555a6e7731d760cf08b81b611e3d60c9c055f4f1da2",
        ),
    ),
}
_MARGIN_SNAPSHOT_REASON = (
    "Each procedure enters the snapshot request with its null, so a guardrail's margin moves "
    "the snapshot of every view that reads it; the rows are otherwise unchanged."
)
READOUT_IDENTITY_DIFFERENCES = tuple(
    ExpectedDifference(
        scenario=scenario,
        ingress=ingress,
        field=f"{view} source_snapshot_id",
        frozen=f"sha256:{frozen}",
        current=f"sha256:{current}",
        reason=_MARGIN_SNAPSHOT_REASON,
    )
    for scenario, ingress, view, frozen, current in (
        (
            "parity_baseline",
            "from_definitions",
            "run",
            "8996dea490ae6485b7c28afcb1428a7fcf1ac31078cb832489b52e6cf2f61c40",
            "f14ed77328bfb9959e8338058aea9d1ba05fbafca6a715786f8e465b9a810d5c",
        ),
        # Every other view's frozen snapshot is the one `tests/test_metric_evidence.py` already
        # migrates it to: the value-scale move (daily, as-of) or the arm-count and
        # score-interval move (lift, breakout).
        (
            "parity_baseline",
            "from_definitions",
            "run_daily",
            "abf157fa97fac9b3cd7002d0e8e4a31b908d852fba1cbf44da1ebb84de4d5713",
            "d5a8d7f980ca46f6f9ed7b902b808d3719e944b86e11529db3601e836c15d320",
        ),
        (
            "parity_baseline",
            "from_definitions",
            "run_asof",
            "72847bbe44ced27b7e5e19d5b043a5e95eb7f4c7c53ed629acd4a59faab7fe8b",
            "f279219fa034637a091bf7c534f05226322631d059a2280c65b421d46bff9c38",
        ),
        (
            "parity_baseline",
            "from_definitions",
            "run_daily_lift",
            "4b525e34af8cb6add5b940b4bfc929ce407d792f4297e7bbb5b1f47b8f1324cb",
            "09d04ee429a1573a5dffbcd8ac0851cc2c1f5592a305188ce89ee42955ee923e",
        ),
        (
            "parity_baseline",
            "from_definitions",
            "run_asof_lift",
            "3b70b680af781304f036556f88b3c9892f544754b78d0a28a4102ce5acf84dc0",
            "d6621f8cfbf4cd5f268b8e275435a888a1d8d304096c489d7905989edf2b177c",
        ),
        (
            "parity_baseline",
            "from_definitions",
            "run_breakout",
            "5ce099c78ba4a5bfe4fe51eff2a4c00131f72d3618fd8a5f3253d59107f81e11",
            "420fe58001a95f2c39c28bcb2369036c8a4087bd843b632c169da22f1cbba523",
        ),
        (
            "mixed_local",
            "from_unit_summary",
            "run",
            "aed46124b71363a2ac2497ab0272919b6c15f437bb9e8d4db330c6e80154c1ec",
            "7b8283b81486fedb3b7a55ae9725c6ab446b3ed22138cee48acc5daf967f34c8",
        ),
    )
)


def without_guardrail_segments(rows: Any, metric: str, *, dropped: int) -> Any:
    """The frozen breakout identity rows without ``metric``'s segments: a breakout cannot test
    a guardrail's margin, so the segment view reads every other metric. Refuses a capture that
    no longer holds exactly ``dropped`` such rows."""
    kept = [item for item in rows if item["row"]["metric"] != metric]
    assert len(rows) - len(kept) == dropped, f"{metric}: {len(rows) - len(kept)} segment rows"
    return kept


def with_expected_difference(
    scenario: str,
    ingress: str,
    expected: Any,
    *,
    differences: Mapping[tuple[str, str], tuple[Difference, ...]] = EXPECTED_DIFFERENCES,
) -> Any:
    """The frozen capture of one case, migrated by each of its recorded differences."""
    for difference in differences.get((scenario, ingress), ()):
        expected = difference.applied_to(expected)
    return expected


#: Mixed CUPED fields gained after the frozen captures, keyed by the field that marks the
#: mapping holding them: a finalized fit (``_projections``) or a fitted payload's state
#: (``triangular_factor``). A frozen value that predates them decodes with their empty
#: defaults and is compared without them.
_GAINED_AFTER_CAPTURE = {
    "_projections": ("covariate_mean_differences", "within_arm_scatter"),
    "triangular_factor": ("within_arm_scatter",),
}
FIT_FIELDS_GAINED_AFTER_CAPTURE = _GAINED_AFTER_CAPTURE["_projections"]


def without_gained_fit_fields(expected: Any, actual: Any) -> Any:
    """``actual`` without the mixed CUPED fields its frozen ``expected`` value predates.

    Only a fit or fit-state mapping is projected, and only a field it gained after the
    capture that the frozen mapping lacks. Every other value is left to the caller's
    comparison. Lists and tuples pair by position.
    """
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        projected = {
            key: without_gained_fit_fields(expected.get(key), value)
            for key, value in actual.items()
        }
        for marker, gained in _GAINED_AFTER_CAPTURE.items():
            if marker in actual and marker in expected:
                for field in gained:
                    if field not in expected:
                        projected.pop(field, None)
        return projected
    if (
        isinstance(expected, (list, tuple))
        and isinstance(actual, (list, tuple))
        and len(expected) == len(actual)
    ):
        paired = [
            without_gained_fit_fields(left, right)
            for left, right in zip(expected, actual, strict=True)
        ]
        return tuple(paired) if isinstance(actual, tuple) else paired
    return actual


# Frozen day-axis rows dated after each panel's declared observation end, which the current
# route no longer returns; the warehouse capture of the same case stops at that end too.
_DAY_ROWS_THROUGH = {
    ("asof-audit-retention-breakout-cohorts", "from_unit_panel"): "2025-01-20",
    ("asof_lift-audit-retention-breakout-cohorts", "from_unit_panel"): "2025-01-20",
}


def through_observation_end(scenario: str, ingress: str, records: Any) -> Any:
    """The frozen consumer-input *records* of one case without the day-axis rows dated after
    its declared observation end. Refuses a recorded case whose frozen rows hold none."""
    end = _DAY_ROWS_THROUGH.get((scenario, ingress))
    if end is None:
        return records
    dropped = 0
    for record in records:
        if record["request_type"] != "DayAxisRequest":
            continue
        for day_slice in record["answer"]["slices"]:
            for name in ("rows", "consumer_rows"):
                rows = day_slice.get(name)
                if rows is None:
                    continue
                kept = [row for row in rows if str(row["ds"]) <= end]
                dropped += len(rows) - len(kept)
                day_slice[name] = kept
    assert dropped, f"{scenario}/{ingress} no longer records day rows after {end}"
    return records


def _binomial_arm_facts(counts: BinomialCounts) -> dict[str, int | float]:
    x_c, n_c, x_t, n_t = counts
    return {"n_control": n_c, "n_treat": n_t, "mean_control": x_c / n_c, "mean_treat": x_t / n_t}


def _without_gained_unit_arm_facts(
    expected: Mapping[str, Any], actual: Mapping[str, Any], projected: dict[str, Any]
) -> dict[str, Any]:
    """``projected`` without the arm unit counts an unclustered ratio or a quantile row gained,
    and the arm ratios a ratio row gained. Their captured evidence holds neither, so the counts
    must be positive integers and the ratios finite."""
    metric_type = actual.get("metric_type")
    if metric_type == "quantile":
        gained: tuple[str, ...] = ("n_control", "n_treat")
    elif metric_type == "ratio" and actual.get("n_clusters") is None:
        gained = ("n_control", "n_treat", "mean_control", "mean_treat")
    else:
        return projected
    for name in gained:
        value = actual.get(name)
        if value is None or expected.get(name) is not None:
            continue
        if name in ("n_control", "n_treat"):
            # A frame holds counts in a float column; a typed row holds integers.
            assert type(value) in (int, float) and value >= 1 and float(value).is_integer(), (
                name,
                value,
            )
        else:
            assert type(value) is float and math.isfinite(value), (name, value)
        if name in expected:
            projected[name] = None
        else:
            del projected[name]
    return projected


def _assert_score_interval(actual: Mapping[str, Any], counts: BinomialCounts) -> None:
    """The live row's additive bounds are the score interval of its counts at ``abs_alpha``."""
    from increment.estimation.engine import binomial_abs_interval

    alpha = actual["abs_alpha"]
    score = binomial_abs_interval(*counts, alpha)[:2]
    assert nested_close(score, (actual["abs_lb"], actual["abs_ub"])), (counts, alpha, score)


def _assert_score_replaced_wald(
    expected: Mapping[str, Any], actual: Mapping[str, Any], counts: BinomialCounts
) -> None:
    """The live row's additive bounds are the score interval of its counts, and the captured
    row's the Wald interval of its own ``abs_diff`` and ``abs_se``, both at ``abs_alpha``."""
    from scipy.stats import norm

    _assert_score_interval(actual, counts)
    alpha = actual["abs_alpha"]
    half_width = float(norm.isf(alpha / 2.0)) * expected["abs_se"]
    wald = (expected["abs_diff"] - half_width, expected["abs_diff"] + half_width)
    assert nested_close(wald, (expected["abs_lb"], expected["abs_ub"])), (alpha, wald)


def without_gained_arm_facts(
    expected: Any,
    actual: Any,
    counts: Mapping[tuple[str, str], BinomialCounts] | None = None,
) -> Any:
    """``actual`` without the arm counts and values rows gained after a capture, and the score
    additive interval exact binomial rows gained in place of the captured Wald interval.

    An arm fact is projected only where its captured row lacks it or holds it as null. On an
    exact binomial row (``reference_kind == "binomial"``) the counts must equal ``n_c`` and
    ``n_t`` exactly, as integers on a typed row; the rates must equal ``x_c / n_c`` and
    ``x_t / n_t`` within the harness tolerance. Both come from the row's own ``binomial_set``
    or, for a flattened row without one, from ``counts`` keyed by ``(metric, group_id)`` as
    ``(x_c, n_c, x_t, n_t)``. Such a row with ``abs_reference_kind == "score"`` whose captured
    row names no additive reference takes the captured ``abs_lb``/``abs_ub`` and a null
    reference once its live bounds equal the score interval of its counts at the row's
    ``abs_alpha``, within the harness tolerance. A captured row with a Wald standard error
    must also hold ``abs_diff`` plus or minus the Normal critical value times ``abs_se``; one
    without, whose Wald interval was undefined, must hold no bounds and no ``abs_alpha``, and
    the live ``abs_alpha`` is projected back to null too. Ratio and quantile rows follow
    ``_without_gained_unit_arm_facts``. Every other field is left to the caller's comparison.
    Lists and tuples pair by position, so a captured model's ``(type name, fields)`` form is
    projected too.
    """
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        projected = {
            key: without_gained_arm_facts(expected.get(key), value, counts)
            for key, value in actual.items()
        }
        if actual.get("reference_kind") != "binomial":
            return _without_gained_unit_arm_facts(expected, actual, projected)
        exact = actual.get("binomial_set")
        if isinstance(exact, tuple) and len(exact) == 2:
            exact = exact[1]
        if isinstance(exact, Mapping):
            row_counts = (exact["x_c"], exact["n_c"], exact["x_t"], exact["n_t"])
        elif counts is not None:
            row_counts = counts[(actual["metric"], actual["group_id"])]
        else:
            return projected
        for field, value in _binomial_arm_facts(row_counts).items():
            if actual.get(field) is None or expected.get(field) is not None:
                continue
            if field in ("n_control", "n_treat"):
                # A frame holds counts in a float column; a typed row holds integers.
                count_types = (int,) if isinstance(exact, Mapping) else (int, float)
                assert type(actual[field]) in count_types and actual[field] == value, (
                    field,
                    value,
                    actual[field],
                )
            else:
                assert nested_close(value, actual[field]), (field, value, actual[field])
            if field in expected:
                projected[field] = None
            else:
                del projected[field]
        if (
            actual.get("abs_reference_kind") == "score"
            and expected.get("abs_reference_kind") is None
        ):
            if expected.get("abs_se") is not None:
                _assert_score_replaced_wald(expected, actual, row_counts)
                restored: tuple[str, ...] = ("abs_lb", "abs_ub", "abs_reference_kind")
            elif all(expected.get(field) is None for field in ("abs_lb", "abs_ub", "abs_alpha")):
                _assert_score_interval(actual, row_counts)
                restored = ("abs_lb", "abs_ub", "abs_alpha", "abs_reference_kind")
            else:
                restored = ()
            for field in restored:
                if field in expected:
                    projected[field] = expected[field]
                else:
                    projected.pop(field, None)
        return projected
    if (
        isinstance(expected, (list, tuple))
        and isinstance(actual, (list, tuple))
        and len(expected) == len(actual)
    ):
        paired = [
            without_gained_arm_facts(left, right, counts)
            for left, right in zip(expected, actual, strict=True)
        ]
        return tuple(paired) if isinstance(actual, tuple) else paired
    return actual


def _registered_family_cell(row: Mapping[str, Any]) -> bool:
    """Whether ``row``'s own sequential evidence registers its cell in a family."""
    result = row.get("sequential_result")
    checkpoint = result.get("checkpoint") if isinstance(result, Mapping) else None
    cell = checkpoint.get("cell") if isinstance(checkpoint, Mapping) else None
    return isinstance(cell, Mapping) and cell.get("family") is True


def without_post_capture_family_size(expected: Any, actual: Any) -> Any:
    """``actual`` without the ``family_size`` a family row gained after its capture.

    Two kinds of row gained it: a fixed-horizon BH segment row, and a row whose own
    ``sequential_result`` registers its cell in a sequential family. Only such a row with a
    positive integer ``family_size`` is projected, and only when its captured row lacks the
    field or holds it as null. Lists and tuples pair by position, as in
    ``without_gained_arm_facts``.
    """
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        projected = {
            key: without_post_capture_family_size(expected.get(key), value)
            for key, value in actual.items()
        }
        size = actual.get("family_size")
        gained = (
            actual.get("family_procedure") == "bh" and actual.get("dimension_value") is not None
        ) or _registered_family_cell(actual)
        if gained and type(size) is int and size >= 1 and expected.get("family_size") is None:
            if "family_size" in expected:
                projected["family_size"] = None
            else:
                del projected["family_size"]
        return projected
    if (
        isinstance(expected, (list, tuple))
        and isinstance(actual, (list, tuple))
        and len(expected) == len(actual)
    ):
        paired = [
            without_post_capture_family_size(left, right)
            for left, right in zip(expected, actual, strict=True)
        ]
        return tuple(paired) if isinstance(actual, tuple) else paired
    return actual


def without_value_row_scale(expected: Any, actual: Any) -> Any:
    """``actual`` without the fixed ``value_scale`` daily and as-of value rows gained after a
    capture.

    Only a row holding a day's (``ds``) arm ``value`` whose captured row lacks the field is
    projected, and only when it holds ``"absolute"``, the one scale such a row records. Lists
    and tuples pair by position, as in ``without_gained_arm_facts``.
    """
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        projected = {
            key: without_value_row_scale(expected.get(key), value) for key, value in actual.items()
        }
        if {"ds", "value", "value_scale"} <= actual.keys() and "value_scale" not in expected:
            assert actual["value_scale"] == "absolute", actual["value_scale"]
            del projected["value_scale"]
        return projected
    if (
        isinstance(expected, (list, tuple))
        and isinstance(actual, (list, tuple))
        and len(expected) == len(actual)
    ):
        paired = [
            without_value_row_scale(left, right)
            for left, right in zip(expected, actual, strict=True)
        ]
        return tuple(paired) if isinstance(actual, tuple) else paired
    return actual


def without_metric_outcome_columns(expected: Any, actual: Any) -> Any:
    """``actual`` consumer inputs without the ``metric_outcome_columns`` each source context
    gained after ``expected`` was captured.

    Only a context whose capture lacks the field is projected, and only when its value follows
    from the captured source inputs: a declared dataframe's metric specs name each metric's value
    column (the numerator for a ratio) and any denominator; every other source names none.
    """
    derived: dict[str, set[str]] = {}
    for source_input in expected["source_inputs"]:
        if source_input["kind"] != "declared_source_frame":
            continue
        for spec in source_input["configuration"]["source_declaration"]["specs"]:
            value = spec["numerator"] if spec["type"] == "ratio" else spec["value_column"]
            derived[spec["name"]] = {value or spec["name"]} | (
                {spec["denominator"]} if spec["denominator"] else set()
            )

    def project(captured: Any, live: Any) -> Any:
        if isinstance(captured, Mapping) and isinstance(live, Mapping):
            fields = dict(live)
            if (
                "metric_missingness" in captured
                and "metric_outcome_columns" not in captured
                and "metric_outcome_columns" in fields
            ):
                columns = fields.pop("metric_outcome_columns")
                assert {name: set(value) for name, value in columns.items()} == derived, columns
            return {key: project(captured.get(key), value) for key, value in fields.items()}
        if isinstance(captured, list) and isinstance(live, list) and len(captured) == len(live):
            return [project(left, right) for left, right in zip(captured, live, strict=True)]
        return live

    return project(expected, actual)


def _relative_late_reference(lift: Mapping[str, Any]) -> tuple[float, float]:
    """The log-scale point and SE a relative LATE lift's symmetric Normal interval implies."""
    from scipy.stats import norm

    z = float(norm.isf(lift["alpha"] / 2.0))
    return (
        math.log1p(lift["value"]),
        (math.log1p(lift["ub"]) - math.log1p(lift["lb"])) / (2.0 * z),
    )


def without_relative_late_log_reference(expected: Any, actual: Any) -> Any:
    """``actual`` without the log-scale reference a relative LATE row gained after a capture.

    Only an ``estimand == "late"`` row on the relative scale is projected, and only when its
    captured lift holds ``log_mean`` and ``log_se`` as null. Both must follow from the row's
    own Normal interval within the harness tolerance: ``log_mean = log1p(value)`` and
    ``log_se = (log1p(ub) - log1p(lb)) / (2 z)``, ``z`` the upper ``alpha / 2`` quantile of
    the lift's ``alpha``. Every other field is left to the caller's comparison.
    """
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return actual
    captured, lift = expected.get("lift"), actual.get("lift")
    if (
        actual.get("estimand") != "late"
        or actual.get("value_scale") != "relative"
        or not isinstance(captured, Mapping)
        or not isinstance(lift, Mapping)
        or captured.get("log_mean") is not None
        or captured.get("log_se") is not None
        or lift.get("log_mean") is None
    ):
        return actual
    implied = dict(zip(("log_mean", "log_se"), _relative_late_reference(lift), strict=True))
    for name, value in implied.items():
        assert nested_close(value, lift[name]), (name, value, lift[name])
    return {**actual, "lift": {**lift, "log_mean": None, "log_se": None}}


def without_relative_late_p_value(expected: Any, actual: Any, rows: Iterable[Any]) -> Any:
    """Saved ``actual`` metadata without the p-value relative LATE cells gained after a capture.

    Only a ``late`` cell record on the relative scale is projected, and only when its captured
    sampling evidence is null. The p-value must be the Normal tail, for the row's alternative,
    of ``(log_mean - log1p(null_lift)) / log_se`` with both taken from the matching row's own
    interval, as in ``without_relative_late_log_reference``; it is compared relative to its
    size. Its hypothesis and method must be the cell's own. Records pair by position, and
    every other field is left to the caller's comparison.
    """
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return actual
    old_records, records = expected.get("cells"), actual.get("cells")
    if not isinstance(old_records, list) or not isinstance(records, list):
        return actual
    if len(old_records) != len(records):
        return actual
    from scipy.stats import norm

    from increment.declarations import CellKey

    rows_by_cell = {CellKey.from_row(row): row for row in rows}
    projected = []
    pairs = cast(
        "list[tuple[Mapping[str, Any], Mapping[str, Any]]]",
        list(zip(old_records, records, strict=True)),
    )
    for old, record in pairs:
        cell, sampling, old_sampling = record["cell"], record.get("sampling"), old.get("sampling")
        if (
            cell.get("estimand") != "late"
            or cell.get("value_scale") != "relative"
            or not isinstance(sampling, Mapping)
            or sampling.get("evidence") is None
            or not isinstance(old_sampling, Mapping)
            or old_sampling.get("evidence") is not None
        ):
            projected.append(record)
            continue
        evidence = sampling["evidence"]
        key = CellKey.model_validate(cell)
        row = rows_by_cell[key].model_dump(mode="json")
        log_mean, log_se = _relative_late_reference(row["lift"])
        statistic = (log_mean - math.log1p(row["null_lift"])) / log_se
        implied = {
            "greater": norm.sf(statistic),
            "less": norm.cdf(statistic),
            "two-sided": 2.0 * norm.sf(abs(statistic)),
        }[row["alternative"]]
        assert evidence["hypothesis"] == asdict(key.hypothesis()), evidence
        assert (evidence["method"], evidence["reference"]) == (cell["method"], "normal"), evidence
        assert math.isclose(evidence["p_value"], implied, rel_tol=TOLERANCE), (evidence, implied)
        projected.append({**record, "sampling": {**sampling, "evidence": None}})
    return {**actual, "cells": projected}
