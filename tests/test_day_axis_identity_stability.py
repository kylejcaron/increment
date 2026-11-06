from __future__ import annotations

from typing import Any, cast

import pytest

from tests.parity_harness.expected_differences import (
    VALUE_SCALE_SNAPSHOT_REASON,
    ExpectedDifference,
)

pytestmark = pytest.mark.integration


def _snapshot_id(result: Any) -> str | None:
    metadata = getattr(result, "metadata", None)
    if metadata is not None:
        return metadata.scope.snapshot_id
    return result[0].source_snapshot_id if result else None


BINOMIAL_ROW_SNAPSHOT_DIFFERENCES = tuple(
    ExpectedDifference(
        scenario="breakout_store_dimension",
        ingress="from_definitions",
        field=f"{view} snapshot_id",
        frozen=frozen,
        current=current,
        reason=(
            "Lift and breakout snapshots digest arm counts and each row's additive reference. "
            "The pre-kernel route left arm counts off exact binomial and unclustered ratio rows "
            "(both lift views), and exact binomial rows now name the score interval as their "
            "additive reference (all three views); every other snapshot ID is unchanged."
        ),
    )
    for view, frozen, current in (
        (
            "daily_lift",
            "sha256:ed34e487c261b462afc2663879586ddb9a0317d4d2feb0f94fd1049778f54678",
            "sha256:13f1c31232d63f01b6b85f537d2c72c53baf79c64d25cfa5abcbe3072371506f",
        ),
        (
            "asof_lift",
            "sha256:e85a35071b539ee939d5d29fb3e5f31c526aa2241f9617323539c8c52d1139dd",
            "sha256:413112f6af4bd1914c5cc492225adfe810f961262ab19a22347b3c4f07e7c649",
        ),
        (
            "breakout",
            "sha256:b36013595a199643bc99407709b3253008aa20dfdbd310dfed3a1144aea57fc9",
            "sha256:fd017e16f0999da4f298a1692e00322a54d7335ce7c4d1b0f467675ee82feddb",
        ),
    )
)
VALUE_SCALE_SNAPSHOT_DIFFERENCES = tuple(
    ExpectedDifference(
        scenario=scenario,
        ingress=ingress,
        field=f"{view} snapshot_id",
        frozen=frozen,
        current=current,
        reason=VALUE_SCALE_SNAPSHOT_REASON,
    )
    for scenario, ingress, view, frozen, current in (
        (
            "breakout_store_dimension",
            "from_definitions",
            "daily",
            "sha256:64f65bc8a092793edc036331f96643d5f76eaf7d31eae2d27fac636bfdb2f504",
            "sha256:10987ef0000f7a04587c0428edcc4f379602106dd98dc994381d082bd403ee67",
        ),
        (
            "breakout_store_dimension",
            "from_definitions",
            "asof",
            "sha256:6e3358e10d620b1a10a51061deeafd5a5b20b11495103138ba611191abd4679d",
            "sha256:b10a1956f1982c017722963ed11813f9c76d1a5ad7548dc4bfb6ad7693151f6c",
        ),
        (
            "frame_panel_analysis",
            "from_unit_panel",
            "daily",
            "sha256:ef7d9e98a4ad8ab76bffa1c7fdf2cae8b0f20480ff184ef020336882d1c18bba",
            "sha256:8179ba5c6343378eacad3aeaed669407ff07cff1cbbfaaf6b2f0d932b8139abb",
        ),
        (
            "frame_panel_analysis",
            "from_unit_panel",
            "asof",
            "sha256:d90f56f993eb4618eaaa46f7d055d5c5caa3ac3be7aa59d848a3ca45791a66cd",
            "sha256:7996d27c616d2bf7aa1a140cbdf83048e93470f596b0aeca6ff1356da29f15e2",
        ),
    )
)


@pytest.mark.parametrize(
    ("ingress", "expected"),
    [
        (
            "from_definitions",
            {
                "daily": "sha256:64f65bc8a092793edc036331f96643d5f76eaf7d31eae2d27fac636bfdb2f504",
                "daily_lift": "sha256:ed34e487c261b462afc2663879586ddb9a0317d4d2feb0f94fd1049778f54678",
                "asof": "sha256:6e3358e10d620b1a10a51061deeafd5a5b20b11495103138ba611191abd4679d",
                "asof_lift": "sha256:e85a35071b539ee939d5d29fb3e5f31c526aa2241f9617323539c8c52d1139dd",
                "breakout": "sha256:b36013595a199643bc99407709b3253008aa20dfdbd310dfed3a1144aea57fc9",
            },
        ),
        (
            "from_unit_panel",
            {
                "daily": "sha256:ef7d9e98a4ad8ab76bffa1c7fdf2cae8b0f20480ff184ef020336882d1c18bba",
                "daily_lift": "sha256:007ee88da88cf335a7221f24d2f86af55d0decfe94741840e01b375c7fda0501",
                "asof": "sha256:d90f56f993eb4618eaaa46f7d055d5c5caa3ac3be7aa59d848a3ca45791a66cd",
                "asof_lift": "sha256:0a93345621af45231054d32b1f7f15d6179af35e35a7a843f6d5cd925f72bfd1",
                "breakout": "sha256:f6f58be83b5f50d0566a84ef089be07d4af68a2a57994ed6ba0177e870b71508",
            },
        ),
    ],
)
def test_day_axis_source_snapshot_ids_match_pre_kernel_route(
    ingress: str, expected: dict[str, str]
):
    """Kernel-4 commit 192a5026 pre-kernel IDs are spec oracles for this exact fixture data."""
    from increment import View
    from tests.analysis_factory import view_readout
    from tests.parity_harness.cases import PARITY_CASES, _close_parity_binding
    from tests.source_conformance import frame_panel_analysis

    if ingress == "from_definitions":
        case = next(case for case in PARITY_CASES if case.id == "breakout_store_dimension")
        analysis = case.build[ingress]()
    else:
        analysis = frame_panel_analysis()
    experiment, resolver = analysis
    try:
        views = {
            "daily": View(view="daily", grain="daily", values=True),
            "daily_lift": View(view="daily", grain="daily"),
            "asof": View(view="asof", grain="asof", values=True),
            "asof_lift": View(view="asof", grain="asof"),
            "breakout": View(view="breakout", grain="total"),
        }
        for difference in (*BINOMIAL_ROW_SNAPSHOT_DIFFERENCES, *VALUE_SCALE_SNAPSHOT_DIFFERENCES):
            if difference.ingress == ingress:
                expected = difference.applied_to(expected)
        assert {
            name: _snapshot_id(view_readout(experiment, resolver, scope))
            for name, scope in views.items()
        } == expected
    finally:
        _close_parity_binding(analysis)


@pytest.mark.parametrize(
    ("scenario", "operation", "expected"),
    [
        (
            "triggered_daily",
            "run_daily",
            "sha256:26a5b7884c72bcee1a286281f2c5b0044a02fa96ebbfd7c5ce4b7220ad9c16a3",
        ),
        (
            "triggered_daily_lift",
            "run_daily_lift",
            "sha256:906d69949cd2699bf8eea037e0148c8558b5d62b1de8a9ea2427190b54d68eb7",
        ),
        (
            "triggered_asof",
            "run_asof",
            "sha256:b5c8643bfd5b02f42eff25575d68ac638a323dad4902e8b5ec44f995b0040557",
        ),
        (
            "triggered_asof_lift",
            "run_asof_lift",
            "sha256:2eef1a7eb816cca7d6300424b35dba4058e14e9f5e3bf92e6dbd2d400fb6e08b",
        ),
        (
            "triggered_encouragement_compliance_asof",
            "run_asof_lift",
            "sha256:250f3b61e63d68f59821d42f3f56bbdd3969d660903abb8e649ddbf6c3541bce",
        ),
    ],
)
def test_triggered_day_axis_source_snapshot_ids_match_frozen_capture(
    scenario: str, operation: str, expected: str
) -> None:
    """Triggered IDs are frozen operations.json.gz pre-kernel from_definitions oracles, migrated
    by a recorded expected difference where the case has one."""
    from tests.parity_harness.cases import PARITY_CASES, _close_parity_binding
    from tests.parity_harness.expected_differences import with_expected_difference
    from tests.parity_harness.views import ViewName, read

    case = next(case for case in PARITY_CASES if case.id == scenario)
    bound = case.build["from_definitions"]()
    try:
        kwargs: dict[str, Any] = {"population": "triggered"}
        if scenario == "triggered_encouragement_compliance_asof":
            kwargs["estimands"] = ("compliance",)
        result = read(bound, cast(ViewName, operation.removeprefix("run_")), **kwargs)
        assert _snapshot_id(result) == with_expected_difference(
            scenario, "from_definitions", expected
        )
    finally:
        _close_parity_binding(bound)
