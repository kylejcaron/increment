from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.integration


def _snapshot_id(result: Any) -> str | None:
    metadata = getattr(result, "metadata", None)
    if metadata is not None:
        return metadata.scope.snapshot_id
    return result[0].source_snapshot_id if result else None


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
    from increment import Scope
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
            "daily": Scope(view="daily", grain="daily", values=True),
            "daily_lift": Scope(view="daily", grain="daily"),
            "asof": Scope(view="asof", grain="asof", values=True),
            "asof_lift": Scope(view="asof", grain="asof"),
            "breakout": Scope(view="breakout", grain="total"),
        }
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
    """Triggered IDs are frozen operations.json.gz pre-kernel from_definitions oracles."""
    from tests.parity_harness.cases import PARITY_CASES

    case = next(case for case in PARITY_CASES if case.id == scenario)
    analysis = case.build["from_definitions"]()
    try:
        kwargs = {"population": "triggered"}
        if scenario == "triggered_encouragement_compliance_asof":
            kwargs["estimands"] = ("compliance",)
        result = getattr(analysis, operation)(**kwargs)
        assert _snapshot_id(result) == expected
    finally:
        analysis.close()
