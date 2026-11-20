"""Check frozen synthetic truths, gate edges, and complete smoke accounting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from itertools import product
from math import expm1, log
from pathlib import Path

import numpy as np
import pytest

from calibration import harm_campaign as campaign
from calibration.harm_bounds import AtomWeights, Band, Cell, Counts4, LeafMap, StudyError
from calibration.harm_campaign import (
    coordinates,
    counter_gate,
    draw_counts,
    evaluate_cell,
    population_truth,
)


def test_truth_and_gate_edges() -> None:
    maps = ((0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 2, 3))
    anchors = (
        ("0", ("0", ".5"), ("0", ".5"), ("0", ".5")),
        (".5", ("0", ".5"), ("0", ".5"), ("0", ".5")),
        ("1", ("1", "1"), ("1", "1"), ("1", "1")),
        ("0", ("0", "0"), ("0", "0"), ("0", "0")),
        (".125", ("0", ".365"), (".125", ".20"), (".125", ".20")),
        (".125", ("0", ".365"), ("0", ".365"), (".125", ".20")),
        (".0195", ("0", ".5"), ("0", ".5"), (".0001", ".0197")),
    )
    for d, (h, *ranges) in enumerate(anchors):
        for leaf_map, (lo, hi) in zip(maps, ranges, strict=True):
            assert population_truth(d, leaf_map) == (Fraction(h), Fraction(lo), Fraction(hi))
    # D5 pools the unseparated S=1 atoms, contributing .365/2 to U, not .20/2.
    assert population_truth(5, (0, 1, 2, 2)) == tuple(map(Fraction, (".125", ".0625", ".2825")))

    assert [counter_gate(k, 20_000, "error")[1] for k in (882, 883, 1121, 1122)] == [
        "green",
        "inconclusive",
        "inconclusive",
        "red",
    ]
    assert [counter_gate(k, 20_000, "power")[1] for k in (15779, 15780, 16217, 16218)] == [
        "red",
        "inconclusive",
        "inconclusive",
        "green",
    ]


def test_split_stream_preserves_only_factual_counts() -> None:
    coord = coordinates()[25]  # D6, n=160, e=.1
    weights = (0.49, 0.49, 0.01, 0.01)
    joint = (
        (0.98, 0.01, 0.01, 0),
        (0, 0.01, 0.01, 0.98),
        (0.01, 0.48, 0.49, 0.02),
        (0.02, 0.49, 0.48, 0.01),
    )
    support = tuple(product(range(4), range(2), range(2), range(2)))
    probs = [weights[g] * joint[g][2 * y0 + y1] * (0.1 if a else 0.9) for g, y0, y1, a in support]
    for split, size in enumerate((53, 107)):
        # Trial 7, with no other coordinates sampled beforehand.
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([20261119082, 6, 0, 1, 7, split]))
        )
        expected = np.zeros((4, 2, 2), dtype=int)
        for count, (g, y0, y1, a) in zip(rng.multinomial(size, probs), support, strict=True):
            expected[g, a, (y0, y1)[a]] += count
        got = draw_counts(coord, 7, split)
        assert [(c.n0, c.y0, c.n1, c.y1) for c in got] == [
            (int(g[0].sum()), int(g[0, 1]), int(g[1].sum()), int(g[1, 1])) for g in expected
        ]


def test_smoke_accounts_for_every_trial() -> None:
    coords = coordinates()
    assert [(c.dgp, c.n, c.e) for c in coords] == list(product(range(7), (160, 4096), (0.5, 0.1)))
    all_gates = []
    for coord in coords:
        cell, decisions, gates, failures = evaluate_cell(coord, 8)
        assert not failures
        assert cell.attempted == 8
        assert len({(m.method, m.mode) for m in cell.methods}) == 10
        for method in cell.methods:
            assert method.attempted == method.valid_bands == 8
            assert method.plugin_trials + method.null_plugin == 8
            assert sum(r.count for r in method.null_plugin_reasons) == method.null_plugin
        assert len({(r.mode, r.target, r.tau) for r in decisions}) == 2 * 10 * 13
        for row in decisions:
            assert row.clear + row.rule_out + row.inconclusive == row.attempted == 8
            assert (
                row.id_abstain + row.sampling_abstain + row.diagnostic_abstain == row.inconclusive
            )
            assert all(loss.vs_never_decide.n == loss.vs_p0.n == 8 for loss in row.losses)
            assert row.false_clear <= gates[0].count and row.false_rule_out <= gates[0].count
            if row.target == "P0":
                assert all(loss.vs_p0.mean == loss.vs_p0.mcse == 0 for loss in row.losses)
        all_gates.extend(gates)
    assert len(all_gates) == 88
    assert sum(record.kind == "power" for record in all_gates) == 4
    assert all(record.attempted == 8 for record in all_gates)
    zero_error = next(
        record for record in all_gates if record.kind == "error" and record.count == 0
    )
    assert zero_error.interval == pytest.approx((0.0, -expm1(log(0.01 / 176) / 8)), abs=1e-13)


@pytest.fixture
def oracle_receipt(tmp_path: Path) -> Path:
    root = Path(campaign.__file__).resolve().parents[1]
    paths = ("calibration/harm_bounds.py", "calibration/harm_oracle.py", "uv.lock")

    def receipt(name: str) -> dict[str, str]:
        return {"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}

    def probability(exact: str) -> dict[str, float | str]:
        return {"exact": exact, "value": float(Fraction(exact))}

    protocol = receipt("calibration/protocols/harm_study_v1.md")
    stamp = "2026-11-19T00:00:00Z"
    rows = []
    thresholds = (0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.32, 0.5)
    for (n0, n1), p, q in product(
        ((4, 4), (9, 11), (25, 20)),
        (0.0, 0.01, 0.2, 0.4, 0.5, 0.99, 1.0),
        (0.0, 0.01, 0.2, 0.4, 0.5, 0.99, 1.0),
    ):
        box = lower = upper = joint = probability("1")
        if (n0, n1, p, q) == (25, 20, 0.2, 0.4):
            denom = "28421709430404007434844970703125"
            box = probability(f"28317335346128600615173559943168/{denom}")
            lower = probability(f"28421709430396677310002672695573/{denom}")
            upper = joint = probability(f"28416127711200596180176134895893/{denom}")
        rows.append(
            {
                "coordinate": {"n0": n0, "n1": n1, "p": p, "q": q},
                "box": box,
                "L": lower,
                "U": upper,
                "joint": joint,
                "decisions": [
                    {"tau": t, "false_clear": probability("0"), "false_rule_out": probability("0")}
                    for t in thresholds
                ],
            }
        )
    path = tmp_path / "oracle.json"
    path.write_text(
        json.dumps(
            {
                "manifest": {
                    "base_revision": "77ab667b227b2005e18bcc3d0f56c23b6aaec651",
                    "source_revision": "8fc12df340d4a0b36f7d4039b2a37f22fcb4eb8e",
                    "protocol_sha256": protocol["sha256"],
                    "sources": [receipt(p) for p in paths],
                    "inputs": [protocol],
                    "invocation": [],
                    "python": "3",
                    "numpy": "2",
                    "scipy": "1",
                },
                "started_at": stamp,
                "ended_at": stamp,
                "status": "passed",
                "tables": 165,
                "marginal_pairs": 81,
                "stratum_table_pairs": 1225,
                "simplex_cases": 5,
                "largest_cp_error": 0,
                "sampling_cells": rows,
                "extrema": [],
                "mismatches": 0,
                "failures": [],
            }
        )
    )
    return path


@pytest.mark.parametrize("action", ("false_rule_out", "false_clear"))
def test_oracle_rejects_wrong_action_exceeding_its_own_endpoint_miss(
    oracle_receipt: Path,
    action: str,
) -> None:
    report = json.loads(oracle_receipt.read_text())
    manifest = report["manifest"]
    protocol = campaign._Receipt.model_validate(manifest["inputs"][0])
    sources = tuple(campaign._Receipt.model_validate(r) for r in manifest["sources"])
    row = next(
        c
        for c in report["sampling_cells"]
        if c["coordinate"] == {"n0": 25, "n1": 20, "p": 0.2, "q": 0.4}
    )
    if action == "false_clear":
        row["L"], row["U"] = row["U"], row["L"]
    row["decisions"][0][action] = {"exact": "1/10000", "value": 0.0001}
    oracle_receipt.write_text(json.dumps(report))
    with pytest.raises(StudyError) as caught:
        campaign._verify_oracle(oracle_receipt, protocol, sources)
    assert caught.value.reason == "oracle_sampling_mismatch"


def test_plugin_counts_rational_threshold_tie(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = Cell(0, 0, 0, 0)
    splits = (
        (Cell(13, 0, 13, 7), empty, Cell(13, 12, 14, 9), empty),
        (Cell(27, 2, 27, 16), empty, Cell(27, 26, 26, 17), empty),
    )
    monkeypatch.setattr(campaign, "draw_counts", lambda _coord, _trial, split: splits[split])
    # The two fixed-S upper endpoints are 2/40 and 14/40 with equal masses.
    _, decisions, _, diagnostics = evaluate_cell(coordinates()[0], 1)
    assert not diagnostics
    for row in (r for r in decisions if r.target == "F" and r.tau == 0.20):
        assert row.plugin is not None
        assert row.plugin.upper_equal == row.plugin.inconclusive == 1


@pytest.mark.parametrize("raise_error", (True, False))
def test_failed_band_keeps_other_methods_and_modes(
    raise_error: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    summarize = campaign.summarize

    def fail(counts: Counts4, leaf_map: LeafMap, *, atom_weights: AtomWeights | None) -> Band:
        if (
            leaf_map != (0, 0, 1, 1)
            or atom_weights is None
            or sum(c.n0 + c.n1 for c in counts) != 160
        ):
            return summarize(counts, leaf_map, atom_weights=atom_weights)
        if raise_error:
            raise StudyError("invalid_cp_interval", (("stage", "sampling"),))
        return replace(
            summarize(counts, leaf_map, atom_weights=atom_weights), outer=(float("nan"), 1.0)
        )

    monkeypatch.setattr(campaign, "summarize", fail)
    cell, rows, counters, failures = evaluate_cell(coordinates()[0], 1)
    assert cell.attempted == 1 and cell.evaluated == 0
    assert not counters and failures
    assert {(r.method, r.mode) for r in cell.methods if r.valid_bands == 0} == {("F", "known")}
    assert all(r.spans.outer.n == r.valid_bands for r in cell.methods)
    available = {(r.target, r.mode): r.evaluated for r in rows if r.tau == 0.5}
    assert available["P0", "known"] == available["P1", "estimated"] == 1
    assert available["P1", "known"] == 0


@pytest.mark.integration
def test_keyboard_interrupt_finalizes_partial_packet(
    tmp_path: Path,
    oracle_receipt: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    draw = campaign.draw_counts

    def interrupted(coord: campaign._Coordinate, trial: int, split: int) -> Counts4:
        if coord.e_index == 1 and trial == 1:
            raise KeyboardInterrupt
        return draw(coord, trial, split)

    monkeypatch.setattr(campaign, "draw_counts", interrupted)
    out = tmp_path / "interrupted"
    assert (
        campaign.main(["--mode", "smoke", "--out", str(out), "--oracle", str(oracle_receipt)]) == 1
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "execution_failure"
    assert manifest["completed_cells"] == 1 and not manifest["certification_eligible"]
    assert manifest["failures"][0]["reason"] == "interrupted"
    assert manifest["oracle_sha256"] == hashlib.sha256(oracle_receipt.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in (out / "cells.jsonl").read_text().splitlines()]
    assert [(r["attempted"], r["evaluated"]) for r in rows] == [(8, 8), (2, 1)]
    assert len(json.loads((out / "gates.json").read_text())["counters"]) == 3
    assert all(
        r["sha256"] == hashlib.sha256((out / r["path"]).read_bytes()).hexdigest()
        for r in manifest["artifacts"]
    )
