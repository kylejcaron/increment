"""Check frozen synthetic truths, gate edges, and complete smoke accounting."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import expm1, log

import numpy as np
import pytest

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
