"""The journal must account for every draw and detect tampering."""

from __future__ import annotations

import json

import pytest

from calibration.journal import JOURNAL_NAME, DrawJournal, JournalError, verify


def _run(directory, *, draws=2500, block_size=1000, exception_limit=4096, exceptional=()):
    with DrawJournal(
        directory, case_id="c1", block_size=block_size, exception_limit=exception_limit
    ) as journal:
        for index in range(draws):
            journal.started(index)
            journal.completed(index, {"draw": index}, exceptional=index in exceptional)
        return journal.close()


def test_journal_accounts_for_every_draw_without_a_file_per_draw(tmp_path):
    totals = _run(tmp_path, draws=2500)
    assert (totals.started, totals.completed) == (2500, 2500)
    # Three blocks for 2500 draws: the partial tail is sealed on close.
    assert totals.blocks == 3
    assert {path.name for path in tmp_path.iterdir()} == {"journal.jsonl", "exceptions.jsonl"}
    assert verify(tmp_path, case_id="c1") == totals


def test_journal_retains_only_exceptional_draws_and_counts_the_overflow(tmp_path):
    totals = _run(tmp_path, draws=1000, exception_limit=2, exceptional={10, 20, 30})
    assert (totals.exceptional, totals.exceptional_dropped) == (3, 1)
    retained = [json.loads(line) for line in (tmp_path / "exceptions.jsonl").read_text().split()]
    assert [row["draw"] for row in retained] == [10, 20]


@pytest.mark.parametrize("attack", ["drop", "edit", "reorder", "truncate_line"])
def test_journal_verification_rejects_a_tampered_chain(tmp_path, attack):
    _run(tmp_path, draws=3000)
    path = tmp_path / JOURNAL_NAME
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    if attack == "drop":
        lines = [lines[0], lines[2]]
    elif attack == "edit":
        row = json.loads(lines[1])
        row["completed"] -= 1
        lines[1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    elif attack == "reorder":
        lines = [lines[1], lines[0], lines[2]]
    else:
        lines[1] = lines[1][: len(lines[1]) // 2]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(JournalError):
        verify(tmp_path, case_id="c1")


def test_journal_chain_is_bound_to_its_case(tmp_path):
    _run(tmp_path, draws=1000)
    with pytest.raises(JournalError):
        verify(tmp_path, case_id="other")


def test_journal_refuses_use_after_close(tmp_path):
    journal = DrawJournal(tmp_path, case_id="c1")
    journal.started(0)
    journal.close()
    with pytest.raises(JournalError):
        journal.started(1)


def test_verification_reports_a_missing_journal(tmp_path):
    with pytest.raises(JournalError):
        verify(tmp_path, case_id="c1")


def test_exact_block_completion_is_retained_in_verified_journal(tmp_path):
    journal = DrawJournal(tmp_path, case_id="c1", block_size=2)
    journal.started(0)
    journal.started(1)
    journal.completed(1, tally=("ok",))
    journal.completed(0, tally=("ok",))

    live = journal.totals()
    assert (live.blocks, live.completed, dict(live.counters)) == (1, 2, {"ok": 2})
    assert verify(tmp_path, case_id="c1") == live
    journal.close()


def test_journal_reopens_and_appends_after_an_interrupted_final_write(tmp_path):
    first = DrawJournal(tmp_path, case_id="c1", block_size=2)
    first.started(0)
    first.completed(0, tally=("ok",))
    first.close()
    path = tmp_path / JOURNAL_NAME
    path.write_bytes(path.read_bytes() + b'{"interrupted":')

    reopened = DrawJournal(tmp_path, case_id="c1", block_size=2)
    reopened.started(1)
    reopened.completed(1, tally=("ok",))
    totals = reopened.close()

    assert (totals.started, totals.completed, dict(totals.counters)) == (2, 2, {"ok": 2})
    assert verify(tmp_path, case_id="c1") == totals


def test_journal_seals_completed_blocks_in_start_order(tmp_path):
    journal = DrawJournal(tmp_path, case_id="c1", block_size=2)
    for index in range(4):
        journal.started(index)
    journal.completed(3, tally=("ok",))
    journal.completed(2, tally=("ok",))
    assert journal.totals().blocks == 0
    journal.completed(1, tally=("ok",))
    assert journal.totals().blocks == 0
    journal.completed(0, tally=("ok",))

    assert journal.totals().blocks == 2
    assert verify(tmp_path, case_id="c1") == journal.totals()
    journal.close()


def test_exception_sidecar_must_match_sealed_exception_count(tmp_path):
    journal = DrawJournal(tmp_path, case_id="c1", block_size=2)
    journal.started(0)
    journal.completed(0, exceptional=True)
    journal.close()
    (tmp_path / "exceptions.jsonl").write_text("")
    with pytest.raises(JournalError):
        verify(tmp_path, case_id="c1")


def test_exception_records_follow_seal_order_when_completions_are_out_of_order(tmp_path):
    journal = DrawJournal(tmp_path, case_id="c1", block_size=2)
    for index in range(3):
        journal.started(index)
    journal.completed(2, {"draw": 2}, exceptional=True)
    journal.completed(0, {"draw": 0}, exceptional=True)
    journal.completed(1)
    journal.close()

    retained = [
        json.loads(line)["draw"]
        for line in (tmp_path / "exceptions.jsonl").read_text().splitlines()
    ]
    assert retained == [0, 2]
    assert verify(tmp_path, case_id="c1").exceptional == 2


@pytest.mark.parametrize("suffix", [b'{"case_id":"c1","draw":2,"record":null}\n', b"\xff"])
def test_reopen_discards_sidecar_written_before_uncommitted_chain(tmp_path, suffix):
    _run(tmp_path, draws=2, block_size=2, exceptional={0})
    sidecar = tmp_path / "exceptions.jsonl"
    sidecar.write_bytes(sidecar.read_bytes() + suffix)
    assert verify(tmp_path, case_id="c1").completed == 2

    reopened = DrawJournal(tmp_path, case_id="c1", block_size=2)
    reopened.started(2)
    reopened.completed(2)
    reopened.close()

    retained = [json.loads(line)["draw"] for line in sidecar.read_text().splitlines()]
    assert retained == [0]


def test_reopen_does_not_repair_a_tampered_committed_chain(tmp_path):
    _run(tmp_path, draws=2, block_size=2)
    journal = tmp_path / JOURNAL_NAME
    before = journal.read_bytes()
    rows = journal.read_text().splitlines()
    row = json.loads(rows[0])
    row["completed"] -= 1
    journal.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    tampered = journal.read_bytes()

    with pytest.raises(JournalError):
        DrawJournal(tmp_path, case_id="c1", block_size=2)
    assert journal.read_bytes() == tampered
    assert tampered != before
