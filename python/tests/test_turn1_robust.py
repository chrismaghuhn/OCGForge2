"""turn1_search / outcome_eval robustness: resume, sharing, picking, engine seeds (synthetic only)."""

import json
import threading
import time
from pathlib import Path

import pytest

from ocgforge import book_policy as bp
from ocgforge import outcome_eval as oe
from ocgforge import turn1_search as ts


class Pos:
    """Just enough of a duel for `position`."""

    def __init__(self, deck, finished=False):
        self.deck, self.finished, self.winner = deck, finished, None
        self.turn, self.phase, self.player = 2, 1, 1

    def snapshot(self):
        cards = [{"controller": 0, "owner": 0, "location": 1, "code": c, "sequence": i, "position": 10}
                 for i, c in enumerate(self.deck)]
        return {"cards": cards, "life_points": (8000, 8000)}

    def keys(self):
        return ["chain.pass"]


def test_position_tells_deck_orders_apart():
    assert ts.position(Pos([1, 2, 3])) != ts.position(Pos([3, 2, 1]))
    assert ts.position(Pos([1, 2, 3])) == ts.position(Pos([1, 2, 3]))


def test_pick_leaves_out_a_candidate_with_an_error_in_any_scenario():
    def play(score, error=False):
        p = ts.Play([], [], None, [])
        p.score, p.value, p.error = score, score, error
        return p
    plays = [[play(0.9), play(0.9, error=True), play(0.9)], [play(0.5), play(0.5), play(0.5)],
             [play(0.5), play(0.5), play(0.5)]]
    assert ts.pick(plays) == 1  # 0 has the best mean but a scenario missing; ties: the earlier candidate
    plays[1][2].score = None
    assert ts.pick(plays) == 2
    assert ts.pick([[play(0.1, error=True)]]) is None


class Seat:
    """An open duel whose pending decision is `player`'s; the net values it `worth` for the actor."""

    def __init__(self, player, worth):
        self.player, self.worth, self.finished, self.winner = player, worth, False, None

    def encode(self, index):
        return {"player": self.player, "worth": self.worth}


def test_terminal_is_seat_zeros_view_whoever_decides():
    net = ts.Net(lambda rows: [([0.0], r["worth"]) for r in rows], {})
    # our own decision at the end of turn 1 (e.g. a chain window of ours): the value as it is;
    # the opponent's: negated
    assert net.terminal([Seat(0, 0.4), Seat(1, 0.4)]) == [0.4, -0.4]
    done = Seat(0, 0.0)
    done.finished, done.winner = True, 1
    assert net.terminal([done]) == [-1.0]


def test_resume_drops_torn_lines_and_lines_of_unfinished_hands(tmp_path):
    out = tmp_path / "x-t1.jsonl"
    p = ts.paths(out)
    p["hands"].write_text('{"hand": "a"}\n{"hand": "b"}\n{"hand": ', encoding="utf-8")
    p["lines"].write_text('{"seed": "a"}\n{"seed": "c"}\n{"seed": "b"', encoding="utf-8")
    p["rollouts"].write_text('{"hand": "a", "score": 1}\n{"hand": "c", "score": 0}\n', encoding="utf-8")
    assert ts.done_hands(out) == {"a", "b"}
    assert ts.read_jsonl(p["lines"]) == [{"seed": "a"}]
    assert ts.read_jsonl(p["hands"]) == [{"hand": "a"}, {"hand": "b"}]
    assert ts.read_jsonl(p["rollouts"]) == [{"hand": "a", "score": 1}]


def test_write_book_finds_the_opponent_of_another_checkout(tmp_path, monkeypatch):
    (tmp_path / "decks").mkdir()
    (tmp_path / "decks" / "maliss.ydk").write_text("")
    (tmp_path / ".cache" / "yugiohmeta" / "decks").mkdir(parents=True)
    (tmp_path / ".cache" / "yugiohmeta" / "decks" / "yummy.ydk").write_text("")
    monkeypatch.setattr(bp.rollout, "REPO", tmp_path)
    assert bp.resolve("/elsewhere/ocgforge/decks/maliss.ydk") == tmp_path / "decks" / "maliss.ydk"
    assert bp.resolve("/elsewhere/.cache/yugiohmeta/decks/yummy.ydk") == tmp_path / ".cache/yugiohmeta/decks/yummy.ydk"
    assert bp.resolve(".cache/yugiohmeta/decks/yummy.ydk") == tmp_path / ".cache/yugiohmeta/decks/yummy.ydk"
    used = []
    monkeypatch.setattr(ts.cc, "puzzle_start", lambda deck, opp, script, seed: used.append(opp) or (lambda: None))
    monkeypatch.setattr(ts.ob, "build_book", lambda lines, relevant: ({}, {}))
    monkeypatch.setattr(ts.ob, "book_cards", lambda records: set())
    records = [{"seed": "a", "opponent": "/elsewhere/.cache/yugiohmeta/decks/yummy.ydk", "script": "", "duel_seed": 1},
               {"seed": "b", "script": "", "duel_seed": 2}]
    ts.write_book(tmp_path / "b-t1.jsonl", Path("deck.ydk"), Path("default.ydk"), records, {})
    assert used == [tmp_path / ".cache/yugiohmeta/decks/yummy.ydk", Path("default.ydk")]


def test_one_bad_request_fails_only_its_own_caller():
    def fn(rows):
        if "bad" in rows:
            raise ValueError("bad row")
        return [([0.0], 0.0) for _ in rows]
    ev = ts.Evaluator(fn, wait=0.2)  # long enough that both requests land in one batch
    out = {}

    def ask(name, rows):
        try:
            out[name] = ev(rows)
        except ValueError as error:
            out[name] = error
    threads = [threading.Thread(target=ask, args=("good", ["x", "y"])), threading.Thread(target=ask, args=("bad", ["bad"]))]
    for t in threads:
        t.start()
        time.sleep(0.01)
    for t in threads:
        t.join()
    ev.close()
    assert out["good"] == [([0.0], 0.0)] * 2 and isinstance(out["bad"], ValueError)


class Seeded:
    """A duel that remembers the engine seeds of its clones (and that a finished one cannot clone)."""

    def __init__(self, finished=False, winner=None, log=None):
        self.finished, self.winner, self.log = finished, winner, log if log is not None else []
        self.turn, self.player = 3, 0

    def clone(self, seed=None):
        if self.finished:
            raise RuntimeError("the duel is over")
        self.log.append(seed)
        return Seeded(log=self.log)

    def keys(self):
        return ["idle.7.0"]

    def encode(self, index):
        return {"player": 0}

    def snapshot(self):
        return {"cards": [], "life_points": (8000, 8000)}


def test_rollouts_share_engine_seeds_and_score_finished_duels():
    net = ts.Net(lambda rows: [([0.0], 0.25) for _ in rows], {})
    a, b = Seeded(), Seeded()
    oe.play_turn2([a, b], net, 2, ["s-k0", "s-k0"])
    assert a.log == b.log and len(set(a.log)) == 2 and None not in a.log  # the same seed per (k, j)
    lost = Seeded(finished=True, winner=1)
    rollouts = oe.play_turn2([lost], net, 2, ["s"])[0]
    assert [r.value for r in rollouts] == [-1.0, -1.0] and not any(r.error for r in rollouts)


def test_failed_hands_end_the_run_with_their_own_exit_code(tmp_path, monkeypatch):
    """main() writes <out>-failed.json and exits FAILED_HANDS (so the driver retries) when a hand raised."""
    assert ts.FAILED_HANDS == 3
    from ocgforge import turn1_iteration as ti
    assert ti.accept(0, 1) and not ti.accept(ts.FAILED_HANDS, 1) and ti.accept(ts.FAILED_HANDS, ti.FAILED_TRIES)
    assert not ti.accept(1, 99)
    path = ts.paths(tmp_path / "x-t1.jsonl")["failed"]
    path.write_text(json.dumps(["h1"]))
    assert ti.failed_hands(tmp_path / "x-t1.jsonl") == ["h1"]


def test_the_gain_is_measured_apart_from_the_grading():
    def play(score):
        p = ts.Play([], [], None, [])
        p.score = score
        return p
    out = ts.paired_gain([play(0.5), play(0.3), play(0.4)], [play(0.1), play(0.3), play(0.2)])
    assert out["gain"] == pytest.approx(0.2) and out["gain_n"] == 3 and out["gain_se"] == pytest.approx(0.1155, abs=1e-3)
    record = {"search": {"gain": 0.2, "gain_se": 0.15, "score": 0.9, "own_score": 0.1, "scorer": "turn2"}}
    assert ts.gains(record) == 0.2  # not the grading's 0.8
    assert ts.keeps(record, 0.1) and not ts.keeps(record, 0.1, min_gain_se=2.0) and not ts.keeps(record, 0.25)
