"""turn1_search: beam search, judging over deck orders, trap branches, output (synthetic duels only)."""

import copy
import json
import threading
from pathlib import Path

import pytest

from ocgforge import opening_book as ob
from ocgforge import turn1_search as ts

DRAWER, GOOD, DUD, PLAIN, ASH = 11110001, 11110002, 11110003, 11110004, 14558127
WEIGHT = {DRAWER: 1, GOOD: 3, DUD: 0, PLAIN: 1}  # what a card on our field is worth to the toy net


class Toy:
    """Seat 0's turn 1: summon a hand card (idle.1), activate DRAWER (idle.5: it goes to the GY and
    the top card of the deck is drawn), or end the turn (idle.7.0). After an activation seat 1 gets
    a chain window, where it may fire ASH from its hand (the draw is negated)."""

    def __init__(self, hand, deck, opponent_hand=()):
        self.hand, self.deck, self.field, self.grave = list(hand), list(deck), [], []
        self.opponent_hand, self.opponent_grave = list(opponent_hand), []
        self.turn, self.phase, self.player, self.turn_player = 1, 4, 0, 0
        self.finished, self.winner, self.window = False, None, False

    def keys(self):
        if self.player == 1:
            return ["chain.pass"] + [f"chain.0.0.{c}.1.2.0" for c in self.opponent_hand if c == ASH and self.window]
        return ([f"idle.1.{i}.{c}.0.2.{i}" for i, c in enumerate(self.hand)]
                + [f"idle.5.{i}.{c}.0.2.{i}" for i, c in enumerate(self.hand) if c == DRAWER] + ["idle.7.0"])

    def act(self, key):
        assert key in self.keys(), key
        if self.player == 1:
            fired = key != "chain.pass"
            if fired:
                self.opponent_hand.remove(ASH)
                self.opponent_grave.append(ASH)
            if self.window and not fired and self.deck:
                self.hand.append(self.deck.pop(0))
            self.window = False
            self.player = 0 if self.turn == 1 else 1
            return
        if key == "idle.7.0":
            self.turn, self.player = 2, 1
            return
        code = int(key.split(".")[3])
        self.hand.remove(code)
        if key.startswith("idle.5."):
            self.grave.append(code)
            self.window, self.player = True, 1
        else:
            self.field.append(code)

    def clone(self):
        return copy.deepcopy(self)

    def snapshot(self):
        def card(controller, location, code, i, public=False):
            return {"controller": controller, "owner": controller, "location": location, "code": code,
                    "sequence": i, "position": 1 if location == 4 else 10, "public": public}
        cards = ([card(0, 2, c, i) for i, c in enumerate(self.hand)] + [card(0, 4, c, i) for i, c in enumerate(self.field)]
                 + [card(0, 0x10, c, i) for i, c in enumerate(self.grave)] + [card(0, 1, c, i) for i, c in enumerate(self.deck)]
                 + [card(1, 2, c, i) for i, c in enumerate(self.opponent_hand)]
                 + [card(1, 0x10, c, i, True) for i, c in enumerate(self.opponent_grave)])
        return {"cards": cards, "life_points": (8000, 8000)}

    def encode(self, index):
        return {"candidate_kind": [0] * len(self.keys()), "player": self.player, "worth": sum(WEIGHT[c] for c in self.field),
                "keys": self.keys()}


def toy_net(row):
    """Likes ending the turn at once (the net's weakness); values our field by WEIGHT."""
    lp = [0.0 if k == "idle.7.0" else -3.0 for k in row["keys"]]
    value = 0.1 * row["worth"]
    return lp, value if row["player"] == 0 else -value


def net():
    return ts.Net(lambda rows: [toy_net(r) for r in rows], {})


def test_beam_beats_the_greedy_net():
    hand = [PLAIN, DRAWER]
    lines = ts.beam_search(Toy(hand, [GOOD]), [], [], net(), ts.Ctx(), beam=4, branch=3)
    assert lines and lines[0].value == pytest.approx(0.4)  # PLAIN, then draw GOOD and summon it
    own = ts.judge([[]], [Toy(hand, [GOOD])], net(), lambda k: ts.Ctx())[0][0]
    assert own.ours == ["idle.7.0"] and own.value == 0.0 and own.on_field == 0


def test_judging_over_deck_orders_drops_a_line_that_only_works_with_this_deck():
    """Drawing pays only where GOOD is on top (scenario 0); summoning DRAWER is a sure +0.1."""
    starts = [Toy([DRAWER], [GOOD])] + [Toy([DRAWER], [DUD]) for _ in range(3)]
    found = [n.ours for n in ts.beam_search(starts[0].clone(), [], [], net(), ts.Ctx(), beam=4, branch=3)]
    draw_line = [f"idle.5.0.{DRAWER}.0.2.0", f"idle.1.0.{GOOD}.0.2.0", "idle.7.0"]
    summon_line = [f"idle.1.0.{DRAWER}.0.2.0", "idle.7.0"]
    assert draw_line in found and summon_line in found
    plays = ts.judge([draw_line, summon_line], starts, net(), lambda k: ts.Ctx())
    assert ts.pick([row[:1] for row in plays]) == 0  # one deck order: the lucky draw wins
    assert ts.pick(plays) == 1  # four: the sure line wins
    # in the other orders the draw line is not offered its GOOD: it diverges, the net ends the turn
    assert all(p.diverged for p in plays[0][1:]) and plays[0][1].ours[-1] == "idle.7.0"


def test_reveal_marks_the_first_answer_after_which_an_unnamed_card_left_the_deck():
    keys = [f"idle.1.0.{PLAIN}.0.2.0", f"idle.5.0.{DRAWER}.0.2.0", "chain.pass", f"idle.1.0.{GOOD}.0.2.0", "idle.7.0"]
    assert ts.reveal_at(lambda: Toy([PLAIN, DRAWER], [GOOD]), keys) == (True, 1)
    assert ts.reveal_at(lambda: Toy([PLAIN], [GOOD]), keys[:1] + keys[-1:]) == (True, None)
    assert ts.reveal_at(lambda: Toy([PLAIN], [GOOD]), keys[1:]) == (False, None)


def test_trap_branch_starts_where_the_trap_fired():
    base = [f"idle.1.0.{PLAIN}.0.2.0", f"idle.5.0.{DRAWER}.0.2.0", f"idle.1.0.{GOOD}.0.2.0", "idle.7.0"]
    bot = ob.hand_trap_bot(ASH, set())
    found = ts.trap_root(Toy([PLAIN, DRAWER], [GOOD], [ASH]), base, ASH, bot, ts.Ctx(bot))
    assert found is not None
    duel, keys, ours = found
    assert ours == base[:2] and keys[-1] == f"chain.0.0.{ASH}.1.2.0" and duel.hand == []  # the draw was negated
    assert ts.trap_root(Toy([PLAIN, DRAWER], [GOOD]), base, ASH, bot, ts.Ctx(bot)) is None  # no trap: never fires


class ToyHand(ts.Hand):
    def start(self, k, trap=None):
        deck = [GOOD] if k == 0 else [DUD]
        return f"script-{k}-{trap}", k, lambda: Toy(self.cards, deck, [trap] if trap else [])


def toy_hand():
    return ToyHand("t1-toy-0-0", Path("toy.ydk"), "Toy", Path("opp.ydk"), ([], []), [], [PLAIN, DRAWER], ASH, 4)


def test_search_hand_keeps_the_line_that_is_best_over_the_deck_orders():
    records, stats = ts.search_hand(toy_hand(), net(), ts.Settings(beam=4, branch=3), set(), lambda c: False)
    assert len(records) == 1
    base = records[0]
    assert stats["base"]["value"] > stats["base"]["own_value"] and stats["base"]["pick"] == "search"
    # drawing pays in one of the 4 orders only: the kept line summons both cards instead
    summons = [k for k in base["line"] if k.startswith("idle.")]
    assert len(summons) == 3 and all(k.startswith("idle.1.") for k in summons[:2]) and summons[2] == "idle.7.0"
    assert base["trap"] is None and base["field"] == 2 and base["value"]["held"] is None and base["fillers"] == []
    assert base["search"]["reveal_at"] is None and base["seed"] == "t1-toy-0-0"
    # the kept line activates nothing, so the hand trap never fires: no branch
    assert stats["trap"]["fired"] is False


def test_branch_follows_the_base_line_until_the_trap():
    s = ts.Settings(beam=4, branch=3)
    hand = toy_hand()
    base = [f"idle.5.1.{DRAWER}.0.2.1", f"idle.1.0.{PLAIN}.0.2.0", "idle.7.0"]
    record, stats, ours = ts.search_branch(hand, ASH, net(), s, set(), lambda c: False, base)
    assert record is not None and record["fired"] and record["trap"] == ASH
    assert ours[0] == base[0] and record["line"][:2] == [base[0], f"chain.0.0.{ASH}.1.2.0"]
    assert record["field"] == 1  # PLAIN; the draw was negated


def test_book_builds_from_the_records():
    records, _ = ts.search_hand(toy_hand(), net(), ts.Settings(beam=4, branch=3), set(), lambda c: False)
    start = toy_hand().start(0)[2]
    lines = [(r, (lambda r=r: Toy(toy_hand().cards, [GOOD], [r["trap"]] if r["trap"] else []))) for r in records]
    relevant = ob.book_cards(records)
    book, _ = ob.build_book(lines, relevant)
    assert {PLAIN, DRAWER} <= relevant and len(book) >= 2
    assert ob.stay(start, book, relevant) >= 2


def test_evaluator_batches_threads_and_passes_errors():
    seen = []

    def fn(rows):
        seen.append(len(rows))
        if "bad" in rows:
            raise ValueError("bad row")
        return [([0.0], float(r)) for r in rows]

    ev = ts.Evaluator(fn, wait=0.05)
    out = {}

    def ask(i):
        out[i] = ev([i, i + 100])

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert {i: [v for _, v in out[i]] for i in out} == {i: [float(i), float(i + 100)] for i in range(4)}
    assert max(seen) > 2  # several threads in one call
    with pytest.raises(ValueError):
        ev(["bad"])
    assert ev([7]) == [([0.0], 7.0)]
    ev.close()


def test_resume_drops_lines_of_unfinished_hands(tmp_path):
    out = tmp_path / "x.jsonl"
    p = ts.paths(out)
    p["hands"].write_text(json.dumps({"hand": "a"}) + "\n", encoding="utf-8")
    p["lines"].write_text("".join(json.dumps({"seed": s}) + "\n" for s in ("a", "a", "b")), encoding="utf-8")
    assert ts.done_hands(out) == {"a"}
    assert [json.loads(t)["seed"] for t in p["lines"].read_text().splitlines()] == ["a", "a"]


def test_summary_counts_lines_and_compares_with_the_net():
    row = {"value": 0.5, "own_value": 0.1, "field": 4, "own_field": 3, "field0": 4, "own_field0": 2, "pick": "search",
           "reveal_at": None, "diverged": 1, "finished": 9, "seconds": 2.0, "replays": True}
    hands = [{"hand": "a", "base": row, "trap": {**row, "pick": "own", "value": 0.1}},
             {"hand": "b", "base": {"error": "no line"}}]
    s = ts.summarize(hands)
    assert s["base"]["kept"] == 1 and s["base"]["errors"] == {"no line": 1} and s["trap"]["picked_own"] == 1.0
    assert s["lines_per_hand"] == 1.0 and s["base"]["better"] == 1.0 and s["trap"]["better"] == 0.0


def test_trap_pool_prefers_hand_traps_that_stop_a_combo():
    names = {1: "Mulcharmy Fuwalos", 2: "Effect Veiler", 3: "Ash Blossom & Joyous Spring", 4: "Unknown"}
    assert ts.trap_pool({1: set(), 2: set(), 3: set(), 4: set()}, names, 2) == [3, 2]
