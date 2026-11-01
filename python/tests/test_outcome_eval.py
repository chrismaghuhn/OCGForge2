"""outcome_eval and the turn-2 scorer of turn1_search (synthetic duels only)."""

import copy
from pathlib import Path

import pytest

from ocgforge import outcome_eval as oe
from ocgforge import turn1_search as ts

GOOD, NEGATE, PLAIN = 22220001, 22220002, 22220003
WEIGHT = {GOOD: 3, NEGATE: 1, PLAIN: 1}


class Turn2:
    """From the end of our turn 1: seat 1 may wipe our field ("wipe", which NEGATE on our field
    stops once: our chain window "chain.negate"), attack an empty field (wins the duel) or end its
    turn; then turn 3 begins at our decision."""

    def __init__(self, field, loop=False):
        self.field, self.grave = list(field), []
        self.turn, self.phase, self.player = 2, 4, 1
        self.finished, self.winner, self.pending, self.loop = False, None, None, loop

    def keys(self):
        if self.turn >= 3:
            return ["idle.7.0", "idle.6.0"]
        if self.player == 0:
            return ["chain.pass", "chain.negate"]
        if self.loop:
            return ["idle.6.0", "idle.6.1"]
        return ["idle.7.0", "wipe"] + (["attack"] if not self.field else [])

    def act(self, key):
        assert key in self.keys(), key
        if self.player == 0:
            if key == "chain.negate":
                self.field.remove(NEGATE)
                self.grave.append(NEGATE)
            else:
                self.grave += self.field
                self.field = []
            self.turn, self.player = 3, 0
            return
        if key == "wipe":
            if NEGATE in self.field:
                self.player = 0
            else:
                self.grave += self.field
                self.field = []
                self.turn, self.player = 3, 0
        elif key == "attack":
            self.finished, self.winner = True, 1
        elif key == "idle.7.0":
            self.turn, self.player = 3, 0

    def clone(self):
        return copy.deepcopy(self)

    def snapshot(self):
        cards = [{"controller": 0, "owner": 0, "location": 4, "code": c, "sequence": i, "position": 1}
                 for i, c in enumerate(self.field)]
        return {"cards": cards, "life_points": (8000, 8000)}

    def encode(self, index):
        return {"candidate_kind": [0] * len(self.keys()), "player": self.player, "keys": self.keys(),
                "worth": sum(WEIGHT[c] for c in self.field)}


def toy_net(row):
    """Seat 1 wants to wipe (or attack an empty field); we negate when we can. Values our field."""
    keys = row["keys"]
    liked = {"wipe", "attack", "idle.6.0"} if row["player"] == 1 else {"chain.negate", "idle.7.0"}
    lp = [0.0 if k in liked else -40.0 for k in keys]
    if "attack" in keys:
        lp[keys.index("wipe")] = -40.0
    value = 0.1 * row["worth"]
    return lp, value if row["player"] == 0 else -value


class Counting:
    def __init__(self):
        self.rows = 0

    def __call__(self, rows):
        self.rows += len(rows)
        return [toy_net(r) for r in rows]


def toy():
    return ts.Net(Counting(), {})


def plays_of(boards, net):
    """One play per board (one scenario each), at the end of our turn 1 as judge(keep=True) leaves it."""
    rows = []
    for board in boards:
        play = ts.Play([], [], Turn2(board), [])
        play.value = play.score = net.terminal([play.duel])[0]
        play.on_field = len(board)
        rows.append([play])
    return rows


def test_a_board_that_holds_beats_a_bigger_one_that_is_wiped():
    big, small = [GOOD, GOOD, PLAIN], [NEGATE, GOOD]
    net = toy()
    plays = plays_of([big, small], net)
    assert ts.pick(plays, "value") == 0  # the end-of-turn value likes the big board
    ts.score_turn2(plays, net, ts.Settings(scorer="turn2", rollouts=2), "h")
    assert plays[0][0].score == pytest.approx(0.0) and plays[0][0].t2["field3"] == 0
    assert plays[1][0].score == pytest.approx(0.3) and plays[1][0].t2["field3"] == 1  # negated: GOOD stays
    assert ts.pick(plays) == 1
    assert all(p.duel is None for row in plays for p in row)


def test_an_empty_field_loses_in_turn_two():
    net = toy()
    rollouts = oe.play_turn2([Turn2([])], net, 3, ["s"])[0]
    assert all(r.value == -1.0 and r.stats["lost"] and r.stats["field"] == 0 for r in rollouts)
    assert oe.summary(rollouts)["lost"] == 1.0


def test_capped_rollouts_are_scored_at_their_pending_decision():
    net = toy()
    rollouts = oe.play_turn2([Turn2([GOOD], loop=True)], net, 1, ["s"], max_decisions=5)[0]
    assert rollouts[0].capped and rollouts[0].decisions == 5 and rollouts[0].value == pytest.approx(0.3)


def test_the_same_position_shares_its_rollouts():
    net = toy()
    plays = plays_of([[GOOD, NEGATE], [GOOD, NEGATE], [PLAIN]], net)
    ts.score_turn2(plays, net, ts.Settings(scorer="turn2", rollouts=4), "h")
    assert plays[0][0].t2 is plays[1][0].t2 and plays[0][0].sig == plays[1][0].sig
    single = toy()
    ts.score_turn2(plays_of([[GOOD, NEGATE], [PLAIN]], single), single, ts.Settings(scorer="turn2", rollouts=4), "h")
    assert net.evaluate.rows == single.evaluate.rows + 1  # one more terminal value at the start, no more rollouts


def test_mix_and_value_scorers():
    assert oe.combine("value", 0.5, -1.0) == 0.5
    assert oe.combine("turn2", 0.5, -1.0) == -1.0
    assert oe.combine("mix", 0.5, -1.0, 0.25) == pytest.approx(0.25 * 0.5 - 0.75)
    assert oe.combine("mix", None, -1.0) is None and oe.combine("turn2", 0.5, None) is None
    net = toy()
    plays = plays_of([[GOOD, GOOD, PLAIN], [NEGATE, GOOD]], net)
    ts.score_turn2(plays, net, ts.Settings(scorer="mix", rollouts=1, mix_weight=0.5), "h")
    assert plays[0][0].score == pytest.approx(0.5 * 0.7) and plays[1][0].score == pytest.approx(0.5 * 0.4 + 0.5 * 0.3)


def test_sampling_follows_the_policy_and_is_seeded():
    import random
    assert oe.sample([0.0, -40.0], random.Random(1)) == 0
    a = [oe.sample([-0.69, -0.69], random.Random("x-0")) for _ in range(5)]
    b = [oe.sample([-0.69, -0.69], random.Random("x-0")) for _ in range(5)]
    assert a == b


def test_gain_uses_the_scorer():
    assert ts.gains({"search": {"value": 0.5, "own_value": 0.1}}) == pytest.approx(0.4)
    record = {"search": {"scorer": "turn2", "value": 0.5, "own_value": 0.1, "score": 0.2, "own_score": 0.25}}
    assert ts.gains(record) == pytest.approx(-0.05)
    assert ts.gains({"search": {"scorer": "turn2", "value": 0.5, "own_value": 0.1}}) is None


def test_compare_counts_other_positions_only():
    cands = [{"value": 0.5, "score": 0.1, "field": 5, "lost": 0.0}, {"value": 0.2, "score": 0.3, "field": 3, "lost": 0.0}]
    rows = [{"scorer": "turn2", "cands": cands, "best_index": 1, "value_index": 0, "own_index": 1,
             "same_as_value": False, "same_as_own": True, "value_same_as_own": False},
            {"scorer": "turn2", "cands": cands, "best_index": 1, "value_index": 0, "own_index": 1,
             "same_as_value": True, "same_as_own": True, "value_same_as_own": True}]
    out = ts.compare(rows)
    assert out["differs"] == 0.5 and out["picked_own"] == 1.0 and out["value_picked_own"] == 0.5
    assert out["kept"]["field"] == 3 and out["value_kept"]["field"] == 5


def test_real_opponents_play_with_their_extra_deck():
    opponents = [(Path(f"opp{i}.ydk"), [100 + i] * 10, [900 + i]) for i in range(3)]
    hand = ts.Hand("t1-x-0-0", Path("x.ydk"), "X", Path("maliss.ydk"), ([1] * 40, [2]), [7] * 40, [1] * 5, None, 4,
                   opponents)
    chosen = [hand.opponent_of(k) for k in range(4)]
    assert chosen == [hand.opponent_of(k) for k in range(4)] and all(c in opponents for c in chosen)
    script, _ = hand.script(0)
    path, main, extra = chosen[0]
    assert f"Debug.AddCard({extra[0]},1,1,64,0,8)" in script and f"Debug.AddCard({main[0]},1,1,2," in script
    plain = ts.Hand("t1-x-0-0", Path("x.ydk"), "X", Path("maliss.ydk"), ([1] * 40, [2]), [7] * 40, [1] * 5, None, 4)
    assert plain.opponent_of(0) == (Path("maliss.ydk"), [7] * 40, []) and ",1,1,64," not in plain.script(0)[0]


def test_search_hand_with_the_turn2_scorer_records_scores_and_candidates():
    from test_turn1_search import net as toy_t1_net, toy_hand

    records, stats = ts.search_hand(toy_hand(), toy_t1_net(), ts.Settings(beam=4, branch=3, scorer="turn2"), set(),
                                    lambda c: False)
    base = stats["base"]
    # the toy's turn 2 never ends: every rollout is capped and scored at seat 1's pending decision
    assert base["scorer"] == "turn2" and base["score"] == pytest.approx(base["value"])
    assert len(base["cands"]) == base["candidates"] and all(c["capped"] == 1.0 for c in base["cands"])
    assert base["best_index"] == base["value_index"] and base["same_as_value"] is True
    assert records[0]["opponent"] == "opp.ydk" and ts.gains(records[0]) == pytest.approx(base["score"] - base["own_score"])
    report = ts.summarize([stats])
    assert report["base"]["scorer"]["differs"] == 0.0 and report["base"]["scorer"]["hands"] == 1


def test_shards_split_the_hands_and_merge_in_order(tmp_path):
    from ocgforge import turn1_iteration as ti

    assert ti.shard_ranges(10, 3) == [(0, 4), (4, 3), (7, 3)] and ti.shard_ranges(2, 4) == [(0, 1), (1, 1)]
    book = ti.book_path(tmp_path, Path("decks/elfnote.ydk"))
    shards = [ti.shard_book(book, i) for i in range(2)]
    shards[0].parent.mkdir()
    for i, shard in enumerate(shards):
        ts.paths(shard)["lines"].write_text(f'{{"seed": "l{i}"}}\n')
        ts.paths(shard)["hands"].write_text(f'{{"hand": "h{i}"}}\n')
    ti.merge_shards(book, shards)
    assert ts.paths(book)["lines"].read_text() == '{"seed": "l0"}\n{"seed": "l1"}\n'
    assert ts.paths(book)["hands"].read_text() == '{"hand": "h0"}\n{"hand": "h1"}\n'
