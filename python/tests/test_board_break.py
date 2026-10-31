"""board_break: the combo scenario keeps seat 0's cards and gets the agent's for seat 1."""

import random

from ocgforge import board_break as bb


def test_with_agent_replaces_only_seat_one():
    script = "\n".join([
        "Debug.ReloadFieldBegin(0,5)",
        "Debug.AddCard(111,0,0,1,1,8)",  # seat 0's deck card at sequence 1 (looks like ",1,1,")
        "Debug.AddCard(222,0,0,2,0,10)",
        "Debug.AddCard(333,1,1,1,0,8)",
        "Debug.AddCard(444,1,1,2,0,10)",
        "Debug.ReloadFieldEnd()",
    ]) + "\n"
    lines = bb.agent_lines([7, 8, 9, 10, 11, 12], random.Random(0), hand_size=5)
    out = bb.with_agent(script, lines, extra=[99]).splitlines()
    assert "Debug.AddCard(111,0,0,1,1,8)" in out and "Debug.AddCard(222,0,0,2,0,10)" in out
    assert not any(line.startswith(("Debug.AddCard(333,", "Debug.AddCard(444,")) for line in out)
    seat_one = [line for line in out if line.startswith("Debug.AddCard(") and line.split("(")[1].split(",")[1] == "1"]
    assert len(seat_one) == 7  # 1 deck + 5 hand + 1 Extra Deck
    assert out[-1] == "Debug.ReloadFieldEnd()"


def test_wanted_boards():
    assert bb.wanted({"checked": True, "field": 1})
    assert not bb.wanted({"checked": False, "field": 6})
    assert bb.wanted({"checked": False, "field": 6}, min_field=5)
    assert not bb.wanted({"checked": False, "field": 4}, min_field=5)
    assert bb.wanted({"checked": False, "field": 0}, every=True)
    old = {"checked": False, "end_board": [{"location": 4}, {"location": 8}, {"location": 2}]}  # no "field"
    assert bb.wanted(old, min_field=2) and not bb.wanted(old, min_field=3)


def test_defender_modes():
    keys = ["chain.pass", "chain.0.0.1.0.4.0"]
    assert bb.defend(keys, "pass") == "chain.pass"
    assert bb.defend(keys, "eager") == "chain.0.0.1.0.4.0"
    assert bb.defend(["card.0", "card.1"], "eager") == "card.0"


class Script:
    """A duel that walks through (turn, player, keys) states; act() records who answered what."""

    def __init__(self, states):
        self.states, self.i, self.acted, self.finished = states, 0, [], False

    @property
    def turn(self):
        return self.states[self.i][0]

    @property
    def player(self):
        return self.states[self.i][1]

    def keys(self):
        return self.states[self.i][2]

    def encode(self, card_index):
        return {"keys": self.keys()}

    def act(self, key):
        assert key in self.keys()
        self.acted.append((self.player, key))
        self.i += 1
        if self.i >= len(self.states):
            self.finished = True


def test_build_board_plays_seat_zero_and_declines_for_seat_one():
    states = [(1, 0, ["idle.1.0.13597785.0.2.0", "idle.7.0"]), (1, 1, ["chain.pass", "chain.0.0.14558127.1.2.0"]),
              (1, 0, ["idle.7.0"]), (2, 1, ["idle.7.0"])]
    duel = bb.build_board(lambda: Script(states), ["idle.1.0.13597785.0.2.0"])
    assert duel is not None and duel.turn == 2 and duel.player == 1
    assert duel.acted == [(0, "idle.1.0.13597785.0.2.0"), (1, "chain.pass"), (0, "idle.7.0")]
    assert bb.build_board(lambda: Script(states), ["idle.1.9.99999999.0.2.0"]) is None  # the line no longer fits


def test_play_turn_asks_the_policy_for_seat_one_only():
    states = [(2, 1, ["idle.1.0.11111111.1.2.0", "idle.7.0"]), (2, 0, ["chain.pass", "chain.0.0.84815190.0.4.5"]),
              (3, 0, ["idle.7.0"])]
    duel = Script(states)
    asked = []

    def policy(rows):
        asked.append(rows[0]["keys"])
        return [[0]]
    bb.play_turn(duel, policy, None, "eager")
    assert asked == [["idle.1.0.11111111.1.2.0", "idle.7.0"]]
    assert duel.acted == [(1, "idle.1.0.11111111.1.2.0"), (0, "chain.0.0.84815190.0.4.5")]
