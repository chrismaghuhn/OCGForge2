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
