import random
from collections import Counter

from ocgforge import worlds as w


def card(code, owner, location, sequence, position=0x8, public=0):
    return {"code": code, "owner": owner, "controller": owner, "location": location, "sequence": sequence,
            "position": position, "public": public, "materials": [], "counters": []}


def test_revealed_cards_are_dealt_to_the_hand_not_the_deck():
    # The opponent's deck comes first in a snapshot; a card it revealed must still land in its hand.
    deck = [card(9000 + i, 1, w.LOCATION_DECK, i) for i in range(10)]
    hand = [card(9100 + i, 1, w.LOCATION_HAND, i, position=0xA) for i in range(3)]
    mine = [card(1, 0, w.LOCATION_HAND, 0, position=0xA)]
    snapshot = {"cards": deck + hand + mine, "life_points": [8000, 8000]}
    decklists = {1: ([2001] + list(range(3000, 3012)), [])}
    for seed in range(5):
        script = w.sample_world(snapshot, 0, decklists, random.Random(seed), revealed=Counter({2001: 1}))
        in_hand = [line for line in script.splitlines() if f",{w.LOCATION_HAND}," in line and "AddCard(2001," in line]
        assert in_hand, script


def test_match_keeps_the_command():
    summon = "idle.0.3.12375297.0.2.1"
    set_ = "idle.3.0.12375297.0.2.1"
    activate = "idle.5.1.12375297.0.2.1"
    assert w.match(summon, [set_, activate]) is None
    assert w.match(summon, [set_, "idle.0.0.12375297.0.2.4"]) == "idle.0.0.12375297.0.2.4"  # other list index and spot
    assert w.match("card.0.2.12375297.0.1.7", ["card.0.5.12375297.0.1.3"]) == "card.0.5.12375297.0.1.3"


def test_canonical_drops_the_list_index():
    assert w.canonical("idle.1.4.13597785.0.2.0") == "idle.1.13597785.0.2.0"
    assert w.canonical("chain.pass") == "chain.pass"
