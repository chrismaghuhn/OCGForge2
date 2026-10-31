"""db_logs: Duelingbook logs (both exports) as starting hand, events and the opponent's hand traps."""

from ocgforge import db_logs

LONG = """# Duelingbook replay 1-2 (posted by baguette). Test hand: Tinia 1-card combo.
Enter M1: Entered Main Phase 1
To B Deck: Returned "Elfnote Regina" from hand (1/5) to bottom of deck
SS ATK: Special Summoned "Elfnote Tinia" from hand (2/2) to M-3 (ATK)
To ST: Placed "Elfnotes: Welcome Home" from Deck to S-3
To GY: Sent "Sunset Beat" from hand (3/4) to GY
To B Deck: Returned "Sunset Beat" from GY to bottom of deck   # misplay, undone
SS DEF: Special Summoned banished "Ecclesia and the Dark Dragon" (1/1) to M-1 (DEF)
To hand: Added "Elfnote Fortuna" from Deck to hand
"""

SHORT = """# Elfnote vs Handtraps | Lucina vs Droll | GalacticJoey (plays the opponent's Droll himself).
A Returned "Medius the Pure" fr hand to deck bottom
A Added "Elfnote Lucina" fr Deck to hand
A Added "Droll & Lock Bird" fr Deck to hand
A Entered Main Phase 1
A SS "Elfnote Lucina" fr hand to M-3
A Eff "Elfnote Lucina" in M-3
A Sent "Droll & Lock Bird" fr hand to GY
A Eff "Droll & Lock Bird" in GY
A Added "Elfnote Power Patron" fr Deck to hand
A NS "Elfnote Power Patron" fr hand to M-4
"""

TABS = """# Branded Elfnote | Medius the Pure | replay 1-3 | users GalacticJoey (A = first)
A\tTo B Deck\tReturned "Elfnote Lucina" from hand (2/4) to bottom of deck
A\tTo hand\tAdded "Medius the Pure" from Deck to hand
A\tNS\tNormal Summoned "Medius the Pure" from hand to M-3 (ATK)
"""


def kinds(log):
    return [(e.card, e.kind) for e in log.events]


def test_long_export_with_a_misplay_undone():
    log = db_logs.parse(LONG)
    assert log.hand == ["Elfnote Tinia", "Sunset Beat"]  # both played from hand
    assert kinds(log) == [("Elfnote Tinia", "summon"), ("Elfnotes: Welcome Home", "place"),
                          ("Ecclesia and the Dark Dragon", "summon"), ("Elfnote Fortuna", "hand")]


def test_short_export_setup_effects_and_the_opponents_hand_trap():
    log = db_logs.parse(SHORT, {"Droll & Lock Bird"})
    assert log.hand == ["Elfnote Lucina"]  # setup adds; the opponent's Droll is not ours
    assert kinds(log) == [("Elfnote Lucina", "summon"), ("Elfnote Lucina", "use"), ("Elfnote Power Patron", "hand"),
                          ("Elfnote Power Patron", "summon")]
    assert log.traps == {"Droll & Lock Bird": ["Elfnote Lucina"]}  # it answered Lucina's effect


def test_opponent_zone_marks_a_hand_trap():
    text = ('# x vs y\nA Placed "Infinite Impermanence" fr hand to S-3\n'
            'A Moved "Infinite Impermanence" fr S-3 to M2-3\nA SS "Elfnote Lucina" fr hand to M-3\n'
            'A Eff "Elfnote Lucina" in M-3\nA Eff "Infinite Impermanence" in M-3\n')
    log = db_logs.parse(text)
    assert log.traps == {"Infinite Impermanence": ["Elfnote Lucina"]}
    assert kinds(log)[-1] == ("Elfnote Lucina", "use")


def test_comments_are_kept_and_do_not_end_the_setup():
    text = ('# Pure Elfnote | Lucina | GalacticJoey\nA "Lucina is our starter"\n'
            'A Added "Elfnote Lucina" fr Deck to hand\nA SS "Elfnote Lucina" fr hand to M-3\n')
    log = db_logs.parse(text)
    assert log.comments == ["Lucina is our starter"]
    assert log.hand == ["Elfnote Lucina"] and kinds(log) == [("Elfnote Lucina", "summon")]


def test_tab_export():
    log = db_logs.parse(TABS)
    assert log.hand == ["Medius the Pure"] and kinds(log) == [("Medius the Pure", "summon")]
