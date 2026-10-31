"""pinned: pinned Discord replays (short Duelingbook logs + the posts) as seeds (synthetic logs)."""

from ocgforge import combos as cb
from ocgforge import pinned
from ocgforge import seeds as sd

CARDS = {"Branded Fusion": 1, "Fallen of Albaz": 2, "Lubellion the Searing Dragon": 3, "Mirrorjade the Iceblade Dragon": 4,
         "Ash Blossom & Joyous Spring": 5, "Guiding Quem, the Virtuous": 6, "Branded in Red": 7,
         "Tri-Brigade Mercourier": 8, "Granguignol the Dusk Dragon": 9, "Aluber the Jester of Despia": 10}
NAMES = {cb.normalize(name): code for name, code in CARDS.items()}
resolve = sd.resolver(NAMES, set(), {c: n for n, c in CARDS.items()})  # no pool: nicknames need aliases

LOG = """# Branded | Branded Fusion + 1 discard | SomePlayer. SS=Special Summoned, fr=from, Eff=Declared effect of.
A Returned "Aluber the Jester of Despia" fr hand to top of deck
A Added "Branded Fusion" fr Deck to hand
A Added "Ash Blossom & Joyous Spring" fr Deck to hand
A Entered Main Phase 1
A Act "Branded Fusion" fr hand to S-3
A Eff "Branded Fusion" in S-3
A Sent "Fallen of Albaz" fr Deck to GY
A SS "Lubellion the Searing Dragon" fr ED to M-1
A Eff "Lubellion the Searing Dragon" in M-1
A Sent "Ash Blossom & Joyous Spring" fr hand to GY
A "a note from the player about this step"
A SS "Mirrorjade the Iceblade Dragon" fr ED to M-1
"""
POSTS = """# header line
== 111-222 | Katsune 01.01.24 | Branded Fusion + 1 discard | lock line
endboard: Jade (used), set BIR, Merc in hand
== 333-444 | Colossus 02.02.24
End Board: Mirrorjade + Quem (Albaz in GY) + Granguignol
"""


def test_posts():
    posts = pinned.read_posts_text(POSTS)
    assert posts["111-222"].author == "KatsuneNiko" and posts["111-222"].variant == "lock line"
    assert posts["333-444"].author == "ScarTheColossus" and posts["333-444"].hand == ""
    assert posts["333-444"].end_board_text().startswith("Mirrorjade + Quem")


def test_board_cards_keep_the_field():
    from ocgforge import aliases
    r = aliases.with_aliases(resolve)
    assert pinned.board_cards("Jade (used), set BIR, Merc in hand", r) == [4, 7]
    assert pinned.board_cards("Mirrorjade + Quem (Albaz in GY) + Granguignol", r) == [4, 6, 9]
    assert pinned.board_cards("Lost search Merc, double Jade through Ad Lib", r) == [4]


def test_pinned_seeds(tmp_path):
    (tmp_path / "111-222.txt").write_text(LOG, encoding="utf-8")
    (tmp_path / "descriptions.txt").write_text(POSTS, encoding="utf-8")
    (seed,) = pinned.pinned_seeds(tmp_path, "Branded", resolve)
    assert seed.id == "pinned-111-222" and seed.origin == "111-222" and seed.source == "replay"
    assert seed.author == "SomePlayer"  # the log's header first, the post's signature otherwise
    assert seed.hand == [1]  # Ash is only discarded: filler
    assert [e.code for e in seed.combo.starting] == [1]
    assert seed.combo.end_board == [4, 7]
    assert "log: a note from the player about this step" in seed.combo.notes
    assert "lock line" in seed.combo.notes
    assert [step[0].code for step in seed.combo.steps][-1] == 4
