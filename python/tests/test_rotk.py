"""rotk: Road of the King report text -> openings and seeds (synthetic report, made-up lines)."""

from ocgforge import combos as cb
from ocgforge import rotk
from ocgforge import seeds as sd

CARDS = {"Elfnote Lucina": 1, "Elfnote Tinia": 2, "Elfnotes: Welcome Home": 3, "Elfnote Regina": 4,
         "Elfnote Power Patron": 5, "Elfnote Seraphim Strelitzia": 6, "Baronne de Fleur": 7, "Accel Synchron": 8,
         "Kewl Tune B2B": 9, "Prohibited Power Patron Portal - Terminus": 10, "Theorealize Medius": 11,
         "Puzzlomino, the Drop-n-Deleter": 12, "Ars Magna of Infinity and Finity": 13, "Junoldo the Shadespirit Power Patron": 14,
         "Fallen of the White Dragon": 15, "The Crimson King": 16, "Incredible Ecclesia, the Virtuous": 17,
         "Crystal Wing Synchro Dragon": 18}
NAMES = {cb.normalize(name): code for name, code in CARDS.items()}
resolve = sd.resolver(NAMES, set(), {c: n for n, c in CARDS.items()})

REPORT = """OCG 2099.01 Metagame Report #1
Metagame Breakdown
40 Elfnote (30 Elfnote, 10 Synchron Elfnote)
12 Branded
Elfnote
Elfnote gained a new card this format, which changes nothing below.
An opening with Prohibited Power Patron Portal – Terminus and Elfnote Lucina + 1 Monster Card is as follow:
Prohibited Power Patron Portal – Terminus ① effect, send Elfnote Power Patron from deck to graveyard, add Theorealized Medius from deck to hand.
Special Summon Elfnote Lucina from hand to center Main Monster Zone.
Elfnote Lucina ② effect, add Elfnote Tinia from deck to hand.
Theorealized Medius ① effect, Special Summon itself from hand, then add Ars Magna the Finite and the Infinite from deck to hand.
Elfnote Power Patron + Theorealized Medius → Accel Synchron .
Accel Synchron + Elfnote Lucina → Baronne de Fleur (to Main Monster Zone 1).
Special Summon Elfnote Tinia from hand to center Main Monster Zone.
Elfnote Tinia ② effect, place Elfnotes: Welcome Home from deck to Spell & Trap Zone.
Elfnotes: Welcome Home ② effect, send Elfnote Tinia from field to graveyard as cost, Special Summon Elfnote Regina from deck (to center Main Monster Zone).
Elfnote Regina ② effect, Special Summon Elfnote Power Patron from deck.
Elfnote Power Patron ② effect, increase Elfnote Regina Level by 3, Elfnote Power Patron + Elfnote Regina → Elfnote Seraphim Strelitzia .
Chain 2: Elfnote Regina ③ effect, add itself from graveyard to hand.
Chain 1: Junoldo the Shadespirit Power Patron ① Pendulum effect, destroy itself, draw 2 cards.
During opponent’s turn,
Elfnote Regina ① effect, send Elfnotes: Welcome Home from field to graveyard as cost, Special Summon itself from hand.
When opponent activates a Monster Effect, Kewl Tune B2B ③ effect, Special Summon Elfnote Power Patron from graveyard, Elfnote Power Patron + Elfnote Regina → Crystal Wing Synchro Dragon .
The end board consists of 4 disruptions:
Baronne de Fleur negates 1 Card or Effect.
Crystal Wing Synchro Dragon negates 1 Monster Effect.
Elfnote Seraphim Strelitzia counts as two here.
In Step 3, this is commentary that ends the list early.
Somebody went 7-1 and finished 1st in "Some Cup" which had 30 participants.
Branded
Branded was played a lot.
Fallen of the White Dragon ① effect, send something from Extra Deck to graveyard as cost, Special Summon itself from hand.
Fallen of the White Dragon ② effect, Special Summon Incredible Ecclesia, the Virtuous from deck.
Incredible Ecclesia, the Virtuous + Fallen of the White Dragon → The Crimson King .
Medius the Pure (banished) + Puzzlomino, the Drop-n-Deleter → Baronne de Fleur .
Conclusion
Elfnote Lucina ② effect, add Elfnote Tinia from deck to hand.
"""


def openings():
    return rotk.parse(REPORT, resolve)


def test_sections_and_hands():
    elf, branded = openings()
    assert (elf.deck, branded.deck) == ("Elfnote", "Branded")
    assert elf.hand == [10, 1]  # Terminus and Lucina; "1 Monster Card" is filler
    assert branded.intro == "" and branded.hand == [15]  # no intro: what it plays from the hand


def test_steps_effects_chains_turns():
    elf = openings()[0]
    first = elf.steps[0]
    assert (first.code, first.effect, first.pendulum) == (10, 1, False)
    assert first.does.startswith("send Elfnote Power Patron from deck to graveyard")
    assert [(e.code, e.action) for e in first.moves] == [(10, "Activate ① effect"), (5, "Send to GY"),
                                                        (11, "Add to hand")]  # Theorealized -> Theorealize
    medius = elf.steps[3]
    assert [(e.code, e.action) for e in medius.moves][1:] == [(11, "Special Summon"), (13, "Add to hand")]
    chain = [s for s in elf.steps if s.chain]
    assert [(s.chain, s.code, s.chain_group) for s in chain] == [(2, 4, 1), (1, 14, 1)]
    assert chain[1].pendulum and chain[1].effect == 1
    level_up = elf.steps[10]
    assert [e.action for e in level_up.moves] == ["Activate ② effect", "Material for Elfnote Seraphim Strelitzia",
                                                 "Material for Elfnote Seraphim Strelitzia", "Summon"]
    opp = [s for s in elf.steps if s.opponent_turn]
    assert len(opp) == 2 and opp[1].trigger == "When opponent activates a Monster Effect" and opp[1].code == 9


def test_end_board_and_disruptions():
    elf, branded = openings()
    # our field when our turn ends: Baronne, Welcome Home, Strelitzia (Junoldo destroyed itself)
    assert elf.end_board == [7, 3, 6]
    assert elf.stated == 4 and [d.code for d in elf.disruptions] == [7, 18, 6]  # the list ends early
    assert elf.disruptions[0].text == "negates 1 Card or Effect."
    assert 16 in branded.end_board and 7 in branded.end_board  # comma names as materials
    assert "Medius the Pure" in elf.unknown or "Medius the Pure" in branded.unknown


def test_seeds(tmp_path):
    path = tmp_path / "ocg-2099-01-metagame-report-1.txt"
    path.write_text(REPORT, encoding="utf-8")
    (seed,) = rotk.rotk_seeds(path, "elfnote", resolve)
    assert seed.source == "guide" and seed.author == "Road of the King"
    assert seed.origin == "ocg-2099-01-metagame-report-1" and seed.hand == [10, 1] and seed.first
    assert seed.combo.end_board == [7, 3, 6]
    steps = [[e.code for e in step] for step in seed.combo.steps]
    assert [4, 4, 14] in steps  # one chain is one step
    assert all(9 not in s for s in steps)  # the opponent's turn is not searched
    assert "Opponent's turn" in seed.combo.notes and "Crystal Wing" in seed.combo.notes
    assert seed.title.startswith("OCG 2099.01 Metagame Report #1 | Elfnote | An opening with")
    assert len(rotk.rotk_seeds(path, "Branded", resolve)) == 1
    assert rotk.rotk_seeds(path, "Toon", resolve) == []
