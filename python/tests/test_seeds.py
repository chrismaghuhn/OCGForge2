"""seeds: seeds from player sheets, Duelingbook logs and community combos; nicknames; grid hands."""

from ocgforge import combos as cb
from ocgforge import seeds as sd

CARDS = {"Aluber the Jester of Despia": 1, "Mirrorjade the Iceblade Dragon": 2, "Branded Banishment": 3,
         "Branded Lost": 4, "Fallen of Albaz": 5, "Droll & Lock Bird": 6, "Elfnote Tinia": 7,
         "Elfnotes: Welcome Home": 8, "Elfnote Regina": 9, "Ash Blossom & Joyous Spring": 10,
         "Elfnote Lucina": 11, "Mulcharmy Fuwalos": 12, "Crystal Wing Synchro Dragon": 13,
         "Elfnote Seraphim Strelitzia": 14, "Elfnotes: Rhapsodia of Madness": 15}
NAMES = {cb.normalize(name): code for name, code in CARDS.items()}
POOL = set(CARDS.values())
CARD_NAME = {code: name for name, code in CARDS.items()}
resolve = sd.resolver(NAMES, POOL, CARD_NAME)


def test_resolver_nicknames_and_parts():
    assert resolve("Aluber") == 1  # unique part of a pool card's name
    assert resolve("Banishment") == 3 and resolve("Lost") == 4  # aliases
    assert resolve("Albaz in gy.") == 5  # location note dropped
    assert resolve("Albaz (version 2)") == 5
    assert resolve("Elfnote") is None  # not unique


def test_sheet_rows(tmp_path):
    path = tmp_path / "sheet.csv"
    path.write_text(',,Combo,Endboard,Replay,Notes\n'
                    ',1,Aluber + 1 Discard,Mirrorjade + Banishment + Lost,https://x,Plays into droll.\n'
                    ',2,Mystery card + 1 Discard,Mirrorjade,https://x,\n'
                    ',3,Aluber (otk),Mirrorjade,https://x,\n', encoding="utf-8")
    seeds = sd.sheet_seeds(path, "Branded", resolve)
    assert [s.hand for s in seeds] == [[1], [1]]
    assert seeds[0].combo.end_board == [2, 3, 4] and seeds[0].vs == [6] and seeds[0].first
    assert not seeds[1].first


def test_list_sheets(tmp_path):
    tsv = tmp_path / "lines.tsv"
    tsv.write_text("Pure Elfnote\tElfnote Lucina + Discard vs Fuwalos\t1-2\n"
                   "Pure Elfnote\tWelcome Home + Any WIND Monster\t1-3\n"
                   "Pure Elfnote\tUnknown Card\t1-4\n", encoding="utf-8")
    seeds = sd.list_seeds(tsv, "Elfnote", resolve)
    assert [(s.hand, s.vs) for s in seeds] == [([11], [12]), ([8], [])]
    csv_path = tmp_path / "sheet1.csv"
    csv_path.write_text('Card,Board,Notes,Verified\n'
                        'Tinia,"Crystal Wing, Strelitzia, Rhapsodia, 4 cards in hand",Loses to Nib,TRUE\n',
                        encoding="utf-8")
    (seed,) = sd.list_seeds(csv_path, "Elfnote", resolve)
    assert seed.hand == [7] and seed.combo.end_board == [13, 14, 15]


def test_borrow_plans_from_community_seed():
    plan = cb.Combo("c", "Lucina 1cc", "Elfnote", 3, [], [[cb.Entry(11, "Elfnote Lucina", "Special Summon")]],
                    end_board=[13])
    seeds = [sd.Seed("c", "Lucina 1cc", "community", [11], plan),
             sd.Seed("s", "Lucina vs Fuwalos", "sheet", [11], cb.Combo("s", "x", "Elfnote", 0, [], []), [12])]
    sd.borrow_plans(seeds)
    assert seeds[1].combo.steps == plan.steps and seeds[1].combo.end_board == [13] and seeds[1].vs == [12]


def test_replay_log(tmp_path):
    log = tmp_path / "r.txt"
    log.write_text("# header\n"
                   "Enter M1: Entered Main Phase 1\n"
                   'To B Deck: Returned "Elfnote Regina" from hand (1/5) to bottom of deck\n'
                   'SS ATK: Special Summoned "Elfnote Tinia" from hand (2/2) to M-3 (ATK)\n'
                   'To ST: Placed "Elfnotes: Welcome Home" from Deck to S-3\n'
                   'To GY: Sent "Elfnote Tinia" from M-3 to GY\n'
                   'To GY: Sent "Ash Blossom & Joyous Spring" from hand (3/4) to GY\n'
                   'To B Deck: Returned "Ash Blossom & Joyous Spring" from GY to bottom of deck   # misplay\n'
                   'SS DEF: Special Summoned "Elfnote Regina" from Deck to M-3 (DEF)\n'
                   'To hand: Returned "Elfnote Regina" from GY to hand\n', encoding="utf-8")
    seed = sd.replay_seed(log, "Elfnote", resolve)
    assert seed.hand == [7, 10]
    steps = [(e.code, e.action) for step in seed.combo.steps for e in step]
    assert steps == [(7, "Special Summon"), (8, "Place"), (7, "Send to GY"), (9, "Special Summon"),
                     (9, "Add to hand")]


def test_grid_hands_cover_starter_and_extender():
    plan = cb.Combo("a", "Aluber", "Branded", 0, [], [[cb.Entry(1, "", "")]])
    seeds = [sd.Seed("a", "Aluber", "community", [1], plan),
             sd.Seed("b", "Tinia + Regina", "community", [7, 9], plan)]
    grid = sd.grid_seeds(seeds, [1, 7, 9], limit=10)
    assert sorted(tuple(sorted(s.hand)) for s in grid) == [(1, 7), (1, 9)]


def test_authors_are_kept(tmp_path):
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(",,,,,Created by April\n,1,Aluber,Mirrorjade,https://x,\n", encoding="utf-8")
    assert sd.sheet_seeds(sheet, "Branded", resolve)[0].author == "April"
    assert sd.log_author("Elfnote vs Handtraps | Lucina vs Droll | GalacticJoey (plays the opponent's Droll)") == "GalacticJoey"
    assert sd.log_author("Branded Elfnote | Medius the Pure | replay 284601-1 | users GalacticJoey (A = first)") == "GalacticJoey"
    assert sd.log_author("Duelingbook replay 1-2 (Elfnote Discord #replays, posted by baguette 13.06.26)") == "baguette"



def test_going_first():
    assert sd.going_first("Lucina 1cc (Featuring Walbaz)")
    assert sd.going_first("use its second effect, then the 2nd copy")
    assert not sd.going_first("Aluber (otk)") and not sd.going_first("Raye going second board breaker")
    assert not sd.going_first("Combo for turn 2")


def test_plan_key_tells_different_combos_with_the_same_hand_apart():
    a = cb.Combo("a", "A", "Elfnote", 0, [], [[cb.Entry(11, "Elfnote Lucina", "Special Summon")]])
    b = cb.Combo("b", "B", "Elfnote", 0, [], [[cb.Entry(7, "Elfnote Tinia", "Special Summon")]])
    sa, sb = sd.Seed("a", "A", "community", [11], a), sd.Seed("b", "B", "community", [11], b)
    assert sd.plan_key(sa) != sd.plan_key(sb)
    assert sd.plan_key(sa) == sd.plan_key(sd.Seed("c", "C", "sheet", [11], a))  # a borrowed plan is the same


def test_combo_author_from_object_or_name():
    assert cb.author_name({"username": "kanna"}) == "kanna"
    assert cb.author_name("kanna") == "kanna" and cb.author_name(None) == ""
