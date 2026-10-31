from ocgforge.combo_check import forced, wants_center

LUCINA, POWER_PATRON, BARONNE = 13597785, 12375297, 84815190
CENTER = {LUCINA}
is_extra = lambda code: code == BARONNE  # noqa: E731
ZONES = [f"place.0.4.{z}" for z in (0, 1, 2, 3, 4)]


def test_wants_center():
    assert wants_center("You can Special Summon this card (from your hand) to your center Main Monster Zone.")
    assert wants_center("This card's original ATK becomes 3000, while in the center Main Monster Zone.")
    assert not wants_center("You can target 1 monster in your center Main Monster Zone; increase its Level by 3.")


def test_position_is_face_up_attack():
    assert forced([f"position.{LUCINA}.1", f"position.{LUCINA}.4"], [], False, CENTER, is_extra) == f"position.{LUCINA}.1"


def test_zones():
    summon = lambda code: [f"idle.1.0.{code}.0.2.0"]  # noqa: E731
    assert forced(ZONES, summon(LUCINA), True, CENTER, is_extra) == "place.0.4.2"
    assert forced(ZONES, summon(POWER_PATRON), True, CENTER, is_extra) == "place.0.4.1"  # center stays free
    assert forced(ZONES + ["place.0.4.5"], summon(BARONNE), True, CENTER, is_extra) == "place.0.4.5"
    # materials chosen after the summon do not change which card is placed
    history = summon(BARONNE) + [f"unselect.select.0.{LUCINA}.0.4.2"]
    assert forced(ZONES + ["place.0.4.5"], history, True, CENTER, is_extra) == "place.0.4.5"


def test_other_questions_are_searched():
    assert forced(["chain.0.0.14558127.0.2.1", "chain.pass"], [], True, CENTER, is_extra) is None
    assert forced(["yes_no.no", "yes_no.yes"], [], True, CENTER, is_extra) is None


def test_plan_choice_takes_the_next_steps_card():
    from ocgforge.combo_check import plan_choice
    regina, fortuna = 56651978, 85976588
    keys = [f"card.0.0.{fortuna}.0.1.4", f"card.0.1.{regina}.0.1.9", "card.0.2.12375297.0.1.3"]
    assert plan_choice(keys, {0, 1, 2}, [regina, fortuna]) == {1}
    assert plan_choice(keys, {0, 2}, [regina, fortuna]) == {0}  # Regina was tried: the next one
    assert plan_choice(keys, {0, 1, 2}, [LUCINA]) == {0, 1, 2}  # nothing of the plan offered
    assert plan_choice(["chain.pass", f"chain.0.0.{regina}.0.2.1"], {0, 1}, [regina]) == {0, 1}


class FakeDuel:
    """A duel that records its answers; clone() copies them."""

    def __init__(self, keys=()):
        self.played = list(keys)
        self.clones = 0

    def clone(self):
        self.clones += 1
        return FakeDuel(self.played)

    def act(self, key):
        if key == "bad":
            raise RuntimeError("rejected")
        self.played.append(key)


def test_grow_clones_all_but_the_last_child_of_a_parent():
    from ocgforge.combo_check import Line, grow
    a, b = FakeDuel(["x"]), FakeDuel(["y"])
    children = [(Line(["x", "1"]), a), (Line(["y", "1"]), b), (Line(["x", "bad"]), a), (Line(["x", "2"]), a)]
    grown = grow(children)
    assert [line.keys for line in grown] == [["x", "1"], ["y", "1"], ["x", "2"]]
    assert [line.duel.played for line in grown] == [["x", "1"], ["y", "1"], ["x", "2"]]
    assert grown[2].duel is a and grown[1].duel is b  # the last child takes its parent's duel
    assert a.clones == 2 and b.clones == 0


def test_checked_line():
    from ocgforge import combos as cb
    from ocgforge.combo_check import Line, checked_line
    combo = cb.Combo("id", "t", "Elfnote", 0, [], [], end_board=[1, 2, 3, 4, 5])
    targets = [(1, None), (2, None)]
    assert checked_line(combo, Line(progress=2, skipped=2, key_cards=4), targets)
    assert not checked_line(combo, Line(progress=2, skipped=3, key_cards=5), targets)
    assert not checked_line(combo, Line(progress=2, key_cards=3), targets)
    assert not checked_line(combo, Line(progress=1, key_cards=5), targets)


def test_advance_steps_of_one_group_in_any_order():
    from collections import Counter
    from ocgforge.combo_check import Line, advance
    STRELITZIA, REGINA, PATRON, TINIA = 42302563, 56651978, 12375297, 59581480
    targets = [(STRELITZIA, None), (REGINA, None), (PATRON, None), (TINIA, 0x4)]
    groups = [0, 0, 0, 1]
    line = Line()
    named = lambda code: f"unselect.select.0.{code}.0.4.1"  # noqa: E731
    # Regina before Strelitzia, inside the same community step: no skip, Regina done ahead
    steps, skipped, ahead = advance(line, targets, named(REGINA), Counter(), 3, groups)
    assert (steps, skipped, ahead) == (0, (), frozenset({1}))
    line.ahead = ahead
    steps, skipped, ahead = advance(line, targets, named(STRELITZIA), Counter(), 3, groups)
    assert (steps, skipped, ahead) == (2, (), frozenset())
    line.progress = 2
    # Tinia arriving on the field before Power Patron was used: a skip
    steps, skipped, ahead = advance(line, targets, None, Counter({(TINIA, 0x4): 1}), 3, groups)
    assert (steps, skipped) == (2, (2,))


def test_advance_named_card_does_not_jump_its_own_earlier_step():
    from collections import Counter
    from ocgforge.combo_check import Line, advance
    LUCINA, FORTUNA = 13597785, 85976588
    targets = [(LUCINA, 0x4), (FORTUNA, None), (LUCINA, None)]
    summon = f"idle.1.0.{LUCINA}.0.2.0"
    # summoning Lucina names her before she arrives: her later "use" step must wait
    assert advance(Line(), targets, summon, Counter(), 3, [0, 1, 2])[:2] == (0, ())
    # a later step of another card may be done by naming it, at the cost of a skip
    assert advance(Line(), targets, f"chain.0.0.{FORTUNA}.0.4.1", Counter(), 3, [0, 1, 2])[:2] == (2, (0,))


def test_advance_card_already_in_place_is_done_when_named_there():
    from collections import Counter
    from ocgforge.combo_check import Line, advance
    MEDIUS = 97556336
    targets = [(MEDIUS, 0x10)]  # "Effect in GY"
    assert advance(Line(), targets, f"chain.0.0.{MEDIUS}.0.16.0", Counter(), 3)[0] == 1
    assert advance(Line(), targets, f"idle.5.0.{MEDIUS}.0.4.0", Counter(), 3)[0] == 0  # named on the field


def test_opponent_entries_are_not_steps():
    from ocgforge import combos as cb
    from ocgforge.combo_check import step_targets
    fuwalos, lucina = 42141493, 13597785
    combo = cb.Combo("id", "vs Fuwa", "Elfnote", 0, [], [[cb.Entry(lucina, "Elfnote Lucina", "Special Summon")],
                                                          [cb.Entry(fuwalos, "Mulcharmy Fuwalos", "Opp chains Fuwalos")]])
    assert step_targets(combo) == [(lucina, 0x4)]
    combo.steps.append([cb.Entry(fuwalos, "Mulcharmy Fuwalos", "Draw")])
    combo.vs_hand_traps = [fuwalos]
    assert step_targets(combo) == [(lucina, 0x4)]


def test_effect_first_steps_are_done_by_naming_the_card():
    from collections import Counter
    from ocgforge import combos as cb
    from ocgforge.combo_check import Line, advance, step_loose, step_targets
    VIDOLIUM, FALLEN = 70488851, 1
    combo = cb.Combo("id", "t", "Elfnote", 0, [], [[cb.Entry(VIDOLIUM, "Vidolium", "Effect to Revive")],
                                                   [cb.Entry(FALLEN, "Fallen", "Special Summon")]])
    assert step_targets(combo)[0] == (VIDOLIUM, 0x4) and step_loose(combo) == frozenset({0})
    targets = step_targets(combo)
    key = f"idle.5.0.{VIDOLIUM}.0.8.4"  # its Pendulum effect activated in the Pendulum Zone
    assert advance(Line(), targets, key, Counter(), 3, [0, 1], step_loose(combo))[0] == 1
    assert advance(Line(), targets, key, Counter(), 3, [0, 1])[0] == 0


def test_trigger_key_names_the_card_of_an_effect_question():
    from ocgforge import worlds
    from ocgforge.combo_check import trigger_key
    row = {"candidate_source": [2, 2], "token_card": [0, 0, 7]}
    key = trigger_key(row, ["yes_no.no", "yes_no.yes"], 1, {7: 12375297})
    assert worlds.card_ref(key)[1] == "12375297"
    assert trigger_key({"candidate_source": [-1, -1], "token_card": []}, ["yes_no.no", "yes_no.yes"], 1, {}) is None


def test_filler_cards_stay_fixed_over_shuffles():
    import random
    from ocgforge import combos as cb
    from ocgforge.combo_check import filler_rng, scenario_script
    combo = cb.Combo("c1", "t", "Elfnote", 0, [cb.Entry(1, "a", "")], [])
    decklist = (list(range(1, 41)), [])

    def seat0(script, location):
        return [line for line in script.splitlines() if line.startswith("Debug.AddCard(") and ",0,0," in line
                and line.split(",")[3] == str(location)]

    scripts = [scenario_script(combo, decklist, list(range(100, 140)), lambda c: False, random.Random(f"c1-{s}"),
                               filler_rng=filler_rng(combo)) for s in range(3)]
    hands = [seat0(s, 2) for s in scripts]
    assert hands[0] == hands[1] == hands[2] and len(hands[0]) == 5 and hands[0][0].startswith("Debug.AddCard(1,")
    assert seat0(scripts[0], 1) != seat0(scripts[1], 1)  # the deck order still differs
    old = [seat0(scenario_script(combo, decklist, [], lambda c: False, random.Random(f"c1-{s}")), 2) for s in range(3)]
    assert len({tuple(h) for h in old}) > 1  # without filler_rng (books before v3) the fillers vary


def test_offered_bonus_prefers_the_next_step():
    from ocgforge.combo_check import STEP_BONUS, offered_bonus
    assert offered_bonus({1, 2}, [2, 1]) == STEP_BONUS
    assert 0 < offered_bonus({1}, [2, 1]) < STEP_BONUS
    assert offered_bonus({3}, [2, 1]) == 0.0


def test_peek_follows_forced_answers_to_the_next_question():
    from ocgforge.combo_check import peek

    class Duel:
        def __init__(self, script, i=0):
            self.script, self.i, self.player, self.finished = script, i, 0, False

        def clone(self):
            return Duel(self.script, self.i)

        def keys(self):
            return self.script[self.i]

        def act(self, key):
            assert key in self.keys()
            self.i += 1

    crystal_wing = 50954680
    script = [["yes_no.no", "yes_no.yes"], ["chain.pass", "chain.0.0.73642296.0.2.2"],
              [f"position.{crystal_wing}.1", f"position.{crystal_wing}.4"],
              [f"idle.1.0.{crystal_wing}.0.64.4", "idle.7.0"]]
    assert peek(Duel(script), "yes_no.no", [], False, set(), lambda c: False, [crystal_wing]) == {crystal_wing}


def test_expected_location_reads_the_head_verb_and_its_object():
    from ocgforge.combo_check import expected_location as loc
    MZONE, HAND, GRAVE, SZONE, REMOVED = 0x4, 0x2, 0x10, 0x8, 0x20
    assert loc("Special Summon - Activate Effect", "Elfnote Tinia") == MZONE
    assert loc("Special Summon to MMZ", "Elfnote Regina") == MZONE
    assert loc("Synchro Summon -> Search Synchro Rumble", "Crimson Dragon") == MZONE
    assert loc("Add back to hand", "Elfnote Regina") == HAND
    assert loc("Send to GY to make Riser Level 6.", "Effect Veiler") == GRAVE
    assert loc("Banish until EP", "Ecclesia and the Dark Dragon") == REMOVED
    assert loc("Place", "Elfnotes: Welcome Home") == SZONE
    assert loc("return to field", "Ecclesia and the Dark Dragon") == MZONE
    assert loc("SS", "Ecclesia and the Dark Dragon") == MZONE
    # the verb's object is another card: this one was only used
    assert loc("Tribute Tinia", "Elfnotes: Welcome Home") is None
    assert loc("Revive Power Patron", "Synchro Rumble") is None
    assert loc("Search Lucina (Chain Link 1 Power Patron -> Chain Link 2 Riser)", "Elfnote Power Patron") is None
    assert loc("Activate Effect -> Place Continuous Trap", "Elfnote Fortuna") is None
    assert loc("banish fallen and cart", "Albion the Branded Dragon", {"fallen", "cart"}) is None
    assert loc("Send Albion", "Albion the Branded Dragon") == GRAVE
    # a material: "into" another card
    assert loc("Synchro 10 into Baronne De Fleur", "Elfnote Lucina") == GRAVE
    assert loc("Synchro 10 into Baronne", "Baronne de Fleur") == MZONE
    assert loc("Resolve Chain link 2", "Guiding Quem") is None
    assert loc("Rip a card from hand", "PSY-Framelord Omega") is None
    assert loc("", "Elfnote Fortuna") is None


class Turns:
    """Seat 1 answers `seat1` questions, then our turn ends (turn 2) or it is our turn again."""

    def __init__(self, seat1: int, then_turn: int = 1):
        self.left, self.then_turn = seat1, then_turn
        self.finished, self.turn, self.player = False, 1, 1 if seat1 else 0

    def keys(self):
        return ["chain.pass", "chain.0.0.14558127.1.2.0"] if self.turn == 1 else ["idle.1.0.11962031.1.2.0", "idle.7.0"]

    def act(self, key):
        self.left -= 1
        if self.left <= 0:
            self.turn, self.player = self.then_turn, 0 if self.then_turn == 1 else 1


def test_opponent_answers_stop_when_our_turn_is_over():
    from ocgforge.combo_check import opponent_answers
    history = []
    opponent_answers(Turns(2), history)
    assert history == ["chain.pass", "chain.pass"]  # declines, then it is our answer again
    duel, history = Turns(1, then_turn=2), []
    opponent_answers(duel, history)
    assert history == ["chain.pass"] and duel.turn == 2  # our turn ended: no answer in the opponent's turn
    duel, history = Turns(10 ** 6), []
    opponent_answers(duel, history, limit=5)
    assert len(history) == 5  # a looping bot stops


def test_opponent_answers_use_the_bot():
    from ocgforge.combo_check import opponent_answers
    history = ["idle.5.0.13597785.0.2.0"]
    opponent_answers(Turns(1), history, lambda duel, h: duel.keys()[1])
    assert history[-1] == "chain.0.0.14558127.1.2.0"
