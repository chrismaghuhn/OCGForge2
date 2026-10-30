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
