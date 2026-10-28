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
