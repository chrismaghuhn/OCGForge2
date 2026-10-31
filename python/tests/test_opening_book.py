"""opening_book: the hand-trap bot of the interruption branches."""

from ocgforge import opening_book as ob


class Bot:
    def __init__(self, keys):
        self._keys = keys

    def keys(self):
        return self._keys


def test_hand_trap_bot_waits_for_its_chokepoint():
    ash, nadir, aluber = 14558127, 87462901, 73819701
    bot = ob.hand_trap_bot(ash, {nadir})
    window = ["chain.pass", f"chain.0.0.{ash}.1.2.0"]
    assert bot(Bot(window), [f"idle.5.0.{aluber}.0.2.0"]) == "chain.pass"  # not a choke point
    assert bot(Bot(window), [f"idle.5.0.{nadir}.0.2.0", "chain.pass"]) == f"chain.0.0.{ash}.1.2.0"
    anywhere = ob.hand_trap_bot(ash, set())
    assert anywhere(Bot(window), [f"idle.5.0.{aluber}.0.2.0"]) == f"chain.0.0.{ash}.1.2.0"
    # after firing, a target question aims at a choke-point card
    fired = [f"idle.5.0.{nadir}.0.2.0", f"chain.0.0.{ash}.1.2.0"]
    assert bot(Bot([f"card.0.0.{aluber}.0.4.0", f"card.0.1.{nadir}.0.4.1"]), fired) == f"card.0.1.{nadir}.0.4.1"


class Solo:
    """A two-decision duel: hand cards `hand`, then our summon moves `starter` to the field."""

    def __init__(self, hand, starter):
        self.hand, self.starter, self.step = list(hand), starter, 0
        self.finished, self.turn, self.phase, self.player = False, 1, 4, 0

    def snapshot(self):
        cards = [{"controller": 0, "location": 2, "code": c, "sequence": i, "position": 10, "public": False}
                 for i, c in enumerate(self.hand)]
        if self.step:
            cards.append({"controller": 0, "location": 4, "code": self.starter, "sequence": 2, "position": 1,
                          "public": True})
        return {"cards": cards}

    def keys(self):
        return [f"idle.1.{i}.{c}.0.2.{i}" for i, c in enumerate(self.hand)] if self.step == 0 else ["idle.7.0"]

    def act(self, key):
        if self.step == 0:
            self.hand.remove(int(key.split(".")[3]))
        self.step += 1
        self.turn = 1 if self.step < 2 else 2


def test_book_cards_leave_out_fillers():
    lucina, filler = 13597785, 42141493
    result = {"line": [f"idle.1.0.{lucina}.0.2.0", f"card.0.0.{filler}.0.2.1"], "hand": [lucina], "fillers": [filler],
              "end_board": [{"code": 24092792, "location": 8}, {"code": 99999999, "location": 2}]}
    assert ob.book_cards([result]) == {lucina, 24092792}


def test_book_lookup_exact_and_loose():
    lucina, a, b = 13597785, 11111111, 22222222
    relevant = {lucina, a}
    own = Solo([lucina, a], lucina)
    book = {ob.state_key(own, relevant): {"answer": f"idle.1.0.{lucina}.0.2.0"}}
    loose = {ob.state_key(own, relevant, False): [((1.0, 3), f"idle.1.0.{lucina}.0.2.0")]}
    assert ob.stay(lambda: Solo([lucina, a], lucina), book, relevant) == 1
    other = lambda: Solo([lucina, b], lucina)  # noqa: E731  a hand the book never saw (another relevant card)
    assert ob.stay(other, book, relevant | {b}) == 0
    assert ob.stay(other, book, relevant | {b}, loose) == 1  # the loose key finds Lucina's summon



def test_book_cards_keep_a_card_another_line_uses_for_real():
    tinia, regina = 59581480, 56651978
    as_filler = {"line": [f"card.0.0.{tinia}.0.2.1"], "hand": [regina], "fillers": [tinia], "end_board": []}
    for_real = {"line": [f"idle.1.0.{tinia}.0.2.0"], "hand": [tinia], "fillers": [], "end_board": []}
    assert tinia not in ob.book_cards([as_filler])
    assert tinia in ob.book_cards([as_filler, for_real])


class Board:
    """A snapshot-only duel for state keys."""

    def __init__(self, cards, keys=("idle.7.0",)):
        self.cards, self._keys, self.phase, self.turn, self.player = cards, list(keys), 4, 1, 0

    def snapshot(self):
        return {"cards": self.cards}

    def keys(self):
        return self._keys


def card(controller, location, code, position=0x5, public=False):
    return {"controller": controller, "owner": controller, "location": location, "code": code, "sequence": 0,
            "position": position, "public": public}


def test_state_key_sees_the_opponents_face_up_and_gy_cards():
    droll, lucina = 94145021, 13597785
    base = [card(0, 4, lucina)]
    plain = ob.state_key(Board(base), {lucina})
    in_gy = ob.state_key(Board(base + [card(1, 0x10, droll)]), {lucina})  # Droll used: in the opponent's GY
    face_up = ob.state_key(Board(base + [card(1, 8, droll, position=0x1)]), {lucina})
    face_down = ob.state_key(Board(base + [card(1, 8, droll, position=0xA)]), {lucina})
    assert plain != in_gy and plain != face_up and face_up != face_down
    hidden = ob.state_key(Board(base + [card(1, 8, 12345678, position=0xA)]), {lucina})
    assert face_down == hidden  # a face-down card has no identity in the key


def test_walk_and_stay_use_the_same_keys():
    lucina, a = 13597785, 11111111
    relevant = {lucina, a}
    walked = ob.walk(lambda: Solo([lucina, a], lucina), [f"idle.1.0.{lucina}.0.2.0"], relevant)
    book = {state: {"answer": answer} for state, answer, _, _ in walked}
    assert ob.stay(lambda: Solo([lucina, a], lucina), book, relevant) == 1


def test_score_puts_unrated_lines_below_rated_ones():
    assert ob.score({"field": 2, "value": {"held": 0.0}}) > ob.score({"field": 9, "value": {"held": None}})


def test_from_lines_checks_its_settings(tmp_path):
    import json

    import pytest
    lines = tmp_path / "book-lines.jsonl"
    lines.write_text("", encoding="utf-8")
    (tmp_path / "book-meta.json").write_text(json.dumps({"deck": "decks/elfnote.ydk", "deck_type": "Elfnote",
                                                         "opponent": "x.ydk"}), encoding="utf-8")
    args = ob.parse_args(["--deck", "decks/branded.ydk", "--deck-type", "Branded", "--checkpoint", "c",
                          "--out", str(tmp_path / "b.jsonl"), "--from-lines", str(lines), "--opponent", "x.ydk"])
    with pytest.raises(SystemExit, match="other settings"):
        ob.load_lines(args)
