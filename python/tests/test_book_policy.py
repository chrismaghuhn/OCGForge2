"""book_policy: the book key as one seat sees it, and the book player (synthetic duels only)."""

import json

import pytest

from ocgforge import book_policy as bp
from ocgforge import opening_book as ob

LUCINA, REGINA, FILLER, DROLL, ASH = 13597785, 56651978, 42141493, 94145021, 14558127
HAND, DECK, MZONE, SZONE, GRAVE = 0x2, 0x1, 0x4, 0x8, 0x10


def card(controller, location, code, sequence=0, position=0x5, public=False):
    return {"controller": controller, "owner": controller, "location": location, "code": code, "sequence": sequence,
            "position": position, "public": public}


class Board:
    """A snapshot-only duel."""

    def __init__(self, cards, keys=("idle.7.0", "idle.6.0"), player=0, turn=1, turn_player=0, phase=4):
        self.cards, self._keys = cards, list(keys)
        self.player, self.turn, self.turn_player, self.phase = player, turn, turn_player, phase

    def snapshot(self):
        return {"cards": [dict(c) for c in self.cards]}

    def keys(self):
        return self._keys


def mirrored(board: Board) -> Board:
    """The same duel with the seats swapped."""
    cards = [{**c, "controller": 1 - c["controller"], "owner": 1 - c["owner"]} for c in board.cards]
    keys = []
    for k in board.keys():
        parts = k.split(".")
        if len(parts) >= 5 and parts[-3] in ("0", "1") and int(parts[-4]) >= 1000:
            parts[-3] = str(1 - int(parts[-3]))
        keys.append(".".join(parts))
    return Board(cards, keys, 1 - board.player, board.turn, 1 - board.turn_player, board.phase)


RELEVANT = {LUCINA, REGINA, DROLL}
OURS = [card(0, HAND, LUCINA, 0), card(0, HAND, FILLER, 1), card(0, MZONE, REGINA, 2, 0x1), card(0, GRAVE, LUCINA),
        card(0, DECK, 11111111)]
THEIRS = [card(1, HAND, ASH, 0), card(1, SZONE, 33333333, 1, 0xA), card(1, GRAVE, DROLL), card(1, DECK, 22222222)]
QUESTION = (f"idle.1.0.{LUCINA}.0.2.0", f"idle.5.0.{REGINA}.0.4.2", "idle.7.0")


def test_seat_zero_key_is_the_books_key():
    for cards in (OURS, OURS + THEIRS, THEIRS):
        board = Board(cards, QUESTION)
        for hand in (True, False):
            assert bp.seat_key(board, RELEVANT, 0, hand) == ob.state_key(board, RELEVANT, hand)


def test_seat_one_sees_the_mirrored_duel_as_seat_zero_does():
    board = Board(OURS + THEIRS, QUESTION)
    for hand in (True, False):
        assert bp.seat_key(mirrored(board), RELEVANT, 1, hand) == bp.seat_key(board, RELEVANT, 0, hand)


def test_key_ignores_what_the_seat_cannot_see():
    base = bp.seat_key(Board(OURS + THEIRS, QUESTION), RELEVANT)
    hidden_changed = [card(1, HAND, DROLL, 0), card(1, HAND, LUCINA, 1),  # other opponent hand
                      card(1, SZONE, REGINA, 1, 0xA),  # another face-down card in the same zone
                      card(1, GRAVE, DROLL), card(1, DECK, LUCINA)]
    our_deck_changed = [c if c["location"] != DECK else card(0, DECK, REGINA) for c in OURS]
    assert bp.seat_key(Board(OURS + hidden_changed, QUESTION), RELEVANT) == base
    assert bp.seat_key(Board(our_deck_changed + THEIRS, QUESTION), RELEVANT) == base
    # what it can see does change the key: our hand, the opponent's face-up card, its GY
    other_hand = [card(0, HAND, REGINA, 0) if c["code"] == LUCINA and c["location"] == HAND else c for c in OURS]
    assert bp.seat_key(Board(other_hand + THEIRS, QUESTION), RELEVANT) != base
    flipped = [card(1, SZONE, DROLL, 1, 0x1) if c["location"] == SZONE else c for c in THEIRS]
    assert bp.seat_key(Board(OURS + flipped, QUESTION), RELEVANT) != base
    assert bp.seat_key(Board(OURS + THEIRS[:2], QUESTION), RELEVANT) != base


def test_names_hidden_spots_questions_about_unseen_cards():
    board = Board(OURS + THEIRS, (f"card.0.0.33333333.1.8.1", "chain.pass"))  # the opponent's face-down card
    assert bp.names_hidden(board, 0)
    assert not bp.names_hidden(Board(OURS + THEIRS, (f"card.0.0.{DROLL}.1.16.0", "chain.pass")), 0)  # its GY
    assert not bp.names_hidden(Board(OURS + THEIRS, QUESTION), 0)  # our own cards
    assert bp.names_hidden(Board(OURS + THEIRS, (f"card.0.0.{LUCINA}.0.2.0", "chain.pass")), 1)  # our hand, for seat 1


def book_of(board: Board, answer: str) -> bp.Book:
    return bp.Book({bp.seat_key(board, RELEVANT): answer}, {bp.seat_key(board, RELEVANT, hand=False): [answer]},
                   RELEVANT, ((), ()))


def test_exact_and_loose_modes():
    seen = Board(OURS, QUESTION)
    book = book_of(seen, QUESTION[1])
    other_hand = Board([card(0, HAND, REGINA, 0)] + OURS[1:], QUESTION)  # a hand the book never saw
    exact, loose = bp.BookPlayer([book], "exact"), bp.BookPlayer([book], "loose")
    assert exact.answer(seen, 0, book, bp.Track()) == QUESTION[1]
    assert exact.answer(other_hand, 0, book, bp.Track()) is None
    assert loose.answer(other_hand, 0, book, bp.Track()) == QUESTION[1]
    assert exact.stats["exact"] == 1 and exact.stats["miss"] == 1 and loose.stats["loose"] == 1
    assert exact.summary()["hit_rate"] == 0.5


def test_loose_stops_once_the_duel_left_the_book():
    seen = Board(OURS, QUESTION)
    book = book_of(seen, QUESTION[1])
    player, track = bp.BookPlayer([book], "loose"), bp.Track()
    unknown = Board(OURS[1:], ("idle.7.0", "idle.6.0"))
    assert player.answer(unknown, 0, book, track) is None and track.left
    assert player.answer(Board([card(0, HAND, REGINA, 0)] + OURS[1:], QUESTION), 0, book, track) is None  # no loose
    assert player.answer(seen, 0, book, track) == QUESTION[1]  # an exact transposition still counts


def test_book_only_in_the_seats_first_turn_going_first_and_real_choices():
    seen = Board(OURS, QUESTION)
    book = book_of(seen, QUESTION[1])
    player = bp.BookPlayer([book], "loose")
    for other in (dict(turn=2), dict(turn_player=1), dict(player=1)):
        assert player.answer(Board(OURS, QUESTION, **other), 0, book, bp.Track()) is None
    assert player.answer(Board(OURS, ("idle.7.0",)), 0, book, bp.Track()) is None  # forced
    assert player.stats["choices"] == 0
    assert player.answer(mirrored(seen), 1, book, bp.Track()) == mirrored(seen).keys()[1]  # the book seat is 1


def test_for_seat_swaps_controller_and_zone_player():
    assert bp.for_seat(f"chain.0.0.{LUCINA}.0.2.0", 1) == f"chain.0.0.{LUCINA}.1.2.0"
    assert bp.for_seat(f"card.0.1.{DROLL}.1.16.0", 1) == f"card.0.1.{DROLL}.0.16.0"
    assert bp.for_seat("place.0.4.2", 1) == "place.1.4.2"
    assert bp.for_seat("chain.pass", 1) == "chain.pass" and bp.for_seat("place.0.4.2", 0) == "place.0.4.2"


def test_real_duel_key_does_not_see_the_opponents_deck():
    """Same seed and book deck, other opponent decks: the opponent's hand differs, the key must not."""
    from ocgforge import rollout, worlds
    cache = rollout.REPO / ".cache"
    if not (cache / "cardscripts").exists():
        pytest.skip("card scripts not fetched")
    from ocgforge._core import Duel
    decks = rollout.REPO / "decks"
    relevant = set(worlds.read_decklist(decks / "elfnote.ydk")[0])
    for seat in (0, 1):
        keys = set()
        for other in ("branded", "artmage", "doomz"):
            pair = ("elfnote", other) if seat == 0 else (other, "elfnote")
            duel = Duel(str(cache / "cardscripts"), str(cache / "card_data.tsv"), *[str(decks / f"{d}.ydk") for d in pair],
                        7, seat)
            while duel.player != seat:
                duel.act(worlds.decline(duel.keys()))
            keys.add(bp.seat_key(duel, relevant, seat))
        assert len(keys) == 1


def test_unknown_mode():
    with pytest.raises(ValueError):
        bp.BookPlayer([], "fuzzy")


class Solo:
    """A two-decision duel: summon a hand card, then end the turn."""

    def __init__(self, hand):
        self.hand, self.step, self.finished = list(hand), 0, False
        self.turn, self.phase, self.player, self.turn_player = 1, 4, 0, 0

    def snapshot(self):
        return {"cards": [card(0, HAND, c, i) for i, c in enumerate(self.hand)]}

    def keys(self):
        return [f"idle.1.{i}.{c}.0.2.{i}" for i, c in enumerate(self.hand)] if self.step == 0 else ["idle.7.0"]

    def act(self, key):
        self.hand.remove(int(key.split(".")[3]))
        self.step += 1


def test_load_rebuilds_book_and_loose_answers(tmp_path, monkeypatch):
    deck = tmp_path / "solo.ydk"
    deck.write_text(f"#main\n{LUCINA}\n{REGINA}\n#extra\n!side\n", encoding="utf-8")
    monkeypatch.setattr(bp.cc, "puzzle_start", lambda d, o, script, seed: (lambda: Solo([LUCINA, REGINA])))
    lines = [{"seed": "s", "trap": None, "field": 1, "value": {"held": 0.5}, "line": [f"idle.1.0.{LUCINA}.0.2.0"],
              "script": "", "duel_seed": 0},
             {"seed": "t", "trap": None, "field": 1, "value": {"held": 0.9}, "line": [f"idle.1.1.{REGINA}.0.2.1"],
              "script": "", "duel_seed": 0}]
    (tmp_path / "b-lines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    (tmp_path / "b-meta.json").write_text(json.dumps({"deck": str(deck), "opponent": str(deck),
                                                      "relevant": [LUCINA, REGINA]}), encoding="utf-8")
    (tmp_path / "b.jsonl").write_text("{}\n", encoding="utf-8")
    book = bp.load(tmp_path / "b.jsonl")
    assert list(book.exact.values()) == [f"idle.1.{REGINA}.0.2.1"]  # the better line's answer (canonical)
    assert list(book.loose.values()) == [[f"idle.1.{REGINA}.0.2.1", f"idle.1.{LUCINA}.0.2.0"]]
    assert book.deck == bp.deck_id(deck) and bp.BookPlayer([book]).book_for(deck) is book
    (tmp_path / "b.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        bp.load(tmp_path / "b.jsonl")
