"""Opening book in play: the seat playing a book deck answers from the book in its first turn.

A book (`opening_book`) is keyed by game state as seat 0 sees it, built with us going first. Here
`seat_key` computes that key for any seat (for seat 0 it is `opening_book.state_key`), and
`BookPlayer` answers a decision from the book when it can, else returns None so the caller asks
the net:
- only on turn 1 with the book seat as turn player (the book never saw a hand going second);
- `exact`: the exact key only; `loose`: the exact key, then the loose key (no hand) as long as the
  duel has not left the book (a real choice the book did not know);
- never when the question names a card the seat cannot see (a face-down or hand card of the
  opponent): the key's question part would carry its identity.
The key only reads what the seat legitimately knows: its own hand, field, GY and banished cards,
the opponent's visible cards (face up, public, in the GY), phase, turn and its own question.

Hits are counted per real choice (more than one answer) in `stats`.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import combo_check as cc
from . import opening_book as ob
from . import rollout, worlds

MODES = ("exact", "loose")
_HAND, _GRAVE, _REMOVED = worlds.LOCATION_HAND, worlds.LOCATION_GRAVE, worlds.LOCATION_REMOVED
_FIELD = (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)


def seat_key(duel, relevant: set[int], seat: int = 0, hand: bool = True, snapshot: dict | None = None) -> str:
    """`opening_book.state_key` as `seat` sees the duel (seat 0: the same hash)."""
    parts = []
    for c in (snapshot or duel.snapshot())["cards"]:
        loc, mine = c["location"], c["controller"] == seat
        if mine and loc == _HAND and not hand:
            continue
        if mine and loc in (_HAND, _GRAVE, _REMOVED):
            parts.append((0, loc, c["code"] if c["code"] in relevant else 0, 0))
        elif mine and loc in _FIELD:
            parts.append((0, loc, c["code"], c["sequence"] * 16 + c["position"]))
        elif not mine and loc in _FIELD + (_GRAVE, _REMOVED):
            parts.append((1, loc, c["code"] if worlds.visible(_as_seen_by(c, seat), 0) else 0, 0))
    question = sorted({(k.split(".")[0], worlds.card_ref(k)[1] if hand and worlds.card_ref(k) and
                        int(worlds.card_ref(k)[1]) in relevant else "") for k in duel.keys()})
    raw = json.dumps([sorted(parts), duel.phase, duel.turn, duel.player ^ seat, question])
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def _as_seen_by(card: dict, seat: int) -> dict:
    """The card with controller and owner relabelled so that `seat` is player 0."""
    return {**card, "controller": card["controller"] ^ seat, "owner": card.get("owner", card["controller"]) ^ seat}


def for_seat(answer: str, seat: int) -> str:
    """A book answer (seat 0's view) as `seat` would see it: the controller of the card it names and
    the player of a zone choice swapped."""
    if seat == 0:
        return answer
    parts = answer.split(".")
    if worlds.card_ref(answer) is not None:
        parts[-3] = str(int(parts[-3]) ^ seat)
    elif parts[0] == "place" and len(parts) > 1 and parts[1].isdigit():
        parts[1] = str(int(parts[1]) ^ seat)
    return ".".join(parts)


def _hidden(card: dict, seat: int) -> bool:
    if card["location"] in (worlds.LOCATION_HAND, worlds.LOCATION_DECK, worlds.LOCATION_EXTRA):
        return not card["public"]
    return not worlds.visible(_as_seen_by(card, seat), 0)


def names_hidden(duel, seat: int, snapshot: dict | None = None) -> bool:
    """Whether one of the question's answers names a card of the other seat that `seat` cannot see
    (its passcode would leak)."""
    cards = None
    for key in duel.keys():
        ref = worlds.card_ref(key)
        if ref is None or int(ref[2]) == seat:
            continue
        if cards is None:
            cards = {(c["controller"], c["location"], c["sequence"]): c for c in (snapshot or duel.snapshot())["cards"]}
        c = cards.get((int(ref[2]), int(ref[3]), int(key.split(".")[-1])))
        if c is None or _hidden(c, seat):
            return True
    return False


@dataclass
class Book:
    """One deck's book: exact state -> answer, loose state -> answers (best first), the cards the key
    tells apart, and the deck it was built for (sorted main and extra passcodes)."""
    exact: dict[str, str]
    loose: dict[str, list[str]]
    relevant: set[int]
    deck: tuple
    name: str = ""


def deck_id(path: Path) -> tuple:
    main, extra = worlds.read_decklist(path)
    return tuple(sorted(main)), tuple(sorted(extra))


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = rollout.REPO / p
    if not p.exists():  # built in another checkout: the same deck file of this one
        p = rollout.REPO / "decks" / p.name
    return p


def load(path: Path) -> Book:
    """The book `path` (opening_book output) with its loose fallback, rebuilt from the lines beside it
    exactly as `opening_book.build_book` does (the loose answers are not stored in the book file)."""
    stem = path.with_name(path.stem)
    meta = json.loads(Path(f"{stem}-meta.json").read_text(encoding="utf-8"))
    deck, opponent = resolve(meta["deck"]), resolve(meta["opponent"])
    relevant = set(meta["relevant"])
    lines = []
    for text in Path(f"{stem}-lines.jsonl").read_text(encoding="utf-8").splitlines():
        result = json.loads(text)
        lines.append((result, cc.puzzle_start(deck, opponent, result["script"], result["duel_seed"])))
    exact, loose = ob.build_book(lines, relevant)
    stored = sum(1 for _ in path.open(encoding="utf-8"))
    if stored != len(exact):
        raise ValueError(f"{path.name}: {stored} states stored, {len(exact)} rebuilt from the lines")
    return Book({s: e["answer"] for s, e in exact.items()}, {s: [a for _, a in answers] for s, answers in loose.items()},
                relevant, deck_id(deck), path.stem)


@dataclass
class Track:
    """One seat in one duel: whether it left the book, and its real choices played from the book."""
    left: bool = False
    depth: int = 0
    used: bool = False


@dataclass
class BookPlayer:
    books: list[Book]
    mode: str = "loose"
    stats: Counter = field(default_factory=Counter)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode {self.mode!r} not in {MODES}")
        self._by_deck = {b.deck: b for b in self.books}

    def book_for(self, deck: Path | tuple) -> Book | None:
        return self._by_deck.get(deck if isinstance(deck, tuple) else deck_id(deck))

    def answer(self, duel, seat: int, book: Book, track: Track) -> str | None:
        """The book's answer for `seat` at this decision, or None (ask the net)."""
        if duel.player != seat or duel.turn != 1 or duel.turn_player != seat:
            return None
        keys = duel.keys()
        if len(keys) < 2:
            return None
        if not track.used:
            track.used = True
            self.stats["duels"] += 1
        self.stats["choices"] += 1
        snapshot = duel.snapshot()
        if names_hidden(duel, seat, snapshot):
            self.stats["hidden"] += 1
            return self._miss(track)
        found = book.exact.get(seat_key(duel, book.relevant, seat, snapshot=snapshot))
        answer = worlds.match(for_seat(found, seat), keys) if found else None
        if answer is not None:
            self.stats["exact"] += 1
        elif self.mode == "loose" and not track.left:
            wide = book.loose.get(seat_key(duel, book.relevant, seat, hand=False, snapshot=snapshot), [])
            answer = next((m for a in wide if (m := worlds.match(for_seat(a, seat), keys)) is not None), None)
            if answer is not None:
                self.stats["loose"] += 1
        if answer is None:
            return self._miss(track)
        if not track.left:
            track.depth += 1
        return answer

    def _miss(self, track: Track) -> None:
        self.stats["miss"] += 1
        if not track.left:
            track.left = True
            self.stats[f"left_at_{min(track.depth, 20)}"] += 1
        return None

    def summary(self) -> dict:
        s = self.stats
        n = max(1, s["choices"])
        return {"mode": self.mode, "duels": s["duels"], "choices": s["choices"],
                "exact_rate": round(s["exact"] / n, 3), "loose_rate": round(s["loose"] / n, 3),
                "hit_rate": round((s["exact"] + s["loose"]) / n, 3), "hidden": s["hidden"],
                "left_at": dict(sorted((int(k[8:]), v) for k, v in s.items() if k.startswith("left_at_"))),
                "never_left": s["duels"] - sum(v for k, v in s.items() if k.startswith("left_at_"))}
