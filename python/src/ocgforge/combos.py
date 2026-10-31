"""Community combos from masterduelmeta.com (tools/fetch_combos.py), as passcodes for our engine.

A combo: starting cards, then steps of entries, each a card with an action text ("Normal Summon",
"Synchro 10 into Baronne", often empty for "use its effect") or a generic note. Combo spreadsheets
add key cards of the end board and lines against specific hand traps. Card names are matched to the
pinned card database (official passcodes first); a combo with a card we do not have is marked
unusable. See docs/combo-teacher-design.md.
"""

from __future__ import annotations

import glob
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .cards import CardTable

CACHE = Path(__file__).resolve().parents[3] / ".cache" / "masterduelmeta"


def normalize(name: str) -> str:
    name = name.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    name = name.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", name).strip().casefold()


def loose(name: str) -> str:
    return "~" + re.sub(r"[^0-9a-z]", "", normalize(name))


def name_index(table: CardTable | None = None) -> dict[str, int]:
    """Normalized (and loose) card name -> passcode; alternate artworks and pre-release duplicates
    lose against the official passcode."""
    table = table or CardTable.from_cdb()
    codes: dict[str, int] = {}
    for card in table.cards:
        official = card.code < 100_000_000
        for key in (normalize(card.name), loose(card.name)):
            current = codes.get(key)
            if current is None or (official and current >= 100_000_000):
                codes[key] = card.code
    return codes


@dataclass
class Entry:
    code: int | None  # the card, None for a generic note or an unknown card
    name: str
    action: str  # action text, "" when the card is just used


@dataclass
class Combo:
    id: str
    title: str
    deck_type: str
    votes: int  # upvotes - downvotes
    starting: list[Entry]
    steps: list[list[Entry]]
    notes: str = ""
    end_board: list[int] = field(default_factory=list)  # key cards, from a spreadsheet
    vs_hand_traps: list[int] = field(default_factory=list)  # the hand traps this line plays through
    missing: list[str] = field(default_factory=list)  # card names we could not match
    author: str = ""  # who wrote it (masterduelmeta user name), for crediting sources

    @property
    def usable(self) -> bool:
        return not self.missing

    def cards(self) -> list[int]:
        """Every card the combo names, in order of first appearance."""
        seen: dict[int, None] = {}
        for entry in [*self.starting, *(e for step in self.steps for e in step)]:
            if entry.code is not None:
                seen.setdefault(entry.code)
        return list(seen)


def latest(endpoint: str) -> list[dict]:
    files = sorted(glob.glob(str(CACHE / f"{endpoint}-*.json")))
    if not files:
        raise FileNotFoundError(f"no {endpoint} in {CACHE}: run tools/fetch_combos.py")
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))


def _entry(raw: dict, names: dict[str, int], missing: list[str]) -> Entry:
    card = raw.get("card")
    if not card:
        return Entry(None, raw.get("generic") or "", raw.get("generic") or "")
    name = card.get("name", "")
    code = names.get(normalize(name), names.get(loose(name)))
    if code is None:
        missing.append(name)
    return Entry(code, name, raw.get("label") or "")


def author_name(author) -> str:
    """The combo's author: a user object ({"username": ...}) or already a name."""
    if isinstance(author, dict):
        return author.get("username") or ""
    return author if isinstance(author, str) else ""


def load(deck_type: str | None = None, names: dict[str, int] | None = None) -> list[Combo]:
    """Combos (optionally of one deck type), with end boards and hand-trap lines from the spreadsheets."""
    names = names or name_index()
    extra: dict[str, dict] = {}  # combo id -> spreadsheet fields
    for sheet in latest("spreadsheets"):
        for tab in sheet.get("tabs", []):
            for section in tab.get("sections", []):
                for item in section.get("combos", []):
                    if isinstance(item, dict) and item.get("combo"):
                        extra.setdefault(item["combo"], item)
    combos = []
    for raw in latest("combos"):
        kind = (raw.get("deckType") or {}).get("name", "")
        if deck_type is not None and kind != deck_type:
            continue
        missing: list[str] = []
        combo = Combo(
            id=raw["_id"], title=raw.get("title", ""), deck_type=kind,
            votes=len(raw.get("upvotes", [])) - len(raw.get("downvotes", [])),
            starting=[_entry(e, names, missing) for e in raw.get("startingCards", [])],
            steps=[[_entry(e, names, missing) for e in step.get("entries", [])] for step in raw.get("steps", [])],
            notes=raw.get("notes") or "", missing=missing, author=author_name(raw.get("author")))
        sheet = extra.get(combo.id, {})
        for key, target in (("endBoard", combo.end_board), ("vsHandTraps", combo.vs_hand_traps)):
            for e in sheet.get(key) or []:
                card = e.get("card", e) if isinstance(e, dict) else None
                code = names.get(normalize(card.get("name", ""))) if isinstance(card, dict) else None
                if code is not None:
                    target.append(code)
        combos.append(combo)
    return combos


def deck_types_for(decklist: set[int], names: dict[str, int] | None = None, top: int = 3,
                   ignore: set[int] = frozenset()) -> list[tuple[float, str]]:
    """Deck types whose combos use most of `decklist`'s cards (weighted: a card many deck types use
    counts less), best first. Matching by name would miss renamed decks."""
    names = names or name_index()
    by_type: dict[str, set[int]] = {}
    for combo in load(names=names):
        by_type.setdefault(combo.deck_type, set()).update(combo.cards())
    spread = Counter(code for cards in by_type.values() for code in cards)
    weight = {code: 1.0 / spread[code] for code in spread}
    mine = decklist - ignore
    total = sum(weight.get(code, 1.0) for code in mine) or 1.0
    scores = [(sum(weight[c] for c in mine & cards) / total, kind) for kind, cards in by_type.items()]
    return sorted(scores, reverse=True)[:top]
