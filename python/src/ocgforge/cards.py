"""Static card facts from the pinned BabelCDB, turned into fixed-size per-card feature rows.

Every card the model can meet gets a row: type/attribute/race ids, archetype (setcode) ids, numeric
stats, rules-text term ids and optionally a frozen text-embedding vector (embed_cards.py). The model
stores these rows as buffers, so a checkpoint is self-contained and needs no database at play time.

Card index 0 is padding and 1 is "unknown" (e.g. a face-down opponent card). Alternate artworks
(alias within 20 of the passcode, the YGOPro convention) share the index of the original card.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PAD = 0
UNK = 1

MAX_TYPES = 8
MAX_SETCODES = 8  # up to 4 setcodes, each as full value and as base archetype
MAX_TEXT = 192  # unigram + bigram ids per card
ARTWORK_ALIAS_RANGE = 20

TYPE_MONSTER = 0x1
TYPE_PENDULUM = 0x1000000
TYPE_LINK = 0x4000000
TYPE_TOKEN = 0x4000

# Per-card numeric features, in this order.
CARD_NUMERIC = (
    "level",  # level / rank / link rating
    "lscale",
    "rscale",
    "atk",
    "atkUnknown",  # printed ATK "?"
    "def",
    "defUnknown",
    "hasDef",  # Link monsters have none
    *(f"linkMarker{i}" for i in range(9)),
)

_TOKEN = re.compile(r"~|[a-z]+(?:'[a-z]+)?|\d+")


def tokenize(text: str, name: str = "") -> list[str]:
    """Card text -> lowercase word tokens; the card's own name becomes `~`, `<nl>` between lines."""
    if name:
        text = text.replace(f'"{name}"', "~").replace(name, "~")
    tokens: list[str] = []
    for line in text.lower().split("\n"):
        words = _TOKEN.findall(line)
        if words:
            tokens.extend(words)
            tokens.append("<nl>")
    return tokens


def text_terms(text: str, name: str = "") -> list[str]:
    """Unigrams followed by bigrams; bigrams keep the word order a bag of words would lose."""
    words = tokenize(text, name)
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def _bits(value: int, width: int) -> list[int]:
    return [bit for bit in range(width) if value >> bit & 1]


@dataclass(frozen=True)
class Card:
    code: int
    name: str
    text: str
    type: int
    attribute: int
    race: int
    level: int  # raw cdb value: level | lscale << 24 | rscale << 16
    atk: int
    defense: int
    setcode: int

    @property
    def setcodes(self) -> list[int]:
        return [v for v in ((self.setcode >> (16 * i)) & 0xFFFF for i in range(4)) if v]


@dataclass
class CardTable:
    """All cards of a card database, indexed for the model."""

    cards: list[Card]
    # passcode (including alternate artworks) -> card index
    index_of: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_cdb(cls, path: Path, include_tokens: bool = True) -> "CardTable":
        query = ("select d.id, d.alias, t.name, t.desc, d.type, d.attribute, d.race, d.level, d.atk, d.def, "
                 "d.setcode from datas d join texts t on t.id = d.id order by d.id")
        with sqlite3.connect(Path(path)) as connection:
            rows = connection.execute(query).fetchall()
        table = cls(cards=[])
        artworks: list[tuple[int, int]] = []
        for code, alias, name, desc, card_type, attribute, race, level, atk, defense, setcode in rows:
            if alias and abs(code - alias) < ARTWORK_ALIAS_RANGE:
                artworks.append((code, alias))
                continue
            if card_type & TYPE_TOKEN and not include_tokens:
                continue
            table.index_of[code] = len(table.cards) + 2
            table.cards.append(Card(code, name, desc or "", card_type, attribute, race, level, atk, defense,
                                    setcode))
        for code, alias in artworks:
            if alias in table.index_of:
                table.index_of[code] = table.index_of[alias]
        return table

    def __len__(self) -> int:
        """Number of index rows, including padding and unknown."""
        return len(self.cards) + 2

    def index(self, code: int) -> int:
        return self.index_of.get(code, UNK)

    def card(self, index: int) -> Card | None:
        return self.cards[index - 2] if index >= 2 else None


@dataclass
class CardFeatures:
    """Per-card feature rows (numpy), plus the vocabularies they were built with."""

    codes: list[int]  # passcode of every card index >= 2
    aliases: dict[int, int]  # alternate artwork passcode -> original passcode
    text_vocab: dict[str, int]
    setcode_vocab: dict[int, int]
    types: np.ndarray  # [N, MAX_TYPES] type-bit ids (bit + 2)
    attribute: np.ndarray  # [N] attribute-bit id
    race: np.ndarray  # [N] race-bit id
    setcodes: np.ndarray  # [N, MAX_SETCODES] setcode-vocab ids
    numeric: np.ndarray  # [N, len(CARD_NUMERIC)]
    text: np.ndarray  # [N, MAX_TEXT] text-vocab ids
    seen: np.ndarray  # [N] cards that occur in training; others play from their features alone

    TYPE_IDS = 32 + 2
    ATTRIBUTE_IDS = 8 + 2
    RACE_IDS = 64 + 2

    @property
    def size(self) -> int:
        return len(self.codes) + 2

    def text_size(self) -> int:
        return len(self.text_vocab) + 2

    def setcode_size(self) -> int:
        return len(self.setcode_vocab) + 2

    def index_map(self) -> dict[int, int]:
        """passcode -> card index, including alternate artworks."""
        mapping = {code: i + 2 for i, code in enumerate(self.codes)}
        for alt, original in self.aliases.items():
            if original in mapping:
                mapping[alt] = mapping[original]
        return mapping

    @classmethod
    def build(cls, table: CardTable, trained_codes: set[int] | None = None, vocab_codes: set[int] | None = None,
              min_term_count: int = 2) -> "CardFeatures":
        """Feature rows for every card in `table`.

        `trained_codes`: cards that occur in training; only these get a learned name embedding
        (default: all). `vocab_codes`: cards whose text and setcodes build the vocabularies
        (default: all); words seen fewer than `min_term_count` times read as unknown."""
        n = len(table)
        trained = trained_codes if trained_codes is not None else {c.code for c in table.cards}
        vocab_source = [c for c in table.cards if vocab_codes is None or c.code in vocab_codes]

        term_counts = Counter(term for c in vocab_source for term in set(text_terms(c.text, c.name)))
        text_vocab = {term: i + 2 for i, term in enumerate(sorted(t for t, k in term_counts.items()
                                                                  if k >= min_term_count))}
        setcode_values = sorted({v for c in vocab_source for s in c.setcodes for v in (s, s & 0x0FFF)})
        setcode_vocab = {value: i + 2 for i, value in enumerate(setcode_values)}

        types = np.zeros((n, MAX_TYPES), dtype=np.int64)
        attribute = np.zeros(n, dtype=np.int64)
        race = np.zeros(n, dtype=np.int64)
        setcodes = np.zeros((n, MAX_SETCODES), dtype=np.int64)
        numeric = np.zeros((n, len(CARD_NUMERIC)), dtype=np.float32)
        text = np.zeros((n, MAX_TEXT), dtype=np.int64)
        seen = np.zeros(n, dtype=bool)
        seen[:2] = True
        for i, card in enumerate(table.cards, start=2):
            type_ids = [b + 2 for b in _bits(card.type, 32)][:MAX_TYPES]
            types[i, : len(type_ids)] = type_ids
            attribute[i] = next((b + 2 for b in _bits(card.attribute, 8)), PAD)
            race[i] = next((b + 2 for b in _bits(card.race, 64)), PAD)
            set_ids = [setcode_vocab.get(v, UNK) for s in card.setcodes for v in (s, s & 0x0FFF)][:MAX_SETCODES]
            setcodes[i, : len(set_ids)] = set_ids
            numeric[i] = card_numeric(card)
            term_ids = [text_vocab.get(t, UNK) for t in text_terms(card.text, card.name)][:MAX_TEXT]
            text[i, : len(term_ids)] = term_ids
            seen[i] = card.code in trained
        aliases = {code: table.cards[index - 2].code for code, index in table.index_of.items()
                   if table.cards[index - 2].code != code}
        return cls([c.code for c in table.cards], aliases, text_vocab, setcode_vocab,
                   types, attribute, race, setcodes, numeric, text, seen)

    def to_json(self) -> dict:
        """Everything except the arrays, which the model keeps as buffers."""
        return {"schema": "ocgforge-card-features@1", "codes": self.codes,
                "aliases": {str(k): v for k, v in self.aliases.items()},
                "textVocab": self.text_vocab, "setcodeVocab": {str(k): v for k, v in self.setcode_vocab.items()}}

    @classmethod
    def from_json(cls, payload: dict) -> "CardFeatures":
        """Vocabularies and empty arrays of the right shape; load_checkpoint fills them from the weights."""
        if payload.get("schema") != "ocgforge-card-features@1":
            raise ValueError(f"unexpected card features schema {payload.get('schema')!r}")
        n = len(payload["codes"]) + 2
        return cls(list(payload["codes"]), {int(k): v for k, v in payload["aliases"].items()},
                   dict(payload["textVocab"]), {int(k): v for k, v in payload["setcodeVocab"].items()},
                   np.zeros((n, MAX_TYPES), np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64),
                   np.zeros((n, MAX_SETCODES), np.int64), np.zeros((n, len(CARD_NUMERIC)), np.float32),
                   np.zeros((n, MAX_TEXT), np.int64), np.zeros(n, bool))


def card_numeric(card: Card) -> list[float]:
    is_link = bool(card.type & TYPE_LINK)
    is_monster = bool(card.type & TYPE_MONSTER)
    pendulum = bool(card.type & TYPE_PENDULUM)
    markers = card.defense if is_link else 0
    return [
        (card.level & 0xFF) / 12.0,
        ((card.level >> 24) & 0xFF) / 13.0 if pendulum else 0.0,
        ((card.level >> 16) & 0xFF) / 13.0 if pendulum else 0.0,
        max(card.atk, 0) / 5000.0 if is_monster else 0.0,
        1.0 if is_monster and card.atk == -2 else 0.0,
        max(card.defense, 0) / 5000.0 if is_monster and not is_link else 0.0,
        1.0 if is_monster and not is_link and card.defense == -2 else 0.0,
        1.0 if is_monster and not is_link else 0.0,
        *(float(markers >> bit & 1) for bit in range(9)),
    ]
