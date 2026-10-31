"""Duelingbook logs as text: card movements and declared effects, nothing else (no engine, no card ids).

Two exports are read:
    SS ATK: Special Summoned "Elfnote Tinia" from hand (2/2) to M-3 (ATK)
    A SS "Elfnote Lucina" fr Deck to M-3        (abbreviated, one actor letter; also tab separated)
    A Eff "Elfnote Lucina" in M-3               (a declared effect)

Duelingbook is a manual simulator: the log records what the player moved, not what the engine
did. So `parse` drops what is not part of the combo:
- setup: the leading block of "Returned X to deck" / "Added X to hand" lines builds the test hand
  (the added cards are the starting hand);
- a move taken back right away (to the GY, then returned to the deck: a misplay);
- in a "vs hand trap" log the player plays the opponent's hand trap himself: its cards move to the
  opponent's zones ("to M2-3"), and its declared effect answers the effect declared just before
  (a choke point). Such cards become `traps`, not events.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LINE = re.compile(r'^(?:[A-Z]\s+|[^:"]+:\s+)?(?P<verb>[A-Za-z]+(?:\s[A-Za-z]+)?)\s+(?:banished\s+)?"(?P<card>.+?)"'
                  r'(?P<rest>(?:\s+(?:fr|from|in|to)\b.*|\s*\(.*|\s*))$')
SUMMON_VERBS = {"ss", "ns", "special summoned", "normal summoned", "summoned", "tribute summoned"}
USE_VERBS = {"eff", "declared", "activated effect"}
KINDS = ("summon", "place", "gy", "hand", "banish", "use")


@dataclass
class Event:
    card: str
    kind: str  # one of KINDS: where the card went, or "use" (its effect was declared)


@dataclass
class Log:
    title: str = ""
    hand: list[str] = field(default_factory=list)  # starting hand (setup adds, cards played from hand)
    events: list[Event] = field(default_factory=list)
    traps: dict[str, list[str]] = field(default_factory=dict)  # opponent's hand trap -> cards it answered


def kind_of(verb: str, rest: str) -> str | None:
    if verb in USE_VERBS:
        return "use"
    if verb in SUMMON_VERBS or re.search(r"to (m-|left emz|right emz|.*emz)", rest):
        return "summon"
    if verb in ("placed", "act", "activated", "set") or "to s-" in rest or "to fz" in rest:
        return "place"
    if verb == "sent" and "gy" in rest:
        return "gy"
    if verb in ("added", "returned") and "to hand" in rest:
        return "hand"
    if verb == "banished":
        return "banish"
    return None


def parse(text: str, hand_traps: set[str] = frozenset()) -> Log:
    """The log's combo: starting hand, events in order, and the opponent's hand traps. Cards named
    in `hand_traps` are the opponent's in a log whose title says "vs"."""
    log = Log()
    opponent: set[str] = set()
    started = False
    last_use = None
    added: set[str] = set()
    for raw in text.splitlines():
        if raw.startswith("#"):
            log.title = log.title or raw.lstrip("# ").strip()
            continue
        line = re.sub(r"\s+#.*$", "", raw).strip()
        line = re.sub(r"^[A-Z]\t[^\t]*\t", "", line)  # tab export: actor, category, text
        m = LINE.match(line)
        if not m:
            continue
        verb, card, rest = m["verb"].lower(), m["card"], m["rest"].lower()
        if "to m2-" in rest or "to s2-" in rest:
            opponent.add(card)  # the opponent's hand trap, put on its side by the logging player
            continue
        if card in opponent or ("vs" in log.title.lower() and card in hand_traps
                                and verb in USE_VERBS | {"sent", "ss"}):
            if verb in USE_VERBS or verb == "sent":
                answered = log.traps.setdefault(card, [])
                if last_use is not None and last_use not in answered:
                    answered.append(last_use)
            continue
        setup = (verb == "returned" and "deck" in rest) or (verb == "added" and "to hand" in rest)
        if not started and setup:
            if verb == "added":
                log.hand.append(card)
            continue
        started = True
        if verb == "returned" and "deck" in rest:
            if log.events and log.events[-1].card == card:
                log.events.pop()  # the move this undoes
            continue
        if ("fr hand" in rest or "from hand" in rest or (verb in USE_VERBS and "in hand" in rest)) \
                and card not in added and card not in log.hand:
            log.hand.append(card)
        kind = kind_of(verb, rest)
        if kind is None:
            continue
        if kind == "use":
            last_use = card
        if kind == "hand":
            added.add(card)
        log.events.append(Event(card, kind))
    log.hand = [c for c in log.hand if c not in log.traps and c not in opponent]
    return log
