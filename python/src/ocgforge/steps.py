"""Community combo steps for the search: which entries count, and where each step's card has to go.

A step is (card, expected location): `expected_location` reads the action text ("Special Summon",
"Tribute Tinia" on Welcome Home = only used, "Synchro 10 into Baronne" on a material = the GY).
Pure text; `combo_check` follows these targets.
"""

from __future__ import annotations

import re

from . import combos as cb
from . import worlds

# What a step's action text says happened to its card, as the location the card has to reach. None
# means "the card was used" (its effect activated, it was selected, ...): any answer naming it.
# The first verb of the text's head (before "->", "(", ",", " - ") decides; a verb whose object is
# another card ("Tribute Tinia" on Welcome Home, "Revive Power Patron" on Synchro Rumble, "Search
# Lucina") means this card was only used.
SUMMONED = ("summon", "summoned", "special", "ss", "ns", "revive", "synchro", "syncho", "xyz", "link", "fusion",
            "ritual")
VERB_LOCATION = {
    **{w: worlds.LOCATION_MZONE for w in SUMMONED},
    **{w: worlds.LOCATION_HAND for w in ("add", "search", "return")},
    **{w: worlds.LOCATION_GRAVE for w in ("send", "tribute", "discard", "discarded", "material", "detach", "mill",
                                          "dump", "gy", "grave")},
    "banish": worlds.LOCATION_REMOVED,
    **{w: worlds.LOCATION_SZONE for w in ("place", "set", "scale")},
}
# Words between a verb and its object that do not name a card.
FILLER = {"to", "from", "the", "a", "an", "back", "it", "itself", "this", "card", "for", "cost", "effect", "summon",
          "hand", "deck", "gy", "grave", "ed", "extra", "mmz", "emz", "center", "centre", "middle", "main", "monster",
          "zone", "until", "ep", "end", "phase", "in", "on", "own", "its", "again", "face", "up", "down", "using",
          "with", "of", "by", "p", "pendulum", "field", "and", "activate", "use", "then", "if", "as", "level"}
HEAD = re.compile(r"->|=>|\(|,|;|:| - | then ")


def _tokens(name: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) >= 3}


def expected_location(action: str, name: str = "", others: set[str] = frozenset()) -> int | None:
    """Where the step's card (`name`) has to arrive, from its action text; `others` are name words of
    the combo's other cards (a lower-case object like "banish fallen and cart" is another card too)."""
    head = HEAD.split(action, maxsplit=1)[0]
    words = re.findall(r"[A-Za-z0-9]+", head)
    lower = [w.lower() for w in words]
    mine = _tokens(name)
    for i, w in enumerate(lower):
        if w not in VERB_LOCATION or (w == "link" and i and lower[i - 1] == "chain"):
            continue
        location = VERB_LOCATION[w]
        rest = lower[i + 1:]
        if w == "return":  # "return to field" is a summon; back to the Extra Deck is nowhere we track
            if "field" in rest:
                return worlds.LOCATION_MZONE
            if "extra" in rest:
                return None
        if "into" in rest:  # "Synchro 10 into Baronne": this card is the material unless it is Baronne
            target = {t for t in rest[rest.index("into") + 1:] if len(t) >= 3}
            return worlds.LOCATION_MZONE if target & mine or not target else worlds.LOCATION_GRAVE
        for j, t in enumerate(rest):
            if t in FILLER or t.isdigit() or t in VERB_LOCATION:
                continue
            if t in mine:
                break  # names this card
            if words[i + 1 + j][0].isupper() or t in others:
                return None  # another card is the object: this one was only used
            break
        return location
    if re.search(r"\b(comes? back|return) to (the )?field\b", head.lower()):
        return worlds.LOCATION_MZONE
    return None


OPPONENT = re.compile(r"\bopp(onent)?('s)?\b", re.IGNORECASE)


def ours(combo: cb.Combo, entry: cb.Entry) -> bool:
    """An entry we play: not the opponent's hand trap of a "vs Fuwalos" line ("Opp chains Fuwalos";
    the puzzle opponent never acts), and naming a card."""
    return entry.code is not None and not OPPONENT.search(entry.action) and entry.code not in combo.vs_hand_traps


def step_entries(combo: cb.Combo) -> list[tuple[int, cb.Entry]]:
    """The entries the search follows, with the index of the community step each belongs to."""
    return [(n, e) for n, step in enumerate(combo.steps) for e in step if ours(combo, e)]


def step_groups(combo: cb.Combo) -> list[int]:
    """For each step entry (as in `step_targets`), the index of the community step it belongs to."""
    return [n for n, _ in step_entries(combo)]


EFFECT_FIRST = re.compile(r"^\s*(activate\s+)?(its\s+|own\s+|hand\s+|gy\s+|pendulum\s+)?(effect|eff)\b", re.IGNORECASE)


def step_loose(combo: cb.Combo) -> frozenset:
    """Steps whose text puts "effect" before the verb ("Effect to Revive" on Vidolium, "Activate Eff
    to SS"): whether the card itself moves or its effect moves another card is open, so naming the
    card by an answer also does them."""
    return frozenset(i for i, (_, e) in enumerate(step_entries(combo)) if EFFECT_FIRST.search(e.action))


def step_targets(combo: cb.Combo) -> list[tuple[int, int | None]]:
    """The combo's steps as (card, expected location)."""
    entries = [e for _, e in step_entries(combo)]
    words = {e.code: _tokens(e.name) for e in [*combo.starting, *entries] if e.code is not None}
    targets = []
    for e in entries:
        others = set().union(*(w for code, w in words.items() if code != e.code)) - words.get(e.code, set())
        targets.append((e.code, expected_location(e.action, e.name, others)))
    return targets
