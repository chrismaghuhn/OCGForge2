"""Worlds for honest search: the duel as the searching player sees it, with what it cannot see sampled.

At the start of a player's main phase, `sample_world` turns the true duel (Duel.snapshot) into a puzzle
script (Debug.ReloadFieldBegin / AddCard / ReloadFieldEnd) for a fresh `_core.Duel.world`:
- known exactly: the searching player's own cards everywhere but the deck, the deck's contents
  (not its order), life points, and every card of the opponent that is face up or public;
- sampled: the opponent's hand, face-down cards and Extra Deck from what is left of its decklist,
  and both decks' order.
The world starts with the searching player's turn (rule 5, attacks allowed) without drawing, so it
matches the real duel at that point except for what the player could not know and what the puzzle
API cannot express (the duel's history: lingering effects, "this turn" flags, revealed cards, the
turn number). Decklists are known to both players, as in a tournament with open deck lists.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE = 0x01, 0x02, 0x04, 0x08
LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA = 0x10, 0x20, 0x40
POS_FACEUP = 0x5
DUEL_ATTACK_FIRST_TURN = 0x2


def canonical(key: str) -> str:
    """A semantic key without its list index: idle/battle keys number the commands in the engine's
    list order, which differs between the real duel and a rebuilt world (it follows card creation)."""
    parts = key.split(".")
    if parts[0] in ("idle", "battle") and len(parts) > 3:
        del parts[2]
    return ".".join(parts)


def decline(keys: list[str]) -> str:
    """The answer that changes least: pass on chains, say no. The real duel has already gone past the
    start of this turn, so a world replays its draw and standby windows without acting in them."""
    for wanted in ("chain.pass", "yes_no.no", "effect_yes_no.no"):
        if wanted in keys:
            return wanted
    return keys[0]


def start_world(spec, script: str, me: int, seed: int, max_steps: int = 50):
    """A world duel advanced to `me`'s first idle command (its main phase), or None if it never gets
    there (the world went wrong)."""
    from ._core import Duel
    from . import rollout

    cache = rollout.REPO / ".cache"
    duel = Duel.world(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(spec.deck0), str(spec.deck1),
                      script, seed, me)
    for _ in range(max_steps):
        if duel.finished:
            return None
        keys = duel.keys()
        if duel.player == me and keys[0].startswith("idle"):
            return duel
        duel.act(decline(keys))
    return None


def read_decklist(path: Path) -> tuple[list[int], list[int]]:
    """Main and Extra Deck passcodes of a .ydk file."""
    main, extra, section = [], [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line in ("#main", "#extra", "!side"):
            section = line
        elif line.isdigit():
            (main if section == "#main" else extra if section == "#extra" else []).append(int(line))
    return main, extra


def visible(card: dict, me: int) -> bool:
    """Whether the searching player `me` knows this card's identity (outside its own deck)."""
    if card["controller"] == me and card["owner"] == me:
        return card["location"] != LOCATION_DECK
    if card["location"] in (LOCATION_DECK,):
        return False
    if card["location"] == LOCATION_GRAVE:
        return True
    return bool(card["public"]) or bool(card["position"] & POS_FACEUP)


def sample_world(snapshot: dict, me: int, decklists: dict[int, tuple[list[int], list[int]]], rng: random.Random,
                 truth: bool = False) -> str:
    """A puzzle script for one world. With truth=True hidden cards keep their real identity (only the
    decks are reshuffled): for checking the world builder against the real duel."""
    opponent = 1 - me
    cards = [dict(c) for c in snapshot["cards"]]
    if not truth:
        # What is left of the opponent's decklist after every card of it the player has seen.
        main, extra = decklists[opponent]
        pool_main, pool_extra = Counter(main), Counter(extra)
        seen = [c for c in cards if c["owner"] == opponent and visible(c, me)]
        for c in seen:
            for code in [c["code"], *c["materials"]]:
                if pool_main[code] > 0:
                    pool_main[code] -= 1
                elif pool_extra[code] > 0:
                    pool_extra[code] -= 1
        main_left = list(pool_main.elements())
        extra_left = list(pool_extra.elements())
        rng.shuffle(main_left)
        rng.shuffle(extra_left)
        for c in cards:
            if c["owner"] != opponent or visible(c, me):
                continue
            from_extra = c["location"] == LOCATION_EXTRA
            source = extra_left if from_extra else main_left
            if source:
                c["code"] = source.pop()
            c["materials"] = [m if m else (main_left.pop() if main_left else m) for m in c["materials"]]
    lines = [f"Debug.ReloadFieldBegin({DUEL_ATTACK_FIRST_TURN},5)", "local c"]
    for player in (0, 1):
        lines.append(f"Debug.SetPlayerInfo({player},{snapshot['life_points'][player]},0,0)")  # only this turn is played
    # Decks in a fresh random order; then everything else in its exact place.
    for player in (0, 1):
        deck = [c for c in cards if c["location"] == LOCATION_DECK and c["controller"] == player]
        rng.shuffle(deck)
        for i, c in enumerate(deck):
            lines.append(f"Debug.AddCard({c['code']},{c['owner']},{player},{LOCATION_DECK},{i},{c['position'] or 8})")
    for c in cards:
        if c["location"] == LOCATION_DECK or not c["code"]:
            continue
        proc = ",true" if c["location"] == LOCATION_MZONE else ""
        lines.append(f"c=Debug.AddCard({c['code']},{c['owner']},{c['controller']},{c['location']},"
                     f"{c['sequence']},{c['position']}{proc})")
        for material in c["materials"]:
            if material:
                lines.append(f"Debug.AddCard({material},{c['owner']},{c['controller']},{c['location']},"
                             f"{c['sequence']},{POS_FACEUP})")
        for counter, count in c["counters"]:
            lines.append(f"if c then Debug.PreAddCounter(c,{counter},{count}) end")
    lines.append("Debug.ReloadFieldEnd()")
    return "\n".join(lines) + "\n"
