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
    """A semantic key without what differs between the real duel and a rebuilt world: the list index
    of idle/battle commands (it follows card creation order) and, for multi-step selections, the
    request hash and engine step."""
    parts = key.split(".")
    if parts[0] in ("idle", "battle") and len(parts) > 3:
        del parts[2]
    elif parts[0] == "cont" and len(parts) > 1:
        del parts[1]
        if "engine" in parts:
            i = parts.index("engine")
            del parts[i:i + 2]
    return ".".join(parts)


def match(key: str, keys: list[str]) -> str | None:
    """The answer among `keys` that is `key` (from another duel): equal, equal in canonical form, or
    equal but for the last field (a card's sequence in a pile whose order differs)."""
    if key in keys:
        return key
    wanted = canonical(key)
    for k in keys:
        if canonical(k) == wanted:
            return k
    loose = wanted.rsplit(".", 1)[0]
    for k in keys:
        if canonical(k).rsplit(".", 1)[0] == loose:
            return k
    card = card_ref(key)  # the same answer about the same card, whatever the list and pile order
    if card is not None:
        for k in keys:
            if card_ref(k) == card:
                return k
    return None


def card_ref(key: str):
    """(answer kind, passcode, controller, location) of an answer about a card, else None. Keys that
    name a card end in passcode, controller, location, sequence."""
    parts = key.split(".")
    if len(parts) < 5 or not all(p.isdigit() for p in parts[-4:]) or int(parts[-4]) < 1000:
        return None
    kind = tuple(p for p in parts[:-4] if not p.isdigit())
    return kind, parts[-4], parts[-3], parts[-2]


def forced_cards(history: list[str], start: dict, me: int) -> dict:
    """Hidden opposing cards at the turn's start whose identity this turn's answers show: an answer
    naming a card in a spot that held a hidden card at the start (a set card activated, say) fixes it.
    Returns {(controller, location, sequence): passcode}."""
    hidden = {(c["controller"], c["location"], c["sequence"]) for c in start["cards"]
              if c["owner"] != me and not visible(c, me) and c["location"] in (LOCATION_MZONE, LOCATION_SZONE)}
    forced = {}
    for key in history:
        ref = card_ref(key)
        if ref is None:
            continue
        parts = key.split(".")
        spot = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
        if spot in hidden and spot not in forced:
            forced[spot] = int(parts[-4])
    return forced


def phase_group(phase: int) -> int:
    """1 = Main Phase 1, 2 = Battle Phase, 3 = Main Phase 2, 0 = anything else."""
    if phase & 0x4:
        return 1
    if phase & 0xF8 and not phase & 0x100:
        return 2
    if phase & 0x100:
        return 3
    return 0


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


def seen_since(before: dict, now: dict, me: int) -> Counter:
    """Opposing cards visible to `me` in `now` beyond those visible in `before` (by passcode)."""
    count = lambda snap: Counter(c["code"] for c in snap["cards"]
                                 if c["owner"] != me and c["code"] and visible(c, me))
    return count(now) - count(before)


def drawn_since(before: dict, now: dict, me: int) -> int:
    """How many cards left `me`'s deck between the two snapshots."""
    size = lambda snap: sum(1 for c in snap["cards"] if c["location"] == LOCATION_DECK and c["controller"] == me)
    return max(0, size(before) - size(now))


def world_now(spec, start: dict, now: dict, history: list[str], me: int, decklists, rng: random.Random,
              seed: int, attempts: int = 6):
    """A world at the current point of this turn: sampled at the turn's start (`start`, the snapshot
    at `me`'s first idle command), consistent with what `me` has seen since (`now`), then this turn's
    answers (`history`, both players') replayed into it. Worlds whose replay breaks (a sampled card
    the opponent cannot have played) are rejected; None after `attempts` tries."""
    revealed, own_top = seen_since(start, now, me), drawn_since(start, now, me)
    forced = forced_cards(history, start, me)
    for attempt in range(attempts):
        script = sample_world(start, me, decklists, rng, revealed=revealed, own_top=own_top, forced=forced)
        duel = start_world(spec, script, me, seed + attempt)
        if duel is None:
            continue
        for key in history:
            found = None
            for _ in range(8):  # questions the real duel did not ask (a sampled card's window): decline
                if duel.finished:
                    break
                keys = duel.keys()
                found = match(key, keys)
                if found is not None or decline(keys) == keys[0] and keys[0] not in ("chain.pass", "yes_no.no"):
                    break
                duel.act(decline(keys))
            if found is None:
                duel = None
                break
            duel.act(found)
        if duel is not None and not duel.finished:
            return duel
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
                 truth: bool = False, revealed: Counter | None = None, own_top: int = 0,
                 forced: dict | None = None) -> str:
    """A puzzle script for one world. With truth=True hidden cards keep their real identity (only the
    decks are reshuffled): for checking the world builder against the real duel.

    For a world of the turn so far (to replay this turn's answers into), `revealed` are opposing
    cards the player has seen since `snapshot` was taken: they are dealt to the opponent's hidden
    cards first. `own_top` keeps that many cards on top of the player's own deck in their real
    order: the cards it has drawn or revealed from it since then. `forced` fixes hidden opposing field
    cards to the passcodes this turn's answers showed for them (forced_cards)."""
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
        for code, count in (revealed or Counter()).items():  # dealt first (popped from the end)
            for _ in range(count):
                for pool in (main_left, extra_left):
                    if code in pool:
                        pool.remove(code)
                        pool.append(code)
                        break
        for spot, code in (forced or {}).items():
            for c in cards:
                if (c["controller"], c["location"], c["sequence"]) == spot and not visible(c, me):
                    c["code"] = code
                    c["forced"] = True
                    for pool in (main_left, extra_left):
                        if code in pool:
                            pool.remove(code)
                            break
        for c in cards:
            if c["owner"] != opponent or visible(c, me) or c.get("forced"):
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
        deck.sort(key=lambda c: c["sequence"])
        keep = own_top if player == me and own_top > 0 else 0
        bottom, top = (deck[:len(deck) - keep], deck[len(deck) - keep:]) if keep else (deck, [])
        rng.shuffle(bottom)
        deck = bottom + top
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
