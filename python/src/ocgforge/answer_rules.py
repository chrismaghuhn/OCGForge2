"""Rules for single answers in the combo search (combo_check.check): fixed positions and zones,
the next step's card in a card choice, and look-ahead on a clone (what the next question offers
after an answer). From playing Lucina 1cc by hand (docs/combo-teacher-design.md, K1 finding).
"""

from __future__ import annotations

from collections import Counter

from . import worlds

POS_ATTACK = 0x1
EMZ = (5, 6)


def placing(keys: list[str]) -> int | None:
    """The card a zone question is about: the last summon, activation or card choice of ours
    (materials and costs are chosen after it, so selections of several cards do not count)."""
    for key in reversed(keys):
        if key.startswith(("idle.", "card.", "chain.")):
            ref = worlds.card_ref(key)
            if ref is not None:
                return int(ref[1])
    return None


def forced(keys: list[str], history: list[str], upcoming_center: bool, center: set[int], is_extra) -> str | None:
    """The answer the rules fix without searching (docs/combo-teacher-design.md, K1 finding), or None.
    - Position: face-up attack.
    - Monster zone: the center for a monster that wants it; the Extra Monster Zone for another Extra
      Deck monster; else a side zone, keeping the center free when the plan still needs it."""
    if all(k.startswith("position.") for k in keys):
        attack = [k for k in keys if k.rsplit(".", 1)[-1].isdigit() and int(k.rsplit(".", 1)[-1]) & POS_ATTACK]
        return (attack or keys)[0]
    zones = {int(k.split(".")[3]): k for k in keys if k.startswith("place.0.4.") and len(k.split(".")) == 4}
    if zones and len(zones) == len(keys):
        code = placing(history)
        if code in center and 2 in zones:
            return zones[2]
        if code is not None and is_extra(code) and code not in center:
            for z in EMZ:
                if z in zones:
                    return zones[z]
        sides = [z for z in (1, 3, 0, 4) if z in zones]
        if sides:
            return zones[sides[0]]
        return zones[2] if 2 in zones and not upcoming_center else next(iter(zones.values()))
    return None


def plan_choice(keys: list[str], allowed: set[int], ahead: list[int]) -> set[int]:
    """In a choice of one card (`card.` answers: a search, a target, a monster to summon), only the
    card of the earliest coming step that is offered; the model must not swap it for another legal
    card (Welcome Home calling Fortuna where the plan wants Regina). Unchanged when none is offered."""
    if not all(k.startswith("card.") for k in keys):
        return allowed
    codes = {i: int(ref[1]) for i in allowed if (ref := worlds.card_ref(keys[i])) is not None}
    for code in ahead:
        chosen = {i for i, c in codes.items() if c == code}
        if chosen:
            return chosen
    return allowed


def peek(duel, key: str, history: list[str], upcoming_center: bool, center: set[int], is_extra,
         ahead: list[int], max_answers: int = 12) -> set[int]:
    """The cards our next real question offers after `key`, on a clone: optional chain windows are
    passed and positions and zones fixed (`forced`) on the way. A yes/no question ("reduce
    Strelitzia's level?") decides whether the plan's next Synchro is still there."""
    return follow(duel, key, history, upcoming_center, center, is_extra, ahead, max_answers)[0]


def deck_cards(duel) -> Counter:
    return Counter(c["code"] for c in duel.snapshot()["cards"] if c["controller"] == 0 and c["location"] == worlds.LOCATION_DECK)


def follow(duel, key: str, history: list[str], upcoming_center: bool, center: set[int], is_extra,
           ahead: list[int], max_answers: int = 12):
    """`peek`, plus the clone at that next question (None if it went wrong).

    Hidden information: a clone holds the real deck order (`Duel.clone(seed)` only reseeds future
    randomness, it does not reshuffle; docs/engine-clone.md). Searching a card from the deck is a
    choice among cards we know, but a draw, excavation or deck mill in between would show what the
    real deck holds. So when any card leaves our deck that the answer itself did not name, the
    look-ahead gives nothing (no offered cards, no clone): the search decides without it. Choices
    that depend on deck order still stay honest only through the shuffles combo_check runs."""
    copy = None
    try:
        before = deck_cards(duel)
        named = {int(ref[1]) for ref in [worlds.card_ref(key)] if ref is not None}
        copy = duel.clone()
        copy.act(key)
        history = history + [key]
        for _ in range(max_answers):
            left = before - deck_cards(copy)
            if any(code not in named for code in left) or sum(left.values()) > len(named):
                return set(), None  # the deck order showed: no look-ahead
            if copy.finished:
                return set(), copy
            keys = copy.keys()
            offered = {int(ref[1]) for k in keys if (ref := worlds.card_ref(k)) is not None}
            optional_chain = "chain.pass" in keys and all(k.startswith("chain.") for k in keys)
            if copy.player != 0 or (optional_chain and not offered & set(ahead)):
                answer = worlds.decline(keys)
            else:
                answer = forced(keys, history, upcoming_center, center, is_extra)
                if answer is None:
                    return offered, copy
            copy.act(answer)
            history = history + [answer]
    except Exception:
        return set(), None
    return set(), copy


def choice_by_outcome(duel, keys: list[str], allowed: set[int], before, pending: list[tuple[int, int | None]],
                      follow_args: tuple) -> set[int]:
    """In a choice of one card, the answers after which (on a clone) the earliest coming step's card
    arrives where the step wants it: Shooting Riser sends Spell Canceller, not the Regina whose
    step is a return to hand. Unchanged when no answer does that."""
    if not all(k.startswith("card.") for k in keys) or len(allowed) < 2:
        return allowed
    best, chosen = len(pending), set()
    for i in allowed:
        _, copy = follow(duel, keys[i], *follow_args)
        if copy is None:
            continue
        mine = Counter((c["code"], c["location"]) for c in copy.snapshot()["cards"] if c["controller"] == 0)
        arrived = {code_loc for code_loc, n in (mine - before).items() if n > 0}
        hit = next((j for j, (code, loc) in enumerate(pending) if loc is not None and (code, loc) in arrived), None)
        if hit is not None and hit < best:
            best, chosen = hit, {i}
        elif hit is not None and hit == best:
            chosen.add(i)
    return chosen or allowed


STEP_BONUS = 20.0  # for an answer naming the next step's card: above the model's log-probabilities


def step_bonus(index: int) -> float:
    """Bonus for naming the card of the `index`-th coming step (0 = the next)."""
    return STEP_BONUS * (1.0 - index / 3.0)


def offered_bonus(offered: set[int], ahead: list[int]) -> float:
    """Bonus for an answer after which the next steps' cards are offered (6 for the next one, as
    for answers naming it)."""
    return max((step_bonus(i) for i, code in enumerate(ahead) if code in offered), default=0.0)


def trigger_key(row: dict, keys: list[str], i: int, codes: dict[int, int]) -> str | None:
    """A key naming the card a yes/no answer is about ("activate Power Patron's effect?"), from the
    encoded decision (candidate source token -> card index -> passcode), or None."""
    try:
        token = row["candidate_source"][i]
        code = codes.get(int(row["token_card"][token])) if token is not None and token >= 0 else None
    except (KeyError, IndexError, TypeError):
        return None
    return f"trigger.{code}.0.0.0" if code else None
