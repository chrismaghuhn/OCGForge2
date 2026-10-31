"""K1 of the combo teacher: replay community combos in our engine (docs/combo-teacher-design.md).

    python -m ocgforge.combo_check --deck decks/elfnote.ydk --deck-type Elfnote --checkpoint runs/x/latest

Each combo becomes a solo puzzle: we go first with its starting cards (filled up to 5 from the
decklist), the opponent holds a hand but never acts. A beam search over the engine's answers follows the combo's
steps. A step is done when its card got where the step's action text says (summoned, added to hand,
sent to the GY, ...; or, for a bare "use", named by an answer), checked on the board after each
answer; the model ranks everything the steps do not decide (materials, zones, targets), and the
book's end-board cards on the field rank lines too. A combo counts as checked when every step is
done in order and at least 80 % of the book's end-board cards are on our field. The
end board and the line (semantic keys) go into the combo book. Several deck shuffles per combo
(--shuffles) keep draws and excavations honest: a combo only counts if it works in all of them.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import combos as cb
from . import features, rollout, worlds

TYPE_EXTRA = 0x40 | 0x2000 | 0x800000 | 0x4000000  # Fusion, Synchro, Xyz, Link
PHASE_END = re.compile(r"^idle\.[6-9]\.\d+$")  # to Battle Phase / End Phase (no card)
# Monsters that want the center Main Monster Zone (summoned there, or stronger while there).
CENTER = re.compile(r"(this card|Summoned|it)[^.]{0,40}\b(to|in) (the|your) center Main Monster Zone"
                    r"|while in the center Main Monster Zone")
# Answers that move the duel on and cannot loop: the board stays the same over several chain windows,
# so the loop guard (same answer on the same board) must not take them away.
PASSIVE = ("chain.pass", "yes_no.", "effect_yes_no.")
POS_ATTACK = 0x1
MAX_SKIPPED = 2  # steps a checked combo may pass over (once-per-turn effects, steps the board cannot show)
EMZ = (5, 6)


def wants_center(text: str) -> bool:
    return bool(CENTER.search(text or ""))


def filler_rng(combo: cb.Combo) -> random.Random:
    """The random source of a combo's filler cards: the same in every shuffle (see scenario_script)."""
    return random.Random(f"{combo.id}-hand")


def scenario_script(combo: cb.Combo, decklist: tuple[list[int], list[int]], opponent: list[int], is_extra,
                    rng: random.Random, hand_size: int = 5, hand: list[int] | None = None,
                    filler_rng: random.Random | None = None, opponent_hand: list[int] = ()) -> str:
    """The puzzle: our hand (the combo's starting cards, filled up from the decklist), our deck and
    Extra Deck (the decklist plus the combo's cards it lacks), an opponent with a deck and a hand it
    never plays (effects like PSY-Framelord Omega's need one).

    With `filler_rng` the cards filling the hand come from it (one draw per combo, the same in every
    shuffle), and `rng` only orders the decks: the shuffles then test what the combo player cannot
    know (deck order: draws, excavations), not a different starting hand. Without it (books before
    v3) `rng` picks the fillers too. `opponent_hand` cards (hand traps for interruption branches) go
    into the opponent's hand first."""
    main, extra = list(decklist[0]), list(decklist[1])
    for code in combo.cards():  # the combo's cards we do not run
        target = extra if is_extra(code) else main
        if code not in target:
            target.append(code)
    wanted = list(hand) if hand is not None else [e.code for e in combo.starting if e.code is not None]
    hand = []
    for code in wanted:
        if code in main:
            main.remove(code)
        hand.append(code)
    if filler_rng is not None:
        pool = sorted(main)
        filler_rng.shuffle(pool)
        for code in pool[:max(0, hand_size - len(hand))]:
            main.remove(code)
            hand.append(code)
    rng.shuffle(main)
    while len(hand) < hand_size and main:
        hand.append(main.pop())
    lines = ["Debug.ReloadFieldBegin(0,5)", "Debug.SetPlayerInfo(0,8000,0,0)", "Debug.SetPlayerInfo(1,8000,0,0)"]
    for i, code in enumerate(main):
        lines.append(f"Debug.AddCard({code},0,0,{worlds.LOCATION_DECK},{i},8)")
    for code in extra:
        lines.append(f"Debug.AddCard({code},0,0,{worlds.LOCATION_EXTRA},0,8)")
    for i, code in enumerate(hand):
        lines.append(f"Debug.AddCard({code},0,0,{worlds.LOCATION_HAND},{i},10)")
    opponent = list(opponent)
    rng.shuffle(opponent)
    for code in reversed(list(opponent_hand)):
        if code in opponent:
            opponent.remove(code)
        opponent.insert(0, code)
    for i, code in enumerate(opponent[hand_size:]):
        lines.append(f"Debug.AddCard({code},1,1,{worlds.LOCATION_DECK},{i},8)")
    for i, code in enumerate(opponent[:hand_size]):  # it never plays them, but effects may target a hand
        lines.append(f"Debug.AddCard({code},1,1,{worlds.LOCATION_HAND},{i},10)")
    lines.append("Debug.ReloadFieldEnd()")
    return "\n".join(lines) + "\n"


_cpu_models: dict[int, object] = {}
_cpu_lock = threading.Lock()


@torch.no_grad()
def candidate_logprobs(model, rows: list[dict], device: str) -> list[list[float]]:
    """The model's log-probabilities of each row's answers. On a shared GPU that runs out of memory
    the batch goes to a CPU copy of the model instead of failing the run."""
    model.eval()
    try:
        inputs = features.collate(rows, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            scores, _ = model(inputs)
    except torch.cuda.OutOfMemoryError:
        with _cpu_lock:
            if id(model) not in _cpu_models:
                _cpu_models[id(model)] = copy.deepcopy(model).to("cpu")
            cpu_model = _cpu_models[id(model)]
        scores, _ = cpu_model(features.collate(rows, "cpu"))
    logprobs = torch.log_softmax(scores.float(), dim=-1).cpu()
    return [logprobs[i, : len(r["candidate_kind"])].tolist() for i, r in enumerate(rows)]


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


def board(duel) -> Counter:
    """(passcode, location) -> count of our cards, for telling what an answer did."""
    return Counter((c["code"], c["location"]) for c in duel.snapshot()["cards"] if c["controller"] == 0)


@dataclass
class Line:
    keys: list[str] = field(default_factory=list)  # every answer, both players
    progress: int = 0  # steps done so far
    score: float = 0.0  # the model's log-probabilities, plus bonuses
    board: Counter = field(default_factory=Counter)  # our cards after the last answer
    key_cards: int = 0  # book end-board cards on our field now
    tried: frozenset = frozenset()  # (board, answer) pairs this line already played: no loops
    skipped: int = 0  # steps passed over without being done
    ahead: frozenset = frozenset()  # later steps of the current community step already done
    passed: tuple = ()  # the skipped steps
    trigger: str | None = None  # for a "yes" to "activate X's effect?": a key naming X (yes/no keys name no card)
    intent: float = 0.0  # bonus for heading to the next step (its card named), kept until a step is done
    duel: object = field(default=None, repr=False, compare=False)  # the engine at this line's end (a clone)


def advance(line: Line, targets: list[tuple[int, int | None]], last_key: str | None, now: Counter,
            lookahead: int, groups: list[int] | None = None, loose: frozenset = frozenset()
            ) -> tuple[int, int, frozenset]:
    """How many steps the last answer completed, how many of them it skipped, and which later steps
    are done ahead of their turn: a step with a location is done when its card arrived there; one
    without, when the answer named its card. Entries of one community step (`groups`: "Strelitzia,
    Regina, Power Patron") may be done in any order at no cost. A few steps may be skipped (the
    combo text names cards the engine never acts on, or an effect that is once per turn), but a
    skip costs the line rank."""
    arrived = {code_loc for code_loc, n in (now - line.board).items() if n > 0}
    named = worlds.card_ref(last_key) if last_key else None
    progress, skipped, ahead = line.progress, [], set(line.ahead)
    for _ in range(4):  # one answer can complete several steps (a summon that is also a material)
        hit = None
        for skip in range(lookahead):
            i = progress + skip
            if i >= len(targets):
                break
            if i in ahead:
                continue
            code, location = targets[i]
            # A step without a location (the card was used) is done by an answer naming its card, but
            # not past an earlier open step of the same card: the answer summoning a card names it
            # too, before it arrives for that earlier step.
            unblocked = not any(targets[j][0] == code for j in range(progress, i) if j not in ahead)
            # A card already where its step wants it (Medius's effect in the GY, Welcome Home
            # placed before) is done when an answer names it there.
            if (location is not None and (code, location) in arrived) or (
                    unblocked and named is not None and int(named[1]) == code
                    and (location is None or i in loose or (named[2] == "0" and named[3] == str(location)))):
                hit = i
                if location is not None:
                    arrived.discard((code, location))
                else:
                    named = None  # one answer names one card for one step
                break
        if hit is None:
            break
        if groups is not None and groups[hit] == groups[progress]:
            ahead.add(hit)  # in its own step: no skip
        else:
            skipped += [i for i in range(progress, hit) if i not in ahead]
            ahead = {i for i in ahead if i > hit}
            progress = hit + 1
        while progress in ahead:
            ahead.discard(progress)
            progress += 1
    return progress - line.progress, tuple(skipped), frozenset(ahead)


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


def grow(children: list[tuple[Line, object]]) -> list[Line]:
    """Play each chosen child's last answer on a clone of its parent's duel; the parent's last child
    takes the parent's duel itself. A child whose answer the engine rejects is dropped."""
    left = Counter(id(parent) for _, parent in children)
    grown = []
    for child, parent in children:
        left[id(parent)] -= 1
        try:
            duel = parent if left[id(parent)] == 0 else parent.clone()
            duel.act(child.keys[-1])
        except Exception:
            continue
        child.duel = duel
        grown.append(child)
    return grown


def peek(duel, key: str, history: list[str], upcoming_center: bool, center: set[int], is_extra,
         ahead: list[int], max_answers: int = 12) -> set[int]:
    """The cards our next real question offers after `key`, on a clone: optional chain windows are
    passed and positions and zones fixed (`forced`) on the way. A yes/no question ("reduce
    Strelitzia's level?") decides whether the plan's next Synchro is still there."""
    try:
        copy = duel.clone()
        copy.act(key)
        history = history + [key]
        for _ in range(max_answers):
            if copy.finished:
                return set()
            keys = copy.keys()
            offered = {int(ref[1]) for k in keys if (ref := worlds.card_ref(k)) is not None}
            optional_chain = "chain.pass" in keys and all(k.startswith("chain.") for k in keys)
            if copy.player != 0 or (optional_chain and not offered & set(ahead)):
                answer = worlds.decline(keys)
            else:
                answer = forced(keys, history, upcoming_center, center, is_extra)
                if answer is None:
                    return offered
            copy.act(answer)
            history = history + [answer]
    except Exception:
        return set()
    return set()


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


def check(combo: cb.Combo, start, model, card_index, device: str, beam: int = 12, branch: int = 4,
          max_decisions: int = 150, lookahead: int = 3, center: set[int] = frozenset(),
          is_extra=lambda code: False, opponent=None) -> tuple[Line, list[tuple[int, int | None]]]:
    """Beam search through the combo from a fresh puzzle duel (`start()`); returns the best line and
    the steps (card, expected location) it had to follow. Lines rank by steps done, then by how many
    of the book's end-board cards are on the field, then by the model's log-probabilities.

    Rules from playing Lucina 1cc by hand: positions and monster zones are fixed (`forced`); in a
    chain window only cards of the next steps may activate, else pass; in a choice of one card only the
    next step's card when offered (`plan_choice`); optional yes/no questions keep
    both answers in the beam (the board decides, not the model: a level change can break a later
    Synchro); end-board cards on the field are never chosen as material or cost."""
    targets = step_targets(combo)
    groups = step_groups(combo)
    loose = step_loose(combo)
    codes: dict[int, int] = {}  # card index -> passcode (originals before alternate artworks)
    for code, index in card_index.items():
        codes.setdefault(index, code)
    key_cards = Counter(combo.end_board)
    plan = {code for code, _ in targets} | set(key_cards)
    center = set(center) & plan
    # Steps first: an end-board card played early (a search taken too soon) must not beat the plan.
    # A summon takes several answers (materials, zone) before its step counts: the bonus for naming
    # the step's card (`intent`) carries the line until then, and is dropped once the step is done or
    # the summon cancelled. It never adds up over a line, so cancelling and retrying earns nothing.
    # Heading for the very next step counts as half a step: its materials leave the field before the
    # summoned card arrives, and the end-board count must not drop the line in between.
    rank = lambda l: (l.progress - l.skipped + (0.5 if l.intent >= STEP_BONUS else 0.0), l.key_cards,  # noqa: E731
                      l.score + l.intent)
    first = start()
    if first is None:
        return Line(), targets
    # Each line holds its own duel: children clone their parent's (Duel.clone, ~5 ms) instead of
    # replaying the whole line into a fresh duel every decision. The keys stay the record: a line
    # replays from `start()` to the same position (main() replays the best one for the end board).
    lines = [Line(board=board(first), duel=first)]
    done: list[Line] = []
    for _ in range(max_decisions):
        expanded: list[tuple[Line, object]] = []
        seen: set = set()  # (board, question, steps) already in the beam: the rest are duplicates
        for line in lines:
            duel, line.duel = line.duel, None
            if duel is None:
                continue
            try:
                while not duel.finished and duel.player != 0:  # the opponent never acts
                    key = opponent(duel, line.keys) if opponent else worlds.decline(duel.keys())
                    duel.act(key)
                    line.keys.append(key)
            except Exception:  # the engine failed on this line: drop it, not the whole run
                continue
            now = board(duel)
            mine = next((k for k in reversed(line.keys) if not k.startswith(("chain.pass", "yes_no.no"))), None)
            if mine == "yes_no.yes" and line.trigger:
                mine = line.trigger  # an optional trigger effect was activated: it uses its card
            steps, skipped, line.ahead = advance(line, targets, mine, now, lookahead, groups, loose)
            line.progress += steps
            line.skipped += len(skipped)
            line.passed += skipped
            if steps:
                line.intent = 0.0
            line.board = now
            field_now = Counter({code: n for (code, loc), n in now.items()
                                 if loc in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)})
            line.key_cards = sum((key_cards & field_now).values())
            # The board has no levels or ATK: yes/no answers (a level change, say) tell such states apart.
            choices = tuple(k for k in line.keys if k.startswith(("yes_no.", "effect_yes_no.", "option.")))
            identity = (hash(tuple(sorted(now.items()))), tuple(duel.keys()) if not duel.finished else (),
                        line.progress, line.skipped, line.ahead, choices)
            if identity in seen:
                continue
            seen.add(identity)
            if duel.finished or duel.turn > 1 or (line.progress == len(targets) and
                                                  line.key_cards >= sum(key_cards.values())):
                done.append(line)
            else:
                expanded.append((line, duel))
        if not expanded:
            break
        rows = [d.encode(card_index) for _, d in expanded]
        logprobs = candidate_logprobs(model, rows, device)
        children = []
        for (line, duel), lp, row in zip(expanded, logprobs, rows):
            keys = duel.keys()
            state = hash(tuple(sorted(line.board.items())))
            pending = [i for i in range(line.progress, len(targets)) if i not in line.ahead][:lookahead]
            ahead = [targets[i][0] for i in pending]
            window = set(ahead)
            # Cards the last few steps named: an end-board card among them may be a material now
            # (Strelitzia into Crystal Wing, revived later for the end board).
            recent = {code for code, _ in targets[max(0, line.progress - 3):line.progress]} | {
                targets[i][0] for i in line.ahead}
            upcoming = window | set(key_cards)
            later_center = any(code in center for code, _ in targets[line.progress:])
            fixed = forced(keys, line.keys, later_center, center, is_extra)
            on_field = {code for (code, loc), n in line.board.items()
                        if n > 0 and loc in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)}
            # An optional chain window (a mandatory one has no pass and is never filtered).
            chain_window = "chain.pass" in keys and all(k.startswith("chain.") for k in keys)
            if chain_window and not any((ref := worlds.card_ref(k)) and int(ref[1]) in window for k in keys):
                fixed = fixed or ("chain.pass" if "chain.pass" in keys else None)
            options = []
            for i, key in enumerate(keys):
                if fixed is not None:
                    if key == fixed:
                        options.append((0.0, 0.0, 0.0, i))
                    continue
                if (state, key) in line.tried and len(keys) > 1 and not key.startswith(PASSIVE):
                    continue  # same answer on the same board again: a summon cancelled and retried
                options.append((0.0, 0.0, 0.0, i))
            if fixed is None:
                allowed = {o[-1] for o in options} or set(range(len(keys)))  # all retried: undo still allowed
                options = []
                allowed = plan_choice(keys, allowed, ahead)
            for i, key in enumerate(keys):
                if fixed is not None or i not in allowed:
                    continue
                ref = worlds.card_ref(key)
                value, bonus = lp[i], 0.0
                if chain_window and ref is not None and int(ref[1]) not in window:
                    continue  # our own hand trap or negation on our own chain
                if (ref is not None and key.startswith(("unselect.", "select.", "card.")) and int(ref[1]) in key_cards
                        and int(ref[1]) in on_field and int(ref[1]) not in window | recent and len(keys) > 1):
                    continue  # an end-board card as material or cost
                if key.startswith(("yes_no.", "effect_yes_no.")):
                    value = 0.5 if key.endswith(".yes") else 0.0  # both stay; the board decides
                    bonus = offered_bonus(peek(duel, key, line.keys, later_center, center, is_extra, ahead), ahead)
                    if key == "yes_no.yes" and (named := trigger_key(row, keys, i, codes)):
                        ref = worlds.card_ref(named)  # yes to a step card's trigger effect counts like naming it
                if ref is not None and int(ref[1]) in window:
                    bonus = step_bonus(ahead.index(int(ref[1])))  # the next step's card first
                elif ref is not None and int(ref[1]) in upcoming:
                    value += 1.0  # an end-board card, but not yet
                elif ref is not None and int(ref[1]) not in plan and line.progress < len(targets):
                    value -= 8.0  # off-plan card (a hand trap fired at nothing, a wrong Synchro, ...)
                elif line.progress < len(targets) and PHASE_END.match(key):
                    value -= 20.0  # leaving Main Phase 1 with the combo unfinished
                if key.endswith(".cancel") and len(keys) > 1:
                    value -= 10.0  # undoing a summon or selection just started: rarely the plan
                options.append((value + bonus, value, bonus, i))
            if not options:  # every answer filtered out: stop here rather than break a rule
                done.append(line)
                continue
            for _, value, bonus, i in sorted(options, reverse=True)[:branch]:
                # the latest card named for a coming step sets the intent (a material choice keeps it)
                intent = 0.0 if keys[i].endswith(".cancel") else bonus or line.intent
                trigger = trigger_key(row, keys, i, codes) if keys[i] == "yes_no.yes" else None
                children.append((Line(line.keys + [keys[i]], line.progress, line.score + value, line.board,
                                      line.key_cards, line.tried | {(state, keys[i])}, line.skipped, line.ahead,
                                      line.passed, trigger, intent), duel))
        lines = grow(sorted(children, key=lambda c: rank(c[0]), reverse=True)[:beam])
    best = max(done + lines, key=rank, default=Line())
    return best, targets


def checked_line(combo: cb.Combo, line: Line, targets: list) -> bool:
    """Checked: every step done in order (at most MAX_SKIPPED passed over) and at least 80 % of the
    book's end-board cards on our field."""
    return bool(targets) and line.progress == len(targets) and line.skipped <= MAX_SKIPPED and (
        not combo.end_board or line.key_cards >= 0.8 * len(combo.end_board))


def end_board(duel) -> list[dict]:
    """Our field and the cards we added to hand: what the combo left us."""
    snapshot = duel.snapshot()
    return [{"code": c["code"], "location": c["location"], "sequence": c["sequence"], "position": c["position"]}
            for c in snapshot["cards"] if c["controller"] == 0
            and c["location"] in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE, worlds.LOCATION_HAND)]


def main(argv=None) -> None:
    from ._core import Duel
    from .cards import CardTable
    from .model import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--deck-type", required=True, help="masterduelmeta deck type, e.g. Elfnote")
    parser.add_argument("--opponent", type=Path, default=rollout.REPO / "decks" / "salamangreat.ydk")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shuffles", type=int, default=3, help="deck orders a combo must work in")
    parser.add_argument("--beam", type=int, default=24)
    parser.add_argument("--branch", type=int, default=4, help="answers tried per line and decision")
    parser.add_argument("--title", help="only combos whose title contains this")
    parser.add_argument("--precise", type=Path,
                        help="hand-written engine steps (JSON list of {title, hand, steps: [[card, action]], "
                             "end_board}): replace the community steps, starting hand and end board of those combos")
    parser.add_argument("--limit", type=int, default=0, help="check only the best-voted N combos")
    parser.add_argument("--workers", type=int, default=4, help="combos/shuffles searched at once (threads)")
    parser.add_argument("--max-decisions", type=int, default=300, help="our answers per line at most")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="CPU threads for the model; the workers already keep --workers cores busy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, help="combo book (JSON lines)")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.torch_threads)

    table = CardTable.from_cdb()
    types = {card.code: card.type for card in table.cards}
    is_extra = lambda code: bool(types.get(code, 0) & TYPE_EXTRA)  # noqa: E731
    center = {card.code for card in table.cards if wants_center(card.text)}
    names = cb.name_index(table)
    model = load_checkpoint(args.checkpoint, args.device, dropout=0.0)
    card_index = model.card_features.index_map()
    decklist = worlds.read_decklist(args.deck)
    opponent = worlds.read_decklist(args.opponent)[0]
    combos = sorted((c for c in cb.load(args.deck_type, names) if c.usable and c.steps),
                    key=lambda c: -c.votes)
    hands: dict[str, list[int]] = {}
    if args.precise:
        code_of = lambda name: names.get(cb.normalize(name), names.get(cb.loose(name)))  # noqa: E731
        specs = {spec["title"]: spec for spec in json.loads(args.precise.read_text(encoding="utf-8"))}
        combos = [c for c in combos if c.title in specs]
        for combo in combos:
            spec = specs[combo.title]
            combo.steps = [[cb.Entry(code_of(card), card, action)] for card, action in spec["steps"]]
            combo.end_board = [code_of(card) for card in spec.get("end_board", [])]
            hands[combo.id] = [code_of(card) for card in spec["hand"]]
    if args.title:
        combos = [c for c in combos if args.title in c.title]
    if args.limit:
        combos = combos[: args.limit]
    cache = rollout.REPO / ".cache"
    verbs = Counter()  # action text -> [used, total]
    verbs_used = Counter()
    located = Counter()  # step texts mapped to a location (True) or only "card named" (False)

    def start_for(script: str, seed: int):
        def start():
            try:
                duel = Duel.world(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(args.deck),
                                  str(args.opponent), script, seed, 0)
                for _ in range(20):  # through the draw and standby phase to our first decision
                    if duel.finished or duel.player == 0:
                        break
                    duel.act(worlds.decline(duel.keys()))
            except Exception:  # a scenario the engine rejects
                return None
            return None if duel.finished else duel
        return start

    def one(combo: cb.Combo, shuffle: int):
        """One deck order of one combo: its own scenario and seed (`scenario(combo, shuffle)`)."""
        began = time.perf_counter()
        rng = random.Random(f"{combo.id}-{shuffle}")
        script = scenario_script(combo, decklist, opponent, is_extra, rng, hand=hands.get(combo.id),
                                 filler_rng=filler_rng(combo))
        start = start_for(script, rng.getrandbits(32))
        line, targets = check(combo, start, model, card_index, args.device, beam=args.beam, branch=args.branch,
                              max_decisions=args.max_decisions,
                              center=center, is_extra=is_extra)
        return line, targets, start, time.perf_counter() - began

    # Combos and shuffles run in threads: the engine releases the GIL (Duel.world, act, clone).
    pool = ThreadPoolExecutor(max(1, args.workers))
    jobs = [[pool.submit(one, combo, shuffle) for shuffle in range(args.shuffles)] for combo in combos]
    results = []
    for n, (combo, futures) in enumerate(zip(combos, jobs)):
        runs = [f.result() for f in futures]
        verdicts = [checked_line(combo, line, targets) for line, targets, _, _ in runs]
        checked = sum(verdicts)
        # The book keeps a line that is checked if there is one, else the furthest.
        shuffle = max(range(len(runs)), key=lambda i: (verdicts[i], runs[i][0].progress - runs[i][0].skipped,
                                                      runs[i][0].key_cards))
        line, targets, start, _ = runs[shuffle]
        seconds = sum(r[3] for r in runs)
        entries = [e for _, e in step_entries(combo)]
        for i, (e, (_, location)) in enumerate(zip(entries, targets)):
            verb = e.action.strip().lower() or "(use)"
            verbs[verb] += 1
            verbs_used[verb] += i < line.progress
            located[location is not None] += 1
        duel = start()  # the line replays from the scenario alone (no clone involved)
        replays = duel is not None
        for key in line.keys if duel is not None else []:
            if duel.finished or key not in duel.keys():
                replays = False
                break
            duel.act(key)
        replays = replays and board(duel) == line.board
        final = end_board(duel) if duel is not None else []
        on_field = [c for c in final if c["location"] in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)]
        result = {"combo": combo.id, "title": combo.title, "votes": combo.votes, "steps": len(targets),
                  "progress": line.progress, "skipped": line.skipped,
                  "skipped_steps": [[i, entries[i].name, entries[i].action] for i in line.passed], "checked_shuffles": checked, "shuffles": args.shuffles,
                  "checked": checked == args.shuffles, "shuffle": shuffle, "fixed_hand": True, "replays": replays, "line": line.keys,
                  "end_board": final, "field": len(on_field), "book_end_board": combo.end_board,
                  "key_cards_reached": (sum(1 for c in combo.end_board if c in {b["code"] for b in on_field})
                                        / len(combo.end_board)) if combo.end_board else None,
                  "seconds": round(seconds, 1)}
        results.append(result)
        print(f"{n + 1}/{len(combos)} {'OK  ' if result['checked'] else 'part'} {line.progress}/{len(targets)} "
              f"steps ({line.skipped} skipped), {checked}/{args.shuffles} shuffles, field {len(on_field)}, "
              f"{seconds:.0f}s{'' if replays else ' NO REPLAY'}  {combo.title[:60]}", flush=True)
    pool.shutdown()
    passed = sum(r["checked"] for r in results)
    print(f"checked: {passed}/{len(results)} combos ({passed / max(1, len(results)):.0%}), mean field "
          f"{sum(r['field'] for r in results) / max(1, len(results)):.2f} (checked: "
          f"{sum(r['field'] for r in results if r['checked']) / max(1, passed):.2f}), search "
          f"{sum(r['seconds'] for r in results):.0f}s ({sum(r['seconds'] for r in results) / max(1, len(results)):.1f}s/combo)")
    print(f"checked in some shuffles: {sum(r['checked_shuffles'] > 0 for r in results)} combos, "
          f"{sum(r['checked_shuffles'] for r in results)}/{len(results) * args.shuffles} shuffles")
    print(f"steps with a location: {located[True]}, only named: {located[False]}")
    print("verb coverage (used/total):", {v: f"{verbs_used[v]}/{verbs[v]}" for v, _ in verbs.most_common(20)})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")


if __name__ == "__main__":
    main()
