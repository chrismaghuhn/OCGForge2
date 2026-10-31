"""Road of the King metagame reports (roadoftheking.com, saved as text) as opening-book seeds.

A report has a "Metagame Breakdown", then one section per deck (a line with the deck's name). A
section may show openings, one line per step:

    An opening with Elfnote Lucina is as follow:            (or "For example, with A and B in hand:")
    Special Summon Elfnote Lucina from hand to center Main Monster Zone.
    Elfnote Lucina ② effect, add Elfnote Power Patron from deck to hand.
    Elfnote Power Patron + Elfnote Lucina → Elfnote Seraphim Strelitzia (to Main Monster Zone 1).
    Chain 2: Elfnote Regina ③ effect, add itself from graveyard to hand.
    Chain 1: Jet Synchron ① effect, add Starjunk Synchron from deck to hand.
    During opponent's Main Phase,
    When opponent activates a Monster Effect, Kewl Tune B2B ③ effect, Special Summon ...
    The end board consists of 7 disruptions:
    Baronne de Fleur negates 1 Card or Effect.               (7 lines, then commentary)

`parse` reads every opening (`Opening`: deck, starting hand, `Step`s with card, effect number,
what it does, chain grouping, own or opponent's turn, the card movements as `combos.Entry`; the
disruptions). `rotk_seeds` turns the openings of one deck into `seeds.Seed`s: our turn's steps (a
chain is one step), the cards on our field at the end of our turn as end board; the opponent's turn
and the disruptions go to the notes. Card names go through `resolve` (`seeds.resolver`), after
Road of the King's translations that differ from the database (`NAMES`).

Hook for seeds.load (the source "guide" also needs a place in its `order`, e.g. "guide": 1):
    seeds += [s for path in args.rotk for s in rotk.rotk_seeds(path, deck_type, resolve)]
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import combos as cb

AUTHOR = "Road of the King"
SOURCE = "guide"
# Road of the King's English names -> the database's (normalized keys).
NAMES = {
    "theorealized medius": "Theorealize Medius",
    "ars magna the finite and the infinite": "Ars Magna of Infinity and Finity",
    "ars magna the corruption and the purification": "Ars Magna of Purification and Corruption",
    "ars magna the division and the union": "Ars Magna of Unification and Separation",
    'ars magna "citrinitas"': 'Ars Magna - "Citrinitas"',
    "mediclius the extraordinary power patron": "Medicurius the Power Patron of Illusions",
    "perfect toon world": "Toon World the Perfect World",
    "dark rebellion xyz dragon, four heavenly dragons": "Dark Rebellion Xyz Dragon of the Four Heavenly Dragons",
}
MIN_STEPS = 3  # an opening without an intro line: at least this many steps in a row
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"
MARKER = re.compile(rf"\s([{CIRCLED}])\s(Pendulum\s)?effect,\s*")
VERBS = (r"(?:(?:Special|Normal|Tribute|Pendulum|Ritual|Fusion|Synchro|Xyz|Link) Summon"
         r"|send|add|place|set|banish|destroy|return|shuffle|draw|discard|increase|decrease|reduce|negate|equip|change|switch|target|attach|detach"
         r"|activate|look|reveal|excavate|declare|gain|inflict|apply|tribute|make)\b")
CLAUSE = re.compile(rf"(?:,\s+(?:then\s+)?|\s+then\s+|\s+and\s+)(?={VERBS})", re.IGNORECASE)
VERB = re.compile(rf"^(?P<verb>{VERBS})\s*(?P<obj>.*)$", re.IGNORECASE)
STOP = re.compile(r"\s+(?:from|to|into|in|as|on|until|by|with|instead|face-up|face-down)\b|\s*\(|\s*,|\s*$")
INTRO = re.compile(r"^(?:for example,\s+)?with (?P<hand>.+?) in hand:\s*$"
                   r"|^(?:an?\s+)?(?:combo\s+)?opening with (?P<hand2>.+?) is as follows?:\s*$", re.IGNORECASE)
CHAIN = re.compile(r"^Chain (\d+):\s*")
OPPONENT_TURN = re.compile(r"^During (?:the )?opponent['’]s\b[^,:]*[,:]?\s*(?P<rest>.*)$", re.IGNORECASE)
OWN_PHASE = re.compile(r"^During (?:the |your )?(?:End|Battle|Standby|Main) Phase( 1| 2)?,\s*", re.IGNORECASE)
END_BOARD = re.compile(r"^The end board consists of (\d+) disruptions?:", re.IGNORECASE)
LINE_VERB = re.compile(r"^(?:(?:Normal|Special|Tribute|Pendulum|Ritual|Fusion|Synchro|Xyz|Link) Summon|Activate|Set)\s")
CONDITION = re.compile(r"^(When|If|During|After|At|Once|While)\b")
GENERIC = re.compile(r"^(\d+|an?|any)\s|\bcard\b", re.IGNORECASE)


@dataclass
class Step:
    text: str  # the line as written
    card: str  # whose effect or action it is ("" for a summon line "A + B → C" or "Normal Summon X")
    code: int | None
    effect: int | None  # ① -> 1; None when no numbered effect is used
    pendulum: bool  # a Pendulum effect ("① Pendulum effect")
    does: str  # verb and object: what the step does ("add Elfnote Tinia from deck to hand")
    chain: int | None = None  # chain link ("Chain 2:")
    chain_group: int | None = None  # steps of one chain share it
    opponent_turn: bool = False
    trigger: str = ""  # "When opponent activates a Monster Effect"
    moves: list[cb.Entry] = field(default_factory=list)  # cards and where they go (combos action texts)


@dataclass
class Disruption:
    card: str
    code: int | None
    text: str  # "negates 1 Card or Effect."


@dataclass
class Opening:
    deck: str
    intro: str  # "An opening with Elfnote Lucina is as follow:"
    hand: list[int]  # starting cards named by the intro (generic parts are filler)
    hand_names: list[str]
    steps: list[Step] = field(default_factory=list)
    disruptions: list[Disruption] = field(default_factory=list)
    stated: int = 0  # "The end board consists of N disruptions"
    end_board: list[int] = field(default_factory=list)  # our field at the end of our turn
    unknown: list[str] = field(default_factory=list)  # card-like names `resolve` did not know
    played_from_hand: list[int] = field(default_factory=list)  # first seen leaving the hand
    seen: set = field(default_factory=set, repr=False)


def translated(resolve):
    """`resolve` after Road of the King's names (`NAMES`), with the text's stray spaces removed."""
    def wrapped(text: str) -> int | None:
        text = re.sub(r"\s+\.$", "", text.strip()).strip()
        if not text:
            return None
        return resolve(NAMES.get(cb.normalize(text), text))
    return wrapped


def _strip(text: str) -> str:
    """No parentheticals ("(to center Main Monster Zone)", "(banished)"), no trailing " ."."""
    return re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", text)).strip().rstrip(".").strip()


def object_card(obj: str, resolve) -> tuple[str, int | None]:
    """The card a clause's object starts with: the longest prefix ending before "from", "to", "(",
    ... that resolves ("Elfnote Lucina from hand to graveyard as cost" -> Elfnote Lucina)."""
    obj = re.sub(r"^(?:up to\s+)?(?:\d+\s+)?", "", obj.strip())
    ends = sorted({m.start() for m in STOP.finditer(obj)}, reverse=True)
    for end in ends:
        text = obj[:end].strip()
        if text and (code := resolve(text)) is not None:
            return text, code
    return (obj[:ends[-1]].strip() if ends else obj), None


def lead_card(text: str, resolve) -> tuple[str, int | None, str]:
    """A disruption line's card: the longest leading run of words that resolves."""
    words = text.split()
    for n in range(len(words), 0, -1):
        if (code := resolve(" ".join(words[:n]))) is not None:
            return " ".join(words[:n]), code, " ".join(words[n:])
    return "", None, text


class _Turn:
    """Card movements of one opening, and our field (multiset) while it is our turn."""

    def __init__(self):
        self.field: Counter = Counter()
        self.order: list[int] = []

    def enter(self, code):
        if code is not None:
            self.field[code] += 1
            self.order.append(code)

    def leave(self, code):
        if code is not None and self.field[code] > 0:
            self.field[code] -= 1

    def board(self) -> list[int]:
        left = Counter(+self.field)
        return [c for c in dict.fromkeys(self.order) if left[c] > 0]


def summon_clause(text: str, resolve, step: Step, turn: _Turn | None, opening: Opening):
    """ "A + B → C": A and B are materials (used), C is summoned."""
    mats, _, result = text.partition("→")
    for m in mats.split(" + "):
        name = _strip(m)
        code = resolve(name)
        if code is None:
            opening.unknown.append(name)
        elif code not in opening.seen and "from hand" in m.lower():
            opening.played_from_hand.append(code)
        opening.seen.add(code)
        step.moves.append(cb.Entry(code, name, f"Material for {_strip(result)}"))
        if turn:
            turn.leave(code)
    name = _strip(result)
    code = resolve(name)
    if code is None:
        opening.unknown.append(name)
    opening.seen.add(code)
    step.moves.append(cb.Entry(code, name, "Summon"))
    if turn:
        turn.enter(code)


def verb_clause(text: str, resolve, step: Step, turn: _Turn | None, opening: Opening):
    m = VERB.match(text.strip())
    if not m:
        return
    verb, obj = m["verb"].lower(), m["obj"]
    lower = obj.lower()
    if re.match(r"^(both\s+)?(itself|this card)\b", lower):
        name, code = step.card, step.code
    else:
        name, code = object_card(obj, resolve)
        if code is None and " and " in obj:  # "Special Summon A and B from Extra Deck"
            parts = obj.split(" and ")
            if all(object_card(p, resolve)[1] is not None for p in parts):
                tail = re.search(r"\s(from|to|in)\s.*$", parts[-1])
                for part in parts:
                    verb_clause(f"{m['verb']} {object_card(part, resolve)[0]}{tail[0] if tail else ''}", resolve,
                                step, turn, opening)
                return
        if code is None:
            if name and name[0].isupper() and not GENERIC.search(name) and verb not in ("increase", "decrease",
                                                                                         "change", "negate"):
                opening.unknown.append(name)
            return
    from_field = bool(re.search(r"\bfrom (the )?field\b|from spell & trap zone", lower))
    on_field = turn is not None and turn.field[code] > 0
    action, enters, leaves = None, False, from_field
    if verb.endswith("summon"):
        action, enters = ("Normal Summon" if verb.startswith("normal") else "Special Summon"), "to hand" not in lower
    elif verb == "send":
        action = "Send to GY" if "graveyard" in lower or "gy" in lower else None
    elif verb == "add":
        action = "Add to hand"
    elif verb in ("place", "set"):
        action, enters = verb.capitalize(), True
    elif verb == "equip":
        action, enters = "Equip", True
    elif verb == "banish":
        action, leaves = "Banish", from_field or ("from" not in lower and on_field)
    elif verb in ("destroy", "tribute"):
        leaves = on_field
    elif verb in ("return", "shuffle"):
        leaves = from_field or (on_field and ("extra deck" in lower or "itself" in lower))
        action = "Add to hand" if "to the hand" in lower or "to hand" in lower else None
    elif verb == "activate" and "pendulum scale" in lower:
        action, enters = "Pendulum Scale", True
    elif verb == "activate":
        action, enters = "Activate", True
    if code not in opening.seen and ("from hand" in lower or verb == "normal summon"
                                     or (verb == "activate" and "from" not in lower)):
        opening.played_from_hand.append(code)
    opening.seen.add(code)
    if action:
        step.moves.append(cb.Entry(code, name, action))
    if turn:
        if leaves:
            turn.leave(code)
        if enters:
            turn.enter(code)


def clauses(text: str, resolve) -> list[str]:
    """A step's text cut into clauses: verb clauses and a trailing summon "A + B → C" (whose first
    material starts after the last ", " that leaves it a card name: "Puzzlomino, the Drop-n-Deleter")."""
    summon = ""
    if "→" in text:
        left = text[:text.index("→")]
        plus = left.find(" + ")
        plus = plus if plus >= 0 else len(left)
        cuts = [m.start() for m in re.finditer(", ", left[:plus])]
        cut = -1
        if cuts and resolve(_strip(left[:plus])) is None:
            hits = [c for c in reversed(cuts) if resolve(_strip(left[c + 2:plus])) is not None]
            # no card after any ", ": cut at the last one when a clause comes first
            cut = hits[0] if hits else (cuts[-1] if VERB.match(left) or CONDITION.match(left) else -1)
        text, summon = (text[:cut], text[cut + 2:]) if cut >= 0 else ("", text)
    parts = [p for p in CLAUSE.split(text) if p.strip()] if text else []
    return parts + ([summon] if summon else [])


def is_step(line: str) -> bool:
    return bool(MARKER.search(line) or "→" in line or CHAIN.match(line) or OPPONENT_TURN.match(line)
                or OWN_PHASE.match(line) or LINE_VERB.match(line))


def parse_step(line: str, resolve, opponent: bool) -> Step:
    body = OWN_PHASE.sub("", line)
    chain = None
    if m := CHAIN.match(body):
        chain, body = int(m[1]), body[m.end():]
    step = Step(line, "", None, None, False, body, chain=chain, opponent_turn=opponent)
    if m := MARKER.search(body):
        pieces = body[:m.start()].split(", ")
        for i in range(len(pieces)):
            name = ", ".join(pieces[i:])
            if (code := resolve(name)) is not None:
                step.card, step.code, step.trigger = name, code, ", ".join(pieces[:i])
                break
        else:
            step.card, step.trigger = pieces[-1], ", ".join(pieces[:-1])
        step.effect, step.pendulum = CIRCLED.index(m[1]) + 1, bool(m[2])
        step.does = body[m.end():].strip().rstrip(".").strip()
        step.moves.append(cb.Entry(step.code, step.card, f"Activate {m[1]} {'Pendulum ' if m[2] else ''}effect"))
    else:
        pieces = body.split(", ")
        if len(pieces) > 1 and CONDITION.match(pieces[0]):
            step.trigger, step.does = pieces[0], ", ".join(pieces[1:]).strip().rstrip(".").strip()
    return step


def parse(text: str, resolve) -> list[Opening]:
    """Every opening of a report, in order."""
    resolve = translated(resolve)
    lines = [re.sub(r"\s+", " ", raw).strip() for raw in text.splitlines()]
    lines = [line for line in lines if line]
    start = next((i for i, line in enumerate(lines) if line == "Metagame Breakdown"), -1) + 1
    while start < len(lines) and re.match(r"^\d+ ", lines[start]):
        start += 1
    openings: list[Opening] = []
    deck = ""
    current: Opening | None = None
    turn: _Turn | None = None
    opponent = False
    group, last_chain = 0, None
    pending = 0  # disruption lines still to read

    def close():
        nonlocal current
        if current is not None and turn is not None and (current.intro or len(current.steps) >= MIN_STEPS):
            if not current.end_board:
                current.end_board = turn.board()
            if not current.hand:  # no intro, or it names no card: what the line plays from the hand
                named = {e.code: e.name for s in current.steps for e in s.moves}
                current.hand = [c for c in dict.fromkeys(current.played_from_hand) if c is not None]
                current.hand_names = [named.get(c, "") for c in current.hand]
            openings.append(current)
        current = None

    for line in lines[start:]:
        name, code, rest = lead_card(line, resolve) if pending else ("", None, "")
        if pending and code is None:
            pending = 0
            close()  # fewer disruption lines than stated (a card counted for each use)
        if pending:
            pending -= 1
            current.disruptions.append(Disruption(name, code, rest or line))
            if not pending:
                close()
            continue
        if is_header(line):
            close()
            if line in ("Conclusion", "References"):
                break
            deck = line
            continue
        if m := INTRO.match(line):
            close()
            hand, hand_names = intro_hand(m["hand"] or m["hand2"], resolve)
            current, turn, opponent, last_chain = Opening(deck, line, hand, hand_names), _Turn(), False, None
            continue
        if current is None:
            if not deck or not is_step(line) or OPPONENT_TURN.match(line):
                continue
            # an opening without an intro line: its hand is what it plays from the hand
            current, turn, opponent, last_chain = Opening(deck, "", [], []), _Turn(), False, None
        if m := END_BOARD.match(line):
            current.stated = pending = int(m[1])
            current.end_board = turn.board() if not opponent else current.end_board
            if not pending:
                close()
            continue
        if not is_step(line):
            close()  # commentary: the opening is over
            continue
        if m := OPPONENT_TURN.match(line):
            if not opponent:
                current.end_board = turn.board()
            opponent = True
            if not m["rest"]:
                continue
            line = m["rest"]
        step = parse_step(line, resolve, opponent)
        if step.chain is not None:
            if last_chain is None or step.chain >= last_chain:
                group += 1
            step.chain_group = group
        last_chain = step.chain
        own = None if opponent else turn
        for clause in clauses(step.does.strip().rstrip(".").strip(), resolve):
            if "→" in clause:
                summon_clause(clause, resolve, step, own, current)
            else:
                verb_clause(clause, resolve, step, own, current)
        current.steps.append(step)
    close()
    return openings


def is_header(line: str) -> bool:
    """A deck section's title: a few words, no sentence."""
    return (len(line.split()) <= 6 and not re.search(r"[.:,;!?)\]]$", line) and "→" not in line
            and not MARKER.search(line) and not line[0].isdigit() and not INTRO.match(line))


def intro_hand(text: str, resolve) -> tuple[list[int], list[str]]:
    """ "Prohibited Power Patron Portal – Terminus and Elfnote Lucina", "Elfnote Lucina + 1 Monster
    Card": the named cards (an "and" inside a card name stays)."""
    codes, names = [], []
    for part in text.split(" + "):
        words = part.split(" and ")
        i = 0
        while i < len(words):
            for j in range(len(words), i, -1):
                name = " and ".join(words[i:j]).strip()
                if (code := resolve(name)) is not None:
                    codes.append(code)
                    names.append(name)
                    i = j
                    break
            else:
                i += 1  # generic ("1 Monster Card") or unknown: filler
    return codes, names


def deck_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def to_seed(opening: Opening, n: int, slug: str, title: str, deck_type: str):
    from .seeds import Seed  # here: seeds may import this module

    steps: list[list[cb.Entry]] = []
    last_group = None
    for step in opening.steps:
        if step.opponent_turn or not step.moves:
            continue
        if step.chain_group is not None and step.chain_group == last_group:
            steps[-1].extend(step.moves)  # one chain: its links in any order
        else:
            steps.append(list(step.moves))
        last_group = step.chain_group
    opp = [s.text for s in opening.steps if s.opponent_turn]
    notes = []
    if opp:
        notes.append("Opponent's turn: " + " / ".join(opp))
    if opening.disruptions:
        notes.append(f"End board ({opening.stated} disruptions): "
                     + " / ".join(f"{d.card} {d.text}".strip() for d in opening.disruptions))
    seed_id = f"rotk-{slug}-{deck_key(opening.deck)}-{n}"
    full = f"{title} | {opening.deck} | {opening.intro.rstrip(':') or 'opening'}"
    hand = list(dict.fromkeys(opening.hand))
    starting = [cb.Entry(c, name, "") for c, name in zip(opening.hand, opening.hand_names)]
    combo = cb.Combo(seed_id, full, deck_type, 0, starting, steps, notes="\n".join(notes), end_board=list(opening.end_board), author=AUTHOR)
    return Seed(seed_id, full, SOURCE, hand, combo, [], True, author=AUTHOR, origin=slug)


def rotk_seeds(path: Path, deck_type: str, resolve) -> list:
    """The openings of `deck_type`'s section (the section title matches it, ignoring case) of a
    saved report, as seeds; openings without a known starting card are left out."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    title = next((line.strip() for line in text.splitlines() if line.strip()), path.stem)
    seeds = []
    for n, opening in enumerate(o for o in parse(text, resolve) if o.deck.casefold() == deck_type.casefold()):
        if opening.hand:
            seeds.append(to_seed(opening, n, path.stem, title, deck_type))
    return seeds
