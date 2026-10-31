"""Seeds for the opening book: what a starting hand should reach, from every source we have.

One format (`Seed`): starting cards (the rest of the hand is filler), a plan for the search
(`combos.Combo`: steps and/or end board, either may be empty), the hand traps the seed plays
through or into (`vs`), the cards a logged hand trap answered (`chokes`), going first or not.

Sources: community combos (masterduelmeta, `combos.load`), the Branded Discord sheet (CSV: starting
hand text like "Aluber + 1 Discard", end board like "Mirrorjade + Banishment + Lost", notes like
"plays into droll"), other sheets (Card/Board CSV, "In Hand" TSV), Duelingbook logs
(`db_logs`), and starter x extender hands no source covers. Nicknames resolve through `ALIASES`
and unique parts of card names in the deck's pool. Hand-trap rules (which card a hand trap waits
for) come from the counters data plus guide rules.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import aliases as _aliases
from . import combos as cb
from . import db_logs
from .steps import step_entries

HAND_SIZE = 5
# Player nicknames -> card names (sheets and guides use them; the card database does not).
ALIASES = {  # nicknames not in aliases.py (hand traps, Elfnote and Fiendsmith cards); aliases.ALIASES is merged below
    "winda": "El Shaddoll Winda", "schism": "Shaddoll Schism", "king": "Rindbrumm the Striking Dragon",
    "branded king": "Rindbrumm the Striking Dragon", "white": "Fallen of the White Dragon",
    "dead mirrorjade": "Mirrorjade the Iceblade Dragon", "dead mirrorajde": "Mirrorjade the Iceblade Dragon",
    "drago": "Red-Eyes Dark Dragoon", "requiem lock": "Fiendsmith's Requiem", "ash": "Ash Blossom & Joyous Spring",
    "imperm": "Infinite Impermanence", "veiler": "Effect Veiler", "droll": "Droll & Lock Bird",
    "belle": "Ghost Belle & Haunted Mansion", "ogre": "Ghost Ogre & Snow Rabbit",
    "nibiru": "Nibiru, the Primal Being", "maxx c": 'Maxx "C"', "fuwa": "Mulcharmy Fuwalos",
    "fuwalos": "Mulcharmy Fuwalos", "crow": "D.D. Crow", "mourner": "Ghost Mourner & Moonlit Chill",
    "harmonia": "Fydraulis Harmonia", "engraver": "Fiendsmith Engraver", "tract": "Fiendsmith's Tract",
    "levi": "Fabled Levilazebul", "levilazebul": "Fabled Levilazebul", "desirae": "Fiendsmith's Desirae",
    "lacrima": "Fiendsmith's Lacrima", "agnumday": "Fiendsmith's Agnumday", "sequence": "Fiendsmith's Sequence",
}
ALIASES.update(_aliases.ALIASES)  # the nickname table of aliases.py (checked against the card database)
INTERRUPTERS = {
    "Ash Blossom & Joyous Spring", "Infinite Impermanence", "Effect Veiler", "Droll & Lock Bird",
    "Ghost Belle & Haunted Mansion", "Ghost Ogre & Snow Rabbit", "Nibiru, the Primal Being", "D.D. Crow",
    "Ghost Mourner & Moonlit Chill", "Bystial Magnamhut", "Bystial Saronir", "Bystial Druiswurm", "Bystial Baldrake",
    "Dominus Impulse", "Dominus Purge", "Fydraulis Harmonia", "Called by the Grave", "Mulcharmy Fuwalos",
}
HAND_TRAP_WORDS = ("ash", "imperm", "veiler", "droll", "belle", "ogre", "nibiru", "maxx c", "fuwa", "crow", "mourner")
DISCARDS = re.compile(r"^(\d+)\s+discards?\b", re.IGNORECASE)
WHERE = re.compile(r"\s+in\s+(the\s+)?(hand|gy|grave|banish\w*)\.?$", re.IGNORECASE)
VERSION = re.compile(r"\s*\((version|example|no|otk|winda|dragoon)[^)]*\)\s*$", re.IGNORECASE)


@dataclass
class Seed:
    id: str
    title: str
    source: str  # "community", "sheet", "replay", "grid"
    hand: list[int]  # starting cards; the rest of the hand is filler
    combo: cb.Combo  # steps and end board for the search (steps may be empty)
    vs: list[int] = field(default_factory=list)  # hand traps the seed is said to play through / into
    first: bool = True  # a going-first line (vs OTKs and going-second lines)
    chokes: dict = field(default_factory=dict)  # hand trap -> cards it answered in this seed's log
    author: str = ""  # who made the line (credited in docs/data-sources.md)
    origin: str = ""  # where it comes from (masterduelmeta, the sheet's or log's file name)


def resolver(names: dict[str, int], pool: set[int], card_name: dict[int, str]):
    """Card name or nickname -> passcode: aliases, exact names, then a unique card of the deck's pool
    whose name contains the text."""
    def resolve(text: str) -> int | None:
        text = VERSION.sub("", WHERE.sub("", text.strip().rstrip("."))).strip()
        key = text.lower()
        full = ALIASES.get(key, text)
        code = names.get(cb.normalize(full), names.get(cb.loose(full)))
        if code is not None:
            return code
        hits = sorted({c for c in pool if key and key in card_name.get(c, "").lower()})
        return hits[0] if len(hits) == 1 else None
    return resolve


def mentioned_hand_traps(text: str, resolve) -> list[int]:
    lower = (text or "").lower()
    found = []
    for word in HAND_TRAP_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lower) and (code := resolve(word)) is not None:
            found.append(code)
    return found


def sheet_seeds(path: Path, deck_type: str, resolve) -> list[Seed]:
    """Rows of a player sheet: number, starting hand ("Aluber + 1 Discard"), end board, replay, notes.
    Generic parts ("any dark monster", "A Normal Summon") and unknown nicknames drop the row."""
    seeds = []
    author = ""
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            credit = next((re.sub(r"^\s*created by\s+", "", c, flags=re.IGNORECASE).strip() for c in row
                           if re.match(r"\s*created by\s", c, re.IGNORECASE)), None)
            author = credit or author
            if len(row) < 4 or not row[1].strip().isdigit() or not row[2].strip():
                continue
            hand, unknown = [], False
            for part in row[2].split("+"):
                part = part.strip()
                if not part or DISCARDS.match(part):
                    continue  # a discard is any card: the fillers
                code = resolve(part)
                if code is None:
                    unknown = True
                    break
                hand.append(code)
            if unknown or not hand:
                continue
            board = [code for part in row[3].split("+") if (code := resolve(part)) is not None]
            notes = row[5] if len(row) > 5 else ""
            combo = cb.Combo(f"sheet-{row[1].strip()}", f"{row[2].strip()} -> {row[3].strip()}", deck_type, 0, [], [],
                             notes=notes, end_board=board)
            seeds.append(Seed(combo.id, combo.title, "sheet", hand, combo, mentioned_hand_traps(notes, resolve),
                              first=going_first(row[2] + " " + notes), author=author, origin=path.name))
    return seeds


def board_parts(text: str, resolve) -> list[int]:
    """End-board text ("Caesar, Strelitzia, 2 Elves/Dragster, Rhapsodia, 4 cards in hand") -> cards:
    the first of alternatives, counts and notes ("cards in hand", "burn 1200", "setup") dropped."""
    cards = []
    for part in re.split(r"[,+]", text):
        part = re.sub(r"^\s*\d+\s+", "", part.strip()).split("/")[0].strip()
        if not part or re.search(r"\b(cards? in hand|burn|setup|draw|lock|otk)\b", part, re.IGNORECASE):
            continue
        if (code := resolve(part)) is not None:
            cards.append(code)
    return cards


def list_seeds(path: Path, deck_type: str, resolve, author: str = "") -> list[Seed]:
    """Other player sheets: a CSV with columns Card (starting hand, "Lucina + Engraver (Desirae)")
    and Board (end board), or a TSV whose second column reads "In Hand: Lucina + Discard vs Fuwalos"
    (hand only: the plan comes from a community seed with the same first card)."""
    seeds = []
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".tsv":
        rows = [line.split("\t") for line in text.splitlines()]
        for n, row in enumerate(rows):
            if len(row) < 2:
                continue
            body = re.sub(r"^In Hand:\s*", "", row[1].strip())
            hand_text, _, vs_text = body.partition(" vs ")
            parts = [p for p in hand_text.split("+") if not re.search(r"discard|\bany\b", p, re.IGNORECASE)]
            hand = [c for part in parts if (c := resolve(part)) is not None]
            if not hand or len(hand) < len(parts):
                continue
            vs = [c for part in re.split(r"[+,&]| and ", vs_text)
                  if (c := resolve(re.sub(r"^\s*(\d+|early|late)\s+", "", part.strip(), flags=re.IGNORECASE)))]
            combo = cb.Combo(f"{path.stem}-{n}", f"{row[0]}: {row[1]}", deck_type, 0, [], [])
            seeds.append(Seed(combo.id, combo.title, "sheet", hand, combo, vs, author=author, origin=path.name))
        return seeds
    for n, row in enumerate(csv.DictReader(text.splitlines())):
        hand_text = re.sub(r"\([^)]*\)", "", row.get("Card") or "")
        hand = [c for part in hand_text.split("+") if (c := resolve(part)) is not None]
        board = board_parts(row.get("Board") or "", resolve)
        if not hand or not board:
            continue
        combo = cb.Combo(f"{path.stem}-{n}", f"{row['Card']} -> {row['Board']}", deck_type, 0, [], [],
                         notes=row.get("Notes") or "", end_board=board)
        seeds.append(Seed(combo.id, combo.title, "sheet", hand, combo, mentioned_hand_traps(row.get("Notes"), resolve),
                          author=author, origin=path.name))
    return seeds


def borrow_plans(seeds: list[Seed]) -> None:
    """A seed with nothing to aim for (a hand-only sheet row) takes the plan of the first community
    seed that starts with the same card."""
    by_first = {}
    for s in seeds:
        if s.source == "community":
            by_first.setdefault(s.hand[0], s)
    for s in seeds:
        if not s.combo.steps and not s.combo.end_board and s.hand[0] in by_first:
            plan = by_first[s.hand[0]].combo
            s.combo = cb.Combo(s.id, s.title, plan.deck_type, 0, plan.starting, plan.steps, end_board=plan.end_board,
                               vs_hand_traps=plan.vs_hand_traps)


def replay_seed(path: Path, deck_type: str, resolve) -> Seed | None:
    """A Duelingbook log (db_logs.parse) as precise steps: each card movement a step whose card has
    to get there, each declared effect a step that uses the card. The opponent's hand traps played
    in a "vs" log become `vs`, and the cards they answered `chokes`."""
    log = db_logs.parse(path.read_text(encoding="utf-8"), INTERRUPTERS)
    moves = [(code, e.card, ACTION[e.kind]) for e in log.events if (code := resolve(e.card)) is not None]
    traps = {code: {c for name in answered if (c := resolve(name)) is not None}
             for trap, answered in log.traps.items() if (code := resolve(trap)) is not None}
    first: dict[int, str] = {}
    for code, _, action in moves:
        first.setdefault(code, action)
    # Discard fodder ("+ 2 discards": a hand card whose first move is to the GY) is no seed card:
    # any filler will do.
    hand = [c for name in log.hand if (c := resolve(name)) is not None and c not in traps
            and first.get(c) != ACTION["gy"]][:HAND_SIZE] or [c for name in log.hand[:1] if (c := resolve(name))]
    if not moves or not hand:
        return None
    combo = cb.Combo(f"replay-{path.stem}", log.title or path.stem, deck_type, 0, [cb.Entry(c, "", "") for c in hand],
                     [[cb.Entry(code, card, action)] for code, card, action in moves])
    return Seed(combo.id, combo.title, "replay", hand, combo, list(traps), chokes=traps, author=log_author(log.title),
                origin=path.name)


def log_author(title: str) -> str:
    """The player a log header credits: "... | GalacticJoey.", "users GalacticJoey", "posted by baguette"."""
    m = (re.search(r"posted by\s+([^\s,;)]+)", title) or re.search(r"users\s+([^\s,;(]+)", title)
         or re.search(r"\|\s*([A-Za-z0-9_]+)(?:\s*\(|\.|\s*$)", title))
    return m.group(1) if m else ""


# db_logs event kinds as action texts combo_check reads (expected_location).
ACTION = {"summon": "Special Summon", "place": "Place", "gy": "Send to GY", "hand": "Add to hand", "banish": "Banish",
          "use": ""}


GOING_SECOND = re.compile(r"\b(otk|ftk|going (2nd|second)|go(ing)? 2nd|turn 2|as (the )?(2nd|second) player|"
                          r"(2nd|second) turn|board ?break(er|ing)?)\b", re.IGNORECASE)


def going_first(text: str) -> bool:
    """A going-first line: its title or notes do not say OTK, going second, turn 2 or board
    breaking ("second effect" or "2nd copy" is no going-second line)."""
    return not GOING_SECOND.search(text or "")


def community_seeds(deck_type: str, names: dict[str, int]) -> list[Seed]:
    seeds = []
    for combo in sorted((c for c in cb.load(deck_type, names) if c.usable and c.steps), key=lambda c: -c.votes):
        hand = [e.code for e in combo.starting if e.code is not None]
        if not hand:
            continue
        seeds.append(Seed(combo.id, combo.title, "community", hand, combo, list(combo.vs_hand_traps),
                          going_first(combo.title + " " + combo.notes),
                          author=combo.author, origin="masterduelmeta"))
    return seeds


def grid_seeds(seeds: list[Seed], deck_main: list[int], limit: int) -> list[Seed]:
    """Starter x extender hands no seed covers: starters are the cards of one-card seeds,
    extenders the second card of two-card seeds; each gets the best seed of its starter as plan."""
    by_starter: dict[int, Seed] = {}
    extenders: Counter = Counter()
    covered = {tuple(sorted(s.hand)) for s in seeds}
    for s in seeds:
        if len(s.hand) == 1:
            by_starter.setdefault(s.hand[0], s)
        elif len(s.hand) == 2:
            extenders.update(s.hand)
    grid = []
    for starter, seed in by_starter.items():
        for extender, _ in extenders.most_common():
            if extender == starter or extender not in deck_main or tuple(sorted((starter, extender))) in covered:
                continue
            covered.add(tuple(sorted((starter, extender))))
            grid.append(Seed(f"grid-{starter}-{extender}", f"{seed.title} + extender {extender}", "grid",
                             [starter, extender], seed.combo, list(seed.vs), seed.first, author=seed.author,
                             origin=seed.origin))
    return grid[:limit]


def interruptions(deck_type: str, resolve, extra: dict[str, list[str]] | None = None) -> dict[int, set[int]]:
    """Hand trap -> choke-point cards: the counters data (cards with impact High/Medium, and the
    choke points of their entry) plus `extra` rules (hand trap name -> card names, from guides)."""
    rules: dict[int, set[int]] = {}
    for entry in cb.latest("counters"):
        if (entry.get("deckType") or {}).get("name") != deck_type:
            continue
        points = {resolve(c.get("name", "")) for c in entry.get("chokepoints") or [] if isinstance(c, dict)} - {None}
        for counter in entry.get("counterCards") or []:
            if counter.get("impact") in ("High", "Medium", None) and (code := resolve(counter["card"]["name"])):
                rules.setdefault(code, set()).update(points)
    for trap, cards in (extra or {}).items():
        if (code := resolve(trap)) is not None:
            rules.setdefault(code, set()).update(c for name in cards if (c := resolve(name)) is not None)
    return rules


def plan_key(seed: Seed) -> tuple:
    """Seeds with this key are the same search: the same starting cards and the same plan (every step
    in order, the end board). A borrowed plan is the same plan."""
    steps = tuple((e.code, e.action) for _, e in step_entries(seed.combo))
    return tuple(sorted(seed.hand)), steps, tuple(sorted(seed.combo.end_board))


def load(args, deck_type: str, names: dict[str, int], resolve, decklist, card_name: dict[int, str]):
    """All seeds of a deck (community combos, `args.sheet`, `args.replays`, `args.lists`, grid
    hands), ordered by `args.priority` titles, going first, then source; one per hand and plan.
    Returns the seeds, the hand-trap rules (hand trap -> choke points; only hand traps the bot can
    play on our turn) and the hand traps by how much they are feared."""
    sources = [s.strip() for s in args.sources.split(",")]
    seeds = community_seeds(deck_type, names)
    if args.sheet:
        seeds += sheet_seeds(args.sheet, deck_type, resolve)
    seeds += [s for path in args.replays if (s := replay_seed(path, deck_type, resolve)) is not None]
    if getattr(args, "rotk", None):  # Road of the King reports (rotk.py)
        from . import rotk
        for path in args.rotk:
            seeds += rotk.rotk_seeds(path, deck_type, resolve)
    if getattr(args, "pinned", None):  # pinned Discord replays with their posts (pinned.py)
        from . import pinned
        seeds += pinned.pinned_seeds(args.pinned, deck_type, resolve)
    for item in args.lists:  # "path" or "path::author"
        path, _, author = str(item).partition("::")
        seeds += list_seeds(Path(path), deck_type, resolve, author)
    borrow_plans(seeds)
    seeds += grid_seeds(seeds, decklist[0], args.grid) if "grid" in sources else []
    seeds = [s for s in seeds if s.source in sources]  # community seeds may still lend their plans
    # Every scenario is built with us going first: going-second lines (OTKs, board breakers) would be
    # searched as first-turn lines, so they stay out.
    seeds = [s for s in seeds if s.first]
    priority = lambda s: next((i for i, t in enumerate(args.priority) if t.lower() in s.title.lower()),  # noqa: E731
                              len(args.priority))
    # Priority titles, then going-first lines, then the sources in the given order, short hands first.
    order = -1 if getattr(args, "long_hands_first", False) else 1
    seeds.sort(key=lambda s: (priority(s), "defunct" in s.title.lower(), sources.index(s.source), order * len(s.hand)))
    seeds = [s for s in seeds if s.combo.steps or s.combo.end_board]  # nothing to aim for otherwise
    unique: dict[tuple, Seed] = {}  # one search per hand and plan; hand traps named by duplicates merge
    for s in seeds:
        key = plan_key(s)
        if key in unique:
            unique[key].vs = list(dict.fromkeys(unique[key].vs + s.vs))
            for trap, cards in s.chokes.items():
                unique[key].chokes.setdefault(trap, set()).update(cards)
        else:
            unique[key] = s
    seeds = list(unique.values())
    if args.max_seeds:
        seeds = seeds[:args.max_seeds]
    extra = json.loads(args.chokepoints.read_text(encoding="utf-8")) if args.chokepoints else None
    rules = interruptions(deck_type, resolve, extra)
    # Interruptions the bot can play on our turn (floodgates and draw-for-the-opponent cards are no
    # branch): ordered by how often seeds name them, then by their choke points.
    rules = {t: points for t, points in rules.items() if card_name.get(t) in INTERRUPTERS}
    named = Counter(t for s in seeds for t in s.vs if t in rules)
    feared = sorted(rules, key=lambda t: (-named[t], -len(rules[t])))
    return seeds, rules, feared
