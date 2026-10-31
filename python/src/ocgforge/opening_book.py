"""Opening book: the best first turn of one deck from many starting hands, with hand-trap branches.

    python -m ocgforge.opening_book --deck decks/branded.ydk --deck-type Branded \\
        --sheet .cache/branded-discord/sheet1.csv --checkpoint runs/x/latest --out runs/combo-book/branded-opening.jsonl

Seeds (what a hand should reach) come from the community combos (masterduelmeta: steps, end board,
the hand traps a line plays through), from a player sheet (CSV: starting hand as text like
"Aluber + 1 Discard", end board like "Mirrorjade + Banishment + Lost", notes like "plays into
droll") and from Duelingbook logs (card movements). Each seed becomes several starting hands: its
cards plus fillers, one random source per hand, and starter x extender hands that no seed covers.
The search (`combo_check.check`, beam search on duel clones) plays each hand; the seed's steps and
end board only guide it, it may find another line.

Interruptions: for each hand trap the deck fears (counters data: card, impact, choke points; plus
`--chokepoints` rules from guides), the hand gets a second search in which the opponent holds that
hand trap and fires it the first time we activate one of its choke-point cards (or at the first
chance, without choke points). The book keeps the best continuation after it.

The book is keyed by game state, not by starting hand: our hand, field, GY and banished cards
(cards no book line ever names count only as a number), the opponent's public cards, the phase
and the question. Transpositions share an entry. Each entry holds the answer of the best line
through it; a line's value is how often our agent fails to break its end board (board_break: the
agent goes second against the board, the bot defends it eagerly), then its size.

Report (`--test-hands`): random hands from the deck played from the book; the share that stays in
the book for 5, 10 and 20 of our answers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import board_break as bb
from . import combo_check as cc
from . import combos as cb
from . import rollout, worlds

HAND_SIZE = 5
# Player nicknames -> card names (sheets and guides use them; the card database does not).
ALIASES = {
    "brafu": "Branded Fusion", "foolish": "Foolish Burial", "dupe": "Fusion Duplication", "lost": "Branded Lost",
    "tragedy": "Despian Tragedy", "granguinol": "Granguignol the Dusk Dragon",
    "granguniol": "Granguignol the Dusk Dragon", "banishment": "Branded Banishment", "winda": "El Shaddoll Winda",
    "masq": "Masquerade the Blazing Dragon", "schism": "Shaddoll Schism", "ad lib": "Ad Libitum of Despia",
    "adlib": "Ad Libitum of Despia", "sarc": "Gold Sarcophagus", "kitt": "Tri-Brigade Springans Kitt",
    "king": "Rindbrumm the Striking Dragon", "branded king": "Rindbrumm the Striking Dragon",
    "lubellion": "Lubellion the Searing Dragon", "cart": "Blazing Cartesia, the Virtuous",
    "white": "Fallen of the White Dragon", "albaz": "Fallen of Albaz", "abaz": "Fallen of Albaz",
    "mirrorajde": "Mirrorjade the Iceblade Dragon", "dead mirrorjade": "Mirrorjade the Iceblade Dragon",
    "dead mirrorajde": "Mirrorjade the Iceblade Dragon", "drago": "Red-Eyes Dark Dragoon",
    "requiem lock": "Fiendsmith's Requiem", "furious": "Borreload Furious Dragon", "edge imp": "Edge Imp Chain",
    "high spirit": "Branded in High Spirits", "ash": "Ash Blossom & Joyous Spring",
    "imperm": "Infinite Impermanence", "veiler": "Effect Veiler", "droll": "Droll & Lock Bird",
    "belle": "Ghost Belle & Haunted Mansion", "ogre": "Ghost Ogre & Snow Rabbit", "nibiru": "Nibiru, the Primal Being",
    "maxx c": 'Maxx "C"', "fuwa": "Mulcharmy Fuwalos", "fuwalos": "Mulcharmy Fuwalos", "crow": "D.D. Crow",
    "mourner": "Ghost Mourner & Moonlit Chill", "harmonia": "Fydraulis Harmonia",
    "medius": "Medius the Pure", "terminus": "Prohibited Power Patron Portal - Terminus",
    "dragster": "F.A. Dawn Dragster", "crystal wing": "Crystal Wing Synchro Dragon",
    "engraver": "Fiendsmith Engraver", "tract": "Fiendsmith's Tract", "levi": "Fabled Levilazebul",
    "levilazebul": "Fabled Levilazebul", "desirae": "Fiendsmith's Desirae", "lacrima": "Fiendsmith's Lacrima",
    "agnumday": "Fiendsmith's Agnumday", "sequence": "Fiendsmith's Sequence",
}
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
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
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
                              first="otk" not in (row[2] + notes).lower()))
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


def list_seeds(path: Path, deck_type: str, resolve) -> list[Seed]:
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
            seeds.append(Seed(combo.id, combo.title, "sheet", hand, combo, vs))
        return seeds
    for n, row in enumerate(csv.DictReader(text.splitlines())):
        hand_text = re.sub(r"\([^)]*\)", "", row.get("Card") or "")
        hand = [c for part in hand_text.split("+") if (c := resolve(part)) is not None]
        board = board_parts(row.get("Board") or "", resolve)
        if not hand or not board:
            continue
        combo = cb.Combo(f"{path.stem}-{n}", f"{row['Card']} -> {row['Board']}", deck_type, 0, [], [],
                         notes=row.get("Notes") or "", end_board=board)
        seeds.append(Seed(combo.id, combo.title, "sheet", hand, combo, mentioned_hand_traps(row.get("Notes"), resolve)))
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


MOVE = re.compile(r'^(?P<what>[^:]+):\s+(?P<verb>\w+(?:\s\w+)?)\s+(banished\s+)?"(?P<card>[^"]+)"\s*(?P<rest>.*)$')


def replay_seed(path: Path, deck_type: str, resolve) -> Seed | None:
    """A Duelingbook log (card movements, "Special Summoned X from Deck to M-3") as precise steps:
    each movement is a step whose card has to get there. Setup lines (cards returned to the deck
    before the combo) are dropped, and so is a move undone right away (to the GY and back)."""
    moves, hand, added = [], [], set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip() if not raw.startswith("#") else ""
        m = MOVE.match(line)
        if not m:
            continue
        verb, card, rest = m["verb"].lower().split()[-1], m["card"], m["rest"].lower()
        code = resolve(card)
        if code is None:
            continue
        if "from hand" in rest and code not in added and code not in hand and not rest.endswith("of deck"):
            hand.append(code)
        if verb == "returned" and ("bottom of deck" in rest or "top of deck" in rest):
            if moves and moves[-1][0] == code:
                moves.pop()  # the move this undoes
            continue
        if verb == "summoned" or "to m-" in rest or "emz" in rest:
            action = "Special Summon"
        elif verb in ("placed", "activated", "set") or "to s-" in rest:
            action = "Place"
        elif verb == "sent" and "gy" in rest:
            action = "Send to GY"
        elif verb in ("added", "returned") and "to hand" in rest:
            action = "Add to hand"
            added.add(code)
        elif verb == "banished":
            action = "Banish"
        else:
            continue
        moves.append((code, card, action))
    if not moves:
        return None
    combo = cb.Combo(f"replay-{path.stem}", path.stem, deck_type, 0, [cb.Entry(c, "", "") for c in hand],
                     [[cb.Entry(code, card, action)] for code, card, action in moves])
    return Seed(combo.id, combo.title, "replay", hand, combo)


def community_seeds(deck_type: str, names: dict[str, int]) -> list[Seed]:
    seeds = []
    for combo in sorted((c for c in cb.load(deck_type, names) if c.usable and c.steps), key=lambda c: -c.votes):
        hand = [e.code for e in combo.starting if e.code is not None]
        if not hand:
            continue
        text = (combo.title + " " + combo.notes).lower()
        first = not any(w in text for w in ("otk", "going 2nd", "going second", "go 2nd", "2nd", "second"))
        seeds.append(Seed(combo.id, combo.title, "community", hand, combo, list(combo.vs_hand_traps), first))
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
                             [starter, extender], seed.combo, list(seed.vs), seed.first))
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


def hand_trap_bot(trap: int, chokepoints: set[int]):
    """Seat 1 with `trap` in hand: activates it the first time it is offered right after we
    activated a choke-point card (any card without choke points); for its follow-up questions it
    picks a choke-point card if offered, else the first answer; declines everything else."""
    def act(duel, history: list[str]) -> str:
        keys = duel.keys()
        offers = [k for k in keys if (ref := worlds.card_ref(k)) is not None and int(ref[1]) == trap
                  and k.startswith(("chain.", "idle."))]
        if offers:
            ours = next((worlds.card_ref(k) for k in reversed(history)
                         if k.startswith(("chain.", "idle.")) and worlds.card_ref(k) and worlds.card_ref(k)[2] == "0"),
                        None)
            if not chokepoints or (ours is not None and int(ours[1]) in chokepoints):
                return offers[0]
            return worlds.decline(keys)
        if fired(history, trap) and not ("chain.pass" in keys and all(k.startswith("chain.") for k in keys)):
            aimed = [k for k in keys if (ref := worlds.card_ref(k)) is not None and int(ref[1]) in chokepoints]
            return (aimed or keys)[0]
        return worlds.decline(keys)
    return act


def fired(history: list[str], trap: int) -> bool:
    return any((ref := worlds.card_ref(k)) is not None and int(ref[1]) == trap and k.startswith(("chain.", "idle."))
               and ref[2] == "1" for k in history)


def state_key(duel, relevant: set[int]) -> str:
    """The game state as the book sees it: our hand, field (with zone and position), GY and
    banished cards, the opponent's public cards, phase and turn, and the question (its kinds and
    the relevant cards it names). Cards no book line names count only as a number, so a filler card
    does not split states."""
    parts = []
    for c in duel.snapshot()["cards"]:
        loc = c["location"]
        if c["controller"] == 0 and loc in (worlds.LOCATION_HAND, worlds.LOCATION_GRAVE, worlds.LOCATION_REMOVED):
            parts.append((0, loc, c["code"] if c["code"] in relevant else 0, 0))
        elif c["controller"] == 0 and loc in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE):
            parts.append((0, loc, c["code"], c["sequence"] * 16 + c["position"]))
        elif c["controller"] == 1 and loc in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE, worlds.LOCATION_GRAVE,
                                              worlds.LOCATION_REMOVED) and c["public"]:
            parts.append((1, loc, c["code"], 0))
    question = sorted({(k.split(".")[0], worlds.card_ref(k)[1] if worlds.card_ref(k) and
                        int(worlds.card_ref(k)[1]) in relevant else "") for k in duel.keys()})
    raw = json.dumps([sorted(parts), duel.phase, duel.turn, duel.player, question])
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def walk(start, keys: list[str], relevant: set[int]) -> list[tuple[str, str, int]]:
    """(state, answer, depth) for each of our answers along a line replayed from its scenario."""
    duel = start()
    out, depth = [], 0
    for key in keys:
        if duel is None or duel.finished or key not in duel.keys():
            break
        if duel.player == 0:
            out.append((state_key(duel, relevant), worlds.canonical(key), depth))
            depth += 1
        duel.act(key)
    return out


def main(argv=None) -> None:
    from ._core import Duel
    from .cards import CardTable
    from .model import load_checkpoint

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--deck", type=Path, required=True)
    p.add_argument("--deck-type", required=True)
    p.add_argument("--sheet", type=Path, help="player sheet CSV (starting hand, end board, replay, notes)")
    p.add_argument("--replays", type=Path, nargs="*", default=[], help="Duelingbook logs (card movements)")
    p.add_argument("--lists", type=Path, nargs="*", default=[], help="other player sheets (Card/Board CSV, In Hand TSV)")
    p.add_argument("--chokepoints", type=Path, help="JSON {hand trap: [card names]} from guides")
    p.add_argument("--priority", nargs="*", default=[], help="seed titles (substrings) to search first")
    p.add_argument("--opponent", type=Path, default=rollout.REPO / "decks" / "salamangreat.ydk")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--hands", type=int, default=3, help="starting hands per seed (other fillers)")
    p.add_argument("--max-seeds", type=int, default=0, help="seeds to search at most (0: all)")
    p.add_argument("--grid", type=int, default=40, help="starter x extender hands at most")
    p.add_argument("--traps", type=int, default=3, help="interruption branches per hand (most feared hand traps)")
    p.add_argument("--first-only", action="store_true", help="only going-first seeds")
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--rate-tries", type=int, default=4, help="board_break tries per end board (0: field size only)")
    p.add_argument("--agent-deck", type=Path, default=rollout.REPO / ".cache" / "yugiohmeta" / "decks" / "snake_eye.ydk")
    p.add_argument("--test-hands", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="book (JSON lines: one state per line)")
    p.add_argument("--dry-run", action="store_true", help="list the seeds and searches, search nothing")
    args = p.parse_args(argv)
    torch.set_num_threads(args.torch_threads)

    table = CardTable.from_cdb()
    card_name = {c.code: c.name for c in table.cards}
    types = {c.code: c.type for c in table.cards}
    is_extra = lambda code: bool(types.get(code, 0) & cc.TYPE_EXTRA)  # noqa: E731
    center = {c.code for c in table.cards if cc.wants_center(c.text)}
    names = cb.name_index(table)
    decklist = worlds.read_decklist(args.deck)
    opponent = worlds.read_decklist(args.opponent)[0]
    pool = set(decklist[0]) | set(decklist[1])
    for combo in cb.load(args.deck_type, names):
        pool |= set(combo.cards()) | set(combo.end_board)
    resolve = resolver(names, pool, card_name)
    model = load_checkpoint(args.checkpoint, args.device, dropout=0.0)
    card_index = model.card_features.index_map()
    cache = rollout.REPO / ".cache"

    seeds = community_seeds(args.deck_type, names)
    if args.sheet:
        seeds += sheet_seeds(args.sheet, args.deck_type, resolve)
    seeds += [s for path in args.replays if (s := replay_seed(path, args.deck_type, resolve)) is not None]
    for path in args.lists:
        seeds += list_seeds(path, args.deck_type, resolve)
    borrow_plans(seeds)
    seeds += grid_seeds(seeds, decklist[0], args.grid)
    if args.first_only:
        seeds = [s for s in seeds if s.first]
    priority = lambda s: next((i for i, t in enumerate(args.priority) if t.lower() in s.title.lower()),  # noqa: E731
                              len(args.priority))
    order = {"replay": 0, "community": 1, "sheet": 2, "grid": 3}
    seeds.sort(key=lambda s: (priority(s), not s.first, order[s.source]))
    seeds = [s for s in seeds if s.combo.steps or s.combo.end_board]  # nothing to aim for otherwise
    unique: dict[tuple, Seed] = {}  # one search per hand and plan; hand traps named by duplicates merge
    for s in seeds:
        key = (tuple(sorted(s.hand)), len(s.combo.steps), tuple(s.combo.end_board))
        if key in unique:
            unique[key].vs = list(dict.fromkeys(unique[key].vs + s.vs))
        else:
            unique[key] = s
    seeds = list(unique.values())
    if args.max_seeds:
        seeds = seeds[:args.max_seeds]
    extra = json.loads(args.chokepoints.read_text(encoding="utf-8")) if args.chokepoints else None
    rules = interruptions(args.deck_type, resolve, extra)
    # Interruptions the bot can play on our turn (floodgates and draw-for-the-opponent cards are no
    # branch): ordered by how often seeds name them, then by their choke points.
    rules = {t: points for t, points in rules.items() if card_name.get(t) in INTERRUPTERS}
    named = Counter(t for s in seeds for t in s.vs if t in rules)
    feared = sorted(rules, key=lambda t: (-named[t], -len(rules[t])))
    print(f"{len(seeds)} seeds ({Counter(s.source for s in seeds)}), hand traps: "
          f"{[card_name.get(t) for t in feared[:8]]}", flush=True)

    def scenario(seed: Seed, variant: int, trap: int | None):
        fill = random.Random(f"{seed.id}-hand-{variant}")
        rng = random.Random(f"{seed.id}-{variant}")
        hand = list(seed.hand)
        rest = [c for c in decklist[0] if c not in hand]
        fill.shuffle(rest)
        hand += rest[:max(0, HAND_SIZE - len(hand))]
        script = cc.scenario_script(seed.combo, decklist, opponent, is_extra, rng, hand=hand,
                                    opponent_hand=[trap] if trap else [])
        seed_value = rng.getrandbits(32)

        def start():
            try:
                duel = Duel.world(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(args.deck),
                                  str(args.opponent), script, seed_value, 0)
                for _ in range(20):
                    if duel.finished or duel.player == 0:
                        break
                    duel.act(worlds.decline(duel.keys()))
            except Exception:
                return None
            return None if duel.finished else duel
        return script, seed_value, start

    def search(seed: Seed, variant: int, trap: int | None):
        began = time.perf_counter()
        script, seed_value, start = scenario(seed, variant, trap)
        bot = hand_trap_bot(trap, rules.get(trap, set())) if trap else None
        line, targets = cc.check(seed.combo, start, model, card_index, args.device, beam=args.beam, center=center,
                                 is_extra=is_extra, opponent=bot)
        duel = start()
        for key in line.keys if duel is not None else []:
            if duel.finished or key not in duel.keys():
                break
            duel.act(key)
        final = cc.end_board(duel) if duel is not None else []
        on_field = [c for c in final if c["location"] in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)]
        return {"seed": seed.id, "title": seed.title, "source": seed.source, "variant": variant,
                "trap": trap, "fired": fired(line.keys, trap) if trap else None, "line": line.keys,
                "steps": len(targets), "progress": line.progress, "skipped": line.skipped,
                "key_cards": line.key_cards, "book_end_board": seed.combo.end_board, "field": len(on_field),
                "end_board": final, "script": script, "duel_seed": seed_value, "first": seed.first,
                "seconds": round(time.perf_counter() - began, 1)}, start

    # Base lines first; each hand's interruption branches use the hand traps its seed names first,
    # then the most feared ones.
    jobs = []
    for seed in seeds:
        traps = list(dict.fromkeys([t for t in seed.vs if t in rules] + feared))[:args.traps]
        for variant in range(args.hands):
            jobs.append((seed, variant, None))
            jobs += [(seed, variant, t) for t in traps]
    print(f"{len(jobs)} searches", flush=True)
    if args.dry_run:
        for seed in seeds:
            print(f"  {seed.source:9s} first={int(seed.first)} hand={[card_name.get(c) for c in seed.hand]} "
                  f"steps={len(cc.step_targets(seed.combo))} board={[card_name.get(c) for c in seed.combo.end_board]} "
                  f"vs={[card_name.get(c) for c in seed.vs]}  {seed.title[:60]}")
        return
    pool_ex = ThreadPoolExecutor(max(1, args.workers))
    futures = [pool_ex.submit(search, *job) for job in jobs]
    lines = []
    for n, future in enumerate(futures):
        try:
            result, start = future.result()
        except Exception as error:  # one broken scenario must not stop the book
            print(f"{n + 1}/{len(futures)} error {str(error)[:120]}", flush=True)
            continue
        lines.append((result, start))
        print(f"{n + 1}/{len(futures)} {result['source']:9s} field {result['field']} "
              f"trap {card_name.get(result['trap'], '-')[:20]:20s} fired {result['fired']} {result['seconds']:.0f}s  "
              f"{result['title'][:50]}", flush=True)
    pool_ex.shutdown()

    # End boards rated by board_break: the agent goes second against each.
    agent_main, agent_extra = worlds.read_decklist(args.agent_deck)
    policy = rollout.ModelPolicy(model, args.device, sample=True, seed=args.seed)
    for result, start in lines:
        # A line with a hand trap fired cannot be rebuilt for board_break (the agent's seat holds other
        # cards): it ranks below the base lines on the states they share, by its field size.
        result["value"] = (rate(result, args, agent_main, agent_extra, policy, card_index, cache, Duel)
                           if result["trap"] is None else {"held": -1.0, "left": None, "tries": 0})
    relevant = set()
    for result, _ in lines:
        for key in result["line"]:
            if (ref := worlds.card_ref(key)) is not None and ref[2] == "0":
                relevant.add(int(ref[1]))
        relevant |= {c["code"] for c in result["end_board"]}
    book: dict[str, dict] = {}
    for result, start in lines:
        score = (result["value"]["held"], result["field"])
        for state, answer, depth in walk(start, result["line"], relevant):
            entry = book.get(state)
            if entry is None or tuple(entry["score"]) < score:
                book[state] = {"state": state, "answer": answer, "score": list(score), "depth": depth,
                               "seed": result["seed"], "trap": result["trap"], "hits": (entry or {}).get("hits", 0)}
            book[state]["hits"] += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(e) + "\n" for e in book.values()), encoding="utf-8")
    lines_out = args.out.with_name(args.out.stem + "-lines.jsonl")
    lines_out.write_text("".join(json.dumps(r) + "\n" for r, _ in lines), encoding="utf-8")
    meta = {"relevant": sorted(relevant)}
    args.out.with_name(args.out.stem + "-meta.json").write_text(json.dumps(meta), encoding="utf-8")
    report(lines, book, relevant, args, decklist, opponent, cache, Duel, card_name)


def rate(result: dict, args, agent_main, agent_extra, policy, card_index, cache, Duel) -> dict:
    """How often the agent fails to win against this end board (held), and the share of the board
    left after its turn (board_break, eager defender)."""
    if args.rate_tries <= 0:
        return {"held": result["field"] / 10, "left": None, "tries": 0}
    mine = list(result["line"])
    answers = None
    held, left, tries = 0, [], 0
    for t in range(args.rate_tries):
        script = bb.with_agent(result["script"], bb.agent_lines(agent_main, random.Random(f"{result['seed']}-{t}")),
                               agent_extra)

        def start(script=script):
            try:
                duel = Duel.world(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(args.deck),
                                  str(args.agent_deck), script, result["duel_seed"], 0)
                for _ in range(20):
                    if duel.finished or duel.player == 0:
                        break
                    duel.act(worlds.decline(duel.keys()))
            except Exception:
                return None
            return None if duel.finished else duel
        if answers is None:  # seat 0's answers of the line, as in its own scenario
            answers = own(result, args, cache, Duel, mine)
            if not answers:
                return {"held": 0.0, "left": None, "tries": 0}
        duel = bb.build_board(start, answers)
        if duel is None:
            continue
        before = bb.field(duel, 0)["cards"]
        try:
            bb.play_turn(duel, policy, card_index, "eager")
        except Exception:
            continue
        tries += 1
        won = duel.finished and duel.winner == 1
        held += not won
        left.append((bb.field(duel, 0)["cards"] if not duel.finished else 0) / max(1, before))
    return {"held": round(held / tries, 3) if tries else 0.0,
            "left": round(sum(left) / len(left), 3) if left else None, "tries": tries}


def own(result: dict, args, cache, Duel, keys: list[str]) -> list[str] | None:
    duel = Duel.world(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(args.deck), str(args.opponent),
                      result["script"], result["duel_seed"], 0)
    for _ in range(20):
        if duel.finished or duel.player == 0:
            break
        duel.act(worlds.decline(duel.keys()))
    mine = []
    for key in keys:
        if duel.finished or key not in duel.keys():
            return mine or None
        if duel.player == 0:
            mine.append(key)
        duel.act(key)
    return mine


def report(lines, book, relevant, args, decklist, opponent, cache, Duel, card_name) -> None:
    """Book size, branches per hand trap, end-board values, and how long random test hands stay in
    the book (our answers played from it before the first state it does not know)."""
    traps = Counter(card_name.get(r["trap"], str(r["trap"])) for r, _ in lines if r["trap"] and r["fired"])
    tried = Counter(card_name.get(r["trap"], str(r["trap"])) for r, _ in lines if r["trap"])
    values = [r["value"]["held"] for r, _ in lines if r["trap"] is None]
    rng = random.Random(args.seed)
    stays = []
    for t in range(args.test_hands):
        main = list(decklist[0])
        rng.shuffle(main)
        hand = main[:HAND_SIZE]
        combo = cb.Combo(f"test-{t}", "test", args.deck_type, 0, [], [])
        script = cc.scenario_script(combo, decklist, opponent, lambda c: False, random.Random(f"test-{t}"), hand=hand)
        try:
            duel = Duel.world(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(args.deck),
                              str(args.opponent), script, t, 0)
        except Exception:
            continue
        depth = 0
        for _ in range(400):
            if duel.finished or duel.turn > 1:
                break
            keys = duel.keys()
            if duel.player != 0:
                duel.act(worlds.decline(keys))
                continue
            entry = book.get(state_key(duel, relevant))
            answer = worlds.match(entry["answer"], keys) if entry else None
            if answer is None:
                break
            duel.act(answer)
            depth += 1
        stays.append(depth)
    n = max(1, len(stays))
    summary = {"states": len(book), "lines": len(lines), "base_lines": len(values),
               "mean_field": round(sum(r["field"] for r, _ in lines if r["trap"] is None) / max(1, len(values)), 2),
               "held_mean": round(sum(values) / max(1, len(values)), 3),
               "test_hands": len(stays), **{f"stay_{k}": round(sum(d >= k for d in stays) / n, 3) for k in (1, 5, 10, 20)},
               "branches_fired": dict(traps), "branches_tried": dict(tried)}
    print(json.dumps(summary), flush=True)
    args.out.with_name(args.out.stem + "-report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
