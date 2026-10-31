"""Turn-1 expert iteration: search good first turns for many random hands, for the net to imitate.

    python -m ocgforge.turn1_search --deck decks/elfnote.ydk --deck-type Elfnote \\
        --checkpoint runs/student-base2/latest --hands 250 --out runs/turn1/elfnote-t1.jsonl

The opening book (`opening_book`) only knows community starting hands. Here every hand is an
honest random 5 cards of the deck, we go first, and a beam search over our turn-1 answers (on duel
clones) looks for the line the net itself rates best:
- Score: the net's value head at the end of our turn 1 (the opponent's first decision of turn 2,
  negated; a duel won or lost in turn 1 scores +-1). Not community-plan progress. During the
  search, open lines rank by the value at their current decision (ours) plus `alpha` times the
  log-probability of the answers they took. Positions and monster zones are fixed by
  `answer_rules.forced`, the opponent declines (or is the hand-trap bot of a branch).
- Honesty: a clone carries the real deck order, so a line that draws, excavates or mills may be
  chosen for what the deck happens to hold. Each hand is therefore played in `--orders` K scenarios
  (the same hand, other deck orders and opponent hands). The search runs in scenario 0; its best
  `--candidates` finished lines and the net's own greedy line are then replayed in all K scenarios
  (our answers only, matched with `worlds.match`; from the first answer that is not offered on, the
  net plays greedily) and the line with the best MEAN end-of-turn value is kept. A choice made
  before the deck shows a card is thus judged over K deck orders; answers after it react to cards
  that are visible by then. Each kept line records `reveal_at`, our first answer after which a card
  we did not name left the deck (None: the line never looks into the deck).
- Hand traps: for each hand, one interruption branch (the deck's hand traps from the counters data,
  in turn by hand): the opponent holds the trap and fires it at the first chance (`opening_book.
  hand_trap_bot`). The branch replays the kept line until the trap fired, then searches on from
  there, so the line never plays around a trap it could not know of, and its states before the trap
  are the base line's (the book gets no conflicting answers there). Unfired branches are dropped.
- Output: `<out>-lines.jsonl` in the opening book's line format (`seed` = hand id, `trap`, `line`,
  `field`, `end_board`, `script`, `duel_seed`, ...) plus a `search` record; after the search, the
  book `<out>` and `<out>-meta.json` are built by `opening_book.build_book`, so `book_imitation` and
  `teach --book` read it unchanged. Line values for the book stay unrated (`value.held` None): one
  line per hand and branch, so lines only meet at the shared prefix of a hand's base and trap line.
- `--min-gain G`: a second book `<out stem>-gain.jsonl` holds only the lines whose mean value
  beats the net's own line (greedy over the same K scenarios; in a branch, its greedy continuation)
  by more than G. Where the net's own line was kept, imitation teaches nothing new.
- Resumable: every finished hand is appended to `<out>-hands.jsonl` (stats) right after its lines;
  `--resume` skips those hands (and drops lines of an unfinished hand).
- Net calls of all search threads go through one `Evaluator`, which batches them on the GPU.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import combo_check as cc
from . import combos as cb
from . import opening_book as ob
from . import rollout, worlds
from .answer_rules import deck_cards, forced
from .seeds import HAND_SIZE, INTERRUPTERS

FIELD = (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)
END_TURN = ("idle.7", "battle.3")  # to the End Phase (board_break.END_TURN)
PASSIVE = ("chain.pass", "yes_no.", "effect_yes_no.")  # never "the same answer again" (combo_check.PASSIVE)
MAX_AUTO = 400  # answers in a row without a real choice of ours (a looping bot must not hang a search)
# Hand traps for the branches, first come first: the cards that stop a combo at once before
# Fuwalos (which only lets the opponent draw; opening_book's ordering picked it far too often).
TRAP_ORDER = ("Ash Blossom & Joyous Spring", "Infinite Impermanence", "Effect Veiler", "Droll & Lock Bird",
              "Ghost Ogre & Snow Rabbit", "Ghost Belle & Haunted Mansion", "Ghost Mourner & Moonlit Chill",
              "D.D. Crow", "Bystial Magnamhut", "Dominus Impulse", "Nibiru, the Primal Being", "Mulcharmy Fuwalos")


class Evaluator:
    """Batches net calls of many threads: `evaluate(rows)` blocks until a worker thread has run
    `fn` on its rows together with whatever other threads asked for meanwhile (up to `max_rows`).
    `fn(rows)` returns one (log-probabilities, value of the acting player) per row."""

    def __init__(self, fn, max_rows: int = 512, wait: float = 0.002):
        self.fn, self.max_rows, self.wait = fn, max_rows, wait
        self.calls = self.rows = 0
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def __call__(self, rows: list) -> list[tuple[list[float], float]]:
        if not rows:
            return []
        future: Future = Future()
        self._queue.put((rows, future))
        return future.result()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            jobs, n, stop = [item], len(item[0]), False
            deadline = time.perf_counter() + self.wait
            while n < self.max_rows:
                try:
                    more = self._queue.get(timeout=max(0.0, deadline - time.perf_counter()))
                except queue.Empty:
                    break
                if more is None:
                    stop = True
                    break
                jobs.append(more)
                n += len(more[0])
            try:
                out = self.fn([r for rows, _ in jobs for r in rows])
            except Exception as error:  # every waiting thread gets the error, the worker lives on
                for _, future in jobs:
                    future.set_exception(error)
            else:
                i = 0
                for rows, future in jobs:
                    future.set_result(out[i:i + len(rows)])
                    i += len(rows)
            self.calls += 1
            self.rows += n
            if stop:
                return


def model_fn(model, device: str, chunk: int = 256):
    """`Evaluator` function of a model: log-softmax of the answer scores and the value head."""
    import torch

    from . import features

    @torch.no_grad()
    def fn(rows: list[dict]) -> list[tuple[list[float], float]]:
        model.eval()
        out = []
        for i in range(0, len(rows), chunk):
            part = rows[i:i + chunk]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                scores, values = model(features.collate(part, device))
            logprobs = torch.log_softmax(scores.float(), dim=-1).cpu()
            values = values.float().cpu().tolist()
            out += [(logprobs[j, :len(r["candidate_kind"])].tolist(), values[j]) for j, r in enumerate(part)]
        return out
    return fn


@dataclass
class Net:
    """The net as the search asks it: answers and value of a duel's pending decision."""
    evaluate: object  # Evaluator (or any rows -> [(logprobs, value)])
    card_index: object

    def ask(self, duels: list) -> list[tuple[list[float], float]]:
        """(log-probabilities of the answers, value for seat 0) of each duel's pending decision."""
        out = self.evaluate([d.encode(self.card_index) for d in duels])
        return [(lp, v if d.player == 0 else -v) for d, (lp, v) in zip(duels, out)]

    def terminal(self, duels: list) -> list[float]:
        """Seat 0's value at the end of turn 1: +-1 (0 for a draw) for a finished duel, else the
        net's value at the pending decision (the opponent's, negated)."""
        values = [None if not d.finished else 0.0 if d.winner == 2 else 1.0 if d.winner == 0 else -1.0 for d in duels]
        open_ = [d for d, v in zip(duels, values) if v is None]
        asked = iter(self.ask(open_))
        return [v if v is not None else next(asked)[1] for v in values]


@dataclass
class Ctx:
    """What every line of one scenario shares: seat 1's answers (None: decline everything) and
    the rules of `forced` (cards that want the center zone, Extra Deck cards)."""
    opponent: object = None
    center: set = field(default_factory=set)
    is_extra: object = lambda code: False
    max_choices: int = 120  # real choices of ours per line at most


def over(duel) -> bool:
    """Our turn 1 is over: the duel ended, or turn 2 began."""
    return duel.finished or duel.turn > 1


def settle(duel, keys: list[str], ours: list[str], ctx: Ctx) -> None:
    """Plays every answer that is no real choice of ours (seat 1's, our single answers, positions
    and zones `forced` fixes) until our next real choice or the end of turn 1; appended to `keys`
    (all answers) and, ours, to `ours`."""
    for _ in range(MAX_AUTO):
        if over(duel):
            return
        options = duel.keys()
        if duel.player != 0:
            key = ctx.opponent(duel, keys) if ctx.opponent else worlds.decline(options)
        elif len(options) == 1:
            key = options[0]
        else:
            key = forced(options, keys, False, ctx.center, ctx.is_extra)
            if key is None:
                return
        mine = duel.player == 0
        duel.act(key)
        keys.append(key)
        if mine:
            ours.append(key)
    raise RuntimeError("no real choice of ours in MAX_AUTO answers")


def identity(duel, keys: list[str]) -> tuple:
    """Lines in the same position at the same question are one line: our cards (with zone and
    position), the opponent's cards as we see them, life points, phase, the question, and our
    yes/no and option answers (the snapshot shows no levels or ATK changes)."""
    snap = duel.snapshot()
    cards = tuple(sorted((c["controller"], c["location"], c["code"] if c["controller"] == 0 or worlds.visible(c, 0)
                          else 0, c["sequence"] if c["location"] in FIELD else 0,
                          c["position"] if c["location"] in FIELD else 0) for c in snap["cards"]
                         if not (c["controller"] == 1 and c["location"] in (worlds.LOCATION_DECK, worlds.LOCATION_HAND))))
    choices = tuple(k for k in keys if k.startswith(("yes_no.", "effect_yes_no.", "option.")))
    return cards, tuple(snap["life_points"]), duel.phase, tuple(duel.keys()), choices


def board(duel) -> int:
    return hash(tuple(sorted(cc.board(duel).items())))


def field_cards(duel) -> int:
    if duel.finished:
        return 0
    return sum(c["controller"] == 0 and c["location"] in FIELD for c in duel.snapshot()["cards"])


@dataclass
class Node:
    keys: list[str]  # every answer from the scenario start, both seats
    ours: list[str]  # our answers among them, in order
    duel: object = None
    logp: float = 0.0  # log-probabilities of our real choices
    value: float = 0.0  # the net's value (ours) at the pending decision; at the end: end-of-turn value
    tried: frozenset = frozenset()  # (board, answer) pairs played already: no summon cancelled and retried
    choices: int = 0
    on_field: int = 0
    lp: list = field(default_factory=list, repr=False)


def beam_search(duel, keys: list[str], ours: list[str], net: Net, ctx: Ctx, beam: int = 8, branch: int = 3,
                alpha: float = 0.02) -> list[Node]:
    """Beam search over our real choices from `duel` (which it consumes) to the end of turn 1.
    Each step: the open lines' children (the `branch` answers the net likes most per line, on
    clones) are settled to their next real choice, the net rates them, and the `beam` best by
    value + alpha * log-probability go on. Returns every finished line with its end-of-turn value,
    best first."""
    root = Node(list(keys), list(ours), duel)
    try:
        settle(root.duel, root.keys, root.ours, ctx)
    except Exception:
        return []
    frontier, finished = [root], []
    while frontier:
        live, ended, seen = [], [], set()
        for node in frontier:
            if over(node.duel):
                ended.append(node)
                continue
            try:
                ident = identity(node.duel, node.ours)
            except Exception:
                continue
            if ident in seen or node.choices >= ctx.max_choices:
                continue
            seen.add(ident)
            live.append(node)
        if ended:
            for node, value in zip(ended, net.terminal([n.duel for n in ended])):
                node.value, node.on_field, node.duel = value, field_cards(node.duel), None
            finished += ended
        if not live:
            break
        for node, (lp, value) in zip(live, net.ask([n.duel for n in live])):
            node.value, node.lp = value, lp
        live.sort(key=lambda n: n.value + alpha * n.logp, reverse=True)
        for node in live[beam:]:
            node.duel = None
        children = []
        for node in live[:beam]:
            options = node.duel.keys()
            state = board(node.duel)
            # No answer again on the same board (a summon cancelled and retried), and no cancel at
            # all: a started summon undone is a detour the book would teach as a real move.
            fresh = [i for i, k in enumerate(options) if ((state, k) not in node.tried or k.startswith(PASSIVE))
                     and not k.endswith(".cancel")]
            for i in sorted(fresh or range(len(options)), key=lambda i: node.lp[i], reverse=True)[:branch]:
                children.append((Node(node.keys + [options[i]], node.ours + [options[i]], None,
                                      node.logp + node.lp[i], node.value, node.tried | {(state, options[i])},
                                      node.choices + 1), node.duel))
            node.lp = []
        frontier = []
        for child in cc.grow(children):
            try:
                settle(child.duel, child.keys, child.ours, ctx)
            except Exception:
                continue
            frontier.append(child)
    finished.sort(key=lambda n: n.value, reverse=True)
    return finished


@dataclass
class Play:
    """One line played out in one scenario: all answers, ours, whether it left the given answers
    (`diverged`: an answer was not offered; the net played on greedily), the end."""
    keys: list[str]
    ours: list[str]
    duel: object
    pending: list[str]
    diverged: bool = False
    followed: int = 0
    choices: int = 0
    value: float = 0.0
    on_field: int = 0
    end_board: list = field(default_factory=list)
    error: bool = False


def play_out(plays: list[Play], net: Net, ctx_of) -> None:
    """Plays all `plays` to the end of turn 1 in lockstep: our answers from `pending` while they
    are offered (`worlds.match`; a single answer that does not match is played without using one
    up), then the net's most likely answer; seat 1 by `ctx_of(play).opponent`. Net calls of all
    plays are batched. Sets value, field and end board."""
    active = list(plays)
    while active:
        asking = []
        for play in active:
            try:
                while not over(play.duel):
                    options = play.duel.keys()
                    ctx = ctx_of(play)
                    if play.duel.player != 0:
                        key = ctx.opponent(play.duel, play.keys) if ctx.opponent else worlds.decline(options)
                    else:
                        key = None
                        if play.pending and not play.diverged:
                            key = worlds.match(play.pending[0], options)
                            if key is not None:
                                play.pending.pop(0)
                                play.followed += 1
                            elif len(options) > 1:
                                play.diverged = True
                        if key is None and len(options) == 1:
                            key = options[0]
                        if key is None and play.choices >= ctx.max_choices:
                            key = next((k for k in options if k.startswith(END_TURN)), worlds.decline(options))
                        if key is None:
                            asking.append(play)
                            break
                        play.ours.append(key)
                    play.duel.act(key)
                    play.keys.append(key)
            except Exception:
                play.error = True
        if asking:
            for play, (lp, _) in zip(asking, net.ask([p.duel for p in asking])):
                options = play.duel.keys()
                key = options[max(range(len(options)), key=lp.__getitem__)]
                play.choices += 1
                try:
                    play.duel.act(key)
                except Exception:
                    play.error = True
                    continue
                play.keys.append(key)
                play.ours.append(key)
        active = [p for p in asking if not p.error]
    done = [p for p in plays if not p.error]
    for play, value in zip(done, net.terminal([p.duel for p in done])):
        play.value, play.on_field = value, field_cards(play.duel)
        play.end_board = [] if play.duel.finished else cc.end_board(play.duel)
    for play in plays:
        play.duel = None


def judge(candidates: list[list[str]], starts: list, net: Net, ctx_of_k) -> list[list[Play]]:
    """Each candidate (our answers) played in every scenario (a clone of its start duel)."""
    plays = [[Play([], [], start.clone(), list(ours)) for start in starts] for ours in candidates]
    lookup = {id(p): k for row in plays for k, p in enumerate(row)}
    play_out([p for row in plays for p in row], net, lambda p: ctx_of_k(lookup[id(p)]))
    return plays


def mean(values) -> float | None:
    values = list(values)
    return round(statistics.fmean(values), 4) if values else None


def pick(plays: list[list[Play]]) -> int | None:
    """The candidate with the best mean end-of-turn value over the scenarios it was played in
    without an engine error (scenario 0 must be one; ties: the better value in scenario 0)."""
    best, best_key = None, None
    for i, row in enumerate(plays):
        if row[0].error:
            continue
        values = [p.value for p in row if not p.error]
        key = (statistics.fmean(values), row[0].value)
        if best_key is None or key > best_key:
            best, best_key = i, key
    return best


def reveal_at(start, keys: list[str]) -> tuple[bool, int | None]:
    """Replays `keys` from a fresh scenario duel: whether they all replay (the book replays them
    so), and the index of our first answer after which a card left our deck that none of our
    answers named in the deck (a search names its card; a draw, excavation or mill does not)."""
    duel = start()
    if duel is None:
        return False, None
    initial, named = deck_cards(duel), Counter()
    first, n = None, 0
    for key in keys:
        if duel.finished or key not in duel.keys():
            return False, first
        mine = duel.player == 0
        ref = worlds.card_ref(key)
        if mine and ref is not None and int(ref[3]) == worlds.LOCATION_DECK:
            named[int(ref[1])] += 1
        duel.act(key)
        n += mine
        if first is None and +((initial - deck_cards(duel)) - named):
            first = max(0, n - 1)
    return True, first


def trap_root(start_duel, ours: list[str], trap: int, bot, ctx: Ctx):
    """The line `ours` replayed in a scenario where seat 1 holds `trap` (bot answers for it) until
    the trap fired: (duel, keys, our answers) at our first decision after it, or None if it never
    fired (or the line no longer fits before it)."""
    duel, keys, mine, pending = start_duel, [], [], list(ours)
    for _ in range(4 * MAX_AUTO):
        if over(duel):
            return None
        options = duel.keys()
        if duel.player != 0:
            key = bot(duel, keys)
        else:
            if ob.fired(keys, trap):
                return duel, keys, mine
            key = worlds.match(pending[0], options) if pending else None
            if key is not None:
                pending.pop(0)
            elif len(options) == 1:
                key = options[0]
            else:
                return None
            mine.append(key)
        duel.act(key)
        keys.append(key)
    return None


@dataclass
class Hand:
    """One random going-first hand of one deck and the K scenarios it is played in."""
    id: str
    deck: Path
    deck_type: str
    opponent: Path
    decklist: tuple
    opponent_cards: list
    cards: list
    trap: int | None
    orders: int

    def script(self, k: int, trap: int | None = None) -> tuple[str, int]:
        """Scenario k: this hand, deck order and opponent hand k (with `trap` in seat 1's hand)."""
        rng = random.Random(f"{self.id}-order-{k}")
        combo = cb.Combo(self.id, "turn1", self.deck_type, 0, [], [])
        script = cc.scenario_script(combo, self.decklist, self.opponent_cards, lambda code: False, rng,
                                    hand=self.cards, opponent_hand=[trap] if trap else [])
        return script, rng.getrandbits(32)

    def start(self, k: int, trap: int | None = None):
        script, seed = self.script(k, trap)
        return script, seed, cc.puzzle_start(self.deck, self.opponent, script, seed)


def random_hand(decklist: tuple, run_seed: int, deck: Path, t: int) -> list[int]:
    main = list(decklist[0])
    random.Random(f"{run_seed}-{deck.stem}-hand-{t}").shuffle(main)
    return main[:HAND_SIZE]


def trap_pool(rules: dict, card_name: dict, size: int) -> list[int]:
    """The hand traps of the branches: the deck's (counters data), in TRAP_ORDER, at most `size`."""
    rank = {name: i for i, name in enumerate(TRAP_ORDER)}
    known = [t for t in rules if card_name.get(t) in rank]
    return sorted(known, key=lambda t: rank[card_name[t]])[:size]


@dataclass
class Settings:
    beam: int = 8
    branch: int = 3
    alpha: float = 0.02
    candidates: int = 4
    max_choices: int = 120


def search_branch(hand: Hand, trap: int | None, net: Net, s: Settings, center: set, is_extra,
                  base: list[str] | None = None) -> tuple[dict | None, dict, list[str]]:
    """One branch of a hand: the base (trap None) or the interruption by `trap` after the base
    line `base` (our answers). Returns (line record or None, stats, our answers of the line)."""
    began = time.perf_counter()
    bot = ob.hand_trap_bot(trap, set()) if trap else None
    ctx = Ctx(bot, center, is_extra, s.max_choices)
    scenarios = [hand.start(k, trap) for k in range(hand.orders)]
    starts = [start() for _, _, start in scenarios]
    stats = {"trap": trap}
    if any(d is None for d in starts):
        return None, {**stats, "error": "scenario"}, []
    root = starts[0].clone()
    keys, ours = [], []
    if trap:
        found = trap_root(root, base, trap, bot, ctx)
        if found is None:
            return None, {**stats, "fired": False}, []
        root, keys, ours = found
    lines = beam_search(root, keys, ours, net, ctx, s.beam, s.branch, s.alpha)
    candidates, seen = [], set()
    for node in lines:
        if tuple(node.ours) not in seen:
            seen.add(tuple(node.ours))
            candidates.append(node.ours)
        if len(candidates) >= s.candidates:
            break
    # The net's own play: greedy in every scenario (after the base line's prefix in a branch).
    own = judge([ours], starts, net, lambda k: ctx)[0]
    own_index = None
    if not own[0].error:
        if tuple(own[0].ours) not in seen:
            candidates.append(own[0].ours)
        own_index = candidates.index(own[0].ours)
    plays = judge(candidates, starts, net, lambda k: ctx)
    best = pick(plays)
    stats.update(finished=len(lines), candidates=len(candidates), seconds=round(time.perf_counter() - began, 1),
                 own_value=mean(p.value for p in own if not p.error), own_field=mean(p.on_field for p in own if not p.error),
                 own_value0=own[0].value, own_field0=own[0].on_field)
    if best is None:
        return None, {**stats, "error": "no line"}, []
    row = plays[best]
    script, duel_seed, start = scenarios[0]
    keys = row[0].keys
    replays, reveal = reveal_at(start, keys)
    stats.update(value=mean(p.value for p in row if not p.error), field=mean(p.on_field for p in row if not p.error),
                 value0=row[0].value, field0=row[0].on_field, pick="own" if best == own_index else "search", beam_value0=lines[0].value if lines else None, diverged=sum(p.diverged for p in row[1:]),
                 replays=replays, reveal_at=reveal, our_answers=len(row[0].ours))
    if not replays:
        return None, {**stats, "error": "no replay"}, []
    if trap and not ob.fired(keys, trap):
        return None, {**stats, "fired": False}, []
    record = {"seed": hand.id, "title": f"turn1 {hand.deck.stem} {hand.id}", "source": "turn1", "author": "",
              "origin": "turn1_search", "variant": 0, "trap": trap, "fired": True if trap else None, "line": keys,
              "steps": 0, "progress": 0, "skipped": 0, "key_cards": 0, "book_end_board": [], "field": row[0].on_field,
              "end_board": row[0].end_board, "script": script, "duel_seed": duel_seed, "first": True,
              "hand": hand.cards, "fillers": [],
              "value": {"held": None, "left": None, "tries": 0, "unbuildable": 0, "errors": 0},
              "search": {k: v for k, v in stats.items() if k != "trap"}, "seconds": stats["seconds"]}
    return record, stats, row[0].ours


def search_hand(hand: Hand, net: Net, s: Settings, center: set, is_extra) -> tuple[list[dict], dict]:
    """The base line of a hand and, after it, the interruption branch of the hand's trap."""
    base, stats, ours = search_branch(hand, None, net, s, center, is_extra)
    out, summary = [], {"hand": hand.id, "cards": hand.cards, "base": stats}
    if base is not None:
        out.append(base)
        if hand.trap:
            branch, summary["trap"], _ = search_branch(hand, hand.trap, net, s, center, is_extra, ours)
            if branch is not None:
                out.append(branch)
    return out, summary


def paths(out: Path) -> dict[str, Path]:
    stem = out.with_name(out.stem)
    return {name: Path(f"{stem}-{name}.{ext}") for name, ext in
            (("lines", "jsonl"), ("hands", "jsonl"), ("meta", "json"), ("report", "json"))}


def done_hands(out: Path) -> set[str]:
    """--resume: the hands already finished; lines of other hands (a crash between a hand's lines
    and its stats) are dropped from the lines file."""
    p = paths(out)
    if not p["hands"].exists():
        return set()
    done = {json.loads(t)["hand"] for t in p["hands"].read_text(encoding="utf-8").splitlines() if t.strip()}
    if p["lines"].exists():
        kept = [t for t in p["lines"].read_text(encoding="utf-8").splitlines() if t.strip() and json.loads(t)["seed"] in done]
        p["lines"].write_text("".join(t + "\n" for t in kept), encoding="utf-8")
    return done


def gains(record: dict) -> float | None:
    """How much better the kept line is than the net's own play: mean end-of-turn value over the K
    scenarios minus that of the net's greedy line (in a branch: its greedy continuation after the
    same prefix); None for records without search stats."""
    search = record.get("search") or {}
    if search.get("value") is None or search.get("own_value") is None:
        return None
    return search["value"] - search["own_value"]


def gain_path(out: Path) -> Path:
    """The filtered book beside `out` (`--min-gain`): <stem>-gain.jsonl, with its own lines and meta."""
    return out.with_name(f"{out.stem}-gain.jsonl")


def write_book(out: Path, deck: Path, opponent: Path, records: list[dict], meta: dict) -> int:
    """Book `out`, its lines file and meta (with the relevant cards) from `records`; the number of states."""
    p = paths(out)
    lines = [(r, cc.puzzle_start(deck, opponent, r["script"], r["duel_seed"])) for r in records]
    relevant = ob.book_cards(records)
    book, _ = ob.build_book(lines, relevant)
    if not p["lines"].exists() or [json.loads(t) for t in p["lines"].read_text(encoding="utf-8").splitlines()] != records:
        p["lines"].write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    out.write_text("".join(json.dumps(e) + "\n" for e in book.values()), encoding="utf-8")
    p["meta"].write_text(json.dumps({**meta, "relevant": sorted(relevant)}), encoding="utf-8")
    return len(book)


def build(out: Path, deck: Path, opponent: Path, meta: dict, min_gain: float | None = None) -> dict:
    """The book of the lines file (opening_book.build_book) beside `out`, its meta and a report.
    With `min_gain`, also the book of the lines whose value beats the net's own line by more than
    that (`gain_path`): a line the net would play anyway (or nearly as well) teaches it nothing.
    Base and branch lines are filtered alike; a branch line whose base line is dropped keeps the
    base line's answers before the hand trap."""
    p = paths(out)
    records = [json.loads(t) for t in p["lines"].read_text(encoding="utf-8").splitlines() if t.strip()]
    states = write_book(out, deck, opponent, records, meta)
    report = summarize([json.loads(t) for t in p["hands"].read_text(encoding="utf-8").splitlines() if t.strip()])
    report.update(states=states, lines=len(records))
    if min_gain is not None:
        kept = [r for r in records if (g := gains(r)) is not None and g > min_gain]
        report["gain"] = {"min_gain": min_gain, "lines": len(kept),
                          "base": sum(r["trap"] is None for r in kept), "trap": sum(r["trap"] is not None for r in kept),
                          "base_of": sum(r["trap"] is None for r in records),
                          "trap_of": sum(r["trap"] is not None for r in records),
                          "states": write_book(gain_path(out), deck, opponent, kept, {**meta, "min_gain": min_gain})}
    p["report"].write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def summarize(hands: list[dict]) -> dict:
    """Search statistics over the hands: lines per hand, end-of-turn value and field of the kept
    lines vs the net's own greedy play (mean over the K scenarios, and in scenario 0)."""
    out = {"hands": len(hands)}
    for branch in ("base", "trap"):
        rows = [h[branch] for h in hands if branch in h]
        kept = [r for r in rows if "value" in r and not r.get("error") and r.get("fired", True) is not False
                and r.get("replays")]
        if not rows:
            continue
        out[branch] = {
            "tried": len(rows), "kept": len(kept), "fired": sum(r.get("fired", True) is not False for r in rows),
            "errors": dict(Counter(r["error"] for r in rows if r.get("error"))),
            "value": mean(r["value"] for r in kept), "own_value": mean(r["own_value"] for r in kept),
            "field": mean(r["field"] for r in kept), "own_field": mean(r["own_field"] for r in kept),
            "field0": mean(r["field0"] for r in kept), "own_field0": mean(r["own_field0"] for r in kept),
            "picked_own": mean(r["pick"] == "own" for r in kept),
            "better": mean(r["value"] > r["own_value"] + 1e-6 for r in kept),
            "reveals": mean(r["reveal_at"] is not None for r in kept),
            "diverged": mean(r["diverged"] for r in kept), "finished": mean(r["finished"] for r in kept),
            "seconds": mean(r["seconds"] for r in kept)}
    out["lines_per_hand"] = round(sum(out.get(b, {}).get("kept", 0) for b in ("base", "trap")) / max(1, len(hands)), 3)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--deck", type=Path, required=True)
    p.add_argument("--deck-type", required=True, help="masterduelmeta deck type (hand traps of the counters data)")
    p.add_argument("--opponent", type=Path, default=rollout.REPO / "decks" / "maliss.ydk")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--hands", type=int, default=200)
    p.add_argument("--first-hand", type=int, default=0)
    p.add_argument("--orders", type=int, default=4, help="K: deck orders (scenarios) each candidate line is judged in")
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--branch", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.02, help="weight of the log-probability in the beam's rank")
    p.add_argument("--candidates", type=int, default=4, help="best beam lines judged over the K orders")
    p.add_argument("--max-choices", type=int, default=120)
    p.add_argument("--traps", type=int, default=4, help="hand traps the branches rotate through (0: no branches)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--min-gain", type=float, default=None,
                   help="also build <out stem>-gain.jsonl from the lines whose mean value beats the net's own line "
                        "by more than this")
    p.add_argument("--build-only", action="store_true", help="build book and report from the lines")
    p.add_argument("--out", type=Path, required=True, help="book (JSON lines); lines, hands, meta, report beside it")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import torch

    from . import seeds as sd
    from .cards import CardTable
    from .model import load_checkpoint

    args = parse_args(argv)
    args.deck, args.opponent = args.deck.resolve(), args.opponent.resolve()
    meta = {"deck": str(args.deck), "deck_type": args.deck_type, "opponent": str(args.opponent), "sources": "turn1"}
    if args.build_only:
        print(json.dumps(build(args.out, args.deck, args.opponent, meta, args.min_gain)), flush=True)
        return
    torch.set_num_threads(args.torch_threads)
    table = CardTable.from_cdb()
    card_name = {c.code: c.name for c in table.cards}
    types = {c.code: c.type for c in table.cards}
    is_extra = lambda code: bool(types.get(code, 0) & cc.TYPE_EXTRA)  # noqa: E731
    center = {c.code for c in table.cards if cc.wants_center(c.text)}
    decklist = worlds.read_decklist(args.deck)
    opponent_cards = worlds.read_decklist(args.opponent)[0]
    resolve = sd.resolver(cb.name_index(table), set(decklist[0]) | set(decklist[1]), card_name)
    rules = {t: v for t, v in sd.interruptions(args.deck_type, resolve).items() if card_name.get(t) in INTERRUPTERS}
    traps = trap_pool(rules, card_name, args.traps)
    print(f"hand traps: {[card_name[t] for t in traps]}", flush=True)
    model = load_checkpoint(args.checkpoint, args.device, dropout=0.0)
    evaluator = Evaluator(model_fn(model, args.device))
    net = Net(evaluator, model.card_features.index_map())
    settings = Settings(args.beam, args.branch, args.alpha, args.candidates, args.max_choices)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    p = paths(args.out)
    done = done_hands(args.out) if args.resume else set()
    if not args.resume:
        for path in (p["lines"], p["hands"]):
            path.write_text("", encoding="utf-8")
    p["meta"].write_text(json.dumps(meta), encoding="utf-8")
    hands = []
    for t in range(args.first_hand, args.first_hand + args.hands):
        hid = f"t1-{args.deck.stem}-{args.seed}-{t}"
        if hid not in done:
            hands.append(Hand(hid, args.deck, args.deck_type, args.opponent, decklist, opponent_cards,
                              random_hand(decklist, args.seed, args.deck, t),
                              traps[t % len(traps)] if traps else None, args.orders))
    print(f"{len(hands)} hands to search ({len(done)} done)", flush=True)
    began = time.perf_counter()
    lock = threading.Lock()
    with ThreadPoolExecutor(max(1, args.workers)) as pool, open(p["lines"], "a", encoding="utf-8") as lines_out, \
            open(p["hands"], "a", encoding="utf-8") as hands_out:
        futures = [pool.submit(search_hand, hand, net, settings, center, is_extra) for hand in hands]
        for n, future in enumerate(futures):
            try:
                records, stats = future.result()
            except Exception as error:  # one broken hand must not stop the run
                print(f"{n + 1}/{len(futures)} error {str(error)[:160]}", flush=True)
                continue
            with lock:
                for record in records:
                    lines_out.write(json.dumps(record) + "\n")
                lines_out.flush()
                hands_out.write(json.dumps(stats) + "\n")
                hands_out.flush()
            b = stats["base"]
            print(f"{n + 1}/{len(futures)} {stats['hand']} value {b.get('value')} (own {b.get('own_value')}) "
                  f"field {b.get('field0')} (own {b.get('own_field0')}) pick {b.get('pick')} "
                  f"trap {card_name.get(hands[n].trap, '-')[:12]} lines {len(records)} {b.get('seconds')}s "
                  f"[{time.perf_counter() - began:.0f}s, {evaluator.rows / max(1, evaluator.calls):.1f} rows/call]",
                  flush=True)
    evaluator.close()
    print(json.dumps(build(args.out, args.deck, args.opponent, meta, args.min_gain)), flush=True)


if __name__ == "__main__":
    main()
