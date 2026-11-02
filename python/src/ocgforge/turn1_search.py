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
  (the same hand, other deck orders and opponent hands) besides the beam's scenario 0. The search
  runs in scenario 0; its best `--candidates` finished lines and the net's own greedy line are then
  replayed in scenarios 1..K, which the beam never saw (our answers only, matched with
  `worlds.match`; from the first answer that is not offered on, the net plays greedily), and the
  line with the best MEAN score is kept (a line with an engine error in any scenario is out). A
  choice made before the deck shows a card is thus judged over K deck orders; answers after it react
  to cards that are visible by then. The book line is the kept line in scenario 0. Each kept line
  records `reveal_at`, our first answer after which a card we did not name left the deck (None: the
  line never looks into the deck).
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
- `--min-gain G`: a second book `<out stem>-gain.jsonl` holds only the lines whose gain over the
  net's own line (greedy; in a branch, its greedy continuation) exceeds G (`--min-gain-se z`: also z
  standard errors). The gain is measured apart from the grading, in `--gain-orders` further
  scenarios with fresh seeds, the same way under every scorer: the grading picked the maximum, so
  its numbers flatter the kept line. Where the net's own line was kept, imitation teaches nothing new.
- Resumable: every finished hand is appended to `<out>-hands.jsonl` (stats) right after its lines
  and rollouts; `--resume` skips those hands (drops lines of an unfinished hand and torn lines).
  Hands that failed make the run exit with FAILED_HANDS (3), listed in `<out>-failed.json`.
- Net calls of all search threads go through one `Evaluator`, which batches them on the GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import queue
import random
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import combo_check as cc
from . import combos as cb
from . import opening_book as ob
from . import outcome_eval as oe
from . import rollout, worlds
from .answer_rules import deck_cards, forced
from .seeds import HAND_SIZE, INTERRUPTERS

FIELD = oe.FIELD
END_TURN = ("idle.7", "battle.3")  # to the End Phase (board_break.END_TURN)
PASSIVE = ("chain.pass", "yes_no.", "effect_yes_no.")  # never "the same answer again" (combo_check.PASSIVE)
FAILED_HANDS = 3  # exit code: the run went through, but some hands failed (retry with --resume)
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
            except Exception:  # retry each request on its own: only the bad one fails (its hand)
                for rows, future in jobs:
                    try:
                        future.set_result(self.fn(rows))
                    except Exception as error:
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
    score: float | None = None  # what `pick` ranks by: the value, or the scorer's (outcome_eval)
    t2: dict = field(default_factory=dict)  # turn-2 outcome stats (outcome_eval.summary)
    sig: object = None  # the position at the end of turn 1 (`position`; set by score_turn2)
    raw: list = field(default_factory=list)  # every rollout of this play: j, score, lost, won (score_turn2)


def play_out(plays: list[Play], net: Net, ctx_of, keep: bool = False) -> None:
    """Plays all `plays` to the end of turn 1 in lockstep: our answers from `pending` while they
    are offered (`worlds.match`; a single answer that does not match is played without using one
    up), then the net's most likely answer; seat 1 by `ctx_of(play).opponent`. Net calls of all
    plays are batched. Sets value (and score), field and end board; `keep` keeps the duels at the
    end of turn 1 (for the turn-2 rollouts)."""
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
        play.score = value
        play.end_board = [] if play.duel.finished else cc.end_board(play.duel)
    for play in plays:
        if not keep or play.error:
            play.duel = None


def judge(candidates: list[list[str]], starts: list, net: Net, ctx_of_k, keep: bool = False) -> list[list[Play]]:
    """Each candidate (our answers) played in every scenario (a clone of its start duel)."""
    plays = [[Play([], [], start.clone(), list(ours)) for start in starts] for ours in candidates]
    lookup = {id(p): k for row in plays for k, p in enumerate(row)}
    play_out([p for row in plays for p in row], net, lambda p: ctx_of_k(lookup[id(p)]), keep)
    return plays


def position(duel) -> tuple:
    """A duel's position for sharing rollouts: every card (hidden ones too) with its place (in the
    deck: its order), life points, turn, phase and the pending question."""
    if duel.finished:
        return ("finished", duel.winner)
    snap = duel.snapshot()
    ordered = FIELD + (worlds.LOCATION_DECK,)  # the deck's order decides the coming draws
    cards = tuple(sorted((c["controller"], c["location"], c["code"], c["sequence"] if c["location"] in ordered else 0,
                          c["position"] if c["location"] in FIELD else 0) for c in snap["cards"]))
    return cards, tuple(snap["life_points"]), duel.turn, duel.phase, duel.player, tuple(duel.keys())


def score_turn2(plays: list[list[Play]], net: Net, s: "Settings", tag: str, k_offset: int = 0) -> None:
    """The turn-2 scorer (`s.scorer` turn2 or mix): every play kept at the end of turn 1 (judge
    with keep) gets M = `s.rollouts` sampled turn 2s (outcome_eval.play_turn2, all plays in one
    lockstep), its outcome stats (`t2`) and its score. Rollouts of scenario k use the same seeds for
    every candidate; candidates that reach the same position in a scenario (`position`: lines that
    differ only in the order of answers) share its rollouts. Column k of `plays` is scenario
    k + `k_offset` (its seeds). Frees the duels."""
    flat = [(k, p) for row in plays for k, p in enumerate(row) if not p.error and p.duel is not None]
    if s.scorer != "value" and flat:
        groups: dict = {}
        for k, p in flat:
            try:
                p.sig = position(p.duel)
            except Exception:
                p.sig = ("unique", id(p))
            groups.setdefault((k, p.sig), []).append(p)
        firsts = [(key[0], members[0]) for key, members in groups.items()]
        outs = oe.play_turn2([p.duel for _, p in firsts], net, s.rollouts, [f"{tag}-k{k + k_offset}" for k, _ in firsts], s.t2_cap)
        for (k, _), members, rollouts in zip(firsts, groups.values(), outs):
            t2 = oe.summary(rollouts)
            # one row per rollout (long format); early: the duel was decided (+-1), else a value estimate
            raw = [{"k": k + k_offset, "j": j, "seed": f"{tag}-k{k + k_offset}-{j}", "score": r.value,
                    "early": bool(r.stats.get("finished")), "lost": r.stats.get("lost"), "won": r.stats.get("won"),
                    "capped": r.capped} for j, r in enumerate(rollouts) if not r.error and r.value is not None]
            for p in members:
                p.t2, p.raw = t2, raw
                p.score = oe.combine(s.scorer, p.value, t2["t2"], s.mix_weight)
    for row in plays:
        for p in row:
            p.duel = None


mean = oe.mean  # rounded mean of the values that are not None (None for none)


def pick(plays: list[list[Play]], by: str = "score") -> int | None:
    """The candidate with the best mean score (`by`: "score", the scorer's, or "value", the
    end-of-turn value) over the grading scenarios. A candidate with an engine error or no score in
    any scenario is left out (a mean over fewer scenarios is not comparable); ties: the earlier
    candidate (the beam's order)."""
    best, best_mean = None, None
    for i, row in enumerate(plays):
        if not row or any(p.error or getattr(p, by) is None for p in row):
            continue
        value = statistics.fmean(getattr(p, by) for p in row)
        if best_mean is None or value > best_mean:
            best, best_mean = i, value
    return best


def reveal_at(start, keys: list[str]) -> tuple[bool, int | None]:
    """Replays `keys` from a fresh scenario duel: whether they all replay (the book replays them
    so), and the index of our first answer after which a card left our deck that none of our
    answers named in the deck (a search names its card; a draw, excavation or mill does not).

    Known gap: only cards that end up outside the deck count. An excavation that looks at cards
    and puts them all back (or a reveal of the top card) is not seen, although the line may react
    to what it showed; telling those apart needs the duel's reveal events."""
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
    opponents: list = field(default_factory=list)  # (deck path, main, extra): real opponents, one per scenario

    def opponent_of(self, k: int) -> tuple[Path, list, list]:
        """Seat 1's deck in scenario k: `opponent` without an Extra Deck (it only declines), or, with
        `opponents`, one of them (a stable choice by hand and k) with its Extra Deck, to play turn 2."""
        if not self.opponents:
            return self.opponent, self.opponent_cards, []
        return self.opponents[random.Random(f"{self.id}-opponent-{k}").randrange(len(self.opponents))]

    def script(self, k: int, trap: int | None = None) -> tuple[str, int]:
        """Scenario k: this hand, deck order and opponent hand k (with `trap` in seat 1's hand)."""
        rng = random.Random(f"{self.id}-order-{k}")
        combo = cb.Combo(self.id, "turn1", self.deck_type, 0, [], [])
        _, main, extra = self.opponent_of(k)
        script = cc.scenario_script(combo, self.decklist, main, lambda code: False, rng,
                                    hand=self.cards, opponent_hand=[trap] if trap else [], opponent_extra=extra)
        return script, rng.getrandbits(32)

    def start(self, k: int, trap: int | None = None):
        script, seed = self.script(k, trap)
        return script, seed, cc.puzzle_start(self.deck, self.opponent_of(k)[0], script, seed)


def random_hand(decklist: tuple, run_seed: int, deck: Path, t: int) -> list[int]:
    main = list(decklist[0])
    random.Random(f"{run_seed}-{deck.stem}-hand-{t}").shuffle(main)
    return main[:HAND_SIZE]


def read_traps(path: Path) -> list[int]:
    """The deck's hand traps from a --traps-file: a JSON list of passcodes (older files: an object
    keyed by passcode; its choke points are ignored, the branches' bot fires at the first chance)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return sorted(int(t) for t in raw)


def write_traps(path: Path, traps) -> None:
    """The deck's resolved hand traps (passcodes) as a --traps-file."""
    path.write_text(json.dumps(sorted(int(t) for t in traps)) + "\n", encoding="utf-8")


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
    scorer: str = "value"  # outcome_eval.SCORERS
    rollouts: int = 2  # M: sampled turn 2s per candidate and scenario
    mix_weight: float = 0.5  # mix: weight of the end-of-turn-1 value
    t2_cap: int = 250  # sampled decisions per turn-2 rollout at most
    validate: int = 0  # >0: fresh turn-2 rollouts (this many) of the kept, value-kept and own line
    validate_orders: int = 0  # >0: the validation plays this many scenarios (default: the search's K)
    log_rollouts: bool = True  # keep every rollout (written to <out>-rollouts.jsonl)
    validate_categories: bool = False  # also validate a mid-ranked, a fragile and a big-board candidate
    validate_chunk: int = 6  # scenarios scored at a time (turn-2 duels held in memory)
    gain_orders: int = 8  # further scenarios the kept line's gain over the own line is measured in
    gain_rollouts: int = 1  # turn-2 rollouts per scenario there
    race: bool = False  # racing instead of K x M grading (see `race`)
    race_k1: int = 8  # stage-1 scenarios (every candidate)
    race_k2: int = 16  # scenarios in all for the stage-2 survivors and the own line
    race_se: float = 2.0  # dropped when the stage-1 best leads by more than this many standard errors
    race_frac: float = 0.75  # ... or leads in at least this share of the scenarios
    race_margin: float = 0.0  # ... with a mean lead above this
    race_c: float = 1.0  # the gain over the own line counts above this many standard errors


def categories(cands: list[dict], positions: list[tuple], best: int, own: int | None) -> dict:
    """Further lines to validate against the own line, each a position other than the kept and the
    own line's: "mid" (the middle of the others by score), "fragile" (the others' highest turn-2 loss
    rate, if any loses) and "big" (the others' biggest field after turn 1, if bigger than the kept)."""
    taken = {positions[best]} | ({positions[own]} if own is not None else set())
    others, seen = [], set(taken)
    for i in sorted(range(len(cands)), key=lambda i: -(cands[i]["score"] if cands[i]["score"] is not None else -9)):
        if positions[i] not in seen and cands[i]["score"] is not None:
            seen.add(positions[i])
            others.append(i)
    out = {"mid": others[len(others) // 2] if others else None}
    fragile = max(others, key=lambda i: cands[i].get("lost") or 0, default=None)
    out["fragile"] = fragile if fragile is not None and (cands[fragile].get("lost") or 0) > 0 else None
    big = max(others, key=lambda i: cands[i].get("field") or 0, default=None)
    out["big"] = big if big is not None and (cands[big].get("field") or 0) > (cands[best].get("field") or 0) else None
    return out


def scored(candidates: list[list[str]], starts: list, net: Net, ctx: Ctx, s: Settings, tag: str, k_first: int,
           rollouts: int | None = None, scorer: str | None = None) -> list[list[Play]]:
    """Each candidate played in every scenario of `starts` (scenario ids from `k_first`) and scored
    (`scorer`, default `s.scorer`; turn2/mix with `rollouts` sampled turn 2s, default `s.rollouts`).
    Turn-2 scoring goes `s.validate_chunk` scenarios at a time: the duels of one chunk are freed
    before the next (memory)."""
    scorer = scorer or s.scorer
    if scorer == "value":
        return judge(candidates, starts, net, lambda k: ctx)
    t2 = dataclasses.replace(s, scorer=scorer, rollouts=rollouts or s.rollouts)
    rows: list[list[Play]] = [[] for _ in candidates]
    for first in range(0, len(starts), max(1, s.validate_chunk)):
        plays = judge(candidates, starts[first:first + s.validate_chunk], net, lambda k: ctx, keep=True)
        score_turn2(plays, net, t2, tag, k_offset=k_first + first)
        for row, part in zip(rows, plays):
            row += part
    return rows


def paired_gain(kept: list[Play], own: list[Play]) -> dict:
    """Mean and standard error of score(kept) - score(own) over the scenarios both were scored in."""
    diffs = [a.score - b.score for a, b in zip(kept, own)
             if not a.error and not b.error and a.score is not None and b.score is not None]
    if not diffs:
        return {"gain": None, "gain_se": None, "gain_n": 0}
    se = statistics.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else None
    return {"gain": round(statistics.fmean(diffs), 4), "gain_se": round(se, 4) if se is not None else None,
            "gain_n": len(diffs)}


def complete(row: list[Play]) -> bool:
    return bool(row) and all(not p.error and p.score is not None for p in row)


def diffs(a: list[Play], b: list[Play]) -> list[float]:
    """Paired scenario differences score(a) - score(b) (both complete rows over the same scenarios)."""
    return [x.score - y.score for x, y in zip(a, b)]


def mean_se(values: list[float]) -> tuple[float, float | None]:
    m = statistics.fmean(values)
    return m, (statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None)


def race(candidates: list[list[str]], own_index: int | None, hand: Hand, trap: int | None, net: Net, ctx: Ctx,
         s: Settings, tag: str) -> tuple[list[list[Play]], int | None, dict]:
    """Racing (`s.race`), all with 1 rollout per scenario, common seeds and engine seeds per scenario:
    1. every candidate (the own line included) in scenarios 1..K1 (the beam's is 0);
    2. a candidate is dropped when it reaches the stage-1 best's positions in every scenario (the same
       line in another order), or when the stage-1 best beats it clearly: its mean paired difference is
       above `race_se` standard errors, or at least `race_frac` of the differences are positive and
       their mean is above `race_margin` (deterministic); the survivors and the own line go on in
       scenarios K1+1..K2;
    3. the kept line is the survivor with the best mean over all K2 scenarios; its gain over the own
       line is measured on the stage-2 scenarios alone (not used to drop anyone), and it counts
       (`gain_ok`) when above `race_c` standard errors.
    Returns (rows per candidate: stage 1, plus stage 2 for those that went on; kept index; stats)."""
    def starts(ks):
        out = [start() for _, _, start in (hand.start(k, trap) for k in ks)]
        return None if any(d is None for d in out) else out

    first = starts(range(1, s.race_k1 + 1))
    second = starts(range(s.race_k1 + 1, s.race_k2 + 1))
    info = {"k1": s.race_k1, "k2": s.race_k2, "stage1": len(candidates)}
    if first is None or second is None or not second:
        return [[] for _ in candidates], None, {**info, "error": "scenario"}
    rows = scored(candidates, first, net, ctx, s, tag, 1, 1)
    ok = [i for i, row in enumerate(rows) if complete(row)]
    if not ok:
        return rows, None, {**info, "error": "no line"}
    means = {i: statistics.fmean(p.score for p in rows[i]) for i in ok}
    lead = max(ok, key=lambda i: (means[i], -i))
    survivors = [lead]
    for i in ok:
        if i == lead:
            continue
        if all(x.sig is not None and x.sig == y.sig for x, y in zip(rows[lead], rows[i])):
            continue  # the same positions as the lead (answers in another order): nothing to race
        d = diffs(rows[lead], rows[i])
        m, se = mean_se(d)
        clear = (se is not None and m > s.race_se * se) or \
            (sum(x > 0 for x in d) >= s.race_frac * len(d) and m > s.race_margin)
        if not clear:
            survivors.append(i)
    go_on = sorted(set(survivors) | ({own_index} if own_index in ok else set()))
    for i, row in zip(go_on, scored([candidates[i] for i in go_on], second, net, ctx, s, tag, s.race_k1 + 1, 1)):
        rows[i] = rows[i] + row
    full = [i for i in survivors if complete(rows[i]) and len(rows[i]) == s.race_k2]
    if not full:
        return rows, None, {**info, "stage2": len(survivors), "error": "no line"}
    total = {i: statistics.fmean(p.score for p in rows[i]) for i in full}
    best = max(full, key=lambda i: (total[i], -i))
    info["stage2"] = len(survivors)
    rivals = [i for i in full if i != best]
    if rivals:  # the decision among the survivors, over all K2 scenarios
        second_best = max(rivals, key=lambda i: (total[i], -i))
        m, se = mean_se(diffs(rows[best], rows[second_best]))
        info.update(margin=round(m, 4), margin_se=round(se, 4) if se is not None else None,
                    noise=se is not None and m < s.race_se * se)
    else:  # nobody survived stage 1 against it
        others = [i for i in ok if i != best]
        info.update(margin=round(means[best] - max(means[i] for i in others), 4) if others else None,
                    margin_se=None, noise=False)
    if best == own_index:
        info.update(gain=0.0, gain_se=None, gain_n=0, gain_ok=False)
    elif own_index is not None and complete(rows[own_index]) and len(rows[own_index]) == s.race_k2:
        m, se = mean_se(diffs(rows[best][s.race_k1:], rows[own_index][s.race_k1:]))
        info.update(gain=round(m, 4), gain_se=round(se, 4) if se is not None else None,
                    gain_n=s.race_k2 - s.race_k1, gain_ok=se is not None and m > s.race_c * se)
    return rows, best, info


def search_branch(hand: Hand, trap: int | None, net: Net, s: Settings, center: set, is_extra,
                  base: list[str] | None = None) -> tuple[dict | None, dict, list[str]]:
    """One branch of a hand: the base (trap None) or the interruption by `trap` after the base
    line `base` (our answers). Returns (line record or None, stats, our answers of the line).

    Scenario 0 is the beam's; the candidates are graded in scenarios 1..K (deck orders the beam
    never saw), and the kept line's gain over the net's own line is measured once more in
    `s.gain_orders` further scenarios with fresh seeds (the grading picked the maximum, so its own
    numbers flatter the kept line). The book line is the kept line replayed in scenario 0."""
    began = time.perf_counter()
    bot = ob.hand_trap_bot(trap, set()) if trap else None
    ctx = Ctx(bot, center, is_extra, s.max_choices)
    scenarios = [hand.start(k, trap) for k in range(hand.orders + 1)]
    starts = [start() for _, _, start in scenarios]
    stats = {"trap": trap}
    if any(d is None for d in starts):
        return None, {**stats, "error": "scenario"}, []
    grade = starts[1:]
    tag = f"{hand.id}-{trap or 'base'}"
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
    # The net's own play: greedy in every grading scenario (after the base line's prefix in a branch).
    own = judge([ours], grade, net, lambda k: ctx)[0]
    own_line = own[0].ours if not own[0].error else None
    own_index = None
    if own_line is not None:
        if tuple(own_line) not in seen:
            candidates.append(own_line)
        own_index = candidates.index(own_line)
    turn2 = s.scorer != "value"
    if s.race:
        plays, best, race_info = race(candidates, own_index, hand, trap, net, ctx, s, tag)
        stats["race"] = {k: v for k, v in race_info.items() if not k.startswith("gain")}
        stats.update({k: v for k, v in race_info.items() if k.startswith("gain")})
    else:
        plays = scored(candidates, grade, net, ctx, s, tag, 1)
        best = pick(plays)
    stats.update(finished=len(lines), candidates=len(candidates), seconds=round(time.perf_counter() - began, 1),
                 own_value=mean(p.value for p in own if not p.error), own_field=mean(p.on_field for p in own if not p.error),
                 own_value0=own[0].value, own_field0=own[0].on_field)
    if turn2:
        # every candidate's numbers, and the line the value scorer would have kept (on the same hand)
        stats["cands"] = [{"value": mean(p.value for p in r if not p.error),
                           "score": mean(p.score for p in r if not p.error and p.score is not None),
                           "field": mean(p.on_field for p in r if not p.error),
                           **{key: mean(p.t2.get(key) for p in r if not p.error and p.t2.get(key) is not None)
                              for key in ("lost", "won", "field3", "lp", "opp_lp", "capped", "decisions")}}
                          for r in plays]
        value_index = pick(plays, "value") if not s.race else \
            pick([row[:s.race_k1] for row in plays], "value")
        same = lambda i, j: None if i is None or j is None else [p.sig for p in plays[i]] == [p.sig for p in plays[j]]  # noqa: E731
        stats.update(scorer=s.scorer, own_index=own_index, best_index=best, value_index=value_index,
                     own_score=stats["cands"][own_index]["score"] if own_index is not None else None,
                     # the same positions after turn 1 in every scenario: the lines differ in order only
                     same_as_value=same(best, value_index), same_as_own=same(best, own_index),
                     value_same_as_own=same(value_index, own_index))
        if s.log_rollouts:  # the scoring rollouts in long format: candidate i, scenario k, rollout j
            stats["score_raw"] = [{"phase": "score", "cand": i, "line": "kept" if i == best else "own" if i == own_index
                                   else f"c{i}", **r} for i, row in enumerate(plays) for p in row if not p.error
                                  for r in p.raw]
        scores = [c["score"] for c in stats["cands"]]
        others = [x for i, x in enumerate(scores) if i != best and x is not None and x != scores[best]]
        stats["margin"] = round(scores[best] - max(others), 4) if best is not None and others and scores[best] is not None \
            else None
        if s.validate and best is not None:
            # fresh rollouts (other seeds) of the kept, the value scorer's and the net's own line (and with
            # `validate_categories` a mid-ranked, a fragile and a big-board candidate)
            picks = {"kept": best, "value_kept": value_index, "own": own_index}
            if s.validate_categories:
                picks.update(categories(stats["cands"], [tuple(p.sig for p in row) for row in plays], best, own_index))
            chosen = sorted({i for i in picks.values() if i is not None})
            checks = [start() for _, _, start in (hand.start(k, trap) for k in range(s.validate_orders))] \
                if s.validate_orders else grade
            k_first = 0 if s.validate_orders else 1
            if all(d is not None for d in checks):
                rows = dict(zip(chosen, scored([candidates[i] for i in chosen], checks, net, ctx, s,
                                               f"{tag}-validate", k_first, s.validate, "turn2")))
                fresh = {i: oe.summary_of([p.t2 for p in row if not p.error and p.t2]) for i, row in rows.items()}
                if s.log_rollouts:  # every single rollout (scenario k, rollout j: the same seeds for every line)
                    stats["validate_raw"] = [{"phase": "validate", "line": name, "cand": i, **r}
                                             for name, i in picks.items() if i is not None
                                             for p in rows[i] if not p.error for r in p.raw]
                stats["validate"] = {name: fresh[i] for name, i in picks.items() if i is not None}
                stats["validate_picks"] = picks
    if best is None:
        return None, {**stats, "error": "no line"}, []
    # The gain over the net's own line, on independent scenarios and seeds (the same for value and turn2)
    if s.race:
        pass  # measured on the race's stage-2 scenarios
    elif best == own_index:
        stats.update(gain=0.0, gain_se=None, gain_n=0)
    elif own_line is not None and s.gain_orders > 0:
        first = hand.orders + 1
        fresh = [start() for _, _, start in (hand.start(k, trap) for k in range(first, first + s.gain_orders))]
        if all(d is not None for d in fresh):
            kept_rows, own_rows = scored([candidates[best], own_line], fresh, net, ctx, s, f"{tag}-gain", first,
                                         s.gain_rollouts)
            stats.update(paired_gain(kept_rows, own_rows))
    row = plays[best]
    script, duel_seed, start = scenarios[0]
    # the book line: the kept line in the beam's scenario 0 (where its answers were found)
    line = judge([candidates[best]], [starts[0]], net, lambda k: ctx)[0][0]
    if line.error:
        return None, {**stats, "error": "no replay"}, []
    keys = line.keys
    replays, reveal = reveal_at(start, keys)
    stats.update(value=mean(p.value for p in row if not p.error), field=mean(p.on_field for p in row if not p.error),
                 score=mean(p.score for p in row if not p.error and p.score is not None),
                 value0=line.value, field0=line.on_field, pick="own" if best == own_index else "search",
                 beam_value0=lines[0].value if lines else None, diverged=sum(p.diverged for p in row),
                 replays=replays, reveal_at=reveal, our_answers=len(line.ours))
    if not replays:
        return None, {**stats, "error": "no replay"}, []
    if trap and not ob.fired(keys, trap):
        return None, {**stats, "fired": False}, []
    record = {"seed": hand.id, "title": f"turn1 {hand.deck.stem} {hand.id}", "source": "turn1", "author": "",
              "origin": "turn1_search", "variant": 0, "trap": trap, "fired": True if trap else None, "line": keys,
              "steps": 0, "progress": 0, "skipped": 0, "key_cards": 0, "book_end_board": [], "field": line.on_field,
              "end_board": line.end_board, "script": script, "duel_seed": duel_seed, "first": True,
              "hand": hand.cards, "fillers": [], "opponent": str(hand.opponent_of(0)[0]),
              "value": {"held": None, "left": None, "tries": 0, "unbuildable": 0, "errors": 0},
              "search": {k: v for k, v in stats.items() if k not in ("trap", "score_raw", "validate_raw")},
              "seconds": stats["seconds"]}
    return record, stats, line.ours


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
            (("lines", "jsonl"), ("hands", "jsonl"), ("meta", "json"), ("report", "json"), ("rollouts", "jsonl"),
             ("failed", "json"))}


def read_jsonl(path: Path) -> list[dict]:
    """The records of a JSON-lines file; a line that does not parse (one torn by a crash) is skipped."""
    out = []
    for text in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(text))
        except ValueError:
            continue
    return out


def done_hands(out: Path) -> set[str]:
    """--resume: the hands already finished; lines of other hands (a crash between a hand's lines
    and its stats) are dropped from the lines file, and every file is rewritten without torn lines."""
    p = paths(out)
    if not p["hands"].exists():
        return set()
    hands = read_jsonl(p["hands"])
    done = {h["hand"] for h in hands}
    p["hands"].write_text("".join(json.dumps(h) + "\n" for h in hands), encoding="utf-8")
    for kind, key in (("lines", "seed"), ("rollouts", "hand")):
        if p[kind].exists():
            kept = [r for r in read_jsonl(p[kind]) if r.get(key) in done]
            p[kind].write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    return done


def gains(record: dict) -> float | None:
    """How much better the kept line is than the net's own play: mean end-of-turn value over the K
    scenarios minus that of the net's greedy line (in a branch: its greedy continuation after the
    same prefix); under a turn-2 scorer the same in its score. None for records without search stats.

    Records with an independent measurement (`gain`: kept minus own line on fresh scenarios and
    seeds, search_branch) use it. Older records fall back to the grading numbers, which flatter the
    kept line (it was picked as their maximum)."""
    search = record.get("search") or {}
    if "gain" in search:
        return search["gain"]
    if search.get("scorer", "value") != "value":
        if search.get("score") is None or search.get("own_score") is None:
            return None
        return search["score"] - search["own_score"]
    if search.get("value") is None or search.get("own_value") is None:
        return None
    return search["value"] - search["own_value"]


def gain_path(out: Path) -> Path:
    """The filtered book beside `out` (`--min-gain`): <stem>-gain.jsonl, with its own lines and meta."""
    return out.with_name(f"{out.stem}-gain.jsonl")


def write_book(out: Path, deck: Path, opponent: Path, records: list[dict], meta: dict) -> int:
    """Book `out`, its lines file and meta (with the relevant cards) from `records`; the number of states."""
    p = paths(out)
    from .book_policy import resolve

    lines = [(r, cc.puzzle_start(deck, resolve(r["opponent"]) if r.get("opponent") else opponent, r["script"],
                                 r["duel_seed"])) for r in records]
    relevant = ob.book_cards(records)
    book, _ = ob.build_book(lines, relevant)
    if not p["lines"].exists() or read_jsonl(p["lines"]) != records:
        p["lines"].write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    out.write_text("".join(json.dumps(e) + "\n" for e in book.values()), encoding="utf-8")
    p["meta"].write_text(json.dumps({**meta, "relevant": sorted(relevant)}), encoding="utf-8")
    return len(book)


def keeps(record: dict, min_gain: float, min_gain_se: float | None = None) -> bool:
    """A line for the gain book: its gain beats `min_gain` and, with `min_gain_se` z, also z standard
    errors of its independent measurement (when it has one)."""
    g = gains(record)
    if g is None or g <= min_gain:
        return False
    search = record.get("search") or {}
    if min_gain_se is None and "gain_ok" in search:  # racing decided it on its stage-2 scenarios
        return bool(search["gain_ok"])
    se = search.get("gain_se")
    return min_gain_se is None or se is None or g > min_gain_se * se


def build(out: Path, deck: Path, opponent: Path, meta: dict, min_gain: float | None = None,
          min_gain_se: float | None = None) -> dict:
    """The book of the lines file (opening_book.build_book) beside `out`, its meta and a report.
    With `min_gain`, also the book of the lines whose value beats the net's own line by more than
    that (`gain_path`): a line the net would play anyway (or nearly as well) teaches it nothing.
    Base and branch lines are filtered alike; a branch line whose base line is dropped keeps the
    base line's answers before the hand trap."""
    p = paths(out)
    records = read_jsonl(p["lines"])
    states = write_book(out, deck, opponent, records, meta)
    report = summarize(read_jsonl(p["hands"]))
    report.update(states=states, lines=len(records))
    if min_gain is not None:
        kept = [r for r in records if keeps(r, min_gain, min_gain_se)]
        report["gain"] = {"min_gain": min_gain, "min_gain_se": min_gain_se, "lines": len(kept),
                          "base": sum(r["trap"] is None for r in kept), "trap": sum(r["trap"] is not None for r in kept),
                          "base_of": sum(r["trap"] is None for r in records),
                          "trap_of": sum(r["trap"] is not None for r in records),
                          "states": write_book(gain_path(out), deck, opponent, kept,
                                               {**meta, "min_gain": min_gain, "min_gain_se": min_gain_se})}
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
        raced = [r for r in kept if r.get("race")]
        if raced:  # racing: how many lines went on, how often the choice was within the noise, gains that count
            out[branch]["race"] = {"stage1": mean(r["race"].get("stage1") for r in raced),
                                   "stage2": mean(r["race"].get("stage2") for r in raced),
                                   "noise": mean(r["race"].get("noise") for r in raced),
                                   "margin": mean(r["race"].get("margin") for r in raced),
                                   "gain": mean(r.get("gain") for r in raced),
                                   "gain_ok": mean(r.get("gain_ok") for r in raced)}
        scored = [r for r in kept if r.get("cands") and r.get("value_index") is not None]
        if scored:
            out[branch]["scorer"] = compare(scored)
    out["lines_per_hand"] = round(sum(out.get(b, {}).get("kept", 0) for b in ("base", "trap")) / max(1, len(hands)), 3)
    return out


def compare(rows: list[dict]) -> dict:
    """Turn-2 scorer vs value scorer on the same hands (each hand's candidates were scored both
    ways): how often they keep another line, and the kept lines of each (and the net's own) by
    end-of-turn value, turn-2 score, field at the end of turn 1, turn 2 lost, our field and both life
    points at turn 3."""
    def side(index_of) -> dict:
        picked = [r["cands"][index_of(r)] for r in rows if index_of(r) is not None]
        return {key: mean(c[key] for c in picked if c.get(key) is not None)
                for key in ("value", "score", "field", "lost", "won", "field3", "lp", "opp_lp", "capped", "decisions")}
    return {"hands": len(rows), "scorer": rows[0].get("scorer"),
            # another position after turn 1 (lines that differ in the order of answers count as the same)
            "differs": mean(not r.get("same_as_value", r["best_index"] == r["value_index"]) for r in rows),
            "picked_own": mean(bool(r.get("same_as_own", r["best_index"] == r["own_index"])) for r in rows),
            "value_picked_own": mean(bool(r.get("value_same_as_own", r["value_index"] == r["own_index"])) for r in rows),
            "kept": side(lambda r: r["best_index"]), "value_kept": side(lambda r: r["value_index"]),
            "own": side(lambda r: r["own_index"]),
            # fresh rollouts (--validate), on the hands that have them for all three lines
            "validate": {name: {key: oe.mean(r["validate"][name].get(key) for r in checked)
                                for key in ("t2", "lost", "field3", "lp", "opp_lp")}
                         for name in ("kept", "value_kept", "own")
                         for checked in [[r for r in rows if len(r.get("validate") or {}) == 3]] if checked},
            "validate_diff": paired([r["validate"]["kept"]["t2"] - r["validate"]["value_kept"]["t2"] for r in rows
                                     if len(r.get("validate") or {}) == 3 and not r.get("same_as_value")
                                     and r["validate"]["kept"]["t2"] is not None
                                     and r["validate"]["value_kept"]["t2"] is not None])}


def paired(diffs: list[float]) -> dict | None:
    """Mean of paired differences with its standard error."""
    if len(diffs) < 2:
        return None
    return {"n": len(diffs), "mean": round(statistics.fmean(diffs), 4),
            "se": round(statistics.stdev(diffs) / len(diffs) ** 0.5, 4)}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--deck", type=Path, required=True)
    p.add_argument("--deck-type", required=True, help="masterduelmeta deck type (hand traps of the counters data)")
    p.add_argument("--opponent", type=Path, default=rollout.REPO / "decks" / "maliss.ydk")
    p.add_argument("--opponents-file", type=Path,
                   help="real opponents instead (one deck per line, relative to the repo): scenario k plays one of "
                        "them, with its Extra Deck (needed for --score turn2/mix)")
    p.add_argument("--score", choices=oe.SCORERS, default="value",
                   help="value: end-of-turn-1 value; turn2: the net's value at our turn 3 after sampled turn 2s "
                        "(or the result); mix: --mix-weight * value + rest * turn2")
    p.add_argument("--rollouts", type=int, default=2, help="M: sampled turn 2s per candidate and scenario")
    p.add_argument("--mix-weight", type=float, default=0.5)
    p.add_argument("--t2-cap", type=int, default=250, help="sampled decisions per turn-2 rollout at most")
    p.add_argument("--validate", type=int, default=0,
                   help="fresh turn-2 rollouts per scenario of the kept, value-kept and own line (unbiased check)")
    p.add_argument("--validate-orders", type=int, default=0, help="scenarios of the validation (0: the search's K)")
    p.add_argument("--log-rollouts", action=argparse.BooleanOptionalAction, default=True,
                   help="write every turn-2 rollout (scoring and validation) to <out stem>-rollouts.jsonl, one row each: "
                        "hand, branch, phase, line, candidate, scenario k, rollout j, seed, score, early (decided), lost, won")
    p.add_argument("--validate-categories", action="store_true",
                   help="also validate a mid-ranked, a fragile and a big-board candidate against the own line")
    p.add_argument("--validate-chunk", type=int, default=6, help="validation scenarios at a time (memory)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--hands", type=int, default=200)
    p.add_argument("--first-hand", type=int, default=0)
    p.add_argument("--orders", type=int, default=4,
                   help="K: deck orders (scenarios) each candidate line is graded in (besides the beam's scenario 0)")
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--branch", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.02, help="weight of the log-probability in the beam's rank")
    p.add_argument("--candidates", type=int, default=4, help="best beam lines judged over the K orders")
    p.add_argument("--max-choices", type=int, default=120)
    p.add_argument("--traps", type=int, default=4, help="hand traps the branches rotate through (0: no branches)")
    p.add_argument("--traps-file", type=Path,
                   help="the deck's hand traps (JSON list of passcodes, from --write-traps) instead of the counters data")
    p.add_argument("--write-traps", type=Path, help="write the deck's resolved hand traps there and exit")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--min-gain", type=float, default=None,
                   help="also build <out stem>-gain.jsonl from the lines whose mean value beats the net's own line "
                        "by more than this")
    p.add_argument("--min-gain-se", type=float, default=None,
                   help="also require the gain to beat this many standard errors of its independent measurement")
    p.add_argument("--gain-orders", type=int, default=8,
                   help="further scenarios (fresh seeds) the kept line's gain over the own line is measured in")
    p.add_argument("--gain-rollouts", type=int, default=1, help="turn-2 rollouts per scenario there")
    p.add_argument("--race", action="store_true",
                   help="racing instead of --orders x --rollouts: every candidate in --race-k1 scenarios x 1 rollout, "
                        "the close ones (and the own line) up to --race-k2")
    p.add_argument("--race-k1", type=int, default=8)
    p.add_argument("--race-k2", type=int, default=16)
    p.add_argument("--race-se", type=float, default=2.0, help="stage 1 drops what the best beats by this many SE")
    p.add_argument("--race-frac", type=float, default=0.75,
                   help="... or beats in this share of the scenarios with a mean lead above --race-margin")
    p.add_argument("--race-margin", type=float, default=0.0)
    p.add_argument("--race-c", type=float, default=1.0, help="the gain over the own line counts above c SE")
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
    meta = {"deck": str(args.deck), "deck_type": args.deck_type, "opponent": str(args.opponent), "sources": "turn1",
            "score": args.score}
    if args.score != "value" and not args.opponents_file and not args.build_only:
        raise SystemExit("--score turn2/mix needs --opponents-file (an opponent that can play turn 2)")
    if args.build_only:
        print(json.dumps(build(args.out, args.deck, args.opponent, meta, args.min_gain, args.min_gain_se)), flush=True)
        return
    torch.set_num_threads(args.torch_threads)
    table = CardTable.from_cdb()
    card_name = {c.code: c.name for c in table.cards}
    types = {c.code: c.type for c in table.cards}
    is_extra = lambda code: bool(types.get(code, 0) & cc.TYPE_EXTRA)  # noqa: E731
    center = {c.code for c in table.cards if cc.wants_center(c.text)}
    decklist = worlds.read_decklist(args.deck)
    opponent_cards = worlds.read_decklist(args.opponent)[0]
    opponents = []
    if args.opponents_file:
        for line in args.opponents_file.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                path = (rollout.REPO / line.strip()).resolve()
                opponents.append((path, *worlds.read_decklist(path)))
    if args.traps_file:
        rules = read_traps(args.traps_file)
    else:
        resolve = sd.resolver(cb.name_index(table), set(decklist[0]) | set(decklist[1]), card_name)
        rules = {t: v for t, v in sd.interruptions(args.deck_type, resolve).items()
                 if card_name.get(t) in INTERRUPTERS}
    if args.write_traps:
        write_traps(args.write_traps, rules)
        return
    traps = trap_pool(rules, card_name, args.traps)
    print(f"hand traps: {[card_name[t] for t in traps]}", flush=True)
    model = load_checkpoint(args.checkpoint, args.device, dropout=0.0)
    evaluator = Evaluator(model_fn(model, args.device))
    net = Net(evaluator, model.card_features.index_map())
    settings = Settings(args.beam, args.branch, args.alpha, args.candidates, args.max_choices, args.score,
                        args.rollouts, args.mix_weight, args.t2_cap, args.validate,
                        args.validate_orders, args.log_rollouts, args.validate_categories, args.validate_chunk,
                        args.gain_orders, args.gain_rollouts, args.race, args.race_k1, args.race_k2, args.race_se,
                        args.race_frac, args.race_margin, args.race_c)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    p = paths(args.out)
    done = done_hands(args.out) if args.resume else set()
    if not args.resume:
        for path in (p["lines"], p["hands"], p["rollouts"]):
            path.write_text("", encoding="utf-8")
    p["meta"].write_text(json.dumps(meta), encoding="utf-8")
    hands = []
    for t in range(args.first_hand, args.first_hand + args.hands):
        hid = f"t1-{args.deck.stem}-{args.seed}-{t}"
        if hid not in done:
            hands.append(Hand(hid, args.deck, args.deck_type, args.opponent, decklist, opponent_cards,
                              random_hand(decklist, args.seed, args.deck, t),
                              traps[t % len(traps)] if traps else None, args.orders, opponents))
    print(f"{len(hands)} hands to search ({len(done)} done)", flush=True)
    began = time.perf_counter()
    failed = []
    with ThreadPoolExecutor(max(1, args.workers)) as pool, open(p["lines"], "a", encoding="utf-8") as lines_out, \
            open(p["hands"], "a", encoding="utf-8") as hands_out, \
            open(p["rollouts"], "a", encoding="utf-8") as rollouts_out:
        futures = {pool.submit(search_hand, hand, net, settings, center, is_extra): hand for hand in hands}
        for n, future in enumerate(as_completed(futures)):  # written as they finish
            hand = futures[future]
            try:
                records, stats = future.result()
            except Exception as error:  # one broken hand must not stop the run; it is retried on --resume
                failed.append(hand.id)
                print(f"{n + 1}/{len(futures)} {hand.id} error {str(error)[:160]}", flush=True)
                continue
            raw = [{"hand": stats["hand"], "branch": branch, **r} for branch in ("base", "trap")
                   for key in ("score_raw", "validate_raw") for r in (stats.get(branch) or {}).pop(key, [])]
            # each file in one write: a crash tears at most its last line (read_jsonl skips it); the
            # hand's stats come last, so a hand counts as done only with its lines and rollouts
            lines_out.write("".join(json.dumps(record) + "\n" for record in records))
            lines_out.flush()
            rollouts_out.write("".join(json.dumps(r) + "\n" for r in raw))
            rollouts_out.flush()
            hands_out.write(json.dumps(stats) + "\n")
            hands_out.flush()
            b = stats["base"]
            score = f"score {b.get('score')} (own {b.get('own_score')}) " if b.get("scorer") else ""
            race_note = f"race {b['race'].get('stage2')}/{b['race'].get('stage1')} noise {b['race'].get('noise')} " \
                if b.get("race") else ""
            print(f"{n + 1}/{len(futures)} {stats['hand']} value {b.get('value')} (own {b.get('own_value')}) {score}"
                  f"gain {b.get('gain')} {race_note}field {b.get('field0')} (own {b.get('own_field0')}) pick {b.get('pick')} "
                  f"trap {card_name.get(hand.trap, '-')[:12]} lines {len(records)} {b.get('seconds')}s "
                  f"[{time.perf_counter() - began:.0f}s, {evaluator.rows / max(1, evaluator.calls):.1f} rows/call]",
                  flush=True)
    evaluator.close()
    print(json.dumps(build(args.out, args.deck, args.opponent, meta, args.min_gain, args.min_gain_se)), flush=True)
    if failed:  # not "done": the driver (or a rerun with --resume) retries these hands
        p["failed"].write_text(json.dumps(failed) + "\n", encoding="utf-8")
        print(f"{len(failed)} hands failed: {failed[:10]}", flush=True)
        raise SystemExit(FAILED_HANDS)
    p["failed"].unlink(missing_ok=True)


if __name__ == "__main__":
    main()
