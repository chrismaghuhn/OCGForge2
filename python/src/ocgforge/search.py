"""Best-of-K turn search: the model plays the rest of its turn K ways and keeps the best line.

Our engine has no state copy, but a duel is fully determined by its seed, decks, starting player
and answers (VectorEnv.info gives all of them), so a line is tried by replaying the duel's answers
into a fresh `_core.Duel` and playing on. From a decision on the searching player's own turn, rollout
0 plays the model's most likely answers and rollouts 1..K-1 sample them; each runs until the turn
passes (or the duel ends) and is scored by the outcome or, mid-duel, by the model's value of the
position from the searching player's side. The best line is a better move than the plain policy's
when search helps, and a training target for it (expert iteration).

    python -m ocgforge.search --checkpoint runs/focus-large-s0/latest --duels 40   # search vs policy
"""

from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import rollout


@dataclass(frozen=True)
class DuelSpec:
    """Everything that fixes a duel besides its answers."""
    deck0: Path
    deck1: Path
    seed: int
    starting_player: int

    def start(self):
        from ._core import Duel

        cache = rollout.REPO / ".cache"
        return Duel(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(self.deck0), str(self.deck1),
                    self.seed, self.starting_player)


def replay(spec: DuelSpec, actions: list[str]):
    duel = spec.start()
    for key in actions:
        duel.act(key)
    return duel


@dataclass
class SearchResult:
    line: list[str]  # answers from the search point to the end of the turn (both players' answers)
    value: float  # the best line's score for the searching player (-1..1)
    greedy_value: float  # the most likely line's score
    values: list[float] = field(default_factory=list)


def search_turn(spec: DuelSpec, actions: list[str], policy, card_index, k: int = 16, max_decisions: int = 80,
                pool: ThreadPoolExecutor | None = None) -> SearchResult:
    """Play the rest of the current turn k ways from the decision after `actions`. `policy` must sample
    (rollouts 1..k-1); rollout 0 follows each decision's most likely answer."""
    pool = pool or ThreadPoolExecutor(max(1, k))
    duels = list(pool.map(lambda _: replay(spec, actions), range(k)))
    me, turn = duels[0].player, duels[0].turn
    lines: list[list[str]] = [[] for _ in range(k)]
    scores: list[float | None] = [None] * k
    for _ in range(max_decisions):
        open_ = [i for i in range(k) if scores[i] is None]
        for i in list(open_):
            duel = duels[i]
            if duel.finished:
                scores[i] = 0.0 if duel.winner is None or duel.winner > 1 else (1.0 if duel.winner == me else -1.0)
                open_.remove(i)
        if not open_:
            break
        rows = [duels[i].encode(card_index) for i in open_]
        choices, logprobs, values = policy(rows)
        greedy = policy.greedy(rows) if 0 in open_ else None
        moves = []
        for n, i in enumerate(open_):
            duel = duels[i]
            if duel.turn != turn:  # the turn has passed: score the position the turn left
                scores[i] = values[n] if duel.player == me else -values[n]
                continue
            choice = greedy[open_.index(0)] if i == 0 else choices[n]
            moves.append((i, duel.keys()[choice]))
        for i, key in moves:
            lines[i].append(key)
        list(pool.map(lambda move: duels[move[0]].act(move[1]), moves))
    for i in range(k):  # cut off by max_decisions: score as it stands
        if scores[i] is None:
            if duels[i].finished:
                winner = duels[i].winner
                scores[i] = 0.0 if winner is None or winner > 1 else (1.0 if winner == me else -1.0)
            else:
                _, _, values = policy([duels[i].encode(card_index)])
                scores[i] = values[0] if duels[i].player == me else -values[0]
    best = max(range(k), key=lambda i: scores[i])
    return SearchResult(lines[best], scores[best], scores[0], scores)


def honest_search(spec: DuelSpec, duel, policy, card_index, lines: int = 8, worlds_per_line: int = 8,
                  max_decisions: int = 80, seed: int = 0, pool: ThreadPoolExecutor | None = None,
                  make_world=None, segment: int = 0) -> SearchResult:
    """Search without peeking: from `duel` at the searching player's first idle command of its turn,
    build `worlds_per_line` worlds from what that player can see (ocgforge.worlds), generate `lines`
    candidate plans (rollout 0 the most likely answers, the others sampled, each in its own world),
    replay every plan in every world (the player's answers follow the plan while it fits, then the
    most likely answer; the opponent's are sampled) and keep the plan with the best mean score.
    The plan is the player's answers only, in canonical form (worlds.canonical).

    By default the worlds are built at the current point, which must be the start of the player's
    main phase; `make_world(i)` builds world i elsewhere (e.g. worlds.world_now mid-turn) or returns
    None. With `segment` (worlds.phase_group) a rollout ends when the duel leaves that phase group
    instead of at the end of the turn."""
    from . import worlds as w

    pool = pool or ThreadPoolExecutor(16)
    me = duel.player
    snapshot = duel.snapshot()
    decklists = {0: w.read_decklist(spec.deck0), 1: w.read_decklist(spec.deck1)}
    rng = random.Random(seed)
    if make_world is None:
        scripts = [w.sample_world(snapshot, me, decklists, random.Random(rng.getrandbits(32)))
                   for _ in range(max(lines, worlds_per_line))]
        make_world = lambda i: w.start_world(spec, scripts[i], me, seed + i)  # noqa: E731

    def run(plans: list[list[str] | None], world_ids: list[int], greedy_mine: list[bool]):
        """Play each (plan, world) to the end of the turn; returns per-run scores and my answers."""
        duels = list(pool.map(make_world, world_ids))
        n = len(duels)
        mine: list[list[str]] = [[] for _ in range(n)]
        steps = [0] * n
        scores: list[float | None] = [None if d is not None else 0.0 for d in duels]
        turn = [d.turn if d is not None else 0 for d in duels]
        for _ in range(max_decisions):
            open_ = [i for i in range(n) if scores[i] is None]
            for i in list(open_):
                d = duels[i]
                if d.finished:
                    scores[i] = 0.0 if d.winner is None or d.winner > 1 else (1.0 if d.winner == me else -1.0)
                    open_.remove(i)
            if not open_:
                break
            rows = [duels[i].encode(card_index) for i in open_]
            sampled, _, values = policy(rows)
            greedy = policy.greedy(rows)
            moves = []
            for n_, i in enumerate(open_):
                d = duels[i]
                if d.turn != turn[i] or (segment and w.phase_group(d.phase) != segment):
                    scores[i] = values[n_] if d.player == me else -values[n_]
                    continue
                keys = d.keys()
                key = None
                if d.player == me and plans[i] is not None and steps[i] < len(plans[i]):
                    key = w.match(plans[i][steps[i]], keys)
                    steps[i] += 1
                if key is None:
                    choice = greedy[n_] if (d.player == me and (plans[i] is not None or greedy_mine[i])) else sampled[n_]
                    key = keys[choice]
                if d.player == me:
                    mine[i].append(w.canonical(key))
                moves.append((i, key))
            list(pool.map(lambda move: duels[move[0]].act(move[1]), moves))
        for i in range(n):
            if scores[i] is None:
                d = duels[i]
                if d.finished:
                    scores[i] = 0.0 if d.winner is None or d.winner > 1 else (1.0 if d.winner == me else -1.0)
                else:
                    _, _, v = policy([d.encode(card_index)])
                    scores[i] = v[0] if d.player == me else -v[0]
        return scores, mine

    # Candidate plans: one rollout per world, the first one greedy.
    _, candidates = run([None] * lines, list(range(lines)), [i == 0 for i in range(lines)])
    plans = [p for p in candidates if p]
    unique = list(dict.fromkeys(tuple(p) for p in plans))
    if not unique:  # no world could be built (or the player had nothing to do)
        return SearchResult([], 0.0, 0.0, [])
    # Every plan in every world.
    grid_plans = [list(p) for p in unique for _ in range(worlds_per_line)]
    grid_worlds = [j for _ in unique for j in range(worlds_per_line)]
    scores, _ = run(grid_plans, grid_worlds, [False] * len(grid_plans))
    means = [sum(scores[k * worlds_per_line:(k + 1) * worlds_per_line]) / worlds_per_line for k in range(len(unique))]
    best = max(range(len(unique)), key=lambda k: means[k])
    greedy_k = unique.index(tuple(candidates[0])) if candidates[0] else best
    return SearchResult(list(unique[best]), means[best], means[greedy_k], means)


class SearchPolicy:
    """Plays like `policy`, but at its first decision of each own turn searches the turn and then
    follows the best line while the duel stays on it (searching again where it leaves it)."""

    def __init__(self, policy, card_index, k: int = 16, pool: ThreadPoolExecutor | None = None):
        self.policy, self.card_index, self.k = policy, card_index, k
        self.pool = pool or ThreadPoolExecutor(k)
        self.plan: list[str] = []
        self.searches = 0
        self.gains: list[float] = []

    def choose(self, spec: DuelSpec, actions: list[str], duel) -> str:
        keys = duel.keys()
        if duel.turn_player != duel.player:  # the opponent's turn: plain policy
            self.plan = []
            return keys[self.policy([duel.encode(self.card_index)])[0][0]]
        if not self.plan or self.plan[0] not in keys:
            result = search_turn(spec, actions, self.policy, self.card_index, self.k, pool=self.pool)
            self.plan = result.line
            self.searches += 1
            self.gains.append(result.value - result.greedy_value)
        if self.plan and self.plan[0] in keys:
            return self.plan.pop(0)
        return keys[self.policy([duel.encode(self.card_index)])[0][0]]

    def observe_other(self, key: str) -> None:
        """The opponent answered during our planned line: follow the plan only if it expected that."""
        if self.plan and self.plan[0] == key:
            self.plan.pop(0)
        else:
            self.plan = []


class HonestSearchPolicy:
    """Plays like `policy`, but searches honestly (honest_search) on its own turns and follows the
    plan while each next answer is legal (else, and on the opponent's turn, the policy's most likely
    answer). Without `phases` it plans the whole turn at its first idle command. With `phases` it
    plans each phase on its own: Main Phase 1 there, the Battle Phase at its first battle command and
    Main Phase 2 at its first idle command, each scored where the phase ends. Worlds for the later
    phases are built at the turn's start and replayed through this turn's answers (worlds.world_now),
    so this-turn effects and what the opponent revealed meanwhile carry over."""

    def __init__(self, policy, card_index, lines: int = 8, worlds_per_line: int = 8,
                 pool: ThreadPoolExecutor | None = None, phases: bool = False):
        self.policy, self.card_index = policy, card_index
        self.lines, self.worlds_per_line, self.phases = lines, worlds_per_line, phases
        self.pool = pool or ThreadPoolExecutor(16)
        self.plan: list[str] = []
        self.turn = None
        self.searched: set[int] = set()
        self.start = None
        self.start_index = 0
        self.searches = 0
        self.failed = 0
        self.gains: list[float] = []

    def search(self, spec: DuelSpec, actions: list[str], duel, segment: int) -> None:
        from . import worlds as w

        make_world = None
        if segment > 1:
            now, history, me = duel.snapshot(), actions[self.start_index:], duel.player
            decklists = {0: w.read_decklist(spec.deck0), 1: w.read_decklist(spec.deck1)}
            base = 1000 * (self.searches + 1)
            make_world = lambda i: w.world_now(spec, self.start, now, history, me, decklists,  # noqa: E731
                                               random.Random(base + i), base + 10 * i)
        result = honest_search(spec, duel, self.policy, self.card_index, self.lines, self.worlds_per_line,
                               seed=self.searches, pool=self.pool, make_world=make_world, segment=segment)
        self.searches += 1
        if not result.values:
            self.failed += 1
        self.plan = result.line
        self.gains.append(result.value - result.greedy_value)

    def choose(self, spec: DuelSpec, actions: list[str], duel) -> str:
        from . import worlds as w

        keys = duel.keys()
        mine = duel.turn_player == duel.player
        if mine and duel.turn != self.turn and keys[0].startswith("idle"):
            self.turn, self.start, self.start_index, self.searched = duel.turn, duel.snapshot(), len(actions), {1}
            self.search(spec, actions, duel, 1 if self.phases else 0)
        elif self.phases and mine and duel.turn == self.turn and keys[0].startswith(("idle", "battle")):
            group = w.phase_group(duel.phase)
            if group in (2, 3) and group not in self.searched:
                self.searched.add(group)
                self.search(spec, actions, duel, group)
        if mine and duel.turn == self.turn and self.plan:
            key = w.match(self.plan.pop(0), keys)
            if key is not None:
                return key
            self.plan = []  # the duel left the plan
        return keys[self.policy.greedy([duel.encode(self.card_index)])[0]]

    def observe_other(self, key: str) -> None:
        pass


class GreedyPolicy(rollout.ModelPolicy):
    """ModelPolicy that samples, plus greedy(rows): each row's most likely answer."""

    def greedy(self, rows):
        sample, self.sample = self.sample, False
        try:
            return self(rows)[0]
        finally:
            self.sample = sample


def match(model, decks: list[Path], duels: int, k: int, device: str, seed: int = 0, honest: bool = False,
          phases: bool = False) -> dict:
    """`duels` duels of a searching player against the plain sampling policy (same model), alternating
    seats: oracle search (SearchPolicy) or, with honest=True, HonestSearchPolicy with k lines x k worlds."""
    rng = random.Random(seed)
    card_index = model.card_features.index_map()
    policy = GreedyPolicy(model, device, sample=True, seed=seed)
    searcher = (HonestSearchPolicy(policy, card_index, k, k, phases=phases) if honest or phases
                else SearchPolicy(policy, card_index, k))
    wins = losses = draws = 0
    started = time.time()
    for n in range(duels):
        spec = DuelSpec(rng.choice(decks), rng.choice(decks), rng.getrandbits(63), rng.randrange(2))
        search_seat = n % 2
        duel, actions = spec.start(), []
        searcher.plan = []
        searcher.turn = None
        while not duel.finished and duel.decision_count < 3000:
            if duel.player == search_seat:
                key = searcher.choose(spec, actions, duel)
            else:
                key = duel.keys()[policy([duel.encode(card_index)])[0][0]]
                if duel.turn_player == search_seat:
                    searcher.observe_other(key)
            duel.act(key)
            actions.append(key)
        winner = duel.winner if duel.finished else None
        if winner is None or winner > 1:
            draws += 1
        elif winner == search_seat:
            wins += 1
        else:
            losses += 1
        print(f"duel {n + 1}: {'search wins' if winner == search_seat else 'draw' if winner is None or winner > 1 else 'policy wins'}"
              f" ({wins}-{losses}-{draws}, {searcher.searches} searches, {time.time() - started:.0f}s)", flush=True)
    gain = sum(searcher.gains) / max(1, len(searcher.gains))
    return {"wins": wins, "losses": losses, "draws": draws, "searches": searcher.searches,
            "failed_searches": getattr(searcher, "failed", 0),
            "mean_value_gain": round(gain, 4), "seconds": round(time.time() - started)}


def main(argv=None) -> None:
    from .model import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decks", type=Path, nargs="+", default=sorted((rollout.REPO / "decks").glob("*.ydk")))
    parser.add_argument("--duels", type=int, default=20)
    parser.add_argument("--k", type=int, default=16, help="rollouts per search")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--honest", action="store_true", help="honest search (k plans x k sampled worlds)")
    parser.add_argument("--phases", action="store_true", help="honest search per phase (Main 1, Battle, Main 2)")
    args = parser.parse_args(argv)
    model = load_checkpoint(args.checkpoint, args.device, dropout=0.0)
    print(match(model, args.decks, args.duels, args.k, args.device, args.seed, args.honest,
                args.phases), flush=True)


if __name__ == "__main__":
    main()
