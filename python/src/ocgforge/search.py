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


class GreedyPolicy(rollout.ModelPolicy):
    """ModelPolicy that samples, plus greedy(rows): each row's most likely answer."""

    def greedy(self, rows):
        sample, self.sample = self.sample, False
        try:
            return self(rows)[0]
        finally:
            self.sample = sample


def match(model, decks: list[Path], duels: int, k: int, device: str, seed: int = 0) -> dict:
    """`duels` duels of SearchPolicy against the plain sampling policy (same model), alternating seats."""
    rng = random.Random(seed)
    card_index = model.card_features.index_map()
    policy = GreedyPolicy(model, device, sample=True, seed=seed)
    searcher = SearchPolicy(policy, card_index, k)
    wins = losses = draws = 0
    started = time.time()
    for n in range(duels):
        spec = DuelSpec(rng.choice(decks), rng.choice(decks), rng.getrandbits(63), rng.randrange(2))
        search_seat = n % 2
        duel, actions = spec.start(), []
        searcher.plan = []
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
    args = parser.parse_args(argv)
    model = load_checkpoint(args.checkpoint, args.device, dropout=0.0)
    print(match(model, args.decks, args.duels, args.k, args.device, args.seed), flush=True)


if __name__ == "__main__":
    main()
