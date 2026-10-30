"""Playing duels in the C++ VectorEnv: self-play rollouts for PPO and matches between two policies.

A policy maps a list of pending decisions (features.py dicts) to (choices, logprobs, values), one
entry per decision; policies with an `on_batch` method also take a padded VectorEnv.recv_batch
batch directly, which saves building per-decision dicts. Every env always has a pending decision;
finished duels restart by themselves.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import features
from .cards import CardFeatures
from .model import Model

REPO = Path(__file__).resolve().parents[3]

Policy = Callable[[Sequence[dict]], tuple[list[int], list[float], list[float]]]


def vector_env(card_features: CardFeatures, decks: Sequence[Path | str], num_envs: int, seed: int = 0,
               threads: int = 0, max_decisions: int = 5000, cache: Path = REPO / ".cache",
               skip_forced: bool = False):
    """A VectorEnv whose card indices match `card_features` (e.g. model.card_features). With
    skip_forced, decisions with a single candidate are answered in C++ and never handed out
    (counted in env.forced_decisions)."""
    from ._core import VectorEnv

    if card_features.size > features.MAX_COMPACT_CARDS:
        raise ValueError(f"{card_features.size} cards do not fit the compact int16 card indices of features.row")
    return VectorEnv(str(cache / "cardscripts"), str(cache / "card_data.tsv"), [str(d) for d in decks],
                     card_features.index_map(), num_envs=num_envs, seed=seed, threads=threads,
                     max_decisions=max_decisions, skip_forced=skip_forced)


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, decisions: Sequence[dict]) -> tuple[list[int], list[float], list[float]]:
        return self._choose([len(d["candidate_kind"]) for d in decisions])

    def on_batch(self, batch: dict) -> tuple[list[int], list[float], list[float]]:
        return self._choose([int(c) for c in batch["candidate_count"]])

    def _choose(self, counts: list[int]) -> tuple[list[int], list[float], list[float]]:
        return [int(self.rng.integers(c)) for c in counts], [-float(np.log(c)) for c in counts], [0.0] * len(counts)


class ModelPolicy:
    """Samples from the model's softmax (or plays the argmax with sample=False), in eval mode."""

    def __init__(self, model: Model, device: torch.device | str = "cpu", sample: bool = True, seed: int = 0,
                 bf16: bool = False):
        """`bf16`: autocast on CUDA, to match a PPOConfig(bf16=True) update."""
        self.model, self.device, self.sample, self.bf16 = model, device, sample, bf16
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, decisions: Sequence[dict]) -> tuple[list[int], list[float], list[float]]:
        return self._choose(features.collate(decisions, self.device))

    def on_batch(self, batch: dict) -> tuple[list[int], list[float], list[float]]:
        return self._choose(features.from_padded(batch, self.device))

    @torch.no_grad()
    def _choose(self, inputs: dict) -> tuple[list[int], list[float], list[float]]:
        self.model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16 and str(self.device).startswith("cuda")):
            scores, values = self.model(inputs)
        values = values.float()
        logprobs = torch.log_softmax(scores.float(), dim=-1).cpu()
        if self.sample:
            choices = torch.multinomial(logprobs.exp(), 1, generator=self.generator).squeeze(1)
        else:
            choices = logprobs.argmax(-1)
        picked = logprobs.gather(1, choices.unsqueeze(1)).squeeze(1)
        return choices.tolist(), picked.tolist(), values.cpu().tolist()


@dataclass
class RolloutStats:
    duels: int = 0
    errors: int = 0
    decisions: int = 0
    wins: list[int] = field(default_factory=lambda: [0, 0, 0])  # player 0, player 1, draw (self-play duels)
    league_duels: int = 0
    learner_decisions: int = 0  # decisions kept for training
    league_decisions: int = 0  # of those, played against a league opponent
    error_messages: list[str] = field(default_factory=list)
    end_turns: list[int] = field(default_factory=list)  # turn count at the end of each finished duel


@dataclass(frozen=True)
class Tempo:
    """Wins lose weight the longer the duel ran: full weight up to turn `start` (turns of both
    players), then `slope` less per turn, never below `floor` (so a late win stays a win). Losses
    stay -1 unless `losses`: shrinking them too would pay a lost position for dragging the duel out."""
    start: int = 5
    slope: float = 0.0
    floor: float = 0.6
    losses: bool = False

    def weight(self, turn: int) -> float:
        return 1.0 if turn <= self.start else max(self.floor, 1.0 - self.slope * (turn - self.start))

    def apply(self, result: float, turn: int) -> float:
        return result * self.weight(turn) if result > 0 or self.losses else result


def _mark_fresh(fresh: list[bool | None], decisions: Sequence[dict]) -> None:
    """On the first look at each env: is its duel at its first decision (not left over)?"""
    for i, decision in enumerate(decisions):
        if fresh[i] is None:
            fresh[i] = decision["duel_decision"] == 0


def _result_for(player: int, winner: int) -> float:
    return 0.0 if winner == 2 else (1.0 if winner == player else -1.0)


def collect(env, policy: Policy, duels: int) -> tuple[list[tuple[list[dict], float]], RolloutStats]:
    """Self-play until `duels` duels have finished. Returns one (decisions, result) trajectory per
    seat of every finished duel, ready for ppo.build_steps. Only duels played from their first
    decision count: duels already running when collect starts (left over from an earlier call) are
    finished but not recorded, and duels still running at the end stay in the env. Failed duels
    (including ones that hit the decision limit) are dropped."""
    pending: dict[tuple[int, int], list[dict]] = {}
    trajectories: list[tuple[list[dict], float]] = []
    stats = RolloutStats()
    fresh: list[bool | None] = [None] * len(env)  # we saw env i's current duel from its first decision
    while stats.duels + stats.errors < duels:
        decisions = env.observe()
        _mark_fresh(fresh, decisions)
        choices, logprobs, values = policy(decisions)
        for i, (decision, choice, logprob, value) in enumerate(zip(decisions, choices, logprobs, values)):
            decision.update(chosen=choice, logprob=logprob, value=value)
            pending.setdefault((i, decision["player"]), []).append(decision)
        done, winners, errors = env.step(choices)
        stats.decisions += len(decisions)
        for i in (i for i, d in enumerate(done) if d):
            seats = [pending.pop((i, p), []) for p in (0, 1)]
            was_fresh, fresh[i] = fresh[i], True
            if not was_fresh:
                continue
            if errors[i]:
                stats.errors += 1
                stats.error_messages.append(errors[i])
                continue
            stats.duels += 1
            stats.wins[min(winners[i], 2)] += 1
            for player, seat in enumerate(seats):
                if seat:
                    trajectories.append((seat, _result_for(player, winners[i])))
    return trajectories, stats


class League:
    """Frozen earlier models the learner plays against in a share of the envs (the rest is plain
    self-play), so it keeps beating what it already beat instead of chasing its latest self.

    Who comes up next is prioritized fictitious self-play: weight (1 - score)^2 on the learner's
    smoothed score over its last `window` duels against each opponent (so the weights follow the
    current learner), mixed with a `uniform` share of uniform picks so no single opponent takes
    over. The pool holds the `size` newest models plus up to `anchors` anchor models (added with
    anchor=True, e.g. every few hundred iterations) for a long memory; the first anchor never leaves."""

    def __init__(self, share: float, size: int, seed: int = 0, window: int = 200, uniform: float = 0.2,
                 anchors: int = 4):
        self.share, self.size, self.window, self.uniform, self.max_anchors = share, size, window, uniform, anchors
        self.policies: dict[str, Policy] = {}
        self.results: dict[str, deque[float]] = {}  # name -> learner scores of the latest duels
        self.recent: list[str] = []
        self.anchors: list[str] = []
        self.rng = np.random.default_rng(seed)

    def add(self, name: str, policy: Policy, anchor: bool = False) -> None:
        self.policies[name] = policy
        self.results.setdefault(name, deque(maxlen=self.window))
        (self.anchors if anchor else self.recent).append(name)
        while len(self.recent) > self.size:
            self._drop(self.recent.pop(0))
        while len(self.anchors) > max(self.max_anchors, 1):
            self._drop(self.anchors.pop(1))

    def _drop(self, name: str) -> None:
        del self.policies[name], self.results[name]

    def pick(self) -> str | None:
        if not self.policies:
            return None
        names = list(self.policies)
        scores = np.array([(sum(self.results[n]) + 1) / (len(self.results[n]) + 2) for n in names])
        weights = (1 - scores) ** 2
        p = (1 - self.uniform) * weights / weights.sum() + self.uniform / len(names)
        return names[self.rng.choice(len(names), p=p / p.sum())]

    def result(self, name: str, learner_score: float) -> None:
        if name in self.results:
            self.results[name].append(learner_score)

    def scores(self) -> dict[str, float]:
        """Learner score against each opponent over its latest duels."""
        return {n: round(sum(r) / len(r), 3) for n, r in self.results.items() if r}


CARD_ACTIONS = ("summon", "special_summon", "set_monster", "set_spell_trap", "activate", "chain_activate", "attack")
_CARD_ACTION_IDS = np.array([features.CANDIDATE_KINDS.index(k) for k in CARD_ACTIONS])


class CardUsage:
    """How often each card was offered as an action (summon, set, activate, attack) and how often the
    learner took it: which cards the policy plays and which it leaves alone."""

    def __init__(self, cards: int):
        self.offered = np.zeros(cards, np.int64)  # decisions in which the card had an action
        self.chosen = np.zeros(cards, np.int64)

    def add(self, batch: dict, rows: list[int], choices: list[int]) -> None:
        if not rows:
            return
        kind = batch["candidate_kind"][rows]
        source = batch["candidate_source"][rows]
        tokens = batch["token_card"][rows]
        card = np.where(source >= 0, np.take_along_axis(tokens, np.maximum(source, 0), 1), batch["candidate_card"][rows])
        action = np.isin(kind, _CARD_ACTION_IDS) & (card > 1)
        row = np.nonzero(action)[0]
        once = np.unique(row * len(self.offered) + card[action]) % len(self.offered)  # once per decision
        np.add.at(self.offered, once, 1)
        picked = np.asarray(choices)
        taken = action[np.arange(len(rows)), picked]
        np.add.at(self.chosen, card[np.arange(len(rows)), picked][taken], 1)

    def reset(self) -> None:
        self.offered[:] = 0
        self.chosen[:] = 0


_EXTRA_TYPE_IDS = (6 + 2, 13 + 2, 23 + 2, 26 + 2)  # Fusion, Synchro, Xyz, Link as CardFeatures.types ids
_MONSTER_ZONE = features.ZONES.index("monster_zone")
_TURN = features.GLOBAL_NUMERIC.index("turn")
_OWN_TURN = features.GLOBAL_NUMERIC.index("isOwnTurn")


class ExtraDeckUsage:
    """The learner's Extra Deck monsters: in how many duels each one reached its field ("played") and
    how often each one stood there when the learner's turn ended ("end board"), counted over all own
    turns and separately for the first turn of the duel (the board a first player sets up). The end
    of a turn is the learner's last decision in it, seen when its next decision is in another turn;
    a duel that ends in the learner's own turn records no end board for that turn."""

    def __init__(self, card_features):
        self.extra = np.isin(card_features.types, _EXTRA_TYPE_IDS).any(axis=1)
        n = len(self.extra)
        self.played = np.zeros(n, np.int64)
        self.end_board = np.zeros(n, np.int64)
        self.first_board = np.zeros(n, np.int64)
        self.turns = 0  # own turns with a recorded end board
        self.first_turns = 0
        self.seen: dict[tuple[int, int], set[int]] = {}  # (env, player) -> extra cards on its field this duel
        self.last: dict[tuple[int, int], tuple[int, list[int]]] = {}  # -> (own turn, extra cards on its field)

    def add(self, ids: list[int], batch: dict, rows: list[int]) -> None:
        """Counts the learner's `rows` of a batch; every row of it (a league opponent's too) may
        start a new duel in its env, which forgets that env's previous duel."""
        for n in np.flatnonzero(batch["duel_decision"] == 0):
            for player in (0, 1):
                self.seen.pop((ids[n], player), None)
                self.last.pop((ids[n], player), None)
        if not rows:
            return
        players = batch["player"]
        cards = batch["token_card"]
        own = (batch["token_side"] == 0) & (batch["token_zone"] == _MONSTER_ZONE)
        for n in rows:
            key = (ids[n], int(players[n]))
            field = [int(c) for c in cards[n][own[n]] if c > 1 and self.extra[c]]
            seen = self.seen.setdefault(key, set())
            for card in set(field) - seen:
                self.played[card] += 1
                seen.add(card)
            turn = round(float(batch["global_numeric"][n][_TURN]) * 20)
            previous = self.last.get(key)
            if previous is not None and previous[0] != turn:
                self._end_turn(*previous)
                del self.last[key]
            if batch["global_numeric"][n][_OWN_TURN] > 0.5:
                self.last[key] = (turn, field)

    def _end_turn(self, turn: int, field: list[int]) -> None:
        self.turns += 1
        np.add.at(self.end_board, field, 1)
        if turn == 1:
            self.first_turns += 1
            np.add.at(self.first_board, field, 1)

    def summary(self) -> dict:
        return {"endBoardTurns": self.turns,
                "endBoardExtra": round(float(self.end_board.sum() / max(1, self.turns)), 3),
                "firstTurnBoards": self.first_turns,
                "firstTurnExtra": round(float(self.first_board.sum() / max(1, self.first_turns)), 3)}

    def reset(self) -> None:
        """Clears the counts; duels in progress keep their state."""
        self.played[:] = 0
        self.end_board[:] = 0
        self.first_board[:] = 0
        self.turns = self.first_turns = 0


def _rows(batch: dict, rows: list[int]) -> dict:
    """The given rows of a recv_batch batch."""
    count = len(batch["token_count"])
    return {k: v[rows] if isinstance(v, np.ndarray) and v.ndim and len(v) == count else v for k, v in batch.items()}


def _play(policy: Policy, batch: dict) -> tuple[list[int], list[float], list[float]]:
    if hasattr(policy, "on_batch"):
        return policy.on_batch(batch)
    return policy([features.row(batch, n) for n in range(len(batch["token_count"]))])


class AsyncCollector:
    """Self-play with env.recv_batch/send in segments: each seat of each env keeps at most `segment`
    decisions; a full segment is handed out with the value of that seat's next decision as its
    bootstrap, so memory does not grow with the length of a duel. The policy gets whichever
    `batch_size` envs are ready first, so neither the engine nor the policy waits for the slowest
    duel. Duels keep running between collect calls; at the end of a call every open segment is
    handed out too, bootstrapped from its seat's latest decision, so at most that one decision per
    seat (played by the policy before the update) carries over. Use an env driven only this way.

    With a `league`, the first `league.share` of the envs play a league opponent (picked when a duel
    starts) on seat `env % 2` and keep only the learner's decisions; until the league has models,
    they play self-play too."""

    def __init__(self, env, batch_size: int, segment: int = 128, league: League | None = None,
                 card_usage: CardUsage | None = None, extra_usage: ExtraDeckUsage | None = None,
                 tempo: Tempo | None = None):
        self.env, self.batch_size, self.segment, self.league = env, batch_size, segment, league
        self.card_usage, self.extra_usage, self.tempo = card_usage, extra_usage, tempo
        self.turn: dict[int, int] = {}  # env -> turn count at its latest decision
        self.pending: dict[tuple[int, int], list[dict]] = {}
        self.league_envs = round(len(env) * league.share) if league else 0
        self.opponent: dict[int, str] = {}  # env -> league opponent of its current duel

    def league_policies(self) -> dict[str, Policy]:
        return self.league.policies if self.league else {}

    def collect(self, policy: Policy, steps: int) -> tuple[list[tuple[list[dict], float, float | None]], RolloutStats]:
        """Plays `steps` decisions (at least) and returns them as segments (decisions, result,
        bootstrap) for ppo.build_steps, minus the latest decision of each seat, which carries over.
        Failed duels (including ones that hit the decision limit) drop their unfinished segments."""
        segments: list[tuple[list[dict], float, float | None]] = []
        stats = RolloutStats()
        while stats.decisions < steps:
            ids, batch, done, winners, errors = self.env.recv_batch(self.batch_size)
            ids = ids.tolist()
            if not ids:  # every env is still handed out, e.g. by an earlier interrupted call
                raise RuntimeError("recv_batch returned no envs: some envs were never sent an action")
            for i, finished, winner, error in zip(ids, done.tolist(), winners.tolist(), errors):
                if not finished:
                    continue
                seats = [self.pending.pop((i, p), []) for p in (0, 1)]
                opponent = self.opponent.pop(i, None)
                if error:
                    stats.errors += 1
                    stats.error_messages.append(error)
                    continue
                turn = self.turn.get(i, 0)
                stats.end_turns.append(turn)
                if opponent is not None:
                    learner = 1 - i % 2
                    self.league.result(opponent, (_result_for(learner, winner) + 1) / 2)
                    stats.league_duels += 1
                    continue_with = [learner]
                else:
                    stats.duels += 1
                    stats.wins[min(winner, 2)] += 1
                    continue_with = [0, 1]
                for player in continue_with:
                    seat = seats[player]
                    if seat:
                        result = _result_for(player, winner)
                        segments.append((seat, self.tempo.apply(result, turn) if self.tempo else result, None))
            turns = batch["global_numeric"][:, _TURN] * 20
            for n, i in enumerate(ids):
                self.turn[i] = round(float(turns[n]))
            for i, first in zip(ids, (batch["duel_decision"] == 0).tolist()):
                if first and i < self.league_envs and i not in self.opponent and (name := self.league.pick()):
                    self.opponent[i] = name
            players = batch["player"].tolist()
            # An opponent dropped from the league mid-duel hands its seat to the learner.
            actor = [name if players[n] == i % 2 and (name := self.opponent.get(i)) in self.league_policies() else None
                     for n, i in enumerate(ids)]
            choices, logprobs, values = [0] * len(ids), [0.0] * len(ids), [0.0] * len(ids)
            for name in set(actor):
                rows = [n for n, a in enumerate(actor) if a == name]
                part = batch if len(rows) == len(ids) else _rows(batch, rows)
                for n, c, lp, v in zip(rows, *_play(policy if name is None else self.league.policies[name], part)):
                    choices[n], logprobs[n], values[n] = c, lp, v
            learner_rows = [n for n, a in enumerate(actor) if a is None]
            if self.card_usage is not None:
                self.card_usage.add(batch, learner_rows, [choices[n] for n in learner_rows])
            if self.extra_usage is not None:
                self.extra_usage.add(ids, batch, learner_rows)
            for n, (i, choice, logprob, value) in enumerate(zip(ids, choices, logprobs, values)):
                if actor[n] is not None:
                    continue  # the league opponent's move: not trained on
                stats.learner_decisions += 1
                stats.league_decisions += i in self.opponent
                decision = features.row(batch, n, copy=True)  # compact: a view would keep the whole batch alive
                decision.update(chosen=choice, logprob=logprob, value=value)
                key = (i, decision["player"])
                seat = self.pending.setdefault(key, [])
                if len(seat) == self.segment:
                    segments.append((seat, 0.0, float(value)))
                    seat = self.pending[key] = []
                seat.append(decision)
            self.env.send(ids, choices)
            stats.decisions += len(ids)
        for key, seat in self.pending.items():
            if len(seat) > 1:
                segments.append((seat[:-1], 0.0, float(seat[-1]["value"])))
                self.pending[key] = seat[-1:]
        return segments, stats


def match(env, policy_a: Policy, policy_b: Policy, duels: int, name_a: str = "a", name_b: str = "b",
          deck_names: Sequence[str] | None = None) -> list[dict]:
    """Duels between two policies, as ladder.py results. In even envs `a` is player 0, in odd envs
    player 1, so both sit on both seats; who goes first is random per duel. Only duels played
    entirely by these two policies are counted, so an env can be reused across matches."""
    results: list[dict] = []
    errors = 0
    fresh: list[bool | None] = [None] * len(env)
    while len(results) + errors < duels:
        decisions = env.observe()
        _mark_fresh(fresh, decisions)
        a_turn = [(i % 2 == 0) == (d["player"] == 0) for i, d in enumerate(decisions)]
        choices = [0] * len(decisions)
        for policy, mine in ((policy_a, True), (policy_b, False)):
            indices = [i for i, turn in enumerate(a_turn) if turn == mine]
            if indices:
                picked, _, _ = policy([decisions[i] for i in indices])
                for i, choice in zip(indices, picked):
                    choices[i] = choice
        decks = [env.decks(i) for i in range(len(decisions))]
        done, winners, error_messages = env.step(choices)
        for i in (i for i, d in enumerate(done) if d):
            was_fresh, fresh[i] = fresh[i], True
            if not was_fresh:
                continue
            if error_messages[i]:
                errors += 1
                continue
            a_player = 0 if i % 2 == 0 else 1
            winner = winners[i]
            deck_a, deck_b = decks[i][a_player], decks[i][1 - a_player]
            results.append({
                "a": name_a, "b": name_b,
                "winner": "draw" if winner == 2 else ("a" if winner == a_player else "b"),
                "deck_a": deck_names[deck_a] if deck_names else str(deck_a),
                "deck_b": deck_names[deck_b] if deck_names else str(deck_b),
            })
    return results
