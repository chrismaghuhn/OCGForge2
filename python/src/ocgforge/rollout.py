"""Playing duels in the C++ VectorEnv: self-play rollouts for PPO and matches between two policies.

A policy maps a list of pending decisions (features.py dicts) to (choices, logprobs, values), one
entry per decision; policies with an `on_batch` method also take a padded VectorEnv.recv_batch
batch directly, which saves building per-decision dicts. Every env always has a pending decision;
finished duels restart by themselves.
"""

from __future__ import annotations

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
               threads: int = 0, max_decisions: int = 5000, cache: Path = REPO / ".cache"):
    """A VectorEnv whose card indices match `card_features` (e.g. model.card_features)."""
    from ._core import VectorEnv

    return VectorEnv(str(cache / "cardscripts"), str(cache / "card_data.tsv"), [str(d) for d in decks],
                     card_features.index_map(), num_envs=num_envs, seed=seed, threads=threads,
                     max_decisions=max_decisions)


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

    def __init__(self, model: Model, device: torch.device | str = "cpu", sample: bool = True, seed: int = 0):
        self.model, self.device, self.sample = model, device, sample
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, decisions: Sequence[dict]) -> tuple[list[int], list[float], list[float]]:
        return self._choose(features.collate(decisions, self.device))

    def on_batch(self, batch: dict) -> tuple[list[int], list[float], list[float]]:
        return self._choose(features.from_padded(batch, self.device))

    @torch.no_grad()
    def _choose(self, inputs: dict) -> tuple[list[int], list[float], list[float]]:
        self.model.eval()
        scores, values = self.model(inputs)
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
    wins: list[int] = field(default_factory=lambda: [0, 0, 0])  # player 0, player 1, draw
    error_messages: list[str] = field(default_factory=list)


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


def collect_async(env, policy: Policy, duels: int, batch_size: int) -> tuple[list[tuple[list[dict], float]], RolloutStats]:
    """Like collect, but with env.recv/send: the policy gets whichever `batch_size` envs are ready
    first, so neither the engine nor the policy waits for the slowest duel. Use an env that is
    driven only this way (not with step)."""
    pending: dict[tuple[int, int], list[dict]] = {}
    trajectories: list[tuple[list[dict], float]] = []
    stats = RolloutStats()
    fresh: dict[int, bool] = {}  # we saw env i's current duel from its first decision
    while stats.duels + stats.errors < duels:
        ids, batch, done, winners, errors = env.recv_batch(batch_size)
        ids = ids.tolist()
        for i, finished, winner, error in zip(ids, done.tolist(), winners.tolist(), errors):
            if not finished:
                continue
            seats = [pending.pop((i, p), []) for p in (0, 1)]
            was_fresh, fresh[i] = fresh.get(i, False), True
            if not was_fresh:
                continue
            if error:
                stats.errors += 1
                stats.error_messages.append(error)
                continue
            stats.duels += 1
            stats.wins[min(winner, 2)] += 1
            for player, seat in enumerate(seats):
                if seat:
                    trajectories.append((seat, _result_for(player, winner)))
        for i, first in zip(ids, (batch["duel_decision"] == 0).tolist()):
            fresh.setdefault(i, first)
        if not ids:  # every env is still handed out, e.g. by an earlier interrupted call
            raise RuntimeError("recv_batch returned no envs: some envs were never sent an action")
        if hasattr(policy, "on_batch"):
            choices, logprobs, values = policy.on_batch(batch)
        else:
            choices, logprobs, values = policy([features.row(batch, n) for n in range(len(ids))])
        for n, (i, choice, logprob, value) in enumerate(zip(ids, choices, logprobs, values)):
            decision = features.row(batch, n, copy=True)  # compact: a view would keep the whole batch alive
            decision.update(chosen=choice, logprob=logprob, value=value)
            pending.setdefault((i, decision["player"]), []).append(decision)
        env.send(ids, choices)
        stats.decisions += len(ids)
    return trajectories, stats


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
