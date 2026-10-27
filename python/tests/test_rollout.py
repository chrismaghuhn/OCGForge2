"""collect/match bookkeeping against a scripted env: seats, result signs, leftover duels."""

import numpy as np

from ocgforge import rollout

from conftest import random_decision


class ScriptedEnv:
    """Each env alternates players 0,1,0,1,...; duel k of env i lasts `lengths[i]` steps and is won
    by `winners[i][k]` (2 = draw, 255 = error)."""

    def __init__(self, lengths, winners):
        self.lengths, self.winners = lengths, winners
        self.step_in_duel = [0] * len(lengths)
        self.duel = [0] * len(lengths)
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.lengths)

    def observe(self):
        out = []
        for i in range(len(self)):
            d = random_decision(self.rng, 5, tokens=2, candidates=2)
            d["player"] = self.step_in_duel[i] % 2
            d["duel_decision"] = self.step_in_duel[i]
            out.append(d)
        return out

    def decks(self, i):
        return (0, 1)

    def step(self, actions):
        done, winner, errors = [], [], []
        for i in range(len(self)):
            self.step_in_duel[i] += 1
            finished = self.step_in_duel[i] == self.lengths[i]
            w = self.winners[i][self.duel[i]] if finished else 255
            done.append(finished)
            winner.append(w if w != 255 else 255)
            errors.append("boom" if finished and w == 255 else "")
            if finished:
                self.step_in_duel[i] = 0
                self.duel[i] += 1
        return done, winner, errors


def test_collect_gives_each_seat_its_own_result():
    env = ScriptedEnv(lengths=[4], winners=[[1, 0, 2]])
    trajectories, stats = rollout.collect(env, rollout.RandomPolicy(), duels=3)
    assert stats.duels == 3 and stats.wins == [1, 1, 1]
    results = [(t[0]["player"], r, len(t)) for t, r in trajectories]
    assert results == [(0, -1.0, 2), (1, 1.0, 2), (0, 1.0, 2), (1, -1.0, 2), (0, 0.0, 2), (1, 0.0, 2)]


def test_failed_duels_are_dropped():
    env = ScriptedEnv(lengths=[2], winners=[[255, 0]])
    trajectories, stats = rollout.collect(env, rollout.RandomPolicy(), duels=2)
    assert stats.errors == 1 and stats.duels == 1 and len(trajectories) == 2


def test_duels_left_running_are_not_recorded_by_the_next_call():
    env = ScriptedEnv(lengths=[4, 6], winners=[[0] * 9, [1] * 9])
    rollout.collect(env, rollout.RandomPolicy(), duels=1)  # env 1 is left mid-duel
    trajectories, _ = rollout.collect(env, rollout.RandomPolicy(), duels=2)
    assert all(len(t) in (2, 3) for t, _ in trajectories)  # whole duels only: 4 or 6 decisions over two seats


def test_match_maps_seats_and_winners():
    # Env 0: a is player 0; env 1: a is player 1. Player 0 wins every duel.
    env = ScriptedEnv(lengths=[2, 2], winners=[[0] * 9, [0] * 9])
    results = rollout.match(env, rollout.RandomPolicy(1), rollout.RandomPolicy(2), duels=4, name_a="A", name_b="B")
    by_env_winner = sorted(r["winner"] for r in results)
    assert by_env_winner.count("a") == by_env_winner.count("b") == len(results) // 2
