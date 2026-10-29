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


def test_league_prefers_opponents_the_learner_struggles_against():
    from ocgforge.rollout import League

    league = League(share=0.5, size=2, seed=0, window=10)
    assert league.pick() is None
    league.add("weak", object())
    league.add("strong", object())
    for _ in range(20):
        league.result("weak", 1.0)
        league.result("strong", 0.0)
    picks = [league.pick() for _ in range(400)]
    assert 300 < picks.count("strong") < 390  # PFSP, but the uniform share still picks "weak"
    for _ in range(10):  # only the latest `window` duels count
        league.result("strong", 1.0)
    assert league.scores()["strong"] == 1.0
    league.add("newest", object())  # size 2: the oldest leaves
    assert set(league.policies) == {"strong", "newest"} and "weak" not in league.scores()


def test_league_anchors_outlive_the_recent_window():
    from ocgforge.rollout import League

    league = League(share=0.5, size=1, anchors=2)
    league.add("first", object(), anchor=True)
    for name in ("a", "b", "c"):
        league.add(name, object())
    league.add("anchor1", object(), anchor=True)
    league.add("anchor2", object(), anchor=True)  # over 2 anchors: the oldest after "first" leaves
    assert set(league.policies) == {"first", "anchor2", "c"}


def test_card_usage_counts_offered_and_chosen_card_actions():
    from ocgforge import features
    from ocgforge.rollout import CardUsage

    kind = features.CANDIDATE_KINDS.index
    batch = {  # two decisions; token 0 is card 5, token 1 is card 6
        "token_card": np.array([[5, 6], [5, 6]]),
        "candidate_kind": np.array([[kind("activate"), kind("summon"), kind("to_end_phase")],
                                    [kind("activate"), kind("activate"), 0]]),
        "candidate_source": np.array([[0, 1, -1], [0, -1, -1]]),
        "candidate_card": np.array([[0, 0, 0], [0, 7, 0]]),  # card 7 only from the deck, no token
    }
    usage = CardUsage(10)
    usage.add(batch, [0, 1], [1, 2])  # summon card 6; then the padding slot is not a card action
    assert usage.offered[[5, 6, 7]].tolist() == [2, 1, 1]
    assert usage.chosen.tolist() == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]


def test_extra_deck_usage_counts_played_cards_and_end_boards():
    from types import SimpleNamespace

    from ocgforge import features
    from ocgforge.rollout import ExtraDeckUsage

    types = np.zeros((10, 2), np.int64)
    types[5, 0], types[6, 1] = 13 + 2, 26 + 2  # card 5 a Synchro, card 6 a Link; card 7 a main deck card
    usage = ExtraDeckUsage(SimpleNamespace(types=types))
    mzone, grave = features.ZONES.index("monster_zone"), features.ZONES.index("graveyard")
    turn, own = features.GLOBAL_NUMERIC.index("turn"), features.GLOBAL_NUMERIC.index("isOwnTurn")

    def step(duel_decision, turn_count, own_turn, tokens):
        """One decision of player 0 in env 3; tokens are (card, side, zone)."""
        g = np.zeros((1, len(features.GLOBAL_NUMERIC)), np.float32)
        g[0, turn], g[0, own] = turn_count / 20, float(own_turn)
        usage.add([3], {"player": np.array([0]), "duel_decision": np.array([duel_decision]), "global_numeric": g,
                        "token_card": np.array([[c for c, _, _ in tokens]]),
                        "token_side": np.array([[s for _, s, _ in tokens]]),
                        "token_zone": np.array([[z for _, _, z in tokens]])}, [0])

    step(0, 1, True, [(7, 0, mzone)])
    step(5, 1, True, [(5, 0, mzone), (6, 1, mzone)])  # the opponent's Link is not ours
    step(9, 1, True, [(5, 0, mzone), (6, 0, mzone), (7, 0, mzone)])  # turn 1 ends with both on the field
    step(12, 2, False, [(5, 0, mzone), (6, 0, grave)])
    step(20, 3, True, [(5, 0, mzone), (6, 0, mzone)])  # card 6 again in the same duel: still played once
    step(25, 5, True, [(6, 0, mzone)])  # no decision in turn 4: turn 3 ends here
    assert usage.played[[5, 6, 7]].tolist() == [1, 1, 0]
    assert usage.end_board[[5, 6]].tolist() == [2, 2] and usage.first_board[[5, 6]].tolist() == [1, 1]
    assert usage.summary() == {"endBoardTurns": 2, "endBoardExtra": 2.0, "firstTurnBoards": 1, "firstTurnExtra": 2.0}
    step(0, 1, False, [(6, 0, mzone)])  # a new duel: turn 5 never ended, card 6 counts again
    assert usage.played[6] == 2 and usage.turns == 2
    step(9, 4, True, [(5, 0, mzone)])  # this duel ends in our own turn 4 with card 5 out
    # The next duel starts with the opponent's decision (a league opponent's row: not counted)...
    usage.add([3], {"player": np.array([1]), "duel_decision": np.array([0]),
                    "global_numeric": np.zeros((1, len(features.GLOBAL_NUMERIC)), np.float32),
                    "token_card": np.array([[0]]), "token_side": np.array([[2]]), "token_zone": np.array([[0]])}, [])
    step(3, 2, False, [(5, 0, mzone)])  # ...so our turn 4 of the last duel is no end board here
    assert usage.turns == 2 and usage.played[5] == 3  # played in all three duels


def test_tempo_keeps_early_results_and_shrinks_late_ones_to_the_floor():
    from ocgforge.rollout import Tempo

    tempo = Tempo(start=5, slope=0.05, floor=0.6)
    assert [tempo.weight(t) for t in (1, 3, 5)] == [1.0, 1.0, 1.0]
    assert [round(tempo.weight(t), 2) for t in (6, 10, 13, 30)] == [0.95, 0.75, 0.6, 0.6]
    assert (round(tempo.apply(1.0, 10), 2), tempo.apply(-1.0, 10), tempo.apply(0.0, 10)) == (0.75, -1.0, 0.0)
    assert round(Tempo(start=5, slope=0.05, floor=0.6, losses=True).apply(-1.0, 10), 2) == -0.75
