"""The C++ side (ocgforge._core): contract with features.py, information safety, determinism, and
one PPO update on real self-play data."""

import os

import numpy as np
import pytest
import torch

from ocgforge import features
from ocgforge.cards import UNK, CardFeatures, CardTable
from ocgforge.model import Model, ModelConfig
from ocgforge.ppo import PPOConfig, build_steps, ppo_update

from conftest import CDB, REPO

if os.environ.get("OCGFORGE_REQUIRE_CORE"):  # CI: a missing or broken module is a failure, not a skip
    from ocgforge import _core as core
else:
    core = pytest.importorskip("ocgforge._core", reason="build with: cmake --preset python && cmake --build --preset python")
from ocgforge import rollout

DECKS = [REPO / "decks" / "swordsoul_tenyi.ydk", REPO / "decks" / "salamangreat.ydk"]


@pytest.fixture(scope="module")
def card_features() -> CardFeatures:
    if not CDB.exists():
        pytest.skip("card database not fetched")
    return CardFeatures.build(CardTable.from_cdb())


def test_cpp_and_python_agree_on_the_input_contract():
    assert tuple(core.decision_names()) == features.DECISIONS
    assert tuple(core.zone_names()) == features.ZONES
    assert tuple(core.phase_names()) == features.PHASES
    assert tuple(core.candidate_kind_names()) == features.CANDIDATE_KINDS
    assert tuple(core.token_numeric_names()) == features.TOKEN_NUMERIC
    assert tuple(core.global_numeric_names()) == features.GLOBAL_NUMERIC
    assert tuple(core.candidate_numeric_names()) == features.CANDIDATE_NUMERIC
    assert core.MAX_TARGETS == features.MAX_TARGETS
    assert tuple(core.event_kind_names()) == features.EVENT_KINDS
    assert tuple(core.event_numeric_names()) == features.EVENT_NUMERIC
    assert core.MAX_EVENTS == features.MAX_EVENTS


def _play_random(env, steps, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        decisions = env.observe()
        yield decisions
        env.step([int(rng.integers(len(d["candidate_kind"]))) for d in decisions])


def test_random_play_gives_valid_safe_inputs(card_features):
    env = rollout.vector_env(card_features, DECKS, num_envs=8, seed=1)
    zones, numeric = features.ZONES, features.TOKEN_NUMERIC
    field = [zones.index(z) for z in ("monster_zone", "spell_trap_zone", "field_zone", "pendulum_zone")]
    face_up, selected = numeric.index("faceUp"), numeric.index("selected")
    face_up_spell_traps = 0
    for decisions in _play_random(env, 400):
        for d in decisions:
            features.check(d)
            assert d["phase"] > 0 and d["decision"] > 0
            # Face-down opponent cards on the field are never identified. (Opponent hand cards
            # only appear at all once the engine has made them public, e.g. after a search.)
            face_down = np.isin(d["token_zone"], field) & (d["token_side"] == 1) & (d["token_numeric"][:, face_up] == 0)
            assert (d["token_card"][face_down] == UNK).all()
            # Spells/traps report POS_FACEUP (both bits); they used to read as face-down.
            spell_traps = d["token_zone"] == zones.index("spell_trap_zone")
            face_up_spell_traps += int((d["token_numeric"][spell_traps, face_up] == 1).sum())
            # A card that can be unselected is marked as selected.
            for kind, source in zip(d["candidate_kind"], d["candidate_source"]):
                if features.CANDIDATE_KINDS[kind] == "unselect_card" and source >= 0:
                    assert d["token_numeric"][source, selected] == 1.0
    assert face_up_spell_traps > 0
    assert env.duel_errors == 0 and env.start_failures == 0


@pytest.mark.parametrize("deck", sorted((REPO / "decks").glob("**/*.ydk")), ids=lambda p: p.stem)
def test_every_deck_can_start_a_duel(card_features, deck):
    env = rollout.vector_env(card_features, [deck], num_envs=1, seed=13)
    assert env.start_failures == 0, env.last_start_failure


def test_same_seed_and_actions_give_the_same_inputs(card_features):
    runs = []
    for _ in range(2):
        env = rollout.vector_env(card_features, DECKS, num_envs=4, seed=7)
        runs.append([[{k: np.asarray(v).tolist() for k, v in d.items()} for d in ds] for ds in _play_random(env, 60)])
    assert runs[0] == runs[1]


def test_invalid_action_is_rejected(card_features):
    env = rollout.vector_env(card_features, DECKS, num_envs=2, seed=3)
    with pytest.raises(IndexError):
        env.step([len(env.observe()[0]["candidate_kind"]), 0])


def test_self_play_rollout_feeds_a_ppo_update(card_features):
    torch.manual_seed(0)
    model = Model(card_features, ModelConfig(d_model=32, heads=2, layers=1, ff=64, dropout=0.0))
    env = rollout.vector_env(card_features, DECKS, num_envs=8, seed=5)
    trajectories, stats = rollout.collect(env, rollout.ModelPolicy(model), duels=4)
    assert stats.duels >= 4 and stats.errors == 0
    assert len(trajectories) == 2 * stats.duels
    assert {result for _, result in trajectories} <= {-1.0, 0.0, 1.0}
    steps = build_steps(trajectories, gamma=1.0, lam=0.95)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    history = ppo_update(model, optimizer, steps, PPOConfig(epochs=1, batch_size=256))
    assert history and np.isfinite(history[0]["loss"])


def test_match_counts_both_seats(card_features):
    env = rollout.vector_env(card_features, DECKS, num_envs=4, seed=9)
    results = rollout.match(env, rollout.RandomPolicy(1), rollout.RandomPolicy(2), duels=6, name_a="r1",
                            name_b="r2", deck_names=["swordsoul", "salamangreat"])
    assert len(results) >= 6
    assert all(r["a"] == "r1" and r["winner"] in ("a", "b", "draw") for r in results)
    assert {r["deck_a"] for r in results} <= {"swordsoul", "salamangreat"}


def test_padded_batches_match_collated_decisions(card_features):
    """recv_batch pads in C++; it must give exactly what collate gives for the same decisions."""
    a = rollout.vector_env(card_features, DECKS, num_envs=12, seed=21, threads=1)
    b = rollout.vector_env(card_features, DECKS, num_envs=12, seed=21, threads=1)
    rng = np.random.default_rng(0)
    for _ in range(40):
        ids_a, decisions, *_ = a.recv(12)
        ids_b, batch, *_ = b.recv_batch(12)
        assert list(ids_a) == list(ids_b)
        collated = features.collate(decisions)
        padded = features.from_padded(batch)
        assert collated.keys() - {"chosen"} == padded.keys()
        for key, value in padded.items():
            assert torch.equal(collated[key], value), key
        for i, decision in enumerate(decisions):
            view, compact = features.row(batch, i), features.row(batch, i, copy=True)
            for key, value in decision.items():
                assert np.array_equal(np.asarray(view[key]), np.asarray(value)), key
                assert np.array_equal(np.asarray(compact[key]), np.asarray(value)), key
        assert torch.equal(features.collate([features.row(batch, i, copy=True) for i in range(len(ids_b))])["token_card"],
                           collated["token_card"])
        actions = [int(rng.integers(len(d["candidate_kind"]))) for d in decisions]
        a.send(ids_a, actions)
        b.send(list(ids_b), actions)


def test_async_segments_feed_a_ppo_update(card_features):
    torch.manual_seed(0)
    model = Model(card_features, ModelConfig(d_model=32, heads=2, layers=1, ff=64, dropout=0.0))
    env = rollout.vector_env(card_features, DECKS, num_envs=8, seed=6)
    collector = rollout.AsyncCollector(env, batch_size=4, segment=16)
    policy = rollout.ModelPolicy(model)
    segments, stats = collector.collect(policy, steps=300)
    assert all(len(seat) == 1 for seat in collector.pending.values())  # only the latest decision carries over
    more, more_stats = collector.collect(policy, steps=300)  # duels carry over between calls
    segments += more
    assert sum(len(s) for s, _, _ in segments) >= 600 - 2 * len(env)  # one decision per seat carries over
    assert all(1 <= len(s) <= 16 for s, _, _ in segments)
    cut = [(s, result) for s, result, bootstrap in segments if bootstrap is not None]
    assert cut and all(result == 0.0 for _, result in cut) and any(len(s) == 16 for s, _ in cut)
    for decisions, _, _ in segments:  # one seat of one duel, in order
        assert len({d["player"] for d in decisions}) == 1
        assert all(a["duel_decision"] < b["duel_decision"] for a, b in zip(decisions, decisions[1:]))
    # Which duel an env plays next depends on thread timing here, so now and then one is stopped by
    # the loop protection or the Lua budget; only that kind of error is fine.
    errors = stats.error_messages + more_stats.error_messages
    assert all("stuck" in m or "budget" in m for m in errors), errors
    assert {result for _, result, _ in segments} <= {-1.0, 0.0, 1.0}
    steps = build_steps(segments, gamma=1.0, lam=0.95)
    history = ppo_update(model, torch.optim.AdamW(model.parameters(), lr=1e-4), steps, PPOConfig(epochs=1, batch_size=256))
    assert history and np.isfinite(history[0]["loss"])


def test_league_envs_keep_only_the_learners_decisions(card_features):
    env = rollout.vector_env(card_features, DECKS, num_envs=4, seed=8)
    league = rollout.League(share=0.5, size=2)
    league.add("random", rollout.RandomPolicy(1))
    collector = rollout.AsyncCollector(env, batch_size=2, segment=16, league=league)
    segments, stats = collector.collect(rollout.RandomPolicy(2), steps=400)
    assert collector.opponent == {0: "random", 1: "random"}
    for env_id, player in collector.pending:
        if env_id < 2:
            assert player == 1 - env_id % 2  # the opponent sits on seat env % 2
    assert {seat for env_id, seat in collector.pending if env_id >= 2} == {0, 1}
    assert sum(len(s) for s, _, _ in segments) < stats.decisions
