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
