import numpy as np
import pytest
import torch

from ocgforge import features
from ocgforge.model import Model, ModelConfig
from ocgforge.ppo import PPOConfig, build_steps, collate_steps, gae, ppo_loss, ppo_update

from conftest import random_decision

SMALL = ModelConfig(d_model=32, heads=2, layers=1, ff=64, dropout=0.0, name_dropout=0.0)


def test_gae_with_terminal_reward_only():
    advantages, returns = gae([0.0, 0.0, 0.0], 1.0, gamma=1.0, lam=1.0)
    assert advantages == [1.0, 1.0, 1.0]
    assert returns == [1.0, 1.0, 1.0]
    advantages, _ = gae([0.5, 0.5], -1.0, gamma=1.0, lam=0.0)
    assert advantages == pytest.approx([0.0, -1.5])  # lam 0: one-step TD errors


def test_build_steps_normalizes_advantages():
    trajectory = [{"value": 0.0}, {"value": 0.2}, {"value": -0.1}]
    steps = build_steps([(trajectory, 1.0), (trajectory, -1.0)], gamma=1.0, lam=0.95)
    advantages = np.array([s["advantage"] for s in steps])
    assert len(steps) == 6
    assert abs(advantages.mean()) < 1e-6 and abs(advantages.std(ddof=1) - 1) < 1e-5


def _rollout(model, rng, n_cards, count):
    """Decisions played by `model` itself (sampled), with logprob and value as a collector records them."""
    samples = [random_decision(rng, n_cards, candidates=3) for _ in range(count)]
    model.eval()
    with torch.no_grad():
        scores, values = model(features.collate(samples))
    logprobs = torch.log_softmax(scores, -1)
    for sample, row, value in zip(samples, logprobs, values):
        chosen = int(torch.multinomial(row.exp(), 1))
        sample.update(chosen=chosen, logprob=float(row[chosen]), value=float(value))
    return samples


def test_loss_on_unchanged_policy_has_ratio_one(card_features):
    torch.manual_seed(0)
    model = Model(card_features, SMALL)
    steps = build_steps([(_rollout(model, np.random.default_rng(0), card_features.size, 16), 1.0)], 1.0, 0.95)
    _, stats = ppo_loss(model, collate_steps(steps), PPOConfig())
    assert abs(stats["approxKl"]) < 1e-6 and stats["clipFraction"] == 0.0


def test_update_makes_rewarded_choices_more_likely(card_features):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    model = Model(card_features, SMALL)
    decisions = _rollout(model, rng, card_features.size, 64)
    # Reward exactly the decisions where candidate 0 was played: one-step trajectories.
    trajectories = [([d], 1.0 if d["chosen"] == 0 else -1.0) for d in decisions]
    steps = build_steps(trajectories, gamma=1.0, lam=0.95)
    batch = features.collate(decisions)

    def mean_prob_first():
        model.eval()
        with torch.no_grad():
            return torch.softmax(model(batch)[0], -1)[:, 0].mean().item()

    before = mean_prob_first()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    config = PPOConfig(epochs=8, batch_size=32, kl_coef=0.0, target_kl=1.0)
    history = ppo_update(model, optimizer, steps, config)
    assert history and history[0]["batches"] == 2
    assert mean_prob_first() > before + 0.05


def test_update_stops_early_at_target_kl(card_features):
    torch.manual_seed(0)
    model = Model(card_features, SMALL)
    decisions = _rollout(model, np.random.default_rng(2), card_features.size, 64)
    steps = build_steps([([d], 1.0 if d["chosen"] == 0 else -1.0) for d in decisions], 1.0, 0.95)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    history = ppo_update(model, optimizer, steps, PPOConfig(epochs=20, batch_size=16, target_kl=0.001))
    assert history[-1]["stoppedEarly"]
    assert len(history) < 20
