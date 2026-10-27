import numpy as np
import torch

from ocgforge import features
from ocgforge.model import Model, ModelConfig, load_checkpoint, save_checkpoint

from conftest import random_decision

SMALL = ModelConfig(d_model=32, heads=2, layers=1, ff=64, dropout=0.0, name_dropout=0.5)


def test_scores_have_one_entry_per_candidate_and_padding_is_masked(card_features):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = Model(card_features, SMALL).eval()
    samples = [random_decision(rng, card_features.size, candidates=c) for c in (2, 5, 1)]
    scores, value = model(features.collate(samples))
    assert scores.shape == (3, 5) and value.shape == (3,)
    for row, sample in zip(scores, samples):
        c = len(sample["candidate_kind"])
        assert torch.isfinite(row[:c]).all() and torch.isinf(row[c:]).all()
    assert (value.abs() <= 1).all()


def test_batching_does_not_change_a_decisions_scores(card_features):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    model = Model(card_features, SMALL).eval()
    small = random_decision(rng, card_features.size, tokens=2, candidates=2)
    large = random_decision(rng, card_features.size, tokens=9, candidates=6)
    alone, _ = model(features.collate([small]))
    batched, _ = model(features.collate([small, large]))
    torch.testing.assert_close(alone[0], batched[0, :2], rtol=1e-4, atol=1e-5)


def test_decision_without_visible_cards(card_features):
    rng = np.random.default_rng(2)
    model = Model(card_features, SMALL).eval()
    scores, _ = model(features.collate([random_decision(rng, card_features.size, tokens=0, candidates=3)]))
    assert torch.isfinite(scores).all()


def test_name_dropout_only_while_training(card_features):
    torch.manual_seed(0)
    model = Model(card_features, ModelConfig(d_model=16, heads=2, layers=1, ff=32, dropout=0.0, name_dropout=1.0))
    cards = torch.tensor([[2, 3]])
    model.eval()
    kept = model.card_embedding(cards) - model.card_static(cards)
    assert torch.allclose(kept, model.name(cards), atol=1e-6)
    model.train()
    # With every name dropped, a card only differs from "unknown" by its static features.
    dropped = model.card_embedding(cards) - model.card_static(cards)
    assert torch.allclose(dropped, model.name(torch.ones_like(cards)), atol=1e-6)


def test_unseen_card_plays_without_learned_name(card_features):
    card_features.seen[3] = False
    model = Model(card_features, SMALL).eval()
    with torch.no_grad():
        model.name.weight[3] += 100.0  # would dominate if it were used
    unseen = model.card_embedding(torch.tensor([3])) - model.card_static(torch.tensor([3]))
    assert torch.allclose(unseen, model.name(torch.tensor([1])), atol=1e-5)


def test_checkpoint_roundtrip(card_features, tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    config = ModelConfig(d_model=32, heads=2, layers=1, ff=64, dropout=0.0, text_embedding_dim=8)
    embeddings = rng.random((card_features.size, 8), dtype=np.float32)
    model = Model(card_features, config, text_embeddings=embeddings).eval()
    batch = features.collate([random_decision(rng, card_features.size) for _ in range(4)])
    expected = model(batch)
    save_checkpoint(model, tmp_path)
    loaded = load_checkpoint(tmp_path).eval()
    for a, b in zip(expected, loaded(batch)):
        torch.testing.assert_close(a, b)
    assert loaded.card_features.text_vocab == card_features.text_vocab
    np.testing.assert_array_equal(loaded.card_features.text, card_features.text)


def test_text_embeddings_are_required_when_enabled(card_features):
    import pytest

    with pytest.raises(ValueError, match="text_embeddings"):
        Model(card_features, ModelConfig(d_model=16, heads=2, layers=1, ff=32, text_embedding_dim=8))


def test_batch_without_any_events(card_features):
    rng = np.random.default_rng(4)
    model = Model(card_features, SMALL).eval()
    samples = [random_decision(rng, card_features.size, tokens=3, candidates=2, events=0) for _ in range(2)]
    scores, _ = model(features.collate(samples))
    assert torch.isfinite(scores).all()
