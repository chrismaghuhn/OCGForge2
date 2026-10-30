import numpy as np
import pytest
import torch

from ocgforge import expert
from ocgforge.model import Model, ModelConfig
from ocgforge.search import HonestSearchPolicy

from conftest import random_decision


class FakeDuel:
    player = 0


def test_plan_posterior_votes_by_score_and_keeps_the_agreeing_plans():
    reported = []
    searcher = HonestSearchPolicy(None, None, on_target=lambda duel, keys, target: reported.append(target),
                                  temperature=0.1)
    searcher.scored = [(["a", "x"], 0.5), (["a", "y"], 0.4), (["b", "z"], 0.2)]
    searcher.played = 0
    searcher.report_target(FakeDuel(), ["a", "b", "c"], "a")
    weights = np.exp(np.array([0.5, 0.4, 0.2]) / 0.1 - 5.0)
    assert reported[0] == pytest.approx([(weights[0] + weights[1]) / weights.sum(), weights[2] / weights.sum(), 0.0])
    assert [plan for plan, _ in searcher.scored] == [["a", "x"], ["a", "y"]]  # "b" left with its plan
    searcher.played = 1
    searcher.report_target(FakeDuel(), ["x", "y"], "x")
    assert reported[1] == pytest.approx([1 / (1 + np.exp(-1.0)), np.exp(-1.0) / (1 + np.exp(-1.0))])


def test_distill_loss_ignores_padding_and_learns_the_target(card_features):
    torch.manual_seed(0)
    model = Model(card_features, ModelConfig(d_model=32, heads=2, layers=1, ff=64, dropout=0.0))
    rng = np.random.default_rng(0)
    samples = []
    for i in range(8):
        d = random_decision(rng, card_features.size, candidates=3 + i % 3)
        target = np.zeros(len(d["candidate_kind"]), np.float32)
        target[1] = 1.0
        samples.append({**d, "target": target, "result": 1.0 if i % 2 else -1.0})
    batch, targets, results = expert._batch(samples, "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        loss, stats = expert.distill_loss(model, batch, targets, results)
        assert torch.isfinite(loss)
        first = first if first is not None else stats["policy"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert stats["policy"] < first and stats["agree"] == 1.0


def test_search_data_reads_complete_shards_and_keeps_the_newest(tmp_path):
    import pickle

    with open(tmp_path / "shard-0-0.pkl", "wb") as f:
        pickle.dump([{"i": i} for i in range(5)], f)
    (tmp_path / "shard-0-1.pkl.tmp").write_bytes(b"half written")
    data = expert.SearchData(tmp_path, window=4)
    assert data.refresh() == 5 and [s["i"] for s in data.samples] == [1, 2, 3, 4]
    with open(tmp_path / "shard-0-2.pkl", "wb") as f:
        pickle.dump([{"i": 9}], f)
    assert data.refresh() == 1 and [s["i"] for s in data.samples] == [2, 3, 4, 9]
    assert data.refresh() == 0 and len(data.sample(10)) == 4


def test_training_imitates_search_shards(tmp_path):
    import json

    from ocgforge import rollout, train

    if not (rollout.REPO / ".cache" / "babelcdb").exists():
        pytest.skip("card database not fetched")
    decks = sorted((rollout.REPO / "decks").glob("*.ydk"))[:2]
    base = ["--size", "small", "--text-embeddings", "none", "--envs", "8", "--threads", "2", "--segment", "16",
            "--steps", "64", "--arena-every", "1000", "--device", "cpu", "--decks", *map(str, decks)]
    train.main(["--out", str(tmp_path / "base"), "--iterations", "1", *base])
    expert.generate_shard(tmp_path / "base" / "latest", decks, 1, 2, 3, tmp_path / "search" / "shard-0-0.pkl", "cpu")
    train.main(["--out", str(tmp_path / "run"), "--iterations", "2", "--init", str(tmp_path / "base" / "latest"),
                "--search-data", str(tmp_path / "search"), "--search-batch", "16", *base])
    metrics = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert metrics[0]["searchSamples"] > 0 and metrics[0]["searchNew"] > 0 and metrics[1]["searchNew"] == 0
    assert all(0 <= m["searchAgree"] <= 1 and m["searchLoss"] >= 0 for m in metrics)
