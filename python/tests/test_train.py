import json

import pytest

from ocgforge import train

core = pytest.importorskip("ocgforge._core", reason="build with: cmake --preset python && cmake --build --preset python")


def test_deck_codes_skip_the_side_deck(tmp_path):
    deck = tmp_path / "d.ydk"
    deck.write_text("#created by x\n#main\n123\n456\n#extra\n789\n!side\n999\n")
    assert train.deck_codes(deck) == {123, 456, 789}


def test_training_runs_an_arena_and_resumes(tmp_path):
    if not (train.rollout.REPO / ".cache" / "babelcdb").exists():
        pytest.skip("card database not fetched")
    args = ["--out", str(tmp_path), "--size", "small", "--text-embeddings", "none", "--envs", "8",
            "--threads", "2", "--segment", "16", "--steps", "64", "--iterations", "2", "--arena-every", "2",
            "--arena-duels", "2", "--arena-envs", "4", "--device", "cpu"]
    train.main(args)
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    arena = [json.loads(line) for line in (tmp_path / "arena.jsonl").read_text().splitlines()]
    assert [m["iteration"] for m in metrics] == [1, 2]
    assert arena[0]["iteration"] == 2
    assert arena[0]["vs_random"] is None or 0.0 <= arena[0]["vs_random"] <= 1.0  # duels can be cut off
    assert (tmp_path / "checkpoints" / "000002" / "model.safetensors").exists()
    cards = json.loads((tmp_path / "cards.jsonl").read_text().splitlines()[0])["cards"]
    assert cards and all(0 <= c["rate"] <= 1 for c in cards)
    args[args.index("--iterations") + 1] = "3"
    train.main([*args, "--resume"])
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [m["iteration"] for m in metrics] == [1, 2, 3]
