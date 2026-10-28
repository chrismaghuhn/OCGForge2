import json

import pytest
import torch

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
    assert arena[0]["iteration"] == 2 and "games_random" in arena[0]
    assert arena[0]["vs_random"] is None or 0.0 <= arena[0]["vs_random"] <= 1.0  # duels can be cut off
    assert (tmp_path / "checkpoints" / "000002" / "model.safetensors").exists()
    cards = json.loads((tmp_path / "cards.jsonl").read_text().splitlines()[0])["cards"]
    assert cards and all(0 <= c["rate"] <= 1 for c in cards)
    with pytest.raises(SystemExit):  # the directory holds a run
        train.main(args)
    args[args.index("--iterations") + 1] = "3"
    train.main([*args, "--resume"])
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [m["iteration"] for m in metrics] == [1, 2, 3]


def test_latest_is_replaced_whole(tmp_path):
    from ocgforge.model import Model, ModelConfig
    from ocgforge.cards import CardFeatures
    from conftest import make_table

    model = Model(CardFeatures.build(make_table(), min_term_count=1), ModelConfig(d_model=16, heads=2, layers=1, ff=32))
    optimizer = torch.optim.AdamW(model.parameters())
    train.save_latest(model, optimizer, tmp_path, 1)
    train.save_latest(model, optimizer, tmp_path, 2)
    assert torch.load(tmp_path / "latest" / train.STATE_FILE)["iteration"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["latest"]


def test_resume_survives_a_cut_off_log_line_and_an_interrupted_save(tmp_path):
    log = tmp_path / "metrics.jsonl"
    log.write_text('{"iteration": 1}\n{"iteration": 2}\n{"iteration": 3}\n{"iterat')
    train.drop_after(log, 2)
    assert log.read_text() == '{"iteration": 1}\n{"iteration": 2}\n'

    from ocgforge.model import Model, ModelConfig
    from ocgforge.cards import CardFeatures
    from conftest import make_table

    model = Model(CardFeatures.build(make_table(), min_term_count=1), ModelConfig(d_model=16, heads=2, layers=1, ff=32))
    optimizer = torch.optim.AdamW(model.parameters())
    train.save_latest(model, optimizer, tmp_path, 1)
    (tmp_path / "latest").rename(tmp_path / "latest.old")  # killed between the two renames
    train.save_latest(model, optimizer, tmp_path, 2)
    assert torch.load(tmp_path / "latest" / train.STATE_FILE)["iteration"] == 2
    assert not (tmp_path / "latest.old").exists()
