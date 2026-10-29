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
    assert all(m["blockedAnswers"] >= 0 and m["loopLosses"] >= 0 for m in metrics)
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

    init = tmp_path / "from-latest"  # a new run from these weights: its first arena opponent is them
    args[args.index("--out") + 1] = str(init)
    train.main([*args, "--init", str(tmp_path / "latest")])
    arena = [json.loads(line) for line in (init / "arena.jsonl").read_text().splitlines()]
    assert (init / "checkpoints" / "000000" / "model.safetensors").exists()
    assert "games_previous" in arena[0]


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


def test_entropy_ramps_from_start_to_the_coefficient():
    from types import SimpleNamespace

    args = SimpleNamespace(entropy=0.03, entropy_start=0.08, entropy_ramp=100)
    assert [round(train.entropy_coef(args, i), 4) for i in (0, 50, 100, 400)] == [0.08, 0.055, 0.03, 0.03]
    assert train.entropy_coef(SimpleNamespace(entropy=0.03, entropy_start=None, entropy_ramp=0), 5) == 0.03


def test_reset_heads_keeps_the_encoder(card_features):
    from ocgforge.model import Model, ModelConfig

    model = Model(card_features, ModelConfig(d_model=32, heads=2, layers=1, ff=64))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    train.reset_heads(model)
    after = model.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed and all(k.startswith(("score.", "value.")) for k in changed)


def test_resume_keeps_the_runs_origin_and_schedules():
    from pathlib import Path
    from types import SimpleNamespace

    args = SimpleNamespace(**train.RESUMED_SETTINGS)
    args.tempo_floor = 0.5  # given again on the command line: wins over run.json
    train.restore_run_settings(args, {"init": "runs/base", "tempo_slope": 0.05, "tempo_floor": 0.6,
                                      "entropy_start": 0.08, "entropy_ramp": 150})
    assert args.init == Path("runs/base") and args.tempo_slope == 0.05 and args.tempo_floor == 0.5
    assert (args.entropy_start, args.entropy_ramp) == (0.08, 150)


def test_extend_seen_marks_the_decks_cards_and_survives_a_save(tmp_path):
    from conftest import make_table
    from ocgforge.cards import UNK, CardFeatures
    from ocgforge.model import Model, ModelConfig, load_checkpoint, save_checkpoint

    table = make_table()
    model = Model(CardFeatures.build(table, trained_codes={1001}, min_term_count=1),
                  ModelConfig(d_model=16, heads=2, layers=1, ff=32))
    deck = tmp_path / "d.ydk"
    deck.write_text("#main\n1001\n1003\n1004\n#extra\n1002\n!side\n1003\n")
    warrior, link, spell = (table.index_of[c] for c in (1001, 1002, 1003))
    trained_before = model.name.weight[warrior].clone()
    assert train.extend_seen(model, [deck]) == 2  # the Link and the spell; the alternate artwork is the Warrior
    assert model.card_seen[[warrior, link, spell]].tolist() == [True, True, True]
    assert torch.equal(model.name.weight[link], model.name.weight[UNK])
    assert torch.equal(model.name.weight[warrior], trained_before)
    save_checkpoint(model, tmp_path / "ckpt")
    assert load_checkpoint(tmp_path / "ckpt").card_features.seen[[link, spell]].tolist() == [True, True]
