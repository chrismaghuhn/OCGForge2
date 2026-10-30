import json

import pytest

from ocgforge import rollout, teach, train


def test_student_learns_from_the_teacher_and_is_saved(tmp_path):
    if not (rollout.REPO / ".cache" / "babelcdb").exists():
        pytest.skip("card database not fetched")
    decks = sorted((rollout.REPO / "decks").glob("*.ydk"))[:2]
    train.main(["--out", str(tmp_path / "teacher"), "--iterations", "1", "--size", "base", "--text-embeddings", "none",
                "--envs", "8", "--threads", "2", "--segment", "16", "--steps", "64", "--arena-every", "1000",
                "--device", "cpu", "--decks", *map(str, decks)])
    teach.main(["--teacher", str(tmp_path / "teacher" / "latest"), "--size", "small", "--rounds", "2", "--steps", "64",
                "--envs", "8", "--threads", "2", "--batch-size", "32", "--micro-batch", "16", "--save-every", "1",
                "--device", "cpu", "--out", str(tmp_path / "student"), "--decks", *map(str, decks)])
    lines = [json.loads(line) for line in (tmp_path / "student" / "teach.jsonl").read_text().splitlines()]
    assert [line["round"] for line in lines] == [1, 2] and all(line["samples"] >= 64 for line in lines)
    assert all(0 <= line["agree"] <= 1 and line["policy"] >= 0 for line in lines)
    from ocgforge.model import load_checkpoint

    student = load_checkpoint(tmp_path / "student" / "latest")
    assert student.config.d_model == train.SIZES["small"].d_model
    assert student.card_features.index_map() == load_checkpoint(tmp_path / "teacher" / "latest").card_features.index_map()
