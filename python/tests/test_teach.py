import json

import numpy as np
import pytest

from ocgforge import rollout, teach, train


@pytest.fixture(scope="module")
def teacher(tmp_path_factory):
    """A tiny trained teacher checkpoint and the decks it played."""
    if not (rollout.REPO / ".cache" / "babelcdb").exists():
        pytest.skip("card database not fetched")
    out = tmp_path_factory.mktemp("teacher")
    decks = sorted((rollout.REPO / "decks").glob("*.ydk"))[:2]
    train.main(["--out", str(out), "--iterations", "1", "--size", "base", "--text-embeddings", "none",
                "--envs", "8", "--threads", "2", "--segment", "16", "--steps", "64", "--arena-every", "1000",
                "--device", "cpu", "--decks", *map(str, decks)])
    return out / "latest", decks


def small_run(teacher_path, decks, out, *extra):
    teach.main(["--teacher", str(teacher_path), "--size", "small", "--steps", "64", "--envs", "8", "--threads", "2",
                "--batch-size", "32", "--micro-batch", "16", "--save-every", "1", "--device", "cpu",
                "--out", str(out), "--decks", *map(str, decks), *extra])
    return [json.loads(line) for line in (out / "teach.jsonl").read_text().splitlines()]


def test_student_learns_from_the_teacher_and_is_saved(teacher, tmp_path):
    teacher_path, decks = teacher
    lines = small_run(teacher_path, decks, tmp_path / "student", "--rounds", "2")
    assert [line["round"] for line in lines] == [1, 2] and all(line["samples"] >= 64 for line in lines)
    assert all(0 <= line["agree"] <= 1 and line["policy"] >= 0 for line in lines)
    assert "bookSamples" not in lines[0]
    from ocgforge.model import load_checkpoint

    student = load_checkpoint(tmp_path / "student" / "latest")
    assert student.config.d_model == train.SIZES["small"].d_model
    assert student.card_features.index_map() == load_checkpoint(teacher_path).card_features.index_map()


def test_book_samples_are_mixed_in_at_their_share(teacher, tmp_path, monkeypatch):
    """--book with a synthetic book: real observations of a fresh duel, the first answer as target."""
    from ocgforge import book_imitation, worlds
    from ocgforge._core import Duel
    from ocgforge.expert import compact
    from ocgforge.model import load_checkpoint

    teacher_path, decks = teacher
    index = load_checkpoint(teacher_path).card_features.index_map()
    cache = rollout.REPO / ".cache"
    duel = Duel(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(decks[0]), str(decks[1]), 3, 0)
    rows = []
    while len(rows) < 6 and not duel.finished:
        keys = duel.keys()
        if len(keys) > 1:
            row = compact(duel.encode(index))
            row["target"], row["result"] = np.eye(len(keys), dtype=np.float32)[0], 0.0
            rows.append(row)
        duel.act(worlds.decline(keys))
    assert len(rows) == 6
    monkeypatch.setattr(book_imitation, "samples", lambda path, index, holdout: (rows[:4], rows[4:]))
    lines = small_run(teacher_path, decks, tmp_path / "student", "--rounds", "1", "--book", str(tmp_path / "x.jsonl"),
                      "--book-share", "0.25")
    start, first = lines
    assert start["round"] == 0 and start["book"] == 4 and start["bookHeld"] == 2
    assert first["bookSamples"] == teach.book_count(first["teacherSamples"], 0.25)
    assert first["samples"] == first["teacherSamples"] + first["bookSamples"] and first["teacherSamples"] >= 64
    assert abs(first["bookSamples"] / first["samples"] - 0.25) < 0.02
    for key in ("bookAgree", "bookHeldAgree", "bookTrainAgree", "teacherAgree"):
        assert 0 <= first[key] <= 1
    assert first["bookTrainPolicy"] >= 0 and first["teacherPolicy"] >= 0


def test_book_options_are_checked(tmp_path):
    assert teach.book_count(300, 0.25) == 100 and teach.book_count(300, 0.0) == 0
    for bad in (["--book-share", "1"], ["--book-share", "-0.1"], ["--book"]):
        with pytest.raises(SystemExit):
            teach.main(["--teacher", "t", "--out", str(tmp_path), *bad])
