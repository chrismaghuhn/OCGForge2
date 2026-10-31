"""book_imitation: samples from book lines (synthetic duels only)."""

import json

import numpy as np
import torch

from ocgforge import book_imitation as bi
from ocgforge import book_policy as bp

LUCINA, REGINA, TINIA = 13597785, 56651978, 59581480


class Solo:
    """Seat 0 summons hand cards one by one; seat 1 gets a one-answer question in between."""

    def __init__(self, hand):
        self.hand, self.finished, self.step = list(hand), False, 0
        self.turn, self.phase, self.player, self.turn_player = 1, 4, 0, 0

    def snapshot(self):
        return {"cards": [{"controller": 0, "owner": 0, "location": 2, "code": c, "sequence": i, "position": 10,
                           "public": False} for i, c in enumerate(self.hand)]}

    def keys(self):
        if self.player == 1:
            return ["chain.pass"]
        return [f"idle.1.{i}.{c}.0.2.{i}" for i, c in enumerate(self.hand)] + ["idle.7.0"]

    def encode(self, index):
        return {"token_card": np.array([index.get(c, 0) for c in self.hand], np.int64),
                "candidate_kind": np.zeros(len(self.keys()), np.int64), "phase": 4}

    def act(self, key):
        if self.player == 0 and key != "idle.7.0":
            self.hand.remove(int(key.split(".")[3]))
        self.player = 1 - self.player if key != "idle.7.0" else self.player
        self.step += 1


def lines_of(tmp_path, monkeypatch, lines):
    deck = tmp_path / "solo.ydk"
    deck.write_text(f"#main\n{LUCINA}\n{REGINA}\n{TINIA}\n#extra\n!side\n", encoding="utf-8")
    monkeypatch.setattr(bp.cc, "puzzle_start", lambda d, o, script, seed: (lambda: Solo([LUCINA, REGINA, TINIA])))
    (tmp_path / "b-lines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    (tmp_path / "b-meta.json").write_text(json.dumps({"deck": str(deck), "opponent": str(deck),
                                                      "relevant": [LUCINA, REGINA, TINIA]}), encoding="utf-8")
    book, _ = bp.ob.build_book([(r, bp.cc.puzzle_start(None, None, "", 0)) for r in lines], {LUCINA, REGINA, TINIA})
    (tmp_path / "b.jsonl").write_text("".join(json.dumps(e) + "\n" for e in book.values()), encoding="utf-8")
    return tmp_path / "b.jsonl"


def line(seed, held, keys):
    return {"seed": seed, "trap": None, "field": 1, "value": {"held": held}, "line": keys, "script": "", "duel_seed": 0}


def test_samples_follow_the_books_best_answer_at_real_choices(tmp_path, monkeypatch):
    weak = line("weak", 0.2, [f"idle.1.0.{LUCINA}.0.2.0", "chain.pass", "idle.7.0"])
    strong = line("strong", 0.9, [f"idle.1.1.{REGINA}.0.2.1", "chain.pass", f"idle.1.1.{TINIA}.0.2.1"])
    path = lines_of(tmp_path, monkeypatch, [weak, strong])
    train, held = bi.samples(path, {LUCINA: 1, REGINA: 2, TINIA: 3})
    assert held == []
    # the start is shared: kept once, with the strong line's answer (Regina, index 1)
    assert len(train) == 3  # start, weak's second state, strong's second state; seat 1's forced passes left out
    first = [s for s in train if len(s["token_card"]) == 3]
    assert len(first) == 1 and first[0]["target"].tolist() == [0, 1, 0, 0]
    assert all(s["target"].sum() == 1 and s["result"] == 0.0 for s in train)


def test_holdout_by_seed_keeps_shared_states_in_training(tmp_path, monkeypatch):
    seeds = [f"seed-{n}" for n in range(40)]
    out = [s for s in seeds if bi.held_out(s, 0.5)]
    assert 5 < len(out) < 35 and out == [s for s in seeds if bi.held_out(s, 0.5)]  # stable
    assert not any(bi.held_out(s, 0.0) for s in seeds)
    a, b = out[0], next(s for s in seeds if s not in out)
    path = lines_of(tmp_path, monkeypatch, [line(a, 0.5, [f"idle.1.2.{TINIA}.0.2.2", "chain.pass", "idle.7.0"]),
                                            line(b, 0.5, [f"idle.1.0.{LUCINA}.0.2.0", "chain.pass", "idle.7.0"])])
    train, held = bi.samples(path, {LUCINA: 1, REGINA: 2, TINIA: 3}, holdout=0.5)
    assert len(train) == 2 and len(held) == 1  # the shared start stays in training
    assert held[0]["token_card"].tolist() == [1, 2]  # after the held-out line summoned Tinia


class Fixed(torch.nn.Module):
    """Scores the candidate at index `best`; value 0.25."""

    def __init__(self, best):
        super().__init__()
        self.best = best

    def forward(self, batch):
        n = batch["n"]
        scores = torch.zeros(n, 4)
        scores[:, self.best] = 1.0
        return scores, torch.full((n,), 0.25)


def test_agreement_and_values(monkeypatch):
    monkeypatch.setattr(bi.features, "collate", lambda rows, device: {"n": len(rows)})
    rows = [{"target": np.array([0, 1, 0], np.float32)}, {"target": np.array([1, 0], np.float32)}]
    assert bi.agreement(Fixed(1), rows, "cpu") == 0.5
    assert bi.agreement(Fixed(1), [], "cpu") is None
    assert [r["result"] for r in bi.with_values(Fixed(0), rows, "cpu")] == [0.25, 0.25]
