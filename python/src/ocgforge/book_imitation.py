"""Training samples from an opening book: the net learns to play the book's answers itself.

Every line beside a book (`<book>-lines.jsonl`) is replayed from its scenario; at each real choice
(two answers or more) of the combo player the encoded observation is kept with the book's answer
for that state as a one-hot target (the book holds the answer of the best line through a state, so
lines that pass through it agree on one answer). A state is kept once. With `holdout`, the lines
of a share of the seeds are held out (their states that train lines also reach stay in training),
to see whether the net learns the book or only remembers it.

`teach --book ...` mixes these samples into the teacher's distillation rounds.

    python -m ocgforge.book_imitation runs/combo-book/elfnote-opening-v2.jsonl --checkpoint runs/x/latest
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import book_policy as bp
from . import features
from . import opening_book as ob
from . import worlds
from .expert import compact


def held_out(seed_id: str, share: float) -> bool:
    """Whether a seed's lines are held out (a stable choice by its id)."""
    return share > 0 and int(hashlib.sha1(seed_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def line_samples(start, keys: list[str], book: bp.Book, card_index) -> list[tuple[str, dict]]:
    """(state, sample) at each real choice of seat 0 along a line replayed from `start`, in turn 1
    only (the book's turn, as in `opening_book.walk`)."""
    duel = start()
    out = []
    for key in keys:
        if duel is None or duel.finished or duel.turn > 1 or key not in duel.keys():
            break
        options = duel.keys()
        if duel.player == 0 and len(options) > 1:
            state = ob.state_key(duel, book.relevant)
            answer = worlds.match(book.exact.get(state, key), options) or key
            sample = compact(duel.encode(card_index))
            target = np.zeros(len(options), np.float32)
            target[options.index(answer)] = 1.0
            sample["target"], sample["result"] = target, 0.0
            out.append((state, sample))
        duel.act(key)
    return out


def samples(path: Path, card_index, holdout: float = 0.0) -> tuple[list[dict], list[dict]]:
    """(train, held-out) samples of the book `path`. The lines and their scenario starts come from
    `book_policy.load`; each line is replayed once more here, as the book build keeps no encodings."""
    book = bp.load(path)
    train, held = {}, {}
    for result, start in book.lines:
        into = held if held_out(result["seed"], holdout) else train
        for state, sample in line_samples(start, result["line"], book, card_index):
            into.setdefault(state, sample)
    return list(train.values()), [s for state, s in held.items() if state not in train]


@torch.no_grad()
def with_values(model, rows: list[dict], device: str, batch: int = 256) -> list[dict]:
    """Sets each sample's value target ("result") to the model's value (the book says nothing about it)."""
    model.eval()
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        _, values = model(features.collate(part, device))
        for row, v in zip(part, values.float().cpu().tolist()):
            row["result"] = float(v)
    return rows


@torch.no_grad()
def agreement(model, rows: list[dict], device: str, batch: int = 256) -> float | None:
    """Share of samples where the model's most likely answer is the book's."""
    if not rows:
        return None
    model.eval()
    hits = 0
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        scores, _ = model(features.collate(part, device))
        best = scores.float().argmax(-1).cpu().tolist()
        hits += sum(int(r["target"].argmax()) == b for r, b in zip(part, best))
    return round(hits / len(rows), 4)


def main(argv=None) -> None:
    from .model import load_checkpoint

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("books", type=Path, nargs="+")
    p.add_argument("--checkpoint", type=Path, required=True, help="whose card table to encode with; its agreement")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    model = load_checkpoint(args.checkpoint, args.device, dropout=0.0)
    for path in args.books:
        train, held = samples(path, model.card_features.index_map(), args.holdout)
        print(json.dumps({"book": path.stem, "train": len(train), "held": len(held),
                          "agree_train": agreement(model, train, args.device),
                          "agree_held": agreement(model, held, args.device)}), flush=True)


if __name__ == "__main__":
    main()
