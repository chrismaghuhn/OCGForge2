"""Distill a checkpoint into a smaller model: the teacher plays self-play, the student learns its moves.

Each round the teacher samples answers for `steps` self-play decisions (both seats, a VectorEnv);
every decision is kept with the teacher's whole answer distribution and its value. The student, a new
model of another size on the same card table and text embeddings, is then trained on that round:
cross-entropy to the teacher's distribution plus the squared error to the teacher's value. Rounds
repeat (fresh positions each time), and every `eval_every` rounds the agreement of the two models'
most likely answers on the newest round is printed. The student is saved as a checkpoint that
train.py can continue from with --init.

    python -m ocgforge.teach --teacher runs/x/latest --size base --decks ... --rounds 200 --out runs/student
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import features, rollout
from .model import Model, load_checkpoint, save_checkpoint


def student_for(teacher: Model, size: str) -> Model:
    """A new model of `size` on the teacher's card table and text embeddings."""
    from .train import SIZES

    config = replace(SIZES[size], dropout=0.0, text_embedding_dim=teacher.config.text_embedding_dim)
    embeddings = teacher.card_textemb.cpu().numpy() if teacher.config.text_embedding_dim else None
    return Model(teacher.card_features, config, text_embeddings=embeddings)


@torch.no_grad()
def collect(env, teacher: Model, steps: int, device: str, generator: torch.Generator, batch_size: int) -> list[dict]:
    """`steps` self-play decisions with the teacher's sampled answers; each kept with the teacher's
    distribution ("target") and value."""
    teacher.eval()
    samples: list[dict] = []
    while len(samples) < steps:
        ids, batch, *_ = env.recv_batch(batch_size)
        scores, values = teacher(features.from_padded(batch, device))
        probs = torch.softmax(scores.float(), dim=-1)
        choices = torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(1).tolist()  # generator lives on the CPU
        probs, values = probs.cpu().numpy(), values.float().cpu().numpy()
        counts = np.asarray(batch["candidate_count"])
        for n in range(len(ids)):
            row = features.row(batch, n, copy=True)
            row["target"] = probs[n, : counts[n]].astype(np.float32)
            row["result"] = float(values[n])
            samples.append(row)
        env.send(ids, choices)
    return samples


def train_round(student: Model, optimizer, samples: list[dict], device: str, batch_size: int, micro_batch: int,
                value_coef: float, rng: random.Random) -> dict:
    """One pass over a round's samples; returns the mean losses."""
    from .expert import _batch, distill_loss

    student.eval()  # like ppo_update: no dropout or name dropout, so the student matches how it will play
    rng.shuffle(samples)
    totals, n = {"policy": 0.0, "value": 0.0, "agree": 0.0}, 0
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        optimizer.zero_grad()
        for micro in range(0, len(batch), micro_batch):
            part = batch[micro:micro + micro_batch]
            loss, stats = distill_loss(student, *_batch(part, device), value_coef=value_coef)
            (loss * len(part) / len(batch)).backward()
            for key in totals:
                totals[key] += stats[key] * len(part)
            n += len(part)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
    return {key: round(value / max(n, 1), 4) for key, value in totals.items()}


def book_samples(args, teacher: Model) -> tuple[list[dict], list[dict]]:
    """--book: the books' training and held-out samples, value targets from the teacher."""
    from .book_imitation import samples, with_values

    train, held = [], []
    for path in args.book:
        t, h = samples(path, teacher.card_features.index_map(), args.book_holdout)
        train += t
        held += h
    return with_values(teacher, train, args.device), with_values(teacher, held, args.device)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--student", type=Path, help="continue this student checkpoint instead of a new one")
    p.add_argument("--size", default="base", help="size of a new student (train.SIZES)")
    p.add_argument("--decks", type=Path, nargs="+", default=sorted((rollout.REPO / "decks").glob("*.ydk")))
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--steps", type=int, default=32768, help="teacher decisions per round")
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--micro-batch", type=int, default=256)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--book", type=Path, nargs="*", default=[],
                   help="opening books whose answers are mixed into every round (book_imitation)")
    p.add_argument("--book-share", type=float, default=0.2, help="share of book samples in a round")
    p.add_argument("--book-holdout", type=float, default=0.1, help="share of book seeds held out (agreement only)")
    args = p.parse_args(argv)

    teacher = load_checkpoint(args.teacher, args.device, dropout=0.0)
    student = (load_checkpoint(args.student, args.device, dropout=0.0) if args.student
               else student_for(teacher, args.size).to(args.device))
    print(f"teacher {sum(t.numel() for t in teacher.parameters()) / 1e6:.1f}M, "
          f"student {sum(t.numel() for t in student.parameters()) / 1e6:.1f}M parameters", flush=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    env = rollout.vector_env(teacher.card_features, args.decks, num_envs=args.envs, seed=args.seed, threads=args.threads)
    generator = torch.Generator().manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "teach.jsonl", "a", encoding="utf-8")
    book, book_held = book_samples(args, teacher) if args.book else ([], [])
    if book:
        from .book_imitation import agreement
        line = {"round": 0, "book": len(book), "bookHeld": len(book_held),
                "bookAgree": agreement(student, book, args.device), "bookHeldAgree": agreement(student, book_held, args.device)}
        log.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
    for round_ in range(1, args.rounds + 1):
        start = time.time()
        samples = collect(env, teacher, args.steps, args.device, generator, max(1, args.envs // 2))
        if book:  # book share s of the round: s / (1 - s) book samples per teacher sample
            samples += rng.choices(book, k=round(len(samples) * args.book_share / (1 - args.book_share)))
        collected = time.time()
        stats = train_round(student, optimizer, samples, args.device, args.batch_size, args.micro_batch,
                            args.value_coef, rng)
        line = {"round": round_, "samples": len(samples), "collectSeconds": round(collected - start, 1),
                "trainSeconds": round(time.time() - collected, 1), **stats}
        if book:
            line.update(bookAgree=agreement(student, book, args.device),
                        bookHeldAgree=agreement(student, book_held, args.device))
        log.write(json.dumps(line) + "\n")
        log.flush()
        print(json.dumps(line), flush=True)
        if round_ % args.save_every == 0 or round_ == args.rounds:
            save_checkpoint(student, args.out / "latest")
    log.close()


if __name__ == "__main__":
    main()
