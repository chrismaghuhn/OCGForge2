"""One turn-1 expert iteration: search hands -> book -> distillation with book imitation.

    python -m ocgforge.turn1_iteration --out runs/turn1/it1 --net runs/student-base2/latest \\
        --teacher runs/base-day-1030 --decks-file runs/decks-3m.txt --hands 250 --rounds 30

1. `turn1_search` for each book deck (--books: deck path and deck type) with the current student
   as the searching net; one book per deck in <out>/<deck>-t1.jsonl.
2. `teach --book <the books> --book-share` continuing the student (lr, rounds, decks as given).

Every stage runs as a subprocess and survives engine crashes (a segfault kills a run every hour or
so): a search is restarted with --resume until it finishes; teach is restarted from the last saved
checkpoint (<out>/student/latest) for the rounds it still owes, with seed `seed + segment` as the
earlier drivers did. The driver itself can be rerun: finished searches are skipped, and teach
picks up the rounds recorded in <out>/student/driver.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import rollout

BOOKS = ("decks/elfnote.ydk:Elfnote", "decks/branded.ydk:Branded")
MAX_TRIES = 8


def saved_rounds(completed: int, save_every: int, ok: bool) -> int:
    """Rounds of a teach segment that are in its checkpoint: all if it finished, else the ones up to
    its last save."""
    return completed if ok else completed - completed % save_every


def book_path(out: Path, deck: Path) -> Path:
    return out / f"{deck.stem}-t1.jsonl"


def done_marker(book: Path) -> Path:
    return book.with_name(f"{book.stem}-done")


def search_done(book: Path, hands: int) -> bool:
    """Finished: a run went through every hand (hands whose search raised are not retried) and built the book."""
    return done_marker(book).exists() and book.exists()


def run(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as out:
        out.write("$ " + " ".join(cmd) + "\n")
        out.flush()
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # a shared GPU fragments
        return subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)


def search(args, deck: Path, deck_type: str) -> Path:
    """The book of one deck (the --min-gain book if a minimum gain is set)."""
    book = book_path(args.out, deck)
    for attempt in range(MAX_TRIES):
        if search_done(book, args.hands):
            return chosen(book, args)
        code = run([sys.executable, "-m", "ocgforge.turn1_search", "--deck", str(deck), "--deck-type", deck_type,
                    "--opponent", str(args.opponent), "--checkpoint", str(args.net), "--hands", str(args.hands),
                    "--orders", str(args.orders), "--beam", str(args.beam), "--branch", str(args.branch),
                    "--workers", str(args.workers), "--seed", str(args.seed), "--resume", "--out", str(book)]
                   + (["--min-gain", str(args.min_gain)] if args.min_gain is not None else []),
                   args.out / f"{deck.stem}-search.log")
        print(f"search {deck.stem}: attempt {attempt} exit {code}", flush=True)
        if code == 0:
            done_marker(book).write_text(str(args.hands))
    if not search_done(book, args.hands):
        raise SystemExit(f"search {deck.stem} did not finish in {MAX_TRIES} tries")
    return chosen(book, args)


def chosen(book: Path, args) -> Path:
    """The book teach imitates: the --min-gain book (built by the search, or now from its lines)."""
    if args.min_gain is None:
        return book
    from .turn1_search import gain_path

    gain = gain_path(book)
    meta = gain.with_name(f"{gain.stem}-meta.json")
    if not gain.exists() or json.loads(meta.read_text()).get("min_gain") != args.min_gain:
        deck_meta = json.loads(book.with_name(f"{book.stem}-meta.json").read_text())
        code = run([sys.executable, "-m", "ocgforge.turn1_search", "--deck", deck_meta["deck"], "--deck-type",
                    deck_meta["deck_type"], "--opponent", deck_meta["opponent"], "--checkpoint", str(args.net),
                    "--build-only", "--min-gain", str(args.min_gain), "--out", str(book)],
                   args.out / f"{book.stem}-build.log")
        if code != 0:
            raise SystemExit(f"building {gain.name} failed")
    return gain


def teach(args, books: list[Path]) -> None:
    """teach --book `books` (none: the control, plain distillation) into <out>/student (<out>/ctrl)."""
    out = args.out / ("student" if books else "ctrl")
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "driver.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": 0, "segment": 0}
    log = out / "teach.jsonl"
    decks = [str(rollout.REPO / d.strip()) for d in args.decks_file.read_text().splitlines()
             if d.strip() and not d.startswith("#")]
    while state["done"] < args.rounds:
        if state["segment"] >= MAX_TRIES:
            raise SystemExit("teach: too many crashes")
        student = out / "latest" if (out / "latest").exists() else args.net
        before = sum('"samples"' in t for t in log.read_text().splitlines()) if log.exists() else 0
        left = args.rounds - state["done"]
        code = run([sys.executable, "-m", "ocgforge.teach", "--teacher", str(args.teacher), "--student", str(student),
                    *(["--book", *map(str, books), "--book-share", str(args.book_share)] if books else []),
                    "--rounds", str(left),
                    "--lr", str(args.lr), "--threads", str(args.teach_threads), "--save-every", str(args.save_every),
                    "--micro-batch", str(args.micro_batch), "--envs", str(args.envs),
                    "--seed", str(args.teach_seed + state["segment"]), "--out", str(out), "--decks", *decks],
                   out / "teach.log")
        after = sum('"samples"' in t for t in log.read_text().splitlines()) if log.exists() else 0
        saved = saved_rounds(after - before, args.save_every, code == 0)
        state = {"done": args.rounds if code == 0 else state["done"] + saved, "segment": state["segment"] + 1}
        state_path.write_text(json.dumps(state))
        print(f"teach segment {state['segment'] - 1}: exit {code}, {after - before} rounds, {saved} saved, "
              f"{state['done']}/{args.rounds}", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--net", type=Path, required=True, help="the student: searches, then is trained on")
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--decks-file", type=Path, required=True, help="teach decks, one per line (relative to the repo)")
    p.add_argument("--books", nargs="+", default=list(BOOKS), help="deck:deck type of each book")
    p.add_argument("--opponent", type=Path, default=rollout.REPO / "decks" / "maliss.ydk")
    p.add_argument("--hands", type=int, default=250)
    p.add_argument("--orders", type=int, default=4)
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--branch", type=int, default=3)
    p.add_argument("--workers", type=int, default=4, help="search threads")
    p.add_argument("--teach-threads", type=int, default=4, help="teach engine threads")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--book-share", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--micro-batch", type=int, default=64,
                   help="teach micro-batch (the gradient is the same; smaller fits beside the desktop on the GPU)")
    p.add_argument("--envs", type=int, default=512, help="teach self-play envs")
    p.add_argument("--teach-seed", type=int, default=100, help="seed of teach segment 0 (the control used 100)")
    p.add_argument("--min-gain", type=float, default=0.1,
                   help="imitate only lines beating the net's own line by more than this (mean value); negative: all")
    p.add_argument("--search-only", action="store_true")
    p.add_argument("--control", action="store_true",
                   help="the control instead: the same teach rounds and seeds without a book, into <out>/ctrl")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.min_gain is not None and args.min_gain < 0:
        args.min_gain = None
    if args.control:
        teach(args, [])
        return
    books = []
    for spec in args.books:
        deck, _, deck_type = spec.rpartition(":")
        books.append(search(args, (rollout.REPO / deck).resolve(), deck_type))
    if not args.search_only:
        teach(args, books)


if __name__ == "__main__":
    main()
