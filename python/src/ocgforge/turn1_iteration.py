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
        return subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT)


def search(args, deck: Path, deck_type: str) -> Path:
    book = book_path(args.out, deck)
    for attempt in range(MAX_TRIES):
        if search_done(book, args.hands):
            return book
        code = run([sys.executable, "-m", "ocgforge.turn1_search", "--deck", str(deck), "--deck-type", deck_type,
                    "--opponent", str(args.opponent), "--checkpoint", str(args.net), "--hands", str(args.hands),
                    "--orders", str(args.orders), "--beam", str(args.beam), "--branch", str(args.branch),
                    "--workers", str(args.workers), "--seed", str(args.seed), "--resume", "--out", str(book)],
                   args.out / f"{deck.stem}-search.log")
        print(f"search {deck.stem}: attempt {attempt} exit {code}", flush=True)
        if code == 0:
            done_marker(book).write_text(str(args.hands))
    if not search_done(book, args.hands):
        raise SystemExit(f"search {deck.stem} did not finish in {MAX_TRIES} tries")
    return book


def teach(args, books: list[Path]) -> None:
    out = args.out / "student"
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
                    "--book", *map(str, books), "--book-share", str(args.book_share), "--rounds", str(left),
                    "--lr", str(args.lr), "--threads", str(args.workers), "--save-every", str(args.save_every),
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
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--book-share", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--teach-seed", type=int, default=100, help="seed of teach segment 0 (the control used 100)")
    p.add_argument("--search-only", action="store_true")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    books = []
    for spec in args.books:
        deck, _, deck_type = spec.rpartition(":")
        books.append(search(args, (rollout.REPO / deck).resolve(), deck_type))
    if not args.search_only:
        teach(args, books)


if __name__ == "__main__":
    main()
