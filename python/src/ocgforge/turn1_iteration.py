"""One turn-1 expert iteration: search hands -> book -> distillation with book imitation.

    python -m ocgforge.turn1_iteration --out runs/turn1/it1 --net runs/student-base2/latest \\
        --teacher runs/base-day-1030 --decks-file runs/decks-3m.txt --hands 250 --rounds 30

1. `turn1_search` for each book deck (--books: deck path and deck type) with the current student
   as the searching net; one book per deck in <out>/<deck>-t1.jsonl.
2. `teach --book <the books> --book-share` continuing the student (lr, rounds, decks as given).

`--score turn2` (or mix) scores the candidate lines by the opponent's real turn 2 (outcome_eval)
against real opponents (`--opponents-file`). `--shards N` splits each deck's hands over N search
processes (the search threads share one interpreter, so one process uses under two cores); every
shard resumes on its own, and their lines are merged into the deck's book when all are done.

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
import time
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


FAILED_HANDS = 3  # turn1_search's exit code: it went through, but some hands failed
FAILED_TRIES = 3  # runs that end with failed hands before the driver accepts them (deterministic failures)


def search_done(book: Path, expected: dict) -> bool:
    """Finished: the done marker holds `expected` (the hand range and seed it was searched for) and
    the book exists. A marker for other hands (an earlier run with other settings in the same
    --out) stops the driver instead of reusing a stale book."""
    marker = done_marker(book)
    if not marker.exists():
        return False
    try:
        found = json.loads(marker.read_text())
    except ValueError:
        found = None
    held = {k: found.get(k) for k in expected} if isinstance(found, dict) else found
    if held != expected:
        raise SystemExit(f"{marker} is for {found}, not {expected}: remove it (and the book) or use another --out")
    return book.exists()


def mark_done(book: Path, expected: dict, failed: list | None = None) -> None:
    done_marker(book).write_text(json.dumps({**expected, **({"failed": failed} if failed else {})}))


def run(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as out:
        out.write("$ " + " ".join(cmd) + "\n")
        out.flush()
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # a shared GPU fragments
        return subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)


def search_cmd(args, deck: Path, deck_type: str, book: Path, first: int, hands: int, min_gain: bool) -> list[str]:
    return ([sys.executable, "-m", "ocgforge.turn1_search", "--deck", str(deck), "--deck-type", deck_type,
             "--opponent", str(args.opponent), "--checkpoint", str(args.net), "--first-hand", str(first),
             "--hands", str(hands), "--orders", str(args.orders), "--beam", str(args.beam), "--branch", str(args.branch),
             "--traps", str(args.traps), "--workers", str(args.workers), "--seed", str(args.seed),
             "--score", args.score, "--rollouts", str(args.rollouts), "--mix-weight", str(args.mix_weight),
             "--t2-cap", str(args.t2_cap), "--validate", str(args.validate), "--gain-orders", str(args.gain_orders),
             "--gain-rollouts", str(args.gain_rollouts), "--resume", "--out", str(book)]
            + (["--race", "--race-k1", str(args.race_k1), "--race-k2", str(args.race_k2), "--race-se", str(args.race_se),
                "--race-frac", str(args.race_frac), "--race-margin", str(args.race_margin), "--race-c", str(args.race_c)]
               if args.race else [])
            + (["--opponents-file", str(args.opponents_file)] if args.opponents_file else [])
            + (gain_args(args) if min_gain else []))


def gain_args(args) -> list[str]:
    return ((["--min-gain", str(args.min_gain)] if args.min_gain is not None else [])
            + (["--min-gain-se", str(args.min_gain_se)] if args.min_gain_se is not None else []))


def shard_ranges(hands: int, shards: int) -> list[tuple[int, int]]:
    """(first hand, hands) of each shard: `hands` split as evenly as possible."""
    size, rest = divmod(hands, shards)
    out, first = [], 0
    for i in range(shards):
        n = size + (i < rest)
        if n:
            out.append((first, n))
        first += n
    return out


def shard_book(book: Path, first: int, hands: int) -> Path:
    """A shard's files, named by its hand range: another --shards count gets other files, never a
    shard of other hands."""
    return book.parent / "shards" / f"{book.stem}-h{first}-{first + hands}.jsonl"


def merge_shards(book: Path, shards: list[Path]) -> None:
    """The deck's lines, hands and rollouts files from the shards' (in shard order); a hand that more
    than one shard holds is taken once, from the first."""
    from .turn1_search import paths, read_jsonl

    hands, owner = [], {}
    for i, shard in enumerate(shards):
        for h in read_jsonl(paths(shard)["hands"]) if paths(shard)["hands"].exists() else []:
            if h["hand"] not in owner:
                owner[h["hand"]] = i
                hands.append(h)
    out = {"hands": hands, "lines": [], "rollouts": []}
    seen_lines = set()
    for i, shard in enumerate(shards):
        for kind, key in (("lines", "seed"), ("rollouts", "hand")):
            if not paths(shard)[kind].exists():
                continue
            for r in read_jsonl(paths(shard)[kind]):
                if owner.get(r.get(key)) != i:
                    continue
                if kind == "lines":
                    ident = (r["seed"], r.get("trap"))
                    if ident in seen_lines:
                        continue
                    seen_lines.add(ident)
                out[kind].append(r)
    for kind, rows in out.items():
        paths(book)[kind].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def accept(code: int, tries: int) -> bool:
    """A search run counts as finished: it succeeded, or it keeps failing the same hands."""
    return code == 0 or (code == FAILED_HANDS and tries >= FAILED_TRIES)


def failed_hands(book: Path) -> list:
    from .turn1_search import paths

    path = paths(book)["failed"]
    return json.loads(path.read_text()) if path.exists() else []


def search(args, deck: Path, deck_type: str) -> Path:
    """The book of one deck (the --min-gain book if a minimum gain is set)."""
    book = book_path(args.out, deck)
    expected = {"first": 0, "hands": args.hands, "seed": args.seed}
    if args.shards > 1:
        return search_sharded(args, deck, deck_type, book, expected)
    for attempt in range(MAX_TRIES):
        if search_done(book, expected):
            return chosen(book, args)
        code = run(search_cmd(args, deck, deck_type, book, 0, args.hands, True), args.out / f"{deck.stem}-search.log")
        print(f"search {deck.stem}: attempt {attempt} exit {code}", flush=True)
        if accept(code, attempt + 1):
            mark_done(book, expected, failed_hands(book))
    if not search_done(book, expected):
        raise SystemExit(f"search {deck.stem} did not finish in {MAX_TRIES} tries")
    return chosen(book, args)


def run_shards(jobs: list[tuple[list[str], Path, Path, dict]], deck: str) -> None:
    """Runs the shard searches (cmd, log, shard book, expected) at once, each restarted until it is
    accepted. However this ends (an error, Ctrl-C), no shard process outlives it."""
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    tries = [0] * len(jobs)
    running: dict[int, subprocess.Popen] = {}
    try:
        while True:
            for i, (cmd, log_path, shard, expected) in enumerate(jobs):
                if i in running or search_done(shard, expected):
                    continue
                if tries[i] >= MAX_TRIES:
                    raise SystemExit(f"search {deck} shard {i} did not finish in {MAX_TRIES} tries")
                tries[i] += 1
                with open(log_path, "a", encoding="utf-8") as log:
                    log.write("$ " + " ".join(cmd) + "\n")
                    log.flush()
                    running[i] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            if not running:
                return
            for i, proc in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                del running[i]
                print(f"search {deck} shard {i}: attempt {tries[i] - 1} exit {code}", flush=True)
                if accept(code, tries[i]):
                    mark_done(jobs[i][2], jobs[i][3], failed_hands(jobs[i][2]))
            time.sleep(5)
    finally:
        for proc in running.values():
            proc.terminate()
        for proc in running.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def search_sharded(args, deck: Path, deck_type: str, book: Path, expected: dict) -> Path:
    """`--shards` search processes at once, each restarted with --resume until it finishes; then
    their lines are merged and the deck's book is built (`--build-only`)."""
    ranges = shard_ranges(args.hands, args.shards)
    books = [shard_book(book, first, n) for first, n in ranges]
    books[0].parent.mkdir(parents=True, exist_ok=True)
    if not search_done(book, expected):
        run_shards([(search_cmd(args, deck, deck_type, shard, first, n, False), shard.with_suffix(".log"), shard,
                     {"first": first, "hands": n, "seed": args.seed}) for (first, n), shard in zip(ranges, books)],
                   deck.stem)
        merge_shards(book, books)
        meta = json.loads(books[0].with_name(f"{books[0].stem}-meta.json").read_text())
        code = run([sys.executable, "-m", "ocgforge.turn1_search", "--deck", meta["deck"], "--deck-type", deck_type,
                    "--opponent", meta["opponent"], "--checkpoint", str(args.net), "--build-only", "--score", args.score,
                    "--out", str(book)] + gain_args(args), args.out / f"{deck.stem}-build.log")
        if code != 0:
            raise SystemExit(f"building {book.name} failed")
        mark_done(book, expected, [h for shard in books for h in failed_hands(shard)])
    return chosen(book, args)


def chosen(book: Path, args) -> Path:
    """The book teach imitates: the --min-gain book (built by the search, or now from its lines)."""
    if args.min_gain is None:
        return book
    from .turn1_search import gain_path

    gain = gain_path(book)
    meta = gain.with_name(f"{gain.stem}-meta.json")
    held = json.loads(meta.read_text()) if meta.exists() else {}
    if not gain.exists() or held.get("min_gain") != args.min_gain or held.get("min_gain_se") != args.min_gain_se:
        deck_meta = json.loads(book.with_name(f"{book.stem}-meta.json").read_text())
        code = run([sys.executable, "-m", "ocgforge.turn1_search", "--deck", deck_meta["deck"], "--deck-type",
                    deck_meta["deck_type"], "--opponent", deck_meta["opponent"], "--checkpoint", str(args.net),
                    "--build-only", "--score", args.score, *gain_args(args), "--out", str(book)],
                   args.out / f"{book.stem}-build.log")
        if code != 0:
            raise SystemExit(f"building {gain.name} failed")
    return gain


def teach(args, books: list[Path]) -> None:
    """teach --book `books` (none: the control, plain distillation) into <out>/student (<out>/ctrl)."""
    out = args.out / (args.teach_dir or ("student" if books else "ctrl"))
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
    p.add_argument("--workers", type=int, default=4, help="search threads (per shard)")
    p.add_argument("--shards", type=int, default=1, help="search processes per deck")
    p.add_argument("--traps", type=int, default=4, help="hand-trap branches (turn1_search --traps; 0: none)")
    p.add_argument("--score", choices=("value", "turn2", "mix"), default="value", help="turn1_search --score")
    p.add_argument("--rollouts", type=int, default=2)
    p.add_argument("--mix-weight", type=float, default=0.5)
    p.add_argument("--t2-cap", type=int, default=250)
    p.add_argument("--validate", type=int, default=0, help="turn1_search --validate")
    p.add_argument("--opponents-file", type=Path, help="turn1_search --opponents-file (needed for turn2/mix)")
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
    p.add_argument("--min-gain-se", type=float, default=None, help="turn1_search --min-gain-se")
    p.add_argument("--race", action="store_true", help="turn1_search --race (racing, with the --race-* settings)")
    p.add_argument("--race-k1", type=int, default=8)
    p.add_argument("--race-k2", type=int, default=16)
    p.add_argument("--race-se", type=float, default=2.0)
    p.add_argument("--race-frac", type=float, default=0.75)
    p.add_argument("--race-margin", type=float, default=0.0)
    p.add_argument("--race-c", type=float, default=1.0)
    p.add_argument("--gain-orders", type=int, default=8, help="turn1_search --gain-orders")
    p.add_argument("--gain-rollouts", type=int, default=1, help="turn1_search --gain-rollouts")
    p.add_argument("--min-gain", type=float, default=0.1,
                   help="imitate only lines beating the net's own line by more than this (mean value); negative: all")
    p.add_argument("--search-only", action="store_true")
    p.add_argument("--book-files", type=Path, nargs="+",
                   help="teach with these books (built elsewhere, e.g. merged search runs) instead of searching")
    p.add_argument("--teach-dir", help="teach into <out>/<this> (default: student, or ctrl with --control)")
    p.add_argument("--control", action="store_true",
                   help="the control instead: the same teach rounds and seeds without a book, into <out>/ctrl")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.min_gain is not None and args.min_gain < 0:
        args.min_gain = None
    if args.control:
        teach(args, [])
        return
    if args.book_files:
        teach(args, [p.resolve() for p in args.book_files])
        return
    books = []
    for spec in args.books:
        deck, _, deck_type = spec.rpartition(":")
        books.append(search(args, (rollout.REPO / deck).resolve(), deck_type))
    if not args.search_only:
        teach(args, books)


if __name__ == "__main__":
    main()
