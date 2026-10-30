#!/usr/bin/env python3
"""Duel.clone() against replaying the actions into a new Duel, at many decisions of random duels.

    python tools/bench_clone.py --duels 6 --every 5

Prints the median times per decision point and the resident memory one clone adds.
Needs the Python module (cmake --preset python && cmake --build --preset python).
"""

from __future__ import annotations

import argparse
import gc
import random
import statistics
import time
from pathlib import Path

from ocgforge._core import Duel

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / ".cache"
PAIRS = [("swordsoul_tenyi", "salamangreat"), ("branded", "maliss"), ("doomz", "elfnote"),
         ("artmage", "blitzclique"), ("light_and_darkness_ritual", "branded"), ("maliss", "doomz")]


def new_duel(deck0: str, deck1: str, seed: int) -> Duel:
    return Duel(str(CACHE / "cardscripts"), str(CACHE / "card_data.tsv"), str(REPO / "decks" / f"{deck0}.ydk"),
                str(REPO / "decks" / f"{deck1}.ydk"), seed, seed % 2)


def rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duels", type=int, default=6)
    p.add_argument("--every", type=int, default=5, help="measure at every n-th decision")
    p.add_argument("--memory-clones", type=int, default=200)
    args = p.parse_args()

    clone_s, replay_s, create_s, depth = [], [], [], []
    for n in range(args.duels):
        deck0, deck1 = PAIRS[n % len(PAIRS)]
        seed = 1000 + n
        rng = random.Random(seed)
        duel = new_duel(deck0, deck1, seed)
        history: list[str] = []
        while not duel.finished:
            if len(history) % args.every == 0:
                start = time.perf_counter()
                copy = duel.clone()
                clone_s.append(time.perf_counter() - start)
                start = time.perf_counter()
                replay = new_duel(deck0, deck1, seed)
                create_s.append(time.perf_counter() - start)
                for key in history:
                    replay.act(key)
                replay_s.append(time.perf_counter() - start)
                assert replay.keys() == copy.keys() == duel.keys()
                depth.append(len(history))
                del copy, replay
            history.append(rng.choice(duel.keys()))
            duel.act(history[-1])

    ms = lambda xs: statistics.median(xs) * 1000  # noqa: E731
    print(f"{len(clone_s)} decision points over {args.duels} duels, median depth {statistics.median(depth):.0f} "
          f"decisions (max {max(depth)})")
    print(f"clone : median {ms(clone_s):.3f} ms  p90 {statistics.quantiles(clone_s, n=10)[-1] * 1000:.3f} ms")
    print(f"replay: median {ms(replay_s):.3f} ms  p90 {statistics.quantiles(replay_s, n=10)[-1] * 1000:.3f} ms "
          f"(new Duel + replaying the actions)")
    print(f"new Duel alone (to its first decision): median {ms(create_s):.3f} ms")
    early = [r / c for r, c, d in zip(replay_s, clone_s, depth) if d < 100]
    if early:
        print(f"speedup at depth < 100: median {statistics.median(early):.1f}x ({len(early)} points)")
    print(f"speedup (median replay / median clone): {statistics.median(replay_s) / statistics.median(clone_s):.1f}x")

    # Memory: many clones of one mid-duel decision, resident set before and after.
    rng = random.Random(7)
    duel = new_duel("branded", "maliss", 7)
    for _ in range(60):
        if duel.finished:
            break
        duel.act(rng.choice(duel.keys()))
    gc.collect()
    before = rss_bytes()
    clones = [duel.clone() for _ in range(args.memory_clones)]
    after = rss_bytes()
    print(f"memory: {(after - before) / len(clones) / 1024:.0f} KiB resident per clone "
          f"({len(clones)} clones at decision {duel.decision_count})")
    del clones


if __name__ == "__main__":
    main()
