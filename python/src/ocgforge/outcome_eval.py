"""Build against break: a turn-1 line scored by what the opponent's turn 2 does to it.

The value head at the end of our turn 1 rewards a big board, not one that holds (turn1-exit,
iteration 2: bigger boards, fewer wins). Here a finished turn-1 line is played on: from the end of
our turn 1 the opponent (its real hand and deck of that scenario) plays turn 2 with the net, sampling
from the policy, and our seat answers with the net too (set traps, negates, quick effects). The
score is the net's value at our first decision of turn 3 (seat 0's view), or the result (+-1, 0 for a
draw) if the duel ended. `rollouts` M samples per end position, all positions in lockstep so that
every net call is one batch.

Scorers (`turn1_search --score`):
    value   the end-of-turn-1 value (as before; no rollouts)
    turn2   the mean turn-2 outcome over the K scenarios x M rollouts
    mix     w * value + (1 - w) * turn2 (`--mix-weight` w)
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

from . import worlds

SCORERS = ("value", "turn2", "mix")
FIELD = (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)
MAX_FORCED = 2000  # single answers per rollout at most (a looping position must not hang the search)


@dataclass
class Rollout:
    """One sampled turn 2 from one end-of-turn-1 position."""
    duel: object
    rng: random.Random
    decisions: int = 0  # sampled (real) decisions of both seats
    forced: int = 0  # single answers played
    capped: bool = False
    error: bool = False
    value: float | None = None  # seat 0: value at our first turn-3 decision, or the result
    stats: dict = field(default_factory=dict)


def turn3(duel) -> bool:
    """Done: the duel ended, or it is our (seat 0's) decision in turn 3 or later."""
    return duel.finished or (duel.turn >= 3 and duel.player == 0)


def sample(logprobs: list[float], rng: random.Random) -> int:
    """An index drawn from the softmax the log-probabilities describe."""
    weights = [math.exp(lp) for lp in logprobs]
    return rng.choices(range(len(weights)), weights)[0]


def end_stats(duel) -> dict:
    """What turn 2 left: lost/won in turn 2, our field cards, both life points."""
    if duel.finished:
        lost, won = duel.winner == 1, duel.winner == 0
        return {"lost": lost, "won": won, "finished": True, "field": 0, "lp": 0 if lost else None,
                "opp_lp": 0 if won else None}
    snap = duel.snapshot()
    ours = sum(c["controller"] == 0 and c["location"] in FIELD for c in snap["cards"])
    lp = list(snap["life_points"])
    return {"lost": False, "won": False, "finished": False, "field": ours, "lp": lp[0], "opp_lp": lp[1]}


def engine_seed(text: str) -> int:
    """A stable 63-bit engine seed from a seed string."""
    return random.Random(f"engine-{text}").getrandbits(63)


def clone(duel, seed: int):
    """`duel.clone(seed)`: the engine reseeded (duels without a seed parameter, e.g. test toys, clone plainly)."""
    try:
        return duel.clone(seed)
    except TypeError:
        return duel.clone()


def play_turn2(duels: list, net, rollouts: int, seeds: list, max_decisions: int = 250) -> list[list[Rollout]]:
    """M = `rollouts` sampled turn 2s from each duel (cloned; `duels` stay untouched). `seeds[i]`
    seeds duel i's rollouts (rollout j: `{seed}-{j}`), so the same scenario gets the same samples for
    every candidate line: the policy samples and the engine's random state (`clone(seed)`, coin
    flips, shuffles) are the same for every candidate. A finished duel is not cloned (the engine
    cannot): its rollouts are its result. `net` is a `turn1_search.Net` (ask: log-probabilities and seat-0 value;
    terminal: seat-0 value of a pending decision or the result). A rollout that reaches
    `max_decisions` sampled decisions stops there and is scored at its pending decision (`capped`).
    Every net call of all rollouts is batched."""
    out: list[list[Rollout]] = []
    for duel, seed in zip(duels, seeds):
        row = []
        for j in range(rollouts):
            r = Rollout(None, random.Random(f"{seed}-{j}"))
            try:
                r.duel = duel if duel.finished else clone(duel, engine_seed(f"{seed}-{j}"))
            except Exception:
                r.error = True
            row.append(r)
        out.append(row)
    active = [r for row in out for r in row if not r.error]
    while active:
        asking = []
        for r in active:
            try:
                while not turn3(r.duel) and r.forced < MAX_FORCED:
                    options = r.duel.keys()
                    if len(options) > 1:
                        break
                    r.duel.act(options[0])
                    r.forced += 1
            except Exception:
                r.error = True
                continue
            if turn3(r.duel):
                continue
            if r.decisions >= max_decisions or r.forced >= MAX_FORCED:
                r.capped = True
                continue
            asking.append(r)
        if asking:
            for r, (lp, _) in zip(asking, net.ask([r.duel for r in asking])):
                options = r.duel.keys()
                r.decisions += 1
                try:
                    r.duel.act(options[sample(lp[:len(options)], r.rng)])
                except Exception:
                    r.error = True
        active = [r for r in asking if not r.error]
    done = [r for row in out for r in row if not r.error]
    try:
        values = net.terminal([r.duel for r in done])
    except Exception:  # a broken encoding: score nothing rather than lose the hand
        values = [None] * len(done)
    for r, value in zip(done, values):
        r.value = value
        try:
            r.stats = end_stats(r.duel)
        except Exception:
            r.stats = {}
    for row in out:
        for r in row:
            r.duel = None
    return out


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def summary(rollouts: list[Rollout]) -> dict:
    """Mean outcome of rollouts (those without an engine error): score, lost/won in turn 2, our field
    and life points and the opponent's at turn 3, capped share, sampled decisions."""
    ok = [r for r in rollouts if not r.error and r.value is not None]
    return {"t2": mean(r.value for r in ok), "n": len(ok), "lost": mean(r.stats.get("lost") for r in ok),
            "won": mean(r.stats.get("won") for r in ok), "field3": mean(r.stats.get("field") for r in ok),
            "lp": mean(r.stats.get("lp") for r in ok), "opp_lp": mean(r.stats.get("opp_lp") for r in ok),
            "capped": mean(r.capped for r in ok), "decisions": mean(r.decisions for r in ok)}


def summary_of(summaries: list[dict]) -> dict:
    """The mean of several `summary` dicts (e.g. one line's scenarios), key by key."""
    keys = [k for k in (summaries[0] if summaries else {}) if k != "n"]
    return {k: mean(s.get(k) for s in summaries) for k in keys}


def combine(scorer: str, value: float | None, t2: float | None, weight: float = 0.5) -> float | None:
    """A play's score under `scorer` from its end-of-turn-1 value and its mean turn-2 outcome."""
    if scorer == "value":
        return value
    if scorer == "turn2":
        return t2
    return None if t2 is None or value is None else weight * value + (1 - weight) * t2
