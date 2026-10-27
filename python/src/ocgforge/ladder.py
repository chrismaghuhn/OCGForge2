"""Elo ladder and matchup statistics over duel results, ported from Argentum P1.

A result is a dict: {"a": player, "b": player, "winner": "a" | "b" | "draw", "deck_a": ..., "deck_b": ...}
(deck fields optional). Ratings are a Bradley-Terry maximum-likelihood fit, draws count as half a
win each way, anchored so `anchor` (default "random") sits at 1500.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

ANCHOR_RATING = 1500.0
ELO_SCALE = 400.0 / math.log(10.0)


def _score_a(result: dict) -> float:
    return {"a": 1.0, "b": 0.0, "draw": 0.5}[result["winner"]]


@dataclass
class MatchupStats:
    player: str
    opponent: str
    games: int = 0
    points: float = 0.0
    by_deck: dict[str, list[float]] = field(default_factory=dict)  # deck -> [games, points]

    @property
    def win_rate(self) -> float:
        return self.points / self.games if self.games else 0.0


def matchups(results: list[dict]) -> dict[tuple[str, str], MatchupStats]:
    """Per ordered (player, opponent) statistics, counting each duel from both sides."""
    stats: dict[tuple[str, str], MatchupStats] = {}
    for result in results:
        score_a = _score_a(result)
        for player, opponent, score, deck in (
            (result["a"], result["b"], score_a, result.get("deck_a", "?")),
            (result["b"], result["a"], 1.0 - score_a, result.get("deck_b", "?")),
        ):
            s = stats.setdefault((player, opponent), MatchupStats(player, opponent))
            s.games += 1
            s.points += score
            deck_stats = s.by_deck.setdefault(deck, [0, 0.0])
            deck_stats[0] += 1
            deck_stats[1] += score
    return stats


def elo(results: list[dict], anchor: str = "random", iterations: int = 500, prior_games: float = 1.0) -> dict[str, float]:
    """Bradley-Terry ratings in Elo units. `anchor` is fixed at 1500 (first player if absent); one
    virtual draw against it keeps players with only wins or only losses finite."""
    players = sorted({r["a"] for r in results} | {r["b"] for r in results})
    if not players:
        return {}
    if anchor not in players:
        anchor = players[0]
    wins: dict[tuple[str, str], float] = defaultdict(float)
    games: dict[tuple[str, str], float] = defaultdict(float)
    for result in results:
        a, b = result["a"], result["b"]
        score = _score_a(result)
        wins[(a, b)] += score
        wins[(b, a)] += 1.0 - score
        games[(a, b)] += 1
        games[(b, a)] += 1
    for p in players:
        if p != anchor:
            wins[(p, anchor)] += prior_games / 2
            wins[(anchor, p)] += prior_games / 2
            games[(p, anchor)] += prior_games
            games[(anchor, p)] += prior_games
    strength = {p: 1.0 for p in players}
    for _ in range(iterations):
        updated = {}
        for p in players:
            total_wins = sum(w for (x, _), w in wins.items() if x == p)
            denominator = sum(n / (strength[p] + strength[o]) for (x, o), n in games.items() if x == p)
            updated[p] = total_wins / denominator if denominator > 0 and total_wins > 0 else strength[p]
        norm = updated[anchor]
        strength = {p: v / norm for p, v in updated.items()}
    return {p: round(ANCHOR_RATING + ELO_SCALE * math.log(strength[p]), 1) for p in players}
