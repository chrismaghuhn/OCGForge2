import pytest

from ocgforge.ladder import elo, matchups


def _results(a, b, wins_a, wins_b, draws=0):
    return ([{"a": a, "b": b, "winner": "a"}] * wins_a + [{"a": a, "b": b, "winner": "b"}] * wins_b
            + [{"a": a, "b": b, "winner": "draw"}] * draws)


def test_anchor_stays_at_1500_and_stronger_player_rates_higher():
    ratings = elo(_results("model", "random", 30, 10) + _results("old", "random", 20, 20))
    assert ratings["random"] == 1500.0
    assert ratings["model"] > ratings["old"] > 1400
    assert ratings["old"] == pytest.approx(1500.0, abs=1.0)


def test_only_wins_stay_finite():
    ratings = elo(_results("model", "random", 10, 0))
    assert 1500.0 < ratings["model"] < 2500.0


def test_matchups_count_both_sides_and_decks():
    results = [{"a": "m", "b": "r", "winner": "a", "deck_a": "swordsoul", "deck_b": "salamangreat"},
               {"a": "r", "b": "m", "winner": "draw", "deck_a": "swordsoul", "deck_b": "salamangreat"}]
    stats = matchups(results)
    assert stats[("m", "r")].games == 2 and stats[("m", "r")].points == 1.5
    assert stats[("m", "r")].by_deck == {"swordsoul": [1, 1.0], "salamangreat": [1, 0.5]}
    assert stats[("r", "m")].win_rate == pytest.approx(0.25)
