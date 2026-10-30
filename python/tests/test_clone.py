"""Duel.clone (the ocgcore fork's OCG_CloneDuel): a clone plays on exactly like its original."""

import os
import random

import numpy as np
import pytest

from ocgforge.cards import CardFeatures, CardTable

from conftest import CDB, REPO

if os.environ.get("OCGFORGE_REQUIRE_CORE"):
    from ocgforge import _core as core
else:
    core = pytest.importorskip("ocgforge._core", reason="build with: cmake --preset python && cmake --build --preset python")

CACHE = REPO / ".cache"
PAIRS = [("swordsoul_tenyi", "salamangreat"), ("branded", "maliss"), ("doomz", "elfnote"),
         ("artmage", "blitzclique"), ("light_and_darkness_ritual", "branded")]


@pytest.fixture(scope="module")
def card_index() -> dict[int, int]:
    if not CDB.exists():
        pytest.skip("card database not fetched")
    return CardFeatures.build(CardTable.from_cdb()).index_map()


def new_duel(deck0: str, deck1: str, seed: int):
    return core.Duel(str(CACHE / "cardscripts"), str(CACHE / "card_data.tsv"), str(REPO / "decks" / f"{deck0}.ydk"),
                     str(REPO / "decks" / f"{deck1}.ydk"), seed, seed % 2)


def view(duel, card_index) -> tuple:
    """What can be told apart without the engine's message order: answers, model input, outcome."""
    if duel.finished:
        return ("finished", duel.winner, duel.decision_count)
    row = {k: np.asarray(v).tolist() for k, v in duel.encode(card_index).items()}
    return (duel.player, duel.decision_count, tuple(duel.keys()), sorted(row.items()), duel.snapshot())


def lockstep(a, b, card_index, rng: random.Random, steps: int, *others) -> int:
    """Plays a and b (and others, unchecked) on with the same random answers; a and b must agree."""
    done = 0
    while done < steps and not a.finished:
        key = rng.choice(a.keys())
        for duel in (a, b, *others):
            duel.act(key)
        assert view(a, card_index) == view(b, card_index)
        done += 1
    return done


@pytest.mark.parametrize("deck0,deck1", PAIRS, ids=lambda d: d)
def test_clone_plays_on_like_the_original(card_index, deck0, deck1):
    seed = sum(map(ord, deck0 + deck1))
    rng = random.Random(seed)
    duel = new_duel(deck0, deck1, seed)
    never_cloned = new_duel(deck0, deck1, seed)  # cloning must not change the source either
    clone_points = steps = 0
    while not duel.finished:
        if duel.decision_count % 7 == 0:
            copy = duel.clone()
            assert view(copy, card_index) == view(duel, card_index) == view(never_cloned, card_index)
            clone_points += 1
            steps += lockstep(duel, copy, card_index, rng, 60, never_cloned)
            continue
        key = rng.choice(duel.keys())
        duel.act(key)
        never_cloned.act(key)
    assert view(duel, card_index) == view(never_cloned, card_index)
    assert clone_points >= 2 and steps >= 60


def test_clone_outlives_its_original_and_matches_a_replay(card_index):
    rng = random.Random(3)
    duel = new_duel("maliss", "swordsoul_tenyi", 3)
    history = []
    for _ in range(30):
        history.append(rng.choice(duel.keys()))
        duel.act(history[-1])
    assert not duel.finished
    copy = duel.clone()
    del duel
    replay = new_duel("maliss", "swordsoul_tenyi", 3)
    for key in history:
        replay.act(key)
    assert view(copy, card_index) == view(replay, card_index)
    while not replay.finished:
        lockstep(replay, copy, card_index, rng, 1000)
    assert copy.finished and copy.winner == replay.winner


def test_reseeded_clone_and_finished_duel(card_index):
    rng = random.Random(5)
    duel = new_duel("doomz", "branded", 5)
    for _ in range(10):
        duel.act(rng.choice(duel.keys()))
    reseeded = duel.clone(seed=99)
    assert reseeded.reseeds == [99] and duel.reseeds == []
    assert reseeded.keys() == duel.keys()
    while not reseeded.finished:
        reseeded.act(rng.choice(reseeded.keys()))
    while not duel.finished:
        duel.act(rng.choice(duel.keys()))
    with pytest.raises(Exception):
        duel.clone()


def test_concurrent_clones_of_one_duel(card_index):
    """Search clones one root from many threads at once (the GIL is released): every such clone must
    match a clone made alone, and the root must not change."""
    from concurrent.futures import ThreadPoolExecutor

    deck0, deck1 = PAIRS[1]
    rng = random.Random(5)
    duel = new_duel(deck0, deck1, 5)
    for _ in range(40):
        duel.act(rng.choice(duel.keys()))
    assert not duel.finished
    before = view(duel, card_index)
    with ThreadPoolExecutor(16) as pool:
        for _ in range(4):
            clones = list(pool.map(lambda _: duel.clone(), range(32)))
            assert view(duel, card_index) == before
            reference = duel.clone()
            moves = random.Random(len(clones))
            for clone in clones[:8]:
                lockstep(reference.clone(), clone, card_index, random.Random(moves.random()), 40)


def test_collected_template_plays_on_like_the_original(card_index):
    """Search fan-out: collect the garbage of one private clone, then clone that template."""
    rng = random.Random(5)
    duel = new_duel("branded", "maliss", 5)
    for _ in range(150):
        duel.act(rng.choice(duel.keys()))
    used = duel.lua_memory["used"]
    template = duel.clone()
    template.collect_lua_garbage()
    assert template.lua_memory["used"] <= used
    assert duel.lua_memory["used"] == used  # the source is untouched
    assert lockstep(duel.clone(), template.clone(), card_index, random.Random(6), 300) > 0


def test_duels_can_be_created_and_encoded_from_threads(card_index):
    """Duel() and Duel.encode release the GIL; duels built on threads match ones built serially."""
    from concurrent.futures import ThreadPoolExecutor

    def build(seed):
        duel = new_duel(*PAIRS[seed % len(PAIRS)], seed)
        return view(duel, card_index)

    seeds = list(range(40, 52))
    with ThreadPoolExecutor(4) as pool:
        threaded = list(pool.map(build, seeds))
    assert threaded == [build(seed) for seed in seeds]
