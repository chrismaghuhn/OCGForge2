from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ocgforge import features
from ocgforge.cards import Card, CardFeatures, CardTable

REPO = Path(__file__).resolve().parents[2]
CDB = REPO / ".cache" / "babelcdb" / "cards.cdb"


@pytest.fixture(scope="session")
def cdb_table() -> CardTable:
    if not CDB.exists():
        pytest.skip("card database not fetched (run cmake --preset debug or tools/setup_deps.py)")
    return CardTable.from_cdb(CDB)


def make_table() -> CardTable:
    """Three small cards: a monster, a Link monster and a spell."""
    cards = [
        Card(1001, "Test Warrior", 'Once per turn: "Test Warrior" gains 500 ATK.', 0x21, 0x10, 0x1, 4, 1800, 1200, 0x1234),
        Card(1002, "Test Link", "2 Effect Monsters\nDraw 1 card.", 0x4000021, 0x20, 0x1000000, 2, 1600, 0o202, 0x5678),
        Card(1003, "Test Spell", "Draw 2 cards.", 0x2, 0, 0, 0, 0, 0, 0),
    ]
    table = CardTable(cards=cards, index_of={c.code: i + 2 for i, c in enumerate(cards)})
    table.index_of[1004] = table.index_of[1001]  # alternate artwork
    return table


@pytest.fixture
def card_features() -> CardFeatures:
    return CardFeatures.build(make_table(), min_term_count=1)


def random_decision(rng: np.random.Generator, n_cards: int, tokens: int | None = None,
                    candidates: int | None = None) -> dict:
    t = int(rng.integers(0, 12)) if tokens is None else tokens
    c = int(rng.integers(1, 8)) if candidates is None else candidates
    return {
        "token_card": rng.integers(1, n_cards, t),
        "token_zone": rng.integers(1, len(features.ZONES), t),
        "token_side": rng.integers(0, 2, t),
        "token_numeric": rng.random((t, len(features.TOKEN_NUMERIC)), dtype=np.float32),
        "global_numeric": rng.random(len(features.GLOBAL_NUMERIC), dtype=np.float32),
        "phase": int(rng.integers(1, len(features.PHASES))),
        "decision": int(rng.integers(1, len(features.DECISIONS))),
        "candidate_kind": rng.integers(1, len(features.CANDIDATE_KINDS), c),
        "candidate_source": rng.integers(-1, max(t, 1), c) if t else np.full(c, -1),
        "candidate_targets": rng.integers(-1, max(t, 1), (c, features.MAX_TARGETS)) if t
        else np.full((c, features.MAX_TARGETS), -1),
        "candidate_numeric": rng.random((c, len(features.CANDIDATE_NUMERIC)), dtype=np.float32),
    }
