import numpy as np

from ocgforge.cards import CARD_NUMERIC, UNK, CardFeatures, text_terms, tokenize

from conftest import REPO, make_table


def test_tokenize_replaces_own_name_and_marks_lines():
    tokens = tokenize('If "Test Warrior" is destroyed:\nDraw 1 card.', "Test Warrior")
    assert tokens == ["if", "~", "is", "destroyed", "<nl>", "draw", "1", "card", "<nl>"]
    assert "~ is" in text_terms('"Test Warrior" is', "Test Warrior")


def test_alternate_artwork_shares_the_card_index():
    table = make_table()
    assert table.index(1004) == table.index(1001)
    assert table.index(999999) == UNK


def test_features_for_link_monster(card_features):
    link = card_features.index_map()[1002]
    numeric = dict(zip(CARD_NUMERIC, card_features.numeric[link]))
    assert numeric["level"] == 2 / 12
    assert numeric["hasDef"] == 0.0 and numeric["def"] == 0.0
    assert numeric["linkMarker1"] == 1.0 and numeric["linkMarker7"] == 1.0  # bottom and top (0o202)
    assert card_features.attribute[link] == 5 + 2  # DARK = bit 5
    assert card_features.race[link] == 24 + 2  # Cyberse = bit 24


def test_setcodes_include_base_archetype(card_features):
    warrior = card_features.index_map()[1001]
    ids = {i for i in card_features.setcodes[warrior] if i}
    expected = {card_features.setcode_vocab[0x1234], card_features.setcode_vocab[0x0234]}
    assert ids == expected


def test_untrained_cards_are_not_seen():
    features = CardFeatures.build(make_table(), trained_codes={1001}, min_term_count=1)
    index = features.index_map()
    assert features.seen[index[1001]] and not features.seen[index[1003]]
    assert features.seen[0] and features.seen[UNK]


def test_json_roundtrip_keeps_vocabularies(card_features):
    restored = CardFeatures.from_json(card_features.to_json())
    assert restored.codes == card_features.codes
    assert restored.text_vocab == card_features.text_vocab
    assert restored.index_map() == card_features.index_map()


def test_real_database_covers_both_decks(cdb_table):
    assert len(cdb_table) > 10_000
    for deck in ("swordsoul_tenyi.ydk", "salamangreat.ydk"):
        for line in (REPO / "decks" / deck).read_text().splitlines():
            if line.strip().isdigit():
                assert cdb_table.index(int(line)) != UNK, (deck, line)


def test_real_database_features_build(cdb_table):
    features = CardFeatures.build(cdb_table)
    assert features.text.shape[0] == len(cdb_table)
    assert np.count_nonzero(features.text[2:].any(axis=1)) > 0.95 * (len(cdb_table) - 2)
