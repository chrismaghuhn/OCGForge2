import numpy as np

from ocgforge.embed_cards import card_sentence, load_text_embeddings

from conftest import make_table


def test_sentence_has_type_line_and_replaces_own_name():
    warrior, link, spell = make_table().cards
    assert card_sentence(warrior).startswith("Monster Effect, LIGHT, Warrior, Level 4. Once per turn: this card")
    assert card_sentence(link).startswith("Monster Effect Link, DARK, Cyberse, Link 2.")
    assert card_sentence(spell) == "Spell. Draw 2 cards."


def test_vectors_land_on_card_indices(card_features, tmp_path):
    path = tmp_path / "emb.npz"
    np.savez(path, codes=np.array([1003, 1001, 4242]), vectors=np.array([[3.0, 3.0], [1.0, 1.0], [9.0, 9.0]]))
    rows = load_text_embeddings(path, card_features)
    index = card_features.index_map()
    assert rows.shape == (card_features.size, 2)
    assert rows[index[1001]].tolist() == [1.0, 1.0] and rows[index[1003]].tolist() == [3.0, 3.0]
    assert rows[index[1002]].tolist() == [0.0, 0.0]
