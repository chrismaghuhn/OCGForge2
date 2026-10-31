"""aliases: every nickname names exactly one card of the pinned database."""

from ocgforge import aliases
from ocgforge import combos as cb
from ocgforge import seeds as sd

# Nicknames that are also the name of another card ("Tragedy" is a card; players mean Despian Tragedy).
SHADOWS = {"tragedy"}


def test_keys_are_lower_case_and_trimmed():
    for key in aliases.ALIASES:
        assert key == key.strip().lower(), key


def test_every_alias_resolves_to_one_card(cdb_table):
    official = [c for c in cdb_table.cards if c.code < 100_000_000]
    by_name: dict[str, set[int]] = {}
    for card in official:
        by_name.setdefault(cb.normalize(card.name), set()).add(card.code)
    names = cb.name_index(cdb_table)
    bad = [(k, n) for k, n in aliases.ALIASES.items() if len(by_name.get(cb.normalize(n), ())) != 1]
    assert not bad, f"not exactly one card: {bad}"
    # A nickname that is itself another card's name hides that card: only on purpose.
    shadows = {k for k, n in aliases.ALIASES.items()
               if names.get(cb.normalize(k), names[cb.normalize(n)]) != names[cb.normalize(n)]}
    assert shadows == SHADOWS


def test_no_conflict_with_seeds_aliases():
    for key, name in aliases.ALIASES.items():
        if key in sd.ALIASES:
            assert sd.ALIASES[key] == name, key


def test_with_aliases_wraps_a_resolver():
    cards = {"Mirrorjade the Iceblade Dragon": 2, "Jade Dragon Mech": 3, "Elfnote Power Patron": 4}
    names = {cb.normalize(n): c for n, c in cards.items()}
    base = sd.resolver(names, set(cards.values()), {c: n for n, c in cards.items()})
    assert base("Jade") is None and base("PP") is None  # ambiguous / unknown in this pool
    resolve = aliases.with_aliases(base)
    assert resolve("Jade") == 2 and resolve("Jade in GY.") == 2 and resolve("PP") == 4
    assert resolve("Jade Dragon Mech") == 3
