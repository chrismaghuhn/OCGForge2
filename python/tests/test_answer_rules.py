"""answer_rules: look-ahead on clones for card choices."""

from collections import Counter

from ocgforge.answer_rules import choice_by_outcome

REGINA, CANCELLER, FORTUNA = 56651978, 34980312, 85976588
DECK, HAND, GY = 0x1, 0x2, 0x10


class Choice:
    """Shooting Riser's send: pick a deck card; it goes to the GY, then an idle question."""

    def __init__(self, cards, picked=False):
        self.cards, self.picked = dict(cards), picked
        self.finished, self.player = False, 0

    def clone(self):
        return Choice(self.cards, self.picked)

    def keys(self):
        if self.picked:
            return ["idle.7.0"]
        return [f"card.0.{i}.{code}.0.1.{i}" for i, code in enumerate(sorted(self.cards))]

    def act(self, key):
        self.cards[int(key.split(".")[3])] = GY
        self.picked = True

    def snapshot(self):
        return {"cards": [{"controller": 0, "code": c, "location": loc} for c, loc in self.cards.items()]}


def test_choice_by_outcome_prefers_the_card_that_arrives_for_a_coming_step():
    duel = Choice({REGINA: DECK, CANCELLER: DECK, FORTUNA: DECK})
    keys = duel.keys()
    before = Counter((c, loc) for c, loc in duel.cards.items())
    follow_args = ([], False, set(), lambda c: False, [REGINA, CANCELLER])
    # Regina's coming step is a return to hand: sending her does not do it; Canceller's GY step does
    chosen = choice_by_outcome(duel, keys, set(range(3)), before, [(REGINA, HAND), (CANCELLER, GY)], follow_args)
    assert [keys[i] for i in chosen] == [k for k in keys if str(CANCELLER) in k]
    # nothing arrives where a step wants it: unchanged
    assert choice_by_outcome(duel, keys, {0, 1, 2}, before, [(REGINA, HAND)], follow_args) == {0, 1, 2}


class Draw(Choice):
    """After the pick, the effect draws the top card of the deck as well."""

    def act(self, key):
        super().act(key)
        top = next(code for code, loc in self.cards.items() if loc == DECK)
        self.cards[top] = HAND

    def clone(self):
        return Draw(self.cards, self.picked)


def test_no_look_ahead_when_the_deck_order_would_show():
    from ocgforge.answer_rules import follow
    duel = Draw({REGINA: DECK, CANCELLER: DECK, FORTUNA: DECK})
    key = duel.keys()[0]
    assert follow(duel, key, [], False, set(), lambda c: False, []) == (set(), None)
    plain = Choice({REGINA: DECK, CANCELLER: DECK, FORTUNA: DECK})
    assert follow(plain, plain.keys()[0], [], False, set(), lambda c: False, [])[1] is not None
