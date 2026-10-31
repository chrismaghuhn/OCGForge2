"""opening_book: the hand-trap bot of the interruption branches."""

from ocgforge import opening_book as ob


class Bot:
    def __init__(self, keys):
        self._keys = keys

    def keys(self):
        return self._keys


def test_hand_trap_bot_waits_for_its_chokepoint():
    ash, nadir, aluber = 14558127, 87462901, 73819701
    bot = ob.hand_trap_bot(ash, {nadir})
    window = ["chain.pass", f"chain.0.0.{ash}.1.2.0"]
    assert bot(Bot(window), [f"idle.5.0.{aluber}.0.2.0"]) == "chain.pass"  # not a choke point
    assert bot(Bot(window), [f"idle.5.0.{nadir}.0.2.0", "chain.pass"]) == f"chain.0.0.{ash}.1.2.0"
    anywhere = ob.hand_trap_bot(ash, set())
    assert anywhere(Bot(window), [f"idle.5.0.{aluber}.0.2.0"]) == f"chain.0.0.{ash}.1.2.0"
    # after firing, a target question aims at a choke-point card
    fired = [f"idle.5.0.{nadir}.0.2.0", f"chain.0.0.{ash}.1.2.0"]
    assert bot(Bot([f"card.0.0.{aluber}.0.4.0", f"card.0.1.{nadir}.0.4.1"]), fired) == f"card.0.1.{nadir}.0.4.1"
