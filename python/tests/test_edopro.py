import pytest

from ocgforge.edopro import protocol as p

core = pytest.importorskip("ocgforge._core", reason="build with: cmake --preset python && cmake --build --preset python")


def test_packets_round_trip_and_join_layout():
    buffer = bytearray(p.join_game("pw") + p.hand_result(2) + p.packet(p.CTOS_HS_READY)[:2])
    packets = p.read_packets(buffer)
    assert [kind for kind, _ in packets] == [p.CTOS_JOIN_GAME, p.CTOS_HAND_RESULT]
    join = packets[0][1]
    assert len(join) == 52 and join[:2] == (0x1354).to_bytes(2, "little") and join[8:12] == "pw".encode("utf-16-le")
    assert join[48:] == bytes(p.CLIENT_VERSION)
    assert buffer == p.packet(p.CTOS_HS_READY)[:2]  # the partial packet stays


def test_client_duel_skips_what_it_cannot_read():
    duel = core.ClientDuel(0)
    assert not duel.feed(bytes([250, 1, 2, 3]))  # no such message type
    assert not duel.pending and duel.skipped and duel.messages == 1
    with pytest.raises(ValueError):
        core.ClientDuel(2)
