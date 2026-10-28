"""EDOPro network packets (client side), for EDOPro 41.0.x (PRO_VERSION 0x1354, ocgcore API 11.0).

A packet is u16 length (little endian, counting the type byte and the payload), u8 type, payload.
Layouts follow EDOPro's gframe/network.h; OCGForge-Ignis I1/I2 checked them against a real server.
"""

from __future__ import annotations

import struct
from pathlib import Path

PRO_VERSION = 0x1354
CLIENT_VERSION = (41, 0, 11, 0)  # EDOPro major, minor; ocgcore API major, minor

# client to server
CTOS_RESPONSE = 0x01
CTOS_UPDATE_DECK = 0x02
CTOS_HAND_RESULT = 0x03
CTOS_TP_RESULT = 0x04
CTOS_PLAYER_INFO = 0x10
CTOS_CREATE_GAME = 0x11
CTOS_JOIN_GAME = 0x12
CTOS_LEAVE_GAME = 0x13
CTOS_TIME_CONFIRM = 0x15
CTOS_HS_TODUELIST = 0x20
CTOS_HS_READY = 0x22
CTOS_HS_START = 0x25
CTOS_REMATCH_RESPONSE = 0xF0

# server to client
STOC_GAME_MSG = 0x01
STOC_ERROR_MSG = 0x02
STOC_SELECT_HAND = 0x03
STOC_SELECT_TP = 0x04
STOC_HAND_RESULT = 0x05
STOC_TP_RESULT = 0x06
STOC_CHANGE_SIDE = 0x07
STOC_WAITING_SIDE = 0x08
STOC_CREATE_GAME = 0x11
STOC_JOIN_GAME = 0x12
STOC_TYPE_CHANGE = 0x13
STOC_LEAVE_GAME = 0x14
STOC_DUEL_START = 0x15
STOC_DUEL_END = 0x16
STOC_REPLAY = 0x17
STOC_TIME_LIMIT = 0x18
STOC_CHAT = 0x19
STOC_HS_PLAYER_ENTER = 0x20
STOC_HS_PLAYER_CHANGE = 0x21
STOC_HS_WATCH_CHANGE = 0x22
STOC_CATCHUP = 0xF0
STOC_REMATCH = 0xF1
STOC_WAITING_REMATCH = 0xF2
STOC_CHAT_2 = 0xF3
STOC_NEW_REPLAY = 0x30

STOC_NAMES = {v: k for k, v in globals().items() if k.startswith("STOC_") and isinstance(v, int)}

PLAYERCHANGE_READY = 0x9
SERVER_HANDSHAKE = 4043399681

MSG_RETRY = 1
MSG_START = 4
MSG_WIN = 5

DUEL_MODE_MR5 = 0x2E800  # Master Rule 2020: the rules the model is trained on

ERRORS = {1: "join error", 2: "deck error", 3: "side error", 4: "version error", 5: "version error"}


def packet(kind: int, payload: bytes = b"") -> bytes:
    if len(payload) + 1 > 0xFFFF:
        raise ValueError(f"packet payload of {len(payload)} bytes is too long")
    return struct.pack("<HB", len(payload) + 1, kind) + payload


def read_packets(buffer: bytearray) -> list[tuple[int, bytes]]:
    """Complete packets at the front of `buffer` (removed from it); a partial one stays."""
    packets = []
    while len(buffer) >= 2:
        (length,) = struct.unpack_from("<H", buffer)
        if length == 0:
            raise ValueError("packet with length 0")
        if len(buffer) < 2 + length:
            break
        packets.append((buffer[2], bytes(buffer[3:2 + length])))
        del buffer[:2 + length]
    return packets


def utf16(text: str, units: int = 20) -> bytes:
    """Fixed-width UTF-16LE text, zero padded (the last unit always 0)."""
    data = text.encode("utf-16-le")[: (units - 1) * 2]
    return data + b"\0" * (units * 2 - len(data))


def player_info(name: str) -> bytes:
    return packet(CTOS_PLAYER_INFO, utf16(name))


def join_game(password: str = "", game_id: int = 0) -> bytes:
    # u16 version, 2 bytes padding, u32 game id, password[20], client version (4 x u8): 52 bytes
    payload = struct.pack("<HxxI", PRO_VERSION, game_id) + utf16(password) + bytes(CLIENT_VERSION)
    return packet(CTOS_JOIN_GAME, payload)


def create_game(password: str, duel_flags: int = 0x2E800, time_limit: int = 0) -> bytes:
    """Create a 1v1 room (a server such as Multirole; EDOPro's own LAN host cannot be created
    remotely): no deck checks, any deck size, Master Rule 2020 unless duel_flags say otherwise.
    HostInfo is 68 bytes (gframe/network.h), then name[20], pass[20], notes[200]."""
    host_info = struct.pack(
        "<IBBBBB3xIBBHII4siiiIIH6H2x",  # deck limits right after extra rules, then padding
        0, 3, 0, 0, 1, 0,  # banlist hash, allowed cards (all), mode, duel rule, no deck check, shuffle
        8000, 5, 1, time_limit,  # LP, starting hand, draw per turn, seconds (0 = none)
        duel_flags >> 32, SERVER_HANDSHAKE, bytes(CLIENT_VERSION),
        1, 1, 1,  # one player per team, best of 1
        duel_flags & 0xFFFFFFFF, 0, 0,  # forbidden types, extra rules
        0, 999, 0, 999, 0, 999)  # main, extra, side deck size limits
    assert len(host_info) == 68
    return packet(CTOS_CREATE_GAME, host_info + utf16("") + utf16(password) + bytes(200))


def update_deck(main_and_extra: list[int], side: list[int]) -> bytes:
    codes = [*main_and_extra, *side]
    return packet(CTOS_UPDATE_DECK, struct.pack(f"<II{len(codes)}I", len(main_and_extra), len(side), *codes))


def hand_result(choice: int) -> bytes:
    """Rock paper scissors: 1, 2 or 3."""
    return packet(CTOS_HAND_RESULT, bytes([choice]))


def tp_result(go_first: bool) -> bytes:
    return packet(CTOS_TP_RESULT, bytes([1 if go_first else 0]))


def response(data: bytes) -> bytes:
    return packet(CTOS_RESPONSE, data)


def duel_flags(host_info: bytes) -> int:
    """The room's duel flags from STOC_JOIN_GAME's HostInfo (duel_flag_high at 20, duel_flag_low at 44)."""
    high, low = struct.unpack_from("<I", host_info, 20)[0], struct.unpack_from("<I", host_info, 44)[0]
    return high << 32 | low


def read_ydk(path: Path) -> tuple[list[int], list[int]]:
    """Main and extra deck (in file order), and side deck passcodes of a .ydk file."""
    main: list[int] = []
    side: list[int] = []
    target = main
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("!side"):
            target = side
        elif line.isdigit():
            target.append(int(line))
    return main, side
