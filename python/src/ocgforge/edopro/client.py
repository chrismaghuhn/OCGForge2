"""Join an EDOPro room as a duelist and play one duel.

    python -m ocgforge.edopro.client --deck decks/elfnote.ydk --checkpoint runs/focus-large-s0/latest \\
        --record runs/edopro/duel.bin

The room is created in EDOPro (LAN/host, Master Rule 2020: the rules the model is trained on). The
bot sends its deck, gets ready, and once the host starts, plays rock paper scissors, goes first if it
may, and answers every question: with --checkpoint the model's most likely answer (--sample samples
instead), without it a uniformly random legal answer. It sees the duel as in training: ClientDuel
builds the same observation from what the server sends this player. --record writes every packet, both directions, for offline replay: per packet u8 direction
(0 = from the server, 1 = to it), u16 length, then the packet without its length prefix.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import struct
import sys
from collections import Counter
from pathlib import Path

from .. import rollout
from . import protocol as p


def main_deck(path: Path) -> list[int]:
    """The main deck passcodes of a .ydk file (without extra and side deck)."""
    codes, section = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(("#", "!")):
            section = line if line in ("#main", "#extra", "!side") else section
        elif line.isdigit() and section == "#main":
            codes.append(int(line))
    return codes


class Bot:
    def __init__(self, deck: Path, name: str, password: str, seed: int, record: Path | None,
                 checkpoint: Path | None = None, sample: bool = False, device: str = "cpu"):
        from .._core import ClientDuel

        self.ClientDuel = ClientDuel
        self.deck = p.read_ydk(deck)
        self.main_deck = main_deck(deck)
        self.flags = 0
        self.policy = self.index = None
        if checkpoint is not None:
            from ..model import load_checkpoint

            model = load_checkpoint(checkpoint, device, dropout=0.0)
            self.policy = rollout.ModelPolicy(model, device, sample=sample, seed=seed)
            self.index = model.card_features.index_map()
        self.name, self.password = name, password
        self.rng = random.Random(seed)
        self.record = open(record, "wb", buffering=0) if record else None  # complete even after a crash
        self.duel = None
        self.writer: asyncio.StreamWriter | None = None
        self.stats = Counter()
        self.done = False

    def log(self, direction: int, kind: int, payload: bytes) -> None:
        if self.record:
            self.record.write(struct.pack("<BHB", direction, len(payload) + 1, kind) + payload)

    def send(self, data: bytes) -> None:
        self.log(1, data[2], data[3:])
        self.writer.write(data)

    async def run(self, host: str, port: int) -> None:
        reader, self.writer = await asyncio.open_connection(host, port)
        self.send(p.player_info(self.name))
        self.send(p.join_game(self.password))
        buffer = bytearray()
        while not self.done:
            chunk = await reader.read(65536)
            if not chunk:
                print("server closed the connection", flush=True)
                break
            buffer += chunk
            for kind, payload in p.read_packets(buffer):
                self.log(0, kind, payload)
                self.on_packet(kind, payload)
            await self.writer.drain()
        self.writer.close()
        if self.record:
            self.record.close()

    def on_packet(self, kind: int, payload: bytes) -> None:
        self.stats[p.STOC_NAMES.get(kind, hex(kind))] += 1
        if kind == p.STOC_JOIN_GAME:
            self.flags = p.duel_flags(payload)
            rules = "Master Rule 2020" if self.flags == p.DUEL_MODE_MR5 else f"duel flags {self.flags:#x}, not Master Rule 2020"
            print(f"joined the room ({rules})", flush=True)
            self.send(p.update_deck(*self.deck))
            self.send(p.packet(p.CTOS_HS_READY))
        elif kind == p.STOC_TYPE_CHANGE:
            seat = payload[0] & 0xF
            if seat == 7:  # NETPLAYER_TYPE_OBSERVER: the duelist seats are taken or a duel is running
                print("joined as a spectator: no free duelist seat (or a duel is running); leaving", flush=True)
                self.send(p.packet(p.CTOS_LEAVE_GAME))
                self.done = True
                return
            print(f"seat {seat}{' (host)' if payload[0] & 0x10 else ''}", flush=True)
        elif kind == p.STOC_ERROR_MSG:
            code = struct.unpack_from("<I", payload, 4)[0] if len(payload) >= 8 else 0  # u8 type, 3 padding, u32
            card = f", card {struct.unpack_from('<I', payload, 20)[0]}" if len(payload) >= 24 else ""  # deck error
            print(f"server error: {p.ERRORS.get(payload[0], payload[0])} {code:#x}{card}", flush=True)
            self.done = True
        elif kind == p.STOC_SELECT_HAND:
            self.send(p.hand_result(self.rng.randint(1, 3)))
        elif kind == p.STOC_SELECT_TP:
            self.send(p.tp_result(go_first=True))
        elif kind == p.STOC_TIME_LIMIT:
            self.send(p.packet(p.CTOS_TIME_CONFIRM))
        elif kind == p.STOC_GAME_MSG:
            self.on_game_message(payload)
        elif kind in (p.STOC_DUEL_END, p.STOC_LEAVE_GAME):
            self.done = True

    def on_game_message(self, message: bytes) -> None:
        if not message:
            return
        if message[0] == p.MSG_START:
            seat = message[1] & 0xF
            card_data = str(rollout.REPO / ".cache" / "card_data.tsv") if self.policy else ""
            self.duel = self.ClientDuel(seat, self.flags & 0xFFFFFFFF, card_data, self.main_deck)
            print(f"duel started, we are player {seat}", flush=True)
        if self.duel is None:
            return
        if message[0] == p.MSG_RETRY:
            self.stats["retries"] += 1
            print("server rejected our last response (MSG_RETRY)", flush=True)
        try:
            asked = self.duel.feed(message)
        except Exception as error:  # a question we cannot answer
            self.stats["unanswerable"] += 1
            print(f"cannot answer message {message[0]}: {error}", flush=True)
            self.send(p.packet(p.CTOS_LEAVE_GAME))
            self.done = True
            return
        if self.duel.finished:
            print(f"duel over, winner {self.duel.winner}", flush=True)
        if not asked:
            return
        while True:
            keys = self.duel.request()["candidates"]
            answer = self.duel.act(keys[self.choose(len(keys))])
            if answer is not None:
                self.stats["responses"] += 1
                self.send(p.response(answer))
                return


    def choose(self, count: int) -> int:
        """Index of the answer: the model's, or random without a model or when it fails."""
        if self.policy is not None:
            try:
                choices, _, _ = self.policy([self.duel.encode(self.index)])
                self.stats["model answers"] += 1
                return choices[0]
            except Exception as error:  # e.g. a question the encoder cannot represent
                self.stats["random answers"] += 1
                print(f"model could not answer ({error}); answering randomly", flush=True)
        return self.rng.randrange(count)


def main(argv=None) -> None:
    import faulthandler

    faulthandler.enable()  # a crash in the C++ part still leaves a trace
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7911)
    parser.add_argument("--password", default="")
    parser.add_argument("--name", default="OCGForge")
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="model checkpoint directory, e.g. runs/x/latest")
    parser.add_argument("--sample", action="store_true", help="sample answers instead of the most likely")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
    bot = Bot(args.deck, args.name, args.password, args.seed, args.record, args.checkpoint, args.sample, args.device)
    try:
        asyncio.run(bot.run(args.host, args.port))
    finally:
        if bot.duel is not None:
            bot.stats["blocked looping answers"] = bot.duel.blocked_answers
        print(dict(bot.stats), flush=True)
        if bot.duel is not None and bot.duel.skipped:
            reasons = Counter(s.split(":", 1)[0] + ":" + s.split(":", 2)[1] for s in bot.duel.skipped)
            print("skipped messages:", dict(reasons.most_common(15)), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
