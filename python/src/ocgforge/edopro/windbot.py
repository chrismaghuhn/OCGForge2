"""Benchmark a checkpoint against WindBot, EDOPro's rule-based AI, on a local Multirole server.

    python -m ocgforge.edopro.windbot --checkpoint runs/focus-large-s0/latest --duels 20

For each of our decks and each WindBot deck, our bot creates a room on the server (--port, Multirole's
room hosting port), WindBot joins it, and both play --duels duels through rematches. Prints and saves
(--out) the win rate per pairing and overall. Needs Multirole running (see docs/edopro-bridge-plan.md)
and mono for WindBot. WindBot plays its own decks; by default those listed for Master Rule 5.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from .. import rollout

WINDBOT = Path.home() / "Apps" / "ProjectIgnis" / "WindBot"
WINDBOT_VERSION = 41 | 0 << 8 | 11 << 16  # EDOPro 41.0, ocgcore API 11.0


def windbot_decks(directory: Path) -> list[str]:
    bots = json.loads((directory / "bots.json").read_text(encoding="utf-8"))
    return sorted({bot["deck"] for bot in bots if 5 in bot.get("masterRules", [])})


async def pairing(args, deck: Path, windbot_deck: str, room: int) -> dict:
    """`args.duels` duels of our bot with `deck` against WindBot with `windbot_deck`."""
    password = f"bench{os.getpid() % 10000}{room}"
    env = {**os.environ, "OMP_NUM_THREADS": str(args.torch_threads)}  # many bots share the CPU
    client = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "ocgforge.edopro.client", "--port", str(args.port), "--create", "--password", password,
        "--deck", str(deck), "--checkpoint", str(args.checkpoint), "--duels", str(args.duels), "--seed", str(room),
        "--name", "OCGForge", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env)
    windbot = None
    result = {"deck": deck.stem, "windbot": windbot_deck, "wins": 0, "losses": 0, "draws": 0, "errors": []}
    seat = None
    deadline = time.time() + args.minutes_per_duel * 60 * args.duels
    try:
        while time.time() < deadline:
            try:
                line = await asyncio.wait_for(client.stdout.readline(), timeout=max(1.0, deadline - time.time()))
            except asyncio.TimeoutError:
                result["errors"].append("timeout")
                break
            if not line:
                break
            text = line.decode(errors="replace").strip()
            if match := re.match(r"room (\d+) created", text):
                windbot = await asyncio.create_subprocess_exec(
                    "mono", "WindBot.exe", "Name=WindBot", f"Deck={windbot_deck}", "Host=127.0.0.1",
                    f"Port={args.port}", f"HostInfo={password}", f"RoomId={match.group(1)}",
                    f"Version={WINDBOT_VERSION}", "Chat=false", f"DbPath={args.windbot_db}",
                    cwd=args.windbot, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            elif match := re.match(r"duel started, we are player (\d)", text):
                seat = int(match.group(1))
            elif match := re.match(r"duel over, winner (\d+)", text):
                winner = int(match.group(1))
                key = "draws" if winner > 1 else "wins" if winner == seat else "losses"
                result[key] += 1
                print(f"  {deck.stem} vs {windbot_deck}: {key[:-1] if key != 'losses' else 'loss'}", flush=True)
            elif text.startswith(("server error", "cannot answer", "model could not answer")):
                result["errors"].append(text)
    finally:
        for process in (client, windbot):
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
    return result


async def run(args) -> list[dict]:
    """Every pairing, --parallel at a time (each in its own room)."""
    slots = asyncio.Semaphore(args.parallel)
    pairs = [(deck, windbot_deck) for deck in args.decks for windbot_deck in args.windbot_decks]

    async def one(room: int, deck: Path, windbot_deck: str) -> dict:
        async with slots:
            r = await pairing(args, deck, windbot_deck, room)
        games = r["wins"] + r["losses"] + r["draws"]
        print(f"{deck.stem} vs {windbot_deck}: {r['wins']}/{games} won"
              f"{', errors: ' + str(len(r['errors'])) if r['errors'] else ''}", flush=True)
        return r

    return list(await asyncio.gather(*(one(room, deck, wb) for room, (deck, wb) in enumerate(pairs, 1))))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decks", type=Path, nargs="+", default=[rollout.REPO / "decks" / "elfnote.ydk"])
    parser.add_argument("--windbot-decks", nargs="+", help="WindBot deck names (default: its Master Rule 5 decks)")
    parser.add_argument("--duels", type=int, default=10, help="duels per pairing")
    parser.add_argument("--port", type=int, default=7920, help="Multirole's room hosting port")
    parser.add_argument("--windbot", type=Path, default=WINDBOT)
    parser.add_argument("--windbot-db", type=Path, default=WINDBOT.parent / "expansions" / "cards.cdb")
    parser.add_argument("--minutes-per-duel", type=float, default=10.0)
    parser.add_argument("--parallel", type=int, default=6, help="pairings played at the same time")
    parser.add_argument("--torch-threads", type=int, default=2, help="CPU threads for each bot's model")
    parser.add_argument("--out", type=Path, help="JSON file for the results")
    args = parser.parse_args(argv)
    args.windbot_decks = args.windbot_decks or windbot_decks(args.windbot)
    results = asyncio.run(run(args))
    wins = sum(r["wins"] for r in results)
    games = sum(r["wins"] + r["losses"] + r["draws"] for r in results)
    print(f"total: {wins}/{games} won ({wins / max(games, 1):.1%})", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "results": results, "wins": wins,
                                        "games": games}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
