"""Board breaking: a scripted bot builds a combo end board, our agent goes second and tries to break it.

The boards come from the combo book (combo_check.py): each checked line is replayed answer for answer
by the combo player (seat 0) in the same puzzle scenario it was found in, then the bot ends its turn.
Our agent (seat 1) gets a fresh hand from its own deck for every attempt and plays its turn with the
model. Meanwhile the bot defends the board by a fixed rule (--defender):

    pass    never responds (the board's monsters only count as bodies)
    eager   activates the first effect it is offered in every chain window, picks the first option
            of every follow-up question (a crude stand-in for a player using every negate)

After the agent's turn the bot's field is compared with the end board: cards left, monsters left,
life points, and whether the duel was won in that turn.

    python -m ocgforge.board_break --book runs/combo-book/elfnote-s1.jsonl --deck decks/elfnote.ydk \\
        --deck-type Elfnote --agent-deck decks/branded.ydk --checkpoint runs/x/latest --tries 8
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from . import combo_check, combos as cb, rollout, worlds

END_TURN = ("idle.7", "battle.3")  # to the End Phase from the Main Phase or the Battle Phase


def agent_lines(agent_deck: list[int], rng: random.Random, hand_size: int = 5) -> list[str]:
    """Puzzle-script lines for seat 1: a shuffled deck and a hand from `agent_deck`."""
    cards = list(agent_deck)
    rng.shuffle(cards)
    lines = [f"Debug.AddCard({code},1,1,{worlds.LOCATION_DECK},{i},8)" for i, code in enumerate(cards[hand_size:])]
    lines += [f"Debug.AddCard({code},1,1,{worlds.LOCATION_HAND},{i},10)" for i, code in enumerate(cards[:hand_size])]
    return lines


def with_agent(script: str, lines: list[str], extra: list[int]) -> str:
    """The combo scenario with seat 1's cards replaced by the agent's deck, hand and Extra Deck."""
    def seat_one(line: str) -> bool:  # Debug.AddCard(code,owner,player,location,sequence,position)
        return line.startswith("Debug.AddCard(") and line[len("Debug.AddCard("):].split(",")[1] == "1"

    kept = [line for line in script.splitlines() if not seat_one(line)]
    end = kept.index("Debug.ReloadFieldEnd()")
    agent = lines + [f"Debug.AddCard({code},1,1,{worlds.LOCATION_EXTRA},0,8)" for code in extra]
    return "\n".join(kept[:end] + agent + kept[end:]) + "\n"


def field(duel, owner: int) -> dict:
    cards = [c for c in duel.snapshot()["cards"]
             if c["controller"] == owner and c["location"] in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)]
    return {"cards": len(cards), "monsters": sum(c["location"] == worlds.LOCATION_MZONE for c in cards),
            "codes": sorted(c["code"] for c in cards)}


def defend(keys: list[str], mode: str) -> str:
    if mode == "eager":
        chains = [k for k in keys if k.startswith("chain.") and k != "chain.pass"]
        if chains:
            return chains[0]
        if keys and keys[0].startswith("chain."):
            return "chain.pass"
        return keys[0]
    return worlds.decline(keys)


def own_answers(start, line: list[str]) -> list[str] | None:
    """The book line's answers of the combo player (seat 0) alone: replayed in the book's own
    scenario, where seat 1 only ever declined. With another hand seat 1 is asked other questions,
    so only seat 0's answers can be replayed."""
    duel = start()
    if duel is None:
        return None
    mine = []
    for key in line:
        if duel.finished or key not in duel.keys():
            return None
        if duel.player == 0:
            mine.append(key)
        duel.act(key)
    return mine


def build_board(start, answers: list[str], max_steps: int = 300):
    """Seat 0 plays its book answers while seat 1 declines everything, then seat 0 ends its turn;
    the duel at seat 1's first decision of turn 2, or None if the line no longer fits (seat 1's
    cards changed something, e.g. an effect that looks at its hand)."""
    duel = start()
    if duel is None:
        return None
    pending = list(answers)
    for _ in range(max_steps):
        if duel.finished:
            return None
        if duel.turn >= 2 and duel.player == 1:
            return duel if not pending else None
        keys = duel.keys()
        if duel.player == 1:
            duel.act(worlds.decline(keys))  # the agent sits out turn 1: the board is built uninterrupted
        elif pending:
            if pending[0] not in keys:
                return None
            duel.act(pending.pop(0))
        else:
            duel.act(next((k for k in keys if k.startswith(END_TURN)), worlds.decline(keys)))
    return None


def play_turn(duel, policy, card_index, defender: str, max_steps: int = 600) -> None:
    """The agent's turn 2 (with the bot answering by `defender`), until turn 3 or the end."""
    for _ in range(max_steps):
        if duel.finished or duel.turn >= 3:
            return
        keys = duel.keys()
        if duel.player == 1:
            choice = policy([duel.encode(card_index)])[0][0]
            duel.act(keys[choice])
        else:
            duel.act(defend(keys, defender))


def main(argv=None) -> None:
    from ._core import Duel
    from .cards import CardTable
    from .model import load_checkpoint

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--book", type=Path, required=True, help="combo book (combo_check --out)")
    p.add_argument("--deck", type=Path, required=True, help="the combo deck the book was made with")
    p.add_argument("--deck-type", required=True)
    p.add_argument("--book-opponent", type=Path, default=rollout.REPO / "decks" / "salamangreat.ydk",
                   help="combo_check --opponent the book was made with (to rebuild its scenarios)")
    p.add_argument("--agent-deck", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tries", type=int, default=8, help="agent hands per board")
    p.add_argument("--defender", choices=("pass", "eager"), default="eager")
    p.add_argument("--all", action="store_true", help="also boards of lines that were not fully checked")
    p.add_argument("--greedy", action="store_true", help="the agent plays its most likely answers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    table = CardTable.from_cdb()
    types = {card.code: card.type for card in table.cards}
    is_extra = lambda code: bool(types.get(code, 0) & combo_check.TYPE_EXTRA)  # noqa: E731
    names = cb.name_index(table)
    combos = {c.id: c for c in cb.load(args.deck_type, names)}
    decklist = worlds.read_decklist(args.deck)
    book_opponent = worlds.read_decklist(args.book_opponent)[0]
    agent_main, agent_extra = worlds.read_decklist(args.agent_deck)
    model = load_checkpoint(args.checkpoint, args.device, dropout=0.0)
    card_index = model.card_features.index_map()
    policy = rollout.ModelPolicy(model, args.device, sample=not args.greedy, seed=args.seed)
    cache = rollout.REPO / ".cache"

    def make_start(script: str, seed: int, seat_one_deck: Path):
        def start():
            try:
                duel = Duel.world(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(args.deck),
                                  str(seat_one_deck), script, seed, 0)
            except Exception:
                return None
            for _ in range(20):  # through the draw and standby phase, as combo_check does
                if duel.finished or duel.player == 0:
                    break
                duel.act(worlds.decline(duel.keys()))
            return None if duel.finished else duel
        return start

    entries = [json.loads(line) for line in args.book.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries = [e for e in entries if args.all or e.get("checked")]
    report = []
    for entry in entries:
        combo = combos.get(entry["combo"])
        if combo is None:
            print(f"{entry['title']}: combo not found, skipped", flush=True)
            continue
        # The scenario exactly as combo_check built it (shuffle 0), then the agent's cards for seat 1.
        rng = random.Random(f"{combo.id}-0")
        script = combo_check.scenario_script(combo, decklist, book_opponent, is_extra, rng)
        seed = rng.getrandbits(32)
        answers = own_answers(make_start(script, seed, args.book_opponent), entry["line"])
        if answers is None:
            print(f"{entry['title']}: the book line does not replay, skipped", flush=True)
            continue
        tries, unplayable = [], 0
        for t in range(args.tries):
            agent_script = with_agent(script, agent_lines(agent_main, random.Random(f"{args.seed}-{combo.id}-{t}")),
                                      agent_extra)

            duel = build_board(make_start(agent_script, seed, args.agent_deck), answers)
            if duel is None:
                unplayable += 1
                continue
            before = field(duel, 0)
            try:
                play_turn(duel, policy, card_index, args.defender)
            except Exception as error:  # an engine error in the agent's turn: count it, go on
                tries.append({"error": str(error)[:200]})
                continue
            after = field(duel, 0) if not duel.finished else {"cards": 0, "monsters": 0, "codes": []}
            won = duel.finished and duel.winner == 1
            lp = duel.snapshot()["life_points"][0] if not duel.finished else 0
            tries.append({"won": won, "before": before["cards"], "left": after["cards"],
                          "monsters_before": before["monsters"], "monsters_left": after["monsters"], "lp": lp})
        ok = [t for t in tries if "error" not in t]
        summary = {"title": entry["title"], "combo": entry["combo"], "tries": len(tries), "errors": len(tries) - len(ok),
                   "unplayable": unplayable}
        if ok:
            summary.update(board=ok[0]["before"], monsters=ok[0]["monsters_before"],
                           won=round(float(np.mean([t["won"] for t in ok])), 3),
                           left=round(float(np.mean([t["left"] for t in ok])), 2),
                           monsters_left=round(float(np.mean([t["monsters_left"] for t in ok])), 2),
                           lp=round(float(np.mean([t["lp"] for t in ok]))))
        print(json.dumps(summary), flush=True)
        report.append({**summary, "detail": tries})
    played = [r for r in report if r.get("tries", 0) > r.get("errors", 0)]
    if played:
        total = Counter()
        for r in played:
            total["boards"] += 1
            total["won"] += r["won"]
            total["cleared"] += r["left"] / max(1, r["board"])
        print(json.dumps({"boards": total["boards"], "won": round(total["won"] / total["boards"], 3),
                          "left_share": round(total["cleared"] / total["boards"], 3), "defender": args.defender}))
    if args.out:
        args.out.write_text("\n".join(json.dumps(r) for r in report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
