"""Opening book: the best first turn of one deck from many starting hands, with hand-trap branches.

    python -m ocgforge.opening_book --deck decks/branded.ydk --deck-type Branded \\
        --sheet .cache/branded-discord/sheet1.csv --checkpoint runs/x/latest --out runs/combo-book/branded-opening.jsonl

Seeds (what a hand should reach) come from `seeds` (community combos, player sheets, Duelingbook
logs, starter x extender hands). Each seed becomes several starting hands: its cards plus
fillers, one random source per hand.
The search (`combo_check.check`, beam search on duel clones) plays each hand; the seed's steps and
end board only guide it, it may find another line.

Interruptions: for each hand trap the deck fears (counters data: card, impact, choke points; plus
`--chokepoints` rules from guides), the hand gets a second search in which the opponent holds that
hand trap and fires it the first time we activate one of its choke-point cards (or at the first
chance, without choke points). The book keeps the best continuation after it.

The book is keyed by game state, not by starting hand: our hand, field, GY and banished cards
(cards no book line ever names count only as a number), the opponent's public cards, the phase
and the question. Transpositions share an entry. Each entry holds the answer of the best line
through it; a line's value is how often our agent fails to break its end board (board_break: the
agent goes second against the board, the bot defends it eagerly), then its size.

Report (`--test-hands`): random hands from the deck played from the book; the share that stays in
the book for 5, 10 and 20 of our answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from . import board_break as bb
from . import combo_check as cc
from . import combos as cb
from . import rollout, worlds
from . import seeds as sd
from .seeds import HAND_SIZE, Seed

def hand_trap_bot(trap: int, chokepoints: set[int]):
    """Seat 1 with `trap` in hand: activates it the first time it is offered right after we
    activated a choke-point card (any card without choke points); for its follow-up questions it
    picks a choke-point card if offered, else the first answer; declines everything else."""
    def act(duel, history: list[str]) -> str:
        keys = duel.keys()
        offers = [k for k in keys if (ref := worlds.card_ref(k)) is not None and int(ref[1]) == trap
                  and k.startswith(("chain.", "idle."))]
        if offers:
            ours = next((worlds.card_ref(k) for k in reversed(history)
                         if k.startswith(("chain.", "idle.")) and worlds.card_ref(k) and worlds.card_ref(k)[2] == "0"),
                        None)
            if not chokepoints or (ours is not None and int(ours[1]) in chokepoints):
                return offers[0]
            return worlds.decline(keys)
        if fired(history, trap) and not ("chain.pass" in keys and all(k.startswith("chain.") for k in keys)):
            aimed = [k for k in keys if (ref := worlds.card_ref(k)) is not None and int(ref[1]) in chokepoints]
            return (aimed or keys)[0]
        return worlds.decline(keys)
    return act


def fired(history: list[str], trap: int) -> bool:
    return any((ref := worlds.card_ref(k)) is not None and int(ref[1]) == trap and k.startswith(("chain.", "idle."))
               and ref[2] == "1" for k in history)


def state_key(duel, relevant: set[int]) -> str:
    """The game state as the book sees it: our hand, field (with zone and position), GY and
    banished cards, the opponent's public cards, phase and turn, and the question (its kinds and
    the relevant cards it names). Cards no book line names count only as a number, so a filler card
    does not split states."""
    parts = []
    for c in duel.snapshot()["cards"]:
        loc = c["location"]
        if c["controller"] == 0 and loc in (worlds.LOCATION_HAND, worlds.LOCATION_GRAVE, worlds.LOCATION_REMOVED):
            parts.append((0, loc, c["code"] if c["code"] in relevant else 0, 0))
        elif c["controller"] == 0 and loc in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE):
            parts.append((0, loc, c["code"], c["sequence"] * 16 + c["position"]))
        elif c["controller"] == 1 and loc in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE, worlds.LOCATION_GRAVE,
                                              worlds.LOCATION_REMOVED) and c["public"]:
            parts.append((1, loc, c["code"], 0))
    question = sorted({(k.split(".")[0], worlds.card_ref(k)[1] if worlds.card_ref(k) and
                        int(worlds.card_ref(k)[1]) in relevant else "") for k in duel.keys()})
    raw = json.dumps([sorted(parts), duel.phase, duel.turn, duel.player, question])
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def walk(start, keys: list[str], relevant: set[int]) -> list[tuple[str, str, int]]:
    """(state, answer, depth) for each of our answers along a line replayed from its scenario."""
    duel = start()
    out, depth = [], 0
    for key in keys:
        if duel is None or duel.finished or key not in duel.keys():
            break
        if duel.player == 0:
            out.append((state_key(duel, relevant), worlds.canonical(key), depth))
            depth += 1
        duel.act(key)
    return out


def main(argv=None) -> None:
    from ._core import Duel
    from .cards import CardTable
    from .model import load_checkpoint

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--deck", type=Path, required=True)
    p.add_argument("--deck-type", required=True)
    p.add_argument("--sheet", type=Path, help="player sheet CSV (starting hand, end board, replay, notes)")
    p.add_argument("--replays", type=Path, nargs="*", default=[], help="Duelingbook logs (card movements)")
    p.add_argument("--lists", nargs="*", default=[],
                   help="other player sheets (Card/Board CSV, In Hand TSV), each as path or path::author")
    p.add_argument("--chokepoints", type=Path, help="JSON {hand trap: [card names]} from guides")
    p.add_argument("--priority", nargs="*", default=[], help="seed titles (substrings) to search first")
    p.add_argument("--opponent", type=Path, default=rollout.REPO / "decks" / "salamangreat.ydk")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--hands", type=int, default=3, help="starting hands per seed (other fillers)")
    p.add_argument("--max-seeds", type=int, default=0, help="seeds to search at most (0: all)")
    p.add_argument("--grid", type=int, default=40, help="starter x extender hands at most")
    p.add_argument("--sources", default="replay,sheet,community,grid",
                   help="seed sources to use, in search order (replay, sheet, community, grid)")
    p.add_argument("--max-searches", type=int, default=0, help="searches at most, in seed order (0: all)")
    p.add_argument("--traps", type=int, default=3, help="interruption branches per hand (most feared hand traps)")
    p.add_argument("--first-only", action="store_true", help="only going-first seeds")
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--rate-tries", type=int, default=4, help="board_break tries per end board (0: field size only)")
    p.add_argument("--agent-deck", type=Path, default=rollout.REPO / ".cache" / "yugiohmeta" / "decks" / "snake_eye.ydk")
    p.add_argument("--test-hands", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="book (JSON lines: one state per line)")
    p.add_argument("--dry-run", action="store_true", help="list the seeds and searches, search nothing")
    args = p.parse_args(argv)
    torch.set_num_threads(args.torch_threads)

    table = CardTable.from_cdb()
    card_name = {c.code: c.name for c in table.cards}
    types = {c.code: c.type for c in table.cards}
    is_extra = lambda code: bool(types.get(code, 0) & cc.TYPE_EXTRA)  # noqa: E731
    center = {c.code for c in table.cards if cc.wants_center(c.text)}
    names = cb.name_index(table)
    decklist = worlds.read_decklist(args.deck)
    opponent = worlds.read_decklist(args.opponent)[0]
    pool = set(decklist[0]) | set(decklist[1])
    for combo in cb.load(args.deck_type, names):
        pool |= set(combo.cards()) | set(combo.end_board)
    resolve = sd.resolver(names, pool, card_name)
    model = load_checkpoint(args.checkpoint, args.device, dropout=0.0)
    card_index = model.card_features.index_map()
    cache = rollout.REPO / ".cache"

    seeds, rules, feared = sd.load(args, args.deck_type, names, resolve, decklist, card_name)
    print(f"{len(seeds)} seeds ({Counter(s.source for s in seeds)}), hand traps: "
          f"{[card_name.get(t) for t in feared[:8]]}", flush=True)

    def scenario(seed: Seed, variant: int, trap: int | None):
        fill = random.Random(f"{seed.id}-hand-{variant}")
        rng = random.Random(f"{seed.id}-{variant}")
        hand = list(seed.hand)
        rest = [c for c in decklist[0] if c not in hand]
        fill.shuffle(rest)
        hand += rest[:max(0, HAND_SIZE - len(hand))]
        script = cc.scenario_script(seed.combo, decklist, opponent, is_extra, rng, hand=hand,
                                    opponent_hand=[trap] if trap else [])
        seed_value = rng.getrandbits(32)

        def start():
            try:
                duel = Duel.world(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(args.deck),
                                  str(args.opponent), script, seed_value, 0)
                for _ in range(20):
                    if duel.finished or duel.player == 0:
                        break
                    duel.act(worlds.decline(duel.keys()))
            except Exception:
                return None
            return None if duel.finished else duel
        return script, seed_value, start

    def search(seed: Seed, variant: int, trap: int | None):
        began = time.perf_counter()
        script, seed_value, start = scenario(seed, variant, trap)
        bot = hand_trap_bot(trap, rules.get(trap, set()) | seed.chokes.get(trap, set())) if trap else None
        line, targets = cc.check(seed.combo, start, model, card_index, args.device, beam=args.beam, center=center,
                                 is_extra=is_extra, opponent=bot)
        duel = start()
        for key in line.keys if duel is not None else []:
            if duel.finished or key not in duel.keys():
                break
            duel.act(key)
        final = cc.end_board(duel) if duel is not None else []
        on_field = [c for c in final if c["location"] in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)]
        return {"seed": seed.id, "title": seed.title, "source": seed.source, "author": seed.author,
                "origin": seed.origin, "variant": variant,
                "trap": trap, "fired": fired(line.keys, trap) if trap else None, "line": line.keys,
                "steps": len(targets), "progress": line.progress, "skipped": line.skipped,
                "key_cards": line.key_cards, "book_end_board": seed.combo.end_board, "field": len(on_field),
                "end_board": final, "script": script, "duel_seed": seed_value, "first": seed.first,
                "seconds": round(time.perf_counter() - began, 1)}, start

    # Base lines first; each hand's interruption branches use the hand traps its seed names first,
    # then the most feared ones.
    jobs = []
    for seed in seeds:
        # its own hand traps first (a log shows where they hit), then the most feared ones
        traps = list(dict.fromkeys([t for t in seed.vs if t in rules or card_name.get(t) in sd.INTERRUPTERS]
                                   + feared))[:args.traps]
        for variant in range(args.hands):
            jobs.append((seed, variant, None))
            jobs += [(seed, variant, t) for t in traps]
    if args.max_searches:
        jobs = jobs[:args.max_searches]
    print(f"{len(jobs)} searches", flush=True)
    if args.dry_run:
        for seed in seeds:
            print(f"  {seed.source:9s} {seed.author[:14]:14s} first={int(seed.first)} hand={[card_name.get(c) for c in seed.hand]} "
                  f"steps={len(cc.step_targets(seed.combo))} board={[card_name.get(c) for c in seed.combo.end_board]} "
                  f"vs={[card_name.get(c) for c in seed.vs]}  {seed.title[:60]}")
        return
    pool_ex = ThreadPoolExecutor(max(1, args.workers))
    futures = [pool_ex.submit(search, *job) for job in jobs]
    lines = []
    for n, future in enumerate(futures):
        try:
            result, start = future.result()
        except Exception as error:  # one broken scenario must not stop the book
            print(f"{n + 1}/{len(futures)} error {str(error)[:120]}", flush=True)
            continue
        lines.append((result, start))
        print(f"{n + 1}/{len(futures)} {result['source']:9s} field {result['field']} "
              f"trap {card_name.get(result['trap'], '-')[:20]:20s} fired {result['fired']} {result['seconds']:.0f}s  "
              f"{result['title'][:50]}", flush=True)
    pool_ex.shutdown()

    # End boards rated by board_break: the agent goes second against each.
    agent_main, agent_extra = worlds.read_decklist(args.agent_deck)
    policy = rollout.ModelPolicy(model, args.device, sample=True, seed=args.seed)
    for result, start in lines:
        # A line with a hand trap fired cannot be rebuilt for board_break (the agent's seat holds other
        # cards): it ranks below the base lines on the states they share, by its field size.
        result["value"] = (rate(result, args, agent_main, agent_extra, policy, card_index, cache, Duel)
                           if result["trap"] is None else {"held": -1.0, "left": None, "tries": 0})
    relevant = set()
    for result, _ in lines:
        for key in result["line"]:
            if (ref := worlds.card_ref(key)) is not None and ref[2] == "0":
                relevant.add(int(ref[1]))
        relevant |= {c["code"] for c in result["end_board"]}
    book: dict[str, dict] = {}
    for result, start in lines:
        score = (result["value"]["held"], result["field"])
        for state, answer, depth in walk(start, result["line"], relevant):
            entry = book.get(state)
            if entry is None or tuple(entry["score"]) < score:
                book[state] = {"state": state, "answer": answer, "score": list(score), "depth": depth,
                               "seed": result["seed"], "trap": result["trap"], "hits": (entry or {}).get("hits", 0)}
            book[state]["hits"] += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(e) + "\n" for e in book.values()), encoding="utf-8")
    lines_out = args.out.with_name(args.out.stem + "-lines.jsonl")
    lines_out.write_text("".join(json.dumps(r) + "\n" for r, _ in lines), encoding="utf-8")
    meta = {"relevant": sorted(relevant)}
    args.out.with_name(args.out.stem + "-meta.json").write_text(json.dumps(meta), encoding="utf-8")
    report(lines, book, relevant, args, decklist, opponent, cache, Duel, card_name)


def rate(result: dict, args, agent_main, agent_extra, policy, card_index, cache, Duel) -> dict:
    """How often the agent fails to win against this end board (held), and the share of the board
    left after its turn (board_break, eager defender)."""
    if args.rate_tries <= 0:
        return {"held": result["field"] / 10, "left": None, "tries": 0}
    mine = list(result["line"])
    answers = None
    held, left, tries = 0, [], 0
    for t in range(args.rate_tries):
        script = bb.with_agent(result["script"], bb.agent_lines(agent_main, random.Random(f"{result['seed']}-{t}")),
                               agent_extra)

        def start(script=script):
            try:
                duel = Duel.world(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(args.deck),
                                  str(args.agent_deck), script, result["duel_seed"], 0)
                for _ in range(20):
                    if duel.finished or duel.player == 0:
                        break
                    duel.act(worlds.decline(duel.keys()))
            except Exception:
                return None
            return None if duel.finished else duel
        if answers is None:  # seat 0's answers of the line, as in its own scenario
            answers = own(result, args, cache, Duel, mine)
            if not answers:
                return {"held": 0.0, "left": None, "tries": 0}
        duel = bb.build_board(start, answers)
        if duel is None:
            continue
        before = bb.field(duel, 0)["cards"]
        try:
            bb.play_turn(duel, policy, card_index, "eager")
        except Exception:
            continue
        tries += 1
        won = duel.finished and duel.winner == 1
        held += not won
        left.append((bb.field(duel, 0)["cards"] if not duel.finished else 0) / max(1, before))
    return {"held": round(held / tries, 3) if tries else 0.0,
            "left": round(sum(left) / len(left), 3) if left else None, "tries": tries}


def own(result: dict, args, cache, Duel, keys: list[str]) -> list[str] | None:
    duel = Duel.world(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(args.deck), str(args.opponent),
                      result["script"], result["duel_seed"], 0)
    for _ in range(20):
        if duel.finished or duel.player == 0:
            break
        duel.act(worlds.decline(duel.keys()))
    mine = []
    for key in keys:
        if duel.finished or key not in duel.keys():
            return mine or None
        if duel.player == 0:
            mine.append(key)
        duel.act(key)
    return mine


def report(lines, book, relevant, args, decklist, opponent, cache, Duel, card_name) -> None:
    """Book size, branches per hand trap, end-board values, and how long random test hands stay in
    the book (our answers played from it before the first state it does not know)."""
    traps = Counter(card_name.get(r["trap"], str(r["trap"])) for r, _ in lines if r["trap"] and r["fired"])
    tried = Counter(card_name.get(r["trap"], str(r["trap"])) for r, _ in lines if r["trap"])
    values = [r["value"]["held"] for r, _ in lines if r["trap"] is None]
    rng = random.Random(args.seed)
    stays = []
    for t in range(args.test_hands):
        main = list(decklist[0])
        rng.shuffle(main)
        hand = main[:HAND_SIZE]
        combo = cb.Combo(f"test-{t}", "test", args.deck_type, 0, [], [])
        script = cc.scenario_script(combo, decklist, opponent, lambda c: False, random.Random(f"test-{t}"), hand=hand)
        try:
            duel = Duel.world(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(args.deck),
                              str(args.opponent), script, t, 0)
        except Exception:
            continue
        depth = 0
        for _ in range(400):
            if duel.finished or duel.turn > 1:
                break
            keys = duel.keys()
            if duel.player != 0:
                duel.act(worlds.decline(keys))
                continue
            entry = book.get(state_key(duel, relevant))
            answer = worlds.match(entry["answer"], keys) if entry else None
            if answer is None:
                break
            duel.act(answer)
            depth += 1
        stays.append(depth)
    n = max(1, len(stays))
    summary = {"states": len(book), "lines": len(lines), "base_lines": len(values),
               "mean_field": round(sum(r["field"] for r, _ in lines if r["trap"] is None) / max(1, len(values)), 2),
               "held_mean": round(sum(values) / max(1, len(values)), 3),
               "test_hands": len(stays), **{f"stay_{k}": round(sum(d >= k for d in stays) / n, 3) for k in (1, 5, 10, 20)},
               "branches_fired": dict(traps), "branches_tried": dict(tried)}
    print(json.dumps(summary), flush=True)
    args.out.with_name(args.out.stem + "-report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
