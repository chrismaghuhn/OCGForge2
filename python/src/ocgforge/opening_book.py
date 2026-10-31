"""Opening book: the best first turn of one deck from many starting hands, with hand-trap branches.

    python -m ocgforge.opening_book --deck decks/branded.ydk --deck-type Branded \\
        --sheet .cache/branded-discord/sheet1.csv --checkpoint runs/x/latest --out runs/combo-book/branded-opening.jsonl

Seeds (what a hand should reach) come from `seeds` (community combos, player sheets, Duelingbook
logs, guides, starter x extender hands); only going-first seeds, as every scenario is built with us
going first. Each seed becomes several starting hands: its cards plus fillers, one random source
per hand. The search (`combo_check.check`, beam search on duel clones) plays each hand; the seed's
steps and end board only guide it, it may find another line.

Interruptions: for each hand trap the deck fears (counters data: card, impact, choke points; plus
`--chokepoints` rules from guides; a seed's own logged hand traps first), the hand gets a second
search in which the opponent holds that hand trap and fires it the first time we activate one of
its choke-point cards (or at the first chance, without choke points).

The book is keyed by game state, not by starting hand: our hand, field, GY and banished cards
(cards that are only ever fillers count as a number), the opponent's visible cards, the phase and
the question. Transpositions share an entry, which holds the answer of the best line through it.
A line's value is how often our agent fails to win against its end board (board_break: the agent
goes second against the board, the bot defends it eagerly), then its size; lines that could not be
rated rank below rated ones. A loose key (no hand) is the fallback for hands the book never saw.

Lines are written as they are found (`<out>-lines.jsonl`), so a crash in rating or book building
loses no search; `--from-lines` rebuilds book and report from them.

Report: the book's own lines, the seeds' cards with other fillers, and honest random hands; the
share that stays in the book for 1, 5, 10 and 20 real choices (exact key, and with the fallback).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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


def state_key(duel, relevant: set[int], hand: bool = True) -> str:
    """The game state as the book sees it: our hand, field (with zone and position), GY and
    banished cards, the opponent's visible cards (face up, revealed, in the GY: `worlds.visible`),
    phase and turn, and the question (its kinds and the relevant cards it names). Cards outside
    `relevant` count only as a number, so a filler card does not split states. Without `hand` (the
    loose key) the hand and the cards the question names are left out: a fallback for hands the
    book never saw, checked by the answer itself."""
    parts = []
    for c in duel.snapshot()["cards"]:
        loc = c["location"]
        if c["controller"] == 0 and loc == worlds.LOCATION_HAND and not hand:
            continue
        if c["controller"] == 0 and loc in (worlds.LOCATION_HAND, worlds.LOCATION_GRAVE, worlds.LOCATION_REMOVED):
            parts.append((0, loc, c["code"] if c["code"] in relevant else 0, 0))
        elif c["controller"] == 0 and loc in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE):
            parts.append((0, loc, c["code"], c["sequence"] * 16 + c["position"]))
        elif c["controller"] == 1 and loc in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE, worlds.LOCATION_GRAVE,
                                              worlds.LOCATION_REMOVED):
            parts.append((1, loc, c["code"] if worlds.visible(c, 0) else 0, 0))
    question = sorted({(k.split(".")[0], worlds.card_ref(k)[1] if hand and worlds.card_ref(k) and
                        int(worlds.card_ref(k)[1]) in relevant else "") for k in duel.keys()})
    raw = json.dumps([sorted(parts), duel.phase, duel.turn, duel.player, question])
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def book_cards(results: list[dict]) -> set[int]:
    """The cards the state key tells apart: the seeds' starting cards, the field end-board cards,
    and every card of ours a book line names where it was not one of that line's fillers. A card
    that is a filler in every line naming it (a hand with other fillers must reach the same states;
    a filler used as a cost then leaves the book there) is not told apart."""
    cards = set()
    for result in results:
        fillers = set(result.get("fillers", []))
        for key in result["line"]:
            if (ref := worlds.card_ref(key)) is not None and ref[2] == "0" and int(ref[1]) not in fillers:
                cards.add(int(ref[1]))
        cards |= set(result.get("hand", [])) | {c["code"] for c in result["end_board"]
                                                if c["location"] in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)}
    return cards


def walk(start, keys: list[str], relevant: set[int]) -> list[tuple[str, str, int, str]]:
    """(state, answer, depth, loose state) for each of our answers along a line replayed from its
    scenario. `stay` computes the same keys at the same decisions."""
    duel = start()
    out, depth = [], 0
    for key in keys:
        if duel is None or duel.finished or key not in duel.keys():
            break
        if duel.player == 0:
            out.append((state_key(duel, relevant), worlds.canonical(key), depth, state_key(duel, relevant, False)))
            depth += 1
        duel.act(key)
    return out


def score(result: dict) -> tuple:
    """A line's value in the book: hold rate (unrated lines below every rated one), then field."""
    held = result.get("value", {}).get("held")
    return (held if held is not None else -1.0, result["field"])


def build_book(lines: list[tuple[dict, object]], relevant: set[int]) -> tuple[dict, dict]:
    """The exact book (state -> best answer) and the loose one (loose state -> answers, best first)."""
    book: dict[str, dict] = {}
    loose: dict[str, list] = {}
    for result, start in lines:
        value = score(result)
        for state, answer, depth, wide in walk(start, result["line"], relevant):
            entry = book.get(state)
            if entry is None or tuple(entry["score"]) < value:
                book[state] = {"state": state, "loose": wide, "answer": answer, "score": list(value), "depth": depth,
                               "seed": result["seed"], "trap": result["trap"], "hits": (entry or {}).get("hits", 0)}
            book[state]["hits"] += 1
            loose.setdefault(wide, []).append((value, answer))
    for answers in loose.values():
        answers.sort(key=lambda a: a[0], reverse=True)
    return book, loose


def stay(start, book: dict, relevant: set[int], loose: dict | None = None, max_answers: int = 400) -> int:
    """How many of our answers a duel plays from the book before the first state it does not know
    (with `loose`: or whose loose state offers none of the book's answers there). Only real choices
    count: questions with one answer and passed chain windows do not."""
    duel = start()
    depth = 0
    for _ in range(max_answers):
        if duel is None or duel.finished or duel.turn > 1:
            break
        keys = duel.keys()
        if duel.player != 0:
            duel.act(worlds.decline(keys))
            continue
        entry = book.get(state_key(duel, relevant))
        answer = worlds.match(entry["answer"], keys) if entry else None
        if answer is None and loose is not None:
            answer = next((m for _, a in loose.get(state_key(duel, relevant, False), [])
                           if (m := worlds.match(a, keys)) is not None), None)
        if answer is None:
            break
        duel.act(answer)
        depth += len(keys) > 1 and answer != "chain.pass"  # real choices only: no forced answers, no passes
    return depth


def rate(result: dict, args, agent, policy, card_index) -> dict:
    """board_break against the line's end board (eager defender): `held` is the share of played
    tries the agent did not win, `left` the share of the board left. Tries whose board cannot be
    built (the agent's cards change the line) or whose agent turn fails count apart, never as a
    broken board; with no played try `held` is None."""
    out = {"held": None, "left": None, "tries": 0, "unbuildable": 0, "errors": 0}
    if args.rate_tries <= 0:
        return out
    try:
        answers = bb.own_answers(cc.puzzle_start(args.deck, args.opponent, result["script"], result["duel_seed"]),
                                 result["line"])
    except Exception:
        answers = None
    if answers is None:
        out["unbuildable"] = args.rate_tries
        return out
    held, left = 0, []
    for t in range(args.rate_tries):
        script = bb.with_agent(result["script"], bb.agent_lines(agent[0], random.Random(f"{result['seed']}-{t}")),
                               agent[1])
        try:
            duel = bb.build_board(cc.puzzle_start(args.deck, args.agent_deck, script, result["duel_seed"]), answers)
        except Exception:
            duel = None
        if duel is None:
            out["unbuildable"] += 1
            continue
        before = bb.field(duel, 0)["cards"]
        try:
            bb.play_turn(duel, policy, card_index, "eager")
        except Exception:
            out["errors"] += 1
            continue
        out["tries"] += 1
        held += not (duel.finished and duel.winner == 1)
        left.append((bb.field(duel, 0)["cards"] if not duel.finished else 0) / max(1, before))
    if out["tries"]:
        out["held"], out["left"] = round(held / out["tries"], 3), round(sum(left) / len(left), 3)
    return out


def report(lines, book, loose, tests: dict[str, list], relevant, args, card_name) -> dict:
    """Book size, branches per hand trap, end-board values, and for each test set (lists of
    scenario starts) the share of hands that stay in the book for 1, 5, 10 and 20 of our answers."""
    base = [r for r, _ in lines if r["trap"] is None]
    rated = [r["value"]["held"] for r in base if r.get("value", {}).get("held") is not None]
    summary = {"states": len(book), "loose_states": len(loose), "lines": len(lines), "base_lines": len(base),
               "mean_field": round(sum(r["field"] for r in base) / max(1, len(base)), 2),
               "rated": len(rated), "held_mean": round(sum(rated) / len(rated), 3) if rated else None,
               "unbuildable": sum(r.get("value", {}).get("unbuildable", 0) for r in base),
               "errors": sum(r.get("value", {}).get("errors", 0) for r in base),
               "branches_fired": dict(Counter(card_name.get(r["trap"], str(r["trap"])) for r, _ in lines
                                              if r["trap"] and r["fired"])),
               "branches_tried": dict(Counter(card_name.get(r["trap"], str(r["trap"])) for r, _ in lines if r["trap"]))}
    for name, starts in tests.items():
        for kind, fallback in (("exact", None), ("loose", loose)):
            depths = [stay(start, book, relevant, fallback) for start in starts]
            n = max(1, len(depths))
            summary[f"{name}_{kind}"] = {"hands": len(depths), **{f"stay_{k}": round(sum(d >= k for d in depths) / n, 3)
                                                                  for k in (1, 5, 10, 20)}}
    return summary


@dataclass
class Setup:
    """What every step of a run shares: cards, decks, seeds, the model."""
    args: argparse.Namespace
    card_name: dict
    is_extra: object
    center: set
    decklist: tuple
    opponent: list
    seeds: list
    rules: dict
    feared: list
    model: object
    card_index: dict


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--deck", type=Path, required=True)
    p.add_argument("--deck-type", required=True)
    p.add_argument("--sheet", type=Path, help="player sheet CSV (starting hand, end board, replay, notes)")
    p.add_argument("--replays", type=Path, nargs="*", default=[], help="Duelingbook logs (card movements)")
    p.add_argument("--lists", nargs="*", default=[],
                   help="other player sheets (Card/Board CSV, In Hand TSV), each as path or path::author")
    p.add_argument("--chokepoints", type=Path, help="JSON {hand trap: [card names]} from guides")
    p.add_argument("--rotk", type=Path, nargs="*", default=[], help="Road of the King reports (text)")
    p.add_argument("--pinned", type=Path, help="directory of pinned Discord replays (<id>.txt, descriptions.txt)")
    p.add_argument("--priority", nargs="*", default=[], help="seed titles (substrings) to search first")
    p.add_argument("--opponent", type=Path, default=rollout.REPO / "decks" / "salamangreat.ydk")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--hands", type=int, default=3, help="starting hands per seed (other fillers)")
    p.add_argument("--max-seeds", type=int, default=0, help="seeds to search at most (0: all)")
    p.add_argument("--grid", type=int, default=40, help="starter x extender hands at most")
    p.add_argument("--sources", default="replay,guide,sheet,community,grid",
                   help="seed sources to use, in search order (replay, guide, sheet, community, grid)")
    p.add_argument("--long-hands-first", action="store_true", help="within a source, hands with more seed cards first")
    p.add_argument("--max-searches", type=int, default=0, help="searches at most, in seed order (0: all)")
    p.add_argument("--traps", type=int, default=3, help="interruption branches per hand (most feared hand traps)")
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--rate-tries", type=int, default=4, help="board_break tries per end board (0: no rating)")
    p.add_argument("--agent-deck", type=Path, default=rollout.REPO / ".cache" / "yugiohmeta" / "decks" / "snake_eye.ydk")
    p.add_argument("--test-hands", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="book (JSON lines: one state per line)")
    p.add_argument("--dry-run", action="store_true", help="list the seeds and searches, search nothing")
    p.add_argument("--from-lines", type=Path, help="lines of an earlier run (<out>-lines.jsonl): book and report only")
    return p.parse_args(argv)


def setup(args) -> Setup:
    from .cards import CardTable
    from .model import load_checkpoint

    table = CardTable.from_cdb()
    card_name = {c.code: c.name for c in table.cards}
    types = {c.code: c.type for c in table.cards}
    names = cb.name_index(table)
    decklist = worlds.read_decklist(args.deck)
    pool = set(decklist[0]) | set(decklist[1])
    for combo in cb.load(args.deck_type, names):
        pool |= set(combo.cards()) | set(combo.end_board)
    resolve = sd.resolver(names, pool, card_name)
    seeds, rules, feared = sd.load(args, args.deck_type, names, resolve, decklist, card_name)
    model = load_checkpoint(args.checkpoint, args.device, dropout=0.0)
    return Setup(args, card_name, lambda code: bool(types.get(code, 0) & cc.TYPE_EXTRA),
                 {c.code for c in table.cards if cc.wants_center(c.text)}, decklist,
                 worlds.read_decklist(args.opponent)[0], seeds, rules, feared, model, model.card_features.index_map())


def scenario(s: Setup, seed: Seed, variant: int, trap: int | None):
    """The puzzle of one starting hand (the seed's cards plus fillers of `variant`), with `trap` in
    the opponent's hand: (script, duel seed, start, fillers)."""
    fill = random.Random(f"{seed.id}-hand-{variant}")
    rng = random.Random(f"{seed.id}-{variant}")
    hand = list(seed.hand)
    rest = [c for c in s.decklist[0] if c not in hand]
    fill.shuffle(rest)
    hand += rest[:max(0, HAND_SIZE - len(hand))]
    script = cc.scenario_script(seed.combo, s.decklist, s.opponent, s.is_extra, rng, hand=hand,
                                opponent_hand=[trap] if trap else [])
    duel_seed = rng.getrandbits(32)
    return script, duel_seed, cc.puzzle_start(s.args.deck, s.args.opponent, script, duel_seed), hand[len(seed.hand):]


def search(s: Setup, seed: Seed, variant: int, trap: int | None) -> tuple[dict, object]:
    began = time.perf_counter()
    script, duel_seed, start, fillers = scenario(s, seed, variant, trap)
    bot = hand_trap_bot(trap, s.rules.get(trap, set()) | seed.chokes.get(trap, set())) if trap else None
    line, targets = cc.check(seed.combo, start, s.model, s.card_index, s.args.device, beam=s.args.beam,
                             center=s.center, is_extra=s.is_extra, opponent=bot)
    duel = start()
    for key in line.keys if duel is not None else []:
        if duel.finished or key not in duel.keys():
            break
        duel.act(key)
    final = cc.end_board(duel) if duel is not None else []
    on_field = [c for c in final if c["location"] in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE)]
    return {"seed": seed.id, "title": seed.title, "source": seed.source, "author": seed.author, "origin": seed.origin,
            "variant": variant, "trap": trap, "fired": fired(line.keys, trap) if trap else None, "line": line.keys,
            "steps": len(targets), "progress": line.progress, "skipped": line.skipped, "key_cards": line.key_cards,
            "book_end_board": seed.combo.end_board, "field": len(on_field), "end_board": final, "script": script,
            "duel_seed": duel_seed, "first": seed.first, "hand": seed.hand, "fillers": fillers,
            "seconds": round(time.perf_counter() - began, 1)}, start


def jobs_for(s: Setup) -> list[tuple]:
    """Each hand's base search, then its interruption branches: the seed's own hand traps first (a
    log shows where they hit), then the most feared ones."""
    jobs = []
    for seed in s.seeds:
        traps = list(dict.fromkeys([t for t in seed.vs if t in s.rules or s.card_name.get(t) in sd.INTERRUPTERS]
                                   + s.feared))[:s.args.traps]
        for variant in range(s.args.hands):
            jobs.append((seed, variant, None))
            jobs += [(seed, variant, t) for t in traps]
    return jobs[:s.args.max_searches] if s.args.max_searches else jobs


def meta_of(args) -> dict:
    return {"deck": str(args.deck), "deck_type": args.deck_type, "opponent": str(args.opponent),
            "sources": args.sources}


def load_lines(args) -> list[tuple[dict, object]]:
    """--from-lines: the stored lines, after checking they were made with the same deck, deck type
    and opponent (their scenarios only replay with those)."""
    meta_path = args.from_lines.with_name(args.from_lines.name.replace("-lines.jsonl", "-meta.json"))
    if not meta_path.exists():
        raise SystemExit(f"--from-lines: no {meta_path.name} beside {args.from_lines.name}")
    stored = json.loads(meta_path.read_text(encoding="utf-8"))
    wanted = meta_of(args)
    wrong = {k: (stored.get(k), wanted[k]) for k in ("deck", "deck_type", "opponent") if stored.get(k) != wanted[k]}
    if wrong:
        raise SystemExit(f"--from-lines: the lines were made with other settings: {wrong}")
    lines = []
    for text in args.from_lines.read_text(encoding="utf-8").splitlines():
        result = json.loads(text)
        lines.append((result, cc.puzzle_start(args.deck, args.opponent, result["script"], result["duel_seed"])))
    return lines


def run_searches(s: Setup, jobs: list[tuple], lines_path: Path) -> list[tuple[dict, object]]:
    """All searches in threads; each line is appended to `lines_path` as soon as it is found."""
    lines = []
    with ThreadPoolExecutor(max(1, s.args.workers)) as pool, open(lines_path, "w", encoding="utf-8") as out:
        futures = [pool.submit(search, s, *job) for job in jobs]
        for n, future in enumerate(futures):
            try:
                result, start = future.result()
            except Exception as error:  # one broken scenario must not stop the book
                print(f"{n + 1}/{len(futures)} error {str(error)[:120]}", flush=True)
                continue
            lines.append((result, start))
            out.write(json.dumps(result) + "\n")
            out.flush()
            print(f"{n + 1}/{len(futures)} {result['source']:9s} field {result['field']} "
                  f"trap {s.card_name.get(result['trap'], '-')[:20]:20s} fired {result['fired']} "
                  f"{result['seconds']:.0f}s  {result['title'][:50]}", flush=True)
    return lines


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    s = setup(args)
    print(f"{len(s.seeds)} seeds ({Counter(seed.source for seed in s.seeds)}), hand traps: "
          f"{[s.card_name.get(t) for t in s.feared[:8]]}", flush=True)
    jobs = jobs_for(s)
    print(f"{len(jobs)} searches", flush=True)
    if args.dry_run:
        for seed in s.seeds:
            print(f"  {seed.source:9s} {seed.author[:14]:14s} hand={[s.card_name.get(c) for c in seed.hand]} "
                  f"steps={len(cc.step_targets(seed.combo))} board={[s.card_name.get(c) for c in seed.combo.end_board]} "
                  f"vs={[s.card_name.get(c) for c in seed.vs]}  {seed.title[:60]}")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_name(args.out.stem)
    lines_path, meta_path = Path(f"{stem}-lines.jsonl"), Path(f"{stem}-meta.json")
    if args.from_lines:
        lines = load_lines(args)
    else:
        meta_path.write_text(json.dumps(meta_of(args)), encoding="utf-8")
        lines = run_searches(s, jobs, lines_path)
    # End boards rated by board_break (base lines only: a line with a hand trap fired cannot be
    # rebuilt there, the agent's seat holds other cards).
    agent = worlds.read_decklist(args.agent_deck)
    policy = rollout.ModelPolicy(s.model, args.device, sample=True, seed=args.seed)
    for result, _ in lines:
        if "value" not in result:
            result["value"] = (rate(result, args, agent, policy, s.card_index) if result["trap"] is None
                               else {"held": None, "left": None, "tries": 0, "unbuildable": 0, "errors": 0})
    lines_path.write_text("".join(json.dumps(r) + "\n" for r, _ in lines), encoding="utf-8")
    relevant = book_cards([r for r, _ in lines])
    book, loose = build_book(lines, relevant)
    args.out.write_text("".join(json.dumps(e) + "\n" for e in book.values()), encoding="utf-8")
    meta_path.write_text(json.dumps({**meta_of(args), "relevant": sorted(relevant)}), encoding="utf-8")
    by_id = {seed.id: seed for seed in s.seeds}
    base = [(r, start) for r, start in lines if r["trap"] is None]
    tests = {"own_lines": [start for _, start in base],
             "seed_hands": [scenario(s, by_id[r["seed"]], 1000 + r["variant"], None)[2] for r, _ in base
                            if r["seed"] in by_id],
             "random_hands": [random_start(t, args, s.decklist, s.opponent) for t in range(args.test_hands)]}
    summary = report(lines, book, loose, tests, relevant, args, s.card_name)
    print(json.dumps(summary), flush=True)
    Path(f"{stem}-report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def random_start(t: int, args, decklist, opponent):
    """A scenario with an honest random hand from the deck (test hand `t`)."""
    main = list(decklist[0])
    rng = random.Random(f"{args.seed}-test-{t}")
    rng.shuffle(main)
    combo = cb.Combo(f"test-{t}", "test", args.deck_type, 0, [], [])
    script = cc.scenario_script(combo, decklist, opponent, lambda c: False, rng, hand=main[:HAND_SIZE])
    return cc.puzzle_start(args.deck, args.opponent, script, t)


if __name__ == "__main__":
    main()
