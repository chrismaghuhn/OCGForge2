"""K1 of the combo teacher: replay community combos in our engine (docs/combo-teacher-design.md).

    python -m ocgforge.combo_check --deck decks/elfnote.ydk --deck-type Elfnote --checkpoint runs/x/latest

Each combo becomes a solo puzzle: we go first with its starting cards (filled up to 5 from the
decklist), the opponent holds a hand but never acts. A beam search over the engine's answers follows the combo's
steps: answers that use the next step's card advance the combo, the model ranks everything else
(materials, zones, targets). A combo counts as checked when every step card was used in order. The
end board and the line (semantic keys) go into the combo book. Several deck shuffles per combo
(--shuffles) keep draws and excavations honest: a combo only counts if it works in all of them.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import combos as cb
from . import features, rollout, worlds

TYPE_EXTRA = 0x40 | 0x2000 | 0x800000 | 0x4000000  # Fusion, Synchro, Xyz, Link
PHASE_END = re.compile(r"^idle\.[6-9]\.\d+$")  # to Battle Phase / End Phase (no card)


def scenario_script(combo: cb.Combo, decklist: tuple[list[int], list[int]], opponent: list[int], is_extra,
                    rng: random.Random, hand_size: int = 5) -> str:
    """The puzzle: our hand (the combo's starting cards, filled up from the decklist), our deck and
    Extra Deck (the decklist plus the combo's cards it lacks), an opponent with a deck and a hand it
    never plays (effects like PSY-Framelord Omega's need one)."""
    main, extra = list(decklist[0]), list(decklist[1])
    for code in combo.cards():  # the combo's cards we do not run
        target = extra if is_extra(code) else main
        if code not in target:
            target.append(code)
    hand = []
    for entry in combo.starting:
        if entry.code is not None and entry.code in main:
            main.remove(entry.code)
            hand.append(entry.code)
    rng.shuffle(main)
    while len(hand) < hand_size and main:
        hand.append(main.pop())
    lines = ["Debug.ReloadFieldBegin(0,5)", "Debug.SetPlayerInfo(0,8000,0,0)", "Debug.SetPlayerInfo(1,8000,0,0)"]
    for i, code in enumerate(main):
        lines.append(f"Debug.AddCard({code},0,0,{worlds.LOCATION_DECK},{i},8)")
    for code in extra:
        lines.append(f"Debug.AddCard({code},0,0,{worlds.LOCATION_EXTRA},0,8)")
    for i, code in enumerate(hand):
        lines.append(f"Debug.AddCard({code},0,0,{worlds.LOCATION_HAND},{i},10)")
    opponent = list(opponent)
    rng.shuffle(opponent)
    for i, code in enumerate(opponent[hand_size:]):
        lines.append(f"Debug.AddCard({code},1,1,{worlds.LOCATION_DECK},{i},8)")
    for i, code in enumerate(opponent[:hand_size]):  # it never plays them, but effects may target a hand
        lines.append(f"Debug.AddCard({code},1,1,{worlds.LOCATION_HAND},{i},10)")
    lines.append("Debug.ReloadFieldEnd()")
    return "\n".join(lines) + "\n"


@torch.no_grad()
def candidate_logprobs(model, rows: list[dict], device: str) -> list[list[float]]:
    model.eval()
    inputs = features.collate(rows, device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
        scores, _ = model(inputs)
    logprobs = torch.log_softmax(scores.float(), dim=-1).cpu()
    return [logprobs[i, : len(r["candidate_kind"])].tolist() for i, r in enumerate(rows)]


@dataclass
class Line:
    keys: list[str] = field(default_factory=list)  # every answer, both players
    progress: int = 0  # step cards used so far
    score: float = 0.0  # 10 per step card, plus the model's log-probabilities
    used: list[int] = field(default_factory=list)  # index of the decision that used each step card


def check(combo: cb.Combo, start, model, card_index, device: str, beam: int = 12, branch: int = 4,
          max_decisions: int = 150, lookahead: int = 3) -> tuple[Line, list[int]]:
    """Beam search through the combo from a fresh puzzle duel (`start()`); returns the best line and
    the step cards (codes, in order) it had to follow."""
    targets = [e.code for step in combo.steps for e in step if e.code is not None]
    lines = [Line()]
    done: list[Line] = []
    for _ in range(max_decisions):
        expanded: list[tuple[Line, object]] = []
        for line in lines:
            duel = start()
            if duel is None:
                continue
            for key in line.keys:
                duel.act(key)
            while not duel.finished and duel.player != 0:  # the opponent never acts
                key = worlds.decline(duel.keys())
                duel.act(key)
                line.keys.append(key)
            if duel.finished or duel.turn > 1 or line.progress == len(targets):
                done.append(line)
            else:
                expanded.append((line, duel))
        if not expanded:
            break
        logprobs = candidate_logprobs(model, [d.encode(card_index) for _, d in expanded], device)
        children = []
        for (line, duel), lp in zip(expanded, logprobs):
            keys = duel.keys()
            options = []
            for i, key in enumerate(keys):
                ref = worlds.card_ref(key)
                advance, value = 0, lp[i]
                if ref is not None:
                    code = int(ref[1])
                    # The next step card, or one a little later: the combo text names cards the
                    # engine never asks about (an effect that just resolves), so steps may be skipped.
                    for skip in range(lookahead):
                        if line.progress + skip < len(targets) and code == targets[line.progress + skip]:
                            advance, value = skip + 1, value + 10.0 - 3.0 * skip
                            break
                elif line.progress < len(targets) and PHASE_END.match(key):
                    value -= 20.0  # leaving Main Phase 1 with the combo unfinished
                options.append((value, i, advance))
            for value, i, advance in sorted(options, reverse=True)[:branch]:
                child = Line(line.keys + [keys[i]], line.progress + advance, line.score + value,
                             line.used + [len(line.keys)] * advance)
                children.append(child)
        lines = sorted(children, key=lambda l: (l.progress, l.score), reverse=True)[:beam]
    best = max(done + lines, key=lambda l: (l.progress, l.score))
    return best, targets


def end_board(duel) -> list[dict]:
    """Our field and the cards we added to hand: what the combo left us."""
    snapshot = duel.snapshot()
    return [{"code": c["code"], "location": c["location"], "sequence": c["sequence"], "position": c["position"]}
            for c in snapshot["cards"] if c["controller"] == 0
            and c["location"] in (worlds.LOCATION_MZONE, worlds.LOCATION_SZONE, worlds.LOCATION_HAND)]


def main(argv=None) -> None:
    from ._core import Duel
    from .cards import CardTable
    from .model import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--deck-type", required=True, help="masterduelmeta deck type, e.g. Elfnote")
    parser.add_argument("--opponent", type=Path, default=rollout.REPO / "decks" / "salamangreat.ydk")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shuffles", type=int, default=3, help="deck orders a combo must work in")
    parser.add_argument("--beam", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="check only the best-voted N combos")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, help="combo book (JSON lines)")
    args = parser.parse_args(argv)

    table = CardTable.from_cdb()
    types = {card.code: card.type for card in table.cards}
    is_extra = lambda code: bool(types.get(code, 0) & TYPE_EXTRA)  # noqa: E731
    names = cb.name_index(table)
    model = load_checkpoint(args.checkpoint, args.device, dropout=0.0)
    card_index = model.card_features.index_map()
    decklist = worlds.read_decklist(args.deck)
    opponent = worlds.read_decklist(args.opponent)[0]
    combos = sorted((c for c in cb.load(args.deck_type, names) if c.usable and c.steps),
                    key=lambda c: -c.votes)
    if args.limit:
        combos = combos[: args.limit]
    cache = rollout.REPO / ".cache"
    verbs = Counter()  # action text -> [used, total]
    verbs_used = Counter()
    results = []
    for n, combo in enumerate(combos):
        checked = 0
        best = None
        for shuffle in range(args.shuffles):
            rng = random.Random(f"{combo.id}-{shuffle}")
            script = scenario_script(combo, decklist, opponent, is_extra, rng)
            seed = rng.getrandbits(32)

            def start(script=script, seed=seed):
                duel = Duel.world(str(cache / "cardscripts"), str(cache / "card_data.tsv"), str(args.deck),
                                  str(args.opponent), script, seed, 0)
                for _ in range(20):  # through the draw and standby phase to our first decision
                    if duel.finished or duel.player == 0:
                        break
                    duel.act(worlds.decline(duel.keys()))
                return None if duel.finished else duel

            line, targets = check(combo, start, model, card_index, args.device, beam=args.beam)
            ok = bool(targets) and line.progress == len(targets)
            checked += ok
            if best is None or line.progress > best[0].progress:
                best = (line, targets, start)
        line, targets, start = best
        entries = [e for step in combo.steps for e in step if e.code is not None]
        for i, e in enumerate(entries):
            verb = e.action.strip().lower() or "(use)"
            verbs[verb] += 1
            verbs_used[verb] += i < line.progress
        duel = start()
        for key in line.keys:
            duel.act(key)
        result = {"combo": combo.id, "title": combo.title, "votes": combo.votes, "steps": len(targets),
                  "progress": line.progress, "checked_shuffles": checked, "shuffles": args.shuffles,
                  "checked": checked == args.shuffles, "line": line.keys, "end_board": end_board(duel),
                  "book_end_board": combo.end_board,
                  "key_cards_reached": (sum(1 for c in combo.end_board if c in {b["code"] for b in end_board(duel)})
                                        / len(combo.end_board)) if combo.end_board else None}
        results.append(result)
        print(f"{n + 1}/{len(combos)} {'OK  ' if result['checked'] else 'part'} {line.progress}/{len(targets)} "
              f"steps, {checked}/{args.shuffles} shuffles  {combo.title[:60]}", flush=True)
    passed = sum(r["checked"] for r in results)
    print(f"checked: {passed}/{len(results)} combos ({passed / max(1, len(results)):.0%})")
    print("verb coverage (used/total):", {v: f"{verbs_used[v]}/{verbs[v]}" for v, _ in verbs.most_common(20)})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")


if __name__ == "__main__":
    main()
